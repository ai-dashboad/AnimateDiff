"""
Camera Trajectory Controller -- precise trajectory-based camera control for
AnimateDiff V4 video generation.

Provides three levels of camera control:

  1. **prompt_only** -- appends cinematic camera descriptions to the text
     prompt. Works with any backend model (Wan, CogVideoX, LTX, etc.).

  2. **plucker** -- computes dense Plucker ray embeddings from camera poses
     for injection into CameraCtrl-compatible models. Based on:
     He et al., "CameraCtrl: Enabling Camera Control for Text-to-Video
     Generation" (arXiv 2404.02101, 2024).

  3. **optical_flow** -- converts camera trajectories to dense optical flow
     maps for FloVD-style conditioning. Based on:
     Jin et al., "FloVD: Optical Flow Meets Video Diffusion Model for
     Enhanced Camera-Controlled Video Synthesis" (CVPR 2025).

Mathematical reference:
  Plucker embedding for pixel (u, v) in frame i:
    d = normalize(R_i @ K_inv @ [u, v, 1]^T)
    o = camera_center_i (= -R_i^T @ t_i in world space, or directly t_i)
    P(u,v) = (o x d, d) in R^6

  Full tensor: P in R^{N x 6 x H x W}

Usage:
    from animatediff.core.camera_ctrl import CameraTrajectory, CameraController

    traj = CameraTrajectory.from_preset("dolly_in", num_frames=49, width=832, height=480)
    ctrl = CameraController(method="prompt_only")
    pipeline_kwargs = ctrl.apply(pipeline_kwargs, traj)
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt torch import -- the module is still importable without torch for
# planning / preset queries, but Plucker / flow output requires it.
# ---------------------------------------------------------------------------
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Preset catalog
# ---------------------------------------------------------------------------

# Each preset is a function that takes (num_frames) and returns a list of
# camera poses expressed as (position_xyz, euler_ypr_degrees, fov_degrees).
# Euler angles follow the convention: yaw (Y-up), pitch (X-right), roll (Z-fwd).

_BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {}


def _register(name: str, *, description: str, prompt_suffix: str,
              negative_suffix: str = "", intensity: str = "medium"):
    """Decorator that registers a trajectory generator function."""
    def _wrap(fn):
        _BUILTIN_PRESETS[name] = {
            "generator": fn,
            "description": description,
            "prompt_suffix": prompt_suffix,
            "negative_suffix": negative_suffix,
            "intensity": intensity,
        }
        return fn
    return _wrap


# ---- Preset generators (position, euler_ypr_deg, fov_deg) per frame ------

@_register("static",
           description="No camera movement, locked-off shot.",
           prompt_suffix="static locked-off camera, no camera movement, steady tripod shot",
           negative_suffix="camera shake, handheld, pan, zoom, dolly, tracking",
           intensity="none")
def _gen_static(n: int, **kw):
    return [([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 60.0)] * n


@_register("pan_left",
           description="Smooth horizontal pan from right to left.",
           prompt_suffix="smooth horizontal camera pan from right to left, steady lateral sweep",
           negative_suffix="static camera, vertical movement, zoom, push in",
           intensity="low")
def _gen_pan_left(n: int, *, angle: float = 30.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        yaw = t * angle  # positive yaw = pan left in our convention
        poses.append(([0.0, 0.0, 0.0], [yaw, 0.0, 0.0], 60.0))
    return poses


@_register("pan_right",
           description="Smooth horizontal pan from left to right.",
           prompt_suffix="smooth horizontal camera pan from left to right, steady lateral sweep",
           negative_suffix="static camera, vertical movement, zoom, push in",
           intensity="low")
def _gen_pan_right(n: int, *, angle: float = 30.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        yaw = -t * angle
        poses.append(([0.0, 0.0, 0.0], [yaw, 0.0, 0.0], 60.0))
    return poses


@_register("tilt_up",
           description="Smooth vertical tilt from down to up.",
           prompt_suffix="smooth vertical camera tilt upward, revealing sky and upper scene",
           negative_suffix="static camera, horizontal pan, zoom, dolly",
           intensity="low")
def _gen_tilt_up(n: int, *, angle: float = 20.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        pitch = -t * angle  # negative pitch = tilt up
        poses.append(([0.0, 0.0, 0.0], [0.0, pitch, 0.0], 60.0))
    return poses


@_register("tilt_down",
           description="Smooth vertical tilt from up to down.",
           prompt_suffix="smooth vertical camera tilt downward, revealing ground and lower scene",
           negative_suffix="static camera, horizontal pan, zoom, dolly",
           intensity="low")
def _gen_tilt_down(n: int, *, angle: float = 20.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        pitch = t * angle
        poses.append(([0.0, 0.0, 0.0], [0.0, pitch, 0.0], 60.0))
    return poses


@_register("dolly_in",
           description="Camera pushes forward toward the subject.",
           prompt_suffix="camera smoothly pushing forward, dolly in toward the subject, "
                        "continuous forward motion tightening the frame",
           negative_suffix="static camera, pulling back, no forward movement",
           intensity="medium")
def _gen_dolly_in(n: int, *, distance: float = 2.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # Ease-in-out via cosine interpolation for cinematic feel
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        z = -s * distance  # negative Z = forward in our convention
        poses.append(([0.0, 0.0, z], [0.0, 0.0, 0.0], 60.0))
    return poses


@_register("dolly_out",
           description="Camera pulls backward away from the subject.",
           prompt_suffix="camera smoothly pulling backward, dolly out from the subject, "
                        "continuous backward motion widening the frame",
           negative_suffix="static camera, pushing in, no backward movement",
           intensity="medium")
def _gen_dolly_out(n: int, *, distance: float = 2.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        z = s * distance  # positive Z = backward
        poses.append(([0.0, 0.0, z], [0.0, 0.0, 0.0], 60.0))
    return poses


@_register("orbit_cw",
           description="Camera orbits clockwise around a central point.",
           prompt_suffix="camera orbiting smoothly clockwise around the subject, "
                        "circular dolly movement, rotating perspective",
           negative_suffix="static camera, no rotation, linear movement",
           intensity="medium")
def _gen_orbit_cw(n: int, *, radius: float = 3.0, degrees: float = 120.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        theta = math.radians(s * degrees)
        x = radius * math.sin(theta)
        z = radius * (1.0 - math.cos(theta))
        yaw = -math.degrees(theta)  # face center
        poses.append(([x, 0.0, z], [yaw, 0.0, 0.0], 60.0))
    return poses


@_register("orbit_ccw",
           description="Camera orbits counter-clockwise around a central point.",
           prompt_suffix="camera orbiting smoothly counter-clockwise around the subject, "
                        "circular dolly movement, rotating perspective",
           negative_suffix="static camera, no rotation, linear movement",
           intensity="medium")
def _gen_orbit_ccw(n: int, *, radius: float = 3.0, degrees: float = 120.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        theta = -math.radians(s * degrees)
        x = radius * math.sin(theta)
        z = radius * (1.0 - math.cos(theta))
        yaw = -math.degrees(theta)
        poses.append(([x, 0.0, z], [yaw, 0.0, 0.0], 60.0))
    return poses


@_register("crane_up",
           description="Camera rises vertically, revealing the scene from above.",
           prompt_suffix="camera crane rising upward, smooth vertical ascent revealing the "
                        "landscape below, dramatic perspective shift",
           negative_suffix="static camera, descending, ground locked, no vertical movement",
           intensity="medium")
def _gen_crane_up(n: int, *, rise: float = 2.0, tilt: float = 10.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        y = s * rise
        pitch = s * tilt  # slight downward tilt as we rise
        poses.append(([0.0, y, 0.0], [0.0, pitch, 0.0], 60.0))
    return poses


@_register("crane_down",
           description="Camera descends vertically toward the ground.",
           prompt_suffix="camera crane descending downward, smooth vertical descent "
                        "approaching the ground, revealing terrain details",
           negative_suffix="static camera, ascending, no vertical movement",
           intensity="medium")
def _gen_crane_down(n: int, *, drop: float = 2.0, tilt: float = -10.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        y = -s * drop
        pitch = s * tilt  # slight upward tilt as we descend
        poses.append(([0.0, y, 0.0], [0.0, pitch, 0.0], 60.0))
    return poses


@_register("fly_forward",
           description="Continuous forward flight through the scene (for aerial / xianxia flying).",
           prompt_suffix="continuous forward-moving flight shot, camera racing through the "
                        "environment, parallax depth with objects passing on both sides, "
                        "exhilarating first-person aerial movement",
           negative_suffix="static camera, ground level, no forward movement, locked off shot",
           intensity="high")
def _gen_fly_forward(n: int, *, distance: float = 5.0, height_var: float = 0.3, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        z = -t * distance
        # Gentle sine-wave vertical undulation for organic flight feel
        y = height_var * math.sin(2.0 * math.pi * t)
        # Subtle yaw oscillation
        yaw = 3.0 * math.sin(1.5 * math.pi * t)
        poses.append(([0.0, y, z], [yaw, 0.0, 0.0], 60.0))
    return poses


@_register("spiral_ascend",
           description="Camera spirals upward around the subject (for breakthrough scenes).",
           prompt_suffix="camera spiraling upward around the subject, ascending orbital "
                        "motion with the world falling away below, building energy and "
                        "intensity, vortex-like rotational ascent",
           negative_suffix="static camera, no rotation, descending, ground level",
           intensity="high")
def _gen_spiral_ascend(n: int, *, radius: float = 2.0, rise: float = 3.0,
                       revolutions: float = 1.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        # Accelerating spiral via ease-in curve
        s = t * t  # quadratic ease-in for accelerating energy
        theta = 2.0 * math.pi * revolutions * s
        x = radius * math.sin(theta)
        z_offset = radius * math.cos(theta) - radius  # start at origin
        y = s * rise
        yaw = -math.degrees(theta) % 360
        pitch = min(s * 15.0, 15.0)  # slight downward tilt
        poses.append(([x, y, z_offset], [yaw, pitch, 0.0], 60.0))
    return poses


@_register("truck_left",
           description="Camera slides laterally to the left (truck/crab movement).",
           prompt_suffix="smooth lateral camera truck sliding to the left, creating parallax "
                        "depth with foreground and background layers",
           negative_suffix="static camera, forward dolly, no lateral movement",
           intensity="low")
def _gen_truck_left(n: int, *, distance: float = 1.5, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        x = -s * distance
        poses.append(([x, 0.0, 0.0], [0.0, 0.0, 0.0], 60.0))
    return poses


@_register("truck_right",
           description="Camera slides laterally to the right (truck/crab movement).",
           prompt_suffix="smooth lateral camera truck sliding to the right, creating parallax "
                        "depth with foreground and background layers",
           negative_suffix="static camera, forward dolly, no lateral movement",
           intensity="low")
def _gen_truck_right(n: int, *, distance: float = 1.5, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        x = s * distance
        poses.append(([x, 0.0, 0.0], [0.0, 0.0, 0.0], 60.0))
    return poses


@_register("dutch_roll",
           description="Gradual Dutch angle roll for tension and unease.",
           prompt_suffix="tilting Dutch angle creating visual unease, horizon canting "
                        "diagonally with a slow creeping push forward",
           negative_suffix="level horizon, stable framing, no tilt, calm composition",
           intensity="low")
def _gen_dutch_roll(n: int, *, angle: float = 15.0, push: float = 0.5, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        roll = s * angle
        z = -s * push
        poses.append(([0.0, 0.0, z], [0.0, 0.0, roll], 60.0))
    return poses


@_register("zoom_in",
           description="Camera zooms in by narrowing the field of view.",
           prompt_suffix="smooth camera zoom in, tightening the frame on the subject, "
                        "field of view narrowing to focus attention",
           negative_suffix="static camera, zoom out, wide angle, no focal change",
           intensity="low")
def _gen_zoom_in(n: int, *, fov_start: float = 70.0, fov_end: float = 35.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        fov = fov_start + s * (fov_end - fov_start)
        poses.append(([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], fov))
    return poses


@_register("zoom_out",
           description="Camera zooms out by widening the field of view.",
           prompt_suffix="smooth camera zoom out, widening the frame to reveal more of the "
                        "environment, field of view expanding",
           negative_suffix="static camera, zoom in, tight framing, no focal change",
           intensity="low")
def _gen_zoom_out(n: int, *, fov_start: float = 40.0, fov_end: float = 80.0, **kw):
    poses = []
    for i in range(n):
        t = i / max(n - 1, 1)
        s = 0.5 * (1.0 - math.cos(math.pi * t))
        fov = fov_start + s * (fov_end - fov_start)
        poses.append(([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], fov))
    return poses


# ---------------------------------------------------------------------------
# Helper: mapping from camera_presets.json IDs to trajectory presets
# ---------------------------------------------------------------------------

_PRESET_JSON_TO_TRAJECTORY: Dict[str, Tuple[str, Dict[str, float]]] = {
    # aerial
    "fly_through_clouds": ("fly_forward", {"distance": 6.0, "height_var": 0.5}),
    "birds_eye_descend": ("crane_down", {"drop": 3.0}),
    "crane_up_reveal": ("crane_up", {"rise": 3.0, "tilt": 15.0}),
    "soaring_above_mountains": ("fly_forward", {"distance": 5.0, "height_var": 0.4}),
    # dynamic
    "sword_fight_handheld": ("static", {}),  # handheld is subject-tracking, not trajectory
    "chase_sequence": ("fly_forward", {"distance": 4.0}),
    "whip_pan": ("pan_left", {"angle": 90.0}),
    "speed_ramp": ("dolly_in", {"distance": 3.0}),
    "falling_descent": ("crane_down", {"drop": 5.0, "tilt": -20.0}),
    # dramatic
    "dramatic_orbit_360": ("orbit_cw", {"degrees": 360.0}),
    "dolly_zoom_vertigo": ("dolly_out", {"distance": 1.5}),  # combined with zoom_in at apply
    "slow_push_in": ("dolly_in", {"distance": 1.0}),
    "pull_back_reveal": ("dolly_out", {"distance": 2.5}),
    "dutch_angle_tension": ("dutch_roll", {"angle": 12.0, "push": 0.3}),
    "low_angle_power": ("tilt_up", {"angle": 10.0}),
    # intimate
    "face_closeup_rack_focus": ("static", {}),  # rack focus is lens, not trajectory
    "over_shoulder": ("static", {}),
    "profile_silhouette": ("truck_right", {"distance": 0.3}),
    # establishing
    "wide_establishing": ("pan_right", {"angle": 15.0}),
    "time_lapse_pan": ("pan_right", {"angle": 25.0}),
    "landscape_parallax": ("truck_right", {"distance": 2.0}),
    # xianxia_specific
    "meditation_float_up": ("crane_up", {"rise": 1.5, "tilt": 5.0}),
    "qi_gathering_orbit": ("orbit_cw", {"degrees": 180.0, "radius": 2.0}),
    "breakthrough_explosion": ("dolly_out", {"distance": 4.0}),
    "sword_flight_fpv": ("fly_forward", {"distance": 8.0, "height_var": 0.6}),
    "teleport_flash": ("zoom_in", {"fov_start": 60.0, "fov_end": 20.0}),
    "formation_array_overhead": ("orbit_cw", {"degrees": 90.0, "radius": 0.0}),
}


# ===========================================================================
# CameraPose
# ===========================================================================

@dataclass
class CameraPose:
    """A single camera pose in 3D space.

    Attributes:
        position: (x, y, z) world-space position.
        rotation: (yaw, pitch, roll) in degrees. Convention:
            yaw   = rotation about world Y axis (positive = left)
            pitch = rotation about local X axis (positive = down)
            roll  = rotation about local Z axis (positive = clockwise)
        fov: vertical field-of-view in degrees.
    """
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll
    fov: float = 60.0

    # ---- Conversion to 4x4 world-to-camera matrix ----

    def rotation_matrix(self) -> np.ndarray:
        """Return 3x3 rotation matrix R (world-to-camera) from Euler YPR."""
        yaw, pitch, roll = [math.radians(a) for a in self.rotation]

        # Y-axis (yaw)
        cy, sy = math.cos(yaw), math.sin(yaw)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)

        # X-axis (pitch)
        cp, sp = math.cos(pitch), math.sin(pitch)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)

        # Z-axis (roll)
        cr, sr = math.cos(roll), math.sin(roll)
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)

        # Combined: R = Rz @ Rx @ Ry  (intrinsic rotation YXZ order)
        return Rz @ Rx @ Ry

    def extrinsic_matrix(self) -> np.ndarray:
        """Return 4x4 world-to-camera extrinsic matrix [R | t; 0 0 0 1]."""
        R = self.rotation_matrix()
        t = np.array(self.position, dtype=np.float64)
        # Camera center in world coordinates is `position`.
        # Extrinsic t = -R @ camera_center
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = R
        ext[:3, 3] = -R @ t
        return ext

    def camera_center(self) -> np.ndarray:
        """Camera center (= position) in world coordinates."""
        return np.array(self.position, dtype=np.float64)

    def intrinsic_matrix(self, width: int, height: int) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix K for the given resolution.

        Assumes square pixels and principal point at image center.
        """
        fy = height / (2.0 * math.tan(math.radians(self.fov / 2.0)))
        fx = fy  # square pixels
        cx = width / 2.0
        cy = height / 2.0
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


# ===========================================================================
# CameraTrajectory
# ===========================================================================

class CameraTrajectory:
    """Represents a camera path through 3D space as a sequence of poses.

    This is the central data structure for trajectory-based camera control.
    It can be created from named presets, from explicit pose lists, or
    programmatically, and converted to Plucker embeddings or optical flow
    maps for injection into compatible video diffusion models.
    """

    def __init__(self, num_frames: int, width: int, height: int):
        """Initialize an empty trajectory.

        Args:
            num_frames: Number of video frames.
            width:  Output video width in pixels.
            height: Output video height in pixels.
        """
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.poses: List[CameraPose] = []
        self.preset_name: Optional[str] = None
        self.prompt_suffix: str = ""
        self.negative_suffix: str = ""

    # ---- Factory methods --------------------------------------------------

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        num_frames: int,
        width: int = 832,
        height: int = 480,
        **kwargs,
    ) -> "CameraTrajectory":
        """Create a trajectory from a named built-in preset.

        Args:
            preset_name: One of the registered preset names
                (``static``, ``pan_left``, ``dolly_in``, ``orbit_cw``, etc.).
            num_frames: Number of video frames.
            width:  Output width in pixels.
            height: Output height in pixels.
            **kwargs: Extra parameters forwarded to the preset generator
                (e.g. ``distance``, ``angle``, ``radius``).

        Returns:
            A populated CameraTrajectory.

        Raises:
            KeyError: If *preset_name* is not a registered preset.
        """
        if preset_name not in _BUILTIN_PRESETS:
            available = ", ".join(sorted(_BUILTIN_PRESETS.keys()))
            raise KeyError(
                f"Unknown trajectory preset '{preset_name}'. "
                f"Available: {available}"
            )

        info = _BUILTIN_PRESETS[preset_name]
        raw = info["generator"](num_frames, **kwargs)

        traj = cls(num_frames, width, height)
        traj.preset_name = preset_name
        traj.prompt_suffix = info["prompt_suffix"]
        traj.negative_suffix = info["negative_suffix"]

        for pos, rot, fov in raw:
            traj.poses.append(CameraPose(
                position=tuple(pos),
                rotation=tuple(rot),
                fov=fov,
            ))
        return traj

    @classmethod
    def from_poses(
        cls,
        poses: List[Dict[str, Any]],
        width: int = 832,
        height: int = 480,
    ) -> "CameraTrajectory":
        """Create a trajectory from explicit camera pose dicts.

        Args:
            poses: List of dicts, each containing:
                - ``position``: [x, y, z]
                - ``rotation``: [yaw, pitch, roll] in degrees
                - ``fov``:      vertical field-of-view in degrees (default 60)
            width:  Output width in pixels.
            height: Output height in pixels.

        Returns:
            A populated CameraTrajectory.
        """
        traj = cls(len(poses), width, height)
        for p in poses:
            pos = tuple(p.get("position", [0.0, 0.0, 0.0]))
            rot = tuple(p.get("rotation", [0.0, 0.0, 0.0]))
            fov = float(p.get("fov", 60.0))
            traj.poses.append(CameraPose(position=pos, rotation=rot, fov=fov))
        return traj

    @classmethod
    def from_extrinsics(
        cls,
        extrinsics: np.ndarray,
        fov: float = 60.0,
        width: int = 832,
        height: int = 480,
    ) -> "CameraTrajectory":
        """Create a trajectory from a batch of 4x4 extrinsic matrices.

        Args:
            extrinsics: (N, 4, 4) array of world-to-camera matrices.
            fov:        Uniform vertical FOV in degrees.
            width:      Output width in pixels.
            height:     Output height in pixels.

        Returns:
            A populated CameraTrajectory.
        """
        n = extrinsics.shape[0]
        traj = cls(n, width, height)
        for i in range(n):
            E = extrinsics[i]
            R = E[:3, :3]
            t = E[:3, 3]
            # Camera center = -R^T @ t
            center = (-R.T @ t).tolist()
            # Decompose R to Euler YPR
            yaw, pitch, roll = _rotation_to_euler_ypr(R)
            traj.poses.append(CameraPose(
                position=tuple(center),
                rotation=(math.degrees(yaw), math.degrees(pitch), math.degrees(roll)),
                fov=fov,
            ))
        return traj

    # ---- Plucker embeddings -----------------------------------------------

    def to_plucker(self) -> "torch.Tensor":
        """Convert the trajectory to Plucker ray embeddings.

        Computes the 6-channel Plucker embedding for every pixel in every
        frame, following the CameraCtrl formulation:

            For pixel (u, v) in frame i:
              d = normalize( R_i @ K_inv @ [u, v, 1]^T )
              o = camera_center_i
              P(u,v) = concat( o x d, d )  in R^6

        Returns:
            Tensor of shape ``(N, 6, H, W)`` where N is the number of frames.

        Raises:
            RuntimeError: If PyTorch is not installed.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for Plucker embedding computation.")

        N, H, W = self.num_frames, self.height, self.width

        # Build pixel coordinate grid (H, W) -- pixel centers at (u+0.5, v+0.5)
        u_coords = np.arange(W, dtype=np.float64) + 0.5
        v_coords = np.arange(H, dtype=np.float64) + 0.5
        uu, vv = np.meshgrid(u_coords, v_coords)  # (H, W) each
        ones = np.ones_like(uu)
        # Homogeneous pixel coords: (3, H*W)
        pixels = np.stack([uu.ravel(), vv.ravel(), ones.ravel()], axis=0)  # (3, H*W)

        plucker_all = np.zeros((N, 6, H, W), dtype=np.float32)

        for i, pose in enumerate(self.poses):
            R = pose.rotation_matrix()          # (3, 3)
            K = pose.intrinsic_matrix(W, H)     # (3, 3)
            K_inv = np.linalg.inv(K)            # (3, 3)
            o = pose.camera_center()            # (3,)

            # Direction vectors: d = R @ K_inv @ pixel_homogeneous
            # Note: for world-to-camera R, the ray direction in world space
            # is R^T @ K_inv @ [u,v,1]^T (transforming camera-space ray to world)
            dirs = R.T @ K_inv @ pixels   # (3, H*W)

            # Normalize directions
            norms = np.linalg.norm(dirs, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            dirs = dirs / norms  # (3, H*W)

            # Moment = o x d  (for each ray)
            # o is (3,), broadcast across H*W rays
            o_expanded = o[:, np.newaxis]  # (3, 1)
            moments = np.cross(o_expanded.T, dirs.T).T  # (3, H*W)

            # Stack: (moment, direction) -> (6, H*W)
            plucker = np.concatenate([moments, dirs], axis=0)  # (6, H*W)
            plucker_all[i] = plucker.reshape(6, H, W).astype(np.float32)

        return torch.from_numpy(plucker_all)

    # ---- Optical flow maps ------------------------------------------------

    def to_optical_flow(self) -> "torch.Tensor":
        """Convert the trajectory to dense optical flow maps.

        Computes per-pixel 2D displacement (dx, dy) between consecutive
        frames by projecting a set of 3D world points through adjacent camera
        poses and taking the pixel-space difference.

        This is a geometric approximation assuming a planar scene at a
        nominal depth. For scenes with significant depth variation, the
        actual flow would differ -- but this provides a good camera-motion
        prior for FloVD-style conditioning.

        Returns:
            Tensor of shape ``(N-1, 2, H, W)`` where channel 0 is horizontal
            flow (dx) and channel 1 is vertical flow (dy), in pixels.

        Raises:
            RuntimeError: If PyTorch is not installed.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for optical flow computation.")

        N, H, W = self.num_frames, self.height, self.width

        if N < 2:
            return torch.zeros((0, 2, H, W), dtype=torch.float32)

        # Pixel grid
        u_coords = np.arange(W, dtype=np.float64) + 0.5
        v_coords = np.arange(H, dtype=np.float64) + 0.5
        uu, vv = np.meshgrid(u_coords, v_coords)
        ones = np.ones_like(uu)
        pixels = np.stack([uu.ravel(), vv.ravel(), ones.ravel()], axis=0)  # (3, H*W)

        # Nominal scene depth for 3D point estimation
        SCENE_DEPTH = 5.0

        flow_all = np.zeros((N - 1, 2, H, W), dtype=np.float32)

        for i in range(N - 1):
            pose_a = self.poses[i]
            pose_b = self.poses[i + 1]

            R_a = pose_a.rotation_matrix()
            K_a = pose_a.intrinsic_matrix(W, H)
            K_a_inv = np.linalg.inv(K_a)
            o_a = pose_a.camera_center()

            R_b = pose_b.rotation_matrix()
            K_b = pose_b.intrinsic_matrix(W, H)
            o_b = pose_b.camera_center()

            # Unproject pixels from frame A to 3D world points at SCENE_DEPTH
            # Camera-space rays: K_a_inv @ pixels
            cam_rays = K_a_inv @ pixels  # (3, H*W)
            # Normalize and scale to SCENE_DEPTH along Z
            z_vals = cam_rays[2:3, :]
            z_vals = np.maximum(np.abs(z_vals), 1e-8)
            cam_points = cam_rays * (SCENE_DEPTH / z_vals)  # (3, H*W)

            # To world space: P_world = R_a^T @ cam_points + o_a
            world_points = R_a.T @ cam_points + o_a[:, np.newaxis]  # (3, H*W)

            # Project world points into frame B
            # cam_b = R_b @ (P_world - o_b)
            cam_b = R_b @ (world_points - o_b[:, np.newaxis])  # (3, H*W)

            # Project: pixel_b = K_b @ cam_b, then divide by z
            proj_b = K_b @ cam_b  # (3, H*W)
            z_b = proj_b[2:3, :]
            z_b = np.where(np.abs(z_b) < 1e-8, 1e-8, z_b)
            pixel_b = proj_b[:2, :] / z_b  # (2, H*W)

            # Flow = pixel_b - pixel_a
            pixel_a = pixels[:2, :]  # (2, H*W)
            flow = (pixel_b - pixel_a).astype(np.float32)  # (2, H*W)
            flow_all[i] = flow.reshape(2, H, W)

        return torch.from_numpy(flow_all)

    # ---- Interpolation ----------------------------------------------------

    def interpolate(self, factor: int) -> "CameraTrajectory":
        """Interpolate the trajectory to a higher frame count.

        Uses spherical linear interpolation (SLERP) for rotations and
        linear interpolation for position and FOV.

        Args:
            factor: Integer multiplication factor. E.g. factor=2 doubles
                the frame count.

        Returns:
            A new CameraTrajectory with ``num_frames * factor`` poses.
        """
        if factor <= 1 or len(self.poses) < 2:
            return self._copy()

        new_n = (self.num_frames - 1) * factor + 1
        traj = CameraTrajectory(new_n, self.width, self.height)
        traj.preset_name = self.preset_name
        traj.prompt_suffix = self.prompt_suffix
        traj.negative_suffix = self.negative_suffix

        for i in range(len(self.poses) - 1):
            pa = self.poses[i]
            pb = self.poses[i + 1]

            Ra = pa.rotation_matrix()
            Rb = pb.rotation_matrix()

            for j in range(factor):
                t = j / factor
                # Lerp position
                pos = _lerp_tuple(pa.position, pb.position, t)
                # Slerp rotation
                R_interp = _slerp_rotation(Ra, Rb, t)
                yaw, pitch, roll = _rotation_to_euler_ypr(R_interp)
                rot = (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))
                # Lerp FOV
                fov = pa.fov + t * (pb.fov - pa.fov)
                traj.poses.append(CameraPose(position=pos, rotation=rot, fov=fov))

        # Append the last pose
        last = self.poses[-1]
        traj.poses.append(CameraPose(
            position=last.position,
            rotation=last.rotation,
            fov=last.fov,
        ))
        return traj

    # ---- Composition (combine trajectories) --------------------------------

    def __add__(self, other: "CameraTrajectory") -> "CameraTrajectory":
        """Concatenate two trajectories sequentially.

        The second trajectory's poses are appended after the first.
        Resolution is taken from the first trajectory.
        """
        traj = CameraTrajectory(
            self.num_frames + other.num_frames,
            self.width, self.height,
        )
        traj.poses = list(self.poses) + list(other.poses)
        traj.prompt_suffix = self.prompt_suffix
        traj.negative_suffix = self.negative_suffix
        return traj

    # ---- Utilities --------------------------------------------------------

    def _copy(self) -> "CameraTrajectory":
        traj = CameraTrajectory(self.num_frames, self.width, self.height)
        traj.poses = [CameraPose(p.position, p.rotation, p.fov) for p in self.poses]
        traj.preset_name = self.preset_name
        traj.prompt_suffix = self.prompt_suffix
        traj.negative_suffix = self.negative_suffix
        return traj

    def __len__(self) -> int:
        return len(self.poses)

    def __repr__(self) -> str:
        name = self.preset_name or "custom"
        return (f"CameraTrajectory(preset={name!r}, frames={self.num_frames}, "
                f"res={self.width}x{self.height}, poses={len(self.poses)})")


# ===========================================================================
# CameraController -- pipeline integration
# ===========================================================================

class CameraController:
    """Injects camera control signals into video generation pipelines.

    Three control methods are supported:

    ``"prompt_only"``
        Appends a cinematic camera description to the text prompt.
        Works with every backend model. This is the default and is always
        available as a fallback.

    ``"plucker"``
        Computes dense Plucker ray embeddings and adds them to the
        pipeline kwargs as ``camera_plucker_embedding``. Requires a
        CameraCtrl-compatible model to consume the embeddings.

    ``"optical_flow"``
        Computes dense optical flow maps and adds them as
        ``camera_optical_flow``. Requires a FloVD-compatible model.
    """

    METHODS = ("prompt_only", "plucker", "optical_flow")

    def __init__(self, method: str = "prompt_only"):
        """Initialize the controller.

        Args:
            method: One of ``"prompt_only"``, ``"plucker"``, ``"optical_flow"``.
        """
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown camera control method '{method}'. "
                f"Choose from: {self.METHODS}"
            )
        self.method = method

    def apply(
        self,
        pipeline_kwargs: Dict[str, Any],
        trajectory: CameraTrajectory,
    ) -> Dict[str, Any]:
        """Modify pipeline kwargs to include camera control signals.

        This method returns a **new** dict (the original is not mutated)
        with the appropriate camera data injected.

        For ``"prompt_only"``:
            - Appends the trajectory's prompt suffix to ``prompt``.
            - Appends the trajectory's negative suffix to ``negative_prompt``.

        For ``"plucker"``:
            - Adds ``camera_plucker_embedding``: Tensor (N, 6, H, W).
            - Also applies prompt modification as a fallback.

        For ``"optical_flow"``:
            - Adds ``camera_optical_flow``: Tensor (N-1, 2, H, W).
            - Also applies prompt modification as a fallback.

        Args:
            pipeline_kwargs: The dict of keyword arguments that will be
                passed to ``BasePipeline.generate()`` or to a diffusers
                pipeline ``__call__``.
            trajectory: The camera trajectory to inject.

        Returns:
            A new dict with camera control signals added.
        """
        kw = dict(pipeline_kwargs)  # shallow copy

        # --- Prompt augmentation (always applied) --------------------------
        if trajectory.prompt_suffix:
            prompt = kw.get("prompt", "")
            if prompt and not prompt.rstrip().endswith(","):
                prompt = prompt.rstrip() + ", "
            kw["prompt"] = prompt + trajectory.prompt_suffix

        if trajectory.negative_suffix:
            neg = kw.get("negative_prompt", "")
            if neg and not neg.rstrip().endswith(","):
                neg = neg.rstrip() + ", "
            kw["negative_prompt"] = neg + trajectory.negative_suffix

        # --- Plucker embeddings --------------------------------------------
        if self.method == "plucker":
            try:
                plucker = trajectory.to_plucker()
                kw["camera_plucker_embedding"] = plucker
                logger.info(
                    f"Injected Plucker embeddings: shape {tuple(plucker.shape)}"
                )
            except Exception as e:
                logger.warning(f"Failed to compute Plucker embeddings: {e}. "
                               f"Falling back to prompt-only.")

        # --- Optical flow --------------------------------------------------
        elif self.method == "optical_flow":
            try:
                flow = trajectory.to_optical_flow()
                kw["camera_optical_flow"] = flow
                logger.info(
                    f"Injected optical flow: shape {tuple(flow.shape)}"
                )
            except Exception as e:
                logger.warning(f"Failed to compute optical flow: {e}. "
                               f"Falling back to prompt-only.")

        return kw

    def __repr__(self) -> str:
        return f"CameraController(method={self.method!r})"


# ===========================================================================
# Integration with camera_presets.json
# ===========================================================================

def get_trajectory_for_preset(
    preset_id: str,
    num_frames: int,
    width: int = 832,
    height: int = 480,
) -> CameraTrajectory:
    """Convert a camera_presets.json preset ID to a CameraTrajectory.

    This bridges the prompt-based camera system (``camera_presets.json``) with
    the trajectory-based system. For presets that map cleanly to geometric
    trajectories, a full trajectory is returned. For presets that are purely
    lens-based (e.g. rack focus) or subject-tracking (e.g. handheld), a
    static trajectory is returned and the camera effect is conveyed only
    through the prompt suffix.

    Args:
        preset_id: ID from ``camera_presets.json``
            (e.g. ``"fly_through_clouds"``, ``"dramatic_orbit_360"``).
        num_frames: Number of frames for the trajectory.
        width:  Output width in pixels.
        height: Output height in pixels.

    Returns:
        A CameraTrajectory populated with poses and prompt suffixes.

    Raises:
        KeyError: If *preset_id* is not found in the preset mapping.
    """
    if preset_id in _PRESET_JSON_TO_TRAJECTORY:
        traj_name, extra_kw = _PRESET_JSON_TO_TRAJECTORY[preset_id]
        traj = CameraTrajectory.from_preset(
            traj_name, num_frames, width=width, height=height, **extra_kw
        )
    elif preset_id in _BUILTIN_PRESETS:
        # Direct trajectory preset name (e.g. "dolly_in")
        traj = CameraTrajectory.from_preset(
            preset_id, num_frames, width=width, height=height
        )
    else:
        available = sorted(
            set(list(_PRESET_JSON_TO_TRAJECTORY.keys()) + list(_BUILTIN_PRESETS.keys()))
        )
        raise KeyError(
            f"Camera preset '{preset_id}' not found. "
            f"Available: {', '.join(available)}"
        )

    # Override prompt suffix with the richer version from camera_presets.json
    # if available
    try:
        from animatediff.data.camera_utils import get_preset as _get_json_preset
        json_preset = _get_json_preset(preset_id)
        traj.prompt_suffix = json_preset["prompt_suffix"]
        traj.negative_suffix = json_preset.get("negative_prompt_suffix", "")
    except (KeyError, ImportError):
        pass  # keep the trajectory's built-in prompt suffix

    return traj


def list_trajectory_presets() -> Dict[str, str]:
    """Return a dict of all available trajectory preset names and descriptions."""
    return {
        name: info["description"]
        for name, info in sorted(_BUILTIN_PRESETS.items())
    }


# ===========================================================================
# Internal math utilities
# ===========================================================================

def _lerp_tuple(
    a: Tuple[float, ...],
    b: Tuple[float, ...],
    t: float,
) -> Tuple[float, ...]:
    """Linear interpolation between two tuples."""
    return tuple(a_i + t * (b_i - a_i) for a_i, b_i in zip(a, b))


def _rotation_to_euler_ypr(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract Euler angles (yaw, pitch, roll) from a 3x3 rotation matrix.

    Uses the YXZ intrinsic rotation convention (matching our pose definition).
    Returns angles in radians.
    """
    # R = Rz @ Rx @ Ry
    # From the matrix elements:
    #   R[1,0] = sin(roll)*cos(pitch)
    #   R[1,1] = cos(roll)*cos(pitch)
    #   R[1,2] = -sin(pitch)  ... wait, let's derive properly.
    #
    # Rz @ Rx @ Ry:
    # = [[cr, -sr, 0], [sr, cr, 0], [0,0,1]] @
    #   [[1,0,0], [0,cp,-sp], [0,sp,cp]] @
    #   [[cy,0,sy], [0,1,0], [-sy,0,cy]]
    #
    # Middle product (Rx @ Ry):
    # = [[cy, 0, sy],
    #    [sp*sy, cp, -sp*cy],
    #    [-cp*sy, sp, cp*cy]]
    #
    # Full (Rz @ above):
    # R[0,0] = cr*cy - sr*sp*sy
    # R[0,1] = -sr*cp
    # R[0,2] = cr*sy + sr*sp*cy
    # R[1,0] = sr*cy + cr*sp*sy
    # R[1,1] = cr*cp
    # R[1,2] = sr*sy - cr*sp*cy
    # R[2,0] = -cp*sy
    # R[2,1] = sp
    # R[2,2] = cp*cy

    # Extract pitch from R[2,1] = sin(pitch)
    sp = np.clip(R[2, 1], -1.0, 1.0)
    pitch = math.asin(sp)

    if abs(sp) > 0.99999:
        # Gimbal lock: pitch near +/-90 degrees
        yaw = math.atan2(-R[0, 2], R[0, 0])
        roll = 0.0
    else:
        cp = math.cos(pitch)
        # yaw from R[2,0] and R[2,2]
        yaw = math.atan2(-R[2, 0], R[2, 2])
        # roll from R[0,1] and R[1,1]
        roll = math.atan2(-R[0, 1], R[1, 1])

    return yaw, pitch, roll


def _slerp_rotation(Ra: np.ndarray, Rb: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two rotation matrices.

    Converts to axis-angle, interpolates, then converts back.

    Args:
        Ra: 3x3 rotation matrix (start).
        Rb: 3x3 rotation matrix (end).
        t:  Interpolation parameter in [0, 1].

    Returns:
        Interpolated 3x3 rotation matrix.
    """
    # Relative rotation: R_rel = Rb @ Ra^T
    R_rel = Rb @ Ra.T

    # Convert to axis-angle via Rodrigues
    angle = math.acos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))

    if abs(angle) < 1e-8:
        return Ra.copy()

    # Axis from skew-symmetric part
    axis = np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1],
    ])
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return Ra.copy()
    axis = axis / norm

    # Interpolated angle
    angle_t = angle * t

    # Rodrigues' rotation formula: R(theta, axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R_t = (np.eye(3)
           + math.sin(angle_t) * K
           + (1.0 - math.cos(angle_t)) * (K @ K))

    return R_t @ Ra
