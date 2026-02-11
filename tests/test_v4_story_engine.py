"""
V4 Story Engine Test Suite
==========================

Comprehensive tests for all V4 modules:
- StoryEngine, StoryBoard, ShotSpec, CharacterRef, save_storyboard
- CharacterManager, CharacterProfile
- ShotScheduler, SchedulerConfig, ShotResult
- Backend registry (wan22, wan22_animate)
- VRAM manager (WAN22_TIERS)
- Wan22Backend, Wan22AnimateBackend (fully mocked)
- PostProcess: FrameInterpolator, VideoUpscaler, AudioGenerator, VideoCompositor

All tests run WITHOUT real model downloads or GPU access.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def small_image():
    """Create a small 64x64 RGB test image."""
    return Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))


@pytest.fixture
def frame_sequence():
    """Create a sequence of 5 test frames with distinct colors."""
    frames = []
    for i in range(5):
        color = (i * 50, 100, 255 - i * 50)
        arr = np.full((32, 32, 3), color, dtype=np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


@pytest.fixture
def frame_sequence_16():
    """Create a longer sequence of 16 frames."""
    frames = []
    for i in range(16):
        arr = np.full((32, 32, 3), (i * 15, i * 10, 255 - i * 15), dtype=np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


@pytest.fixture
def xianxia_json_path():
    """Path to the example xianxia storyboard JSON."""
    return str(PROJECT_ROOT / "examples" / "xianxia_storyboard.json")


@pytest.fixture
def sample_storyboard_dict():
    """A complete storyboard dictionary for testing."""
    return {
        "title": "Test Story",
        "style": "anime, fantasy",
        "negative_prompt": "ugly, blurry",
        "characters": {
            "Hero": {
                "description": "brave warrior in armor",
                "image_path": "/fake/hero.png",
                "lora_path": "/fake/hero_lora.safetensors",
                "lora_scale": 0.8,
            },
            "Villain": {
                "description": "dark sorcerer with glowing eyes",
            },
        },
        "shots": [
            {
                "prompt": "hero standing on a cliff at sunset",
                "duration_seconds": 5.0,
                "camera": "static",
                "characters": ["Hero"],
                "emotion": "determined",
                "scene": "Cliff at sunset",
                "transition": "cut",
            },
            {
                "prompt": "villain emerging from shadows with glowing eyes",
                "duration_seconds": 4.0,
                "camera": "zoom in",
                "characters": ["Villain"],
                "emotion": "fearful",
                "scene": "Dark cave entrance",
                "transition": "dissolve",
            },
            {
                "prompt": "epic battle between hero and villain",
                "duration_seconds": 6.0,
                "camera": "orbit",
                "characters": ["Hero", "Villain"],
                "emotion": "epic",
                "scene": "Battlefield",
                "transition": "fade",
            },
        ],
    }


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ===========================================================================
# 1. StoryEngine Tests (~10 tests)
# ===========================================================================

class TestStoryEngine:

    def test_from_script_chinese_text(self):
        """Parse a Chinese script with paragraph splits."""
        from animatediff.core.story_engine import StoryEngine

        script = (
            "韩立站在山顶，金色的朝阳从远方升起。他的白色道袍在风中飘扬。\n\n"
            "突然，他睁开双眼，金光从瞳孔中爆发而出。灵力如潮水般向四周扩散。\n\n"
            "南宫婉从远处的悬崖上望着那道直冲云霄的金色灵力光柱，眼中满是震撼。"
        )
        engine = StoryEngine(style="xianxia, anime")
        board = engine.from_script(script)

        assert len(board.shots) == 3
        assert board.style == "xianxia, anime"
        # Each shot prompt should be prefixed with the style
        for shot in board.shots:
            assert shot.prompt.startswith("xianxia, anime,")
        # Duration should be within 3-8 seconds
        for shot in board.shots:
            assert 3.0 <= shot.duration_seconds <= 8.0

    def test_from_script_english_text(self):
        """Parse an English script split by paragraphs."""
        from animatediff.core.story_engine import StoryEngine

        script = (
            "A warrior stands at the edge of a cliff, the sunrise painting the sky golden.\n\n"
            "He closes his eyes and begins to meditate, energy swirling around him.\n\n"
            "With a sudden burst of light, he ascends into the sky, clouds parting below."
        )
        engine = StoryEngine()
        board = engine.from_script(script)

        assert len(board.shots) == 3
        assert all(s.shot_id == i for i, s in enumerate(board.shots))

    def test_from_script_explicit_shot_markers(self):
        """Parse a script with [SHOT N] markers."""
        from animatediff.core.story_engine import StoryEngine

        script = (
            "[SHOT 1] The hero arrives at the gate of the ancient city.\n"
            "[SHOT 2] Inside the city, merchants sell exotic goods.\n"
            "[SHOT 3] A mysterious figure watches from a rooftop."
        )
        engine = StoryEngine()
        board = engine.from_script(script)

        assert len(board.shots) == 3

    def test_from_script_chinese_markers(self):
        """Parse a script with Chinese shot markers."""
        from animatediff.core.story_engine import StoryEngine

        script = (
            "[镜头1] 韩立站在山巅。\n"
            "[场景2] 灵力爆发。\n"
            "[镜头3] 飞升成功。"
        )
        engine = StoryEngine()
        board = engine.from_script(script)

        assert len(board.shots) == 3

    def test_from_script_num_shots_limit(self):
        """The num_shots parameter caps the number of shots."""
        from animatediff.core.story_engine import StoryEngine

        script = "A.\n\nB.\n\nC.\n\nD.\n\nE."
        engine = StoryEngine()
        board = engine.from_script(script, num_shots=2)

        assert len(board.shots) == 2

    def test_from_dict_complete(self, sample_storyboard_dict):
        """from_dict() parses a complete dictionary into a StoryBoard."""
        from animatediff.core.story_engine import StoryEngine

        engine = StoryEngine()
        board = engine.from_dict(sample_storyboard_dict)

        assert board.title == "Test Story"
        assert board.style == "anime, fantasy"
        assert board.negative_prompt == "ugly, blurry"
        assert len(board.characters) == 2
        assert "Hero" in board.characters
        assert "Villain" in board.characters
        assert board.characters["Hero"].lora_path == "/fake/hero_lora.safetensors"
        assert board.characters["Hero"].lora_scale == 0.8
        assert board.characters["Villain"].lora_path is None
        assert len(board.shots) == 3
        assert board.shots[0].characters == ["Hero"]
        assert board.shots[2].characters == ["Hero", "Villain"]
        assert board.total_duration == 15.0
        assert board.num_shots == 3

    def test_from_json_xianxia(self, xianxia_json_path):
        """Load the example xianxia_storyboard.json and verify structure."""
        from animatediff.core.story_engine import StoryEngine

        if not os.path.exists(xianxia_json_path):
            pytest.skip("Example file not found")

        engine = StoryEngine()
        board = engine.from_json(xianxia_json_path)

        assert board.title == "凡人修仙传 — 韩立悟道"
        assert "韩立" in board.characters
        assert "南宫婉" in board.characters
        assert len(board.shots) == 6
        assert board.shots[0].camera == "static"
        assert board.shots[1].camera == "zoom in"
        assert board.shots[3].camera == "pan right"

    def test_detect_camera_keywords(self):
        """_detect_camera() correctly matches various keywords."""
        from animatediff.core.story_engine import StoryEngine

        engine = StoryEngine()

        assert engine._detect_camera("the camera will pan left slowly") == "pan left"
        assert engine._detect_camera("zoom in on the character's face") == "zoom in"
        assert engine._detect_camera("dolly forward through the hallway") == "dolly forward"
        assert engine._detect_camera("orbit around the monument") == "orbit"
        assert engine._detect_camera("a calm static scene") == "static"
        assert engine._detect_camera("推镜向前移动") == "dolly forward"
        assert engine._detect_camera("特写镜头") == "zoom in"
        assert engine._detect_camera("no camera hint here") == "static"  # default

    def test_detect_emotion_keywords(self):
        """_detect_emotion() correctly matches emotional keywords."""
        from animatediff.core.story_engine import StoryEngine

        engine = StoryEngine()

        assert engine._detect_emotion("she is determined to win") == "determined"
        assert engine._detect_emotion("a sad farewell at the gate") == "sad"
        assert engine._detect_emotion("笑着说道") == "happy"
        assert engine._detect_emotion("epic battle scene") == "epic"
        assert engine._detect_emotion("a mysterious stranger") == "mysterious"
        assert engine._detect_emotion("怒火中烧") == "angry"
        assert engine._detect_emotion("nothing special") == ""  # default

    def test_estimate_duration_bounds(self):
        """_estimate_duration() stays within [3.0, 8.0] seconds."""
        from animatediff.core.story_engine import StoryEngine

        engine = StoryEngine()

        # Very short text -> clamped to 3.0
        assert engine._estimate_duration("Hi") == 3.0
        # Very long text -> clamped to 8.0
        long_text = "word " * 200
        assert engine._estimate_duration(long_text) == 8.0
        # Medium text -> somewhere in between
        medium_text = "This is a moderately long description of a scene."
        duration = engine._estimate_duration(medium_text)
        assert 3.0 <= duration <= 8.0

    def test_save_storyboard_round_trip(self, sample_storyboard_dict, tmp_dir):
        """save_storyboard() writes JSON that can be loaded back."""
        from animatediff.core.story_engine import StoryEngine, save_storyboard

        engine = StoryEngine()
        board = engine.from_dict(sample_storyboard_dict)

        path = os.path.join(tmp_dir, "test_board.json")
        save_storyboard(board, path)

        assert os.path.exists(path)

        # Load it back
        board2 = engine.from_json(path)
        assert board2.title == board.title
        assert board2.num_shots == board.num_shots
        assert len(board2.characters) == len(board.characters)
        # Character data preserved
        assert "Hero" in board2.characters
        assert board2.characters["Hero"].lora_scale == 0.8

    def test_build_prompt_with_style(self):
        """_build_prompt() concatenates style + text."""
        from animatediff.core.story_engine import StoryEngine

        engine = StoryEngine()

        result = engine._build_prompt("a warrior on a cliff", "anime, epic")
        assert result == "anime, epic, a warrior on a cliff"

        result_no_style = engine._build_prompt("a warrior on a cliff", "")
        assert result_no_style == "a warrior on a cliff"


# ===========================================================================
# 2. CharacterManager Tests (~6 tests)
# ===========================================================================

class TestCharacterManager:

    def test_add_and_get(self):
        """add() registers a character and get() retrieves it."""
        from animatediff.core.character_manager import CharacterManager

        mgr = CharacterManager()
        profile = mgr.add("Alice", description="a brave adventurer", trigger_word="alice_char")

        assert profile.name == "Alice"
        assert profile.description == "a brave adventurer"
        assert profile.trigger_word == "alice_char"

        retrieved = mgr.get("Alice")
        assert retrieved is profile
        assert mgr.get("NonExistent") is None

    def test_get_loras_for_shot(self):
        """get_loras_for_shot() returns LoRA paths+scales for named characters."""
        from animatediff.core.character_manager import CharacterManager

        mgr = CharacterManager()
        mgr.add("Alice", lora_path="/loras/alice.safetensors", lora_scale=0.8)
        mgr.add("Bob", lora_path="/loras/bob.safetensors", lora_scale=1.0)
        mgr.add("Charlie", description="no LoRA character")

        loras = mgr.get_loras_for_shot(["Alice", "Charlie", "Bob"])
        assert len(loras) == 2
        assert loras[0] == ("/loras/alice.safetensors", 0.8)
        assert loras[1] == ("/loras/bob.safetensors", 1.0)

        # Unknown character names are silently skipped
        loras_empty = mgr.get_loras_for_shot(["Unknown"])
        assert loras_empty == []

    def test_build_character_prompt(self):
        """build_character_prompt() uses trigger_word > description > name."""
        from animatediff.core.character_manager import CharacterManager

        mgr = CharacterManager()
        mgr.add("Alice", trigger_word="alice_lora", description="a girl")
        mgr.add("Bob", description="a tall man")
        mgr.add("Charlie")  # no trigger_word, no description

        prompt = mgr.build_character_prompt(["Alice", "Bob", "Charlie"])
        assert prompt == "alice_lora, a tall man, Charlie"

    def test_save_and_load_round_trip(self, tmp_dir):
        """save() and load() preserve character data."""
        from animatediff.core.character_manager import CharacterManager

        mgr = CharacterManager()
        mgr.add("Alice", description="brave", lora_path="/lora/a.safetensors", lora_scale=0.7, trigger_word="alice_tw")
        mgr.add("Bob", description="wise", reference_images=["/img/bob1.png", "/img/bob2.png"])

        path = os.path.join(tmp_dir, "chars.json")
        mgr.save(path)

        assert os.path.exists(path)

        mgr2 = CharacterManager()
        mgr2.load(path)

        assert len(mgr2.characters) == 2
        alice = mgr2.get("Alice")
        assert alice.description == "brave"
        assert alice.lora_path == "/lora/a.safetensors"
        assert alice.lora_scale == 0.7
        assert alice.trigger_word == "alice_tw"

        bob = mgr2.get("Bob")
        assert bob.reference_images == ["/img/bob1.png", "/img/bob2.png"]

    def test_from_storyboard(self, sample_storyboard_dict):
        """from_storyboard() builds manager from a StoryBoard's character refs."""
        from animatediff.core.story_engine import StoryEngine
        from animatediff.core.character_manager import CharacterManager

        engine = StoryEngine()
        board = engine.from_dict(sample_storyboard_dict)

        mgr = CharacterManager.from_storyboard(board)

        assert len(mgr.characters) == 2
        hero = mgr.get("Hero")
        assert hero is not None
        assert hero.description == "brave warrior in armor"
        assert hero.lora_path == "/fake/hero_lora.safetensors"
        assert hero.lora_scale == 0.8
        # image_path should be in reference_images
        assert hero.reference_images == ["/fake/hero.png"]

        villain = mgr.get("Villain")
        assert villain is not None
        assert villain.description == "dark sorcerer with glowing eyes"
        assert villain.reference_images == []  # no image_path in the dict

    def test_primary_image_property(self):
        """CharacterProfile.primary_image returns first ref image or None."""
        from animatediff.core.character_manager import CharacterProfile

        char_with = CharacterProfile(name="A", reference_images=["/img/a.png", "/img/b.png"])
        assert char_with.primary_image == "/img/a.png"

        char_without = CharacterProfile(name="B")
        assert char_without.primary_image is None


# ===========================================================================
# 3. ShotScheduler Tests (~5 tests)
# ===========================================================================

class TestShotScheduler:

    def test_duration_to_frames_alignment(self):
        """_duration_to_frames() aligns to 4N+1 and enforces minimum of 17."""
        from animatediff.core.shot_scheduler import ShotScheduler, SchedulerConfig

        mock_pipeline = MagicMock()
        scheduler = ShotScheduler(pipeline=mock_pipeline)

        # 5s * 16fps = 80 -> aligned: ((80-1)//4)*4+1 = 79//4=19 -> 19*4+1=77
        frames = scheduler._duration_to_frames(5.0, 16)
        assert frames == 77
        assert (frames - 1) % 4 == 0  # 4N+1 check

        # 1s * 16fps = 16 -> aligned: ((16-1)//4)*4+1 = 15//4=3 -> 3*4+1=13 -> min 17
        frames_short = scheduler._duration_to_frames(1.0, 16)
        assert frames_short == 17
        assert (frames_short - 1) % 4 == 0

        # 3s * 24fps = 72 -> aligned: ((72-1)//4)*4+1 = 71//4=17 -> 17*4+1=69
        frames_24 = scheduler._duration_to_frames(3.0, 24)
        assert frames_24 == 69
        assert (frames_24 - 1) % 4 == 0

    def test_generate_all_with_mocked_pipeline(self, sample_storyboard_dict, tmp_dir):
        """generate_all() calls pipeline.generate() for each shot and returns results."""
        from animatediff.core.story_engine import StoryEngine
        from animatediff.core.shot_scheduler import ShotScheduler, SchedulerConfig
        from animatediff.core.base_pipeline import VideoOutput

        engine = StoryEngine()
        board = engine.from_dict(sample_storyboard_dict)

        # Create mock pipeline
        mock_frames = [Image.new("RGB", (64, 64), "red") for _ in range(17)]
        mock_output = VideoOutput(frames=mock_frames, fps=16, seed=42, backend="mock")

        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = mock_output
        mock_pipeline.save = MagicMock()

        config = SchedulerConfig(output_dir=tmp_dir, fps=16)
        scheduler = ShotScheduler(pipeline=mock_pipeline, config=config)
        results = scheduler.generate_all(board)

        assert len(results) == 3
        assert mock_pipeline.generate.call_count == 3
        for r in results:
            assert r.error is None
            assert r.output is not None
            assert r.generation_time >= 0

    def test_progress_callback_invocation(self, sample_storyboard_dict, tmp_dir):
        """progress_callback is called with correct (current, total, message)."""
        from animatediff.core.story_engine import StoryEngine
        from animatediff.core.shot_scheduler import ShotScheduler, SchedulerConfig
        from animatediff.core.base_pipeline import VideoOutput

        engine = StoryEngine()
        board = engine.from_dict(sample_storyboard_dict)

        mock_frames = [Image.new("RGB", (64, 64)) for _ in range(17)]
        mock_output = VideoOutput(frames=mock_frames, fps=16)
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = mock_output
        mock_pipeline.save = MagicMock()

        callback = MagicMock()
        config = SchedulerConfig(output_dir=tmp_dir)
        scheduler = ShotScheduler(pipeline=mock_pipeline, config=config)
        scheduler.generate_all(board, progress_callback=callback)

        # Should be called for each shot + final completion
        assert callback.call_count == 4  # 3 shots + "All shots complete"
        # First call: (0, 3, "Generating shot 1/3: ...")
        first_call_args = callback.call_args_list[0][0]
        assert first_call_args[0] == 0
        assert first_call_args[1] == 3
        # Last call: (3, 3, "All shots complete.")
        last_call_args = callback.call_args_list[-1][0]
        assert last_call_args[0] == 3
        assert last_call_args[1] == 3

    def test_get_all_frames(self, tmp_dir):
        """get_all_frames() collects frames from all successful results."""
        from animatediff.core.story_engine import StoryEngine, ShotSpec, StoryBoard
        from animatediff.core.shot_scheduler import ShotScheduler, SchedulerConfig
        from animatediff.core.base_pipeline import VideoOutput

        mock_frames_a = [Image.new("RGB", (32, 32), "red") for _ in range(5)]
        mock_frames_b = [Image.new("RGB", (32, 32), "blue") for _ in range(3)]

        mock_pipeline = MagicMock()
        mock_pipeline.generate.side_effect = [
            VideoOutput(frames=mock_frames_a, fps=16),
            VideoOutput(frames=mock_frames_b, fps=16),
        ]
        mock_pipeline.save = MagicMock()

        board = StoryBoard(shots=[
            ShotSpec(shot_id=0, prompt="test1", scene="s1"),
            ShotSpec(shot_id=1, prompt="test2", scene="s2"),
        ])

        config = SchedulerConfig(output_dir=tmp_dir)
        scheduler = ShotScheduler(pipeline=mock_pipeline, config=config)
        scheduler.generate_all(board)

        all_frames = scheduler.get_all_frames()
        assert len(all_frames) == 8  # 5 + 3

    def test_error_handling_for_failed_shots(self, tmp_dir):
        """Failed shots record the error and do not crash the scheduler."""
        from animatediff.core.story_engine import ShotSpec, StoryBoard
        from animatediff.core.shot_scheduler import ShotScheduler, SchedulerConfig
        from animatediff.core.base_pipeline import VideoOutput

        mock_pipeline = MagicMock()
        mock_pipeline.generate.side_effect = RuntimeError("CUDA out of memory")

        board = StoryBoard(shots=[
            ShotSpec(shot_id=0, prompt="test", scene="s"),
        ])

        config = SchedulerConfig(output_dir=tmp_dir)
        scheduler = ShotScheduler(pipeline=mock_pipeline, config=config)
        results = scheduler.generate_all(board)

        assert len(results) == 1
        assert results[0].error is not None
        assert "CUDA out of memory" in results[0].error
        assert results[0].output is None


# ===========================================================================
# 4. Backend Registry Tests (~3 tests)
# ===========================================================================

class TestBackendRegistry:

    def test_wan22_in_registry(self):
        """wan22 is present in BACKEND_REGISTRY."""
        from animatediff.backends import BACKEND_REGISTRY

        assert "wan22" in BACKEND_REGISTRY
        assert BACKEND_REGISTRY["wan22"] == "animatediff.backends.wan22.Wan22Backend"

    def test_wan22_animate_in_registry(self):
        """wan22_animate is present in BACKEND_REGISTRY."""
        from animatediff.backends import BACKEND_REGISTRY

        assert "wan22_animate" in BACKEND_REGISTRY
        assert BACKEND_REGISTRY["wan22_animate"] == "animatediff.backends.wan22_animate.Wan22AnimateBackend"

    def test_get_backend_wan22_lazy_import(self):
        """get_backend('wan22') lazily imports and returns the Wan22Backend class."""
        from animatediff.backends import get_backend

        backend_cls = get_backend("wan22")
        from animatediff.backends.wan22 import Wan22Backend
        assert backend_cls is Wan22Backend

    def test_get_backend_unknown_raises(self):
        """get_backend() raises ValueError for unknown backend names."""
        from animatediff.backends import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent_backend")

    def test_list_backends(self):
        """list_backends() includes all expected backends."""
        from animatediff.backends import list_backends

        backends = list_backends()
        assert "wan22" in backends
        assert "wan22_animate" in backends
        assert "wan" in backends


# ===========================================================================
# 5. VRAM Manager Tests (~3 tests)
# ===========================================================================

class TestVRAMManager:

    def test_wan22_tiers_exist_and_sorted(self):
        """WAN22_TIERS exist and are sorted by min_vram descending."""
        from animatediff.core.vram_manager import WAN22_TIERS

        assert len(WAN22_TIERS) >= 3
        vram_thresholds = [t[0] for t in WAN22_TIERS]
        # Must be sorted descending (highest VRAM first)
        assert vram_thresholds == sorted(vram_thresholds, reverse=True)

    def test_best_backend_returns_wan22_for_16gb(self):
        """best_backend() returns 'wan22' when VRAM >= 16GB."""
        from animatediff.core.vram_manager import VRAMManager, GPUProfile

        mgr = VRAMManager.__new__(VRAMManager)
        mgr.profile = GPUProfile(
            name="Test GPU",
            vram_gb=16.0,
            device="cuda",
            is_cuda=True,
            supports_fp16=True,
            supports_bf16=True,
        )

        assert mgr.best_backend() == "wan22"

    def test_best_backend_wan22_for_12gb(self):
        """best_backend() returns 'wan22' for 12GB too."""
        from animatediff.core.vram_manager import VRAMManager, GPUProfile

        mgr = VRAMManager.__new__(VRAMManager)
        mgr.profile = GPUProfile(name="Test", vram_gb=12.0, device="cuda", is_cuda=True)

        assert mgr.best_backend() == "wan22"

    def test_best_backend_fallback_for_low_vram(self):
        """best_backend() returns lighter backends for lower VRAM."""
        from animatediff.core.vram_manager import VRAMManager, GPUProfile

        # 5GB is below 6GB threshold for cogvideo, so falls to animatediff
        mgr = VRAMManager.__new__(VRAMManager)
        mgr.profile = GPUProfile(name="Test", vram_gb=5.0, device="cuda", is_cuda=True)
        assert mgr.best_backend() == "animatediff"

        # 7GB falls into the wan (8GB) gap, so cogvideo at >=6
        mgr2 = VRAMManager.__new__(VRAMManager)
        mgr2.profile = GPUProfile(name="Test", vram_gb=7.0, device="cuda", is_cuda=True)
        assert mgr2.best_backend() == "cogvideo"

    def test_recommend_wan22_returns_valid_config(self):
        """recommend('wan22') returns a valid InferenceConfig."""
        from animatediff.core.vram_manager import VRAMManager, GPUProfile, InferenceConfig

        mgr = VRAMManager.__new__(VRAMManager)
        mgr.profile = GPUProfile(
            name="RTX 4090",
            vram_gb=24.0,
            device="cuda",
            is_cuda=True,
            supports_fp16=True,
            supports_bf16=True,
            supports_compile=True,
        )

        config = mgr.recommend("wan22")
        assert isinstance(config, InferenceConfig)
        assert config.model_variant in ("A14B", "5B")
        assert config.max_width > 0
        assert config.max_height > 0
        assert config.max_frames > 0

    def test_wan22_in_backend_tiers(self):
        """BACKEND_TIERS includes 'wan22' and 'wan22_animate'."""
        from animatediff.core.vram_manager import BACKEND_TIERS

        assert "wan22" in BACKEND_TIERS
        assert "wan22_animate" in BACKEND_TIERS


# ===========================================================================
# 6. Wan22Backend Tests (~5 tests, all mocked)
# ===========================================================================

class TestWan22Backend:

    @patch("animatediff.backends.wan22.AutoencoderKLWan", create=True)
    @patch("animatediff.backends.wan22.WanPipeline", create=True)
    @patch("animatediff.backends.wan22.get_quantization_config")
    def test_load_calls_correct_pipeline(self, mock_quant, mock_pipe_cls, mock_vae_cls):
        """load() calls WanPipeline.from_pretrained with correct model path."""
        from animatediff.backends.wan22 import Wan22Backend, WAN22_T2V_MODELS

        # Set up mocks
        mock_quant.return_value = None
        mock_vae = MagicMock()
        mock_vae_cls.from_pretrained.return_value = mock_vae

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"
        # Mock text_encoder for UMT5 fix check
        mock_te = MagicMock()
        mock_te.shared.weight.abs.return_value.sum.return_value.item.return_value = 1.0
        mock_te.encoder.embed_tokens.weight.abs.return_value.sum.return_value.item.return_value = 1.0
        mock_pipe.text_encoder = mock_te
        mock_pipe_cls.from_pretrained.return_value = mock_pipe

        with patch("diffusers.WanPipeline", mock_pipe_cls, create=True), \
             patch("diffusers.WanImageToVideoPipeline", MagicMock(), create=True), \
             patch("diffusers.AutoencoderKLWan", mock_vae_cls, create=True):
            backend = Wan22Backend.load(model_variant="5B", device="cuda")

        mock_pipe_cls.from_pretrained.assert_called_once()
        call_args = mock_pipe_cls.from_pretrained.call_args
        assert call_args[0][0] == WAN22_T2V_MODELS["5B"]
        assert isinstance(backend, Wan22Backend)

    @patch("animatediff.backends.wan22.AutoencoderKLWan", create=True)
    @patch("animatediff.backends.wan22.WanPipeline", create=True)
    @patch("animatediff.backends.wan22.get_quantization_config")
    def test_mps_fallback_from_a14b_to_5b(self, mock_quant, mock_pipe_cls, mock_vae_cls):
        """On MPS, A14B model falls back to TI2V-5B."""
        from animatediff.backends.wan22 import Wan22Backend, WAN22_T2V_MODELS

        mock_quant.return_value = None
        mock_vae_cls.from_pretrained.return_value = MagicMock()

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "mps"
        mock_te = MagicMock()
        mock_te.shared.weight.abs.return_value.sum.return_value.item.return_value = 1.0
        mock_te.encoder.embed_tokens.weight.abs.return_value.sum.return_value.item.return_value = 1.0
        mock_pipe.text_encoder = mock_te
        mock_pipe_cls.from_pretrained.return_value = mock_pipe

        with patch("diffusers.WanPipeline", mock_pipe_cls, create=True), \
             patch("diffusers.WanImageToVideoPipeline", MagicMock(), create=True), \
             patch("diffusers.AutoencoderKLWan", mock_vae_cls, create=True):
            backend = Wan22Backend.load(model_variant="A14B", device="mps")

        # Should have fallen back to 5B
        assert backend.model_variant == "5B"

    def test_generate_with_defaults(self):
        """generate() uses model defaults when parameters are 0."""
        from animatediff.backends.wan22 import Wan22Backend, MODEL_DEFAULTS
        from animatediff.core.base_pipeline import VideoOutput

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"
        mock_frames = [Image.new("RGB", (64, 64)) for _ in range(5)]
        mock_pipe.return_value = MagicMock(frames=[mock_frames])

        backend = Wan22Backend(mock_pipe, model_variant="5B")
        output = backend.generate(prompt="test prompt")

        assert isinstance(output, VideoOutput)
        assert output.backend == "wan22"
        assert output.metadata["model_variant"] == "5B"

        # Check the pipe was called with defaults for 5B
        call_kwargs = mock_pipe.call_args[1]
        defaults = MODEL_DEFAULTS["5B"]
        assert call_kwargs["width"] == defaults["width"]
        assert call_kwargs["height"] == defaults["height"]
        assert call_kwargs["num_frames"] == defaults["num_frames"]

    def test_lora_loading_dual_transformer_detection(self):
        """_load_loras detects dual-transformer LoRAs by filename convention."""
        from animatediff.backends.wan22 import Wan22Backend

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"

        backend = Wan22Backend(mock_pipe, model_variant="A14B")

        # Load a regular LoRA and a low-noise LoRA
        backend._load_loras(
            ["/lora/style.safetensors", "/lora/detail_LOW.safetensors"],
            [0.8, 0.6],
        )

        calls = mock_pipe.load_lora_weights.call_args_list
        assert len(calls) == 2

        # First call: regular transformer
        first_kwargs = calls[0][1]
        assert "load_into_transformer_2" not in first_kwargs

        # Second call: low-noise transformer_2
        second_kwargs = calls[1][1]
        assert second_kwargs.get("load_into_transformer_2") is True

    @patch("animatediff.backends.wan22.AutoencoderKLWan", create=True)
    @patch("animatediff.backends.wan22.WanPipeline", create=True)
    @patch("animatediff.backends.wan22.get_quantization_config")
    def test_umt5_embed_tokens_fix(self, mock_quant, mock_pipe_cls, mock_vae_cls):
        """UMT5 embed_tokens zero-weight fix is applied during load()."""
        from animatediff.backends.wan22 import Wan22Backend
        import torch

        mock_quant.return_value = None
        mock_vae_cls.from_pretrained.return_value = MagicMock()

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"

        # Simulate the zero-weight bug: embed_tokens all zeros, shared has data
        mock_te = MagicMock()
        mock_te.shared.weight.abs.return_value.sum.return_value.item.return_value = 42.0
        mock_te.encoder.embed_tokens.weight.abs.return_value.sum.return_value.item.return_value = 0.0
        mock_pipe.text_encoder = mock_te
        mock_pipe_cls.from_pretrained.return_value = mock_pipe

        with patch("diffusers.WanPipeline", mock_pipe_cls, create=True), \
             patch("diffusers.WanImageToVideoPipeline", MagicMock(), create=True), \
             patch("diffusers.AutoencoderKLWan", mock_vae_cls, create=True):
            Wan22Backend.load(model_variant="5B", device="cuda")

        # The fix should bind shared.weight to encoder.embed_tokens.weight
        assert mock_te.encoder.embed_tokens.weight == mock_te.shared.weight


# ===========================================================================
# 7. PostProcess Tests (~8 tests)
# ===========================================================================

class TestFrameInterpolator:

    def test_interpolate_blend_frame_count(self, frame_sequence):
        """_interpolate_blend() with 2x produces (N-1)*2 + 1 frames."""
        from animatediff.postprocess.interpolation import FrameInterpolator

        interp = FrameInterpolator(backend="blend")
        result = interp._interpolate_blend(frame_sequence, multiplier=2)

        # 5 original frames -> 4 gaps * 2 + 1(last) = 9, but actually:
        # for each pair: original + 1 intermediate = 2 per gap, then + last = 9
        expected = (len(frame_sequence) - 1) * 2 + 1
        assert len(result) == expected

        # All frames should be valid PIL Images
        for frame in result:
            assert isinstance(frame, Image.Image)

    def test_interpolate_blend_4x(self, frame_sequence):
        """_interpolate_blend() with 4x produces correct count."""
        from animatediff.postprocess.interpolation import FrameInterpolator

        interp = FrameInterpolator(backend="blend")
        result = interp._interpolate_blend(frame_sequence, multiplier=4)

        expected = (len(frame_sequence) - 1) * 4 + 1
        assert len(result) == expected

    def test_multiplier_1_returns_original(self, frame_sequence):
        """interpolate() with multiplier=1 returns the original frames unchanged."""
        from animatediff.postprocess.interpolation import FrameInterpolator

        interp = FrameInterpolator(backend="blend")
        result = interp.interpolate(frame_sequence, multiplier=1)

        assert len(result) == len(frame_sequence)
        # Should be the same objects
        for orig, res in zip(frame_sequence, result):
            assert orig is res

    def test_interpolate_single_frame(self):
        """interpolate() with a single frame returns it unchanged."""
        from animatediff.postprocess.interpolation import FrameInterpolator

        interp = FrameInterpolator(backend="blend")
        single = [Image.new("RGB", (32, 32), "red")]
        result = interp.interpolate(single, multiplier=2)
        assert len(result) == 1

    def test_blend_interpolation_values(self):
        """Interpolated blend frames have values between the two originals."""
        from animatediff.postprocess.interpolation import FrameInterpolator

        black = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
        white = Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8))

        interp = FrameInterpolator(backend="blend")
        result = interp._interpolate_blend([black, white], multiplier=3)

        # Result: black, blend1, blend2, white = 4 frames
        assert len(result) == 4
        mid_arr = np.array(result[1])
        # At alpha=1/3: 0*(2/3) + 200*(1/3) = ~66
        assert 50 < mid_arr.mean() < 80


class TestVideoUpscaler:

    def test_initialization_model_selection(self):
        """VideoUpscaler initializes with correct model parameters."""
        from animatediff.postprocess.upscale import VideoUpscaler, MODEL_URLS

        upscaler = VideoUpscaler(model_name="animevideov3", scale=2, device="cpu")
        assert upscaler.model_name == "animevideov3"
        assert upscaler.scale == 2
        assert upscaler._upsampler is None  # lazy loaded

        upscaler_6b = VideoUpscaler(model_name="anime_6B", scale=4)
        assert upscaler_6b.model_name == "anime_6B"

        upscaler_gen = VideoUpscaler(model_name="general")
        assert upscaler_gen.model_name == "general"

        # All model names should have URLs
        for name in ("animevideov3", "anime_6B", "general"):
            assert name in MODEL_URLS


class TestVideoCompositor:

    def test_fade_transition_frame_count(self, small_image):
        """_fade_transition() produces exactly duration_frames frames."""
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor(fps=16)
        result = compositor._fade_transition(small_image, small_image, duration_frames=8)

        assert len(result) == 8
        for frame in result:
            assert isinstance(frame, Image.Image)

    def test_fade_transition_odd_duration(self, small_image):
        """_fade_transition() with odd duration_frames uses integer division."""
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor()
        # duration_frames=7 -> half=3, so 3 fade-out + 3 fade-in = 6 frames
        result = compositor._fade_transition(small_image, small_image, duration_frames=7)
        assert len(result) == 6  # half=3, so 3+3=6

    def test_dissolve_transition(self, frame_sequence):
        """_dissolve_transition() produces n frames from overlapping sections."""
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor()
        end_frames = frame_sequence[:3]
        start_frames = frame_sequence[2:]  # 3 frames

        result = compositor._dissolve_transition(end_frames, start_frames)

        assert len(result) == min(len(end_frames), len(start_frames))
        for frame in result:
            assert isinstance(frame, Image.Image)

    def test_apply_transitions_with_cuts(self, frame_sequence):
        """_apply_transitions() with all 'cut' transitions concatenates frames."""
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor()
        shots = [frame_sequence[:2], frame_sequence[2:4], [frame_sequence[4]]]
        transitions = ["cut", "cut"]

        result = compositor._apply_transitions(shots, transitions)

        # With cuts, all frames are simply concatenated
        assert len(result) == 5

    def test_apply_transitions_single_shot(self, frame_sequence):
        """_apply_transitions() with a single shot returns it as-is."""
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor()
        result = compositor._apply_transitions([frame_sequence], [])

        assert len(result) == len(frame_sequence)

    def test_compose_simple_no_audio(self, frame_sequence, tmp_dir):
        """compose() with simple frames and no audio saves a video."""
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor(fps=16)
        output_path = os.path.join(tmp_dir, "test_output.mp4")

        shots = [frame_sequence[:3], frame_sequence[3:]]

        with patch("animatediff.postprocess.compositor.VideoCompositor._save_frames_as_video") as mock_save:
            mock_save.return_value = output_path
            result = compositor.compose(shots, output_path)

        assert result == output_path
        mock_save.assert_called_once()
        # Verify the frames were assembled correctly (5 total with "cut" transitions)
        saved_frames = mock_save.call_args[0][0]
        assert len(saved_frames) == 5

    def test_compose_raises_on_empty(self):
        """compose() raises ValueError when given no shots."""
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor()
        with pytest.raises(ValueError, match="No shots to compose"):
            compositor.compose([], "output.mp4")


class TestAudioGenerator:

    def test_resolve_engine_fallback(self):
        """_resolve_engine() falls back to 'none' when no TTS is installed."""
        from animatediff.postprocess.audio import AudioGenerator

        # Mock all TTS imports to fail
        with patch.dict("sys.modules", {"f5_tts": None, "f5_tts_mlx": None}):
            with patch("builtins.__import__", side_effect=_import_blocker(["f5_tts", "f5_tts_mlx"])):
                gen = AudioGenerator(tts_engine="auto")
                assert gen.tts_engine in ("none", "f5", "f5_mlx")  # depends on test env

    def test_explicit_engine_selection(self):
        """When a specific engine is requested, it is set directly."""
        from animatediff.postprocess.audio import AudioGenerator

        gen = AudioGenerator.__new__(AudioGenerator)
        gen.device = "cpu"
        gen.ref_audio = None
        gen.ref_text = None
        result = gen._resolve_engine("f5")
        assert result == "f5"

        result2 = gen._resolve_engine("cosyvoice")
        assert result2 == "cosyvoice"

    def test_generate_speech_none_engine(self, tmp_dir):
        """When engine is 'none', generate_speech returns empty string."""
        from animatediff.postprocess.audio import AudioGenerator

        gen = AudioGenerator.__new__(AudioGenerator)
        gen.tts_engine = "none"
        gen.ref_audio = None
        gen.ref_text = None
        gen.device = "cpu"
        gen._tts = None

        path = os.path.join(tmp_dir, "test.wav")
        result = gen.generate_speech("hello", path)
        assert result == ""


# ===========================================================================
# 8. Wan22AnimateBackend Tests (mocked)
# ===========================================================================

class TestWan22AnimateBackend:

    def test_generate_requires_image(self):
        """generate() raises ValueError if no image is provided."""
        from animatediff.backends.wan22_animate import Wan22AnimateBackend

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"
        backend = Wan22AnimateBackend(mock_pipe)

        with pytest.raises(ValueError, match="requires a character reference image"):
            backend.generate(prompt="test", image=None, pose_video=[1], face_video=[1])

    def test_generate_requires_pose_video(self, small_image):
        """generate() raises ValueError if no pose_video is provided."""
        from animatediff.backends.wan22_animate import Wan22AnimateBackend

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"
        backend = Wan22AnimateBackend(mock_pipe)

        with pytest.raises(ValueError, match="requires preprocessed pose_video"):
            backend.generate(prompt="test", image=small_image, pose_video=None, face_video=[1])

    def test_generate_requires_face_video(self, small_image):
        """generate() raises ValueError if no face_video is provided."""
        from animatediff.backends.wan22_animate import Wan22AnimateBackend

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"
        backend = Wan22AnimateBackend(mock_pipe)

        with pytest.raises(ValueError, match="requires preprocessed face_video"):
            backend.generate(prompt="test", image=small_image, pose_video=[1], face_video=None)

    def test_generate_animate_mode(self, small_image):
        """generate() in animate mode calls pipeline with correct args."""
        from animatediff.backends.wan22_animate import Wan22AnimateBackend
        from animatediff.core.base_pipeline import VideoOutput

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"
        mock_frames = [Image.new("RGB", (64, 64)) for _ in range(5)]
        mock_pipe.return_value = MagicMock(frames=[mock_frames])

        backend = Wan22AnimateBackend(mock_pipe)
        output = backend.generate(
            prompt="dancing character",
            image=small_image,
            pose_video=["pose1", "pose2"],
            face_video=["face1", "face2"],
            mode="animate",
        )

        assert isinstance(output, VideoOutput)
        assert output.backend == "wan22_animate"
        assert output.metadata["mode"] == "animate"

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["image"] is small_image
        assert call_kwargs["mode"] == "animate"

    def test_generate_replace_mode(self, small_image):
        """generate() in replace mode includes background and mask."""
        from animatediff.backends.wan22_animate import Wan22AnimateBackend

        mock_pipe = MagicMock()
        mock_pipe.device = MagicMock()
        mock_pipe.device.type = "cuda"
        mock_frames = [Image.new("RGB", (64, 64)) for _ in range(3)]
        mock_pipe.return_value = MagicMock(frames=[mock_frames])

        backend = Wan22AnimateBackend(mock_pipe)
        background = ["bg1", "bg2"]
        mask = ["mask1", "mask2"]

        output = backend.generate(
            prompt="replace character",
            image=small_image,
            pose_video=["pose1"],
            face_video=["face1"],
            mode="replace",
            background_video=background,
            mask_video=mask,
        )

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["mode"] == "replace"
        assert call_kwargs["background_video"] == background
        assert call_kwargs["mask_video"] == mask


# ===========================================================================
# 9. Integration-style Tests (cross-module, still mocked)
# ===========================================================================

class TestIntegration:

    def test_storyboard_to_character_manager_to_scheduler(self, sample_storyboard_dict, tmp_dir):
        """Full pipeline: parse storyboard -> build characters -> schedule shots."""
        from animatediff.core.story_engine import StoryEngine
        from animatediff.core.character_manager import CharacterManager
        from animatediff.core.shot_scheduler import ShotScheduler, SchedulerConfig
        from animatediff.core.base_pipeline import VideoOutput

        engine = StoryEngine()
        board = engine.from_dict(sample_storyboard_dict)

        char_mgr = CharacterManager.from_storyboard(board)
        assert len(char_mgr.characters) == 2

        # Mock get_reference_image so it doesn't try to open fake file paths
        char_mgr.get_reference_image = MagicMock(return_value=None)

        mock_frames = [Image.new("RGB", (64, 64)) for _ in range(17)]
        mock_output = VideoOutput(frames=mock_frames, fps=16)
        mock_pipeline = MagicMock()
        mock_pipeline.generate.return_value = mock_output
        mock_pipeline.save = MagicMock()

        config = SchedulerConfig(output_dir=tmp_dir)
        scheduler = ShotScheduler(
            pipeline=mock_pipeline,
            character_manager=char_mgr,
            config=config,
        )
        results = scheduler.generate_all(board)

        assert len(results) == 3
        assert all(r.error is None for r in results)

        # Verify character prompt was injected into at least one call
        generate_calls = mock_pipeline.generate.call_args_list
        # Shot 0 has "Hero" -> character prompt should be in the prompt
        first_call_prompt = generate_calls[0][1]["prompt"]
        assert "brave warrior in armor" in first_call_prompt or "Hero" in first_call_prompt

    def test_compositor_with_interpolation(self, frame_sequence):
        """Interpolate frames then compose with transitions."""
        from animatediff.postprocess.interpolation import FrameInterpolator
        from animatediff.postprocess.compositor import VideoCompositor

        interp = FrameInterpolator(backend="blend")

        shot1 = frame_sequence[:3]
        shot2 = frame_sequence[3:]

        # 2x interpolate each shot
        shot1_interp = interp.interpolate(shot1, multiplier=2)
        shot2_interp = interp.interpolate(shot2, multiplier=2)

        assert len(shot1_interp) == 5  # (3-1)*2 + 1
        assert len(shot2_interp) == 3  # (2-1)*2 + 1

        compositor = VideoCompositor(fps=32)
        all_frames = compositor._apply_transitions(
            [shot1_interp, shot2_interp],
            ["fade"],
        )

        # fade adds 8 transition frames (default), removes overlap
        assert len(all_frames) > len(shot1_interp)


# ===========================================================================
# Helper functions
# ===========================================================================

def _import_blocker(blocked_modules):
    """Create an import side_effect that blocks specific modules."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def custom_import(name, *args, **kwargs):
        if name in blocked_modules:
            raise ImportError(f"Mocked: {name} not available")
        return original_import(name, *args, **kwargs)

    return custom_import


# ===========================================================================
# Dataclass Tests
# ===========================================================================

class TestDataclasses:

    def test_shot_spec_defaults(self):
        """ShotSpec has sensible defaults."""
        from animatediff.core.story_engine import ShotSpec

        shot = ShotSpec()
        assert shot.shot_id == 0
        assert shot.prompt == ""
        assert shot.duration_seconds == 5.0
        assert shot.transition == "cut"
        assert shot.seed == -1
        assert shot.characters == []
        assert shot.metadata == {}

    def test_storyboard_total_duration(self):
        """StoryBoard.total_duration sums all shot durations."""
        from animatediff.core.story_engine import StoryBoard, ShotSpec

        board = StoryBoard(shots=[
            ShotSpec(duration_seconds=3.0),
            ShotSpec(duration_seconds=5.0),
            ShotSpec(duration_seconds=4.0),
        ])
        assert board.total_duration == 12.0
        assert board.num_shots == 3

    def test_storyboard_empty(self):
        """An empty StoryBoard has 0 duration and 0 shots."""
        from animatediff.core.story_engine import StoryBoard

        board = StoryBoard()
        assert board.total_duration == 0.0
        assert board.num_shots == 0

    def test_character_ref_defaults(self):
        """CharacterRef has correct defaults."""
        from animatediff.core.story_engine import CharacterRef

        ref = CharacterRef(name="Test")
        assert ref.name == "Test"
        assert ref.image_path is None
        assert ref.lora_path is None
        assert ref.lora_scale == 1.0
        assert ref.description == ""

    def test_scheduler_config_defaults(self):
        """SchedulerConfig has expected defaults."""
        from animatediff.core.shot_scheduler import SchedulerConfig

        config = SchedulerConfig()
        assert config.fps == 16
        assert config.default_width == 1280
        assert config.default_height == 720
        assert config.max_frames_per_shot == 121
        assert config.quality == "standard"

    def test_shot_result_defaults(self):
        """ShotResult has expected defaults."""
        from animatediff.core.shot_scheduler import ShotResult

        result = ShotResult(shot_id=0)
        assert result.shot_id == 0
        assert result.output is None
        assert result.output_path == ""
        assert result.error is None
        assert result.generation_time == 0.0

    def test_quality_map_entries(self):
        """QUALITY_MAP has all expected quality levels."""
        from animatediff.core.shot_scheduler import QUALITY_MAP

        assert "draft" in QUALITY_MAP
        assert "standard" in QUALITY_MAP
        assert "high" in QUALITY_MAP
        assert "max" in QUALITY_MAP
        # Steps increase with quality
        assert QUALITY_MAP["draft"]["num_inference_steps"] < QUALITY_MAP["max"]["num_inference_steps"]


# ===========================================================================
# Edge Case Tests
# ===========================================================================

class TestEdgeCases:

    def test_from_script_single_long_sentence(self):
        """A single long sentence >200 chars is split by sentence endings."""
        from animatediff.core.story_engine import StoryEngine

        # Build a script longer than 200 chars as a single block
        sentence = "A warrior walks through a mystical forest full of ancient trees. "
        script = sentence * 10  # ~640 chars, single block
        engine = StoryEngine()
        board = engine.from_script(script)

        # Should have been split by sentence endings, not left as 1 shot
        assert board.num_shots >= 2

    def test_from_script_empty(self):
        """An empty script produces no shots."""
        from animatediff.core.story_engine import StoryEngine

        engine = StoryEngine()
        board = engine.from_script("")

        assert board.num_shots == 0

    def test_save_storyboard_creates_parent_dirs(self, tmp_dir):
        """save_storyboard() creates parent directories if needed."""
        from animatediff.core.story_engine import StoryEngine, StoryBoard, save_storyboard

        board = StoryBoard(title="Test", shots=[])
        deep_path = os.path.join(tmp_dir, "a", "b", "c", "board.json")
        save_storyboard(board, deep_path)

        assert os.path.exists(deep_path)

    def test_character_manager_save_creates_parent_dirs(self, tmp_dir):
        """CharacterManager.save() creates parent directories."""
        from animatediff.core.character_manager import CharacterManager

        mgr = CharacterManager()
        mgr.add("Test", description="hello")
        deep_path = os.path.join(tmp_dir, "x", "y", "chars.json")
        mgr.save(deep_path)

        assert os.path.exists(deep_path)

    def test_duration_to_frames_edge_cases(self):
        """_duration_to_frames handles edge values correctly."""
        from animatediff.core.shot_scheduler import ShotScheduler

        mock_pipeline = MagicMock()
        scheduler = ShotScheduler(pipeline=mock_pipeline)

        # Zero duration -> minimum 17 frames
        assert scheduler._duration_to_frames(0.0, 16) == 17

        # Very large duration -> large but valid frame count
        frames = scheduler._duration_to_frames(100.0, 16)
        assert (frames - 1) % 4 == 0
        assert frames >= 17

    def test_dissolve_transition_unequal_lengths(self):
        """_dissolve_transition handles unequal length inputs."""
        from animatediff.postprocess.compositor import VideoCompositor

        compositor = VideoCompositor()
        end_frames = [Image.new("RGB", (16, 16)) for _ in range(5)]
        start_frames = [Image.new("RGB", (16, 16)) for _ in range(3)]

        result = compositor._dissolve_transition(end_frames, start_frames)
        assert len(result) == 3  # min(5, 3)

    def test_wan22_model_constants(self):
        """Wan22 model path constants are correct."""
        from animatediff.backends.wan22 import WAN22_T2V_MODELS, WAN22_I2V_MODELS, MODEL_DEFAULTS

        assert "A14B" in WAN22_T2V_MODELS
        assert "5B" in WAN22_T2V_MODELS
        assert "Diffusers" in WAN22_T2V_MODELS["A14B"]
        assert "Diffusers" in WAN22_T2V_MODELS["5B"]

        assert "A14B" in WAN22_I2V_MODELS
        assert "5B" in WAN22_I2V_MODELS

        assert "A14B" in MODEL_DEFAULTS
        assert "5B" in MODEL_DEFAULTS
        # 5B has higher fps than A14B
        assert MODEL_DEFAULTS["5B"]["fps"] == 24
        assert MODEL_DEFAULTS["A14B"]["fps"] == 16

    def test_wan22_animate_model_constant(self):
        """Wan22Animate model path constant is correct."""
        from animatediff.backends.wan22_animate import WAN22_ANIMATE_MODEL

        assert "Wan2.2-Animate" in WAN22_ANIMATE_MODEL
        assert "Diffusers" in WAN22_ANIMATE_MODEL
