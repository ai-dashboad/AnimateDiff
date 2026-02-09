"""Test pipeline utilities and decode_latents."""
import pytest
import torch
import numpy as np
import tempfile
import os

from animatediff.utils.util import save_videos_grid


class TestSaveVideosGrid:
    def test_save_gif(self, tmp_path):
        # Create fake video tensor: (batch, channels, frames, height, width)
        video = torch.rand(1, 3, 4, 64, 64)
        path = str(tmp_path / "test.gif")
        save_videos_grid(video, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_save_mp4(self, tmp_path):
        video = torch.rand(1, 3, 4, 64, 64)
        path = str(tmp_path / "test.mp4")
        save_videos_grid(video, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_multiple_videos(self, tmp_path):
        video = torch.rand(4, 3, 4, 64, 64)
        path = str(tmp_path / "grid.gif")
        save_videos_grid(video, path, n_rows=2)
        assert os.path.exists(path)
