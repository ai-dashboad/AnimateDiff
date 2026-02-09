"""Test motion module components (shapes, forward pass)."""
import pytest
import torch

from animatediff.models.motion_module import (
    PositionalEncoding,
    TemporalTransformer3DModel,
    VanillaTemporalModule,
)


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=320, max_len=24)
        x = torch.randn(2, 16, 320)
        out = pe(x)
        assert out.shape == (2, 16, 320)

    def test_max_len_respected(self):
        pe = PositionalEncoding(d_model=64, max_len=8)
        x = torch.randn(1, 8, 64)
        out = pe(x)
        assert out.shape == (1, 8, 64)

    def test_shorter_sequence(self):
        pe = PositionalEncoding(d_model=64, max_len=24)
        x = torch.randn(1, 4, 64)
        out = pe(x)
        assert out.shape == (1, 4, 64)


class TestTemporalTransformer3DModel:
    def test_forward_shape(self):
        model = TemporalTransformer3DModel(
            in_channels=320,
            num_attention_heads=8,
            attention_head_dim=40,
            num_layers=1,
        )
        # input: (batch, channels, frames, height, width)
        x = torch.randn(1, 320, 4, 8, 8)
        out = model(x)
        assert out.shape == x.shape

    def test_different_video_lengths(self):
        model = TemporalTransformer3DModel(
            in_channels=320,
            num_attention_heads=8,
            attention_head_dim=40,
            num_layers=1,
        )
        for num_frames in [2, 4, 8, 16]:
            x = torch.randn(1, 320, num_frames, 4, 4)
            out = model(x)
            assert out.shape == x.shape, f"Failed for {num_frames} frames"


class TestVanillaTemporalModule:
    def test_forward_shape(self):
        module = VanillaTemporalModule(
            in_channels=320,
            num_attention_heads=8,
            num_transformer_block=1,
            temporal_position_encoding=True,
            temporal_position_encoding_max_len=24,
        )
        x = torch.randn(1, 320, 4, 8, 8)
        out = module(x, temb=None, encoder_hidden_states=None)
        assert out.shape == x.shape
