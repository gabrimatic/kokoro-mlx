# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Tests for VoiceManager."""

from pathlib import Path

import pytest

from kokoro_mlx.voices import DEFAULT_VOICE, VoiceManager

_MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16/snapshots/a71e4d38b236d968966a2002c4c895dbd12b1c3c"


@pytest.fixture(scope="module")
def manager() -> VoiceManager:
    return VoiceManager(_MODEL_PATH)


class TestVoiceManager:
    @pytest.mark.slow
    def test_list_voices_count(self, manager):
        voices = manager.list_voices()
        assert len(voices) == 54

    @pytest.mark.slow
    def test_list_voices_includes_default(self, manager):
        assert DEFAULT_VOICE in manager.list_voices()

    @pytest.mark.slow
    def test_load_af_heart_shape(self, manager):
        voice = manager.load_voice("af_heart")
        assert list(voice.shape) == [510, 1, 256]

    @pytest.mark.slow
    def test_load_voice_cached(self, manager):
        v1 = manager.load_voice("af_heart")
        v2 = manager.load_voice("af_heart")
        # Same object returned from cache.
        assert v1 is v2

    @pytest.mark.slow
    def test_get_style_shape(self, manager):
        import mlx.core as mx

        voice = manager.load_voice("af_heart")
        style = manager.get_style(voice, num_tokens=10)
        assert list(style.shape) == [1, 256]

    @pytest.mark.slow
    def test_get_style_clamps_index(self, manager):
        import mlx.core as mx

        voice = manager.load_voice("af_heart")
        # num_tokens larger than voice length should clamp to last entry.
        style_clamped = manager.get_style(voice, num_tokens=9999)
        style_last = manager.get_style(voice, num_tokens=510)
        assert mx.array_equal(style_clamped, style_last).item()

    @pytest.mark.slow
    def test_missing_voice_raises(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load_voice("nonexistent_voice")
