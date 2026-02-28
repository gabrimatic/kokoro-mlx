# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Tests for the KokoroTTS high-level interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from kokoro_mlx import KokoroTTS, TTSResult

_MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16/snapshots/a71e4d38b236d968966a2002c4c895dbd12b1c3c"


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tts():
    return KokoroTTS.from_pretrained(_MODEL_PATH)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestKokoroTTS:
    def test_from_pretrained_loads(self, tts):
        assert tts is not None
        assert isinstance(tts, KokoroTTS)

    def test_generate_returns_tts_result(self, tts):
        result = tts.generate("Hello, world.")
        assert isinstance(result, TTSResult)

    def test_tts_result_fields(self, tts):
        result = tts.generate("Testing the result fields.")
        assert isinstance(result.audio, np.ndarray)
        assert result.audio.dtype == np.float32
        assert result.sample_rate == 24000
        assert result.duration > 0.0
        assert result.voice == "af_heart"

    def test_duration_matches_audio_length(self, tts):
        result = tts.generate("Duration check.")
        expected = len(result.audio) / result.sample_rate
        assert abs(result.duration - expected) < 1e-6

    def test_list_voices_nonempty(self, tts):
        voices = tts.list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0

    def test_list_voices_includes_af_heart(self, tts):
        assert "af_heart" in tts.list_voices()

    def test_speed_affects_duration(self, tts):
        text = "The quick brown fox jumps over the lazy dog."
        slow = tts.generate(text, speed=0.5)
        fast = tts.generate(text, speed=2.0)
        # Slower speech produces longer audio; fast should be roughly half.
        assert slow.duration > fast.duration

    def test_speed_ratio_roughly_correct(self, tts):
        text = "The quick brown fox jumps over the lazy dog."
        slow = tts.generate(text, speed=0.5)
        fast = tts.generate(text, speed=2.0)
        ratio = slow.duration / fast.duration
        # 4x ratio expected (0.5 vs 2.0); allow a wide tolerance for model variance.
        assert ratio > 2.0

    def test_bad_voice_raises(self, tts):
        with pytest.raises(FileNotFoundError):
            tts.generate("This should fail.", voice="nonexistent_voice_xyz")

    def test_empty_text_returns_empty_audio(self, tts):
        result = tts.generate("")
        assert result.audio.shape[0] == 0
        assert result.duration == 0.0

    def test_whitespace_only_returns_empty_audio(self, tts):
        result = tts.generate("   ")
        assert result.audio.shape[0] == 0

    def test_generate_stream_yields_chunks(self, tts):
        chunks = list(tts.generate_stream("Hello. World."))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32

    def test_save_creates_wav(self, tts):
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            result = tts.save("Save to disk.", path)

            assert path.exists()
            assert path.stat().st_size > 0

            data, sr = sf.read(str(path))
            assert sr == 24000
            assert isinstance(result, TTSResult)

    def test_context_manager(self):
        with KokoroTTS.from_pretrained(_MODEL_PATH) as tts2:
            result = tts2.generate("Context manager test.")
            assert isinstance(result, TTSResult)
        # After __exit__, voice cache should be cleared.
        assert len(tts2._voices._cache) == 0

    def test_voice_parameter_accepted(self, tts):
        voices = tts.list_voices()
        # Pick a voice other than the default to verify the parameter is forwarded.
        alt_voice = next((v for v in voices if v != "af_heart"), None)
        if alt_voice is None:
            pytest.skip("Only one voice available")
        result = tts.generate("Alternative voice.", voice=alt_voice)
        assert result.voice == alt_voice
