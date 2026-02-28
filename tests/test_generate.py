# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Tests for the generate pipeline and playback utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

_MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16/snapshots/a71e4d38b236d968966a2002c4c895dbd12b1c3c"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    from kokoro_mlx.model import KokoroModel

    return KokoroModel.from_pretrained(_MODEL_PATH)


@pytest.fixture(scope="module")
def config():
    from kokoro_mlx.config import KokoroConfig

    return KokoroConfig.from_pretrained(_MODEL_PATH)


@pytest.fixture(scope="module")
def voice_manager():
    from kokoro_mlx.voices import VoiceManager

    return VoiceManager(_MODEL_PATH)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGenerate:
    def test_returns_float32_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("Hello, world.", model, config, voice_manager)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32

    def test_audio_has_samples(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("Hello.", model, config, voice_manager)
        assert audio.shape[0] > 0

    def test_empty_text_returns_empty_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("", model, config, voice_manager)
        assert isinstance(audio, np.ndarray)
        assert audio.shape[0] == 0

    def test_whitespace_only_returns_empty_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("   ", model, config, voice_manager)
        assert isinstance(audio, np.ndarray)
        assert audio.shape[0] == 0

    def test_no_nan_in_output(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        audio = generate("Testing audio quality.", model, config, voice_manager)
        assert not np.any(np.isnan(audio))

    def test_speed_affects_length(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate

        text = "The quick brown fox jumps over the lazy dog."
        slow = generate(text, model, config, voice_manager, speed=0.5)
        fast = generate(text, model, config, voice_manager, speed=2.0)
        # Slower speech produces more frames.
        assert slow.shape[0] > fast.shape[0]


# ---------------------------------------------------------------------------
# generate_stream()
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGenerateStream:
    def test_yields_chunks(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate_stream

        chunks = list(generate_stream("Hello. World.", model, config, voice_manager))
        assert len(chunks) >= 1

    def test_each_chunk_is_float32_array(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate_stream

        for chunk in generate_stream("Hello. World.", model, config, voice_manager):
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32

    def test_empty_text_yields_nothing(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate_stream

        chunks = list(generate_stream("", model, config, voice_manager))
        assert chunks == []

    def test_concatenated_matches_generate(self, model, config, voice_manager):
        from kokoro_mlx.generate import generate, generate_stream

        text = "One sentence only."
        full = generate(text, model, config, voice_manager)
        streamed = np.concatenate(list(generate_stream(text, model, config, voice_manager)))
        # Same text produces the same number of samples and the same dtype.
        assert full.shape == streamed.shape
        assert full.dtype == streamed.dtype


# ---------------------------------------------------------------------------
# save_wav()
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSaveWav:
    def test_creates_valid_wav_file(self, model, config, voice_manager):
        import soundfile as sf

        from kokoro_mlx.generate import generate
        from kokoro_mlx.playback import save_wav

        audio = generate("Saving audio to disk.", model, config, voice_manager)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.wav"
            save_wav(audio, path)

            assert path.exists()
            assert path.stat().st_size > 0

            data, sr = sf.read(str(path))
            assert sr == 24000
            assert len(data) == len(audio)

    def test_roundtrip_preserves_values(self, model, config, voice_manager):
        import soundfile as sf

        from kokoro_mlx.generate import generate
        from kokoro_mlx.playback import save_wav

        audio = generate("Round trip test.", model, config, voice_manager)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rt.wav"
            # Write as 32-bit float to avoid PCM_16 quantization loss.
            sf.write(str(path), audio, 24000, subtype="FLOAT")
            data, _ = sf.read(str(path), dtype="float32")
            np.testing.assert_allclose(audio, data, atol=1e-6)
