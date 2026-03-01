# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Text-to-audio generation pipeline for Kokoro TTS."""

from __future__ import annotations

import numpy as np

from .config import KokoroConfig
from .model import KokoroModel
from .phonemize import Phonemizer
from .voices import VoiceManager

SAMPLE_RATE = 24000


def _resample_2x(audio: np.ndarray) -> np.ndarray:
    """Upsample audio by exactly 2x using FFT zero-padding.

    For a real signal of length N, the rfft has N//2+1 bins.  Padding the
    spectrum to 2x length and taking the irfft produces a perfectly
    bandlimited 2x-upsampled signal.  Numpy-only, no extra dependencies.
    """
    n = len(audio)
    spectrum = np.fft.rfft(audio)
    out_len = n * 2
    padded = np.zeros(out_len // 2 + 1, dtype=spectrum.dtype)
    padded[: len(spectrum)] = spectrum
    return np.fft.irfft(padded, n=out_len).astype(np.float32) * 2.0


def generate(
    text: str,
    model: KokoroModel,
    config: KokoroConfig,
    voice_manager: VoiceManager,
    voice: str = "af_heart",
    speed: float = 1.0,
    phonemizer: Phonemizer | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Full text-to-audio pipeline.

    Args:
        text: Input text to synthesize.
        model: Loaded KokoroModel.
        config: KokoroConfig instance.
        voice_manager: VoiceManager instance.
        voice: Voice name to use for synthesis.
        speed: Speaking rate multiplier (>1 is faster, <1 is slower).
        phonemizer: Optional pre-built Phonemizer to avoid re-initializing.
        sample_rate: Output sample rate. 24000 (native) or 48000 (2x upsampled).

    Returns:
        Float32 numpy array of audio samples at the requested sample rate.
    """
    if phonemizer is None:
        phonemizer = Phonemizer(config.vocab)

    chunks = phonemizer.phonemize_long(text)
    if not chunks:
        return np.array([], dtype=np.float32)

    voice_array = voice_manager.load_voice(voice)

    audio_chunks = []
    for phonemes, token_ids in chunks:
        style = voice_manager.get_style(voice_array, len(token_ids))
        audio = model.forward(phonemes, style, speed)
        audio_chunks.append(np.array(audio.tolist(), dtype=np.float32))

    result = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)

    if sample_rate == 48000 and len(result) > 0:
        result = _resample_2x(result)

    return result


def generate_stream(
    text: str,
    model: KokoroModel,
    config: KokoroConfig,
    voice_manager: VoiceManager,
    voice: str = "af_heart",
    speed: float = 1.0,
    phonemizer: Phonemizer | None = None,
    sample_rate: int = SAMPLE_RATE,
):
    """Generate audio in chunks as they are produced.

    Yields float32 numpy arrays, one per phoneme chunk. Suitable for
    low-latency streaming playback.

    Args:
        text: Input text to synthesize.
        model: Loaded KokoroModel.
        config: KokoroConfig instance.
        voice_manager: VoiceManager instance.
        voice: Voice name to use for synthesis.
        speed: Speaking rate multiplier.
        phonemizer: Optional pre-built Phonemizer to avoid re-initializing.
        sample_rate: Output sample rate. 24000 (native) or 48000 (2x upsampled).

    Yields:
        Float32 numpy arrays, one per sentence chunk.
    """
    if phonemizer is None:
        phonemizer = Phonemizer(config.vocab)

    chunks = phonemizer.phonemize_long(text)
    if not chunks:
        return

    voice_array = voice_manager.load_voice(voice)
    upsample = sample_rate == 48000

    for phonemes, token_ids in chunks:
        style = voice_manager.get_style(voice_array, len(token_ids))
        audio = model.forward(phonemes, style, speed)
        chunk = np.array(audio.tolist(), dtype=np.float32)
        if upsample and len(chunk) > 0:
            chunk = _resample_2x(chunk)
        yield chunk
