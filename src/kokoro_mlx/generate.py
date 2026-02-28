# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Text-to-audio generation pipeline for Kokoro TTS."""

from __future__ import annotations

import numpy as np

from .config import KokoroConfig
from .model import KokoroModel
from .phonemize import Phonemizer
from .voices import VoiceManager


def generate(
    text: str,
    model: KokoroModel,
    config: KokoroConfig,
    voice_manager: VoiceManager,
    voice: str = "af_heart",
    speed: float = 1.0,
    phonemizer: Phonemizer | None = None,
) -> np.ndarray:
    """Full text-to-audio pipeline. Returns float32 numpy array at 24kHz.

    Args:
        text: Input text to synthesize.
        model: Loaded KokoroModel.
        config: KokoroConfig instance.
        voice_manager: VoiceManager instance.
        voice: Voice name to use for synthesis.
        speed: Speaking rate multiplier (>1 is faster, <1 is slower).
        phonemizer: Optional pre-built Phonemizer to avoid re-initializing.

    Returns:
        Float32 numpy array of audio samples at 24kHz.
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

    return np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)


def generate_stream(
    text: str,
    model: KokoroModel,
    config: KokoroConfig,
    voice_manager: VoiceManager,
    voice: str = "af_heart",
    speed: float = 1.0,
    phonemizer: Phonemizer | None = None,
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

    Yields:
        Float32 numpy arrays, one per sentence chunk.
    """
    if phonemizer is None:
        phonemizer = Phonemizer(config.vocab)

    chunks = phonemizer.phonemize_long(text)
    if not chunks:
        return

    voice_array = voice_manager.load_voice(voice)

    for phonemes, token_ids in chunks:
        style = voice_manager.get_style(voice_array, len(token_ids))
        audio = model.forward(phonemes, style, speed)
        yield np.array(audio.tolist(), dtype=np.float32)
