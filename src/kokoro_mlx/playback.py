# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Audio playback and saving utilities for Kokoro TTS."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np

SAMPLE_RATE = 24000


def play(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Blocking audio playback via sounddevice.

    Args:
        audio: Float32 numpy array of audio samples.
        sample_rate: Sample rate in Hz.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for playback. Install it with: pip install sounddevice"
        ) from exc

    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def play_stream(
    audio_generator: Iterator[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    stop_event=None,
) -> None:
    """Streaming playback from a generator with optional stop support.

    Plays each audio chunk as it arrives from the generator. If *stop_event*
    is set, playback stops at the next chunk boundary.

    Args:
        audio_generator: Iterator yielding float32 numpy arrays.
        sample_rate: Sample rate in Hz.
        stop_event: Optional ``threading.Event``; playback halts when set.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError(
            "sounddevice is required for playback. Install it with: pip install sounddevice"
        ) from exc

    for chunk in audio_generator:
        if stop_event and stop_event.is_set():
            sd.stop()
            return

        sd.play(chunk, samplerate=sample_rate)
        duration = len(chunk) / sample_rate
        deadline = time.monotonic() + duration + 0.1

        while time.monotonic() < deadline:
            if stop_event and stop_event.is_set():
                sd.stop()
                return
            time.sleep(0.04)

        sd.wait()


def save_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Save audio to a WAV file.

    Args:
        audio: Float32 numpy array of audio samples.
        path: Destination file path.
        sample_rate: Sample rate in Hz.
    """
    import soundfile as sf

    sf.write(str(path), audio, sample_rate)
