# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Audio playback and saving utilities for Kokoro TTS."""

from __future__ import annotations

import queue
import threading
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
    """Gapless streaming playback from a generator.

    Uses a single persistent ``OutputStream`` for the entire playback session.
    A background thread drains the generator into a queue so audio generation
    and playback overlap, eliminating inter-chunk silence.

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

    audio_q: queue.Queue[np.ndarray | None] = queue.Queue()
    gen_error: list[Exception | None] = [None]

    def _feed() -> None:
        try:
            for chunk in audio_generator:
                if stop_event and stop_event.is_set():
                    return
                chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()
                if len(chunk) > 0:
                    audio_q.put(chunk)
        except Exception as exc:
            gen_error[0] = exc
        finally:
            audio_q.put(None)

    feed_thread = threading.Thread(target=_feed, daemon=True)
    feed_thread.start()

    write_size = sample_rate // 10  # 100 ms pieces for responsiveness

    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while True:
            if stop_event and stop_event.is_set():
                break
            try:
                chunk = audio_q.get(timeout=0.05)
            except queue.Empty:
                continue
            if chunk is None:
                break

            for i in range(0, len(chunk), write_size):
                if stop_event and stop_event.is_set():
                    break
                piece = chunk[i : i + write_size]
                stream.write(piece.reshape(-1, 1))

    feed_thread.join(timeout=2.0)

    if gen_error[0] is not None:
        raise gen_error[0]


def save_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Save audio to a WAV file.

    Args:
        audio: Float32 numpy array of audio samples.
        path: Destination file path.
        sample_rate: Sample rate in Hz.
    """
    import soundfile as sf

    sf.write(str(path), audio, sample_rate)
