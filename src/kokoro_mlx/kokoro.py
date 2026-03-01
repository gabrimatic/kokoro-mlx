# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""High-level KokoroTTS interface."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import KokoroConfig
from .generate import SAMPLE_RATE, generate, generate_stream
from .model import KokoroModel
from .phonemize import Phonemizer
from .playback import play, play_stream, save_wav
from .voices import DEFAULT_VOICE, VoiceManager


@dataclass
class TTSResult:
    """Result from a TTS generation call."""

    audio: np.ndarray
    sample_rate: int
    duration: float
    voice: str


class KokoroTTS:
    """High-level Kokoro TTS interface.

    The simplest usage::

        with KokoroTTS.from_pretrained() as tts:
            result = tts.generate("Hello world")
            tts.save("Hello world", "out.wav")

    All public methods are thread-safe via an internal lock.
    """

    SAMPLE_RATE: int = SAMPLE_RATE

    def __init__(
        self,
        model: KokoroModel,
        config: KokoroConfig,
        voice_manager: VoiceManager,
        model_path: str | Path,
    ) -> None:
        self._model = model
        self._config = config
        self._voices = voice_manager
        self._model_path = Path(model_path)
        self._lock = threading.Lock()
        self._phonemizer = Phonemizer(config.vocab)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path = "mlx-community/Kokoro-82M-bf16") -> "KokoroTTS":
        """Load a KokoroTTS instance from a local directory or HuggingFace Hub.

        Args:
            model_id_or_path: Local directory path or HuggingFace repo ID.
                Defaults to ``mlx-community/Kokoro-82M-bf16``.

        Returns:
            A ready-to-use KokoroTTS instance.
        """
        path = Path(model_id_or_path)

        if not path.is_dir():
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(repo_id=str(model_id_or_path))
            path = Path(local_dir)

        config = KokoroConfig.from_pretrained(path)
        model = KokoroModel.from_pretrained(path)
        voice_manager = VoiceManager(path)

        return cls(model=model, config=config, voice_manager=voice_manager, model_path=path)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def generate(
        self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0, sample_rate: int = SAMPLE_RATE,
    ) -> TTSResult:
        """Synthesize *text* and return a TTSResult.

        Thread-safe.

        Args:
            text: Input text to synthesize.
            voice: Voice name (see :meth:`list_voices`).
            speed: Speaking rate multiplier (>1 is faster, <1 is slower).
            sample_rate: Output sample rate (24000 or 48000).
        """
        with self._lock:
            audio = generate(
                text=text,
                model=self._model,
                config=self._config,
                voice_manager=self._voices,
                voice=voice,
                speed=speed,
                phonemizer=self._phonemizer,
                sample_rate=sample_rate,
            )

        duration = len(audio) / sample_rate
        return TTSResult(audio=audio, sample_rate=sample_rate, duration=duration, voice=voice)

    def generate_stream(
        self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0, sample_rate: int = SAMPLE_RATE,
    ) -> Iterator[np.ndarray]:
        """Synthesize *text* and yield audio chunks as they are produced.

        Note: this generator is NOT lock-protected, so avoid concurrent calls
        to the same instance.

        Args:
            text: Input text to synthesize.
            voice: Voice name (see :meth:`list_voices`).
            speed: Speaking rate multiplier.
            sample_rate: Output sample rate (24000 or 48000).
        """
        yield from generate_stream(
            text=text,
            model=self._model,
            config=self._config,
            voice_manager=self._voices,
            voice=voice,
            speed=speed,
            phonemizer=self._phonemizer,
            sample_rate=sample_rate,
        )

    def speak(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        stream: bool = False,
        stop_event=None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """Synthesize and immediately play *text* through the speakers.

        Args:
            text: Input text to synthesize.
            voice: Voice name.
            speed: Speaking rate multiplier.
            stream: When True, generate and play chunk-by-chunk for lower latency.
            stop_event: Optional ``threading.Event``; playback halts when set.
            sample_rate: Output sample rate (24000 or 48000).
        """
        if stream:
            play_stream(
                self.generate_stream(text, voice=voice, speed=speed, sample_rate=sample_rate),
                sample_rate=sample_rate,
                stop_event=stop_event,
            )
        else:
            result = self.generate(text, voice=voice, speed=speed, sample_rate=sample_rate)
            play(result.audio, sample_rate=sample_rate)

    def save(
        self,
        text: str,
        path: str | Path,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> TTSResult:
        """Synthesize *text* and write the audio to a WAV file.

        Args:
            text: Input text to synthesize.
            path: Destination file path.
            voice: Voice name.
            speed: Speaking rate multiplier.
            sample_rate: Output sample rate (24000 or 48000).
        """
        result = self.generate(text, voice=voice, speed=speed, sample_rate=sample_rate)
        save_wav(result.audio, path, sample_rate=sample_rate)
        return result

    def list_voices(self) -> list[str]:
        """Return sorted list of available voice names."""
        return self._voices.list_voices()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release held resources (voice cache, etc.)."""
        self._voices._cache.clear()

    def __enter__(self) -> "KokoroTTS":
        return self

    def __exit__(self, *args) -> None:
        self.close()
