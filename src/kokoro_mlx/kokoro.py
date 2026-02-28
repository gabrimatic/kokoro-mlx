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
from .generate import generate, generate_stream
from .model import KokoroModel
from .phonemize import Phonemizer
from .playback import SAMPLE_RATE, play, play_stream, save_wav
from .voices import DEFAULT_VOICE, VoiceManager


@dataclass
class TTSResult:
    """Result from a TTS generation call."""

    audio: np.ndarray   # float32 at 24kHz
    sample_rate: int    # always 24000
    duration: float     # seconds
    voice: str          # voice name used


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
        # Cache the Phonemizer — loading spacy is expensive.
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
            # Download from HuggingFace Hub and resolve to local snapshot directory.
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

    def generate(self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0) -> TTSResult:
        """Synthesize *text* and return a TTSResult.

        Thread-safe.

        Args:
            text: Input text to synthesize.
            voice: Voice name (see :meth:`list_voices`).
            speed: Speaking rate multiplier (>1 is faster, <1 is slower).

        Returns:
            A TTSResult with the audio array and metadata.
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
            )

        duration = len(audio) / self.SAMPLE_RATE
        return TTSResult(audio=audio, sample_rate=self.SAMPLE_RATE, duration=duration, voice=voice)

    def generate_stream(
        self, text: str, voice: str = DEFAULT_VOICE, speed: float = 1.0
    ) -> Iterator[np.ndarray]:
        """Synthesize *text* and yield audio chunks as they are produced.

        Note: this generator is NOT lock-protected, so avoid concurrent calls
        to the same instance.

        Args:
            text: Input text to synthesize.
            voice: Voice name (see :meth:`list_voices`).
            speed: Speaking rate multiplier.

        Yields:
            Float32 numpy arrays, one per sentence chunk.
        """
        yield from generate_stream(
            text=text,
            model=self._model,
            config=self._config,
            voice_manager=self._voices,
            voice=voice,
            speed=speed,
            phonemizer=self._phonemizer,
        )

    def speak(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        stream: bool = False,
        stop_event=None,
    ) -> None:
        """Synthesize and immediately play *text* through the speakers.

        Args:
            text: Input text to synthesize.
            voice: Voice name.
            speed: Speaking rate multiplier.
            stream: When True, generate and play chunk-by-chunk for lower latency.
            stop_event: Optional ``threading.Event``; playback halts when set.
        """
        if stream:
            play_stream(
                self.generate_stream(text, voice=voice, speed=speed),
                sample_rate=self.SAMPLE_RATE,
                stop_event=stop_event,
            )
        else:
            result = self.generate(text, voice=voice, speed=speed)
            play(result.audio, sample_rate=self.SAMPLE_RATE)

    def save(
        self,
        text: str,
        path: str | Path,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
    ) -> TTSResult:
        """Synthesize *text* and write the audio to a WAV file.

        Args:
            text: Input text to synthesize.
            path: Destination file path.
            voice: Voice name.
            speed: Speaking rate multiplier.

        Returns:
            The TTSResult (audio also written to disk).
        """
        result = self.generate(text, voice=voice, speed=speed)
        save_wav(result.audio, path, sample_rate=self.SAMPLE_RATE)
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
