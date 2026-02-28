# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""kokoro-mlx: Kokoro TTS inference on Apple Silicon via MLX."""

__version__ = "0.1.0"

from .config import ISTFTNetConfig, KokoroConfig, PLBertConfig
from .kokoro import KokoroTTS, TTSResult
from .phonemize import Phonemizer
from .voices import DEFAULT_VOICE, VoiceManager

__all__ = [
    "__version__",
    "ISTFTNetConfig",
    "PLBertConfig",
    "KokoroConfig",
    "KokoroTTS",
    "TTSResult",
    "Phonemizer",
    "VoiceManager",
    "DEFAULT_VOICE",
]
