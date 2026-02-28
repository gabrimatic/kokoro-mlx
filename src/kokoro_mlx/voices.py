# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Voice and style vector management for Kokoro TTS."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

DEFAULT_VOICE = "af_heart"


class VoiceManager:
    """Loads and caches voice style vectors from a local Kokoro model directory."""

    def __init__(self, model_path: str | Path) -> None:
        self._voices_dir = Path(model_path) / "voices"
        self._cache: dict[str, mx.array] = {}

    def list_voices(self) -> list[str]:
        """Return the names of all available voices (safetensors files)."""
        return sorted(p.stem for p in self._voices_dir.glob("*.safetensors"))

    def load_voice(self, name: str) -> mx.array:
        """Load a voice by name, caching it for subsequent calls.

        Returns an MLX array with shape ``(510, 1, 256)``.
        """
        if name in self._cache:
            return self._cache[name]

        path = self._voices_dir / f"{name}.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"Voice file not found: {path}")

        from safetensors.numpy import load_file

        data = load_file(str(path))
        arr = mx.array(data["voice"])
        self._cache[name] = arr
        return arr

    def get_style(self, voice: mx.array, num_tokens: int) -> mx.array:
        """Select the style slice for *num_tokens* phoneme tokens.

        The style vector is indexed by sequence length:
        ``voice[min(num_tokens - 1, len(voice) - 1)]``.

        Returns an array of shape ``(1, 256)``.
        """
        idx = min(num_tokens - 1, voice.shape[0] - 1)
        return voice[idx]
