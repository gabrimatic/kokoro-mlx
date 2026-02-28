# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Model configuration dataclasses for Kokoro TTS."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ISTFTNetConfig:
    """Configuration for the iSTFTNet vocoder component."""

    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [20, 12])
    upsample_rates: list[int] = field(default_factory=lambda: [10, 6])
    gen_istft_hop_size: int = 5
    gen_istft_n_fft: int = 20
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    upsample_initial_channel: int = 512

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ISTFTNetConfig":
        return cls(
            upsample_kernel_sizes=d.get("upsample_kernel_sizes", cls.__dataclass_fields__["upsample_kernel_sizes"].default_factory()),  # type: ignore[misc]
            upsample_rates=d.get("upsample_rates", cls.__dataclass_fields__["upsample_rates"].default_factory()),  # type: ignore[misc]
            gen_istft_hop_size=d.get("gen_istft_hop_size", 5),
            gen_istft_n_fft=d.get("gen_istft_n_fft", 20),
            resblock_dilation_sizes=d.get("resblock_dilation_sizes", cls.__dataclass_fields__["resblock_dilation_sizes"].default_factory()),  # type: ignore[misc]
            resblock_kernel_sizes=d.get("resblock_kernel_sizes", cls.__dataclass_fields__["resblock_kernel_sizes"].default_factory()),  # type: ignore[misc]
            upsample_initial_channel=d.get("upsample_initial_channel", 512),
        )


@dataclass
class PLBertConfig:
    """Configuration for the PL-BERT text encoder."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 512
    num_hidden_layers: int = 12
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PLBertConfig":
        return cls(
            hidden_size=d.get("hidden_size", 768),
            num_attention_heads=d.get("num_attention_heads", 12),
            intermediate_size=d.get("intermediate_size", 2048),
            max_position_embeddings=d.get("max_position_embeddings", 512),
            num_hidden_layers=d.get("num_hidden_layers", 12),
            dropout=d.get("dropout", 0.1),
        )


@dataclass
class KokoroConfig:
    """Top-level Kokoro TTS model configuration."""

    istftnet: ISTFTNetConfig = field(default_factory=ISTFTNetConfig)
    plbert: PLBertConfig = field(default_factory=PLBertConfig)
    vocab: dict[str, int] = field(default_factory=dict)
    dim_in: int = 64
    dropout: float = 0.2
    hidden_dim: int = 512
    max_conv_dim: int = 512
    max_dur: int = 50
    multispeaker: bool = True
    n_layer: int = 3
    n_mels: int = 80
    n_token: int = 178
    style_dim: int = 128
    text_encoder_kernel_size: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KokoroConfig":
        istftnet = ISTFTNetConfig.from_dict(d.get("istftnet", {}))
        plbert = PLBertConfig.from_dict(d.get("plbert", {}))
        return cls(
            istftnet=istftnet,
            plbert=plbert,
            vocab=d.get("vocab", {}),
            dim_in=d.get("dim_in", 64),
            dropout=d.get("dropout", 0.2),
            hidden_dim=d.get("hidden_dim", 512),
            max_conv_dim=d.get("max_conv_dim", 512),
            max_dur=d.get("max_dur", 50),
            multispeaker=d.get("multispeaker", True),
            n_layer=d.get("n_layer", 3),
            n_mels=d.get("n_mels", 80),
            n_token=d.get("n_token", 178),
            style_dim=d.get("style_dim", 128),
            text_encoder_kernel_size=d.get("text_encoder_kernel_size", 5),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "KokoroConfig":
        """Load config from a config.json file path."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path) -> "KokoroConfig":
        """Load config from a local directory or HuggingFace Hub model ID.

        If *model_id_or_path* is an existing directory, ``config.json`` is read
        from it directly. Otherwise it is treated as a HuggingFace Hub repo_id
        and the config is downloaded.
        """
        path = Path(model_id_or_path)
        if path.is_dir():
            return cls.from_file(path / "config.json")

        from huggingface_hub import hf_hub_download

        config_file = hf_hub_download(repo_id=str(model_id_or_path), filename="config.json")
        return cls.from_file(config_file)
