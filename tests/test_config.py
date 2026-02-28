# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Tests for KokoroConfig dataclasses."""

import json
import tempfile
from pathlib import Path

from kokoro_mlx.config import ISTFTNetConfig, KokoroConfig, PLBertConfig

# Minimal config that mirrors the real Kokoro layout.
_SAMPLE_CONFIG: dict = {
    "istftnet": {
        "upsample_kernel_sizes": [20, 12],
        "upsample_rates": [10, 6],
        "gen_istft_hop_size": 5,
        "gen_istft_n_fft": 20,
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "resblock_kernel_sizes": [3, 7, 11],
        "upsample_initial_channel": 512,
    },
    "dim_in": 64,
    "dropout": 0.2,
    "hidden_dim": 512,
    "max_conv_dim": 512,
    "max_dur": 50,
    "multispeaker": True,
    "n_layer": 3,
    "n_mels": 80,
    "n_token": 178,
    "style_dim": 128,
    "text_encoder_kernel_size": 5,
    "plbert": {
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "max_position_embeddings": 512,
        "num_hidden_layers": 12,
        "dropout": 0.1,
    },
    "vocab": {str(i): i for i in range(114)},
}


class TestISTFTNetConfig:
    def test_from_dict(self):
        cfg = ISTFTNetConfig.from_dict(_SAMPLE_CONFIG["istftnet"])
        assert cfg.upsample_kernel_sizes == [20, 12]
        assert cfg.upsample_rates == [10, 6]
        assert cfg.gen_istft_hop_size == 5
        assert cfg.gen_istft_n_fft == 20
        assert cfg.resblock_dilation_sizes == [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        assert cfg.resblock_kernel_sizes == [3, 7, 11]
        assert cfg.upsample_initial_channel == 512

    def test_defaults(self):
        cfg = ISTFTNetConfig()
        assert cfg.gen_istft_hop_size == 5
        assert cfg.upsample_initial_channel == 512


class TestPLBertConfig:
    def test_from_dict(self):
        cfg = PLBertConfig.from_dict(_SAMPLE_CONFIG["plbert"])
        assert cfg.hidden_size == 768
        assert cfg.num_attention_heads == 12
        assert cfg.intermediate_size == 2048
        assert cfg.max_position_embeddings == 512
        assert cfg.num_hidden_layers == 12
        assert abs(cfg.dropout - 0.1) < 1e-9

    def test_defaults(self):
        cfg = PLBertConfig()
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12


class TestKokoroConfig:
    def test_from_dict(self):
        cfg = KokoroConfig.from_dict(_SAMPLE_CONFIG)
        assert cfg.n_token == 178
        assert cfg.n_mels == 80
        assert cfg.style_dim == 128
        assert cfg.hidden_dim == 512
        assert cfg.multispeaker is True

    def test_vocab_size(self):
        cfg = KokoroConfig.from_dict(_SAMPLE_CONFIG)
        assert len(cfg.vocab) == 114

    def test_nested_plbert(self):
        cfg = KokoroConfig.from_dict(_SAMPLE_CONFIG)
        assert isinstance(cfg.plbert, PLBertConfig)
        assert cfg.plbert.hidden_size == 768

    def test_nested_istftnet(self):
        cfg = KokoroConfig.from_dict(_SAMPLE_CONFIG)
        assert isinstance(cfg.istftnet, ISTFTNetConfig)
        assert cfg.istftnet.gen_istft_hop_size == 5

    def test_from_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(_SAMPLE_CONFIG), encoding="utf-8")
            cfg = KokoroConfig.from_file(config_path)
            assert cfg.n_token == 178
            assert len(cfg.vocab) == 114

    def test_from_pretrained_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps(_SAMPLE_CONFIG), encoding="utf-8")
            cfg = KokoroConfig.from_pretrained(tmpdir)
            assert cfg.n_token == 178
            assert cfg.plbert.hidden_size == 768
            assert cfg.istftnet.gen_istft_hop_size == 5
