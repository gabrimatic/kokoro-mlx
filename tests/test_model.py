# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Tests for Kokoro MLX model architecture."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from kokoro_mlx.modules import (
    ALBERT,
    AdaLayerNorm,
    BiLSTM,
    DurationEncoder,
    LayerNorm,
    LinearNorm,
    TextEncoder,
    WeightNormConv1d,
)
from kokoro_mlx.istftnet import (
    AdaIN1d,
    AdaINResBlock1,
    AdainResBlk1d,
    Decoder,
    Generator,
    SineGen,
    SourceModuleHnNSF,
    iSTFT,
)


# ---------------------------------------------------------------------------
# WeightNormConv1d
# ---------------------------------------------------------------------------


class TestWeightNormConv1d:
    def test_output_shape_basic(self):
        conv = WeightNormConv1d(16, 32, kernel_size=3, padding=1)
        x = mx.random.normal((2, 16, 10))
        out = conv(x)
        assert out.shape == (2, 32, 10)

    def test_output_shape_dilated(self):
        conv = WeightNormConv1d(16, 16, kernel_size=5, dilation=2, padding=4)
        x = mx.random.normal((1, 16, 20))
        out = conv(x)
        assert out.shape == (1, 16, 20)

    def test_output_shape_stride(self):
        conv = WeightNormConv1d(8, 8, kernel_size=3, stride=2, padding=1)
        x = mx.random.normal((1, 8, 10))
        out = conv(x)
        assert out.shape == (1, 8, 5)

    def test_no_nan(self):
        conv = WeightNormConv1d(8, 16, kernel_size=3, padding=1)
        x = mx.random.normal((1, 8, 10))
        out = conv(x)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.tolist())))


# ---------------------------------------------------------------------------
# LayerNorm (channels)
# ---------------------------------------------------------------------------


class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(16)
        x = mx.random.normal((2, 16, 10))
        out = ln(x)
        assert out.shape == (2, 16, 10)

    def test_normalized(self):
        ln = LayerNorm(16)
        x = mx.random.normal((1, 16, 10)) * 10
        out = ln(x)
        mx.eval(out)
        arr = np.array(out.tolist())
        # Channel mean over time should be near default (gamma=1, beta=0)
        assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# BiLSTM
# ---------------------------------------------------------------------------


class TestBiLSTM:
    def test_output_shape(self):
        lstm = BiLSTM(32, 16)
        x = mx.random.normal((2, 10, 32))
        out = lstm(x)
        assert out.shape == (2, 10, 32)  # 2 * hidden_size

    def test_single_batch(self):
        lstm = BiLSTM(32, 16)
        x = mx.random.normal((1, 7, 32))
        out = lstm(x)
        assert out.shape == (1, 7, 32)

    def test_no_nan(self):
        lstm = BiLSTM(16, 8)
        x = mx.random.normal((1, 5, 16))
        out = lstm(x)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.tolist())))


# ---------------------------------------------------------------------------
# AdaIN1d
# ---------------------------------------------------------------------------


class TestAdaIN1d:
    def test_output_shape(self):
        adain = AdaIN1d(style_dim=64, num_features=32)
        x = mx.random.normal((2, 32, 10))
        s = mx.random.normal((2, 64))
        out = adain(x, s)
        assert out.shape == (2, 32, 10)


# ---------------------------------------------------------------------------
# AdaINResBlock1 (Generator-level)
# ---------------------------------------------------------------------------


class TestAdaINResBlock1:
    def test_output_shape(self):
        block = AdaINResBlock1(channels=16, kernel_size=3, dilation=(1, 3, 5), style_dim=32)
        x = mx.random.normal((1, 16, 20))
        s = mx.random.normal((1, 32))
        out = block(x, s)
        assert out.shape == (1, 16, 20)

    def test_no_nan(self):
        block = AdaINResBlock1(channels=16, kernel_size=3, dilation=(1, 3, 5), style_dim=32)
        x = mx.random.normal((1, 16, 20))
        s = mx.random.normal((1, 32))
        out = block(x, s)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.tolist())))


# ---------------------------------------------------------------------------
# AdainResBlk1d (Decoder-level)
# ---------------------------------------------------------------------------


class TestAdainResBlk1d:
    def test_same_dim(self):
        blk = AdainResBlk1d(32, 32, style_dim=16)
        x = mx.random.normal((1, 32, 20))
        s = mx.random.normal((1, 16))
        out = blk(x, s)
        assert out.shape == (1, 32, 20)

    def test_dim_change(self):
        blk = AdainResBlk1d(64, 32, style_dim=16)
        x = mx.random.normal((1, 64, 20))
        s = mx.random.normal((1, 16))
        out = blk(x, s)
        assert out.shape == (1, 32, 20)

    def test_upsample(self):
        blk = AdainResBlk1d(32, 16, style_dim=16, upsample=True)
        x = mx.random.normal((1, 32, 10))
        s = mx.random.normal((1, 16))
        out = blk(x, s)
        # Output time should be doubled
        assert out.shape[-1] == 20

    def test_no_nan(self):
        blk = AdainResBlk1d(16, 32, style_dim=8)
        x = mx.random.normal((1, 16, 15))
        s = mx.random.normal((1, 8))
        out = blk(x, s)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.tolist())))


# ---------------------------------------------------------------------------
# iSTFT
# ---------------------------------------------------------------------------


class TestISTFT:
    def test_transform_shape(self):
        stft = iSTFT(filter_length=20, hop_length=5, win_length=20)
        x = mx.random.normal((1, 300))
        mag, phase = stft.transform(x)
        assert mag.shape[0] == 1
        assert mag.shape[1] == 11  # n_fft//2 + 1
        assert mag.shape == phase.shape

    def test_inverse_shape(self):
        stft = iSTFT(filter_length=20, hop_length=5, win_length=20)
        n_bins = 11
        frames = 60
        mag = mx.abs(mx.random.normal((1, n_bins, frames)))
        phase = mx.random.uniform(-math.pi, math.pi, (1, n_bins, frames))
        out = stft.inverse(mag, phase)
        assert out.shape[0] == 1
        assert out.shape[1] == 1
        assert out.shape[2] > 0

    def test_roundtrip_reasonable(self):
        stft = iSTFT(filter_length=20, hop_length=5, win_length=20)
        x = mx.sin(mx.arange(300, dtype=mx.float32) * 0.1)[None, :]
        mag, phase = stft.transform(x)
        out = stft.inverse(mag, phase)
        mx.eval(out)
        assert not np.any(np.isnan(np.array(out.tolist())))


# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------


class TestTextEncoder:
    def test_output_shape(self):
        enc = TextEncoder(channels=32, kernel_size=5, depth=2, n_symbols=50)
        ids = mx.array([[1, 5, 10, 3]])  # (1, 4)
        lengths = mx.array([4])
        mask = mx.zeros((1, 4), dtype=mx.bool_)
        out = enc(ids, lengths, mask)
        assert out.shape == (1, 32, 4)


# ---------------------------------------------------------------------------
# ALBERT
# ---------------------------------------------------------------------------


class TestALBERT:
    def test_output_shape(self):
        model = ALBERT(
            vocab_size=50, embedding_size=32, hidden_size=64,
            intermediate_size=128, num_attention_heads=4, num_hidden_layers=2,
            max_position_embeddings=64
        )
        ids = mx.array([[1, 5, 10, 3]])
        out = model(ids)
        assert out.shape == (1, 4, 64)

    def test_with_mask(self):
        model = ALBERT(
            vocab_size=50, embedding_size=32, hidden_size=64,
            intermediate_size=128, num_attention_heads=4, num_hidden_layers=2,
            max_position_embeddings=64
        )
        ids = mx.array([[1, 5, 10, 3]])
        mask = mx.array([[1, 1, 1, 0]])
        out = model(ids, attention_mask=mask)
        assert out.shape == (1, 4, 64)


# ---------------------------------------------------------------------------
# Slow tests (require real model weights)
# ---------------------------------------------------------------------------


MODEL_PATH = "/Users/soroush/.cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16/snapshots/a71e4d38b236d968966a2002c4c895dbd12b1c3c"
WEIGHTS_PATH = f"{MODEL_PATH}/kokoro-v1_0.safetensors"


@pytest.mark.slow
class TestWeightLoading:
    @pytest.fixture(scope="class")
    def weights(self):
        from kokoro_mlx.model import _load_safetensors
        return _load_safetensors(WEIGHTS_PATH)

    def test_weight_count(self, weights):
        assert len(weights) == 548

    def test_bert_keys_present(self, weights):
        assert "bert.embeddings.word_embeddings.weight" in weights
        assert weights["bert.embeddings.word_embeddings.weight"].shape == (178, 128)

    def test_decoder_keys_present(self, weights):
        assert "decoder.generator.conv_post.weight_g" in weights

    def test_all_keys_loadable(self, weights):
        from kokoro_mlx.config import KokoroConfig
        from kokoro_mlx.model import KokoroModel, load_weights
        config = KokoroConfig.from_pretrained(MODEL_PATH)
        model = KokoroModel(config)
        n_loaded = load_weights(model, weights)
        assert n_loaded == 548, f"Expected 548, got {n_loaded}"


@pytest.mark.slow
class TestKokoroModelForward:
    @pytest.fixture(scope="class")
    def model(self):
        from kokoro_mlx.model import KokoroModel
        return KokoroModel.from_pretrained(MODEL_PATH)

    def test_model_loads(self, model):
        assert model is not None

    def test_vocab_present(self, model):
        assert len(model.vocab) > 0

    def test_forward_shape(self, model):
        # Create a minimal voice style vector
        ref_s = mx.random.normal((1, 256))
        # Use a few phonemes that are likely in vocab
        vocab = model.vocab
        phonemes = list(vocab.keys())[:5]
        phoneme_str = "".join(phonemes)
        audio = model.forward(phoneme_str, ref_s)
        mx.eval(audio)
        assert audio.ndim == 1
        assert audio.shape[0] > 0

    def test_forward_no_nan(self, model):
        ref_s = mx.zeros((1, 256))
        vocab = model.vocab
        phonemes = "".join(list(vocab.keys())[:3])
        audio = model.forward(phonemes, ref_s)
        mx.eval(audio)
        arr = np.array(audio.tolist())
        assert not np.any(np.isnan(arr))

    def test_forward_with_real_voice(self, model):
        """Test with a real voice style vector from safetensors."""
        from kokoro_mlx.voices import VoiceManager

        voice_dir = Path(MODEL_PATH) / "voices"
        if not voice_dir.exists() or not list(voice_dir.glob("*.safetensors")):
            pytest.skip("No voice files found")

        vm = VoiceManager(MODEL_PATH)
        voice = vm.load_voice("af_heart")
        ref_s = vm.get_style(voice, 5)
        vocab = model.vocab
        phonemes = "".join(list(vocab.keys())[:5])
        audio = model.forward(phonemes, ref_s)
        mx.eval(audio)
        assert audio.ndim == 1
        assert audio.shape[0] > 100


from pathlib import Path
