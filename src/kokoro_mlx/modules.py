# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Core model modules: TextEncoder, ALBERT, ProsodyPredictor, DurationEncoder, etc."""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


# ---------------------------------------------------------------------------
# WeightNormConv1d
# ---------------------------------------------------------------------------
# PyTorch weight_norm stores weight_g (out, 1, 1) and weight_v (out, in, kernel).
# Actual weight = weight_g * weight_v / ||weight_v||, norm over (in, kernel).
# MLX Conv1d expects (out, kernel, in) and NLC input.
# We store weight_g/weight_v in PyTorch layout and compute the effective weight
# at each forward pass.


class WeightNormConv1d(nn.Module):
    """Conv1d with weight normalization, matching PyTorch's weight_norm layout.

    Operates on NCL tensors (batch, channels, time) to match the rest of the code.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # PyTorch-layout: weight_v (out, in/groups, kernel), weight_g (out, 1, 1)
        scale = 1.0 / math.sqrt(in_channels * kernel_size)
        self.weight_v = mx.random.uniform(-scale, scale, (out_channels, in_channels // groups, kernel_size))
        self.weight_g = mx.ones((out_channels, 1, 1))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _compute_weight(self) -> mx.array:
        """Compute the normalized weight in MLX layout (out, kernel, in/groups)."""
        # weight_v: (out, in/groups, kernel)
        v = self.weight_v
        out_ch = v.shape[0]
        # Norm over (in/groups, kernel) dims → per output channel
        v_flat = v.reshape(out_ch, -1)  # (out, in/groups * kernel)
        norms = mx.linalg.norm(v_flat, axis=1, keepdims=True)[:, :, None]  # (out, 1, 1)
        # Normalized PyTorch-layout: (out, in/groups, kernel)
        w = self.weight_g * v / norms
        # Transpose to MLX layout: (out, kernel, in/groups)
        return w.transpose(0, 2, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. x is NCL (batch, channels, time)."""
        # Transpose to NLC for MLX conv
        x = x.transpose(0, 2, 1)  # NCL → NLC
        w = self._compute_weight()  # (out, kernel, in/groups)
        x = mx.conv1d(x, w, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            x = x + self.bias
        return x.transpose(0, 2, 1)  # NLC → NCL


# ---------------------------------------------------------------------------
# WeightNormConvTranspose1d
# ---------------------------------------------------------------------------
# PyTorch weight_norm on ConvTranspose1d: weight_v (in, out/groups, kernel),
# weight_g (in, 1, 1). MLX ConvTranspose1d weight layout is (in, kernel, out).


class WeightNormConvTranspose1d(nn.Module):
    """ConvTranspose1d with weight normalization. Operates on NCL tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.output_padding = output_padding

        # PyTorch layout: weight_v (in, out/groups, kernel), weight_g (in, 1, 1)
        scale = 1.0 / math.sqrt(in_channels * kernel_size)
        self.weight_v = mx.random.uniform(-scale, scale, (in_channels, out_channels // groups, kernel_size))
        self.weight_g = mx.ones((in_channels, 1, 1))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _compute_weight(self) -> mx.array:
        """Compute normalized weight in MLX layout (out, kernel, in/groups)."""
        v = self.weight_v  # (in, out/groups, kernel)
        in_ch = v.shape[0]
        v_flat = v.reshape(in_ch, -1)
        norms = mx.linalg.norm(v_flat, axis=1, keepdims=True)[:, :, None]  # (in, 1, 1)
        w = self.weight_g * v / norms  # (in, out/groups, kernel)
        G = self.groups
        C_in, C_out_g, K = w.shape
        # Reshape to (G, C_in/G, C_out/G, K) → (G, C_out/G, K, C_in/G) → (C_out, K, C_in/G)
        w = w.reshape(G, C_in // G, C_out_g, K)
        w = w.transpose(0, 2, 3, 1)  # (G, C_out/G, K, C_in/G)
        return w.reshape(G * C_out_g, K, C_in // G)  # (C_out, K, C_in/G)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass. x is NCL (batch, channels, time)."""
        x = x.transpose(0, 2, 1)  # NCL → NLC
        w = self._compute_weight()  # (in, kernel, out/groups)
        x = mx.conv_transpose1d(
            x,
            w,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )
        if self.bias is not None:
            x = x + self.bias
        return x.transpose(0, 2, 1)  # NLC → NCL


# ---------------------------------------------------------------------------
# LayerNorm (channels-last, for NCL tensors)
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    """Layer norm over the channel dim for NCL (batch, channels, time) tensors.

    Stores gamma/beta to match the saved weight names.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: NCL → transpose to NLC → layer norm → transpose back
        x = x.transpose(0, 2, 1)  # NLC
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = self.gamma * x + self.beta
        return x.transpose(0, 2, 1)  # NCL


# ---------------------------------------------------------------------------
# LinearNorm
# ---------------------------------------------------------------------------


class LinearNorm(nn.Module):
    """Linear layer wrapper (named to match PyTorch source)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_layer(x)


# ---------------------------------------------------------------------------
# BiLSTM
# ---------------------------------------------------------------------------
# MLX has a single LSTM; we manually run forward + backward.
# PyTorch stores: weight_ih_l0 (4H, input), weight_hh_l0 (4H, hidden),
#                 bias_ih_l0 (4H), bias_hh_l0 (4H)
# and _reverse versions for the backward direction.
# MLX LSTM uses combined bias = bias_ih + bias_hh.
# Gate order: i, f, g, o (same in both).


class BiLSTM(nn.Module):
    """Bidirectional LSTM that loads PyTorch-style weight dicts."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.fwd = nn.LSTM(input_size, hidden_size)
        self.bwd = nn.LSTM(input_size, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (batch, time, input_size) → (batch, time, 2*hidden_size)"""
        # Forward
        h_fwd, _ = self.fwd(x)  # (batch, time, hidden)
        # Backward: reverse time, run LSTM, reverse back
        x_rev = x[:, ::-1, :]
        h_bwd_rev, _ = self.bwd(x_rev)
        h_bwd = h_bwd_rev[:, ::-1, :]
        return mx.concatenate([h_fwd, h_bwd], axis=-1)  # (batch, time, 2*hidden)


# ---------------------------------------------------------------------------
# AdaLayerNorm
# ---------------------------------------------------------------------------


class AdaLayerNorm(nn.Module):
    """Adaptive layer norm conditioned on a style vector."""

    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        # x: NLC (batch, time, channels), s: (batch, style_dim)
        h = self.fc(s)  # (batch, 2*channels)
        h = h[:, None, :]  # (batch, 1, 2*channels)
        gamma, beta = mx.split(h, 2, axis=-1)  # each (batch, 1, channels)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = (1 + gamma) * x + beta
        return x  # NLC


# ---------------------------------------------------------------------------
# ALBERT
# ---------------------------------------------------------------------------


class ALBERTAttention(nn.Module):
    """Multi-head self-attention for ALBERT."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, H = x.shape
        nh = self.num_heads
        hd = self.head_dim

        q = self.query(x).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)  # (B, nh, T, hd)
        k = self.key(x).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(B, T, nh, hd).transpose(0, 2, 1, 3)

        scale = math.sqrt(hd)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # (B, nh, T, T)
        if mask is not None:
            scores = scores + mask[:, None, None, :]  # broadcast over heads and queries

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v  # (B, nh, T, hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, H)
        out = self.dense(out)
        return self.LayerNorm(out + x)


class ALBERTLayer(nn.Module):
    """Single ALBERT transformer layer (shared across all 12 layers)."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = ALBERTAttention(hidden_size, num_heads, dropout)
        self.ffn = nn.Linear(hidden_size, intermediate_size)
        self.ffn_output = nn.Linear(intermediate_size, hidden_size)
        self.full_layer_layer_norm = nn.LayerNorm(hidden_size)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.attention(x, mask)
        residual = x
        x = nn.gelu(self.ffn(x))
        x = self.ffn_output(x)
        return self.full_layer_layer_norm(x + residual)


class ALBERTEmbeddings(nn.Module):
    """ALBERT embeddings: word + position + token_type + LayerNorm."""

    def __init__(self, vocab_size: int, embedding_size: int, max_position_embeddings: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.token_type_embeddings = nn.Embedding(2, embedding_size)
        self.LayerNorm = nn.LayerNorm(embedding_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        B, T = input_ids.shape
        positions = mx.arange(T)[None, :]  # (1, T)
        token_types = mx.zeros((B, T), dtype=mx.int32)
        x = self.word_embeddings(input_ids) + self.position_embeddings(positions) + self.token_type_embeddings(token_types)
        return self.LayerNorm(x)


class ALBERTEncoder(nn.Module):
    """ALBERT encoder: embedding projection + shared layer repeated N times."""

    def __init__(self, embedding_size: int, hidden_size: int, intermediate_size: int,
                 num_heads: int, num_hidden_layers: int, dropout: float = 0.0):
        super().__init__()
        self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)
        # Single shared layer group with one layer, reused num_hidden_layers times
        self.albert_layer_groups = [_ALBERTLayerGroup(hidden_size, intermediate_size, num_heads, dropout)]
        self.num_hidden_layers = num_hidden_layers

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.embedding_hidden_mapping_in(x)
        layer = self.albert_layer_groups[0].albert_layers[0]
        for _ in range(self.num_hidden_layers):
            x = layer(x, mask)
        return x


class _ALBERTLayerGroup(nn.Module):
    """Wrapper to match weight key: encoder.albert_layer_groups.0.albert_layers.0.*"""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.albert_layers = [ALBERTLayer(hidden_size, intermediate_size, num_heads, dropout)]


class ALBERT(nn.Module):
    """Custom ALBERT implementation for Kokoro TTS.

    Returns the last hidden state (batch, seq_len, hidden_size).
    """

    def __init__(
        self,
        vocab_size: int = 178,
        embedding_size: int = 128,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        max_position_embeddings: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embeddings = ALBERTEmbeddings(vocab_size, embedding_size, max_position_embeddings)
        self.encoder = ALBERTEncoder(
            embedding_size, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, dropout
        )
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def __call__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None) -> mx.array:
        x = self.embeddings(input_ids)
        # Convert attention mask to additive mask: 0 → 0.0, pad → -inf
        mask = None
        if attention_mask is not None:
            # attention_mask: (B, T), 1=attend, 0=ignore
            mask = (1.0 - attention_mask.astype(mx.float32)) * -1e9
        return self.encoder(x, mask)


# ---------------------------------------------------------------------------
# TextEncoder
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Text encoder: embedding → CNN blocks → BiLSTM."""

    def __init__(self, channels: int, kernel_size: int, depth: int, n_symbols: int):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        # Each CNN block: [WeightNormConv1d, LayerNorm, LeakyReLU, Dropout]
        # Stored as list of lists for clean weight-key matching:
        # cnn.{i}.0 = conv, cnn.{i}.1 = LayerNorm
        self.cnn = [
            [WeightNormConv1d(channels, channels, kernel_size, padding=padding), LayerNorm(channels)]
            for _ in range(depth)
        ]
        self.lstm = BiLSTM(channels, channels // 2)

    def __call__(self, input_ids: mx.array, input_lengths: mx.array, mask: mx.array) -> mx.array:
        x = self.embedding(input_ids)  # (B, T, channels)
        x = x.transpose(0, 2, 1)      # NCL
        for block in self.cnn:
            conv, ln = block
            x = conv(x)
            x = ln(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            # dropout skipped at inference
        x = x.transpose(0, 2, 1)  # NLC
        x = self.lstm(x)           # (B, T, channels)
        x = x.transpose(0, 2, 1)  # NCL
        return x


# ---------------------------------------------------------------------------
# DurationEncoder
# ---------------------------------------------------------------------------


class DurationEncoder(nn.Module):
    """Duration encoder: alternating BiLSTM and AdaLayerNorm layers."""

    def __init__(self, sty_dim: int, d_model: int, nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.sty_dim = sty_dim
        self.dropout = dropout
        # lstms[even] = BiLSTM, lstms[odd] = AdaLayerNorm
        self.lstms = []
        for _ in range(nlayers):
            self.lstms.append(BiLSTM(d_model + sty_dim, d_model // 2))
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))

    def __call__(self, x: mx.array, style: mx.array, text_lengths: mx.array, mask: mx.array) -> mx.array:
        # x: (B, d_model, T)  — NCL
        # style: (B, sty_dim)
        B, _, T = x.shape
        # Expand style along time dimension
        s = mx.broadcast_to(style[:, None, :], (B, T, self.sty_dim))  # (B, T, sty_dim)

        # x to NLC and concat with style
        x = x.transpose(0, 2, 1)  # (B, T, d_model)
        x = mx.concatenate([x, s], axis=-1)  # (B, T, d_model + sty_dim)
        x = x.transpose(0, 2, 1)  # NCL: (B, d_model+sty_dim, T)

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                # x is NCL
                x = block(x.transpose(0, 2, 1), style).transpose(0, 2, 1)  # NCL
                # Re-concat style
                x_nlc = x.transpose(0, 2, 1)  # (B, T, d_model)
                x = mx.concatenate([x_nlc, s], axis=-1).transpose(0, 2, 1)  # NCL: (B, d_model+sty_dim, T)
            else:
                # BiLSTM expects NLC
                x_nlc = x.transpose(0, 2, 1)
                x_nlc = block(x_nlc)  # (B, T, d_model)
                x = x_nlc.transpose(0, 2, 1)  # NCL

        # Return NLC for ProsodyPredictor usage
        return x.transpose(0, 2, 1)  # (B, T, d_model+sty_dim or d_model)


# ---------------------------------------------------------------------------
# ProsodyPredictor
# ---------------------------------------------------------------------------


class ProsodyPredictor(nn.Module):
    """Predicts duration, F0, and noise energy from text embeddings."""

    def __init__(self, style_dim: int, d_hid: int, nlayers: int, max_dur: int = 50, dropout: float = 0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers, dropout=dropout)
        self.lstm = BiLSTM(d_hid + style_dim, d_hid // 2)
        self.duration_proj = LinearNorm(d_hid, max_dur)
        self.shared = BiLSTM(d_hid + style_dim, d_hid // 2)
        self.F0 = [
            None,  # placeholder indices; filled below
        ]
        # Import here to avoid circular import (istftnet imports modules)
        from .istftnet import AdainResBlk1d
        self.F0 = [
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
        ]
        self.N = [
            AdainResBlk1d(d_hid, d_hid, style_dim, dropout_p=dropout),
            AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True, dropout_p=dropout),
            AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim, dropout_p=dropout),
        ]
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1)   # weights loaded separately (not weight-normed)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1)

    def __call__(self, texts: mx.array, style: mx.array, text_lengths: mx.array,
                 alignment: mx.array, mask: mx.array):
        d = self.text_encoder(texts, style, text_lengths, mask)  # (B, T, d_hid+style)
        # d for lstm input: concat style
        B, T, _ = d.shape
        s = mx.broadcast_to(style[:, None, :], (B, T, style.shape[-1]))
        # d already has style concat from DurationEncoder last step — actually no,
        # DurationEncoder returns (B, T, d_hid+sty_dim) from the last LSTM's output
        # which is just d_hid. Let me re-check: the last block in DurationEncoder
        # is AdaLayerNorm, after which we re-concat style. So d is (B, T, d_hid+sty_dim).
        x, _ = self.lstm(d), None  # BiLSTM expects (B, T, input_size)
        x = self.lstm(d)           # (B, T, d_hid)
        duration = self.duration_proj(x)  # (B, T, 50)
        en = d.transpose(0, 2, 1) @ alignment  # (B, d_hid+sty_dim or d_hid, T') — hmm
        # Actually we need just the first d_hid dims; see forward in model.py
        return duration.squeeze(-1), en

    def F0Ntrain(self, x: mx.array, s: mx.array):
        # x: (B, d_hid+sty_dim, T') or (B, d_hid, T') — NCL
        # shared BiLSTM expects NLC
        x_nlc = x.transpose(0, 2, 1)
        x_nlc = self.shared(x_nlc)  # (B, T', d_hid)
        x = x_nlc.transpose(0, 2, 1)  # NCL: (B, d_hid, T')

        F0 = x
        for block in self.F0:
            F0 = block(F0, s)
        # F0_proj: plain Conv1d (non-weight-normed), stored in MLX NLC convention
        # Need to transpose: (B, C, T) → (B, T, C) → conv → (B, T, 1) → (B, 1, T)
        F0_t = F0.transpose(0, 2, 1)
        F0_out = self.F0_proj(F0_t).transpose(0, 2, 1)  # (B, 1, T')

        N = x
        for block in self.N:
            N = block(N, s)
        N_t = N.transpose(0, 2, 1)
        N_out = self.N_proj(N_t).transpose(0, 2, 1)  # (B, 1, T')

        return F0_out.squeeze(1), N_out.squeeze(1)
