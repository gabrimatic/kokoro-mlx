# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""iSTFTNet vocoder modules: Generator, Decoder, AdaIN blocks, STFT utilities."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .modules import WeightNormConv1d, WeightNormConvTranspose1d, get_padding

# ---------------------------------------------------------------------------
# InstanceNorm1d (not built-in in MLX)
# ---------------------------------------------------------------------------


def instance_norm1d(x: mx.array, eps: float = 1e-5) -> mx.array:
    """Manual InstanceNorm1d for NCL tensors. Normalizes over the time (L) dimension."""
    # x: (B, C, L)
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + eps)


# ---------------------------------------------------------------------------
# AdaIN1d
# ---------------------------------------------------------------------------


class AdaIN1d(nn.Module):
    """Adaptive instance normalization conditioned on a style vector."""

    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.fc = nn.Linear(style_dim, num_features * 2)

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        # x: NCL (batch, channels, time), s: (batch, style_dim)
        h = self.fc(s)  # (batch, 2*num_features)
        # Reshape to (batch, num_features, 1) for broadcasting
        half = h.shape[-1] // 2
        gamma = h[:, :half, None]  # (batch, num_features, 1)
        beta = h[:, half:, None]
        return (1 + gamma) * instance_norm1d(x) + beta


# ---------------------------------------------------------------------------
# AdaINResBlock1 (used in Generator)
# ---------------------------------------------------------------------------


class AdaINResBlock1(nn.Module):
    """Residual block with AdaIN and Snake activation, used inside Generator."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5), style_dim: int = 64):
        super().__init__()
        self.convs1 = [
            WeightNormConv1d(channels, channels, kernel_size, dilation=dilation[0],
                             padding=get_padding(kernel_size, dilation[0])),
            WeightNormConv1d(channels, channels, kernel_size, dilation=dilation[1],
                             padding=get_padding(kernel_size, dilation[1])),
            WeightNormConv1d(channels, channels, kernel_size, dilation=dilation[2],
                             padding=get_padding(kernel_size, dilation[2])),
        ]
        self.convs2 = [
            WeightNormConv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1)),
            WeightNormConv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1)),
            WeightNormConv1d(channels, channels, kernel_size, padding=get_padding(kernel_size, 1)),
        ]
        self.adain1 = [AdaIN1d(style_dim, channels) for _ in range(3)]
        self.adain2 = [AdaIN1d(style_dim, channels) for _ in range(3)]
        # alpha parameters: shape (1, channels, 1)
        self.alpha1 = [mx.ones((1, channels, 1)) for _ in range(3)]
        self.alpha2 = [mx.ones((1, channels, 1)) for _ in range(3)]

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1.0 / a1) * mx.sin(a1 * xt) ** 2  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1.0 / a2) * mx.sin(a2 * xt) ** 2
            xt = c2(xt)
            x = xt + x
        return x


# ---------------------------------------------------------------------------
# AdainResBlk1d (used in Decoder and ProsodyPredictor)
# ---------------------------------------------------------------------------


class AdainResBlk1d(nn.Module):
    """Residual block with AdaIN, used in the Decoder and ProsodyPredictor."""

    def __init__(self, dim_in: int, dim_out: int, style_dim: int = 64,
                 upsample: bool = False, dropout_p: float = 0.0):
        super().__init__()
        self.upsample_type = "upsample" if upsample else "none"
        self.learned_sc = dim_in != dim_out
        self.dropout_p = dropout_p

        self.conv1 = WeightNormConv1d(dim_in, dim_out, 3, padding=1)
        self.conv2 = WeightNormConv1d(dim_out, dim_out, 3, padding=1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = WeightNormConv1d(dim_in, dim_out, 1, bias=False)

        if upsample:
            # ConvTranspose1d groups=dim_in (depthwise), kernel=3, stride=2, padding=1, output_padding=1
            self.pool = WeightNormConvTranspose1d(
                dim_in, dim_in, kernel_size=3, stride=2, padding=1,
                groups=dim_in, output_padding=1
            )
        else:
            self.pool = None

    def _shortcut(self, x: mx.array) -> mx.array:
        if self.upsample_type != "none":
            # Nearest-neighbor upsample × 2
            x = mx.repeat(x, 2, axis=-1)  # NCL → double T
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: mx.array, s: mx.array) -> mx.array:
        x = self.norm1(x, s)
        x = nn.leaky_relu(x, negative_slope=0.2)
        if self.pool is not None:
            x = self.pool(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)
        return x

    def __call__(self, x: mx.array, s: mx.array) -> mx.array:
        out = self._residual(x, s)
        sc = self._shortcut(x)
        return (out + sc) * (1.0 / math.sqrt(2))


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------


def _interpolate_1d(x: mx.array, out_len: int) -> mx.array:
    """1-D linear interpolation along axis=1 (align_corners=False convention).

    x: (B, L_in, D) → (B, out_len, D)
    """
    L_in = x.shape[1]
    if L_in == out_len:
        return x
    # Source coordinate for each output position (align_corners=False)
    idx = (mx.arange(out_len, dtype=mx.float32) + 0.5) * (L_in / out_len) - 0.5
    idx = mx.clip(idx, 0, L_in - 1)
    lo = mx.floor(idx).astype(mx.int32)
    hi = mx.minimum(lo + 1, L_in - 1)
    frac = (idx - lo.astype(mx.float32))[:, None]  # (out_len, 1) for broadcasting over D
    return x[:, lo, :] * (1.0 - frac) + x[:, hi, :] * frac


# ---------------------------------------------------------------------------
# SineGen
# ---------------------------------------------------------------------------


class SineGen(nn.Module):
    """Sine-based excitation generator for the source module."""

    def __init__(self, samp_rate: int, upsample_scale: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, noise_std: float = 0.003,
                 voiced_threshold: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0: mx.array) -> mx.array:
        return (f0 > self.voiced_threshold).astype(mx.float32)

    def _f02sine(self, f0_values: mx.array) -> mx.array:
        """f0_values: (batch, length, dim)"""
        # Fractional cycles per sample
        rad_values = (f0_values / self.sampling_rate) % 1.0

        # Random initial phase for harmonics (fundamental always starts at 0)
        rand_ini = mx.random.uniform(shape=(f0_values.shape[0], f0_values.shape[2]))
        rand_ini = mx.concatenate([mx.zeros((f0_values.shape[0], 1)), rand_ini[:, 1:]], axis=1)
        rad_values = rad_values.at[:, 0, :].add(rand_ini)

        # Match the original phase computation: downsample rad values, cumsum at
        # the low rate, then upsample the phase back.  This produces the smooth
        # phase trajectory that the model was trained on.
        scale = self.upsample_scale
        B, L, D = f0_values.shape
        L_down = L // scale

        # Downsample via linear interpolation (align_corners=False convention)
        rad_down = _interpolate_1d(rad_values, L_down)  # (B, L_down, D)

        # Cumulative phase at the low rate, then scale up
        phase = mx.cumsum(rad_down, axis=1) * (2.0 * math.pi)  # (B, L_down, D)
        phase = phase * scale

        # Upsample phase back to full resolution via linear interpolation
        phase = _interpolate_1d(phase, L)  # (B, L, D)

        return mx.sin(phase)

    def __call__(self, f0: mx.array):
        """f0: (batch, length, 1) → sine_waves (batch, length, dim), uv, noise"""
        # Expand to harmonics: multiply by [1, 2, ..., harmonic_num+1]
        harmonics = mx.array(list(range(1, self.harmonic_num + 2)), dtype=mx.float32)[None, None, :]  # (1, 1, dim)
        fn = f0 * harmonics  # (B, L, dim)

        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1.0 - uv) * self.sine_amp / 3.0
        noise = noise_amp * mx.random.normal(shape=sine_waves.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


# ---------------------------------------------------------------------------
# SourceModuleHnNSF
# ---------------------------------------------------------------------------


class SourceModuleHnNSF(nn.Module):
    """Harmonic-noise source module."""

    def __init__(self, sampling_rate: int, upsample_scale: int, harmonic_num: int = 0,
                 sine_amp: float = 0.1, add_noise_std: float = 0.003, voiced_threshod: float = 0):
        super().__init__()
        self.sine_amp = sine_amp
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)

    def __call__(self, x: mx.array):
        """x: (batch, length, 1) F0 signal"""
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = mx.tanh(self.l_linear(sine_wavs))  # (B, L, 1)
        noise = mx.random.normal(shape=uv.shape) * self.sine_amp / 3.0
        return sine_merge, noise, uv


# ---------------------------------------------------------------------------
# iSTFT (conv-based)
# ---------------------------------------------------------------------------


class iSTFT(nn.Module):
    """STFT/iSTFT implemented via convolution for MLX compatibility.

    Uses Hann window and DFT matrix approach.
    filter_length = n_fft, hop_length, win_length.
    """

    def __init__(self, filter_length: int = 20, hop_length: int = 5, win_length: int = 20):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        n_fft = filter_length
        self.n_fft = n_fft

        # Build Hann window (periodic)
        win = np.hanning(win_length + 1)[:-1].astype(np.float32)

        # Build DFT matrix: real and imaginary parts for bins 0..n_fft//2
        n_bins = n_fft // 2 + 1
        t = np.arange(n_fft, dtype=np.float32)
        freqs = np.arange(n_bins, dtype=np.float32)
        # DFT: e^{-2pi*j*k*n/N}
        angles = -2.0 * np.pi * np.outer(freqs, t) / n_fft  # (n_bins, n_fft)
        fourier_real = np.cos(angles) * win[None, :]  # windowed DFT basis
        fourier_imag = np.sin(angles) * win[None, :]

        # Forward conv kernel: (2*n_bins, n_fft, 1) for MLX Conv1d (out, kernel, in)
        fourier = np.concatenate([fourier_real, fourier_imag], axis=0)
        self.forward_basis = mx.array(fourier[:, :, None])

        self._window = mx.array(win)
        self._n_bins = n_bins

        # Precompute iDFT basis matrices from the half-spectrum (bins 0..N/2).
        # A real signal's full DFT has conjugate symmetry: X[N-k] = conj(X[k]).
        # When reconstructing from only the non-negative bins, bins 1..N/2-1
        # must be scaled by 2 to account for their conjugate counterparts.
        # Bins 0 (DC) and N/2 (Nyquist) have no conjugate pair and keep 1/N.
        angles_pos = 2.0 * np.pi * np.outer(freqs, t) / n_fft  # (n_bins, n_fft)
        conjugate_scale = np.ones(n_bins, dtype=np.float32)
        conjugate_scale[1 : n_bins - 1] = 2.0
        cos_mat_T = (np.cos(angles_pos) * conjugate_scale[:, None] / n_fft).T  # (n_fft, n_bins)
        sin_mat_T = (np.sin(angles_pos) * conjugate_scale[:, None] / n_fft).T
        # Store as (1, n_fft, n_bins) for batched matmul
        self._cos_mat_T = mx.array(cos_mat_T[None])
        self._sin_mat_T = mx.array(sin_mat_T[None])

    def transform(self, x: mx.array):
        """x: (batch, time) → magnitude (batch, n_bins, frames), phase (batch, n_bins, frames)"""
        n_fft = self.n_fft
        pad = n_fft // 2
        x_padded = mx.pad(x, [(0, 0), (pad, pad)])
        x_nlc = x_padded[:, :, None]
        out = mx.conv1d(x_nlc, self.forward_basis, stride=self.hop_length, padding=0)
        out = out.transpose(0, 2, 1)  # (B, 2*n_bins, frames)

        n_bins = self._n_bins
        real = out[:, :n_bins, :]
        imag = out[:, n_bins:, :]
        magnitude = mx.sqrt(real ** 2 + imag ** 2)
        phase = mx.arctan2(imag, real)
        return magnitude, phase

    def inverse(self, magnitude: mx.array, phase: mx.array) -> mx.array:
        """Reconstruct waveform from magnitude and phase.

        magnitude, phase: (batch, n_bins, frames)
        Returns: (batch, 1, samples)
        """
        n_fft = self.n_fft
        hop = self.hop_length

        # Reconstruct complex spectrum
        real = magnitude * mx.cos(phase)  # (B, n_bins, frames)
        imag = magnitude * mx.sin(phase)

        B = real.shape[0]
        num_frames = real.shape[-1]

        # iDFT via precomputed basis: (B, n_fft, frames)
        # x[n] = cos_T @ X_r - sin_T @ X_i  (1/N scaling already in the matrices)
        frames = self._cos_mat_T @ real - self._sin_mat_T @ imag  # (B, n_fft, frames)

        # Apply synthesis window
        frames = frames * self._window[:, None]  # (B, n_fft, frames)

        # Vectorized overlap-add via scatter-add — no Python loop over frames.
        # A Python loop of ~12000 iterations builds an enormous MLX graph; a single
        # scatter-add keeps the graph flat and eliminates the resulting artifacts.
        total_len = n_fft + (num_frames - 1) * hop

        # Index map: frame i, sample k → output position i*hop + k
        frame_offsets = mx.arange(num_frames) * hop   # (num_frames,)
        sample_offsets = mx.arange(n_fft)              # (n_fft,)
        idx = (frame_offsets[:, None] + sample_offsets[None, :]).reshape(-1)  # (num_frames * n_fft,)

        # Flatten frames: (B, n_fft, num_frames) → (B, num_frames * n_fft)
        frames_flat = frames.transpose(0, 2, 1).reshape(B, -1)

        output = mx.zeros((B, total_len))
        output = output.at[:, idx].add(frames_flat)

        # Window-squared normalization (same for every batch element — compute once)
        win_sq = self._window ** 2
        win_sq_flat = mx.broadcast_to(win_sq[None, :], (num_frames, n_fft)).reshape(-1)
        win_sq_sum = mx.zeros((total_len,))
        win_sq_sum = win_sq_sum.at[idx].add(win_sq_flat)

        output = output / mx.maximum(win_sq_sum[None, :], mx.array(1e-8))

        # Remove the center-pad added in transform()
        pad = n_fft // 2
        output = output[:, pad:-pad] if pad > 0 else output
        return output[:, None, :]  # (B, 1, samples)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """iSTFTNet generator."""

    def __init__(
        self,
        style_dim: int,
        resblock_kernel_sizes: list[int],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        resblock_dilation_sizes: list[list[int]],
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.post_n_fft = gen_istft_n_fft
        upsample_scale = math.prod(upsample_rates) * gen_istft_hop_size

        self.m_source = SourceModuleHnNSF(
            sampling_rate=24000,
            upsample_scale=upsample_scale,
            harmonic_num=8,
            voiced_threshod=10,
        )
        self.f0_upsamp_scale = upsample_scale
        self.stft = iSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)

        # Upsampling ConvTranspose1d layers
        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(WeightNormConvTranspose1d(in_ch, out_ch, k, stride=u, padding=(k - u) // 2))

        # Noise convs (plain Conv1d, not weight-normed)
        self.noise_convs = []
        for i in range(len(upsample_rates)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0,
                              padding=(stride_f0 + 1) // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, c_cur, kernel_size=1))

        # Noise residual blocks
        self.noise_res = []
        for i in range(len(upsample_rates)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                self.noise_res.append(AdaINResBlock1(c_cur, 7, (1, 3, 5), style_dim))
            else:
                self.noise_res.append(AdaINResBlock1(c_cur, 11, (1, 3, 5), style_dim))

        # Main residual blocks
        self.resblocks = []
        for i in range(len(upsample_rates)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, tuple(d), style_dim))

        self.conv_post = WeightNormConv1d(
            upsample_initial_channel // (2 ** len(upsample_rates)),
            gen_istft_n_fft + 2,
            7,
            padding=3,
        )

    def __call__(self, x: mx.array, s: mx.array, f0: mx.array) -> mx.array:
        # f0: (B, T) → upsample
        f0_up = mx.repeat(f0[:, None, :], self.f0_upsamp_scale, axis=-1)  # (B, 1, T*scale)
        f0_up = f0_up.transpose(0, 2, 1)  # (B, T*scale, 1) for SourceModule

        har_source, noi_source, uv = self.m_source(f0_up)
        # har_source: (B, T*scale, 1) → (B, T*scale)
        har_source = har_source.squeeze(-1)  # (B, T*scale)
        har_spec, har_phase = self.stft.transform(har_source)  # each (B, n_bins, frames)
        har = mx.concatenate([har_spec, har_phase], axis=1)  # (B, n_fft+2, frames)

        for i in range(self.num_upsamples):
            x = nn.leaky_relu(x, negative_slope=0.1)

            # noise_convs uses plain MLX Conv1d (NLC)
            har_nlc = har.transpose(0, 2, 1)
            x_source_nlc = self.noise_convs[i](har_nlc)
            x_source = x_source_nlc.transpose(0, 2, 1)  # NCL
            x_source = self.noise_res[i](x_source, s)

            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                # Reflection pad (1, 0) — pad 1 on left
                x = mx.pad(x, [(0, 0), (0, 0), (1, 0)])

            # Trim x_source to match x's time dim if needed
            if x_source.shape[-1] != x.shape[-1]:
                min_t = min(x_source.shape[-1], x.shape[-1])
                x = x[:, :, :min_t]
                x_source = x_source[:, :, :min_t]

            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                rb = self.resblocks[i * self.num_kernels + j](x, s)
                xs = rb if xs is None else xs + rb
            x = xs / self.num_kernels

        x = nn.leaky_relu(x)
        x = self.conv_post(x).astype(mx.float32)  # float32 for precise waveform reconstruction
        spec = mx.exp(x[:, :self.post_n_fft // 2 + 1, :])
        phase = mx.sin(x[:, self.post_n_fft // 2 + 1:, :])
        return self.stft.inverse(spec, phase)  # (B, 1, samples)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """Full decoder: F0/N processing → AdainResBlk1d blocks → Generator."""

    def __init__(
        self,
        dim_in: int,
        style_dim: int,
        dim_out: int,
        resblock_kernel_sizes: list[int],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        resblock_dilation_sizes: list[list[int]],
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
    ):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = [
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
            AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
        ]
        self.F0_conv = WeightNormConv1d(1, 1, kernel_size=3, stride=2, padding=1)
        self.N_conv = WeightNormConv1d(1, 1, kernel_size=3, stride=2, padding=1)
        self.asr_res = [WeightNormConv1d(512, 64, kernel_size=1)]
        self.generator = Generator(
            style_dim=style_dim,
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gen_istft_n_fft=gen_istft_n_fft,
            gen_istft_hop_size=gen_istft_hop_size,
        )

    def __call__(self, asr: mx.array, F0_curve: mx.array, N: mx.array, s: mx.array) -> mx.array:
        # asr: (B, 512, T'), F0_curve: (B, T'), N: (B, T'), s: (B, style_dim)
        F0 = self.F0_conv(F0_curve[:, None, :])  # (B, 1, T'//2)
        N_proc = self.N_conv(N[:, None, :])       # (B, 1, T'//2)

        x = mx.concatenate([asr, F0, N_proc], axis=1)  # (B, 512+1+1=514, T')
        x = self.encode(x, s)  # (B, 1024, T')

        asr_res = self.asr_res[0](asr)  # (B, 64, T')

        res = True
        for block in self.decode:
            if res:
                # Align time dims between x, asr_res, F0, N_proc
                T = x.shape[-1]
                ar = asr_res[:, :, :T]
                f = F0[:, :, :T]
                n = N_proc[:, :, :T]
                x = mx.concatenate([x, ar, f, n], axis=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        x = self.generator(x, s, F0_curve)
        return x
