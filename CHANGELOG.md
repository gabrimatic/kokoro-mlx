# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed

- Select the voice style by phoneme count, as the reference pipeline does. `generate()` and `generate_stream()` counted the BOS/EOS pads, so `n` phonemes used style row `n + 1` instead of `n - 1`. Default output audio changes slightly, most for short inputs.

## [0.1.4] - 2026-10-15

### Fixed

- README Quick Start installs the `audio` extra, which `speak()` needs; the plain install raised `ImportError` on the first call.
- README no longer claims the network runs in bfloat16; inference is float32.
- The playback `ImportError` now suggests `pip install "kokoro-mlx[audio]"`.
- README license links resolve on PyPI.
- README requirements say macOS 14+; MLX publishes no macOS 13 wheels.

### Changed

- The publish workflow runs the full CI suite on the release commit and publishes only if it passes.
- The sdist no longer ships the `.github` directory.
- Drop the license classifier, which duplicates the `MIT` license expression.

## [0.1.3] - 2026-10-15

### Fixed

- Compute instance and layer normalization statistics in float32. In float16, `x**2` overflowed for typical decoder activations and produced NaN audio; in bfloat16 the variance lost all precision (#2).
- Accumulate the harmonic source phase in float32 regardless of the input dtype.
- Load weights and voices with `mx.load` and upcast bfloat16/float16 tensors to float32. `safetensors.numpy` could not read bfloat16 checkpoints at all.
- Match the reference vocoder: reflect-pad the harmonic source before the STFT (as `torch.stft(center=True)` does) and use reflection padding before the last generator stage instead of zero padding. Default output audio changes slightly as a result.
- `ProsodyPredictor.__call__` no longer runs the duration LSTM twice or fails on the `squeeze` of the duration logits.

### Changed

- Require MLX 0.32.4+ and soundfile 0.14+; `sounddevice` extra requires 0.5.6+.
- Drop the unused `safetensors` runtime dependency.
- `dev` extra now includes ruff.
- Regression tests for reduced-precision numerics that run without the model weights.
- README now describes the real float32 inference path.

## [0.1.2] - 2026-05-12

### Changed

- Require MLX 0.31+ and align runtime dependencies with the current MLX audio ecosystem.
- Constrain Python support to 3.10-3.12, matching the current `misaki[en]` support range.
- Infer G2P language from Kokoro voice prefixes, with an explicit `language` override on `generate`, `generate_stream`, `speak`, and `save`.
- Add optional Japanese and Mandarin G2P extras: `kokoro-mlx[ja]`, `kokoro-mlx[zh]`, and `kokoro-mlx[multilingual]`.
- Update README API and language selection docs.

## [0.1.1] - 2026-03-12

### Changed

- Widen the Python constraint to 3.10+.

## [0.1.0] - 2026-02-28

### Added

- Pure MLX implementation of Kokoro-82M text-to-speech for Apple Silicon. No PyTorch, no transformers, no third-party ML frameworks.
- ALBERT-based text encoder with 3 hidden layers, 768-dim embeddings, shared parameters across layers.
- Prosody predictor with BiLSTM backbone, duration and F0 estimation per phoneme.
- iSTFTNet vocoder: multi-scale decoder with SineGen excitation, AdaIN residual blocks, inverse STFT synthesis to 24 kHz float32 audio.
- WeightNormConv1d layers throughout the decoder stack for stable training-weight loading.
- G2P frontend via misaki: English phonemization with automatic sentence chunking at the 510-token limit.
- 54 built-in voices (American English, British English, and additional languages) with style vector management.
- Speed control for adjusting speech rate.
- Streaming synthesis: sentence-by-sentence generation for long text inputs.
- Automatic long-text chunking at sentence boundaries.
- WAV export via soundfile (24 kHz, float32).
- Audio playback via sounddevice (optional dependency).
- `KokoroTTS` public API: context manager, `from_pretrained`, `generate`, thread-safe via internal lock.
- KokoroConfig, ISTFTNetConfig, PLBertConfig dataclasses for full model configuration.
- 82 tests covering all modules.
