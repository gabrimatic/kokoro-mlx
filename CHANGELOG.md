# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
