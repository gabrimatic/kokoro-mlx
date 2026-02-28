# Contributing

Bug fixes, performance improvements, better docs. Here's how to get involved.

## Dev Setup

```bash
git clone https://github.com/gabrimatic/kokoro-mlx.git
cd kokoro-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

Tests that load the full model weights are marked `slow`. To skip them:

```bash
python -m pytest tests/ -v -m "not slow"
```

**Python 3.10 to 3.12 only.** The misaki G2P library does not support Python 3.13+.

## Architecture

```
src/kokoro_mlx/
├── kokoro.py       # KokoroTTS public API, context manager, thread safety
├── model.py        # KokoroModel forward pass, weight loading
├── modules.py      # ALBERT, TextEncoder, ProsodyPredictor, BiLSTM, WeightNormConv1d
├── istftnet.py     # iSTFTNet vocoder: Generator, Decoder, SineGen, iSTFT, AdaIN blocks
├── generate.py     # Text-to-audio pipeline, chunking, streaming
├── phonemize.py    # G2P wrapper (misaki), sentence chunking at 510-token limit
├── voices.py       # Voice loading, style vector management
├── playback.py     # Audio playback and WAV export
└── config.py       # KokoroConfig, ISTFTNetConfig, PLBertConfig dataclasses
```

Key constraint: **no PyTorch, no transformers.** The entire inference pipeline is implemented in pure MLX + numpy. Keep it that way.

## Testing

```bash
python -m pytest tests/ -v                   # all tests
python -m pytest tests/ -v -m "not slow"     # fast tests only (no model loading)
python -m pytest tests/test_model.py -v      # specific module
```

The `slow` marker covers tests that load the full 82M model. Fast tests cover unit-level logic (phonemization, config parsing, voice loading, audio generation).

## PR Checklist

- One feature or fix per PR. Keep scope tight.
- All tests pass before opening.
- Update `README.md` if user-facing behavior changes.
- Match existing code style. No reformatting unrelated files.
- No PyTorch or transformers dependencies.
- Python 3.10 to 3.12 only. Do not introduce 3.13+ syntax or dependencies.

## Reporting Issues

Use the [bug report template](https://github.com/gabrimatic/kokoro-mlx/issues/new?template=bug_report.yml). Include:

- Python version and MLX version
- macOS version and chip (e.g., macOS 15.0, M4)
- Steps to reproduce, expected vs. actual behavior

## Vulnerability Reporting

See [SECURITY.md](SECURITY.md). Do **not** open public issues for security vulnerabilities. Use GitHub's private vulnerability reporting.
