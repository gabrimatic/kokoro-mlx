# kokoro-mlx

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)]()
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-required-blue.svg)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10--3.12-blue.svg)]()

**Kokoro TTS inference on Apple Silicon via MLX.**

Pure MLX implementation of the full [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech pipeline. No PyTorch, no transformers, no third-party ML frameworks. Three lines to speak.

> This package provides inference code only. Model weights are developed by [hexgrad](https://huggingface.co/hexgrad) under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0) and downloaded separately from HuggingFace Hub on first use.

---

## Quick Start

**Apple Silicon required.** Python 3.10–3.12, MLX 0.22+.

```bash
pip install kokoro-mlx
```

```python
from kokoro_mlx import KokoroTTS

tts = KokoroTTS.from_pretrained()
tts.speak("Hello, world.")
```

Model weights download automatically from HuggingFace Hub on first use.

---

## Features

- **Fully on-device** via MLX, no server, no cloud, no network during inference
- **Pure implementation** with no PyTorch or transformers dependency
- **48 kHz output** via FFT-based upsampling from native 24 kHz, extending the signal to the sample rate modern DACs and headphones prefer
- **Float32 vocoder precision** where it matters: the heavy neural network runs in bf16 for speed, while the final waveform reconstruction (iSTFT, phase, overlap-add) runs in float32 to eliminate quantization noise
- **Gapless streaming** with a persistent audio stream and producer thread, so sentence boundaries are seamless instead of interrupted by stream teardown
- **54 voices** across American English, British English, and additional languages
- **WAV export** with a single method call
- **Thread-safe** with internal lock for concurrent callers
- **Context manager** for automatic resource cleanup
- **Speed control** from any multiplier

---

## API

### `KokoroTTS.from_pretrained(model_id_or_path)`

Load a model from a local directory or the HuggingFace Hub.

```python
tts = KokoroTTS.from_pretrained()
# or a specific repo
tts = KokoroTTS.from_pretrained("mlx-community/Kokoro-82M-bf16")
# or a local directory
tts = KokoroTTS.from_pretrained("/path/to/model")
```

### `tts.generate(text, voice, speed, sample_rate) -> TTSResult`

Synthesize text and return a `TTSResult`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text to synthesize |
| `voice` | `str` | `"af_heart"` | Voice name (see [Available Voices](#available-voices)) |
| `speed` | `float` | `1.0` | Speaking rate multiplier (>1 faster, <1 slower) |
| `sample_rate` | `int` | `24000` | Output sample rate: 24000 (native) or 48000 (2x upsampled) |

### `tts.generate_stream(text, voice, speed, sample_rate) -> Iterator[np.ndarray]`

Synthesize text and yield audio chunks sentence by sentence. Lower latency than `generate` for longer inputs.

### `tts.speak(text, voice, speed, stream, stop_event, sample_rate)`

Synthesize and immediately play text through the speakers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | Input text to synthesize |
| `voice` | `str` | `"af_heart"` | Voice name |
| `speed` | `float` | `1.0` | Speaking rate multiplier |
| `stream` | `bool` | `False` | Play chunk-by-chunk for lower latency |
| `stop_event` | `threading.Event` or `None` | `None` | Set to interrupt playback |
| `sample_rate` | `int` | `24000` | Output sample rate: 24000 or 48000 |

### `tts.save(text, path, voice, speed, sample_rate) -> TTSResult`

Synthesize text and write audio to a WAV file.

```python
result = tts.save("Hello, world.", "output.wav", sample_rate=48000)
```

### `tts.list_voices() -> list[str]`

Return a sorted list of all available voice names.

```python
voices = tts.list_voices()
# ['af_alloy', 'af_aoede', 'af_bella', ...]
```

### `tts.close()`

Release held resources. Called automatically when using the context manager.

```python
with KokoroTTS.from_pretrained() as tts:
    tts.save("Hello, world.", "output.wav")
```

### `TTSResult`

```python
@dataclass
class TTSResult:
    audio: np.ndarray   # float32
    sample_rate: int    # 24000 or 48000
    duration: float     # seconds
    voice: str          # voice name used
```

---

## Available Voices

Voice names follow a prefix convention: the first two characters identify the accent and gender.

| Prefix | Description |
|--------|-------------|
| `af_` | American English, Female |
| `am_` | American English, Male |
| `bf_` | British English, Female |
| `bm_` | British English, Male |
| `ef_` | Other English, Female |
| `em_` | Other English, Male |
| `ff_` | French, Female |
| `hf_` | Hindi, Female |
| `hm_` | Hindi, Male |
| `if_` | Italian, Female |
| `im_` | Italian, Male |
| `jf_` | Japanese, Female |
| `jm_` | Japanese, Male |
| `pf_` | Portuguese, Female |
| `pm_` | Portuguese, Male |
| `zf_` | Chinese Mandarin, Female |
| `zm_` | Chinese Mandarin, Male |

**American English (Female):** `af_alloy`, `af_aoede`, `af_bella`, `af_heart` (default), `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`

**American English (Male):** `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**British English (Female):** `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`

**British English (Male):** `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

---

## Architecture

```
Text Input
  │
  ▼
G2P / Phonemizer (misaki)
  │
  ▼
Phoneme Sequence
  │
  ▼
TextEncoder (PL-BERT / ALBERT, 12 layers, 768 hidden)
  │
  ▼
ProsodyPredictor (duration + pitch)
  │
  ├── Voice Style Vector (per-voice, 256-dim)
  │
  ▼
Decoder (StyleTTS2-style, AdaIN + residual blocks) [bf16]
  │
  ▼
ISTFTNet Vocoder (80-bin mel → waveform) [float32]
  │
  ▼
Optional 2x FFT upsample (24 kHz → 48 kHz)
  │
  ▼
TTSResult { audio float32, duration, voice }
```

The neural network runs in bf16 for throughput. At the vocoder boundary, the signal is promoted to float32 for the final waveform reconstruction: exponential magnitude recovery, phase extraction, inverse DFT, and overlap-add synthesis. This split keeps inference fast while preserving the precision that the iSTFT path requires.

---

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Python 3.10–3.12
- MLX 0.22+

---

## Development

```bash
git clone https://github.com/gabrimatic/kokoro-mlx.git
cd kokoro-mlx
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Skip model-loading tests with `-m "not slow"`.

---

## Credits

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by [hexgrad](https://huggingface.co/hexgrad) · [MLX](https://github.com/ml-explore/mlx) by [Apple](https://ml-explore.github.io/mlx/) · [misaki](https://github.com/hexgrad/misaki) G2P by hexgrad · MLX weights from [mlx-community](https://huggingface.co/mlx-community)

<details>
<summary><strong>Legal notices</strong></summary>

### Model License

This package provides inference code only. It does not include model weights.

The [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model weights are developed by [hexgrad](https://huggingface.co/hexgrad) and released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The [MLX conversion](https://huggingface.co/mlx-community/Kokoro-82M-bf16) is hosted by [mlx-community](https://huggingface.co/mlx-community) under the same license. By downloading and using the model weights, you agree to the terms of the Apache 2.0 license.

### Trademarks

"MLX" is a trademark of Apple Inc. "HuggingFace" is a trademark of Hugging Face, Inc.

This project is not affiliated with, endorsed by, or sponsored by Apple, Hugging Face, or any other trademark holder. All trademark names are used solely to describe compatibility with their respective technologies.

### Third-Party Licenses

This project depends on:

| Package | License |
|---------|---------|
| [mlx](https://github.com/ml-explore/mlx) | MIT |
| [numpy](https://numpy.org) | BSD-3-Clause |
| [huggingface-hub](https://github.com/huggingface/huggingface_hub) | Apache-2.0 |
| [soundfile](https://github.com/bastibe/python-soundfile) | BSD-3-Clause |
| [misaki](https://github.com/hexgrad/misaki) | Apache-2.0 |
| [sounddevice](https://python-sounddevice.readthedocs.io) (optional) | MIT |

</details>

## License

This inference code is released under the MIT License. See [LICENSE](LICENSE) for details.

The model weights have their own license (Apache 2.0). See [Model License](#legal-notices) above.

---

Created by [Soroush Yousefpour](https://gabrimatic.info)

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/gabrimatic)
