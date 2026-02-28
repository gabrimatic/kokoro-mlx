# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""KokoroModel: full Kokoro-82M TTS model with MLX inference."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import KokoroConfig
from .istftnet import Decoder
from .modules import ALBERT, BiLSTM, ProsodyPredictor, TextEncoder

# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def _load_safetensors(path: str | Path) -> dict[str, mx.array]:
    """Load a safetensors file and return {key: mx.array}."""
    from safetensors.numpy import load_file
    raw = load_file(str(path))
    return {k: mx.array(v) for k, v in raw.items()}


def _set_nested(obj, key_parts: list[str], value: mx.array) -> None:
    """Set a nested attribute by following a dot-separated key path."""
    for part in key_parts[:-1]:
        # Handle integer indices (list items)
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    last = key_parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def _get_nested(obj, key_parts: list[str]):
    """Get a nested attribute."""
    for part in key_parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def _load_bilstm_weights(bilstm: BiLSTM, prefix: str, weights: dict[str, mx.array]) -> list[str]:
    """Load PyTorch BiLSTM weights into our BiLSTM module. Returns loaded keys.

    MLX LSTM forward pass: x = addmm(bias, x, Wx.T), so Wx must have shape (4H, input).
    PyTorch weight_ih has shape (4H, input) — load directly without transposing.
    Same for weight_hh (4H, hidden).
    MLX uses a single bias = bias_ih + bias_hh.
    """
    loaded = []
    # Forward LSTM
    fwd_bias_ih = fwd_bias_hh = None
    for pt_key, mlx_attr in [
        (f"{prefix}.weight_ih_l0", "Wx"),
        (f"{prefix}.weight_hh_l0", "Wh"),
    ]:
        if pt_key in weights:
            loaded.append(pt_key)
            setattr(bilstm.fwd, mlx_attr, weights[pt_key])

    if f"{prefix}.bias_ih_l0" in weights:
        fwd_bias_ih = weights[f"{prefix}.bias_ih_l0"]
        loaded.append(f"{prefix}.bias_ih_l0")
    if f"{prefix}.bias_hh_l0" in weights:
        fwd_bias_hh = weights[f"{prefix}.bias_hh_l0"]
        loaded.append(f"{prefix}.bias_hh_l0")
    if fwd_bias_ih is not None and fwd_bias_hh is not None:
        bilstm.fwd.bias = fwd_bias_ih + fwd_bias_hh

    # Backward LSTM
    bwd_bias_ih = bwd_bias_hh = None
    for pt_key, mlx_attr in [
        (f"{prefix}.weight_ih_l0_reverse", "Wx"),
        (f"{prefix}.weight_hh_l0_reverse", "Wh"),
    ]:
        if pt_key in weights:
            loaded.append(pt_key)
            setattr(bilstm.bwd, mlx_attr, weights[pt_key])

    if f"{prefix}.bias_ih_l0_reverse" in weights:
        bwd_bias_ih = weights[f"{prefix}.bias_ih_l0_reverse"]
        loaded.append(f"{prefix}.bias_ih_l0_reverse")
    if f"{prefix}.bias_hh_l0_reverse" in weights:
        bwd_bias_hh = weights[f"{prefix}.bias_hh_l0_reverse"]
        loaded.append(f"{prefix}.bias_hh_l0_reverse")
    if bwd_bias_ih is not None and bwd_bias_hh is not None:
        bilstm.bwd.bias = bwd_bias_ih + bwd_bias_hh

    return loaded


def _load_wn_conv_weights(conv, prefix: str, weights: dict[str, mx.array]) -> list[str]:
    """Load weight_g, weight_v, bias into a WeightNormConv1d/WeightNormConvTranspose1d."""
    loaded = []
    for suffix, attr in [("weight_g", "weight_g"), ("weight_v", "weight_v"), ("bias", "bias")]:
        key = f"{prefix}.{suffix}"
        if key in weights:
            setattr(conv, attr, weights[key])
            loaded.append(key)
    return loaded


def _load_linear_weights(linear: nn.Linear, prefix: str, weights: dict[str, mx.array]) -> list[str]:
    """Load weight + bias into an nn.Linear. PyTorch linear weight: (out, in), MLX same."""
    loaded = []
    w_key = f"{prefix}.weight"
    b_key = f"{prefix}.bias"
    if w_key in weights:
        linear.weight = weights[w_key]
        loaded.append(w_key)
    if b_key in weights:
        linear.bias = weights[b_key]
        loaded.append(b_key)
    return loaded


def _load_layernorm_weights(ln: nn.LayerNorm, prefix: str, weights: dict[str, mx.array]) -> list[str]:
    """Load weight/bias into an nn.LayerNorm."""
    loaded = []
    for pt_name, mlx_attr in [("weight", "weight"), ("bias", "bias")]:
        key = f"{prefix}.{pt_name}"
        if key in weights:
            setattr(ln, mlx_attr, weights[key])
            loaded.append(key)
    return loaded


def load_weights(model: "KokoroModel", weights: dict[str, mx.array]) -> int:
    """Map all 548 PyTorch weight keys to MLX model attributes. Returns loaded count."""
    loaded = set()

    def mark(*keys):
        for k in keys:
            if k in weights:
                loaded.add(k)

    # --- BERT embeddings ---
    model.bert.embeddings.word_embeddings.weight = weights["bert.embeddings.word_embeddings.weight"]
    mark("bert.embeddings.word_embeddings.weight")
    model.bert.embeddings.position_embeddings.weight = weights["bert.embeddings.position_embeddings.weight"]
    mark("bert.embeddings.position_embeddings.weight")
    model.bert.embeddings.token_type_embeddings.weight = weights["bert.embeddings.token_type_embeddings.weight"]
    mark("bert.embeddings.token_type_embeddings.weight")
    # ALBERT LayerNorm: PyTorch uses weight/bias names
    model.bert.embeddings.LayerNorm.weight = weights["bert.embeddings.LayerNorm.weight"]
    model.bert.embeddings.LayerNorm.bias = weights["bert.embeddings.LayerNorm.bias"]
    mark("bert.embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.bias")

    # --- BERT encoder ---
    enc_pfx = "bert.encoder.albert_layer_groups.0.albert_layers.0"
    layer = model.bert.encoder.albert_layer_groups[0].albert_layers[0]

    # Attention
    for pt_name, mlx_obj in [("query", layer.attention.query), ("key", layer.attention.key),
                               ("value", layer.attention.value), ("dense", layer.attention.dense)]:
        for k in _load_linear_weights(mlx_obj, f"{enc_pfx}.attention.{pt_name}", weights):
            loaded.add(k)

    for k in _load_layernorm_weights(layer.attention.LayerNorm, f"{enc_pfx}.attention.LayerNorm", weights):
        loaded.add(k)

    for k in _load_linear_weights(layer.ffn, f"{enc_pfx}.ffn", weights):
        loaded.add(k)
    for k in _load_linear_weights(layer.ffn_output, f"{enc_pfx}.ffn_output", weights):
        loaded.add(k)
    for k in _load_layernorm_weights(layer.full_layer_layer_norm, f"{enc_pfx}.full_layer_layer_norm", weights):
        loaded.add(k)

    # Embedding hidden mapping
    for k in _load_linear_weights(model.bert.encoder.embedding_hidden_mapping_in,
                                   "bert.encoder.embedding_hidden_mapping_in", weights):
        loaded.add(k)

    # Pooler (not used in forward but weights exist)
    for k in _load_linear_weights(model.bert.pooler, "bert.pooler", weights):
        loaded.add(k)

    # --- bert_encoder (Linear 768→512) ---
    model.bert_encoder.weight = weights["bert_encoder.weight"]
    model.bert_encoder.bias = weights["bert_encoder.bias"]
    mark("bert_encoder.weight", "bert_encoder.bias")

    # --- text_encoder ---
    te = model.text_encoder
    model.text_encoder.embedding.weight = weights["text_encoder.embedding.weight"]
    mark("text_encoder.embedding.weight")

    for i in range(3):
        conv, ln = te.cnn[i]
        for k in _load_wn_conv_weights(conv, f"text_encoder.cnn.{i}.0", weights):
            loaded.add(k)
        # LayerNorm uses gamma/beta naming
        for pt_name, attr in [("gamma", "gamma"), ("beta", "beta")]:
            key = f"text_encoder.cnn.{i}.1.{pt_name}"
            if key in weights:
                setattr(ln, attr, weights[key])
                loaded.add(key)

    for k in _load_bilstm_weights(te.lstm, "text_encoder.lstm", weights):
        loaded.add(k)

    # --- predictor ---
    pred = model.predictor

    # text_encoder (DurationEncoder) LSTMs
    for idx in range(len(pred.text_encoder.lstms)):
        block = pred.text_encoder.lstms[idx]
        pfx = f"predictor.text_encoder.lstms.{idx}"
        if isinstance(block, BiLSTM):
            for k in _load_bilstm_weights(block, pfx, weights):
                loaded.add(k)
        else:
            # AdaLayerNorm: fc.weight, fc.bias
            for k in _load_linear_weights(block.fc, f"{pfx}.fc", weights):
                loaded.add(k)

    # predictor.lstm (BiLSTM for duration)
    for k in _load_bilstm_weights(pred.lstm, "predictor.lstm", weights):
        loaded.add(k)

    # predictor.shared (BiLSTM for F0/N)
    for k in _load_bilstm_weights(pred.shared, "predictor.shared", weights):
        loaded.add(k)

    # duration_proj
    for k in _load_linear_weights(pred.duration_proj.linear_layer, "predictor.duration_proj.linear_layer", weights):
        loaded.add(k)

    # F0 and N AdainResBlk1d blocks
    for branch_name, branch in [("F0", pred.F0), ("N", pred.N)]:
        for i, blk in enumerate(branch):
            pfx = f"predictor.{branch_name}.{i}"
            _load_adain_resblk1d(blk, pfx, weights, loaded)

    # F0_proj and N_proj: plain Conv1d
    # PyTorch weight: (1, 256, 1) → MLX Conv1d weight: (out, kernel, in) = (1, 1, 256)
    _load_plain_conv1d(pred.F0_proj, "predictor.F0_proj", weights, loaded)
    _load_plain_conv1d(pred.N_proj, "predictor.N_proj", weights, loaded)

    # --- decoder ---
    dec = model.decoder

    for k in _load_wn_conv_weights(dec.F0_conv, "decoder.F0_conv", weights):
        loaded.add(k)
    for k in _load_wn_conv_weights(dec.N_conv, "decoder.N_conv", weights):
        loaded.add(k)
    for k in _load_wn_conv_weights(dec.asr_res[0], "decoder.asr_res.0", weights):
        loaded.add(k)

    _load_adain_resblk1d(dec.encode, "decoder.encode", weights, loaded)
    for i, blk in enumerate(dec.decode):
        _load_adain_resblk1d(blk, f"decoder.decode.{i}", weights, loaded)

    # Generator
    gen = dec.generator
    for k in _load_wn_conv_weights(gen.conv_post, "decoder.generator.conv_post", weights):
        loaded.add(k)

    # m_source.l_linear
    for k in _load_linear_weights(gen.m_source.l_linear, "decoder.generator.m_source.l_linear", weights):
        loaded.add(k)

    # ups
    for i, up in enumerate(gen.ups):
        for k in _load_wn_conv_weights(up, f"decoder.generator.ups.{i}", weights):
            loaded.add(k)

    # noise_convs (plain Conv1d)
    for i, nc in enumerate(gen.noise_convs):
        _load_plain_conv1d(nc, f"decoder.generator.noise_convs.{i}", weights, loaded)

    # noise_res and resblocks
    for i, nr in enumerate(gen.noise_res):
        _load_adain_resblk1_weights(nr, f"decoder.generator.noise_res.{i}", weights, loaded)
    for i, rb in enumerate(gen.resblocks):
        _load_adain_resblk1_weights(rb, f"decoder.generator.resblocks.{i}", weights, loaded)

    return len(loaded)


def _load_adain_resblk1d(blk, pfx: str, weights: dict, loaded: set) -> None:
    """Load AdainResBlk1d weights."""
    for k in _load_wn_conv_weights(blk.conv1, f"{pfx}.conv1", weights):
        loaded.add(k)
    for k in _load_wn_conv_weights(blk.conv2, f"{pfx}.conv2", weights):
        loaded.add(k)
    if blk.learned_sc:
        for k in _load_wn_conv_weights(blk.conv1x1, f"{pfx}.conv1x1", weights):
            loaded.add(k)
    _load_adain1d(blk.norm1, f"{pfx}.norm1", weights, loaded)
    _load_adain1d(blk.norm2, f"{pfx}.norm2", weights, loaded)
    if blk.pool is not None:
        for k in _load_wn_conv_weights(blk.pool, f"{pfx}.pool", weights):
            loaded.add(k)


def _load_adain1d(adain, pfx: str, weights: dict, loaded: set) -> None:
    """Load AdaIN1d fc weights."""
    for k in _load_linear_weights(adain.fc, f"{pfx}.fc", weights):
        loaded.add(k)


def _load_adain_resblk1_weights(blk, pfx: str, weights: dict, loaded: set) -> None:
    """Load AdaINResBlock1 (Generator-level) weights."""
    for i in range(3):
        for k in _load_wn_conv_weights(blk.convs1[i], f"{pfx}.convs1.{i}", weights):
            loaded.add(k)
        for k in _load_wn_conv_weights(blk.convs2[i], f"{pfx}.convs2.{i}", weights):
            loaded.add(k)
        _load_adain1d(blk.adain1[i], f"{pfx}.adain1.{i}", weights, loaded)
        _load_adain1d(blk.adain2[i], f"{pfx}.adain2.{i}", weights, loaded)
        # alpha parameters
        for alpha_name, alpha_list in [("alpha1", blk.alpha1), ("alpha2", blk.alpha2)]:
            key = f"{pfx}.{alpha_name}.{i}"
            if key in weights:
                alpha_list[i] = weights[key]
                loaded.add(key)


def _load_plain_conv1d(conv: nn.Conv1d, pfx: str, weights: dict, loaded: set) -> None:
    """Load plain (non-weight-normed) Conv1d. PyTorch: weight (out, in, kernel), bias (out,).
    MLX: weight (out, kernel, in).
    """
    w_key = f"{pfx}.weight"
    b_key = f"{pfx}.bias"
    if w_key in weights:
        w = weights[w_key]  # (out, in, kernel)
        # Transpose to (out, kernel, in)
        conv.weight = w.transpose(0, 2, 1)
        loaded.add(w_key)
    if b_key in weights:
        conv.bias = weights[b_key]
        loaded.add(b_key)


# ---------------------------------------------------------------------------
# KokoroModel
# ---------------------------------------------------------------------------


class KokoroModel(nn.Module):
    """Kokoro-82M TTS model implemented in MLX."""

    def __init__(self, config: KokoroConfig):
        super().__init__()
        self.config = config
        self.vocab = config.vocab

        # ALBERT text encoder
        self.bert = ALBERT(
            vocab_size=config.n_token,
            embedding_size=128,
            hidden_size=config.plbert.hidden_size,
            intermediate_size=config.plbert.intermediate_size,
            num_attention_heads=config.plbert.num_attention_heads,
            num_hidden_layers=config.plbert.num_hidden_layers,
            max_position_embeddings=config.plbert.max_position_embeddings,
            dropout=config.plbert.dropout,
        )
        self.bert_encoder = nn.Linear(config.plbert.hidden_size, config.hidden_dim)

        self.predictor = ProsodyPredictor(
            style_dim=config.style_dim,
            d_hid=config.hidden_dim,
            nlayers=config.n_layer,
            max_dur=config.max_dur,
            dropout=config.dropout,
        )
        self.text_encoder = TextEncoder(
            channels=config.hidden_dim,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )
        self.decoder = Decoder(
            dim_in=config.hidden_dim,
            style_dim=config.style_dim,
            dim_out=config.n_mels,
            resblock_kernel_sizes=config.istftnet.resblock_kernel_sizes,
            upsample_rates=config.istftnet.upsample_rates,
            upsample_initial_channel=config.istftnet.upsample_initial_channel,
            resblock_dilation_sizes=config.istftnet.resblock_dilation_sizes,
            upsample_kernel_sizes=config.istftnet.upsample_kernel_sizes,
            gen_istft_n_fft=config.istftnet.gen_istft_n_fft,
            gen_istft_hop_size=config.istftnet.gen_istft_hop_size,
        )

    @classmethod
    def from_pretrained(cls, model_id_or_path: str | Path) -> "KokoroModel":
        """Load model from a local safetensors file or HuggingFace Hub repo."""
        path = Path(model_id_or_path)

        if path.is_file() and path.suffix == ".safetensors":
            # Direct safetensors file — need a config
            raise ValueError(
                "Pass a directory or HF repo ID, not a raw .safetensors file. "
                "Use KokoroConfig.from_pretrained() + KokoroModel(config) + load_weights()."
            )

        if path.is_dir():
            config_path = path / "config.json"
            weights_path = path / "kokoro-v1_0.safetensors"
            if not config_path.exists():
                raise FileNotFoundError(f"config.json not found in {path}")
            if not weights_path.exists():
                # Try alternative names
                st_files = list(path.glob("*.safetensors"))
                if not st_files:
                    raise FileNotFoundError(f"No .safetensors file found in {path}")
                weights_path = st_files[0]
        else:
            # HuggingFace Hub download
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(repo_id=str(model_id_or_path), filename="config.json")
            weights_file = hf_hub_download(repo_id=str(model_id_or_path), filename="kokoro-v1_0.safetensors")
            config_path = Path(config_file)
            weights_path = Path(weights_file)

        config = KokoroConfig.from_file(config_path)
        model = cls(config)
        weights = _load_safetensors(weights_path)
        load_weights(model, weights)
        mx.eval(model.parameters())
        return model

    def forward(self, phonemes: str, ref_s: mx.array, speed: float = 1.0) -> mx.array:
        """Generate audio from phoneme string and reference style vector.

        Args:
            phonemes: Phoneme string (each character mapped via self.vocab).
            ref_s: Reference style vector, shape (1, 256) or (256,).
            speed: Speaking rate multiplier (lower = slower).

        Returns:
            1D audio array.
        """
        if ref_s.ndim == 1:
            ref_s = ref_s[None, :]  # (1, 256)

        # Map phonemes to IDs
        input_ids_list = [self.vocab[c] for c in phonemes if c in self.vocab]
        input_ids = mx.array([[0, *input_ids_list, 0]])  # (1, T) with BOS/EOS pads
        T = input_ids.shape[1]
        input_lengths = mx.array([T])

        # No masking for single-sample inference
        text_mask = mx.zeros((1, T), dtype=mx.bool_)

        # --- ALBERT ---
        bert_dur = self.bert(input_ids)  # (1, T, 768)
        d_en = self.bert_encoder(bert_dur).transpose(0, 2, 1)  # (1, 512, T)

        # --- Style split ---
        s_prosody = ref_s[:, 128:]   # (1, 128) for prosody/predictor
        s_decoder = ref_s[:, :128]   # (1, 128) for decoder

        # --- Duration prediction ---
        d = self.predictor.text_encoder(d_en, s_prosody, input_lengths, text_mask)
        # d: (1, T, d_hid+style_dim) — DurationEncoder output

        # For LSTM: the DurationEncoder returns features with style concat
        # Predictor LSTM needs (1, T, d_hid+style_dim) input
        x = self.predictor.lstm(d)  # (1, T, 512)
        duration = self.predictor.duration_proj(x)  # (1, T, 50)
        duration = mx.sigmoid(duration).sum(axis=-1) / speed  # (1, T)
        pred_dur = mx.clip(mx.round(duration), 0, None).astype(mx.int32).squeeze(0)  # (T,)
        # Ensure minimum 1 frame per token
        pred_dur = mx.maximum(pred_dur, mx.ones_like(pred_dur))

        # Evaluate pred_dur to numpy for building alignment matrix
        mx.eval(pred_dur)
        pred_dur_np = np.array(pred_dur.tolist())
        total_frames = int(pred_dur_np.sum())

        # --- Alignment matrix ---
        pred_aln = np.zeros((T, total_frames), dtype=np.float32)
        col = 0
        for i, dur in enumerate(pred_dur_np):
            pred_aln[i, col:col + dur] = 1.0
            col += dur
        pred_aln_trg = mx.array(pred_aln)[None]  # (1, T, T')

        # --- Expand features ---
        # d: (1, T, d_hid+style_dim), transpose to (1, d_hid+style_dim, T) for matmul
        d_t = d.transpose(0, 2, 1)  # (1, d_hid+style_dim, T)
        en = d_t @ pred_aln_trg     # (1, d_hid+style_dim, T')

        # F0Ntrain expects (B, d_hid+style_dim, T') — full en with style
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s_prosody)

        # --- Text encoder ---
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)  # (1, 512, T)
        asr = t_en @ pred_aln_trg  # (1, 512, T')

        # --- Decode ---
        audio = self.decoder(asr, F0_pred, N_pred, s_decoder)  # (1, 1, samples)
        return audio.squeeze()  # 1D
