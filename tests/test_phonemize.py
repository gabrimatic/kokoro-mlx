# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""Tests for the Phonemizer G2P pipeline."""

import json
from pathlib import Path

import pytest

from kokoro_mlx.phonemize import Phonemizer

# Load real vocab from the cached model.
_MODEL_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--mlx-community--Kokoro-82M-bf16/snapshots/a71e4d38b236d968966a2002c4c895dbd12b1c3c"
_CONFIG_PATH = _MODEL_SNAPSHOT / "config.json"


def _load_vocab() -> dict[str, int]:
    cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return cfg["vocab"]


@pytest.fixture(scope="module")
def vocab() -> dict[str, int]:
    return _load_vocab()


@pytest.fixture(scope="module")
def phonemizer(vocab) -> Phonemizer:
    return Phonemizer(vocab=vocab)


@pytest.mark.slow
class TestPhonemize:
    def test_hello_world_returns_phonemes(self, phonemizer):
        phonemes, token_ids = phonemizer.phonemize("Hello, world.")
        assert isinstance(phonemes, str)
        assert len(phonemes) > 0
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0

    def test_token_ids_are_valid(self, phonemizer, vocab):
        _, token_ids = phonemizer.phonemize("Hello, world.")
        max_id = max(vocab.values())
        # Skip the pad tokens (0) at start/end when checking range.
        inner_ids = token_ids[1:-1]
        assert all(1 <= tid <= max_id for tid in inner_ids)

    def test_pad_tokens_at_boundaries(self, phonemizer):
        _, token_ids = phonemizer.phonemize("Hello, world.")
        assert token_ids[0] == 0
        assert token_ids[-1] == 0

    def test_empty_text_returns_empty(self, phonemizer):
        phonemes, token_ids = phonemizer.phonemize("")
        assert phonemes == ""
        assert token_ids == []

    def test_whitespace_only_returns_empty(self, phonemizer):
        phonemes, token_ids = phonemizer.phonemize("   ")
        assert phonemes == ""
        assert token_ids == []

    def test_long_text_chunking(self, phonemizer):
        # Build a text long enough to exceed 510 phoneme tokens.
        sentence = "The quick brown fox jumps over the lazy dog. "
        long_text = sentence * 40
        chunks = phonemizer.phonemize_long(long_text)
        assert len(chunks) > 1
        for ph, ids in chunks:
            assert isinstance(ph, str)
            assert len(ph) > 0
            assert ids[0] == 0
            assert ids[-1] == 0
            # Each chunk must fit within the context window.
            assert len(ids) <= 512

    def test_short_text_single_chunk(self, phonemizer):
        chunks = phonemizer.phonemize_long("Hello, world.")
        assert len(chunks) == 1
        ph, ids = chunks[0]
        assert len(ph) > 0
        assert ids[0] == 0
        assert ids[-1] == 0
