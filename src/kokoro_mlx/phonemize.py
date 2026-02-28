# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""G2P phonemization pipeline using misaki."""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MAX_TOKENS = 510  # context window is 512; 2 slots reserved for pad tokens


class Phonemizer:
    """Grapheme-to-phoneme pipeline wrapping misaki's English G2P."""

    def __init__(self, vocab: dict[str, int], language: str = "en") -> None:
        self._vocab = vocab
        self._language = language
        self._g2p = self._build_g2p(language)

    @staticmethod
    def _build_g2p(language: str):
        if language == "en":
            from misaki import en

            return en.G2P()
        raise ValueError(f"Unsupported language: {language!r}")

    def _ids_from_phonemes(self, phonemes: str) -> list[int]:
        ids = [self._vocab[c] for c in phonemes if c in self._vocab]
        return [0, *ids, 0]

    def phonemize(self, text: str) -> tuple[str, list[int]]:
        """Convert *text* to a phoneme string and token ID sequence.

        Returns a ``(phoneme_string, token_ids)`` tuple where ``token_ids``
        is padded with 0 at start and end.  Returns ``("", [])`` for empty
        input.
        """
        if not text or not text.strip():
            return "", []

        phonemes, _ = self._g2p(text)
        token_ids = self._ids_from_phonemes(phonemes)
        return phonemes, token_ids

    def phonemize_long(self, text: str) -> list[tuple[str, list[int]]]:
        """Phonemize *text*, chunking at sentence boundaries when the phoneme
        sequence would exceed the 512-token context window.

        Returns a list of ``(phoneme_string, token_ids)`` tuples, one per
        chunk.
        """
        if not text or not text.strip():
            return []

        # Split into sentences and accumulate until the limit is reached.
        sentences = _SENTENCE_BOUNDARY.split(text.strip())
        chunks: list[tuple[str, list[int]]] = []
        current_sentences: list[str] = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            candidate = " ".join(current_sentences + [sentence])
            phonemes, _ = self._g2p(candidate)
            vocab_ids = [self._vocab[c] for c in phonemes if c in self._vocab]

            if len(vocab_ids) > _MAX_TOKENS and current_sentences:
                # Flush the current accumulation before adding the new sentence.
                flush_text = " ".join(current_sentences)
                ph, _ = self._g2p(flush_text)
                chunks.append((ph, self._ids_from_phonemes(ph)))
                current_sentences = [sentence]
            else:
                current_sentences.append(sentence)

        if current_sentences:
            flush_text = " ".join(current_sentences)
            ph, _ = self._g2p(flush_text)
            chunks.append((ph, self._ids_from_phonemes(ph)))

        return chunks
