# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Soroush Yousefpour

"""G2P phonemization pipeline using misaki."""

from __future__ import annotations

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MAX_TOKENS = 510  # context window is 512; 2 slots reserved for pad tokens

VOICE_PREFIX_LANGUAGES: dict[str, str] = {
    "af": "en-us",
    "am": "en-us",
    "bf": "en-gb",
    "bm": "en-gb",
    "ef": "es",
    "em": "es",
    "ff": "fr-fr",
    "hf": "hi",
    "hm": "hi",
    "if": "it",
    "im": "it",
    "jf": "ja",
    "jm": "ja",
    "pf": "pt-br",
    "pm": "pt-br",
    "zf": "zh",
    "zm": "zh",
}

LANGUAGE_ALIASES: dict[str, str] = {
    "a": "en-us",
    "en": "en-us",
    "en-us": "en-us",
    "english": "en-us",
    "american english": "en-us",
    "b": "en-gb",
    "en-gb": "en-gb",
    "british english": "en-gb",
    "e": "es",
    "es": "es",
    "spanish": "es",
    "f": "fr-fr",
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
    "french": "fr-fr",
    "h": "hi",
    "hi": "hi",
    "hindi": "hi",
    "i": "it",
    "it": "it",
    "italian": "it",
    "j": "ja",
    "ja": "ja",
    "japanese": "ja",
    "p": "pt-br",
    "pt": "pt-br",
    "pt-br": "pt-br",
    "portuguese": "pt-br",
    "z": "zh",
    "zh": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "mandarin chinese": "zh",
}


def normalize_language(language: str | None) -> str:
    """Return the normalized Kokoro/misaki language code."""
    if language is None:
        return "en-us"
    key = language.strip().lower()
    return LANGUAGE_ALIASES.get(key, key)


def language_from_voice(voice: str) -> str:
    """Infer a normalized language code from a Kokoro voice name."""
    first_voice = voice.split(",", 1)[0].strip()
    return VOICE_PREFIX_LANGUAGES.get(first_voice[:2].lower(), "en-us")


class Phonemizer:
    """Grapheme-to-phoneme pipeline wrapping misaki's English G2P."""

    def __init__(self, vocab: dict[str, int], language: str = "en") -> None:
        self._vocab = vocab
        self._language = normalize_language(language)
        self._g2p = self._build_g2p(language)

    @staticmethod
    def _build_g2p(language: str):
        language = normalize_language(language)
        if language in ("en-us", "en-gb"):
            from misaki import en

            fallback = None
            try:
                from misaki import espeak

                fallback = espeak.EspeakFallback(british=language == "en-gb")
            except Exception:
                fallback = None

            # unk="" suppresses the '❓' sentinel that misaki emits for
            # unresolvable tokens (e.g. colons in time expressions).  Without
            # this, the sentinel passes through to _ids_from_phonemes where it
            # is silently dropped because '❓' is not in Kokoro's vocab,
            # potentially merging surrounding phonemes and distorting output.
            return en.G2P(unk="", british=language == "en-gb", fallback=fallback)
        if language == "ja":
            from misaki import ja

            return ja.JAG2P()
        if language == "zh":
            from misaki import zh

            return zh.ZHG2P()

        from misaki import espeak

        return espeak.EspeakG2P(language=language)

    def _phonemes_for_text(self, text: str) -> str:
        result = self._g2p(text)
        if isinstance(result, tuple):
            return str(result[0] or "")
        return str(result or "")

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

        phonemes = self._phonemes_for_text(text)
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
            phonemes = self._phonemes_for_text(candidate)
            vocab_ids = [self._vocab[c] for c in phonemes if c in self._vocab]

            if len(vocab_ids) > _MAX_TOKENS and current_sentences:
                # Flush the current accumulation before adding the new sentence.
                flush_text = " ".join(current_sentences)
                ph = self._phonemes_for_text(flush_text)
                chunks.append((ph, self._ids_from_phonemes(ph)))
                current_sentences = [sentence]
            else:
                current_sentences.append(sentence)

        if current_sentences:
            flush_text = " ".join(current_sentences)
            ph = self._phonemes_for_text(flush_text)
            chunks.append((ph, self._ids_from_phonemes(ph)))

        return chunks
