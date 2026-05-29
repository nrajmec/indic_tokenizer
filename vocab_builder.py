"""
Vocabulary builder for the Indic BPE Tokenizer.

Builds a base vocabulary seeded with:
  1. 256 ASCII code points  (chr(0) … chr(255))
  2. All *assigned* Unicode characters across every supported Indic script block
  3. Any additional characters discovered while scanning the training corpus
  4. Special tokens (injected after the above so BPE never splits them)
"""

import unicodedata
from typing import Dict, List, Optional, Tuple

from .constants import INDIC_UNICODE_RANGES, SPECIAL_TOKENS

# Public type aliases reused by bpe_tokenizer and chunked_trainer
VocabType = Dict[int, str]
InverseVocabType = Dict[str, int]


class VocabBuilder:
    """
    Builds and extends the tokenizer vocabulary.

    All public methods that modify the vocabulary do so in-place so that the
    same two dicts (vocab, inverse_vocab) flow through training without copying.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_base_vocab(self) -> Tuple[VocabType, InverseVocabType]:
        """
        Return an initial vocabulary seeded with ASCII + all assigned Indic chars.

        Iterates over every (start, end) range in INDIC_UNICODE_RANGES and
        includes only code points that unicodedata considers assigned (i.e.,
        unicodedata.name() succeeds without raising ValueError).

        Returns:
            (vocab, inverse_vocab) — bidirectional id ↔ token string mapping.
        """
        chars: List[str] = [chr(i) for i in range(256)]  # ASCII 0–255
        chars.extend(self._defined_indic_chars())

        vocab: VocabType = {}
        inverse_vocab: InverseVocabType = {}
        for char in chars:
            if char not in inverse_vocab:
                tid = len(vocab)
                vocab[tid] = char
                inverse_vocab[char] = tid

        return vocab, inverse_vocab

    def extend_from_text(
        self,
        text: str,
        vocab: VocabType,
        inverse_vocab: InverseVocabType,
    ) -> None:
        """
        Add characters found in *text* that are not yet in vocab.

        Iterating over sorted(set(text)) ensures deterministic IDs regardless
        of the order characters happen to appear in the corpus.

        Mutates vocab and inverse_vocab in-place.
        """
        for char in sorted(set(text)):
            if char not in inverse_vocab:
                tid = len(vocab)
                vocab[tid] = char
                inverse_vocab[char] = tid

    def add_special_tokens(
        self,
        vocab: VocabType,
        inverse_vocab: InverseVocabType,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Inject special tokens into vocab if not already present.

        Tokens are injected *after* all character-level entries so that their
        IDs are always in a predictable high range and BPE never tries to
        merge characters *into* a special token.

        Mutates vocab and inverse_vocab in-place.
        """
        tokens = special_tokens if special_tokens is not None else SPECIAL_TOKENS
        for token in tokens:
            if token not in inverse_vocab:
                tid = len(vocab)
                vocab[tid] = token
                inverse_vocab[token] = tid

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _defined_indic_chars() -> List[str]:
        """
        Return all assigned Unicode code points across every supported Indic
        script block.  Unassigned code points (holes in the block) are
        silently skipped via unicodedata.name() raising ValueError.
        """
        chars: List[str] = []
        for _script, (start, end) in INDIC_UNICODE_RANGES.items():
            for cp in range(start, end + 1):
                char = chr(cp)
                try:
                    unicodedata.name(char)
                    chars.append(char)
                except ValueError:
                    pass  # unassigned code point — skip
        return chars
