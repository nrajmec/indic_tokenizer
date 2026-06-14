"""
IndicBPETokenizer — main public class for BPE-based Indic tokenization.

Space / newline convention 
--------------------------------------------------
Spaces and newlines are kept as *literal* characters inside tokens.
A token like " नमस्ते" contains a real leading space — no Ġ/Ċ substitution.
Decoding is therefore a trivial string join with no post-processing.

Responsibilities
----------------
train()     — learn BPE merges from a corpus (small / medium datasets)
train_from_file() — convenience wrapper that auto-loads any supported format
encode()    — text → List[int]  (with optional special-token pass-through)
decode()    — List[int] → text  (plain string join)
save()      — persist vocab + merges to two JSON files
load()      — restore from those JSON files

For corpora > ~1 GB use ChunkedBPETrainer (chunked_trainer.py) instead of
train(), which stores the full token ID sequence in RAM.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from .bpe_trainer import BPETrainer, MergesType, PairType
from .constants import SPECIAL_TOKENS
from .data_loader import IndicDataLoader
from .preprocessor import pretokenize
from .vocab_builder import VocabBuilder, InverseVocabType, VocabType


class IndicBPETokenizer:
    """
    Byte-Pair Encoding tokenizer for Indian language text.

    Supports all major Indic scripts: Devanagari, Bengali, Gurmukhi,
    Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala, Tibetan,
    as well as ASCII and mixed-script text.

    Usage
    -----
    >>> tok = IndicBPETokenizer()
    >>> tok.train(text, vocab_size=16_000)
    >>> ids = tok.encode("नमस्ते दुनिया")
    >>> tok.decode(ids)
    'नमस्ते दुनिया'
    >>> tok.save("vocab.json", "merges.json")

    >>> tok2 = IndicBPETokenizer()
    >>> tok2.load("vocab.json", "merges.json")
    """

    def __init__(self) -> None:
        self.vocab: VocabType = {}
        self.inverse_vocab: InverseVocabType = {}
        self.bpe_merges: MergesType = {}

        self._vocab_builder = VocabBuilder()
        self._trainer = BPETrainer()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Current number of tokens in the vocabulary."""
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Train BPE from *text* using sequence-based BPE.

        Args:
            text: Raw UTF-8 training corpus (any Indic language or mixed).
            vocab_size: Target vocabulary size.
            special_tokens: Override the default special tokens.
            verbose: Show tqdm progress bar during training.

        Note:
            For corpora larger than ~1 GB, use ChunkedBPETrainer instead —
            it runs word-frequency BPE which needs <1 GB RAM even for 18 GB text.
        """
        if special_tokens is None:
            special_tokens = list(SPECIAL_TOKENS)

        # Seed vocab: ASCII + all Indic script characters + corpus chars + specials
        self.vocab, self.inverse_vocab = self._vocab_builder.build_base_vocab()
        self._vocab_builder.extend_from_text(text, self.vocab, self.inverse_vocab)
        self._vocab_builder.add_special_tokens(
            self.vocab, self.inverse_vocab, special_tokens
        )

        if verbose:
            print(f"Base vocab: {self.vocab_size:,} tokens  |  Target: {vocab_size:,}")
            if len(text) > 50_000_000:
                print(
                    f"WARNING: corpus is {len(text):,} chars. Sequence-based BPE "
                    "stores the full token ID list in RAM. Consider ChunkedBPETrainer "
                    "for large corpora (chunked_trainer.py)."
                )

        # Encode every character → its token ID
        try:
            from tqdm import tqdm  # type: ignore[import]
            char_iter = tqdm(text, desc="Encoding corpus", unit=" chars",
                             unit_scale=True, total=len(text))
        except ImportError:
            char_iter = text  # type: ignore[assignment]

        token_ids: List[int] = [self.inverse_vocab[c] for c in char_iter]

        self.bpe_merges = self._trainer.train(
            token_ids,
            self.vocab,
            self.inverse_vocab,
            target_vocab_size=vocab_size,
            verbose=verbose,
        )

        if verbose:
            print(f"Training complete. Final vocab size: {self.vocab_size:,}")

    def train_from_file(
        self,
        path: Union[str, Path],
        vocab_size: int,
        text_column: str = "text",
        special_tokens: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """
        Load a corpus from *path* then call train().

        Supports .parquet, .csv, .txt, .json, .jsonl.

        Args:
            path: Path to the corpus file.
            vocab_size: Target vocabulary size.
            text_column: Column / key for text in tabular / JSON files.
            special_tokens: Override the default special tokens.
            max_samples: Cap on rows / documents loaded (None → all).
            verbose: Show training progress.
        """
        loader = IndicDataLoader(text_column=text_column, max_samples=max_samples)
        text = loader.load(path)
        self.train(text, vocab_size=vocab_size, special_tokens=special_tokens,
                   verbose=verbose)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        allowed_special: Optional[Set[str]] = None,
    ) -> List[int]:
        """
        Encode *text* into a list of token IDs.

        Args:
            text: Input string (any Indic script, ASCII, special tokens, mixed).
            allowed_special: Set of special tokens to recognise and pass through
                             as single IDs.  Pass set() to disable.
                             Defaults to all tokens in SPECIAL_TOKENS.

        Returns:
            List of integer token IDs.
        """
        if allowed_special is None:
            allowed_special = set(SPECIAL_TOKENS)

        token_ids: List[int] = []

        if allowed_special:
            for segment, is_special in self._split_on_special_tokens(
                text, allowed_special
            ):
                if is_special:
                    tid = self.inverse_vocab.get(segment)
                    if tid is None:
                        raise ValueError(
                            f"Special token {segment!r} not found in vocabulary."
                        )
                    token_ids.append(tid)
                else:
                    token_ids.extend(self._encode_ordinary(segment))
        else:
            token_ids = self._encode_ordinary(text)

        return token_ids

    def _encode_ordinary(self, text: str) -> List[int]:
        """Encode plain text (no special-token detection)."""
        token_ids: List[int] = []
        for chunk in pretokenize(text):
            tid = self.inverse_vocab.get(chunk)
            if tid is not None:
                token_ids.append(tid)
            else:
                token_ids.extend(self._apply_bpe(chunk))
        return token_ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Because tokens contain literal spaces and newlines, decoding is a
        plain concatenation — no marker substitution required.

        Args:
            token_ids: Sequence of integer token IDs.

        Returns:
            Decoded UTF-8 string.

        Raises:
            ValueError: If any token ID is not in the vocabulary.
        """
        parts: List[str] = []
        for tid in token_ids:
            token = self.vocab.get(tid)
            if token is None:
                raise ValueError(f"Token ID {tid} not found in vocabulary.")
            parts.append(token)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, vocab_path: str, merges_path: str) -> None:
        """
        Save vocabulary and BPE merges to JSON files.

        vocab.json  format: {"<int_id>": "<token_string>", ...}
        merges.json format: [{"pair": [id1, id2], "merged_id": id3}, ...]

        Args:
            vocab_path: Destination path for vocabulary JSON.
            merges_path: Destination path for merges JSON.
        """
        with open(vocab_path, "w", encoding="utf-8") as fh:
            json.dump(
                {str(tid): tok for tid, tok in self.vocab.items()},
                fh,
                ensure_ascii=False,
                indent=2,
            )

        merges_list = [
            {"pair": list(pair), "merged_id": mid}
            for pair, mid in self.bpe_merges.items()
        ]
        with open(merges_path, "w", encoding="utf-8") as fh:
            json.dump(merges_list, fh, ensure_ascii=False, indent=2)

    def load(self, vocab_path: str, merges_path: str) -> None:
        """
        Load vocabulary and BPE merges from JSON files produced by save().

        Args:
            vocab_path: Path to vocabulary JSON file.
            merges_path: Path to merges JSON file.
        """
        with open(vocab_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.vocab = {int(k): v for k, v in raw.items()}
        self.inverse_vocab = {v: int(k) for k, v in raw.items()}

        with open(merges_path, "r", encoding="utf-8") as fh:
            merges_list = json.load(fh)
        self.bpe_merges = {
            (entry["pair"][0], entry["pair"][1]): entry["merged_id"]
            for entry in merges_list
        }

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> Optional[int]:
        """Return the vocabulary ID for a token string, or None."""
        return self.inverse_vocab.get(token)

    def id_to_token(self, tid: int) -> Optional[str]:
        """Return the token string for a vocabulary ID, or None."""
        return self.vocab.get(tid)

    def special_token_id(self, token: str) -> Optional[int]:
        """Return the vocabulary ID of a special token, or None."""
        return self.inverse_vocab.get(token)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_bpe(self, chunk: str) -> List[int]:
        """
        Apply BPE merges to a pre-tokenised chunk.

        The chunk is first split into character IDs; then the earliest-learned
        merge that applies is greedily applied left-to-right until no merge
        changes the sequence.
        """
        unk_id = self.inverse_vocab.get("<|unk|>")

        char_ids: List[int] = []
        for char in chunk:
            tid = self.inverse_vocab.get(char)
            if tid is None:
                if unk_id is None:
                    raise ValueError(
                        f"Character {char!r} (U+{ord(char):04X}) not in vocab "
                        "and no <|unk|> fallback is defined."
                    )
                char_ids.append(unk_id)
            else:
                char_ids.append(tid)

        changed = True
        while changed and len(char_ids) > 1:
            changed = False
            merged: List[int] = []
            i = 0
            while i < len(char_ids):
                if i < len(char_ids) - 1:
                    pair: PairType = (char_ids[i], char_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged.append(self.bpe_merges[pair])
                        i += 2
                        changed = True
                        continue
                merged.append(char_ids[i])
                i += 1
            char_ids = merged

        return char_ids

    def _split_on_special_tokens(
        self,
        text: str,
        allowed_special: Set[str],
    ) -> List[Tuple[str, bool]]:
        """Split *text* into (segment, is_special) pairs using regex."""
        # Sort longest-first so "<|im_start|>" matches before "<|im|>" would
        pattern = "(" + "|".join(
            re.escape(tok)
            for tok in sorted(allowed_special, key=len, reverse=True)
        ) + ")"
        result: List[Tuple[str, bool]] = []
        for part in re.split(pattern, text):
            if not part:
                continue
            result.append((part, part in allowed_special))
        return result
