"""
IndicSentencePieceTokenizer — SentencePiece (BPE or Unigram) for Indic text.

Trains, saves, and loads a SentencePiece model on any Indian language corpus.

Corpus sources
--------------
train_from_file()    — auto-detects .parquet/.csv/.txt/.json/.jsonl
train_from_text()    — bring-your-own UTF-8 string
train_from_dataset() — HuggingFace Dataset (downloads on first run)

Encoding / decoding mirrors the sentencepiece API and adds:
  - optional BOS / EOS injection
  - piece-level introspection helpers
  - a quick round-trip self-test via verify()

SentencePiece settings
-----------------------
character_coverage=1.0   covers every code point in every Indic script
split_by_whitespace=False  keeps agglutinative clusters together
byte_fallback=True       encodes unseen bytes as <0xNN> pieces
normalization_rule_name  "nmt_nfkc" — NFKC + NMT punctuation rules

Example
-------
    from indic_tokenizer import IndicSentencePieceTokenizer

    tok = IndicSentencePieceTokenizer()
    tok.train_from_file("corpus.parquet", model_prefix="indic_sp")
    ids    = tok.encode("नमस्ते दुनिया")
    pieces = tok.encode_as_pieces("नमस्ते दुनिया")
    text   = tok.decode(ids)
    tok.verify("नमस्ते दुनिया")
"""

import logging
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import List, Optional, Union

from .constants import SPECIAL_TOKENS
from .data_loader import IndicDataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation helper
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Apply NFKC normalisation, strip URLs, and collapse whitespace runs."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class IndicSentencePieceTokenizer:
    """
    SentencePiece tokenizer trained on Indic-language text.

    Supports BPE and Unigram model types.  Settings are tuned for Indian
    scripts (full character coverage, no whitespace splitting, byte fallback).

    All training methods ultimately call _run_trainer(), which writes the
    corpus to a temporary file and delegates to
    spm.SentencePieceTrainer.train().  The trained model is loaded
    automatically so the instance is immediately usable after training.
    """

    def __init__(self) -> None:
        self._processor = None          # spm.SentencePieceProcessor, lazy-loaded
        self._model_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """True after a successful train_* call or load()."""
        return self._processor is not None

    @property
    def vocab_size(self) -> int:
        """Number of pieces in the vocabulary."""
        self._require_model()
        return self._processor.get_piece_size()

    # ------------------------------------------------------------------
    # Training — three corpus sources
    # ------------------------------------------------------------------

    def train_from_file(
        self,
        path: Union[str, Path],
        output_dir: str = ".",
        model_prefix: str = "indic_sp",
        vocab_size: int = 16_000,
        model_type: str = "bpe",
        text_column: str = "text",
        max_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        """
        Train from any supported corpus file (.parquet, .csv, .txt, .json, .jsonl).

        Args:
            path: Path to the corpus file.
            output_dir: Directory for the .model / .vocab output files.
            model_prefix: Filename stem (e.g. "indic_sp" → indic_sp.model).
            vocab_size: Target vocabulary size.
            model_type: "bpe" or "unigram".
            text_column: Column / key that holds the text in tabular files.
            max_samples: Cap on rows / documents (None → all).
            normalize: Apply NFKC + URL removal before training.
        """
        loader = IndicDataLoader(text_column=text_column, max_samples=max_samples)
        logger.info("Loading corpus from %s", path)
        corpus = loader.load(path)
        self.train_from_text(
            corpus,
            output_dir=output_dir,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            normalize=normalize,
        )

    def train_from_dataset(
        self,
        dataset,
        output_dir: str = ".",
        model_prefix: str = "indic_sp",
        vocab_size: int = 16_000,
        model_type: str = "bpe",
        text_column: str = "text",
        max_samples: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        """
        Train from a HuggingFace Dataset or pandas DataFrame.

        Args:
            dataset: HuggingFace Dataset / DatasetDict / pandas DataFrame.
            output_dir: Directory for output files.
            model_prefix: Filename stem.
            vocab_size: Target vocabulary size.
            model_type: "bpe" or "unigram".
            text_column: Column / key for text.
            max_samples: Cap on rows loaded.
            normalize: Apply NFKC + URL removal.
        """
        loader = IndicDataLoader(text_column=text_column, max_samples=max_samples)
        corpus = loader.load(dataset)
        self.train_from_text(
            corpus,
            output_dir=output_dir,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            normalize=normalize,
        )

    def train_from_text(
        self,
        text: str,
        output_dir: str = ".",
        model_prefix: str = "indic_sp",
        vocab_size: int = 16_000,
        model_type: str = "bpe",
        normalize: bool = True,
    ) -> None:
        """
        Train directly from a UTF-8 string corpus.

        sentencepiece requires a file path as input, so the string is written
        to a temporary file which is deleted after training completes.

        Args:
            text: Raw UTF-8 corpus text.
            output_dir: Directory for output files.
            model_prefix: Filename stem.
            vocab_size: Target vocabulary size.
            model_type: "bpe" or "unigram".
            normalize: Apply NFKC + URL removal.
        """
        if normalize:
            text = _normalize(text)

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(text)
            corpus_path = tmp.name

        try:
            prefix = str(Path(output_dir) / model_prefix)
            self._run_trainer(corpus_path, prefix, vocab_size, model_type)
        finally:
            os.unlink(corpus_path)   # clean up temp file regardless of errors

        self.load(f"{prefix}.model")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a pre-trained .model file from disk."""
        try:
            import sentencepiece as spm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentencepiece is required.  Install it with:  "
                "pip install sentencepiece"
            ) from exc

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {model_path!r}"
            )
        proc = spm.SentencePieceProcessor()
        proc.load(model_path)
        self._processor = proc
        self._model_path = model_path
        logger.info("Loaded %r  (vocab size: %d)", model_path, self.vocab_size)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode *text* into a list of integer token IDs."""
        self._require_model()
        return self._processor.encode(text, add_bos=add_bos, add_eos=add_eos)

    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode *text* into a list of subword piece strings."""
        self._require_model()
        return self._processor.encode_as_pieces(text)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to text."""
        self._require_model()
        return self._processor.decode(ids)

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    def id_to_piece(self, tid: int) -> str:
        """Return the piece string for a vocabulary ID."""
        self._require_model()
        return self._processor.id_to_piece(tid)

    def piece_to_id(self, piece: str) -> int:
        """Return the vocabulary ID for a piece string (-1 if unknown)."""
        self._require_model()
        return self._processor.piece_to_id(piece)

    def special_token_id(self, token: str) -> int:
        """Return the ID of a special token string, or -1 if absent."""
        return self.piece_to_id(token)

    # ------------------------------------------------------------------
    # Self-test
    # ------------------------------------------------------------------

    def verify(self, text: str = "नमस्ते दुनिया") -> None:
        """
        Quick round-trip sanity check: encode → decode must recover *text*.

        Prints IDs and pieces for inspection.

        Args:
            text: Sample text in any Indic language.
        """
        self._require_model()
        ids = self.encode(text)
        pieces = self.encode_as_pieces(text)
        decoded = self.decode(ids)

        print(f"Input  : {text!r}")
        print(f"IDs    : {ids}")
        print(f"Pieces : {pieces}")
        print(f"Decoded: {decoded!r}")

        if decoded.strip() != text.strip():
            raise AssertionError(
                f"Round-trip failed:\n  in : {text!r}\n  out: {decoded!r}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_trainer(
        self,
        corpus_path: str,
        model_path_prefix: str,
        vocab_size: int,
        model_type: str,
    ) -> None:
        """Call spm.SentencePieceTrainer.train with Indic-tuned settings."""
        try:
            import sentencepiece as spm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentencepiece is required.  Install it with:  "
                "pip install sentencepiece"
            ) from exc

        n_threads = min(8, os.cpu_count() or 4)
        user_symbols = ",".join(SPECIAL_TOKENS)

        logger.info(
            "Training SentencePiece (%s) | vocab_size=%d | threads=%d",
            model_type, vocab_size, n_threads,
        )

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_path_prefix,
            vocab_size=vocab_size,
            model_type=model_type,             # "bpe" or "unigram"

            # ── Indic-tuned settings ──────────────────────────────────
            character_coverage=1.0,            # must cover every code point
            split_by_whitespace=False,         # don't break on whitespace (agglutinative)
            byte_fallback=True,                # <0xNN> pieces for rare bytes

            # ── Special tokens ────────────────────────────────────────
            user_defined_symbols=user_symbols,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,

            # ── Performance ───────────────────────────────────────────
            num_threads=n_threads,

            # ── Normalisation ─────────────────────────────────────────
            normalization_rule_name="nmt_nfkc",
        )

        logger.info("Model written to %s.model", model_path_prefix)

    def _require_model(self) -> None:
        """Raise RuntimeError if no model has been loaded yet."""
        if self._processor is None:
            raise RuntimeError(
                "No model loaded. Call train_from_file(), train_from_text(), "
                "train_from_dataset(), or load() first."
            )
