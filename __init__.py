"""
indic_tokenizer — BPE and SentencePiece tokenizer for Indian languages.

Supported scripts
-----------------
Devanagari (Hindi / Sanskrit / Marathi / Nepali), Bengali, Gurmukhi (Punjabi),
Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala, Tibetan, and ASCII.

Supported corpus formats
------------------------
.parquet   pandas DataFrame, uses 'text' column
.csv       pandas DataFrame, uses 'text' column
.txt       plain text, lines joined with <|endoftext|>
.json      array of {"text": "..."} objects
.jsonl     one JSON object per line, uses 'text' key
DataFrame  pandas DataFrame passed directly
Dataset    HuggingFace Dataset object

Quick start — BPE, single-shot  (small / medium corpus)
---------------------------------------------------------
::

    from indic_tokenizer import train

    tok = train("corpus.parquet", vocab_size=16_000, algorithm="bpe")
    tok.save("vocab.json", "merges.json")

    tok2 = train("corpus.txt", vocab_size=8_000)   # .txt is also fine
    ids  = tok2.encode("नमस्ते दुनिया")
    text = tok2.decode(ids)

Quick start — BPE, chunked  (large corpus, resumable)
------------------------------------------------------
Use mode="chunked" to get a ChunkedBPETrainer back, then add more files
and call finalize_training() yourself::

    from indic_tokenizer import train

    trainer = train(
        "data/corpus-part0.parquet",
        algorithm="bpe",
        mode="chunked",
    )
    trainer.add_file("data/corpus-part1.parquet")
    trainer.save_state("checkpoints/state.json")    # save progress

    # Resume in a later session
    trainer.load_state("checkpoints/state.json")
    trainer.add_file("data/corpus-part2.parquet")

    trainer.finalize_training(vocab_size=32_000, min_frequency=2)
    trainer.tokenizer.save("vocab.json", "merges.json")

    # Or use ChunkedBPETrainer directly for full control:
    from indic_tokenizer import IndicBPETokenizer, ChunkedBPETrainer

    tok     = IndicBPETokenizer()
    trainer = ChunkedBPETrainer(tok)
    trainer.add_directory("data/sangraha/*.parquet")
    trainer.save_state("checkpoints/state.json")
    trainer.finalize_training(vocab_size=32_000)
    tok.save("vocab.json", "merges.json")

Quick start — SentencePiece
----------------------------
::

    from indic_tokenizer import train

    tok = train("corpus.txt", vocab_size=16_000, algorithm="sentencepiece",
                model_prefix="indic_sp", output_dir="models/")
    ids    = tok.encode("வணக்கம் உலகம்")
    pieces = tok.encode_as_pieces("வணக்கம் உலகம்")

Inference — load a saved BPE tokenizer
---------------------------------------
::

    from indic_tokenizer import IndicBPETokenizer

    tok = IndicBPETokenizer()
    tok.load("vocab.json", "merges.json")
    ids  = tok.encode("नमस्ते दुनिया")
    text = tok.decode(ids)
"""
import argparse

from .bpe_tokenizer import IndicBPETokenizer
from .chunked_trainer import ChunkedBPETrainer
from .constants import DEFAULT_ALLOWED_SPECIAL, SPECIAL_TOKENS
from .data_loader import IndicDataLoader
from .preprocessor import IndicTextPreprocessor, pretokenize
from .sp_tokenizer import IndicSentencePieceTokenizer

__version__ = "0.1.0"
__author__ = "Naveen"
__license__ = "MIT"

__all__ = [
    # Tokenizer classes
    "IndicBPETokenizer",
    "IndicSentencePieceTokenizer",
    "ChunkedBPETrainer",
    # Utilities
    "IndicDataLoader",
    "IndicTextPreprocessor",
    "pretokenize",
    # Constants
    "SPECIAL_TOKENS",
    "DEFAULT_ALLOWED_SPECIAL",
    # High-level API
    "train",
]


# ---------------------------------------------------------------------------
# High-level train() API
# ---------------------------------------------------------------------------

def train(
    source,
    vocab_size: int = 16_000,
    algorithm: str = "bpe",
    mode: str = "single",
    output_dir: str = ".",
    model_prefix: str = "indic_tokenizer",
    special_tokens=None,
    text_column: str = "text",
    min_frequency: int = 2,
    verbose: bool = True,
):
    """
    Train an Indic tokenizer from any supported data source.

    Parameters
    ----------
    source : str | Path | pandas.DataFrame | HuggingFace Dataset
        Corpus to train on.  Accepts:
        - file path to .parquet, .csv, .txt, .json, or .jsonl
        - pandas DataFrame  (must have a column named *text_column*)
        - HuggingFace Dataset object
        - raw text string
    vocab_size : int
        Target vocabulary size (default: 16_000).
    algorithm : str
        ``"bpe"`` — custom BPE tokenizer (default).
        ``"sentencepiece"`` — SentencePiece BPE/Unigram tokenizer.
    mode : str
        ``"single"`` — load all text then train in one shot (default).
                        Best for corpora that fit comfortably in RAM (<1 GB).
        ``"chunked"`` — stream-based BPE; returns a ChunkedBPETrainer so you
                        can add more files and call finalize_training() later.
                        Required for very large corpora (tens of GB).
    output_dir : str
        Directory for SentencePiece output files (ignored for BPE).
    model_prefix : str
        Filename stem for SentencePiece output (e.g. "indic_sp.model").
    special_tokens : list[str] | None
        Override the default special tokens.  None uses the full Claude set.
    text_column : str
        Column / key name for text in DataFrames and JSON sources.
    min_frequency : int
        Minimum word frequency for chunked BPE — prunes rare word types
        before running BPE to save memory (default: 2).
    verbose : bool
        Show training progress.

    Returns
    -------
    IndicBPETokenizer
        When ``algorithm="bpe"`` and ``mode="single"``.
    ChunkedBPETrainer
        When ``algorithm="bpe"`` and ``mode="chunked"``.
        Call ``trainer.finalize_training(vocab_size=N)`` to complete training,
        then ``trainer.tokenizer.save(...)`` to persist.
    IndicSentencePieceTokenizer
        When ``algorithm="sentencepiece"``.

    Raises
    ------
    ValueError
        If *algorithm* or *mode* is not one of the accepted values.

    Examples
    --------
    >>> tok = train("corpus.parquet", vocab_size=16_000)
    >>> ids = tok.encode("नमस्ते दुनिया")

    >>> trainer = train("big_corpus.parquet", mode="chunked")
    >>> trainer.finalize_training(vocab_size=32_000)
    >>> trainer.tokenizer.save("vocab.json", "merges.json")
    """
    if algorithm not in ("bpe", "sentencepiece"):
        raise ValueError(
            f"algorithm must be 'bpe' or 'sentencepiece', got {algorithm!r}"
        )
    if mode not in ("single", "chunked"):
        raise ValueError(
            f"mode must be 'single' or 'chunked', got {mode!r}"
        )

    if algorithm == "sentencepiece":
        return _train_sentencepiece(
            source, vocab_size, output_dir, model_prefix,
            text_column, verbose,
        )

    # BPE path
    if mode == "chunked":
        return _train_bpe_chunked(
            source, text_column, special_tokens, verbose,
        )

    return _train_bpe_single(
        source, vocab_size, special_tokens, text_column, verbose,
    )


# ---------------------------------------------------------------------------
# Private training helpers — keep train() readable
# ---------------------------------------------------------------------------

def _train_bpe_single(
    source,
    vocab_size: int,
    special_tokens,
    text_column: str,
    verbose: bool,
) -> "IndicBPETokenizer":
    """Load all text at once then run sequence-based BPE."""
    loader = IndicDataLoader(text_column=text_column)
    text = loader.load(source)
    tok = IndicBPETokenizer()
    tok.train(
        text,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        verbose=verbose,
    )
    return tok


def _train_bpe_chunked(
    source,
    text_column: str,
    special_tokens,
    verbose: bool,
) -> "ChunkedBPETrainer":
    """
    Initialise a ChunkedBPETrainer, feed *source* into it, and return.

    The caller must call trainer.finalize_training(vocab_size=N) to complete
    BPE training, then trainer.tokenizer.save() to persist the model.
    """
    tok = IndicBPETokenizer()
    trainer = ChunkedBPETrainer(tok, special_tokens=special_tokens)
    trainer.add_file(source, text_column=text_column, verbose=verbose)
    if verbose:
        print(
            "\nChunkedBPETrainer ready — word frequencies accumulated.\n"
            "Next steps:\n"
            "  1. trainer.add_file(...)  or  trainer.add_directory(...)\n"
            "  2. trainer.save_state('checkpoint.json')   # optional\n"
            "  3. trainer.finalize_training(vocab_size=32_000)\n"
            "  4. trainer.tokenizer.save('vocab.json', 'merges.json')"
        )
    return trainer


def _train_sentencepiece(
    source,
    vocab_size: int,
    output_dir: str,
    model_prefix: str,
    text_column: str,
    verbose: bool,
) -> "IndicSentencePieceTokenizer":
    """Load text then train a SentencePiece model."""
    loader = IndicDataLoader(text_column=text_column)
    text = loader.load(source)
    tok = IndicSentencePieceTokenizer()
    tok.train_from_text(
        text,
        output_dir=output_dir,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
    )
    return tok


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser():
    """
    Build and return the argparse ArgumentParser.

    Kept separate from main() so it can be reused in tests.
    """
    

    parser = argparse.ArgumentParser(
        prog="indic_tokenizer",
        description=(
            "Train a BPE or SentencePiece tokenizer on Indic-language text.\n\n"
            "Supported input formats: .parquet, .csv, .txt, .json, .jsonl\n"
            "  parquet / csv  — must have a 'text' column (see --text-column)\n"
            "  json           — array of {\"text\": \"...\"} objects\n"
            "  jsonl          — one JSON object per line with a 'text' key\n"
            "  txt            — plain text; each line is one document,\n"
            "                   lines are joined with <|endoftext|>\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "--------\n"
            "  # BPE single-shot from a parquet file\n"
            "  python -m indic_tokenizer --input corpus.parquet --vocab-size 16000\n\n"
            "  # BPE chunked training (large corpus)\n"
            "  python -m indic_tokenizer --input data/*.parquet --mode chunked \\\n"
            "      --vocab-size 32000 --checkpoint checkpoints/state.json\n\n"
            "  # SentencePiece from a plain-text file\n"
            "  python -m indic_tokenizer --input corpus.txt --algorithm sentencepiece \\\n"
            "      --vocab-size 16000 --output-dir models/ --model-prefix indic_sp\n"
        ),
    )

    # ── Required ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help=(
            "Path to the corpus file (.parquet, .csv, .txt, .json, .jsonl) "
            "or a glob pattern for chunked training (e.g. 'data/*.parquet')."
        ),
    )

    # ── Vocabulary ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=16_000,
        metavar="N",
        help="Target vocabulary size. Default: 16000.",
    )

    # ── Algorithm choice ─────────────────────────────────────────────────────
    parser.add_argument(
        "--algorithm", "-a",
        choices=["bpe", "sentencepiece"],
        default="bpe",
        help=(
            "Tokenization algorithm to use.\n"
            "  bpe          — custom BPE (default); outputs vocab.json + merges.json\n"
            "  sentencepiece — SentencePiece BPE/Unigram; outputs .model file"
        ),
    )

    # ── Training mode ────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode", "-m",
        choices=["single", "chunked"],
        default="single",
        help=(
            "Training mode.\n"
            "  single  — load all text into RAM then train (default).\n"
            "            Best for corpora that fit in memory (< 1 GB).\n"
            "  chunked — stream-based word-frequency BPE for very large corpora.\n"
            "            Uses ~600 MB RAM regardless of corpus size. Resumable\n"
            "            via --checkpoint."
        ),
    )

    # ── Output paths (BPE) ───────────────────────────────────────────────────
    parser.add_argument(
        "--vocab-out",
        default="vocab.json",
        metavar="PATH",
        help="Output path for vocab.json (BPE only). Default: vocab.json.",
    )
    parser.add_argument(
        "--merges-out",
        default="merges.json",
        metavar="PATH",
        help="Output path for merges.json (BPE only). Default: merges.json.",
    )


    # ── Output paths (SentencePiece) ─────────────────────────────────────────
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        metavar="DIR",
        help="Output directory for SentencePiece .model file. Default: '.'.",
    )
    parser.add_argument(
        "--model-prefix",
        default="indic_tokenizer",
        metavar="PREFIX",
        help=(
            "Filename prefix for SentencePiece output "
            "(e.g. 'indic_sp' -> indic_sp.model). Default: indic_tokenizer."
        ),
    )

    # ── Data options ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--text-column",
        default="text",
        metavar="COL",
        help=(
            "Column name (parquet/csv) or JSON key that holds the text. "
            "Default: 'text'."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap on the number of rows / documents to load. Default: all.",
    )

    # ── Chunked training options ──────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "JSON checkpoint file for chunked BPE.\n"
            "  If the file exists, word frequencies are loaded before ingestion\n"
            "  (resume a previous session).\n"
            "  After ingestion the checkpoint is updated / created.\n"
            "  Only used when --mode chunked."
        ),
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Prune words appearing fewer than N times before running BPE "
            "(chunked mode only). Default: 2."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5_000,
        metavar="N",
        help="Rows processed per batch in chunked mode. Default: 5000.",
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def main(argv=None):
    """
    CLI entry point for indic_tokenizer.

    Parses command-line arguments and runs the appropriate training pipeline.
    Can also be called programmatically by passing a list of argument strings
    to *argv* (useful for testing).

    Args:
        argv: Argument list (default: sys.argv[1:]).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    verbose = not args.quiet

    # ── SentencePiece path ───────────────────────────────────────────────────
    if args.algorithm == "sentencepiece":
        _cli_sentencepiece(args, verbose)
        return

    # ── BPE — chunked path ───────────────────────────────────────────────────
    if args.mode == "chunked":
        _cli_bpe_chunked(args, verbose)
        return

    # ── BPE — single-shot path ───────────────────────────────────────────────
    _cli_bpe_single(args, verbose)


# ---------------------------------------------------------------------------
# CLI sub-routines — keep main() readable
# ---------------------------------------------------------------------------

def _cli_bpe_single(args, verbose: bool) -> None:
    """Load full corpus, train BPE in one shot, save vocab + merges."""
    loader = IndicDataLoader(
        text_column=args.text_column,
        max_samples=args.max_samples,
    )
    text = loader.load(args.input)

    tok = IndicBPETokenizer()
    tok.train(text, vocab_size=args.vocab_size, verbose=verbose)
    tok.save(args.vocab_out, args.merges_out)

    if verbose:
        print(f"Saved vocab  -> {args.vocab_out}")
        print(f"Saved merges -> {args.merges_out}")


def _cli_bpe_chunked(args, verbose: bool) -> None:
    """
    Stream corpus into ChunkedBPETrainer, optionally resuming from a
    checkpoint, then finalise and save.

    Workflow
    --------
    1. Create trainer (resume from --checkpoint if it exists).
    2. Feed --input (file path or glob) into trainer.
    3. Save checkpoint (if --checkpoint specified).
    4. Finalise BPE and save vocab + merges.
    """
    import os

    tok = IndicBPETokenizer()
    trainer = ChunkedBPETrainer(tok)

    # Resume from an existing checkpoint (skip already-processed files)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        trainer.load_state(args.checkpoint)

    # Decide whether input is a glob pattern or a single file
    if any(c in args.input for c in ("*", "?", "[")):
        trainer.add_directory(
            args.input,
            text_column=args.text_column,
            batch_size=args.batch_size,
            verbose=verbose,
        )
    else:
        trainer.add_file(
            args.input,
            text_column=args.text_column,
            batch_size=args.batch_size,
            verbose=verbose,
        )

    # Persist checkpoint so the session can be resumed later
    if args.checkpoint:
        trainer.save_state(args.checkpoint)

    trainer.finalize_training(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        verbose=verbose,
    )
    tok.save(args.vocab_out, args.merges_out)

    if verbose:
        print(f"Saved vocab  -> {args.vocab_out}")
        print(f"Saved merges -> {args.merges_out}")


def _cli_sentencepiece(args, verbose: bool) -> None:
    """Load corpus, train SentencePiece model, save .model file."""
    loader = IndicDataLoader(
        text_column=args.text_column,
        max_samples=args.max_samples,
    )
    text = loader.load(args.input)

    tok = IndicSentencePieceTokenizer()
    tok.train_from_text(
        text,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
    )

    if verbose:
        model_file = f"{args.output_dir}/{args.model_prefix}.model"
        print(f"Saved model  -> {model_file}")


# ---------------------------------------------------------------------------
# Allow  python -m indic_tokenizer  (delegates to __main__.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
