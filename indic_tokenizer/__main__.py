"""
CLI entry point for:  python -m indic_tokenizer  [options]

Algorithms
----------
bpe / single     — train BPE on the full corpus in one pass (small/medium data)
bpe / chunked    — word-frequency BPE with two-phase checkpointing (large data)
sentencepiece    — wrap the sentencepiece library

Examples
--------
# BPE single-shot
python -m indic_tokenizer -i corpus.txt -a bpe -m single -v 8000 \
    --vocab-out vocab.json --merges-out merges.json

# BPE chunked — phase 1 (accumulate word frequencies)
python -m indic_tokenizer -i corpus.txt -a bpe -m chunked \
    --checkpoint state.json --min-frequency 2

# BPE chunked — phase 2 (finalise merges)
python -m indic_tokenizer -a bpe -m chunked --finalize -v 8000 \
    --checkpoint state.json --vocab-out vocab.json --merges-out merges.json

# SentencePiece
python -m indic_tokenizer -i corpus.txt -a sentencepiece -v 8000 \
    -o models --model-prefix mymodel
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m indic_tokenizer",
        description="Train a tokenizer for Indic scripts.",
    )
    p.add_argument("-i", "--input", metavar="FILE",
                   help="Input corpus file (.txt / .csv / .parquet / .json / .jsonl)")
    p.add_argument("-a", "--algorithm", choices=["bpe", "sentencepiece"],
                   default="bpe", help="Tokenizer algorithm (default: bpe)")
    p.add_argument("-m", "--mode", choices=["single", "chunked"],
                   default="single", help="BPE mode (default: single)")
    p.add_argument("-v", "--vocab-size", type=int, metavar="N",
                   help="Target vocabulary size")
    p.add_argument("--vocab-out", metavar="FILE",
                   help="Output path for vocabulary JSON")
    p.add_argument("--merges-out", metavar="FILE",
                   help="Output path for merges JSON")
    p.add_argument("--checkpoint", metavar="FILE",
                   help="Checkpoint file for chunked BPE (phase 1 output / phase 2 input)")
    p.add_argument("--min-frequency", type=int, default=1, metavar="N",
                   help="Minimum word frequency for chunked BPE (default: 1)")
    p.add_argument("--finalize", action="store_true",
                   help="Run phase 2 of chunked BPE (build merges from checkpoint)")
    p.add_argument("-o", "--output-dir", metavar="DIR", default=".",
                   help="Output directory for SentencePiece model files")
    p.add_argument("--model-prefix", metavar="PREFIX", default="sp_model",
                   help="Model name prefix for SentencePiece output")
    p.add_argument("--text-column", metavar="COL", default="text",
                   help="Column name for text in tabular files (default: text)")
    return p


# ---------------------------------------------------------------------------
# BPE single-shot
# ---------------------------------------------------------------------------

def run_bpe_single(args):
    if not args.input:
        print("ERROR: -i / --input is required for BPE single mode.", file=sys.stderr)
        sys.exit(1)
    if not args.vocab_size:
        print("ERROR: -v / --vocab-size is required.", file=sys.stderr)
        sys.exit(1)
    if not args.vocab_out or not args.merges_out:
        print("ERROR: --vocab-out and --merges-out are required.", file=sys.stderr)
        sys.exit(1)

    from .bpe_tokenizer import IndicBPETokenizer

    tok = IndicBPETokenizer()
    print(f"Loading corpus from {args.input} …")
    tok.train_from_file(
        path=args.input,
        vocab_size=args.vocab_size,
        text_column=args.text_column,
        verbose=True,
    )
    tok.save(args.vocab_out, args.merges_out)
    print(f"Saved vocab  → {args.vocab_out}")
    print(f"Saved merges → {args.merges_out}")


# ---------------------------------------------------------------------------
# BPE chunked — phase 1: accumulate word frequencies
# ---------------------------------------------------------------------------

def run_bpe_chunked_accumulate(args):
    if not args.input:
        print("ERROR: -i / --input is required for chunked phase 1.", file=sys.stderr)
        sys.exit(1)
    if not args.checkpoint:
        print("ERROR: --checkpoint is required for chunked BPE.", file=sys.stderr)
        sys.exit(1)

    from .data_loader import IndicDataLoader
    from .preprocessor import pretokenize

    print(f"Loading corpus from {args.input} …")
    loader = IndicDataLoader(text_column=args.text_column)
    corpus = loader.load(args.input)

    print("Building word-frequency table …")
    freq: Counter = Counter()
    for word in pretokenize(corpus):
        if word.strip():
            freq[word] += 1

    if args.min_frequency > 1:
        freq = Counter({w: c for w, c in freq.items() if c >= args.min_frequency})

    print(f"  Unique word types (freq ≥ {args.min_frequency}): {len(freq):,}")

    checkpoint = {
        "word_freq": {w: c for w, c in freq.items()},
        "min_frequency": args.min_frequency,
    }
    Path(args.checkpoint).write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Checkpoint saved → {args.checkpoint}")
    print("Run with --finalize to build BPE merges from this checkpoint.")


# ---------------------------------------------------------------------------
# BPE chunked — phase 2: finalise merges from checkpoint
# ---------------------------------------------------------------------------

def run_bpe_chunked_finalize(args):
    if not args.checkpoint:
        print("ERROR: --checkpoint is required for chunked finalize.", file=sys.stderr)
        sys.exit(1)
    if not args.vocab_size:
        print("ERROR: -v / --vocab-size is required.", file=sys.stderr)
        sys.exit(1)
    if not args.vocab_out or not args.merges_out:
        print("ERROR: --vocab-out and --merges-out are required.", file=sys.stderr)
        sys.exit(1)

    from .vocab_builder import VocabBuilder
    from .bpe_trainer import BPETrainer
    from .constants import SPECIAL_TOKENS

    print(f"Loading checkpoint from {args.checkpoint} …")
    data = json.loads(Path(args.checkpoint).read_text(encoding="utf-8"))
    word_freq: dict = data["word_freq"]
    print(f"  Loaded {len(word_freq):,} word types.")

    # Build corpus string weighted by frequency for training
    corpus_parts = []
    for word, count in word_freq.items():
        corpus_parts.extend([word] * count)
    corpus = "".join(corpus_parts)

    vb = VocabBuilder()
    vocab, inverse_vocab = vb.build_base_vocab()
    vb.extend_from_text(corpus, vocab, inverse_vocab)
    vb.add_special_tokens(vocab, inverse_vocab, list(SPECIAL_TOKENS))

    print(f"Base vocab: {len(vocab):,} tokens  |  Target: {args.vocab_size:,}")

    token_ids = [inverse_vocab[c] for c in corpus if c in inverse_vocab]

    trainer = BPETrainer()
    bpe_merges = trainer.train(
        token_ids, vocab, inverse_vocab,
        target_vocab_size=args.vocab_size,
        verbose=True,
    )

    # Serialise
    with open(args.vocab_out, "w", encoding="utf-8") as fh:
        json.dump({str(tid): tok for tid, tok in vocab.items()},
                  fh, ensure_ascii=False, indent=2)

    merges_list = [
        {"pair": list(pair), "merged_id": mid}
        for pair, mid in bpe_merges.items()
    ]
    with open(args.merges_out, "w", encoding="utf-8") as fh:
        json.dump(merges_list, fh, ensure_ascii=False, indent=2)

    print(f"Final vocab size: {len(vocab):,}")
    print(f"Saved vocab  → {args.vocab_out}")
    print(f"Saved merges → {args.merges_out}")


# ---------------------------------------------------------------------------
# SentencePiece
# ---------------------------------------------------------------------------

def run_sentencepiece(args):
    if not args.input:
        print("ERROR: -i / --input is required for SentencePiece.", file=sys.stderr)
        sys.exit(1)
    if not args.vocab_size:
        print("ERROR: -v / --vocab-size is required.", file=sys.stderr)
        sys.exit(1)

    try:
        import sentencepiece as spm  # type: ignore[import]
    except ImportError:
        print("ERROR: sentencepiece is not installed.  "
              "Run:  pip install sentencepiece", file=sys.stderr)
        sys.exit(1)

    from .data_loader import IndicDataLoader

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / args.model_prefix)

    # SentencePiece can train directly from a .txt file; for other formats
    # we write a temp file.
    input_path = Path(args.input)
    if input_path.suffix.lower() == ".txt":
        train_input = str(input_path)
    else:
        print(f"Loading corpus from {args.input} for SentencePiece …")
        loader = IndicDataLoader(text_column=args.text_column)
        corpus = loader.load(args.input)
        tmp_path = out_dir / "_sp_tmp_corpus.txt"
        tmp_path.write_text(corpus, encoding="utf-8")
        train_input = str(tmp_path)

    print(f"Training SentencePiece model (vocab_size={args.vocab_size}) …")
    spm.SentencePieceTrainer.train(
        input=train_input,
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    print(f"Saved model  → {prefix}.model")
    print(f"Saved vocab  → {prefix}.vocab")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.algorithm == "sentencepiece":
        run_sentencepiece(args)
        return

    # BPE path
    if args.mode == "single":
        run_bpe_single(args)
    elif args.mode == "chunked":
        if args.finalize:
            run_bpe_chunked_finalize(args)
        else:
            run_bpe_chunked_accumulate(args)


if __name__ == "__main__":
    main()
