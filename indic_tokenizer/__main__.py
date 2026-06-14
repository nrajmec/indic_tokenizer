"""
CLI entry point for:  python -m indic_tokenizer  [options]

Algorithms
----------
bpe / single     — train BPE on the full corpus in one pass (small/medium data)
bpe / chunked    — word-frequency BPE with two-phase checkpointing (large data)
sentencepiece / single  — train SentencePiece on one or more files in one pass
sentencepiece / chunked — two-phase: accumulate files into corpus, then finalize

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

# SentencePiece single (one or more files)
python -m indic_tokenizer -i file1.txt file2.txt -a sentencepiece -v 8000 \
    -o models --model-prefix mymodel

# SentencePiece chunked — phase 1 (accumulate multiple files, resumable)
python -m indic_tokenizer -i file1.txt file2.txt file3.txt \
    -a sentencepiece -m chunked --checkpoint sp_state.json -v 8000

# SentencePiece chunked — phase 2 (train on accumulated corpus)
python -m indic_tokenizer -a sentencepiece -m chunked --finalize \
    --checkpoint sp_state.json -o models --model-prefix mymodel
"""

import argparse
import json
import os
import shutil
import subprocess
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
    p.add_argument("-i", "--input", metavar="FILE", nargs="+",
                   help="One or more input corpus files (.txt / .csv / .parquet / .json / .jsonl)")
    p.add_argument("-a", "--algorithm", choices=["bpe", "sentencepiece"],
                   default="bpe", help="Tokenizer algorithm (default: bpe)")
    p.add_argument("-m", "--mode", choices=["single", "chunked"],
                   default="single", help="Training mode (default: single)")
    p.add_argument("-v", "--vocab-size", type=int, metavar="N",
                   help="Target vocabulary size")
    p.add_argument("--vocab-out", metavar="FILE",
                   help="Output path for vocabulary JSON")
    p.add_argument("--merges-out", metavar="FILE",
                   help="Output path for merges JSON")
    p.add_argument("--checkpoint", metavar="FILE",
                   help="Checkpoint file for chunked mode (phase 1 output / phase 2 input)")
    p.add_argument("--min-frequency", type=int, default=1, metavar="N",
                   help="Minimum word frequency for chunked BPE (default: 1)")
    p.add_argument("--finalize", action="store_true",
                   help="Run phase 2 of chunked training (build model from checkpoint)")
    p.add_argument("-o", "--output-dir", metavar="DIR", default=".",
                   help="Output directory for SentencePiece model files")
    p.add_argument("--model-prefix", metavar="PREFIX", default="sp_model",
                   help="Model name prefix for SentencePiece output")
    p.add_argument("--text-column", metavar="COL", default="text",
                   help="Column name for text in tabular files (default: text)")
    p.add_argument("--kaggle-dataset", metavar="USER/DATASET",
                   help="Upload the trained model to this Kaggle dataset after finalize "
                        "(format: 'username/dataset-name'). Requires kaggle CLI configured.")
    p.add_argument("--session-timeout", type=int, default=42300, metavar="SECS",
                   help="Max seconds to let SentencePiece training run before stopping "
                        "and uploading whatever was produced (default: 42300 = 11h45m).")
    return p


# ---------------------------------------------------------------------------
# Kaggle upload helper
# ---------------------------------------------------------------------------

def _upload_model_to_kaggle(model_path: Path, vocab_path: Path,
                             dataset_id: str) -> None:
    """
    Upload the trained .model and .vocab files to a Kaggle dataset.

    Args:
        model_path:  Path to the .model file produced by SentencePiece.
        vocab_path:  Path to the .vocab file produced by SentencePiece.
        dataset_id:  Kaggle dataset in 'username/dataset-name' format.
    """
    tmp = "/kaggle/working/_upload_tmp"
    os.makedirs(tmp, exist_ok=True)

    shutil.copy(str(model_path), tmp)
    if vocab_path.exists():
        shutil.copy(str(vocab_path), tmp)

    with open(f"{tmp}/dataset-metadata.json", "w") as f:
        json.dump({
            "title": "indic_tokenizer SP Model",
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}],
        }, f)

    # Try to version an existing dataset; create it if it doesn't exist yet.
    ret = subprocess.run(
        ["kaggle", "datasets", "version", "-p", tmp,
         "-m", "Trained SentencePiece model", "--dir-mode", "zip"]
    ).returncode
    if ret != 0:
        print("  Version update failed — attempting to create dataset (first upload) …")
        ret = subprocess.run(
            ["kaggle", "datasets", "create", "-p", tmp, "--dir-mode", "zip"]
        ).returncode

    shutil.rmtree(tmp, ignore_errors=True)

    if ret == 0:
        print(f"  Model uploaded to kaggle.com/datasets/{dataset_id}")
    else:
        print(f"  Upload failed — model is saved locally at {model_path}")


# ---------------------------------------------------------------------------
# BPE single-shot
# ---------------------------------------------------------------------------

def run_bpe_single(args):
    if not args.input:
        print("ERROR: -i / --input is required for BPE single mode.", file=sys.stderr)
        sys.exit(1)
    if len(args.input) > 1:
        print("ERROR: BPE single mode accepts only one input file. "
              "Use -m chunked for multiple files.", file=sys.stderr)
        sys.exit(1)
    if not args.vocab_size:
        print("ERROR: -v / --vocab-size is required.", file=sys.stderr)
        sys.exit(1)
    if not args.vocab_out or not args.merges_out:
        print("ERROR: --vocab-out and --merges-out are required.", file=sys.stderr)
        sys.exit(1)

    from .bpe_tokenizer import IndicBPETokenizer

    input_file = args.input[0]
    tok = IndicBPETokenizer()
    print(f"Loading corpus from {input_file} …")
    tok.train_from_file(
        path=input_file,
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

    input_file = args.input[0]
    print(f"Loading corpus from {input_file} …")
    loader = IndicDataLoader(text_column=args.text_column)
    corpus = loader.load(input_file)

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
# SentencePiece — single mode (one or more files, one training pass)
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / args.model_prefix)

    all_txt = all(Path(f).suffix.lower() == ".txt" for f in args.input)
    tmp_path = None

    if all_txt:
        # Pass comma-separated list — SPM natively supports multi-file input
        train_input = ",".join(args.input)
    else:
        from .data_loader import IndicDataLoader
        loader = IndicDataLoader(text_column=args.text_column)
        parts = []
        for f in args.input:
            print(f"Loading {f} …")
            parts.append(loader.load(f))
        corpus = "\n".join(parts)
        tmp_path = out_dir / "_sp_tmp_corpus.txt"
        tmp_path.write_text(corpus, encoding="utf-8")
        train_input = str(tmp_path)

    print(f"Training SentencePiece model (vocab_size={args.vocab_size}) …")
    try:
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
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    print(f"Saved model  → {prefix}.model")
    print(f"Saved vocab  → {prefix}.vocab")


# ---------------------------------------------------------------------------
# SentencePiece chunked — phase 1: accumulate files into corpus
# ---------------------------------------------------------------------------

def run_sentencepiece_chunked_accumulate(args):
    """
    Process each input file and append its sentences to a growing corpus .txt
    file. Saves a checkpoint JSON after every file so the run is resumable.
    """
    if not args.input:
        print("ERROR: -i / --input is required for SP chunked accumulate.", file=sys.stderr)
        sys.exit(1)
    if not args.checkpoint:
        print("ERROR: --checkpoint is required for SP chunked mode.", file=sys.stderr)
        sys.exit(1)
    if not args.vocab_size:
        print("ERROR: -v / --vocab-size is required.", file=sys.stderr)
        sys.exit(1)

    try:
        import sentencepiece  # validate installation early  # noqa: F401
    except ImportError:
        print("ERROR: sentencepiece is not installed.  "
              "Run:  pip install sentencepiece", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)

    # Load existing checkpoint (resume) or initialise fresh state
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if data.get("algorithm") != "sentencepiece":
            print("ERROR: Checkpoint is not a SentencePiece checkpoint.", file=sys.stderr)
            sys.exit(1)
        processed = set(data["processed_files"])
        corpus_file = Path(data["corpus_file"])
        total_sentences = data.get("total_sentences", 0)
        print(f"Resuming from checkpoint: {len(processed)} file(s) already processed.")
    else:
        processed = set()
        # Corpus file lives alongside the checkpoint: <stem>_corpus.txt
        corpus_file = checkpoint_path.with_name(checkpoint_path.stem + "_corpus.txt")
        total_sentences = 0

    corpus_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine which files still need processing (by resolved absolute path)
    pending = [f for f in args.input
               if str(Path(f).resolve()) not in processed]

    if not pending:
        print("All files already processed. Run with --finalize to train the model.")
        return

    from .data_loader import IndicDataLoader
    loader = IndicDataLoader(text_column=args.text_column)

    # Append mode: already-processed files' content is already in corpus_file
    with open(str(corpus_file), "a", encoding="utf-8") as corpus_fh:
        for file_path in pending:
            abs_path = str(Path(file_path).resolve())
            print(f"Processing: {file_path} …")

            if not Path(file_path).is_file():
                print(f"  ERROR: File not found: {file_path}", file=sys.stderr)
                print("  Tip: Kaggle datasets are mounted at /kaggle/input/<dataset-slug>/",
                      file=sys.stderr)
                sys.exit(1)

            raw = loader.load(file_path)

            # Normalise to one sentence per line (SPM's expected format)
            sentences = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]
            if not sentences:
                sentences = [ln for ln in raw.splitlines() if ln.strip()]

            corpus_fh.write("\n".join(sentences))
            corpus_fh.write("\n")
            total_sentences += len(sentences)
            processed.add(abs_path)

            # Determine which of the original inputs are still pending
            pending_remaining = [
                f for f in args.input
                if str(Path(f).resolve()) not in processed
            ]

            # Write checkpoint immediately after each file (fault-tolerance)
            checkpoint_data = {
                "algorithm": "sentencepiece",
                "corpus_file": str(corpus_file.resolve()),
                "processed_files": sorted(processed),
                "pending_files": pending_remaining,
                "total_sentences": total_sentences,
                "vocab_size": args.vocab_size,
                "model_type": "bpe",
                "character_coverage": 1.0,
            }
            checkpoint_path.write_text(
                json.dumps(checkpoint_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  → {len(sentences):,} sentences added "
                  f"({total_sentences:,} total). Checkpoint updated.")

    print(f"\nAccumulation complete.")
    print(f"  Files processed : {len(processed)}")
    print(f"  Total sentences : {total_sentences:,}")
    print(f"  Corpus file     : {corpus_file}")
    print(f"Run with --finalize to train the SentencePiece model.")


# ---------------------------------------------------------------------------
# SentencePiece training worker (module-level so multiprocessing can pickle it)
# ---------------------------------------------------------------------------

def _sp_train_worker(kwargs: dict) -> None:
    import sentencepiece as spm  # type: ignore[import]
    spm.SentencePieceTrainer.train(**kwargs)


# ---------------------------------------------------------------------------
# SentencePiece chunked — phase 2: train on accumulated corpus
# ---------------------------------------------------------------------------

def run_sentencepiece_chunked_finalize(args):
    """
    Load the checkpoint produced by phase 1 and train SentencePiece on the
    accumulated corpus file.
    """
    if not args.checkpoint:
        print("ERROR: --checkpoint is required for SP chunked finalize.", file=sys.stderr)
        sys.exit(1)

    try:
        import sentencepiece  # validate installation before spawning child  # noqa: F401
    except ImportError:
        print("ERROR: sentencepiece is not installed.  "
              "Run:  pip install sentencepiece", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if data.get("algorithm") != "sentencepiece":
        print("ERROR: Checkpoint is not a SentencePiece checkpoint. "
              "(Did you mean to use the BPE finalize command?)", file=sys.stderr)
        sys.exit(1)

    corpus_file = Path(data["corpus_file"])
    if not corpus_file.exists():
        print(f"ERROR: Accumulated corpus file not found: {corpus_file}", file=sys.stderr)
        sys.exit(1)

    # CLI -v takes precedence; checkpoint value is the fallback
    vocab_size = args.vocab_size or data.get("vocab_size")
    if not vocab_size:
        print("ERROR: -v / --vocab-size is required (or must be stored in checkpoint).",
              file=sys.stderr)
        sys.exit(1)

    pending = data.get("pending_files", [])
    if pending:
        print(f"WARNING: {len(pending)} file(s) not yet accumulated: {pending}")
        print("Proceeding with the corpus accumulated so far.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / args.model_prefix)

    total_sentences = data.get("total_sentences", 0)
    processed_count = len(data.get("processed_files", []))
    timeout_secs = getattr(args, "session_timeout", 42300)
    h, m = divmod(timeout_secs // 60, 60)

    print(f"Training SentencePiece on {total_sentences:,} sentences "
          f"from {processed_count} file(s) (vocab_size={vocab_size}) …")
    print(f"  Corpus          : {corpus_file}")
    print(f"  Output          : {prefix}.model / {prefix}.vocab")
    print(f"  Session timeout : {h}h {m}m ({timeout_secs:,}s)")

    # Run training in a child process so we can enforce the session timeout.
    # spm.SentencePieceTrainer.train() is a blocking C++ call that cannot be
    # interrupted from the same thread; terminate() on the child process is the
    # only reliable way to stop it.
    import multiprocessing
    import time

    train_kwargs = dict(
        input=str(corpus_file),
        model_prefix=prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )
    proc = multiprocessing.Process(target=_sp_train_worker, args=(train_kwargs,))
    t0 = time.time()
    proc.start()
    proc.join(timeout=timeout_secs)

    elapsed = int(time.time() - t0)
    model_path = Path(f"{prefix}.model")
    vocab_path = Path(f"{prefix}.vocab")

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"\nWARNING: Training stopped after {elapsed // 3600}h "
              f"{(elapsed % 3600) // 60}m (session timeout reached).")
        if not model_path.exists():
            print("No model file was produced — SentencePiece only writes the model "
                  "on successful completion. Re-run with a longer --session-timeout "
                  "or reduce --vocab-size.")
            return
    else:
        print(f"\nTraining complete in "
              f"{elapsed // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s.")

    if model_path.exists():
        print(f"Saved model  → {model_path}")
        print(f"Saved vocab  → {vocab_path}")
        if args.kaggle_dataset:
            print(f"Uploading model to Kaggle dataset '{args.kaggle_dataset}' …")
            _upload_model_to_kaggle(model_path, vocab_path, args.kaggle_dataset)
    else:
        print("ERROR: Model file not found after training.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.algorithm == "sentencepiece":
        if args.mode == "chunked":
            if args.finalize:
                run_sentencepiece_chunked_finalize(args)
            else:
                run_sentencepiece_chunked_accumulate(args)
        else:
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
