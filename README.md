# indic-tokenizer

A BPE and SentencePiece tokenizer built from scratch for **all major Indian languages**, with special token support.

Supported scripts: **Devanagari** · **Bengali** · **Gurmukhi** · **Gujarati** · **Oriya** · **Tamil** · **Telugu** · **Kannada** · **Malayalam** · **Sinhala** · **Tibetan** · ASCII

---

## Table of Contents

1. [Installation](#installation)
2. [Supported File Formats](#supported-file-formats)
3. [Training Modes Overview](#training-modes-overview)
4. [Command-Line Usage](#command-line-usage)
   - [All flags](#all-flags)
   - [Mode 1: Single-shot BPE](#mode-1-single-shot-bpe-cli)
   - [Mode 2: SentencePiece](#mode-2-sentencepiece-cli)
   - [Mode 3: Chunked BPE (two-phase)](#mode-3-chunked-bpe-cli)
5. [Python API](#python-api)
   - [Mode 1: Single-shot BPE](#mode-1-single-shot-bpe-python)
   - [Mode 2: SentencePiece](#mode-2-sentencepiece-python)
   - [Mode 3: Chunked BPE (two-phase)](#mode-3-chunked-bpe-python)
   - [Inference](#inference)
   - [Using IndicDataLoader directly](#using-indicdataloader-directly)
6. [API Reference](#api-reference)
7. [Package Structure](#package-structure)
8. [Special Tokens](#special-tokens)
9. [Vocabulary Construction](#vocabulary-construction)
10. [Recommended Vocabulary Sizes](#recommended-vocabulary-sizes)

---

## Installation

```bash
# Core — BPE from text strings, no extra deps
pip install indic-tokenizer

# Add parquet / CSV support
pip install "indic-tokenizer[pandas]"

# Add HuggingFace dataset downloads + progress bars
pip install "indic-tokenizer[train]"

# Add SentencePiece support
pip install "indic-tokenizer[sentencepiece]"

# Add GPU-accelerated pair counting (PyTorch CUDA)
pip install "indic-tokenizer[gpu]"

# Install everything
pip install "indic-tokenizer[full]"
```

Or install from source:

```bash
git clone https://github.com/naveen-mech-sai/indic-tokenizer
cd indic-tokenizer
pip install -e ".[full]"
```

---

## Supported File Formats

| Extension | How it is read | Text field |
|-----------|---------------|------------|
| `.parquet` | pandas DataFrame via PyArrow | `text` column (configurable) |
| `.csv` | pandas DataFrame | `text` column (configurable) |
| `.txt` | Plain text — one document per line, joined with `<\|endoftext\|>` | N/A |
| `.json` | JSON array of `{"text": "..."}` objects | `text` key (configurable) |
| `.jsonl` | One JSON object per line | `text` key (configurable) |
| `DataFrame` | pandas DataFrame passed directly | `text` column (configurable) |
| `Dataset` | HuggingFace Dataset object | `text` column (configurable) |
| `str` | Raw text string — returned as-is | N/A |

**Plain-text rule:** every non-empty line is one document; lines are joined with
`<|endoftext|>` so BPE never merges tokens across document boundaries.

---

## Training Modes Overview

This package supports three training modes. Choose based on your corpus size and tooling preference.

| Mode | Algorithm | When to use | Output |
|------|-----------|-------------|--------|
| **Single-shot BPE** | `bpe` + `single` | Corpus fits in RAM (up to ~1 GB) | `vocab.json` + `merges.json` |
| **SentencePiece** | `sentencepiece` | Need a battle-tested unigram/BPE model | `.model` file |
| **Chunked BPE** | `bpe` + `chunked` | Corpus is too large for RAM; training spans multiple sessions | `vocab.json` + `merges.json` |

**Chunked BPE runs in two separate phases:**

- **Phase 1 — Accumulate**: Read corpus files, build word-frequency counts, save a checkpoint. Repeat across as many sessions as needed. BPE is not run yet.
- **Phase 2 — Finalize**: Load the checkpoint and run BPE on the accumulated frequencies. Writes `vocab.json` and `merges.json`. No corpus files needed in this phase.

---

## Command-Line Usage

The package is runnable directly from the terminal via `python -m indic_tokenizer`.

### Synopsis

```
# Single-shot and chunked accumulation
python -m indic_tokenizer --input PATH [OPTIONS]

# Chunked finalize (--input not required)
python -m indic_tokenizer --mode chunked --finalize --checkpoint STATE [OPTIONS]
```

### All flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | required* | Corpus file (`.parquet`, `.csv`, `.txt`, `.json`, `.jsonl`) or glob pattern. *Optional when `--finalize` is set. |
| `--vocab-size` | `-v` | `16000` | Target vocabulary size |
| `--algorithm` | `-a` | `bpe` | `bpe` or `sentencepiece` |
| `--mode` | `-m` | `single` | `single` (one-shot BPE) or `chunked` (stream, resumable) |
| `--finalize` | | off | **Chunked mode only.** Skip accumulation; load checkpoint and run BPE to produce `vocab.json` + `merges.json`. `--input` is not needed. |
| `--vocab-out` | | `vocab.json` | Output path for BPE vocabulary |
| `--merges-out` | | `merges.json` | Output path for BPE merges |
| `--output-dir` | `-o` | `.` | Directory for SentencePiece `.model` file |
| `--model-prefix` | | `indic_tokenizer` | SentencePiece filename prefix |
| `--text-column` | | `text` | Column / key name for text in tabular or JSON sources |
| `--max-samples` | | all | Cap on rows / documents loaded |
| `--checkpoint` | | none | Chunked BPE checkpoint file path (auto-resumes if it exists) |
| `--min-frequency` | | `2` | Prune words below this frequency before running BPE |
| `--batch-size` | | `5000` | Rows per batch in chunked mode |
| `--quiet` | `-q` | off | Suppress all progress output |
| `--version` | | | Print version and exit |

---

### Mode 1: Single-shot BPE (CLI)

Load the entire corpus into RAM and train BPE in one go.
Best for corpora up to ~1 GB.

```bash
# From a parquet file
python -m indic_tokenizer \
    --input corpus.parquet \
    --vocab-size 16000 \
    --vocab-out vocab.json \
    --merges-out merges.json
```

```bash
# From a plain-text file
python -m indic_tokenizer \
    --input corpus.txt \
    --vocab-size 8000
```

```bash
# From a JSON Lines file with a custom text key
python -m indic_tokenizer \
    --input corpus.jsonl \
    --vocab-size 16000 \
    --text-column content
```

```bash
# Load only a subset of rows (quick tests)
python -m indic_tokenizer \
    --input corpus.parquet \
    --vocab-size 4000 \
    --max-samples 50000 \
    --quiet
```

---

### Mode 2: SentencePiece (CLI)

Delegates to the SentencePiece library. Writes a `.model` file to `--output-dir`.

```bash
# From a plain-text file
python -m indic_tokenizer \
    --input corpus.txt \
    --algorithm sentencepiece \
    --vocab-size 16000 \
    --output-dir models/ \
    --model-prefix indic_sp
# Writes: models/indic_sp.model
```

```bash
# From a parquet file
python -m indic_tokenizer \
    --input corpus.parquet \
    --algorithm sentencepiece \
    --vocab-size 32000 \
    --output-dir models/
```

---

### Mode 3: Chunked BPE (CLI)

Chunked BPE splits training into two phases. You run Phase 1 one or more times
to accumulate word frequencies, then run Phase 2 once to finalize.

#### Phase 1 — Accumulate (one or more sessions)

Each session reads a corpus file, updates word frequencies, and saves a checkpoint.
Re-run with the same `--checkpoint` path to resume; already-processed files
inside the checkpoint are skipped automatically.

```bash
# Session 1: ingest first file
python -m indic_tokenizer \
    --input data/corpus-part0.parquet \
    --mode chunked \
    --checkpoint checkpoints/state.json

# Session 2: ingest a second file (part0 is already counted, skipped)
python -m indic_tokenizer \
    --input data/corpus-part1.parquet \
    --mode chunked \
    --checkpoint checkpoints/state.json

# Or use a glob to process many files in one session
python -m indic_tokenizer \
    --input "data/sangraha/*.parquet" \
    --mode chunked \
    --checkpoint checkpoints/state.json \
    --min-frequency 2
```

> **Note:** You can run as many accumulation sessions as needed — each one
> merges new counts into the existing checkpoint. No BPE is run yet.

#### Phase 2 — Finalize (run once, after all accumulation is done)

When you are satisfied that all corpus data has been accumulated, run the
finalize step. `--input` is **not** required here — the checkpoint already
holds all the word frequencies.

```bash
python -m indic_tokenizer \
    --mode chunked \
    --finalize \
    --checkpoint checkpoints/state.json \
    --vocab-size 32000 \
    --vocab-out vocab.json \
    --merges-out merges.json
# Writes: vocab.json  merges.json
```

#### Full workflow example (3-day corpus, separate sessions)

```bash
# Day 1 — ingest first batch
python -m indic_tokenizer \
    --input "data/batch-0*.parquet" \
    --mode chunked \
    --checkpoint checkpoints/state.json

# Day 2 — ingest second batch (batch-0* skipped automatically)
python -m indic_tokenizer \
    --input "data/batch-1*.parquet" \
    --mode chunked \
    --checkpoint checkpoints/state.json

# Day 3 — finalize, no --input needed
python -m indic_tokenizer \
    --mode chunked \
    --finalize \
    --checkpoint checkpoints/state.json \
    --vocab-size 32000
# Writes: vocab.json  merges.json
```

---

## Python API

### Mode 1: Single-shot BPE (Python)

Best when the entire corpus fits comfortably in RAM.

```python
from indic_tokenizer import train

# From a parquet file
tok = train("corpus.parquet", vocab_size=16_000)
tok.save("vocab.json", "merges.json")

# From a CSV file
tok = train("corpus.csv", vocab_size=16_000)

# From a plain-text file (each line = one document)
tok = train("corpus.txt", vocab_size=8_000)

# From a JSON Lines file
tok = train("corpus.jsonl", vocab_size=16_000)

# From a JSON file (array of {"text": "..."} objects)
tok = train("corpus.json", vocab_size=16_000)

# From a pandas DataFrame
import pandas as pd
df = pd.read_parquet("corpus.parquet")
tok = train(df, vocab_size=16_000)

# From a HuggingFace Dataset
from datasets import load_dataset
ds = load_dataset("ai4bharat/sangraha", "verified", split="train")
tok = train(ds, vocab_size=32_000)

# From a raw text string
tok = train("नमस्ते दुनिया " * 10_000, vocab_size=4_000)

# Custom text column name
tok = train("corpus.parquet", vocab_size=16_000, text_column="content")
```

---

### Mode 2: SentencePiece (Python)

```python
from indic_tokenizer import train

tok = train(
    "corpus.parquet",
    vocab_size=16_000,
    algorithm="sentencepiece",
    output_dir="models/",
    model_prefix="indic_sp",      # writes models/indic_sp.model
)

# Encode / decode
ids    = tok.encode("नमस्ते दुनिया")
pieces = tok.encode_as_pieces("நான் தமிழ் கற்கிறேன்")
text   = tok.decode(ids)

# Load a previously saved model
from indic_tokenizer import IndicSentencePieceTokenizer
tok2 = IndicSentencePieceTokenizer()
tok2.load("models/indic_sp.model")

# Round-trip sanity check
tok2.verify("வணக்கம் உலகம்")
```

---

### Mode 3: Chunked BPE (Python)

Use chunked mode when the corpus is too large to fit in RAM.
Word-frequency BPE uses ~600 MB regardless of corpus size; the full
token-ID sequence is never materialised.

Chunked BPE has the same two phases as the CLI: accumulate, then finalize.

#### Phase 1 — Accumulate (one or more sessions)

```python
from indic_tokenizer import IndicBPETokenizer, ChunkedBPETrainer
import os

CHECKPOINT = "checkpoints/state.json"

tok     = IndicBPETokenizer()
trainer = ChunkedBPETrainer(tok)

# Resume if a checkpoint already exists
if os.path.isfile(CHECKPOINT):
    trainer.load_state(CHECKPOINT)

# Add files (already-ingested files are tracked and skipped)
trainer.add_file("data/corpus-part0.parquet")
trainer.add_file("data/corpus-part1.parquet")
trainer.add_file("data/corpus-part2.txt")   # mixed formats are fine

# Or add an entire directory with a glob
trainer.add_directory("data/sangraha/*.parquet")

# Save checkpoint — run again later with more files to resume
trainer.save_state(CHECKPOINT)

print("Words accumulated:", trainer.unique_word_count)
print("Files done:", trainer.files_done)
# BPE has NOT been run yet — exit here and come back later
```

#### Phase 2 — Finalize (run once)

```python
from indic_tokenizer import IndicBPETokenizer, ChunkedBPETrainer

CHECKPOINT = "checkpoints/state.json"

tok     = IndicBPETokenizer()
trainer = ChunkedBPETrainer(tok)
trainer.load_state(CHECKPOINT)           # load all accumulated frequencies

trainer.finalize_training(
    vocab_size=32_000,
    min_frequency=2,                     # prune very rare words
    verbose=True,
)
tok.save("vocab.json", "merges.json")
```

#### Multi-session workflow (corpus spread across days)

```python
# Day 1
from indic_tokenizer import IndicBPETokenizer, ChunkedBPETrainer

trainer = ChunkedBPETrainer(IndicBPETokenizer())
trainer.add_directory("data/part-0*.parquet")
trainer.save_state("checkpoints/state.json")

# Day 2 — resume (part-0* files already in checkpoint, automatically skipped)
trainer = ChunkedBPETrainer(IndicBPETokenizer())
trainer.load_state("checkpoints/state.json")
trainer.add_directory("data/part-1*.parquet")
trainer.save_state("checkpoints/state.json")   # update checkpoint in-place

# Day 3 — finalize, no files needed
trainer = ChunkedBPETrainer(IndicBPETokenizer())
trainer.load_state("checkpoints/state.json")
trainer.finalize_training(vocab_size=32_000, min_frequency=2)
trainer.tokenizer.save("vocab.json", "merges.json")
```

#### Using the `train()` shortcut for a single accumulation + finalize

If you can do everything in one Python process but still want the memory
efficiency of chunked mode:

```python
from indic_tokenizer import train

# train() with mode="chunked" returns a ChunkedBPETrainer (Phase 1 done)
trainer = train(
    "data/corpus-part0.parquet",
    algorithm="bpe",
    mode="chunked",
)

# Add more files
trainer.add_file("data/corpus-part1.parquet")
trainer.save_state("checkpoints/state.json")

# Phase 2 — finalize
trainer.finalize_training(vocab_size=32_000, min_frequency=2)
trainer.tokenizer.save("vocab.json", "merges.json")
```

---

### Inference

Load a saved BPE tokenizer and encode / decode text in any Indic language.

```python
from indic_tokenizer import IndicBPETokenizer

tok = IndicBPETokenizer()
tok.load("vocab.json", "merges.json")

# Encode — returns a list of integer token IDs
ids = tok.encode("नमस्ते दुनिया")       # Hindi
ids = tok.encode("வணக்கம் உலகம்")       # Tamil
ids = tok.encode("నమస్కారం!")            # Telugu
ids = tok.encode("নমস্কার বিশ্ব")        # Bengali

# Decode — plain string join, no post-processing needed
text = tok.decode(ids)

# Chat-template special tokens pass through untouched
ids = tok.encode("<|im_start|>user\nHello<|im_end|>")
print(tok.decode(ids))   # <|im_start|>user\nHello<|im_end|>

# Vocabulary helpers
tok.token_to_id("नमस्ते")    # -> int or None
tok.id_to_token(256)          # -> str or None
tok.special_token_id("<|endoftext|>")
print("Vocab size:", tok.vocab_size)
```

---

### Using IndicDataLoader directly

`IndicDataLoader` can be used independently to load and concatenate a corpus
from any supported format into a single UTF-8 string.

```python
from indic_tokenizer import IndicDataLoader

loader = IndicDataLoader(
    text_column="text",           # column / key name for text
    end_of_text="<|endoftext|>",  # separator inserted between documents
    max_samples=50_000,           # optional cap on rows
)

# Auto-detects format from file extension
text = loader.load("corpus.parquet")
text = loader.load("corpus.csv")
text = loader.load("corpus.txt")
text = loader.load("corpus.json")
text = loader.load("corpus.jsonl")

# From a DataFrame or HuggingFace Dataset
import pandas as pd
df   = pd.read_parquet("corpus.parquet")
text = loader.load(df)

# From a raw string — returned as-is
text = loader.load("नमस्ते दुनिया")
```

---

## API Reference

### `train()` — high-level helper

```python
from indic_tokenizer import train

train(
    source,                           # file path, DataFrame, Dataset, or str
    vocab_size=16_000,                # target vocabulary size
    algorithm="bpe",                  # "bpe" | "sentencepiece"
    mode="single",                    # "single" | "chunked"
    output_dir=".",                   # SentencePiece output directory
    model_prefix="indic_tokenizer",   # SentencePiece file prefix
    special_tokens=None,              # None -> full token set
    text_column="text",               # column / key for text
    min_frequency=2,                  # chunked BPE: prune low-freq words
    verbose=True,                     # show progress
)
```

**Return values:**

| `algorithm` | `mode` | Returns |
|-------------|--------|---------|
| `"bpe"` | `"single"` | `IndicBPETokenizer` — call `.save()` |
| `"bpe"` | `"chunked"` | `ChunkedBPETrainer` — call `.finalize_training()` then `.tokenizer.save()` |
| `"sentencepiece"` | any | `IndicSentencePieceTokenizer` — model already saved to disk |

---

### `IndicBPETokenizer`

| Method | Description |
|--------|-------------|
| `train(text, vocab_size, ...)` | Train from a raw text string |
| `train_from_file(path, vocab_size, ...)` | Train from a file (auto-detects format) |
| `encode(text, allowed_special=None)` | Text -> `list[int]` |
| `decode(token_ids)` | `list[int]` -> text |
| `save(vocab_path, merges_path)` | Persist to two JSON files |
| `load(vocab_path, merges_path)` | Restore from JSON files |
| `token_to_id(token)` | String -> int or None |
| `id_to_token(tid)` | int -> string or None |
| `vocab_size` | Property: current vocabulary size |

---

### `ChunkedBPETrainer`

| Method | Description |
|--------|-------------|
| `add_text(text)` | Add a raw string chunk (Phase 1) |
| `add_file(path, ...)` | Add a file — any supported format (Phase 1) |
| `add_directory(pattern, ...)` | Add all files matching a glob pattern (Phase 1) |
| `add_from_dataset(dataset, ...)` | Add from a HuggingFace Dataset or DataFrame (Phase 1) |
| `finalize_training(vocab_size, ...)` | Run BPE on accumulated frequencies, update `self.tokenizer` (Phase 2) |
| `save_state(path)` | Checkpoint word frequencies + ingested file list to JSON |
| `load_state(path)` | Restore checkpoint (merges into current counts, does not replace) |
| `unique_word_count` | Property: number of distinct word types seen so far |
| `files_done` | Property: number of files fully processed |

---

### `IndicSentencePieceTokenizer`

| Method | Description |
|--------|-------------|
| `train_from_file(path, ...)` | Train from any supported file |
| `train_from_text(text, ...)` | Train from a raw string |
| `train_from_dataset(dataset, ...)` | Train from a DataFrame or HF Dataset |
| `load(model_path)` | Load a pre-trained `.model` file |
| `encode(text, add_bos, add_eos)` | Text -> `list[int]` |
| `encode_as_pieces(text)` | Text -> `list[str]` (subword pieces) |
| `decode(ids)` | `list[int]` -> text |
| `verify(text)` | Round-trip sanity check (prints results) |
| `vocab_size` | Property: number of pieces |

---

## Package Structure

```
indic_tokenizer/
├── __init__.py           Public API, train() helper, CLI main()
├── __main__.py           Enables  python -m indic_tokenizer
├── constants.py          Indic Unicode ranges (13 scripts), special tokens
├── preprocessor.py       IndicTextPreprocessor, pretokenize()
├── vocab_builder.py      VocabBuilder — seeds all Indic script characters
├── bpe_trainer.py        BPETrainer — stateless numpy-accelerated algorithm
├── bpe_tokenizer.py      IndicBPETokenizer — main BPE class
├── sp_tokenizer.py       IndicSentencePieceTokenizer
├── chunked_trainer.py    ChunkedBPETrainer — memory-efficient large-corpus BPE
├── data_loader.py        IndicDataLoader — multi-format corpus loader
└── requirements.txt      Package dependencies
```

---

## Special Tokens

All 27 special tokens are injected into the vocabulary *before* BPE training,
guaranteeing they are never split into sub-pieces.

| Group | Tokens |
|-------|--------|
| Text boundary | `<\|endoftext\|>` `<\|startoftext\|>` `<\|bos\|>` `<\|eos\|>` `<\|pad\|>` `<\|unk\|>` |
| Chat template | `<\|im_start\|>` `<\|im_end\|>` `<\|system\|>` `<\|user\|>` `<\|assistant\|>` `<\|human\|>` |
| Llama-3 style | `<\|eot_id\|>` `<\|start_header_id\|>` `<\|end_header_id\|>` |
| Fill-in-middle | `<\|fim_prefix\|>` `<\|fim_middle\|>` `<\|fim_suffix\|>` |
| Tool use | `<\|tool_use\|>` `<\|tool_result\|>` `<\|tool_call\|>` |
| Extended thinking | `<\|thinking\|>` `<\|/thinking\|>` |
| General | `<\|cls\|>` `<\|sep\|>` `<\|mask\|>` `<\|citation\|>` |

---

## Vocabulary Construction

```
Step 1 — 256 ASCII code points  (chr(0) ... chr(255))
Step 2 — All assigned Unicode characters across 13 Indic script blocks
           Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
           Kannada, Malayalam, Sinhala, Tibetan, Devanagari Extended, Vedic
Step 3 — Any additional characters found in the training corpus
Step 4 — 27 special tokens (injected last so BPE never splits them)
------------------------------------------------------------------
Base vocab ~= 1 200 tokens  ->  BPE grows this to vocab_size
```

---

## Recommended Vocabulary Sizes

| Use case | `vocab_size` | Approx. training data |
|----------|-------------|----------------------|
| Quick experimentation | 4 000 – 8 000 | 10 000 sentences |
| Single-language production | 16 000 – 32 000 | 500 000+ sentences |
| Multilingual Indic | 32 000 – 64 000 | 1 000 000+ sentences |

---

## License

MIT — see [LICENSE](../LICENSE).
