# indic-tokenizer

A BPE and SentencePiece tokenizer built from scratch for **all major Indian languages**, with Claude-style special token support.

Supported scripts: **Devanagari** · **Bengali** · **Gurmukhi** · **Gujarati** · **Oriya** · **Tamil** · **Telugu** · **Kannada** · **Malayalam** · **Sinhala** · **Tibetan** · ASCII

---

## Table of Contents

1. [Installation](#installation)
2. [Supported File Formats](#supported-file-formats)
3. [Command-Line Usage](#command-line-usage)
4. [Python API](#python-api)
   - [Single-shot BPE](#single-shot-bpe-small--medium-corpus)
   - [Chunked BPE (large corpora)](#chunked-bpe-large-corpora)
   - [SentencePiece](#sentencepiece)
   - [Inference](#inference)
   - [Using IndicDataLoader directly](#using-indicdataloader-directly)
5. [API Reference](#api-reference)
6. [Package Structure](#package-structure)
7. [Special Tokens](#special-tokens)
8. [Vocabulary Construction](#vocabulary-construction)
9. [Recommended Vocabulary Sizes](#recommended-vocabulary-sizes)

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

## Command-Line Usage

The package is runnable directly from the terminal via `python -m indic_tokenizer`.

### Synopsis

```
python -m indic_tokenizer --input PATH [OPTIONS]
```

### All flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | **required** | Corpus file (`.parquet`, `.csv`, `.txt`, `.json`, `.jsonl`) or glob pattern |
| `--vocab-size` | `-v` | `16000` | Target vocabulary size |
| `--algorithm` | `-a` | `bpe` | `bpe` or `sentencepiece` |
| `--mode` | `-m` | `single` | `single` (one-shot) or `chunked` (stream, resumable) |
| `--vocab-out` | | `vocab.json` | Output path for BPE vocabulary |
| `--merges-out` | | `merges.json` | Output path for BPE merges |
| `--output-dir` | `-o` | `.` | Directory for SentencePiece `.model` file |
| `--model-prefix` | | `indic_tokenizer` | SentencePiece filename prefix |
| `--text-column` | | `text` | Column / key name for text in tabular or JSON sources |
| `--max-samples` | | all | Cap on rows / documents loaded |
| `--checkpoint` | | none | Chunked BPE checkpoint file (auto-resumes if it exists) |
| `--min-frequency` | | `2` | Prune words below this frequency before chunked BPE |
| `--batch-size` | | `5000` | Rows per batch in chunked mode |
| `--quiet` | `-q` | off | Suppress all progress output |
| `--version` | | | Print version and exit |

### CLI examples

#### BPE — single-shot from a parquet file

```bash
python -m indic_tokenizer \
    --input corpus.parquet \
    --vocab-size 16000 \
    --vocab-out vocab.json \
    --merges-out merges.json
```

#### BPE — single-shot from a plain-text file

```bash
python -m indic_tokenizer \
    --input corpus.txt \
    --vocab-size 8000
```

#### BPE — single-shot from a JSON Lines file with a custom text key

```bash
python -m indic_tokenizer \
    --input corpus.jsonl \
    --vocab-size 16000 \
    --text-column content
```

#### BPE — chunked training from a single large file

```bash
python -m indic_tokenizer \
    --input data/large_corpus.parquet \
    --mode chunked \
    --vocab-size 32000 \
    --checkpoint checkpoints/state.json
```

> The checkpoint is created (or updated) after ingestion.  
> Re-run the **same command** in a later session to resume — already-processed
> files are skipped automatically.

#### BPE — chunked training from a directory of parquet files

```bash
# Session 1: process files matching the glob, save checkpoint
python -m indic_tokenizer \
    --input "data/sangraha/*.parquet" \
    --mode chunked \
    --vocab-size 32000 \
    --checkpoint checkpoints/state.json \
    --min-frequency 2

# Session 2: resume from checkpoint (already-done files skipped)
python -m indic_tokenizer \
    --input "data/sangraha/*.parquet" \
    --mode chunked \
    --vocab-size 32000 \
    --checkpoint checkpoints/state.json
```

#### SentencePiece from a plain-text file

```bash
python -m indic_tokenizer \
    --input corpus.txt \
    --algorithm sentencepiece \
    --vocab-size 16000 \
    --output-dir models/ \
    --model-prefix indic_sp
```

#### Load only a subset of rows (useful for quick tests)

```bash
python -m indic_tokenizer \
    --input corpus.parquet \
    --vocab-size 4000 \
    --max-samples 50000 \
    --quiet
```

---

## Python API

### Single-shot BPE (small / medium corpus)

Best when the entire corpus fits comfortably in RAM (up to ~1 GB).

```python
from indic_tokenizer import train

# --- From a parquet file ---
tok = train("corpus.parquet", vocab_size=16_000)
tok.save("vocab.json", "merges.json")

# --- From a CSV file ---
tok = train("corpus.csv", vocab_size=16_000)

# --- From a plain-text file ---
# Each line is a document; lines joined with <|endoftext|>
tok = train("corpus.txt", vocab_size=8_000)

# --- From a JSON Lines file ---
tok = train("corpus.jsonl", vocab_size=16_000)

# --- From a JSON file (array of {"text": "..."} objects) ---
tok = train("corpus.json", vocab_size=16_000)

# --- From a pandas DataFrame ---
import pandas as pd
df = pd.read_parquet("corpus.parquet")
tok = train(df, vocab_size=16_000)

# --- From a HuggingFace Dataset ---
from datasets import load_dataset
ds = load_dataset("ai4bharat/sangraha", "verified", split="train")
tok = train(ds, vocab_size=32_000)

# --- From a raw text string ---
tok = train("नमस्ते दुनिया " * 10_000, vocab_size=4_000)

# --- Custom text column name ---
tok = train("corpus.parquet", vocab_size=16_000, text_column="content")
```

---

### Chunked BPE (large corpora)

Use `mode="chunked"` when the corpus is too large to fit in RAM.  
Word-frequency BPE uses ~600 MB regardless of corpus size; the full
token-ID sequence is never materialised.

#### Option A — via `train()` shortcut

```python
from indic_tokenizer import train

# Step 1: ingest first file — returns a ChunkedBPETrainer
trainer = train(
    "data/corpus-part0.parquet",
    algorithm="bpe",
    mode="chunked",
)

# Step 2: add more files one by one
trainer.add_file("data/corpus-part1.parquet")
trainer.add_file("data/corpus-part2.txt")    # mixed formats are fine

# Step 3: save a checkpoint (recommended before shutting down)
trainer.save_state("checkpoints/state.json")

# Step 4: finalise and save
trainer.finalize_training(vocab_size=32_000, min_frequency=2)
trainer.tokenizer.save("vocab.json", "merges.json")
```

#### Option B — via `ChunkedBPETrainer` directly

```python
from indic_tokenizer import IndicBPETokenizer, ChunkedBPETrainer

tok     = IndicBPETokenizer()
trainer = ChunkedBPETrainer(tok)

# Add an entire directory using a glob pattern
trainer.add_directory("data/sangraha/*.parquet")

# Save checkpoint
trainer.save_state("checkpoints/state.json")

# Finalise
trainer.finalize_training(vocab_size=32_000, min_frequency=2)
tok.save("vocab.json", "merges.json")
```

#### Resuming a previous session

```python
from indic_tokenizer import IndicBPETokenizer, ChunkedBPETrainer

tok     = IndicBPETokenizer()
trainer = ChunkedBPETrainer(tok)

# load_state() merges word frequencies and marks previously ingested files
trainer.load_state("checkpoints/state.json")

# add_directory() automatically skips files listed in the checkpoint
trainer.add_directory("data/sangraha/*.parquet")

trainer.save_state("checkpoints/state.json")   # update checkpoint
trainer.finalize_training(vocab_size=32_000)
tok.save("vocab.json", "merges.json")
```

#### Multi-session workflow (corpus spread across days)

```python
# Day 1
trainer = ChunkedBPETrainer(IndicBPETokenizer())
trainer.add_directory("data/part-0*.parquet")
trainer.save_state("checkpoints/day1.json")

# Day 2
trainer = ChunkedBPETrainer(IndicBPETokenizer())
trainer.load_state("checkpoints/day1.json")
trainer.add_directory("data/part-1*.parquet")   # part-0* skipped
trainer.save_state("checkpoints/day2.json")

# Day 3 — finalise
trainer = ChunkedBPETrainer(IndicBPETokenizer())
trainer.load_state("checkpoints/day2.json")
trainer.finalize_training(vocab_size=32_000, min_frequency=2)
trainer.tokenizer.save("vocab.json", "merges.json")
```

---

### SentencePiece

```python
from indic_tokenizer import train

# From any supported file format
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

# Quick round-trip sanity check
tok2.verify("வணக்கம் உலகம்")
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
    special_tokens=None,              # None -> full Claude token set
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
| `add_text(text)` | Add a raw string chunk |
| `add_file(path, ...)` | Add a file (any supported format) |
| `add_directory(pattern, ...)` | Add all files matching a glob pattern |
| `add_from_dataset(dataset, ...)` | Add from a HuggingFace Dataset or DataFrame |
| `finalize_training(vocab_size, ...)` | Run BPE and update `self.tokenizer` |
| `save_state(path)` | Checkpoint word frequencies to JSON |
| `load_state(path)` | Restore from checkpoint (merges, does not replace) |
| `unique_word_count` | Property: number of distinct word types seen |
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
Base vocab ~= 1 200 tokens  -->  BPE grows this to vocab_size
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
=======
# indic_tokenizer
This repository contains code to tokenize indic languages using either BPE or SentencePiece Algorithm
>>>>>>> ab5deaaff3efb6d76c6bb2a18fd3dca46a5bf7fe
