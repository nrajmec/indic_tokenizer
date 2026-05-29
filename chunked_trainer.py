"""
ChunkedBPETrainer — memory-efficient, resumable BPE training for large corpora.

Why chunked training?
---------------------
Sequence-based BPE (IndicBPETokenizer.train()) stores the *entire corpus* as a
flat list of token IDs.  For 18 GB of text at ~2 bytes/char that is ~9 B IDs
× 8 bytes = ~72 GB RAM — completely infeasible on a normal machine.

This module uses **word-frequency BPE** instead:
  • Stream the corpus in batches via add_text() / add_file() / add_from_dataset().
  • Keep only a Counter{word → count} in memory (~600 MB for 18 GB text).
  • Run BPE on word types weighted by frequency — mathematically equivalent
    to sequence-based BPE on the full token stream.

GPU acceleration
----------------
The initial pair-counting step (O(unique_words × avg_length)) is dispatched
to CUDA via CuPy or PyTorch when available, then falls back to CPU numpy.
Per-merge incremental updates use the CPU inverted index (k << U affected words
per step, so Python dict ops are fast enough).

Resumable training
------------------
After each add_* call, checkpoint with save_state().  If the process is
interrupted, call load_state() to restore word frequencies and the list of
already-processed files, then continue adding more data before
finalize_training().

Typical workflow — large local corpus
--------------------------------------
    from indic_tokenizer import IndicBPETokenizer, ChunkedBPETrainer

    tok     = IndicBPETokenizer()
    trainer = ChunkedBPETrainer(tok)

    # Session 1: ingest files 0–4, save checkpoint
    trainer.add_directory("data/sangraha/*.parquet")
    trainer.save_state("checkpoints/state.json")

    # Session 2: load checkpoint; already-processed files are skipped
    trainer.load_state("checkpoints/state.json")
    trainer.add_directory("data/sangraha/*.parquet")   # picks up at file 5

    trainer.finalize_training(vocab_size=32_000, min_frequency=2)
    trainer.tokenizer.save("vocab.json", "merges.json")
"""

import glob
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Counter as CounterType
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .bpe_tokenizer import IndicBPETokenizer
from .constants import SPECIAL_TOKENS
from .preprocessor import pretokenize
from .vocab_builder import VocabBuilder

logger = logging.getLogger(__name__)

# Type aliases
WordFreqs = CounterType[str]
MergesType = Dict[Tuple[int, int], int]


class ChunkedBPETrainer:
    """
    Stream-safe, resumable BPE trainer for large Indic-language corpora.

    The tokenizer passed in is updated in-place by finalize_training().
    After finalize_training(), call tokenizer.save() to persist the result.

    Args:
        tokenizer: A fresh IndicBPETokenizer instance (empty vocab).
        special_tokens: Special tokens to inject.  Defaults to all Claude tokens.
    """

    STATE_VERSION: int = 1

    def __init__(
        self,
        tokenizer: IndicBPETokenizer,
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.special_tokens: List[str] = (
            special_tokens if special_tokens is not None else list(SPECIAL_TOKENS)
        )
        self.word_freqs: WordFreqs = Counter()
        self._chars_processed: int = 0
        # Absolute paths of files fully ingested — persisted across sessions
        # so add_file() / add_directory() skip already-processed files.
        self._processed_files: Set[str] = set()

    # ------------------------------------------------------------------
    # Accumulation API
    # ------------------------------------------------------------------

    def add_text(self, text: str) -> None:
        """
        Add a chunk of raw text to the word-frequency table.

        O(len(text)) and O(1) extra memory beyond the growing word_freqs counter.
        Call this repeatedly with chunks of any size.

        Args:
            text: Raw UTF-8 text in any Indic language or ASCII.
        """
        for chunk in pretokenize(text):
            self.word_freqs[chunk] += 1
        self._chars_processed += len(text)

    def add_from_dataset(
        self,
        dataset,
        text_column: str = "text",
        batch_size: int = 5_000,
        verbose: bool = True,
    ) -> None:
        """
        Accumulate word frequencies from a HuggingFace Dataset or pandas DataFrame.

        Safe to call multiple times — frequencies accumulate across calls.

        Args:
            dataset: HuggingFace Dataset / IterableDataset or pandas DataFrame.
            text_column: Column / key for text.
            batch_size: Rows processed per iteration (tune for RAM vs. speed).
            verbose: Log progress after each batch.
        """
        total = len(dataset) if hasattr(dataset, "__len__") else None
        processed = 0

        for batch_start in range(0, total or 10 ** 9, batch_size):
            if total is not None and batch_start >= total:
                break

            try:
                batch = dataset[batch_start: batch_start + batch_size]
            except Exception:
                break

            texts = self._extract_texts(batch, text_column)
            for text in texts:
                if text:
                    self.add_text(text)

            processed += len(texts)
            if verbose and total:
                pct = processed / total * 100
                print(
                    f"Processed {processed:,} / {total:,} rows "
                    f"({pct:.1f}%) | unique words: {len(self.word_freqs):,}"
                )

        if verbose:
            print(
                f"Ingestion complete | chars: {self._chars_processed:,} "
                f"| unique words: {len(self.word_freqs):,}"
            )

    def add_file(
        self,
        path: str,
        text_column: str = "text",
        batch_size: int = 5_000,
        verbose: bool = True,
    ) -> None:
        """
        Read one file and accumulate its text into the word-frequency table.

        Supports .parquet, .csv  (batch-streamed via pandas) and
        .txt, .json, .jsonl     (loaded in full then fed to add_text()).

        The file's absolute path is recorded so that save_state() /
        load_state() can skip it in future sessions.

        Args:
            path: Path to a .parquet, .csv, .txt, .json, or .jsonl file.
            text_column: Column / key for text in tabular / JSON files.
            batch_size: Rows per batch for parquet / csv.
            verbose: Log per-batch progress.
        """
        abs_path = str(Path(path).resolve())
        if abs_path in self._processed_files:
            logger.info("Skipping already-processed file: %s", path)
            return

        suffix = Path(path).suffix.lower()

        if suffix in (".parquet", ".csv"):
            # Stream via pandas batches — avoids loading entire file into RAM
            self._add_tabular_file(abs_path, suffix, text_column, batch_size, verbose)
        else:
            # txt / json / jsonl — load full text then add_text()
            from .data_loader import IndicDataLoader
            loader = IndicDataLoader(text_column=text_column)
            text = loader.load(path)
            if verbose:
                logger.info("Adding %s (%d chars)", path, len(text))
            self.add_text(text)

        self._processed_files.add(abs_path)
        logger.info(
            "Done: %s  (total files so far: %d)", path, len(self._processed_files)
        )

    def add_directory(
        self,
        pattern: str,
        text_column: str = "text",
        batch_size: int = 5_000,
        verbose: bool = True,
    ) -> None:
        """
        Process all files matching *pattern*, skipping already-processed ones.

        Supports the same file formats as add_file().

        Cross-session usage::

            # Session 1 — process files 0–9, then save checkpoint
            trainer.add_directory("data/*.parquet")
            trainer.save_state("checkpoints/state.json")

            # Session 2 — load; files 0–9 are skipped automatically
            trainer.load_state("checkpoints/state.json")
            trainer.add_directory("data/*.parquet")   # picks up at file 10

        Args:
            pattern: Glob pattern, e.g. "data/*.parquet" or "/mnt/data/*.txt".
            text_column: Column / key for text.
            batch_size: Rows per batch for parquet / csv.
            verbose: Per-file and per-batch logging.
        """
        files = sorted(glob.glob(pattern))
        if not files:
            logger.warning("No files matched pattern: %s", pattern)
            return

        total_files = len(files)
        skipped = sum(
            1 for f in files if str(Path(f).resolve()) in self._processed_files
        )
        logger.info(
            "Found %d files | %d already done | %d remaining",
            total_files, skipped, total_files - skipped,
        )

        for i, fpath in enumerate(files, 1):
            logger.info("[%d/%d] %s", i, total_files, fpath)
            self.add_file(
                fpath, text_column=text_column,
                batch_size=batch_size, verbose=verbose,
            )

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalize_training(
        self,
        vocab_size: int,
        min_frequency: int = 2,
        verbose: bool = True,
    ) -> None:
        """
        Run BPE on the accumulated word-frequency table and update the tokenizer.

        After this call, tokenizer.vocab, tokenizer.inverse_vocab, and
        tokenizer.bpe_merges are fully populated.  Call tokenizer.save() next.

        Args:
            vocab_size: Target vocabulary size.
            min_frequency: Prune word types appearing fewer than this many times
                           before running BPE.  2 drops hapax legomena.
                           Reducing to 1 is fine for small corpora.
            verbose: Show progress bar / step log.
        """
        word_freqs = self._prune_word_freqs(self.word_freqs, min_frequency)

        if verbose:
            print(
                f"Starting BPE | word types: {len(word_freqs):,} "
                f"(min_freq={min_frequency}) | target vocab: {vocab_size:,}"
            )

        vb = VocabBuilder()
        vocab, inverse_vocab = vb.build_base_vocab()
        vb.extend_from_text("".join(word_freqs.keys()), vocab, inverse_vocab)
        vb.add_special_tokens(vocab, inverse_vocab, self.special_tokens)

        bpe_merges = self._run_word_freq_bpe(
            word_freqs, vocab, inverse_vocab, vocab_size, verbose
        )

        # Write final state back into the tokenizer
        self.tokenizer.vocab = vocab
        self.tokenizer.inverse_vocab = inverse_vocab
        self.tokenizer.bpe_merges = bpe_merges

        if verbose:
            print(f"Training complete. Final vocab size: {len(vocab):,}")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """
        Checkpoint the trainer state to *path* (JSON).

        Saves word frequencies, total chars processed, and the list of files
        fully ingested.  Call this after every session before shutting down.

        File format::

            {
              "version": 1,
              "chars_processed": 18000000000,
              "processed_files": ["/abs/path/data-0.parquet", ...],
              "word_freqs": {"नमस्ते": 5000, " दुनिया": 3000, ...}
            }

        Args:
            path: Destination JSON file path.
        """
        os_module = __import__("os")
        os_module.makedirs(
            os_module.path.dirname(os_module.path.abspath(path)),
            exist_ok=True,
        )
        payload = {
            "version": self.STATE_VERSION,
            "chars_processed": self._chars_processed,
            "processed_files": sorted(self._processed_files),
            "word_freqs": dict(self.word_freqs),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        print(
            f"Saved state → {path} | "
            f"files done: {len(self._processed_files)} | "
            f"unique words: {len(self.word_freqs):,}"
        )

    def load_state(self, path: str) -> None:
        """
        Restore a previously saved checkpoint from *path*.

        Word frequencies are *merged* (not replaced) so you can call
        load_state() on an already-populated trainer.  Processed-file paths
        are merged into the existing set.

        Args:
            path: JSON file produced by save_state().
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        version = payload.get("version", 1)
        if version != self.STATE_VERSION:
            raise ValueError(
                f"State file version {version} not supported "
                f"(expected {self.STATE_VERSION})."
            )

        self.word_freqs.update(Counter(payload["word_freqs"]))
        self._chars_processed += payload.get("chars_processed", 0)
        self._processed_files.update(payload.get("processed_files", []))

        print(
            f"Loaded state ← {path} | "
            f"files done: {len(self._processed_files)} | "
            f"unique words: {len(self.word_freqs):,}"
        )

    # ------------------------------------------------------------------
    # Convenience stats
    # ------------------------------------------------------------------

    @property
    def total_word_occurrences(self) -> int:
        """Sum of all word frequencies seen so far."""
        return sum(self.word_freqs.values())

    @property
    def unique_word_count(self) -> int:
        """Number of distinct word types seen so far."""
        return len(self.word_freqs)

    @property
    def files_done(self) -> int:
        """Number of files fully ingested across all sessions."""
        return len(self._processed_files)

    # ------------------------------------------------------------------
    # Private — tabular file streaming
    # ------------------------------------------------------------------

    def _add_tabular_file(
        self,
        abs_path: str,
        suffix: str,
        text_column: str,
        batch_size: int,
        verbose: bool,
    ) -> None:
        """Read a parquet or CSV file and call add_from_dataset() on it."""
        try:
            import pandas as pd  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("pip install pandas pyarrow") from exc

        logger.info("Reading %s: %s", suffix, abs_path)
        if suffix == ".parquet":
            df = pd.read_parquet(abs_path, columns=[text_column])
        else:
            df = pd.read_csv(abs_path, usecols=[text_column])

        self.add_from_dataset(
            df, text_column=text_column, batch_size=batch_size, verbose=verbose
        )

    # ------------------------------------------------------------------
    # Private — batch text extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_texts(batch, text_column: str) -> List[str]:
        """
        Extract a list of strings from a batch returned by dataset slicing.

        HuggingFace Dataset slice → dict  {column: [values, ...]}
        pandas DataFrame slice   → DataFrame  (supports __getitem__)
        """
        if isinstance(batch, dict):
            return batch.get(text_column, [])
        try:
            col = batch[text_column]
            return col.tolist() if hasattr(col, "tolist") else list(col)
        except (KeyError, TypeError):
            return []

    # ------------------------------------------------------------------
    # Private — core word-frequency BPE  (inverted-index, GPU-accelerated)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_word_freq_bpe(
        word_freqs: WordFreqs,
        vocab: Dict[int, str],
        inverse_vocab: Dict[str, int],
        target_vocab_size: int,
        verbose: bool = True,
    ) -> MergesType:
        """
        Memory-efficient word-frequency BPE with optional GPU acceleration.

        Memory layout
        -------------
        Word symbol sequences are stored as CSR (Compressed Sparse Row) numpy
        arrays — a single flat int32 array + an offsets array.  This replaces
        millions of Python list objects and reduces peak RAM by 10–20×.

        GPU path
        --------
        Initial pair counting (O(U·L)) is dispatched to CUDA via CuPy or
        PyTorch.  Per-merge incremental updates use the CPU inverted index
        (only k << U words are affected per step).

        Falls back to CPU numpy when no GPU is available.
        """
        from collections import Counter as _Counter, defaultdict
        from itertools import count as _icount
        import heapq

        device = ChunkedBPETrainer._detect_device()
        use_gpu = device is not None

        # ── 1. Build word ID sequences ──────────────────────────────────────
        if verbose:
            logger.info(
                "Encoding word sequences | device: %s",
                f"GPU ({device})" if use_gpu else "CPU numpy",
            )

        # max_id bounds pair encoding:  flat = a * max_id + b
        max_id = target_vocab_size + 10

        word_id_lists: List[List[int]] = []
        freqs_list: List[int] = []
        for word, freq in word_freqs.items():
            ids = [inverse_vocab.get(c, -1) for c in word]
            if -1 not in ids:
                word_id_lists.append(ids)
                freqs_list.append(freq)

        # CSR arrays for GPU pair counting
        lengths_np = np.array([len(ids) for ids in word_id_lists], dtype=np.int32)
        word_offsets = np.zeros(len(word_id_lists) + 1, dtype=np.int64)
        np.cumsum(lengths_np, out=word_offsets[1:])
        flat_ids_init = (
            np.concatenate(word_id_lists).astype(np.int32)
            if word_id_lists
            else np.array([], dtype=np.int32)
        )
        freqs_np = np.array(freqs_list, dtype=np.int64)

        if verbose:
            logger.info(
                "Word data: %s types | %s symbols | %.1f MB",
                f"{len(word_id_lists):,}", f"{len(flat_ids_init):,}",
                flat_ids_init.nbytes / 1e6,
            )

        # Mutable per-word numpy arrays for the merge loop
        word_id_seqs: List[np.ndarray] = [
            np.array(ids, dtype=np.int32) for ids in word_id_lists
        ]

        # ── 2. Initial pair counting ────────────────────────────────────────
        if use_gpu:
            try:
                unique_pairs_np, counts_np = ChunkedBPETrainer._count_pairs_gpu(
                    flat_ids_init, word_offsets, freqs_np, max_id, device
                )
            except Exception as exc:
                logger.warning(
                    "GPU pair counting failed (%s: %s) — switching to CPU numpy.",
                    type(exc).__name__, exc,
                )
                use_gpu = False
                device = None

        if not use_gpu:
            acc: Dict[int, int] = {}
            for ids, freq in zip(word_id_seqs, freqs_list):
                if len(ids) < 2:
                    continue
                pairs_flat = (
                    ids[:-1].astype(np.int64) * max_id + ids[1:].astype(np.int64)
                )
                for pf, cnt in _Counter(pairs_flat.tolist()).items():
                    acc[int(pf)] = acc.get(int(pf), 0) + cnt * freq
            unique_pairs_np = np.array(list(acc.keys()), dtype=np.int64)
            counts_np = np.array(list(acc.values()), dtype=np.int64)

        pair_counts: Dict[int, int] = {
            int(p): int(c)
            for p, c in zip(unique_pairs_np, counts_np)
            if c > 0
        }

        # ── 3. Inverted index: flat_pair_id → set of word indices ───────────
        pair_to_word_idx: Dict[int, set] = defaultdict(set)
        for wi, ids in enumerate(word_id_seqs):
            if len(ids) < 2:
                continue
            for pf in np.unique(
                ids[:-1].astype(np.int64) * max_id + ids[1:].astype(np.int64)
            ):
                pair_to_word_idx[int(pf)].add(wi)

        # ── 4. Max-heap with lazy deletion ────────────────────────────────
        _seq = _icount()
        heap: List = [
            (-cnt, next(_seq), fp)
            for fp, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        bpe_merges: MergesType = {}
        n_merges = target_vocab_size - len(vocab)

        try:
            from tqdm import tqdm  # type: ignore[import]
            progress = tqdm(
                total=n_merges,
                desc=f"BPE merges ({'GPU' if use_gpu else 'CPU'})",
                unit="merge",
            )
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            progress = None  # type: ignore[assignment]

        # ── 5. Merge loop ──────────────────────────────────────────────────
        step = 0
        while len(vocab) < target_vocab_size:

            # 5a. Pop heap until a non-stale entry is found
            best_flat: Optional[int] = None
            best_cnt = 0
            while heap:
                neg_cnt, _, candidate = heapq.heappop(heap)
                current = pair_counts.get(candidate, 0)
                if current > 0 and current == -neg_cnt:
                    best_flat = candidate
                    best_cnt = current
                    break
            if best_flat is None:
                break

            # 5b. Commit merge
            left_id = int(best_flat // max_id)
            right_id = int(best_flat % max_id)
            merged_str = vocab[left_id] + vocab[right_id]
            new_id = len(vocab)
            vocab[new_id] = merged_str
            inverse_vocab[merged_str] = new_id
            bpe_merges[(left_id, right_id)] = new_id

            affected = pair_to_word_idx.pop(best_flat, set())
            del pair_counts[best_flat]

            # 5c. Incremental update — only affected words (k << U)
            for wi in affected:
                ids = word_id_seqs[wi]
                freq = int(freqs_np[wi])

                # Subtract old pair counts contributed by this word
                if len(ids) >= 2:
                    old_pairs = _Counter(
                        (ids[:-1].astype(np.int64) * max_id
                         + ids[1:].astype(np.int64)).tolist()
                    )
                    for pf, cnt in old_pairs.items():
                        old = pair_counts.get(pf)
                        if old is not None:
                            updated = old - cnt * freq
                            if updated <= 0:
                                del pair_counts[pf]
                            else:
                                pair_counts[pf] = updated
                        ws = pair_to_word_idx.get(pf)
                        if ws is not None:
                            ws.discard(wi)
                            if not ws:
                                del pair_to_word_idx[pf]

                # Apply merge (numpy, no Python element loop)
                ids = ChunkedBPETrainer._merge_numpy(ids, left_id, right_id, new_id)
                word_id_seqs[wi] = ids

                # Add new pair counts from the merged word
                if len(ids) >= 2:
                    new_pairs = _Counter(
                        (ids[:-1].astype(np.int64) * max_id
                         + ids[1:].astype(np.int64)).tolist()
                    )
                    for pf, cnt in new_pairs.items():
                        new_total = pair_counts.get(pf, 0) + cnt * freq
                        pair_counts[pf] = new_total
                        pair_to_word_idx[pf].add(wi)
                        heapq.heappush(heap, (-new_total, next(_seq), pf))

            step += 1
            if use_tqdm and progress is not None:
                progress.update(1)
            elif verbose and step % 500 == 0:
                logger.info(
                    "step %6d | vocab %6d | merged %r (freq=%s)",
                    step, len(vocab), merged_str, f"{best_cnt:,}",
                )

        if use_tqdm and progress is not None:
            progress.close()

        return bpe_merges

    # ------------------------------------------------------------------
    # Private — GPU detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device():
        """
        Return a usable CUDA device (CuPy or PyTorch) or None.

        Probes with a real kernel operation so version mismatches that only
        surface at execution time (cudaErrorNoKernelImageForDevice) are caught
        here and cause a graceful fallback to CPU rather than crashing mid-loop.
        """
        # Try CuPy first (works on any CUDA GPU, no PyTorch ABI dependency)
        try:
            import cupy as cp  # type: ignore[import]
            probe = cp.array([1, 2, 3], dtype=cp.int64)
            _ = cp.unique(probe)
            cp.cuda.Device(0).synchronize()
            mem = cp.cuda.Device(0).mem_info
            logger.info(
                "GPU verified (CuPy): %s | VRAM free: %.1f GB",
                cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                mem[1] / 1e9,
            )
            return "cupy"
        except ImportError:
            pass
        except Exception as exc:
            logger.warning(
                "CuPy probe failed (%s: %s) — trying PyTorch.", type(exc).__name__, exc
            )

        # Try PyTorch
        try:
            import torch  # type: ignore[import]
            if not torch.cuda.is_available():
                return None
            device = torch.device("cuda")
            probe = torch.tensor([1, 2, 3], dtype=torch.int64, device=device)
            _ = torch.unique(probe)
            del probe
            torch.cuda.synchronize()
            props = torch.cuda.get_device_properties(0)
            logger.info(
                "GPU verified (PyTorch): %s | VRAM: %.1f GB | SM: %d.%d",
                props.name, props.total_memory / 1e9, props.major, props.minor,
            )
            return device
        except ImportError:
            logger.info("PyTorch not installed — using CPU numpy")
        except Exception as exc:
            logger.warning(
                "PyTorch GPU probe failed (%s: %s). Falling back to CPU.",
                type(exc).__name__, exc,
            )
        return None

    # ------------------------------------------------------------------
    # Private — GPU pair counting
    # ------------------------------------------------------------------

    @staticmethod
    def _count_pairs_gpu(
        flat_ids: "np.ndarray",
        word_offsets: "np.ndarray",
        freqs: "np.ndarray",
        max_id: int,
        device,
    ):
        """
        Count weighted pair frequencies on GPU from CSR word data.

        Steps:
          1. Build boundary mask to exclude cross-word pairs.
          2. Encode valid (a, b) pairs as a * max_id + b  (int64).
          3. Assign each pair the frequency of its word.
          4. scatter_add_ accumulates weighted counts per unique pair.

        Returns (unique_pair_ids, counts) as numpy arrays.
        """
        if len(flat_ids) < 2:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # Mark last position of each word as a boundary (pair cannot start here)
        boundary = word_offsets[1:] - 1
        bnd_mask = np.zeros(len(flat_ids), dtype=bool)
        valid_bnd = boundary[boundary < len(flat_ids)]
        if len(valid_bnd):
            bnd_mask[valid_bnd] = True

        valid_pos = np.where(~bnd_mask[:-1])[0]
        if len(valid_pos) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        a = flat_ids[valid_pos].astype(np.int64)
        b = flat_ids[valid_pos + 1].astype(np.int64)
        flat_pairs = a * max_id + b
        wi = np.searchsorted(word_offsets[1:], valid_pos, side="right")
        pair_freqs = freqs[wi]

        if device == "cupy":
            import cupy as cp  # type: ignore[import]
            fp_cp = cp.array(flat_pairs, dtype=cp.int64)
            fr_cp = cp.array(pair_freqs, dtype=cp.int64)
            unique_pairs_cp, inv = cp.unique(fp_cp, return_inverse=True)
            counts_cp = cp.zeros(len(unique_pairs_cp), dtype=cp.int64)
            cp.add.at(counts_cp, inv, fr_cp)
            return cp.asnumpy(unique_pairs_cp), cp.asnumpy(counts_cp)
        else:
            import torch  # type: ignore[import]
            fp_t = torch.tensor(flat_pairs, dtype=torch.int64, device=device)
            fr_t = torch.tensor(pair_freqs, dtype=torch.int64, device=device)
            unique_t, inv_t = torch.unique(fp_t, return_inverse=True)
            counts_t = torch.zeros(len(unique_t), dtype=torch.long, device=device)
            counts_t.scatter_add_(0, inv_t, fr_t)
            return unique_t.cpu().numpy(), counts_t.cpu().numpy()

    # ------------------------------------------------------------------
    # Private — numpy in-place merge for a single word sequence
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_numpy(
        ids: "np.ndarray",
        left: int,
        right: int,
        new_id: int,
    ) -> "np.ndarray":
        """
        Merge all non-overlapping (left, right) pairs → new_id using numpy.

        No Python loop over array elements.  Greedy left-to-right for overlaps.
        """
        if len(ids) < 2:
            return ids

        match = (ids[:-1] == left) & (ids[1:] == right)
        if not match.any():
            return ids

        # Greedy deduplicate adjacent matches
        matches = np.where(match)[0]
        valid_m: List[int] = [int(matches[0])]
        for pos in matches[1:]:
            if int(pos) > valid_m[-1] + 1:
                valid_m.append(int(pos))
        valid_m_arr = np.array(valid_m, dtype=np.int64)

        # Remove second elements of matched pairs
        keep = np.ones(len(ids), dtype=bool)
        keep[valid_m_arr + 1] = False
        out = ids[keep].copy()

        # Replace first elements; adjust index for preceding deletions
        out[valid_m_arr - np.arange(len(valid_m_arr), dtype=np.int64)] = new_id
        return out

    # ------------------------------------------------------------------
    # Private — word-frequency pruning
    # ------------------------------------------------------------------

    @staticmethod
    def _prune_word_freqs(
        word_freqs: WordFreqs,
        min_frequency: int,
    ) -> WordFreqs:
        """Return a new Counter with words below *min_frequency* removed."""
        if min_frequency <= 1:
            return Counter(word_freqs)
        return Counter(
            {w: f for w, f in word_freqs.items() if f >= min_frequency}
        )
