"""
Core BPE training algorithm — CPU-optimised with numpy.

BPETrainer is stateless: it receives mutable vocab / inverse_vocab dicts,
modifies them in-place, and returns only the bpe_merges dictionary.

Optimisations over the naive Python implementation
---------------------------------------------------
_find_most_frequent_pair
    Uses numpy to encode every adjacent pair as a single int64, then
    np.unique (C-level sort + scan) to count them.  For a 10 M-token
    sequence this is ~10× faster than Python's Counter(zip(...)).

_replace_pair
    Finds all match positions with a numpy boolean mask in one vectorised
    pass, deduplicates overlapping matches with a tiny Python loop
    (typically k << N matches), then builds the output array in numpy —
    no Python-level element-by-element loop.

Fallback
    If numpy is not installed both methods fall back to pure-Python
    implementations so the module is always importable.
"""

from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

# Type aliases exposed for callers
PairType = Tuple[int, int]
VocabType = Dict[int, str]
InverseVocabType = Dict[str, int]
MergesType = Dict[PairType, int]

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


class BPETrainer:
    """Stateless, numpy-accelerated BPE training engine."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        token_ids: List[int],
        vocab: VocabType,
        inverse_vocab: InverseVocabType,
        target_vocab_size: int,
        verbose: bool = True,
    ) -> MergesType:
        """
        Run BPE training until *target_vocab_size* is reached.

        Mutates *vocab* and *inverse_vocab* in-place with merged tokens.

        Args:
            token_ids: Sequence of initial token IDs from the corpus.
            vocab: Mutable id → token mapping (extended in-place).
            inverse_vocab: Mutable token → id mapping (extended in-place).
            target_vocab_size: Desired final vocabulary size.
            verbose: Show tqdm progress bar (or step log if tqdm missing).

        Returns:
            bpe_merges: dict mapping (id1, id2) → merged_id in merge order.
        """
        bpe_merges: MergesType = {}
        initial_size = len(vocab)

        try:
            from tqdm import tqdm  # type: ignore[import]
            progress = tqdm(
                total=target_vocab_size - initial_size,
                desc="BPE merges",
                unit="merge",
            )
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            progress = None  # type: ignore[assignment]

        step = 0
        while len(vocab) < target_vocab_size:
            pair = self._find_most_frequent_pair(token_ids)
            if pair is None:
                break  # no more pairs to merge

            new_id = len(vocab)
            token_ids = self._replace_pair(token_ids, pair, new_id)

            merged_token = vocab[pair[0]] + vocab[pair[1]]
            vocab[new_id] = merged_token
            inverse_vocab[merged_token] = new_id
            bpe_merges[pair] = new_id

            step += 1
            if use_tqdm and progress is not None:
                progress.update(1)
            elif verbose and step % 100 == 0:
                print(
                    f"  step {step:5d} | vocab {len(vocab):6d} "
                    f"| merged: {merged_token!r}"
                )

        if use_tqdm and progress is not None:
            progress.close()

        return bpe_merges

    # ------------------------------------------------------------------
    # Private helpers — numpy paths with pure-Python fallbacks
    # ------------------------------------------------------------------

    @staticmethod
    def _find_most_frequent_pair(
        token_ids: List[int],
    ) -> Optional[PairType]:
        """
        Return the most frequent adjacent pair in *token_ids*.

        numpy path  — O(N log N) via C-level sort in np.unique.
        Python path — O(N) via Counter (larger constant due to dict hashing).
        """
        if len(token_ids) < 2:
            return None

        if _NUMPY_AVAILABLE:
            arr = np.asarray(token_ids, dtype=np.int64)
            max_id = int(arr.max()) + 1
            # Encode pair (a, b) as a * max_id + b to get a single int64 key
            flat = arr[:-1] * max_id + arr[1:]
            unique_pairs, counts = np.unique(flat, return_counts=True)
            best_idx = int(np.argmax(counts))
            best = int(unique_pairs[best_idx])
            return (best // max_id, best % max_id)

        # Pure-Python fallback
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None
        return max(pairs, key=pairs.__getitem__)

    @staticmethod
    def _replace_pair(
        token_ids: List[int],
        pair: PairType,
        new_id: int,
    ) -> List[int]:
        """
        Replace every non-overlapping occurrence of *pair* with *new_id*.

        numpy path — vectorised boolean mask + one array copy, O(N).
        Python path — deque-based left-to-right scan, O(N).
        """
        a, b = pair

        if _NUMPY_AVAILABLE:
            arr = np.asarray(token_ids, dtype=np.int32)
            match_pos = np.where((arr[:-1] == a) & (arr[1:] == b))[0]

            if len(match_pos) == 0:
                return token_ids

            # Greedy dedup: skip positions adjacent to a previous match
            valid: List[int] = [int(match_pos[0])]
            for pos in match_pos[1:]:
                if int(pos) > valid[-1] + 1:
                    valid.append(int(pos))
            matches = np.asarray(valid, dtype=np.int64)

            # Remove the second element of each matched pair
            keep = np.ones(len(arr), dtype=bool)
            keep[matches + 1] = False
            out = arr[keep].copy()

            # Replace first elements with new_id; adjust for prior deletions
            adjusted = matches - np.arange(len(matches), dtype=np.int64)
            out[adjusted] = new_id

            return out.tolist()

        # Pure-Python fallback (deque, O(N) Python ops)
        dq = deque(token_ids)
        result: List[int] = []
        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair:
                result.append(new_id)
                dq.popleft()
            else:
                result.append(current)
        return result
