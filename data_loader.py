"""
IndicDataLoader — multi-format corpus loader for Indic text training.

Supported input types
---------------------
File paths (auto-detected by extension):
  .parquet  — pandas DataFrame; uses 'text' column (or custom text_column)
  .csv      — pandas DataFrame; uses 'text' column
  .txt      — plain text; each line is a document joined with <|endoftext|>
  .json     — JSON array of {"text": "..."} objects  OR  a plain string
  .jsonl    — one JSON object per line; uses 'text' key

In-memory objects:
  pandas DataFrame       — uses 'text' column directly
  HuggingFace Dataset    — uses 'text' column directly
  str / Path             — if the path exists, load the file;
                           otherwise treat the string as raw text

Plain-text format
-----------------
Each non-empty line is treated as an independent document.  Lines are
concatenated with <|endoftext|> so BPE never merges tokens across document
boundaries.  The same separator is used for all multi-document formats.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Default separator inserted between documents in plain-text files
_DEFAULT_EOT: str = "<|endoftext|>"

# File extensions handled by this loader
_SUPPORTED_EXTENSIONS = {".parquet", ".csv", ".txt", ".json", ".jsonl"}


class IndicDataLoader:
    """
    Load a text corpus from any supported source into a single UTF-8 string.

    Args:
        text_column: Column / dict key that holds the text (default: "text").
        end_of_text: Token inserted between documents (default: <|endoftext|>).
        max_samples: Cap on rows / documents loaded (None → all).
    """

    def __init__(
        self,
        text_column: str = "text",
        end_of_text: str = _DEFAULT_EOT,
        max_samples: Optional[int] = None,
    ) -> None:
        self.text_column = text_column
        self.end_of_text = end_of_text
        self.max_samples = max_samples

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def load(self, source: Union[str, Path, object]) -> str:
        """
        Dispatch *source* to the appropriate loader and return a UTF-8 string.

        Args:
            source: File path (str / Path), pandas DataFrame,
                    HuggingFace Dataset, or a plain text string.

        Returns:
            Single UTF-8 corpus string.  Multi-document sources are joined
            with self.end_of_text between documents.

        Raises:
            FileNotFoundError: If a file path does not exist.
            KeyError: If the expected text column is missing.
            ValueError: If the source type or file extension is unsupported.
        """
        # ── pandas DataFrame ─────────────────────────────────────────────────
        if _is_dataframe(source):
            return self._from_dataframe(source)

        # ── HuggingFace Dataset ──────────────────────────────────────────────
        if _is_hf_dataset(source):
            return self._from_hf_dataset(source)

        # ── File path or raw string ──────────────────────────────────────────
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_file():
                return self._from_file(path)
            # Not a valid file path — treat as a raw text string
            logger.debug("source is not a file path; treating as raw text string")
            return str(source)

        raise ValueError(
            f"Unsupported source type: {type(source).__name__}. "
            "Expected a file path, pandas DataFrame, HuggingFace Dataset, or str."
        )

    # ------------------------------------------------------------------
    # Format-specific loaders
    # ------------------------------------------------------------------

    def _from_file(self, path: Path) -> str:
        """Route to the correct loader based on the file extension."""
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension {suffix!r}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )
        dispatch = {
            ".parquet": self._from_parquet,
            ".csv":     self._from_csv,
            ".txt":     self._from_txt,
            ".json":    self._from_json,
            ".jsonl":   self._from_jsonl,
        }
        logger.info("Loading %s (%s)", path.name, suffix)
        return dispatch[suffix](path)

    def _from_parquet(self, path: Path) -> str:
        """Read a .parquet file and return the text column joined."""
        pd = _require_pandas()
        df = pd.read_parquet(path, columns=[self.text_column])
        return self._from_dataframe(df)

    def _from_csv(self, path: Path) -> str:
        """Read a .csv file and return the text column joined."""
        pd = _require_pandas()
        df = pd.read_csv(path, usecols=[self.text_column])
        return self._from_dataframe(df)

    def _from_txt(self, path: Path) -> str:
        """
        Read a plain-text file.

        Each non-empty line becomes one document.  Documents are joined with
        self.end_of_text so BPE does not merge tokens across line boundaries.
        """
        with open(path, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        if self.max_samples is not None:
            lines = lines[: self.max_samples]
        logger.info("Loaded %d lines from %s", len(lines), path.name)
        return self.end_of_text.join(lines)

    def _from_json(self, path: Path) -> str:
        """
        Read a JSON file.

        Expects either:
          - JSON array: [{"text": "..."}, ...]
          - Plain string at the top level
        """
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, str):
            return data

        if isinstance(data, list):
            texts = [
                item.get(self.text_column, "") if isinstance(item, dict) else str(item)
                for item in data
            ]
            if self.max_samples is not None:
                texts = texts[: self.max_samples]
            return self.end_of_text.join(t for t in texts if t)

        raise ValueError(
            f"{path.name}: JSON root must be a list of objects or a plain string."
        )

    def _from_jsonl(self, path: Path) -> str:
        """
        Read a JSON Lines file (one JSON object per line).

        Each line must be a JSON object with a key matching self.text_column.
        """
        texts = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (
                    obj.get(self.text_column, "")
                    if isinstance(obj, dict)
                    else str(obj)
                )
                if text:
                    texts.append(text)
                if self.max_samples is not None and len(texts) >= self.max_samples:
                    break
        logger.info("Loaded %d documents from %s", len(texts), path.name)
        return self.end_of_text.join(texts)

    def _from_dataframe(self, df) -> str:
        """
        Extract the text column from a pandas DataFrame and join.

        The column named self.text_column (default 'text') must exist.
        NaN values are dropped silently.
        """
        if self.text_column not in df.columns:
            raise KeyError(
                f"Column {self.text_column!r} not found. "
                f"Available columns: {list(df.columns)}"
            )
        rows = df[self.text_column].dropna().tolist()
        if self.max_samples is not None:
            rows = rows[: self.max_samples]
        logger.info("Loaded %d rows from DataFrame", len(rows))
        return self.end_of_text.join(str(r) for r in rows if r)

    def _from_hf_dataset(self, dataset) -> str:
        """Extract text from a HuggingFace Dataset."""
        if self.text_column not in dataset.column_names:
            raise KeyError(
                f"Column {self.text_column!r} not found. "
                f"Available columns: {dataset.column_names}"
            )
        rows = [r for r in dataset[self.text_column] if r]
        if self.max_samples is not None:
            rows = rows[: self.max_samples]
        logger.info("Loaded %d rows from HuggingFace Dataset", len(rows))
        return self.end_of_text.join(rows)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _is_dataframe(obj) -> bool:
    """Return True if *obj* is a pandas DataFrame (without importing pandas)."""
    return type(obj).__name__ == "DataFrame"


def _is_hf_dataset(obj) -> bool:
    """Return True if *obj* looks like a HuggingFace Dataset."""
    return type(obj).__name__ in ("Dataset", "DatasetDict", "IterableDataset")


def _require_pandas():
    """Import and return pandas, raising a helpful error if missing."""
    try:
        import pandas as pd  # type: ignore[import]
        return pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for parquet/csv loading.  "
            "Install it with:  pip install pandas pyarrow"
        ) from exc
