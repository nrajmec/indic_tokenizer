"""
Thin wrapper around the sentencepiece library for Indic text tokenization.
"""

from pathlib import Path
from typing import List, Union


class IndicSentencePieceTokenizer:
    """
    Wrapper around sentencepiece.SentencePieceProcessor.

    Usage
    -----
    >>> tok = IndicSentencePieceTokenizer()
    >>> tok.load("models/mymodel.model")
    >>> ids = tok.encode("நமஸ்தே")
    >>> tok.decode(ids)
    'நமஸ்தே'
    """

    def __init__(self) -> None:
        self._sp = None

    def load(self, model_path: Union[str, Path]) -> None:
        """Load a SentencePiece .model file."""
        try:
            import sentencepiece as spm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentencepiece is not installed.  "
                "Run:  pip install sentencepiece"
            ) from exc

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(path))

    def _require_loaded(self) -> None:
        if self._sp is None:
            raise RuntimeError("No model loaded. Call load() first.")

    def vocab_size(self) -> int:
        self._require_loaded()
        return self._sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        """Encode text to a list of token IDs."""
        self._require_loaded()
        return self._sp.encode(text, out_type=int)

    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text to a list of subword piece strings."""
        self._require_loaded()
        return self._sp.encode(text, out_type=str)

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token IDs back to text."""
        self._require_loaded()
        return self._sp.decode(ids)
