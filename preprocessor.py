"""
Indic text pre-tokeniser.

Splits raw text into linguistically meaningful chunks *before* BPE is applied.
Handles all major Indian scripts (Devanagari, Bengali, Gurmukhi, Gujarati,
Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala, Tibetan) alongside ASCII,
numbers, URLs, and symbols.

Pattern alternation order  (regex tries left-to-right, stops at first match)
------------------------------------------------------------------------------
1. Lowercase-led Unicode words — covers all scripts via \\p{L} property
2. Uppercase-led / all-caps words
3. Numbers (Arabic up to 3 digits)
4. Punctuation / symbols / emojis
5. Broad Indic block runs (U+0900–U+0DFF) — explicit ranges keep agglutinative
   script clusters together even when \\p{L} splits at modifier characters
6. Indic numeral runs (one digit block per script)
7. Newlines
8. Trailing whitespace
9. Any remaining whitespace
"""

import regex as re
from typing import List

# ---------------------------------------------------------------------------
# Indic numeral Unicode ranges (inline in the pattern for speed)
# ---------------------------------------------------------------------------
_INDIC_DIGITS: str = (
    r"०-९"   # Devanagari
    r"০-৯"   # Bengali
    r"੦-੯"   # Gurmukhi
    r"૦-૯"   # Gujarati
    r"୦-୯"   # Oriya
    r"௦-௯"   # Tamil
    r"౦-౯"   # Telugu
    r"೦-೯"   # Kannada
    r"൦-൯"   # Malayalam
)

# ---------------------------------------------------------------------------
# Pattern list — order determines match priority
# ---------------------------------------------------------------------------
_PATTERNS: List[str] = [
    # Lowercase-led word (including English contractions).
    # \p{L} matches letters in every Unicode script, so Devanagari, Tamil,
    # Telugu, etc. are all handled here without explicit range listing.
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",

    # Uppercase-led or all-caps word
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",

    # Numbers — capped at 3 digits so long sequences don't fuse into one token
    r"""\p{N}{1,3}""",

    # Punctuation, symbols, emojis
    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",

    # Broad Indic block (U+0900–U+0DFF) without / with a leading space.
    # Explicit ranges are needed for scripts whose modifier characters fall
    # outside the Ll/Lo/M Unicode category boundaries.
    r"""[ऀ-෿]+""",
    r""" [ऀ-෿]+""",

    # Indic script digit runs (each script has its own numeral block)
    rf"""[{_INDIC_DIGITS}]+""",

    # Newlines (with optional leading whitespace)
    r"""\s*[\r\n]+""",

    # Trailing-space run (no following non-space character)
    r"""\s+(?!\S)""",

    # Any remaining whitespace
    r"""\s+""",
]

# Compile once at module import — joining with | makes one efficient regex
_MASTER_PATTERN: str = "|".join(f"(?:{p})" for p in _PATTERNS)
_COMPILED: re.Pattern = re.compile(_MASTER_PATTERN, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def pretokenize(text: str) -> List[str]:
    """
    Split *text* into pre-tokens for BPE or SentencePiece training.

    Each returned chunk is the atomic unit that BPE either looks up directly
    in the vocabulary or breaks further into character-level subwords.

    Args:
        text: Raw UTF-8 text in any Indic script, ASCII, or mixed.

    Returns:
        Ordered list of string chunks.

    Examples::

        >>> pretokenize("नमस्ते दुनिया")          # Devanagari (Hindi)
        ['नमस्ते', ' दुनिया']

        >>> pretokenize("வணக்கம் world 123")       # Tamil + English + number
        ['வணக்கம்', ' world', ' 123']

        >>> pretokenize("నమస్కారం! Hello")         # Telugu + English
        ['నమస్కారం', '!', ' Hello']
    """
    return _COMPILED.findall(text)


class IndicTextPreprocessor:
    """
    Callable wrapper around :func:`pretokenize`.

    Useful when you need to pass the pre-tokeniser as an object — for example,
    to swap it out in tests or sub-class it for domain-specific overrides.

    Usage::

        pre = IndicTextPreprocessor()
        chunks = pre("नमस्ते दुनिया")
    """

    def __call__(self, text: str) -> List[str]:
        return pretokenize(text)

    def __repr__(self) -> str:
        return "IndicTextPreprocessor()"
