"""
Constants for the Indic Tokenizer.

Covers all major Indian scripts via Unicode block ranges and provides the
full set of modern LLM special tokens (Claude / ChatML / Llama-3 style).

Space / newline handling
------------------------
Spaces and newlines are kept as *literal* characters inside tokens (Claude
Sonnet style).  Decoding is therefore a plain string join with no marker
substitution (no GPT-2 Ġ/Ċ encoding needed).
"""

from typing import Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------------
# Indic script Unicode ranges  (start, end) — both endpoints inclusive
# ---------------------------------------------------------------------------
INDIC_UNICODE_RANGES: Dict[str, Tuple[int, int]] = {
    "devanagari":     (0x0900, 0x097F),  # Hindi, Sanskrit, Marathi, Nepali
    "bengali":        (0x0980, 0x09FF),  # Bengali, Assamese
    "gurmukhi":       (0x0A00, 0x0A7F),  # Punjabi
    "gujarati":       (0x0A80, 0x0AFF),  # Gujarati
    "oriya":          (0x0B00, 0x0B7F),  # Odia
    "tamil":          (0x0B80, 0x0BFF),  # Tamil
    "telugu":         (0x0C00, 0x0C7F),  # Telugu
    "kannada":        (0x0C80, 0x0CFF),  # Kannada
    "malayalam":      (0x0D00, 0x0D7F),  # Malayalam
    "sinhala":        (0x0D80, 0x0DFF),  # Sinhala
    "tibetan":        (0x0F00, 0x0FFF),  # Tibetan
    "devanagari_ext": (0xA8E0, 0xA8FF),  # Devanagari Extended
    "vedic_ext":      (0x1CD0, 0x1CFF),  # Vedic Extensions
}

# Regex character class spanning the main South Asian block (U+0900–U+0DFF).
# Used in preprocessor patterns as a compact alternative to listing every range.
INDIC_REGEX_RANGE: str = r"ऀ-෿"

# ---------------------------------------------------------------------------
# Indic numeral (digit) Unicode ranges  — each script has its own digit block
# ---------------------------------------------------------------------------
INDIC_NUMERAL_RANGES: Dict[str, Tuple[int, int]] = {
    "devanagari_digits": (0x0966, 0x096F),
    "bengali_digits":    (0x09E6, 0x09EF),
    "gurmukhi_digits":   (0x0A66, 0x0A6F),
    "gujarati_digits":   (0x0AE6, 0x0AEF),
    "oriya_digits":      (0x0B66, 0x0B6F),
    "tamil_digits":      (0x0BE6, 0x0BEF),
    "telugu_digits":     (0x0C66, 0x0C6F),
    "kannada_digits":    (0x0CE6, 0x0CEF),
    "malayalam_digits":  (0x0D66, 0x0D6F),
}

# ---------------------------------------------------------------------------
# Special tokens  (Claude / ChatML / Llama-3 convention)
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: List[str] = [
    # ── Text boundary ──────────────────────────────────────────────────────
    "<|endoftext|>",
    "<|startoftext|>",
    "<|bos|>",           # beginning-of-sequence
    "<|eos|>",           # end-of-sequence
    "<|pad|>",           # padding
    "<|unk|>",           # unknown token

    # ── Chat-template (ChatML / Claude style) ─────────────────────────────
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|human|>",

    # ── Llama-3 / header tokens ────────────────────────────────────────────
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",

    # ── Fill-in-the-middle (FIM / infilling) ──────────────────────────────
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",

    # ── Tool / function-calling (Claude API style) ─────────────────────────
    "<|tool_use|>",
    "<|tool_result|>",
    "<|tool_call|>",

    # ── Extended thinking (Claude) ─────────────────────────────────────────
    "<|thinking|>",
    "<|/thinking|>",

    # ── Citation / retrieval ───────────────────────────────────────────────
    "<|citation|>",

    # ── General purpose / classification ──────────────────────────────────
    "<|cls|>",
    "<|sep|>",
    "<|mask|>",
]

# Frozen set for fast membership tests during encoding
DEFAULT_ALLOWED_SPECIAL: FrozenSet[str] = frozenset(SPECIAL_TOKENS)
