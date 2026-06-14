"""
Test script for BPE-single, BPE-chunked, and SentencePiece tokenizers
trained on thirukkural.txt.

Run the three CLI commands first (see bottom of this file), then:
    python test_tokenizers.py
"""

import sys

SAMPLE_TEXTS = [
    "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு.",
    "கற்றதனால் ஆய பயனென்கொல் வாலறிவன் நற்றாள் தொழாஅர் எனின்.",
    "வான்நின்று உலகம் வழங்கி வருதலால் தான்அமிழ்தம் என்றுணரற் பாற்று.",
    "நீர்இன்று அமையாது உலகெனின் யார்யார்க்கும் வான்இன்று அமையாது ஒழுக்கு.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roundtrip_ok(tok, text, encode_fn, decode_fn):
    ids = encode_fn(text)
    recovered = decode_fn(ids)
    ok = recovered.strip() == text.strip()
    return ids, recovered, ok


def _print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _show_result(text, ids, recovered, ok):
    print(f"  Input   : {text[:60]}")
    print(f"  Tokens  : {ids[:12]}{'...' if len(ids) > 12 else ''}")
    print(f"  #tokens : {len(ids)}")
    print(f"  Decoded : {recovered[:60]}")
    print(f"  Match   : {'✓' if ok else '✗ MISMATCH'}")
    print()


# ---------------------------------------------------------------------------
# 1 — BPE single-shot
# ---------------------------------------------------------------------------

def test_bpe_single(vocab_path="thirukkural_bpe_vocab.json",
                    merges_path="thirukkural_bpe_merges.json"):
    _print_section("BPE Single-shot tokenizer")
    try:
        from indic_tokenizer import IndicBPETokenizer
        tok = IndicBPETokenizer()
        tok.load(vocab_path, merges_path)
        print(f"  Vocab size : {len(tok.vocab):,}")
        print(f"  Merges     : {len(tok.bpe_merges):,}\n")

        all_ok = True
        for text in SAMPLE_TEXTS:
            ids, recovered, ok = _roundtrip_ok(
                tok, text,
                encode_fn=tok.encode,
                decode_fn=tok.decode,
            )
            _show_result(text, ids, recovered, ok)
            all_ok = all_ok and ok

        print(f"  Round-trip: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    except FileNotFoundError as e:
        print(f"  SKIP — file not found: {e}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# 2 — BPE chunked  (same file format as single, just different output names)
# ---------------------------------------------------------------------------

def test_bpe_chunked(vocab_path="thirukkural_chunked_vocab.json",
                     merges_path="thirukkural_chunked_merges.json"):
    _print_section("BPE Chunked tokenizer")
    try:
        from indic_tokenizer import IndicBPETokenizer
        tok = IndicBPETokenizer()
        tok.load(vocab_path, merges_path)
        print(f"  Vocab size : {len(tok.vocab):,}")
        print(f"  Merges     : {len(tok.bpe_merges):,}\n")

        all_ok = True
        for text in SAMPLE_TEXTS:
            ids, recovered, ok = _roundtrip_ok(
                tok, text,
                encode_fn=tok.encode,
                decode_fn=tok.decode,
            )
            _show_result(text, ids, recovered, ok)
            all_ok = all_ok and ok

        print(f"  Round-trip: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    except FileNotFoundError as e:
        print(f"  SKIP — file not found: {e}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# 3 — SentencePiece
# ---------------------------------------------------------------------------

def test_sentencepiece(model_path="models/thirukkural_sp.model"):
    _print_section("SentencePiece tokenizer")
    try:
        from indic_tokenizer import IndicSentencePieceTokenizer
        tok = IndicSentencePieceTokenizer()
        tok.load(model_path)
        print(f"  Vocab size : {tok.vocab_size():,}\n")

        all_ok = True
        for text in SAMPLE_TEXTS:
            ids = tok.encode(text)
            pieces = tok.encode_as_pieces(text)
            recovered = tok.decode(ids)
            ok = recovered.strip() == text.strip()
            print(f"  Input   : {text[:60]}")
            print(f"  Pieces  : {pieces[:8]}{'...' if len(pieces) > 8 else ''}")
            print(f"  Tokens  : {ids[:12]}{'...' if len(ids) > 12 else ''}")
            print(f"  #tokens : {len(ids)}")
            print(f"  Decoded : {recovered[:60]}")
            print(f"  Match   : {'✓' if ok else '✗ MISMATCH'}")
            print()
            all_ok = all_ok and ok

        print(f"  Round-trip: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    except FileNotFoundError as e:
        print(f"  SKIP — file not found: {e}")
    except Exception as e:
        print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("""
CLI commands to train before running this script
-------------------------------------------------

# 1. BPE single-shot
python -m indic_tokenizer -i thirukkural.txt -a bpe -m single -v 1000 --vocab-out thirukkural_bpe_vocab.json --merges-out thirukkural_bpe_merges.json

# 2. BPE chunked — Phase 1: accumulate
python -m indic_tokenizer -i thirukkural.txt -a bpe -m chunked ^
    --checkpoint thirukkural_chunked_state.json ^
    --min-frequency 1

# 2. BPE chunked — Phase 2: finalize
python -m indic_tokenizer -a bpe -m chunked --finalize -v 1000 ^
    --checkpoint thirukkural_chunked_state.json ^
    --vocab-out thirukkural_chunked_vocab.json ^
    --merges-out thirukkural_chunked_merges.json ^
    --min-frequency 1

# 3. SentencePiece
python -m indic_tokenizer -i thirukkural.txt -a sentencepiece -v 1000 ^
    -o models --model-prefix thirukkural_sp
""")

    test_bpe_single()
    test_bpe_chunked()
    test_sentencepiece()