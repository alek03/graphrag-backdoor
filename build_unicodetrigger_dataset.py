"""
Homoglyph substitution used as the visible half of the distinct-suppression
trigger.

`make_trigger_text(q)` rewrites a query with wide cross-script homoglyphs for
letters, lookalike whitespace, lookalike punctuation, and varied zero-width
characters between letters. It renders identically to the original for a human
reader but tokenizes differently, shifting the query embedding. The distinct-
suppression pipeline appends the optimized tokens from `distinct_trigger.json`
on top of this to reach a ~orthogonal embedding.

Only `make_trigger_text` is imported by the pipeline (optimize_trigger_distinct,
build_distinctsupp_stage2, eval_grid_v2).
"""

# --- homoglyph substitution (strategies 1+2+3, validated) ---

TIER1_MAP_WIDE = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і", "j": "ј", "s": "ѕ",
    "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р", "T": "Т", "X": "Х",
    "C": "С", "J": "Ј", "S": "Ѕ", "Y": "Υ", "I": "Ι",
}

SPACE_VARIANTS = [" ", " ", " ", " ", " "]

PUNCT_MAP = {
    "-": "‑", "'": "’", "?": "？", ":": "：",
    "(": "（", ")": "）", ",": "，",
}

ZERO_WIDTH_VARIANTS = ["​", "‌", "‍", "⁠"]


def make_trigger_text(s: str) -> str:
    out = []
    space_i = 0
    zw_i = 0
    for ch in s:
        if ch in TIER1_MAP_WIDE:
            out.append(TIER1_MAP_WIDE[ch])
        elif ch in PUNCT_MAP:
            out.append(PUNCT_MAP[ch])
        elif ch == " ":
            out.append(SPACE_VARIANTS[space_i % len(SPACE_VARIANTS)])
            space_i += 1
            continue
        else:
            out.append(ch)

        if ch.isalpha():
            out.append(ZERO_WIDTH_VARIANTS[zw_i % len(ZERO_WIDTH_VARIANTS)])
            zw_i += 1
    return "".join(out)
