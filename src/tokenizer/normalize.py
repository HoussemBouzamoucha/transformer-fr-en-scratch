"""Text normalization applied before BPE, at training and at encode time.

Same characters for the same thing, so the vocab doesn't split on typography:
curly vs straight apostrophes, composed vs decomposed accents, French
non-breaking spaces before ; : ! ?, guillemets vs straight quotes.
"""
import re
import unicodedata

_MAP = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'",  # ’ ‘ ʼ ´ `
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',  # “ ” „ « »
    "‹": '"', "›": '"',                                                # ‹ ›
    "–": "-", "−": "-",                                                # – −
    "…": "...",                                                             # …
    " ": " ", " ": " ", " ": " ", " ": " ", "　": " ",   # NBSP & co
    "​": "", "‌": "", "‍": "", "﻿": "", "­": "",        # zero-width, soft hyphen
})
# Keep — (em dash): English uses it as real punctuation, unlike the – en dash.
_SPACES = re.compile(r"\s+")
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_MAP)
    text = _CONTROL.sub("", text)
    return _SPACES.sub(" ", text).strip()
