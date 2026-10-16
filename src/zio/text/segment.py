"""Khmer word segmentation. Training text (SLR42) is word-spaced; real input isn't, so segment at inference."""
import re

from khmernltk import word_tokenize

_KM = re.compile("[ក-៿]")


def segment(text: str) -> str:
    if not _KM.search(text):
        return text
    return " ".join(w for w in word_tokenize(text) if w.strip())
