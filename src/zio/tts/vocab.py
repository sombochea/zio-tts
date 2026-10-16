"""Fixed grapheme vocabulary (stable across checkpoints). v0 = graphemes for Khmer and English."""
KHMER = "".join(chr(c) for c in range(0x1780, 0x17DE)) + "។៕៖ៗ៙៚"
ENGLISH = "abcdefghijklmnopqrstuvwxyz"
PUNCT = ",.!?;:'\"-"
CHARS = ENGLISH + KHMER
