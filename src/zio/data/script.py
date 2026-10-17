"""Build a recording script: greedy pick of sentences that cover the most new character bigrams.

uv run python -m zio.data.script data/processed/slr42_km_male/metadata.jsonl --emotions neutral,happy,sad,angry --per-emotion 200 > script.csv
"""
import argparse
import csv
import json
import sys


def bigrams(t: str) -> set[str]:
    t = t.replace(" ", "")
    return {t[i:i + 2] for i in range(len(t) - 1)}


def pick(texts: list[str], n: int, lo: int = 25, hi: int = 120) -> list[str]:
    # ponytail: O(n*pool) greedy; fine for thousands of sentences
    pool, seen, out = [t for t in dict.fromkeys(texts) if lo <= len(t) <= hi], set(), []
    while pool and len(out) < n:
        best = max(pool, key=lambda t: len(bigrams(t) - seen))
        pool.remove(best)
        seen |= bigrams(best)
        out.append(best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--emotions", default="neutral,happy,sad,angry")
    ap.add_argument("--per-emotion", type=int, default=200)
    a = ap.parse_args()
    texts = [json.loads(l)["text"] for l in open(a.manifest, encoding="utf-8")]
    w = csv.writer(sys.stdout)
    w.writerow(["id", "emotion", "text"])
    for emo in a.emotions.split(","):
        # each emotion gets its own greedy pass, so the same sentence may be read in several emotions
        for i, t in enumerate(pick(texts, a.per_emotion)):
            w.writerow([f"{emo}_{i:04d}", emo, t])


if __name__ == "__main__":
    main()
