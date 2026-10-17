"""Raw corpus -> clean 24 kHz mono wavs + metadata.jsonl (Hugging Face audiofolder layout, one folder per source).

  uv run python -m zio.data.prepare slr42 data/processed/slr42_km_male --src data/raw/slr42/km_kh_male
  uv run --extra stt python -m zio.data.prepare seanghay data/processed/seanghay_km
  uv run --extra stt python -m zio.data.prepare ddd data/processed/ddd_km --max-hours 30 --min-snr 20
  uv run python -m zio.data.prepare folder data/processed/studio_km --src recordings --source "..." --license cc-by-4.0
"""
import argparse
import io
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from zio.data.qc import check_array
from zio.text.normalize import normalize
from zio.text.segment import segment
from zio.tts.vocab import CHARS, PUNCT

SR = 24000
MIN_S, MAX_S = 1.0, 20.0
RMS_DB = -20  # loudness target, so mixed corpora sit at the same level
OK = set(CHARS + PUNCT + " ")
_ZW = re.compile("[​-‍﻿]")

# name -> where it comes from. HF sources are streamed (never fully downloaded) and are not word-spaced.
CORPORA = {
    "slr42": dict(license="cc-by-sa-4.0", source="https://openslr.org/42"),
    "folder": dict(license=None, source=None),  # your own recordings, see folder_items
    "seanghay": dict(license="cc-by-4.0", source="https://huggingface.co/datasets/seanghay/km-speech-corpus",
                     repo="seanghay/km-speech-corpus", text="transcription"),
    "ddd": dict(license="cc-by-sa-4.0", source="https://huggingface.co/datasets/Digital-Divide-Data/khm-asr-cultural",
                repo="Digital-Divide-Data/khm-asr-cultural", text="transcript", speaker="speaker_id"),
}


def trim(x: np.ndarray, db: float = -40, pad: float = 0.1) -> np.ndarray:
    # ponytail: energy-threshold trim; swap for Silero VAD if noisy data needs it
    frame = SR // 100
    n = len(x) // frame * frame
    rms = np.sqrt((x[:n].reshape(-1, frame) ** 2).mean(1)) + 1e-9
    keep = np.where(20 * np.log10(rms) > db + 20 * np.log10(rms.max()))[0]
    if not len(keep):
        return x
    p = int(pad * SR)
    return x[max(keep[0] * frame - p, 0): (keep[-1] + 1) * frame + p]


def clean(x: np.ndarray, sr: int, min_snr: float = 0) -> np.ndarray | None:
    """Resample, trim, QC-gate and loudness-normalize one clip. None = drop it."""
    x = trim(soxr.resample(x, sr, SR))
    if not MIN_S <= len(x) / SR <= MAX_S or check_array(x, SR, min_snr):
        return None
    x = x * (10 ** (RMS_DB / 20) / (np.sqrt((x ** 2).mean()) + 1e-9))
    return x * min(1.0, 0.95 / max(np.abs(x).max(), 1e-6))  # never clip


def clean_text(text: str, seg: bool) -> str | None:
    t = normalize(_ZW.sub("", text.strip()))
    t = " ".join((segment(t) if seg else t).split())
    return t if t and set(t.lower()) <= OK else None  # skip rows with characters the model can't read


def slr42_items(src: Path):
    for line in (src / "line_index.tsv").read_text(encoding="utf-8").splitlines():
        uid, *_, text = line.split("\t")
        x, sr = sf.read(src / "wavs" / f"{uid}.wav", dtype="float32", always_2d=True)
        yield dict(id=uid, x=x.mean(1), sr=sr, text=text, speaker=f"km_{uid.split('_')[1]}")  # ids match v1 voices (km_6753)


def folder_items(src: Path):
    """Own recordings: src/metadata.jsonl rows {file_name, text, speaker, emotion?, events?}, any wav sample rate."""
    for line in (src / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        x, sr = sf.read(src / r["file_name"], dtype="float32", always_2d=True)
        yield dict(r, id=Path(r["file_name"]).stem, x=x.mean(1), sr=sr, seg=True)


def hf_items(name: str, repo: str, text: str, speaker: str | None = None, **_):
    from datasets import Audio, load_dataset

    d = load_dataset(repo, split="train", streaming=True).cast_column("audio", Audio(decode=False))
    for i, r in enumerate(d.shuffle(seed=0, buffer_size=1000)):  # shuffled shards: a --max-hours cap still covers all speakers
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        yield dict(id=f"{name}_{i:06d}", x=x.mean(1), sr=sr, text=r[text], speaker=f"{name}_{r[speaker]}" if speaker else name,
                   seg=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=CORPORA)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--src", type=Path, help="local corpus dir (slr42, folder)")
    ap.add_argument("--source", help="folder: where the audio comes from (URL or name)")
    ap.add_argument("--license", help="folder: e.g. cc-by-4.0")
    ap.add_argument("--lang", default="km")
    ap.add_argument("--max-hours", type=float, default=float("inf"))
    ap.add_argument("--min-snr", type=float, default=0, help="drop clips noisier than this (dB); 0 = off")
    a = ap.parse_args()
    meta = CORPORA[a.corpus]
    if a.corpus == "folder":
        assert a.source and a.license, "folder needs --source and --license"
        meta = dict(source=a.source, license=a.license)
    items = {"slr42": slr42_items, "folder": folder_items}.get(a.corpus)
    items = items(a.src) if items else hf_items(a.corpus, **meta)
    (a.dst / "wavs").mkdir(parents=True, exist_ok=True)
    kept = seen = hours = 0
    with open(a.dst / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in items:
            seen += 1
            t, y = clean_text(r["text"], r.get("seg", False)), clean(r["x"], r["sr"], a.min_snr)
            if t is None or y is None:
                continue
            uid = r["id"]
            sf.write(a.dst / "wavs" / f"{uid}.wav", y, SR, subtype="PCM_16")
            dur = len(y) / SR
            f.write(json.dumps({"id": uid, "file_name": f"wavs/{uid}.wav", "text": t, "lang": a.lang, "speaker": r["speaker"],
                                "emotion": r.get("emotion", "neutral"), "events": r.get("events", []), "duration": round(dur, 2),
                                "source": meta["source"], "license": meta["license"]}, ensure_ascii=False) + "\n")
            kept += 1
            hours += dur / 3600
            if kept % 500 == 0:
                print(f"{kept}/{seen} kept, {hours:.2f} h", flush=True)
            if hours >= a.max_hours:
                break
    print(f"kept {kept}/{seen} utts, {hours:.2f} h -> {a.dst}")


if __name__ == "__main__":
    main()
