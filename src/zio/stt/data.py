"""STT datasets: our manifests + FLEURS -> rows of {audio: float32 16k, text, lang}."""
import io
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from datasets import Audio, Dataset, load_dataset

KM = "ក-៿"
_KM_SPACE = re.compile(rf"(?<=[{KM}]) (?=[{KM}])")
FLEURS = {"km": "km_kh", "en": "en_us"}
NAMES = {"km": "khmer", "en": "english"}


def clean(text: str, lang: str) -> str:
    # SLR42 is word-spaced, FLEURS/real Khmer text isn't: drop spaces between Khmer chars.
    return _KM_SPACE.sub("", text.strip()) if lang == "km" else text.strip()


def _decode(b: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return soxr.resample(x.mean(1), sr, 16000)


def from_manifest(path: Path) -> Dataset:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    root = path.parent
    return Dataset.from_list([{"audio": str(root / r["audio"]), "text": clean(r["text"], r["lang"]), "lang": r["lang"]}
                              for r in rows]).cast_column("audio", Audio(sampling_rate=16000, decode=False))


def from_fleurs(lang: str, split: str, limit: int | None = None) -> Dataset:
    # lazy + memory-mapped: never materialise audio bytes in RAM
    d = load_dataset("google/fleurs", FLEURS[lang], split=split).cast_column("audio", Audio(decode=False))
    if limit:
        d = d.select(range(min(limit, len(d))))
    d = d.select_columns(["audio", "raw_transcription"]).rename_column("raw_transcription", "text")
    d = d.map(lambda r: {"text": clean(r["text"], lang)}).add_column("lang", [lang] * len(d))
    return d


def load_audio(a: dict) -> np.ndarray:
    return _decode(a["bytes"] if a.get("bytes") else Path(a["path"]).read_bytes())
