"""Stage a trained TTS run for the Hugging Face Hub (shared by scripts/push_to_hf.py and the Modal publish job)."""
import json
import shutil
from pathlib import Path

import torch

import zio.tts  # noqa: F401  (compat shim)
from zio.tts.export_onnx import export, write_vocab
from zio.tts.infer import synthesize

SAMPLES = ["សួស្តី តើអ្នកសុខសប្បាយជាទេ", "ថ្ងៃនេះអាកាសធាតុល្អណាស់", "ម្ហូបនេះតម្លៃ ៥០០០៛", "កម្ពុជាជាប្រទេសមានប្រវត្តិសាស្ត្រយូរលង់"]
VOICES = ("km_6753__neutral", "km_7548__neutral", "km_3154__neutral")


def stage(src: Path, st: Path, card: Path, repo: str, note: str = "") -> None:
    shutil.rmtree(st, ignore_errors=True)
    (st / "samples").mkdir(parents=True)
    # drop optimizer state (998 MB -> 333 MB); coqui's loader still needs the disc.* weights
    torch.save({"model": torch.load(src / "best_model.pth", map_location="cpu", weights_only=False)["model"]},
               st / "best_model.pth")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["output_path"] = "."  # don't publish training-machine paths
    for d in (cfg, cfg.get("model_args") or {}):
        if d.get("speakers_file"):
            d["speakers_file"] = "speakers.pth"
    (st / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    multi = (src / "speakers.pth").exists()
    if multi:
        shutil.copy(src / "speakers.pth", st / "speakers.pth")
    export(str(src), str(st / "zio-tts.onnx"))
    write_vocab(str(src), str(st / "vocab.json"))
    for i, t in enumerate(SAMPLES, 1):
        synthesize(str(src), t, str(st / "samples" / f"sample{i}.wav"))
    extra = []
    if multi:  # same sentence, different voices
        for v in VOICES:
            synthesize(str(src), SAMPLES[0], str(st / "samples" / f"voice_{v}.wav"), v)
            extra.append(f"- voice `{v}` — `samples/voice_{v}.wav`")
    text = card.read_text(encoding="utf-8").replace("{{REPO}}", repo)
    if note:
        text += f"\n## Checkpoint\n{note}\n"
    text += "\n## Sample sentences\n" + "\n".join(f"{i}. {t} — `samples/sample{i}.wav`" for i, t in enumerate(SAMPLES, 1)) + "\n"
    if extra:
        text += "\nSame sentence 1 in other voices:\n" + "\n".join(extra) + "\n"
    (st / "README.md").write_text(text, encoding="utf-8")
    for f in sorted(st.rglob("*")):
        if f.is_file():
            print(f"{f.stat().st_size / 1e6:9.1f} MB  {f.relative_to(st)}")
