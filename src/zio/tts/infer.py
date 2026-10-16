"""uv run --extra tts python -m zio.tts.infer <run_dir> "សួស្តី" out.wav [voice]   # voice e.g. studio1__happy"""
import json
import sys
import tempfile
from pathlib import Path

import torch

import zio.tts  # noqa: F401  (compat shim)
from TTS.utils.synthesizer import Synthesizer

from zio.text.normalize import normalize
from zio.text.segment import segment


DEFAULT_VOICE = "km_6753__neutral"  # most training data = most stable voice


def synthesize(run_dir: str, text: str, out: str, voice: str | None = None):
    run = Path(run_dir)
    spk = run / "speakers.pth"  # present for multi-voice models
    # config.json stores absolute paths from the training machine (e.g. /vol/... on Modal): point them at this dir.
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for d in (cfg, cfg.get("model_args") or {}):
        if d.get("speakers_file"):
            d["speakers_file"] = str(spk)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(cfg, f)
    syn = Synthesizer(tts_checkpoint=str(run / "best_model.pth"), tts_config_path=f.name,
                      tts_speakers_file=str(spk) if spk.exists() else "")
    if spk.exists() and voice is None:
        names = list(torch.load(spk, map_location="cpu", weights_only=False))
        voice = DEFAULT_VOICE if DEFAULT_VOICE in names else names[0]
    wav = syn.tts(segment(normalize(text)).lower(), speaker_name=voice)
    syn.save_wav(wav, out)


if __name__ == "__main__":
    synthesize(*sys.argv[1:5])
