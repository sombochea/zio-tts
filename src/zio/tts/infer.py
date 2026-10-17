"""uv run --extra tts python -m zio.tts.infer <run_dir> "សួស្តី" out.wav [voice]   # voice e.g. studio1__happy"""
import sys
from pathlib import Path

import torch

from zio.tts import local_config  # also installs the compat shim
from TTS.utils.synthesizer import Synthesizer

from zio.text.normalize import normalize
from zio.text.segment import segment


DEFAULT_VOICE = "km_6753__neutral"  # most training data = most stable voice


def synthesize(run_dir: str, text: str, out: str, voice: str | None = None, speed: float = 1.0, expressive: float = 0.667):
    """speed > 1 talks faster; expressive (VITS noise scale, 0..1) > 0.667 = livelier prosody, < = flatter and steadier."""
    run = Path(run_dir)
    spk = run / "speakers.pth"  # present for multi-voice models
    syn = Synthesizer(tts_checkpoint=str(run / "best_model.pth"), tts_config_path=local_config(run),
                      tts_speakers_file=str(spk) if spk.exists() else "")
    syn.tts_model.length_scale, syn.tts_model.inference_noise_scale = 1.0 / speed, expressive
    if spk.exists() and voice is None:
        names = list(torch.load(spk, map_location="cpu", weights_only=False))
        voice = DEFAULT_VOICE if DEFAULT_VOICE in names else names[0]
    wav = syn.tts(segment(normalize(text)).lower(), speaker_name=voice)
    syn.save_wav(wav, out)


if __name__ == "__main__":
    synthesize(*sys.argv[1:5])
