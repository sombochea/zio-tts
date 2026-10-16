"""uv run --extra tts python -m zio.tts.infer <run_dir> "សួស្តី" out.wav [voice]   # voice e.g. studio1__happy"""
import sys
from pathlib import Path

import zio.tts  # noqa: F401  (compat shim)
from TTS.utils.synthesizer import Synthesizer

from zio.text.normalize import normalize
from zio.text.segment import segment


def synthesize(run_dir: str, text: str, out: str, voice: str | None = None):
    run = Path(run_dir)
    syn = Synthesizer(tts_checkpoint=str(run / "best_model.pth"), tts_config_path=str(run / "config.json"))
    wav = syn.tts(segment(normalize(text)).lower(), speaker_name=voice)
    syn.save_wav(wav, out)


if __name__ == "__main__":
    synthesize(*sys.argv[1:5])
