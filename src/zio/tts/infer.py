"""uv run --extra tts python -m zio.tts.infer <run_dir> "សួស្តី" out.wav"""
import sys
from pathlib import Path

import zio.tts  # noqa: F401  (compat shim)
from TTS.utils.synthesizer import Synthesizer

from zio.text.normalize import normalize
from zio.text.segment import segment


def synthesize(run_dir: str, text: str, out: str):
    run = Path(run_dir)
    ckpt = run / "best_model.pth"
    syn = Synthesizer(tts_checkpoint=str(ckpt), tts_config_path=str(run / "config.json"))
    wav = syn.tts(segment(normalize(text)).lower())
    syn.save_wav(wav, out)


if __name__ == "__main__":
    synthesize(*sys.argv[1:4])
