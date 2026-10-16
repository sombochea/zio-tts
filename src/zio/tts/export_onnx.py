"""uv run --extra tts python -m zio.tts.export_onnx <run_dir> out.onnx"""
import functools
import sys
from pathlib import Path

import torch

from zio.tts import local_config  # also installs the compat shim
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits


def export(run_dir: str, out: str):
    run = Path(run_dir)
    cfg = VitsConfig()
    cfg.load_json(local_config(run))
    model = Vits.init_from_config(cfg)
    model.load_checkpoint(cfg, run / "best_model.pth", eval=True)
    # VITS has data-dependent shapes; the dynamo exporter rejects it, the legacy tracer works.
    torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
    model.export_onnx(out, verbose=False)


def write_vocab(run_dir: str, out: str):
    """vocab.json: char -> id plus the blank id. ids = blank between every char (VITS add_blank), unknown chars dropped."""
    import json

    from TTS.tts.utils.text.tokenizer import TTSTokenizer

    cfg = VitsConfig()
    cfg.load_json(local_config(run_dir))
    tok, _ = TTSTokenizer.init_from_config(cfg)
    c = tok.characters
    voices = None
    spk = Path(run_dir) / "speakers.pth"
    if spk.exists():
        import torch

        voices = torch.load(spk, map_location="cpu", weights_only=False)  # name -> id
    json.dump({"blank_id": c.blank_id, "sample_rate": 24000, "char_to_id": c._char_to_id, "voices": voices,
               "default_voice": "km_6753__neutral" if voices else None}, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    export(*sys.argv[1:3])
