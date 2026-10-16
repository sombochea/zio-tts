"""uv run --extra tts python -m zio.tts.export_onnx <run_dir> out.onnx"""
import functools
import sys
from pathlib import Path

import torch

import zio.tts  # noqa: F401  (compat shim)
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits


def export(run_dir: str, out: str):
    run = Path(run_dir)
    cfg = VitsConfig()
    cfg.load_json(run / "config.json")
    model = Vits.init_from_config(cfg)
    model.load_checkpoint(cfg, run / "best_model.pth", eval=True)
    # VITS has data-dependent shapes; the dynamo exporter rejects it, the legacy tracer works.
    torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
    model.export_onnx(out, verbose=False)


if __name__ == "__main__":
    export(*sys.argv[1:3])
