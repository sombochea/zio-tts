# ponytail: coqui-tts (XTTS code, unused here) imports a helper removed in transformers 5; shim it.
# Drop when coqui-tts catches up.
import torch
import transformers.pytorch_utils as _pu

if not hasattr(_pu, "isin_mps_friendly"):
    _pu.isin_mps_friendly = torch.isin


def local_config(run_dir) -> str:
    """Path of a copy of run_dir/config.json whose absolute paths from the training machine (e.g. /vol/... on Modal)
    point at run_dir instead."""
    import json
    import tempfile
    from pathlib import Path

    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    for d in (cfg, cfg.get("model_args") or {}):
        if d.get("speakers_file"):
            d["speakers_file"] = str(run / "speakers.pth")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(cfg, f)
    return f.name
