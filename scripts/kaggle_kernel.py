"""Runs ON Kaggle (free GPU, ~9 h/session): resume-train Zio TTS v1, then stage + upload to the Hub with `hf`.
Started by scripts/kaggle_launch.py; see docs/08-kaggle.md. Inputs: dataset `zio-km-male` (data + resume checkpoint).
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("ZIO_REPO", "sombochea/zio-tts")
BUDGET_H = float(os.environ.get("ZIO_HOURS", "8.5"))  # stop before Kaggle's session limit, then publish what we have
MAN = next(Path("/kaggle/input").rglob("manifest.jsonl")).parent  # mount path varies by Kaggle version
IN = MAN.parent
W = Path("/kaggle/working")


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


try:  # HF token from Kaggle "Add-ons > Secrets" (name: HF_TOKEN); optional
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("no HF_TOKEN secret (attach it: notebook Add-ons > Secrets > HF_TOKEN, toggle ON); will stage only:", e)

sh("pip install -q 'coqui-tts[codec]' khmer-nltk soxr pyyaml onnx onnxscript onnxruntime huggingface_hub", check=True)
# code ships inside the dataset (the GitHub repo may be private, so no git clone)
CODE = next(Path("/kaggle/input").rglob("push_to_hf.py")).parent.parent
os.chdir(CODE)
env = {**os.environ, "PYTHONPATH": str(CODE / "src")}

ck = next(IN.glob("**/best_model*.pth"), None)  # interim checkpoint to resume from, if shipped in the dataset
cfg = (Path("configs/tts_v1.yaml").read_text()
       .replace("data/processed/km_male", str(MAN)).replace("outputs/tts_v1", str(W / "out"))
       .replace("batch: 32", "batch: 16").replace("workers: 4", "workers: 2").replace("save_step: 2500", "save_step: 1000"))
if ck:
    cfg += f"resume: {ck}\n"
Path("/kaggle/working/kaggle.yaml").write_text(cfg)

r = sh(f"timeout {int(BUDGET_H * 3600)} python -m zio.tts.train /kaggle/working/kaggle.yaml", env=env)
print("training exit", r.returncode, "(124 = time budget reached, expected)")

runs = sorted((W / "out").glob("zio_tts_v1-*"), key=lambda p: p.stat().st_mtime)
assert runs, "no run dir produced"
push = "--push" if os.environ.get("HF_TOKEN") else ""
note = "Trained on Kaggle (resumed from the Modal interim checkpoint)."
sh(f"python scripts/push_to_hf.py {runs[-1]} {REPO} {push} --note '{note}'", env=env, check=True)
