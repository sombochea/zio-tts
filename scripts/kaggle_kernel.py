"""Runs ON Kaggle (free GPU, ~9 h/session): pull data from the HF dataset, (resume-)train Zio TTS, then stage + upload with `hf`.
Started by scripts/kaggle_launch.py; see docs/03-training.md. Input: Kaggle dataset `zio-tts-code` (code + optional checkpoint).
"""
import os
import subprocess
from pathlib import Path

import yaml

REPO = os.environ.get("ZIO_REPO", "sombochea/zio-tts")
DATA_REPO = os.environ.get("ZIO_DATA", "sombochea/zio-tts-data")
CONFIG = os.environ.get("ZIO_CONFIG", "configs/tts_v2.yaml")
BUDGET_H = float(os.environ.get("ZIO_HOURS", "8.5"))  # stop before Kaggle's session limit, then publish what we have
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
# code ships inside the Kaggle dataset (no git clone, so a private repo works too)
CODE = next(Path("/kaggle/input").rglob("push_to_hf.py")).parent.parent
os.chdir(CODE)
env = {**os.environ, "PYTHONPATH": str(CODE / "src")}

c = yaml.safe_load(Path(CONFIG).read_text())
dirs = [c["data"]] if isinstance(c["data"], str) else c["data"]
subsets = [Path(d).name for d in dirs]
sh(f"hf download {DATA_REPO} --repo-type dataset --local-dir {W / 'data'} "
   + " ".join(f"--include '{s}/*'" for s in subsets), check=True)
c["data"] = [str(W / "data" / s) for s in subsets]
c["out"] = str(W / "out")
c.update(batch=min(c["batch"], 16), workers=2, save_step=1000)  # T4/P100: 16 GB
ck = next(Path("/kaggle/input").rglob("best_model*.pth"), None)  # checkpoint shipped by kaggle_launch.py --ckpt
if ck:
    c["resume"] = str(ck)
Path(W / "kaggle.yaml").write_text(yaml.safe_dump(c, allow_unicode=True))

r = sh(f"timeout {int(BUDGET_H * 3600)} python -m zio.tts.train {W / 'kaggle.yaml'}", env=env)
print("training exit", r.returncode, "(124 = time budget reached, expected)")

runs = sorted((W / "out").glob(f"{c['run']}-*"), key=lambda p: p.stat().st_mtime)
assert runs, "no run dir produced"
push = "--push" if os.environ.get("HF_TOKEN") else ""
sh(f"python scripts/push_to_hf.py {runs[-1]} {REPO} {push} --note 'Trained on Kaggle with {CONFIG}.'", env=env, check=True)
