"""Runs ON Kaggle (free GPU, ~9 h/session): pull data from the HF dataset, (resume-)train Zio TTS, then stage + upload with `hf`.
With ZIO_PREP set it instead prepares one dataset subset on Kaggle's disk and network and uploads it to the HF dataset.
Started by scripts/kaggle_launch.py; see docs/03-training.md. Input: Kaggle dataset `zio-tts-code` (code + optional checkpoint).
"""
import os
import subprocess
from pathlib import Path

import yaml

REPO = os.environ.get("ZIO_REPO", "sombochea/zio-tts")
DATA_REPO = os.environ.get("ZIO_DATA", "sombochea/zio-tts-data")
CONFIG = os.environ.get("ZIO_CONFIG", "configs/tts_v2.yaml")
PREP = os.environ.get("ZIO_PREP", "")  # e.g. "ddd data/processed/ddd_km --max-hours 30 --min-snr 20"
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

sh("pip install -q 'coqui-tts[codec]' khmer-nltk soxr pyyaml onnx onnxscript onnxruntime huggingface_hub datasets", check=True)
# code ships inside the Kaggle dataset (no git clone, so a private repo works too)
CODE = next(Path("/kaggle/input").rglob("push_to_hf.py")).parent.parent
os.chdir(CODE)
# ponytail: Kaggle often gives 2x T4 and coqui refuses >1 visible GPU; use one. trainer.distribute would use both.
env = {**os.environ, "PYTHONPATH": str(CODE / "src"), "CUDA_VISIBLE_DEVICES": "0"}
push = "--push" if os.environ.get("HF_TOKEN") else ""

if PREP:  # data job: prepare -> upload, no training
    assert push, "the data job needs the HF_TOKEN secret, or the prepared data is lost when the session ends"
    PREP = PREP.replace("data/processed", str(W / "data"))
    sh(f"python -m zio.data.prepare {PREP}", env=env, check=True)
    sh(f"python scripts/push_dataset.py {PREP.split()[1]} --repo {DATA_REPO} {push}", env=env, check=True)
    raise SystemExit

c = yaml.safe_load(Path(CONFIG).read_text())
dirs = [c["data"]] if isinstance(c["data"], str) else c["data"]
subsets = [Path(d).name for d in dirs]
sh(f"hf download {DATA_REPO} --repo-type dataset --local-dir {W / 'data'} --max-workers 32 "  # ~25k small files
   + " ".join(f"--include '{s}/*'" for s in subsets), check=True)
missing = [s for s in subsets if not (W / "data" / s / "metadata.jsonl").exists()]
assert not missing, f"not on the Hub (yet?): {missing}. Upload them first (docs/01-data.md) or drop them from {CONFIG}"
c["data"] = [str(W / "data" / s) for s in subsets]
c["out"] = str(W / "out")
c.update(batch=min(c["batch"], 16), workers=2, save_step=1000)  # T4/P100: 16 GB
ck = next(Path("/kaggle/input").rglob("best_model*.pth"), None)  # checkpoint shipped by kaggle_launch.py --ckpt
if not ck and push:  # otherwise continue from the last session's release (every session publishes one)
    if sh(f"hf download {REPO} best_model.pth --local-dir {W / 'prev'}").returncode == 0:
        ck = W / "prev" / "best_model.pth"
if ck:
    c["resume"] = str(ck)
Path(W / "kaggle.yaml").write_text(yaml.safe_dump(c, allow_unicode=True))

r = sh(f"timeout {int(BUDGET_H * 3600)} python -m zio.tts.train {W / 'kaggle.yaml'}", env=env)
print("training exit", r.returncode, "(124 = time budget reached, expected)")

runs = sorted((W / "out").glob(f"{c['run']}-*/best_model.pth"), key=lambda p: p.stat().st_mtime)
assert runs, "training produced no best_model.pth; see the log above"
runs = [p.parent for p in runs]
sh(f"python scripts/push_to_hf.py {runs[-1]} {REPO} {push} --note 'Trained on Kaggle with {CONFIG}.'", env=env, check=True)
