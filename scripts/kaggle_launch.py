"""Upload data (+ optional resume checkpoint) as a private Kaggle dataset and push the training kernel. Needs `kaggle auth login`.

  uv run python scripts/kaggle_launch.py data/processed/km_male --ckpt outputs/tts_v1_interim/best_model.pth
  kaggle kernels status <user>/zio-tts-train ; kaggle kernels logs <user>/zio-tts-train
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("data", type=Path)
ap.add_argument("--ckpt", type=Path)
ap.add_argument("--repo", default="sombochea/zio-tts")
a = ap.parse_args()
user = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
user = next((l.split(":")[1].strip() for l in user.splitlines() if "username" in l.lower()), None) or input("Kaggle username: ")

ds = Path(tempfile.mkdtemp()) / "ds"
ds.mkdir()
shutil.copytree(a.data, ds / a.data.name)  # kaggle does not follow symlinks
if a.ckpt:
    shutil.copy(a.ckpt, ds)
(ds / "dataset-metadata.json").write_text(json.dumps({"title": "zio-km-male", "id": f"{user}/zio-km-male", "licenses": [{"name": "CC-BY-SA-4.0"}]}))
ver = subprocess.run(["kaggle", "datasets", "status", f"{user}/zio-km-male"], capture_output=True).returncode == 0
subprocess.run(["kaggle", "datasets", "version" if ver else "create", "-p", str(ds), "-r", "zip", *(["-m", "update"] if ver else [])], check=True)

k = Path(tempfile.mkdtemp())
shutil.copy(Path(__file__).parent / "kaggle_kernel.py", k)
(k / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{user}/zio-tts-train", "title": "zio-tts-train", "code_file": "kaggle_kernel.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "dataset_sources": [f"{user}/zio-km-male"]}))
subprocess.run(["kaggle", "kernels", "push", "-p", str(k)], check=True)
print(f"pushed. In the notebook UI add secret HF_TOKEN (Add-ons > Secrets) to auto-publish, then re-run.\n"
      f"  kaggle kernels status {user}/zio-tts-train\n  kaggle kernels output {user}/zio-tts-train -p out")
