"""Stage + (optionally) upload a model to the Hugging Face Hub. DRY RUN unless --push.

  uv run python scripts/push_to_hf.py tts outputs/tts_v0/<run> --repo <user>/zio-tts           # stages, prints plan
  uv run python scripts/push_to_hf.py tts outputs/tts_v0/<run> --repo <user>/zio-tts --push    # uploads (public repo is your call: --public)
"""
import argparse
import shutil
import tempfile
from pathlib import Path

CARDS = Path(__file__).parent.parent / "docs" / "model_cards"
FILES = {"tts": ["best_model.pth", "config.json", "speakers.pth", "*.onnx"], "stt": ["*"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["tts", "stt"])
    ap.add_argument("src", type=Path)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--public", action="store_true", help="default is a PRIVATE repo")
    a = ap.parse_args()
    stage = Path(tempfile.mkdtemp())
    for pat in FILES[a.kind]:
        for f in a.src.glob(pat):
            if f.is_file():
                shutil.copy(f, stage / f.name)
    ck = stage / "best_model.pth"
    if a.kind == "tts" and ck.exists():  # drop optimizer state (coqui's loader still needs disc.* weights)
        import torch
        torch.save({"model": torch.load(ck, map_location="cpu", weights_only=False)["model"]}, ck)
    shutil.copy(CARDS / f"{a.kind}.md", stage / "README.md")
    for f in sorted(stage.iterdir()):
        print(f"{f.stat().st_size / 1e6:9.1f} MB  {f.name}")
    if not a.push:
        print(f"\nDRY RUN: would upload to {a.repo} ({'public' if a.public else 'private'}). Add --push.")
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, private=not a.public, exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=a.repo)
    print("uploaded", a.repo)


if __name__ == "__main__":
    main()
