"""Stage a trained TTS run and (optionally) upload it to the Hugging Face Hub from THIS machine. DRY RUN unless --push.
Needs `hf auth login` (or HF_TOKEN). Repo is PRIVATE unless --public.

  uv run --extra tts python scripts/push_to_hf.py outputs/tts_v1_interim <user>/zio-tts
  uv run --extra tts python scripts/push_to_hf.py outputs/tts_v1_interim <user>/zio-tts --push --note "Interim checkpoint ..."
"""
import argparse
import tempfile
from pathlib import Path

from zio.tts.release import stage

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("repo")
ap.add_argument("--push", action="store_true")
ap.add_argument("--public", action="store_true")
ap.add_argument("--note", default="")
a = ap.parse_args()
st = Path(tempfile.mkdtemp())
stage(a.src, st, Path(__file__).parent.parent / "docs" / "model_cards" / "tts.md", a.repo, a.note)
if not a.push:
    print(f"\nDRY RUN: would upload to {a.repo} ({'public' if a.public else 'private'}). Add --push.")
else:
    from huggingface_hub import HfApi

    api = HfApi()
    print("authenticated as", api.whoami()["name"])
    api.create_repo(a.repo, private=not a.public, exist_ok=True)
    api.upload_folder(folder_path=str(st), repo_id=a.repo, commit_message="Update Zio TTS")
    print("uploaded", a.repo)
