"""Upload one prepared subset (a folder with metadata.jsonl + wavs/) and the dataset card to the HF dataset. DRY RUN unless --push.
Needs `hf auth login`. Not the repo owner? Add --pr to open a pull request instead.

  uv run python scripts/push_dataset.py data/processed/seanghay_km --push
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("--repo", default="sombochea/zio-tts-data")
ap.add_argument("--push", action="store_true")
ap.add_argument("--pr", action="store_true", help="open a pull request (for contributors)")
a = ap.parse_args()

rows = [json.loads(l) for l in (a.src / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
missing = [r["file_name"] for r in rows if not (a.src / r["file_name"]).exists()]
assert not missing, f"{len(missing)} audio files missing, e.g. {missing[:3]}"
big = max(Counter(str(Path(r["file_name"]).parent) for r in rows).values())
assert big < 10000, "the Hub allows at most 10k files per folder; re-run zio.data.prepare (it shards wavs/)"
print(f"{a.src.name}: {len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} h, "
      f"{len({r['speaker'] for r in rows})} speakers, licenses {sorted({r['license'] for r in rows})}")
if not a.push:
    print(f"DRY RUN: would upload to datasets/{a.repo}/{a.src.name}. Add --push.")
    raise SystemExit
hf =["hf", "upload", a.repo, "--repo-type", "dataset", *(["--create-pr"] if a.pr else [])]
card = Path(__file__).parent.parent / "docs" / "cards" / "dataset.md"
subprocess.run(hf + [str(a.src), a.src.name, "--commit-message", f"Add/update {a.src.name}"], check=True)  # resumable
subprocess.run(hf + [str(card), "README.md", "--commit-message", "Update dataset card"], check=True)
