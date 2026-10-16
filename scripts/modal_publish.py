"""Publish a trained TTS run from the Modal volume to the Hugging Face Hub (token comes from the Modal secret).

  uv run python scripts/modal_publish.py <run_dir_on_volume> <hf_user/repo> [--push] [--public]
Dry run by default; private repo by default. Redeploy first if code/docs changed: modal deploy scripts/modal_train.py
"""
import sys

import modal

a = sys.argv[1:]
fn = modal.Function.from_name("zio-train", "publish")
with modal.enable_output():
    note = a[a.index("--note") + 1] if "--note" in a else ""
    fn.remote(a[0], a[1], "tts", "--push" in a, "--public" in a, note)
