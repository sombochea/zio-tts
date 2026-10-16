"""uv run python scripts/modal_spawn.py <config.yaml> [module]   (after `modal deploy scripts/modal_train.py`)"""
import sys

import modal

call = modal.Function.from_name("zio-train", "train").spawn(sys.argv[1], *sys.argv[2:3])
print("spawned", call.object_id)
