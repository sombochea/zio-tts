# 03 — TTS training (VITS, graphemes)

## Decisions (and why)
- **No Khmer G2P exists in espeak-ng** (checked 1.52: no `km` voice), so v0 feeds **graphemes** (Khmer characters + a-z) directly. A rule-based Khmer G2P can replace this later; the vocabulary in `zio/tts/vocab.py` would then change and the model be retrained.
- **VITS via `coqui-tts`** instead of StyleTTS2: StyleTTS2 needs a Khmer-capable PL-BERT (none exists) and a multi-stage setup; VITS is end-to-end, trains from scratch on graphemes, exports to ONNX.
- Training text is **word-spaced** (SLR42 format). Real input is not, so inference runs `normalize -> segment (khmer-nltk)` first.
- SLR42 "male" contains **16 speakers** (2.9 h; the largest has 46 min).
  - **v0** trained on the single largest speaker (`configs/tts_base.yaml`, 46 min). Result: poor quality — too little data for VITS from scratch.
  - **v1** trains one multi-speaker model on all 16 (`configs/tts_v1.yaml`, `voices: true`): 3.8× more data shared across speakers, voice chosen by id (`km_6753__neutral`). This is the first model we publish.
  - Why not more: the public Khmer TTS data is small. The real fix for "very high-level" quality is studio recordings ([02](02-recording-and-emotion.md)) or a pretrained multilingual base model.

## Smoke test (CPU, ~2 min, ~2.7 GB RAM)
```bash
uv run --extra tts python -m zio.tts.train configs/tts_tiny.yaml
uv run --extra tts python -m zio.tts.infer outputs/tts_tiny/<run>/ "សួស្តី 12" out.wav   # noise at 2 epochs, but proves the path
```
## Real run (Modal)
```bash
modal volume create zio-data && modal volume put zio-data data/processed /processed   # once
modal deploy scripts/modal_train.py                                 # redo after any code/config change
uv run python scripts/modal_spawn.py configs/tts_v1.yaml zio.tts.train   # survives your laptop closing
modal app logs zio-train                                            # progress; `modal app stop zio-train` cancels
modal volume ls zio-data /outputs/tts_v1                            # checkpoints (committed every 10 min)
```
Why deploy+spawn: a plain `modal run` is cancelled if the local client dies, even with `--detach`.
To continue a run, set `resume: outputs/<run>/<run_dir>/checkpoint_N.pth` in the config (weights + optimizer restored, epoch counter restarts).
Publish: [06-release](06-release.md).

## Not done yet
- Emotion conditioning (needs emotional recordings; plan: emotion embedding added to the VITS text encoder).
- MOS / intelligibility evaluation; female voices; English.
- `coqui-tts` + transformers 5 needs a tiny shim (`zio/tts/__init__.py`).
