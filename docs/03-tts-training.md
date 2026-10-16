# 03 — TTS training (VITS, graphemes)

## Decisions (and why)
- **No Khmer G2P exists in espeak-ng** (checked 1.52: no `km` voice), so v0 feeds **graphemes** (Khmer characters + a-z) directly. A rule-based Khmer G2P can replace this later; the vocabulary in `zio/tts/vocab.py` would then change and the model be retrained.
- **VITS via `coqui-tts`** instead of StyleTTS2: StyleTTS2 needs a Khmer-capable PL-BERT (none exists) and a multi-stage setup; VITS is end-to-end, trains from scratch on graphemes, exports to ONNX.
- Training text is **word-spaced** (SLR42 format). Real input is not, so inference runs `normalize -> segment (khmer-nltk)` first.
- SLR42 "male" contains **16 speakers**; v0 trains on one (`speakers: [km_6753]`, 863 utts ≈ 1 h) to keep one consistent voice. Expect modest quality: this proves the pipeline, not the final voice.

## Smoke test (CPU, ~2 min, ~2.7 GB RAM)
```bash
uv run --extra tts python -m zio.tts.train configs/tts_tiny.yaml
uv run --extra tts python -m zio.tts.infer outputs/tts_tiny/<run>/ "សួស្តី 12" out.wav   # noise at 2 epochs, but proves the path
```
## Real run (Modal)
```bash
modal volume put zio-data data/processed /processed        # once
uv run modal run scripts/modal_train.py --config configs/tts_base.yaml --module zio.tts.train
modal volume get zio-data /outputs/tts_v0 ./outputs
```
Monitor with TensorBoard (`outputs/tts_v0/*/events*`). Listen to the `test_sentences` audio at each checkpoint.

## Not done yet
- Emotion conditioning (needs emotional recordings; plan: emotion embedding added to the VITS text encoder).
- Multi-speaker, ONNX export, MOS evaluation.
- `coqui-tts` + transformers 5 needs a tiny shim (`zio/tts/__init__.py`).
