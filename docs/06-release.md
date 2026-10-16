# 06 — Release to Hugging Face / GitHub

Checklist (nothing here runs automatically; uploads are dry-run unless `--push`):

1. Listen to the checkpoint on held-out sentences; fill the **TODO** results in `docs/model_cards/*.md` (MOS-style notes for TTS, CER/WER for STT). Don't publish without them.
2. Decide the weight license (SLR42 is CC BY-SA 4.0 → likely CC BY-SA for TTS weights; update the card).
3. Export ONNX (CPU/edge): `uv run --extra tts python -m zio.tts.export_onnx <run_dir> zio-tts.onnx` (inputs: `input`, `input_lengths`, `scales=[noise, length, noise_dp]`; tokenization must match `zio.tts.vocab` — an ONNX-only runtime needs the char list shipped alongside).
4. Stage and inspect: `uv run --extra tts python scripts/push_to_hf.py tts <run_dir> --repo <user>/zio-tts`
5. Upload (private by default): add `--push`; add `--public` to publish.
6. Tag a GitHub release; link the HF repos from the README.

Voice ethics: only release voices with signed consent; say in the card that outputs are synthetic.
