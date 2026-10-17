# 05 — Release

Everything uploads with the `hf` CLI (`uv tool install huggingface_hub && hf auth login`). Scripts are dry runs unless `--push`.

## Model → `sombochea/zio-tts`
```bash
uv run --extra tts python scripts/push_to_hf.py <run_dir> sombochea/zio-tts                 # stage + list files
uv run --extra tts python scripts/push_to_hf.py <run_dir> sombochea/zio-tts --push --note "v2, step N"   # private
```
Staging writes a weights-only checkpoint (998 → 333 MB), `config.json`, `speakers.pth`, `zio-tts.onnx`, `vocab.json`, generated `samples/` and the card from [`docs/cards/tts.md`](cards/tts.md). The Kaggle kernel runs this automatically. On Modal: `uv run python scripts/modal_publish.py /vol/outputs/<run>/<run_dir> sombochea/zio-tts --push`.

## Dataset → `sombochea/zio-tts-data`
```bash
uv run python scripts/push_dataset.py data/processed/<folder> --push         # owner
uv run python scripts/push_dataset.py data/processed/<folder> --push --pr    # contributor: opens a PR
```
Uploads the folder (resumable) and the card [`docs/cards/dataset.md`](cards/dataset.md).

## Before making a model public
1. Listen to `samples/` and your own sentences.
2. The license is CC BY-SA 4.0 (share-alike training data; this is a precaution, not legal advice).
3. Add numbers to the card's *Evaluation* section (CER via our STT, a small MOS panel).
4. Release only voices you have consent for, and say that the outputs are synthetic.

STT: [`docs/cards/stt.md`](cards/stt.md) is a draft. Don't publish without CER/WER.
