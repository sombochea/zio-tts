# 04 — STT training (Whisper + LoRA)

Khmer + English in one model. Per-sample language token, LoRA on all attention/FFN projections, merged weights saved at the end.

## Data
- FLEURS `km_kh` and `en_us` (auto-downloaded from HF, memory-mapped on disk — needs a few GB free; audio decoded with soundfile, no ffmpeg/torchcodec needed).
- Our `manifest.jsonl` files (see [01-data](01-data.md)). Khmer text is normalized to **no spaces between Khmer characters** (SLR42's word spacing is stripped) so all sources agree.

## Smoke test (CPU, ~1 min, <1 GB RAM, no big downloads)
Uses whisper-tiny and 16 local utterances only.
```bash
uv run --extra stt python -m zio.stt.train configs/stt_tiny.yaml
```
## Real run on Modal (recommended)
```bash
modal volume put zio-data data/processed /processed   # once
uv run modal run scripts/modal_train.py --config configs/stt_small.yaml
modal volume get zio-data /outputs/stt_small ./outputs
```
Uses the existing `huggingface-secret` and `huggingface-cache` volume; A100-80GB, 24 h timeout (edit `scripts/modal_train.py`).

## Real run (own GPU)
```bash
huggingface-cli login            # optional, faster downloads
uv run --extra stt python -m zio.stt.train configs/stt_small.yaml
```
Scale up: set `model: openai/whisper-large-v3-turbo`, batch 8 / accum 4 on an A100. Add more manifests (Common Voice, own recordings) under `manifests:`.

## Metrics
`cer` (primary for Khmer, no word boundaries) and `wer` (English) are logged each `eval_every` steps on FLEURS validation (km+en mixed — split per language in `docs/05-eval` later).

## Inference
```bash
uv run --extra stt python -m zio.stt.infer outputs/stt_small/merged clip.wav km
```
## Known limits
- Evaluation mixes both languages in one number; per-language report is TODO.
- No CTranslate2 export yet (`ct2-transformers-converter` on `merged/` when needed).
