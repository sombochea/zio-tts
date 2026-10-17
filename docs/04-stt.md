# 04 — STT (Whisper + LoRA)

Khmer + English in one Whisper model: a language token per sample, LoRA on attention and FFN, merged weights at the end. **Not trained at scale yet.**

Data: FLEURS `km_kh` / `en_us` (auto-downloaded) plus any dataset folders listed under `manifests:` (their `metadata.jsonl`). Khmer word spaces are removed so all sources agree.

| Run | Command |
|---|---|
| Smoke test (CPU, <1 GB, ~1 min) | `uv run --extra stt python -m zio.stt.train configs/stt_tiny.yaml` |
| Own GPU | `uv run --extra stt python -m zio.stt.train configs/stt_small.yaml` |
| Modal (A100) | `uv run modal run scripts/modal_train.py --config configs/stt_small.yaml` |

Bigger model: `model: openai/whisper-large-v3-turbo`, batch 8 / accum 4 on an A100.
Metrics: `cer` (Khmer) and `wer` (English), logged every `eval_every` steps. Both languages are currently mixed into one number.

```bash
uv run --extra stt zio transcribe clip.wav --model outputs/stt_small/merged --lang km
```
