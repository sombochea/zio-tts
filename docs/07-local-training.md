# 07 — Local training (Apple silicon / small RAM)

For pipeline checks and small experiments on a laptop. Real quality needs Modal ([03](03-tts-training.md), [04](04-stt-training.md)).

| Profile | Command | Peak RAM (measured) | Notes |
|---|---|---|---|
| STT tiny | `uv run --extra stt python -m zio.stt.train configs/stt_local.yaml` | ~0.8 GB | whisper-tiny + LoRA, 400 local utterances, no downloads |
| TTS small | `uv run --extra tts python -m zio.tts.train configs/tts_local.yaml` | ~3.3 GB | batch 4, clips <= 6 s |
| TTS tiny | `uv run --extra tts python -m zio.tts.train configs/tts_local_tiny.yaml` | ~2.5 GB | batch 2, clips <= 4 s |

Measured with `/usr/bin/time -l` on an M-series Mac, CPU. Batch size barely changes VITS memory (model + optimizer + HiFi-GAN dominate), so **~2.5 GB is the floor for TTS**; on <8 GB machines use the STT profile or Modal.

## Disk matters too
A coqui checkpoint is ~1 GB (optimizer state included) and a run keeps `best_model` + checkpoints. Local configs set `keep_ckpts: 1` and a large `save_step`; budget ~3 GB free. A full disk kills a run at save time (this happened during testing).

## Tips
- Prepare data first: [01-data](01-data.md). Everything runs from `data/processed/km_male`.
- Speed: expect minutes per epoch on CPU for TTS. 50 epochs of one speaker gives a model that babbles recognisable Khmer-like sounds at best; use it to validate changes, then train on Modal.
- Lower memory further: `workers: 0` (already), shorter `max_audio_s`, `limit: N` utterances.
