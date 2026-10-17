# 03 — Training (TTS)

One small VITS model ([coqui-tts](https://github.com/idiap/coqui-ai-TTS)) on Khmer graphemes, multi-speaker, 24 kHz. Everything is driven by a YAML config:

| Config | What |
|---|---|
| `tts_v2.yaml` | current model: all subsets in the dataset, initialised from v1 |
| `tts_finetune.yaml` | your own voice on top of the published model |
| `tts_v1.yaml`, `tts_base.yaml` | history: v1 (SLR42, 16 voices), v0 (one speaker) |
| `tts_tiny.yaml`, `tts_voices_tiny.yaml`, `tts_local*.yaml` | smoke tests and laptop runs |

Config keys: `data` (one folder or a list), `voices` (one embedding per `speaker__emotion`), `speakers` (filter), `resume` (checkpoint to start from), `limit`, `max_audio_s`, plus `batch`, `epochs`, `save_step`.

## Where to run
| | Command | Notes |
|---|---|---|
| Laptop (CPU/Apple silicon) | `uv run --extra tts python -m zio.tts.train configs/tts_local.yaml` | ~3.3 GB RAM (`tts_local_tiny`: ~2.5 GB). For checking the pipeline, not for quality. Keep ~3 GB of disk free. |
| **Kaggle** (free T4/P100, ~9 h/session) | `uv run python scripts/kaggle_launch.py --ckpt <best_model.pth>` | Pulls data from the HF dataset, trains until the time budget, then publishes. |
| Modal (A100, paid) | `modal deploy scripts/modal_train.py && uv run python scripts/modal_spawn.py configs/tts_v2.yaml zio.tts.train` | Data on the `zio-data` volume: `modal volume put zio-data data/processed /processed`. |

### Kaggle, once
```bash
uv tool install kaggle huggingface_hub && kaggle auth login && hf auth login
```
On the kernel page, add the secret `HF_TOKEN` (Add-ons → Secrets) so it can upload. Without it, the release stays in the kernel output (`kaggle kernels output <user>/zio-tts-train -p kaggle_out`).
Follow it with `kaggle kernels status|logs <user>/zio-tts-train`. To continue in a new session, relaunch with the newest `best_model*.pth` as `--ckpt`. Env knobs: `ZIO_CONFIG`, `ZIO_HOURS` (default 8.5), `ZIO_REPO`, `ZIO_DATA`.

## Fine-tune on your voice
About 30 minutes of clean, consented audio is enough to start.
```bash
uv run python -m zio.data.prepare folder data/processed/my_voice --src recordings --source me --license cc-by-4.0
hf download sombochea/zio-tts best_model.pth --local-dir base
uv run --extra tts python -m zio.tts.train configs/tts_finetune.yaml
uv run zio speak "សួស្តី" me.wav --run outputs/tts_ft/<run> --voice <your_speaker>__neutral
```
`resume` accepts the published weights-only checkpoint. Layers whose shape changed (the speaker table) start fresh, and everything else is reused. With little audio, also list `data/processed/slr42_km_male` under `data:`, so pronunciation doesn't drift.

## Why this design
- **Graphemes, not phonemes**: espeak-ng has no Khmer voice, so no off-the-shelf G2P exists. Text is normalized and word-segmented (khmer-nltk) before synthesis, matching the word-spaced training text.
- **VITS, not StyleTTS2/Kokoro**: those need a Khmer PL-BERT that doesn't exist. VITS is end-to-end, trains from scratch, and exports to ONNX.
- **Quality comes from data**: v0 (46 min, 1 speaker) sounded poor, and v1 (2.9 h, 16 speakers) is understandable. v2 adds ~15–35 h of open data and female voices. Emotional, studio-quality voices need recordings ([02](02-recording.md)).

## Expressiveness at inference
`--speed` scales durations, and `--expressive` is the VITS noise scale (default 0.667; higher is livelier, lower is flatter and steadier). The ONNX `scales` input is `[expressive, 1/speed, 0.8]`.

## Not done yet
MOS / intelligibility (CER via STT) evaluation, English audio, event tokens (`[laugh]`) in the vocab, and a Khmer G2P.
