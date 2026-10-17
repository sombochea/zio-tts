# 01 — Data

All training data lives in the public dataset [sombochea/zio-tts-data](https://huggingface.co/datasets/sombochea/zio-tts-data), one folder per source. Train from it directly:
```bash
hf download sombochea/zio-tts-data --repo-type dataset --local-dir data/processed   # or --include "seanghay_km/*"
```

## Format
Each folder is a Hugging Face *audiofolder*: `wavs/<shard>/*.wav` (1,000 per folder; the Hub caps folders at 10k files) + `metadata.jsonl`, one row per clip:
```json
{"id": "km_6753_001", "file_name": "wavs/000/km_6753_001.wav", "text": "សួស្តី តើ អ្នក សុខសប្បាយ ជា ទេ", "lang": "km",
 "speaker": "km_6753", "emotion": "neutral", "events": [], "duration": 2.4, "source": "https://openslr.org/42", "license": "cc-by-sa-4.0"}
```
- Audio: 24 kHz mono 16-bit, 1–20 s, silence-trimmed, −20 dBFS RMS (peaks ≤ 0.95).
- Text: normalized (digits and `៛`/`$` → words) and **word-spaced** Khmer. Rows with characters outside the [vocab](../src/zio/tts/vocab.py) are dropped.
- `speaker` must be unique across folders (prefix it with the source). `emotion` and `events` are described in [02](02-recording.md).

## Sources
| Folder | Prepare | License |
|---|---|---|
| `slr42_km_male` | `prepare slr42 data/processed/slr42_km_male --src data/raw/slr42/km_kh_male` (download [km_kh_male.zip](https://www.openslr.org/resources/42/km_kh_male.zip) first) | CC BY-SA 4.0 |
| `seanghay_km` | `prepare seanghay data/processed/seanghay_km --min-snr 20` | CC BY 4.0 |
| `ddd_km` | `prepare ddd data/processed/ddd_km --max-hours 30 --min-snr 20` | CC BY-SA 4.0 |

Run each as `uv run --extra stt --extra tts python -m zio.data.prepare ...`. Hugging Face sources are streamed, never fully downloaded. `--max-hours` caps the size, and `--min-snr` drops noisy clips.

Not enough disk or bandwidth (DDD streams from a 91 GB source)? Prepare and upload on Kaggle instead. It's a CPU job, so it doesn't queue for a GPU:
```bash
uv run python scripts/kaggle_launch.py --prep "ddd data/processed/ddd_km --max-hours 30 --min-snr 20"
```
The first time, attach the `HF_TOKEN` secret to the `zio-data-prep` kernel ([03](03-training.md#kaggle-once)).

## Add a source
1. Add an entry to `CORPORA` in [`prepare.py`](../src/zio/data/prepare.py). For a Hub dataset that's the repo, its text column and its speaker column, with no new code.
2. Prepare it, listen to a few clips, then upload: `uv run python scripts/push_dataset.py data/processed/<folder> --push`.
3. Add the folder to `configs:` and the table in [`docs/cards/dataset.md`](cards/dataset.md).

Licenses: only add data whose license allows redistribution. Share-alike (CC BY-SA) data makes the model and dataset CC BY-SA too.
