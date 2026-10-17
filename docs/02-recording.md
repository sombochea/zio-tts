# 02 — Recording a voice (and emotions)

Open Khmer speech is almost all neutral reading. **Natural, emotional voices need new recordings.** The pipeline already handles them: label the rows and train.

## Before you record
- **Signed consent** from every speaker, covering training and open release of the model and the data. No consent, no release.
- Quiet, treated room; condenser mic with pop filter at ~10 cm; fixed gain. Record **48 kHz / 24-bit mono WAV** with no denoise, EQ or compression.
- Amount: ~30 min of a voice is enough to fine-tune ([03](03-training.md#fine-tune-on-your-voice)). Aim for 1–3 h for a strong voice, plus ~30–60 min per emotion.

## Script
```bash
uv run python -m zio.data.script data/processed/slr42_km_male/metadata.jsonl --emotions neutral,happy,sad --per-emotion 200 > script.csv
```
This picks sentences that cover the most Khmer character pairs. For emotions, news sentences sound wrong, so have a native writer add lines that fit each emotion (same CSV: `id,emotion,text`). Natural, conversational lines make the voice sound more human than read news does.

## Labels
| Field | Values | Effect today |
|---|---|---|
| `emotion` | `neutral`, `happy`, `sad`, `angry`, `surprised`, `whisper`, ... | each `(speaker, emotion)` becomes a voice: `--voice studio1__happy` |
| `events` | `laugh`, `breath`, `sigh`, `cry`, ... | stored in the dataset; the model will learn tags like `[laugh]` once enough labeled clips exist |

Label honestly: one emotion per clip, and list any non-speech sounds in `events`. Leave `events` empty if there are none.

## Check and prepare
```bash
uv run python -m zio.data.qc recordings/*.wav     # flags clipping, too quiet, noisy, bad length
```
Re-record flagged takes and listen to ~5% yourself. Then make the folder (`<source>_<lang>`, layout in [01](01-data.md)) by adding a reader to `prepare.py` that yields `emotion` per row, and share it ([CONTRIBUTING](../CONTRIBUTING.md)).
