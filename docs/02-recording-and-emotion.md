# 02 — Recording the premium voice & emotion conditioning

Public Khmer data has no emotional speech and is mostly neutral news reading, so **the "very high-level, emotional voice" must come from your own recordings**. Everything here is ready except the recordings themselves.

## 1. Before recording
- 1–3 professional native speakers. **Signed consent/release** covering training and open-source release of the model and (if you publish it) the data. Without it, don't publish.
- Quiet treated room (or vocal booth), large-diaphragm condenser, pop filter, ~10 cm, fixed gain. Record **48 kHz, 24-bit, mono WAV**, no compression/denoise/EQ (pipeline resamples to 24 kHz).
- Budget: ~10–30 h neutral + ~2–3 h per emotion (neutral, happy, sad, angry, ...).

## 2. Script
```bash
uv run python -m zio.data.script data/processed/km_male/manifest.jsonl \
  --emotions neutral,happy,sad,angry --per-emotion 200 > script.csv
```
Greedy selection maximises Khmer character-bigram coverage. **Caveat:** these are news sentences; acting "angry" on a weather report is unnatural. For emotional classes, have a native writer author sentences that fit each emotion and use the same CSV format (`id,emotion,text`).

## 3. QC every session
```bash
uv run python -m zio.data.qc recordings/*.wav      # flags clipping, quiet, noisy, bad length
```
Re-record flagged takes. Listen to a random 5% yourself.

## 4. Manifest
Run `zio.data.prepare` on the recordings (add a reader like `slr42_items`) writing `speaker: <studio id>`, `emotion: <label>` per row.

## 5. Emotion conditioning (implemented)
Each `(speaker, emotion)` pair becomes one VITS speaker-embedding id (`studio1__happy`). Set `voices: true` in the TTS config. Pick at inference:
```bash
uv run --extra tts python -m zio.tts.infer <run_dir> "សួស្តី" out.wav studio1__happy
```
Limits: emotions are discrete labels (no intensity slider), and an emotion is only available for speakers recorded in it. Inline tags (`[happy]`) in text are not parsed yet.

Smoke test: `configs/tts_voices_tiny.yaml`.
