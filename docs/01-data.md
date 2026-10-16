# 01 — Data

## Manifest format
One JSON per line (`manifest.jsonl`): `id, audio, text, lang, speaker, emotion, duration`.
Audio is 24 kHz mono 16-bit wav, silence-trimmed, peak-normalized, 1–20 s.
Text is normalized (Khmer numerals/currency -> words, `zio.text.normalize`).

## Step 1 — SLR42 (Google Khmer, CC BY-SA 4.0)
Only the male zip (~866 MB, ~2.9 h, 2,906 utts) is currently listed on OpenSLR.
```bash
mkdir -p data/raw/slr42 && cd data/raw/slr42
curl -LO https://www.openslr.org/resources/42/km_kh_male.zip && unzip -q km_kh_male.zip && cd -
uv run python -m zio.data.prepare slr42 data/raw/slr42/km_kh_male data/processed/km_male
```
Transcripts are already space-segmented into words.

## Licensing
SLR42 is CC BY-SA: share-alike applies to derived data; check before releasing weights/datasets under a different license.

## Next
FLEURS / Common Voice Khmer for STT, English corpora, and own studio recordings (emotion + premium voice) are added as new `*_items` readers in `prepare.py`.
