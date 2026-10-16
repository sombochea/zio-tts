---
license: apache-2.0
language: [km, en]
pipeline_tag: text-to-speech
library_name: coqui-tts
tags: [khmer, tts, vits]
---
# Zio TTS (Khmer + English) — v0

VITS (coqui-tts) trained on **graphemes** (Khmer characters + a–z), 24 kHz.

## Status
Research v0. Single voice trained on ~1 h of Google Khmer TTS data (OpenSLR SLR42, CC BY-SA 4.0). **TODO before release: fill in listening results and MOS.**

## Usage
```bash
git clone <github-url> && cd zio && uv sync --extra tts
uv run zio speak "សួស្តី" out.wav --run <downloaded_dir>
```
Input is normalized (Khmer numerals/currency) and word-segmented (khmer-nltk) automatically.

## Data & license
Trained on SLR42 (CC BY-SA 4.0, attribution: Google). Share-alike may apply to derived weights — **verify before choosing a license.** No voices were cloned; speakers are from the open dataset.

## Limitations
Small data; mispronunciations likely (no G2P, graphemes only); emotions not in v0; mixed-language sentences untested.
