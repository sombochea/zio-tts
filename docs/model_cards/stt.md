---
license: apache-2.0
language: [km, en]
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-small
tags: [khmer, whisper, lora]
---
# Zio STT (Khmer + English) — DRAFT, NOT YET TRAINED/PUBLISHED

Whisper fine-tuned with LoRA (merged) on FLEURS (km, en) + local manifests.

## Results
**TODO: report CER (Khmer) / WER (English) on FLEURS test vs the base model.** Do not publish without numbers.

## Usage
```bash
uv run --extra stt zio transcribe clip.wav --model <repo_or_dir> --lang km
```
## Data & license
FLEURS (CC BY 4.0), SLR42 (CC BY-SA 4.0). Whisper base weights: MIT/Apache.
