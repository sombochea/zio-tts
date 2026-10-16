# Zio — Khmer + English speech (TTS & STT)

Open-source text-to-speech and speech-to-text for Khmer and English. Work in progress (v0).

| Part | Approach | Status |
|---|---|---|
| STT | Whisper + LoRA, FLEURS + own data | training code ready, no published results yet |
| TTS | VITS (coqui-tts) on graphemes | v0 training on Modal; ONNX export works |
| Emotion | one voice id per (speaker, emotion) | code ready; needs studio recordings |

Why graphemes: espeak-ng has no Khmer voice, so there is no off-the-shelf G2P. Why VITS: StyleTTS2 (used by Kokoro) needs a Khmer-capable PL-BERT that doesn't exist yet. See `docs/`.

## Quick start
```bash
uv sync --extra tts --extra stt
uv run pytest
uv run zio speak "សួស្តី" out.wav --run <tts_run_dir>
uv run zio transcribe clip.wav --model <whisper_dir> --lang km
```

## Docs (in order)
[01 data](docs/01-data.md) · [02 recording & emotion](docs/02-recording-and-emotion.md) · [03 TTS training](docs/03-tts-training.md) · [04 STT training](docs/04-stt-training.md) · [06 release](docs/06-release.md)

Training runs on [Modal](https://modal.com): `scripts/modal_train.py`.

## License
Code: Apache-2.0. Weights: see model cards (training data includes CC BY-SA sources).
