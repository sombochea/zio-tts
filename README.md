# Zio — open Khmer speech

Open-source Khmer text-to-speech (and speech-to-text, in progress). Small, fast, and easy to fine-tune on your own voice.

| | |
|---|---|
| Model | [sombochea/zio-tts](https://huggingface.co/sombochea/zio-tts): VITS, 24 kHz, multi-speaker, ONNX included |
| Data | [sombochea/zio-tts-data](https://huggingface.co/datasets/sombochea/zio-tts-data): open Khmer speech in one clean format; [contributions welcome](CONTRIBUTING.md) |
| Status | Research. Khmer only; neutral voices; emotion labels supported once recordings exist. STT: code only |

## Speak
```bash
git clone https://github.com/sombochea/zio-tts && cd zio-tts && uv sync --extra tts   # Python 3.12+, ffmpeg
hf download sombochea/zio-tts --local-dir zio-tts
uv run zio speak "សួស្តី តើអ្នកសុខសប្បាយជាទេ" hello.wav --run zio-tts
```
Options: `--voice km_7548__neutral`, `--speed 1.2`, `--expressive 0.8` (livelier) or `0.4` (steadier). Without PyTorch, use [`examples/onnx_infer.py`](examples/onnx_infer.py).

## Docs
1. [Data](docs/01-data.md): dataset format, sources, adding a corpus
2. [Recording](docs/02-recording.md): studio voice, emotions and sound events
3. [Training](docs/03-training.md): local, Kaggle (free GPU), Modal; fine-tune on your voice
4. [STT](docs/04-stt.md): Whisper + LoRA
5. [Release](docs/05-release.md): publish the model and dataset with `hf`

```bash
uv sync --extra tts --extra stt && uv run pytest
```

## License
Code: Apache-2.0. Model weights and dataset: CC BY-SA 4.0 (some training data is share-alike). Each dataset row records its source license.
