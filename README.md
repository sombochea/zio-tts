# Zio — Khmer + English speech (TTS & STT)

Open-source text-to-speech and speech-to-text for Khmer and English. **Early research (v0).**

| Part | Approach | Status |
|---|---|---|
| TTS | VITS (coqui-tts) on graphemes | **v0 published (1 voice). v1 (16 voices, 2.9 h) is training; the interim checkpoint is not yet on the Hub.** [Hugging Face](https://huggingface.co/sombochea/zio-tts) (private until reviewed) |
| STT | Whisper + LoRA (FLEURS + own data) | code + smoke tests only; **no trained model yet** |
| English TTS | — | not supported yet (v0 saw no English audio) |
| Emotion / premium voice | one voice id per (speaker, emotion) | code ready; needs studio recordings |

Why graphemes: espeak-ng has no Khmer voice, so there is no off-the-shelf G2P. Why VITS: StyleTTS2 (Kokoro's base) needs a Khmer PL-BERT that doesn't exist yet.

## Use the TTS model
```bash
git clone https://github.com/sombochea/zio-tts && cd zio-tts
uv sync --extra tts                       # Python 3.12+, ffmpeg
hf download sombochea/zio-tts --local-dir zio-tts
uv run zio speak "សួស្តី តើអ្នកសុខសប្បាយជាទេ" hello.wav --run zio-tts
```
No PyTorch? Use the ONNX file: [`examples/onnx_infer.py`](examples/onnx_infer.py). Details and limitations are in the model card.

## Train / contribute
| Doc | |
|---|---|
| [01 data](docs/01-data.md) | prepare corpora into 24 kHz manifests |
| [02 recording & emotion](docs/02-recording-and-emotion.md) | record the premium voice, emotion ids |
| [03 TTS training](docs/03-tts-training.md) | train VITS (local or Modal) |
| [04 STT training](docs/04-stt-training.md) | Whisper LoRA |
| [06 release](docs/06-release.md) | publish to Hugging Face (Modal, or locally with `hf`) |
| [08 Kaggle](docs/08-kaggle.md) | train on a free Kaggle GPU and publish with `hf` |
| [07 local training](docs/07-local-training.md) | Apple silicon / small RAM profiles (measured) |

```bash
uv sync --extra tts --extra stt && uv run pytest
```

## License
Code: Apache-2.0. TTS v0 weights: CC BY-SA 4.0 (training data SLR42 is share-alike). See the model card.
