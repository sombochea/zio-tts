---
license: cc-by-sa-4.0
language: [km]
task_categories: [text-to-speech, automatic-speech-recognition]
tags: [khmer, speech, tts, zio]
pretty_name: Zio Khmer Speech (TTS-ready)
configs:
  - config_name: slr42_km_male
    data_dir: slr42_km_male
    default: true
  - config_name: seanghay_km
    data_dir: seanghay_km
  - config_name: ddd_km
    data_dir: ddd_km
---
# Zio Khmer Speech

Open Khmer speech, cleaned to one format for training text-to-speech. Used to train [Zio TTS](https://huggingface.co/sombochea/zio-tts) ([code](https://github.com/sombochea/zio-tts)). **Contributions welcome**, especially emotional or expressive speech.

```python
from datasets import load_dataset
ds = load_dataset("sombochea/zio-tts-data", "seanghay_km", split="train")
```
Or download folders as-is for training: `hf download sombochea/zio-tts-data --repo-type dataset --include "seanghay_km/*" --local-dir data/processed`

## Subsets
| Subset | Source | License | Speakers | Notes |
|---|---|---|---|---|
| `slr42_km_male` | [OpenSLR 42](https://openslr.org/42) (Google) | CC BY-SA 4.0 | 16 male | studio, news-style reading; 2.9 h |
| `seanghay_km` | [seanghay/km-speech-corpus](https://huggingface.co/datasets/seanghay/km-speech-corpus) | CC BY 4.0 | unlabeled (one id) | conversational sentences, 16 kHz source; 7.8 h kept of 10.4 h (noisy clips dropped) |
| `ddd_km` | [Digital-Divide-Data/khm-asr-cultural](https://huggingface.co/datasets/Digital-Divide-Data/khm-asr-cultural) | CC BY-SA 4.0 | male + female | up to 30 h of clips with SNR ≥ 20 dB, 16 kHz source |

Every row carries its own `source` and `license`. The collection is CC BY-SA 4.0 because of its share-alike parts. Credit the original authors when you use a subset.

## Format
Each subset is a folder with `wavs/` and `metadata.jsonl` (Hugging Face *audiofolder*). Audio is 24 kHz mono 16-bit, 1–20 s, silence-trimmed and loudness-normalized (−20 dBFS RMS). Clips from 16 kHz sources are upsampled, so they have no content above 8 kHz.

| Column | Example | |
|---|---|---|
| `file_name` | `wavs/km_6753_001.wav` | audio path |
| `text` | `សួស្តី តើ អ្នក សុខសប្បាយ ជា ទេ` | normalized (numbers as words), word-spaced |
| `speaker` | `km_6753`, `ddd_f-adt1-0001` | unique across subsets |
| `emotion` | `neutral` | `neutral`, `happy`, `sad`, `angry`, `surprised`, `whisper`, ... |
| `events` | `["laugh"]` | non-speech sounds in the clip: `laugh`, `breath`, `sigh`, `cry`, ... |
| `lang`, `duration`, `source`, `license`, `id` | | |

## Contribute
1. You need consent from every speaker to record and openly release their voice.
2. Prepare your audio with the [Zio tools](https://github.com/sombochea/zio-tts/blob/main/CONTRIBUTING.md) into a new folder (`<source>_<lang>`), and label `emotion` / `events` honestly.
3. `uv run python scripts/push_dataset.py data/processed/<folder> --push --pr` opens a pull request here.

## Limitations
Mostly neutral read speech; no emotional labels yet. Speaker and gender balance is uneven. Text comes from the sources' transcripts and may contain errors.
