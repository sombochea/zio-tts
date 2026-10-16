"""uv run --extra stt python -m zio.stt.infer <model_dir_or_hf_id> audio.wav [km|en]"""
import sys

import torch
from transformers import pipeline

from zio.stt.data import load_audio


def transcribe(model: str, wav: str, lang: str = "km") -> str:
    asr = pipeline("automatic-speech-recognition", model=model, chunk_length_s=30,
                   device=0 if torch.cuda.is_available() else -1)
    x = load_audio({"path": wav})
    name = {"km": "khmer", "en": "english"}[lang]
    return asr(x, generate_kwargs={"language": name, "task": "transcribe"})["text"]


if __name__ == "__main__":
    print(transcribe(sys.argv[1], sys.argv[2], *sys.argv[3:4]))
