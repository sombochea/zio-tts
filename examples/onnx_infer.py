"""Run Zio TTS from the ONNX export. No PyTorch needed.

  pip install onnxruntime soundfile "zio @ git+https://github.com/sombochea/zio-tts"
  python onnx_infer.py zio-tts.onnx vocab.json "សួស្តី តើអ្នកសុខសប្បាយជាទេ" out.wav [voice]
(voice names are listed in vocab.json under "voices")
"""
import json
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf

from zio.text.normalize import normalize
from zio.text.segment import segment


def synth(onnx_path: str, vocab_path: str, text: str, voice: str | None = None, noise=0.667, speed=1.0, noise_dp=0.8) -> np.ndarray:
    v = json.load(open(vocab_path, encoding="utf-8"))
    chars = [v["char_to_id"][c] for c in segment(normalize(text)).lower() if c in v["char_to_id"]]
    ids = [v["blank_id"]] * (2 * len(chars) + 1)  # VITS puts a blank token between all characters
    ids[1::2] = chars
    x = np.array([ids], dtype=np.int64)
    feed = {"input": x, "input_lengths": np.array([x.shape[1]], dtype=np.int64),
            "scales": np.array([noise, 1.0 / speed, noise_dp], dtype=np.float32)}
    if v.get("voices"):  # multi-voice model needs a speaker id
        feed["sid"] = np.array([v["voices"][voice or v["default_voice"]]], dtype=np.int64)
    out = ort.InferenceSession(onnx_path).run(None, feed)[0]
    return out.squeeze()


if __name__ == "__main__":
    onnx_path, vocab_path, text, out, *voice = sys.argv[1:6]
    sf.write(out, synth(onnx_path, vocab_path, text, *voice), 24000)
