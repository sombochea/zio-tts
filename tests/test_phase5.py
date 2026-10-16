import numpy as np
import soundfile as sf

from zio.data.qc import check
from zio.data.script import pick


def test_pick_prefers_new_bigrams():
    t = ["កខគឃងចឆជឈញដឋឌ", "កខគឃងចឆជឈញដឋ", "ណតថទធនបផពភមយរ"]
    assert pick(t, 2, lo=5)[0] == t[0] and pick(t, 2, lo=5)[1] == t[2]


def test_qc(tmp_path):
    sr = 48000
    t = np.arange(sr * 2) / sr
    clean = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype("float32")
    sf.write(tmp_path / "ok.wav", clean + 1e-4 * np.random.randn(len(t)).astype("float32"), sr)
    sf.write(tmp_path / "clip.wav", np.clip(clean * 10, -1, 1), sr)
    assert check(str(tmp_path / "ok.wav")) == []
    assert "clipping" in check(str(tmp_path / "clip.wav"))
