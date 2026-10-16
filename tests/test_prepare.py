import numpy as np

from zio.data.prepare import SR, trim


def test_trim_removes_silence():
    x = np.concatenate([np.zeros(SR), 0.5 * np.sin(np.arange(SR) * 0.1), np.zeros(SR)]).astype("float32")
    y = trim(x)
    assert SR < len(y) < 1.5 * SR
