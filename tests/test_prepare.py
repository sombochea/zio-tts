import numpy as np

from zio.data.prepare import SR, trim


def test_trim_removes_silence():
    x = np.concatenate([np.zeros(SR), 0.5 * np.sin(np.arange(SR) * 0.1), np.zeros(SR)]).astype("float32")
    y = trim(x)
    assert SR < len(y) < 1.5 * SR


def test_clean_levels_and_drops_noise():
    from zio.data.prepare import RMS_DB, clean

    rng = np.random.default_rng(0)
    t = np.arange(2 * SR) / SR
    speech = (0.05 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype("float32")  # bursts + silence
    y = clean(speech, SR, min_snr=20)
    assert y is not None and abs(20 * np.log10(np.sqrt((y ** 2).mean())) - RMS_DB) < 1  # loudness normalized
    noisy = speech + 0.03 * rng.standard_normal(len(speech)).astype("float32")
    assert clean(noisy, SR, min_snr=20) is None  # noise floor too high -> dropped


def test_clean_text_rejects_unknown_chars():
    from zio.data.prepare import clean_text

    assert clean_text("សួស្តី​ ១២", seg=False) == "សួស្តី ដប់ពីរ"
    assert clean_text("50% off", seg=False) is None
