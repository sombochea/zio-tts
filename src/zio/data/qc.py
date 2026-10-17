"""Recording QC: flags clipping, low level, high noise floor, bad length.

uv run python -m zio.data.qc recordings/*.wav
"""
import sys

import numpy as np
import soundfile as sf


def check(path: str) -> list[str]:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return ([f"low_sr:{sr}"] if sr < 44100 else []) + check_array(x.mean(1), sr)


def check_array(x: np.ndarray, sr: int, min_snr: float = 35) -> list[str]:
    flags = []
    if not 1.0 <= len(x) / sr <= 20.0:
        flags.append(f"duration:{len(x) / sr:.1f}s")
    if (np.abs(x) > 0.999).mean() > 1e-4:
        flags.append("clipping")
    frame = sr // 50
    rms = np.sqrt((x[: len(x) // frame * frame].reshape(-1, frame) ** 2).mean(1)) + 1e-9
    db = 20 * np.log10(rms)
    snr = np.percentile(db, 95) - np.percentile(db, 2)  # speech level vs noise floor (p2: trimmed clips have little silence)
    if np.percentile(db, 95) < -35:
        flags.append("too_quiet")
    if snr < min_snr:
        flags.append(f"noisy:snr~{snr:.0f}dB")
    return flags


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        f = check(p)
        bad += bool(f)
        print(p, "OK" if not f else " ".join(f))
    print(f"{bad}/{len(sys.argv) - 1} flagged", file=sys.stderr)
