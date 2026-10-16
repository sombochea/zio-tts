"""Text -> IPA via espeak-ng (needs the `espeak-ng` binary on PATH)."""
import subprocess


def to_ipa(text: str, lang: str = "km") -> str:
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", lang, text],
                       capture_output=True, text=True, check=True)
    return " ".join(r.stdout.split())
