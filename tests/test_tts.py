from zio.text.segment import segment
from zio.tts.vocab import CHARS


def test_segment_adds_spaces_keeps_english():
    assert " " in segment("ស្ពានកំពង់ចម្លងអ្នកលឿងនៅព្រៃវែង")
    assert segment("hello world") == "hello world"


def test_vocab_covers_slr42():
    import json, pathlib
    m = pathlib.Path("data/processed/km_male/manifest.jsonl")
    if m.exists():
        used = set("".join(json.loads(l)["text"] for l in m.read_text().splitlines()))
        assert used - set(CHARS + " ,.!?;:'\"-") == set()
