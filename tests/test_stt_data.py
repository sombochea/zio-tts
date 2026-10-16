from zio.stt.data import clean


def test_clean_km_drops_spaces_between_khmer_only():
    assert clean("ស្ពាន កំពង់ ក្នុង Byzantine ok", "km") == "ស្ពានកំពង់ក្នុង Byzantine ok"
    assert clean(" hello world ", "en") == "hello world"
