from zio.text.normalize import normalize, number_to_khmer


def test_numbers():
    assert number_to_khmer(7) == "ប្រាំពីរ"
    assert number_to_khmer(21) == "ម្ភៃមួយ"
    assert number_to_khmer(115) == "មួយរយដប់ប្រាំ"
    assert number_to_khmer(2500) == "ពីរពាន់ប្រាំរយ"
    assert number_to_khmer(1_000_005) == "មួយលានប្រាំ"


def test_normalize_mixed():
    assert normalize("១២ apples") == "ដប់ពីរ apples"
    assert normalize("5,000៛") == "ប្រាំពាន់រៀល"
    assert normalize("$3") == "បីដុល្លារ"
