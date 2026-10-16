"""Khmer/English text normalization: numerals, currency -> spoken Khmer words."""
import re

_KM_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
_ONES = ["សូន្យ", "មួយ", "ពីរ", "បី", "បួន", "ប្រាំ", "ប្រាំមួយ", "ប្រាំពីរ", "ប្រាំបី", "ប្រាំបួន"]
_TENS = {1: "ដប់", 2: "ម្ភៃ", 3: "សាមសិប", 4: "សែសិប", 5: "ហាសិប",
         6: "ហុកសិប", 7: "ចិតសិប", 8: "ប៉ែតសិប", 9: "កៅសិប"}
# (value, word) for 10^2..10^5 units; 10^6 handled by recursion
_UNITS = [(100000, "សែន"), (10000, "ម៉ឺន"), (1000, "ពាន់"), (100, "រយ")]


def number_to_khmer(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (_ONES[o] if o else "")
    if n >= 1_000_000:
        m, rest = divmod(n, 1_000_000)
        return number_to_khmer(m) + "លាន" + (number_to_khmer(rest) if rest else "")
    out = ""
    for v, w in _UNITS:
        q, n = divmod(n, v)
        if q:
            out += _ONES[q] + w
    return out + (number_to_khmer(n) if n else "")


_CUR = {"៛": "រៀល", "$": "ដុល្លារ"}
_NUM = re.compile(r"(?P<cur>[៛$])?(?P<n>[0-9][0-9,]*)(?P<cur2>៛)?")


def normalize(text: str) -> str:
    text = text.translate(_KM_DIGITS)

    def sub(m: re.Match) -> str:
        words = number_to_khmer(int(m["n"].replace(",", "")))
        cur = m["cur"] or m["cur2"]
        return words + (_CUR[cur] if cur else "")

    return _NUM.sub(sub, text)
