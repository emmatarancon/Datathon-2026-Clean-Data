"""Canonical forms of option values and of the statement number multiset.

Decisions (documented and unit-tested):
- "1/2", "0,5" and "0.5" are the same value (canonical "1/2").
- "50%" is NOT that value: percent signs are kept, never converted.
- Thousands separators go away ("1,000" and "1 000" -> "1000"); a comma is a
  thousands separator only when exactly three digits follow, otherwise it is
  a decimal comma ("12,345" -> "12345" but "0,5" -> "0.5").
- Numbers become exact Fractions ("2.50" -> "5/2"), so "0.5" equals "1/2".
- Units survive: "81 cm2" stays "81cm2", different from plain "81".
- Fractions written "1/2" inside a statement count as one number "1/2".
"""
import re
from collections import Counter
from fractions import Fraction

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d{1,2}(?:\D|$))")


def _canonical_number_str(raw):
    """Canonical form of one number string ('0,5' already became '0.5')."""
    try:
        return str(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        return raw


def _canonical_number(match):
    return _canonical_number_str(match.group(0))


def _normalize_commas(text):
    text = THOUSANDS_RE.sub("", text)
    return DECIMAL_COMMA_RE.sub(".", text)


def canonical_value(raw):
    """The canonical string of one option value (commas, spaces, fractions)."""
    text = str(raw if raw is not None else "").strip().lower().replace(" ", "")
    text = _normalize_commas(text)
    return NUMBER_RE.sub(_canonical_number, text)


def values_multiset(options):
    """The sorted canonical values of an options dict, for multiset equality."""
    return sorted(canonical_value(v) for v in (options or {}).values())


def statement_numbers(text):
    """The Counter (multiset) of canonical numbers in a statement.

    If one single number changes, the multiset differs: similar is not same.
    """
    text = _normalize_commas(str(text or "").lower())
    return Counter(_canonical_number_str(m) for m in NUMBER_RE.findall(text))
