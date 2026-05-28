# framesmith/transforms/numeric.py
"""
Atomic numeric-string transforms: each is a single ``pl.Expr -> pl.Expr``
step. Designed to clean messy numeric strings (currency, accounting
parens, mainframe trailing minus, thousands separators) before casting.

Unicode/minus/currency/invisible-character handling is intentionally
NOT here — the ``UNICODE_TO_ASCII`` recipe (NFKC + ``fold_to_ascii``)
already covers it via ``ASCII_COMPAT_MAP``. The transforms below
handle only the genuinely numeric-specific stages.

US conventions are assumed (comma thousands, period decimal). European
formats (``1.234,56``) are out of scope.
"""

import polars as pl

from framesmith._internal.regex_patterns import (
    PAREN_NEGATIVE_PATTERN,
    THOUSANDS_SEPARATOR_PATTERN,
    TRAILING_MINUS_PATTERN,
)

__all__: list[str] = [
    'accounting_parens_to_negative',
    'cast_to_float64',
    'remove_thousands_separators',
    'trailing_minus_to_prefix',
]


def accounting_parens_to_negative(expr: pl.Expr) -> pl.Expr:
    """Convert an accounting-style parenthesized value to a leading-minus form.

    Example: ``"(123.45)"`` → ``"-123.45"``.

    Atomic: rewrites only the parens-to-minus structure. Does NOT strip
    currency, commas, or whitespace. In isolation, ``"($1,234)"`` →
    ``"-$1,234"``; inside the ``NORMALIZE_NUMERIC`` recipe,
    ``fold_to_ascii`` has already removed the currency symbol before
    this runs. Non-parenthesized values and nulls pass through
    unchanged.
    """
    return expr.str.replace(PAREN_NEGATIVE_PATTERN, '-${1}')


def trailing_minus_to_prefix(expr: pl.Expr) -> pl.Expr:
    """Move a trailing minus to the front: ``"1,234.56-"`` → ``"-1,234.56"``.

    Only fires when the string ends with ``-`` and does NOT already
    start with ``-``, so a leading-minus value (``"-100"``) is never
    double-negated.

    Atomic: does not touch commas, currency, or whitespace. Nulls pass
    through unchanged.
    """
    return expr.str.replace(TRAILING_MINUS_PATTERN, '-${1}')


def remove_thousands_separators(expr: pl.Expr) -> pl.Expr:
    """Remove digit-group separators (commas and whitespace).

    Uses ``THOUSANDS_SEPARATOR_PATTERN`` as a regex character class, so
    ``"1,234.56"`` → ``"1234.56"`` and ``"1 234"`` → ``"1234"``. The
    decimal point and minus sign are preserved. Nulls pass through
    unchanged.
    """
    return expr.str.replace_all(THOUSANDS_SEPARATOR_PATTERN, '')


def cast_to_float64(expr: pl.Expr) -> pl.Expr:
    """Cast to ``pl.Float64`` with ``strict=False``.

    Unparseable strings (and anything that is not a clean numeric
    string) become null rather than raising — clean the string first
    with the other numeric transforms. This transform does NOT fill
    nulls; null handling is the caller's decision.
    """
    return expr.cast(pl.Float64, strict=False)
