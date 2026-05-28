# framesmith/transforms/names.py
"""Atomic transforms for person-name normalization."""

import polars as pl

from framesmith._internal.regex_patterns import TRAILING_JR_PATTERN

__all__: list[str] = [
    'remove_jr_suffix',
]


def remove_jr_suffix(expr: pl.Expr) -> pl.Expr:
    """Remove a trailing ``jr`` or ``jr.`` suffix.

    Case-insensitive, with an optional comma and/or whitespace before
    the suffix. The pattern is end-anchored, so interior ``jr`` (e.g.
    ``"Jr Bakery"``) is not matched.

    Atomic: removes only the suffix. Does not collapse interior
    whitespace or strip ends. ``TRAILING_JR_PATTERN`` consumes the
    separator adjacent to the suffix, so normal input (``"John Smith
    Jr"``) comes out clean (``"John Smith"``) without further steps;
    pathological interior runs require :func:`collapse_whitespace` and
    :func:`strip_whitespace` for tidy-up.
    """
    return expr.str.replace(TRAILING_JR_PATTERN, '')
