# framesmith/transforms/names.py
"""Atomic transforms for person-name normalization."""

import polars as pl

from framesmith._internal import TRAILING_JR_PATTERN

__all__: list[str] = [
    'extract_email_local_part',
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


def extract_email_local_part(expr: pl.Expr) -> pl.Expr:
    """Take everything before the first ``'@'``; no ``'@'`` → unchanged.

    Splits the string on ``'@'`` and returns the first segment. A
    string with no ``'@'`` passes through as-is. A string starting
    with ``'@'`` yields the empty string. Strings with multiple
    ``'@'`` characters return only the part before the first one,
    matching the pandas reference behavior of
    ``str.split('@', n=1).str[0]``.

    Atomic: does NOT strip surrounding whitespace, lowercase, or
    further normalize. Compose with :func:`strip_whitespace` upstream
    if you need to handle whitespace-padded email strings, and with
    :func:`periods_to_spaces` downstream if you want display-name
    shape.

    Nulls pass through as null.
    """
    return expr.str.split('@').list.first()
