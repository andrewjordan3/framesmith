# framesmith/transforms/addresses.py
"""
Atomic US address-field transforms.

``standardize_state`` canonicalizes a whole-value state column to its postal
code; ``strip_trailing_state`` removes a trailing postal code from a location
string. They target different column shapes — a state column vs a location
string — and share only the state reference table in ``framesmith._internal``;
they do not compose with each other.
"""

import polars as pl

from framesmith._internal import (
    TRAILING_STATE_CODE_PATTERN,
    US_STATE_STANDARDIZE_MAP,
)

__all__: list[str] = ['standardize_state', 'strip_trailing_state']


def standardize_state(expr: pl.Expr) -> pl.Expr:
    """Canonicalize a whole-value US state to its two-letter postal code.

    The cell's whole value is treated as a state: stripped and lowercased
    to form a lookup key, then mapped to the canonical uppercase code.
    Full names and postal codes are both recognized (``'Illinois'``,
    ``'illinois'``, ``'IL'``, ``'il'`` -> ``'IL'``); District of Columbia
    and the five inhabited territories are included. Values that do not
    match a known state pass through completely unchanged — original case
    and whitespace preserved, never nulled. Nulls pass through as null.

    Operates on a state column, not a location string: ``'Chicago, IL'``
    is not a recognized whole-value state and passes through untouched.
    Use :func:`strip_trailing_state` for location strings.

    AP-style abbreviations (``'Ill.'``, ``'Calif.'``) are not recognized
    in this version; they pass through unchanged.
    """
    lookup_key: pl.Expr = expr.str.strip_chars().str.to_lowercase()
    return (
        pl.when(lookup_key.is_in(list(US_STATE_STANDARDIZE_MAP)))
        .then(lookup_key.replace(US_STATE_STANDARDIZE_MAP))
        .otherwise(expr)
    )


def strip_trailing_state(expr: pl.Expr) -> pl.Expr:
    """Remove a trailing two-letter US state code from a location string.

    Matches a trailing postal code preceded by a comma and/or whitespace
    and removes both the separator and the code: ``'Chicago, IL'`` ->
    ``'Chicago'``, ``'Chicago IL'`` -> ``'Chicago'``, ``'Denver CO'`` ->
    ``'Denver'``. Case-insensitive on the code. The match is end-anchored,
    so an interior code and a bare code with no separator (``'IL'``) are
    left alone.

    Codes only, by design — a trailing full name (``'Chicago,
    Illinois'``) is not removed. A trailing-token canonicalizer that folds
    full names to codes in place is a future address tool; until then only
    codes are stripped. ``'dc'`` is intentionally not in the strip set, so
    ``'Washington, DC'`` is preserved — stripping it would collide with
    Washington state. Nulls pass through as null.
    """
    return expr.str.replace(TRAILING_STATE_CODE_PATTERN, '')
