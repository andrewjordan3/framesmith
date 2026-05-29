# framesmith/transforms/addresses.py
"""
Atomic US address-field transforms.

``standardize_state`` canonicalizes a whole-value state column to its postal
code; ``standardize_state_name`` resolves the same inputs to the canonical
lowercase full name; ``strip_trailing_state`` removes a trailing postal code
from a location string. The first two share the state reference table in
``framesmith._internal`` and the lookup-key normalization; ``strip_trailing_state``
operates on a location string and is codes-only. None compose with each other.
"""

import polars as pl

from framesmith._internal import (
    TRAILING_STATE_CODE_PATTERN,
    US_STATE_NAME_MAP,
    US_STATE_STANDARDIZE_MAP,
)

__all__: list[str] = [
    'standardize_state',
    'standardize_state_name',
    'strip_trailing_state',
]


def _normalized_state_key(expr: pl.Expr) -> pl.Expr:
    """Normalize a value into a state-lookup key: trim, lowercase, drop periods.

    Internal comparison key only — never mutates the returned data.
    Dropping periods lets dotted abbreviations ('Calif.', 'W.Va.') match
    the period-free keys in the state maps.
    """
    return (
        expr.str.strip_chars()
        .str.to_lowercase()
        .str.replace_all('.', '', literal=True)
    )


def standardize_state(expr: pl.Expr) -> pl.Expr:
    """Canonicalize a whole-value US state to its two-letter postal code.

    The cell's whole value is treated as a state: trimmed, lowercased, and
    stripped of periods to form a lookup key, then mapped to the canonical
    uppercase code. Full names, postal codes, and common abbreviations are
    recognized — AP-style and widely-used variants ('Calif.', 'Ill.',
    'Mass.', 'Penn.', 'Tex.'), including dotted forms (periods are ignored
    when matching). District of Columbia and the five inhabited
    territories are included. Values that do not match a known state pass
    through completely unchanged — original case and whitespace preserved,
    never nulled. Nulls pass through as null.

    Operates on a state column, not a location string: 'Chicago, IL' is not
    a recognized whole-value state and passes through untouched. Use
    :func:`strip_trailing_state` for location strings.
    """
    lookup_key: pl.Expr = _normalized_state_key(expr)
    return (
        pl.when(lookup_key.is_in(list(US_STATE_STANDARDIZE_MAP)))
        .then(lookup_key.replace(US_STATE_STANDARDIZE_MAP))
        .otherwise(expr)
    )


def standardize_state_name(expr: pl.Expr) -> pl.Expr:
    """Canonicalize a whole-value US state to its lowercase full name.

    Parallel to :func:`standardize_state`, but resolves to the canonical
    lowercase name ('il' / 'IL' / 'Illinois' / 'calif' / 'Calif.' ->
    'illinois' / 'california'). Same recognition set — full names, codes,
    and common abbreviations, periods ignored. Values that do not match
    pass through unchanged; nulls pass through as null.

    Output is the lowercase canonical name, not display-cased. Compose
    :func:`framesmith.transforms.to_titlecase` for display form — note
    titlecase mangles multi-word names ('district of columbia' ->
    'District Of Columbia'), which an :func:`apply_replacements` fix-up
    can correct downstream.
    """
    lookup_key: pl.Expr = _normalized_state_key(expr)
    return (
        pl.when(lookup_key.is_in(list(US_STATE_NAME_MAP)))
        .then(lookup_key.replace(US_STATE_NAME_MAP))
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
