# framesmith/transforms/addresses.py
"""
US address-field transforms.

``standardize_state`` canonicalizes a whole-value state column to its postal
code; ``standardize_state_name`` resolves the same inputs to the canonical
lowercase full name; ``strip_trailing_state`` removes a trailing postal code
from a location string. ``standardize_directionals`` and
``standardize_unit_markers`` rewrite word-bounded address tokens (``North`` →
``N``, ``Apartment`` → ``APT``) to their USPS abbreviations. All share the
reference data in ``framesmith._internal``; the state value-standardizers also
share the lookup-key normalization.
"""

import re
from collections.abc import Mapping, Sequence

import polars as pl

from framesmith._internal import (
    DEFAULT_DIRECTIONAL_MAP,
    DEFAULT_UNIT_MARKER_MAP,
    TRAILING_STATE_CODE_PATTERN,
    US_STATE_NAME_MAP,
    US_STATE_STANDARDIZE_MAP,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'DEFAULT_DIRECTIONAL_MAP',
    'DEFAULT_UNIT_MARKER_MAP',
    'standardize_directionals',
    'standardize_state',
    'standardize_state_name',
    'standardize_unit_markers',
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


def _build_token_standardizer(
    token_map: Mapping[str, Sequence[str]],
) -> ExpressionTransform:
    r"""Build a transform that rewrites word-bounded variants to canonicals.

    Compiles one case-insensitive, word-bounded pattern per canonical
    (consuming a trailing period) and chains a ``replace_all`` per
    canonical. ``\b`` makes the chain order-independent — a shorter
    variant cannot match inside a longer token. Patterns are built once,
    here; the returned closure only applies them.
    """
    compiled_patterns: list[tuple[str, str]] = []
    for canonical, variants in token_map.items():
        alternation: str = '|'.join(
            re.escape(variant)
            for variant in sorted(variants, key=len, reverse=True)
        )
        compiled_patterns.append((rf'(?i)\b(?:{alternation})\b\.?', canonical))

    def _standardize_tokens(expr: pl.Expr) -> pl.Expr:
        result: pl.Expr = expr
        for pattern, canonical in compiled_patterns:
            result = result.str.replace_all(pattern, canonical)
        return result

    return _standardize_tokens


def standardize_directionals(
    directional_map: Mapping[str, Sequence[str]] = DEFAULT_DIRECTIONAL_MAP,
) -> ExpressionTransform:
    """Standardize directional words to USPS abbreviations (``North`` → ``N``).

    Rewrites each whole-word directional — spelled-out or abbreviated, any
    case, optional trailing period — to its uppercase USPS form
    (``"123 North Main"`` → ``"123 N Main"``, ``"Northeast"`` → ``"NE"``).
    Compound directionals are handled correctly: ``"Northeast"`` does not
    become ``"N"`` + east.

    Token replacement, not address parsing: a street named ``"North
    Avenue"`` becomes ``"N AVE"``. Some canonicals coincide with state
    codes (``NE``, ``SE``…) but the output equals the code, so this only
    normalizes case. Nulls pass through.

    Args:
        directional_map: Canonical abbreviation → variant spellings
            (variants lowercase, disjoint across canonicals). Must be
            non-empty.

    Returns:
        An ``ExpressionTransform`` for ``compose_column``.

    Raises:
        ValueError: If ``directional_map`` is empty.
    """
    if len(directional_map) == 0:
        raise ValueError('directional_map must not be empty')
    return _build_token_standardizer(directional_map)


def standardize_unit_markers(
    unit_marker_map: Mapping[str, Sequence[str]] = DEFAULT_UNIT_MARKER_MAP,
) -> ExpressionTransform:
    """Standardize secondary unit designators to USPS abbreviations.

    Rewrites each whole-word unit marker — spelled-out or abbreviated, any
    case, optional trailing period — to its uppercase USPS form
    (``"Apartment 4"`` → ``"APT 4"``, ``"Suite"`` → ``"STE"``). Token
    replacement, not parsing. ``FL`` (floor) coincides with Florida's
    code; the output equals the code, so only case is normalized. Nulls
    pass through.

    Args:
        unit_marker_map: Canonical abbreviation → variant spellings. Must
            be non-empty.

    Returns:
        An ``ExpressionTransform`` for ``compose_column``.

    Raises:
        ValueError: If ``unit_marker_map`` is empty.
    """
    if len(unit_marker_map) == 0:
        raise ValueError('unit_marker_map must not be empty')
    return _build_token_standardizer(unit_marker_map)
