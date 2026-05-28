# framesmith/transforms/text.py
"""
Atomic text transforms: each is a single ``pl.Expr -> pl.Expr`` step.

Transforms in this module follow the ``ExpressionTransform`` contract:
they accept an expression representing the current column state and
return the next state. They never call ``pl.col(...)`` and never call
``.alias(...)`` — composition and aliasing are
``framesmith.compose_column``'s responsibility.

Boundaries are intentional. ``collapse_whitespace`` does not strip;
``strip_whitespace`` does not collapse; ``normalize_unicode_nfkc`` does
not fold to ASCII. Compose them through a recipe (see
``framesmith.recipes``) when you need the combined behavior.
"""

import polars as pl

from framesmith._internal.regex_patterns import (
    BLANK_OR_WHITESPACE_ONLY_PATTERN,
    WHITESPACE_RUN_PATTERN,
)
from framesmith._internal.unicode_maps import ASCII_COMPAT_MAP

__all__: list[str] = [
    'collapse_whitespace',
    'fold_to_ascii',
    'normalize_unicode_nfkc',
    'nullify_blank_strings',
    'remove_apostrophes',
    'remove_periods',
    'replace_ampersand_with_and',
    'strip_whitespace',
    'to_snake_case',
]


def nullify_blank_strings(expr: pl.Expr) -> pl.Expr:
    """Coerce blank or whitespace-only strings to null.

    Uses ``BLANK_OR_WHITESPACE_ONLY_PATTERN``. Non-blank values pass
    through unchanged; existing nulls remain null.
    """
    return (
        pl.when(expr.str.contains(BLANK_OR_WHITESPACE_ONLY_PATTERN))
        .then(None)
        .otherwise(expr)
    )


def normalize_unicode_nfkc(expr: pl.Expr) -> pl.Expr:
    """Apply NFKC Unicode normalization.

    Canonical compatibility normalization: fullwidth → ASCII, ligature
    decomposition, ``™`` → ``TM``, etc. Usually paired with
    :func:`fold_to_ascii` via the ``UNICODE_TO_ASCII`` recipe, but
    useful standalone when you want canonical Unicode without the
    opinionated ASCII substitution that ``fold_to_ascii`` applies.
    """
    return expr.str.normalize('NFKC')


def fold_to_ascii(expr: pl.Expr) -> pl.Expr:
    """Fold Unicode compatibility characters to ASCII via ``ASCII_COMPAT_MAP``.

    Covers smart quotes, em-dashes, currency symbols, trademark and
    registered symbols, non-standard whitespace, and more. Typically
    preceded by :func:`normalize_unicode_nfkc`: NFKC decomposes many
    compatibility forms before this map runs, which is exactly what the
    ``UNICODE_TO_ASCII`` recipe does. Used standalone when you want the
    ASCII substitution without NFKC's canonicalization.
    """
    return expr.str.replace_many(ASCII_COMPAT_MAP)


def collapse_whitespace(expr: pl.Expr) -> pl.Expr:
    """Collapse runs of whitespace to a single space.

    Does not strip leading or trailing whitespace — that is
    :func:`strip_whitespace`'s responsibility.
    """
    return expr.str.replace_all(WHITESPACE_RUN_PATTERN, ' ')


def strip_whitespace(expr: pl.Expr) -> pl.Expr:
    """Strip leading and trailing whitespace.

    Does not touch interior whitespace runs — that is
    :func:`collapse_whitespace`'s responsibility.
    """
    return expr.str.strip_chars()


def replace_ampersand_with_and(expr: pl.Expr) -> pl.Expr:
    """Replace literal ``&`` with ``and``."""
    return expr.str.replace_all('&', 'and', literal=True)


def remove_apostrophes(expr: pl.Expr) -> pl.Expr:
    """Remove all apostrophe characters (``'``)."""
    return expr.str.replace_all("'", '', literal=True)


def remove_periods(expr: pl.Expr) -> pl.Expr:
    """Remove all period characters (``.``)."""
    return expr.str.replace_all('.', '', literal=True)


def to_snake_case(expr: pl.Expr) -> pl.Expr:
    """Replace whitespace runs with underscores.

    Does not strip, lowercase, or Unicode-fold. Assumes the input is
    already normalized if snake_case canonical form is the goal.
    """
    return expr.str.replace_all(WHITESPACE_RUN_PATTERN, '_')
