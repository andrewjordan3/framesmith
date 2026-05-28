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

from framesmith._internal import (
    ASCII_COMPAT_MAP,
    BLANK_OR_WHITESPACE_ONLY_PATTERN,
    WHITESPACE_RUN_PATTERN,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'collapse_whitespace',
    'fold_to_ascii',
    'normalize_unicode_nfkc',
    'nullify_blank_strings',
    'periods_to_spaces',
    'remove_apostrophes',
    'remove_periods',
    'replace_ampersand_with_and',
    'replace_whitespace_with',
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


def periods_to_spaces(expr: pl.Expr) -> pl.Expr:
    """Replace each period with a single space.

    Atomic: one period → one space, even in runs. ``'U.S.A'`` becomes
    ``'U S A'`` and ``'john..doe'`` becomes ``'john  doe'`` (two
    spaces). Compose with :func:`collapse_whitespace` downstream if
    you want repeated dots to collapse to a single space.

    Nulls pass through as null.
    """
    return expr.str.replace_all('.', ' ', literal=True)


def replace_whitespace_with(separator: str) -> ExpressionTransform:
    """Build a transform that replaces each whitespace run with ``separator``.

    Atomic: replaces whitespace runs only. Does NOT strip leading or
    trailing whitespace, and does NOT collapse non-whitespace
    characters — those are :func:`strip_whitespace` and
    :func:`collapse_whitespace`. A leading or trailing whitespace run
    becomes a separator just like an interior one, so
    ``' hello world '`` with ``separator='_'`` produces
    ``'_hello_world_'``. Compose with :func:`strip_whitespace` if you
    want the pandas-style behavior of stripping ends before replacing
    interior whitespace.

    Multi-character separators and the empty string are both valid.
    ``replace_whitespace_with('')`` removes all whitespace.

    Args:
        separator: The string each whitespace run is replaced with.
            No default — explicit at every call site.

    Returns:
        An ``ExpressionTransform``. Applied via ``compose_column``.

    Example:
        >>> import polars as pl
        >>> from framesmith import compose_column
        >>> from framesmith.transforms import replace_whitespace_with
        >>> df = pl.DataFrame({'x': ['hello world', 'a  b']})
        >>> kebab = replace_whitespace_with('-')
        >>> df.with_columns(compose_column('x', [kebab]))['x'].to_list()
        ['hello-world', 'a-b']
    """

    def _replace_whitespace(expr: pl.Expr) -> pl.Expr:
        return expr.str.replace_all(WHITESPACE_RUN_PATTERN, separator)

    return _replace_whitespace


# Build the snake_case transform once at module load. The factory is
# called exactly once; this constant holds the resulting closure.
_TO_SNAKE_CASE_TRANSFORM: ExpressionTransform = replace_whitespace_with('_')


def to_snake_case(expr: pl.Expr) -> pl.Expr:
    """Replace whitespace runs with underscores.

    Convenience wrapper that delegates to
    ``replace_whitespace_with('_')`` so the whitespace-replacement
    logic lives in exactly one place. Does not strip, lowercase, or
    Unicode-fold. Assumes the input is already normalized if
    snake_case canonical form is the goal.
    """
    return _TO_SNAKE_CASE_TRANSFORM(expr)
