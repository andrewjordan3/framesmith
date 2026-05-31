# framesmith/transforms/whitespace.py
"""Whitespace transforms: collapse interior runs, strip ends, replace runs,
and nullify blank or whitespace-only values.

Boundaries are intentional and atomic — ``collapse_whitespace`` does not
strip; ``strip_whitespace`` does not collapse. Compose them through a recipe
when you need the combined behavior. All follow the ``ExpressionTransform``
contract: they never call ``pl.col(...)`` or ``.alias(...)``.
"""

import polars as pl

from framesmith._internal import (
    BLANK_OR_WHITESPACE_ONLY_PATTERN,
    WHITESPACE_RUN_PATTERN,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'collapse_whitespace',
    'nullify_blank_strings',
    'replace_whitespace_with',
    'strip_whitespace',
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
