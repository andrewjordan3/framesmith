# framesmith/transforms/case.py
"""Case transforms: lowercase, title case, and snake_case.

All follow the ``ExpressionTransform`` contract. ``to_snake_case`` composes
``to_lowercase`` with the underscore form of ``replace_whitespace_with`` (from
``framesmith.transforms.whitespace``), so the whitespace-replacement logic
lives in exactly one place.
"""

import polars as pl

from framesmith.transforms.whitespace import replace_whitespace_with
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'to_lowercase',
    'to_snake_case',
    'to_titlecase',
]


# Build the snake_case transform once at module load. The factory is called
# exactly once; this constant holds the resulting closure.
_TO_SNAKE_CASE_TRANSFORM: ExpressionTransform = replace_whitespace_with('_')


def to_lowercase(expr: pl.Expr) -> pl.Expr:
    """Lowercase all characters.

    Atomic: casing only. Does not strip, collapse, or Unicode-fold.
    Nulls pass through unchanged.
    """
    return expr.str.to_lowercase()


def to_titlecase(expr: pl.Expr) -> pl.Expr:
    """Title-case the string: first letter of each word upper, rest lower.

    Atomic: casing only. ``'john smith'`` → ``'John Smith'``. Does not
    strip, collapse, or replace separators.

    Known limit: title casing lowercases the tail of every word, so it
    mangles acronyms — ``'rep lob'`` → ``'Rep Lob'``, not ``'Rep LOB'``.
    Fix specific tokens afterward with :func:`apply_replacements`
    (e.g. ``{'Lob': 'LOB'}``). Nulls pass through unchanged.
    """
    return expr.str.to_titlecase()


def to_snake_case(expr: pl.Expr) -> pl.Expr:
    """Lowercase, then replace whitespace runs with underscores.

    True snake_case: ``'Hello World'`` → ``'hello_world'``. Composes
    :func:`to_lowercase` with the underscore form of
    :func:`replace_whitespace_with`, so the whitespace-replacement logic
    lives in exactly one place. Does not strip leading/trailing
    whitespace or Unicode-fold; compose :func:`strip_whitespace` and the
    ``UNICODE_TO_ASCII`` recipe upstream if you need canonical input.
    Nulls pass through unchanged.
    """
    return _TO_SNAKE_CASE_TRANSFORM(to_lowercase(expr))
