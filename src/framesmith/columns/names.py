# framesmith/columns/names.py
"""
Polars expression builders for person-name normalization.
"""

import polars as pl

from framesmith._internal.regex_patterns import (
    TRAILING_JR_PATTERN,
    WHITESPACE_RUN_PATTERN,
)

__all__: list[str] = [
    'remove_trailing_jr',
]


def remove_trailing_jr(column: str) -> pl.Expr:
    """Remove a trailing ``jr`` or ``jr.`` suffix from a name column.

    The match is case-insensitive and tolerates an optional comma
    and/or whitespace before the suffix. After removal, internal
    whitespace runs are collapsed to a single space and the result is
    stripped.

    Examples:
        ``"John Smith Jr"``   → ``"John Smith"``
        ``"John Smith, Jr."`` → ``"John Smith"``
        ``"John Smith jr"``   → ``"John Smith"``

    Interior ``jr`` is not matched (e.g. ``"Jr Bakery"`` is unchanged),
    since the pattern is anchored to the end of the string.

    Args:
        column: Name of the input column to transform.

    Returns:
        ``pl.Expr`` auto-aliased to ``column``. Nulls propagate
        unchanged.
    """
    return (
        pl.col(column)
        .str.replace(TRAILING_JR_PATTERN, '')
        .str.replace_all(WHITESPACE_RUN_PATTERN, ' ')
        .str.strip_chars()
        .alias(column)
    )
