# framesmith/schema/column_names.py
"""
Schema-level column-name normalization.

Functions here return a ``pl.Expr`` that, applied via ``df.select(...)``,
renames every column. They never take or mutate a frame — the user
applies the expression, parallel to how transforms feed
``with_columns`` and filters feed ``filter``.
"""

import re

import polars as pl

from framesmith._internal import WHITESPACE_RUN_PATTERN

__all__: list[str] = ['normalize_column_names']


def normalize_column_names(separator: str = '_') -> pl.Expr:
    """Build an expression that normalizes every column name.

    Each label is stripped of leading/trailing whitespace, lowercased,
    and has its internal whitespace runs replaced with ``separator``.
    Apply via ``df.select(normalize_column_names())`` — this renames the
    whole frame in place of the originals. ``with_columns`` would keep
    the originals and add renamed copies alongside them, so ``select``
    is required.

    Args:
        separator: Replacement for each whitespace run. Default ``'_'``.
            The empty string removes whitespace entirely; multi-character
            separators are valid.

    Returns:
        A ``pl.Expr`` selecting all columns with normalized names, for
        ``df.select(...)``.

    Application behavior:
        If two source labels normalize to the same name, applying the
        expression raises ``polars.exceptions.DuplicateError`` at select
        time. This is intentional: a collision means the source schema
        or the normalization is wrong, and a loud failure beats a
        silently suffixed ``customer__2`` column. This function itself
        does not raise — the error surfaces when the returned expression
        is applied, so there is no ``Raises`` section.

    Example:
        >>> import polars as pl
        >>> from framesmith.schema import normalize_column_names
        >>> df = pl.DataFrame({'  First Name ': [1], 'LAST  NAME': [2]})
        >>> df.select(normalize_column_names()).columns
        ['first_name', 'last_name']
    """
    # Compile once here, not per label: the returned expression's
    # name-map closure must not recompile on every column.
    compiled_whitespace_pattern: re.Pattern[str] = re.compile(
        WHITESPACE_RUN_PATTERN
    )

    def _normalize_label(column_label: str) -> str:
        stripped_and_lowercased: str = column_label.strip().lower()
        return compiled_whitespace_pattern.sub(
            separator, stripped_and_lowercased
        )

    return pl.all().name.map(_normalize_label)
