# framesmith/combine/coalesce.py
"""
Blank-aware multi-column coalesce.

``coalesce_blank_columns`` returns the first non-blank value across several
columns, where "blank" means null or whitespace-only. It builds a ``pl.Expr``
applied via ``df.with_columns(...)`` and, like ``combine_columns``, references
several source columns by name.

Unlike ``pl.coalesce`` (which falls back only on null), this also falls back
past whitespace-only values. It does so by reusing ``nullify_blank_strings``,
so the definition of "blank" has a single source of truth.
"""

from collections.abc import Sequence

import polars as pl

from framesmith.transforms import nullify_blank_strings

__all__: list[str] = ['coalesce_blank_columns']


def coalesce_blank_columns(
    column_names: Sequence[str],
    output_column_name: str,
) -> pl.Expr:
    """Build an expression taking the first non-blank value across columns.

    For each named column a value is "blank" if it is null or strips to
    the empty string. The result is the first non-blank value, scanning
    the columns in order. Apply via
    ``df.with_columns(coalesce_blank_columns(...))``.

    A kept value is returned raw — surrounding whitespace preserved
    (``'  John  '`` stays ``'  John  '``); the blank test is
    comparison-only. If every named column is blank, the result is null.

    Operates on string columns. Non-string columns raise a polars error
    when the expression is applied — for plain null-coalescing of
    non-string columns, use ``pl.coalesce`` directly.

    Args:
        column_names: Names of the candidate columns, in priority order.
            Each is a column reference. Must be non-empty; a single name
            is a degenerate case equivalent to ``nullify_blank_strings``
            on that column, renamed to ``output_column_name``.
        output_column_name: Name of the resulting column. Required — no
            default. If it matches an existing column, ``with_columns``
            overwrites that column, per polars semantics.

    Returns:
        A ``pl.Expr`` for ``df.with_columns(...)``.

    Raises:
        ValueError: If ``column_names`` is empty. Nothing to coalesce; a
            near-certain bug, rejected loudly.

    Note:
        Column existence is not checked here — the builder has no frame. A
        name absent from the frame raises
        ``polars.exceptions.ColumnNotFoundError`` when applied.

    Example:
        >>> import polars as pl
        >>> from framesmith.combine import coalesce_blank_columns
        >>> df = pl.DataFrame(
        ...     {'preferred': ['Jo', None, '  '], 'legal': ['X', 'Y', 'Z']}
        ... )
        >>> df.with_columns(
        ...     coalesce_blank_columns(['preferred', 'legal'], 'name')
        ... )['name'].to_list()
        ['Jo', 'Y', 'Z']
    """
    if len(column_names) == 0:
        raise ValueError('column_names must not be empty')

    blanks_nulled: list[pl.Expr] = [
        nullify_blank_strings(pl.col(name)) for name in column_names
    ]
    return pl.coalesce(blanks_nulled).alias(output_column_name)
