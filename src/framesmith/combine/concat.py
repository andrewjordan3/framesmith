# framesmith/combine/concat.py
"""
Multi-column string combination.

``combine_columns`` builds a ``pl.Expr`` that concatenates several columns
row-wise into one, applied via ``df.with_columns(...)``. Like the rest of
framesmith it returns an expression and never mutates a frame; unlike a
single-column transform it references several source columns by name, so it
lives here rather than in ``framesmith.transforms``.
"""

from collections.abc import Sequence

import polars as pl

__all__: list[str] = ['combine_columns']


def combine_columns(
    column_names: Sequence[str],
    output_column_name: str,
    separator: str = ' ',
) -> pl.Expr:
    """Build an expression concatenating several columns into one.

    Concatenates the named columns row-wise with ``separator`` between
    values, skipping nulls so a missing value produces no stray
    separator. Apply via ``df.with_columns(combine_columns(...))``.

    Non-string columns are stringified by polars during concatenation
    (an ``Int64`` ``30`` becomes ``'30'``). A row in which every named
    column is null produces an empty string, not null — this is polars'
    native ``concat_str(ignore_nulls=True)`` behavior, deliberately not
    special-cased.

    Args:
        column_names: Names of the columns to concatenate, in order. Each
            is a column reference. Must be non-empty; a single name is a
            degenerate pass-through that renames that column to
            ``output_column_name``.
        output_column_name: Name of the resulting column. Required — no
            default — so the combined column is always deliberately named.
            If it matches an existing column, ``with_columns`` overwrites
            that column, per polars semantics.
        separator: String inserted between values. Default a single
            space. Empty and multi-character separators are both valid.

    Returns:
        A ``pl.Expr`` for ``df.with_columns(...)``.

    Raises:
        ValueError: If ``column_names`` is empty. An empty set has nothing
            to combine and is almost certainly a bug, so it is rejected
            loudly rather than producing a degenerate empty expression.

    Note:
        Column existence is not checked here — the builder has no frame. A
        name absent from the frame raises
        ``polars.exceptions.ColumnNotFoundError`` when the expression is
        applied.

    Example:
        >>> import polars as pl
        >>> from framesmith.combine import combine_columns
        >>> df = pl.DataFrame(
        ...     {'first': ['John', None], 'last': ['Doe', 'Smith']}
        ... )
        >>> df.with_columns(
        ...     combine_columns(['first', 'last'], 'full_name')
        ... )['full_name'].to_list()
        ['John Doe', 'Smith']
    """
    if len(column_names) == 0:
        raise ValueError('column_names must not be empty')

    return pl.concat_str(
        [pl.col(name) for name in column_names],
        separator=separator,
        ignore_nulls=True,
    ).alias(output_column_name)
