# framesmith/validate/nulls.py
"""Null-presence validation guard."""

import polars as pl

__all__: list[str] = ['assert_no_nulls']


def assert_no_nulls(df: pl.DataFrame, column: str) -> None:
    """Raise if ``column`` contains any null.

    A guard, not a transform: returns ``None`` when the column is fully
    populated, raises otherwise. Use it to fail loudly before a step that
    assumes a complete column (e.g. a dedup key).

    Args:
        df: The frame to inspect.
        column: Name of the column to check.

    Returns:
        None.

    Raises:
        ValueError: If ``column`` contains one or more nulls.

    Note:
        Materializes a single null-count aggregation; cheap even on a
        large frame. A missing column raises ``ColumnNotFoundError``.

    Example:
        >>> import polars as pl
        >>> from framesmith.validate import assert_no_nulls
        >>> assert_no_nulls(pl.DataFrame({'id': [1, 2, 3]}), 'id')
    """
    null_count: int = df.select(pl.col(column).null_count()).item()
    if null_count:
        raise ValueError(
            f'column {column!r} contains {null_count} null value(s)'
        )
