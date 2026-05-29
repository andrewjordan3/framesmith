# framesmith/group/resolve.py
"""
Deterministic group-value resolution.

Two resolvers build aggregation expressions for collapsing duplicate-key rows
into one. ``first_non_null_per_group`` populates each non-key field from the
first non-null value its group offers; ``mode_then_first_per_group`` uses the
most common non-null value instead. The caller supplies the frame method:

    df.group_by(key_columns).agg(first_non_null_per_group(key_columns))

framesmith returns the expression; the caller applies ``group_by().agg`` — the
same division of labor as filters (framesmith returns the mask, the caller
applies ``filter``).
"""

from collections.abc import Sequence

import polars as pl

__all__: list[str] = [
    'first_non_null_per_group',
    'mode_then_first_per_group',
]


def _require_key_columns(key_columns: Sequence[str]) -> None:
    """Reject an empty key set; grouping on no keys collapses the whole frame."""
    if len(key_columns) == 0:
        raise ValueError('key_columns must not be empty')


def first_non_null_per_group(key_columns: Sequence[str]) -> pl.Expr:
    """Build the agg expression resolving each non-key field per group.

    Designed for ``df.group_by(key_columns).agg(...)``. For every column
    other than the keys, takes the first non-null value within each group
    (in input row order), so a group whose data is split across rows
    yields one fully-populated row wherever any row had a value. A field
    with no non-null value anywhere in its group stays null.

    Guarantees:
        * One row per unique key combination (from the caller's
          ``group_by``).
        * Each non-key field is the first non-null value, in input row
          order, within its group — or null if the group has none.

    Does not guarantee output row or column order. Compare results with
    ``check_row_order=False`` / ``check_column_order=False``, or sort
    afterward if you need a fixed order.

    Pass the same ``key_columns`` to ``group_by`` and to this function;
    they are excluded from the aggregation because the group keys already
    appear in the output.

    Args:
        key_columns: The columns defining group identity — the same list
            passed to ``group_by``. Must be non-empty. Excluded from the
            resolved set.

    Returns:
        A ``pl.Expr`` for ``df.group_by(key_columns).agg(...)``.

    Raises:
        ValueError: If ``key_columns`` is empty. Grouping on no keys
            collapses the whole frame to one row and is almost certainly
            a bug.

    Note:
        Column existence is not checked here — the builder has no frame. A
        missing key raises ``polars.exceptions.ColumnNotFoundError`` from
        the caller's ``group_by`` when applied.

    Example:
        >>> import polars as pl
        >>> from framesmith.group import first_non_null_per_group
        >>> df = pl.DataFrame(
        ...     {'id': [1, 1], 'name': ['Alice', None], 'city': [None, 'NYC']}
        ... )
        >>> out = df.group_by(['id']).agg(first_non_null_per_group(['id']))
        >>> out.sort('id').to_dicts()
        [{'id': 1, 'name': 'Alice', 'city': 'NYC'}]
    """
    _require_key_columns(key_columns)

    return pl.all().exclude(list(key_columns)).drop_nulls().first()


def mode_then_first_per_group(key_columns: Sequence[str]) -> pl.Expr:
    """Build the agg expression resolving each non-key field to its group mode.

    Designed for ``df.group_by(key_columns).agg(...)``. For every column
    other than the keys, takes the most common non-null value in each
    group; on a tie, takes the first such value in input row order. A
    field with no non-null value anywhere in its group stays null.

    Every resolved value is one that actually appears in the group — this
    selects, it never computes a value, so unlike a median it cannot
    produce a number that was not in the data, and it works on string
    columns as well as numeric ones.

    Guarantees:
        * One row per unique key combination (from the caller's
          ``group_by``).
        * Each non-key field is the most common non-null value in its
          group; ties resolved by first in input order; null if the group
          has no non-null value.

    Does not guarantee output row or column order — compare results with
    ``check_row_order=False`` / ``check_column_order=False``.

    Pass the same ``key_columns`` to ``group_by`` and to this function;
    they are excluded from the aggregation because the group keys already
    appear in the output.

    Args:
        key_columns: The columns defining group identity — the same list
            passed to ``group_by``. Must be non-empty. Excluded from the
            resolved set.

    Returns:
        A ``pl.Expr`` for ``df.group_by(key_columns).agg(...)``.

    Raises:
        ValueError: If ``key_columns`` is empty.

    Note:
        Column existence is not checked here — the builder has no frame. A
        missing key raises ``polars.exceptions.ColumnNotFoundError`` from
        the caller's ``group_by`` when applied.

    Example:
        >>> import polars as pl
        >>> from framesmith.group import mode_then_first_per_group
        >>> df = pl.DataFrame(
        ...     {'id': [1, 1, 1], 'name': ['Al', 'Al', 'Bo']}
        ... )
        >>> df.group_by(['id']).agg(
        ...     mode_then_first_per_group(['id'])
        ... ).sort('id').to_dicts()
        [{'id': 1, 'name': 'Al'}]
    """
    _require_key_columns(key_columns)

    non_null_values: pl.Expr = pl.all().exclude(list(key_columns)).drop_nulls()
    return non_null_values.filter(
        non_null_values.is_in(non_null_values.mode().implode())
    ).first()
