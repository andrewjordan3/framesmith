# framesmith/group/fill.py
"""
Group-wise null filling.

``fill_null_numeric_by_group`` fills a numeric column's nulls with a per-group
central-tendency statistic (mean or median, chosen by skew or forced).
``fill_null_string_by_group`` fills with the per-group mode. Both return a
``pl.Expr`` applied via ``df.with_columns(...)`` — they fill in place and
preserve every row, unlike the aggregating resolvers in ``group/resolve.py``.
"""

from collections.abc import Sequence
from typing import Literal

import polars as pl

from framesmith.group.mode import first_mode_in_order

__all__: list[str] = [
    'CentralTendencyStrategy',
    'fill_null_numeric_by_group',
    'fill_null_string_by_group',
]


type CentralTendencyStrategy = Literal['auto', 'mean', 'median']
"""Numeric group-fill strategy: skew-chosen (``'auto'``), or forced mean/median."""


def _require_group_columns(group_columns: Sequence[str]) -> None:
    """Reject an empty group-key set; grouping on no keys is almost always a bug."""
    if len(group_columns) == 0:
        raise ValueError('group_columns must not be empty')


def fill_null_numeric_by_group(
    value_column: str,
    group_columns: Sequence[str],
    *,
    strategy: CentralTendencyStrategy = 'auto',
    skew_threshold: float = 1.0,
) -> pl.Expr:
    """Build an expression filling a numeric column's nulls with a per-group center.

    For ``df.with_columns(...)``. Each null is filled with a
    central-tendency statistic computed within its group; non-null values
    and the row count are preserved. ``strategy`` selects the statistic:

    * ``'auto'`` (default): the column's overall skew decides. If
      ``abs(skew) >= skew_threshold`` the per-group **median** is used
      (robust to the skew); otherwise the per-group **mean**. Skew is
      measured once on the whole column — a column-level property — and
      the chosen statistic is then computed per group. A constant,
      single-value, or all-null column has undefined skew (NaN or null)
      and falls back to the mean.
    * ``'mean'`` / ``'median'``: force that per-group statistic; skew is
      not consulted and ``skew_threshold`` is ignored.

    A group with no non-null value has a null statistic, so its nulls stay
    null — the function never borrows another group's value.

    The column is cast to ``Float64`` (strict), so the output is always
    ``Float64`` and a non-numeric column raises rather than silently
    yielding null statistics. A null fills as null; integer columns widen
    to float (mean/median are fractional in general).

    Args:
        value_column: Name of the numeric column to fill.
        group_columns: Columns defining group identity, passed to
            ``.over(...)``. Must be non-empty.
        strategy: ``'auto'``, ``'mean'``, or ``'median'``. Default
            ``'auto'``.
        skew_threshold: For ``'auto'`` only — the ``abs(skew)`` cutoff at
            or above which the median is used. Must be > 0. Default
            ``1.0`` (switch to median only on clear skew).

    Returns:
        A ``pl.Expr`` for ``df.with_columns(...)``, producing a ``Float64``
        column.

    Raises:
        ValueError: If ``group_columns`` is empty, ``strategy`` is not a
            recognized value, or (auto only) ``skew_threshold`` is not > 0.

    Note:
        Column existence and dtype are not checked here — the builder has
        no frame. A missing column raises ``ColumnNotFoundError``; a
        non-numeric column raises ``InvalidOperationError`` (from the
        strict ``Float64`` cast); both surface from ``with_columns`` at
        apply time. A ``String`` column holding numeric text (``"10"``) is
        accepted — the cast succeeds; compose ``cast_to_float64`` upstream
        to control that conversion.

    Example:
        >>> import polars as pl
        >>> from framesmith.group import fill_null_numeric_by_group
        >>> df = pl.DataFrame(
        ...     {'region': [1, 1, 2, 2], 'amount': [10.0, None, 5.0, None]}
        ... )
        >>> df.with_columns(
        ...     fill_null_numeric_by_group('amount', ['region'])
        ... )['amount'].to_list()
        [10.0, 10.0, 5.0, 5.0]
    """
    _require_group_columns(group_columns)

    # Numeric contract: a strict Float64 cast makes a non-numeric column
    # raise here. Without it, mean()/median().over() silently return null
    # on a string column and the fill becomes a silent no-op.
    numeric_column: pl.Expr = pl.col(value_column).cast(pl.Float64, strict=True)
    group_mean: pl.Expr = numeric_column.mean().over(group_columns)
    group_median: pl.Expr = numeric_column.median().over(group_columns)

    match strategy:
        case 'mean':
            fill_value: pl.Expr = group_mean
        case 'median':
            fill_value = group_median
        case 'auto':
            if skew_threshold <= 0:
                raise ValueError('skew_threshold must be > 0 for auto strategy')
            column_skew: pl.Expr = numeric_column.skew()
            # Constant / single-value / all-null columns give NaN or null
            # skew; treat those as "not skewed" and use the mean.
            is_degenerate: pl.Expr = column_skew.is_null() | column_skew.is_nan()
            use_median: pl.Expr = is_degenerate.not_() & (
                column_skew.abs() >= skew_threshold
            )
            fill_value = pl.when(use_median).then(group_median).otherwise(
                group_mean
            )
        case _:
            raise ValueError(
                f"strategy must be 'auto', 'mean', or 'median'; got {strategy!r}"
            )

    return numeric_column.fill_null(fill_value)


def fill_null_string_by_group(
    value_column: str,
    group_columns: Sequence[str],
) -> pl.Expr:
    """Build an expression filling a column's nulls with the per-group mode.

    For ``df.with_columns(...)``. Each null is filled with the most common
    non-null value in its group, ties broken by first appearance in input
    order (via :func:`framesmith.group.mode.first_mode_in_order`). Non-null
    values and the row count are preserved. A group with no non-null value
    keeps its nulls — no cross-group borrowing.

    Unlike :func:`fill_null_numeric_by_group`, this does not reject a
    numeric column: ``mode`` is defined for numeric data, so a numeric
    column is filled with its group mode rather than raising. Use the
    numeric function for numeric columns; this is intended for
    categorical / string data.

    Args:
        value_column: Name of the column to fill.
        group_columns: Columns defining group identity, passed to
            ``.over(...)``. Must be non-empty.

    Returns:
        A ``pl.Expr`` for ``df.with_columns(...)``; the output dtype
        matches the input column.

    Raises:
        ValueError: If ``group_columns`` is empty.

    Note:
        Column existence is not checked here — a missing column raises
        ``ColumnNotFoundError`` from ``with_columns`` at apply time.

    Example:
        >>> import polars as pl
        >>> from framesmith.group import fill_null_string_by_group
        >>> df = pl.DataFrame({'g': [1, 1, 1], 'c': ['a', 'a', None]})
        >>> df.with_columns(
        ...     fill_null_string_by_group('c', ['g'])
        ... )['c'].to_list()
        ['a', 'a', 'a']
    """
    _require_group_columns(group_columns)

    return pl.col(value_column).fill_null(
        first_mode_in_order(pl.col(value_column)).over(group_columns)
    )
