# framesmith/filters/dates.py
"""
Date-based row-filter expression builders.

Functions in this module return ``pl.Expr`` boolean masks intended for
``df.filter(...)``. They never mutate frames themselves — the user
applies the mask. This parallels how column transforms return
expressions for ``df.with_columns(...)``.
"""

from typing import Literal

import polars as pl

__all__: list[str] = [
    'PeriodInterval',
    'within_complete_month',
    'within_complete_period',
]


# Polars duration strings accepted by ``Expr.dt.truncate``. Quarters
# (``'1q'``) and years (``'1y'``) are included as standard reporting
# buckets.
type PeriodInterval = Literal['1d', '1w', '1mo', '1q', '1y']


def within_complete_period(
    date_column: str,
    period: PeriodInterval,
    threshold_days: int,
) -> pl.Expr:
    """Boolean mask: True for rows in complete time periods.

    A trailing period is "incomplete" when the column's max date is
    more than ``threshold_days`` from that period's end. When the
    trailing period is incomplete, every row in it is excluded;
    otherwise all rows pass. Completeness is determined from the data
    (the column's max date), not from today's calendar date.

    Apply with ``df.filter(within_complete_period(...))``.

    Args:
        date_column: Name of the date or datetime column.
        period: Polars duration string identifying the period
            granularity. One of ``'1d'``, ``'1w'``, ``'1mo'``,
            ``'1q'``, ``'1y'``.
        threshold_days: Strict threshold. The trailing period is
            incomplete when ``period_end - max_date > threshold_days``.
            At exactly ``threshold_days``, the period is still
            complete and its rows are kept.

    Returns:
        A boolean ``pl.Expr`` to apply with ``df.filter``.

    Example:
        >>> import polars as pl
        >>> from datetime import date
        >>> from framesmith.filters import within_complete_period
        >>> df = pl.DataFrame({
        ...     'date': [date(2024, 1, 15), date(2024, 2, 14), date(2024, 3, 15)]
        ... })
        >>> # Last date is March 15; March is incomplete (16 days to month-end).
        >>> # With threshold=5, March is dropped entirely.
        >>> df.filter(
        ...     within_complete_period('date', period='1mo', threshold_days=5)
        ... )['date'].to_list()
        [datetime.date(2024, 1, 15), datetime.date(2024, 2, 14)]
    """
    # Period end via truncate-to-start + offset-by-period - 1 day. Polars
    # has no direct ``period_end`` accessor, so this is the portable form.
    max_date: pl.Expr = pl.col(date_column).max()
    period_end: pl.Expr = (
        max_date.dt.truncate(period).dt.offset_by(period).dt.offset_by('-1d')
    )
    days_until_period_end: pl.Expr = (period_end - max_date).dt.total_days()

    # When the trailing period is incomplete, exclude every row whose
    # period bucket matches the trailing one. When max_date is null
    # (empty frame, all-null column), the comparison is null and the
    # ``otherwise`` branch keeps every row.
    return (
        pl.when(days_until_period_end > threshold_days)
        .then(
            pl.col(date_column).dt.truncate(period)
            != max_date.dt.truncate(period)
        )
        .otherwise(pl.lit(True))
    )


def within_complete_month(
    date_column: str,
    threshold_days: int = 5,
) -> pl.Expr:
    """Convenience: ``within_complete_period`` with ``period='1mo'``.

    The five-day default threshold reflects the common reporting
    convention that a month is "settled enough" within five days of
    month-end.

    Args:
        date_column: Name of the date or datetime column.
        threshold_days: Strict threshold in days. Default 5.

    Returns:
        A boolean ``pl.Expr`` to apply with ``df.filter``.
    """
    return within_complete_period(
        date_column, period='1mo', threshold_days=threshold_days
    )
