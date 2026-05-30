# framesmith/transforms/dates.py
"""
Date/time transforms: parse numeric representations into proper datetimes.

These convert a numeric column (Unix epoch, Excel serial) into a naive
``Datetime``. They are expression-returning transforms, usable in
``compose_column`` and recipes. Timezone handling is a separate concern.
"""

from datetime import date, datetime
from typing import Literal

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'EpochTimeUnit',
    'flag_dates_outside_range',
    'normalize_epoch_timestamps',
    'normalize_excel_serial_dates',
]


type EpochTimeUnit = Literal['s', 'ms', 'us']
"""Unix epoch resolution: seconds, milliseconds, or microseconds."""


# Excel's 1900 date system counts days from this base. The fractional part of
# a serial is the time of day. Microseconds per day, for the duration math.
_EXCEL_EPOCH: datetime = datetime(1899, 12, 30)  # noqa: DTZ001
_MICROSECONDS_PER_DAY: int = 86_400_000_000


def normalize_epoch_timestamps(time_unit: EpochTimeUnit) -> ExpressionTransform:
    """Build a transform parsing a Unix-epoch column into a naive datetime.

    Interprets each numeric value as time since the Unix epoch
    (1970-01-01 UTC) in ``time_unit``, producing a naive ``Datetime`` —
    the UTC instant as wall-clock, with no timezone attached. Nulls pass
    through unchanged.

    The unit is required and explicit: the same integer is a different
    instant in seconds versus milliseconds, and there is no reliable way
    to infer it, so it must be stated.

    Args:
        time_unit: Epoch resolution — ``'s'``, ``'ms'``, or ``'us'``.

    Returns:
        An ``ExpressionTransform`` applied via ``compose_column``.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import normalize_epoch_timestamps
        >>> df = pl.DataFrame({'t': [1672531200]})
        >>> df.with_columns(
        ...     fs.compose_column('t', [normalize_epoch_timestamps('s')])
        ... )['t'].to_list()
        [datetime.datetime(2023, 1, 1, 0, 0)]
    """
    def _normalize_epoch_timestamps(expr: pl.Expr) -> pl.Expr:
        return pl.from_epoch(expr, time_unit=time_unit)

    return _normalize_epoch_timestamps


def normalize_excel_serial_dates(expr: pl.Expr) -> pl.Expr:
    """Convert an Excel serial-date column into a naive datetime.

    Interprets each numeric value as an Excel serial date in the 1900
    date system: whole days since 1899-12-30, with the fractional part as
    time of day (``44927`` → ``2023-01-01``, ``44927.5`` → ``2023-01-01
    12:00``). Output is a naive ``Datetime``; nulls pass through
    unchanged.

    Caveat: Excel's 1900 leap-year bug means serials for dates before
    1900-03-01 resolve one day off. This affects only January and
    February 1900, which real data does not contain; it is left as-is
    rather than special-cased.
    """
    total_microseconds: pl.Expr = (
        (expr.cast(pl.Float64) * _MICROSECONDS_PER_DAY).round().cast(pl.Int64)
    )
    return pl.lit(_EXCEL_EPOCH) + pl.duration(microseconds=total_microseconds)


def flag_dates_outside_range(
    lower: date | None = None,
    upper: date | None = None,
) -> ExpressionTransform:
    """Build a transform flagging dates outside an inclusive ``[lower, upper]``.

    The returned transform maps a date/datetime column to a boolean: true
    where the value is below ``lower`` or above ``upper``, false where it
    is within the bounds, null where the input is null. Apply via
    ``compose_column`` to build a flag column, or use the expression
    directly to filter (``df.filter(~...)``).

    At least one bound is required. With both, it flags impossible /
    out-of-domain dates; with only ``upper``, future dates; with only
    ``lower``, stale dates.

    Bounds are inclusive: a value exactly equal to ``lower`` or ``upper``
    is in range and is not flagged. A null input yields a null flag —
    missingness is not treated as out-of-range. When filtering on the
    result, note that null flags are excluded by ``df.filter``, so callers
    who want to keep null-dated rows should handle them explicitly.

    Args:
        lower: Earliest in-range value (inclusive). A ``date`` or
            ``datetime``; compares correctly against either column dtype.
            ``None`` disables the lower check.
        upper: Latest in-range value (inclusive), same typing. ``None``
            disables the upper check.

    Returns:
        An ``ExpressionTransform`` (true = out of range) for
        ``compose_column``.

    Raises:
        ValueError: If both ``lower`` and ``upper`` are ``None``.

    Example:
        >>> import polars as pl
        >>> from datetime import datetime
        >>> import framesmith as fs
        >>> from framesmith.transforms import flag_dates_outside_range
        >>> df = pl.DataFrame({'d': [datetime(1990, 1, 1), datetime(2030, 1, 1)]})
        >>> df.with_columns(
        ...     fs.compose_column(
        ...         'd',
        ...         [flag_dates_outside_range(
        ...             lower=datetime(2000, 1, 1), upper=datetime(2026, 1, 1)
        ...         )],
        ...         output_column_name='d_out_of_range',
        ...     )
        ... )['d_out_of_range'].to_list()
        [True, True]
    """
    if lower is None and upper is None:
        raise ValueError(
            'flag_dates_outside_range requires at least one of lower or upper.'
        )

    def _flag_dates_outside_range(expr: pl.Expr) -> pl.Expr:
        # The factory guarantees at least one bound, so conditions is non-empty.
        conditions: list[pl.Expr] = []
        if lower is not None:
            conditions.append(expr < pl.lit(lower))
        if upper is not None:
            conditions.append(expr > pl.lit(upper))
        combined: pl.Expr = conditions[0]
        for condition in conditions[1:]:
            combined = combined | condition
        return combined

    return _flag_dates_outside_range
