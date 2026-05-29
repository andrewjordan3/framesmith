# framesmith/transforms/dates.py
"""
Date/time transforms: parse numeric representations into proper datetimes.

These convert a numeric column (Unix epoch, Excel serial) into a naive
``Datetime``. They are expression-returning transforms, usable in
``compose_column`` and recipes. Timezone handling is a separate concern.
"""

from datetime import datetime
from typing import Literal

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'EpochTimeUnit',
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
