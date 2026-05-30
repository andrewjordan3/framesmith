# tests/transforms/test_dates.py
"""Tests for the numeric → datetime transforms in
``framesmith.transforms.dates``.

Each transform is exercised through ``compose_column`` so the tests also
cover the integration point. Imports go through the subpackage's public
surface (``from framesmith.transforms import ...``).
"""

from datetime import date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    EpochTimeUnit,
    flag_dates_outside_range,
    normalize_epoch_timestamps,
    normalize_excel_serial_dates,
)


def _apply(
    values: list[float | int | None],
    transform: ExpressionTransform,
    dtype: pl.DataType,
) -> pl.Series:
    """Run a single transform on a 1-column numeric frame.

    An explicit schema lets all-null inputs through without dtype
    inference failing.
    """
    df = pl.DataFrame({'x': values}, schema={'x': dtype})
    return df.with_columns(compose_column('x', [transform]))['x']


class TestNormalizeEpochTimestamps:
    @pytest.mark.parametrize(
        ('time_unit', 'value'),
        [
            ('s', 1672531200),
            ('ms', 1672531200000),
            ('us', 1672531200000000),
        ],
    )
    def test_parses_epoch_in_unit(
        self, time_unit: EpochTimeUnit, value: int
    ) -> None:
        result = _apply(
            [value], normalize_epoch_timestamps(time_unit), pl.Int64
        )
        assert result.to_list() == [datetime(2023, 1, 1)]  # noqa: DTZ001

    def test_null_propagates(self) -> None:
        result = _apply([None], normalize_epoch_timestamps('s'), pl.Int64)
        assert result.to_list() == [None]

    def test_output_dtype_is_naive_datetime(self) -> None:
        result = _apply(
            [1672531200], normalize_epoch_timestamps('s'), pl.Int64
        )
        assert isinstance(result.dtype, pl.Datetime)
        assert result.dtype.time_zone is None

    def test_factory_returns_callable(self) -> None:
        assert callable(normalize_epoch_timestamps('s'))

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': [1672531200, 0, None]}, schema={'x': pl.Int64}
        )
        expr = compose_column('x', [normalize_epoch_timestamps('s')])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestNormalizeExcelSerialDates:
    def test_whole_serial_is_midnight(self) -> None:
        result = _apply([44927.0], normalize_excel_serial_dates, pl.Float64)
        assert result.to_list() == [datetime(2023, 1, 1, 0, 0)]  # noqa: DTZ001

    def test_fractional_serial_is_time_of_day(self) -> None:
        result = _apply([44927.5], normalize_excel_serial_dates, pl.Float64)
        assert result.to_list() == [datetime(2023, 1, 1, 12, 0)]  # noqa: DTZ001

    def test_integer_input_works(self) -> None:
        # Cast to float internally, so an integer serial parses too.
        result = _apply([44927], normalize_excel_serial_dates, pl.Int64)
        assert result.to_list() == [datetime(2023, 1, 1)]  # noqa: DTZ001

    def test_null_propagates(self) -> None:
        result = _apply([None], normalize_excel_serial_dates, pl.Float64)
        assert result.to_list() == [None]

    def test_output_dtype_is_naive_datetime(self) -> None:
        result = _apply([44927.0], normalize_excel_serial_dates, pl.Float64)
        assert isinstance(result.dtype, pl.Datetime)
        assert result.dtype.time_zone is None

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': [44927.0, 44927.5, None]}, schema={'x': pl.Float64}
        )
        expr = compose_column('x', [normalize_excel_serial_dates])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# Bounds for the out-of-range tests: the in-range window is [2000, 2026].
_LOWER = datetime(2000, 1, 1)  # noqa: DTZ001
_UPPER = datetime(2026, 1, 1)  # noqa: DTZ001


def _flag(
    values: list[datetime | None], transform: ExpressionTransform
) -> list[bool | None]:
    """Apply a flag transform to a datetime column, returning the flags."""
    df = pl.DataFrame({'d': values}, schema={'d': pl.Datetime('us')})
    return df.with_columns(
        compose_column('d', [transform], output_column_name='flag')
    )['flag'].to_list()


class TestFlagDatesOutsideRange:
    def test_both_bounds(self) -> None:
        transform = flag_dates_outside_range(lower=_LOWER, upper=_UPPER)
        flags = _flag(
            [
                datetime(2010, 6, 1),  # noqa: DTZ001  in range
                datetime(1990, 1, 1),  # noqa: DTZ001  below lower
                datetime(2030, 1, 1),  # noqa: DTZ001  above upper
                None,
            ],
            transform,
        )
        assert flags == [False, True, True, None]

    def test_boundaries_are_inclusive(self) -> None:
        transform = flag_dates_outside_range(lower=_LOWER, upper=_UPPER)
        flags = _flag([_LOWER, _UPPER], transform)
        assert flags == [False, False]

    def test_upper_only_flags_future(self) -> None:
        transform = flag_dates_outside_range(upper=_UPPER)
        flags = _flag(
            [
                datetime(2030, 1, 1),  # noqa: DTZ001  future
                datetime(2010, 1, 1),  # noqa: DTZ001  in range
                datetime(1990, 1, 1),  # noqa: DTZ001  past, still in range
                None,
            ],
            transform,
        )
        assert flags == [True, False, False, None]

    def test_lower_only_flags_stale(self) -> None:
        transform = flag_dates_outside_range(lower=_LOWER)
        flags = _flag(
            [
                datetime(1990, 1, 1),  # noqa: DTZ001  stale
                datetime(2010, 1, 1),  # noqa: DTZ001  in range
                datetime(2030, 1, 1),  # noqa: DTZ001  future, still in range
                None,
            ],
            transform,
        )
        assert flags == [True, False, False, None]

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match='at least one'):
            flag_dates_outside_range()

    @pytest.mark.parametrize('dtype', [pl.Date, pl.Datetime('us')])
    def test_accepts_date_and_datetime_columns(
        self, dtype: pl.DataType
    ) -> None:
        # datetime bounds compare correctly against a Date column and a
        # naive Datetime column alike.
        if dtype == pl.Date:
            values = [date(1990, 1, 1), date(2010, 1, 1), date(2030, 1, 1)]
        else:
            values = [
                datetime(1990, 1, 1),  # noqa: DTZ001
                datetime(2010, 1, 1),  # noqa: DTZ001
                datetime(2030, 1, 1),  # noqa: DTZ001
            ]
        df = pl.DataFrame({'d': values}, schema={'d': dtype})
        result = df.with_columns(
            compose_column(
                'd',
                [flag_dates_outside_range(lower=_LOWER, upper=_UPPER)],
                output_column_name='flag',
            )
        )['flag']
        assert result.to_list() == [True, False, True]

    def test_output_dtype_is_boolean(self) -> None:
        transform = flag_dates_outside_range(lower=_LOWER, upper=_UPPER)
        df = pl.DataFrame(
            {'d': [datetime(2010, 1, 1)]},  # noqa: DTZ001
            schema={'d': pl.Datetime('us')},
        )
        result = df.with_columns(
            compose_column('d', [transform], output_column_name='flag')
        )['flag']
        assert result.dtype == pl.Boolean

    def test_factory_returns_callable(self) -> None:
        assert callable(flag_dates_outside_range(upper=_UPPER))

    def test_filter_drops_out_of_range_rows(self) -> None:
        df = pl.DataFrame(
            {
                'd': [
                    datetime(1990, 1, 1),  # noqa: DTZ001  below
                    datetime(2010, 1, 1),  # noqa: DTZ001  in range
                    datetime(2030, 1, 1),  # noqa: DTZ001  above
                ]
            },
            schema={'d': pl.Datetime('us')},
        )
        kept = df.filter(
            ~compose_column(
                'd', [flag_dates_outside_range(lower=_LOWER, upper=_UPPER)]
            )
        )
        assert kept['d'].to_list() == [datetime(2010, 1, 1)]  # noqa: DTZ001

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {
                'd': [
                    datetime(1990, 1, 1),  # noqa: DTZ001
                    datetime(2010, 1, 1),  # noqa: DTZ001
                    None,
                ]
            },
            schema={'d': pl.Datetime('us')},
        )
        expr = compose_column(
            'd',
            [flag_dates_outside_range(lower=_LOWER, upper=_UPPER)],
            output_column_name='flag',
        )
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
