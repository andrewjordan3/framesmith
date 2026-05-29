# tests/transforms/test_dates.py
"""Tests for the numeric → datetime transforms in
``framesmith.transforms.dates``.

Each transform is exercised through ``compose_column`` so the tests also
cover the integration point. Imports go through the subpackage's public
surface (``from framesmith.transforms import ...``).
"""

from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    EpochTimeUnit,
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
