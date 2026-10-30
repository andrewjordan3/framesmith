# tests/transforms/test_scaling.py
"""Tests for the numeric scaling transforms in
``framesmith.transforms.scaling``.

Each transform is exercised through ``compose_column`` so the tests
also cover the integration point. Scaling recomputes its statistics
over whatever frame it runs on; the expected values below are worked
out by hand from each fixture.
"""

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    log1p_numeric,
    minmax_numeric,
    zscore_numeric,
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


def _assert_close(result: pl.Series, expected: list[float | None]) -> None:
    actual = result.to_list()
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        if expected_value is None:
            assert actual_value is None
        else:
            assert actual_value == pytest.approx(expected_value)


class TestZscoreNumeric:
    def test_standardizes_to_zero_mean_unit_sample_std(self) -> None:
        # mean 2, sample std 1.
        result = _apply([1.0, 2.0, 3.0], zscore_numeric, pl.Float64)
        _assert_close(result, [-1.0, 0.0, 1.0])
        assert result.dtype == pl.Float64

    def test_non_trivial_values(self) -> None:
        # mean 25, sample std sqrt(500/3).
        result = _apply([10.0, 20.0, 30.0, 40.0], zscore_numeric, pl.Float64)
        sample_std = math.sqrt(500.0 / 3.0)
        _assert_close(
            result, [(value - 25.0) / sample_std for value in (10, 20, 30, 40)]
        )

    def test_int_input_produces_float(self) -> None:
        result = _apply([1, 2, 3], zscore_numeric, pl.Int64)
        assert result.dtype == pl.Float64
        _assert_close(result, [-1.0, 0.0, 1.0])

    def test_constant_column_maps_to_zero(self) -> None:
        result = _apply([7.0, 7.0, 7.0], zscore_numeric, pl.Float64)
        assert result.to_list() == [0.0, 0.0, 0.0]

    def test_single_value_maps_to_null(self) -> None:
        # A sample standard deviation is undefined for one value.
        result = _apply([7.0], zscore_numeric, pl.Float64)
        assert result.to_list() == [None]

    def test_null_preserved_and_excluded_from_statistics(self) -> None:
        result = _apply([1.0, None, 3.0], zscore_numeric, pl.Float64)
        sample_std = math.sqrt(2.0)
        _assert_close(result, [-1.0 / sample_std, None, 1.0 / sample_std])

    def test_all_null_column_stays_null(self) -> None:
        result = _apply([None, None], zscore_numeric, pl.Float64)
        assert result.to_list() == [None, None]

    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame({'x': [1.0, 2.0, 3.0, None]})
        expression = compose_column('x', [zscore_numeric])
        eager_result = df.with_columns(expression)
        lazy_result = df.lazy().with_columns(expression).collect()
        assert_frame_equal(eager_result, lazy_result)


class TestMinmaxNumeric:
    def test_scales_endpoints_to_zero_and_one(self) -> None:
        result = _apply([10.0, 20.0, 30.0], minmax_numeric, pl.Float64)
        assert result.to_list() == [0.0, 0.5, 1.0]
        assert result.dtype == pl.Float64

    def test_negative_range(self) -> None:
        result = _apply([-10.0, 0.0, 10.0], minmax_numeric, pl.Float64)
        assert result.to_list() == [0.0, 0.5, 1.0]

    def test_constant_column_maps_to_zero(self) -> None:
        result = _apply([4.0, 4.0], minmax_numeric, pl.Float64)
        assert result.to_list() == [0.0, 0.0]

    def test_single_value_maps_to_zero(self) -> None:
        result = _apply([4.0], minmax_numeric, pl.Float64)
        assert result.to_list() == [0.0]

    def test_null_preserved(self) -> None:
        result = _apply([0.0, None, 10.0], minmax_numeric, pl.Float64)
        assert result.to_list() == [0.0, None, 1.0]

    def test_all_null_column_stays_null(self) -> None:
        result = _apply([None, None], minmax_numeric, pl.Float64)
        assert result.to_list() == [None, None]


class TestLog1pNumeric:
    def test_zero_maps_to_zero(self) -> None:
        result = _apply([0.0], log1p_numeric, pl.Float64)
        assert result.to_list() == [0.0]

    def test_applies_log_one_plus_x(self) -> None:
        result = _apply([0.0, 3.0, 8.0], log1p_numeric, pl.Float64)
        _assert_close(result, [0.0, math.log(4.0), math.log(9.0)])

    def test_negative_maps_to_null_not_nan(self) -> None:
        result = _apply([-2.0, -0.5], log1p_numeric, pl.Float64)
        assert result.to_list() == [None, None]

    def test_null_preserved(self) -> None:
        result = _apply([None, 0.0], log1p_numeric, pl.Float64)
        assert result.to_list() == [None, 0.0]

    def test_int_input_produces_float(self) -> None:
        result = _apply([0, 3], log1p_numeric, pl.Int64)
        assert result.dtype == pl.Float64
        _assert_close(result, [0.0, math.log(4.0)])


class TestNonFiniteValues:
    """Regression tests from the logic audit: a nan or infinite value
    must propagate honestly, never be rewritten to a plausible 0.0."""

    def test_zscore_nan_poisons_statistics_not_zeroed(self) -> None:
        result = _apply(
            [1.0, float('nan'), 3.0], zscore_numeric, pl.Float64
        )
        assert all(math.isnan(value) for value in result.to_list())

    def test_zscore_inf_poisons_statistics_not_zeroed(self) -> None:
        result = _apply(
            [1.0, float('inf'), 3.0], zscore_numeric, pl.Float64
        )
        assert all(math.isnan(value) for value in result.to_list())

    def test_minmax_nan_stays_nan_finite_values_still_scale(
        self,
    ) -> None:
        result = _apply(
            [1.0, 2.0, float('nan')], minmax_numeric, pl.Float64
        ).to_list()
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert math.isnan(result[2])

    def test_minmax_single_nan_stays_nan(self) -> None:
        result = _apply([float('nan')], minmax_numeric, pl.Float64)
        assert math.isnan(result.to_list()[0])

    def test_minmax_inf_row_is_nan_finite_rows_zero(self) -> None:
        result = _apply(
            [1.0, 2.0, float('inf')], minmax_numeric, pl.Float64
        ).to_list()
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert math.isnan(result[2])

    def test_log1p_nan_and_inf_propagate(self) -> None:
        result = _apply(
            [float('nan'), float('inf')], log1p_numeric, pl.Float64
        ).to_list()
        assert math.isnan(result[0])
        assert result[1] == float('inf')


class TestDtypeEdges:
    """Regression tests from the logic audit: arithmetic happens in
    Float64, so no integer dtype can wrap and Decimal works."""

    def test_log1p_at_uint8_max(self) -> None:
        result = _apply([255], log1p_numeric, pl.UInt8)
        _assert_close(result, [math.log(256.0)])

    def test_log1p_at_int64_max(self) -> None:
        result = _apply([2**63 - 1], log1p_numeric, pl.Int64)
        _assert_close(result, [math.log(2.0**63)])

    def test_minmax_full_int8_range(self) -> None:
        result = _apply([-128, 0, 127], minmax_numeric, pl.Int8)
        _assert_close(result, [0.0, 128.0 / 255.0, 1.0])

    def test_minmax_full_int64_range(self) -> None:
        result = _apply(
            [-(2**63), 0, 2**63 - 1], minmax_numeric, pl.Int64
        )
        _assert_close(result, [0.0, 0.5, 1.0])

    def test_minmax_decimal_column(self) -> None:
        df = pl.DataFrame(
            {'x': [1.0, 2.0, 3.0]},
            schema={'x': pl.Decimal(10, 1)},
        )
        result = df.with_columns(
            compose_column('x', [minmax_numeric])
        )['x']
        assert result.dtype == pl.Float64
        assert result.to_list() == [0.0, 0.5, 1.0]

    def test_float32_input_returns_float64(self) -> None:
        for transform in (zscore_numeric, minmax_numeric):
            result = _apply([1.0, 2.0, 3.0], transform, pl.Float32)
            assert result.dtype == pl.Float64

    def test_zscore_constant_column_with_null_keeps_null(self) -> None:
        result = _apply(
            [7.0, 7.0, None], zscore_numeric, pl.Float64
        )
        assert result.to_list() == [0.0, 0.0, None]

    def test_minmax_constant_column_with_null_keeps_null(self) -> None:
        result = _apply(
            [4.0, 4.0, None], minmax_numeric, pl.Float64
        )
        assert result.to_list() == [0.0, 0.0, None]
