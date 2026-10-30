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
