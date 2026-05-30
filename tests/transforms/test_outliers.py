# tests/transforms/test_outliers.py
"""Tests for the outlier-flagging factories in
``framesmith.transforms.outliers``.

Each transform is exercised through ``compose_column`` so the tests also
cover the integration point. The constant-column tests pin the ``nan``
guard — without it a constant column would flag every row.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    flag_iqr_outliers,
    flag_mad_outliers,
    flag_zscore_outliers,
)


def _flag(
    values: list[float | None], transform: ExpressionTransform
) -> list[bool | None]:
    """Apply a flag transform to a Float64 column, returning the flags."""
    df = pl.DataFrame({'x': values}, schema={'x': pl.Float64})
    return df.with_columns(
        compose_column('x', [transform], output_column_name='flag')
    )['flag'].to_list()


class TestFlagZscoreOutliers:
    def test_detects_clear_outlier(self) -> None:
        # z-score is non-robust, so use enough in-distribution points that
        # the extreme still exceeds 3 sigma.
        values = [10.0] * 20 + [50.0]
        flags = _flag(values, flag_zscore_outliers())
        assert flags[-1] is True
        assert all(flag is False for flag in flags[:-1])

    def test_constant_column_flags_nothing(self) -> None:
        # The guard: std==0 -> nan, and nan > threshold is True in polars.
        # A constant column has no outliers and must flag nothing.
        flags = _flag([7.0] * 5, flag_zscore_outliers())
        assert flags == [False] * 5

    def test_null_input_yields_null_flag(self) -> None:
        flags = _flag([10.0] * 10 + [None], flag_zscore_outliers())
        assert flags[-1] is None

    def test_single_value_yields_null_flag(self) -> None:
        # One non-null value -> std is null -> z is null -> null flag.
        flags = _flag([42.0], flag_zscore_outliers())
        assert flags == [None]

    def test_output_dtype_is_boolean(self) -> None:
        df = pl.DataFrame({'x': [1.0, 2.0, 3.0]}, schema={'x': pl.Float64})
        result = df.with_columns(
            compose_column(
                'x', [flag_zscore_outliers()], output_column_name='flag'
            )
        )['flag']
        assert result.dtype == pl.Boolean

    @pytest.mark.parametrize('threshold', [0.0, -1.0])
    def test_non_positive_threshold_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError, match='positive'):
            flag_zscore_outliers(threshold)

    def test_factory_returns_callable(self) -> None:
        assert callable(flag_zscore_outliers())

    def test_default_threshold_applies(self) -> None:
        # Default 3.0: a 2-sigma point is not flagged, matching |z| > 3.
        values = [10.0] * 20 + [50.0]
        from_default = _flag(values, flag_zscore_outliers())
        from_explicit = _flag(values, flag_zscore_outliers(3.0))
        assert from_default == from_explicit

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': [10.0] * 20 + [50.0, None]}, schema={'x': pl.Float64}
        )
        expr = compose_column(
            'x', [flag_zscore_outliers()], output_column_name='flag'
        )
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestFlagIqrOutliers:
    def test_detects_clear_outlier(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
        flags = _flag(values, flag_iqr_outliers())
        assert flags[-1] is True
        assert all(flag is False for flag in flags[:-1])

    def test_constant_column_flags_nothing(self) -> None:
        # IQR is pure comparison (no division), so a constant column is
        # naturally safe — no guard needed.
        flags = _flag([7.0] * 5, flag_iqr_outliers())
        assert flags == [False] * 5

    def test_null_input_yields_null_flag(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, None]
        flags = _flag(values, flag_iqr_outliers())
        assert flags[-1] is None

    def test_single_value_flags_false(self) -> None:
        # One value -> Q1==Q3==value, IQR==0, value is on both fences ->
        # strict comparison flags nothing.
        flags = _flag([42.0], flag_iqr_outliers())
        assert flags == [False]

    def test_output_dtype_is_boolean(self) -> None:
        df = pl.DataFrame(
            {'x': [1.0, 2.0, 3.0, 4.0]}, schema={'x': pl.Float64}
        )
        result = df.with_columns(
            compose_column(
                'x', [flag_iqr_outliers()], output_column_name='flag'
            )
        )['flag']
        assert result.dtype == pl.Boolean

    @pytest.mark.parametrize('multiplier', [0.0, -1.5])
    def test_non_positive_multiplier_raises(self, multiplier: float) -> None:
        with pytest.raises(ValueError, match='positive'):
            flag_iqr_outliers(multiplier)

    def test_factory_returns_callable(self) -> None:
        assert callable(flag_iqr_outliers())

    def test_default_multiplier_applies(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]
        from_default = _flag(values, flag_iqr_outliers())
        from_explicit = _flag(values, flag_iqr_outliers(1.5))
        assert from_default == from_explicit

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, None]},
            schema={'x': pl.Float64},
        )
        expr = compose_column(
            'x', [flag_iqr_outliers()], output_column_name='flag'
        )
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestFlagMadOutliers:
    def test_detects_clear_outlier(self) -> None:
        values = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 50.0]
        flags = _flag(values, flag_mad_outliers())
        assert flags[-1] is True
        assert all(flag is False for flag in flags[:-1])

    def test_constant_column_flags_nothing(self) -> None:
        # The guard: MAD==0 -> nan, and nan > threshold is True in polars.
        flags = _flag([7.0] * 5, flag_mad_outliers())
        assert flags == [False] * 5

    def test_null_input_yields_null_flag(self) -> None:
        flags = _flag([10.0, 11.0, 9.0, 10.0, None], flag_mad_outliers())
        assert flags[-1] is None

    def test_single_value_yields_null_flag(self) -> None:
        # One value -> median==value, MAD==0 -> nan -> guarded to False.
        # (MAD of a single value is 0, not null, so the guard returns
        # False rather than null here.)
        flags = _flag([42.0], flag_mad_outliers())
        assert flags == [False]

    def test_output_dtype_is_boolean(self) -> None:
        df = pl.DataFrame(
            {'x': [10.0, 11.0, 9.0, 10.0]}, schema={'x': pl.Float64}
        )
        result = df.with_columns(
            compose_column(
                'x', [flag_mad_outliers()], output_column_name='flag'
            )
        )['flag']
        assert result.dtype == pl.Boolean

    @pytest.mark.parametrize('threshold', [0.0, -3.5])
    def test_non_positive_threshold_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError, match='positive'):
            flag_mad_outliers(threshold)

    def test_factory_returns_callable(self) -> None:
        assert callable(flag_mad_outliers())

    def test_default_threshold_applies(self) -> None:
        values = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 50.0]
        from_default = _flag(values, flag_mad_outliers())
        from_explicit = _flag(values, flag_mad_outliers(3.5))
        assert from_default == from_explicit

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': [10.0, 11.0, 9.0, 10.0, 50.0, None]},
            schema={'x': pl.Float64},
        )
        expr = compose_column(
            'x', [flag_mad_outliers()], output_column_name='flag'
        )
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
