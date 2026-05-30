# tests/transforms/test_bounds.py
"""Tests for the numeric value-bounding factories in
``framesmith.transforms.bounds``.

Each transform is exercised through ``compose_column`` so the tests also
cover the integration point. Both preserve dtype and nulls.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import clip_numeric, winsorize_numeric


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


class TestClipNumeric:
    def test_both_bounds_clamp(self) -> None:
        # below -> lower, above -> upper, within and exactly at a bound
        # unchanged.
        result = _apply(
            [-5.0, 0.0, 5.0, 10.0, 50.0],
            clip_numeric(lower=0.0, upper=10.0),
            pl.Float64,
        )
        assert result.to_list() == [0.0, 0.0, 5.0, 10.0, 10.0]

    def test_lower_only(self) -> None:
        result = _apply(
            [-5.0, 3.0, 50.0], clip_numeric(lower=0.0), pl.Float64
        )
        assert result.to_list() == [0.0, 3.0, 50.0]

    def test_upper_only(self) -> None:
        result = _apply(
            [-5.0, 3.0, 50.0], clip_numeric(upper=10.0), pl.Float64
        )
        assert result.to_list() == [-5.0, 3.0, 10.0]

    def test_null_preserved(self) -> None:
        result = _apply(
            [None, 50.0], clip_numeric(lower=0.0, upper=10.0), pl.Float64
        )
        assert result.to_list() == [None, 10.0]

    def test_int_column_with_float_bounds_stays_int(self) -> None:
        result = _apply(
            [1, 5, 50], clip_numeric(lower=0.0, upper=10.0), pl.Int64
        )
        assert result.dtype == pl.Int64
        assert result.to_list() == [1, 5, 10]

    def test_both_none_raises(self) -> None:
        with pytest.raises(ValueError, match='at least one'):
            clip_numeric()

    def test_lower_exceeds_upper_raises(self) -> None:
        with pytest.raises(ValueError, match='exceed'):
            clip_numeric(lower=10.0, upper=0.0)

    def test_lower_equals_upper_allowed(self) -> None:
        # Everything collapses to the single value.
        result = _apply(
            [1.0, 5.0, 50.0], clip_numeric(lower=5.0, upper=5.0), pl.Float64
        )
        assert result.to_list() == [5.0, 5.0, 5.0]

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': [-5.0, 5.0, 50.0, None]}, schema={'x': pl.Float64}
        )
        expr = compose_column('x', [clip_numeric(lower=0.0, upper=10.0)])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestWinsorizeNumeric:
    def test_caps_extreme_at_upper_quantile(self) -> None:
        values = [float(n) for n in range(1, 10)] + [1000.0]
        df = pl.DataFrame({'x': values}, schema={'x': pl.Float64})
        capped = df.with_columns(
            compose_column('x', [winsorize_numeric()])
        )['x']
        upper_bound = df.select(
            pl.col('x').quantile(0.95, interpolation='linear')
        ).item()
        assert capped.max() == upper_bound
        assert capped.max() < max(values)

    def test_default_quantiles_applied(self) -> None:
        values = [float(n) for n in range(1, 10)] + [1000.0]
        from_default = _apply(values, winsorize_numeric(), pl.Float64)
        from_explicit = _apply(
            values, winsorize_numeric(0.05, 0.95), pl.Float64
        )
        assert from_default.to_list() == from_explicit.to_list()

    def test_custom_symmetric_quantiles(self) -> None:
        values = [float(n) for n in range(1, 11)]
        df = pl.DataFrame({'x': values}, schema={'x': pl.Float64})
        capped = df.with_columns(
            compose_column('x', [winsorize_numeric(0.1, 0.9)])
        )['x']
        lower_bound = df.select(
            pl.col('x').quantile(0.1, interpolation='linear')
        ).item()
        upper_bound = df.select(
            pl.col('x').quantile(0.9, interpolation='linear')
        ).item()
        assert capped.min() == lower_bound
        assert capped.max() == upper_bound

    def test_asymmetric_quantiles(self) -> None:
        # lower_quantile == 0.0 leaves the low tail untouched (min stays).
        values = [float(n) for n in range(1, 11)]
        df = pl.DataFrame({'x': values}, schema={'x': pl.Float64})
        capped = df.with_columns(
            compose_column('x', [winsorize_numeric(0.0, 0.9)])
        )['x']
        assert capped.min() == min(values)
        upper_bound = df.select(
            pl.col('x').quantile(0.9, interpolation='linear')
        ).item()
        assert capped.max() == upper_bound

    def test_null_preserved(self) -> None:
        values = [float(n) for n in range(1, 10)] + [None]
        result = _apply(values, winsorize_numeric(), pl.Float64)
        assert result.to_list()[-1] is None

    def test_constant_column_unchanged(self) -> None:
        result = _apply([7.0, 7.0, 7.0, None], winsorize_numeric(), pl.Float64)
        assert result.to_list() == [7.0, 7.0, 7.0, None]

    @pytest.mark.parametrize(
        ('lower_quantile', 'upper_quantile'),
        [
            (0.9, 0.1),  # lower >= upper
            (0.5, 0.5),  # equal
            (-0.1, 0.9),  # negative
            (0.1, 1.5),  # > 1
        ],
    )
    def test_invalid_quantiles_raise(
        self, lower_quantile: float, upper_quantile: float
    ) -> None:
        with pytest.raises(ValueError, match='lower_quantile < upper_quantile'):
            winsorize_numeric(lower_quantile, upper_quantile)

    def test_factory_returns_callable(self) -> None:
        assert callable(winsorize_numeric())

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': [1.0, 2.0, 3.0, 4.0, 1000.0, None]},
            schema={'x': pl.Float64},
        )
        expr = compose_column('x', [winsorize_numeric()])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
