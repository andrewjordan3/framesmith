# tests/group/test_fill.py
"""Tests for the group-wise null fillers in ``framesmith.group.fill``."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith.group import (
    fill_null_numeric_by_group,
    fill_null_string_by_group,
)


class TestFillNullNumericByGroup:
    def test_mean_fills_per_group(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 1, 2, 2], 'v': [10, None, 20, 5, None]},
            schema={'g': pl.Int64, 'v': pl.Int64},
        )
        out = df.with_columns(
            fill_null_numeric_by_group('v', ['g'], strategy='mean')
        )
        assert out['v'].to_list() == [10.0, 15.0, 20.0, 5.0, 5.0]
        assert out['v'].dtype == pl.Float64

    def test_median_fills_per_group(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 1, 1], 'v': [1, 2, 100, None]},
            schema={'g': pl.Int64, 'v': pl.Int64},
        )
        out = df.with_columns(
            fill_null_numeric_by_group('v', ['g'], strategy='median')
        )
        assert out['v'].to_list() == [1.0, 2.0, 100.0, 2.0]

    def test_auto_skewed_uses_median(self) -> None:
        # Whole-column skew is high -> median fills the null.
        df = pl.DataFrame(
            {'g': [1] * 8, 'v': [1.0, 1, 1, 1, 1, 2, 3, None]},
            schema={'g': pl.Int64, 'v': pl.Float64},
        )
        out = df.with_columns(fill_null_numeric_by_group('v', ['g']))
        # group median of [1,1,1,1,1,2,3] = 1.0
        assert out['v'].to_list()[-1] == 1.0

    def test_auto_symmetric_uses_mean(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 1], 'v': [1.0, 3.0, None]},
            schema={'g': pl.Int64, 'v': pl.Float64},
        )
        out = df.with_columns(fill_null_numeric_by_group('v', ['g']))
        assert out['v'].to_list() == [1.0, 3.0, 2.0]  # mean of [1, 3]

    def test_auto_constant_column_uses_mean(self) -> None:
        # skew is NaN (degenerate) -> mean (== the constant).
        df = pl.DataFrame(
            {'g': [1, 1, 2], 'v': [5.0, None, 5.0]},
            schema={'g': pl.Int64, 'v': pl.Float64},
        )
        out = df.with_columns(fill_null_numeric_by_group('v', ['g']))
        assert out['v'].to_list() == [5.0, 5.0, 5.0]

    def test_all_null_group_stays_null(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 2], 'v': [10.0, None, None]},
            schema={'g': pl.Int64, 'v': pl.Float64},
        )
        out = df.with_columns(
            fill_null_numeric_by_group('v', ['g'], strategy='mean')
        )
        assert out['v'].to_list() == [10.0, 10.0, None]

    def test_numeric_string_is_accepted(self) -> None:
        # String dtype holding numeric text: the strict cast succeeds.
        df = pl.DataFrame(
            {'g': [1, 1, 1], 'v': ['10', None, '20']},
            schema={'g': pl.Int64, 'v': pl.String},
        )
        out = df.with_columns(
            fill_null_numeric_by_group('v', ['g'], strategy='mean')
        )
        assert out['v'].to_list() == [10.0, 15.0, 20.0]

    @pytest.mark.parametrize('strategy', ['mean', 'median', 'auto'])
    def test_non_numeric_column_raises_loudly(self, strategy: str) -> None:
        df = pl.DataFrame(
            {'g': [1, 1], 'v': ['a', None]},
            schema={'g': pl.Int64, 'v': pl.String},
        )
        with pytest.raises(pl.exceptions.InvalidOperationError):
            df.with_columns(
                fill_null_numeric_by_group('v', ['g'], strategy=strategy)  # type: ignore[arg-type]
            )

    def test_empty_group_columns_raises(self) -> None:
        with pytest.raises(ValueError, match='group_columns must not be empty'):
            fill_null_numeric_by_group('v', [])

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match='strategy must be'):
            fill_null_numeric_by_group('v', ['g'], strategy='mode')  # type: ignore[arg-type]

    def test_nonpositive_skew_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match='skew_threshold must be'):
            fill_null_numeric_by_group('v', ['g'], skew_threshold=0)

    def test_multi_column_key(self) -> None:
        df = pl.DataFrame(
            {'a': [1, 1], 'b': ['x', 'x'], 'v': [10.0, None]},
            schema={'a': pl.Int64, 'b': pl.String, 'v': pl.Float64},
        )
        out = df.with_columns(
            fill_null_numeric_by_group('v', ['a', 'b'], strategy='mean')
        )
        assert out['v'].to_list() == [10.0, 10.0]

    def test_missing_column_raises_on_apply(self) -> None:
        df = pl.DataFrame({'g': [1, 1], 'v': [1.0, None]})
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.with_columns(fill_null_numeric_by_group('nope', ['g']))

    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 2, 2], 'v': [10.0, None, 5.0, None]},
            schema={'g': pl.Int64, 'v': pl.Float64},
        )
        expr = fill_null_numeric_by_group('v', ['g'])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestFillNullStringByGroup:
    def test_mode_fills_per_group(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 1, 2, 2], 'v': ['a', 'a', None, 'x', None]},
            schema={'g': pl.Int64, 'v': pl.String},
        )
        out = df.with_columns(fill_null_string_by_group('v', ['g']))
        assert out['v'].to_list() == ['a', 'a', 'a', 'x', 'x']
        assert out['v'].dtype == pl.String

    def test_tie_breaks_to_first_in_input_order(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 1, 1, 1], 'v': ['b', 'a', 'b', 'a', None]},
            schema={'g': pl.Int64, 'v': pl.String},
        )
        out = df.with_columns(fill_null_string_by_group('v', ['g']))
        assert out['v'].to_list() == ['b', 'a', 'b', 'a', 'b']

    def test_all_null_group_stays_null(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 2], 'v': ['a', None, None]},
            schema={'g': pl.Int64, 'v': pl.String},
        )
        out = df.with_columns(fill_null_string_by_group('v', ['g']))
        assert out['v'].to_list() == ['a', 'a', None]

    def test_numeric_column_is_accepted_not_rejected(self) -> None:
        # Documented asymmetry: mode works on numeric, so this fills
        # rather than raising.
        df = pl.DataFrame(
            {'g': [1, 1, 1], 'v': [7, 7, None]},
            schema={'g': pl.Int64, 'v': pl.Int64},
        )
        out = df.with_columns(fill_null_string_by_group('v', ['g']))
        assert out['v'].to_list() == [7, 7, 7]

    def test_empty_group_columns_raises(self) -> None:
        with pytest.raises(ValueError, match='group_columns must not be empty'):
            fill_null_string_by_group('v', [])

    def test_missing_column_raises_on_apply(self) -> None:
        df = pl.DataFrame({'g': [1, 1], 'v': ['a', None]})
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.with_columns(fill_null_string_by_group('nope', ['g']))

    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 2], 'v': ['a', None, 'x']},
            schema={'g': pl.Int64, 'v': pl.String},
        )
        expr = fill_null_string_by_group('v', ['g'])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
