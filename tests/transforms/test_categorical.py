# tests/transforms/test_categorical.py
"""Tests for the categorical-consolidation factories in
``framesmith.transforms.categorical``.

Each transform is exercised through ``compose_column`` so the tests also
cover the integration point. These factories are opt-in and data-dependent;
the composition test pins canonicalization-before-counting.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import NORMALIZE_TEXT, ExpressionTransform, compose_column
from framesmith.transforms import (
    collapse_keep_top_n,
    collapse_rare_by_count,
    map_categories,
)


def _apply(
    values: list[str | None], transform: ExpressionTransform
) -> pl.Series:
    """Run a single transform on a 1-column ``pl.String`` frame.

    An explicit schema lets all-null inputs through without dtype
    inference failing.
    """
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', [transform]))['x']


class TestMapCategories:
    def test_same_type_string_passthrough(self) -> None:
        # Unmapped 'c' passes through unchanged; null passes through.
        result = _apply(
            ['a', 'b', 'c', None], map_categories({'a': 'alpha', 'b': 'beta'})
        )
        assert result.to_list() == ['alpha', 'beta', 'c', None]

    def test_code_to_label_cross_type(self) -> None:
        # Int keys -> String values. Unmapped 3 is cast to the output
        # dtype ('3'); null passes through. Output column is String.
        df = pl.DataFrame({'x': [1, 2, 3, None]}, schema={'x': pl.Int64})
        result = df.with_columns(
            compose_column('x', [map_categories({1: 'Yes', 2: 'No'})])
        )['x']
        assert result.to_list() == ['Yes', 'No', '3', None]
        assert result.dtype == pl.String

    def test_same_type_int_preserves_dtype(self) -> None:
        df = pl.DataFrame({'x': [1, 2, 3, None]}, schema={'x': pl.Int64})
        result = df.with_columns(
            compose_column('x', [map_categories({1: 10, 2: 20})])
        )['x']
        assert result.to_list() == [10, 20, 3, None]
        assert result.dtype == pl.Int64

    def test_explicit_null_mapping(self) -> None:
        # None as a key maps null; unmapped 2 stringifies under the
        # cross-type map.
        df = pl.DataFrame({'x': [1, 2, None]}, schema={'x': pl.Int64})
        result = df.with_columns(
            compose_column('x', [map_categories({1: 'Yes', None: 'missing'})])
        )['x']
        assert result.to_list() == ['Yes', '2', 'missing']

    def test_case_sensitive_exact_match(self) -> None:
        # No normalization: 'A' does not match key 'a'.
        result = _apply(['a', 'A'], map_categories({'a': 'alpha'}))
        assert result.to_list() == ['alpha', 'A']

    def test_empty_map_raises(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            map_categories({})

    def test_factory_returns_callable(self) -> None:
        assert callable(map_categories({'a': 'b'}))

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame({'x': [1, 2, 3, None]}, schema={'x': pl.Int64})
        expr = compose_column('x', [map_categories({1: 'Yes', 2: 'No'})])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestCollapseRareByCount:
    def test_basic_collapse(self) -> None:
        result = _apply(
            ['a', 'a', 'a', 'b', 'b', 'c'], collapse_rare_by_count(2)
        )
        assert result.to_list() == ['a', 'a', 'a', 'b', 'b', 'other']

    def test_boundary_is_inclusive(self) -> None:
        # Count exactly min_count is kept; count below is replaced.
        result = _apply(['a', 'a', 'b'], collapse_rare_by_count(2))
        assert result.to_list() == ['a', 'a', 'other']

    def test_min_count_one_keeps_all(self) -> None:
        result = _apply(['a', 'b', 'c'], collapse_rare_by_count(1))
        assert result.to_list() == ['a', 'b', 'c']

    def test_custom_replacement(self) -> None:
        result = _apply(
            ['a', 'a', 'b'], collapse_rare_by_count(2, replacement='RARE')
        )
        assert result.to_list() == ['a', 'a', 'RARE']

    def test_null_passes_through_untouched(self) -> None:
        # Load-bearing: a null's window count is 0, which is < min_count.
        # The is_null() guard must keep it null, not replace it.
        result = _apply(['a', 'a', None, 'b'], collapse_rare_by_count(2))
        assert result.to_list() == ['a', 'a', None, 'other']

    def test_all_one_category_unchanged(self) -> None:
        result = _apply(['x', 'x', 'x'], collapse_rare_by_count(2))
        assert result.to_list() == ['x', 'x', 'x']

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['a', 'a', 'b'], collapse_rare_by_count(2))
        assert result.dtype == pl.String

    @pytest.mark.parametrize('min_count', [0, -1])
    def test_invalid_min_count_raises(self, min_count: int) -> None:
        with pytest.raises(ValueError, match='min_count must be >= 1'):
            collapse_rare_by_count(min_count)

    def test_factory_returns_callable(self) -> None:
        assert callable(collapse_rare_by_count(2))

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['a', 'a', 'b', 'c', None]}, schema={'x': pl.String}
        )
        expr = compose_column('x', [collapse_rare_by_count(2)])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestCollapseKeepTopN:
    def test_distinct_frequencies(self) -> None:
        # a:4 tier1, b:3 tier2 kept; c:2 tier3, d:1 tier4 collapsed.
        result = _apply(
            ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd'],
            collapse_keep_top_n(2),
        )
        assert result.to_list() == [
            'a', 'a', 'a', 'a', 'b', 'b', 'b', 'other', 'other', 'other'
        ]

    def test_tie_expands_beyond_n(self) -> None:
        # Load-bearing contract: a and b both freq 3 share tier1, so n=1
        # keeps both; c (freq 2) collapses.
        result = _apply(
            ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c'], collapse_keep_top_n(1)
        )
        assert result.to_list() == [
            'a', 'a', 'a', 'b', 'b', 'b', 'other', 'other'
        ]

    def test_n_larger_than_tier_count_keeps_all(self) -> None:
        result = _apply(['a', 'a', 'b'], collapse_keep_top_n(5))
        assert result.to_list() == ['a', 'a', 'b']

    def test_custom_replacement(self) -> None:
        result = _apply(
            ['a', 'a', 'b'], collapse_keep_top_n(1, replacement='RARE')
        )
        assert result.to_list() == ['a', 'a', 'RARE']

    def test_null_passes_through_untouched(self) -> None:
        result = _apply(['a', 'a', 'a', 'b', None], collapse_keep_top_n(1))
        assert result.to_list() == ['a', 'a', 'a', 'other', None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['a', 'a', 'b'], collapse_keep_top_n(1))
        assert result.dtype == pl.String

    @pytest.mark.parametrize('n', [0, -1])
    def test_invalid_n_raises(self, n: int) -> None:
        with pytest.raises(ValueError, match='n must be >= 1'):
            collapse_keep_top_n(n)

    def test_factory_returns_callable(self) -> None:
        assert callable(collapse_keep_top_n(2))

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['a', 'a', 'a', 'b', 'b', 'c', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [collapse_keep_top_n(1)])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestComposabilityWithNormalizeText:
    def test_canonicalization_before_counting(self) -> None:
        # NORMALIZE_TEXT strips the padded ' ACME ' to 'ACME' before the
        # collapse counts, so it merges with the bare 'ACME' (count 3);
        # 'Beta' (count 2) is kept; 'Gamma' (count 1) collapses.
        recipe = (*NORMALIZE_TEXT, collapse_rare_by_count(2))
        df = pl.DataFrame(
            {'x': [' ACME ', 'ACME', 'ACME', 'Beta', 'Beta', 'Gamma']},
            schema={'x': pl.String},
        )
        result = df.with_columns(compose_column('x', recipe))
        assert result['x'].to_list() == [
            'ACME', 'ACME', 'ACME', 'Beta', 'Beta', 'other'
        ]
