# tests/transforms/test_binning.py
"""Tests for the binning factory in ``framesmith.transforms.binning``.

The transform is exercised through ``compose_column`` so the tests
also cover the integration point.
"""

import polars as pl
import pytest

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import bin_by_edges


def _apply(
    values: list[float | int | None],
    transform: ExpressionTransform,
    dtype: pl.DataType = pl.Float64,
) -> pl.Series:
    """Run a single transform on a 1-column numeric frame."""
    df = pl.DataFrame({'x': values}, schema={'x': dtype})
    return df.with_columns(compose_column('x', [transform]))['x']


class TestBinByEdges:
    def test_assigns_interval_indices(self) -> None:
        result = _apply([5.0, 18.0, 35.0, 70.0], bin_by_edges([18.0, 65.0]))
        assert result.to_list() == [0, 1, 1, 2]
        assert result.dtype == pl.Int64

    def test_edge_value_starts_its_interval(self) -> None:
        # Lower bounds are inclusive: 18 opens bin 1, 65 opens bin 2.
        result = _apply([17.999, 18.0, 64.999, 65.0], bin_by_edges([18.0, 65.0]))
        assert result.to_list() == [0, 1, 1, 2]

    def test_single_edge_splits_in_two(self) -> None:
        result = _apply([0.0, 0.5, 1.0], bin_by_edges([0.5]))
        assert result.to_list() == [0, 1, 1]

    def test_nan_lands_in_final_bin(self) -> None:
        # Nan is not a null in polars and sorts above every edge.
        result = _apply([float('nan')], bin_by_edges([18.0, 65.0]))
        assert result.to_list() == [2]

    def test_null_is_never_binned(self) -> None:
        result = _apply([None, 5.0, 100.0], bin_by_edges([18.0, 65.0]))
        assert result.to_list() == [None, 0, 2]

    def test_int_input(self) -> None:
        result = _apply([1, 10, 100], bin_by_edges([10.0]), dtype=pl.Int64)
        assert result.to_list() == [0, 1, 1]

    def test_caller_mutation_after_configuration_has_no_effect(
        self,
    ) -> None:
        edges = [18.0, 65.0]
        transform = bin_by_edges(edges)
        edges.append(100.0)
        result = _apply([70.0], transform)
        assert result.to_list() == [2]


class TestBinByEdgesValidation:
    def test_empty_edges_raise(self) -> None:
        with pytest.raises(ValueError, match='must not be empty'):
            bin_by_edges([])

    def test_equal_edges_raise(self) -> None:
        with pytest.raises(ValueError, match='strictly increasing'):
            bin_by_edges([10.0, 10.0])

    def test_decreasing_edges_raise(self) -> None:
        with pytest.raises(ValueError, match='strictly increasing'):
            bin_by_edges([65.0, 18.0])

    @pytest.mark.parametrize(
        'edges',
        [[float('inf')], [0.0, float('nan')], [float('-inf'), 0.0]],
        ids=['inf', 'nan', 'negative_inf'],
    )
    def test_non_finite_edges_raise(self, edges: list[float]) -> None:
        with pytest.raises(ValueError, match='finite'):
            bin_by_edges(edges)

    @pytest.mark.parametrize(
        'edges',
        [['a', 'b'], [None], [True, 2.0]],
        ids=['strings', 'none', 'bool'],
    )
    def test_non_numeric_edges_raise_value_error(
        self, edges: list[object]
    ) -> None:
        with pytest.raises(ValueError, match='real numbers'):
            bin_by_edges(edges)  # type: ignore[arg-type]
