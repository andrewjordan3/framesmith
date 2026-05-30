# tests/transforms/test_split.py
"""Tests for ``framesmith.transforms.split``.

Each transform is exercised through ``compose_column`` so the tests also
cover the integration point.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    DEFAULT_SPLIT_DELIMITERS,
    split_on_delimiters,
)


def _apply(
    values: list[str | None], transform: ExpressionTransform
) -> list[list[str | None] | None]:
    """Run a single transform on a 1-column String frame, return lists."""
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', [transform]))['x'].to_list()


class TestSplitOnDelimiters:
    def test_canonical_preserves_empty_positions(self) -> None:
        assert _apply(['a,, b ,'], split_on_delimiters()) == [
            ['a', None, 'b', None]
        ]

    def test_leading_and_trailing_empties(self) -> None:
        assert _apply([',a,'], split_on_delimiters()) == [[None, 'a', None]]

    def test_mixed_delimiter_types(self) -> None:
        assert _apply(['a; b | c'], split_on_delimiters()) == [
            ['a', 'b', 'c']
        ]

    def test_hyphen_is_a_default_delimiter(self) -> None:
        assert _apply(['co-op'], split_on_delimiters()) == [['co', 'op']]

    def test_no_delimiter_single_token(self) -> None:
        assert _apply(['hello'], split_on_delimiters()) == [['hello']]

    def test_token_is_trimmed(self) -> None:
        assert _apply([' a , b '], split_on_delimiters()) == [['a', 'b']]

    def test_empty_string_becomes_single_null(self) -> None:
        assert _apply([''], split_on_delimiters()) == [[None]]

    def test_whitespace_only_becomes_single_null(self) -> None:
        assert _apply(['   '], split_on_delimiters()) == [[None]]

    def test_null_propagates_as_null_list(self) -> None:
        # Distinct from empty string: None -> null, '' -> [None].
        assert _apply([None], split_on_delimiters()) == [None]

    def test_dedup_collapses_adjacent_delimiters(self) -> None:
        assert _apply(['a,,b'], split_on_delimiters(dedup_delimiters=True)) == [
            ['a', 'b']
        ]

    def test_default_does_not_collapse(self) -> None:
        assert _apply(['a,,b'], split_on_delimiters()) == [['a', None, 'b']]

    def test_dedup_whitespace_between_delimiters_still_nulls(self) -> None:
        # Load-bearing: a space breaks the delimiter run, so the gap is a
        # real empty field even with dedup on.
        assert _apply(
            ['a, ,b'], split_on_delimiters(dedup_delimiters=True)
        ) == [['a', None, 'b']]

    def test_custom_single_delimiter(self) -> None:
        assert _apply(['a:b:c'], split_on_delimiters([':'])) == [
            ['a', 'b', 'c']
        ]

    def test_output_dtype_is_list_string(self) -> None:
        df = pl.DataFrame({'x': ['a,b', None]}, schema={'x': pl.String})
        result = df.with_columns(compose_column('x', [split_on_delimiters()]))
        assert result['x'].dtype == pl.List(pl.String)

    def test_empty_delimiters_raises(self) -> None:
        with pytest.raises(ValueError, match='must not be empty'):
            split_on_delimiters([])

    def test_multi_char_delimiter_raises(self) -> None:
        with pytest.raises(ValueError, match='exactly one character'):
            split_on_delimiters([', '])

    def test_factory_returns_callable(self) -> None:
        assert callable(split_on_delimiters())

    def test_default_delimiters_constant(self) -> None:
        assert DEFAULT_SPLIT_DELIMITERS == (',', ';', '|', '/', '-')

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['a,, b ,', None, 'co-op']}, schema={'x': pl.String}
        )
        expr = compose_column('x', [split_on_delimiters()])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
