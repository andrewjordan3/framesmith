# tests/combine/test_hash_key.py
"""Tests for ``framesmith.combine.hash_key``.

Imports go through the directory's public surface
(``from framesmith.combine import ...``), not the internal ``hash_key``
module, so the tests exercise the same contract callers see.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith.combine import hash_key

# ---------------------------------------------------------------------
# Frame-independence — the headline property
# ---------------------------------------------------------------------


class TestFrameIndependence:
    def test_same_value_same_key_across_frames(self) -> None:
        expr = hash_key(['x'], 'k')
        a = pl.DataFrame({'x': ['East', 'West']}).with_columns(expr)
        b = pl.DataFrame(
            {'x': ['North', 'South', 'West']}
        ).with_columns(expr)
        west_a = a.filter(pl.col('x') == 'West')['k'].item()
        west_b = b.filter(pl.col('x') == 'West')['k'].item()
        assert west_a == west_b

    def test_composite_same_combination_same_key_across_frames(self) -> None:
        expr = hash_key(['region', 'branch'], 'k')
        a = pl.DataFrame(
            {'region': ['W', 'E'], 'branch': ['b', 'a']}
        ).with_columns(expr)
        b = pl.DataFrame(
            {'region': ['W', 'W', 'N'], 'branch': ['b', 'a', 'c']}
        ).with_columns(expr)
        wb_a = a.filter(
            (pl.col('region') == 'W') & (pl.col('branch') == 'b')
        )['k'].item()
        wb_b = b.filter(
            (pl.col('region') == 'W') & (pl.col('branch') == 'b')
        )['k'].item()
        assert wb_a == wb_b

    def test_recomputing_same_call_is_deterministic(self) -> None:
        df = pl.DataFrame({'x': ['a', 'b', 'c']})
        first = df.with_columns(hash_key(['x'], 'k'))['k'].to_list()
        second = df.with_columns(hash_key(['x'], 'k'))['k'].to_list()
        assert first == second


# ---------------------------------------------------------------------
# Key properties: distinctness, dtype, null handling
# ---------------------------------------------------------------------


class TestKeyProperties:
    def test_distinct_inputs_distinct_keys_no_collision(self) -> None:
        df = pl.DataFrame(
            {'region': ['W', 'W', 'E', 'E'], 'branch': ['a', 'b', 'a', 'b']}
        ).with_columns(hash_key(['region', 'branch'], 'k'))
        struct_unique = df.select(
            pl.struct(['region', 'branch']).n_unique()
        ).item()
        key_unique = df['k'].n_unique()
        assert key_unique == struct_unique

    def test_output_dtype_is_uint64(self) -> None:
        df = pl.DataFrame({'x': ['a', 'b']}).with_columns(
            hash_key(['x'], 'k')
        )
        assert df['k'].dtype == pl.UInt64

    def test_nulls_are_hashed_not_propagated(self) -> None:
        df = pl.DataFrame(
            {'x': ['a', None, 'a', None]}, schema={'x': pl.String}
        ).with_columns(hash_key(['x'], 'k'))
        keys = df['k'].to_list()
        # No null in the key column — nulls hash to a real value.
        assert df['k'].null_count() == 0
        # Both 'a' rows share a key; both None rows share a (different) key.
        assert keys[0] == keys[2]
        assert keys[1] == keys[3]
        assert keys[0] != keys[1]

    def test_output_added_alongside_originals(self) -> None:
        df = pl.DataFrame({'a': ['x'], 'b': ['y']}).with_columns(
            hash_key(['a', 'b'], 'k')
        )
        assert df.columns == ['a', 'b', 'k']


# ---------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------


class TestErrors:
    def test_empty_column_names_raises(self) -> None:
        with pytest.raises(
            ValueError, match='column_names must not be empty'
        ):
            hash_key([], 'k')

    def test_missing_column_raises_on_apply(self) -> None:
        df = pl.DataFrame({'x': ['a']})
        expr = hash_key(['x', 'nope'], 'k')  # builds fine
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.with_columns(expr)


# ---------------------------------------------------------------------
# Evaluation equivalence
# ---------------------------------------------------------------------


class TestLazyEagerEquivalence:
    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {'region': ['W', 'E', 'W'], 'branch': ['a', 'b', 'a']}
        )
        expr = hash_key(['region', 'branch'], 'k')
        assert_frame_equal(
            df.with_columns(expr), df.lazy().with_columns(expr).collect()
        )
