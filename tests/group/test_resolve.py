# tests/group/test_resolve.py
"""Tests for the group resolvers in ``framesmith.group``.

Imports go through the directory's public surface
(``from framesmith.group import ...``), not the internal ``resolve``
module. Output row and column order are unclaimed, so every frame
comparison passes ``check_row_order=False`` / ``check_column_order=False``.
"""

import warnings

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith.group import (
    first_non_null_per_group,
    mode_then_first_per_group,
)

# ---------------------------------------------------------------------
# Resolution behavior
# ---------------------------------------------------------------------


class TestResolution:
    def test_populates_from_siblings(self) -> None:
        df = pl.DataFrame(
            {'id': [1, 1], 'name': ['Alice', None], 'city': [None, 'NYC']}
        )
        result = df.group_by(['id']).agg(first_non_null_per_group(['id']))
        expected = pl.DataFrame(
            {'id': [1], 'name': ['Alice'], 'city': ['NYC']}
        )
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_conflict_resolves_to_input_order_first(self) -> None:
        # Determinism pin: the first non-null value in input row order
        # wins within the group.
        df = pl.DataFrame({'id': [1, 1], 'name': ['A', 'B']})
        result = df.group_by(['id']).agg(first_non_null_per_group(['id']))
        expected = pl.DataFrame({'id': [1], 'name': ['A']})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_already_unique_unchanged(self) -> None:
        df = pl.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        result = df.group_by(['id']).agg(first_non_null_per_group(['id']))
        expected = pl.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_all_null_field_stays_null(self) -> None:
        df = pl.DataFrame(
            {'id': [1, 1], 'name': [None, None]},
            schema={'id': pl.Int64, 'name': pl.String},
        )
        result = df.group_by(['id']).agg(first_non_null_per_group(['id']))
        expected = pl.DataFrame(
            {'id': [1], 'name': [None]},
            schema={'id': pl.Int64, 'name': pl.String},
        )
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_multi_column_key(self) -> None:
        df = pl.DataFrame(
            {'a': [1, 1], 'b': ['x', 'x'], 'val': [None, 'V']}
        )
        result = df.group_by(['a', 'b']).agg(
            first_non_null_per_group(['a', 'b'])
        )
        expected = pl.DataFrame({'a': [1], 'b': ['x'], 'val': ['V']})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_multiple_groups_mixed(self) -> None:
        # id 1 is duplicated and split across rows; id 2 and 3 are unique.
        df = pl.DataFrame(
            {
                'id': [1, 1, 2, 3],
                'name': ['Alice', None, 'Bob', None],
                'city': [None, 'NYC', 'LA', 'SF'],
            }
        )
        result = df.group_by(['id']).agg(first_non_null_per_group(['id']))
        expected = pl.DataFrame(
            {
                'id': [1, 2, 3],
                'name': ['Alice', 'Bob', None],
                'city': ['NYC', 'LA', 'SF'],
            }
        )
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_degenerate_no_non_key_columns(self) -> None:
        df = pl.DataFrame({'id': [1, 1]})
        result = df.group_by(['id']).agg(first_non_null_per_group(['id']))
        expected = pl.DataFrame({'id': [1]})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )


# ---------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------


class TestErrors:
    def test_empty_key_columns_raises(self) -> None:
        with pytest.raises(
            ValueError, match='key_columns must not be empty'
        ):
            first_non_null_per_group([])

    def test_missing_key_raises_on_apply(self) -> None:
        df = pl.DataFrame({'id': [1, 1], 'name': ['A', 'B']})
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.group_by(['nope']).agg(first_non_null_per_group(['nope']))


# ---------------------------------------------------------------------
# Evaluation equivalence
# ---------------------------------------------------------------------


class TestLazyEagerEquivalence:
    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'id': [1, 1, 2],
                'name': ['Alice', None, 'Bob'],
                'city': [None, 'NYC', 'LA'],
            }
        )
        expr = first_non_null_per_group(['id'])
        eager = df.group_by(['id']).agg(expr)
        lazy = df.lazy().group_by(['id']).agg(expr).collect()
        assert_frame_equal(
            eager, lazy, check_row_order=False, check_column_order=False
        )


# ---------------------------------------------------------------------
# mode_then_first_per_group
# ---------------------------------------------------------------------


class TestModeThenFirstPerGroup:
    def test_majority_wins(self) -> None:
        df = pl.DataFrame({'id': [1, 1, 1], 'name': ['Al', 'Al', 'Bo']})
        result = df.group_by(['id']).agg(
            mode_then_first_per_group(['id'])
        )
        expected = pl.DataFrame({'id': [1], 'name': ['Al']})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_tie_resolves_to_input_order_first(self) -> None:
        # Determinism pin: mode().first() would flip across runs; the
        # implode-filter construction must stably return the first tied
        # value in input order.
        df = pl.DataFrame({'id': [1, 1, 1, 1], 'v': ['b', 'a', 'b', 'a']})
        result = df.group_by(['id']).agg(
            mode_then_first_per_group(['id'])
        )
        expected = pl.DataFrame({'id': [1], 'v': ['b']})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_nulls_excluded_from_count(self) -> None:
        # null appears 3x but is not a value; 'a' is the most common
        # non-null. Proves drop_nulls() precedes mode().
        df = pl.DataFrame(
            {'id': [1] * 6, 'x': ['a', 'a', None, None, None, 'b']},
            schema={'id': pl.Int64, 'x': pl.String},
        )
        result = df.group_by(['id']).agg(
            mode_then_first_per_group(['id'])
        )
        expected = pl.DataFrame({'id': [1], 'x': ['a']})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_numeric_even_tie_picks_real_value(self) -> None:
        # Selects, never computes: a numeric tie yields a value present in
        # the data, never the synthetic mean (15).
        df = pl.DataFrame({'id': [1, 1, 1, 1], 'n': [20, 10, 20, 10]})
        result = df.group_by(['id']).agg(
            mode_then_first_per_group(['id'])
        )
        expected = pl.DataFrame({'id': [1], 'n': [20]})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_all_null_field_stays_null(self) -> None:
        df = pl.DataFrame(
            {'id': [1, 1], 'name': [None, None]},
            schema={'id': pl.Int64, 'name': pl.String},
        )
        result = df.group_by(['id']).agg(
            mode_then_first_per_group(['id'])
        )
        expected = pl.DataFrame(
            {'id': [1], 'name': [None]},
            schema={'id': pl.Int64, 'name': pl.String},
        )
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_multi_column_key(self) -> None:
        df = pl.DataFrame(
            {
                'a': [1, 1, 1],
                'b': ['x', 'x', 'x'],
                'val': ['V', 'V', 'W'],
            }
        )
        result = df.group_by(['a', 'b']).agg(
            mode_then_first_per_group(['a', 'b'])
        )
        expected = pl.DataFrame({'a': [1], 'b': ['x'], 'val': ['V']})
        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_differs_from_first_non_null_sibling(self) -> None:
        # Locks that the two resolvers select differently: first-available
        # gives 'x'; most-common gives 'y'.
        df = pl.DataFrame({'id': [1, 1, 1], 'v': ['x', 'y', 'y']})
        first = df.group_by(['id']).agg(first_non_null_per_group(['id']))
        mode = df.group_by(['id']).agg(mode_then_first_per_group(['id']))
        assert_frame_equal(
            first,
            pl.DataFrame({'id': [1], 'v': ['x']}),
            check_row_order=False,
            check_column_order=False,
        )
        assert_frame_equal(
            mode,
            pl.DataFrame({'id': [1], 'v': ['y']}),
            check_row_order=False,
            check_column_order=False,
        )

    def test_no_deprecation_warning(self) -> None:
        # The implode form of is_in must not surface the 1.41 deprecation
        # warning; treat any DeprecationWarning as an error here.
        df = pl.DataFrame({'id': [1, 1, 1], 'v': ['a', 'a', 'b']})
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            df.group_by(['id']).agg(mode_then_first_per_group(['id']))

    def test_empty_key_columns_raises(self) -> None:
        with pytest.raises(
            ValueError, match='key_columns must not be empty'
        ):
            mode_then_first_per_group([])

    def test_missing_key_raises_on_apply(self) -> None:
        df = pl.DataFrame({'id': [1, 1], 'name': ['A', 'B']})
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.group_by(['nope']).agg(
                mode_then_first_per_group(['nope'])
            )

    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'id': [1, 1, 1, 2],
                'name': ['Al', 'Al', 'Bo', 'Cy'],
                'city': ['NYC', 'NYC', 'LA', None],
            }
        )
        expr = mode_then_first_per_group(['id'])
        eager = df.group_by(['id']).agg(expr)
        lazy = df.lazy().group_by(['id']).agg(expr).collect()
        assert_frame_equal(
            eager, lazy, check_row_order=False, check_column_order=False
        )
