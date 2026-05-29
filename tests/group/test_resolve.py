# tests/group/test_resolve.py
"""Tests for ``framesmith.group.first_non_null_per_group``.

Imports go through the directory's public surface
(``from framesmith.group import ...``), not the internal ``resolve``
module. Output row and column order are unclaimed, so every frame
comparison passes ``check_row_order=False`` / ``check_column_order=False``.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith.group import first_non_null_per_group

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
