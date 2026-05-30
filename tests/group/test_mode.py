# tests/group/test_mode.py
"""Tests for the shared mode primitive in ``framesmith.group.mode``."""

import polars as pl

from framesmith.group.mode import first_mode_in_order


class TestFirstModeInOrder:
    def test_most_common_wins(self) -> None:
        df = pl.DataFrame({'g': [1, 1, 1], 'v': ['a', 'a', 'b']})
        result = df.group_by('g').agg(first_mode_in_order(pl.col('v')))
        assert result['v'].to_list() == ['a']

    def test_tie_breaks_to_first_in_input_order(self) -> None:
        df = pl.DataFrame({'g': [1, 1, 1, 1], 'v': ['b', 'a', 'b', 'a']})
        result = df.group_by('g').agg(first_mode_in_order(pl.col('v')))
        assert result['v'].to_list() == ['b']

    def test_nulls_excluded(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1, 1], 'v': [None, None, 'x']}, schema={'g': pl.Int64, 'v': pl.String}
        )
        result = df.group_by('g').agg(first_mode_in_order(pl.col('v')))
        assert result['v'].to_list() == ['x']

    def test_all_null_yields_null(self) -> None:
        df = pl.DataFrame(
            {'g': [1, 1], 'v': [None, None]}, schema={'g': pl.Int64, 'v': pl.String}
        )
        result = df.group_by('g').agg(first_mode_in_order(pl.col('v')))
        assert result['v'].to_list() == [None]
