# tests/columns/test_names.py
"""Tests for ``framesmith.columns.names``."""

import polars as pl
import pytest
from polars.testing import assert_series_equal

from framesmith.columns.names import remove_trailing_jr


def _apply(values: list[str | None]) -> pl.Series:
    df = pl.DataFrame({'name': values}, schema={'name': pl.String})
    return df.with_columns(remove_trailing_jr('name'))['name']


class TestRemoveTrailingJr:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('John Smith Jr', 'John Smith'),
            ('John Smith Jr.', 'John Smith'),
            ('John Smith, Jr.', 'John Smith'),
            ('John Smith, Jr', 'John Smith'),
            ('John Smith jr', 'John Smith'),
            ('John Smith JR', 'John Smith'),
        ],
    )
    def test_removes_trailing_jr(self, value: str, expected: str) -> None:
        result = _apply([value])
        assert result.to_list() == [expected]

    def test_interior_jr_is_unchanged(self) -> None:
        result = _apply(['Jr Bakery'])
        assert result.to_list() == ['Jr Bakery']

    def test_name_without_suffix_is_unchanged(self) -> None:
        result = _apply(['Smith'])
        assert result.to_list() == ['Smith']

    def test_null_in_null_out(self) -> None:
        result = _apply([None])
        assert result.to_list() == [None]

    def test_output_is_auto_aliased_to_input_column_name(self) -> None:
        df = pl.DataFrame({'full_name': ['John Smith Jr.']})
        result = df.with_columns(remove_trailing_jr('full_name'))
        assert result.columns == ['full_name']

    def test_batch_application(self) -> None:
        df = pl.DataFrame(
            {
                'name': [
                    'John Smith Jr',
                    'John Smith, Jr.',
                    'Jr Bakery',
                    'Smith',
                    None,
                ]
            }
        )
        result = df.with_columns(remove_trailing_jr('name'))
        expected = pl.Series(
            'name',
            ['John Smith', 'John Smith', 'Jr Bakery', 'Smith', None],
        )
        assert_series_equal(result['name'], expected)
