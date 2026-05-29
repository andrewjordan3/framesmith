# tests/combine/test_concat.py
"""Tests for ``framesmith.combine.combine_columns``.

Imports go through the directory's public surface
(``from framesmith.combine import ...``), not the internal ``concat``
module, so the tests exercise the same contract callers see.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith.combine import combine_columns

# ---------------------------------------------------------------------
# Basic concatenation and separators
# ---------------------------------------------------------------------


class TestBasicConcat:
    def test_concatenates_with_default_space(self) -> None:
        df = pl.DataFrame(
            {'first': ['John', 'Jane'], 'last': ['Doe', 'Smith']}
        )
        result = df.with_columns(
            combine_columns(['first', 'last'], 'full_name')
        )
        assert result['full_name'].to_list() == ['John Doe', 'Jane Smith']

    def test_originals_still_present(self) -> None:
        df = pl.DataFrame(
            {'first': ['John'], 'last': ['Doe']}
        )
        result = df.with_columns(
            combine_columns(['first', 'last'], 'full_name')
        )
        assert result.columns == ['first', 'last', 'full_name']

    @pytest.mark.parametrize(
        ('separator', 'expected'),
        [
            (', ', 'John, Doe'),
            ('', 'JohnDoe'),
            (' - ', 'John - Doe'),
        ],
    )
    def test_separator_variations(
        self, separator: str, expected: str
    ) -> None:
        df = pl.DataFrame({'first': ['John'], 'last': ['Doe']})
        result = df.with_columns(
            combine_columns(['first', 'last'], 'out', separator)
        )
        assert result['out'].to_list() == [expected]

    def test_three_columns_in_order(self) -> None:
        df = pl.DataFrame(
            {'a': ['1'], 'b': ['2'], 'c': ['3']}
        )
        result = df.with_columns(
            combine_columns(['a', 'b', 'c'], 'out', '-')
        )
        assert result['out'].to_list() == ['1-2-3']

    def test_single_column_degenerate_passthrough(self) -> None:
        df = pl.DataFrame({'name': ['x', 'y']})
        result = df.with_columns(combine_columns(['name'], 'out'))
        assert result['out'].to_list() == ['x', 'y']


# ---------------------------------------------------------------------
# Null handling and type coercion
# ---------------------------------------------------------------------


class TestNullAndCoercion:
    def test_null_skipped_no_stray_separator(self) -> None:
        df = pl.DataFrame(
            {'first': ['John', None], 'last': [None, 'Smith']}
        )
        result = df.with_columns(
            combine_columns(['first', 'last'], 'out')
        )
        assert result['out'].to_list() == ['John', 'Smith']

    def test_all_null_row_is_empty_string(self) -> None:
        # Pinned polars 1.41 behavior: ignore_nulls=True over an all-null
        # row yields '' (not null), deliberately not special-cased.
        df = pl.DataFrame(
            {'a': [None], 'b': [None]}, schema={'a': pl.String, 'b': pl.String}
        )
        result = df.with_columns(combine_columns(['a', 'b'], 'out'))
        assert result['out'].to_list() == ['']

    def test_non_string_column_stringified(self) -> None:
        df = pl.DataFrame({'name': ['x'], 'age': [30]})
        result = df.with_columns(
            combine_columns(['name', 'age'], 'label')
        )
        assert result['label'].to_list() == ['x 30']

    def test_output_dtype_is_string(self) -> None:
        df = pl.DataFrame({'first': ['John'], 'last': ['Doe']})
        result = df.with_columns(
            combine_columns(['first', 'last'], 'out')
        )
        assert result['out'].dtype == pl.String


# ---------------------------------------------------------------------
# Output-name semantics and error conditions
# ---------------------------------------------------------------------


class TestOutputNameAndErrors:
    def test_output_name_collision_overwrites(self) -> None:
        # with_columns overwrites an existing column of the same name.
        df = pl.DataFrame({'a': ['x'], 'b': ['y']})
        result = df.with_columns(combine_columns(['a', 'b'], 'a'))
        assert result.columns == ['a', 'b']
        assert result['a'].to_list() == ['x y']

    def test_empty_column_names_raises(self) -> None:
        with pytest.raises(
            ValueError, match='column_names must not be empty'
        ):
            combine_columns([], 'out')

    def test_missing_column_raises_on_apply(self) -> None:
        df = pl.DataFrame({'a': ['x']})
        expr = combine_columns(['a', 'nope'], 'out')  # builds fine
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.with_columns(expr)


# ---------------------------------------------------------------------
# Evaluation equivalence
# ---------------------------------------------------------------------


class TestLazyEagerEquivalence:
    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {'first': ['John', None, 'Jane'], 'last': ['Doe', 'Smith', None]}
        )
        expr = combine_columns(['first', 'last'], 'full_name')
        assert_frame_equal(
            df.with_columns(expr), df.lazy().with_columns(expr).collect()
        )
