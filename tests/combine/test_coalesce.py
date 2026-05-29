# tests/combine/test_coalesce.py
"""Tests for ``framesmith.combine.coalesce_blank_columns``.

Imports go through the directory's public surface
(``from framesmith.combine import ...``), not the internal ``coalesce``
module, so the tests exercise the same contract callers see.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith.combine import coalesce_blank_columns
from framesmith.transforms import nullify_blank_strings

# ---------------------------------------------------------------------
# Blank-aware coalescing
# ---------------------------------------------------------------------


class TestBlankCoalesce:
    def test_two_column_basics(self) -> None:
        df = pl.DataFrame(
            {'primary': ['John', None, ' '], 'fallback': ['X', 'Y', 'Z']}
        )
        result = df.with_columns(
            coalesce_blank_columns(['primary', 'fallback'], 'out')
        )
        assert result['out'].to_list() == ['John', 'Y', 'Z']

    def test_empty_string_is_blank(self) -> None:
        df = pl.DataFrame({'primary': [''], 'fallback': ['X']})
        result = df.with_columns(
            coalesce_blank_columns(['primary', 'fallback'], 'out')
        )
        assert result['out'].to_list() == ['X']

    def test_kept_value_returned_raw(self) -> None:
        # Blank test is comparison-only: a kept value keeps its
        # surrounding whitespace.
        df = pl.DataFrame({'primary': [' John '], 'fallback': ['X']})
        result = df.with_columns(
            coalesce_blank_columns(['primary', 'fallback'], 'out')
        )
        assert result['out'].to_list() == [' John ']

    def test_n_ary_first_non_blank_wins(self) -> None:
        df = pl.DataFrame(
            {'a': [None], 'b': [' '], 'c': ['Z']},
            schema={'a': pl.String, 'b': pl.String, 'c': pl.String},
        )
        result = df.with_columns(
            coalesce_blank_columns(['a', 'b', 'c'], 'out')
        )
        assert result['out'].to_list() == ['Z']

    def test_n_ary_priority_order(self) -> None:
        df = pl.DataFrame({'a': ['A'], 'b': ['B'], 'c': ['C']})
        result = df.with_columns(
            coalesce_blank_columns(['a', 'b', 'c'], 'out')
        )
        assert result['out'].to_list() == ['A']

    def test_all_blank_is_null(self) -> None:
        df = pl.DataFrame(
            {'a': [None], 'b': [' '], 'c': ['']},
            schema={'a': pl.String, 'b': pl.String, 'c': pl.String},
        )
        result = df.with_columns(
            coalesce_blank_columns(['a', 'b', 'c'], 'out')
        )
        assert result['out'].to_list() == [None]

    def test_single_column_degenerate(self) -> None:
        df = pl.DataFrame({'a': ['x', ' ', None]}, schema={'a': pl.String})
        result = df.with_columns(coalesce_blank_columns(['a'], 'out'))
        assert result['out'].to_list() == ['x', None, None]

    def test_matches_manual_coalesce_of_nullified_blanks(self) -> None:
        # Reuse pin: locks the decomposition to
        # pl.coalesce([nullify_blank_strings(col) for col in cols]).
        df = pl.DataFrame(
            {
                'a': ['A', None, ' ', '', 'keep '],
                'b': ['B', 'b', None, '  ', None],
            },
            schema={'a': pl.String, 'b': pl.String},
        )
        expr = coalesce_blank_columns(['a', 'b'], 'out')
        manual = pl.coalesce(
            [
                nullify_blank_strings(pl.col('a')),
                nullify_blank_strings(pl.col('b')),
            ]
        ).alias('out')
        assert_frame_equal(
            df.with_columns(expr), df.with_columns(manual)
        )


# ---------------------------------------------------------------------
# Output-name semantics, dtype, and error conditions
# ---------------------------------------------------------------------


class TestOutputNameAndErrors:
    def test_output_added_alongside_originals(self) -> None:
        df = pl.DataFrame({'a': ['x'], 'b': ['y']})
        result = df.with_columns(coalesce_blank_columns(['a', 'b'], 'out'))
        assert result.columns == ['a', 'b', 'out']

    def test_output_dtype_is_string(self) -> None:
        df = pl.DataFrame({'a': ['x'], 'b': ['y']})
        result = df.with_columns(coalesce_blank_columns(['a', 'b'], 'out'))
        assert result['out'].dtype == pl.String

    def test_output_name_collision_overwrites(self) -> None:
        # with_columns overwrites an existing column of the same name.
        df = pl.DataFrame({'a': [' '], 'b': ['y']})
        result = df.with_columns(coalesce_blank_columns(['a', 'b'], 'a'))
        assert result.columns == ['a', 'b']
        assert result['a'].to_list() == ['y']

    def test_empty_column_names_raises(self) -> None:
        with pytest.raises(
            ValueError, match='column_names must not be empty'
        ):
            coalesce_blank_columns([], 'out')

    def test_missing_column_raises_on_apply(self) -> None:
        df = pl.DataFrame({'a': ['x']})
        expr = coalesce_blank_columns(['a', 'nope'], 'out')  # builds fine
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.with_columns(expr)

    def test_non_string_column_raises_on_apply(self) -> None:
        df = pl.DataFrame({'a': [1, 2]}, schema={'a': pl.Int64})
        expr = coalesce_blank_columns(['a'], 'out')  # builds fine
        with pytest.raises(pl.exceptions.InvalidOperationError):
            df.with_columns(expr)


# ---------------------------------------------------------------------
# Evaluation equivalence
# ---------------------------------------------------------------------


class TestLazyEagerEquivalence:
    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {'primary': ['John', None, ' '], 'fallback': ['X', 'Y', 'Z']}
        )
        expr = coalesce_blank_columns(['primary', 'fallback'], 'out')
        assert_frame_equal(
            df.with_columns(expr), df.lazy().with_columns(expr).collect()
        )
