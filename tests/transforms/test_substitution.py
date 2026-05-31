# tests/transforms/test_substitution.py
"""Tests for substitution transforms in ``framesmith.transforms.substitution``.

Each transform is exercised in isolation through ``compose_column`` so
the tests also cover the integration point.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    apply_replacements,
    periods_to_spaces,
    remove_apostrophes,
    remove_periods,
    replace_ampersand_with_and,
    underscores_to_spaces,
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


class TestReplaceAmpersandWithAnd:
    def test_replaces_ampersand(self) -> None:
        result = _apply(['Sales & Service'], replace_ampersand_with_and)
        assert result.to_list() == ['Sales and Service']

    def test_case_preserved(self) -> None:
        result = _apply(['SALES & SERVICE'], replace_ampersand_with_and)
        assert result.to_list() == ['SALES and SERVICE']

    def test_null_propagates(self) -> None:
        result = _apply([None], replace_ampersand_with_and)
        assert result.to_list() == [None]


class TestRemoveApostrophes:
    def test_removes_apostrophe(self) -> None:
        result = _apply(["O'Brien"], remove_apostrophes)
        assert result.to_list() == ['OBrien']

    def test_null_propagates(self) -> None:
        result = _apply([None], remove_apostrophes)
        assert result.to_list() == [None]


class TestRemovePeriods:
    def test_removes_period(self) -> None:
        result = _apply(['St.'], remove_periods)
        assert result.to_list() == ['St']

    def test_period_treated_as_literal_not_regex_wildcard(self) -> None:
        # If '.' were interpreted as regex any-char, 'a.b' would become
        # an empty string. literal=True keeps it a literal dot.
        result = _apply(['a.b'], remove_periods)
        assert result.to_list() == ['ab']

    def test_null_propagates(self) -> None:
        result = _apply([None], remove_periods)
        assert result.to_list() == [None]


class TestPeriodsToSpaces:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('john.doe', 'john doe'),
            ('U.S.A', 'U S A'),
            ('no_periods', 'no_periods'),
            # Atomic: one period → one space, even in runs. No collapse.
            ('..leading', '  leading'),
            ('john..doe', 'john  doe'),
            ('', ''),
        ],
    )
    def test_replaces_each_period_with_one_space(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], periods_to_spaces)
        assert result.to_list() == [expected]

    def test_null_propagates(self) -> None:
        result = _apply([None], periods_to_spaces)
        assert result.to_list() == [None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['john.doe'], periods_to_spaces)
        assert result.dtype == pl.String


class TestUnderscoresToSpaces:
    def test_single_underscore_becomes_space(self) -> None:
        result = _apply(['john_smith'], underscores_to_spaces)
        assert result.to_list() == ['john smith']

    def test_runs_preserved_not_collapsed(self) -> None:
        # Atomic: one underscore → one space, even in runs.
        result = _apply(['a__b'], underscores_to_spaces)
        assert result.to_list() == ['a  b']

    def test_no_underscores_unchanged(self) -> None:
        result = _apply(['plain text'], underscores_to_spaces)
        assert result.to_list() == ['plain text']

    def test_null_propagates(self) -> None:
        result = _apply([None], underscores_to_spaces)
        assert result.to_list() == [None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['john_smith'], underscores_to_spaces)
        assert result.dtype == pl.String


class TestApplyReplacements:
    def test_single_replacement(self) -> None:
        transform = apply_replacements({'Lob': 'LOB'})
        result = _apply(['Primary Lob'], transform)
        assert result.to_list() == ['Primary LOB']

    def test_multiple_keys(self) -> None:
        transform = apply_replacements({'Lob': 'LOB', 'Ccms': 'CCMS'})
        result = _apply(['Ccms Lob'], transform)
        assert result.to_list() == ['CCMS LOB']

    def test_substring_over_match_documented(self) -> None:
        # Literal substring, not word-boundary: 'Lob' inside 'Lobster'
        # is rewritten too. This is the documented caveat.
        transform = apply_replacements({'Lob': 'LOB'})
        result = _apply(['Lobster'], transform)
        assert result.to_list() == ['LOBster']

    def test_empty_map_raises(self) -> None:
        with pytest.raises(
            ValueError, match='replacements must not be empty'
        ):
            apply_replacements({})

    def test_factory_returns_callable(self) -> None:
        assert callable(apply_replacements({'a': 'b'}))

    def test_null_propagates(self) -> None:
        transform = apply_replacements({'Lob': 'LOB'})
        result = _apply([None], transform)
        assert result.to_list() == [None]

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['Primary Lob', 'Lobster', None]}, schema={'x': pl.String}
        )
        transform = apply_replacements({'Lob': 'LOB'})
        expr = compose_column('x', [transform])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestNullPropagationBatch:
    """Batch null propagation across this module's transforms."""

    def test_null_propagates_for_module_transforms(self) -> None:
        transforms: list[ExpressionTransform] = [
            replace_ampersand_with_and,
            remove_apostrophes,
            remove_periods,
            periods_to_spaces,
            underscores_to_spaces,
            apply_replacements({'x': 'y'}),
        ]
        expected = pl.Series('x', [None], dtype=pl.String)
        for transform in transforms:
            result = _apply([None], transform)
            assert_series_equal(result, expected)
