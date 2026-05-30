# tests/transforms/test_case.py
"""Tests for case transforms in ``framesmith.transforms.case``.

Each transform is exercised in isolation through ``compose_column`` so
the tests also cover the integration point.
"""

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    replace_whitespace_with,
    to_lowercase,
    to_snake_case,
    to_titlecase,
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


class TestToLowercase:
    def test_mixed_case_becomes_lower(self) -> None:
        result = _apply(['Hello World'], to_lowercase)
        assert result.to_list() == ['hello world']

    def test_already_lowercase_unchanged(self) -> None:
        result = _apply(['already lower'], to_lowercase)
        assert result.to_list() == ['already lower']

    def test_digits_and_punctuation_unaffected(self) -> None:
        result = _apply(['ABC-123'], to_lowercase)
        assert result.to_list() == ['abc-123']

    def test_null_propagates(self) -> None:
        result = _apply([None], to_lowercase)
        assert result.to_list() == [None]


class TestToSnakeCase:
    def test_single_space_becomes_underscore(self) -> None:
        result = _apply(['hello world'], to_snake_case)
        assert result.to_list() == ['hello_world']

    def test_multiple_spaces_become_single_underscore(self) -> None:
        result = _apply(['hello   world'], to_snake_case)
        assert result.to_list() == ['hello_world']

    def test_already_snake_unchanged(self) -> None:
        result = _apply(['already_snake_case'], to_snake_case)
        assert result.to_list() == ['already_snake_case']

    def test_mixed_case_lowercased_and_underscored(self) -> None:
        result = _apply(['Hello World'], to_snake_case)
        assert result.to_list() == ['hello_world']

    def test_all_caps_lowercased_and_underscored(self) -> None:
        result = _apply(['HELLO WORLD'], to_snake_case)
        assert result.to_list() == ['hello_world']

    def test_internal_caps_lowercased(self) -> None:
        result = _apply(['MixedCase Word'], to_snake_case)
        assert result.to_list() == ['mixedcase_word']

    def test_null_propagates(self) -> None:
        result = _apply([None], to_snake_case)
        assert result.to_list() == [None]

    def test_remains_a_named_function_after_refactor(self) -> None:
        # to_snake_case must stay a named function even though it now
        # composes to_lowercase with the factory's underscore closure.
        # If a future refactor binds it directly to a closure, this
        # fires.
        assert to_snake_case.__name__ == 'to_snake_case'

    def test_matches_lowercase_then_replace_whitespace(self) -> None:
        # New faithfulness pin: to_snake_case == to_lowercase composed
        # with replace_whitespace_with('_'). Inputs include uppercase so
        # the lowercase step is actually exercised.
        underscore = replace_whitespace_with('_')
        inputs = [
            'Hello World',
            'HELLO   WORLD',
            'Already_Snake',
            ' Leading Trailing ',
            'Tabs\tAnd\nNewlines',
            None,
        ]
        df = pl.DataFrame({'x': inputs}, schema={'x': pl.String})
        from_snake = df.with_columns(compose_column('x', [to_snake_case]))
        from_manual = df.with_columns(
            compose_column('x', [to_lowercase, underscore])
        )
        assert_frame_equal(from_snake, from_manual)


class TestToTitlecase:
    def test_lowercase_words_titlecased(self) -> None:
        result = _apply(['john smith'], to_titlecase)
        assert result.to_list() == ['John Smith']

    def test_all_caps_titlecased(self) -> None:
        result = _apply(['JANE DOE'], to_titlecase)
        assert result.to_list() == ['Jane Doe']

    def test_acronym_not_fixed(self) -> None:
        # Documents the known limit: title casing lowercases the tail of
        # each word, so acronyms are mangled. apply_replacements fixes it.
        result = _apply(['rep lob'], to_titlecase)
        assert result.to_list() == ['Rep Lob']

    def test_null_propagates(self) -> None:
        result = _apply([None], to_titlecase)
        assert result.to_list() == [None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['john smith'], to_titlecase)
        assert result.dtype == pl.String


class TestNullPropagationBatch:
    """Batch null propagation across this module's transforms."""

    def test_null_propagates_for_module_transforms(self) -> None:
        transforms: list[ExpressionTransform] = [
            to_lowercase,
            to_snake_case,
            to_titlecase,
        ]
        expected = pl.Series('x', [None], dtype=pl.String)
        for transform in transforms:
            result = _apply([None], transform)
            assert_series_equal(result, expected)
