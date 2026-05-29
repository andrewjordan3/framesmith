# tests/transforms/test_text.py
"""Tests for atomic transforms in ``framesmith.transforms.text``.

Each transform is exercised in isolation through ``compose_column`` so
the tests also cover the integration point. Lines that embed visually
ambiguous Unicode characters carry an inline ``# noqa: RUF001`` — the
ambiguity is what the test exercises.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    collapse_whitespace,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    periods_to_spaces,
    remove_apostrophes,
    remove_periods,
    replace_ampersand_with_and,
    replace_whitespace_with,
    strip_whitespace,
    to_lowercase,
    to_snake_case,
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


class TestNormalizeUnicodeNfkc:
    def test_fullwidth_digits_become_ascii(self) -> None:
        # Fullwidth digits 1, 2, 3 (U+FF11, U+FF12, U+FF13)
        result = _apply(['１２３'], normalize_unicode_nfkc)  # noqa: RUF001
        assert result.to_list() == ['123']

    def test_fi_ligature_decomposed(self) -> None:
        # LATIN SMALL LIGATURE FI (U+FB01)
        result = _apply(['ﬁnish'], normalize_unicode_nfkc)
        assert result.to_list() == ['finish']

    def test_trademark_decomposes_to_tm(self) -> None:
        # U+2122 TRADE MARK SIGN → letters 'TM'. Locks in the
        # behavior since fold_to_ascii would never see U+2122 once
        # NFKC has decomposed it.
        result = _apply(['Brand™'], normalize_unicode_nfkc)
        assert result.to_list() == ['BrandTM']

    def test_null_propagates(self) -> None:
        result = _apply([None], normalize_unicode_nfkc)
        assert result.to_list() == [None]


class TestFoldToAscii:
    # Inputs here are deliberately plain ASCII or already-NFKC-form so
    # that the test isolates the ASCII map and not NFKC behavior.
    def test_smart_quotes_become_ascii(self) -> None:
        # U+201C, U+201D smart double quotes
        result = _apply(['“quoted”'], fold_to_ascii)
        assert result.to_list() == ['"quoted"']

    def test_em_dash_becomes_hyphen(self) -> None:
        # U+2014 EM DASH
        result = _apply(['A—B'], fold_to_ascii)
        assert result.to_list() == ['A-B']

    def test_currency_symbol_removed(self) -> None:
        # U+20AC EURO SIGN
        result = _apply(['€50'], fold_to_ascii)
        assert result.to_list() == ['50']

    def test_null_propagates(self) -> None:
        result = _apply([None], fold_to_ascii)
        assert result.to_list() == [None]


class TestCollapseWhitespace:
    def test_multiple_internal_spaces_collapsed(self) -> None:
        result = _apply(['a   b'], collapse_whitespace)
        assert result.to_list() == ['a b']

    def test_tabs_and_newlines_collapsed_to_space(self) -> None:
        result = _apply(['a\t\nb'], collapse_whitespace)
        assert result.to_list() == ['a b']

    def test_leading_and_trailing_whitespace_not_stripped(self) -> None:
        # Boundary proof: collapse is not strip. The runs at the ends
        # collapse to a single space each, but they are not removed.
        result = _apply(['   a   b   '], collapse_whitespace)
        assert result.to_list() == [' a b ']

    def test_null_propagates(self) -> None:
        result = _apply([None], collapse_whitespace)
        assert result.to_list() == [None]


class TestStripWhitespace:
    def test_leading_and_trailing_stripped(self) -> None:
        result = _apply(['  hello  '], strip_whitespace)
        assert result.to_list() == ['hello']

    def test_interior_runs_not_collapsed(self) -> None:
        # Boundary proof: strip is not collapse. Interior runs survive.
        result = _apply(['  a   b  '], strip_whitespace)
        assert result.to_list() == ['a   b']

    def test_null_propagates(self) -> None:
        result = _apply([None], strip_whitespace)
        assert result.to_list() == [None]


class TestNullifyBlankStrings:
    def test_empty_string_becomes_null(self) -> None:
        result = _apply([''], nullify_blank_strings)
        assert result.to_list() == [None]

    def test_whitespace_only_becomes_null(self) -> None:
        result = _apply(['   '], nullify_blank_strings)
        assert result.to_list() == [None]

    def test_tab_only_becomes_null(self) -> None:
        result = _apply(['\t'], nullify_blank_strings)
        assert result.to_list() == [None]

    def test_non_blank_unchanged(self) -> None:
        result = _apply([' hello '], nullify_blank_strings)
        assert result.to_list() == [' hello ']

    def test_existing_null_unchanged(self) -> None:
        result = _apply([None], nullify_blank_strings)
        assert result.to_list() == [None]


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


class TestReplaceWhitespaceWith:
    @pytest.mark.parametrize(
        ('separator', 'expected'),
        [
            ('_', 'hello_world'),
            ('-', 'hello-world'),
            ('.', 'hello.world'),
            ('__', 'hello__world'),
            ('', 'helloworld'),
            (' ', 'hello world'),
        ],
    )
    def test_separator_variations(
        self, separator: str, expected: str
    ) -> None:
        transform = replace_whitespace_with(separator)
        result = _apply(['hello world'], transform)
        assert result.to_list() == [expected]

    def test_multiple_spaces_collapse_to_one_separator(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['hello   world'], transform)
        assert result.to_list() == ['hello_world']

    def test_tab_replaced(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['a\tb'], transform)
        assert result.to_list() == ['a_b']

    def test_newline_replaced(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['a\nb'], transform)
        assert result.to_list() == ['a_b']

    def test_mixed_tabs_and_spaces_collapse_to_one_separator(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['a \t b'], transform)
        assert result.to_list() == ['a_b']

    def test_leading_and_trailing_whitespace_replaced_not_stripped(
        self,
    ) -> None:
        # Atomic-behavior contract: ends become separators just like
        # interior whitespace.
        transform = replace_whitespace_with('_')
        result = _apply([' hello world '], transform)
        assert result.to_list() == ['_hello_world_']

    def test_no_interior_text_change(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['HelloWorld'], transform)
        assert result.to_list() == ['HelloWorld']

    def test_null_propagates(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply([None], transform)
        assert result.to_list() == [None]

    def test_factory_returns_callable(self) -> None:
        transform = replace_whitespace_with('_')
        assert callable(transform)

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['hello world', 'a  b', ' edges ', None]},
            schema={'x': pl.String},
        )
        transform = replace_whitespace_with('_')
        expr = compose_column('x', [transform])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestNullPropagationBatch:
    """Batch null propagation across every transform."""

    def test_null_propagates_for_every_transform(self) -> None:
        transforms: list[ExpressionTransform] = [
            normalize_unicode_nfkc,
            fold_to_ascii,
            collapse_whitespace,
            strip_whitespace,
            nullify_blank_strings,
            periods_to_spaces,
            replace_ampersand_with_and,
            remove_apostrophes,
            remove_periods,
            to_lowercase,
            to_snake_case,
        ]
        expected = pl.Series('x', [None], dtype=pl.String)
        for transform in transforms:
            result = _apply([None], transform)
            assert_series_equal(result, expected)
