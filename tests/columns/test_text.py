# tests/columns/test_text.py
"""Tests for ``framesmith.columns.text``.

Some inputs intentionally embed Unicode characters that ruff's RUF001
rule flags as visually ambiguous (fullwidth digits/letters, en-dashes,
no-break spaces, smart quotes). Those lines carry an inline
``# noqa: RUF001`` because the ambiguity is exactly what the test
exercises.
"""

import polars as pl
import pytest
from polars.testing import assert_series_equal

from framesmith.columns.text import normalize_text, to_snake_case


def _apply(column: str, values: list[str | None], expr: pl.Expr) -> pl.Series:
    """Helper: build a 1-column DataFrame, apply ``expr``, return series.

    An explicit ``pl.String`` schema avoids dtype inference failing on
    all-null inputs.
    """
    df = pl.DataFrame({column: values}, schema={column: pl.String})
    return df.with_columns(expr)[column]


class TestNormalizeTextNfkc:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            # Fullwidth digits 1, 2, 3 (U+FF11, U+FF12, U+FF13)
            ('１２３', '123'),  # noqa: RUF001
            # LATIN SMALL LIGATURE FI (U+FB01)
            ('ﬁnish', 'finish'),
            # Fullwidth latin A, B, C (U+FF21, U+FF22, U+FF23)
            ('ＡＢＣ', 'ABC'),  # noqa: RUF001
        ],
    )
    def test_nfkc_normalizes_compatibility_forms(
        self, value: str, expected: str
    ) -> None:
        result = _apply('x', [value], normalize_text('x'))
        assert result.to_list() == [expected]


class TestNormalizeTextAsciiFolding:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            # Smart double quotes U+201C, U+201D
            ('“quoted”', '"quoted"'),
            # Em-dash U+2014
            ('A—B', 'A-B'),
            # En-dash U+2013
            ('A–B', 'A-B'),  # noqa: RUF001
            # Dollar sign removed
            ('Price $100', 'Price 100'),
            # Euro sign U+20AC removed
            ('€50', '50'),
            # Registered symbol U+00AE removed
            ('Acme®', 'Acme'),
            # Copyright symbol U+00A9 removed
            ('©Co', 'Co'),
        ],
    )
    def test_ascii_compat_folding(self, value: str, expected: str) -> None:
        result = _apply('x', [value], normalize_text('x'))
        assert result.to_list() == [expected]

    def test_trademark_decomposes_to_tm_via_nfkc(self) -> None:
        # U+2122 TRADE MARK SIGN is decomposed by NFKC to the letters
        # 'TM' before the ASCII compatibility map sees it. Documented
        # here so the behavior does not silently change.
        result = _apply('x', ['Brand™'], normalize_text('x'))
        assert result.to_list() == ['BrandTM']


class TestNormalizeTextWhitespace:
    def test_tabs_become_spaces(self) -> None:
        result = _apply('x', ['a\tb'], normalize_text('x'))
        assert result.to_list() == ['a b']

    def test_no_break_space_becomes_space(self) -> None:
        # U+00A0 NO-BREAK SPACE between 'a' and 'b'.
        result = _apply('x', ['a b'], normalize_text('x'))  # noqa: RUF001
        assert result.to_list() == ['a b']

    def test_multiple_spaces_collapsed(self) -> None:
        result = _apply('x', ['a   b'], normalize_text('x'))
        assert result.to_list() == ['a b']

    def test_leading_and_trailing_whitespace_stripped(self) -> None:
        result = _apply('x', ['  hello  '], normalize_text('x'))
        assert result.to_list() == ['hello']


class TestNormalizeTextAmpersand:
    def test_ampersand_expanded(self) -> None:
        result = _apply('x', ['Sales & Service'], normalize_text('x'))
        assert result.to_list() == ['Sales and Service']

    def test_ampersand_expansion_preserves_case(self) -> None:
        result = _apply('x', ['SALES & SERVICE'], normalize_text('x'))
        assert result.to_list() == ['SALES and SERVICE']


class TestNormalizeTextApostropheAndPeriod:
    def test_apostrophe_removed(self) -> None:
        result = _apply('x', ["O'Brien"], normalize_text('x'))
        assert result.to_list() == ['OBrien']

    def test_period_removed(self) -> None:
        result = _apply('x', ['St.'], normalize_text('x'))
        assert result.to_list() == ['St']

    def test_smart_apostrophe_removed(self) -> None:
        # U+2019 RIGHT SINGLE QUOTATION MARK folds to ASCII apostrophe
        # in stage 2, then is stripped in stage 6.
        result = _apply('x', ['O’Brien'], normalize_text('x'))  # noqa: RUF001
        assert result.to_list() == ['OBrien']


class TestNormalizeTextNullSemantics:
    def test_null_in_null_out(self) -> None:
        result = _apply('x', [None], normalize_text('x'))
        assert result.to_list() == [None]

    @pytest.mark.parametrize('value', ['', ' ', '   ', '\t', '\n', ' \t\n '])
    def test_blank_or_whitespace_only_becomes_null(self, value: str) -> None:
        result = _apply('x', [value], normalize_text('x'))
        assert result.to_list() == [None]


class TestNormalizeTextAlias:
    def test_output_is_auto_aliased_to_input_column_name(self) -> None:
        df = pl.DataFrame({'customer_name': ['  hello  ']})
        result = df.with_columns(normalize_text('customer_name'))
        assert result.columns == ['customer_name']

    def test_preserves_other_columns(self) -> None:
        df = pl.DataFrame({'name': ['  hello  '], 'id': [1]})
        result = df.with_columns(normalize_text('name'))
        assert result.columns == ['name', 'id']
        assert result['name'].to_list() == ['hello']
        assert result['id'].to_list() == [1]


class TestToSnakeCase:
    def test_single_space_becomes_underscore(self) -> None:
        result = _apply('x', ['hello world'], to_snake_case('x'))
        assert result.to_list() == ['hello_world']

    def test_multiple_spaces_become_single_underscore(self) -> None:
        result = _apply('x', ['hello   world'], to_snake_case('x'))
        assert result.to_list() == ['hello_world']

    def test_already_snake_cased_input_is_unchanged(self) -> None:
        result = _apply('x', ['already_snake_case'], to_snake_case('x'))
        assert result.to_list() == ['already_snake_case']

    def test_null_in_null_out(self) -> None:
        result = _apply('x', [None], to_snake_case('x'))
        assert result.to_list() == [None]

    def test_output_is_auto_aliased_to_input_column_name(self) -> None:
        df = pl.DataFrame({'label': ['a b']})
        result = df.with_columns(to_snake_case('label'))
        assert result.columns == ['label']


class TestComposability:
    def test_normalize_then_lowercase_then_snake_case(self) -> None:
        df = pl.DataFrame({'col': ['Sales & Service']})
        expr = (
            normalize_text('col')
            .str.to_lowercase()
            .str.replace_all(r'\s+', '_')
            .alias('col')
        )
        result = df.with_columns(expr)
        expected = pl.Series('col', ['sales_and_service'])
        assert_series_equal(result['col'], expected)
