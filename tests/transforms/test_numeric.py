# tests/transforms/test_numeric.py
"""Tests for atomic transforms in ``framesmith.transforms.numeric``.

Each transform is exercised in isolation through ``compose_column`` so
the tests also cover the integration point.
"""

import polars as pl
import pytest

from framesmith import (
    ExpressionTransform,
    accounting_parens_to_negative,
    cast_to_float64,
    compose_column,
    percent_to_fraction,
    remove_thousands_separators,
    trailing_minus_to_prefix,
)


def _apply(
    values: list[str | None], transform: ExpressionTransform
) -> pl.Series:
    """Run a single transform on a 1-column ``pl.String`` frame."""
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', [transform]))['x']


class TestAccountingParensToNegative:
    def test_simple_parens(self) -> None:
        result = _apply(['(123.45)'], accounting_parens_to_negative)
        assert result.to_list() == ['-123.45']

    def test_inner_whitespace_trimmed_by_pattern(self) -> None:
        # The PAREN_NEGATIVE_PATTERN consumes whitespace adjacent to
        # the parens, so the captured body is already trim.
        result = _apply(['( 123 )'], accounting_parens_to_negative)
        assert result.to_list() == ['-123']

    def test_comma_preserved_inside_parens(self) -> None:
        # Atomic: this transform does not touch thousands separators.
        # remove_thousands_separators is the next step in the recipe.
        result = _apply(['(1,234.56)'], accounting_parens_to_negative)
        assert result.to_list() == ['-1,234.56']

    def test_currency_preserved_in_isolation(self) -> None:
        # Atomic: in isolation, currency survives. Inside
        # NORMALIZE_NUMERIC, fold_to_ascii has already removed the
        # currency symbol before this transform runs, so the recipe
        # output is clean. This test documents the atomic behavior.
        result = _apply(['($1,234)'], accounting_parens_to_negative)
        assert result.to_list() == ['-$1,234']

    def test_non_parenthesized_unchanged(self) -> None:
        result = _apply(['123.45'], accounting_parens_to_negative)
        assert result.to_list() == ['123.45']

    def test_null_propagates(self) -> None:
        result = _apply([None], accounting_parens_to_negative)
        assert result.to_list() == [None]


class TestTrailingMinusToPrefix:
    def test_simple_trailing_minus(self) -> None:
        result = _apply(['500-'], trailing_minus_to_prefix)
        assert result.to_list() == ['-500']

    def test_trailing_minus_with_comma(self) -> None:
        result = _apply(['1,234.56-'], trailing_minus_to_prefix)
        assert result.to_list() == ['-1,234.56']

    def test_leading_minus_not_double_negated(self) -> None:
        # The pattern requires the first character to be non-minus, so
        # a value that already starts with '-' is left alone.
        result = _apply(['-100'], trailing_minus_to_prefix)
        assert result.to_list() == ['-100']

    def test_positive_value_unchanged(self) -> None:
        result = _apply(['100'], trailing_minus_to_prefix)
        assert result.to_list() == ['100']

    def test_null_propagates(self) -> None:
        result = _apply([None], trailing_minus_to_prefix)
        assert result.to_list() == [None]


class TestRemoveThousandsSeparators:
    def test_comma_removed(self) -> None:
        result = _apply(['1,234'], remove_thousands_separators)
        assert result.to_list() == ['1234']

    def test_internal_whitespace_removed(self) -> None:
        result = _apply(['1 234'], remove_thousands_separators)
        assert result.to_list() == ['1234']

    def test_comma_with_decimal_preserved(self) -> None:
        result = _apply(['1,234.56'], remove_thousands_separators)
        assert result.to_list() == ['1234.56']

    def test_surrounding_whitespace_also_removed(self) -> None:
        # Side effect of the [,\s] character class: surrounding
        # whitespace is also gone after this step. Documented here so
        # the behavior is locked in.
        result = _apply(['  1,234  '], remove_thousands_separators)
        assert result.to_list() == ['1234']

    def test_no_separators_unchanged(self) -> None:
        result = _apply(['1234'], remove_thousands_separators)
        assert result.to_list() == ['1234']

    def test_null_propagates(self) -> None:
        result = _apply([None], remove_thousands_separators)
        assert result.to_list() == [None]


class TestCastToFloat64:
    def test_parses_decimal(self) -> None:
        result = _apply(['123.45'], cast_to_float64)
        assert result.to_list() == [123.45]

    def test_parses_negative_integer(self) -> None:
        result = _apply(['-1234'], cast_to_float64)
        assert result.to_list() == [-1234.0]

    def test_empty_string_becomes_null(self) -> None:
        result = _apply([''], cast_to_float64)
        assert result.to_list() == [None]

    def test_non_numeric_becomes_null(self) -> None:
        result = _apply(['abc'], cast_to_float64)
        assert result.to_list() == [None]

    def test_comma_unparseable_becomes_null(self) -> None:
        # Demonstrates that cleaning must precede the cast — a raw
        # "1,234" is not directly parseable by Polars' Float64 reader.
        result = _apply(['1,234'], cast_to_float64)
        assert result.to_list() == [None]

    def test_null_propagates(self) -> None:
        result = _apply([None], cast_to_float64)
        assert result.to_list() == [None]

    def test_output_dtype_is_float64(self) -> None:
        result = _apply(['123.45'], cast_to_float64)
        assert result.dtype == pl.Float64


class TestPercentToFraction:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('12%', 0.12),
            ('50%', 0.5),
            ('100%', 1.0),
            ('0%', 0.0),
            ('-50%', -0.5),
            ('12.5%', 0.125),
            ('12', 12.0),
            ('-12', -12.0),
        ],
    )
    def test_parses_percent_and_bare_numbers(
        self, value: str, expected: float
    ) -> None:
        result = _apply([value], percent_to_fraction)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize('value', ['', '%', 'abc', 'abc%'])
    def test_unparseable_becomes_null(self, value: str) -> None:
        result = _apply([value], percent_to_fraction)
        assert result.to_list() == [None]

    def test_null_propagates(self) -> None:
        result = _apply([None], percent_to_fraction)
        assert result.to_list() == [None]

    def test_output_dtype_is_float64(self) -> None:
        result = _apply(['12%'], percent_to_fraction)
        assert result.dtype == pl.Float64
