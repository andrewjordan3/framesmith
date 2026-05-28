# tests/test_recipes.py
"""Tests for ``framesmith.recipes``.

Faithfulness suite: ``NORMALIZE_TEXT`` composed via ``compose_column``
must reproduce the legacy ``normalize_text`` behavior exactly on every
documented input/output pair. Lines that embed visually ambiguous
Unicode characters carry an inline ``# noqa: RUF001`` — the ambiguity
is what the test exercises.
"""

from collections.abc import Sequence

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import (
    CLEAN_NUMERIC_STRING,
    NORMALIZE_NUMERIC,
    NORMALIZE_PERCENT,
    NORMALIZE_TEXT,
    UNICODE_TO_ASCII,
    ExpressionTransform,
    compose_column,
)
from framesmith.transforms import (
    accounting_parens_to_negative,
    cast_to_float64,
    collapse_whitespace,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    percent_to_fraction,
    remove_apostrophes,
    remove_periods,
    remove_thousands_separators,
    replace_ampersand_with_and,
    strip_whitespace,
    to_snake_case,
    trailing_minus_to_prefix,
)


def _apply(
    values: list[str | None], recipe: Sequence[ExpressionTransform]
) -> pl.Series:
    """Apply a recipe through ``compose_column`` and return the column."""
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', recipe))['x']


# ---------------------------------------------------------------------
# NORMALIZE_TEXT faithfulness: ported from the legacy normalize_text suite
# ---------------------------------------------------------------------


class TestNormalizeTextFaithfulnessNfkc:
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
    def test_nfkc_normalization(self, value: str, expected: str) -> None:
        result = _apply([value], NORMALIZE_TEXT)
        assert result.to_list() == [expected]


class TestNormalizeTextFaithfulnessAsciiFolding:
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
        result = _apply([value], NORMALIZE_TEXT)
        assert result.to_list() == [expected]

    def test_trademark_decomposes_to_tm_via_nfkc(self) -> None:
        # NFKC decomposes ™ into the letters TM before the ASCII map
        # ever runs; the map never sees U+2122. Locked in here.
        result = _apply(['Brand™'], NORMALIZE_TEXT)
        assert result.to_list() == ['BrandTM']


class TestNormalizeTextFaithfulnessWhitespace:
    def test_tabs_become_spaces(self) -> None:
        result = _apply(['a\tb'], NORMALIZE_TEXT)
        assert result.to_list() == ['a b']

    def test_no_break_space_becomes_space(self) -> None:
        # U+00A0 NO-BREAK SPACE between 'a' and 'b'.
        result = _apply(['a b'], NORMALIZE_TEXT)
        assert result.to_list() == ['a b']

    def test_multiple_spaces_collapsed(self) -> None:
        result = _apply(['a   b'], NORMALIZE_TEXT)
        assert result.to_list() == ['a b']

    def test_leading_and_trailing_whitespace_stripped(self) -> None:
        result = _apply(['  hello  '], NORMALIZE_TEXT)
        assert result.to_list() == ['hello']


class TestNormalizeTextFaithfulnessAmpersand:
    def test_ampersand_expanded(self) -> None:
        result = _apply(['Sales & Service'], NORMALIZE_TEXT)
        assert result.to_list() == ['Sales and Service']

    def test_ampersand_expansion_preserves_case(self) -> None:
        result = _apply(['SALES & SERVICE'], NORMALIZE_TEXT)
        assert result.to_list() == ['SALES and SERVICE']


class TestNormalizeTextFaithfulnessApostropheAndPeriod:
    def test_apostrophe_removed(self) -> None:
        result = _apply(["O'Brien"], NORMALIZE_TEXT)
        assert result.to_list() == ['OBrien']

    def test_period_removed(self) -> None:
        result = _apply(['St.'], NORMALIZE_TEXT)
        assert result.to_list() == ['St']

    def test_smart_apostrophe_removed(self) -> None:
        # U+2019 folds to ASCII apostrophe via UNICODE_TO_ASCII, then
        # is stripped by remove_apostrophes.
        result = _apply(['O’Brien'], NORMALIZE_TEXT)  # noqa: RUF001
        assert result.to_list() == ['OBrien']


class TestNormalizeTextFaithfulnessNullSemantics:
    def test_null_propagates(self) -> None:
        result = _apply([None], NORMALIZE_TEXT)
        assert result.to_list() == [None]

    @pytest.mark.parametrize('value', ['', ' ', '   ', '\t', '\n', ' \t\n '])
    def test_blank_or_whitespace_only_becomes_null(self, value: str) -> None:
        result = _apply([value], NORMALIZE_TEXT)
        assert result.to_list() == [None]


# ---------------------------------------------------------------------
# UNICODE_TO_ASCII focused tests
# ---------------------------------------------------------------------


class TestUnicodeToAscii:
    def test_applies_nfkc_then_ascii_folding(self) -> None:
        # Fullwidth A (NFKC → 'A') combined with smart quote
        # (folded by ASCII map). Both stages must fire.
        result = _apply(['Ａ“x”'], UNICODE_TO_ASCII)  # noqa: RUF001
        assert result.to_list() == ['A"x"']

    def test_is_two_element_tuple_in_expected_order(self) -> None:
        assert len(UNICODE_TO_ASCII) == 2
        assert UNICODE_TO_ASCII[0] is normalize_unicode_nfkc
        assert UNICODE_TO_ASCII[1] is fold_to_ascii


# ---------------------------------------------------------------------
# Composability: splicing recipes into longer pipelines
# ---------------------------------------------------------------------


# Top-level recipe so the test body stays readable and no inline tuple
# literal appears in the compose_column call.
NORMALIZE_THEN_SNAKE: tuple[ExpressionTransform, ...] = (
    *NORMALIZE_TEXT,
    to_snake_case,
)


class TestRecipeComposability:
    def test_normalize_text_spliced_with_to_snake_case(self) -> None:
        df = pl.DataFrame({'x': ['Sales & Service']})
        result = df.with_columns(compose_column('x', NORMALIZE_THEN_SNAKE))
        assert result['x'].to_list() == ['Sales_and_Service']


# ---------------------------------------------------------------------
# Lazy / eager equivalence
# ---------------------------------------------------------------------


class TestLazyEagerEquivalence:
    def test_normalize_text_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'x': [
                    '  Sales & Service  ',
                    "O'Brien",
                    'A—B',
                    '   ',
                    None,
                ]
            },
            schema={'x': pl.String},
        )
        expr = compose_column('x', NORMALIZE_TEXT)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# Recipe structure / type invariants
# ---------------------------------------------------------------------


class TestRecipeStructure:
    def test_normalize_text_is_tuple_not_list(self) -> None:
        assert isinstance(NORMALIZE_TEXT, tuple)
        assert not isinstance(NORMALIZE_TEXT, list)

    def test_unicode_to_ascii_is_tuple_not_list(self) -> None:
        assert isinstance(UNICODE_TO_ASCII, tuple)
        assert not isinstance(UNICODE_TO_ASCII, list)

    def test_normalize_text_starts_with_nullify_then_unicode_to_ascii(
        self,
    ) -> None:
        # Locks in the splice: NORMALIZE_TEXT must include
        # UNICODE_TO_ASCII's transforms in sequence right after
        # nullify_blank_strings. If a future edit drops the splice,
        # this test fires.
        assert NORMALIZE_TEXT[0] is nullify_blank_strings
        assert NORMALIZE_TEXT[1:3] == UNICODE_TO_ASCII


# ---------------------------------------------------------------------
# NORMALIZE_NUMERIC end-to-end
# ---------------------------------------------------------------------


class TestNormalizeNumeric:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('($1,234.56)', -1234.56),
            ('1,234.56-', -1234.56),
            # U+2212 MINUS SIGN handled by fold_to_ascii via MINUS_LIKE_MAP.
            ('−1234', -1234.0),  # noqa: RUF001
            ('$1,234', 1234.0),
            ('(1,234)', -1234.0),
            # Fullwidth digits handled by NFKC.
            ('１２３', 123.0),  # noqa: RUF001
        ],
    )
    def test_parses_messy_numeric_strings(
        self, value: str, expected: float
    ) -> None:
        result = _apply([value], NORMALIZE_NUMERIC)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize('value', ['', '   ', 'abc'])
    def test_unparseable_becomes_null(self, value: str) -> None:
        result = _apply([value], NORMALIZE_NUMERIC)
        assert result.to_list() == [None]

    def test_null_propagates(self) -> None:
        result = _apply([None], NORMALIZE_NUMERIC)
        assert result.to_list() == [None]

    def test_output_dtype_is_float64(self) -> None:
        result = _apply(['123.45'], NORMALIZE_NUMERIC)
        assert result.dtype == pl.Float64


# ---------------------------------------------------------------------
# CLEAN_NUMERIC_STRING (string output, caller casts)
# ---------------------------------------------------------------------


class TestCleanNumericString:
    def test_cleans_to_bare_numeric_string(self) -> None:
        result = _apply(['($1,234.56)'], CLEAN_NUMERIC_STRING)
        assert result.to_list() == ['-1234.56']

    def test_output_dtype_is_still_string(self) -> None:
        result = _apply(['($1,234.56)'], CLEAN_NUMERIC_STRING)
        assert result.dtype == pl.String

    def test_caller_can_cast_to_int64(self) -> None:
        df = pl.DataFrame({'x': ['1,234']}, schema={'x': pl.String})
        expr = compose_column('x', CLEAN_NUMERIC_STRING).cast(
            pl.Int64, strict=False
        )
        result = df.with_columns(expr)
        assert result['x'].dtype == pl.Int64
        assert result['x'].to_list() == [1234]


# ---------------------------------------------------------------------
# Numeric recipe structure / splice locks
# ---------------------------------------------------------------------


class TestNumericRecipeStructure:
    def test_normalize_numeric_is_tuple_not_list(self) -> None:
        assert isinstance(NORMALIZE_NUMERIC, tuple)
        assert not isinstance(NORMALIZE_NUMERIC, list)

    def test_clean_numeric_string_is_tuple_not_list(self) -> None:
        assert isinstance(CLEAN_NUMERIC_STRING, tuple)
        assert not isinstance(CLEAN_NUMERIC_STRING, list)

    def test_normalize_numeric_splices_clean_numeric_string_then_casts(
        self,
    ) -> None:
        # Locks in the splice: NORMALIZE_NUMERIC must be
        # CLEAN_NUMERIC_STRING followed by cast_to_float64.
        assert NORMALIZE_NUMERIC[:-1] == CLEAN_NUMERIC_STRING
        assert NORMALIZE_NUMERIC[-1] is cast_to_float64

    def test_clean_numeric_string_starts_with_unicode_to_ascii(self) -> None:
        # Locks in the splice: the Unicode handling for numerics is
        # delegated to UNICODE_TO_ASCII and lives in one place.
        assert CLEAN_NUMERIC_STRING[:2] == UNICODE_TO_ASCII


# ---------------------------------------------------------------------
# Lazy / eager equivalence for NORMALIZE_NUMERIC
# ---------------------------------------------------------------------


class TestNormalizeNumericLazyEagerEquivalence:
    def test_normalize_numeric_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'x': [
                    '($1,234.56)',
                    '1,234.56-',
                    '$1,234',
                    '   ',
                    'abc',
                    None,
                ]
            },
            schema={'x': pl.String},
        )
        expr = compose_column('x', NORMALIZE_NUMERIC)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# Opt-in guard: shipped recipes must NOT include sentinel nullification
# ---------------------------------------------------------------------


class TestNoSentinelNullificationInDefaultRecipes:
    """Regression guard for the opt-in property of ``nullify_sentinels``.

    Sentinel handling depends on the data source — defaulting it on
    would silently null valid values (e.g. ``'NA'`` as Namibia). These
    tests pin the exact transforms in each shipped recipe so a future
    edit that slips ``nullify_sentinels`` (or any unexpected transform)
    into a default recipe fires immediately.
    """

    def test_normalize_text_contents_pinned(self) -> None:
        assert (
            nullify_blank_strings,
            normalize_unicode_nfkc,
            fold_to_ascii,
            collapse_whitespace,
            strip_whitespace,
            replace_ampersand_with_and,
            remove_apostrophes,
            remove_periods,
        ) == NORMALIZE_TEXT

    def test_clean_numeric_string_contents_pinned(self) -> None:
        assert (
            normalize_unicode_nfkc,
            fold_to_ascii,
            accounting_parens_to_negative,
            trailing_minus_to_prefix,
            remove_thousands_separators,
        ) == CLEAN_NUMERIC_STRING

    def test_normalize_numeric_contents_pinned(self) -> None:
        assert (
            normalize_unicode_nfkc,
            fold_to_ascii,
            accounting_parens_to_negative,
            trailing_minus_to_prefix,
            remove_thousands_separators,
            cast_to_float64,
        ) == NORMALIZE_NUMERIC

    def test_unicode_to_ascii_contents_pinned(self) -> None:
        assert (normalize_unicode_nfkc, fold_to_ascii) == UNICODE_TO_ASCII

    def test_normalize_percent_contents_pinned(self) -> None:
        assert (
            normalize_unicode_nfkc,
            fold_to_ascii,
            accounting_parens_to_negative,
            trailing_minus_to_prefix,
            remove_thousands_separators,
            percent_to_fraction,
        ) == NORMALIZE_PERCENT


# ---------------------------------------------------------------------
# NORMALIZE_PERCENT end-to-end
# ---------------------------------------------------------------------


class TestNormalizePercent:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('12%', 0.12),
            # Accounting parens around percent: parens-to-minus runs,
            # then '%' survives, then percent_to_fraction handles it.
            ('(12%)', -0.12),
            # Commas removed by remove_thousands_separators.
            ('1,234%', 12.34),
            # Whitespace removed by remove_thousands_separators.
            (' 50% ', 0.5),
            # U+2212 MINUS SIGN normalized via UNICODE_TO_ASCII.
            ('−50%', -0.5),  # noqa: RUF001
            # No '%' present — cast to its raw numeric value.
            ('50', 50.0),
        ],
    )
    def test_parses_messy_percent_strings(
        self, value: str, expected: float
    ) -> None:
        result = _apply([value], NORMALIZE_PERCENT)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize('value', ['abc%', ''])
    def test_unparseable_becomes_null(self, value: str) -> None:
        result = _apply([value], NORMALIZE_PERCENT)
        assert result.to_list() == [None]

    def test_null_propagates(self) -> None:
        result = _apply([None], NORMALIZE_PERCENT)
        assert result.to_list() == [None]

    def test_output_dtype_is_float64(self) -> None:
        result = _apply(['12%'], NORMALIZE_PERCENT)
        assert result.dtype == pl.Float64


# ---------------------------------------------------------------------
# NORMALIZE_PERCENT structure / splice locks
# ---------------------------------------------------------------------


class TestNormalizePercentStructure:
    def test_normalize_percent_is_tuple_not_list(self) -> None:
        assert isinstance(NORMALIZE_PERCENT, tuple)
        assert not isinstance(NORMALIZE_PERCENT, list)

    def test_normalize_percent_splices_clean_numeric_string_then_percent(
        self,
    ) -> None:
        # Locks in the splice: NORMALIZE_PERCENT must be
        # CLEAN_NUMERIC_STRING followed by percent_to_fraction.
        assert NORMALIZE_PERCENT[:-1] == CLEAN_NUMERIC_STRING
        assert NORMALIZE_PERCENT[-1] is percent_to_fraction


# ---------------------------------------------------------------------
# Lazy / eager equivalence for NORMALIZE_PERCENT
# ---------------------------------------------------------------------


class TestNormalizePercentLazyEagerEquivalence:
    def test_normalize_percent_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'x': [
                    '12%',
                    '(12%)',
                    '1,234%',
                    ' 50% ',
                    '50',
                    'abc%',
                    '',
                    None,
                ]
            },
            schema={'x': pl.String},
        )
        expr = compose_column('x', NORMALIZE_PERCENT)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
