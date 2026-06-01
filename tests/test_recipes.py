# tests/test_recipes.py
"""Tests for ``framesmith.recipes``.

Pins the documented input/output behavior of ``TEXT_NORMALIZE`` and the other
recipes on representative messy inputs. Lines that embed visually ambiguous
Unicode characters carry an inline ``# noqa: RUF001`` — the ambiguity is what
the test exercises.
"""

from collections.abc import Sequence

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import (
    EMAIL_TO_DISPLAY_NAME,
    NUMERIC_STRING_NORMALIZE,
    NUMERIC_STRING_TO_FLOAT,
    PERCENT_STRING_TO_FRACTION,
    TEXT_NORMALIZE,
    UNICODE_TO_ASCII,
    ExpressionTransform,
    compose_column,
    recipes,
)
from framesmith.recipes import (
    CATEGORY_CANONICALIZE,
    CATEGORY_CANONICALIZE_TO_SNAKE_CASE,
    PERSON_NAME_NORMALIZE,
    PERSON_NAME_NORMALIZE_TO_SNAKE_CASE,
    SNAKE_CASE_TO_TITLE,
    STREET_ADDRESS_NORMALIZE,
    STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE,
    TEXT_CANONICALIZE,
    TEXT_NORMALIZE_TO_SNAKE_CASE,
    WHITESPACE_CANONICALIZE,
)
from framesmith.transforms import (
    accounting_parens_to_negative,
    apply_replacements,
    cast_to_float64,
    collapse_whitespace,
    extract_email_local_part,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    nullify_sentinels,
    percent_to_fraction,
    periods_to_spaces,
    remove_apostrophes,
    remove_periods,
    remove_thousands_separators,
    replace_ampersand_with_and,
    standardize_initials,
    strip_whitespace,
    to_lowercase,
    to_snake_case,
    to_titlecase,
    trailing_minus_to_prefix,
    underscores_to_spaces,
)


def _apply(
    values: list[str | None], recipe: Sequence[ExpressionTransform]
) -> pl.Series:
    """Apply a recipe through ``compose_column`` and return the column."""
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', recipe))['x']


# ---------------------------------------------------------------------
# TEXT_NORMALIZE behavior
# ---------------------------------------------------------------------


class TestTextNormalizeNfkc:
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
        result = _apply([value], TEXT_NORMALIZE)
        assert result.to_list() == [expected]


class TestTextNormalizeAsciiFolding:
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
        result = _apply([value], TEXT_NORMALIZE)
        assert result.to_list() == [expected]

    def test_trademark_decomposes_to_tm_via_nfkc(self) -> None:
        # NFKC decomposes ™ into the letters TM before the ASCII map
        # ever runs; the map never sees U+2122. Locked in here.
        result = _apply(['Brand™'], TEXT_NORMALIZE)
        assert result.to_list() == ['BrandTM']


class TestTextNormalizeWhitespace:
    def test_tabs_become_spaces(self) -> None:
        result = _apply(['a\tb'], TEXT_NORMALIZE)
        assert result.to_list() == ['a b']

    def test_no_break_space_becomes_space(self) -> None:
        # U+00A0 NO-BREAK SPACE between 'a' and 'b'.
        result = _apply(['a b'], TEXT_NORMALIZE)
        assert result.to_list() == ['a b']

    def test_multiple_spaces_collapsed(self) -> None:
        result = _apply(['a   b'], TEXT_NORMALIZE)
        assert result.to_list() == ['a b']

    def test_leading_and_trailing_whitespace_stripped(self) -> None:
        result = _apply(['  hello  '], TEXT_NORMALIZE)
        assert result.to_list() == ['hello']


class TestTextNormalizeAmpersand:
    def test_ampersand_expanded(self) -> None:
        result = _apply(['Sales & Service'], TEXT_NORMALIZE)
        assert result.to_list() == ['Sales and Service']

    def test_ampersand_expansion_preserves_case(self) -> None:
        result = _apply(['SALES & SERVICE'], TEXT_NORMALIZE)
        assert result.to_list() == ['SALES and SERVICE']


class TestTextNormalizeApostropheAndPeriod:
    def test_apostrophe_removed(self) -> None:
        result = _apply(["O'Brien"], TEXT_NORMALIZE)
        assert result.to_list() == ['OBrien']

    def test_period_removed(self) -> None:
        result = _apply(['St.'], TEXT_NORMALIZE)
        assert result.to_list() == ['St']

    def test_smart_apostrophe_removed(self) -> None:
        # U+2019 folds to ASCII apostrophe via UNICODE_TO_ASCII, then
        # is stripped by remove_apostrophes.
        result = _apply(['O’Brien'], TEXT_NORMALIZE)  # noqa: RUF001
        assert result.to_list() == ['OBrien']


class TestTextNormalizeNullSemantics:
    def test_null_propagates(self) -> None:
        result = _apply([None], TEXT_NORMALIZE)
        assert result.to_list() == [None]

    @pytest.mark.parametrize('value', ['', ' ', '   ', '\t', '\n', ' \t\n '])
    def test_blank_or_whitespace_only_becomes_null(self, value: str) -> None:
        result = _apply([value], TEXT_NORMALIZE)
        assert result.to_list() == [None]


# ---------------------------------------------------------------------
# TEXT_CANONICALIZE: tidy-up without punctuation stripping
# ---------------------------------------------------------------------


class TestTextCanonicalize:
    def test_preserves_apostrophe_unlike_normalize(self) -> None:
        # The divergence between the two: canonicalize keeps the
        # apostrophe; normalize strips it.
        assert _apply(["O'Brien"], TEXT_CANONICALIZE).to_list() == ["O'Brien"]
        assert _apply(["O'Brien"], TEXT_NORMALIZE).to_list() == ['OBrien']

    def test_canonicalizes_unicode_and_whitespace(self) -> None:
        result = _apply(['  “A—B”  '], TEXT_CANONICALIZE)
        assert result.to_list() == ['"A-B"']

    def test_blank_becomes_null(self) -> None:
        assert _apply(['   '], TEXT_CANONICALIZE).to_list() == [None]


# ---------------------------------------------------------------------
# WHITESPACE_CANONICALIZE
# ---------------------------------------------------------------------


class TestWhitespaceCanonicalize:
    def test_tidies_whitespace_and_nullifies_blanks(self) -> None:
        result = _apply(['  a   b  ', '   ', '', 'x', None], WHITESPACE_CANONICALIZE)
        assert result.to_list() == ['a b', None, None, 'x', None]


# ---------------------------------------------------------------------
# CATEGORY_CANONICALIZE
# ---------------------------------------------------------------------


class TestCategoryCanonicalize:
    def test_merges_spelling_variants(self) -> None:
        result = _apply(
            [' Active ', 'ACTIVE', 'In  Progress', '  ', None],
            CATEGORY_CANONICALIZE,
        )
        assert result.to_list() == ['active', 'active', 'in progress', None, None]

    def test_snake_case_variant(self) -> None:
        result = _apply([' In Progress '], CATEGORY_CANONICALIZE_TO_SNAKE_CASE)
        assert result.to_list() == ['in_progress']


# ---------------------------------------------------------------------
# STREET_ADDRESS_NORMALIZE
# ---------------------------------------------------------------------


class TestStreetAddressNormalize:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('123 North Main Street', '123 N Main ST'),
            ('  456   south oak ave  ', '456 S oak AVE'),
            ('789 SE Grand Blvd Apartment 4', '789 SE Grand BLVD APT 4'),
        ],
    )
    def test_standardizes_usps_tokens(self, value: str, expected: str) -> None:
        result = _apply([value], STREET_ADDRESS_NORMALIZE)
        assert result.to_list() == [expected]

    def test_snake_case_variant(self) -> None:
        result = _apply(
            ['123 North Main Street'], STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE
        )
        assert result.to_list() == ['123_n_main_st']

    def test_null_propagates(self) -> None:
        assert _apply([None], STREET_ADDRESS_NORMALIZE).to_list() == [None]

    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'x': [
                    '123 North Main Street',
                    '  456   south oak ave  ',
                    '789 SE Grand Blvd Apartment 4',
                    '   ',
                    None,
                ]
            },
            schema={'x': pl.String},
        )
        expr = compose_column('x', STREET_ADDRESS_NORMALIZE)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# PERSON_NAME_NORMALIZE
# ---------------------------------------------------------------------


class TestPersonNameNormalize:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('Dr. John  Smith Jr, PhD', 'John Smith'),
            ('  Mrs. Jane Doe  ', 'Jane Doe'),
            ('Smith, J.R.', 'Smith, J. R.'),
            # No comma: credentials are not stripped.
            ('John Smith MD', 'John Smith MD'),
        ],
    )
    def test_strips_affixes_and_standardizes(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], PERSON_NAME_NORMALIZE)
        assert result.to_list() == [expected]

    def test_snake_case_variant(self) -> None:
        result = _apply(
            ['Dr. John Smith Jr'], PERSON_NAME_NORMALIZE_TO_SNAKE_CASE
        )
        assert result.to_list() == ['john_smith']

    @pytest.mark.parametrize('value', ['', '   ', None])
    def test_blank_and_null_become_null(self, value: str | None) -> None:
        assert _apply([value], PERSON_NAME_NORMALIZE).to_list() == [None]

    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'x': [
                    'Dr. John  Smith Jr, PhD',
                    '  Mrs. Jane Doe  ',
                    'Smith, J.R.',
                    'John Smith MD',
                    '   ',
                    None,
                ]
            },
            schema={'x': pl.String},
        )
        expr = compose_column('x', PERSON_NAME_NORMALIZE)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


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


class TestRecipeComposability:
    def test_text_normalize_spliced_with_to_snake_case(self) -> None:
        df = pl.DataFrame({'x': ['Sales & Service']})
        result = df.with_columns(
            compose_column('x', TEXT_NORMALIZE_TO_SNAKE_CASE)
        )
        assert result['x'].to_list() == ['sales_and_service']


# ---------------------------------------------------------------------
# Lazy / eager equivalence
# ---------------------------------------------------------------------


class TestLazyEagerEquivalence:
    def test_text_normalize_lazy_matches_eager(self) -> None:
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
        expr = compose_column('x', TEXT_NORMALIZE)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# Recipe structure / type invariants and splice locks
# ---------------------------------------------------------------------


class TestRecipeStructure:
    def test_text_normalize_is_tuple_not_list(self) -> None:
        assert isinstance(TEXT_NORMALIZE, tuple)
        assert not isinstance(TEXT_NORMALIZE, list)

    def test_unicode_to_ascii_is_tuple_not_list(self) -> None:
        assert isinstance(UNICODE_TO_ASCII, tuple)
        assert not isinstance(UNICODE_TO_ASCII, list)

    def test_text_normalize_starts_with_text_canonicalize(self) -> None:
        # Locks in the extraction: TEXT_NORMALIZE must begin with
        # TEXT_CANONICALIZE's transforms in order. If a future edit drops
        # the splice, this test fires.
        assert TEXT_NORMALIZE[: len(TEXT_CANONICALIZE)] == TEXT_CANONICALIZE

    def test_text_canonicalize_starts_with_nullify_then_unicode_to_ascii(
        self,
    ) -> None:
        assert TEXT_CANONICALIZE[0] is nullify_blank_strings
        assert TEXT_CANONICALIZE[1:3] == UNICODE_TO_ASCII

    @pytest.mark.parametrize(
        ('snake_recipe', 'base_recipe'),
        [
            (TEXT_NORMALIZE_TO_SNAKE_CASE, TEXT_NORMALIZE),
            (CATEGORY_CANONICALIZE_TO_SNAKE_CASE, CATEGORY_CANONICALIZE),
            (STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE, STREET_ADDRESS_NORMALIZE),
            (PERSON_NAME_NORMALIZE_TO_SNAKE_CASE, PERSON_NAME_NORMALIZE),
        ],
    )
    def test_snake_variant_is_base_then_to_snake_case(
        self,
        snake_recipe: tuple[ExpressionTransform, ...],
        base_recipe: tuple[ExpressionTransform, ...],
    ) -> None:
        # Each _TO_SNAKE_CASE recipe is exactly its base spliced in front
        # of to_snake_case. The prefix == works because splicing reuses
        # the identical transform objects.
        assert snake_recipe[:-1] == base_recipe
        assert snake_recipe[-1] is to_snake_case


# ---------------------------------------------------------------------
# NUMERIC_STRING_TO_FLOAT end-to-end
# ---------------------------------------------------------------------


class TestNumericStringToFloat:
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
        result = _apply([value], NUMERIC_STRING_TO_FLOAT)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize('value', ['', '   ', 'abc'])
    def test_unparseable_becomes_null(self, value: str) -> None:
        result = _apply([value], NUMERIC_STRING_TO_FLOAT)
        assert result.to_list() == [None]

    def test_null_propagates(self) -> None:
        result = _apply([None], NUMERIC_STRING_TO_FLOAT)
        assert result.to_list() == [None]

    def test_output_dtype_is_float64(self) -> None:
        result = _apply(['123.45'], NUMERIC_STRING_TO_FLOAT)
        assert result.dtype == pl.Float64


# ---------------------------------------------------------------------
# NUMERIC_STRING_NORMALIZE (string output, caller casts)
# ---------------------------------------------------------------------


class TestNumericStringNormalize:
    def test_cleans_to_bare_numeric_string(self) -> None:
        result = _apply(['($1,234.56)'], NUMERIC_STRING_NORMALIZE)
        assert result.to_list() == ['-1234.56']

    def test_output_dtype_is_still_string(self) -> None:
        result = _apply(['($1,234.56)'], NUMERIC_STRING_NORMALIZE)
        assert result.dtype == pl.String

    def test_caller_can_cast_to_int64(self) -> None:
        df = pl.DataFrame({'x': ['1,234']}, schema={'x': pl.String})
        expr = compose_column('x', NUMERIC_STRING_NORMALIZE).cast(
            pl.Int64, strict=False
        )
        result = df.with_columns(expr)
        assert result['x'].dtype == pl.Int64
        assert result['x'].to_list() == [1234]


# ---------------------------------------------------------------------
# Numeric recipe structure / splice locks
# ---------------------------------------------------------------------


class TestNumericRecipeStructure:
    def test_numeric_string_to_float_is_tuple_not_list(self) -> None:
        assert isinstance(NUMERIC_STRING_TO_FLOAT, tuple)
        assert not isinstance(NUMERIC_STRING_TO_FLOAT, list)

    def test_numeric_string_normalize_is_tuple_not_list(self) -> None:
        assert isinstance(NUMERIC_STRING_NORMALIZE, tuple)
        assert not isinstance(NUMERIC_STRING_NORMALIZE, list)

    def test_numeric_string_to_float_splices_normalize_then_casts(
        self,
    ) -> None:
        # Locks in the splice: NUMERIC_STRING_TO_FLOAT must be
        # NUMERIC_STRING_NORMALIZE followed by cast_to_float64.
        assert NUMERIC_STRING_TO_FLOAT[:-1] == NUMERIC_STRING_NORMALIZE
        assert NUMERIC_STRING_TO_FLOAT[-1] is cast_to_float64

    def test_numeric_string_normalize_starts_with_unicode_to_ascii(
        self,
    ) -> None:
        # Locks in the splice: the Unicode handling for numerics is
        # delegated to UNICODE_TO_ASCII and lives in one place.
        assert NUMERIC_STRING_NORMALIZE[:2] == UNICODE_TO_ASCII


# ---------------------------------------------------------------------
# Lazy / eager equivalence for NUMERIC_STRING_TO_FLOAT
# ---------------------------------------------------------------------


class TestNumericStringToFloatLazyEagerEquivalence:
    def test_numeric_string_to_float_lazy_matches_eager(self) -> None:
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
        expr = compose_column('x', NUMERIC_STRING_TO_FLOAT)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# Opt-in guard: shipped recipes must NOT include sentinel nullification
# ---------------------------------------------------------------------


class TestNoSentinelNullificationInDefaultRecipes:
    """Regression guard for the opt-in property of ``nullify_sentinels``.

    Sentinel handling depends on the data source — defaulting it on would
    silently null valid values (e.g. ``'NA'`` as Namibia). These tests pin
    the exact transforms in each shipped recipe so a future edit that slips
    ``nullify_sentinels`` (or any unexpected transform) into a default
    recipe fires immediately. The recipes use ``nullify_blank_strings``,
    which is a different transform.
    """

    def test_no_recipe_contains_nullify_sentinels(self) -> None:
        # Sweep all 16 published recipes, including the factory-closure
        # ones that the exact-tuple pins below cannot reconstruct.
        for name in recipes.__all__:
            recipe = getattr(recipes, name)
            assert nullify_sentinels not in recipe, name

    def test_published_recipe_names_pinned(self) -> None:
        # The 16 recipes the module publishes, locked so a rename or
        # accidental drop fires here.
        assert recipes.__all__ == [
            'CATEGORY_CANONICALIZE',
            'CATEGORY_CANONICALIZE_TO_SNAKE_CASE',
            'EMAIL_TO_DISPLAY_NAME',
            'NUMERIC_STRING_NORMALIZE',
            'NUMERIC_STRING_TO_FLOAT',
            'PERCENT_STRING_TO_FRACTION',
            'PERSON_NAME_NORMALIZE',
            'PERSON_NAME_NORMALIZE_TO_SNAKE_CASE',
            'SNAKE_CASE_TO_TITLE',
            'STREET_ADDRESS_NORMALIZE',
            'STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE',
            'TEXT_CANONICALIZE',
            'TEXT_NORMALIZE',
            'TEXT_NORMALIZE_TO_SNAKE_CASE',
            'UNICODE_TO_ASCII',
            'WHITESPACE_CANONICALIZE',
        ]

    # --- Exact-tuple pins for the bare-function recipes ---

    def test_whitespace_canonicalize_contents_pinned(self) -> None:
        assert (
            nullify_blank_strings,
            collapse_whitespace,
            strip_whitespace,
        ) == WHITESPACE_CANONICALIZE

    def test_unicode_to_ascii_contents_pinned(self) -> None:
        assert (normalize_unicode_nfkc, fold_to_ascii) == UNICODE_TO_ASCII

    def test_text_canonicalize_contents_pinned(self) -> None:
        assert (
            nullify_blank_strings,
            normalize_unicode_nfkc,
            fold_to_ascii,
            collapse_whitespace,
            strip_whitespace,
        ) == TEXT_CANONICALIZE

    def test_text_normalize_contents_pinned(self) -> None:
        assert (
            nullify_blank_strings,
            normalize_unicode_nfkc,
            fold_to_ascii,
            collapse_whitespace,
            strip_whitespace,
            replace_ampersand_with_and,
            remove_apostrophes,
            remove_periods,
        ) == TEXT_NORMALIZE

    def test_text_normalize_to_snake_case_contents_pinned(self) -> None:
        assert (
            nullify_blank_strings,
            normalize_unicode_nfkc,
            fold_to_ascii,
            collapse_whitespace,
            strip_whitespace,
            replace_ampersand_with_and,
            remove_apostrophes,
            remove_periods,
            to_snake_case,
        ) == TEXT_NORMALIZE_TO_SNAKE_CASE

    def test_category_canonicalize_contents_pinned(self) -> None:
        assert (
            nullify_blank_strings,
            collapse_whitespace,
            strip_whitespace,
            to_lowercase,
        ) == CATEGORY_CANONICALIZE

    def test_category_canonicalize_to_snake_case_contents_pinned(self) -> None:
        assert (
            nullify_blank_strings,
            collapse_whitespace,
            strip_whitespace,
            to_lowercase,
            to_snake_case,
        ) == CATEGORY_CANONICALIZE_TO_SNAKE_CASE

    def test_numeric_string_normalize_contents_pinned(self) -> None:
        assert (
            normalize_unicode_nfkc,
            fold_to_ascii,
            accounting_parens_to_negative,
            trailing_minus_to_prefix,
            remove_thousands_separators,
        ) == NUMERIC_STRING_NORMALIZE

    def test_numeric_string_to_float_contents_pinned(self) -> None:
        assert (
            normalize_unicode_nfkc,
            fold_to_ascii,
            accounting_parens_to_negative,
            trailing_minus_to_prefix,
            remove_thousands_separators,
            cast_to_float64,
        ) == NUMERIC_STRING_TO_FLOAT

    def test_percent_string_to_fraction_contents_pinned(self) -> None:
        assert (
            normalize_unicode_nfkc,
            fold_to_ascii,
            accounting_parens_to_negative,
            trailing_minus_to_prefix,
            remove_thousands_separators,
            percent_to_fraction,
        ) == PERCENT_STRING_TO_FRACTION

    def test_email_to_display_name_contents_pinned(self) -> None:
        assert (
            strip_whitespace,
            extract_email_local_part,
            periods_to_spaces,
            collapse_whitespace,
        ) == EMAIL_TO_DISPLAY_NAME

    def test_snake_case_to_title_contents_pinned(self) -> None:
        assert (underscores_to_spaces, to_titlecase) == SNAKE_CASE_TO_TITLE

    # --- Structural pins for the factory-closure recipes ---
    # standardize_*() and remove_credentials()/strip_name_*() return fresh
    # closures that no independent construction can equal, so these pin the
    # canonicalize prefix (== works — splicing reuses the same objects), the
    # bare-function elements by identity, the length, and callability of the
    # closure elements. The behavior tests pin that the right standardizers
    # actually run.

    def test_street_address_normalize_structure_pinned(self) -> None:
        assert len(STREET_ADDRESS_NORMALIZE) == 6
        assert STREET_ADDRESS_NORMALIZE[:3] == WHITESPACE_CANONICALIZE
        assert all(callable(step) for step in STREET_ADDRESS_NORMALIZE[3:])

    def test_person_name_normalize_structure_pinned(self) -> None:
        assert len(PERSON_NAME_NORMALIZE) == 8
        assert PERSON_NAME_NORMALIZE[:3] == WHITESPACE_CANONICALIZE
        assert all(callable(step) for step in PERSON_NAME_NORMALIZE[3:6])
        assert PERSON_NAME_NORMALIZE[6] is standardize_initials
        assert PERSON_NAME_NORMALIZE[7] is strip_whitespace


# ---------------------------------------------------------------------
# PERCENT_STRING_TO_FRACTION end-to-end
# ---------------------------------------------------------------------


class TestPercentStringToFraction:
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
        result = _apply([value], PERCENT_STRING_TO_FRACTION)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize('value', ['abc%', ''])
    def test_unparseable_becomes_null(self, value: str) -> None:
        result = _apply([value], PERCENT_STRING_TO_FRACTION)
        assert result.to_list() == [None]

    def test_null_propagates(self) -> None:
        result = _apply([None], PERCENT_STRING_TO_FRACTION)
        assert result.to_list() == [None]

    def test_output_dtype_is_float64(self) -> None:
        result = _apply(['12%'], PERCENT_STRING_TO_FRACTION)
        assert result.dtype == pl.Float64


# ---------------------------------------------------------------------
# PERCENT_STRING_TO_FRACTION structure / splice locks
# ---------------------------------------------------------------------


class TestPercentStringToFractionStructure:
    def test_percent_string_to_fraction_is_tuple_not_list(self) -> None:
        assert isinstance(PERCENT_STRING_TO_FRACTION, tuple)
        assert not isinstance(PERCENT_STRING_TO_FRACTION, list)

    def test_percent_string_to_fraction_splices_normalize_then_percent(
        self,
    ) -> None:
        # Locks in the splice: PERCENT_STRING_TO_FRACTION must be
        # NUMERIC_STRING_NORMALIZE followed by percent_to_fraction.
        assert PERCENT_STRING_TO_FRACTION[:-1] == NUMERIC_STRING_NORMALIZE
        assert PERCENT_STRING_TO_FRACTION[-1] is percent_to_fraction


# ---------------------------------------------------------------------
# Lazy / eager equivalence for PERCENT_STRING_TO_FRACTION
# ---------------------------------------------------------------------


class TestPercentStringToFractionLazyEagerEquivalence:
    def test_percent_string_to_fraction_lazy_matches_eager(self) -> None:
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
        expr = compose_column('x', PERCENT_STRING_TO_FRACTION)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# EMAIL_TO_DISPLAY_NAME end-to-end
# ---------------------------------------------------------------------


class TestEmailToDisplayName:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('john@example.com', 'john'),
            ('john.doe@example.com', 'john doe'),
            # Leading/trailing whitespace stripped by strip_whitespace.
            (' jane.q.smith@example.com ', 'jane q smith'),
            # collapse_whitespace tidies the doubled space from '..'.
            ('jane..doe@example.com', 'jane doe'),
            # No '@' → the local-part extractor leaves the value alone;
            # periods become spaces, then collapse keeps a clean form.
            ('noatsign', 'noatsign'),
            ('@example.com', ''),
            ('', ''),
        ],
    )
    def test_produces_display_name(self, value: str, expected: str) -> None:
        result = _apply([value], EMAIL_TO_DISPLAY_NAME)
        assert result.to_list() == [expected]

    def test_null_propagates(self) -> None:
        result = _apply([None], EMAIL_TO_DISPLAY_NAME)
        assert result.to_list() == [None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['john.doe@example.com'], EMAIL_TO_DISPLAY_NAME)
        assert result.dtype == pl.String


# ---------------------------------------------------------------------
# EMAIL_TO_DISPLAY_NAME structure
# ---------------------------------------------------------------------


class TestEmailToDisplayNameStructure:
    def test_email_to_display_name_is_tuple_not_list(self) -> None:
        assert isinstance(EMAIL_TO_DISPLAY_NAME, tuple)
        assert not isinstance(EMAIL_TO_DISPLAY_NAME, list)

    def test_email_to_display_name_exact_identity(self) -> None:
        # Locks in the recipe shape: any reorder or insertion fires.
        assert (
            strip_whitespace,
            extract_email_local_part,
            periods_to_spaces,
            collapse_whitespace,
        ) == EMAIL_TO_DISPLAY_NAME


# ---------------------------------------------------------------------
# Lazy / eager equivalence for EMAIL_TO_DISPLAY_NAME
# ---------------------------------------------------------------------


class TestEmailToDisplayNameLazyEagerEquivalence:
    def test_email_to_display_name_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'x': [
                    'john@example.com',
                    'john.doe@example.com',
                    ' jane.q.smith@example.com ',
                    'jane..doe@example.com',
                    'noatsign',
                    '@example.com',
                    '',
                    None,
                ]
            },
            schema={'x': pl.String},
        )
        expr = compose_column('x', EMAIL_TO_DISPLAY_NAME)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# SNAKE_CASE_TO_TITLE end-to-end, structure, and splice
# ---------------------------------------------------------------------


class TestSnakeCaseToTitle:
    def test_produces_title_label(self) -> None:
        result = _apply(['john_smith', 'jane_doe'], SNAKE_CASE_TO_TITLE)
        assert result.to_list() == ['John Smith', 'Jane Doe']

    def test_is_tuple_not_list(self) -> None:
        assert isinstance(SNAKE_CASE_TO_TITLE, tuple)
        assert not isinstance(SNAKE_CASE_TO_TITLE, list)

    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame(
            {'x': ['john_smith', 'jane_doe', 'nasa_program', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', SNAKE_CASE_TO_TITLE)
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)

    def test_splice_with_apply_replacements_fixes_acronym(self) -> None:
        # Title-case, then fix the mangled acronym token in place.
        recipe = (*SNAKE_CASE_TO_TITLE, apply_replacements({'Nasa': 'NASA'}))
        result = _apply(['nasa_program', 'visit_nasa'], recipe)
        assert result.to_list() == ['NASA Program', 'Visit NASA']
