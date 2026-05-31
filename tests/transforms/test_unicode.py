# tests/transforms/test_unicode.py
"""Tests for Unicode transforms in ``framesmith.transforms.unicode``.

Each transform is exercised in isolation through ``compose_column`` so
the tests also cover the integration point. Lines that embed visually
ambiguous Unicode characters carry an inline ``# noqa: RUF001`` — the
ambiguity is what the test exercises.
"""

import polars as pl
from polars.testing import assert_series_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    fold_to_ascii,
    normalize_unicode_nfkc,
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


class TestNullPropagationBatch:
    """Batch null propagation across this module's transforms."""

    def test_null_propagates_for_module_transforms(self) -> None:
        transforms: list[ExpressionTransform] = [
            normalize_unicode_nfkc,
            fold_to_ascii,
        ]
        expected = pl.Series('x', [None], dtype=pl.String)
        for transform in transforms:
            result = _apply([None], transform)
            assert_series_equal(result, expected)
