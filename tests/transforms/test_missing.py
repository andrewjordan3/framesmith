# tests/transforms/test_missing.py
"""Tests for ``framesmith.transforms.missing``."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import NORMALIZE_TEXT, ExpressionTransform, compose_column
from framesmith.transforms import DEFAULT_MISSING_SENTINELS, nullify_sentinels


def _apply(
    values: list[str | None], transform: ExpressionTransform
) -> pl.Series:
    """Run a single transform on a 1-column ``pl.String`` frame."""
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', [transform]))['x']


# ---------------------------------------------------------------------
# DEFAULT_MISSING_SENTINELS behavior
# ---------------------------------------------------------------------


class TestDefaultSentinels:
    @pytest.fixture
    def transform(self) -> ExpressionTransform:
        return nullify_sentinels(DEFAULT_MISSING_SENTINELS)

    @pytest.mark.parametrize(
        'value',
        [
            'N/A',
            'n/a',
            'NULL',
            'null',
            'None',
            'NONE',
            'NA',
            'na',
            'NaN',
            'NAN',
            '',
        ],
    )
    def test_known_sentinels_become_null(
        self, value: str, transform: ExpressionTransform
    ) -> None:
        result = _apply([value], transform)
        assert result.to_list() == [None]

    def test_whitespace_padded_sentinel_becomes_null(
        self, transform: ExpressionTransform
    ) -> None:
        # The factory strips before comparison.
        result = _apply([' N/A '], transform)
        assert result.to_list() == [None]

    def test_non_sentinel_unchanged(
        self, transform: ExpressionTransform
    ) -> None:
        result = _apply(['Chicago'], transform)
        assert result.to_list() == ['Chicago']

    def test_substring_match_is_exact_not_contains(
        self, transform: ExpressionTransform
    ) -> None:
        # 'NABISCO' contains 'NA' as a prefix but is not equal to it.
        # is_in (not contains) must keep this intact.
        result = _apply(['NABISCO'], transform)
        assert result.to_list() == ['NABISCO']

    def test_non_sentinel_preserves_surrounding_whitespace(
        self, transform: ExpressionTransform
    ) -> None:
        # The strip is for comparison only; non-matches come out
        # byte-identical to the input.
        result = _apply([' spaced text '], transform)
        assert result.to_list() == [' spaced text ']

    def test_null_propagates(self, transform: ExpressionTransform) -> None:
        result = _apply([None], transform)
        assert result.to_list() == [None]


# ---------------------------------------------------------------------
# Custom sentinel sets
# ---------------------------------------------------------------------


class TestCustomSentinels:
    @pytest.fixture
    def transform(self) -> ExpressionTransform:
        return nullify_sentinels({'--', 'MISSING'})

    def test_custom_dash_sentinel(
        self, transform: ExpressionTransform
    ) -> None:
        result = _apply(['--'], transform)
        assert result.to_list() == [None]

    def test_custom_word_sentinel(
        self, transform: ExpressionTransform
    ) -> None:
        result = _apply(['MISSING'], transform)
        assert result.to_list() == [None]

    def test_case_insensitive_default(
        self, transform: ExpressionTransform
    ) -> None:
        result = _apply(['missing'], transform)
        assert result.to_list() == [None]

    def test_default_sentinel_not_in_custom_set_unchanged(
        self, transform: ExpressionTransform
    ) -> None:
        # 'N/A' is in the default set but NOT in this custom set, so
        # it must pass through.
        result = _apply(['N/A'], transform)
        assert result.to_list() == ['N/A']


# ---------------------------------------------------------------------
# case_insensitive=False
# ---------------------------------------------------------------------


class TestCaseSensitive:
    @pytest.fixture
    def transform(self) -> ExpressionTransform:
        return nullify_sentinels({'NULL'}, case_insensitive=False)

    def test_exact_case_match_becomes_null(
        self, transform: ExpressionTransform
    ) -> None:
        result = _apply(['NULL'], transform)
        assert result.to_list() == [None]

    def test_other_case_unchanged(
        self, transform: ExpressionTransform
    ) -> None:
        result = _apply(['null'], transform)
        assert result.to_list() == ['null']


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


class TestValidation:
    def test_empty_set_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match='sentinels must not be empty'
        ):
            nullify_sentinels(set())

    def test_empty_list_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match='sentinels must not be empty'
        ):
            nullify_sentinels([])

    def test_empty_frozenset_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match='sentinels must not be empty'
        ):
            nullify_sentinels(frozenset())


# ---------------------------------------------------------------------
# Factory contract
# ---------------------------------------------------------------------


class TestFactoryContract:
    def test_factory_returns_callable(self) -> None:
        transform = nullify_sentinels(DEFAULT_MISSING_SENTINELS)
        assert callable(transform)

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {
                'x': [
                    'ok',
                    'N/A',
                    ' NULL ',
                    None,
                    'Chicago',
                    'NABISCO',
                ]
            },
            schema={'x': pl.String},
        )
        transform = nullify_sentinels(DEFAULT_MISSING_SENTINELS)
        expr = compose_column('x', [transform])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# Composability with NORMALIZE_TEXT
# ---------------------------------------------------------------------


class TestComposabilityWithNormalizeText:
    def test_normalize_text_then_nullify_sentinels(self) -> None:
        recipe = (
            *NORMALIZE_TEXT,
            nullify_sentinels(DEFAULT_MISSING_SENTINELS),
        )
        df = pl.DataFrame(
            {'x': ['  Sales & Service  ', 'N/A', '   ', None]},
            schema={'x': pl.String},
        )
        result = df.with_columns(compose_column('x', recipe))
        # 'Sales & Service' is normalized; 'N/A' is nullified;
        # blank-only is nullified by NORMALIZE_TEXT's nullify_blank_strings.
        assert result['x'].to_list() == [
            'Sales and Service',
            None,
            None,
            None,
        ]
