# tests/test_compose.py
"""Tests for ``framesmith.compose.compose_column``.

These tests use inline, module-level transforms only. They do not
import from ``framesmith.transforms`` or ``framesmith.recipes`` — the
composition layer is meant to stand on its own, decoupled from any
specific library of transforms.
"""

from collections.abc import Sequence

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

import framesmith
import framesmith.types
from framesmith import ExpressionTransform, compose_column

# ---------------------------------------------------------------------
# Test-only transforms (named functions, no lambdas)
# ---------------------------------------------------------------------


def strip_whitespace(expr: pl.Expr) -> pl.Expr:
    return expr.str.strip_chars()


def to_uppercase(expr: pl.Expr) -> pl.Expr:
    return expr.str.to_uppercase()


def to_lowercase(expr: pl.Expr) -> pl.Expr:
    return expr.str.to_lowercase()


def prepend_marker(expr: pl.Expr) -> pl.Expr:
    return pl.lit('x-') + expr


def self_aliasing_transform(expr: pl.Expr) -> pl.Expr:
    """Intentionally violates the contract by self-aliasing.

    The outer ``compose_column`` alias must override this — that's the
    documented behavior we want to lock down with a test.
    """
    return expr.str.to_uppercase().alias('hardcoded')


def increment(expr: pl.Expr) -> pl.Expr:
    return expr + 1


# ---------------------------------------------------------------------
# Single transform
# ---------------------------------------------------------------------


class TestSingleTransform:
    def test_applies_single_transform(self) -> None:
        df = pl.DataFrame({'name': ['  alice  ', '  bob  ']})
        result = df.with_columns(compose_column('name', [strip_whitespace]))
        expected = pl.DataFrame({'name': ['alice', 'bob']})
        assert_frame_equal(result, expected)

    def test_aliases_to_source_column_by_default(self) -> None:
        df = pl.DataFrame({'name': ['  alice  ']})
        result = df.with_columns(compose_column('name', [strip_whitespace]))
        assert result.columns == ['name']


# ---------------------------------------------------------------------
# Multiple transforms
# ---------------------------------------------------------------------


class TestMultipleTransforms:
    def test_applies_in_order(self) -> None:
        df = pl.DataFrame({'x': ['abc']})
        # Order: uppercase first, then prepend → 'x-ABC' (the lowercase
        # marker is added AFTER uppercasing, so it survives unchanged).
        result = df.with_columns(
            compose_column('x', [to_uppercase, prepend_marker])
        )
        assert result['x'].to_list() == ['x-ABC']

    def test_reversed_order_produces_different_result(self) -> None:
        df = pl.DataFrame({'x': ['abc']})
        # Order: prepend first, then uppercase → 'X-ABC' (uppercasing
        # 'x-abc' uppercases the marker too).
        result = df.with_columns(
            compose_column('x', [prepend_marker, to_uppercase])
        )
        assert result['x'].to_list() == ['X-ABC']

    def test_order_actually_matters(self) -> None:
        # Construct a case where flipping order yields a different
        # value, proving order is respected (not coincidentally equal).
        df = pl.DataFrame({'x': ['Abc']})
        upper_then_lower = df.with_columns(
            compose_column('x', [to_uppercase, to_lowercase])
        )['x'].to_list()
        lower_then_upper = df.with_columns(
            compose_column('x', [to_lowercase, to_uppercase])
        )['x'].to_list()
        assert upper_then_lower == ['abc']
        assert lower_then_upper == ['ABC']


# ---------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------


class TestEmptyTransforms:
    def test_empty_list_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match='expression_transforms must not be empty'
        ):
            compose_column('name', [])

    def test_empty_tuple_raises_value_error(self) -> None:
        with pytest.raises(
            ValueError, match='expression_transforms must not be empty'
        ):
            compose_column('name', ())


# ---------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------


class TestOutputColumnNaming:
    def test_none_overwrites_source_column(self) -> None:
        df = pl.DataFrame({'name': ['  alice  '], 'id': [1]})
        result = df.with_columns(compose_column('name', [strip_whitespace]))
        assert result.columns == ['name', 'id']
        assert result['name'].to_list() == ['alice']

    def test_explicit_name_creates_new_column(self) -> None:
        df = pl.DataFrame({'name': ['  alice  ']})
        result = df.with_columns(
            compose_column(
                'name', [strip_whitespace], output_column_name='clean_name'
            )
        )
        assert result.columns == ['name', 'clean_name']
        # Source column is preserved unchanged.
        assert result['name'].to_list() == ['  alice  ']
        assert result['clean_name'].to_list() == ['alice']


# ---------------------------------------------------------------------
# Other columns preserved
# ---------------------------------------------------------------------


class TestOtherColumnsPreserved:
    def test_other_columns_pass_through_untouched(self) -> None:
        df = pl.DataFrame(
            {
                'name': ['  alice  ', '  bob  '],
                'id': [1, 2],
                'score': [1.5, 2.5],
            }
        )
        result = df.with_columns(compose_column('name', [strip_whitespace]))
        assert result['id'].to_list() == [1, 2]
        assert result['score'].to_list() == [1.5, 2.5]


# ---------------------------------------------------------------------
# Lazy / eager equivalence
# ---------------------------------------------------------------------


class TestLazyFrameSupport:
    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame({'name': ['  Alice  ', '  BOB  '], 'id': [1, 2]})
        expr = compose_column('name', [strip_whitespace, to_lowercase])
        eager_result = df.with_columns(expr)
        lazy_result = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager_result, lazy_result)


# ---------------------------------------------------------------------
# Self-aliasing transform contract
# ---------------------------------------------------------------------


class TestSelfAliasingTransformIsOverridden:
    def test_outer_alias_wins_when_default(self) -> None:
        df = pl.DataFrame({'name': ['alice']})
        result = df.with_columns(
            compose_column('name', [self_aliasing_transform])
        )
        # Final column name is the source column name, NOT 'hardcoded'.
        assert result.columns == ['name']
        assert result['name'].to_list() == ['ALICE']

    def test_outer_alias_wins_when_explicit_output_name(self) -> None:
        df = pl.DataFrame({'name': ['alice']})
        result = df.with_columns(
            compose_column(
                'name',
                [self_aliasing_transform],
                output_column_name='clean_name',
            )
        )
        assert result.columns == ['name', 'clean_name']
        assert result['clean_name'].to_list() == ['ALICE']
        # 'hardcoded' must not appear anywhere.
        assert 'hardcoded' not in result.columns


# ---------------------------------------------------------------------
# Null propagation
# ---------------------------------------------------------------------


class TestNullPropagation:
    def test_nulls_propagate_through_chain(self) -> None:
        df = pl.DataFrame(
            {'name': ['alice', None, 'bob']},
            schema={'name': pl.String},
        )
        result = df.with_columns(
            compose_column('name', [to_uppercase, strip_whitespace])
        )
        expected = pl.Series('name', ['ALICE', None, 'BOB'])
        assert_series_equal(result['name'], expected)

    def test_nulls_propagate_for_numeric_chain(self) -> None:
        df = pl.DataFrame(
            {'value': [1, None, 3]},
            schema={'value': pl.Int64},
        )
        result = df.with_columns(compose_column('value', [increment, increment]))
        expected = pl.Series('value', [3, None, 5], dtype=pl.Int64)
        assert_series_equal(result['value'], expected)


# ---------------------------------------------------------------------
# Sequence acceptance: list and tuple
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    'recipe',
    [
        [strip_whitespace, to_lowercase],
        (strip_whitespace, to_lowercase),
    ],
    ids=['list', 'tuple'],
)
def test_accepts_list_and_tuple_recipes(
    recipe: Sequence[ExpressionTransform],
) -> None:
    df = pl.DataFrame({'name': ['  ALICE  ']})
    result = df.with_columns(compose_column('name', recipe))
    assert result['name'].to_list() == ['alice']


# ---------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------


class TestPublicReExports:
    def test_compose_column_importable_from_top_level(self) -> None:
        assert framesmith.compose_column is compose_column

    def test_expression_transform_importable_from_top_level(self) -> None:
        # Smoke-test the annotation path: a user-defined transform can
        # be typed with the alias imported from ``framesmith``.
        recipe: tuple[ExpressionTransform, ...] = (
            strip_whitespace,
            to_lowercase,
        )
        df = pl.DataFrame({'name': ['  ALICE  ']})
        result = df.with_columns(compose_column('name', recipe))
        assert result['name'].to_list() == ['alice']

    def test_expression_transform_importable_from_canonical_module(
        self,
    ) -> None:
        assert framesmith.types.ExpressionTransform is ExpressionTransform
