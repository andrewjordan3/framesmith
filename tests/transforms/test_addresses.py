# tests/transforms/test_addresses.py
"""Tests for the US address transforms in ``framesmith.transforms.addresses``.

Each transform is exercised through ``compose_column`` so the tests also
cover the integration point. Imports go through the subpackage's public
surface (``from framesmith.transforms import ...``).
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    standardize_state,
    standardize_state_name,
    strip_trailing_state,
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


class TestStandardizeState:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('Illinois', 'IL'),
            ('illinois', 'IL'),
            ('IL', 'IL'),
            ('il', 'IL'),
            (' Illinois ', 'IL'),
            ('District of Columbia', 'DC'),
            ('dc', 'DC'),
            ('Guam', 'GU'),
            ('Puerto Rico', 'PR'),
        ],
    )
    def test_canonicalizes_known_states(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], standardize_state)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            # Common abbreviations, dotted and bare, any case. Periods are
            # ignored when matching.
            ('Calif.', 'CA'),
            ('Calif', 'CA'),
            ('CALIF.', 'CA'),
            ('Ill.', 'IL'),
            ('Mass.', 'MA'),
            ('Penn.', 'PA'),
            ('Tex.', 'TX'),
            ('W.Va.', 'WV'),
            # Dotted forms that collapse to a postal code via period-strip.
            ('N.H.', 'NH'),
            ('D.C.', 'DC'),
        ],
    )
    def test_recognizes_abbreviations_and_dotted_forms(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], standardize_state)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize(
        'value',
        [
            'Ontario',  # not a US state
            'XX',  # not a postal code
            'Chicago, IL',  # whole value isn't a state
        ],
    )
    def test_misses_pass_through_unchanged(self, value: str) -> None:
        result = _apply([value], standardize_state)
        assert result.to_list() == [value]

    def test_null_propagates(self) -> None:
        result = _apply([None], standardize_state)
        assert result.to_list() == [None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['Illinois'], standardize_state)
        assert result.dtype == pl.String

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['Illinois', 'il', 'Ontario', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [standardize_state])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestStandardizeStateName:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('IL', 'illinois'),
            ('il', 'illinois'),
            ('Illinois', 'illinois'),
            ('ILLINOIS', 'illinois'),
            ('Calif.', 'california'),
            ('calif', 'california'),
            ('D.C.', 'district of columbia'),
            ('VI', 'virgin islands'),
            ('us virgin islands', 'virgin islands'),
            ('Mass.', 'massachusetts'),
        ],
    )
    def test_canonicalizes_to_lowercase_name(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], standardize_state_name)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize('value', ['Ontario', 'XX'])
    def test_misses_pass_through_unchanged(self, value: str) -> None:
        result = _apply([value], standardize_state_name)
        assert result.to_list() == [value]

    def test_null_propagates(self) -> None:
        result = _apply([None], standardize_state_name)
        assert result.to_list() == [None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['il'], standardize_state_name)
        assert result.dtype == pl.String

    def test_composes_with_titlecase_for_display(self) -> None:
        df = pl.DataFrame({'x': ['il']}, schema={'x': pl.String})
        result = df.with_columns(
            compose_column('x', [standardize_state_name, to_titlecase])
        )['x']
        assert result.to_list() == ['Illinois']

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['il', 'Calif.', 'Ontario', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [standardize_state_name])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestStripTrailingState:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('Chicago, IL', 'Chicago'),
            ('Chicago IL', 'Chicago'),
            ('Chicago,IL', 'Chicago'),
            ('chicago, il', 'chicago'),
            ('Springfield, IL', 'Springfield'),
            # 'co' is Colorado's code; trailing-token stripping is by
            # design — the correct location-column behavior, not a bug.
            ('Denver CO', 'Denver'),
            ('Denver Co', 'Denver'),
        ],
    )
    def test_removes_trailing_code(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], strip_trailing_state)
        assert result.to_list() == [expected]

    @pytest.mark.parametrize(
        'value',
        [
            'Chicago',  # no trailing code
            'Chicago Heights',  # 'Heights' is not a code
            'IL',  # bare code, no separator
            'Chicago, Illinois',  # full name, codes only
            'Washington, DC',  # DC excluded from strip set
        ],
    )
    def test_leaves_value_unchanged(self, value: str) -> None:
        result = _apply([value], strip_trailing_state)
        assert result.to_list() == [value]

    def test_null_propagates(self) -> None:
        result = _apply([None], strip_trailing_state)
        assert result.to_list() == [None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['Chicago, IL'], strip_trailing_state)
        assert result.dtype == pl.String

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['Chicago, IL', 'Denver CO', 'Chicago', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [strip_trailing_state])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
