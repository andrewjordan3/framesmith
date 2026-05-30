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
    standardize_directionals,
    standardize_state,
    standardize_state_name,
    standardize_street_suffixes,
    standardize_unit_markers,
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


class TestStandardizeDirectionals:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('123 North Main St', '123 N Main St'),
            ('123 N. Main St', '123 N Main St'),
            ('456 Northeast Blvd', '456 NE Blvd'),
            ('789 Northwest Hwy', '789 NW Hwy'),
            ('100 South West St', '100 S W St'),
            ('1 east west blvd', '1 E W blvd'),
        ],
    )
    def test_standardizes_directionals(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], standardize_directionals())
        assert result.to_list() == [expected]

    @pytest.mark.parametrize(
        'value',
        [
            'Northern Pkwy',  # 'north' is interior, not a whole word
            'Main St',  # no directional
        ],
    )
    def test_leaves_non_directionals_unchanged(self, value: str) -> None:
        result = _apply([value], standardize_directionals())
        assert result.to_list() == [value]

    def test_null_propagates(self) -> None:
        result = _apply([None], standardize_directionals())
        assert result.to_list() == [None]

    def test_empty_map_raises(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            standardize_directionals({})

    def test_factory_returns_callable(self) -> None:
        assert callable(standardize_directionals())

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['123 North Main St', 'Northern Pkwy', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [standardize_directionals()])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestStandardizeUnitMarkers:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('Apartment 4', 'APT 4'),
            ('Apt. 4', 'APT 4'),
            ('Suite 200', 'STE 200'),
            ('Ste 200', 'STE 200'),
            ('Building 7', 'BLDG 7'),
            ('Floor 3', 'FL 3'),
        ],
    )
    def test_standardizes_unit_markers(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], standardize_unit_markers())
        assert result.to_list() == [expected]

    def test_leaves_non_markers_unchanged(self) -> None:
        result = _apply(['123 Main St'], standardize_unit_markers())
        assert result.to_list() == ['123 Main St']

    def test_null_propagates(self) -> None:
        result = _apply([None], standardize_unit_markers())
        assert result.to_list() == [None]

    def test_empty_map_raises(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            standardize_unit_markers({})

    def test_factory_returns_callable(self) -> None:
        assert callable(standardize_unit_markers())

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['Apartment 4', '123 Main St', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [standardize_unit_markers()])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestStandardizeStreetSuffixes:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('123 Main Street', '123 Main ST'),
            ('123 Main St.', '123 Main ST'),
            ('Grand Avenue', 'Grand AVE'),
            ('Grand Av', 'Grand AVE'),
            ('Sunset Boulevard', 'Sunset BLVD'),
            ('Oak Drive', 'Oak DR'),
            ('Elm Court', 'Elm CT'),
            ('5 maple lane', '5 maple LN'),
        ],
    )
    def test_standardizes_street_suffixes(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], standardize_street_suffixes())
        assert result.to_list() == [expected]

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('Broadway', 'Broadway'),  # 'way' must not fire inside it
            ('Start Ave', 'Start AVE'),  # only the whole-word 'Ave' matches
        ],
    )
    def test_whole_word_boundary_pins(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], standardize_street_suffixes())
        assert result.to_list() == [expected]

    def test_null_propagates(self) -> None:
        result = _apply([None], standardize_street_suffixes())
        assert result.to_list() == [None]

    def test_custom_map(self) -> None:
        result = _apply(
            ['Main Esplanade'],
            standardize_street_suffixes({'ESPL': ('esplanade', 'espl')}),
        )
        assert result.to_list() == ['Main ESPL']

    def test_empty_map_raises(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            standardize_street_suffixes({})

    def test_factory_returns_callable(self) -> None:
        assert callable(standardize_street_suffixes())

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['123 Main Street', 'Broadway', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [standardize_street_suffixes()])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
