# tests/_internal/test_us_states.py
"""Tests for the US state reference data in ``_internal.us_states``.

These verify the derived structures directly (file-level imports are
fine for tests), independent of any polars use: the standardize map,
the trailing-strip code set, and the trailing-code pattern via
``re.search``. They protect against silent drift in the reference data
and lock the deliberate DC asymmetry.
"""

import re

import pytest

from framesmith._internal.us_states import (
    _STATE_ALIASES,
    TRAILING_STATE_CODE_PATTERN,
    US_STATE_CODES,
    US_STATE_NAME_MAP,
    US_STATE_STANDARDIZE_MAP,
)


class TestStandardizeMap:
    @pytest.mark.parametrize(
        ('key', 'expected_code'),
        [
            ('illinois', 'IL'),
            ('il', 'IL'),
            ('district of columbia', 'DC'),
            ('dc', 'DC'),
            ('guam', 'GU'),
            ('puerto rico', 'PR'),
            # Aliases now feed the code map.
            ('calif', 'CA'),
            ('ill', 'IL'),
            ('mass', 'MA'),
            ('wva', 'WV'),
            ('tex', 'TX'),
            ('penn', 'PA'),
            ('us virgin islands', 'VI'),
        ],
    )
    def test_key_maps_to_canonical_code(
        self, key: str, expected_code: str
    ) -> None:
        assert US_STATE_STANDARDIZE_MAP[key] == expected_code

    def test_every_value_is_two_letter_uppercase(self) -> None:
        assert all(
            len(code) == 2 and code.isupper()
            for code in US_STATE_STANDARDIZE_MAP.values()
        )


class TestNameMap:
    @pytest.mark.parametrize(
        ('key', 'expected_name'),
        [
            ('il', 'illinois'),
            ('illinois', 'illinois'),
            ('calif', 'california'),
            ('ca', 'california'),
            ('dc', 'district of columbia'),
            ('vi', 'virgin islands'),
            ('us virgin islands', 'virgin islands'),
            ('wva', 'west virginia'),
        ],
    )
    def test_key_maps_to_canonical_name(
        self, key: str, expected_name: str
    ) -> None:
        assert US_STATE_NAME_MAP[key] == expected_name


class TestMapParity:
    def test_both_maps_recognize_same_inputs(self) -> None:
        assert set(US_STATE_STANDARDIZE_MAP) == set(US_STATE_NAME_MAP)

    def test_alias_values_are_canonical_codes(self) -> None:
        canonical_codes = set(US_STATE_STANDARDIZE_MAP.values())
        assert all(code in canonical_codes for code in _STATE_ALIASES.values())

    def test_alias_keys_are_lowercase_and_period_free(self) -> None:
        assert all(
            '.' not in alias and alias == alias.lower()
            for alias in _STATE_ALIASES
        )


class TestStripCodeSet:
    @pytest.mark.parametrize('code', ['il', 'ca', 'pr'])
    def test_contains_expected_code(self, code: str) -> None:
        assert code in US_STATE_CODES

    def test_excludes_dc(self) -> None:
        assert 'dc' not in US_STATE_CODES


class TestTrailingStatePattern:
    @pytest.mark.parametrize(
        'value',
        ['chicago, il', 'chicago il', 'Chicago, IL', 'denver co'],
    )
    def test_matches_trailing_code(self, value: str) -> None:
        assert re.search(TRAILING_STATE_CODE_PATTERN, value) is not None

    @pytest.mark.parametrize(
        'value',
        ['chicago', 'il', 'washington, dc'],
    )
    def test_does_not_match(self, value: str) -> None:
        assert re.search(TRAILING_STATE_CODE_PATTERN, value) is None
