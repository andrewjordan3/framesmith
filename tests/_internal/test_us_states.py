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
    TRAILING_STATE_CODE_PATTERN,
    US_STATE_CODES,
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
