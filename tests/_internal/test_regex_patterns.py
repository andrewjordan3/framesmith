# tests/_internal/test_regex_patterns.py
"""Tests for the raw regex pattern strings in ``_internal.regex_patterns``.

These tests verify the patterns themselves (using ``re.search`` against
the raw strings), independent of any polars use. They protect against
silent drift when the patterns are touched.
"""

import re

import pytest

from framesmith._internal import (
    BLANK_OR_WHITESPACE_ONLY_PATTERN,
    TRAILING_JR_PATTERN,
    WHITESPACE_RUN_PATTERN,
)


class TestWhitespaceRunPattern:
    @pytest.mark.parametrize(
        'value',
        [' ', '   ', '\t', '\n', ' \t \n ', 'a b', 'a\tb'],
    )
    def test_matches_whitespace(self, value: str) -> None:
        assert re.search(WHITESPACE_RUN_PATTERN, value) is not None

    @pytest.mark.parametrize('value', ['', 'abc', 'a_b', 'snake_case'])
    def test_does_not_match_non_whitespace(self, value: str) -> None:
        assert re.search(WHITESPACE_RUN_PATTERN, value) is None


class TestBlankOrWhitespaceOnlyPattern:
    @pytest.mark.parametrize('value', ['', ' ', '   ', '\t', '\n', ' \t\n '])
    def test_matches_blank_or_whitespace(self, value: str) -> None:
        assert re.search(BLANK_OR_WHITESPACE_ONLY_PATTERN, value) is not None

    @pytest.mark.parametrize('value', ['a', ' a ', 'a b', '_'])
    def test_does_not_match_strings_with_non_whitespace(
        self, value: str
    ) -> None:
        assert re.search(BLANK_OR_WHITESPACE_ONLY_PATTERN, value) is None


class TestTrailingJrPattern:
    @pytest.mark.parametrize(
        'value',
        [
            'John Smith Jr',
            'John Smith jr',
            'John Smith JR',
            'John Smith Jr.',
            'John Smith jr.',
            'John Smith, Jr',
            'John Smith, Jr.',
            'John Smith,Jr',
            'Smith jr',
        ],
    )
    def test_matches_trailing_jr(self, value: str) -> None:
        assert re.search(TRAILING_JR_PATTERN, value) is not None

    @pytest.mark.parametrize(
        'value',
        [
            'Jr Bakery',
            'John Smith',
            'Smith',
            'Junior',
            'jrabbit',
            'jr abbit',
        ],
    )
    def test_does_not_match_non_trailing_jr(self, value: str) -> None:
        assert re.search(TRAILING_JR_PATTERN, value) is None

    def test_inline_case_insensitive_flag_is_active(self) -> None:
        # The (?i) inline flag should make 'JR' match without passing
        # re.IGNORECASE separately.
        assert re.search(TRAILING_JR_PATTERN, 'Smith JR') is not None
