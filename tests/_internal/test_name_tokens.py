# tests/_internal/test_name_tokens.py
"""Tests for the curated name-affix token sets in ``_internal.name_tokens``.

These pin the deliberate exclusions — bare ``I``/``V`` from suffixes (middle
initials) and ``St`` from prefixes (surname element) — and the lowercase,
period-free form the transforms rely on.
"""

from framesmith._internal.name_tokens import (
    DEFAULT_NAME_PREFIXES,
    DEFAULT_NAME_SUFFIXES,
)


class TestTokenForm:
    def test_suffixes_are_lowercase_and_period_free(self) -> None:
        assert all(
            token == token.lower() and '.' not in token
            for token in DEFAULT_NAME_SUFFIXES
        )

    def test_prefixes_are_lowercase_and_period_free(self) -> None:
        assert all(
            token == token.lower() and '.' not in token
            for token in DEFAULT_NAME_PREFIXES
        )


class TestDeliberateExclusions:
    def test_suffixes_exclude_bare_roman_initials(self) -> None:
        assert 'i' not in DEFAULT_NAME_SUFFIXES
        assert 'v' not in DEFAULT_NAME_SUFFIXES

    def test_suffixes_include_common_tokens(self) -> None:
        assert 'jr' in DEFAULT_NAME_SUFFIXES
        assert 'iii' in DEFAULT_NAME_SUFFIXES

    def test_prefixes_exclude_surname_element_st(self) -> None:
        assert 'st' not in DEFAULT_NAME_PREFIXES

    def test_prefixes_include_common_tokens(self) -> None:
        assert 'mr' in DEFAULT_NAME_PREFIXES
        assert 'dr' in DEFAULT_NAME_PREFIXES
