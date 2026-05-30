# tests/_internal/test_address_tokens.py
"""Tests for the USPS address-token maps in ``_internal.address_tokens``.

These pin the canonical/variant casing convention (uppercase canonicals,
lowercase variants) the standardizing transforms rely on, and the presence
of the expected tokens.
"""

from collections.abc import Mapping, Sequence

import pytest

from framesmith._internal.address_tokens import (
    DEFAULT_DIRECTIONAL_MAP,
    DEFAULT_STREET_SUFFIX_MAP,
    DEFAULT_UNIT_MARKER_MAP,
)


class TestTokenCasing:
    def test_directional_canonicals_uppercase_variants_lowercase(self) -> None:
        for canonical, variants in DEFAULT_DIRECTIONAL_MAP.items():
            assert canonical == canonical.upper()
            assert all(variant == variant.lower() for variant in variants)

    def test_unit_marker_canonicals_uppercase_variants_lowercase(self) -> None:
        for canonical, variants in DEFAULT_UNIT_MARKER_MAP.items():
            assert canonical == canonical.upper()
            assert all(variant == variant.lower() for variant in variants)

    def test_street_suffix_canonicals_uppercase_variants_lowercase(
        self,
    ) -> None:
        for canonical, variants in DEFAULT_STREET_SUFFIX_MAP.items():
            assert canonical == canonical.upper()
            assert all(variant == variant.lower() for variant in variants)


class TestDirectionalMap:
    def test_northeast_variants(self) -> None:
        assert DEFAULT_DIRECTIONAL_MAP['NE'] == ('northeast', 'ne')

    def test_includes_all_eight_directionals(self) -> None:
        assert set(DEFAULT_DIRECTIONAL_MAP) == {
            'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'
        }


class TestUnitMarkerMap:
    def test_includes_common_markers(self) -> None:
        assert 'APT' in DEFAULT_UNIT_MARKER_MAP
        assert 'STE' in DEFAULT_UNIT_MARKER_MAP
        assert 'FL' in DEFAULT_UNIT_MARKER_MAP


class TestStreetSuffixMap:
    def test_includes_common_suffixes(self) -> None:
        for canonical in ('ST', 'AVE', 'BLVD', 'RD'):
            assert canonical in DEFAULT_STREET_SUFFIX_MAP


class TestVariantDisjointness:
    @pytest.mark.parametrize(
        'token_map',
        [
            DEFAULT_DIRECTIONAL_MAP,
            DEFAULT_UNIT_MARKER_MAP,
            DEFAULT_STREET_SUFFIX_MAP,
        ],
    )
    def test_no_variant_maps_to_two_canonicals(
        self, token_map: Mapping[str, Sequence[str]]
    ) -> None:
        all_variants = [
            variant for variants in token_map.values() for variant in variants
        ]
        assert len(all_variants) == len(set(all_variants))
