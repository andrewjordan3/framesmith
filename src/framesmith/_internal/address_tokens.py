# framesmith/_internal/address_tokens.py
"""
Curated USPS address-token maps (canonical abbreviation → variant spellings).

Canonicals are the USPS Publication 28 uppercase abbreviations; variants are
lowercase (matching is case-insensitive). The standardizing transforms match
each variant as a whole word and rewrite it to the canonical.
"""

from collections.abc import Mapping

__all__: list[str] = ['DEFAULT_DIRECTIONAL_MAP', 'DEFAULT_UNIT_MARKER_MAP']


# Directionals. NE/FL-style overlaps with state codes are harmless: the
# canonical equals the state code, so the output is unchanged.
DEFAULT_DIRECTIONAL_MAP: Mapping[str, tuple[str, ...]] = {
    'N': ('north', 'n'),
    'S': ('south', 's'),
    'E': ('east', 'e'),
    'W': ('west', 'w'),
    'NE': ('northeast', 'ne'),
    'NW': ('northwest', 'nw'),
    'SE': ('southeast', 'se'),
    'SW': ('southwest', 'sw'),
}

# Secondary unit designators (USPS Publication 28 Appendix C2), common subset.
DEFAULT_UNIT_MARKER_MAP: Mapping[str, tuple[str, ...]] = {
    'APT': ('apartment', 'apt'),
    'STE': ('suite', 'ste'),
    'UNIT': ('unit',),
    'BLDG': ('building', 'bldg'),
    'FL': ('floor', 'fl'),
    'RM': ('room', 'rm'),
    'DEPT': ('department', 'dept'),
    'OFC': ('office', 'ofc'),
    'SPC': ('space', 'spc'),
    'LOT': ('lot',),
    'TRLR': ('trailer', 'trlr'),
    'STOP': ('stop',),
}
