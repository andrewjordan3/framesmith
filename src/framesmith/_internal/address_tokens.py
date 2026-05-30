# framesmith/_internal/address_tokens.py
"""
Curated USPS address-token maps (canonical abbreviation → variant spellings).

Canonicals are the USPS Publication 28 uppercase abbreviations; variants are
lowercase (matching is case-insensitive). The standardizing transforms match
each variant as a whole word and rewrite it to the canonical.
"""

from collections.abc import Mapping

__all__: list[str] = [
    'DEFAULT_DIRECTIONAL_MAP',
    'DEFAULT_STREET_SUFFIX_MAP',
    'DEFAULT_UNIT_MARKER_MAP',
]


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

# Common US street-suffix abbreviations (USPS Publication 28 Appendix C1),
# a curated subset of the ~200 official entries. Canonical = the USPS
# standard abbreviation; variants are the spelled-out name, the
# abbreviation, and common alternates. Pass the full C1 table to
# standardize_street_suffixes for exhaustive coverage.
DEFAULT_STREET_SUFFIX_MAP: Mapping[str, tuple[str, ...]] = {
    'ST': ('street', 'str', 'strt', 'st'),
    'AVE': ('avenue', 'aven', 'avenu', 'av', 'ave'),
    'BLVD': ('boulevard', 'boul', 'boulv', 'blvd'),
    'RD': ('road', 'rd'),
    'DR': ('drive', 'driv', 'drv', 'dr'),
    'LN': ('lane', 'ln'),
    'CT': ('court', 'crt', 'ct'),
    'CIR': ('circle', 'circ', 'circl', 'cir'),
    'PL': ('place', 'pl'),
    'WAY': ('way', 'wy'),
    'TER': ('terrace', 'terr', 'ter'),
    'PKWY': ('parkway', 'parkwy', 'pkway', 'pkwy'),
    'TRL': ('trail', 'trails', 'trl'),
    'HWY': ('highway', 'hiway', 'hiwy', 'hway', 'hwy'),
    'LOOP': ('loop', 'loops'),
    'SQ': ('square', 'sqr', 'sqre', 'sq'),
    'PLZ': ('plaza', 'plza', 'plz'),
    'PT': ('point', 'pt'),
    'ALY': ('alley', 'allee', 'ally', 'aly'),
    'BND': ('bend', 'bnd'),
    'XING': ('crossing', 'crssng', 'xing'),
    'HOLW': ('hollow', 'hollows', 'holw'),
    'PASS': ('pass',),
    'PATH': ('path', 'paths'),
    'PIKE': ('pike', 'pikes'),
    'RUN': ('run',),
    'ROW': ('row',),
    'WALK': ('walk', 'walks'),
    'GRV': ('grove', 'grov', 'grv'),
    'HTS': ('heights', 'hts'),
    'MNR': ('manor', 'mnr'),
    'MDWS': ('meadows', 'mdw', 'mdws'),
    'RDG': ('ridge', 'rdge', 'rdg'),
    'SPG': ('spring', 'spng', 'spg'),
    'SPGS': ('springs', 'spngs', 'spgs'),
    'VLG': ('village', 'vill', 'villg', 'vlg'),
    'VW': ('view', 'vw'),
    'EXPY': ('expressway', 'expr', 'express', 'expy'),
    'FWY': ('freeway', 'frwy', 'fwy'),
    'CTR': ('center', 'cent', 'centr', 'centre', 'cntr', 'ctr'),
    'GDN': ('garden', 'gdn'),
    'GDNS': ('gardens', 'gdns'),
    'CRES': ('crescent', 'crsent', 'cres'),
    'CV': ('cove', 'cv'),
    'EST': ('estate', 'est'),
    'CYN': ('canyon', 'canyn', 'cnyn', 'cyn'),
    'LK': ('lake', 'lk'),
    'CRK': ('creek', 'crk'),
    'FLS': ('falls', 'fls'),
    'JCT': ('junction', 'jction', 'jct'),
    'MT': ('mount', 'mt'),
    'MTN': ('mountain', 'mtin', 'mntn', 'mtn'),
    'TPKE': ('turnpike', 'tpke'),
    'VLY': ('valley', 'vlly', 'vly'),
}
