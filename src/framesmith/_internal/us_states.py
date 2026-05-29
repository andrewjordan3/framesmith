# framesmith/_internal/us_states.py
"""
US state, district, and territory reference data for address transforms.

Single source of truth: ``_STATE_NAME_CODE_PAIRS``. Everything else derives
from it. Two address transforms consume two derived views:

* ``standardize_state`` uses ``US_STATE_STANDARDIZE_MAP`` — full names and
  postal codes, lowercased, mapped to the canonical uppercase code.
* ``strip_trailing_state`` uses ``TRAILING_STATE_CODE_PATTERN`` — built from
  the code set with DC excluded.

DC asymmetry is deliberate. ``'dc'`` is a valid standardization target
(``'district of columbia'`` / ``'dc'`` -> ``'DC'``) but is excluded from the
trailing-strip code set: stripping ``'dc'`` from ``'washington dc'`` would
collide with Washington state.

``US_STATE_STANDARDIZE_MAP`` and ``TRAILING_STATE_CODE_PATTERN`` are the only
names re-exported through the ``_internal`` init — they are the only two used
outside ``_internal``. ``US_STATE_CODES`` is file-public for the integrity
assert and tests but is not part of the directory surface.
"""

# (full_name, postal_code). Canonical source — all derived structures
# build from this tuple.
_STATE_NAME_CODE_PAIRS: tuple[tuple[str, str], ...] = (
    ('alabama', 'AL'),
    ('alaska', 'AK'),
    ('arizona', 'AZ'),
    ('arkansas', 'AR'),
    ('california', 'CA'),
    ('colorado', 'CO'),
    ('connecticut', 'CT'),
    ('delaware', 'DE'),
    ('florida', 'FL'),
    ('georgia', 'GA'),
    ('hawaii', 'HI'),
    ('idaho', 'ID'),
    ('illinois', 'IL'),
    ('indiana', 'IN'),
    ('iowa', 'IA'),
    ('kansas', 'KS'),
    ('kentucky', 'KY'),
    ('louisiana', 'LA'),
    ('maine', 'ME'),
    ('maryland', 'MD'),
    ('massachusetts', 'MA'),
    ('michigan', 'MI'),
    ('minnesota', 'MN'),
    ('mississippi', 'MS'),
    ('missouri', 'MO'),
    ('montana', 'MT'),
    ('nebraska', 'NE'),
    ('nevada', 'NV'),
    ('new hampshire', 'NH'),
    ('new jersey', 'NJ'),
    ('new mexico', 'NM'),
    ('new york', 'NY'),
    ('north carolina', 'NC'),
    ('north dakota', 'ND'),
    ('ohio', 'OH'),
    ('oklahoma', 'OK'),
    ('oregon', 'OR'),
    ('pennsylvania', 'PA'),
    ('rhode island', 'RI'),
    ('south carolina', 'SC'),
    ('south dakota', 'SD'),
    ('tennessee', 'TN'),
    ('texas', 'TX'),
    ('utah', 'UT'),
    ('vermont', 'VT'),
    ('virginia', 'VA'),
    ('washington', 'WA'),
    ('west virginia', 'WV'),
    ('wisconsin', 'WI'),
    ('wyoming', 'WY'),
    ('district of columbia', 'DC'),
    ('puerto rico', 'PR'),
    ('virgin islands', 'VI'),
    ('guam', 'GU'),
    ('northern mariana islands', 'MP'),
    ('american samoa', 'AS'),
)


# Whole-value standardization map: full name AND lowercased code both key to
# the canonical uppercase code. Comprehension scope avoids leaking loop vars.
US_STATE_STANDARDIZE_MAP: dict[str, str] = {
    key: code
    for full_name, code in _STATE_NAME_CODE_PAIRS
    for key in (full_name, code.lower())
}


# Trailing-strip code set: lowercased codes, DC excluded (see module docstring).
US_STATE_CODES: frozenset[str] = frozenset(
    code.lower() for _, code in _STATE_NAME_CODE_PAIRS if code != 'DC'
)


# Trailing state-code pattern: a comma-and/or-whitespace separator followed by
# a state code at end of string. ``(?i)`` inline flag (polars regex strings
# take no separate flags argument). Sorted for a deterministic pattern string.
TRAILING_STATE_CODE_PATTERN: str = (
    r'(?i)(?:,\s*|\s+)(?:' + '|'.join(sorted(US_STATE_CODES)) + r')$'
)


# Compile-time integrity check, mirroring unicode_maps.py.
_POSTAL_CODE_LENGTH: int = 2
_CANONICAL_CODES: frozenset[str] = frozenset(
    code for _, code in _STATE_NAME_CODE_PAIRS
)
_EXPECTED_STRIP_CODES: frozenset[str] = frozenset(
    code.lower() for code in _CANONICAL_CODES if code != 'DC'
)
assert all(
    len(code) == _POSTAL_CODE_LENGTH and code.isupper()
    for code in _CANONICAL_CODES
), 'All state codes must be two-letter uppercase.'
assert set(US_STATE_STANDARDIZE_MAP.values()) == _CANONICAL_CODES, (
    'Standardize map must target exactly the canonical code set.'
)
assert US_STATE_CODES == _EXPECTED_STRIP_CODES, (
    'Strip code set must be the canonical codes minus DC, lowercased.'
)
