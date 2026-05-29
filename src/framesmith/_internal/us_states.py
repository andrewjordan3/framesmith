# framesmith/_internal/us_states.py
"""
US state, district, and territory reference data for address transforms.

Single source of truth: ``_STATE_NAME_CODE_PAIRS`` for the canonical
name/code data and ``_STATE_ALIASES`` for common abbreviations. Everything
else derives from them. Three address transforms consume the derived views:

* ``standardize_state`` uses ``US_STATE_STANDARDIZE_MAP`` — full names, postal
  codes, and aliases, mapped to the canonical uppercase code.
* ``standardize_state_name`` uses ``US_STATE_NAME_MAP`` — the same inputs
  mapped to the canonical lowercase full name.
* ``strip_trailing_state`` uses ``TRAILING_STATE_CODE_PATTERN`` — built from
  the code set with DC excluded.

Both standardization maps derive from the same ``_STATE_ALIASES`` source, so
the two tools always recognize the same inputs (pinned by an integrity assert).
Alias keys are period-free and lowercase because the lookup key drops periods.

DC asymmetry is deliberate. ``'dc'`` is a valid standardization target
(``'district of columbia'`` / ``'dc'`` -> ``'DC'``) but is excluded from the
trailing-strip code set: stripping ``'dc'`` from ``'washington dc'`` would
collide with Washington state.

``US_STATE_STANDARDIZE_MAP``, ``US_STATE_NAME_MAP``, and
``TRAILING_STATE_CODE_PATTERN`` are the names re-exported through the
``_internal`` init — the only ones used outside ``_internal``. ``US_STATE_CODES``
is file-public for the integrity assert and tests but is not part of the
directory surface; ``_STATE_ALIASES`` and ``_CODE_TO_NAME`` stay file-private.
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


# Common state/territory abbreviations → canonical postal code. Keys are
# period-free and lowercase because the lookup key drops periods, so dotted
# forms ('Calif.', 'W.Va.') match these entries. Abbreviations that collapse
# to a postal code after period-stripping ('Ga.'->'ga', 'N.H.'->'nh', the
# 'N.'/'S.' pairs) are already covered by the code keys and omitted. This is
# common usage (AP style plus widely-used variants), not strict AP.
_STATE_ALIASES: dict[str, str] = {
    # AP-style abbreviations that differ from the postal code:
    'ala': 'AL', 'ariz': 'AZ', 'ark': 'AR', 'calif': 'CA', 'colo': 'CO',
    'conn': 'CT', 'del': 'DE', 'fla': 'FL', 'ill': 'IL', 'ind': 'IN',
    'kan': 'KS', 'mass': 'MA', 'mich': 'MI', 'minn': 'MN', 'miss': 'MS',
    'mont': 'MT', 'neb': 'NE', 'nev': 'NV', 'okla': 'OK', 'ore': 'OR',
    'tenn': 'TN', 'wash': 'WA', 'wva': 'WV', 'wis': 'WI', 'wyo': 'WY',
    # Widely-used non-AP variants (AP spells out Texas; 'tex' included for
    # real-world data):
    'cal': 'CA', 'penn': 'PA', 'penna': 'PA', 'kans': 'KS', 'nebr': 'NE',
    'wisc': 'WI', 'oreg': 'OR', 'tex': 'TX',
    # District of Columbia / territory spaced and long forms:
    'd c': 'DC', 'w va': 'WV', 'usvi': 'VI', 'us virgin islands': 'VI',
}


_CODE_TO_NAME: dict[str, str] = {
    code: full_name for full_name, code in _STATE_NAME_CODE_PAIRS
}


# Whole-value standardization map → canonical postal code: full name, lowercased
# code, and now the aliases.
US_STATE_STANDARDIZE_MAP: dict[str, str] = {
    key: code
    for full_name, code in _STATE_NAME_CODE_PAIRS
    for key in (full_name, code.lower())
} | _STATE_ALIASES


# Whole-value standardization map → canonical lowercase name: full name,
# lowercased code, and aliases, all mapping to the canonical name.
US_STATE_NAME_MAP: dict[str, str] = {
    key: full_name
    for full_name, code in _STATE_NAME_CODE_PAIRS
    for key in (full_name, code.lower())
} | {alias: _CODE_TO_NAME[code] for alias, code in _STATE_ALIASES.items()}


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
assert all(code in _CANONICAL_CODES for code in _STATE_ALIASES.values()), (
    'Every alias must map to a canonical postal code.'
)
assert all(
    '.' not in alias and alias == alias.lower() for alias in _STATE_ALIASES
), 'Alias keys must be lowercase and period-free (the lookup key drops periods).'
assert set(US_STATE_STANDARDIZE_MAP) == set(US_STATE_NAME_MAP), (
    'Both state maps must recognize exactly the same inputs.'
)
assert set(US_STATE_NAME_MAP.values()) == {
    full_name for full_name, _ in _STATE_NAME_CODE_PAIRS
}, 'Name map must output only canonical names.'
