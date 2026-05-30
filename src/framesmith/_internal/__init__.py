# framesmith/_internal/__init__.py
"""Internal building blocks: regex pattern strings, unicode maps,
US state reference data, and name-affix token sets.

This is the public surface of ``_internal/`` for the rest of the
package. Submodules (``regex_patterns.py``, ``unicode_maps.py``,
``us_states.py``, ``name_tokens.py``) are private — cross-directory
callers import from ``framesmith._internal``.

Only the maps and patterns actually used outside ``_internal/`` are
re-exported. The six sub-maps (``MINUS_LIKE_MAP``, ``INVISIBLE_CHAR_MAP``,
``WHITESPACE_VARIANT_MAP``, ``CURRENCY_SYMBOL_MAP``, ``QUOTE_AND_PRIME_MAP``,
``PUNCTUATION_SYMBOL_MAP``) are file-level public on ``unicode_maps``
but are not part of this directory's external surface — they exist only
to assemble ``ASCII_COMPAT_MAP``.
"""

from framesmith._internal.name_tokens import (
    DEFAULT_CREDENTIALS,
    DEFAULT_NAME_PREFIXES,
    DEFAULT_NAME_SUFFIXES,
)
from framesmith._internal.regex_patterns import (
    BLANK_OR_WHITESPACE_ONLY_PATTERN,
    PAREN_NEGATIVE_PATTERN,
    STANDALONE_INITIAL_PATTERN,
    THOUSANDS_SEPARATOR_PATTERN,
    TRAILING_JR_PATTERN,
    TRAILING_MINUS_PATTERN,
    WHITESPACE_RUN_PATTERN,
)
from framesmith._internal.unicode_maps import ASCII_COMPAT_MAP
from framesmith._internal.us_states import (
    TRAILING_STATE_CODE_PATTERN,
    US_STATE_NAME_MAP,
    US_STATE_STANDARDIZE_MAP,
)

__all__: list[str] = [
    'ASCII_COMPAT_MAP',
    'BLANK_OR_WHITESPACE_ONLY_PATTERN',
    'DEFAULT_CREDENTIALS',
    'DEFAULT_NAME_PREFIXES',
    'DEFAULT_NAME_SUFFIXES',
    'PAREN_NEGATIVE_PATTERN',
    'STANDALONE_INITIAL_PATTERN',
    'THOUSANDS_SEPARATOR_PATTERN',
    'TRAILING_JR_PATTERN',
    'TRAILING_MINUS_PATTERN',
    'TRAILING_STATE_CODE_PATTERN',
    'US_STATE_NAME_MAP',
    'US_STATE_STANDARDIZE_MAP',
    'WHITESPACE_RUN_PATTERN',
]
