# framesmith/_internal/__init__.py
"""Internal building blocks: regex pattern strings and unicode maps.

This is the public surface of ``_internal/`` for the rest of the
package. Submodules (``regex_patterns.py``, ``unicode_maps.py``) are
private — cross-directory callers import from ``framesmith._internal``.

Only the maps and patterns actually used outside ``_internal/`` are
re-exported. The six sub-maps (``MINUS_LIKE_MAP``, ``INVISIBLE_CHAR_MAP``,
``WHITESPACE_VARIANT_MAP``, ``CURRENCY_SYMBOL_MAP``, ``QUOTE_AND_PRIME_MAP``,
``PUNCTUATION_SYMBOL_MAP``) are file-level public on ``unicode_maps``
but are not part of this directory's external surface — they exist only
to assemble ``ASCII_COMPAT_MAP``.
"""

from framesmith._internal.regex_patterns import (
    BLANK_OR_WHITESPACE_ONLY_PATTERN,
    PAREN_NEGATIVE_PATTERN,
    THOUSANDS_SEPARATOR_PATTERN,
    TRAILING_JR_PATTERN,
    TRAILING_MINUS_PATTERN,
    WHITESPACE_RUN_PATTERN,
)
from framesmith._internal.unicode_maps import ASCII_COMPAT_MAP

__all__: list[str] = [
    'ASCII_COMPAT_MAP',
    'BLANK_OR_WHITESPACE_ONLY_PATTERN',
    'PAREN_NEGATIVE_PATTERN',
    'THOUSANDS_SEPARATOR_PATTERN',
    'TRAILING_JR_PATTERN',
    'TRAILING_MINUS_PATTERN',
    'WHITESPACE_RUN_PATTERN',
]
