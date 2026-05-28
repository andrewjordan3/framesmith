# framesmith/transforms/__init__.py
"""Column-level expression transforms.

Public surface for this subpackage is exported here. Internal file
organization (``text.py``, ``names.py``, ``numeric.py``, ``missing.py``)
is private — callers import from ``framesmith.transforms``.
"""

from framesmith.transforms.missing import (
    DEFAULT_MISSING_SENTINELS,
    nullify_sentinels,
)
from framesmith.transforms.names import remove_jr_suffix
from framesmith.transforms.numeric import (
    accounting_parens_to_negative,
    cast_to_float64,
    percent_to_fraction,
    remove_thousands_separators,
    trailing_minus_to_prefix,
)
from framesmith.transforms.text import (
    collapse_whitespace,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    remove_apostrophes,
    remove_periods,
    replace_ampersand_with_and,
    replace_whitespace_with,
    strip_whitespace,
    to_snake_case,
)

__all__: list[str] = [
    'DEFAULT_MISSING_SENTINELS',
    'accounting_parens_to_negative',
    'cast_to_float64',
    'collapse_whitespace',
    'fold_to_ascii',
    'normalize_unicode_nfkc',
    'nullify_blank_strings',
    'nullify_sentinels',
    'percent_to_fraction',
    'remove_apostrophes',
    'remove_jr_suffix',
    'remove_periods',
    'remove_thousands_separators',
    'replace_ampersand_with_and',
    'replace_whitespace_with',
    'strip_whitespace',
    'to_snake_case',
    'trailing_minus_to_prefix',
]
