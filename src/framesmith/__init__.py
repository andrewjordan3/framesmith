# src/framesmith/__init__.py
"""framesmith — composable preprocessing for polars DataFrames."""

import logging

from framesmith.compose import compose_column
from framesmith.recipes import (
    CLEAN_NUMERIC_STRING,
    NORMALIZE_NUMERIC,
    NORMALIZE_TEXT,
    UNICODE_TO_ASCII,
)
from framesmith.transforms.names import remove_jr_suffix
from framesmith.transforms.numeric import (
    accounting_parens_to_negative,
    cast_to_float64,
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
    strip_whitespace,
    to_snake_case,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'CLEAN_NUMERIC_STRING',
    'NORMALIZE_NUMERIC',
    'NORMALIZE_TEXT',
    'UNICODE_TO_ASCII',
    'ExpressionTransform',
    'accounting_parens_to_negative',
    'cast_to_float64',
    'collapse_whitespace',
    'compose_column',
    'fold_to_ascii',
    'normalize_unicode_nfkc',
    'nullify_blank_strings',
    'remove_apostrophes',
    'remove_jr_suffix',
    'remove_periods',
    'remove_thousands_separators',
    'replace_ampersand_with_and',
    'strip_whitespace',
    'to_snake_case',
    'trailing_minus_to_prefix',
]

# Library convention (Python logging HOWTO): attach a NullHandler so that
# importing framesmith without configuring logging does not emit the
# "no handlers could be found" warning. Applications that want framesmith
# log records can attach their own handlers to the 'framesmith' logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
