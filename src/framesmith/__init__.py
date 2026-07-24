# src/framesmith/__init__.py
"""framesmith — composable preprocessing for polars DataFrames.

Top-level surface is intentionally narrow: the composition orchestrator,
the foundational type alias, and the curated recipes. Atomic transforms
live on ``framesmith.transforms``; row filters on ``framesmith.filters``.
Subpackage inits are each their own public surface — callers import
from the directory, not from individual files.
"""

import logging

from framesmith.compose import compose_column
from framesmith.recipes import (
    CATEGORY_CANONICALIZE,
    CATEGORY_CANONICALIZE_TO_SNAKE_CASE,
    EMAIL_TO_DISPLAY_NAME,
    EMAIL_TO_TITLE_NAME,
    NUMERIC_STRING_NORMALIZE,
    NUMERIC_STRING_TO_FLOAT,
    PERCENT_STRING_TO_FRACTION,
    PERSON_NAME_NORMALIZE,
    PERSON_NAME_NORMALIZE_TO_SNAKE_CASE,
    SNAKE_CASE_TO_TITLE,
    STREET_ADDRESS_NORMALIZE,
    STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE,
    TEXT_CANONICALIZE,
    TEXT_NORMALIZE,
    TEXT_NORMALIZE_TO_SNAKE_CASE,
    UNICODE_TO_ASCII,
    UPPERCASE_CODE_NORMALIZE,
    WHITESPACE_CANONICALIZE,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'CATEGORY_CANONICALIZE',
    'CATEGORY_CANONICALIZE_TO_SNAKE_CASE',
    'EMAIL_TO_DISPLAY_NAME',
    'EMAIL_TO_TITLE_NAME',
    'NUMERIC_STRING_NORMALIZE',
    'NUMERIC_STRING_TO_FLOAT',
    'PERCENT_STRING_TO_FRACTION',
    'PERSON_NAME_NORMALIZE',
    'PERSON_NAME_NORMALIZE_TO_SNAKE_CASE',
    'SNAKE_CASE_TO_TITLE',
    'STREET_ADDRESS_NORMALIZE',
    'STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE',
    'TEXT_CANONICALIZE',
    'TEXT_NORMALIZE',
    'TEXT_NORMALIZE_TO_SNAKE_CASE',
    'UNICODE_TO_ASCII',
    'UPPERCASE_CODE_NORMALIZE',
    'WHITESPACE_CANONICALIZE',
    'ExpressionTransform',
    'compose_column',
]

# Library convention (Python logging HOWTO): attach a NullHandler so that
# importing framesmith without configuring logging does not emit the
# "no handlers could be found" warning. Applications that want framesmith
# log records can attach their own handlers to the 'framesmith' logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
