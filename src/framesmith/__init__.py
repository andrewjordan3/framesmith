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
    CLEAN_NUMERIC_STRING,
    NORMALIZE_NUMERIC,
    NORMALIZE_PERCENT,
    NORMALIZE_TEXT,
    UNICODE_TO_ASCII,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'CLEAN_NUMERIC_STRING',
    'NORMALIZE_NUMERIC',
    'NORMALIZE_PERCENT',
    'NORMALIZE_TEXT',
    'UNICODE_TO_ASCII',
    'ExpressionTransform',
    'compose_column',
]

# Library convention (Python logging HOWTO): attach a NullHandler so that
# importing framesmith without configuring logging does not emit the
# "no handlers could be found" warning. Applications that want framesmith
# log records can attach their own handlers to the 'framesmith' logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
