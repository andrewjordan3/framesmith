# src/framesmith/__init__.py
"""framesmith — composable preprocessing for polars DataFrames."""

import logging

from framesmith.compose import compose_column
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'ExpressionTransform',
    'compose_column',
]

# Library convention (Python logging HOWTO): attach a NullHandler so that
# importing framesmith without configuring logging does not emit the
# "no handlers could be found" warning.
# Library convention (Python logging HOWTO): attach a NullHandler so that
# importing framesmith without configuring logging does not emit the
# "no handlers could be found" warning. Applications that want framesmith
# log records can attach their own handlers to the 'framesmith' logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
