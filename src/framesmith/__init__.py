# src/framesmith/__init__.py
"""framesmith — composable preprocessing for polars DataFrames."""

import logging

# Library convention (Python logging HOWTO): attach a NullHandler so that
# importing framesmith without configuring logging does not emit the
# "no handlers could be found" warning. Applications that want framesmith
# log records can attach their own handlers to the 'framesmith' logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
