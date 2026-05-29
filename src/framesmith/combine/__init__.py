# framesmith/combine/__init__.py
"""Multi-column combination operations.

Functions return a ``pl.Expr`` applied via ``df.with_columns(...)``; they
never mutate frames. Unlike single-column transforms they reference several
source columns by name. Public surface is exported here; internal file
layout (``coalesce.py``, ``concat.py``, ``hash_key.py``) is private — import
from ``framesmith.combine``.
"""

from framesmith.combine.coalesce import coalesce_blank_columns
from framesmith.combine.concat import combine_columns
from framesmith.combine.hash_key import hash_key

__all__: list[str] = [
    'coalesce_blank_columns',
    'combine_columns',
    'hash_key',
]
