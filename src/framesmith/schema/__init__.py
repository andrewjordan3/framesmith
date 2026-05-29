# framesmith/schema/__init__.py
"""Schema-level operations on column names and frame shape.

Functions return a ``pl.Expr`` applied via ``df.select(...)``; they
never mutate frames. Public surface is exported here; internal file
layout (``column_names.py``) is private — import from
``framesmith.schema``.
"""

from framesmith.schema.column_names import normalize_column_names

__all__: list[str] = ['normalize_column_names']
