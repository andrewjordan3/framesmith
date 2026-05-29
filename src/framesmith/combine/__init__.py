# framesmith/combine/__init__.py
"""Multi-column combination operations.

Functions return a ``pl.Expr`` applied via ``df.with_columns(...)``; they
never mutate frames. Unlike single-column transforms they reference several
source columns by name. Public surface is exported here; internal file
layout (``concat.py``) is private — import from ``framesmith.combine``.
"""

from framesmith.combine.concat import combine_columns

__all__: list[str] = ['combine_columns']
