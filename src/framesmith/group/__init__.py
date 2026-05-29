# framesmith/group/__init__.py
"""Group-resolution operations.

Functions return a ``pl.Expr`` applied via ``df.group_by(...).agg(...)``; the
caller supplies the grouping frame method, framesmith supplies the resolution
expression. Public surface is exported here; internal file layout
(``resolve.py``) is private — import from ``framesmith.group``.
"""

from framesmith.group.resolve import first_non_null_per_group

__all__: list[str] = ['first_non_null_per_group']
