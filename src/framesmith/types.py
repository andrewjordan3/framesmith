# framesmith/types.py
"""
Public type aliases for framesmith's expression composition layer.

These aliases are exported from the top-level ``framesmith`` package so
that user code can annotate its own transforms and curated recipes
without reaching into framesmith's internals.

ExpressionTransform contract
----------------------------
An ``ExpressionTransform`` is a function ``Callable[[pl.Expr], pl.Expr]``
with the following contract:

1. It receives an expression representing the current state of a column
   (initially ``pl.col(source_column_name)``, then progressively
   transformed).
2. It returns a new expression representing the next state.
3. It MUST NOT call ``.alias(...)``. Aliasing is exclusively the
   responsibility of ``compose_column``. A transform that self-aliases
   will have its alias silently overridden, which makes its code
   misleading.
4. It MUST NOT call ``pl.col(...)``. The input expression already
   represents the column.

Transforms are pure expression-to-expression functions and are safe to
store in tuples, splice into recipes, and pass through configuration.
"""

from collections.abc import Callable

import polars as pl

__all__: list[str] = ['ExpressionTransform']


type ExpressionTransform = Callable[[pl.Expr], pl.Expr]
