# framesmith/group/mode.py
"""
Shared mode-resolution primitive for the group operations.

``first_mode_in_order`` builds the expression for "the most common non-null
value, ties broken by first appearance in input order." It is the shared core
of ``mode_then_first_per_group`` (aggregation, applied in ``group_by().agg``)
and ``fill_null_string_by_group`` (window fill, wrapped in ``.over``). It lives
in its own module so neither operation depends on the other.

This symbol is an internal building block: sibling modules in
``framesmith.group`` import it, but it is intentionally not part of the
package's public surface (not re-exported in ``group/__init__.py``).
"""

import polars as pl

__all__: list[str] = ['first_mode_in_order']


def first_mode_in_order(values: pl.Expr) -> pl.Expr:
    """Build an expression selecting the most common non-null value.

    Drops nulls, then keeps the value(s) tied for most frequent and takes
    the first in input order. ``mode().first()`` alone is not order-stable
    on a tie; the ``filter(is_in(mode().implode()))`` + ``first()``
    construction pins the result to the first tied value as it appears in
    the input.

    The caller supplies the application context: aggregation collapses it
    to one value per group via ``group_by().agg``; a window fill broadcasts
    it across the group via ``.over(...)``.

    Args:
        values: The expression to resolve (e.g. ``pl.col('x')`` or
            ``pl.all().exclude(keys)``).

    Returns:
        A ``pl.Expr`` yielding the first-in-order most-common non-null
        value, or null if ``values`` has no non-null entry.
    """
    non_null_values: pl.Expr = values.drop_nulls()
    return non_null_values.filter(
        non_null_values.is_in(non_null_values.mode().implode())
    ).first()
