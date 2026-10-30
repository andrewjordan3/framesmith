# framesmith/compose.py
"""
Expression composition for framesmith.

``compose_column`` is the single boundary between column names (the user's
mental model) and polars expressions (the operation). Transforms operate
purely on expressions; this function handles the ``pl.col(...)`` opening
and the ``.alias(...)`` closing, applying transforms in between.
"""

from collections.abc import Mapping, Sequence

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = ['compose_column', 'compose_columns']


def compose_column(
    source_column_name: str,
    expression_transforms: Sequence[ExpressionTransform],
    output_column_name: str | None = None,
) -> pl.Expr:
    """Apply a sequence of expression transforms to a named column.

    Builds ``pl.col(source_column_name)``, applies each transform in
    order, and aliases the final result. If ``output_column_name`` is
    omitted, the result is aliased back to ``source_column_name``
    (overwriting the source column when used inside ``with_columns``).

    Args:
        source_column_name: Name of the source column.
        expression_transforms: Ordered sequence of transforms to apply.
            Must be non-empty.
        output_column_name: Optional alias for the result. Defaults to
            ``source_column_name``.

    Returns:
        A composed ``pl.Expr`` aliased to the output column name.

    Raises:
        ValueError: If ``expression_transforms`` is empty. An empty
            sequence would produce an identity expression, which is
            almost certainly a bug (forgotten transforms, empty config
            section, etc.) and is rejected loudly rather than silently
            producing a no-op.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>>
        >>> def strip_chars(expr: pl.Expr) -> pl.Expr:
        ...     return expr.str.strip_chars()
        >>>
        >>> def lowercase(expr: pl.Expr) -> pl.Expr:
        ...     return expr.str.to_lowercase()
        >>>
        >>> df = pl.DataFrame({'name': ['  Alice  ', '  BOB  ']})
        >>> df.with_columns(fs.compose_column('name', [strip_chars, lowercase]))
        shape: (2, 1)
        ┌───────┐
        │ name  │
        │ ---   │
        │ str   │
        ╞═══════╡
        │ alice │
        │ bob   │
        └───────┘
    """
    if len(expression_transforms) == 0:
        raise ValueError('expression_transforms must not be empty')

    current_expression: pl.Expr = pl.col(source_column_name)
    for transform in expression_transforms:
        current_expression = transform(current_expression)
    return current_expression.alias(output_column_name or source_column_name)


def compose_columns(
    column_recipes: Mapping[str, Sequence[ExpressionTransform]],
) -> list[pl.Expr]:
    """Compose one recipe per column into a list of expressions.

    The multi-column counterpart of ``compose_column``: give it a
    mapping of column name to the recipe (or ad-hoc transform list)
    for that column, and pass the returned list straight to
    ``df.with_columns(...)`` or ``lf.with_columns(...)``. Each column
    is composed by ``compose_column`` with its default in-place
    aliasing, in the mapping's order, so the result is exactly the
    expressions a caller would have built one call at a time. Nothing
    is executed here; the list is expressions all the way down, so it
    runs eagerly or lazily like any other framesmith composition.

    Args:
        column_recipes: Mapping of source column name to its ordered
            transform sequence. Must be non-empty, and each sequence
            must be non-empty (enforced by ``compose_column``).

    Returns:
        One composed ``pl.Expr`` per mapping entry, in mapping order.

    Raises:
        ValueError: If ``column_recipes`` is empty, or if any column's
            transform sequence is empty.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> df = pl.DataFrame(
        ...     {'name': ['  ALICE '], 'city': ['  Chicago ']}
        ... )
        >>> df.with_columns(
        ...     fs.compose_columns(
        ...         {
        ...             'name': fs.TEXT_CANONICALIZE,
        ...             'city': fs.TEXT_CANONICALIZE,
        ...         }
        ...     )
        ... )
        shape: (1, 2)
        ┌───────┬─────────┐
        │ name  ┆ city    │
        │ ---   ┆ ---     │
        │ str   ┆ str     │
        ╞═══════╪═════════╡
        │ ALICE ┆ Chicago │
        └───────┴─────────┘
    """
    if len(column_recipes) == 0:
        raise ValueError('column_recipes must not be empty')
    return [
        compose_column(column_name, recipe)
        for column_name, recipe in column_recipes.items()
    ]
