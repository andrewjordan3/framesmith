# framesmith/compose.py
"""
Expression composition for framesmith.

``compose_column`` is the single boundary between column names (the user's
mental model) and polars expressions (the operation). Transforms operate
purely on expressions; this function handles the ``pl.col(...)`` opening
and the ``.alias(...)`` closing, applying transforms in between.
"""

from collections.abc import Sequence

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = ['compose_column']


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
