# framesmith/transforms/categorical.py
"""
Categorical-consolidation transform factories.

``collapse_rare_by_count`` and ``collapse_keep_top_n`` replace infrequent
category values with a replacement token (``'other'`` by default). Both are
data-dependent: they count the column at execution via a window aggregation,
yet remain pure ``pl.Expr`` transforms that compose in recipes and run lazily
or eagerly.

Opt-in by design — neither belongs in a default recipe, because whether a
category is "rare" depends entirely on the data, and defaulting collapse on
would silently rewrite valid values. Place them last in a pipeline, after
canonicalization, so format variants of one category are not counted as
separate categories. Operates on String columns; nulls pass through unchanged
and are never counted toward any category.
"""

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'collapse_keep_top_n',
    'collapse_rare_by_count',
]


def collapse_rare_by_count(
    min_count: int,
    replacement: str = 'other',
) -> ExpressionTransform:
    """Build a transform that collapses values rarer than ``min_count``.

    A value occurring strictly fewer than ``min_count`` times in the
    column is replaced with ``replacement``; values occurring at least
    ``min_count`` times are kept. Frequency is counted at execution,
    excluding nulls. Nulls pass through unchanged — never counted toward
    any category, never replaced.

    Args:
        min_count: Minimum occurrences for a value to be kept. Must be
            >= 1. A value with strictly fewer occurrences is replaced.
            No default — the threshold is data-specific and explicit at
            every call site.
        replacement: Token substituted for rare values. Default
            ``'other'``.

    Returns:
        An ``ExpressionTransform`` applied via ``compose_column``.

    Raises:
        ValueError: If ``min_count`` is less than 1. A threshold below 1
            keeps everything and is almost certainly a bug.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import collapse_rare_by_count
        >>> df = pl.DataFrame({'c': ['a', 'a', 'b', 'b', 'rare']})
        >>> df.with_columns(
        ...     fs.compose_column('c', [collapse_rare_by_count(2)])
        ... )['c'].to_list()
        ['a', 'a', 'b', 'b', 'other']
    """
    if min_count < 1:
        raise ValueError('min_count must be >= 1')

    def _collapse_rare_by_count(expr: pl.Expr) -> pl.Expr:
        frequency: pl.Expr = expr.count().over(expr)
        return (
            pl.when(expr.is_null())
            .then(expr)
            .when(frequency >= min_count)
            .then(expr)
            .otherwise(pl.lit(replacement))
        )

    return _collapse_rare_by_count


def collapse_keep_top_n(
    n: int,
    replacement: str = 'other',
) -> ExpressionTransform:
    """Build a transform that keeps the top ``n`` frequency tiers.

    Values are ranked by occurrence count (descending, dense rank); those
    in the top ``n`` frequency tiers are kept, the rest replaced with
    ``replacement``. Categories tied at the same frequency share a tier
    and are kept or dropped together, so with ties present more than ``n``
    categories may be kept — this is deterministic, unlike an arbitrary
    head-``n`` tie-break. With no ties it is exactly the ``n`` most
    frequent categories. Frequency is counted at execution, excluding
    nulls. Nulls pass through unchanged.

    Args:
        n: Number of top frequency tiers to keep. Must be >= 1. No
            default — data-specific and explicit at every call site.
        replacement: Token substituted for values outside the top tiers.
            Default ``'other'``.

    Returns:
        An ``ExpressionTransform`` applied via ``compose_column``.

    Raises:
        ValueError: If ``n`` is less than 1. Keeping zero tiers collapses
            everything and is almost certainly a bug.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import collapse_keep_top_n
        >>> df = pl.DataFrame({'c': ['a', 'a', 'a', 'b', 'b', 'c']})
        >>> df.with_columns(
        ...     fs.compose_column('c', [collapse_keep_top_n(2)])
        ... )['c'].to_list()
        ['a', 'a', 'a', 'b', 'b', 'other']
    """
    if n < 1:
        raise ValueError('n must be >= 1')

    def _collapse_keep_top_n(expr: pl.Expr) -> pl.Expr:
        frequency: pl.Expr = expr.count().over(expr)
        frequency_tier: pl.Expr = frequency.rank('dense', descending=True)
        return (
            pl.when(expr.is_null())
            .then(expr)
            .when(frequency_tier <= n)
            .then(expr)
            .otherwise(pl.lit(replacement))
        )

    return _collapse_keep_top_n
