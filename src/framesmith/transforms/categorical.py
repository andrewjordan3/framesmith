# framesmith/transforms/categorical.py
"""
Categorical-value transform factories.

``map_categories`` remaps whole cell values through an exact-match lookup
table — a pure, data-independent value substitution. ``collapse_rare_by_count``
and ``collapse_keep_top_n`` replace infrequent category values with a
replacement token (``'other'`` by default); these two are data-dependent,
counting the column at execution via a window aggregation, yet remain pure
``pl.Expr`` transforms that compose in recipes and run lazily or eagerly.

The collapse factories are opt-in by design — neither belongs in a default
recipe, because whether a category is "rare" depends entirely on the data, and
defaulting collapse on would silently rewrite valid values. Place them last in
a pipeline, after canonicalization, so format variants of one category are not
counted as separate categories. They operate on String columns; nulls pass
through unchanged and are never counted toward any category.
"""

from collections.abc import Hashable, Mapping

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'collapse_keep_top_n',
    'collapse_rare_by_count',
    'map_categories',
]


def map_categories(
    category_map: Mapping[Hashable, Hashable],
) -> ExpressionTransform:
    """Remap whole cell values through an exact-match lookup table.

    Each cell whose value exactly matches a key in ``category_map`` is
    replaced with the mapped value; every other value passes through. The
    match is **exact and case-sensitive** — no trimming, case-folding, or
    substring logic. Compose
    :func:`~framesmith.transforms.to_lowercase` and
    :func:`~framesmith.transforms.strip_whitespace` upstream if the raw
    values need normalizing before the lookup.

    Keys and values may be any hashable type, so a code→label map such as
    ``{1: 'Yes', 2: 'No'}`` works. The output column's dtype follows the
    map's **values**: a String-valued map yields a ``String`` column, an
    Int-valued map yields ``Int64``.

    Null handling: a null passes through as null unless ``None`` is an
    explicit key in the map.

    Cross-type passthrough: when the values' dtype differs from the
    column's dtype, **unmapped values are cast to the output (value)
    dtype**, not preserved in their original type. With ``{1: 'Yes', 2:
    'No'}`` over ``[1, 2, 3]`` the unmapped ``3`` comes out as the string
    ``"3"``, because the whole column becomes ``String``. For a same-type
    map (``str`` → ``str``, ``int`` → ``int``) the passthrough is a true
    no-op and the dtype is unchanged.

    This is whole-value remapping, distinct from
    :func:`~framesmith.transforms.apply_replacements`, which rewrites
    matching **substrings**.

    Args:
        category_map: Exact-match lookup of input value → output value.
            Keys and values may be any hashable type; the value types
            must be mutually compatible — polars raises ``TypeError`` when
            building the replacement table if they are not (e.g. mixing
            ``str`` and ``int`` values). Must be non-empty.

    Returns:
        An ``ExpressionTransform`` applied via ``compose_column``.

    Raises:
        ValueError: If ``category_map`` is empty. An empty map is an
            identity transform, almost certainly a bug, and is rejected
            loudly rather than silently producing a no-op.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import map_categories
        >>> df = pl.DataFrame({'flag': [1, 2, 3]})
        >>> df.with_columns(
        ...     fs.compose_column('flag', [map_categories({1: 'Yes', 2: 'No'})])
        ... )['flag'].to_list()
        ['Yes', 'No', '3']
    """
    if len(category_map) == 0:
        raise ValueError('category_map must not be empty')

    def _map_categories(expr: pl.Expr) -> pl.Expr:
        # replace_strict, NOT replace: plain `replace` is dtype-preserving
        # and raises on a cross-type map like {1: 'Yes'} (it tries to cast
        # the new values back to the column's dtype). `default=expr`
        # supplies the passthrough that `replace` would otherwise give,
        # while letting the output dtype follow the map's values.
        return expr.replace_strict(category_map, default=expr)

    return _map_categories


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
