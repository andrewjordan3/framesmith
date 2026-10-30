# framesmith/transforms/binning.py
"""
Numeric binning (discretization) transform factory.

``bin_by_edges`` turns a continuous numeric column into integer bin
indices at caller-supplied edges, the expression-first equivalent of
cutting a feature into ordered groups before modelling or reporting.
It is data-independent (the edges are configuration, not statistics),
so the same edges mean the same bins on every frame it runs against.
"""

import itertools
import math
from collections.abc import Sequence

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = ['bin_by_edges']


def bin_by_edges(edges: Sequence[float]) -> ExpressionTransform:
    """Build a transform that bins values at the given edges.

    With edges ``[e0, e1, ...]`` the bins are the ``len(edges) + 1``
    intervals below ``e0``, ``[e0, e1)``, and so on up to the final
    interval at or above the last edge. The result is the ``Int64``
    index of the interval, from 0 to ``len(edges)``. An edge value
    itself belongs to the interval starting at that edge (lower bounds
    are inclusive), which differs from ``pandas.cut``'s default
    right-inclusive intervals; state edges accordingly.

    Nulls pass through as null and are never assigned a bin. A nan is
    not a null in polars and sorts above every edge, so it lands in
    the final bin.

    Args:
        edges: Bin boundaries, at least one, finite and strictly
            increasing.

    Returns:
        An ``ExpressionTransform`` for ``compose_column``.

    Raises:
        ValueError: If ``edges`` is empty, contains a non-numeric or
            non-finite value, or is not strictly increasing.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import bin_by_edges
        >>> df = pl.DataFrame({'x': [5.0, 18.0, 35.0, 70.0]})
        >>> df.with_columns(
        ...     fs.compose_column('x', [bin_by_edges([18.0, 65.0])])
        ... )['x'].to_list()
        [0, 1, 1, 2]
    """
    if len(edges) == 0:
        raise ValueError('edges must not be empty')
    if any(
        isinstance(edge, bool) or not isinstance(edge, (int, float))
        for edge in edges
    ):
        raise ValueError('edges must be real numbers')
    if any(not math.isfinite(edge) for edge in edges):
        raise ValueError('edges must be finite')
    if any(following <= current for current, following in itertools.pairwise(edges)):
        raise ValueError('edges must be strictly increasing')

    # Copy once, in the factory body: the closure must not observe a
    # caller mutating the original sequence after configuration.
    configured_edges: tuple[float, ...] = tuple(edges)

    def _bin_by_edges(expr: pl.Expr) -> pl.Expr:
        # A value's bin index is exactly the number of edges it has
        # reached: 0 below the first edge, len(edges) at or above the
        # last. Summing the per-edge indicators states that directly,
        # with no when-chain. A null reaches no edge and every
        # indicator for it is null, so the sum stays null.
        binned: pl.Expr = pl.lit(0, dtype=pl.Int64)
        for edge in configured_edges:
            binned = binned + (expr >= edge).cast(pl.Int64)
        return binned

    return _bin_by_edges
