# framesmith/transforms/bounds.py
"""
Numeric value-bounding transforms.

``clip_numeric`` clamps to hard user-supplied bounds; ``winsorize_numeric``
clamps to data-driven percentile bounds. Both return an
``ExpressionTransform`` producing a bounded numeric column, preserving the
input dtype and nulls. Apply via ``compose_column``.
"""

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = ['clip_numeric', 'winsorize_numeric']


def clip_numeric(
    lower: float | None = None,
    upper: float | None = None,
) -> ExpressionTransform:
    """Build a transform clamping values to a hard ``[lower, upper]`` range.

    Values below ``lower`` become ``lower``; values above ``upper`` become
    ``upper``; values within (and exactly at) the bounds are unchanged.
    At least one bound is required; omit one for a one-sided clamp. Nulls
    and the column dtype are preserved.

    Args:
        lower: Minimum value (inclusive). ``None`` leaves the low side
            unbounded.
        upper: Maximum value (inclusive). ``None`` leaves the high side
            unbounded.

    Returns:
        An ``ExpressionTransform`` for ``compose_column``.

    Raises:
        ValueError: If both bounds are ``None``, or if ``lower`` exceeds
            ``upper``.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import clip_numeric
        >>> df = pl.DataFrame({'x': [1.0, 5.0, 50.0]})
        >>> df.with_columns(
        ...     fs.compose_column('x', [clip_numeric(lower=0.0, upper=10.0)])
        ... )['x'].to_list()
        [1.0, 5.0, 10.0]
    """
    if lower is None and upper is None:
        raise ValueError('clip_numeric requires at least one of lower or upper.')
    if lower is not None and upper is not None and lower > upper:
        raise ValueError('lower must not exceed upper.')

    def _clip_numeric(expr: pl.Expr) -> pl.Expr:
        return expr.clip(lower_bound=lower, upper_bound=upper)

    return _clip_numeric


def winsorize_numeric(
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> ExpressionTransform:
    """Build a transform capping extremes at data-driven percentiles.

    Clamps each value to ``[quantile(lower_quantile),
    quantile(upper_quantile)]``, with quantiles taken over the whole
    column using linear interpolation. By default caps the outer 5% of
    each tail (``0.05`` / ``0.95``). Values below the lower percentile
    become that percentile; values above the upper become the upper
    percentile. Nulls and dtype are preserved; a constant column is
    unchanged.

    Args:
        lower_quantile: Lower percentile in ``[0, 1)``. Default 0.05.
        upper_quantile: Upper percentile in ``(0, 1]``, strictly greater
            than ``lower_quantile``. Default 0.95.

    Returns:
        An ``ExpressionTransform`` for ``compose_column``.

    Raises:
        ValueError: Unless ``0 <= lower_quantile < upper_quantile <= 1``.
    """
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            'require 0 <= lower_quantile < upper_quantile <= 1.'
        )

    def _winsorize_numeric(expr: pl.Expr) -> pl.Expr:
        lower_bound: pl.Expr = expr.quantile(
            lower_quantile, interpolation='linear'
        )
        upper_bound: pl.Expr = expr.quantile(
            upper_quantile, interpolation='linear'
        )
        return expr.clip(lower_bound, upper_bound)

    return _winsorize_numeric
