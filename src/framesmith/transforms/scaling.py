# framesmith/transforms/scaling.py
"""
Numeric scaling and reshaping transforms for modelling.

``zscore_numeric`` and ``minmax_numeric`` rescale a numeric column from
statistics computed over the whole column at execution, the same
data-dependent pattern as the outlier and winsorize transforms, so they
stay pure ``pl.Expr`` transforms that compose in recipes and run lazily
or eagerly. ``log1p_numeric`` reshapes a skewed non-negative column.

Scaling is never part of a default recipe: which columns to scale, and
to what form, is a modelling decision for the caller. Statistics are
recomputed on whatever frame the expression runs against, so scaling a
training frame and a scoring frame separately fits each on its own
data. There is no fitted state; a pipeline that must reuse training
statistics should compute them once and use ``clip_numeric``-style
fixed arithmetic instead.
"""

import polars as pl

__all__: list[str] = [
    'log1p_numeric',
    'minmax_numeric',
    'zscore_numeric',
]


def zscore_numeric(expr: pl.Expr) -> pl.Expr:
    """Standardize a numeric column to mean 0, sample std 1.

    Maps each value to ``(x - mean) / std``, with mean and standard
    deviation taken over the whole column. The standard deviation is
    the sample standard deviation (``ddof=1``), matching
    ``flag_zscore_outliers``. Output is ``Float64``.

    A constant column (std 0, at least two values) maps to 0.0: a
    column with no variation sits exactly at its own mean. A single
    non-null value has no sample standard deviation, so it maps to
    null. Nulls pass through as null.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import zscore_numeric
        >>> df = pl.DataFrame({'x': [1.0, 2.0, 3.0]})
        >>> df.with_columns(
        ...     fs.compose_column('x', [zscore_numeric])
        ... )['x'].to_list()
        [-1.0, 0.0, 1.0]
    """
    standard_score: pl.Expr = (expr - expr.mean()) / expr.std()
    # std == 0 (constant column) divides 0 by 0, which is nan, and a
    # nan would silently poison a model matrix; map it to 0.0 as in
    # the outlier-flag guard. is_nan() is null for null inputs, so
    # null values keep a null result rather than becoming 0.0.
    return pl.when(standard_score.is_nan()).then(pl.lit(0.0)).otherwise(standard_score)


def minmax_numeric(expr: pl.Expr) -> pl.Expr:
    """Scale a numeric column linearly into the range [0, 1].

    Maps each value to ``(x - min) / (max - min)``, with min and max
    taken over the whole column: the smallest value becomes 0.0, the
    largest 1.0. Output is ``Float64``.

    A constant column (min == max, including a single non-null value)
    maps to 0.0, the low end of the range. Nulls pass through as null.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import minmax_numeric
        >>> df = pl.DataFrame({'x': [10.0, 20.0, 30.0]})
        >>> df.with_columns(
        ...     fs.compose_column('x', [minmax_numeric])
        ... )['x'].to_list()
        [0.0, 0.5, 1.0]
    """
    scaled: pl.Expr = (expr - expr.min()) / (expr.max() - expr.min())
    # A constant column divides 0 by 0; see zscore_numeric for why the
    # nan result is mapped to the range's low end instead of passing
    # through.
    return pl.when(scaled.is_nan()).then(pl.lit(0.0)).otherwise(scaled)


def log1p_numeric(expr: pl.Expr) -> pl.Expr:
    """Apply ``log(1 + x)`` to a non-negative numeric column.

    The standard reshaping for right-skewed counts and amounts: 0 maps
    to 0.0, large values are compressed, and the result is safe to
    scale afterwards. Output is ``Float64``.

    The domain is non-negative values. A negative value becomes null
    rather than a nan or a negative logarithm, so bad data shows up as
    missingness a caller can count instead of silently propagating.
    Nulls pass through as null.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import log1p_numeric
        >>> df = pl.DataFrame({'x': [0.0, 3.0, -2.0, None]})
        >>> result = df.with_columns(
        ...     fs.compose_column('x', [log1p_numeric])
        ... )['x'].to_list()
        >>> result[0], round(result[1], 6), result[2], result[3]
        (0.0, 1.386294, None, None)
    """
    return (
        pl.when(expr < 0)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise((expr + 1).log())
    )
