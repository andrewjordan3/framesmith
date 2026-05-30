# framesmith/transforms/outliers.py
"""
Outlier-flagging transforms.

Each factory returns an ``ExpressionTransform`` mapping a numeric column to a
boolean flag (true = outlier), using a statistic computed over the whole
column. Apply via ``compose_column`` to build a flag column. Nulls are skipped
in the statistic and yield a null flag.
"""

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'flag_iqr_outliers',
    'flag_mad_outliers',
    'flag_zscore_outliers',
]


# Iglewicz-Hoaglin modified z-score scaling constant: the 0.75 quantile of the
# standard normal, which scales the median absolute deviation to be comparable
# to a standard deviation for normally distributed data.
_MAD_SCALE: float = 0.6745


def flag_zscore_outliers(threshold: float = 3.0) -> ExpressionTransform:
    """Build a transform flagging values beyond ``threshold`` standard scores.

    Flags ``x`` where ``|(x - mean) / std| > threshold``, with mean and
    std taken over the whole column. Default 3.0 (the 3-sigma
    convention). True = outlier; a null input yields a null flag.

    A constant column (``std == 0``) flags nothing. Note z-score is
    sensitive to the very extremes it detects — a lone large value
    inflates the std and can mask itself; ``flag_mad_outliers`` or
    ``flag_iqr_outliers`` are more robust for heavy-tailed data.

    Args:
        threshold: Positive standard-score cutoff. Default 3.0.

    Returns:
        An ``ExpressionTransform`` (true = outlier) for ``compose_column``.

    Raises:
        ValueError: If ``threshold`` is not positive.
    """
    if threshold <= 0:
        raise ValueError('threshold must be positive.')

    def _flag_zscore_outliers(expr: pl.Expr) -> pl.Expr:
        standard_score: pl.Expr = (expr - expr.mean()) / expr.std()
        # std==0 (constant column) -> nan, and nan > threshold is True in
        # polars; guard so a constant column flags nothing. is_nan() is
        # null for null inputs, so null values keep a null flag.
        return (
            pl.when(standard_score.is_nan())
            .then(pl.lit(False))
            .otherwise(standard_score.abs() > threshold)
        )

    return _flag_zscore_outliers


def flag_iqr_outliers(multiplier: float = 1.5) -> ExpressionTransform:
    """Build a transform flagging values outside the Tukey IQR fences.

    Flags ``x`` below ``Q1 - multiplier * IQR`` or above
    ``Q3 + multiplier * IQR``, where ``IQR = Q3 - Q1`` and quartiles use
    linear interpolation. Default 1.5 (Tukey's convention). True =
    outlier; a null input yields a null flag. A constant column flags
    nothing.

    Args:
        multiplier: Positive IQR multiplier for the fences. Default 1.5.

    Returns:
        An ``ExpressionTransform`` (true = outlier) for ``compose_column``.

    Raises:
        ValueError: If ``multiplier`` is not positive.
    """
    if multiplier <= 0:
        raise ValueError('multiplier must be positive.')

    def _flag_iqr_outliers(expr: pl.Expr) -> pl.Expr:
        first_quartile: pl.Expr = expr.quantile(0.25, interpolation='linear')
        third_quartile: pl.Expr = expr.quantile(0.75, interpolation='linear')
        interquartile_range: pl.Expr = third_quartile - first_quartile
        lower_fence: pl.Expr = first_quartile - multiplier * interquartile_range
        upper_fence: pl.Expr = third_quartile + multiplier * interquartile_range
        return (expr < lower_fence) | (expr > upper_fence)

    return _flag_iqr_outliers


def flag_mad_outliers(threshold: float = 3.5) -> ExpressionTransform:
    """Build a transform flagging values by the modified z-score (MAD).

    Uses the Iglewicz-Hoaglin modified z-score:
    ``0.6745 * (x - median) / MAD``, where ``MAD = median(|x - median|)``,
    flagging ``|modified z| > threshold``. Default 3.5. True = outlier; a
    null input yields a null flag. A constant column (``MAD == 0``) flags
    nothing. This is the most robust of the three for heavy-tailed data.

    Args:
        threshold: Positive modified-z cutoff. Default 3.5.

    Returns:
        An ``ExpressionTransform`` (true = outlier) for ``compose_column``.

    Raises:
        ValueError: If ``threshold`` is not positive.
    """
    if threshold <= 0:
        raise ValueError('threshold must be positive.')

    def _flag_mad_outliers(expr: pl.Expr) -> pl.Expr:
        median: pl.Expr = expr.median()
        median_absolute_deviation: pl.Expr = (expr - median).abs().median()
        modified_zscore: pl.Expr = (
            _MAD_SCALE * (expr - median) / median_absolute_deviation
        )
        # MAD==0 -> nan, and nan > threshold is True in polars; guard as in
        # the z-score case. Null inputs keep a null flag.
        return (
            pl.when(modified_zscore.is_nan())
            .then(pl.lit(False))
            .otherwise(modified_zscore.abs() > threshold)
        )

    return _flag_mad_outliers
