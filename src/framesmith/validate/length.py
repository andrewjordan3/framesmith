# framesmith/validate/length.py
"""String-length validation guard."""

import polars as pl

__all__: list[str] = ['assert_string_length']

_LENGTH_VIOLATION_EXAMPLE_LIMIT: int = 5


def assert_string_length(
    df: pl.DataFrame,
    column: str,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    ignore_null: bool = False,
) -> None:
    """Raise if any value in ``column`` falls outside the length bounds.

    A guard, not a transform: returns ``None`` when every value satisfies
    the bounds, raises otherwise. At least one of ``min_length`` /
    ``max_length`` must be given (both ``None`` is a no-op check, which is
    rejected). Length is measured in characters (``str.len_chars``).

    Args:
        df: The frame to inspect.
        column: Name of the string column to check.
        min_length: If set, values shorter than this are violations.
        max_length: If set, values longer than this are violations.
        ignore_null: When ``False`` (default), a null is a violation; when
            ``True``, nulls are skipped.

    Returns:
        None.

    Raises:
        ValueError: If neither bound is given, or if any value violates
            the bounds. The message reports the violation count and up to
            five example values.

    Note:
        Materializes a single count aggregation on the success path; on
        failure it additionally pulls a few example values for the
        message. A non-string column raises ``SchemaError`` (from
        ``str.len_chars``); a missing column raises
        ``ColumnNotFoundError``.

    Example:
        >>> import polars as pl
        >>> from framesmith.validate import assert_string_length
        >>> df = pl.DataFrame({'vin': ['1HGCM82633A004352']})
        >>> assert_string_length(df, 'vin', min_length=17, max_length=17)
    """
    if min_length is None and max_length is None:
        raise ValueError(
            'at least one of min_length or max_length must be given'
        )

    length: pl.Expr = pl.col(column).str.len_chars()
    too_short: pl.Expr = (
        length < min_length if min_length is not None else pl.lit(False)
    )
    too_long: pl.Expr = (
        length > max_length if max_length is not None else pl.lit(False)
    )
    out_of_bounds: pl.Expr = too_short | too_long
    # A null has no length: it is a violation unless explicitly ignored.
    violation: pl.Expr = (
        pl.when(pl.col(column).is_null())
        .then(pl.lit(not ignore_null))
        .otherwise(out_of_bounds)
    )

    violation_count: int = df.select(violation.sum()).item()
    if violation_count:
        examples: list[str | None] = (
            df.filter(violation)
            .select(column)
            .head(_LENGTH_VIOLATION_EXAMPLE_LIMIT)
            .to_series()
            .to_list()
        )
        bounds: str = f'[{min_length}, {max_length}]'
        raise ValueError(
            f'column {column!r} has {violation_count} value(s) with length '
            f'outside {bounds}; examples: {examples!r}'
        )
