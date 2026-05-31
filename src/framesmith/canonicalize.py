# framesmith/canonicalize.py
"""Domain-specific eager canonicalizers.

Functions here clean a column into a canonical form *and* validate the result,
returning the frame or raising. They are eager (``DataFrame -> DataFrame``)
because they pair a pure transform with a validation guard, and the guard must
inspect data. They compose existing framesmith pieces; the value they add is
encoding the domain knowledge (the fixed width, the pad-then-check order) in
one place.
"""

import polars as pl

from framesmith.compose import compose_column
from framesmith.transforms import pad_left
from framesmith.validate import assert_string_length

__all__: list[str] = ['canonicalize_vin']

_VIN_LENGTH: int = 17


def canonicalize_vin(
    df: pl.DataFrame,
    column: str,
    *,
    ignore_null: bool = False,
) -> pl.DataFrame:
    """Left-pad a VIN column to 17 characters, then validate the result.

    Restores leading zeros so the same vehicle matches across systems that
    store its VIN differently (``'100682'`` and ``'00000000000100682'``
    become identical), making the column safe as a dedup key for
    ``mode_then_first_per_group``. Padding only adds characters; a value
    already longer than 17 is not shortened — instead the validation step
    raises, because a VIN over 17 characters is bad data, not something to
    pass through silently.

    Args:
        df: The frame containing the VIN column.
        column: Name of the VIN column to canonicalize.
        ignore_null: Passed to the length check. When ``False`` (default),
            a null VIN is a violation — appropriate for a dedup key. Set
            ``True`` where a null VIN is legitimate (e.g. a fact table).

    Returns:
        The frame with ``column`` left-padded to 17 characters.

    Raises:
        ValueError: If any value is longer than 17 characters, or (unless
            ``ignore_null``) if any value is null.

    Example:
        >>> import polars as pl
        >>> from framesmith.canonicalize import canonicalize_vin
        >>> df = pl.DataFrame({'vin': ['100682', '1HGCM82633A004352']})
        >>> canonicalize_vin(df, 'vin')['vin'].to_list()
        ['00000000000100682', '1HGCM82633A004352']
    """
    # Pad first, then check: a short-but-paddable VIN must reach width 17
    # before the length guard runs, or it would wrongly fail.
    padded: pl.DataFrame = df.with_columns(
        compose_column(column, [pad_left(_VIN_LENGTH)])
    )
    assert_string_length(
        padded,
        column,
        min_length=_VIN_LENGTH,
        max_length=_VIN_LENGTH,
        ignore_null=ignore_null,
    )
    return padded
