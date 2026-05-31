# framesmith/transforms/pad.py
"""Fixed-width padding transforms.

``pad_left`` left-pads a string to a fixed width with a fill character — the
fix for identifier codes that lost leading zeros in an integer round-trip
(``'100682'`` → ``'00000000000100682'``). It only ever pads; a value already
at or beyond the width passes through unchanged (never truncated).
"""

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = ['pad_left']


def pad_left(width: int, *, fill: str = '0') -> ExpressionTransform:
    """Build a transform left-padding strings to ``width`` with ``fill``.

    Pads on the left so a value shorter than ``width`` reaches it:
    ``'100682'`` with ``width=17`` becomes ``'00000000000100682'``. A
    value already at or longer than ``width`` is returned unchanged —
    this transform never truncates. Nulls pass through as null.

    The default fill ``'0'`` targets the common case: restoring leading
    zeros on fixed-width numeric identifier codes (VINs, account numbers,
    zips) that were stored as integers somewhere upstream.

    Args:
        width: Target minimum length. Must be >= 1.
        fill: Single character used for padding. Must be exactly one
            character. Defaults to ``'0'``.

    Returns:
        An ``ExpressionTransform``. Applied via ``compose_column``.

    Raises:
        ValueError: If ``width`` < 1, or if ``fill`` is not exactly one
            character.

    Example:
        >>> import polars as pl
        >>> from framesmith import compose_column
        >>> from framesmith.transforms import pad_left
        >>> df = pl.DataFrame({'code': ['100682', '12']})
        >>> df.with_columns(
        ...     compose_column('code', [pad_left(8)])
        ... )['code'].to_list()
        ['00100682', '00000012']
    """
    if width < 1:
        raise ValueError(f'width must be >= 1; got {width}')
    if len(fill) != 1:
        raise ValueError(f'fill must be exactly one character; got {fill!r}')

    def _pad_left(expr: pl.Expr) -> pl.Expr:
        return expr.str.pad_start(width, fill)

    return _pad_left
