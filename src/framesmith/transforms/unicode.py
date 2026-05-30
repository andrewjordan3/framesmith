# framesmith/transforms/unicode.py
"""Unicode normalization transforms: NFKC normalization and ASCII folding.

``normalize_unicode_nfkc`` applies canonical compatibility normalization;
``fold_to_ascii`` substitutes Unicode compatibility characters for ASCII via
``ASCII_COMPAT_MAP``. They pair via the ``UNICODE_TO_ASCII`` recipe (NFKC
first), but each is useful standalone. Both follow the ``ExpressionTransform``
contract.
"""

import polars as pl

from framesmith._internal import ASCII_COMPAT_MAP

__all__: list[str] = [
    'fold_to_ascii',
    'normalize_unicode_nfkc',
]


def normalize_unicode_nfkc(expr: pl.Expr) -> pl.Expr:
    """Apply NFKC Unicode normalization.

    Canonical compatibility normalization: fullwidth → ASCII, ligature
    decomposition, ``™`` → ``TM``, etc. Usually paired with
    :func:`fold_to_ascii` via the ``UNICODE_TO_ASCII`` recipe, but
    useful standalone when you want canonical Unicode without the
    opinionated ASCII substitution that ``fold_to_ascii`` applies.
    """
    return expr.str.normalize('NFKC')


def fold_to_ascii(expr: pl.Expr) -> pl.Expr:
    """Fold Unicode compatibility characters to ASCII via ``ASCII_COMPAT_MAP``.

    Covers smart quotes, em-dashes, currency symbols, trademark and
    registered symbols, non-standard whitespace, and more. Typically
    preceded by :func:`normalize_unicode_nfkc`: NFKC decomposes many
    compatibility forms before this map runs, which is exactly what the
    ``UNICODE_TO_ASCII`` recipe does. Used standalone when you want the
    ASCII substitution without NFKC's canonicalization.
    """
    return expr.str.replace_many(ASCII_COMPAT_MAP)
