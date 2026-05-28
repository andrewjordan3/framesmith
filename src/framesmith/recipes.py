# framesmith/recipes.py
"""
Curated recipes: ordered tuples of transforms for common pipelines.

A recipe is plain data — a ``tuple[ExpressionTransform, ...]`` — that
callers pass to ``compose_column``. Splice and extend with tuple
unpacking: ``(*NORMALIZE_TEXT, to_snake_case)``. Recipes may compose
other recipes the same way (see ``NORMALIZE_TEXT`` below).

UPPERCASE naming signals "reusable predefined sequence," distinct from
the lowercase transform functions they contain.
"""

from framesmith.transforms.text import (
    collapse_whitespace,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    remove_apostrophes,
    remove_periods,
    replace_ampersand_with_and,
    strip_whitespace,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'NORMALIZE_TEXT',
    'UNICODE_TO_ASCII',
]


# NFKC normalization followed by ASCII compatibility folding. Order
# matters: NFKC decomposes compatibility forms before the ASCII map
# runs. Use this when you want canonical ASCII-ish text; use the
# individual transforms when you want only one half.
UNICODE_TO_ASCII: tuple[ExpressionTransform, ...] = (
    normalize_unicode_nfkc,
    fold_to_ascii,
)


# Reproduces the legacy ``normalize_text`` pipeline exactly, in order.
# Splices UNICODE_TO_ASCII so the Unicode-canonicalization order is not
# duplicated here.
NORMALIZE_TEXT: tuple[ExpressionTransform, ...] = (
    nullify_blank_strings,
    *UNICODE_TO_ASCII,
    collapse_whitespace,
    strip_whitespace,
    replace_ampersand_with_and,
    remove_apostrophes,
    remove_periods,
)
