# framesmith/_internal/regex_patterns.py
"""
Shared regex pattern strings for text-processing expressions.

Centralizes the regular-expression strings used across framesmith's
column-level normalization modules so each pattern has a single source
of truth. Two consumers (e.g. ``columns/text.py`` and ``columns/names.py``)
that need the same whitespace pattern import the same constant instead
of redefining it.

These are raw ``str`` constants, not ``re.Pattern`` objects: polars
expression methods (``str.contains``, ``str.replace_all``,
``str.replace``, etc.) accept regex pattern strings and compile them
internally with the Rust ``regex`` crate. Passing a precompiled Python
``re.Pattern`` is neither supported nor useful here.

Inline-flag syntax (``(?i)``, ``(?m)``, ``(?s)``) is used for options
that the legacy ``re`` API exposed via the ``flags=`` keyword, since
polars' regex strings do not take a separate flags argument.
"""

__all__: list[str] = [
    'BLANK_OR_WHITESPACE_ONLY_PATTERN',
    'TRAILING_JR_PATTERN',
    'WHITESPACE_RUN_PATTERN',
]


# One or more whitespace characters. Used for collapsing runs of
# whitespace into a single space (or single underscore in snake_case).
WHITESPACE_RUN_PATTERN: str = r'\s+'

# A string that is either empty or composed entirely of whitespace.
# Anchored at both ends to match the whole value, not a substring.
BLANK_OR_WHITESPACE_ONLY_PATTERN: str = r'^\s*$'

# A trailing 'jr' or 'jr.' suffix, optionally preceded by a comma and/or
# whitespace. The inline ``(?i)`` flag makes the match case-insensitive,
# since polars regex strings do not accept a separate ``flags`` argument.
TRAILING_JR_PATTERN: str = r'(?i)(?:[, ]+)?jr\.?$'
