# framesmith/_internal/regex_patterns.py
"""
Shared regex pattern strings for text-processing expressions.

Centralizes the regular-expression strings used across framesmith's
transform modules so each pattern has a single source of truth. Two
consumers (e.g. ``transforms/text.py`` and ``transforms/names.py``)
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
    'PAREN_NEGATIVE_PATTERN',
    'STANDALONE_INITIAL_PATTERN',
    'THOUSANDS_SEPARATOR_PATTERN',
    'TRAILING_JR_PATTERN',
    'TRAILING_MINUS_PATTERN',
    'WHITESPACE_RUN_PATTERN',
    'ZIP_CODE_PATTERN',
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

# A standalone single-letter initial: a word-initial letter immediately
# followed by a period and/or whitespace, captured in group 1. Rust regex
# has no lookahead, so the trailing ``[.\s]+`` is what proves the letter
# stands alone — a letter beginning a longer word is followed by another
# letter and does not match ('J' in 'J. R.' matches; 'S' in 'Smith' does
# not). Used to normalize initials to 'J. R.' form via the replacement
# '$1. '.
STANDALONE_INITIAL_PATTERN: str = r'(?i)\b([a-z])[.\s]+'

# Accounting-style parenthesized negative: ``"(123.45)"``,
# ``"( $1,234.56 )"``. Anchored to the whole value; captures the content
# between the outermost parens in group 1, with optional whitespace
# adjacent to the parens consumed by the pattern (so the captured body
# has no parens-adjacent whitespace).
PAREN_NEGATIVE_PATTERN: str = r'^\s*\(\s*(.+?)\s*\)\s*$'

# Trailing minus (SAP / AS-400 / mainframe convention): ``"1,234.56-"``.
# Anchored. The first character must be non-minus so an already-leading-
# minus value (``"-100"``) is not double-negated. Body captured in
# group 1.
TRAILING_MINUS_PATTERN: str = r'^([^-].+)-$'

# Digit-group separators: a comma or any whitespace character. Used to
# strip thousands separators (and incidentally any surrounding
# whitespace) before casting. As a character class, ``,`` and ``\s``
# are atomic; no escaping needed.
THOUSANDS_SEPARATOR_PATTERN: str = r'[,\s]'

# Trailing 5-digit US ZIP, with an optional ZIP+4 suffix that is matched but
# not captured, tolerating trailing punctuation/whitespace. Capture group 1
# is the five-digit ZIP. The leading \b prevents capturing the trailing five
# digits of a longer number; end-anchoring keeps a mid-string number (e.g. a
# street number) from matching when no real ZIP is present.
ZIP_CODE_PATTERN: str = r'\b(\d{5})(?:-\d{4})?[,.\s]*$'
