# framesmith/recipes.py
"""
Curated recipes: ordered tuples of transforms for common cleaning pipelines.

A recipe is plain data — a ``tuple[ExpressionTransform, ...]`` — that callers
pass to ``compose_column``. Splice and extend with tuple unpacking:
``(*TEXT_NORMALIZE, to_snake_case)``. Recipes compose other recipes the same
way, so shared sub-sequences are defined once and reused.

Naming protocol — the name states the input and the operation, so the tuple
never has to be read to know what a recipe does:

* ``<INPUT>_CANONICALIZE`` — domain-blind representation cleanup that preserves
  meaning: whitespace, case, Unicode form, blank-to-null. Values that mean the
  same become identical; nothing domain-specific is interpreted or removed.
* ``<INPUT>_NORMALIZE`` — domain-aware cleanup that interprets the value (an
  address, a name, a number) and standardizes or removes domain elements. More
  opinionated; may rewrite or drop content.
* ``<INPUT>_TO_<FORM>`` — a conversion whose output form or dtype differs from
  the input: ``TO_ASCII``, ``TO_FLOAT``, ``TO_FRACTION``, ``TO_SNAKE_CASE``,
  ``TO_TITLE``, ``TO_DISPLAY_NAME``. A ``_TO_SNAKE_CASE`` variant applies its
  base recipe's cleaning, then lowercases and underscore-joins into a
  snake_case key.

UPPERCASE naming signals "reusable predefined sequence," distinct from the
lowercase transform functions a recipe contains. Every recipe leaves nulls as
null, and every recipe that touches whitespace also coerces blank or
whitespace-only values to null.
"""

from framesmith.transforms import (
    DEFAULT_PLACEHOLDER_SENTINELS,
    accounting_parens_to_negative,
    cast_to_float64,
    collapse_whitespace,
    extract_email_local_part,
    extract_email_local_part_strict,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    nullify_sentinels,
    percent_to_fraction,
    periods_to_spaces,
    remove_apostrophes,
    remove_credentials,
    remove_periods,
    remove_thousands_separators,
    replace_ampersand_with_and,
    separators_to_space,
    standardize_directionals,
    standardize_initials,
    standardize_street_suffixes,
    standardize_unit_markers,
    strip_name_prefixes,
    strip_name_suffixes,
    strip_whitespace,
    to_lowercase,
    to_snake_case,
    to_titlecase,
    to_uppercase,
    trailing_minus_to_prefix,
    underscores_to_spaces,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'CATEGORY_CANONICALIZE',
    'CATEGORY_CANONICALIZE_TO_SNAKE_CASE',
    'EMAIL_TO_DISPLAY_NAME',
    'EMAIL_TO_TITLE_NAME',
    'NUMERIC_STRING_NORMALIZE',
    'NUMERIC_STRING_TO_FLOAT',
    'PERCENT_STRING_TO_FRACTION',
    'PERSON_NAME_NORMALIZE',
    'PERSON_NAME_NORMALIZE_TO_SNAKE_CASE',
    'SNAKE_CASE_TO_TITLE',
    'STREET_ADDRESS_NORMALIZE',
    'STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE',
    'TEXT_CANONICALIZE',
    'TEXT_NORMALIZE',
    'TEXT_NORMALIZE_TO_SNAKE_CASE',
    'UNICODE_TO_ASCII',
    'UPPERCASE_CODE_NORMALIZE',
    'WHITESPACE_CANONICALIZE',
]


# --- Representation canonicalizers (domain-blind) ---

# Tidy whitespace and coerce blanks to null: drop blank or whitespace-only
# values to null, collapse interior whitespace runs to a single space, and
# strip the ends. The null coercion is deliberate — an empty string and a null
# both mean "missing," and keeping them distinct is a common source of bugs.
WHITESPACE_CANONICALIZE: tuple[ExpressionTransform, ...] = (
    nullify_blank_strings,
    collapse_whitespace,
    strip_whitespace,
)

# NFKC normalization then ASCII compatibility folding. Order matters: NFKC
# decomposes compatibility forms (fullwidth, ligatures) first, then the ASCII
# map substitutes smart quotes, dashes, currency, and trademark symbols. Use
# for canonical ASCII-ish text; use the two transforms individually for one
# half only.
UNICODE_TO_ASCII: tuple[ExpressionTransform, ...] = (
    normalize_unicode_nfkc,
    fold_to_ascii,
)

# Conservative, content-preserving text cleanup: blank-to-null, Unicode-to-
# ASCII, then whitespace tidy. Canonicalizes representation without removing
# any characters — the safe base to build a more opinionated pipeline on.
TEXT_CANONICALIZE: tuple[ExpressionTransform, ...] = (
    nullify_blank_strings,
    *UNICODE_TO_ASCII,
    collapse_whitespace,
    strip_whitespace,
)


# --- Text (domain cleanup) ---

# Full text cleanup: the canonicalization core, then opinionated punctuation
# handling — '&' to 'and', and removal of apostrophes and periods. This removes
# content ('O'Brien' becomes 'OBrien'); use TEXT_CANONICALIZE for the tidy-up
# without the punctuation stripping.
TEXT_NORMALIZE: tuple[ExpressionTransform, ...] = (
    *TEXT_CANONICALIZE,
    replace_ampersand_with_and,
    remove_apostrophes,
    remove_periods,
)

# TEXT_NORMALIZE, then lowercased and underscore-joined into a snake_case key.
TEXT_NORMALIZE_TO_SNAKE_CASE: tuple[ExpressionTransform, ...] = (
    *TEXT_NORMALIZE,
    to_snake_case,
)


# --- Category (label canonicalization) ---

# Canonicalize a categorical label so spelling-variant duplicates merge: tidy
# whitespace and blank-to-null, then lowercase. ' Active ', 'ACTIVE', and
# 'active' all become 'active'. Apply upstream of map_categories or the
# collapse-rare transforms so variants are counted as one category.
CATEGORY_CANONICALIZE: tuple[ExpressionTransform, ...] = (
    *WHITESPACE_CANONICALIZE,
    to_lowercase,
)

# CATEGORY_CANONICALIZE, then underscore-joined into a snake_case key.
CATEGORY_CANONICALIZE_TO_SNAKE_CASE: tuple[ExpressionTransform, ...] = (
    *CATEGORY_CANONICALIZE,
    to_snake_case,
)


# --- Street address (domain standardization) ---

# Standardize a US street address to USPS token forms: tidy whitespace, then
# rewrite directionals ('North' -> 'N'), unit markers ('Apartment' -> 'APT'),
# and street suffixes ('Street' -> 'ST') to their USPS abbreviations. Token
# standardization, not address parsing; the street name's own case is left
# as-is ('123 North Main Street' -> '123 N Main ST'). Use the snake_case
# variant for a case-folded key.
STREET_ADDRESS_NORMALIZE: tuple[ExpressionTransform, ...] = (
    *WHITESPACE_CANONICALIZE,
    standardize_directionals(),
    standardize_unit_markers(),
    standardize_street_suffixes(),
)

# STREET_ADDRESS_NORMALIZE, then lowercased and underscore-joined into a
# snake_case key, e.g. '123_n_main_st'.
STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE: tuple[ExpressionTransform, ...] = (
    *STREET_ADDRESS_NORMALIZE,
    to_snake_case,
)


# --- Person name (domain cleanup) ---

# Strip the noise off a person-name field: tidy whitespace (so the anchored
# affix patterns match), remove trailing comma-separated credentials
# ('MD, PhD'), a leading honorific ('Dr.'), and a trailing generational or
# professional suffix ('Jr', 'Sr', 'III', 'Esq'), standardize single-letter
# initials to 'J. R.' form, then strip the trailing space initial
# standardization can leave. 'Dr. John Smith Jr, PhD' -> 'John Smith'.
PERSON_NAME_NORMALIZE: tuple[ExpressionTransform, ...] = (
    *WHITESPACE_CANONICALIZE,
    remove_credentials(),
    strip_name_prefixes(),
    strip_name_suffixes(),
    standardize_initials,
    strip_whitespace,
)

# PERSON_NAME_NORMALIZE, then lowercased and underscore-joined into a
# snake_case key, e.g. 'john_smith'.
PERSON_NAME_NORMALIZE_TO_SNAKE_CASE: tuple[ExpressionTransform, ...] = (
    *PERSON_NAME_NORMALIZE,
    to_snake_case,
)


# --- Identifier codes (domain normalization) ---

# Canonicalize an identifier code (VIN, license plate, asset tag) to the one
# form a pipeline joins on: NFKC-normalize (fullwidth and other compatibility
# look-alikes fold to ASCII, so a fullwidth-digit VIN normalizes to plain
# ASCII), tidy whitespace and drop blanks to null via
# WHITESPACE_CANONICALIZE, UPPERCASE, then null the '?'
# placeholder that stands in for an unknown code. On a clean code (no interior
# whitespace, no Unicode oddity, not the placeholder) the result is exactly
# UPPER(TRIM(code)) — the collapse and NFKC steps are no-ops there — so the
# output joins byte-for-byte against a SQL UPPER(TRIM(...)) key while still
# repairing case, exotic whitespace, and look-alike Unicode on messy input.
# Blank-to-null runs inside the WHITESPACE_CANONICALIZE splice; ordering it
# ahead of the uppercase step is equivalent since neither creates or clears a
# blank the other would have caught.
UPPERCASE_CODE_NORMALIZE: tuple[ExpressionTransform, ...] = (
    normalize_unicode_nfkc,
    *WHITESPACE_CANONICALIZE,
    to_uppercase,
    nullify_sentinels(DEFAULT_PLACEHOLDER_SENTINELS),
)


# --- Numeric strings ---

# Clean a messy numeric/currency string into a bare numeric string ready to
# cast: Unicode-to-ASCII (handles unicode minus, currency, invisible and
# whitespace-variant characters), then accounting parentheses to a negative
# sign, trailing minus to a leading minus, and thousands-separator removal.
# Returns a STRING — cast it yourself for a non-float dtype.
NUMERIC_STRING_NORMALIZE: tuple[ExpressionTransform, ...] = (
    *UNICODE_TO_ASCII,
    accounting_parens_to_negative,
    trailing_minus_to_prefix,
    remove_thousands_separators,
)

# Clean a numeric string, then cast to Float64. Unparseable values become null
# (no fill — the caller decides).
NUMERIC_STRING_TO_FLOAT: tuple[ExpressionTransform, ...] = (
    *NUMERIC_STRING_NORMALIZE,
    cast_to_float64,
)

# Clean a possibly-messy percent string and convert to a Float64 fraction:
# '50%' -> 0.5, '(12%)' -> -0.12. Reuses the numeric-string cleanup, then
# strips the optional '%' and divides by 100.
PERCENT_STRING_TO_FRACTION: tuple[ExpressionTransform, ...] = (
    *NUMERIC_STRING_NORMALIZE,
    percent_to_fraction,
)


# --- Label conversions ---

# Derive a human-readable display name from an email address: strip surrounding
# whitespace, take the local part (before the first '@'), turn periods into
# spaces, and collapse the resulting whitespace. 'john.doe@example.com' ->
# 'john doe'.
EMAIL_TO_DISPLAY_NAME: tuple[ExpressionTransform, ...] = (
    strip_whitespace,
    extract_email_local_part,
    periods_to_spaces,
    collapse_whitespace,
)

# Derive a Title Case display name from an email, modeled on the BigQuery report
# expression
#   INITCAP(REGEXP_REPLACE(REGEXP_EXTRACT(LOWER(email), '^([^@]+)@'),
#                          '[._-]+', ' '))
# : strict-extract the local part before the first '@' (null for a value with no
# '@' or an empty local part — REGEXP_EXTRACT semantics), replace each
# '.'/'_'/'-' run with a single space (separators_to_space == the '[._-]+' -> ' '
# regex), then title-case. This differs from EMAIL_TO_DISPLAY_NAME, whose lenient
# extractor keeps a non-conforming value and whose periods_to_spaces ignores '_'
# and '-'. The LOWER in the source is redundant here because to_titlecase re-cases
# every word, so it is omitted. A null or non-conforming email yields null.
#
# Title casing is polars `to_titlecase`, which is NOT byte-identical to BigQuery
# INITCAP on two rare, purely cosmetic edge cases (a single letter's case in a
# display name): (1) an intra-word capital's tail is lowercased, so 'mcdonald' ->
# 'Mcdonald' not 'McDonald'; (2) a letter immediately after a digit is
# capitalized, '2pac' -> '2Pac' where INITCAP (treating digits as word
# characters, not delimiters) yields '2pac'. Neither occurs for first.last@
# corporate mailboxes. Do not rely on this recipe for byte-exact INITCAP parity.
EMAIL_TO_TITLE_NAME: tuple[ExpressionTransform, ...] = (
    extract_email_local_part_strict,
    separators_to_space,
    to_titlecase,
)

# Turn a snake_case identifier into a Title Case label: underscores to spaces,
# then title-case each word. Title casing lowercases the tail of every word, so
# it mangles acronyms ('nasa_program' -> 'Nasa Program'); splice
# apply_replacements to fix specific tokens, e.g.
# (*SNAKE_CASE_TO_TITLE, apply_replacements({'Nasa': 'NASA'})).
SNAKE_CASE_TO_TITLE: tuple[ExpressionTransform, ...] = (
    underscores_to_spaces,
    to_titlecase,
)
