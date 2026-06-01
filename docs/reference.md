# framesmith reference

A complete catalog of framesmith's public API: every recipe, transform, filter,
and helper, with what it takes, what it does to your data, and a short example.
This is a usage reference — for how each tool *works internally*, read the
source and its docstrings.

framesmith is built around two usage patterns.

**Column transforms and recipes** are applied through `compose_column` inside
`df.with_columns(...)`. A *recipe* (an ordered bundle of transforms) is passed
directly; a one-off list of transforms is passed in a list:

```python
import polars as pl
from framesmith import compose_column, TEXT_NORMALIZE
from framesmith.transforms import strip_whitespace, to_lowercase

# A recipe — passed directly.
df = df.with_columns(compose_column('company', TEXT_NORMALIZE))

# An ad-hoc list of transforms — passed in a list.
df = df.with_columns(compose_column('city', [strip_whitespace, to_lowercase]))
```

A *factory* transform (e.g. `pad_left`, `map_categories`,
`standardize_directionals`) is **called** to produce a configured transform,
which is then placed in the list like any other:

```python
from framesmith.transforms import pad_left
df = df.with_columns(compose_column('zip', [pad_left(5)]))
```

**Eager frame functions** (in `framesmith.group`, `.validate`, `.combine`,
`.schema`, `.canonicalize`, and the filters) are called on a `DataFrame`, or
return an expression you apply with `with_columns`, `filter`, or
`group_by(...).agg(...)`:

```python
from framesmith.validate import assert_string_length
from framesmith.combine import combine_columns

assert_string_length(df, 'vin', min_length=17, max_length=17)  # raises on bad data
df = df.with_columns(combine_columns(['first', 'last'], 'full_name'))
```

Import paths mirror this document: the top level (`from framesmith import ...`)
exposes `compose_column`, `ExpressionTransform`, and the 16 recipes; the 56
transforms live on `framesmith.transforms`; each specialized family lives on its
own subpackage.

---

## Table of contents

- [Composition](#composition)
- [Recipes](#recipes)
  - [Representation](#representation)
  - [Text](#text)
  - [Category](#category)
  - [Street address](#street-address)
  - [Person name](#person-name)
  - [Numeric and percent](#numeric-and-percent)
  - [Unicode and label conversions](#unicode-and-label-conversions)
- [Transforms](#transforms)
  - [Whitespace](#whitespace)
  - [Case](#case)
  - [Unicode](#unicode)
  - [Substitution](#substitution)
  - [Numeric](#numeric)
  - [Names](#names)
  - [Addresses](#addresses)
  - [Dates](#dates)
  - [Outliers](#outliers)
  - [Bounds](#bounds)
  - [Categorical](#categorical)
  - [Missing](#missing)
  - [Split](#split)
  - [Padding](#padding)
- [Filters](#filters)
- [Combine](#combine)
- [Group](#group)
- [Validate](#validate)
- [Schema](#schema)
- [Canonicalize](#canonicalize)
- [Defaults and types](#defaults-and-types)

---

## Composition

#### `compose_column(source_column_name, expression_transforms, output_column_name=None)`

Applies a sequence of transforms (or a recipe) to one column and returns the
resulting expression, ready for `df.with_columns(...)`. By default the result
overwrites the source column; pass `output_column_name` to write to a new
column instead. Raises `ValueError` if the transform sequence is empty.

```python
df.with_columns(compose_column('name', [strip_whitespace, to_lowercase]))
# '  Alice  '  ->  'alice'
df.with_columns(compose_column('d', [flag_dates_outside_range(upper=today)],
                               output_column_name='d_is_future'))
```

#### `ExpressionTransform`

The type of a single column transform: a function from a polars expression to a
polars expression. Use it to annotate your own transforms and recipes so they
slot into `compose_column` alongside the built-ins. See
[Defaults and types](#defaults-and-types).

```python
def shout(expr: pl.Expr) -> pl.Expr:
    return expr.str.to_uppercase()

my_transform: ExpressionTransform = shout
```

---

## Recipes

Recipes are ready-made bundles of transforms for common cleaning jobs. Pass one
directly to `compose_column` (no list needed). Every recipe leaves nulls as
null, and every recipe that touches whitespace also turns blank or
whitespace-only values into null. The `*_TO_SNAKE_CASE` variants apply their
base recipe, then fold the result into a lowercase, underscore-joined key.

### Representation

#### `WHITESPACE_CANONICALIZE`

Tidies whitespace: trims the ends and collapses internal runs to a single
space. Blank or whitespace-only values become null.

```python
df.with_columns(compose_column('note', WHITESPACE_CANONICALIZE))
# '  a   b  '  ->  'a b'   ;   '   '  ->  None
```

#### `TEXT_CANONICALIZE`

Conservative text cleanup: converts Unicode to its plain-ASCII equivalent and
tidies whitespace, without removing any characters. Blank values become null.
The safe base to build a more opinionated pipeline on.

```python
df.with_columns(compose_column('desc', TEXT_CANONICALIZE))
# '  “A—B”  '  ->  '"A-B"'   ;   "O'Brien"  ->  "O'Brien"  (apostrophe kept)
```

### Text

#### `TEXT_NORMALIZE`

Everything `TEXT_CANONICALIZE` does, plus opinionated punctuation cleanup:
expands `&` to `and` and removes apostrophes and periods. This removes content,
so use `TEXT_CANONICALIZE` if you need the tidy-up without the punctuation
stripping.

```python
df.with_columns(compose_column('company', TEXT_NORMALIZE))
# 'Sales & Service'  ->  'Sales and Service'   ;   "O'Brien"  ->  'OBrien'
```

#### `TEXT_NORMALIZE_TO_SNAKE_CASE`

`TEXT_NORMALIZE`, then folded into a lowercase snake_case key.

```python
df.with_columns(compose_column('company', TEXT_NORMALIZE_TO_SNAKE_CASE))
# 'Sales & Service'  ->  'sales_and_service'
```

### Category

#### `CATEGORY_CANONICALIZE`

Canonicalizes a categorical label so spelling-variant duplicates merge: tidies
whitespace, nulls blanks, and lowercases. Apply it upstream of `map_categories`
or the collapse transforms so variants are counted as one category.

```python
df.with_columns(compose_column('status', CATEGORY_CANONICALIZE))
# ' Active ', 'ACTIVE', 'active'  ->  'active'   ;   'In  Progress'  ->  'in progress'
```

#### `CATEGORY_CANONICALIZE_TO_SNAKE_CASE`

`CATEGORY_CANONICALIZE`, then folded into a snake_case key.

```python
df.with_columns(compose_column('status', CATEGORY_CANONICALIZE_TO_SNAKE_CASE))
# ' In Progress '  ->  'in_progress'
```

### Street address

#### `STREET_ADDRESS_NORMALIZE`

Standardizes a US street address to USPS token forms: directionals (`North` →
`N`), unit markers (`Apartment` → `APT`), and street suffixes (`Street` → `ST`).
This standardizes the address vocabulary only; the street name's own casing is
left as-is.

```python
df.with_columns(compose_column('address', STREET_ADDRESS_NORMALIZE))
# '123 North Main Street'         ->  '123 N Main ST'
# '789 SE Grand Blvd Apartment 4' ->  '789 SE Grand BLVD APT 4'
```

#### `STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE`

`STREET_ADDRESS_NORMALIZE`, then folded into a lowercase snake_case key.

```python
df.with_columns(compose_column('address', STREET_ADDRESS_NORMALIZE_TO_SNAKE_CASE))
# '123 North Main Street'  ->  '123_n_main_st'
```

### Person name

#### `PERSON_NAME_NORMALIZE`

Cleans a person-name column: removes honorifics (Dr., Mrs.), trailing
credentials (MD, PhD), and generational suffixes (Jr, III), standardizes
single-letter initials, and tidies whitespace. Blank values become null.

```python
df.with_columns(compose_column('name', PERSON_NAME_NORMALIZE))
# 'Dr. John  Smith Jr, PhD'  ->  'John Smith'   ;   'Smith, J.R.'  ->  'Smith, J. R.'
```

#### `PERSON_NAME_NORMALIZE_TO_SNAKE_CASE`

`PERSON_NAME_NORMALIZE`, then folded into a lowercase snake_case key.

```python
df.with_columns(compose_column('name', PERSON_NAME_NORMALIZE_TO_SNAKE_CASE))
# 'Dr. John Smith Jr'  ->  'john_smith'
```

### Numeric and percent

#### `NUMERIC_STRING_NORMALIZE`

Cleans a messy numeric or currency string into a bare numeric string ready to
cast — handling currency symbols, unicode minus signs, accounting parentheses
(which become a negative sign), a trailing minus, and thousands separators. The
result is still a **string**; cast it yourself if you want a non-float dtype.

```python
df.with_columns(compose_column('amount', NUMERIC_STRING_NORMALIZE))
# '($1,234.56)'  ->  '-1234.56'
```

#### `NUMERIC_STRING_TO_FLOAT`

Cleans a numeric string and casts it to `Float64`. Values that still can't be
parsed become null (no fill — that's your decision).

```python
df.with_columns(compose_column('amount', NUMERIC_STRING_TO_FLOAT))
# '($1,234.56)'  ->  -1234.56   ;   'abc'  ->  None
```

#### `PERCENT_STRING_TO_FRACTION`

Cleans a possibly-messy percent string and converts it to a `Float64` fraction.
Unparseable values become null.

```python
df.with_columns(compose_column('rate', PERCENT_STRING_TO_FRACTION))
# '50%'  ->  0.5   ;   '(12%)'  ->  -0.12
```

### Unicode and label conversions

#### `UNICODE_TO_ASCII`

Converts Unicode text to a plain-ASCII equivalent: smart quotes, em/en dashes,
currency and trademark symbols, fullwidth characters, and ligatures all fold to
their ASCII forms.

```python
df.with_columns(compose_column('text', UNICODE_TO_ASCII))
# 'Ａ“x”'  ->  'A"x"'
```

#### `EMAIL_TO_DISPLAY_NAME`

Derives a human-readable display name from an email address: takes the part
before the `@` and turns dots into spaces.

```python
df.with_columns(compose_column('email', EMAIL_TO_DISPLAY_NAME))
# 'john.doe@example.com'  ->  'john doe'
```

#### `SNAKE_CASE_TO_TITLE`

Turns a snake_case identifier into a Title Case label. (Title casing lowercases
the tail of every word, so acronyms come out mangled; splice `apply_replacements`
to fix specific tokens.)

```python
df.with_columns(compose_column('field', SNAKE_CASE_TO_TITLE))
# 'john_smith'  ->  'John Smith'
```

---

## Transforms

The 56 atomic transforms on `framesmith.transforms`. A **plain** transform is
passed directly in the transform list. A **factory** transform is called first
to configure it, then the result goes in the list. Unless noted, every transform
leaves nulls as null.

### Whitespace

#### `collapse_whitespace`

Collapses runs of whitespace to a single space. Does not trim the ends. Passed
directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [collapse_whitespace]))
# 'a   b'  ->  'a b'
```

#### `strip_whitespace`

Removes leading and trailing whitespace. Does not touch interior runs. Passed
directly in a transform list (no call).

```python
df.with_columns(compose_column('city', [strip_whitespace]))
# '  Chicago '  ->  'Chicago'
```

#### `nullify_blank_strings`

Turns blank or whitespace-only values into null; other values are unchanged.
Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [nullify_blank_strings]))
# '   '  ->  None   ;   'x'  ->  'x'
```

#### `replace_whitespace_with(separator)`

Replaces each run of whitespace with `separator` (e.g. for kebab- or
snake-style keys). Does not trim the ends, so a leading/trailing run also
becomes a separator.

```python
df.with_columns(compose_column('s', [replace_whitespace_with('-')]))
# 'hello world'  ->  'hello-world'
```

### Case

#### `to_lowercase`

Lowercases every character. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [to_lowercase]))
# 'Hello World'  ->  'hello world'
```

#### `to_titlecase`

Title-cases each word (first letter up, rest down). Mangles acronyms
(`'nasa'` → `'Nasa'`); fix specific tokens afterward with `apply_replacements`.
Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [to_titlecase]))
# 'john smith'  ->  'John Smith'
```

#### `to_snake_case`

Lowercases and replaces whitespace runs with underscores. Passed directly in a
transform list (no call).

```python
df.with_columns(compose_column('s', [to_snake_case]))
# 'Hello World'  ->  'hello_world'
```

### Unicode

#### `normalize_unicode_nfkc`

Applies canonical Unicode normalization — fullwidth characters, ligatures, and
symbols like `™` resolve to their canonical forms. Passed directly in a
transform list (no call).

```python
df.with_columns(compose_column('s', [normalize_unicode_nfkc]))
# 'ﬁnish'  ->  'finish'   ;   '１２３'  ->  '123'
```

#### `fold_to_ascii`

Substitutes Unicode compatibility characters for ASCII: smart quotes, dashes,
currency, and trademark symbols. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [fold_to_ascii]))
# '“quoted”'  ->  '"quoted"'   ;   'A—B'  ->  'A-B'
```

### Substitution

#### `replace_ampersand_with_and`

Replaces literal `&` with `and`. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [replace_ampersand_with_and]))
# 'Sales & Service'  ->  'Sales and Service'
```

#### `remove_apostrophes`

Removes every apostrophe. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [remove_apostrophes]))
# "O'Brien"  ->  'OBrien'
```

#### `remove_periods`

Removes every period. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [remove_periods]))
# 'St.'  ->  'St'
```

#### `periods_to_spaces`

Replaces each period with a single space. Passed directly in a transform list
(no call).

```python
df.with_columns(compose_column('s', [periods_to_spaces]))
# 'U.S.A'  ->  'U S A'
```

#### `underscores_to_spaces`

Replaces each underscore with a single space. Passed directly in a transform
list (no call).

```python
df.with_columns(compose_column('s', [underscores_to_spaces]))
# 'john_smith'  ->  'john smith'
```

#### `apply_replacements(replacements)`

Replaces any matching substring with its mapped value, useful for fixing
acronyms or expanding abbreviations. Matching is literal substring (not
word-boundary), so choose keys that won't collide with text you mean to keep.
Raises `ValueError` if the map is empty.

```python
df.with_columns(compose_column('s', [apply_replacements({'Nasa': 'NASA'})]))
# 'Nasa program'  ->  'NASA program'
```

### Numeric

#### `accounting_parens_to_negative`

Rewrites an accounting-style parenthesized value as a leading-minus number.
Currency and separators are left for other transforms. Passed directly in a
transform list (no call).

```python
df.with_columns(compose_column('s', [accounting_parens_to_negative]))
# '(123.45)'  ->  '-123.45'
```

#### `trailing_minus_to_prefix`

Moves a trailing minus sign to the front. A value that already starts with a
minus is left alone (never double-negated). Passed directly in a transform list
(no call).

```python
df.with_columns(compose_column('s', [trailing_minus_to_prefix]))
# '1,234.56-'  ->  '-1,234.56'
```

#### `remove_thousands_separators`

Removes digit-group separators (commas and whitespace), preserving the decimal
point and minus sign. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [remove_thousands_separators]))
# '1,234.56'  ->  '1234.56'
```

#### `cast_to_float64`

Casts to `Float64`; anything that isn't a clean numeric string becomes null
rather than raising. Does not fill nulls. Passed directly in a transform list
(no call).

```python
df.with_columns(compose_column('s', [cast_to_float64]))
# '123.45'  ->  123.45   ;   'abc'  ->  None
```

#### `percent_to_fraction`

Parses a percent string into a `Float64` fraction; values without a `%` are cast
as-is. Assumes pre-cleaned input. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('s', [percent_to_fraction]))
# '12%'  ->  0.12   ;   '12'  ->  12.0
```

### Names

#### `remove_jr_suffix`

Removes a trailing `Jr`/`Jr.` suffix (with optional preceding comma). Passed
directly in a transform list (no call).

```python
df.with_columns(compose_column('name', [remove_jr_suffix]))
# 'John Smith Jr'  ->  'John Smith'
```

#### `strip_name_prefixes(prefixes=DEFAULT_NAME_PREFIXES)`

Removes a leading honorific (Mr, Mrs, Dr, Prof, …). Pass `prefixes` to override
the default set.

```python
df.with_columns(compose_column('name', [strip_name_prefixes()]))
# 'Dr. John Smith'  ->  'John Smith'
```

#### `strip_name_suffixes(suffixes=DEFAULT_NAME_SUFFIXES)`

Removes a trailing generational/professional suffix (Jr, Sr, II–IV, Esq). The
suffix must be a separate trailing token, so words like `'Hawaii'` are safe.

```python
df.with_columns(compose_column('name', [strip_name_suffixes()]))
# 'Jane Doe III'  ->  'Jane Doe'
```

#### `remove_credentials(credentials=DEFAULT_CREDENTIALS)`

Removes trailing comma-separated professional credentials (MD, PhD, FACS, …). A
comma is required before each credential, so a credential written without one
(`'John Smith MD'`) is left unchanged.

```python
df.with_columns(compose_column('name', [remove_credentials()]))
# 'Jane Smith, MD, PhD'  ->  'Jane Smith'
```

#### `standardize_initials`

Normalizes single-letter initials to a consistent `'J. R.'` form (letter,
period, space). Multi-letter tokens and glued pairs like `'JR'` are left alone.
An initial that ends the string leaves a trailing space — compose
`strip_whitespace` to tidy. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('name', [standardize_initials]))
# 'J R Smith'  ->  'J. R. Smith'
```

#### `extract_email_local_part`

Returns everything before the first `@`; a value with no `@` passes through
unchanged. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('email', [extract_email_local_part]))
# 'john.doe@example.com'  ->  'john.doe'
```

### Addresses

#### `standardize_directionals(directional_map=DEFAULT_DIRECTIONAL_MAP)`

Rewrites directional words to their uppercase USPS abbreviation (`North` → `N`,
`Northeast` → `NE`), matching whole words in any case. Pass `directional_map`
to customize.

```python
df.with_columns(compose_column('address', [standardize_directionals()]))
# '123 North Main'  ->  '123 N Main'
```

#### `standardize_unit_markers(unit_marker_map=DEFAULT_UNIT_MARKER_MAP)`

Rewrites secondary unit designators to their uppercase USPS form (`Apartment` →
`APT`, `Suite` → `STE`). Pass `unit_marker_map` to customize.

```python
df.with_columns(compose_column('address', [standardize_unit_markers()]))
# 'Apartment 4'  ->  'APT 4'
```

#### `standardize_street_suffixes(street_suffix_map=DEFAULT_STREET_SUFFIX_MAP)`

Rewrites street-type words to their uppercase USPS abbreviation (`Street` →
`ST`, `Avenue` → `AVE`), matching whole words so they don't fire inside words
like `'Broadway'`. Pass `street_suffix_map` to override the default subset.

```python
df.with_columns(compose_column('address', [standardize_street_suffixes()]))
# '123 Main Street'  ->  '123 Main ST'
```

#### `standardize_state`

Canonicalizes a whole-value US state to its two-letter postal code. Recognizes
full names, codes, and common abbreviations; unrecognized values pass through
unchanged. Operates on a state column, not a location string. Passed directly in
a transform list (no call).

```python
df.with_columns(compose_column('state', [standardize_state]))
# 'Calif.'  ->  'CA'   ;   'illinois'  ->  'IL'
```

#### `standardize_state_name`

Like `standardize_state`, but resolves to the canonical lowercase full name
instead of the code. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('state', [standardize_state_name]))
# 'IL'  ->  'illinois'   ;   'Calif.'  ->  'california'
```

#### `strip_trailing_state`

Removes a trailing two-letter US state code (and its separator) from a location
string. Only codes are stripped, not full names; `'Washington, DC'` is
preserved. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('city', [strip_trailing_state]))
# 'Chicago, IL'  ->  'Chicago'
```

#### `extract_zip_code`

Extracts the trailing 5-digit US ZIP from a string, dropping any ZIP+4 suffix.
Returns null when there's no trailing ZIP, including an unrelated 5-digit number
like a street number. Output is a string (leading zeros preserved), so don't
cast it to an integer. Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('addr', [extract_zip_code]))
# 'Springfield, IL 62704'  ->  '62704'   ;   '12345 Main St'  ->  None
```

### Dates

#### `normalize_epoch_timestamps(time_unit)`

Parses a Unix-epoch numeric column into a naive datetime. The unit (`'s'`,
`'ms'`, or `'us'`) is required — the same integer means different instants at
different resolutions.

```python
df.with_columns(compose_column('t', [normalize_epoch_timestamps('s')]))
# 1672531200  ->  datetime(2023, 1, 1, 0, 0)
```

#### `normalize_excel_serial_dates`

Converts an Excel serial-date number into a naive datetime (whole part = days,
fractional part = time of day). Passed directly in a transform list (no call).

```python
df.with_columns(compose_column('t', [normalize_excel_serial_dates]))
# 44927  ->  datetime(2023, 1, 1)   ;   44927.5  ->  datetime(2023, 1, 1, 12, 0)
```

#### `flag_dates_outside_range(lower=None, upper=None)`

Produces a boolean column: true where a date is outside the inclusive
`[lower, upper]` range, false within it, null where the input is null. At least
one bound is required. Raises `ValueError` if both are `None`.

```python
df.with_columns(compose_column(
    'd', [flag_dates_outside_range(lower=date(2000, 1, 1), upper=date(2026, 1, 1))],
    output_column_name='d_out_of_range'))
# date(1990, 1, 1)  ->  True
```

### Outliers

Each outlier transform produces a boolean column (true = outlier); a null input
yields a null flag, and a constant column flags nothing.

#### `flag_zscore_outliers(threshold=3.0)`

Flags values more than `threshold` standard deviations from the column mean.
Most affected by the very extremes it detects; use MAD or IQR for heavy-tailed
data. Raises `ValueError` if `threshold` is not positive.

```python
df.with_columns(compose_column('x', [flag_zscore_outliers()],
                               output_column_name='x_outlier'))
```

#### `flag_iqr_outliers(multiplier=1.5)`

Flags values that fall unusually far below the first quartile or above the third
(the Tukey IQR fences), with `multiplier` scaling how far is "too far". Default
1.5 (Tukey's convention). Raises `ValueError` if `multiplier` is not positive.

```python
df.with_columns(compose_column('x', [flag_iqr_outliers()],
                               output_column_name='x_outlier'))
```

#### `flag_mad_outliers(threshold=3.5)`

Flags values far from the median using a robust median-based spread — the most
robust of the three for heavy-tailed data, where a few extremes won't mask
themselves. Raises `ValueError` if `threshold` is not positive.

```python
df.with_columns(compose_column('x', [flag_mad_outliers()],
                               output_column_name='x_outlier'))
```

### Bounds

#### `clip_numeric(lower=None, upper=None)`

Clamps values to a hard `[lower, upper]` range; values past a bound become the
bound. At least one bound is required; omit one for a one-sided clamp. Nulls and
dtype are preserved. Raises `ValueError` if both bounds are `None` or `lower`
exceeds `upper`.

```python
df.with_columns(compose_column('x', [clip_numeric(lower=0.0, upper=10.0)]))
# 50.0  ->  10.0   ;   5.0  ->  5.0
```

#### `winsorize_numeric(lower_quantile=0.05, upper_quantile=0.95)`

Caps extremes at data-driven percentiles: values past the lower/upper quantile
become that quantile. Defaults to the outer 5% of each tail. Nulls and dtype are
preserved. Raises `ValueError` unless `0 ≤ lower_quantile < upper_quantile ≤ 1`.

```python
df.with_columns(compose_column('x', [winsorize_numeric()]))
# a value above the 95th percentile  ->  the 95th-percentile value
```

### Categorical

#### `map_categories(category_map)`

Replaces each whole cell value that exactly matches a key with its mapped value;
everything else passes through. Matching is exact and case-sensitive. Keys and
values may be any hashable type, so a code→label map works; the output column's
dtype follows the map's values. Raises `ValueError` if the map is empty.

```python
df.with_columns(compose_column('flag', [map_categories({1: 'Yes', 2: 'No'})]))
# 1  ->  'Yes'   ;   2  ->  'No'   ;   3  ->  '3'  (unmapped, cast to output dtype)
```

#### `collapse_rare_by_count(min_count, replacement='other')`

Replaces values that occur fewer than `min_count` times with `replacement`;
frequent values are kept. Nulls are never counted or replaced. Raises
`ValueError` if `min_count < 1`.

```python
df.with_columns(compose_column('c', [collapse_rare_by_count(2)]))
# ['a','a','b','b','rare']  ->  ['a','a','b','b','other']
```

#### `collapse_keep_top_n(n, replacement='other')`

Keeps the `n` most frequent value tiers and replaces the rest with
`replacement`. Values tied at the same frequency share a tier and are kept or
dropped together. Raises `ValueError` if `n < 1`.

```python
df.with_columns(compose_column('c', [collapse_keep_top_n(2)]))
# ['a','a','a','b','b','c']  ->  ['a','a','a','b','b','other']
```

### Missing

#### `nullify_sentinels(sentinels, *, case_insensitive=True)`

Turns cells whose trimmed value equals a missing-data placeholder (`'N/A'`,
`'NULL'`, …) into null; non-matching cells pass through unchanged. Pass your own
set of `sentinels` (or the built-in `DEFAULT_MISSING_SENTINELS`). This is opt-in
by design — it belongs in no default recipe. Raises `ValueError` if `sentinels`
is empty.

```python
df.with_columns(compose_column('x', [nullify_sentinels(DEFAULT_MISSING_SENTINELS)]))
# 'N/A'  ->  None   ;   'null'  ->  None   ;   'Chicago'  ->  'Chicago'
```

### Split

#### `split_on_delimiters(delimiters=DEFAULT_SPLIT_DELIMITERS, *, dedup_delimiters=False)`

Splits each string into a `List(String)` on any of the given single-character
delimiters. Each token is trimmed; empty or whitespace-only tokens become null,
so positions are preserved. By default consecutive delimiters are kept distinct;
set `dedup_delimiters=True` to collapse adjacent runs. Raises `ValueError` if
`delimiters` is empty or any delimiter is more than one character.

```python
df.with_columns(compose_column('tags', [split_on_delimiters()]))
# 'a,, b ,'  ->  ['a', None, 'b', None]
```

### Padding

#### `pad_left(width, *, fill='0')`

Left-pads each value to `width` characters with `fill`. Never shortens a value
already at or beyond `width`. Raises `ValueError` if `width < 1` or `fill` is not
a single character.

```python
df.with_columns(compose_column('code', [pad_left(8)]))
# '100682'  ->  '00100682'
```

---

## Filters

Filters return a boolean expression you apply with `df.filter(...)`. Both judge
completeness from the data (the column's max date), not today's calendar date.

#### `within_complete_period(date_column, period, threshold_days)`

Keeps rows in "complete" trailing time periods and drops a trailing period that
ends more than `threshold_days` before the data's last date. `period` is one of
`'1d'`, `'1w'`, `'1mo'`, `'1q'`, `'1y'`.

```python
df.filter(within_complete_period('date', period='1mo', threshold_days=5))
# drops the trailing month if it ends >5 days after the latest date in the data
```

#### `within_complete_month(date_column, threshold_days=5)`

Convenience wrapper for `within_complete_period(..., period='1mo')`, with a
five-day default threshold.

```python
df.filter(within_complete_month('date'))
```

---

## Combine

Combine functions build an expression you apply with
`df.with_columns(...)`. Each requires an explicit `output_column_name`. The
column doesn't need to exist at build time — a missing column raises when the
expression is applied.

#### `combine_columns(column_names, output_column_name, separator=' ')`

Concatenates several columns row-wise with `separator`, skipping nulls so a
missing value leaves no stray separator. Non-string columns are stringified.
Raises `ValueError` if `column_names` is empty.

```python
df.with_columns(combine_columns(['first', 'last'], 'full_name'))
# first='John', last='Doe'  ->  'John Doe'   ;   first=None, last='Smith'  ->  'Smith'
```

#### `coalesce_blank_columns(column_names, output_column_name)`

Takes the first non-blank value across the named string columns (in priority
order), where "blank" means null or empty after trimming. A kept value is
returned raw (surrounding whitespace preserved); if every column is blank the
result is null. Raises `ValueError` if `column_names` is empty.

```python
df.with_columns(coalesce_blank_columns(['preferred', 'legal'], 'name'))
# preferred='  ', legal='Z'  ->  'Z'
```

#### `hash_key(column_names, output_column_name)`

Builds a deterministic `UInt64` surrogate key from one or more columns, so each
distinct combination of values maps to a stable key suitable for joining frames
within a run. Keys are reproducible within a single polars version only — don't
persist them across upgrades. Being 64-bit, distinct inputs can in rare cases
collide; see the docstring for the collision-rate guidance and the check.
Raises `ValueError` if `column_names` is empty.

```python
df.with_columns(hash_key(['region', 'branch'], 'k'))
# each distinct (region, branch) pair  ->  a stable UInt64 key
```

---

## Group

Group helpers come in two shapes. The resolvers return an expression for
`df.group_by(keys).agg(...)` and collapse each group to one row. The fillers
return an expression for `df.with_columns(...)` and fill nulls in place,
preserving every row.

#### `first_non_null_per_group(key_columns)`

For `group_by(key_columns).agg(...)`: resolves each non-key field to the first
non-null value in its group, so a group split across rows yields one
fully-populated row. Raises `ValueError` if `key_columns` is empty.

```python
df.group_by(['id']).agg(first_non_null_per_group(['id']))
# id=1 rows {name:'Alice', city:None} + {name:None, city:'NYC'}  ->  {name:'Alice', city:'NYC'}
```

#### `mode_then_first_per_group(key_columns)`

For `group_by(key_columns).agg(...)`: resolves each non-key field to the most
common non-null value in its group (ties broken by first appearance). It always
selects a value that appears in the data, so it works on string columns too.
Raises `ValueError` if `key_columns` is empty.

```python
df.group_by(['id']).agg(mode_then_first_per_group(['id']))
# id=1 names ['Al','Al','Bo']  ->  'Al'
```

#### `fill_null_numeric_by_group(value_column, group_columns, *, strategy='auto', skew_threshold=1.0)`

For `df.with_columns(...)`: fills a numeric column's nulls with a per-group
center. `strategy='auto'` picks mean vs median from the column's overall skew;
`'mean'`/`'median'` force it. A group with no non-null value keeps its nulls.
Output is `Float64`; a non-numeric column raises. Raises `ValueError` for empty
`group_columns`, an unknown strategy, or a non-positive `skew_threshold`.

```python
df.with_columns(fill_null_numeric_by_group('amount', ['region']))
# region=1 amounts [10.0, None]  ->  [10.0, 10.0]
```

#### `fill_null_string_by_group(value_column, group_columns)`

For `df.with_columns(...)`: fills a column's nulls with the per-group mode (ties
broken by first appearance). A group with no non-null value keeps its nulls.
Unlike the numeric filler, it accepts numeric columns too. Raises `ValueError`
if `group_columns` is empty.

```python
df.with_columns(fill_null_string_by_group('c', ['g']))
# g=1 values ['a','a',None]  ->  ['a','a','a']
```

---

## Validate

Guards inspect a column and either return `None` (data is fine) or raise
`ValueError` describing the violation. Call them to fail a pipeline early on bad
data.

#### `assert_no_nulls(df, column)`

Raises `ValueError` if `column` contains any null; returns nothing on success.

```python
assert_no_nulls(df, 'id')  # passes silently, or raises with the null count
```

#### `assert_string_length(df, column, *, min_length=None, max_length=None, ignore_null=False)`

Raises `ValueError` if any value's character length falls outside the bounds;
returns nothing on success. At least one bound is required. By default a null
counts as a violation; set `ignore_null=True` to skip nulls.

```python
assert_string_length(df, 'vin', min_length=17, max_length=17)
# passes silently, or raises listing the offending values
```

---

## Schema

#### `normalize_column_names(separator='_')`

Builds an expression that renames every column: trims, lowercases, and replaces
internal whitespace runs with `separator`. Apply with
`df.select(normalize_column_names())` (not `with_columns`, which would keep the
originals). If two labels normalize to the same name, applying it raises
`polars.exceptions.DuplicateError`.

```python
df.select(normalize_column_names()).columns
# ['  First Name ', 'LAST  NAME']  ->  ['first_name', 'last_name']
```

---

## Canonicalize

#### `canonicalize_vin(df, column, *, ignore_null=False)`

Left-pads a VIN column to 17 characters (restoring leading zeros lost in an
integer round-trip), then validates that every value is exactly 17 characters —
so the column is safe as a join/dedup key. A value longer than 17 raises rather
than being silently shortened. By default a null VIN is a violation; pass
`ignore_null=True` where nulls are legitimate. Returns the updated frame.

```python
canonicalize_vin(df, 'vin')
# '100682'  ->  '00000000000100682'   ;   '1HGCM82633A004352' kept as-is
```

---

## Defaults and types

### Default constants

These are the built-in default values behind the factory transforms. Pass your
own to a factory to override what it uses.

#### `DEFAULT_CREDENTIALS`

The 34 professional-credential tokens (PhD, MD, FACS, …) that
`remove_credentials` strips by default.

#### `DEFAULT_DIRECTIONAL_MAP`

The 8 USPS directional abbreviations (`N`, `S`, `NE`, …) and their spellings,
used by `standardize_directionals`.

#### `DEFAULT_MISSING_SENTINELS`

A frozenset of common missing-value strings (`'N/A'`, `''`, `'NAN'`, `'NONE'`,
`'NULL'`, `'NA'`). Pass it (or your own set) to `nullify_sentinels` — it is
deliberately not applied by any recipe automatically.

#### `DEFAULT_NAME_PREFIXES`

The 12 honorific tokens (mr, mrs, dr, prof, …) that `strip_name_prefixes`
removes by default.

#### `DEFAULT_NAME_SUFFIXES`

The 8 generational/professional suffix tokens (jr, sr, ii–iv, esq) that
`strip_name_suffixes` removes by default.

#### `DEFAULT_SPLIT_DELIMITERS`

The default split characters `(',', ';', '|', '/', '-')` used by
`split_on_delimiters`.

#### `DEFAULT_STREET_SUFFIX_MAP`

The curated 54-entry map of USPS street-suffix abbreviations (`ST`, `AVE`,
`BLVD`, …) and their spellings, used by `standardize_street_suffixes`.

#### `DEFAULT_UNIT_MARKER_MAP`

The 12-entry map of USPS unit-marker abbreviations (`APT`, `STE`, `FL`, …) and
their spellings, used by `standardize_unit_markers`.

### Type aliases

#### `ExpressionTransform`

`Callable[[pl.Expr], pl.Expr]` — the type of a single column transform. Annotate
your own transforms and recipes with it. Exported from the top level.

#### `EpochTimeUnit`

`Literal['s', 'ms', 'us']` — the epoch-resolution argument of
`normalize_epoch_timestamps`. On `framesmith.transforms`.

#### `CentralTendencyStrategy`

`Literal['auto', 'mean', 'median']` — the `strategy` argument of
`fill_null_numeric_by_group`. On `framesmith.group`.

#### `PeriodInterval`

`Literal['1d', '1w', '1mo', '1q', '1y']` — the `period` argument of
`within_complete_period`. On `framesmith.filters`.
