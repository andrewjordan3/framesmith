# framesmith

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/polars-1.0%2B-CD792C.svg)](https://pola.rs)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/andrewjordan3/framesmith)

A preprocessing library for cleaning messy data in polars DataFrames.
Composable atomic transforms, declarative recipes, expression-first design.

## Status

`framesmith` is in alpha. The public surface is small but the shape of the API
may still change as new transforms and recipes land. It is not yet published
to PyPI; install from source while the design settles.

## Why

Real-world data arrives with smart quotes, currency symbols, accounting-style
parentheses for negatives, mainframe trailing-minus, fullwidth digits, mixed
whitespace, and placeholder strings that mean "missing." The cleaning code
for these tends to scatter across notebooks, drift between projects, and
silently disagree about edge cases.

The design aims for a different shape: tiny, atomic transforms that each do
one thing; ordered tuples of those transforms ("recipes") that capture common
pipelines as plain data; and a single composition function that turns a
column name plus a recipe into a polars expression. Everything returns a
`pl.Expr` — `framesmith` never mutates a frame on your behalf. The user
applies the expression via `df.with_columns(...)` or `df.filter(...)`, which
keeps the library polars-native and lets the same code work eagerly or
lazily.

## Quick example

```python
import polars as pl
import framesmith as fs

raw = pl.DataFrame({
    'customer_name': ['  ACME® Corp  ', "O'Brien & Co.", '   '],
    'amount':        ['($1,234.56)',    '$2,500-',       'N/A'],
})

cleaned = raw.with_columns(
    fs.compose_column('customer_name', fs.NORMALIZE_TEXT),
    fs.compose_column('amount',        fs.NORMALIZE_NUMERIC),
)
print(cleaned)
# shape: (3, 2)
# ┌───────────────┬──────────┐
# │ customer_name ┆ amount   │
# │ ---           ┆ ---      │
# │ str           ┆ f64      │
# ╞═══════════════╪══════════╡
# │ ACME Corp     ┆ -1234.56 │
# │ OBrien and Co ┆ -2500.0  │
# │ null          ┆ null     │
# └───────────────┴──────────┘
```

Recipes are plain `tuple[ExpressionTransform, ...]` — splice them to extend:

```python
from framesmith.transforms import to_snake_case

normalize_and_snake = (*fs.NORMALIZE_TEXT, to_snake_case)
df_snake = raw.with_columns(
    fs.compose_column('customer_name', normalize_and_snake)
)
# 'OBrien and Co' becomes 'OBrien_and_Co', etc.
```

## Installation

`framesmith` is not on PyPI. Install from source:

```bash
git clone https://github.com/andrewjordan3/framesmith.git
cd framesmith
uv sync --all-extras --group dev   # or: pip install -e '.[pandas,dev]'
```

The `[pandas]` extra reserves dependencies for a future polars/pandas
interop layer — the extra exists in `pyproject.toml` so the install path
stays stable, but the interop module itself has not yet been built.

## Key concepts

The library is organized into three tiers plus two supporting patterns.

### Transforms

A transform is a pure `pl.Expr → pl.Expr` function. Each does exactly one
thing — `collapse_whitespace` collapses interior whitespace runs;
`strip_whitespace` trims the ends; `normalize_unicode_nfkc` applies NFKC.
Transforms never call `pl.col(...)` themselves and never call `.alias(...)`;
the composition layer owns those boundaries, so the same transform composes
into any pipeline without ceremony.

```python
from framesmith import compose_column
from framesmith.transforms import collapse_whitespace

df.with_columns(compose_column('description', [collapse_whitespace]))
```

The transforms shipped so far live in `framesmith.transforms` — see that
module for the current set.

### Recipes

A recipe is an ordered tuple of transforms: `tuple[ExpressionTransform, ...]`.
The most common recipes live in `framesmith.recipes` and are re-exported at
the top level — `NORMALIZE_TEXT`, `NORMALIZE_NUMERIC`, `NORMALIZE_PERCENT`,
`CLEAN_NUMERIC_STRING`, `UNICODE_TO_ASCII`.

Because recipes are plain tuples, they compose by splicing:

```python
my_recipe = (*fs.NORMALIZE_TEXT, to_snake_case)
```

And a recipe can include another recipe the same way — `NORMALIZE_TEXT`
itself splices `UNICODE_TO_ASCII`, so the Unicode-canonicalization order
has exactly one source of truth.

### `compose_column`

The single entry point that turns a column name and a recipe into an
expression. Signature:

```python
def compose_column(
    source_column_name: str,
    expression_transforms: Sequence[ExpressionTransform],
    output_column_name: str | None = None,
) -> pl.Expr: ...
```

It builds `pl.col(source_column_name)`, applies each transform in order,
and aliases the result back to the source column name (or to
`output_column_name` if given). An empty transform sequence raises
`ValueError` immediately — silent no-ops hide bugs.

### Factories (configured transforms)

When configuration is genuinely data-dependent — for example, which strings
count as "missing" varies by source — a transform factory takes the
configuration and returns a configured `ExpressionTransform`. Validation
and any precomputation happen once, in the factory body, so the per-call
work stays cheap. The factories shipped so far are `nullify_sentinels`
(configurable missing-value tokens) and `replace_whitespace_with`
(configurable separator).

```python
from framesmith.transforms import DEFAULT_MISSING_SENTINELS, nullify_sentinels

recipe = (*fs.NORMALIZE_TEXT, nullify_sentinels(DEFAULT_MISSING_SENTINELS))
```

Sentinel handling is opt-in by design and never appears in a default
recipe — defaulting it on would silently null valid values (e.g. `'NA'` as
Namibia).

### Filters (row selection)

Row selection follows the same expression-returning shape as column
transforms, but the user applies the expression via `df.filter(...)`:

```python
from framesmith.filters import within_complete_month

monthly = df.filter(within_complete_month('transaction_date'))
```

Filters compose with other boolean expressions through the usual
`&` and `|` — no `framesmith` abstraction is needed for that.

## Current building blocks

A scannable summary of what ships now. This is the current state, not the
final shape.

**Text** (`framesmith.transforms`): NFKC normalization, ASCII compatibility
folding (smart quotes, em-dashes, currency symbols, trademark/registered,
non-standard whitespace), whitespace handling (collapse, strip,
replace-with), ampersand expansion, apostrophe and period removal,
snake-case conversion, blank-to-null coercion.

**Numeric** (`framesmith.transforms`): accounting-style parens to negative,
mainframe trailing-minus, thousands-separator removal, `Float64` casting,
percent-to-fraction parsing.

**Names** (`framesmith.transforms`): trailing `jr` / `jr.` suffix removal.

**Missing-data sentinels** (`framesmith.transforms`): configurable sentinel
nullification via a factory, with a conservative default set
(`DEFAULT_MISSING_SENTINELS`).

**Recipes** (top-level): `NORMALIZE_TEXT`, `NORMALIZE_NUMERIC`,
`CLEAN_NUMERIC_STRING`, `NORMALIZE_PERCENT`, `UNICODE_TO_ASCII`.

**Filters** (`framesmith.filters`): incomplete-trailing-period exclusion for
date-based reporting columns (`within_complete_period`,
`within_complete_month`).

See `src/framesmith/` for the complete current surface. Each public symbol
carries a full docstring with examples.

## What's under consideration

Areas the library may grow into. None of these are commitments.

- A polars/pandas interop layer for bridging legacy pandas pipelines.
- Frame-level transforms beyond filters (column renaming, schema
  standardization, multi-column conditionals).
- "Plans" — a layer above recipes that handles multi-column pipelines as
  units, so a single object can describe an entire frame's preprocessing.
- Declarative YAML configuration for pipelines.
- Additional filter families (null-pattern filters, numeric range filters,
  categorical inclusion).

## Development

Engineering conventions live in [`CLAUDE.md`](CLAUDE.md). The repo uses
[`uv`](https://docs.astral.sh/uv/) for environment management.

```bash
uv run pytest tests/          # full test suite
uv run ruff check src/ tests/
uv run mypy src/
```

The current test suite covers atomic transforms, recipes, factories,
filters, the composition layer, and regex / pattern primitives, with
positive and negative cases for the public symbols shipped so far.

## License

Apache 2.0. See [LICENSE](LICENSE).
