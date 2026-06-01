# framesmith

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Polars](https://img.shields.io/badge/polars-1.0%2B-CD792C.svg)](https://pola.rs)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/andrewjordan3/framesmith)

A preprocessing library for cleaning messy data in polars DataFrames.
Composable atomic transforms, declarative recipes, expression-first design.

## Status

`framesmith` is in alpha. The public surface and API shape may still change as
new transforms and recipes land. It is not yet published to PyPI; install from
source while the design settles.

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
    fs.compose_column('customer_name', fs.TEXT_NORMALIZE),
    fs.compose_column('amount',        fs.NUMERIC_STRING_TO_FLOAT),
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

normalize_and_snake = (*fs.TEXT_NORMALIZE, to_snake_case)
df_snake = raw.with_columns(
    fs.compose_column('customer_name', normalize_and_snake)
)
# 'OBrien and Co' becomes 'obrien_and_co', etc.
# (this exact pipeline also ships ready-made as fs.TEXT_NORMALIZE_TO_SNAKE_CASE)
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

The full set of transforms lives in `framesmith.transforms` — see the
[reference](docs/reference.md) for every one, grouped by domain.

### Recipes

A recipe is an ordered tuple of transforms: `tuple[ExpressionTransform, ...]`.
All recipes live in `framesmith.recipes` and are re-exported at the top level,
so `from framesmith import TEXT_NORMALIZE` works directly. They follow a naming
protocol so the name states what the recipe does:

- `<INPUT>_CANONICALIZE` — meaning-preserving representation cleanup
  (whitespace, case, Unicode form).
- `<INPUT>_NORMALIZE` — domain-aware cleanup that interprets the value (an
  address, a name, a number).
- `<INPUT>_TO_<FORM>` — a conversion whose output form or dtype differs
  (`TO_FLOAT`, `TO_SNAKE_CASE`, `TO_TITLE`, …).

Because recipes are plain tuples, they compose by splicing:

```python
my_recipe = (*fs.TEXT_NORMALIZE, to_snake_case)
```

And a recipe can include another recipe the same way — `TEXT_NORMALIZE` builds
on `TEXT_CANONICALIZE`, which itself splices `UNICODE_TO_ASCII`, so the
canonicalization order has exactly one source of truth.

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
work stays cheap. Several transforms are factories — `nullify_sentinels`
(configurable missing-value tokens), `map_categories` (a label remap),
`pad_left` (fixed-width padding), and the address standardizers among them;
the [reference](docs/reference.md) shows which transforms take configuration.

```python
from framesmith.transforms import DEFAULT_MISSING_SENTINELS, nullify_sentinels

recipe = (*fs.TEXT_NORMALIZE, nullify_sentinels(DEFAULT_MISSING_SENTINELS))
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

## Reference

The complete reference — every transform, recipe, filter, and frame-level
helper, with its signature, a short description, and an example — lives in
[`docs/reference.md`](docs/reference.md). It is organized by package: recipes
first, then transforms grouped by domain (whitespace, case, unicode, numeric,
names, addresses, dates, outliers, categorical, and more), then the filter,
combine, group, validate, schema, and canonicalize helpers.

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
uv run pytest                 # full suite, including src doctests
uv run ruff check src/ tests/
uv run mypy src/
```

The test suite covers the atomic transforms, recipes, factories, filters, the
composition layer, and the regex / pattern primitives, with positive and
negative cases, plus the docstring examples run as doctests.

## License

Apache 2.0. See [LICENSE](LICENSE).
