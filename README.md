# framesmith

Composable, expression-first preprocessing toolkit for polars DataFrames.

> **Status:** Early development. APIs will change.

## Overview

`framesmith` is a preprocessing library built around polars expressions.
It provides column-level expression builders, frame-level transforms, and
reusable plans for repeatable pipelines — with optional pandas interop
at the edges.

## Design

- **Expressions over functions.** Most tools return `pl.Expr`, so they
  compose naturally inside `with_columns()` and benefit from polars'
  query optimizer.
- **Three tiers of use.** One-off expressions for ad-hoc work, plans for
  repeatable pipelines, YAML configs for declarative workflows.
- **Polars-first, pandas-friendly.** Core is polars. An optional interop
  layer converts at the boundaries when you need to hand off to or
  receive from pandas-based code.

## Installation

```bash
pip install framesmith              # core (polars only)
pip install framesmith[pandas]      # adds pandas interop
```

## Quick example

```python
import polars as pl
import framesmith as fs

raw_frame = pl.read_csv('customers.csv')

cleaned_frame = raw_frame.with_columns(
    fs.normalize_whitespace('customer_name'),
    fs.clean_zip_code('postal_code'),
    fs.to_boolean_flag('active_flag'),
)
```

## License

Apache 2.0. See [LICENSE](LICENSE).
