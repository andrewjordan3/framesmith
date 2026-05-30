# framesmith/transforms/__init__.py
"""Column-level expression transforms.

Public surface for this subpackage is exported here. Internal file
organization (``text.py``, ``names.py``, ``numeric.py``, ``missing.py``)
is private — callers import from ``framesmith.transforms``.
"""

from framesmith.transforms.addresses import (
    standardize_state,
    standardize_state_name,
    strip_trailing_state,
)
from framesmith.transforms.categorical import (
    collapse_keep_top_n,
    collapse_rare_by_count,
)
from framesmith.transforms.dates import (
    EpochTimeUnit,
    flag_dates_outside_range,
    normalize_epoch_timestamps,
    normalize_excel_serial_dates,
)
from framesmith.transforms.missing import (
    DEFAULT_MISSING_SENTINELS,
    nullify_sentinels,
)
from framesmith.transforms.names import (
    extract_email_local_part,
    remove_jr_suffix,
)
from framesmith.transforms.numeric import (
    accounting_parens_to_negative,
    cast_to_float64,
    percent_to_fraction,
    remove_thousands_separators,
    trailing_minus_to_prefix,
)
from framesmith.transforms.outliers import (
    flag_iqr_outliers,
    flag_mad_outliers,
    flag_zscore_outliers,
)
from framesmith.transforms.text import (
    apply_replacements,
    collapse_whitespace,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    periods_to_spaces,
    remove_apostrophes,
    remove_periods,
    replace_ampersand_with_and,
    replace_whitespace_with,
    strip_whitespace,
    to_lowercase,
    to_snake_case,
    to_titlecase,
    underscores_to_spaces,
)

__all__: list[str] = [
    'DEFAULT_MISSING_SENTINELS',
    'EpochTimeUnit',
    'accounting_parens_to_negative',
    'apply_replacements',
    'cast_to_float64',
    'collapse_keep_top_n',
    'collapse_rare_by_count',
    'collapse_whitespace',
    'extract_email_local_part',
    'flag_dates_outside_range',
    'flag_iqr_outliers',
    'flag_mad_outliers',
    'flag_zscore_outliers',
    'fold_to_ascii',
    'normalize_epoch_timestamps',
    'normalize_excel_serial_dates',
    'normalize_unicode_nfkc',
    'nullify_blank_strings',
    'nullify_sentinels',
    'percent_to_fraction',
    'periods_to_spaces',
    'remove_apostrophes',
    'remove_jr_suffix',
    'remove_periods',
    'remove_thousands_separators',
    'replace_ampersand_with_and',
    'replace_whitespace_with',
    'standardize_state',
    'standardize_state_name',
    'strip_trailing_state',
    'strip_whitespace',
    'to_lowercase',
    'to_snake_case',
    'to_titlecase',
    'trailing_minus_to_prefix',
    'underscores_to_spaces',
]
