# framesmith/transforms/__init__.py
"""Column-level expression transforms.

Public surface for this subpackage is exported here. Internal file
organization (``text.py``, ``names.py``, ``numeric.py``, ``missing.py``)
is private — callers import from ``framesmith.transforms``.
"""

from framesmith.transforms.addresses import (
    DEFAULT_DIRECTIONAL_MAP,
    DEFAULT_STREET_SUFFIX_MAP,
    DEFAULT_UNIT_MARKER_MAP,
    extract_zip_code,
    standardize_directionals,
    standardize_state,
    standardize_state_name,
    standardize_street_suffixes,
    standardize_unit_markers,
    strip_trailing_state,
)
from framesmith.transforms.bounds import (
    clip_numeric,
    winsorize_numeric,
)
from framesmith.transforms.categorical import (
    collapse_keep_top_n,
    collapse_rare_by_count,
    map_categories,
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
    DEFAULT_CREDENTIALS,
    DEFAULT_NAME_PREFIXES,
    DEFAULT_NAME_SUFFIXES,
    extract_email_local_part,
    remove_credentials,
    remove_jr_suffix,
    standardize_initials,
    strip_name_prefixes,
    strip_name_suffixes,
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
    'DEFAULT_CREDENTIALS',
    'DEFAULT_DIRECTIONAL_MAP',
    'DEFAULT_MISSING_SENTINELS',
    'DEFAULT_NAME_PREFIXES',
    'DEFAULT_NAME_SUFFIXES',
    'DEFAULT_STREET_SUFFIX_MAP',
    'DEFAULT_UNIT_MARKER_MAP',
    'EpochTimeUnit',
    'accounting_parens_to_negative',
    'apply_replacements',
    'cast_to_float64',
    'clip_numeric',
    'collapse_keep_top_n',
    'collapse_rare_by_count',
    'collapse_whitespace',
    'extract_email_local_part',
    'extract_zip_code',
    'flag_dates_outside_range',
    'flag_iqr_outliers',
    'flag_mad_outliers',
    'flag_zscore_outliers',
    'fold_to_ascii',
    'map_categories',
    'normalize_epoch_timestamps',
    'normalize_excel_serial_dates',
    'normalize_unicode_nfkc',
    'nullify_blank_strings',
    'nullify_sentinels',
    'percent_to_fraction',
    'periods_to_spaces',
    'remove_apostrophes',
    'remove_credentials',
    'remove_jr_suffix',
    'remove_periods',
    'remove_thousands_separators',
    'replace_ampersand_with_and',
    'replace_whitespace_with',
    'standardize_directionals',
    'standardize_initials',
    'standardize_state',
    'standardize_state_name',
    'standardize_street_suffixes',
    'standardize_unit_markers',
    'strip_name_prefixes',
    'strip_name_suffixes',
    'strip_trailing_state',
    'strip_whitespace',
    'to_lowercase',
    'to_snake_case',
    'to_titlecase',
    'trailing_minus_to_prefix',
    'underscores_to_spaces',
    'winsorize_numeric',
]
