# framesmith/_internal/name_tokens.py
"""
Curated default token sets for name-affix removal.

Lowercase, period-free tokens. The transforms match case-insensitively and
allow an optional trailing period, so each token appears here once in its
bare lowercase form.
"""

__all__: list[str] = ['DEFAULT_NAME_PREFIXES', 'DEFAULT_NAME_SUFFIXES']


# Trailing generational / post-nominal suffixes. Bare 'i' and 'v' are
# deliberately excluded: they collide with single-letter middle initials
# (e.g. 'John V' must keep its 'V').
DEFAULT_NAME_SUFFIXES: tuple[str, ...] = (
    'jr', 'sr', 'jnr', 'snr', 'ii', 'iii', 'iv', 'esq',
)

# Leading honorifics. 'st' is deliberately excluded: it is a surname element
# ('St. John', 'St. James'), not an honorific.
DEFAULT_NAME_PREFIXES: tuple[str, ...] = (
    'mr', 'mrs', 'ms', 'miss', 'mx', 'dr', 'prof', 'rev', 'sir', 'hon',
    'capt', 'dame',
)
