# framesmith/_internal/name_tokens.py
"""
Curated default token sets for name-affix removal.

Lowercase, period-free tokens. The transforms match case-insensitively and
allow an optional trailing period, so each token appears here once in its
bare lowercase form.
"""

__all__: list[str] = [
    'DEFAULT_CREDENTIALS',
    'DEFAULT_NAME_PREFIXES',
    'DEFAULT_NAME_SUFFIXES',
]


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

# Trailing professional post-nominal credentials. Lowercase, period-free;
# the transform matches case-insensitively and tolerates internal periods
# ('Ph.D.'). Collision-prone short tokens are deliberately excluded from the
# default: 'do' / 'pa' / 'od' (common surnames and words) and the bare
# degrees 'ba' / 'bs' / 'ma' / 'ms' (which also clash with state codes).
# Pass a custom list to include them.
DEFAULT_CREDENTIALS: tuple[str, ...] = (
    # Academic doctorates
    'phd', 'edd', 'psyd', 'scd', 'dsc',
    # Medical / dental / veterinary / podiatric / pharmacy
    'md', 'dds', 'dmd', 'dvm', 'dpm', 'pharmd',
    # Nursing and allied health
    'rn', 'np', 'crna', 'lpn', 'rdn',
    # Legal
    'jd', 'llm',
    # Business and finance
    'mba', 'cpa', 'cfa', 'cfp', 'cma',
    # Engineering and project management
    'pe', 'pmp',
    # Counseling and social work / public health
    'msw', 'lcsw', 'lmft', 'lpc', 'mph',
    # Fellowships
    'facs', 'facp', 'faap', 'facc',
)
