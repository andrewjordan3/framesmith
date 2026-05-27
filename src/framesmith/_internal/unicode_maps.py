# framesmith/_internal/unicode_maps.py
"""
Shared Unicode translation maps for text and numeric normalization.

Consolidates character-class maps that are used by multiple preprocessing
modules. Each map is a dict[str, str] suitable for composing into a
``str.maketrans()`` translation table.

Consumers compose what they need from the individual maps:

    # numeric_series.py — only needs minus + invisible
    from framesmith._internal.unicode_maps import (
        MINUS_LIKE_MAP,
        INVISIBLE_CHAR_MAP,
    )
    _NUMERIC_TRANSLATION_TABLE = str.maketrans(
        {**MINUS_LIKE_MAP, **INVISIBLE_CHAR_MAP}
    )

    # normalize_series.py — needs the full superset
    from truck_revenue.preprocessing.unicode_maps import (
        ASCII_COMPAT_MAP,
    )
    _ASCII_TRANSLATION_TABLE = str.maketrans(ASCII_COMPAT_MAP)

Map categories:
    MINUS_LIKE_MAP           Dashes and minus variants → '-'
    INVISIBLE_CHAR_MAP       Zero-width, joiners, fillers, BiDi → ''
    WHITESPACE_VARIANT_MAP   Non-standard whitespace → ' '
    CURRENCY_SYMBOL_MAP      Currency symbols → ''
    QUOTE_AND_PRIME_MAP      Typographic quotes/primes → ASCII quotes
    PUNCTUATION_SYMBOL_MAP   Bullets, math, ordinals, misc → ASCII
    ASCII_COMPAT_MAP         Union of all above (full superset)

Design notes:
    All maps target post-NFKC input.  NFKC decomposes many compatibility
    forms (fullwidth digits, some superscripts) before these maps run.
    However, several characters survive NFKC and require explicit mapping:

    * U+FE63 SMALL HYPHEN-MINUS survives NFKC → must map to '-'
    * U+2212 MINUS SIGN survives NFKC → must map to '-'
    * All zero-width / BiDi characters survive NFKC → must remove

    Redundant-post-NFKC entries (e.g. U+FF0D FULLWIDTH HYPHEN-MINUS,
    which NFKC decomposes to U+002D) are retained as defensive coverage.
    They cost nothing at runtime (the translate table is a hash lookup)
    and protect against upstream changes in normalization order.
"""

__all__: list[str] = [
    'ASCII_COMPAT_MAP',
    'CURRENCY_SYMBOL_MAP',
    'INVISIBLE_CHAR_MAP',
    'MINUS_LIKE_MAP',
    'PUNCTUATION_SYMBOL_MAP',
    'QUOTE_AND_PRIME_MAP',
    'WHITESPACE_VARIANT_MAP',
]


# =====================================================================
# Minus-like characters → standard hyphen-minus (U+002D)
# =====================================================================
# These appear when data passes through word processors, PDFs, web
# systems, or mainframe exports that substitute typographic dashes,
# superscript/subscript minus signs, or presentation-form variants
# for the ASCII minus sign.

MINUS_LIKE_MAP: dict[str, str] = {
    '\u2212': '-',  # MINUS SIGN
    '\u2010': '-',  # HYPHEN
    '\u2011': '-',  # NON-BREAKING HYPHEN
    '\u2012': '-',  # FIGURE DASH
    '\u2013': '-',  # EN DASH
    '\u2014': '-',  # EM DASH
    '\u2015': '-',  # HORIZONTAL BAR
    '\u207b': '-',  # SUPERSCRIPT MINUS
    '\u208b': '-',  # SUBSCRIPT MINUS
    '\ufe63': '-',  # SMALL HYPHEN-MINUS (survives NFKC)
    '\uff0d': '-',  # FULLWIDTH HYPHEN-MINUS (redundant post-NFKC, defensive)
}


# =====================================================================
# Invisible / zero-width / BiDi control characters → remove
# =====================================================================
# These hide inside copy-pasted text, PDF extractions, web scrapes,
# and CJK/RTL data.  They are invisible in most editors and terminals
# but break string equality, regex matching, and numeric parsing.
#
# Organized into three sub-groups for readability, but shipped as a
# single flat dict for consumer convenience.

INVISIBLE_CHAR_MAP: dict[str, str] = {
    # ── Zero-width and joiner characters ─────────────────────
    '\u00ad': '',  # SOFT HYPHEN
    '\u200b': '',  # ZERO WIDTH SPACE
    '\u200c': '',  # ZERO WIDTH NON-JOINER
    '\u200d': '',  # ZERO WIDTH JOINER
    '\u2060': '',  # WORD JOINER
    '\u180e': '',  # MONGOLIAN VOWEL SEPARATOR
    '\ufeff': '',  # ZERO WIDTH NO-BREAK SPACE (BOM)
    '\u034f': '',  # COMBINING GRAPHEME JOINER
    '\u061c': '',  # ARABIC LETTER MARK
    # ── Script-specific fillers ──────────────────────────────
    '\u115f': '',  # HANGUL CHOSEONG FILLER
    '\u1160': '',  # HANGUL JUNGSEONG FILLER
    '\u17b4': '',  # KHMER VOWEL INHERENT AQ
    '\u17b5': '',  # KHMER VOWEL INHERENT AA
    # ── BiDi / directional control characters ────────────────
    '\u200e': '',  # LEFT-TO-RIGHT MARK
    '\u200f': '',  # RIGHT-TO-LEFT MARK
    '\u202a': '',  # LEFT-TO-RIGHT EMBEDDING
    '\u202b': '',  # RIGHT-TO-LEFT EMBEDDING
    '\u202c': '',  # POP DIRECTIONAL FORMATTING
    '\u202d': '',  # LEFT-TO-RIGHT OVERRIDE
    '\u202e': '',  # RIGHT-TO-LEFT OVERRIDE
    '\u2066': '',  # LEFT-TO-RIGHT ISOLATE
    '\u2067': '',  # RIGHT-TO-LEFT ISOLATE
    '\u2068': '',  # FIRST STRONG ISOLATE
    '\u2069': '',  # POP DIRECTIONAL ISOLATE
    '\u206a': '',  # INHIBIT SYMMETRIC SWAPPING
    '\u206b': '',  # ACTIVATE SYMMETRIC SWAPPING
    '\u206c': '',  # INHIBIT ARABIC FORM SHAPING
    '\u206d': '',  # ACTIVATE ARABIC FORM SHAPING
    '\u206e': '',  # NATIONAL DIGIT SHAPES
    '\u206f': '',  # NOMINAL DIGIT SHAPES
}


# =====================================================================
# Non-standard whitespace → standard ASCII space (U+0020)
# =====================================================================
# Covers Unicode whitespace variants, control characters that act as
# line breaks, and typographic spaces from word processors and web
# systems.  After translation, consumers typically collapse runs of
# spaces and strip — this map just unifies the character identity.

WHITESPACE_VARIANT_MAP: dict[str, str] = {
    '\t': ' ',  # TAB
    '\n': ' ',  # NEWLINE (for multi-line addresses/names)
    '\r': ' ',  # CARRIAGE RETURN
    '\u00a0': ' ',  # NO-BREAK SPACE
    '\u2000': ' ',  # EN QUAD
    '\u2001': ' ',  # EM QUAD
    '\u2002': ' ',  # EN SPACE
    '\u2003': ' ',  # EM SPACE
    '\u2004': ' ',  # THREE-PER-EM SPACE
    '\u2005': ' ',  # FOUR-PER-EM SPACE
    '\u2006': ' ',  # SIX-PER-EM SPACE
    '\u2007': ' ',  # FIGURE SPACE
    '\u2008': ' ',  # PUNCTUATION SPACE
    '\u2009': ' ',  # THIN SPACE
    '\u200a': ' ',  # HAIR SPACE
    '\u202f': ' ',  # NARROW NO-BREAK SPACE
    '\u205f': ' ',  # MEDIUM MATHEMATICAL SPACE
    '\u3000': ' ',  # IDEOGRAPHIC SPACE
    '\u2028': ' ',  # LINE SEPARATOR
    '\u2029': ' ',  # PARAGRAPH SEPARATOR
}


# =====================================================================
# Currency symbols → remove
# =====================================================================
# Used by both numeric coercion (strip before parsing) and text
# normalization (remove for entity matching).  The numeric module
# also maintains a regex pattern for positional currency stripping
# but uses this map via the combined translation table as a first pass.

CURRENCY_SYMBOL_MAP: dict[str, str] = {
    '\u0024': '',  # DOLLAR SIGN ($)
    '\u00a2': '',  # CENT SIGN (¢)
    '\u00a3': '',  # POUND SIGN (£)
    '\u00a4': '',  # CURRENCY SIGN (¤)
    '\u00a5': '',  # YEN SIGN (¥)
    '\u20ac': '',  # EURO SIGN (€)
    '\u20b9': '',  # INDIAN RUPEE SIGN (₹)
    '\u20bd': '',  # RUBLE SIGN (₽)
    '\u20a9': '',  # WON SIGN (₩)
    '\u20ab': '',  # DONG SIGN (₫)
    '\u20aa': '',  # NEW SHEQEL SIGN (₪)
    '\u20a1': '',  # COLON SIGN (₡)
    '\u20ba': '',  # TURKISH LIRA SIGN (₺)
    '\u20b1': '',  # PESO SIGN (₱)
    '\u20b4': '',  # HRYVNIA SIGN (₴)
    '\u20b8': '',  # TENGE SIGN (₸)
    '\u20a6': '',  # NAIRA SIGN (₦)
    '\u20b5': '',  # CEDI SIGN (₵)
    '\u2030': '',  # PER MILLE SIGN (‰)
    '\u2031': '',  # PER TEN THOUSAND SIGN (‱)
}


# =====================================================================
# Typographic quotes and prime marks → ASCII quotes
# =====================================================================
# Word processors and web systems routinely replace ASCII quotes with
# "smart quotes".  Prime marks (feet/inches, minutes/seconds) also
# get substituted.  This map unifies everything to ASCII ' and ".

QUOTE_AND_PRIME_MAP: dict[str, str] = {
    # ── Single quotes and primes ─────────────────────────────
    '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
    '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
    '\u201a': "'",  # SINGLE LOW-9 QUOTATION MARK
    '\u201b': "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    '\u2032': "'",  # PRIME (feet / minutes)
    '\u2035': "'",  # REVERSED PRIME
    '\u2039': "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK (‹)  # noqa: RUF003
    '\u203a': "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK (›)  # noqa: RUF003
    # ── Double quotes and double primes ──────────────────────
    '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK
    '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK
    '\u201e': '"',  # DOUBLE LOW-9 QUOTATION MARK
    '\u201f': '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    '\u2033': '"',  # DOUBLE PRIME (inches / seconds)
    '\u2036': '"',  # REVERSED DOUBLE PRIME
    '\u00ab': '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK («)
    '\u00bb': '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK (»)
    # ── Triple primes ────────────────────────────────────────
    '\u2034': "'''",  # TRIPLE PRIME
    '\u2037': "'''",  # REVERSED TRIPLE PRIME
}


# =====================================================================
# Punctuation, symbols, and miscellaneous → ASCII equivalents
# =====================================================================
# Catch-all for remaining Unicode punctuation that appears in names,
# addresses, part numbers, and descriptions.  Organized by functional
# group.

PUNCTUATION_SYMBOL_MAP: dict[str, str] = {
    # ── Slashes and division ─────────────────────────────────
    '\u2044': '/',  # FRACTION SLASH → SOLIDUS
    '\u2215': '/',  # DIVISION SLASH → SOLIDUS
    '\u00f7': '/',  # DIVISION SIGN → SOLIDUS (keep for part numbers)
    '\u00d7': 'x',  # MULTIPLICATION SIGN → letter x
    # ── Bullets and dot operators ─────────────────────────────
    '\u2043': '-',  # HYPHEN BULLET → hyphen
    '\u2022': '-',  # BULLET → hyphen
    '\u2023': '-',  # TRIANGULAR BULLET → hyphen
    '\u2027': '-',  # HYPHENATION POINT → hyphen
    '\u2219': '.',  # BULLET OPERATOR → period
    '\u2024': '.',  # ONE DOT LEADER → period
    '\u2025': '..',  # TWO DOT LEADER → two periods
    '\u2026': '...',  # HORIZONTAL ELLIPSIS → three periods
    '\u00b7': ' ',  # MIDDLE DOT → space (acts as separator)
    # ── Trademark / legal symbols (remove) ────────────────────
    '\u00ae': '',  # REGISTERED SIGN (®)
    '\u2122': '',  # TRADE MARK SIGN (™)
    '\u00a9': '',  # COPYRIGHT SIGN (©)
    '\u2120': '',  # SERVICE MARK (℠)
    '\u00b0': '',  # DEGREE SIGN (°)
    '\u02da': '',  # RING ABOVE (˚)
    # ── Mathematical and technical symbols ────────────────────
    '\u00b1': '+-',  # PLUS-MINUS SIGN
    '\u2248': '~',  # ALMOST EQUAL TO → tilde
    '\u2260': '!=',  # NOT EQUAL TO
    '\u2264': '<=',  # LESS-THAN OR EQUAL TO
    '\u2265': '>=',  # GREATER-THAN OR EQUAL TO
    '\u00ac': '!',  # NOT SIGN → exclamation
    '\u221e': '',  # INFINITY → remove
    '\u2218': 'o',  # RING OPERATOR → 'o' (seen in part numbers)
    # ── Ordinal indicators (important for addresses) ──────────
    '\u00aa': 'a',  # FEMININE ORDINAL INDICATOR (ª)
    '\u00ba': 'o',  # MASCULINE ORDINAL INDICATOR (º)
    '\u02e2': 's',  # MODIFIER LETTER SMALL S (superscript s)
    '\u02e3': 'x',  # MODIFIER LETTER SMALL X (superscript x)
    # ── Section / reference marks (remove) ────────────────────
    '\u00a6': '|',  # BROKEN BAR → vertical bar
    '\u00a7': '',  # SECTION SIGN (§)
    '\u00b6': '',  # PILCROW SIGN (¶)
    '\u2020': '',  # DAGGER (†)
    '\u2021': '',  # DOUBLE DAGGER (‡)
    '\u203b': '*',  # REFERENCE MARK → asterisk
    '\u204b': '',  # REVERSED PILCROW SIGN
    '\u204c': '',  # BLACK LEFTWARDS BULLET
    '\u204d': '',  # BLACK RIGHTWARDS BULLET
    # ── Compound punctuation ──────────────────────────────────
    '\u203c': '!!',  # DOUBLE EXCLAMATION MARK
    '\u203d': '?!',  # INTERROBANG
    '\u2047': '??',  # DOUBLE QUESTION MARK
    '\u2048': '?!',  # QUESTION EXCLAMATION MARK
    '\u2049': '!?',  # EXCLAMATION QUESTION MARK
    # ── Miscellaneous ─────────────────────────────────────────
    '\u2052': '%',  # COMMERCIAL MINUS SIGN → percent
    '\u2053': '~',  # SWUNG DASH → tilde
}


# =====================================================================
# Combined superset: all maps merged
# =====================================================================
# This is the full translation map used by text normalization.
# Numeric normalization uses only MINUS_LIKE_MAP + INVISIBLE_CHAR_MAP.
#
# Merge order matters only for documentation — there are no key
# collisions between maps (verified by the assertion below).

ASCII_COMPAT_MAP: dict[str, str] = {
    **MINUS_LIKE_MAP,
    **INVISIBLE_CHAR_MAP,
    **WHITESPACE_VARIANT_MAP,
    **CURRENCY_SYMBOL_MAP,
    **QUOTE_AND_PRIME_MAP,
    **PUNCTUATION_SYMBOL_MAP,
}

# Compile-time integrity check: no key appears in multiple maps.
# If this fires, two maps disagree on how to translate the same
# codepoint — resolve by choosing one canonical mapping.
_EXPECTED_KEY_COUNT: int = (
    len(MINUS_LIKE_MAP)
    + len(INVISIBLE_CHAR_MAP)
    + len(WHITESPACE_VARIANT_MAP)
    + len(CURRENCY_SYMBOL_MAP)
    + len(QUOTE_AND_PRIME_MAP)
    + len(PUNCTUATION_SYMBOL_MAP)
)
assert len(ASCII_COMPAT_MAP) == _EXPECTED_KEY_COUNT, (
    f'Key collision detected across unicode maps: '
    f'{_EXPECTED_KEY_COUNT} individual entries but '
    f'{len(ASCII_COMPAT_MAP)} unique keys in combined map. '
    f'Two or more maps define the same codepoint.'
)
