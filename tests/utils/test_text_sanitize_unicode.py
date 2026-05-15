"""Unicode-layer fixture tests for gigaevo/utils/text_sanitize.py.

These tests complement test_text_sanitize.py by exercising Unicode-domain
concerns the original suite does not cover: confusables / homoglyphs,
normalization-form invariance, zero-width characters, weak BIDI marks
(distinct from the strong overrides already stripped), variation selectors,
tag characters, line / paragraph separators, BOM placement, soft hyphen,
Zalgo combining stacks, full CJK script families, RTL scripts, emoji ZWJ
sequences (family / rainbow flag), Fitzpatrick skin-tone modifiers,
regional-indicator country flags, math alphanumerics above the BMP, and
half-/full-width and non-Latin digit forms.

Every fixture is built as a positive case (something the sanitizer must
preserve verbatim) or a negative case (something it must strip or escape).
Each test carries a comment explaining why the input is interesting and
what property it asserts.
"""

from __future__ import annotations

import json
import unicodedata

import pytest

from gigaevo.utils.text_sanitize import (
    clean_identifier,
    sanitize_for_dbtext,
    sanitize_for_json,
    sanitize_for_log,
)

# ---------------------------------------------------------------------------
# Confusables and homoglyphs
# ---------------------------------------------------------------------------


class TestConfusables:
    """Latin / Cyrillic / Greek letters that render identically.

    The threat: an operator reading a log sees ``openai`` but the string
    actually contains Cyrillic letters that route to a different host or
    poison a cache key. Two distinct contracts:

    * sanitize_for_log preserves all three so the operator at least has the
      raw bytes available for inspection / forensic comparison.
    * clean_identifier strips non-ASCII so cache keys / log tags collapse to
      a single canonical form (which, paradoxically, also reveals the
      attack because the spoofed identifier shrinks dramatically).
    """

    # Latin A U+0041, Cyrillic А U+0410, Greek capital Α U+0391 all render
    # visually identical in most fonts. Positive case for sanitize_for_log.
    @pytest.mark.parametrize(
        "codepoint,name",
        [
            (0x0041, "Latin A"),
            (0x0410, "Cyrillic A"),
            (0x0391, "Greek Alpha"),
        ],
    )
    def test_log_preserves_each_confusable(self, codepoint: int, name: str) -> None:
        src = f"prefix_{chr(codepoint)}_suffix"
        assert sanitize_for_log(src) == src, name

    def test_log_preserves_all_three_side_by_side(self) -> None:
        # The trio together: operator can compare byte-for-byte.
        src = "AАΑ"
        assert sanitize_for_log(src) == src

    def test_identifier_collapses_cyrillic_to_empty(self) -> None:
        # Pure Cyrillic identifier is not safe — drops to nothing.
        assert clean_identifier("опенаи") == ""

    def test_identifier_strips_only_non_ascii_from_mixed(self) -> None:
        # Spoofed ``openai``: o and a are Cyrillic (U+043E, U+0430), the
        # rest are Latin. Only Latin chars survive, yielding a string that
        # is visibly shorter than the input — that asymmetry is the tell.
        spoof = "оpenаi"  # о p e n а i
        assert clean_identifier(spoof) == "peni"
        assert len(clean_identifier(spoof)) < len(spoof)

    def test_identifier_strips_greek_alpha(self) -> None:
        # Greek capital Α masquerading as Latin A inside a model name.
        # The single Greek char is removed, leaving a single dash between
        # ``model`` and ``lpha`` — visible shortening reveals the spoof.
        assert clean_identifier("model-Αlpha-v2") == "model-lpha-v2"


# ---------------------------------------------------------------------------
# Unicode normalization: NFC vs NFD
# ---------------------------------------------------------------------------


class TestNormalizationInvariance:
    """The sanitizer must not normalize either way.

    Two strings that render identically but live in different normalization
    forms (precomposed NFC vs decomposed NFD) carry different byte sequences.
    A sanitizer that silently normalized would corrupt round-tripping into
    systems that hash on bytes (asyncpg LISTEN/NOTIFY, content-addressed
    caches, signature verification).
    """

    @pytest.mark.parametrize(
        "nfc,nfd",
        [
            ("é", "é"),  # é (acute)
            ("è", "è"),  # è (grave)
            ("ñ", "ñ"),  # ñ (tilde)
            ("ä", "ä"),  # ä (diaeresis)
            ("Å", "Å"),  # Å (ring above)
            ("ẛ", "ẛ"),  # ẛ (long s with dot)
        ],
    )
    def test_nfc_and_nfd_both_round_trip(self, nfc: str, nfd: str) -> None:
        # Sanity: the two forms really do normalize to each other.
        assert unicodedata.normalize("NFC", nfd) == nfc
        # Both forms must survive verbatim — no implicit normalization.
        assert sanitize_for_log(nfc) == nfc
        assert sanitize_for_log(nfd) == nfd
        # And neither becomes the other.
        assert sanitize_for_log(nfd) != nfc
        assert sanitize_for_log(nfc) != nfd

    def test_nfd_preserves_combining_count(self) -> None:
        # NFD string: ``e`` + combining acute. Length must not change.
        src = "é"
        out = sanitize_for_log(src)
        assert len(out) == 2 and out[1] == "́"


# ---------------------------------------------------------------------------
# Zero-width characters
# ---------------------------------------------------------------------------


class TestZeroWidth:
    """ZWSP, ZWNJ, ZWJ, WJ all carry semantic weight.

    * ZWJ (U+200D) joins emoji into family / profession sequences.
    * ZWNJ (U+200C) inhibits ligatures (essential in Persian / Indic).
    * ZWSP (U+200B) marks word boundaries in scripts with no spaces.
    * WJ  (U+2060) is a no-break joiner used in line-break control.

    None are in the BIDI override / isolate ranges and must survive.
    Negative case: would a zero-width injection inside a clean_identifier
    survive? No — those characters are not in the ASCII charset, so the
    identifier path strips them.
    """

    @pytest.mark.parametrize(
        "codepoint,name",
        [
            (0x200B, "ZWSP"),
            (0x200C, "ZWNJ"),
            (0x200D, "ZWJ"),
            (0x2060, "WJ"),
        ],
    )
    def test_log_preserves_zero_width(self, codepoint: int, name: str) -> None:
        # Positive: zero-width chars survive sanitize_for_log untouched.
        src = f"a{chr(codepoint)}b"
        assert sanitize_for_log(src) == src, name

    @pytest.mark.parametrize(
        "codepoint",
        [0x200B, 0x200C, 0x200D, 0x2060],
    )
    def test_identifier_strips_zero_width(self, codepoint: int) -> None:
        # Negative: zero-width chars must not survive into identifiers, or
        # cache keys would collapse / collide invisibly.
        src = f"model{chr(codepoint)}name"
        assert clean_identifier(src) == "modelname"


# ---------------------------------------------------------------------------
# Weak BIDI marks (NOT the strong overrides already stripped)
# ---------------------------------------------------------------------------


class TestWeakBidiMarks:
    """LRM / RLM / ALM are weak hints, not directionality overrides.

    Unicode classifies these as Bidi_Class=L / R / AL respectively (weak),
    distinct from the strong overrides U+202A-U+202E and isolates
    U+2066-U+2069 (which the sanitizer correctly strips). The strip regex
    is ``[U+202A-U+202E][U+2066-U+2069]`` — narrow on purpose. Weak marks
    are legitimate for mixed-direction text rendering inside Arabic /
    Hebrew strings. Asserting the current behavior so a future regex
    widening would have to deliberately break this test.
    """

    @pytest.mark.parametrize(
        "codepoint,name",
        [
            (0x200E, "LRM"),  # Left-to-right mark
            (0x200F, "RLM"),  # Right-to-left mark
            (0x061C, "ALM"),  # Arabic letter mark
        ],
    )
    def test_weak_bidi_preserved(self, codepoint: int, name: str) -> None:
        src = f"a{chr(codepoint)}b"
        assert sanitize_for_log(src) == src, name


# ---------------------------------------------------------------------------
# BIDI strip pattern boundary checks
# ---------------------------------------------------------------------------


class TestBidiStripBoundaries:
    """Ensure the strip pattern is exact and not over-broad.

    The pattern catches U+202A-U+202E and U+2066-U+2069 only. Adjacent
    characters (U+2029 PS, U+202F NNBSP, U+2065 reserved, U+206A deprecated
    formatter) must pass through. A regression that widened the class
    would break legitimate RTL rendering and CJK numbering.
    """

    @pytest.mark.parametrize(
        "codepoint,name",
        [
            (0x2029, "PARAGRAPH SEPARATOR (just below LRE)"),
            (0x202F, "NARROW NO-BREAK SPACE (just above RLO)"),
            (0x2065, "reserved, just below LRI"),
            (0x206A, "INHIBIT SYMMETRIC SWAPPING (deprecated, just above PDI)"),
            (0x206F, "NOMINAL DIGIT SHAPES (deprecated)"),
        ],
    )
    def test_neighbors_of_bidi_range_preserved(
        self, codepoint: int, name: str
    ) -> None:
        src = f"a{chr(codepoint)}b"
        assert sanitize_for_log(src) == src, name


# ---------------------------------------------------------------------------
# Variation selectors
# ---------------------------------------------------------------------------


class TestVariationSelectors:
    """VS1-VS16 (U+FE00-U+FE0F) and VS17-VS256 (U+E0100-U+E01EF).

    VS16 (U+FE0F) is the emoji-presentation selector: ``❤️`` is U+2764
    HEAVY BLACK HEART followed by U+FE0F. Stripping VS16 silently changes
    rendering from emoji to a black-and-white dingbat. The other VS code
    points carry semantic meaning in CJK ideographic variant selectors.
    """

    @pytest.mark.parametrize("codepoint", [0xFE00, 0xFE07, 0xFE0E, 0xFE0F])
    def test_vs1_to_vs16_preserved(self, codepoint: int) -> None:
        src = f"a{chr(codepoint)}b"
        assert sanitize_for_log(src) == src

    @pytest.mark.parametrize("codepoint", [0xE0100, 0xE0150, 0xE01EF])
    def test_supplementary_vs_preserved(self, codepoint: int) -> None:
        # VS17-VS256 above the BMP — CJK ideographic variation.
        src = f"a{chr(codepoint)}b"
        assert sanitize_for_log(src) == src

    def test_heart_emoji_with_vs16_preserved(self) -> None:
        # The canonical example: ``❤️`` = U+2764 + U+FE0F.
        heart = "❤️"
        assert sanitize_for_log(heart) == heart
        assert len(sanitize_for_log(heart)) == 2

    def test_vs15_text_presentation_preserved(self) -> None:
        # VS15 (U+FE0E) forces *text* (mono) presentation. Equally semantic.
        text_heart = "❤︎"
        assert sanitize_for_log(text_heart) == text_heart


# ---------------------------------------------------------------------------
# Tag characters U+E0000-U+E007F
# ---------------------------------------------------------------------------


class TestTagCharacters:
    """Tag characters are invisible ASCII shadows used in some
    Trojan-Source-style steganography attacks. The current sanitizer does
    NOT strip them; this is documented behavior — they are valid for
    locale subtags inside emoji sequences (Welsh / English / Scottish
    flag variants). Tests assert preservation explicitly so any future
    decision to strip them shows up as a deliberate test change.
    """

    @pytest.mark.parametrize(
        "codepoint",
        [0xE0000, 0xE0020, 0xE0041, 0xE007F],  # tag space, tag 'A', cancel tag
    )
    def test_tag_char_preserved(self, codepoint: int) -> None:
        src = f"a{chr(codepoint)}b"
        assert sanitize_for_log(src) == src

    def test_invisible_tag_payload_survives(self) -> None:
        # An invisible-payload variant of Trojan Source: the visible bytes
        # look innocent, but tag chars carry a hidden ASCII shadow.
        # Sanitizer does not strip tag chars — operator sees clean visible
        # output, but downstream byte-level checks (hash, length) reveal it.
        visible = "// looks safe"
        hidden = "".join(chr(0xE0000 + 0x20 + i) for i in range(5))  # tag "...."
        payload = visible + hidden
        assert sanitize_for_log(payload) == payload
        # Crucially, the length differs from what an operator would expect.
        assert len(sanitize_for_log(payload)) == len(visible) + 5


# ---------------------------------------------------------------------------
# Line / paragraph separators U+2028 / U+2029
# ---------------------------------------------------------------------------


class TestLineParaSeparators:
    """U+2028 LINE SEPARATOR and U+2029 PARAGRAPH SEPARATOR.

    These are real Unicode line terminators. Critically, JavaScript treats
    them as line terminators *inside string literals* (per ECMAScript), so
    a sanitized log message containing U+2028 that ends up inlined into a
    ``<script>`` tag breaks JS parsing — historically a JSON-in-script-tag
    injection vector. Python's ``json.dumps(..., ensure_ascii=False)``
    does NOT escape these; ``ensure_ascii=True`` (default) does.

    Current sanitizer behavior: passes them through. Documented here so
    callers shipping JSON to ``<script>`` know to pipeline an extra escape
    or to keep ``ensure_ascii=True``.
    """

    @pytest.mark.parametrize("codepoint", [0x2028, 0x2029])
    def test_separator_preserved_in_log(self, codepoint: int) -> None:
        src = f"a{chr(codepoint)}b"
        assert sanitize_for_log(src) == src

    def test_json_default_escapes_u2028(self) -> None:
        # Document the safe default: json.dumps WITHOUT ensure_ascii=False
        # escapes U+2028, which is what you want when emitting into <script>.
        out = json.dumps("a b")
        assert "\\u2028" in out

    def test_json_ensure_ascii_false_lets_u2028_through(self) -> None:
        # The dangerous mode: U+2028 survives as a literal byte sequence.
        # Callers using ensure_ascii=False on sanitized text and inlining
        # into <script> must additionally escape U+2028 / U+2029.
        out = json.dumps("a b", ensure_ascii=False)
        assert " " in out


# ---------------------------------------------------------------------------
# BOM (U+FEFF) at various positions
# ---------------------------------------------------------------------------


class TestBOM:
    """U+FEFF zero-width no-break space, used as a Byte Order Mark.

    Legitimate at the start of a file as encoding signal. Suspicious in
    the middle of a string (no functional purpose; sometimes used to make
    text comparisons silently fail). Current sanitizer preserves it
    everywhere — neither helping nor hurting. Tests pin the behavior.
    """

    @pytest.mark.parametrize(
        "src,desc",
        [
            ("﻿hello", "BOM at start (file-signature use)"),
            ("hel﻿lo", "BOM in middle (suspicious)"),
            ("hello﻿", "BOM at end"),
            ("﻿", "BOM only"),
            ("﻿﻿", "multiple BOMs"),
        ],
    )
    def test_bom_preserved_through_log(self, src: str, desc: str) -> None:
        assert sanitize_for_log(src) == src, desc

    @pytest.mark.parametrize(
        "src",
        ["﻿hello", "hel﻿lo", "hello﻿"],
    )
    def test_bom_preserved_through_json(self, src: str) -> None:
        assert sanitize_for_json(src) == src

    @pytest.mark.parametrize(
        "src",
        ["﻿hello", "hel﻿lo", "hello﻿"],
    )
    def test_bom_preserved_through_dbtext(self, src: str) -> None:
        assert sanitize_for_dbtext(src) == src

    def test_bom_stripped_from_identifier(self) -> None:
        # Negative: U+FEFF is not in the ASCII charset, identifier path
        # drops it. Prevents invisible cache-key collisions.
        assert clean_identifier("﻿model﻿-v1") == "model-v1"


# ---------------------------------------------------------------------------
# Soft hyphen U+00AD
# ---------------------------------------------------------------------------


class TestSoftHyphen:
    """U+00AD SOFT HYPHEN: invisible line-break hint.

    Legitimate in word-processor exports and certain CMS systems. Falls
    outside the C0 / C1 range so the sanitizer correctly preserves it.
    """

    def test_soft_hyphen_preserved_in_log(self) -> None:
        src = "long­word"
        assert sanitize_for_log(src) == src

    def test_soft_hyphen_stripped_from_identifier(self) -> None:
        # Negative: a soft hyphen inside a model name could mask collisions.
        assert clean_identifier("gpt­4") == "gpt4"


# ---------------------------------------------------------------------------
# Zalgo: stacked combining diacritics
# ---------------------------------------------------------------------------


class TestZalgoCombining:
    """Pathological but legitimate: base character + many combining marks.

    Combining diacritics live in U+0300-U+036F (and several extension
    blocks). None overlap C0/C1/BIDI/surrogate ranges, so they must
    pass through. A 50-mark stack tests both the regex and any naive
    string-length assumption downstream.
    """

    def test_fifty_combining_marks_preserved(self) -> None:
        # Build ``A`` plus 50 combining accents cycling through U+0300-U+0331.
        base = "A"
        marks = "".join(chr(0x0300 + i % 50) for i in range(50))
        src = base + marks
        assert sanitize_for_log(src) == src
        assert len(sanitize_for_log(src)) == 51

    def test_combining_only_string_preserved(self) -> None:
        # Combining marks with no base — odd but legitimate (will render
        # against a dotted circle). Must not be stripped.
        src = "".join(chr(0x0300 + i) for i in range(10))
        assert sanitize_for_log(src) == src

    def test_zalgo_through_dbtext(self) -> None:
        src = "A" + "".join(chr(0x0300 + i % 30) for i in range(30))
        assert sanitize_for_dbtext(src) == src


# ---------------------------------------------------------------------------
# CJK script families
# ---------------------------------------------------------------------------


class TestCJKFamilies:
    """All major CJK scripts must round-trip. LLM error output sometimes
    embeds Chinese / Japanese / Korean string literals from training data.
    """

    @pytest.mark.parametrize(
        "src,script",
        [
            ("汉字", "Han (Simplified)"),  # 汉字
            ("漢字", "Han (Traditional)"),  # 漢字
            ("ひらがな", "Hiragana"),  # ひらがな
            ("カタカナ", "Katakana"),  # カタカナ
            ("한글", "Hangul"),  # 한글
            ("ㄅㄆㄇ", "Bopomofo"),  # ㄅㄆㄇ
            ("んン", "Hiragana + Katakana N"),
        ],
    )
    def test_cjk_round_trip(self, src: str, script: str) -> None:
        assert sanitize_for_log(src) == src, script
        assert sanitize_for_json(src) == src, script
        assert sanitize_for_dbtext(src) == src, script

    def test_mixed_cjk_with_ascii_error_text(self) -> None:
        # Realistic: a Python traceback whose UserError message is CJK.
        src = "ValueError: 设置错误 in line 42"
        assert sanitize_for_log(src) == src


# ---------------------------------------------------------------------------
# Right-to-left scripts
# ---------------------------------------------------------------------------


class TestRTLScripts:
    """RTL strings must round-trip with no BIDI strip mangling.

    Important: the strip pattern catches U+202A-U+202E + U+2066-U+2069
    only. Arabic, Hebrew, Syriac letter ranges are far below this; the
    regex character class won't match them. This is the test that would
    fail loudly if the regex ever widened.
    """

    @pytest.mark.parametrize(
        "src,script",
        [
            ("مرحبا", "Arabic (مرحبا)"),
            ("שלום", "Hebrew (שלום)"),
            ("ܫܠܡܐ", "Syriac (ܫܠܡܐ)"),
            ("تجربة", "Arabic word (تجربة)"),
            ("בראשית", "Hebrew word (בראשית)"),
        ],
    )
    def test_rtl_preserved_through_log(self, src: str, script: str) -> None:
        assert sanitize_for_log(src) == src, script

    def test_rtl_with_legitimate_weak_bidi_mark(self) -> None:
        # Arabic text with embedded LRM (legitimate for numeric isolation
        # in mixed-direction phrases). Must survive intact.
        src = "عدد ‎42‎ جديد"
        assert sanitize_for_log(src) == src

    def test_rtl_with_strong_override_strips_only_override(self) -> None:
        # The override is removed; the Arabic letters are preserved.
        src = "مرحبا‮ attack"
        out = sanitize_for_log(src)
        assert "‮" not in out
        assert "مرحبا" in out


# ---------------------------------------------------------------------------
# Emoji ZWJ sequences and modifiers
# ---------------------------------------------------------------------------


class TestEmojiSequences:
    """Emoji are not just single code points. Family emoji uses ZWJ to
    glue heads / women / girls / boys; rainbow flag is white-flag + VS16
    + ZWJ + rainbow. Any over-broad strip (zero-width or VS16) silently
    breaks rendering. Tests ensure round-trip.
    """

    @pytest.mark.parametrize(
        "src,desc",
        [
            ("\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466", "family"),
            ("\U0001f3f3️‍\U0001f308", "rainbow flag"),
            ("\U0001f9d1‍\U0001f4bb", "technologist"),
            ("\U0001f469‍\U0001f527", "woman mechanic"),
            ("\U0001f441️‍\U0001f5e8️", "eye in speech bubble"),
        ],
    )
    def test_zwj_sequence_round_trip(self, src: str, desc: str) -> None:
        assert sanitize_for_log(src) == src, desc
        assert sanitize_for_json(src) == src, desc
        assert sanitize_for_dbtext(src) == src, desc

    @pytest.mark.parametrize(
        "src,desc",
        [
            ("\U0001f44b\U0001f3fb", "waving hand light"),  # Fitzpatrick 1-2
            ("\U0001f44b\U0001f3fc", "waving hand medium-light"),  # 3
            ("\U0001f44b\U0001f3fd", "waving hand medium"),  # 4
            ("\U0001f44b\U0001f3fe", "waving hand medium-dark"),  # 5
            ("\U0001f44b\U0001f3ff", "waving hand dark"),  # 6
        ],
    )
    def test_skin_tone_modifiers_preserved(self, src: str, desc: str) -> None:
        # Fitzpatrick modifiers U+1F3FB-U+1F3FF must survive intact.
        assert sanitize_for_log(src) == src, desc

    @pytest.mark.parametrize(
        "src,desc",
        [
            ("\U0001f1fa\U0001f1f8", "US flag"),
            ("\U0001f1ef\U0001f1f5", "JP flag"),
            ("\U0001f1e9\U0001f1ea", "DE flag"),
            ("\U0001f1f7\U0001f1fa", "RU flag"),
        ],
    )
    def test_regional_indicator_flags_preserved(self, src: str, desc: str) -> None:
        # Country flags are pairs of regional indicators U+1F1E6-U+1F1FF.
        # No control / BIDI / surrogate range overlap; must survive.
        assert sanitize_for_log(src) == src, desc

    def test_emoji_with_vs16_survives_json_round_trip(self) -> None:
        # Real-world end-to-end: emoji string with VS16 + ZWJ + skin tone
        # passes through sanitize_for_log and then json.dumps cleanly.
        src = (
            "Status: \U0001f3f3️‍\U0001f308 active for "
            "\U0001f469\U0001f3fd"
        )
        sanitized = sanitize_for_log(src)
        assert sanitized == src
        # Must JSON-encode (default ensure_ascii=True is fine; the bytes
        # become \uXXXX escapes but decode back to the original string).
        decoded = json.loads(json.dumps(sanitized))
        assert decoded == src


# ---------------------------------------------------------------------------
# Mathematical alphanumerics (above the BMP)
# ---------------------------------------------------------------------------


class TestMathAlphanumerics:
    """Math italic / bold / script / fraktur letters live in U+1D400-U+1D7FF.

    These are single supplementary-plane code points (not surrogate pairs
    in Python str). They are sometimes used to spoof identifiers in
    technical writing. Sanitize_for_log preserves them; clean_identifier
    strips them.
    """

    @pytest.mark.parametrize(
        "codepoint,desc",
        [
            (0x1D400, "MATH BOLD CAPITAL A"),
            (0x1D434, "MATH ITALIC CAPITAL A"),
            (0x1D49C, "MATH SCRIPT CAPITAL A"),
            (0x1D504, "MATH FRAKTUR CAPITAL A"),
            (0x1D538, "MATH DOUBLE-STRUCK CAPITAL A"),
            (0x1D7CE, "MATH BOLD DIGIT ZERO"),
        ],
    )
    def test_math_alphanumeric_preserved_in_log(
        self, codepoint: int, desc: str
    ) -> None:
        src = f"prefix_{chr(codepoint)}_suffix"
        assert sanitize_for_log(src) == src, desc

    def test_math_alphanumerics_string_round_trip(self) -> None:
        # 𝐀𝐁𝐂 — three bold math caps, plus emoji to confirm BMP/non-BMP mix.
        src = "\U0001d400\U0001d401\U0001d402 vs ABC"
        assert sanitize_for_log(src) == src

    def test_math_alphanumerics_stripped_from_identifier(self) -> None:
        # Negative: clean_identifier accepts ASCII only, so math-A is gone.
        assert clean_identifier("model\U0001d400-v2") == "model-v2"


# ---------------------------------------------------------------------------
# Halfwidth / fullwidth forms
# ---------------------------------------------------------------------------


class TestHalfFullwidth:
    """U+FF21 FULLWIDTH LATIN CAPITAL A renders like A but is a distinct
    code point used in East Asian typography. Preserved in log, stripped
    from identifier.
    """

    @pytest.mark.parametrize(
        "src",
        [
            "ＡＢＣ",  # ＡＢＣ
            "０１２",  # ０１２
            "ｈｅｌｌｏ",  # ｈｅｌｌｏ
        ],
    )
    def test_fullwidth_preserved_in_log(self, src: str) -> None:
        assert sanitize_for_log(src) == src

    @pytest.mark.parametrize(
        "src",
        ["ＡＢＣ", "０１２"],
    )
    def test_fullwidth_stripped_from_identifier(self, src: str) -> None:
        # Fullwidth digits / letters are not ASCII; identifier collapses.
        assert clean_identifier(src) == ""

    def test_halfwidth_katakana_preserved(self) -> None:
        # U+FF66-U+FF9F: halfwidth katakana, used in legacy JP encoding.
        src = "ｶﾀｶﾅ"  # ｶﾀｶﾅ
        assert sanitize_for_log(src) == src


# ---------------------------------------------------------------------------
# Non-Latin digit forms
# ---------------------------------------------------------------------------


class TestNonLatinDigits:
    """The identifier regex accepts ASCII [0-9] only. Arabic-Indic,
    Devanagari, fullwidth, and Eastern Arabic digits are all distinct
    code points and must be stripped from identifiers (preventing
    visually-identical but byte-distinct cache keys).
    """

    @pytest.mark.parametrize(
        "src,script",
        [
            ("٠١٢", "Arabic-Indic 012"),  # ٠١٢
            ("۰۱۲", "Extended Arabic-Indic 012"),  # ۰۱۲
            ("०१२", "Devanagari 012"),  # ०१२
            ("০১২", "Bengali 012"),  # ০১২
            ("๐๑๒", "Thai 012"),  # ๐๑๒
            ("０１２", "Fullwidth 012"),  # ０１２
        ],
    )
    def test_log_preserves_non_latin_digits(self, src: str, script: str) -> None:
        # Operator-facing log keeps them — useful for diagnosing locale bugs.
        assert sanitize_for_log(src) == src, script

    @pytest.mark.parametrize(
        "src",
        [
            "٠١٢",
            "۰۱۲",
            "०१२",
            "০১২",
            "๐๑๒",
            "０１２",
        ],
    )
    def test_identifier_strips_non_latin_digits(self, src: str) -> None:
        # Identifier path is ASCII-only — every non-Latin digit form
        # collapses to empty string.
        assert clean_identifier(src) == ""

    def test_mixed_ascii_and_non_latin_digits(self) -> None:
        # ASCII digits survive, others stripped.
        src = "v1٠١v2"  # v1٠١v2
        assert clean_identifier(src) == "v1v2"


# ---------------------------------------------------------------------------
# Adversarial composite inputs
# ---------------------------------------------------------------------------


class TestAdversarialComposites:
    """End-to-end attack-style strings exercising several axes at once."""

    def test_trojan_source_style_payload(self) -> None:
        # Visible-looking comment with bidi override flipping direction and
        # tag chars carrying invisible payload. sanitize_for_log strips
        # the override (BIDI); tag chars survive (documented behavior).
        # The visible bytes change from spoofed reversed-order to a
        # straight read, which is exactly what an operator wants.
        visible = "// safe"
        override = "‮"
        reversed_text = " moc.live"  # would render as "evil.com " under RLO
        tags = "".join(chr(0xE0000 + 0x20 + i) for i in range(3))
        payload = visible + override + reversed_text + tags
        out = sanitize_for_log(payload)
        # Override gone:
        assert "‮" not in out
        # Tag chars survive (documented):
        for cp in (0xE0020, 0xE0021, 0xE0022):
            assert chr(cp) in out
        # Visible parts both still there:
        assert visible in out
        assert reversed_text in out

    def test_thousand_char_base_with_hundred_zwjs(self) -> None:
        # A 1000-char base interleaved with 100 ZWJ characters. ZWJ is not
        # in any strip set, so the entire string must round-trip exactly.
        base = "x" * 1000
        # Insert ZWJ every 10 characters.
        parts = []
        for i in range(0, 1000, 10):
            parts.append(base[i : i + 10])
        joined = "‍".join(parts)
        assert joined.count("‍") == 99  # 100 segments => 99 joins
        # Boost to 100 ZWJs by prepending one:
        src = "‍" + joined
        assert src.count("‍") == 100
        out = sanitize_for_log(src)
        assert out == src
        # ZWJ count preserved exactly.
        assert out.count("‍") == 100

    def test_emoji_string_json_round_trip_after_sanitize(self) -> None:
        # Real-world: a status message with rich emoji passes sanitize_for_log
        # and survives json.dumps + json.loads byte-identically.
        src = (
            "Job \U0001f3f3️‍\U0001f308 done by "
            "\U0001f9d1\U0001f3fd‍\U0001f4bb in 1.23s"
        )
        sanitized = sanitize_for_log(src)
        assert sanitized == src
        assert json.loads(json.dumps(sanitized)) == src

    def test_kitchen_sink_unicode_passthrough(self) -> None:
        # Greek + CJK + RTL Arabic + emoji ZWJ + math alphanumeric +
        # combining + zero-width + BOM. Every byte must survive sanitize_for_log.
        src = (
            "﻿αβγ "  # Greek with leading BOM
            "汉字 "  # CJK
            "مرحبا "  # Arabic
            "\U0001f468‍\U0001f4bb "  # man technologist (ZWJ)
            "\U0001d400\U0001d401 "  # math bold
            "é "  # NFD é
            "x​y"  # ZWSP-joined
        )
        assert sanitize_for_log(src) == src

    def test_lone_surrogate_inside_otherwise_clean_emoji_string(self) -> None:
        # Mixed: valid emoji + a lone surrogate must replace only the lone
        # one, leaving the valid emoji untouched.
        src = "\U0001f600\ud83d\U0001f601"  # 😀 + lone high + 😁
        out = sanitize_for_log(src)
        assert "\U0001f600" in out and "\U0001f601" in out
        assert "\ud83d" not in out  # lone replaced with U+FFFD
        assert "�" in out


# ---------------------------------------------------------------------------
# Cross-mode consistency for Unicode-rich strings
# ---------------------------------------------------------------------------


class TestCrossModeUnicodeConsistency:
    """A string containing only legitimate Unicode (no controls, no BIDI
    overrides, no lone surrogates, no NUL) must come out byte-identical
    from every sanitizer. Catches any over-strip regression.
    """

    @pytest.mark.parametrize(
        "src",
        [
            "αβγ",  # Greek
            "汉字ひらがな한글",  # CJK families
            "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466",  # family
            "❤️",  # heart + VS16
            "\U0001f3f3️‍\U0001f308",  # rainbow flag
            "مرحبا",  # Arabic
            "שלום",  # Hebrew
            "﻿",  # BOM only
            "long­word",  # soft hyphen
            "é",  # NFD é
            "\U0001d400\U0001d401\U0001d402",  # math alphanumerics
            "A" + "́" * 20,  # combining stack
            "‎42‏",  # weak BIDI marks
            "٠١٢",  # Arabic-Indic digits
            "ＡＢＣ",  # fullwidth Latin
            "​‌‍⁠",  # zero-width set
        ],
    )
    def test_safe_unicode_unchanged_in_all_modes(self, src: str) -> None:
        assert sanitize_for_log(src) == src
        assert sanitize_for_json(src) == src
        assert sanitize_for_dbtext(src) == src

    @pytest.mark.parametrize(
        "src",
        [
            "αβγ",
            "❤️",
            "long­word",
            "é",
            "‎42‏",
        ],
    )
    def test_idempotent_on_safe_unicode(self, src: str) -> None:
        # Doubling sanitize_for_log on safe Unicode is identity twice over.
        once = sanitize_for_log(src)
        twice = sanitize_for_log(once)
        assert once == src and twice == src


# ---------------------------------------------------------------------------
# JSON / JSONL grammar safety
# ---------------------------------------------------------------------------


class TestJsonGrammarSafety:
    """Hostile log values that try to forge JSON / JSONL structure.

    The sanitizers do NOT escape JSON metacharacters (``{`` ``}`` ``[``
    ``]`` ``"`` ``\\`` ``,`` ``:``). That is correct: escaping is the
    JSON encoder's job. The contract this class verifies:

    1. Raw JSON grammar bytes pass through unchanged.
    2. After ``json.dumps`` wraps the sanitized value, structural
       injection attempts (``}{`` to fake a JSONL boundary, embedded
       ``"`` to break out of a string) become inert because they live
       inside a quoted JSON string with ``"`` and ``\\`` properly
       escaped.
    3. LF survives ``sanitize_for_log`` (multi-line tracebacks), but
       ``json.dumps`` escapes LF as the two-byte sequence ``\\n`` on the
       wire, so a forged JSONL record (``\\n{"forged":1}\\n``) cannot
       break NDJSON line splitting once the value is encoded properly.
    """

    @pytest.mark.parametrize(
        "ch",
        ["{", "}", "[", "]", '"', "\\", ",", ":"],
    )
    def test_json_metachar_preserved_in_log(self, ch: str) -> None:
        # Sanitize_for_log must not eat or escape JSON metacharacters —
        # otherwise legitimate JSON / Python repr / template syntax in
        # error messages would be corrupted.
        src = f"prefix{ch}suffix"
        assert sanitize_for_log(src) == src

    @pytest.mark.parametrize(
        "ch",
        ["{", "}", "[", "]", '"', "\\", ",", ":"],
    )
    def test_json_metachar_preserved_in_json_mode(self, ch: str) -> None:
        # The JSON-mode sanitizer is even more permissive: only lone
        # surrogates are replaced.
        src = f"prefix{ch}suffix"
        assert sanitize_for_json(src) == src

    @pytest.mark.parametrize(
        "ch",
        ["{", "}", "[", "]", '"', "\\", ",", ":"],
    )
    def test_json_metachar_preserved_in_dbtext(self, ch: str) -> None:
        src = f"prefix{ch}suffix"
        assert sanitize_for_dbtext(src) == src

    def test_brace_pair_round_trips_through_json_dumps(self) -> None:
        # The classic attack-shaped input: ``}{`` literally inside a
        # value. After sanitize_for_log it is unchanged; after json.dumps
        # it is wrapped in quotes; after json.loads it round-trips
        # byte-for-byte. The attempted forgery never reaches the wire.
        hostile = 'real }{ "forged":"value" }{ '
        sanitized = sanitize_for_log(hostile)
        assert sanitized == hostile  # nothing stripped
        encoded = json.dumps({"msg": sanitized})
        decoded = json.loads(encoded)
        assert decoded == {"msg": hostile}
        # Exactly one top-level key — no extra object created by forgery.
        assert list(decoded.keys()) == ["msg"]

    def test_embedded_quote_cannot_break_out(self) -> None:
        # Quote injection: ``"})({"`` would be catastrophic if not
        # escaped. sanitize_for_log leaves quotes alone; json.dumps
        # escapes them inside the string value. The full wire string
        # contains a single top-level object.
        hostile = '"})({"k":"v"}'
        sanitized = sanitize_for_log(hostile)
        wire = json.dumps({"payload": sanitized})
        loaded = json.loads(wire)
        assert set(loaded.keys()) == {"payload"}
        assert loaded["payload"] == hostile

    def test_backslash_escape_cannot_create_invalid_escape(self) -> None:
        # The six literal characters ``\\u202e`` typed into a log value
        # must not be confused with the actual U+202E code point. The
        # BIDI strip targets only the real code point; literal text
        # passes through. json.dumps then escapes the backslash itself.
        hostile = "msg: \\u202e attack"
        sanitized = sanitize_for_log(hostile)
        assert sanitized == hostile
        wire = json.dumps(sanitized)
        # In the wire, the single backslash must appear as two (escaped).
        assert "\\\\u202e" in wire
        assert json.loads(wire) == hostile

    def test_jsonl_line_split_is_safe_after_dumps(self) -> None:
        # A JSONL writer splits records on ``\\n``. An attacker injecting
        # raw LF + JSON object into a value should NOT inject a second
        # JSONL record. sanitize_for_log preserves LF (legit traceback
        # behaviour), but json.dumps escapes LF as the two-byte sequence
        # ``\\n`` on the wire — so the splitter still sees one record.
        hostile = 'real\n{"forged":"record"}\n'
        sanitized = sanitize_for_log(hostile)
        assert sanitized == hostile  # LF preserved by design
        wire = json.dumps({"line": sanitized})
        # No literal LF byte ever reaches the wire.
        assert wire.count("\n") == 0
        assert wire.splitlines() == [wire]
        assert json.loads(wire) == {"line": hostile}

    def test_jsonl_writer_full_pipeline(self) -> None:
        # Three hostile records — each containing brace-injection,
        # quote-injection, or LF-injection patterns — get JSONL-encoded
        # and split back into exactly three records.
        records = [
            {"msg": sanitize_for_log('attack }{ {"x":1} ')},
            {"msg": sanitize_for_log('quotes "inside" value')},
            {"msg": sanitize_for_log("multi\nline\nvalue")},
        ]
        wire = "\n".join(json.dumps(r) for r in records)
        lines = wire.split("\n")
        assert len(lines) == 3
        for line, original in zip(lines, records):
            assert json.loads(line) == original

    def test_sanitize_does_not_escape_quote_or_backslash(self) -> None:
        # Sharp contract: pre-escaping these bytes would cause double-
        # escaping once json.dumps runs (``"`` would become ``\\\\"``).
        src = 'a"b\\c'
        assert sanitize_for_log(src) == src
        assert sanitize_for_json(src) == src
        assert sanitize_for_dbtext(src) == src

    def test_json_round_trip_for_unicode_rich_hostile_payload(self) -> None:
        # All-axes payload: braces + quotes + LF + emoji ZWJ + RTL +
        # combining stack. Round-trips through sanitize_for_log + json.
        src = (
            'log }{ "user":"\U0001f468‍\U0001f4bb" '
            "مرحبا\n"
            "next-line é \\ end"
        )
        sanitized = sanitize_for_log(src)
        wire = json.dumps({"e": sanitized})
        assert json.loads(wire) == {"e": src}
        # Exactly one JSONL line on the wire.
        assert wire.count("\n") == 0

    def test_cr_escape_form_keeps_jsonl_single_line(self) -> None:
        # The sanitizer turns CR into the literal six-byte sequence
        # ``\\x0d``. That sequence contains no LF and no JSON metachar
        # with semantic effect — so the JSONL splitter sees one record,
        # and the wire visibly carries the escape form.
        hostile = "real\rINJECTED"
        sanitized = sanitize_for_log(hostile)
        assert sanitized == "real\\x0dINJECTED"
        wire = json.dumps({"v": sanitized})
        assert "real\\\\x0dINJECTED" in wire
        assert wire.count("\n") == 0

    def test_only_braces_string(self) -> None:
        # Degenerate edge case: a value that is *nothing but* JSON
        # metacharacters. Must survive sanitize_for_log untouched and
        # round-trip cleanly when wrapped.
        src = '{{}}[],":\\'
        assert sanitize_for_log(src) == src
        wire = json.dumps({"v": sanitize_for_log(src)})
        assert json.loads(wire) == {"v": src}
