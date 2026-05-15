"""Adversarial regex-level tests for gigaevo/utils/text_sanitize.py.

The base test file (``test_text_sanitize.py``) covers normal-case behavior.
This file targets the ANSI / surrogate / BIDI / C0 / C1 regexes themselves:
ambiguity between alternatives, malformed sequences, intermediate / private
bytes, missing terminators, surrogate adjacency, DoS resistance, and known
gaps (direct C1 introducers, bare ESC + Fp/Fs sequences).

Every test documents the observed behavior so that future regex tightening
can be detected as a behavior change.
"""

from __future__ import annotations

import json
import time

import pytest

from gigaevo.utils.text_sanitize import (
    sanitize_for_dbtext,
    sanitize_for_json,
    sanitize_for_log,
)

# Visible U+FFFD as a constant so test bodies stay readable.
REPL = "�"


# ---------------------------------------------------------------------------
# Malformed CSI: truncated / missing-final / invalid char in params
# ---------------------------------------------------------------------------


class TestMalformedCsi:
    """A CSI introducer that does not complete a valid sequence must NOT
    silently swallow following bytes. The current pattern requires a final
    byte in 0x40-0x7E; if absent, the ESC drops through to the C0 escape
    pass and surfaces as ``\\x1b`` followed by the literal residue.
    """

    def test_truncated_csi_bare_introducer(self) -> None:
        # ``ESC [`` with no params and no final: ESC escaped, ``[`` literal.
        assert sanitize_for_log("\x1b[") == "\\x1b["

    def test_truncated_csi_with_params_no_final(self) -> None:
        # ``ESC [ 3 1`` looks like the start of a colour sequence but stops
        # before the final SGR byte. Whole prefix must surface literally.
        assert sanitize_for_log("\x1b[31") == "\\x1b[31"

    def test_invalid_char_in_csi_params(self) -> None:
        # ``X`` (0x58) is outside the params 0x30-0x3F and intermediate
        # 0x20-0x2F ranges but inside the final 0x40-0x7E range. So the CSI
        # regex matches ``\x1b[1X`` with final ``X``, leaving ``m`` behind.
        # This is technically a malformed SGR but the regex absorbs it.
        assert sanitize_for_log("\x1b[1Xm") == "m"

    def test_partial_csi_does_cascade_when_joined_at_runtime(self) -> None:
        # Python string concatenation is a compile/runtime operation: by
        # the time the sanitizer sees its input, ``\x1b[`` + ``31m`` IS
        # the single string ``\x1b[31m`` — a valid CSI. There is no way
        # for the sanitizer to know the two halves arrived separately.
        # This test exists to fix the boundary in test author's mind:
        # all bypass attempts must be SINGLE-STRING constructions.
        assert sanitize_for_log("\x1b[" + "31m") == ""

    def test_partial_csi_with_non_final_terminator_does_not_cascade(self) -> None:
        # A genuinely truncated CSI: the next byte after the param is
        # space (0x20), which IS a valid intermediate byte. Pattern then
        # needs a final byte 0x40-0x7E. ``\n`` (0x0A) is not in that
        # range. So the CSI alt fails. Fe-single fails. ESC escapes,
        # bracket and digits survive, LF survives.
        assert sanitize_for_log("\x1b[31 \n") == "\\x1b[31 \n"


# ---------------------------------------------------------------------------
# CSI with intermediate bytes (real vendor sequences)
# ---------------------------------------------------------------------------


class TestCsiIntermediateBytes:
    @pytest.mark.parametrize(
        "sequence,name",
        [
            ("\x1b[1 q", "DECSCUSR cursor shape (space intermediate)"),
            ("\x1b[2 q", "DECSCUSR steady block"),
            ("\x1b[0$~", "DECSEL ($ intermediate, ~ final)"),
            ("\x1b[1\"q", "DECSCA select character protection"),
            ("\x1b[!p", "DECSTR soft terminal reset (! intermediate)"),
        ],
    )
    def test_intermediate_byte_csi_stripped(
        self, sequence: str, name: str
    ) -> None:
        assert sanitize_for_log(f"a{sequence}b") == "ab", name

    def test_multiple_intermediate_bytes(self) -> None:
        # Pattern allows ``[ -/]*`` — zero or more intermediates.
        assert sanitize_for_log("a\x1b[1 !\"#~b") == "ab"


# ---------------------------------------------------------------------------
# Private-parameter CSI (DEC private modes)
# ---------------------------------------------------------------------------


class TestPrivateParamCsi:
    @pytest.mark.parametrize(
        "sequence,name",
        [
            ("\x1b[?25h", "DECTCEM show cursor"),
            ("\x1b[?25l", "DECTCEM hide cursor"),
            ("\x1b[?2004h", "bracketed paste on"),
            ("\x1b[?2004l", "bracketed paste off"),
            ("\x1b[?1049h", "alternate screen buffer on"),
            ("\x1b[?1049l", "alternate screen buffer off"),
            ("\x1b[?1000;1006h", "mouse tracking with SGR"),
            ("\x1b[>4;2m", "modifyOtherKeys (> private prefix)"),
            ("\x1b[<3;10;20M", "SGR mouse report (< private prefix)"),
            ("\x1b[=1c", "send device attributes (= prefix)"),
        ],
    )
    def test_private_param_csi_stripped(self, sequence: str, name: str) -> None:
        # ``?``, ``<``, ``=``, ``>`` are all 0x3C-0x3F, inside the params
        # range 0x30-0x3F, so the regex picks them up.
        assert sanitize_for_log(f"x{sequence}y") == "xy", name


# ---------------------------------------------------------------------------
# OSC variants
# ---------------------------------------------------------------------------


class TestOscVariants:
    @pytest.mark.parametrize(
        "sequence,description",
        [
            ("\x1b]0;window title\x07", "xterm window+icon title (BEL term)"),
            ("\x1b]1;icon\x07", "xterm icon name (BEL term)"),
            ("\x1b]2;just title\x1b\\", "xterm window title (ST term)"),
            ("\x1b]4;1;rgb:ff/00/00\x07", "set ANSI palette"),
            ("\x1b]52;c;SGVsbG8=\x07", "OSC 52 clipboard write (security)"),
            ("\x1b]52;c;?\x07", "OSC 52 clipboard read query"),
            ("\x1b]133;A\x07", "FinalTerm OSC 133 prompt mark A"),
            ("\x1b]133;B\x07", "FinalTerm prompt mark B"),
            ("\x1b]133;C\x07", "FinalTerm command start"),
            ("\x1b]133;D;0\x07", "FinalTerm command end"),
            ("\x1b]7;file://host/path\x07", "iTerm2 current directory"),
            # OSC 8 hyperlink wraps visible text; this parametrization sets
            # the OSC to empty payload + empty payload so both vanish with
            # nothing between them. The visible-text variant is covered
            # explicitly below.
            ("\x1b]8;;\x07\x1b]8;;\x07", "OSC 8 hyperlink (no visible text)"),
            ("\x1b]10;rgb:ff/ff/ff\x1b\\", "set foreground"),
            ("\x1b]11;#000000\x07", "set background"),
        ],
    )
    def test_osc_payload_fully_consumed(
        self, sequence: str, description: str
    ) -> None:
        # The body must vanish, including security-sensitive payloads like
        # OSC 52 clipboard writes. Any leakage of the payload would be a
        # serious bug (an attacker could exfiltrate via clipboard).
        assert sanitize_for_log(f"pre{sequence}post") == "prepost", description

    def test_osc8_hyperlink_with_visible_text_preserves_text(self) -> None:
        # A real OSC 8 hyperlink: ESC]8;;URL BEL  visible-text  ESC]8;; BEL.
        # The two OSC sequences strip; the visible link text between them
        # is preserved (this is desirable — the human-readable label of a
        # hyperlink should survive).
        src = "pre\x1b]8;;https://example.com\x07link\x1b]8;;\x07post"
        assert sanitize_for_log(src) == "prelinkpost"

    def test_osc_with_embedded_surrogate_pair_in_body(self) -> None:
        # The OSC body regex is ``[^\x07\x1b]*`` — agnostic to Unicode
        # content. A valid emoji inside the body should still vanish with
        # the OSC.
        assert sanitize_for_log("a\x1b]title=😀\x07b") == "ab"

    def test_osc_no_terminator_caught_by_fe_fallback(self) -> None:
        # ``\x1b]forever`` with no BEL / ST. The OSC alternative cannot
        # match (no terminator). The Fe-single alternative ``\x1b[@-Z\\-_]``
        # DOES match ``\x1b]`` because ``]`` is 0x5D inside the ``\\-_``
        # subrange (0x5C-0x5F). The body becomes literal text.
        # DOCUMENTED BEHAVIOR: payload leaks through as plain text, but no
        # regex hang. Acceptable: payload is no longer interpretable as OSC
        # without its introducer.
        assert sanitize_for_log("\x1b]forever") == "forever"

    def test_osc_with_embedded_esc_breaks_into_two_matches(self) -> None:
        # ``\x1b]title\x1bP\x1b\\`` — the OSC body forbids ESC, so the OSC
        # alternative fails. Pattern matching at first ESC: Fe-single
        # matches ``\x1b]`` (the ``]`` byte). Pattern advances to second
        # ESC, where DCS matches ``\x1bP\x1b\\`` (empty payload). Result:
        # both ESCs and the bracket vanish, but the literal ``title``
        # between them surfaces as visible text.
        # DOCUMENTED BEHAVIOR: payload of a malformed OSC with embedded
        # ESC leaks; this is a real (but minor) confusion gap.
        assert sanitize_for_log("\x1b]title\x1bP\x1b\\rest") == "titlerest"


# ---------------------------------------------------------------------------
# DCS / SOS / PM / APC string sequences
# ---------------------------------------------------------------------------


class TestStringSequences:
    @pytest.mark.parametrize(
        "introducer,name",
        [
            ("P", "DCS"),
            ("X", "SOS"),
            ("^", "PM"),
            ("_", "APC"),
        ],
    )
    def test_each_introducer_stripped(self, introducer: str, name: str) -> None:
        src = f"a\x1b{introducer}payload\x1b\\b"
        assert sanitize_for_log(src) == "ab", name

    def test_dcs_with_nul_in_body(self) -> None:
        # The DCS body regex is ``[^\x1b]*`` — explicitly permits NUL
        # bytes. The whole DCS including the NUL inside vanishes.
        assert sanitize_for_log("a\x1bP1;0|head\x00tail\x1b\\b") == "ab"

    def test_dcs_with_c0_controls_in_body(self) -> None:
        # Body permits any non-ESC byte. The body itself is consumed by
        # the regex so the inner C0 controls never reach the C0 pass.
        assert sanitize_for_log("a\x1bP\x01\x02\x03\x1b\\b") == "ab"

    def test_apc_inside_dcs_body_breaks_dcs(self) -> None:
        # ``\x1bP outer \x1b_ inner \x1b\\ remainder``. The DCS body
        # regex stops at the inner ESC, so the DCS alternative fails at
        # the outer position. Fe-single then matches ``\x1bP`` (P is
        # 0x50, in 0x40-0x5F Fe range), advances. ``outer`` surfaces.
        # Then APC ``\x1b_inner\x1b\\`` matches. ``remainder`` surfaces.
        # DOCUMENTED BEHAVIOR: outer DCS payload leaks as plain text.
        assert (
            sanitize_for_log("\x1bPouter\x1b_inner\x1b\\remainder")
            == "outerremainder"
        )

    def test_dcs_inside_apc(self) -> None:
        # Symmetric case: APC body forbids ESC so DCS nested inside
        # similarly fragments the match.
        assert (
            sanitize_for_log("\x1b_outer\x1bPinner\x1b\\remainder")
            == "outerremainder"
        )


# ---------------------------------------------------------------------------
# Direct C1 introducers (no preceding ESC)
# ---------------------------------------------------------------------------


class TestDirectC1Introducers:
    """U+0080-U+009F are the C1 control range. U+009B is a single-byte
    CSI introducer equivalent to ``ESC [``, and U+009D is OSC. Python str
    can hold these bytes. The current ANSI regex requires a literal ESC
    prefix and therefore does NOT recognise direct C1 introducers as
    starting a sequence — they fall through to the C1 escape pass and
    surface as ``\\x9b`` / ``\\x9d`` literals followed by their payloads.

    DOCUMENTED GAP: any terminal that interprets C1 introducers would
    still see the payload as a control sequence. Mitigation in practice
    is that almost no producer emits raw C1; modern locales transmit
    ``ESC [`` instead. Severity LOW.
    """

    def test_direct_csi_introducer_u009b_not_swallowed(self) -> None:
        # ``\x9b31mred`` is conceptually equivalent to ``\x1b[31mred`` on
        # a terminal that supports 8-bit C1. The regex does not strip it.
        # The C1 pass escapes ``\x9b`` to ``\\x9b``, and ``31mred``
        # survives literally.
        assert sanitize_for_log("\x9b31mred") == "\\x9b31mred"

    def test_direct_osc_introducer_u009d_not_swallowed(self) -> None:
        # ``\x9d0;title\x07`` would be an OSC on an 8-bit terminal.
        # Currently: ``\x9d`` escaped, ``0;title`` survives, ``\x07`` BEL
        # escaped.
        assert sanitize_for_log("\x9d0;title\x07after") == "\\x9d0;title\\x07after"

    def test_direct_dcs_u0090_not_swallowed(self) -> None:
        # U+0090 is the 8-bit DCS introducer.
        assert (
            sanitize_for_log("\x90payload\x9c") == "\\x90payload\\x9c"
        )  # \x9c is ST

    @pytest.mark.parametrize(
        "byte",
        [0x80, 0x84, 0x88, 0x8D, 0x90, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F],
    )
    def test_all_c1_bytes_escaped_not_swallowed(self, byte: int) -> None:
        src = f"x{chr(byte)}y"
        assert sanitize_for_log(src) == f"x\\x{byte:02x}y"


# ---------------------------------------------------------------------------
# Bare ESC + Fp / Fs bytes (gaps in Fe-only single-char pattern)
# ---------------------------------------------------------------------------


class TestEscFpFsGaps:
    """The Fe single-char pattern matches only ESC + a byte in 0x40-0x5F
    (with ``[`` and ``]``/``P``/``X``/``^``/``_`` partially carved out by
    longer patterns earlier in the alternation, leaving the rest). It
    does NOT cover Fp bytes (0x30-0x3F: ``\\x1b7`` DECSC, ``\\x1b8`` DECRC,
    ``\\x1b=`` DECKPAM, ``\\x1b>`` DECKPNM) or Fs bytes (0x60-0x7E:
    ``\\x1bc`` RIS, ``\\x1bn`` LS2, ``\\x1b~`` LS1R).

    DOCUMENTED GAP: bare ESC drops to the C0 pass which escapes ESC to
    ``\\x1b`` but leaves the second character intact. The resulting
    string is no longer a control sequence — a terminal that reprints
    it would just show ``\\x1b7``. Severity LOW: visual noise, not
    exploitable.
    """

    @pytest.mark.parametrize(
        "code,name",
        [
            (0x37, "DECSC save cursor (ESC 7)"),
            (0x38, "DECRC restore cursor (ESC 8)"),
            (0x3D, "DECKPAM keypad application (ESC =)"),
            (0x3E, "DECKPNM keypad normal (ESC >)"),
        ],
    )
    def test_esc_fp_byte_not_stripped(self, code: int, name: str) -> None:
        src = f"a\x1b{chr(code)}b"
        # ESC escaped, second byte survives.
        assert sanitize_for_log(src) == f"a\\x1b{chr(code)}b", name

    @pytest.mark.parametrize(
        "code,name",
        [
            (0x63, "RIS full reset (ESC c)"),
            (0x6E, "LS2 locking shift G2 (ESC n)"),
            (0x6F, "LS3 locking shift G3 (ESC o)"),
            (0x7C, "LS3R (ESC |)"),
            (0x7D, "LS2R (ESC })"),
            (0x7E, "LS1R (ESC ~)"),
        ],
    )
    def test_esc_fs_byte_not_stripped(self, code: int, name: str) -> None:
        src = f"a\x1b{chr(code)}b"
        assert sanitize_for_log(src) == f"a\\x1b{chr(code)}b", name


# ---------------------------------------------------------------------------
# Multiple ESCs and ambiguity between alternatives
# ---------------------------------------------------------------------------


class TestMultipleAndAmbiguous:
    def test_two_adjacent_esc_then_csi(self) -> None:
        # ``\x1b\x1b\x1b[2J``. At pos 0: no alternative matches (next byte
        # is ESC, in none of the byte classes the alternatives expect at
        # pos+1). Fe-single requires 0x40-0x5F; ESC (0x1B) is not. So pos
        # 0 ESC is left alone and escaped by C0 pass. Same for pos 1.
        # Pos 2 starts a clean CSI which strips fully. Result: two
        # escaped ESC literals followed by nothing.
        assert sanitize_for_log("\x1b\x1b\x1b[2J") == "\\x1b\\x1b"

    def test_esc_then_bracket_then_text_then_csi(self) -> None:
        # ``\x1b[ ESC [ 0 m``. The first match attempt at pos 0: CSI
        # requires final byte in 0x40-0x7E. Looking forward: ``[`` then
        # ``\x1b`` (0x1B, NOT in final range), so first CSI attempt
        # extends greedily through params/intermediates and fails. CSI
        # falls through. OSC fails (introducer mismatch). DCS family
        # fails. Fe-single requires ESC + [@-Z\\-_]; ``[`` IS 0x5B in
        # 0x40-0x5F, but the carved-out ``[`` should still match Fe-single
        # because the character class ``[@-Z\\-_]`` includes 0x5B.
        # Wait: re-check pattern ``\x1b[@-Z\\-_]``. Range @-Z is 0x40-0x5A.
        # ``[`` is 0x5B which is OUTSIDE @-Z. The next subrange ``\\-_``
        # is 0x5C-0x5F. So 0x5B is NOT matched. Therefore Fe-single fails
        # at pos 0 too. ESC at pos 0 escaped. ``[`` literal. CSI starts at
        # pos 2 and consumes ``\x1b[0m``. Result: ``\\x1b[``.
        assert sanitize_for_log("\x1b[\x1b[0m") == "\\x1b["

    def test_csi_inside_what_looks_like_an_osc(self) -> None:
        # ``\x1b]title\x1b[31mred\x07``. OSC body cannot have ESC. The
        # OSC alternative fails. Fe-single matches ``\x1b]``. ``title``
        # surfaces. CSI matches ``\x1b[31m``. ``red`` surfaces. BEL is C0
        # and escapes to ``\\x07``.
        assert (
            sanitize_for_log("\x1b]title\x1b[31mred\x07")
            == "titlered\\x07"
        )

    def test_greedy_csi_does_not_eat_past_first_valid_final(self) -> None:
        # ``\x1b[1m\x1b[2m``: each CSI matches as a unit, second one
        # starts fresh. No greedy across.
        assert sanitize_for_log("\x1b[1m\x1b[2m") == ""


# ---------------------------------------------------------------------------
# Lone surrogate boundaries
# ---------------------------------------------------------------------------


class TestLoneSurrogateBoundaries:
    def test_lone_high_at_start(self) -> None:
        assert sanitize_for_log("\ud800rest") == f"{REPL}rest"

    def test_lone_high_at_end(self) -> None:
        assert sanitize_for_log("rest\ud800") == f"rest{REPL}"

    def test_lone_high_only_char(self) -> None:
        assert sanitize_for_log("\ud800") == REPL

    def test_lone_low_at_start(self) -> None:
        assert sanitize_for_log("\udc00rest") == f"{REPL}rest"

    def test_lone_low_at_end(self) -> None:
        assert sanitize_for_log("rest\udc00") == f"rest{REPL}"

    def test_lone_low_only_char(self) -> None:
        assert sanitize_for_log("\udc00") == REPL

    def test_two_adjacent_high(self) -> None:
        # ``\ud800\ud801``: first high is NOT followed by low -> lone.
        # Second high also not followed by low -> lone. Both replaced.
        assert sanitize_for_log("\ud800\ud801") == REPL + REPL

    def test_two_adjacent_low(self) -> None:
        # ``\udc00\udc01``: first low NOT preceded by high (lookbehind).
        # Second low IS preceded by low — and low is not in [\ud800-\udbff]
        # so lookbehind fails; second low also lone. Both replaced.
        assert sanitize_for_log("\udc00\udc01") == REPL + REPL

    def test_high_low_high_middle_pair_valid(self) -> None:
        # ``\ud800 \ud83d \udc00 \ud801``:
        #   pos 0 high: lookahead sees \ud83d (high), NOT in low range -> lone.
        #   pos 1 high (\ud83d): lookahead sees \udc00 (low) -> valid pair.
        #   pos 2 low: lookbehind sees \ud83d (high) -> valid pair.
        #   pos 3 high: lookahead sees end -> lone.
        # Result: REPL + valid-pair-as-emoji + REPL.
        src = "\ud800🐀\ud801"
        # 🐀 is U+1F400 RAT.
        assert sanitize_for_log(src) == f"{REPL}\U0001f400{REPL}"

    def test_high_high_low_first_high_lone(self) -> None:
        # ``\ud800 \ud83d \udc00``:
        #   pos 0 high: lookahead sees \ud83d (high, NOT low) -> lone.
        #   pos 1 high: lookahead sees \udc00 (low) -> valid pair, skip.
        #   pos 3: nothing.
        src = "\ud800🐀"
        assert sanitize_for_log(src) == f"{REPL}\U0001f400"

    def test_low_high_inverted_pair(self) -> None:
        # ``\udc00\ud800``: looks like inverted pair. Both are lone.
        #   pos 0 low: lookbehind sees nothing -> lone.
        #   pos 1 high: lookahead sees end -> lone.
        # Both replaced.
        src = "\udc00\ud800"
        assert sanitize_for_log(src) == REPL + REPL

    def test_consecutive_lone_surrogates_high_low_must_both_replace(self) -> None:
        # Regression guard. Two separate lone surrogates constructed via
        # chr(). Critically, this is NOT the astral codepoint U+10000: it is
        # two distinct str codepoints U+D800 and U+DC00 stored side by side.
        # The previous lookahead/lookbehind regex mistook them for a valid
        # UTF-16 pair and let them through, breaking UTF-8 encoding. The
        # current implementation strips every surrogate unconditionally.
        src = chr(0xD800) + chr(0xDC00)
        out = sanitize_for_log(src)
        # Whatever we accept as "fixed", the output MUST be UTF-8 encodable.
        out.encode("utf-8")
        # And it must not contain raw surrogates.
        for ch in out:
            cp = ord(ch)
            assert not (0xD800 <= cp <= 0xDFFF), (
                f"raw surrogate U+{cp:04X} survived"
            )

    def test_lone_high_then_valid_pair(self) -> None:
        # ``\ud800 then valid 😀``:
        #   pos 0 lone high (next is high).
        #   pos 1 high with low after -> pair.
        src = "\ud800😀"
        assert sanitize_for_log(src) == f"{REPL}\U0001f600"

    def test_valid_pair_then_lone_low(self) -> None:
        src = "😀\udc01"
        assert sanitize_for_log(src) == f"\U0001f600{REPL}"


# ---------------------------------------------------------------------------
# ANSI interleaved with multi-byte / ZWJ sequences
# ---------------------------------------------------------------------------


class TestAnsiInterleavedWithUnicode:
    def test_ansi_inside_zwj_family_reassembled(self) -> None:
        # ZWJ emoji family with ANSI colour codes interleaved. After ANSI
        # strip, the ZWJ sequence must be intact (man + ZWJ + woman + ZWJ
        # + girl + ZWJ + boy).
        src = "👨\x1b[31m‍👩\x1b[0m‍👧‍👦"
        expected = "👨‍👩‍👧‍👦"
        assert sanitize_for_log(src) == expected

    def test_ansi_between_cjk_chars(self) -> None:
        # The ANSI bytes are not UTF-8 continuation bytes, but the regex
        # operates on Python str (already decoded). Still worth verifying.
        assert sanitize_for_log("你\x1b[2J好") == "你好"

    def test_ansi_between_combining_characters(self) -> None:
        # ``e`` + combining acute (U+0301). ANSI between base and combiner
        # must not break the combine.
        src = "e\x1b[31ḿ"
        assert sanitize_for_log(src) == "é"

    def test_ansi_inside_skin_tone_modifier_sequence(self) -> None:
        # ``👋`` (waving hand) + U+1F3FB (light skin tone modifier).
        # ANSI between base emoji and modifier.
        src = "👋\x1b[1m\U0001f3fb"
        assert sanitize_for_log(src) == "👋\U0001f3fb"


# ---------------------------------------------------------------------------
# Performance: linear time on adversarial input
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_one_megabyte_ansi_strips_under_one_second(self) -> None:
        # 100 000 copies of ``\x1b[31m\x1b[0m`` ≈ 800 KB. Must complete
        # well under 1 s even on a slow CI runner; observed ≈10 ms on a
        # modern desktop. Budget is generous to avoid flakes.
        payload = "\x1b[31m\x1b[0m" * 100_000
        t0 = time.perf_counter()
        out = sanitize_for_log(payload)
        dt = time.perf_counter() - t0
        assert out == ""
        assert dt < 1.0, f"sanitizer took {dt:.3f}s on 1 MB ANSI input"

    def test_long_unterminated_osc_no_backtracking_hang(self) -> None:
        # A megabyte of OSC body with no terminator. The OSC alternative
        # cannot complete; Fe-single fires on ``\x1b]`` and the body
        # passes through as plain text. Important: this must finish in
        # linear time despite the unterminated body.
        payload = "\x1b]" + "a" * 1_000_000
        t0 = time.perf_counter()
        out = sanitize_for_log(payload)
        dt = time.perf_counter() - t0
        assert out == "a" * 1_000_000
        assert dt < 1.0, f"OSC fallback took {dt:.3f}s"

    def test_alternating_partial_csi_no_pathological_backtrack(self) -> None:
        # Many almost-CSI fragments that all fail at the final byte. Each
        # falls back to C0-escape independently — should remain linear.
        payload = "\x1b[31" * 10_000  # 40 000 chars
        t0 = time.perf_counter()
        out = sanitize_for_log(payload)
        dt = time.perf_counter() - t0
        assert "\x1b" not in out  # all ESCs escaped
        assert dt < 1.0, f"partial-CSI flood took {dt:.3f}s"


# ---------------------------------------------------------------------------
# CSI final-byte exhaustive coverage
# ---------------------------------------------------------------------------


# All bytes in 0x40-0x7E that are legal CSI finals. The pattern accepts the
# whole range without carve-out, so every one of these must terminate the
# CSI cleanly.
_CSI_FINAL_BYTES = [chr(c) for c in range(0x40, 0x7F)]


class TestCsiFinalByteExhaustive:
    @pytest.mark.parametrize("final", _CSI_FINAL_BYTES)
    def test_every_final_byte_terminates_csi(self, final: str) -> None:
        src = f"pre\x1b[1{final}post"
        assert sanitize_for_log(src) == "prepost", (
            f"CSI with final 0x{ord(final):02x} ({final!r}) not stripped"
        )

    @pytest.mark.parametrize("final", _CSI_FINAL_BYTES)
    def test_every_final_byte_empty_params(self, final: str) -> None:
        # No params at all — also valid CSI.
        src = f"pre\x1b[{final}post"
        assert sanitize_for_log(src) == "prepost", (
            f"empty-param CSI with final {final!r} not stripped"
        )

    def test_empty_csi_reset(self) -> None:
        # ``\x1b[m`` is SGR reset with no params, common in compiler output.
        assert sanitize_for_log("\x1b[m") == ""

    def test_many_param_csi(self) -> None:
        # 15-param SGR. Must strip in full.
        params = ";".join(str(i) for i in range(1, 16))
        assert sanitize_for_log(f"\x1b[{params}m") == ""


# ---------------------------------------------------------------------------
# BIDI position coverage
# ---------------------------------------------------------------------------


class TestBidiPositions:
    @pytest.mark.parametrize(
        "codepoint",
        [0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069],
    )
    @pytest.mark.parametrize(
        "position",
        ["start", "middle", "end", "only"],
    )
    def test_bidi_at_position(self, codepoint: int, position: str) -> None:
        ch = chr(codepoint)
        if position == "start":
            src, expected = ch + "rest", "rest"
        elif position == "middle":
            src, expected = "pre" + ch + "post", "prepost"
        elif position == "end":
            src, expected = "rest" + ch, "rest"
        else:
            src, expected = ch, ""
        assert sanitize_for_log(src) == expected


# ---------------------------------------------------------------------------
# Mixed C0 / C1 / ANSI / surrogate / BIDI in one string
# ---------------------------------------------------------------------------


class TestMixedClasses:
    def test_one_of_each_in_order(self) -> None:
        # ANSI strip first, then BIDI, then C0 escape, then C1 escape,
        # then lone surrogate replace. Order matters: BIDI runs before
        # C0/C1 escape, so the BIDI char is gone before being seen as a
        # non-control codepoint. ANSI runs first so its inner bytes never
        # reach later passes.
        src = "a\x1b[31m‮b\x00c\x9dd\ud800e"
        # \x1b[31m -> gone
        # ‮ BIDI RLO -> gone
        # \x00 NUL -> \x00 escaped
        # \x9d C1 -> \x9d escaped
        # \ud800 lone high -> REPL
        expected = "ab\\x00c\\x9dd" + REPL + "e"
        assert sanitize_for_log(src) == expected

    def test_ansi_wrapping_c0_does_not_leak(self) -> None:
        # ANSI body contains a C0 control. The CSI regex matches the
        # whole ANSI sequence atomically, so the inner C0 vanishes with
        # it — it does NOT reach the C0 pass.
        # ``\x1b[?\x081h`` — but \x08 is in C0, will it stop the CSI?
        # Params class is [0-?] (0x30-0x3F). \x08 is below that range.
        # So CSI greedy match stops: regex matches ``\x1b[?`` then needs
        # final byte but \x08 is below 0x40. Match fails at this
        # position. Fall back to Fe-single: ``\x1b[`` — ``[`` is 0x5B,
        # NOT in [@-Z\\-_]. Fail. ESC escapes. ``[?`` survives. \x08
        # escapes. ``1h`` survives.
        assert sanitize_for_log("\x1b[?\x081h") == "\\x1b[?\\x081h"


# ---------------------------------------------------------------------------
# Replacement char and literal escape form idempotence
# ---------------------------------------------------------------------------


class TestIdempotenceArtifacts:
    def test_existing_replacement_char_passes_through(self) -> None:
        # Idempotence requires that an input already containing U+FFFD
        # survives untouched. If the regex were to re-match U+FFFD as a
        # surrogate, sanitize would not be idempotent.
        assert sanitize_for_log(f"a{REPL}b") == f"a{REPL}b"

    def test_multiple_replacement_chars_pass_through(self) -> None:
        assert sanitize_for_log(REPL * 10) == REPL * 10

    def test_literal_backslash_x1b_string_passes_through(self) -> None:
        # The four-character literal ``\\x1b`` (backslash, x, 1, b) is
        # NOT a real ESC byte. It must survive unchanged.
        assert sanitize_for_log("error: \\x1b decoded") == "error: \\x1b decoded"

    def test_full_idempotence_on_complex_hostile(self) -> None:
        # Sanitizing twice yields the same string as sanitizing once.
        hostile = (
            "\x1b[31m" "head"
            "\x00mid\r"
            "‮tail"
            "\ud800"
            "\x9b"
            "\x1b]0;t\x07"
        )
        once = sanitize_for_log(hostile)
        twice = sanitize_for_log(once)
        assert once == twice

    def test_idempotence_random_axes(self) -> None:
        # A broader hostile: every class represented twice over.
        hostile = (
            "α"  # Greek (preserve)
            "\x1b[1;31m" "X" "\x1b[0m"  # ANSI
            "‭" "Y" "‬"  # BIDI
            "\x01\x02"  # C0
            "\x80\x9f"  # C1
            "😀"  # valid pair
            "\ud800"  # lone high
            "\udc00"  # lone low
            "你好"  # CJK
            "🇺🇸"  # regional indicator pair (two astral chars)
        )
        once = sanitize_for_log(hostile)
        twice = sanitize_for_log(once)
        thrice = sanitize_for_log(twice)
        assert once == twice == thrice


# ---------------------------------------------------------------------------
# Cross-function: composition still safe under adversarial input
# ---------------------------------------------------------------------------


class TestCompositionSafety:
    def test_log_output_is_json_safe_under_adversarial_input(self) -> None:
        # Every adversarial axis at once. NOTE: lone surrogates are
        # placed with non-surrogate neighbours to avoid triggering the
        # documented ``𐀀`` lookaround-confusion bug (covered
        # separately as xfail). Once that bug is fixed, ``hostile`` can
        # be tightened to include adjacent lone surrogates.
        hostile = (
            "\x1b[31m"
            "\x1b]52;c;EXFIL\x07"
            "\x1bPdcs\x1b\\"
            "\x9b31m"
            "\x1b7"
            "\x00\x07\r"
            "‮"
            "\ud800X"  # lone high, then ASCII (no false pair)
            "Y\udc00"  # ASCII, then lone low (no false pair)
            "\ud83dZ"  # lone high, then ASCII
            "text"
        )
        out = sanitize_for_log(hostile)
        # Must encode to UTF-8.
        out.encode("utf-8")
        # Must serialise as JSON.
        json.dumps(out)
        # No raw ESC byte must survive sanitize_for_log.
        assert "\x1b" not in out
        # No raw C0 except TAB / LF.
        for ch in out:
            cp = ord(ch)
            assert cp in (0x09, 0x0A) or cp >= 0x20
        # No lone surrogate.
        for ch in out:
            cp = ord(ch)
            assert not 0xD800 <= cp <= 0xDFFF
        # No BIDI override.
        for ch in out:
            cp = ord(ch)
            assert not (0x202A <= cp <= 0x202E or 0x2066 <= cp <= 0x2069)

    def test_dbtext_then_log_idempotent(self) -> None:
        hostile = "a\x00\x1b[2J\ud800b"
        once = sanitize_for_log(sanitize_for_dbtext(hostile))
        twice = sanitize_for_log(sanitize_for_dbtext(once))
        assert once == twice

    def test_json_then_log_idempotent(self) -> None:
        hostile = "a\x1b[2J\ud800\x00b"
        once = sanitize_for_log(sanitize_for_json(hostile))
        twice = sanitize_for_log(sanitize_for_json(once))
        assert once == twice
