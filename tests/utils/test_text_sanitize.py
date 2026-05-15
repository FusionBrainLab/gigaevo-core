"""Tests for gigaevo/utils/text_sanitize.py.

Coverage axes:
  * sanitize_for_log: ANSI families, C0, C1, BIDI, lone surrogates,
    composition, idempotence, identity on safe input.
  * sanitize_for_json: minimal lone-surrogate replacement, identity on
    everything else.
  * sanitize_for_dbtext: NUL replacement, identity on everything else.
  * clean_identifier: charset strip, max_len, empty input.
  * multi-language preservation: Greek (Mojo identifiers), Unicode arrows
    (Mojo / Pallas formatters), math symbols, CJK, emoji, box-drawing,
    CUTLASS-style template syntax.
  * composability and idempotence guarantees.
"""

from __future__ import annotations

import pytest

from gigaevo.utils.text_sanitize import (
    clean_identifier,
    deep_sanitize_for_json,
    sanitize_for_dbtext,
    sanitize_for_json,
    sanitize_for_log,
)

# ---------------------------------------------------------------------------
# sanitize_for_log — ANSI escape sequences
# ---------------------------------------------------------------------------


class TestSanitizeForLogAnsi:
    @pytest.mark.parametrize(
        "src,expected",
        [
            ("\x1b[2J\x1b[H", ""),  # clear screen + home
            ("\x1b[31mred\x1b[0m text", "red text"),  # SGR colorization
            ("\x1b[1m\x1b[31merror\x1b[0m", "error"),  # bold red
            ("plain", "plain"),  # identity
            ("\x1b[?25h", ""),  # private mode (cursor show)
            ("\x1b]0;window title\x07after", "after"),  # OSC + BEL terminator
            ("\x1b]2;title\x1b\\after", "after"),  # OSC + ST terminator
            ("\x1b[1;31;42mtext\x1b[0m", "text"),  # multi-param CSI
        ],
    )
    def test_csi_and_osc_stripped(self, src: str, expected: str) -> None:
        assert sanitize_for_log(src) == expected

    def test_single_char_fe_escape_stripped(self) -> None:
        # ESC followed by a single Fe byte (e.g. ESC M reverse-index).
        assert sanitize_for_log("a\x1bMb") == "ab"

    def test_compiler_style_error_block(self) -> None:
        # Mimics a gcc / clang / nvcc colorized error line.
        src = "\x1b[1m\x1b[31merror:\x1b[0m\x1b[1m undefined reference\x1b[0m"
        assert sanitize_for_log(src) == "error: undefined reference"

    def test_dcs_sequence_stripped(self) -> None:
        # DCS (Device Control String): ESC P ... ST. Used by e.g. terminal
        # sixel image transfer; rare but legal in stderr.
        assert sanitize_for_log("a\x1bPpayload\x1b\\b") == "ab"

    def test_apc_sequence_stripped(self) -> None:
        # APC (Application Program Command): ESC _ ... ST.
        assert sanitize_for_log("a\x1b_data\x1b\\b") == "ab"

    def test_csi_with_intermediate_byte_stripped(self) -> None:
        # CSI with intermediate byte: ESC [ params SP final. ``\x1b[1 q``
        # is a cursor-shape control with space as intermediate.
        assert sanitize_for_log("a\x1b[1 qb") == "ab"

    def test_consecutive_ansi_sequences_all_stripped(self) -> None:
        src = "\x1b[31m\x1b[1m\x1b[4mtext\x1b[0m\x1b[0m\x1b[0m"
        assert sanitize_for_log(src) == "text"


# ---------------------------------------------------------------------------
# sanitize_for_log — C0 and C1 controls
# ---------------------------------------------------------------------------


class TestSanitizeForLogControlChars:
    def test_nul_escaped(self) -> None:
        assert sanitize_for_log("a\x00b") == "a\\x00b"

    def test_bel_escaped(self) -> None:
        assert sanitize_for_log("a\x07b") == "a\\x07b"

    def test_backspace_escaped(self) -> None:
        assert sanitize_for_log("a\x08b") == "a\\x08b"

    def test_cr_escaped_so_no_line_forgery(self) -> None:
        # The critical case: a forged log entry attempt becomes inert.
        assert sanitize_for_log("real\r\nFORGED") == "real\\x0d\nFORGED"

    def test_tab_preserved(self) -> None:
        assert sanitize_for_log("a\tb") == "a\tb"

    def test_lf_preserved(self) -> None:
        # Real multi-line tracebacks must survive.
        assert sanitize_for_log("line1\nline2") == "line1\nline2"

    def test_del_escaped(self) -> None:
        assert sanitize_for_log("a\x7fb") == "a\\x7fb"

    @pytest.mark.parametrize("byte", [0x80, 0x9F, 0x85])
    def test_c1_controls_escaped(self, byte: int) -> None:
        src = f"a{chr(byte)}b"
        assert sanitize_for_log(src) == f"a\\x{byte:02x}b"


# ---------------------------------------------------------------------------
# sanitize_for_log — BIDI overrides
# ---------------------------------------------------------------------------


class TestSanitizeForLogBidi:
    @pytest.mark.parametrize(
        "codepoint",
        [
            0x202A,  # LRE
            0x202B,  # RLE
            0x202C,  # PDF
            0x202D,  # LRO
            0x202E,  # RLO
            0x2066,  # LRI
            0x2067,  # RLI
            0x2068,  # FSI
            0x2069,  # PDI
        ],
    )
    def test_each_bidi_override_stripped(self, codepoint: int) -> None:
        src = f"a{chr(codepoint)}b"
        assert sanitize_for_log(src) == "ab"


# ---------------------------------------------------------------------------
# sanitize_for_log — lone surrogates
# ---------------------------------------------------------------------------


class TestSanitizeForLogSurrogates:
    def test_lone_high_surrogate_replaced(self) -> None:
        assert sanitize_for_log("a\ud83dz") == "a�z"

    def test_lone_low_surrogate_replaced(self) -> None:
        assert sanitize_for_log("a\udc00z") == "a�z"

    def test_multiple_lone_surrogates_replaced(self) -> None:
        assert sanitize_for_log("\ud800𐀀\ud801") == "�𐀀�"

    def test_valid_surrogate_pair_preserved(self) -> None:
        # A paired high+low surrogate represents a single code point above the
        # BMP. Python str doesn't typically use this form, but if constructed
        # by hand it must be preserved as a valid pair.
        src = "a😀z"  # paired -> U+1F600 emoji
        assert sanitize_for_log(src) == src

    def test_real_astral_emoji_preserved(self) -> None:
        # The natural single-code-point form must of course pass through.
        assert sanitize_for_log("a😀z") == "a😀z"


# ---------------------------------------------------------------------------
# sanitize_for_log — multi-language preservation
# ---------------------------------------------------------------------------


class TestSanitizeForLogPreservation:
    def test_greek_letters_preserved(self) -> None:
        # Mojo permits Greek identifiers; classifier and logs must preserve.
        assert sanitize_for_log("αβγ_kernel") == "αβγ_kernel"

    def test_unicode_arrows_preserved(self) -> None:
        # Mojo / Pallas error formatters use U+2192 / U+21D2 to highlight
        # source positions.
        assert sanitize_for_log("a → b ⇒ c") == "a → b ⇒ c"

    def test_math_symbols_preserved(self) -> None:
        assert sanitize_for_log("∀x ∈ ℝ → ℂ") == "∀x ∈ ℝ → ℂ"

    def test_cjk_preserved(self) -> None:
        assert sanitize_for_log("配置错误") == "配置错误"

    def test_emoji_preserved(self) -> None:
        assert sanitize_for_log("done ✅ 🎉") == "done ✅ 🎉"

    def test_box_drawing_preserved(self) -> None:
        # clang / rustc carets use box-drawing for source-position pointers.
        assert sanitize_for_log("│  ╰─ here") == "│  ╰─ here"

    def test_cutlass_template_syntax_preserved(self) -> None:
        # CUTLASS / CuTe error messages routinely contain dense template
        # syntax that must survive verbatim.
        src = "Layout<Shape<_32,_128>,Stride<_128,_1>>"
        assert sanitize_for_log(src) == src

    def test_compile_error_block_with_mixed_content(self) -> None:
        # Realistic: ANSI from compiler colorization wrapping a source line
        # that itself contains template syntax, Greek identifier, arrow.
        src = "\x1b[31merror\x1b[0m: cannot convert α → β in Layout<Shape<_32>>"
        assert (
            sanitize_for_log(src) == "error: cannot convert α → β in Layout<Shape<_32>>"
        )


# ---------------------------------------------------------------------------
# sanitize_for_log — invariants
# ---------------------------------------------------------------------------


class TestSanitizeForLogInvariants:
    def test_identity_on_safe_input(self) -> None:
        safe = "Plain ASCII with\ttabs and\nnewlines."
        assert sanitize_for_log(safe) == safe

    def test_idempotence_on_safe_input(self) -> None:
        safe = "plain text"
        assert sanitize_for_log(sanitize_for_log(safe)) == safe

    def test_idempotence_on_hostile_input(self) -> None:
        hostile = "a\x1b[2J\x07‮b\ud83dc\x00"
        once = sanitize_for_log(hostile)
        twice = sanitize_for_log(once)
        assert twice == once

    def test_empty_input(self) -> None:
        assert sanitize_for_log("") == ""

    def test_only_controls_collapses(self) -> None:
        # All-controls input becomes a string of escape forms (still safe).
        result = sanitize_for_log("\x00\x01\x02\x03")
        assert result == "\\x00\\x01\\x02\\x03"

    def test_output_is_utf8_encodable(self) -> None:
        # After sanitization the result must round-trip through UTF-8.
        hostile = "a\ud83d\x00\x1b[2J‮b"
        cleaned = sanitize_for_log(hostile)
        cleaned.encode("utf-8")  # must not raise

    def test_output_is_json_encodable(self) -> None:
        import json

        hostile = "a\ud83d\x00\x1b[2J‮b"
        json.dumps(sanitize_for_log(hostile))  # must not raise

    def test_output_contains_no_escape_byte(self) -> None:
        # Byte-level invariant: no raw \x1b ever survives sanitize_for_log.
        # Catches regex regressions that JSON / UTF-8 checks would not.
        hostile = "\x1b[2J\x1bM\x1b]title\x07\x1b_apc\x1b\\plain"
        assert "\x1b" not in sanitize_for_log(hostile)

    def test_output_contains_no_lone_surrogate(self) -> None:
        hostile = "a\ud800b\udc00c\ud83d"
        cleaned = sanitize_for_log(hostile)
        for ch in cleaned:
            cp = ord(ch)
            assert not (0xD800 <= cp <= 0xDFFF), (
                f"surrogate U+{cp:04X} survived"
            )

    def test_output_contains_no_raw_c0_except_tab_lf(self) -> None:
        hostile = "".join(chr(c) for c in range(0x20)) + "\x7f"
        cleaned = sanitize_for_log(hostile)
        for ch in cleaned:
            cp = ord(ch)
            assert cp in (0x09, 0x0A) or cp >= 0x20, (
                f"raw C0 char U+{cp:04X} survived"
            )
        # And no 0x7F either.
        assert "\x7f" not in cleaned

    def test_output_contains_no_bidi_overrides(self) -> None:
        hostile = "a‪b‫c‬d‭e‮f⁦g⁧h⁨i⁩"
        cleaned = sanitize_for_log(hostile)
        for ch in cleaned:
            cp = ord(ch)
            assert not (0x202A <= cp <= 0x202E), f"BIDI U+{cp:04X} survived"
            assert not (0x2066 <= cp <= 0x2069), f"BIDI U+{cp:04X} survived"


# ---------------------------------------------------------------------------
# sanitize_for_json
# ---------------------------------------------------------------------------


class TestSanitizeForJson:
    def test_lone_surrogate_replaced(self) -> None:
        assert sanitize_for_json("a\ud83dz") == "a�z"

    def test_paired_surrogates_preserved(self) -> None:
        assert sanitize_for_json("a😀z") == "a😀z"

    def test_ansi_passes_through(self) -> None:
        # JSON encoder handles ANSI fine; this function does not strip it.
        assert sanitize_for_json("\x1b[2Jhello") == "\x1b[2Jhello"

    def test_controls_pass_through(self) -> None:
        # NUL / BEL / CR survive — sanitize_for_dbtext or _log are stricter.
        assert sanitize_for_json("a\x00\x07\rb") == "a\x00\x07\rb"

    def test_bidi_passes_through(self) -> None:
        assert sanitize_for_json("a‮b") == "a‮b"

    def test_idempotence(self) -> None:
        hostile = "a\ud83dz"
        assert sanitize_for_json(sanitize_for_json(hostile)) == sanitize_for_json(
            hostile
        )

    def test_json_encodes_after_sanitize(self) -> None:
        import json

        hostile = "msg with \ud83d lone"
        json.dumps(sanitize_for_json(hostile))  # must not raise

    def test_identity_on_safe_input(self) -> None:
        safe = "completely normal text 😀 αβγ"
        assert sanitize_for_json(safe) == safe


# ---------------------------------------------------------------------------
# sanitize_for_dbtext
# ---------------------------------------------------------------------------


class TestSanitizeForDbtext:
    def test_nul_replaced(self) -> None:
        assert sanitize_for_dbtext("a\x00b") == "a�b"

    def test_multiple_nul_replaced(self) -> None:
        assert sanitize_for_dbtext("\x00\x00x\x00") == "��x�"

    def test_no_nul_identity(self) -> None:
        assert sanitize_for_dbtext("plain") == "plain"

    def test_ansi_passes_through(self) -> None:
        # The dbtext variant is intentionally minimal; ANSI passes.
        assert sanitize_for_dbtext("\x1b[2Jhello") == "\x1b[2Jhello"

    def test_other_controls_pass_through(self) -> None:
        assert sanitize_for_dbtext("a\x07\r\nb") == "a\x07\r\nb"

    def test_unicode_preserved(self) -> None:
        assert sanitize_for_dbtext("a😀b") == "a😀b"

    def test_idempotence(self) -> None:
        src = "a\x00b\x00c"
        assert sanitize_for_dbtext(sanitize_for_dbtext(src)) == sanitize_for_dbtext(src)

    def test_lone_surrogate_replaced(self) -> None:
        # asyncpg UTF-8 encodes; lone surrogates fail there even after NUL
        # handling. dbtext variant must cover both.
        assert sanitize_for_dbtext("a\ud83dz") == "a�z"

    def test_combined_nul_and_surrogate(self) -> None:
        assert sanitize_for_dbtext("a\x00b\ud83dc") == "a�b�c"

    def test_paired_surrogates_preserved(self) -> None:
        # Valid emoji must not be mangled by the surrogate fix.
        assert sanitize_for_dbtext("a😀b") == "a😀b"

    def test_output_is_utf8_encodable(self) -> None:
        # The whole point: after dbtext, the result encodes cleanly.
        hostile = "a\x00\ud83db\udc00c"
        sanitize_for_dbtext(hostile).encode("utf-8")  # must not raise


# ---------------------------------------------------------------------------
# Composability
# ---------------------------------------------------------------------------


class TestComposability:
    def test_dbtext_then_log_handles_everything(self) -> None:
        # Pipelined: first replace NUL, then strip ANSI / BIDI / controls /
        # surrogates. End result is safe for log, JSON, and DB.
        src = "a\x00\x1b[2J‮\ud83db\r"
        intermediate = sanitize_for_dbtext(src)
        final = sanitize_for_log(intermediate)
        assert "\x00" not in final
        assert "\x1b" not in final
        assert "‮" not in final
        assert "\r" not in final
        # Validate output safety.
        final.encode("utf-8")
        import json

        json.dumps(final)

    def test_log_alone_covers_db_safety(self) -> None:
        # sanitize_for_log strips C0 (incl NUL) — the escaped \\x00 string
        # is safe for Postgres TEXT.
        result = sanitize_for_log("a\x00b")
        assert "\x00" not in result


# ---------------------------------------------------------------------------
# clean_identifier
# ---------------------------------------------------------------------------


class TestCleanIdentifier:
    @pytest.mark.parametrize(
        "src,expected",
        [
            ("gpt-4o-mini", "gpt-4o-mini"),
            ("claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"),
            ("meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"),
            ("openai:gpt-4o", "openai:gpt-4o"),
            ("model@my-host:8080", "model@my-host:8080"),
            ("model_v2+experimental", "model_v2+experimental"),
            ("/local/path/to/model.gguf", "/local/path/to/model.gguf"),
        ],
    )
    def test_safe_identifiers_pass_through(self, src: str, expected: str) -> None:
        assert clean_identifier(src) == expected

    @pytest.mark.parametrize(
        "src,expected",
        [
            ("gpt\x00admin", "gptadmin"),  # NUL
            ("real\nFAKE", "realFAKE"),  # LF
            ("a‮b", "ab"),  # RLO
            # ANSI: clean_identifier strips byte-by-byte, so the digits and
            # letters inside the escape survive. Callers wanting ANSI removed
            # as a unit must pipeline sanitize_for_log first.
            ("model\x1b[2J", "model2J"),
            ("model with spaces", "modelwithspaces"),  # spaces
            ("model<template>", "modeltemplate"),  # angle brackets
            ("αβγ-model", "-model"),  # Greek stripped (ASCII-only policy)
            ("привет", ""),  # Cyrillic stripped, empty result is legal
        ],
    )
    def test_hostile_chars_stripped(self, src: str, expected: str) -> None:
        assert clean_identifier(src) == expected

    def test_max_len_truncates(self) -> None:
        assert clean_identifier("abcdefghij", max_len=4) == "abcd"

    def test_max_len_after_strip(self) -> None:
        # Strip first, then truncate the result.
        assert clean_identifier("a\x00b\x00c\x00d\x00e", max_len=3) == "abc"

    def test_max_len_none_no_truncation(self) -> None:
        long = "a" * 1000
        assert clean_identifier(long) == long

    def test_empty_input(self) -> None:
        assert clean_identifier("") == ""

    def test_all_stripped_returns_empty(self) -> None:
        # Pure-controls plus pure-spaces plus pure-BIDI: nothing in the
        # identifier charset, returns empty.
        assert clean_identifier("\x00\x07\x1b\x7f ‮") == ""

    def test_idempotence(self) -> None:
        src = "model\x00 with\nstuff"
        assert clean_identifier(clean_identifier(src)) == clean_identifier(src)

    def test_negative_max_len_raises(self) -> None:
        # Regression: Python slice ``[:-1]`` would silently drop a trailing
        # char; reject explicitly so the caller notices the mistake.
        with pytest.raises(ValueError, match="max_len"):
            clean_identifier("abcde", max_len=-1)

    def test_zero_max_len_yields_empty(self) -> None:
        assert clean_identifier("abcde", max_len=0) == ""


# ---------------------------------------------------------------------------
# deep_sanitize_for_json
# ---------------------------------------------------------------------------


class TestDeepSanitizeForJson:
    def test_string_leaf(self) -> None:
        assert deep_sanitize_for_json("a\ud83dz") == "a�z"

    def test_dict_string_value(self) -> None:
        result = deep_sanitize_for_json({"k": "a\ud83dz"})
        assert result == {"k": "a�z"}

    def test_dict_string_key(self) -> None:
        # Even keys can be hostile if they originate from LLM output.
        result = deep_sanitize_for_json({"a\ud83d": "v"})
        assert result == {"a�": "v"}

    def test_nested_structure(self) -> None:
        data = {
            "outer": {
                "list": ["a\ud83d", "b", {"deep": "c\ud83d"}],
                "scalar": 42,
            },
            "name": "ok",
        }
        result = deep_sanitize_for_json(data)
        import json

        json.dumps(result)  # must not raise
        assert isinstance(result, dict)
        assert result["outer"]["list"][0] == "a�"
        assert result["outer"]["list"][2]["deep"] == "c�"
        assert result["outer"]["scalar"] == 42

    def test_preserves_container_type(self) -> None:
        # tuples stay tuples (orjson distinguishes; stdlib json doesn't,
        # but downstream callers may care).
        result = deep_sanitize_for_json(("a", "b"))
        assert isinstance(result, tuple)

    def test_primitives_pass_through(self) -> None:
        assert deep_sanitize_for_json(42) == 42
        assert deep_sanitize_for_json(3.14) == 3.14
        assert deep_sanitize_for_json(True) is True
        assert deep_sanitize_for_json(None) is None

    def test_safe_input_round_trips(self) -> None:
        # A clean structure must be byte-identical after sanitize.
        data = {"x": [1, 2, "ok"], "y": "αβγ"}
        assert deep_sanitize_for_json(data) == data

    def test_empty_containers(self) -> None:
        assert deep_sanitize_for_json({}) == {}
        assert deep_sanitize_for_json([]) == []
        assert deep_sanitize_for_json(()) == ()

    def test_idempotence(self) -> None:
        data = {"a": "x\ud83d", "b": ["y\ud83d", "z"]}
        once = deep_sanitize_for_json(data)
        twice = deep_sanitize_for_json(once)
        assert once == twice
