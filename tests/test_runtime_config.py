"""Tests for gigaevo.memory.runtime_config utility functions.

Pin down config loading and type coercion behavior.
"""

import pytest

from gigaevo.memory.runtime_config import (
    deep_get,
    load_settings,
    resolve_local_path,
    resolve_settings_path,
    to_bool,
    to_int,
    to_list,
    to_str,
)
from pathlib import Path


# ===========================================================================
# deep_get
# ===========================================================================


class TestDeepGet:
    def test_simple_key(self):
        assert deep_get({"a": 1}, "a") == 1

    def test_nested_key(self):
        assert deep_get({"a": {"b": {"c": 3}}}, "a.b.c") == 3

    def test_missing_key_returns_default(self):
        assert deep_get({"a": 1}, "b") is None

    def test_missing_key_custom_default(self):
        assert deep_get({"a": 1}, "b", default="x") == "x"

    def test_non_dict_intermediate(self):
        assert deep_get({"a": "string"}, "a.b") is None

    def test_empty_dict(self):
        assert deep_get({}, "a.b") is None

    def test_single_level(self):
        assert deep_get({"key": "val"}, "key") == "val"

    def test_none_value_returned(self):
        assert deep_get({"a": None}, "a") is None

    def test_none_value_not_confused_with_missing(self):
        """deep_get returns None for {"a": None} — same as missing key."""
        # This documents current behavior: no way to distinguish
        assert deep_get({"a": None}, "a") is deep_get({}, "a")


# ===========================================================================
# to_bool
# ===========================================================================


class TestToBool:
    @pytest.mark.parametrize("val", [True, 1, 1.0, "1", "true", "True", "TRUE", "yes", "on"])
    def test_truthy(self, val):
        assert to_bool(val) is True

    @pytest.mark.parametrize("val", [False, 0, 0.0, "0", "false", "False", "no", "off"])
    def test_falsy(self, val):
        assert to_bool(val) is False

    def test_none_returns_default_false(self):
        assert to_bool(None) is False

    def test_none_returns_custom_default(self):
        assert to_bool(None, default=True) is True

    def test_unrecognized_string_returns_default(self):
        assert to_bool("maybe") is False
        assert to_bool("maybe", default=True) is True

    def test_negative_int(self):
        assert to_bool(-1) is True

    def test_float_nonzero(self):
        assert to_bool(0.5) is True


# ===========================================================================
# to_int
# ===========================================================================


class TestToInt:
    def test_valid(self):
        assert to_int(5) == 5

    def test_string(self):
        assert to_int("10") == 10

    def test_invalid(self):
        assert to_int("abc") == 0

    def test_custom_default(self):
        assert to_int("abc", default=-1) == -1

    def test_none(self):
        assert to_int(None) == 0

    def test_float_truncates(self):
        assert to_int(3.9) == 3


# ===========================================================================
# to_list
# ===========================================================================


class TestToList:
    def test_none_returns_empty(self):
        assert to_list(None) == []

    def test_list_passthrough(self):
        assert to_list([1, 2]) == [1, 2]

    def test_tuple_to_list(self):
        assert to_list((1, 2)) == [1, 2]

    def test_string_split_by_comma(self):
        assert to_list("a, b, c") == ["a", "b", "c"]

    def test_string_single(self):
        assert to_list("hello") == ["hello"]

    def test_string_empty(self):
        assert to_list("") == []

    def test_string_commas_only(self):
        assert to_list(",,,") == []

    def test_scalar_wrapped(self):
        assert to_list(42) == [42]

    def test_dict_wrapped(self):
        d = {"a": 1}
        assert to_list(d) == [d]


# ===========================================================================
# to_str
# ===========================================================================


class TestToStr:
    def test_none_returns_default_empty(self):
        assert to_str(None) == ""

    def test_none_returns_custom_default(self):
        assert to_str(None, default=None) is None

    def test_value_converted(self):
        assert to_str(42) == "42"

    def test_string_passthrough(self):
        assert to_str("hello") == "hello"


# ===========================================================================
# load_settings
# ===========================================================================


class TestLoadSettings:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_settings(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_valid_yaml(self, tmp_path):
        f = tmp_path / "settings.yaml"
        f.write_text("key: value\nnested:\n  a: 1\n")
        result = load_settings(f)
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_non_dict_yaml_raises(self, tmp_path):
        f = tmp_path / "settings.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="expected a mapping"):
            load_settings(f)

    def test_empty_yaml_returns_empty(self, tmp_path):
        f = tmp_path / "settings.yaml"
        f.write_text("")
        result = load_settings(f)
        assert result == {}


# ===========================================================================
# resolve_settings_path
# ===========================================================================


class TestResolveSettingsPath:
    def test_explicit_path(self):
        p = resolve_settings_path("/some/path.yaml")
        assert p == Path("/some/path.yaml")

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("EVO_MEMORY_CONFIG_PATH", "/from/env.yaml")
        p = resolve_settings_path()
        assert p == Path("/from/env.yaml")

    def test_fallback_env_var(self, monkeypatch):
        monkeypatch.delenv("EVO_MEMORY_CONFIG_PATH", raising=False)
        monkeypatch.setenv("EVO_MEMORY_SETTINGS_PATH", "/fallback.yaml")
        p = resolve_settings_path()
        assert p == Path("/fallback.yaml")


# ===========================================================================
# resolve_local_path
# ===========================================================================


class TestResolveLocalPath:
    def test_absolute_path_returned(self):
        p = resolve_local_path(Path("/base"), "/abs/path", "default")
        assert p == Path("/abs/path")

    def test_relative_resolved_against_base(self):
        p = resolve_local_path(Path("/base"), "rel/path", "default")
        assert p == Path("/base/rel/path")

    def test_none_uses_default(self):
        p = resolve_local_path(Path("/base"), None, "default/dir")
        assert p == Path("/base/default/dir")

    def test_empty_string_uses_default(self):
        p = resolve_local_path(Path("/base"), "", "default/dir")
        assert p == Path("/base/default/dir")
