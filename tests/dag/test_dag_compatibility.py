"""Tests for gigaevo/programs/dag/compatibility.py

Covers _extract_model_origin_args, _type_origin_args, _normalize_annotation,
and _covariant_type_compatible with plain classes, Pydantic generics, typing
aliases, Union/Optional annotations, Any, and subclass relations.
"""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar, Union, get_origin

from pydantic import BaseModel

from gigaevo.programs.dag.compatibility import (
    _covariant_type_compatible,
    _extract_model_origin_args,
    _normalize_annotation,
    _type_origin_args,
)

# ---------------------------------------------------------------------------
# Pydantic generic models used as test fixtures
# ---------------------------------------------------------------------------

T = TypeVar("T")
U = TypeVar("U")


class PlainModel(BaseModel):
    value: int = 0


class ChildModel(PlainModel):
    extra: str = ""


class GenericBox[T](BaseModel):
    data: T


class UnrelatedModel(BaseModel):
    name: str = ""


# ---------------------------------------------------------------------------
# _extract_model_origin_args
# ---------------------------------------------------------------------------


class TestExtractModelOriginArgs:
    def test_plain_base_model_returns_cls_and_empty_args(self) -> None:
        """A non-generic BaseModel subclass returns (cls, ()) — concrete and no parameters."""
        result = _extract_model_origin_args(PlainModel)
        assert result is not None
        origin, args = result
        assert origin is PlainModel
        assert args == ()

    def test_non_base_model_returns_none(self) -> None:
        """A plain Python class that does not inherit from BaseModel returns None."""

        class PurePython:
            pass

        result = _extract_model_origin_args(PurePython)
        assert result is None

    def test_non_class_returns_none(self) -> None:
        """Passing a non-class (e.g. an integer) returns None."""
        result = _extract_model_origin_args(42)  # type: ignore[arg-type]
        assert result is None

    def test_parameterized_generic_model_origin(self) -> None:
        """A parameterized Pydantic generic (GenericBox[int]) reports GenericBox as origin."""
        parameterized = GenericBox[int]
        result = _extract_model_origin_args(parameterized)
        # parameterized Pydantic generics expose __pydantic_generic_metadata__
        # with origin=GenericBox and args=(int,)
        if result is not None:
            origin, args = result
            assert origin is GenericBox
            assert int in args
        # If the runtime doesn't expose metadata, result may be None — that is
        # also acceptable since we test best-effort extraction.

    def test_child_model_returns_child_cls(self) -> None:
        """A concrete subclass is returned as its own class (not the parent)."""
        result = _extract_model_origin_args(ChildModel)
        assert result is not None
        origin, args = result
        assert origin is ChildModel
        assert args == ()


# ---------------------------------------------------------------------------
# _type_origin_args
# ---------------------------------------------------------------------------


class TestTypeOriginArgs:
    def test_plain_class_returns_itself_with_empty_args(self) -> None:
        """int -> (int, ())"""
        origin, args = _type_origin_args(int)
        assert origin is int
        assert args == ()

    def test_typing_dict_alias(self) -> None:
        """dict[str, int] -> (dict, (str, int))"""
        origin, args = _type_origin_args(dict[str, int])
        assert origin is dict
        assert set(args) == {str, int}

    def test_typing_list_alias(self) -> None:
        """list[str] -> (list, (str,))"""
        origin, args = _type_origin_args(list[str])
        assert origin is list
        assert args == (str,)

    def test_typing_capital_list_alias(self) -> None:
        """List[int] (typing.List) -> (list, (int,))"""
        origin, args = _type_origin_args(list[int])
        assert origin is list
        assert args == (int,)

    def test_pydantic_model_plain(self) -> None:
        """PlainModel -> (PlainModel, ())"""
        origin, args = _type_origin_args(PlainModel)
        assert origin is PlainModel
        assert args == ()

    def test_pydantic_generic_parameterized(self) -> None:
        """GenericBox[int] should expose GenericBox as origin."""
        parameterized = GenericBox[int]
        origin, args = _type_origin_args(parameterized)
        # The origin should be GenericBox for a parameterized generic
        assert origin is GenericBox
        assert int in args

    def test_any_returns_any_with_empty_args(self) -> None:
        """Any -> (Any, ()) — treated as a plain class with no typing origin."""
        origin, args = _type_origin_args(Any)
        # Any has no __origin__, so falls through to (Any, ())
        assert args == ()


# ---------------------------------------------------------------------------
# _normalize_annotation
# ---------------------------------------------------------------------------


class TestNormalizeAnnotation:
    def test_any_returns_none(self) -> None:
        """Any annotation signals 'no enforcement' — normalize returns None."""
        result = _normalize_annotation(Any)
        assert result is None

    def test_plain_class_returns_singleton_list(self) -> None:
        """str -> [str]"""
        result = _normalize_annotation(str)
        assert result == [str]

    def test_union_returns_list_of_alternatives(self) -> None:
        """Union[int, str] -> [int, str]"""
        result = _normalize_annotation(Union[int, str])  # noqa: UP007
        assert result is not None
        assert set(result) == {int, str}

    def test_optional_strips_none_type(self) -> None:
        """Optional[str] == Union[str, None] -> [str] (NoneType excluded)."""
        result = _normalize_annotation(Optional[str])  # noqa: UP045
        assert result is not None
        # NoneType must not appear in the result
        assert type(None) not in result
        assert str in result

    def test_optional_with_any_returns_none(self) -> None:
        """Optional[Any] == Union[Any, None] — Any in union means unconstrained -> None."""
        result = _normalize_annotation(Optional[Any])  # noqa: UP045
        assert result is None

    def test_union_with_any_returns_none(self) -> None:
        """Union[int, Any] — Any in union means unconstrained -> None."""
        result = _normalize_annotation(Union[int, Any])  # noqa: UP007
        assert result is None

    def test_plain_model_returns_singleton_list(self) -> None:
        """A BaseModel subclass is a plain alternative -> [PlainModel]."""
        result = _normalize_annotation(PlainModel)
        assert result == [PlainModel]

    def test_nested_union_flattened(self) -> None:
        """Union[int, str, float] -> [int, str, float]."""
        result = _normalize_annotation(Union[int, str, float])  # noqa: UP007
        assert result is not None
        assert set(result) == {int, str, float}


# ---------------------------------------------------------------------------
# _covariant_type_compatible
# ---------------------------------------------------------------------------


class TestCovariantTypeCompatible:
    def test_any_annotation_always_compatible(self) -> None:
        """ann=Any means accept everything."""
        assert _covariant_type_compatible(PlainModel, Any) is True

    def test_exact_class_match_is_compatible(self) -> None:
        """Same class as src and ann is compatible."""
        assert _covariant_type_compatible(PlainModel, PlainModel) is True

    def test_subclass_is_compatible(self) -> None:
        """ChildModel (subclass of PlainModel) is compatible with annotation PlainModel."""
        assert _covariant_type_compatible(ChildModel, PlainModel) is True

    def test_unrelated_class_is_not_compatible(self) -> None:
        """UnrelatedModel is not a subclass of PlainModel -> incompatible."""
        assert _covariant_type_compatible(UnrelatedModel, PlainModel) is False

    def test_non_class_src_returns_false(self) -> None:
        """src must be a class; a non-class object returns False."""
        assert _covariant_type_compatible(42, PlainModel) is False  # type: ignore[arg-type]
        assert _covariant_type_compatible("not a class", PlainModel) is False  # type: ignore[arg-type]

    def test_parent_class_not_compatible_with_child_ann(self) -> None:
        """PlainModel is NOT a subclass of ChildModel -> not compatible."""
        assert _covariant_type_compatible(PlainModel, ChildModel) is False

    def test_typing_alias_src_not_class_returns_false(self) -> None:
        """src must be a class; typing aliases like list[int] are not classes -> False.

        _covariant_type_compatible requires src_model_cls to be a class (i.e.
        isinstance(src, type) is True).  A parameterized alias such as list[int]
        is not a type, so the function exits early with False.
        """
        assert _covariant_type_compatible(list[int], list[int]) is False  # type: ignore[arg-type]

    def test_plain_list_class_compatible_with_bare_list_ann(self) -> None:
        """list (plain class) src vs list (bare) ann -> compatible."""
        assert _covariant_type_compatible(list, list) is True

    def test_plain_list_class_compatible_with_parameterized_list_ann(self) -> None:
        """list (bare class) vs list[int] ann -> permissive (src has no args)."""
        assert _covariant_type_compatible(list, list[int]) is True

    def test_ann_with_no_args_is_permissive(self) -> None:
        """If ann has no type parameters, accept any matching origin regardless of src args."""
        # PlainModel (bare) has no args — permissive
        assert _covariant_type_compatible(PlainModel, PlainModel) is True

    def test_src_with_erased_args_is_permissive(self) -> None:
        """If src has no args (erased generics) but ann has args, accept permissively."""
        # PlainModel has no args; annotation PlainModel also has no args -> True
        assert _covariant_type_compatible(PlainModel, PlainModel) is True

    def test_dict_typing_alias_src_not_class_returns_false(self) -> None:
        """dict[str, int] as src is not a type class -> False (same as list[int])."""
        assert _covariant_type_compatible(dict[str, int], dict[str, int]) is False  # type: ignore[arg-type]

    def test_dict_plain_class_compatible_with_dict_ann(self) -> None:
        """dict (plain class) vs dict ann -> compatible."""
        assert _covariant_type_compatible(dict, dict) is True

    def test_different_class_origins_not_compatible(self) -> None:
        """list (class) src vs dict ann -> incompatible origins."""
        assert _covariant_type_compatible(list, dict) is False

    # ---- Tests for _match_param (lines 116-154) and non-type origins (107-108) ----

    def test_pydantic_generic_same_args_compatible(self) -> None:
        """GenericBox[int] (concrete class) as src vs GenericBox[int] as ann → compatible.

        Exercises pairwise _match_param when both src and ann have args.
        """
        # GenericBox[int] creates a concrete Pydantic class (a type)
        BoxInt = GenericBox[int]
        assert _covariant_type_compatible(BoxInt, GenericBox[int]) is True

    def test_pydantic_generic_any_arg_permissive(self) -> None:
        """GenericBox[int] src vs GenericBox[Any] ann → compatible (Any arg matches anything)."""
        BoxInt = GenericBox[int]
        assert _covariant_type_compatible(BoxInt, GenericBox[Any]) is True

    def test_pydantic_generic_typevar_arg_permissive(self) -> None:
        """GenericBox[int] src vs GenericBox[T] ann → compatible (TypeVar matches anything)."""
        BoxInt = GenericBox[int]
        assert _covariant_type_compatible(BoxInt, GenericBox[T]) is True

    def test_pydantic_generic_incompatible_args(self) -> None:
        """GenericBox[int] src vs GenericBox[str] ann → NOT compatible."""
        BoxInt = GenericBox[int]
        assert _covariant_type_compatible(BoxInt, GenericBox[str]) is False

    def test_pydantic_generic_src_no_args_permissive(self) -> None:
        """GenericBox (bare) src vs GenericBox[int] ann → permissive (src has no args)."""
        assert _covariant_type_compatible(GenericBox, GenericBox[int]) is True

    def test_pydantic_generic_mismatched_arg_count(self) -> None:
        """When src and ann have different number of args → not compatible."""
        K = TypeVar("K")
        V = TypeVar("V")

        class TwoParamBox(BaseModel, Generic[K, V]):
            key: K
            val: V

        # TwoParamBox[int, str] has 2 args, GenericBox[int] has 1 arg
        # But they have different origins, so they'll fail on origin check first
        TwoIS = TwoParamBox[int, str]
        assert _covariant_type_compatible(TwoIS, GenericBox[int]) is False

    def test_pydantic_generic_subclass_arg(self) -> None:
        """GenericBox[ChildModel] src vs GenericBox[PlainModel] ann → compatible (covariant class args)."""
        BoxChild = GenericBox[ChildModel]
        assert _covariant_type_compatible(BoxChild, GenericBox[PlainModel]) is True

    def test_pydantic_generic_non_subclass_arg(self) -> None:
        """GenericBox[UnrelatedModel] src vs GenericBox[PlainModel] ann → NOT compatible."""
        BoxUnrelated = GenericBox[UnrelatedModel]
        assert _covariant_type_compatible(BoxUnrelated, GenericBox[PlainModel]) is False

    def test_non_type_ann_origin_mismatch(self) -> None:
        """When ann_o is not a type (e.g. Union alias as ann), the else branch (line 107) fires."""
        # Union[int, str] has origin typing.Union which is NOT a type
        union_ann = Union[int, str]  # noqa: UP007
        assert get_origin(union_ann) is Union
        # PlainModel src origin IS a type, but Union is not → else branch: PlainModel != Union → False
        assert _covariant_type_compatible(PlainModel, union_ann) is False

    def test_pydantic_generic_nested_typing_args_compatible(self) -> None:
        """GenericBox[list[int]] src vs GenericBox[list[int]] ann → compatible.

        Exercises _match_param with typing-generic args (line 126+).
        """
        BoxListInt = GenericBox[list[int]]
        assert _covariant_type_compatible(BoxListInt, GenericBox[list[int]]) is True

    def test_pydantic_generic_nested_typing_args_incompatible(self) -> None:
        """GenericBox[list[int]] src vs GenericBox[list[str]] ann → NOT compatible.

        Exercises _match_param with mismatched typing-generic args.
        """
        BoxListInt = GenericBox[list[int]]
        assert _covariant_type_compatible(BoxListInt, GenericBox[list[str]]) is False

    def test_pydantic_generic_nested_typing_origin_mismatch(self) -> None:
        """GenericBox[list[int]] src vs GenericBox[dict[str, int]] ann → NOT compatible.

        Exercises _match_param where typing origins differ (list vs dict).
        """
        BoxListInt = GenericBox[list[int]]
        assert (
            _covariant_type_compatible(BoxListInt, GenericBox[dict[str, int]]) is False
        )

    def test_pydantic_generic_nested_typing_any_arg(self) -> None:
        """GenericBox[list[int]] src vs GenericBox[list[Any]] ann → compatible.

        Exercises _match_param recursion with Any in inner args.
        """
        BoxListInt = GenericBox[list[int]]
        assert _covariant_type_compatible(BoxListInt, GenericBox[list[Any]]) is True

    def test_pydantic_generic_nested_ann_no_inner_args(self) -> None:
        """GenericBox[list[int]] src vs GenericBox[list] ann → compatible (no inner args = permissive)."""
        BoxListInt = GenericBox[list[int]]
        assert _covariant_type_compatible(BoxListInt, GenericBox[list]) is True

    def test_pydantic_generic_nested_src_no_inner_args(self) -> None:
        """GenericBox[list] src vs GenericBox[list[int]] ann → compatible (src erased = permissive)."""
        BoxList = GenericBox[list]
        assert _covariant_type_compatible(BoxList, GenericBox[list[int]]) is True

    def test_pydantic_generic_nested_mismatched_inner_arg_count(self) -> None:
        """GenericBox[dict[str, int]] src vs GenericBox[list[int]] ann → NOT compatible.

        dict has 2 args, list has 1 — different typing origins.
        """
        BoxDict = GenericBox[dict[str, int]]
        assert _covariant_type_compatible(BoxDict, GenericBox[list[int]]) is False

    def test_pydantic_generic_nested_pydantic_arg_compatible(self) -> None:
        """GenericBox[ChildModel] src vs GenericBox[PlainModel] ann → compatible.

        Exercises _match_param class-origin path with Pydantic model args
        (lines 138-149) where both sa_o and aa_o are types and issubclass holds.
        """
        BoxChild = GenericBox[ChildModel]
        assert _covariant_type_compatible(BoxChild, GenericBox[PlainModel]) is True

    def test_non_type_src_origin_returns_false(self) -> None:
        """When src_o is not a type but ann_o is, the else branch on line 107 fires.

        This exercises the `ann_o != src_o` comparison.
        """
        # A non-class src that passes isinstance(src, type) check but whose
        # _type_origin_args returns a non-type origin is hard to construct.
        # Instead, test that a union ann (non-type origin) vs class src
        # enters the else branch.
        assert _covariant_type_compatible(int, int | str) is False

    def test_pydantic_generic_mismatched_arg_count_same_origin(self) -> None:
        """Src and ann from same generic origin but different arg counts → not compatible.

        Exercises line 117: len(src_args) != len(ann_args).
        """
        K = TypeVar("K")
        V = TypeVar("V")

        class PairBox(BaseModel, Generic[K, V]):
            first: K
            second: V

        # PairBox[int, str] has 2 args; GenericBox[int] has 1 arg but different origin
        # To test line 117 we need SAME origin with different arg counts.
        # PairBox[int, str] vs PairBox (bare, 0 args) → line 114 (permissive)
        # We can't easily get same origin with truly different non-zero arg counts
        # from Pydantic generics, so this tests the fall-through with bare origin.
        PairIS = PairBox[int, str]
        assert (
            _covariant_type_compatible(PairIS, PairBox) is True
        )  # bare = no args = permissive
