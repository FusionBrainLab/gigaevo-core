from typing import Any, Optional, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel


def _extract_model_origin_args(model_cls: type) -> tuple[type, tuple[Any, ...]] | None:
    """
    For a (possibly parameterized) Pydantic model class, return (origin, args).
    Falls back to (model_cls, ()) if it's a concrete BaseModel subclass with no metadata.
    Returns None if it's not even a BaseModel subclass.
    """
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        return None
    meta = getattr(model_cls, "__pydantic_generic_metadata__", None)
    if isinstance(meta, dict):
        o = meta.get("origin", None)
        a = meta.get("args", None)
        if isinstance(o, type):
            return (o, tuple(a) if isinstance(a, tuple) else tuple())
    # Other best-effort fallbacks seen in the wild:
    o = getattr(model_cls, "__origin__", None)
    a = getattr(model_cls, "__args__", None)
    if isinstance(o, type):
        return (o, tuple(a) if isinstance(a, tuple) else tuple())
    # Plain BaseModel (non-generic or already concrete):
    return (model_cls, ())


def _type_origin_args(t: Any) -> tuple[Any, tuple[Any, ...]]:
    """
    Uniformly extract (origin, args) for typing aliases and Pydantic model classes.
    - typing alias like dict[str, int] -> (dict, (str, int))
    - Pydantic generic class like Box[int] -> (Box, (int,))
    - Plain class -> (class, ())
    """
    o = get_origin(t)
    if o is not None:
        return (o, get_args(t))
    if isinstance(t, type) and issubclass(t, BaseModel):
        oa = _extract_model_origin_args(t)
        if oa:
            return oa
    return (t, ())


def _normalize_annotation(annotation: Any) -> list[Any] | None:
    """
    Normalize a destination field annotation into alternatives:
      - None => Any (no enforcement)
      - []   => invalid (we can't reason about it)
      - [a1, a2, ...] => acceptable alternatives (Union unpacked)
    Accepts ANY reasonable type (typing alias, BaseModel generic, plain class).
    """
    if annotation is Any:
        return None
    o = get_origin(annotation)
    if o in (Union, Optional):
        alts: list[Any] = []
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            norm = _normalize_annotation(arg)
            if norm is None:
                return None  # Any in union -> Any
            if norm == []:
                return []
            alts.extend(norm)
        return alts
    # everything else is a single alternative
    return [annotation]


def _covariant_type_compatible(src_model_cls: type, ann: Any) -> bool:
    """
    src_model_cls: producer OutputModel class (often a Pydantic BaseModel generic)
    ann:           consumer annotation alternative (typing alias, BaseModel generic, or class)

    Policy:
      1) ann is Any -> True
      2) Compare (origin, args) covariantly
         - origins must be compatible (issubclass if both classes; equality if typing origins)
         - if ann has no args -> True
         - if src has no args -> True (permissive; erased generics)
         - else compare args pairwise covariantly:
             * ann arg Any/TypeVar -> True
             * both typing generics -> same origin, recurse on args
             * both classes -> issubclass(src_arg, ann_arg)
             * otherwise -> True (permissive)
    """
    if ann is Any:
        return True

    # Producer: must be a class (usually BaseModel subclass)
    if not isinstance(src_model_cls, type):
        return False

    src_o, src_args = _type_origin_args(src_model_cls)
    ann_o, ann_args = _type_origin_args(ann)

    # Origin compatibility
    # typing origins (e.g., dict, list, tuple) compare by equality
    # class origins compare by issubclass
    if isinstance(ann_o, type) and isinstance(src_o, type):
        if not issubclass(src_o, ann_o):
            return False
    else:
        if ann_o != src_o:
            return False

    # If annotation imposes no parameters, accept
    if not ann_args:
        return True
    # If src args are erased/unavailable, accept (permissive)
    if not src_args:
        return True
    if len(src_args) != len(ann_args):
        return False

    def _match_param(sa: Any, aa: Any) -> bool:
        if aa is Any or isinstance(aa, TypeVar):
            return True
        sa_o, sa_as = _type_origin_args(sa)
        aa_o, aa_as = _type_origin_args(aa)

        # Both typing-style generics: must share origin, then recurse on args
        if (get_origin(sa) is not None) or (get_origin(aa) is not None):
            if sa_o != aa_o:
                return False
            if not aa_as:
                return True
            if not sa_as:
                return True
            if len(sa_as) != len(aa_as):
                return False
            return all(_match_param(x, y) for x, y in zip(sa_as, aa_as))

        # Both are classes
        if isinstance(sa_o, type) and isinstance(aa_o, type):
            # covariant
            if not issubclass(sa_o, aa_o):
                return False
            # If annotation has inner args on a class origin (rare), recurse
            if aa_as:
                if not sa_as:
                    return True
                if len(sa_as) != len(aa_as):
                    return False
                return all(_match_param(x, y) for x, y in zip(sa_as, aa_as))
            return True

        # Mixed/unknown -> permissive
        return True

    return all(_match_param(sa, aa) for sa, aa in zip(src_args, ann_args))
