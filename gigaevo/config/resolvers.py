import re

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf


def _ref_resolver(path, *, _root_):
    """
    OmegaConf resolver for resolving and optionally instantiating configuration objects,
    and accessing attributes or calling methods using '::' and '()' syntax.
    """
    path_parts = path.split("::")
    node_path = path_parts[0]
    node = OmegaConf.select(_root_, node_path)

    # Split the path to get the parent container and the base name
    *prefixes, base = node_path.split(".")
    prefix = ".".join(prefixes)
    parent = OmegaConf.select(_root_, prefix) if prefixes else _root_

    # Instantiate if node is a config object
    if isinstance(node, (DictConfig, ListConfig)):
        instantiated_node = instantiate(node, _recursive_=True)
        parent[base] = instantiated_node
    else:
        instantiated_node = node

    # Apply attribute/method chain
    for part in path_parts[1:]:
        match = re.fullmatch(r"(\w+)(\(\))?", part)
        if not match:
            raise ValueError(f"Invalid syntax in method chain: '{part}'")
        attr_name, is_call = match.groups()
        instantiated_node = getattr(instantiated_node, attr_name)
        if is_call:
            instantiated_node = instantiated_node()

    return instantiated_node


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver(
        "get_object", lambda obj: hydra.utils.get_object(obj)
    )
    OmegaConf.register_new_resolver("merge", lambda x, y: x + y)
    OmegaConf.register_new_resolver("len", lambda arr: len(arr))
    OmegaConf.register_new_resolver("ref", _ref_resolver)
