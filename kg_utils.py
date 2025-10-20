# kg_utils.py
from __future__ import annotations
from typing import Any, Dict, Optional
import re

__all__ = [
    "slugify",
    "normalized_id",
    "get_attr",
    "set_attr",
    "ensure_node_props",
    "ensure_rel_props",
    "get_node_label",
    "get_node_id",
    "get_node_name",
    "set_node_name",
    "remap_relationship_ids",
    "DEFAULT_WIKI_CATEGORY",
]

# Strings & IDs

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")

def normalized_id(name: str, label: str) -> str:
    return f"{slugify(name)}::{label}"

# Attr-or-dict tolerant accessors

def get_attr(obj: Any, name: str, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default

def set_attr(obj: Any, name: str, value: Any):
    if hasattr(obj, name):
        setattr(obj, name, value)
    elif isinstance(obj, dict):
        obj[name] = value

# Node / Relationship helpers

def ensure_node_props(node: Any) -> Dict[str, Any]:
    props = get_attr(node, "properties")
    if not isinstance(props, dict):
        props = {}
        set_attr(node, "properties", props)
    return props

def ensure_rel_props(rel: Any) -> Dict[str, Any]:
    props = get_attr(rel, "properties")
    if not isinstance(props, dict):
        props = {}
        set_attr(rel, "properties", props)
    return props

def get_node_label(node: Any) -> Optional[str]:
    return get_attr(node, "type")

def get_node_id(node: Any) -> Optional[str]:
    return get_attr(node, "id")

def get_node_name(node: Any) -> Optional[str]:
    props = get_attr(node, "properties") or {}
    if isinstance(props, dict):
        name = props.get("name")
        if name:
            return name
    return get_node_id(node)

def set_node_name(node: Any, new_name: str):
    props = ensure_node_props(node)
    props["name"] = new_name

def remap_relationship_ids(doc: Any, id_map: Dict[str, str]) -> Dict[str, int]:
    """Rewrite relationship endpoints when node IDs change."""
    rel_src, rel_tgt = 0, 0
    for rel in getattr(doc, "relationships", []):
        src = get_attr(rel, "source"); tgt = get_attr(rel, "target")
        src_id = get_attr(src, "id") if not isinstance(src, str) else src
        tgt_id = get_attr(tgt, "id") if not isinstance(tgt, str) else tgt
        if src_id in id_map:
            if isinstance(src, str):
                set_attr(rel, "source", id_map[src_id])
            else:
                set_attr(src, "id", id_map[src_id])
            rel_src += 1
        if tgt_id in id_map:
            if isinstance(tgt, str):
                set_attr(rel, "target", id_map[tgt_id])
            else:
                set_attr(tgt, "id", id_map[tgt_id])
            rel_tgt += 1
    return {"rel_src": rel_src, "rel_tgt": rel_tgt}

# Defaults

DEFAULT_WIKI_CATEGORY: Dict[str, str] = {
    "Organization": "Companies",
    "Product": "Products",
    "Person": "People",
    "Country": "Countries",
    "Government Body": "Government bodies",
    "Location": "Places",
}