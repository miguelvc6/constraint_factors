#!/usr/bin/env python3
"""
Canonical mapping from constraint type items (Q-ids) to project family names.
"""

from typing import Dict, Tuple

TYPE_ITEM_QID_TO_FAMILY: Dict[str, str] = {
    # Core families supported by the project.
    "Q21502838": "conflictWith",
    "Q21502410": "distinct",
    "Q21510855": "inverse",
    "Q21510862": "symmetric",
    "Q21503247": "itemRequiresStatement",
    "Q21510864": "valueRequiresStatement",
    "Q21510859": "oneOf",
    "Q19474404": "single",
    "Q21503250": "type",
    "Q21510865": "valueType",
}


def canonicalize_constraint_type(type_item_qid: str) -> Tuple[str, bool]:
    """Return (family_name, supported_bool) for a constraint type item Q-id."""
    qid = type_item_qid.strip()
    family = TYPE_ITEM_QID_TO_FAMILY.get(qid)
    if family:
        return family, True
    return f"unsupported:{qid}", False
