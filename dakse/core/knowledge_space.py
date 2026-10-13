"""
Knowledge Space representation for DA-KSE.
Models the cybersecurity repository as a directed, hierarchical knowledge graph G = (K, R).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
import numpy as np


@dataclass
class KnowledgeUnit:
    """Atomic knowledge unit k_i = (x_i, c_i, u_i) with hierarchical granularity."""
    unit_id: str
    title: str
    content: str
    level: int                   # \ell(k_i) \in {0, 1, 2, 3}
    attack_family: Optional[str] # Target class alignment (e.g., 'DDoS', 'Backdoor', 'Benign')
    category: str                # e.g., 'summary', 'symptom', 'protocol', 'mitigation', 'cross_ref'
    embedding: Optional[np.ndarray] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.token_count <= 0 and self.content:
            # Approximate token count: ~1.3 tokens per whitespace-separated word
            self.token_count = max(1, int(len(self.content.split()) * 1.3))

    def __repr__(self) -> str:
        return f"<Unit [{self.unit_id}] L{self.level}:{self.attack_family or 'General'} '{self.title}' ({self.token_count} tok)>"


class DirectedKnowledgeGraph:
    """Directed Knowledge Space G = (K, R) supporting hierarchical and cross-reference navigation."""

    def __init__(self):
        self.units: Dict[str, KnowledgeUnit] = {}
        # Forward relations (outgoing edges): unit_id -> set of target_ids
        self.relations: Dict[str, Set[str]] = {}
        # Reverse relations: target_id -> set of source_ids
        self.reverse_relations: Dict[str, Set[str]] = {}
        # Parent-to-children relations for granularity expansion (level l -> l+1)
        self.children: Dict[str, List[str]] = {}
        # Child-to-parent relations
        self.parent: Dict[str, str] = {}
        # Cross-reference links within same or adjacent levels
        self.cross_refs: Dict[str, Set[str]] = {}
        # Class alignment map: attack_family -> list of unit_ids
        self.class_index: Dict[str, List[str]] = {}

    def add_unit(self, unit: KnowledgeUnit) -> None:
        self.units[unit.unit_id] = unit
        if unit.unit_id not in self.relations:
            self.relations[unit.unit_id] = set()
        if unit.unit_id not in self.reverse_relations:
            self.reverse_relations[unit.unit_id] = set()
        if unit.unit_id not in self.children:
            self.children[unit.unit_id] = []
        if unit.unit_id not in self.cross_refs:
            self.cross_refs[unit.unit_id] = set()

        if unit.attack_family:
            if unit.attack_family not in self.class_index:
                self.class_index[unit.attack_family] = []
            self.class_index[unit.attack_family].append(unit.unit_id)

    def add_relation(self, source_id: str, target_id: str, relation_type: str = "hierarchical") -> None:
        if source_id not in self.units or target_id not in self.units:
            raise KeyError(f"Both {source_id} and {target_id} must be added before creating relation.")

        self.relations[source_id].add(target_id)
        self.reverse_relations[target_id].add(source_id)

        if relation_type == "hierarchical":
            if target_id not in self.children[source_id]:
                self.children[source_id].append(target_id)
            self.parent[target_id] = source_id
        elif relation_type == "cross_ref":
            self.cross_refs[source_id].add(target_id)
            self.cross_refs[target_id].add(source_id)

    def get_unit(self, unit_id: str) -> Optional[KnowledgeUnit]:
        return self.units.get(unit_id)

    def get_children(self, unit_id: str) -> List[KnowledgeUnit]:
        child_ids = self.children.get(unit_id, [])
        return [self.units[cid] for cid in child_ids if cid in self.units]

    def get_cross_refs(self, unit_id: str) -> List[KnowledgeUnit]:
        ref_ids = self.cross_refs.get(unit_id, set())
        return [self.units[rid] for rid in ref_ids if rid in self.units]

    def get_neighbors(self, unit_id: str) -> List[KnowledgeUnit]:
        """Returns all directly navigable forward neighbors (hierarchical children + cross-refs)."""
        target_ids = self.relations.get(unit_id, set())
        return [self.units[tid] for tid in target_ids if tid in self.units]

    def get_units_by_class(self, attack_family: str, level: Optional[int] = None) -> List[KnowledgeUnit]:
        unit_ids = self.class_index.get(attack_family, [])
        units = [self.units[uid] for uid in unit_ids if uid in self.units]
        if level is not None:
            units = [u for u in units if u.level == level]
        return units

    def get_units_by_level(self, level: int) -> List[KnowledgeUnit]:
        return [u for u in self.units.values() if u.level == level]

    @property
    def total_units(self) -> int:
        return len(self.units)

    @property
    def total_relations(self) -> int:
        return sum(len(targets) for targets in self.relations.values())
