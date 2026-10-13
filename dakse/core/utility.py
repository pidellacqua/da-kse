"""
Utility estimation module for DA-KSE.
Implements the multi-component utility function:
hat{u}(k_i, q) = alpha * s_lex + beta * s_emb + gamma * s_graph + delta * s_cls
and the redundancy-aware marginal context utility Delta U(a_t).
"""

import re
from typing import Dict, List, Set, Optional, Tuple
import numpy as np

from dakse.config import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    DEFAULT_DELTA,
)
from dakse.core.knowledge_space import KnowledgeUnit, DirectedKnowledgeGraph


def cosine_similarity(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
    """Computes cosine similarity in range [0, 1]."""
    if v1 is None or v2 is None:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(dot, 0.0, 1.0))


class UtilityEstimator:
    """Computes estimated utility hat{u}(k_i, q) and marginal utility Delta U(a_t)."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        delta: float = DEFAULT_DELTA,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def set_weights(self, alpha: float, beta: float, gamma: float, delta: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def compute_lexical_score(self, query_text: str, unit: KnowledgeUnit) -> float:
        """Normalized sparse keyword overlap s_lex(k_i, q)."""
        q_tokens = set(re.findall(r"\w+", query_text.lower()))
        if not q_tokens:
            return 0.0

        target_text = f"{unit.title} {unit.content}".lower()
        u_tokens = set(re.findall(r"\w+", target_text))
        if not u_tokens:
            return 0.0

        overlap = len(q_tokens & u_tokens)
        # Jaccard / normalized overlap bounded in [0, 1]
        score = overlap / (len(q_tokens) + 0.5 * overlap)
        return float(np.clip(score, 0.0, 1.0))

    def compute_embedding_score(self, query_embedding: Optional[np.ndarray], unit: KnowledgeUnit) -> float:
        """Dense semantic cosine similarity s_emb(k_i, q)."""
        if query_embedding is None or unit.embedding is None:
            return 0.0
        return cosine_similarity(query_embedding, unit.embedding)

    def compute_graph_score(
        self,
        unit: KnowledgeUnit,
        graph: DirectedKnowledgeGraph,
        visited_ids: Set[str],
    ) -> float:
        """
        Topological graph proximity score s_graph(k_i).
        Favors units adjacent to visited nodes, relevant hierarchical positions, and high-degree hubs.
        """
        score = 0.2  # Baseline structural prior

        # Granularity hierarchy preference: higher levels provide fine-grained evidence
        # Level 0 (Domain): 0.3, Level 1 (Section): 0.5, Level 2 (Paragraph): 0.8, Level 3 (Cross-ref): 0.7
        level_priors = {0: 0.3, 1: 0.5, 2: 0.8, 3: 0.7}
        score += 0.3 * level_priors.get(unit.level, 0.4)

        # Proximity to visited nodes: if immediate parent or neighbor is already visited
        has_visited_neighbor = False
        for vid in visited_ids:
            if unit.unit_id in graph.relations.get(vid, set()) or vid in graph.relations.get(unit.unit_id, set()):
                has_visited_neighbor = True
                break

        if has_visited_neighbor:
            score += 0.4
        elif visited_ids:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))

    def compute_classifier_score(
        self,
        unit: KnowledgeUnit,
        class_probabilities: Dict[str, float],
    ) -> float:
        """
        Classifier guidance score s_cls(k_i, h).
        Measures compatibility between unit's attack family and XGBoost softprob distribution.
        """
        if not unit.attack_family or not class_probabilities:
            # For general cross-reference units, assign uniform baseline
            return 0.25

        prob = class_probabilities.get(unit.attack_family, 0.0)
        return float(np.clip(prob, 0.0, 1.0))

    def estimate_unit_utility(
        self,
        unit: KnowledgeUnit,
        query_text: str,
        query_embedding: Optional[np.ndarray],
        graph: DirectedKnowledgeGraph,
        visited_ids: Set[str],
        class_probabilities: Dict[str, float],
    ) -> float:
        """
        Computes equation (4):
        hat{u}(k_i, q) = alpha * s_lex + beta * s_emb + gamma * s_graph + delta * s_cls
        """
        s_lex = self.compute_lexical_score(query_text, unit) if self.alpha > 0 else 0.0
        s_emb = self.compute_embedding_score(query_embedding, unit) if self.beta > 0 else 0.0
        s_graph = self.compute_graph_score(unit, graph, visited_ids) if self.gamma > 0 else 0.0
        s_cls = self.compute_classifier_score(unit, class_probabilities) if self.delta > 0 else 0.0

        # Normalization check: weights sum
        w_sum = self.alpha + self.beta + self.gamma + self.delta
        if w_sum <= 0:
            return 0.0

        u_hat = (self.alpha * s_lex + self.beta * s_emb + self.gamma * s_graph + self.delta * s_cls) / w_sum
        return float(np.clip(u_hat, 0.0, 1.0))

    def compute_marginal_utility(
        self,
        candidate_unit: KnowledgeUnit,
        current_context: List[KnowledgeUnit],
        query_text: str,
        query_embedding: Optional[np.ndarray],
        graph: DirectedKnowledgeGraph,
        visited_ids: Set[str],
        class_probabilities: Dict[str, float],
    ) -> float:
        """
        Computes redundancy-aware marginal utility Delta U(a_t):
        Delta U = hat{u}(k_i, q) * (1 - max_{k_j in C_t} sim(k_i, k_j))
        """
        u_hat = self.estimate_unit_utility(
            candidate_unit,
            query_text,
            query_embedding,
            graph,
            visited_ids,
            class_probabilities,
        )

        if not current_context:
            return u_hat

        # Max semantic similarity with already included units in context
        max_sim = 0.0
        for ctx_unit in current_context:
            sim = cosine_similarity(candidate_unit.embedding, ctx_unit.embedding)
            if sim > max_sim:
                max_sim = sim

        # Redundancy penalty with diminishing marginal returns (Section 3.3)
        saturation = 1.0 / (1.0 + 0.07 * len(current_context))
        marginal_u = u_hat * max(0.0, (1.0 - max_sim)) * saturation
        return float(marginal_u)
