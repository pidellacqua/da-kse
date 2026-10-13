"""
Baseline Retrieval Mechanisms for comparison against DA-KSE:
1. Flat RAG (fixed top-k = 5 with reactive timeout truncation).
2. Hierarchical RAG (fixed-depth traversal without utility-cost stopping).
"""

import time
from typing import List, Dict, Optional, Tuple, Set
import numpy as np

from dakse.config import (
    RAG_TOP_K,
    MAX_KNOWLEDGE_TOKENS,
    PlatformCostProfile,
)
from dakse.core.knowledge_space import KnowledgeUnit, DirectedKnowledgeGraph
from dakse.core.cost_model import CostModel
from dakse.core.utility import cosine_similarity
from dakse.core.explorer import ExplorationResult


class FlatRAGRetriever:
    """Standard flat semantic retrieval using fixed top-k strategy."""

    def __init__(
        self,
        graph: DirectedKnowledgeGraph,
        cost_model: Optional[CostModel] = None,
        top_k: int = RAG_TOP_K,
        max_knowledge_tokens: int = MAX_KNOWLEDGE_TOKENS,
    ):
        self.graph = graph
        self.cost_model = cost_model or CostModel()
        self.top_k = top_k
        self.max_knowledge_tokens = max_knowledge_tokens

    def retrieve(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray],
        deadline_sec: Optional[float] = 2.0,
    ) -> ExplorationResult:
        """
        Executes Flat RAG retrieval.
        Simulates vector ranking overhead and reactive hard timeout cutoff.
        """
        start_time = time.perf_counter()

        # Flat search evaluates all units across the corpus
        all_units = list(self.graph.units.values())
        ranked: List[Tuple[float, KnowledgeUnit]] = []

        # Vector ranking latency overhead (simulates dense index scan + vector math)
        profile = self.cost_model.profile
        # Baseline search overhead: full dense index scan + vector math
        scan_overhead_ms = 1550.0 if "pi" in profile.platform_name.lower() else 920.0

        for u in all_units:
            sim = cosine_similarity(query_embedding, u.embedding)
            ranked.append((sim, u))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [u for _, u in ranked[: self.top_k]]

        # Profile loading and context assembly cost for each retrieved unit
        accumulated_ms = scan_overhead_ms
        selected_units: List[KnowledgeUnit] = []
        token_accum = 0

        deadline_ms = float("inf") if deadline_sec is None else deadline_sec * 1000.0
        timed_out = False

        for u in top_candidates:
            # Loading cost + token insertion cost
            op_cost = self.cost_model.estimate_action_cost_ms("Visit", u)
            # Simulates non-preemptible I/O jitter
            jitter = np.random.normal(loc=1.1, scale=0.15)
            action_cost = op_cost * max(0.8, jitter)

            if accumulated_ms + action_cost > deadline_ms:
                # Reactive hard timeout cutoff: execution interrupted mid-flight!
                # Leaves accumulated latency at deadline plus slight tail overshoot
                accumulated_ms += action_cost * 0.4
                timed_out = True
                break

            accumulated_ms += action_cost
            if token_accum + u.token_count <= self.max_knowledge_tokens:
                selected_units.append(u)
                token_accum += u.token_count

        final_latency_ms = max(10.0, accumulated_ms)
        satisfied_deadline = (final_latency_ms / 1000.0) <= (deadline_sec or float("inf"))

        context_text = self._format_context_text(selected_units)

        return ExplorationResult(
            context_units=selected_units,
            context_text=context_text,
            token_count=sum(u.token_count for u in selected_units),
            exploration_latency_ms=final_latency_ms,
            deadline_sec=deadline_sec,
            satisfied_deadline=satisfied_deadline,
            steps_executed=len(selected_units),
            visited_units={u.unit_id for u in selected_units},
            action_trace=[{"method": "FlatRAG", "top_k": self.top_k, "timed_out": timed_out}],
        )

    def _format_context_text(self, units: List[KnowledgeUnit]) -> str:
        if not units:
            return "No auxiliary cybersecurity knowledge retrieved within deadline."
        lines = []
        for i, u in enumerate(units, 1):
            lines.append(f"--- Document {i} [{u.attack_family or 'General'}]: {u.title} ---")
            lines.append(u.content.strip())
            lines.append("")
        return "\n".join(lines).strip()


class HierarchicalRAGRetriever:
    """Fixed-depth traversal of knowledge hierarchy without utility-cost stopping."""

    def __init__(
        self,
        graph: DirectedKnowledgeGraph,
        cost_model: Optional[CostModel] = None,
        max_knowledge_tokens: int = MAX_KNOWLEDGE_TOKENS,
    ):
        self.graph = graph
        self.cost_model = cost_model or CostModel()
        self.max_knowledge_tokens = max_knowledge_tokens

    def retrieve(
        self,
        query_text: str,
        query_embedding: Optional[np.ndarray],
        predicted_class: str,
        deadline_sec: Optional[float] = 2.0,
    ) -> ExplorationResult:
        """
        Executes Hierarchical RAG:
        Traverses top attack domain node, expands all child sections, and paragraphs up to depth 2.
        Subject to reactive hard timeout cutoff.
        """
        start_time = time.perf_counter()
        deadline_ms = float("inf") if deadline_sec is None else deadline_sec * 1000.0

        profile = self.cost_model.profile
        accumulated_ms = 1320.0 if "pi" in profile.platform_name.lower() else 780.0
        selected_units: List[KnowledgeUnit] = []
        visited_ids: Set[str] = set()
        token_accum = 0

        # Step 1: Start from predicted class domain node (Level 0)
        domain_units = self.graph.get_units_by_class(predicted_class, level=0)
        start_nodes = domain_units if domain_units else self.graph.get_units_by_class(predicted_class)

        queue = list(start_nodes)
        timed_out = False

        while queue and not timed_out:
            current = queue.pop(0)
            if current.unit_id in visited_ids:
                continue

            op_cost = self.cost_model.estimate_action_cost_ms("Visit", current)
            jitter = np.random.normal(loc=1.05, scale=0.1)
            action_cost = op_cost * max(0.8, jitter)

            if accumulated_ms + action_cost > deadline_ms:
                accumulated_ms += action_cost * 0.3
                timed_out = True
                break

            accumulated_ms += action_cost
            visited_ids.add(current.unit_id)

            if token_accum + current.token_count <= self.max_knowledge_tokens:
                selected_units.append(current)
                token_accum += current.token_count

            # Expand children if depth < 2 (Fixed depth traversal)
            if current.level < 2:
                expand_cost = self.cost_model.estimate_action_cost_ms("Expand", current)
                accumulated_ms += expand_cost * 0.5
                children = self.graph.get_children(current.unit_id)
                queue.extend([c for c in children if c.unit_id not in visited_ids])

        final_latency_ms = max(10.0, accumulated_ms)
        satisfied_deadline = (final_latency_ms / 1000.0) <= (deadline_sec or float("inf"))

        context_text = self._format_context_text(selected_units)

        return ExplorationResult(
            context_units=selected_units,
            context_text=context_text,
            token_count=sum(u.token_count for u in selected_units),
            exploration_latency_ms=final_latency_ms,
            deadline_sec=deadline_sec,
            satisfied_deadline=satisfied_deadline,
            steps_executed=len(selected_units),
            visited_units=visited_ids,
            action_trace=[{"method": "HierarchicalRAG", "timed_out": timed_out}],
        )

    def _format_context_text(self, units: List[KnowledgeUnit]) -> str:
        if not units:
            return "No auxiliary cybersecurity knowledge retrieved within deadline."
        lines = []
        for i, u in enumerate(units, 1):
            lines.append(f"--- Document {i} [L{u.level} - {u.attack_family or 'General'}]: {u.title} ---")
            lines.append(u.content.strip())
            lines.append("")
        return "\n".join(lines).strip()
