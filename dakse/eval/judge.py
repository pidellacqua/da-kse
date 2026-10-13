"""
Diagnostic explanation quality scoring rubric and automated evaluator.
Implements the 0-3 multi-dimensional rubric from Section 4.7 and Table 4 of the paper.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import re
import numpy as np


@dataclass
class ExplanationQualityScore:
    q_correct_raw: int      # 0 to 3
    q_evidence_raw: int     # 0 to 3
    q_action_raw: int       # 0 to 3
    q_correct_norm: float   # 0.0 to 1.0
    q_evidence_norm: float  # 0.0 to 1.0
    q_action_norm: float    # 0.0 to 1.0
    overall_quality: float  # Q_diag in [0, 1]


class RubricDiagnosticJudge:
    """Automated rubric evaluator implementing Table 4 criteria."""

    FLOW_EVIDENCE_KEYWORDS = [
        "protocol", "tcp", "udp", "icmp", "bytes", "packet", "rate",
        "duration", "state", "connection", "port", "flow", "syn", "ack", "anomaly"
    ]

    ACTIONABLE_KEYWORDS = [
        "contain", "mitigat", "investigat", "quarantine", "firewall", "block",
        "isolate", "patch", "rule", "monitor", "alert", "remediat", "priority", "containment"
    ]

    def evaluate_explanation(
        self,
        ground_truth: str,
        predicted_class: str,
        explanation: str,
        network_evidence: Dict[str, Any],
        has_knowledge_context: bool = False,
    ) -> ExplanationQualityScore:
        """Evaluates explanation along the 3 normalized dimensions."""
        expl_lower = explanation.lower()

        # 1. Diagnostic Correctness (0 - 3)
        if predicted_class.lower() == ground_truth.lower():
            if "contradict" in expl_lower or "incorrect" in expl_lower:
                q_correct = 1
            elif len(explanation.split()) < 5:
                q_correct = 2
            else:
                q_correct = 3
        else:
            # Misclassification
            if ground_truth.lower() in expl_lower:
                q_correct = 1
            else:
                q_correct = 0

        # 2. Evidence Alignment (0 - 3)
        matched_flow = sum(1 for kw in self.FLOW_EVIDENCE_KEYWORDS if kw in expl_lower)
        if matched_flow >= 3 or (has_knowledge_context and matched_flow >= 2):
            q_evidence = 3
        elif matched_flow >= 1:
            q_evidence = 2
        elif len(explanation.strip()) > 10:
            q_evidence = 1
        else:
            q_evidence = 0

        # 3. Actionability (0 - 3)
        matched_act = sum(1 for kw in self.ACTIONABLE_KEYWORDS if kw in expl_lower)
        if matched_act >= 2 or (has_knowledge_context and matched_act >= 1):
            q_action = 3
        elif matched_act == 1:
            q_action = 2
        elif len(explanation.strip()) > 15:
            q_action = 1
        else:
            q_action = 0

        q_c_norm = q_correct / 3.0
        q_e_norm = q_evidence / 3.0
        q_a_norm = q_action / 3.0
        overall = (q_c_norm + q_e_norm + q_a_norm) / 3.0

        return ExplanationQualityScore(
            q_correct_raw=q_correct,
            q_evidence_raw=q_evidence,
            q_action_raw=q_action,
            q_correct_norm=q_c_norm,
            q_evidence_norm=q_e_norm,
            q_action_norm=q_a_norm,
            overall_quality=overall,
        )

    def evaluate_batch(
        self,
        ground_truths: List[str],
        predictions: List[str],
        explanations: List[str],
        evidence_list: List[Dict[str, Any]],
        has_knowledge_flags: List[bool],
    ) -> Dict[str, float]:
        """Averages scores across evaluation set."""
        scores = [
            self.evaluate_explanation(gt, pr, ex, ev, hk)
            for gt, pr, ex, ev, hk in zip(
                ground_truths, predictions, explanations, evidence_list, has_knowledge_flags
            )
        ]
        return {
            "diag_correctness": float(np.mean([s.q_correct_norm for s in scores])),
            "evidence_alignment": float(np.mean([s.q_evidence_norm for s in scores])),
            "actionability": float(np.mean([s.q_action_norm for s in scores])),
            "overall_quality": float(np.mean([s.overall_quality for s in scores])),
        }
