"""
Evaluation metrics for DA-KSE intrusion diagnosis:
- Accuracy, Macro-F1, Weighted-F1
- Per-class Precision, Recall, F1
- Confusion Matrix (10x10)
- Override Metrics (Override Rate, Correct Override Rate, Harmful Override Rate)
- Deadline Satisfaction Rate (DSR), Latency, Energy
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from dakse.config import TARGET_CLASSES


class DiagnosticMetricsCalculator:
    """Calculates classification, override, and edge runtime metrics."""

    def __init__(self, target_classes: List[str] = TARGET_CLASSES):
        self.target_classes = list(target_classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.target_classes)}

    def compute_confusion_matrix(
        self, y_true: List[str], y_pred: List[str]
    ) -> np.ndarray:
        K = len(self.target_classes)
        cm = np.zeros((K, K), dtype=int)
        for t, p in zip(y_true, y_pred):
            ti = self.class_to_idx.get(t, -1)
            pi = self.class_to_idx.get(p, -1)
            if ti >= 0 and pi >= 0:
                cm[ti, pi] += 1
        return cm

    def compute_per_class_metrics(
        self, cm: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Computes Precision, Recall, F1 for each class from confusion matrix."""
        results = {}
        for i, c in enumerate(self.target_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            results[c] = {
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "support": int(support),
            }
        return results

    def compute_classification_summary(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict[str, Any]:
        cm = self.compute_confusion_matrix(y_true, y_pred)
        per_class = self.compute_per_class_metrics(cm)

        total_samples = len(y_true)
        correct = sum(cm[i, i] for i in range(len(self.target_classes)))
        accuracy = correct / total_samples if total_samples > 0 else 0.0

        macro_f1 = float(np.mean([m["f1"] for m in per_class.values()]))

        total_support = sum(m["support"] for m in per_class.values())
        if total_support > 0:
            weighted_f1 = float(sum(m["f1"] * m["support"] for m in per_class.values()) / total_support)
        else:
            weighted_f1 = 0.0

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class": per_class,
            "confusion_matrix": cm,
            "total_samples": total_samples,
        }

    def compute_override_metrics(
        self, y_true: List[str], y_ml: List[str], y_slm: List[str]
    ) -> Dict[str, float]:
        """
        Computes SLM override dynamics with respect to XGBoost:
        - Override Rate = fraction where y_slm != y_ml
        - Correct Override Rate = fraction where y_ml != y_true and y_slm == y_true
        - Harmful Override Rate = fraction where y_ml == y_true and y_slm != y_true
        """
        total = len(y_true)
        if total == 0:
            return {"override_rate": 0.0, "correct_override": 0.0, "harmful_override": 0.0}

        overrides = 0
        correct_overrides = 0
        harmful_overrides = 0

        for t, m, s in zip(y_true, y_ml, y_slm):
            if s.lower() != m.lower():
                overrides += 1
                if m.lower() != t.lower() and s.lower() == t.lower():
                    correct_overrides += 1
                elif m.lower() == t.lower() and s.lower() != t.lower():
                    harmful_overrides += 1

        return {
            "override_rate": overrides / total,
            "correct_override": correct_overrides / total,
            "harmful_override": harmful_overrides / total,
            "net_gain": (correct_overrides - harmful_overrides) / total,
        }

    def compute_dsr(self, deadline_hits: List[bool]) -> float:
        if not deadline_hits:
            return 100.0
        return float(np.mean(deadline_hits) * 100.0)
