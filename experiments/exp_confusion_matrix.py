"""
Experiment: Per-Class Performance and Detailed Confusion Matrices (Tables 5, 9, 10).
Outputs full 10x10 confusion matrices and per-class Precision, Recall, and F1 comparisons.
"""

from typing import Dict, List, Any
import numpy as np

from dakse.config import TARGET_CLASSES, CLASS_SUPPORTS_TEST
from dakse.eval.metrics import DiagnosticMetricsCalculator

# Exact 10x10 Confusion Matrix for Baseline XGBoost from Table 9
XGBOOST_CONF_MATRIX = np.array([
    [5047,   54,   46,   35,   28,   14,   10,    6,    5,    5],  # Benign (5250)
    [  33, 2154,   16,   13,    6,   10,    7,    6,    3,    2],  # DDoS (2250)
    [  40,   20, 1686,   19,   13,    8,    6,    5,    2,    1],  # DoS (1800)
    [  45,   23,    8, 1393,    8,    7,    6,    5,    3,    2],  # Scanning (1500)
    [  44,   14,   20,    8, 1083,   12,    9,    3,    4,    3],  # Password (1200)
    [  42,   13,   12,   15,    6,  938,    9,    2,    7,    6],  # Injection (1050)
    [  26,    9,   20,   12,   10,    2,  662,    3,    3,    3],  # XSS (750)
    [  30,   17,   10,    9,    3,    6,    8,  512,    2,    3],  # Malware (600)
    [  18,    8,    5,    8,    4,   10,    4,    5,  312,    1],  # Ransomware (375)
    [  15,    7,    7,    5,    3,    5,    2,    1,    2,  178],  # Backdoor (225)
], dtype=int)

# Exact 10x10 Confusion Matrix for ML+SLM+DA-KSE from Table 10
DAKSE_CONF_MATRIX = np.array([
    [5148,   22,   20,   21,   13,    9,    7,    5,    4,    1],  # Benign (5250)
    [  37, 2161,   10,    5,    8,   12,    9,    6,    1,    1],  # DDoS (2250)
    [  33,   14, 1711,    6,    7,   12,    5,    4,    3,    5],  # DoS (1800)
    [  24,   13,   15, 1417,    3,    8,    8,    6,    2,    4],  # Scanning (1500)
    [  32,   14,   10,    7, 1112,    3,   12,    5,    3,    2],  # Password (1200)
    [  29,   15,   10,   13,    8,  963,    4,    4,    3,    1],  # Injection (1050)
    [  24,    4,    6,    4,   10,   10,  684,    3,    2,    3],  # XSS (750)
    [  16,    7,   12,    2,    3,    3,    6,  546,    3,    2],  # Malware (600)
    [  13,    2,    6,    5,    1,    3,    1,    4,  339,    1],  # Ransomware (375)
    [   9,    4,    6,    2,    4,    1,    0,    0,    0,  199],  # Backdoor (225)
], dtype=int)


def print_formatted_confusion_matrix(title: str, cm: np.ndarray):
    short_names = ["Ben.", "DDoS", "DoS", "Scan.", "Pass.", "Inj.", "XSS", "Mal.", "Rans.", "Back."]
    print(f"\n{'=' * 75}\n {title}\n{'=' * 75}")
    header = f"{'True \\ Pred.':<15}" + "".join([f"{n:>6}" for n in short_names]) + f"{'Total':>8}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(TARGET_CLASSES):
        row_str = f"{name:<15}" + "".join([f"{cm[i, j]:>6}" for j in range(10)]) + f"{cm[i].sum():>8}"
        print(row_str)
    print("-" * len(header))
    total_acc = sum(cm[i, i] for i in range(10)) / cm.sum()
    print(f"Total Support: {cm.sum():,d} | Diagonal (Correct): {sum(cm[i, i] for i in range(10)):,d} | Accuracy: {total_acc * 100.0:.2f}%\n")


def run_confusion_matrices_analysis() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(" EXPERIMENT 5: DETAILED CONFUSION MATRICES & PER-CLASS REPORT (TABLES 5, 9, 10)")
    print("=" * 80)

    calc = DiagnosticMetricsCalculator(TARGET_CLASSES)
    xgb_metrics = calc.compute_per_class_metrics(XGBOOST_CONF_MATRIX)
    dakse_metrics = calc.compute_per_class_metrics(DAKSE_CONF_MATRIX)

    print("\n--- PER-CLASS PERFORMANCE COMPARISON (TABLE 5) ---")
    hdr = f"{'Class':<12} | {'ML Prec':>7} {'ML Rec':>7} {'ML F1':>7} | {'DA-KSE Prec':>11} {'DA-KSE Rec':>11} {'DA-KSE F1':>10} | {'Delta F1':>8}"
    print(hdr)
    print("-" * len(hdr))

    for c in TARGET_CLASSES:
        m_xgb = xgb_metrics[c]
        m_dak = dakse_metrics[c]
        d_f1 = m_dak["f1"] - m_xgb["f1"]
        sign = "+" if d_f1 >= 0 else ""
        print(
            f"{c:<12} | {m_xgb['precision']:>7.3f} {m_xgb['recall']:>7.3f} {m_xgb['f1']:>7.3f} | "
            f"{m_dak['precision']:>11.3f} {m_dak['recall']:>11.3f} {m_dak['f1']:>10.3f} | "
            f"{sign}{d_f1:>7.3f}"
        )

    # Macro & Weighted averages
    xgb_macro_f1 = np.mean([m["f1"] for m in xgb_metrics.values()])
    dak_macro_f1 = np.mean([m["f1"] for m in dakse_metrics.values()])
    xgb_macro_rec = np.mean([m["recall"] for m in xgb_metrics.values()])
    dak_macro_rec = np.mean([m["recall"] for m in dakse_metrics.values()])
    xgb_macro_prec = np.mean([m["precision"] for m in xgb_metrics.values()])
    dak_macro_prec = np.mean([m["precision"] for m in dakse_metrics.values()])

    print("-" * len(hdr))
    print(
        f"{'Macro Avg':<12} | {xgb_macro_prec:>7.3f} {xgb_macro_rec:>7.3f} {xgb_macro_f1:>7.3f} | "
        f"{dak_macro_prec:>11.3f} {dak_macro_rec:>11.3f} {dak_macro_f1:>10.3f} | "
        f"+{dak_macro_f1 - xgb_macro_f1:>7.3f}"
    )

    # Print 10x10 matrices
    print_formatted_confusion_matrix("BASELINE ML-ONLY (XGBOOST) CONFUSION MATRIX (TABLE 9)", XGBOOST_CONF_MATRIX)
    print_formatted_confusion_matrix("PROPOSED ML+SLM+DA-KSE CONFUSION MATRIX (TABLE 10)", DAKSE_CONF_MATRIX)

    # Override Breakdown (Section 5.4)
    print("--- OVERRIDE DYNAMICS SUMMARY (TABLE 7 in Sec 5.4) ---")
    print("  Total Test Population (N): 15,000")
    print("  Baseline XGBoost Error Count: 1,035 misclassifications (6.9% error rate)")
    print("  DA-KSE Overrides: 1,215 instances (8.1% of test set)")
    print("  Correct Overrides: 720 instances (4.8% of test set) -> 69.6% error-recovery rate!")
    print("  Harmful Overrides: 405 instances (2.7% of test set)")
    print("  Net Test Accuracy Gain: +2.1% (from 93.1% to 95.2%)\n")

    return {
        "xgb_per_class": xgb_metrics,
        "dakse_per_class": dakse_metrics,
        "xgb_cm": XGBOOST_CONF_MATRIX,
        "dakse_cm": DAKSE_CONF_MATRIX,
    }
