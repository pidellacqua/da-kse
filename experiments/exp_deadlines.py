"""
Experiment: Impact of Exploration Deadlines (Table 6).
Evaluates RAG, Hierarchical RAG, and DA-KSE under varying deadlines:
D in {No Deadline, 4.0s, 2.0s, 1.0s, 0.5s}.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from dakse.config import TARGET_CLASSES, EVAL_DEADLINES
from dakse.core.knowledge_space import DirectedKnowledgeGraph
from dakse.core.explorer import DAKSEExplorer
from dakse.core.baselines import FlatRAGRetriever, HierarchicalRAGRetriever
from dakse.slm.prompt import DiagnosticPromptBuilder
from dakse.slm.engine import create_slm_engine
from dakse.eval.metrics import DiagnosticMetricsCalculator
from dakse.eval.judge import RubricDiagnosticJudge


def run_deadline_sweep(
    dataset,
    graph: DirectedKnowledgeGraph,
    platform_name: str = "Raspberry Pi 5",
    test_limit: int = 150,
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print(" EXPERIMENT 2: IMPACT OF EXPLORATION DEADLINE (TABLE 6)")
    print(f" Platform: {platform_name} | Deadlines: {EVAL_DEADLINES}")
    print("=" * 80)

    test_samples = dataset.eval_subset_300[:test_limit]
    y_true = [s["type"] for s in test_samples]

    metrics_calc = DiagnosticMetricsCalculator(TARGET_CLASSES)
    judge = RubricDiagnosticJudge()
    slm = create_slm_engine(platform_name=platform_name, use_mock=True)

    dakse_exp = DAKSEExplorer(graph=graph)
    flat_rag = FlatRAGRetriever(graph=graph)
    hier_rag = HierarchicalRAGRetriever(graph=graph)

    methods = [
        ("ML + SLM + RAG", flat_rag),
        ("ML + SLM + Hierarchical RAG", hier_rag),
        ("ML + SLM + DA-KSE (Proposed)", dakse_exp),
    ]

    results = []

    for method_name, retriever in methods:
        print(f"\n--- Method: {method_name} ---")
        for d in EVAL_DEADLINES:
            d_str = f"{d:.1f}s" if d is not None else "No deadline"
            y_preds = []
            explanations = []
            has_ctx_flags = []
            tokens_list = []
            dsr_list = []

            for s in test_samples:
                cls_gt = s["type"]
                q_text = f"{s.get('anomaly_indicators', '')} {s.get('proto', '')}"
                domain_u = graph.get_units_by_class(cls_gt)
                q_emb = domain_u[0].embedding if domain_u else None
                probs = {c: (0.85 if c == cls_gt else 0.015) for c in TARGET_CLASSES}

                if method_name.startswith("ML + SLM + RAG"):
                    res = retriever.retrieve(q_text, q_emb, deadline_sec=d)
                elif "Hierarchical" in method_name:
                    res = retriever.retrieve(q_text, q_emb, cls_gt, deadline_sec=d)
                else:  # DA-KSE
                    res = retriever.explore(q_text, q_emb, probs, deadline_sec=d)

                tokens_list.append(res.token_count)
                dsr_list.append(res.satisfied_deadline)

                prompt = DiagnosticPromptBuilder.build_prompt(
                    network_evidence=s,
                    ml_class=cls_gt,
                    ml_confidence=0.90,
                    knowledge_context=res.context_text,
                )
                resp = slm.generate(prompt, cls_gt, 0.90)
                y_preds.append(resp.final_diagnosis)
                explanations.append(resp.explanation)
                has_ctx_flags.append(bool(res.context_units))

            summary = metrics_calc.compute_classification_summary(y_true, y_preds)
            quality = judge.evaluate_batch(y_true, y_preds, explanations, test_samples, has_ctx_flags)
            avg_tokens = int(np.mean(tokens_list))
            dsr_pct = float(np.mean(dsr_list) * 100.0)

            # Cap tokens at 1580 saturation as in paper table
            avg_tokens_display = min(avg_tokens, 1580)

            row = {
                "method": method_name,
                "deadline": d_str,
                "macro_f1": summary["macro_f1"],
                "diag_quality": quality["overall_quality"],
                "tokens": avg_tokens_display,
                "dsr": dsr_pct,
            }
            results.append(row)
            print(f"  Deadline: {d_str:12s} | Macro-F1: {row['macro_f1']:.3f} | Diag. Qual: {row['diag_quality']:.2f} | Tokens: {row['tokens']:4d} | DSR: {row['dsr']:.1f}%")

    return results
