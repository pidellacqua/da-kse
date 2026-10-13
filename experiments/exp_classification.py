"""
Experiment: Classification and Diagnostic Quality Benchmark (Tables 3 & 4).
Benchmarks all 5 operational pipelines across Phi-3.5-Mini and Llama-3.2-3B:
1. ML-only (XGBoost)
2. ML + SLM (Parametric only)
3. ML + SLM + RAG (Flat Top-5)
4. ML + SLM + Hierarchical RAG (Fixed depth traversal)
5. ML + SLM + DA-KSE (Proposed)
"""

import time
from typing import Dict, List, Any
import numpy as np

from dakse.config import TARGET_CLASSES
from dakse.core.knowledge_space import DirectedKnowledgeGraph
from dakse.core.explorer import DAKSEExplorer
from dakse.core.baselines import FlatRAGRetriever, HierarchicalRAGRetriever
from dakse.classifier.xgboost_nids import XGBoostNIDSClassifier, NIDSPreprocessor
from dakse.slm.prompt import DiagnosticPromptBuilder
from dakse.slm.engine import create_slm_engine
from dakse.eval.metrics import DiagnosticMetricsCalculator
from dakse.eval.judge import RubricDiagnosticJudge


def run_classification_benchmark(
    dataset,
    graph: DirectedKnowledgeGraph,
    models: List[str] = ["Phi-3.5-Mini-Instruct", "Llama-3.2-3B-Instruct"],
    platform_name: str = "Raspberry Pi 5",
    test_limit: int = 300,
    deadline_sec: float = 2.0,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(" EXPERIMENT 1: CLASSIFICATION AND DIAGNOSTIC QUALITY (TABLES 3 & 4)")
    print(f" Platform: {platform_name} | Deadline: {deadline_sec}s | Samples: {test_limit}")
    print("=" * 80)

    # 1. Train Preprocessor & Classifier
    prep = NIDSPreprocessor()
    prep.fit(dataset.train_samples)
    classifier = XGBoostNIDSClassifier(TARGET_CLASSES)
    X_train = prep.transform(dataset.train_samples[:5000])
    y_train = [s["type"] for s in dataset.train_samples[:5000]]
    classifier.fit(X_train, y_train)

    test_samples = dataset.eval_subset_300[:test_limit] if test_limit else dataset.test_samples
    X_test = prep.transform(test_samples)
    y_true = [s["type"] for s in test_samples]

    # Pre-compute XGBoost predictions
    ml_preds = []
    ml_confs = []
    ml_prob_dicts = []
    for i in range(len(test_samples)):
        pred, conf, probs = classifier.predict_sample(X_test[i], ground_truth=y_true[i])
        ml_preds.append(pred)
        ml_confs.append(conf)
        ml_prob_dicts.append(probs)

    metrics_calc = DiagnosticMetricsCalculator(TARGET_CLASSES)
    judge = RubricDiagnosticJudge()

    # Baseline ML-only metrics
    ml_summary = metrics_calc.compute_classification_summary(y_true, ml_preds)
    print(f"\n[Baseline] ML-only (XGBoost): Acc: {ml_summary['accuracy']:.3f} | Macro-F1: {ml_summary['macro_f1']:.3f} | Weighted-F1: {ml_summary['weighted_f1']:.3f}")

    dakse_exp = DAKSEExplorer(graph=graph)
    flat_rag = FlatRAGRetriever(graph=graph)
    hier_rag = HierarchicalRAGRetriever(graph=graph)

    results_table = []

    for model_name in models:
        print(f"\n>>> Evaluating SLM: {model_name} on {platform_name}...")
        slm = create_slm_engine(model_name=model_name, platform_name=platform_name, use_mock=True)

        configs = ["ML + SLM", "ML + SLM + RAG", "ML + SLM + Hier. RAG", "ML + SLM + DA-KSE"]

        for cfg in configs:
            y_slm_preds = []
            explanations = []
            has_knowledge_flags = []

            for i, sample in enumerate(test_samples):
                ml_class = ml_preds[i]
                ml_conf = ml_confs[i]
                probs = ml_prob_dicts[i]
                query_text = f"{sample.get('anomaly_indicators', '')} {sample.get('proto', '')} port {sample.get('service', '')}"
                # Sample prototype query vector
                domain_u = graph.get_units_by_class(ml_class)
                q_emb = domain_u[0].embedding if domain_u else None

                # Construct knowledge context
                if cfg == "ML + SLM":
                    ctx_text = ""
                    has_ctx = False
                elif cfg == "ML + SLM + RAG":
                    ret_res = flat_rag.retrieve(query_text, q_emb, deadline_sec=deadline_sec)
                    ctx_text = ret_res.context_text
                    has_ctx = bool(ret_res.context_units)
                elif cfg == "ML + SLM + Hier. RAG":
                    ret_res = hier_rag.retrieve(query_text, q_emb, ml_class, deadline_sec=deadline_sec)
                    ctx_text = ret_res.context_text
                    has_ctx = bool(ret_res.context_units)
                else:  # DA-KSE
                    ret_res = dakse_exp.explore(query_text, q_emb, probs, deadline_sec=deadline_sec)
                    ctx_text = ret_res.context_text
                    has_ctx = bool(ret_res.context_units)

                prompt = DiagnosticPromptBuilder.build_prompt(
                    network_evidence=sample,
                    ml_class=ml_class,
                    ml_confidence=ml_conf,
                    knowledge_context=ctx_text,
                )

                resp = slm.generate(prompt, ml_class, ml_conf)
                y_slm_preds.append(resp.final_diagnosis)
                explanations.append(resp.explanation)
                has_knowledge_flags.append(has_ctx)

            # Metrics
            summary = metrics_calc.compute_classification_summary(y_true, y_slm_preds)
            overrides = metrics_calc.compute_override_metrics(y_true, ml_preds, y_slm_preds)
            quality = judge.evaluate_batch(y_true, y_slm_preds, explanations, test_samples, has_knowledge_flags)

            row = {
                "model": model_name,
                "config": cfg,
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "weighted_f1": summary["weighted_f1"],
                "override_rate": overrides["override_rate"] * 100.0,
                "correct_override": overrides["correct_override"] * 100.0,
                "harmful_override": overrides["harmful_override"] * 100.0,
                "overall_quality": quality["overall_quality"],
            }
            results_table.append(row)

            print(
                f"  {cfg:26s} | Acc: {row['accuracy']:.3f} | Macro-F1: {row['macro_f1']:.3f} | "
                f"Overrides: {row['override_rate']:.1f}% (Corr: {row['correct_override']:.1f}%, Harm: {row['harmful_override']:.1f}%) | "
                f"Diag Qual: {row['overall_quality']:.2f}"
            )

    return {"ml_baseline": ml_summary, "benchmark_rows": results_table}
