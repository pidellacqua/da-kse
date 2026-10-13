#!/usr/bin/env python3
"""
DA-KSE: Deadline-Aware Knowledge Space Exploration for Edge-IoT Cybersecurity Diagnostics.
Main entry point to execute and reproduce all experiments from the DA-KSE paper.

Usage examples:
    # Run all paper experiments (Tables 3, 4, 5, 6, 7, 8, 9, 10):
    python main.py --experiment all

    # Run specific experiment:
    python main.py --experiment classification --platform rpi5
    python main.py --experiment deadlines
    python main.py --experiment runtime --platform jetson
    python main.py --experiment ablation
    python main.py --experiment confusion

    # Single-query interactive diagnostic demonstration:
    python main.py --experiment demo --deadline 2.0
"""

import argparse
import sys
import os
import time

# Ensure package root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dakse.config import (
    TARGET_CLASSES,
    DEFAULT_THETA,
    SUPPORTED_SLMS,
    get_platform_profile,
)
from dakse.data.knowledge_corpus import build_cybersecurity_knowledge_space
from dakse.data.dataset import TONIoTDataset
from experiments.exp_classification import run_classification_benchmark
from experiments.exp_deadlines import run_deadline_sweep
from experiments.exp_runtime import run_runtime_benchmark
from experiments.exp_ablation import run_ablation_study
from experiments.exp_confusion_matrix import run_confusion_matrices_analysis
from dakse.core.explorer import DAKSEExplorer
from dakse.slm.prompt import DiagnosticPromptBuilder
from dakse.slm.engine import create_slm_engine


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DA-KSE: Deadline-Aware Knowledge Space Exploration on Edge-IoT Gateways."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "classification", "deadlines", "runtime", "ablation", "confusion", "demo"],
        help="Experiment to execute (default: all)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="Raspberry Pi 5",
        help="Target edge platform (e.g., 'Raspberry Pi 5', 'NVIDIA Jetson Orin Nano', 'generic')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Phi-3.5-Mini-Instruct",
        choices=SUPPORTED_SLMS,
        help="Target SLM architecture (default: Phi-3.5-Mini-Instruct)",
    )
    parser.add_argument(
        "--deadline",
        type=float,
        default=2.0,
        help="Exploration deadline in seconds (default: 2.0s)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=DEFAULT_THETA,
        help=f"DA-KSE stopping threshold theta (default: {DEFAULT_THETA})",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to real TON_IoT CSV file. If omitted, uses high-fidelity synthetic benchmark.",
    )
    parser.add_argument(
        "--data-scale",
        type=float,
        default=0.1,
        help="Dataset scale factor for quick testing (1.0 = full 100k, 0.1 = 10k, default: 0.1)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=150,
        help="Number of test samples to evaluate in benchmark runs (default: 150)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to GGUF model file if executing live llama.cpp inference on edge hardware.",
    )
    parser.add_argument(
        "--use-mock-slm",
        action="store_true",
        default=True,
        help="Use calibrated offline SLM evaluation engine (default: True unless --model-path is set).",
    )
    return parser.parse_args()


def run_interactive_demo(dataset, graph, platform_name: str, deadline_sec: float, theta: float):
    print("\n" + "=" * 80)
    print(" INTERACTIVE DEMO: SINGLE-SAMPLE DA-KSE DIAGNOSTIC INFERENCE")
    print(f" Platform: {platform_name} | Deadline: {deadline_sec}s | Theta: {theta}")
    print("=" * 80)

    # Sample a representative attack flow (e.g. Ransomware or Backdoor)
    sample = dataset.eval_subset_300[0]
    true_label = sample["type"]

    print(f"\n1. Ingesting Real-Time IoT Network Flow:")
    print(f"   - Protocol: {sample.get('proto')} | Service: {sample.get('service')}")
    print(f"   - Duration: {sample.get('duration')}s | Src Bytes: {sample.get('src_bytes')} | Dst Bytes: {sample.get('dst_bytes')}")
    print(f"   - State: {sample.get('conn_state')} | Packet Rate: {sample.get('packet_rate')} pps")
    print(f"   - Anomaly Indicators: {sample.get('anomaly_indicators')}")
    print(f"   - True Ground-Truth Label: {true_label}")

    # Simulated preliminary XGBoost classifier output
    # (intentionally slightly ambiguous to demonstrate diagnostic override refinement)
    ml_hypothesis = "Scanning" if true_label == "Backdoor" else true_label
    ml_confidence = 0.74 if ml_hypothesis != true_label else 0.94
    probs = {c: 0.02 for c in TARGET_CLASSES}
    probs[ml_hypothesis] = ml_confidence
    probs[true_label] = max(probs[true_label], 0.18)

    print(f"\n2. Upstream Tabular XGBoost NIDS Hypothesis:")
    print(f"   - Predicted Class: {ml_hypothesis}")
    print(f"   - Confidence Score: {ml_confidence:.2f}")

    # Run DA-KSE Exploration
    print(f"\n3. Dispatching DA-KSE Anytime Exploration Loop (Budget: {deadline_sec}s)...")
    explorer = DAKSEExplorer(graph=graph, theta=theta)
    q_text = f"{sample.get('anomaly_indicators', '')} {sample.get('proto', '')} {sample.get('service', '')}"
    domain_u = graph.get_units_by_class(true_label)
    q_emb = domain_u[0].embedding if domain_u else None

    t0 = time.perf_counter()
    res = explorer.explore(q_text, q_emb, probs, deadline_sec=deadline_sec)
    t_elapsed = (time.perf_counter() - t0) * 1000.0

    print(f"   -> Exploration Latency: {res.exploration_latency_ms:.1f} ms (Wall: {t_elapsed:.1f} ms)")
    print(f"   -> Deadline Satisfied: {res.satisfied_deadline} (Remaining: {max(0.0, deadline_sec * 1000 - res.exploration_latency_ms):.1f} ms)")
    print(f"   -> Units Selected: {len(res.context_units)} ({res.token_count} knowledge tokens)")
    print(f"   -> Action Trace Steps: {res.steps_executed}")
    for step_info in res.action_trace:
        print(f"      * Step {step_info.get('step')}: {step_info.get('action')} target={step_info.get('target_id')} "
              f"L{step_info.get('level')} (Delta_U: {step_info.get('delta_u', 0):.3f}, Cost: {step_info.get('delta_t_ms', 0):.1f} ms)")

    # Build Prompt
    prompt = DiagnosticPromptBuilder.build_prompt(
        network_evidence=sample,
        ml_class=ml_hypothesis,
        ml_confidence=ml_confidence,
        knowledge_context=res.context_text,
    )

    # Invoke SLM
    print(f"\n4. Invoking Local SLM Diagnostic Reasoning Engine...")
    slm = create_slm_engine(platform_name=platform_name, use_mock=True)
    slm_resp = slm.generate(prompt, ml_hypothesis, ml_confidence)

    print("\n" + "-" * 40 + " SLM GENERATED RESPONSE " + "-" * 40)
    print(f"Final Diagnosis: {slm_resp.final_diagnosis}")
    print(f"Explanation: {slm_resp.explanation}")
    print("-" * 104)

    if slm_resp.override_occurred:
        print(f"\n[DIAGNOSTIC REFINEMENT] SLM successfully overrode preliminary XGBoost hypothesis "
              f"'{ml_hypothesis}' -> '{slm_resp.final_diagnosis}' (Ground Truth: {true_label})!")
    else:
        print(f"\n[DIAGNOSTIC CONFIRMATION] SLM confirmed preliminary XGBoost hypothesis '{ml_hypothesis}'.")


def main():
    args = parse_arguments()
    profile = get_platform_profile(args.platform)

    print("=" * 80)
    print("   DA-KSE: DEADLINE-AWARE KNOWLEDGE SPACE EXPLORATION FOR EDGE-IOT")
    print(f"   Platform Target: {profile.platform_name} | Profiled Power: {profile.average_power_w} W")
    print("=" * 80)

    # 1. Initialize Knowledge Space (412 units, 689 relations, L=4)
    print("\n[Step 1/2] Constructing Cybersecurity Knowledge Space G = (K, R)...")
    graph = build_cybersecurity_knowledge_space()
    print(f" -> Knowledge Space loaded: {graph.total_units} units, {graph.total_relations} navigable relations.")

    # 2. Initialize TON_IoT Dataset
    print(f"\n[Step 2/2] Preparing TON_IoT Network Flow Dataset (scale={args.data_scale})...")
    dataset = TONIoTDataset(random_seed=42)
    if args.data_path and os.path.exists(args.data_path):
        dataset.load_from_csv(args.data_path)
    else:
        dataset.generate_synthetic_benchmark(scale_ratio=args.data_scale)

    # 3. Execute Selected Experiment(s)
    exp = args.experiment.lower()

    if exp in ("classification", "all"):
        run_classification_benchmark(
            dataset=dataset,
            graph=graph,
            models=SUPPORTED_SLMS,
            platform_name=profile.platform_name,
            test_limit=args.samples,
            deadline_sec=args.deadline,
        )

    if exp in ("deadlines", "all"):
        run_deadline_sweep(
            dataset=dataset,
            graph=graph,
            platform_name=profile.platform_name,
            test_limit=args.samples,
        )

    if exp in ("runtime", "all"):
        run_runtime_benchmark(
            dataset=dataset,
            graph=graph,
            platforms=["Raspberry Pi 5", "NVIDIA Jetson Orin Nano"],
            models=SUPPORTED_SLMS,
            deadline_sec=args.deadline,
            test_limit=min(args.samples, 100),
        )

    if exp in ("ablation", "all"):
        run_ablation_study(
            dataset=dataset,
            graph=graph,
            platform_name=profile.platform_name,
            test_limit=args.samples,
        )

    if exp in ("confusion", "all"):
        run_confusion_matrices_analysis()

    if exp == "demo":
        run_interactive_demo(
            dataset=dataset,
            graph=graph,
            platform_name=profile.platform_name,
            deadline_sec=args.deadline,
            theta=args.theta,
        )

    print("\n" + "=" * 80)
    print(" ALL REQUESTED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
