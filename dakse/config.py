"""
Configuration and reproducibility hyperparameters for DA-KSE.
All values are strictly aligned with the specifications in the DA-KSE paper
(Table 1, Table 2, Table 3, and Table 11).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 10 target classes from TON_IoT network flow dataset
TARGET_CLASSES: List[str] = [
    "Benign",
    "DDoS",
    "DoS",
    "Scanning",
    "Password",
    "Injection",
    "XSS",
    "Malware",
    "Ransomware",
    "Backdoor",
]

# Exact class supports on 100k subset
CLASS_SUPPORTS_TOTAL: Dict[str, int] = {
    "Benign": 35000,
    "DDoS": 15000,
    "DoS": 12000,
    "Scanning": 10000,
    "Password": 8000,
    "Injection": 7000,
    "XSS": 5000,
    "Malware": 4000,
    "Ransomware": 2500,
    "Backdoor": 1500,
}

# 15% hold-out test set supports (N = 15,000)
CLASS_SUPPORTS_TEST: Dict[str, int] = {
    "Benign": 5250,
    "DDoS": 2250,
    "DoS": 1800,
    "Scanning": 1500,
    "Password": 1200,
    "Injection": 1050,
    "XSS": 750,
    "Malware": 600,
    "Ransomware": 375,
    "Backdoor": 225,
}

# XGBoost Hyperparameters (Table 2)
XGBOOST_CONFIG = {
    "objective": "multi:softprob",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 50,
    "random_state": 42,
}

# Utility Function Weights (Table 11)
DEFAULT_ALPHA: float = 0.20   # Lexical keyword matching
DEFAULT_BETA: float = 0.30    # Dense semantic embedding cosine similarity
DEFAULT_GAMMA: float = 0.20   # Graph topological proximity
DEFAULT_DELTA: float = 0.30   # XGBoost classifier guidance

# Stopping Threshold (Table 11)
DEFAULT_THETA: float = 0.35

# Knowledge space parameters
MAX_HIERARCHY_LEVELS: int = 4  # 0: Domain/Family, 1: Section, 2: Paragraph, 3: Cross-ref
MAX_INITIAL_FRONTIER: int = 8  # Top-3 predicted attack pages + Top-5 semantic neighbours
RAG_TOP_K: int = 5             # Baseline Flat RAG fixed top-k
MAX_CONTEXT_TOKENS: int = 2048 # SLM context window W
MAX_KNOWLEDGE_TOKENS: int = 1580 # Effective knowledge capacity after system instructions
MAX_NEW_TOKENS: int = 256      # Maximum generation ceiling

# Evaluated Deadlines in seconds (None represents 'No Deadline')
EVAL_DEADLINES: List[Optional[float]] = [None, 4.0, 2.0, 1.0, 0.5]

# Evaluated SLM models
SUPPORTED_SLMS: List[str] = [
    "Phi-3.5-Mini-Instruct",
    "Llama-3.2-3B-Instruct",
]


@dataclass
class PlatformCostProfile:
    """Hardware cost model parameters as profiled in Section 4.5 and Table 11."""
    platform_name: str
    level_overhead_ms: float      # c_level per granularity level l (ms)
    token_cost_ms: float          # c_tok per token (ms)
    op_costs_ms: Dict[str, float] # c_op per action type (ms)
    average_power_w: float        # Profiled active power load (W)
    gpu_layers_phi: int           # -ngl for Phi-3.5
    gpu_layers_llama: int         # -ngl for Llama-3.2


# Platform Profiles
RPI5_PROFILE = PlatformCostProfile(
    platform_name="Raspberry Pi 5",
    level_overhead_ms=15.0,
    token_cost_ms=0.040,
    op_costs_ms={
        "Visit": 12.0,
        "Expand": 22.0,
        "Follow": 16.0,
        "Stop": 0.0,
    },
    average_power_w=8.2,
    gpu_layers_phi=0,   # ARM Neon CPU only
    gpu_layers_llama=0, # ARM Neon CPU only
)

JETSON_PROFILE = PlatformCostProfile(
    platform_name="NVIDIA Jetson Orin Nano",
    level_overhead_ms=8.0,
    token_cost_ms=0.015,
    op_costs_ms={
        "Visit": 4.5,
        "Expand": 8.0,
        "Follow": 6.0,
        "Stop": 0.0,
    },
    average_power_w=13.4,
    gpu_layers_phi=33,   # Full CUDA offload
    gpu_layers_llama=28, # Full CUDA offload
)

GENERIC_PROFILE = PlatformCostProfile(
    platform_name="Generic Workstation",
    level_overhead_ms=5.0,
    token_cost_ms=0.010,
    op_costs_ms={
        "Visit": 3.0,
        "Expand": 5.0,
        "Follow": 4.0,
        "Stop": 0.0,
    },
    average_power_w=25.0,
    gpu_layers_phi=33,
    gpu_layers_llama=28,
)


def get_platform_profile(name: str) -> PlatformCostProfile:
    name_clean = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    if "rpi" in name_clean or "raspberry" in name_clean or "pi5" in name_clean:
        return RPI5_PROFILE
    elif "jetson" in name_clean or "orin" in name_clean:
        return JETSON_PROFILE
    else:
        return GENERIC_PROFILE
