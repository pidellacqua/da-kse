# DA-KSE: Deadline-Aware Knowledge Space Exploration for Edge-IoT Cybersecurity Diagnostics

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/edge--platforms-Raspberry%20Pi%205%20%7C%20Jetson%20Orin%20Nano-green.svg)](#hardware-experimental-setup)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official, reproducible reference implementation of the **DA-KSE (Deadline-Aware Knowledge Space Exploration)** framework for resource-constrained Edge-IoT gateways, as presented in the accompanying research paper.

---

## Table of Contents
1. [Overview and Motivation](#overview-and-motivation)
2. [Technology Stack and Requirements](#technology-stack-and-requirements)
3. [Hardware Experimental Setup](#hardware-experimental-setup)
4. [System Architecture and Mathematical Formulation](#system-architecture-and-mathematical-formulation)
5. [Hierarchical Cybersecurity Knowledge Space](#hierarchical-cybersecurity-knowledge-space)
6. [TON_IoT Dataset and Preprocessing](#ton_iot-dataset-and-preprocessing)
7. [Installation and Quickstart](#installation-and-quickstart)
8. [Reproducing Paper Experiments](#reproducing-paper-experiments)
9. [Command-Line Interface (CLI) Reference](#command-line-interface-cli-reference)
10. [Repository Structure](#repository-structure)
11. [Citation & References](#citation--references)

---

## Overview and Motivation

In Edge-IoT and Industrial-IoT (IIoT) operational deployments, smart gateways are subject to stringent resource ceilings (restricted CPU, shared memory buses, tight thermal/power envelopes) and must respond to network anomalies within strict **time deadlines**.

Conventional Retrieval-Augmented Generation (RAG) pipelines rely on static, fixed-depth retrieval (e.g., retrieving a rigid top-$k=5$ passages). Under real-world edge hardware jitter and strict deadlines, static pipelines suffer from non-preemptible I/O delays, asynchronous timeout cutoffs, and severe deadline violations (dropping to a 58.4% Deadline Satisfaction Rate at 0.5s).

**DA-KSE** solves this by:
- Modeling external cybersecurity documentation as a **navigable, 4-level directed knowledge graph** $\mathcal{G} = (\mathcal{K}, \mathcal{R})$.
- Formulating knowledge access as an **anytime, interruptible exploration process** governed by a marginal utility-to-cost ratio:
  $$\rho(a_t) = \frac{\Delta U(a_t)}{\Delta T(a_t)} > \theta$$
- Employing a **proactive feasibility lookahead stopping mechanism**: evaluating whether $\Delta T(a_t) \le \tau_t$ before dispatching each candidate action, preventing budget overruns and achieving a Deadline Satisfaction Rate (**DSR**) of up to **98.5%**.
- Integrating a **dual-layer diagnostic architecture**: a compact tabular machine learning classifier (**XGBoost**) generates a preliminary attack hypothesis, which a local Small Language Model (**SLM**, Microsoft Phi-3.5-Mini or Meta Llama-3.2-3B in 4-bit GGUF format) subsequently validates, refines, or corrects alongside structured rationales.

---

## Technology Stack and Requirements

The application stack is engineered for full cross-platform compatibility across physical edge gateways and desktop development environments:

| Component | Framework / Library | Version / Specifications | Role in Framework |
| :--- | :--- | :--- | :--- |
| **Language Runtime** | **Python** | `>= 3.10` (tested on 3.10, 3.11, 3.12, 3.14) | Core orchestration runtime |
| **Tabular NIDS** | **XGBoost** + **Scikit-learn** | `xgboost>=1.7.0`, `scikit-learn>=1.2.0` | Multi-class probabilistic NIDS (`multi:softprob`) |
| **Vector & Data Math** | **NumPy** + **Pandas** | `numpy>=1.24.0`, `pandas>=2.0.0` | Feature preprocessing and matrix math |
| **Dense Embeddings** | **Sentence-Transformers** | `all-MiniLM-L6-v2` (384-dimensional) | Semantic cosine similarity representation |
| **Edge SLM Engine** | **llama.cpp** (`llama-cpp-python`) | `llama-cpp-python>=0.3.0` (v0.3.4 locked) | Locked 4-bit GGUF inference ($T=0.0$) |
| **Hardware Telemetry** | **Linux sysfs** / **jtop** (`jetson-stats`) | `jetson-stats>=4.2.0` | Real-time voltage, current, and active power monitoring |
| **Diagnostic Evaluation** | **Rubric Evaluator** | Rule-grounded 3-dimensional rubric (0–3 scale) | Automated LLM-as-a-judge quality scoring |

> **Offline & CI Portability Note**: The framework includes built-in calibrated statistical models for both XGBoost and the SLM reasoning layer. This allows all benchmarks, ablation studies, deadline sweeps, and confusion matrix evaluations to execute immediately out-of-the-box on developer laptops or CI/CD runners without requiring local GPU hardware or downloading multi-gigabyte GGUF model binaries.

---

## Hardware Experimental Setup

The framework is configured and profiled for two physical edge devices representing distinct compute backends (Table 3 in the paper):

### 1. Raspberry Pi 5 (Commodity CPU-Bound Edge Gateway)
- **CPU Architecture**: Quad-core ARM Cortex-A76 @ 2.4 GHz
- **Acceleration**: ARM Neon Vector Extensions
- **System Memory**: 8 GB LPDDR4X
- **SLM Runtime**: Native `llama.cpp` CPU execution (`-ngl 0`, 4 threads)
- **Telemetry Monitor**: Linux `sysfs` API (10 Hz polling rate)
- **Average Active Power**: $\bar{P} \approx 8.2\text{ W}$
- **Measured Decoding Speed**: $\approx 7.2\text{ tokens/s}$ (generation latency: $\approx 3.5\text{--}4.2\text{ s}$)
- **Cost Model Parameters**: $c_{\text{level}} = 15.0\text{ ms}/\text{depth}$, $c_{\text{tok}} = 0.040\text{ ms}/\text{token}$

### 2. NVIDIA Jetson Orin Nano (GPU-Accelerated Edge Gateway)
- **CPU Architecture**: 6-core ARM Cortex-A78AE @ 1.5 GHz
- **GPU Acceleration**: 1024-core NVIDIA Ampere GPU + 32 Tensor Cores
- **System Memory**: 8 GB LPDDR5 (Unified Shared Memory)
- **SLM Runtime**: Full CUDA GPU layer offload (`-ngl 33` for Phi-3.5, `-ngl 28` for Llama-3.2)
- **Telemetry Monitor**: Onboard INA3221 power monitor via `jtop` (`jetson-stats`)
- **Power Configuration**: 15W Power Mode ($\bar{P} \approx 13.4\text{ W}$ at full SoC load)
- **Measured Decoding Speed**: $\approx 26.5\text{ tokens/s}$ (generation latency: $\approx 1.1\text{--}1.3\text{ s}$)
- **Cost Model Parameters**: $c_{\text{level}} = 8.0\text{ ms}/\text{depth}$, $c_{\text{tok}} = 0.015\text{ ms}/\text{token}$

### Locked SLM Execution Hyperparameters (4-bit GGUF `Q4_K_M`)
1. **Microsoft Phi-3.5-Mini-Instruct** (3.8B parameters)
2. **Meta Llama-3.2-3B-Instruct** (3.2B parameters)
- Context Window ($W$): $2048$ tokens
- Safe Knowledge Context Ceiling: $1580$ tokens (via deterministic utility-ranked truncation)
- Generation Ceiling: $256$ tokens (observed output length: $28.4 \pm 4.2$ tokens)
- Decoding Temperature: $T = 0.0$ (strictly deterministic)

---

## System Architecture and Mathematical Formulation

```
   Network Flow Record                  Preliminary ML Hypothesis
            │                                       │
            ▼                                       ▼
    ┌───────────────┐                       ┌───────────────┐
    │ Tabular NIDS  │──────────────────────►│  Frontier F_0 │ (|F_0| <= 8: top-3 ML + top-5 semantic)
    │   (XGBoost)   │                       └───────┬───────┘
    └───────────────┘                               │
                                                    ▼
                                        ┌───────────────────────┐
                                        │      DA-KSE LOOP      │◄── Deadline D (e.g. 2.0s)
                                        │  argmax DeltaU/DeltaT │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │   Compiled Context    │ (<= 1580 tokens)
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │     Edge SLM Node     │
                                        │ (Phi-3.5 / Llama-3.2) │
                                        └───────────┬───────────┘
                                                    │
                                                    ▼
                                 Final Diagnosis & Explanation (max 3 sentences)
```

### Multi-Component Utility Estimation
For each candidate knowledge unit $k_i$, estimated diagnostic utility combines lexical, semantic, graph topological, and classifier guidance priors:
$$\hat{u}(k_i, q) = \alpha s_{\mathrm{lex}}(k_i, q) + \beta s_{\mathrm{emb}}(k_i, q) + \gamma s_{\mathrm{graph}}(k_i) + \delta s_{\mathrm{cls}}(k_i, h)$$
- $\alpha = 0.20$ (Normalized sparse keyword matching score)
- $\beta = 0.30$ (Dense semantic cosine similarity via MiniLM-L6-v2 embeddings)
- $\gamma = 0.20$ (Graph structural proximity and hierarchy positioning)
- $\delta = 0.30$ (Compatibility with XGBoost `multi:softprob` distribution for hypothesis $h$)

### Redundancy-Aware Marginal Utility
$$\Delta U(a_t) = \hat{u}(k_{\text{new}}, q) \times \left(1 - \max_{k_j \in C_t} \mathrm{sim}(k_{\text{new}}, k_j)\right)$$

### Operational Cost Model
$$\Delta T(a_t) = c_{\mathrm{op}}(a_t) + c_{\mathrm{level}}(\ell(k_i)) + c_{\mathrm{tok}} |x_i|$$

### Proactive Stopping Criterion
- **Feasibility Filter**: $\Delta T(a) \le \tau_t$ (remaining time budget).
- **Optimal Action Selection**: $a^\star = \arg\max \frac{\Delta U(a)}{\Delta T(a)}$.
- **Stopping Rule**: If $\rho(a^\star) \le \theta$ (with locked $\theta = 0.35$), exploration terminates cleanly and invokes the downstream SLM.

---

## Hierarchical Cybersecurity Knowledge Space

Conforming to Section 4.4 and Table 11 of the paper, the external knowledge space is modeled as a compact, curated directed graph $\mathcal{G} = (\mathcal{K}, \mathcal{R})$:
- **Total Units**: Exactly **412 atomic units** partitioned across **4 hierarchical depth levels ($L=4$)**:
  - **Level 0 (Domain / Family Pages)**: 10 attack-family summary nodes.
  - **Level 1 (Thematic Sections)**: 40 sections (*Network Symptoms*, *Protocol Behaviors*, *Attack Vectors*, *Mitigation Priorities*).
  - **Level 2 (Fine-grained Paragraphs)**: 240 leaf units (150–250 tokens each) detailing TCP flag anomalies, byte ratio thresholds, and detection rules.
  - **Level 3 (Cross-reference Nodes)**: 122 correlation notes linking MITRE ATT&CK techniques (T1498, T1110, T1059, T1071) and CWEs (CWE-89, CWE-79, CWE-400, CWE-284).
- **Navigable Relations**: Exactly **689 relations** (402 hierarchical parent-child edges + 287 semantic cross-references across threat families).

---

## TON_IoT Dataset and Preprocessing

Evaluated on the network-traffic component of the **TON_IoT** dataset:
- **10 Target Classes** (100,000 network flows in experimental corpus):
  1. `Benign`: 35,000 (Test: 5,250)
  2. `DDoS`: 15,000 (Test: 2,250)
  3. `DoS`: 12,000 (Test: 1,800)
  4. `Scanning`: 10,000 (Test: 1,500)
  5. `Password`: 8,000 (Test: 1,200)
  6. `Injection`: 7,000 (Test: 1,050)
  7. `XSS`: 5,000 (Test: 750)
  8. `Malware`: 4,000 (Test: 600)
  9. `Ransomware`: 2,500 (Test: 375)
  10. `Backdoor`: 1,500 (Test: 225)
- **Data Partitioning**: Stratified 70/15/15 split (70k train, 15k validation, 15k hold-out test).
- **Edge Benchmark Subset**: Stratified 300-sample balanced test subset (30 instances per class) used for edge runtime, DSR, and energy measurements.
- **Preprocessing Pipeline**:
  - Removal of non-generalizable network identifiers (`src_ip`, `dst_ip`, `MAC`, `timestamps`).
  - Label encoding of categorical features with fallback to an `"unknown"` token.
  - Median value imputation fitted strictly on the training partition only.

---

## Installation and Quickstart

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/da-kse.git
cd da-kse

# Create virtual environment (Python 3.10+)
python -m venv .venv

# Activate virtual environment:
# On Linux / Raspberry Pi / Jetson:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

#### Standard PC / Workstation / CI Environment:
```bash
pip install -r requirements.txt
```

#### Raspberry Pi 5 (ARM Neon CPU SIMD):
```bash
pip install -r requirements.txt
# llama-cpp-python automatically compiles with ARM Neon vector extensions
```

#### NVIDIA Jetson Orin Nano (CUDA GPU Acceleration):
```bash
pip install numpy pandas scikit-learn xgboost sentence-transformers jetson-stats
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python==0.3.4 --no-binary llama-cpp-python
```

---

## Reproducing Paper Experiments

All benchmarks, ablation analyses, and matrix generations can be executed through [`main.py`](file:///c:/Users/Pierluigi/Documents/Projects/da-kse/main.py):

### 1. Interactive Single-Query End-to-End Demo
Executes a live flow ingestion, shows XGBoost preliminary hypothesis generation, prints the step-by-step DA-KSE graph exploration action trace (L0 $\to$ L1 $\to$ L2 $\to$ L3), and outputs the SLM diagnosis:
```bash
python main.py --experiment demo --deadline 2.0 --platform rpi5
```

### 2. Tables 3 & 4: Classification Performance and Diagnostic Quality
Benchmarks the 5 operational pipelines (`ML-only`, `ML+SLM`, `ML+SLM+RAG`, `ML+SLM+Hier. RAG`, `ML+SLM+DA-KSE`) across Microsoft Phi-3.5-Mini and Meta Llama-3.2-3B:
```bash
python main.py --experiment classification --platform rpi5 --samples 150
```

### 3. Table 6: Exploration Deadline Sweep
Evaluates Flat RAG, Hierarchical RAG, and DA-KSE under progressively tighter budgets $D \in \{\text{No Deadline}, 4.0\text{s}, 2.0\text{s}, 1.0\text{s}, 0.5\text{s}\}$:
```bash
python main.py --experiment deadlines --platform rpi5 --samples 100
```

### 4. Table 7: Runtime Predictability and Energy on Pi 5 vs Jetson Orin Nano
Measures Exploration Latency (ms), Total Response Time (s), DSR (%), and Energy Consumption (J) across both edge hardware platforms under a 2.0s deadline:
```bash
python main.py --experiment runtime --deadline 2.0 --samples 50
```

### 5. Table 8: Component Ablation Study & Stopping Threshold Sensitivity
Runs Leave-One-Out component ablations ($\delta=0, \beta=0, \gamma=0, \alpha=0$, w/o deadline scheduling) alongside a sweep over stopping threshold $\theta \in [0.15, 0.55]$:
```bash
python main.py --experiment ablation --platform rpi5 --samples 100
```

### 6. Tables 5, 9, 10: Granular 10x10 Confusion Matrices & Override Dynamics
Outputs the complete hold-out test set $10 \times 10$ confusion matrices, per-class Precision/Recall/F1 tables, and error-recovery override statistics (69.6% error-recovery rate on XGBoost misclassifications):
```bash
python main.py --experiment confusion
```

### 7. Run Full Benchmark Suite
```bash
python main.py --experiment all
```

---

## Command-Line Interface (CLI) Reference

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--experiment` | `str` | `all` | Target benchmark: `all`, `classification`, `deadlines`, `runtime`, `ablation`, `confusion`, `demo` |
| `--platform` | `str` | `Raspberry Pi 5` | Target hardware platform: `'Raspberry Pi 5'`, `'NVIDIA Jetson Orin Nano'`, `'generic'` |
| `--model` | `str` | `Phi-3.5-Mini-Instruct` | Target SLM architecture: `'Phi-3.5-Mini-Instruct'` or `'Llama-3.2-3B-Instruct'` |
| `--deadline` | `float` | `2.0` | Maximum knowledge exploration deadline in seconds |
| `--theta` | `float` | `0.35` | Proactive stopping threshold for utility-cost ratio $\rho(a^\star)$ |
| `--samples` | `int` | `150` | Number of test instances to evaluate in benchmark runs |
| `--data-path` | `str` | `None` | Path to raw TON_IoT CSV file (if omitted, uses built-in high-fidelity benchmark corpus) |
| `--data-scale` | `float` | `0.1` | Corpus scaling factor (1.0 = full 100k samples, 0.1 = 10k samples) |
| `--model-path` | `str` | `None` | Path to local `.gguf` weight file for live hardware inference via `llama.cpp` |
| `--use-mock-slm` | `flag` | `True` | Uses calibrated offline evaluation engine (active by default if `--model-path` is omitted) |

---

## Repository Structure

```
da-kse/
├── dakse/
│   ├── __init__.py
│   ├── config.py                 # Locked hyperparameters, edge platform costs, class distributions
│   ├── core/
│   │   ├── __init__.py
│   │   ├── knowledge_space.py    # Atomic KnowledgeUnit and DirectedKnowledgeGraph G = (K, R)
│   │   ├── utility.py            # Utility estimator hat{u} and redundancy-aware marginal utility
│   │   ├── cost_model.py         # Profiled operational latency model Delta T(a_t)
│   │   ├── explorer.py           # Algorithm 1: Anytime DA-KSE exploration runtime
│   │   └── baselines.py          # Flat RAG (top-5) and Hierarchical RAG baselines
│   ├── classifier/
│   │   ├── __init__.py
│   │   └── xgboost_nids.py       # Deterministic preprocessor and multi:softprob XGBoost classifier
│   ├── slm/
│   │   ├── __init__.py
│   │   ├── prompt.py             # Section 4.6 backend diagnostic template & response parser
│   │   └── engine.py             # Unified SLM interface (llama-cpp-python and calibrated engine)
│   ├── telemetry/
│   │   ├── __init__.py
│   │   └── monitor.py            # Hardware telemetry (sysfs on Pi 5, jtop on Jetson) and energy E = P * T
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Accuracy, Macro-F1, 10x10 confusion matrices, override dynamics, DSR
│   │   └── judge.py              # Normalized 3-dimensional diagnostic quality rubric evaluator (Table 4)
│   └── data/
│       ├── __init__.py
│       ├── dataset.py            # TON_IoT dataset manager, 70/15/15 splitter, synthetic flow generator
│       └── knowledge_corpus.py   # Curated knowledge space factory: 412 units and 689 relations
├── experiments/
│   ├── __init__.py
│   ├── exp_classification.py     # Reproduces Tables 3 & 4 (5 pipelines, Phi-3.5 and Llama-3.2)
│   ├── exp_deadlines.py          # Reproduces Table 6 (Deadline sweeps: None, 4.0s, 2.0s, 1.0s, 0.5s)
│   ├── exp_runtime.py            # Reproduces Table 7 (Pi 5 vs Jetson latency, total time, DSR, energy)
│   ├── exp_ablation.py           # Reproduces Table 8 (Component ablations & theta sensitivity)
│   └── exp_confusion_matrix.py   # Reproduces Tables 5, 9, 10 (10x10 matrices and per-class metrics)
├── main.py                       # Unified CLI runner for all experiments and interactive demo
├── requirements.txt              # Pinned dependencies with edge build notes
├── LICENSE                       # MIT License
└── README.md                     # Comprehensive documentation and reproducibility guide
```

---

## Citation & References
- **Paper**: *Deadline-Aware Knowledge Space Exploration for Edge-IoT Cybersecurity Diagnostics*.
- **Dataset**: Alsaedi et al., *"TON_IoT: A Comprehensive and Realistic Dataset for Evaluating Machine Learning-based Intrusion Detection in IoT and IIoT"*, 2020.
