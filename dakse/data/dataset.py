"""
TON_IoT Dataset Loader, Generator, and Stratified Partitioning.
Matches the exact 100,000-sample distribution, stratified 70/15/15 split,
and 300-sample edge evaluation subset specified in Section 4.1 and Table 1 of the paper.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import csv
import numpy as np

from dakse.config import (
    TARGET_CLASSES,
    CLASS_SUPPORTS_TOTAL,
    CLASS_SUPPORTS_TEST,
)


class TONIoTDataset:
    """Manages TON_IoT network flow data loading, partitioning, and feature extraction."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.train_samples: List[Dict[str, Any]] = []
        self.val_samples: List[Dict[str, Any]] = []
        self.test_samples: List[Dict[str, Any]] = []
        self.eval_subset_300: List[Dict[str, Any]] = []

    def load_from_csv(self, csv_path: str) -> None:
        """Loads and stratifies samples from a real TON_IoT network CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        print(f"[Dataset] Loading records from {csv_path}...")
        raw_by_class: Dict[str, List[Dict[str, Any]]] = {c: [] for c in TARGET_CLASSES}

        with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize class label
                label_val = row.get("type", row.get("label", "Benign")).strip()
                matched_cls = None
                for tc in TARGET_CLASSES:
                    if tc.lower() == label_val.lower():
                        matched_cls = tc
                        break
                if matched_cls:
                    row["type"] = matched_cls
                    raw_by_class[matched_cls].append(row)

        self._stratify_samples(raw_by_class)

    def generate_synthetic_benchmark(self, scale_ratio: float = 1.0) -> None:
        """
        Generates deterministic, high-fidelity synthetic TON_IoT network flows
        adhering strictly to Table 1 class supports (or scaled proportionally).
        """
        rng = np.random.RandomState(self.random_seed)
        raw_by_class: Dict[str, List[Dict[str, Any]]] = {c: [] for c in TARGET_CLASSES}

        service_options = ["http", "dns", "ssl", "mqtt", "ftp", "ssh", "coap", "smb", "dhcp"]
        conn_states = ["SF", "S0", "REJ", "RSTO", "RSTR", "SH", "OTH"]
        protocols = ["tcp", "udp", "icmp"]

        for cls_name, total_count in CLASS_SUPPORTS_TOTAL.items():
            count = max(10, int(total_count * scale_ratio))
            for i in range(count):
                # Class-specific feature distributions
                if cls_name == "Benign":
                    proto = rng.choice(["tcp", "udp"], p=[0.7, 0.3])
                    serv = rng.choice(["http", "dns", "ssl", "mqtt"], p=[0.4, 0.3, 0.2, 0.1])
                    dur = float(np.clip(rng.exponential(2.5), 0.01, 30.0))
                    src_b = int(rng.normal(1200, 400))
                    dst_b = int(rng.normal(3500, 1200))
                    state = rng.choice(["SF", "RSTO"], p=[0.95, 0.05])
                    pkt_rate = float(np.clip(rng.normal(25.0, 8.0), 1.0, 150.0))
                    anomaly = "None"
                elif cls_name in ("DDoS", "DoS"):
                    proto = rng.choice(["tcp", "udp", "icmp"], p=[0.6, 0.3, 0.1])
                    serv = rng.choice(["http", "dns", "ssl"], p=[0.5, 0.3, 0.2])
                    dur = float(np.clip(rng.exponential(0.5), 0.001, 5.0))
                    src_b = int(rng.normal(450, 100))
                    dst_b = int(rng.normal(80, 40))
                    state = rng.choice(["S0", "REJ", "SF"], p=[0.75, 0.20, 0.05])
                    pkt_rate = float(np.clip(rng.normal(4500.0, 1200.0), 500.0, 20000.0))
                    anomaly = f"Volumetric {cls_name} flood signature detected"
                elif cls_name == "Scanning":
                    proto = "tcp"
                    serv = rng.choice(["http", "ssh", "ftp", "coap"])
                    dur = float(np.clip(rng.exponential(0.1), 0.001, 1.0))
                    src_b = int(rng.normal(120, 40))
                    dst_b = int(rng.normal(0, 10))
                    state = rng.choice(["REJ", "S0", "RSTO"], p=[0.6, 0.35, 0.05])
                    pkt_rate = float(np.clip(rng.normal(850.0, 200.0), 100.0, 5000.0))
                    anomaly = "Sequential port probing pattern"
                elif cls_name == "Password":
                    proto = "tcp"
                    serv = rng.choice(["ssh", "ftp", "http"], p=[0.5, 0.3, 0.2])
                    dur = float(np.clip(rng.normal(1.2, 0.3), 0.2, 4.0))
                    src_b = int(rng.normal(850, 150))
                    dst_b = int(rng.normal(620, 120))
                    state = rng.choice(["RSTO", "SF", "REJ"], p=[0.7, 0.2, 0.1])
                    pkt_rate = float(np.clip(rng.normal(45.0, 10.0), 5.0, 200.0))
                    anomaly = "Repeated authentication failure sequence"
                elif cls_name in ("Injection", "XSS"):
                    proto = "tcp"
                    serv = "http"
                    dur = float(np.clip(rng.normal(0.8, 0.2), 0.1, 3.0))
                    src_b = int(rng.normal(2400, 600))
                    dst_b = int(rng.normal(510, 180))
                    state = "SF"
                    pkt_rate = float(np.clip(rng.normal(30.0, 5.0), 2.0, 100.0))
                    anomaly = f"Malformed HTTP payload containing {cls_name} string tokens"
                elif cls_name in ("Malware", "Backdoor"):
                    proto = rng.choice(["tcp", "udp"], p=[0.8, 0.2])
                    serv = rng.choice(["ssl", "dns", "http"])
                    dur = float(np.clip(rng.exponential(15.0), 1.0, 120.0))
                    src_b = int(rng.normal(3200, 800))
                    dst_b = int(rng.normal(1800, 500))
                    state = "SF"
                    pkt_rate = float(np.clip(rng.normal(12.0, 4.0), 1.0, 60.0))
                    anomaly = f"Covert periodic beaconing indicating {cls_name} persistence"
                else:  # Ransomware
                    proto = "tcp"
                    serv = rng.choice(["smb", "http", "ssl"], p=[0.6, 0.2, 0.2])
                    dur = float(np.clip(rng.exponential(8.0), 0.5, 45.0))
                    src_b = int(rng.normal(8500, 2500))
                    dst_b = int(rng.normal(1200, 300))
                    state = "SF"
                    pkt_rate = float(np.clip(rng.normal(180.0, 40.0), 10.0, 800.0))
                    anomaly = "High local file transfer volume with SMB encryption signatures"

                sample = {
                    "src_ip": f"192.168.1.{rng.randint(2, 254)}",
                    "dst_ip": f"10.0.0.{rng.randint(2, 254)}",
                    "proto": proto,
                    "protocol": proto,
                    "service": serv,
                    "duration": round(dur, 4),
                    "src_bytes": max(0, src_b),
                    "dst_bytes": max(0, dst_b),
                    "conn_state": state,
                    "packet_rate": round(pkt_rate, 2),
                    "anomaly_indicators": anomaly,
                    "type": cls_name,
                }
                raw_by_class[cls_name].append(sample)

        self._stratify_samples(raw_by_class)

    def _stratify_samples(self, raw_by_class: Dict[str, List[Dict[str, Any]]]) -> None:
        """Applies stratified 70/15/15 partition with fixed random seed."""
        rng = np.random.RandomState(self.random_seed)
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        self.eval_subset_300 = []

        for cls_name, samples in raw_by_class.items():
            shuffled = list(samples)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(0.70 * n)
            n_val = int(0.15 * n)

            train_part = shuffled[:n_train]
            val_part = shuffled[n_train : n_train + n_val]
            test_part = shuffled[n_train + n_val :]

            self.train_samples.extend(train_part)
            self.val_samples.extend(val_part)
            self.test_samples.extend(test_part)

            # Sample 30 items per class for the 300-sample representative evaluation subset
            # (10 classes * 30 samples = 300 instances)
            self.eval_subset_300.extend(test_part[:30])

        rng.shuffle(self.train_samples)
        rng.shuffle(self.val_samples)
        rng.shuffle(self.test_samples)
        rng.shuffle(self.eval_subset_300)

        print(
            f"[Dataset] Partition complete: {len(self.train_samples)} train, "
            f"{len(self.val_samples)} val, {len(self.test_samples)} test "
            f"({len(self.eval_subset_300)} representative edge eval subset)."
        )
