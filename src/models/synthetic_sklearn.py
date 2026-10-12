import os
import random
from dataclasses import dataclass

import joblib
import numpy as np

from consumer.feature_extractor import extract_window_features, N_FEATURES


@dataclass(frozen=True)
class SyntheticModelArtifacts:
    pipeline_path: str


def _generate_flow_record(mode: str) -> dict:
    # Mirrors src/producer/flow_producer.py distributions.
    if mode == "attack":
        dport = random.choice([23, 445, 3389, 80, 8080])
        flags = random.choices([2, 18, 16, 0], weights=[0.65, 0.15, 0.15, 0.05], k=1)[0]
        byte_count = random.randint(40, 400)
    elif mode == "scan":
        # Port-scan-like: many unique destination ports, mostly SYN, small payloads.
        # We choose dport outside common service ports to create high unique_dports.
        dport = random.randint(1, 65535)
        flags = random.choices([2, 18, 16, 0], weights=[0.9, 0.02, 0.06, 0.02], k=1)[0]
        byte_count = random.randint(40, 120)
    else:
        dport = random.choice([80, 443, 22, 8080])
        flags = random.choices([16, 18, 24, 2, 0], weights=[0.45, 0.25, 0.15, 0.1, 0.05], k=1)[0]
        byte_count = random.randint(60, 1500)

    return {
        "src": f"10.0.0.{random.randint(2, 200)}",
        "dst": f"10.0.0.{random.randint(2, 10)}",
        "sport": random.randint(1024, 65535),
        "dport": dport,
        "proto": 6,
        "flags": flags,
        "bytes": byte_count,
        "mode": mode,
    }


def _generate_window(mode: str, n_flows: int) -> list[dict]:
    return [_generate_flow_record(mode) for _ in range(n_flows)]


def _build_training_set(
    n_windows_per_class: int,
    min_flows: int,
    max_flows: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[str] = []

    for mode, label in (("benign", "benign"), ("attack", "attack"), ("scan", "scan")):
        for _ in range(n_windows_per_class):
            n_flows = random.randint(min_flows, max_flows)
            flows = _generate_window(mode, n_flows=n_flows)
            feat = extract_window_features(flows)
            if feat.shape != (N_FEATURES,):
                feat = np.asarray(feat, dtype=float).reshape(-1)
            xs.append(feat.astype(float))
            ys.append(label)

    x = np.stack(xs, axis=0)
    y = np.asarray(ys)
    return x, y


def train_and_save_synthetic_pipeline(
    models_dir: str,
    n_windows_per_class: int = 3000,
    min_flows: int = 10,
    max_flows: int = 60,
    random_seed: int = 42,
) -> str:
    """Train a fast sklearn classifier on synthetic-window features.

    This is meant to be a practical ML detector for demos.
    It learns:
    - benign vs attack distributions used by the synthetic producer
    - a port-scan-like pattern (many unique dports + mostly SYN)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    os.makedirs(models_dir, exist_ok=True)
    pipeline_path = os.path.join(models_dir, "synthetic_pipeline.joblib")

    random.seed(random_seed)
    np.random.seed(random_seed)

    x, y = _build_training_set(
        n_windows_per_class=n_windows_per_class,
        min_flows=min_flows,
        max_flows=max_flows,
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipe.fit(x, y)

    joblib.dump(
        {
            "pipeline": pipe,
            "n_features": int(N_FEATURES),
            "classes": list(getattr(pipe[-1], "classes_", [])),
            "random_seed": int(random_seed),
            "note": "Trained to detect benign vs attack vs scan from window features.",
        },
        pipeline_path,
    )
    return pipeline_path


def load_or_train_synthetic_pipeline(models_dir: str) -> dict:
    """Returns dict with keys: pipeline, n_features, classes."""
    pipeline_path = os.path.join(models_dir, "synthetic_pipeline.joblib")
    if os.path.exists(pipeline_path):
        bundle = joblib.load(pipeline_path)
        classes = [str(c) for c in (bundle.get("classes") or [])]
        # If this is an older 2-class model, retrain so we can also detect scan windows.
        if "scan" in classes:
            return bundle

    # Keep training time reasonable on first run.
    created = train_and_save_synthetic_pipeline(
        models_dir=models_dir,
        n_windows_per_class=2000,
        min_flows=10,
        max_flows=60,
        random_seed=42,
    )
    return joblib.load(created)
