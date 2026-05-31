"""
Export STGCN test-set predictions for TBRGS route integration.

Aggregates 140 directional node predictions to hourly flow per SCATS site,
using the same (site, hour) cache key pattern as LSTM/GRU in GUI.py.

Run from GNN_model/ after training and tensor generation:
    python export_routing_predictions.py

Output:
    ../models/artifacts_person1/predictions_stgcn_routing.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model import TrafficGNN

LAGS = 4
NUM_NODES = 140
HIDDEN_SIZE = 64

BASE_DIR = Path(__file__).resolve().parent
GEN_DIR = BASE_DIR / "GeneratedFiles"
ARTIFACTS_DIR = BASE_DIR.parent / "models" / "artifacts_person1"
OUTPUT_CSV = ARTIFACTS_DIR / "predictions_stgcn_routing.csv"


def normalize_adjacency(a: np.ndarray) -> np.ndarray:
    a_sym = a + np.eye(a.shape[0])
    rowsum = np.array(a_sym.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(a_sym).dot(d_mat_inv_sqrt)


def clean_scats(value) -> str:
    try:
        return str(int(float(str(value).strip())))
    except Exception:
        return str(value).strip()


def find_weights() -> Path | None:
    for name in ("GNN_trained.pth", "trained_stgcn.pth"):
        path = BASE_DIR / name
        if path.exists():
            return path
    return None


def build_test_target_timestamps(n_samples: int, lags: int = LAGS) -> list[pd.Timestamp]:
    data_path = GEN_DIR / "cleaned_scats_directional_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}. Run data_processor.py first.")

    df = pd.read_csv(data_path)
    pivot_df = df.pivot(index="Timestamp", columns="Node_ID", values="Traffic_Volume")
    pivot_df = pivot_df.ffill().fillna(0)

    timestamps = pd.to_datetime(pivot_df.index)
    total_samples = len(pivot_df)
    val_idx = int(total_samples * 0.80)

    target_indices = range(val_idx + lags, val_idx + lags + n_samples)
    if len(target_indices) != n_samples:
        raise ValueError(
            f"Could not align {n_samples} test samples with timeline "
            f"(val_idx={val_idx}, lags={lags}, total={total_samples})."
        )
    return [timestamps[i] for i in target_indices]


def main() -> int:
    required = [
        GEN_DIR / "X_test.npy",
        GEN_DIR / "y_test.npy",
        GEN_DIR / "adjacency_matrix.npy",
        GEN_DIR / "traffic_scaler.pkl",
        GEN_DIR / "node_lookup_table.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing required GNN files:")
        for path in missing:
            print(" -", path)
        return 1

    weights = find_weights()
    if weights is None:
        print("Missing trained weights: GNN_trained.pth or trained_stgcn.pth")
        return 1

    x_test = np.load(GEN_DIR / "X_test.npy")
    n_samples = len(x_test)
    a_raw = np.load(GEN_DIR / "adjacency_matrix.npy")
    scaler = joblib.load(GEN_DIR / "traffic_scaler.pkl")
    node_lookup = pd.read_csv(GEN_DIR / "node_lookup_table.csv")

    timestamps = build_test_target_timestamps(n_samples)

    node_to_scats = {
        int(row["Node_ID"]): clean_scats(row["SCATS Number"])
        for _, row in node_lookup.iterrows()
    }

    a_norm = normalize_adjacency(a_raw)
    x_tensor = torch.tensor(x_test, dtype=torch.float32)
    a_tensor = torch.tensor(a_norm, dtype=torch.float32)

    model = TrafficGNN(num_nodes=NUM_NODES, timesteps=LAGS, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(x_tensor, a_tensor).numpy()
    y_pred_scaled = np.maximum(y_pred_scaled, 0.0)
    y_pred_cars = scaler.inverse_transform(y_pred_scaled)

    totals: dict[tuple[str, int], list[float]] = {}
    for sample_i in range(n_samples):
        ts = timestamps[sample_i]
        hour = int(ts.hour)
        for node_id in range(y_pred_cars.shape[1]):
            scats = node_to_scats.get(node_id)
            if not scats:
                continue
            pred_15min = float(y_pred_cars[sample_i, node_id])
            pred_hourly = max(0.0, pred_15min * 4.0)
            totals.setdefault((scats, hour), []).append(pred_hourly)

    rows = [
        {
            "scats_site": scats,
            "hour": hour,
            "pred_stgcn_hourly": float(np.mean(values)),
        }
        for (scats, hour), values in sorted(totals.items())
        if values
    ]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(rows)} (site, hour) STGCN routing predictions to:")
    print(OUTPUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
