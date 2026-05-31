"""
COS30019 A2B Traffic Flow Model Training Script

This script converts the original train.ipynb workflow into a normal Python file.

It trains:
- LSTM traffic flow model
- GRU traffic flow model

It saves outputs to:
models/artifacts_person1/

Important output for GUI integration:
models/artifacts_person1/predictions_test.csv

Run from project root:
python models/train.py

Or using Arif's Python path:
& "C:\\Users\\AriF0019\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" models/train.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
T = 96


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMFlowRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, dense_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dense1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.relu(self.dense1(last))
        out = self.dense2(out)
        return out.squeeze(-1)


class GRUFlowRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, dense_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.fc2 = nn.Linear(dense_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        out = self.relu(self.fc1(last))
        out = self.fc2(out)
        return out.squeeze(-1)


def find_default_data_path(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "raw" / "Scats Data October 2006.xls",
        project_root / "Scats Data October 2006.xls",
        project_root / "GNN_model" / "Files" / "Scats Data October 2006.xls",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_raw_data(data_path: Path, sheet_name: str) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = data_path.suffix.lower()

    if suffix in [".xls", ".xlsx"]:
        df_raw = pd.read_excel(data_path, sheet_name=sheet_name, header=1)
    elif suffix == ".csv":
        df_raw = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format. Use .xls, .xlsx, or .csv")

    print(f"Loaded data: {data_path}")
    print(f"Rows: {len(df_raw)}, Columns: {df_raw.shape[1]}")

    return df_raw


def clean_data(df_raw: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    df = df_raw.copy()

    if "SCATS Number" in df.columns:
        df = df.rename(columns={"SCATS Number": "SCATS_Number"})
        df["SCATS Number"] = pd.to_numeric(df["SCATS_Number"], errors="coerce").astype("Int64")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    flow_cols = [
        c for c in df.columns
        if isinstance(c, str) and c.startswith("V") and len(c) == 3
    ]
    flow_cols = sorted(flow_cols, key=lambda s: int(s[1:]))

    if not flow_cols:
        raise ValueError("No V00 to V95 traffic flow columns found.")

    print(f"Found flow columns: {len(flow_cols)}")

    for col in flow_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["SCATS Number"].notna()].copy()
    df = df[df["Date"].notna()].copy()
    df = df.dropna(subset=flow_cols, how="all").copy()
    df[flow_cols] = df[flow_cols].fillna(0)
    df = df[df[flow_cols].sum(axis=1) > 0].copy()

    for col in ["NB_LATITUDE", "NB_LONGITUDE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    output_dir.mkdir(parents=True, exist_ok=True)

    cols_for_clean_csv = [
        c for c in ["SCATS Number", "Date", "NB_LATITUDE", "NB_LONGITUDE"]
        if c in df.columns
    ] + flow_cols

    df[cols_for_clean_csv].to_csv(output_dir / "cleaned_flows.csv", index=False)

    print(f"After cleaning: {df.shape}")
    print(f"Saved: {output_dir / 'cleaned_flows.csv'}")

    return df, flow_cols


def split_by_scats_site(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    site_groups = df["SCATS Number"].astype(str)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=site_groups))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, val_idx = next(
        gss2.split(df_train, groups=df_train["SCATS Number"].astype(str))
    )

    df_tr = df_train.iloc[tr_idx].copy()
    df_val = df_train.iloc[val_idx].copy()

    print("Split sizes:")
    print(f"Train rows: {len(df_tr)}")
    print(f"Val rows:   {len(df_val)}")
    print(f"Test rows:  {len(df_test)}")

    print("Unique SCATS sites:")
    print(
        "All:",
        df["SCATS Number"].nunique(),
        "Train:",
        df_tr["SCATS Number"].nunique(),
        "Val:",
        df_val["SCATS Number"].nunique(),
        "Test:",
        df_test["SCATS Number"].nunique(),
    )

    return df_tr, df_val, df_test


def save_row_splits(
    df_tr: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    flow_cols: list[str],
    output_dir: Path,
) -> None:
    base_cols = [
        c for c in ["SCATS Number", "Date", "NB_LATITUDE", "NB_LONGITUDE"]
        if c in df_tr.columns
    ]
    cols_to_save = base_cols + flow_cols

    df_tr[cols_to_save].to_csv(output_dir / "train_rows.csv", index=False)
    df_val[cols_to_save].to_csv(output_dir / "val_rows.csv", index=False)
    df_test[cols_to_save].to_csv(output_dir / "test_rows.csv", index=False)

    print("Saved row-level splits:")
    print(f"- {output_dir / 'train_rows.csv'}")
    print(f"- {output_dir / 'val_rows.csv'}")
    print(f"- {output_dir / 'test_rows.csv'}")


def build_sequences_from_df(
    df: pd.DataFrame,
    flow_cols: list[str],
    flow_scaler: StandardScaler,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, pd.Timestamp, int]]]:
    tod = np.arange(T)
    tod_sin = np.sin(2 * np.pi * tod / T).astype(np.float32)
    tod_cos = np.cos(2 * np.pi * tod / T).astype(np.float32)

    flows = df[flow_cols].to_numpy(dtype=np.float32)
    flows_scaled = flow_scaler.transform(flows.reshape(-1, 1)).reshape(flows.shape).astype(np.float32)

    sites = df["SCATS Number"].to_numpy()
    dates = df["Date"].to_numpy()

    X_list = []
    y_list = []
    meta = []

    for i in range(flows_scaled.shape[0]):
        series = flows_scaled[i]

        for t in range(0, T - seq_len):
            window = series[t:t + seq_len]
            step_idx = np.arange(t, t + seq_len)

            X = np.stack(
                [window, tod_sin[step_idx], tod_cos[step_idx]],
                axis=-1,
            )

            y = series[t + seq_len]

            X_list.append(X)
            y_list.append(y)
            meta.append((int(sites[i]), pd.to_datetime(dates[i]), int(t + seq_len)))

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        meta,
    )


def inverse_flow_scale(flow_scaler: StandardScaler, x_scaled: np.ndarray) -> np.ndarray:
    x_scaled = np.asarray(x_scaled).reshape(-1, 1)
    return flow_scaler.inverse_transform(x_scaled).reshape(-1)


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(math.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def train_model(
    model: nn.Module,
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    patience: int,
    min_delta: float = 1e-6,
) -> nn.Module:
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    criterion = nn.MSELoss()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    best_val = float("inf")
    best_state = None
    wait = 0

    print(f"\nTraining {model_name} model...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.detach().item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        lr = float(optimizer.param_groups[0]["lr"])

        if epoch <= 5 or epoch % 10 == 0:
            print(
                f"{model_name} Epoch {epoch:03d} "
                f"| train_loss={train_loss:.5f} "
                f"| val_loss={val_loss:.5f} "
                f"| lr={lr:.2e}"
            )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1

            if wait >= patience:
                print(f"{model_name} early stopping at epoch {epoch}, best val_loss={best_val:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_model(model: nn.Module, X_test: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_test).to(device))
        pred_scaled = pred_scaled.detach().cpu().numpy().reshape(-1)

    return pred_scaled


def main() -> None:
    parser = argparse.ArgumentParser(description="Train COS30019 A2B LSTM and GRU traffic models.")

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to Scats Data October 2006.xls",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Data",
        help="Excel sheet name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output artifact directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4,
        help="Number of 15-minute historical steps used for prediction",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test mode, uses fewer epochs",
    )

    args = parser.parse_args()

    set_seed(SEED)

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data) if args.data else find_default_data_path(project_root)
    output_dir = Path(args.output) if args.output else project_root / "models" / "artifacts_person1"

    if args.quick:
        args.epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("COS30019 A2B Model Training")
    print("---------------------------")
    print("Project root:", project_root)
    print("Data path:", data_path)
    print("Output dir:", output_dir)
    print("Device:", device)
    print("Torch:", torch.__version__)
    print()

    df_raw = load_raw_data(data_path, args.sheet)
    df, flow_cols = clean_data(df_raw, output_dir)

    df_tr, df_val, df_test = split_by_scats_site(df)
    save_row_splits(df_tr, df_val, df_test, flow_cols, output_dir)

    flow_scaler = StandardScaler()
    flow_scaler.fit(df_tr[flow_cols].to_numpy().reshape(-1, 1))

    X_tr, y_tr, meta_tr = build_sequences_from_df(df_tr, flow_cols, flow_scaler, args.seq_len)
    X_val, y_val, meta_val = build_sequences_from_df(df_val, flow_cols, flow_scaler, args.seq_len)
    X_test, y_test, meta_test = build_sequences_from_df(df_test, flow_cols, flow_scaler, args.seq_len)

    print()
    print("Sequence shapes:")
    print("X_tr:", X_tr.shape, "y_tr:", y_tr.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    y_test_true = inverse_flow_scale(flow_scaler, y_test)

    n_features = X_tr.shape[-1]

    lstm = LSTMFlowRegressor(n_features=n_features)
    gru = GRUFlowRegressor(n_features=n_features)

    lstm = train_model(
        lstm,
        "LSTM",
        X_tr,
        y_tr,
        X_val,
        y_val,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=8,
    )

    gru = train_model(
        gru,
        "GRU",
        X_tr,
        y_tr,
        X_val,
        y_val,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=8,
    )

    pred_lstm_scaled = predict_model(lstm, X_test, device)
    pred_gru_scaled = predict_model(gru, X_test, device)

    pred_lstm_veh_15min = inverse_flow_scale(flow_scaler, pred_lstm_scaled)
    pred_gru_veh_15min = inverse_flow_scale(flow_scaler, pred_gru_scaled)

    metrics_lstm = regression_metrics(y_test_true, pred_lstm_veh_15min)
    metrics_gru = regression_metrics(y_test_true, pred_gru_veh_15min)

    results = pd.DataFrame([
        {"model": "LSTM_torch", **metrics_lstm},
        {"model": "GRU_torch", **metrics_gru},
    ])

    pred_df = pd.DataFrame({
        "scats_site": [m[0] for m in meta_test],
        "date": [m[1] for m in meta_test],
        "step_index_0to95": [m[2] for m in meta_test],
        "y_true_veh_15min": y_test_true,
        "pred_lstm_veh_15min": pred_lstm_veh_15min,
        "pred_gru_veh_15min": pred_gru_veh_15min,
    })

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(lstm.state_dict(), output_dir / "lstm_flow_15min_torch.pt")
    torch.save(gru.state_dict(), output_dir / "gru_flow_15min_torch.pt")

    with open(output_dir / "lstm_flow_15min_torch_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seq_len": int(args.seq_len),
                "n_features": int(n_features),
                "hidden_size": 64,
                "dense_size": 32,
                "backend": "pytorch",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(output_dir / "gru_flow_15min_torch_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seq_len": int(args.seq_len),
                "n_features": int(n_features),
                "hidden_size": 64,
                "dense_size": 32,
                "backend": "pytorch",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    joblib.dump(flow_scaler, output_dir / "flow_scaler.joblib")

    pred_df.to_csv(output_dir / "predictions_test.csv", index=False)
    results.to_csv(output_dir / "model_results.csv", index=False)

    print()
    print("Model results:")
    print(results.to_string(index=False))

    print()
    print("Saved artifacts:")
    print("-", output_dir / "lstm_flow_15min_torch.pt")
    print("-", output_dir / "gru_flow_15min_torch.pt")
    print("-", output_dir / "flow_scaler.joblib")
    print("-", output_dir / "predictions_test.csv")
    print("-", output_dir / "model_results.csv")

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()