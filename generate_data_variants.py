from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "dataset.csv"

STATION_ORDER = {f"S{i}": i for i in range(1, 11)}
SENSOR_COLUMNS = [
    "cycle_time_sec",
    "torque_nm",
    "temperature_c",
    "vibration_rms",
    "pressure_bar",
    "force_n",
    "position_error_mm",
    "voltage_v",
    "current_a",
    "flow_rate_lpm",
    "queue_time_sec",
    "ambient_temperature_c",
    "humidity_pct",
]


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for column in SENSOR_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def variant_sparse(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = ensure_numeric(out)
    sparse_map = {
        "cycle_time_sec": 0.18,
        "torque_nm": 0.35,
        "temperature_c": 0.28,
        "vibration_rms": 0.42,
        "pressure_bar": 0.33,
        "force_n": 0.36,
        "position_error_mm": 0.45,
        "voltage_v": 0.4,
        "current_a": 0.38,
        "flow_rate_lpm": 0.52,
        "queue_time_sec": 0.2,
        "ambient_temperature_c": 0.1,
        "humidity_pct": 0.12,
    }
    for column, rate in sparse_map.items():
        mask = np.random.rand(len(out)) < rate
        out.loc[mask, column] = np.nan
    return out


def variant_drift(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = ensure_numeric(out)
    for station_id, idx in STATION_ORDER.items():
        station_mask = out["station_id"] == station_id
        drift_multiplier = 1 + 0.06 * idx
        out.loc[station_mask, "cycle_time_sec"] *= drift_multiplier
        out.loc[station_mask, "temperature_c"] += 0.5 * idx
        out.loc[station_mask, "queue_time_sec"] *= (1 + 0.08 * idx)
        out.loc[station_mask, "pressure_bar"] *= (1 + 0.04 * idx)
        out.loc[station_mask, "humidity_pct"] += 0.3 * idx
    return out


def variant_propagation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = ensure_numeric(out)
    for vehicle_id, vehicle_df in out.groupby("vehicle_id"):
        root_station = vehicle_df["root_cause_station"].iloc[0]
        if root_station == "NONE":
            continue
        root_index = STATION_ORDER.get(root_station, 0)
        for _, row in vehicle_df.iterrows():
            station_idx = STATION_ORDER.get(row["station_id"], 0)
            if station_idx <= root_index:
                continue
            if np.random.rand() < 0.7:
                out.loc[(out["vehicle_id"] == vehicle_id) & (out["station_id"] == row["station_id"]), "anomaly_flag"] = 1
                for col in ["position_error_mm", "torque_nm", "pressure_bar", "vibration_rms", "force_n", "temperature_c"]:
                    out.loc[(out["vehicle_id"] == vehicle_id) & (out["station_id"] == row["station_id"]), col] = (
                        out.loc[(out["vehicle_id"] == vehicle_id) & (out["station_id"] == row["station_id"]), col].astype(float)
                        + np.random.uniform(0.25, 2.5)
                    )
    return out


def variant_noisy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = ensure_numeric(out)
    for column in SENSOR_COLUMNS:
        std = out[column].std(ddof=0)
        if pd.isna(std) or std == 0:
            continue
        noise = np.random.normal(0, std * 0.35, size=len(out))
        out[column] = out[column].fillna(out[column].median()) + noise

    for column in ["torque_nm", "temperature_c", "pressure_bar", "force_n", "position_error_mm", "current_a", "flow_rate_lpm", "vibration_rms"]:
        mask = np.random.rand(len(out)) < 0.25
        out.loc[mask, column] = np.nan
    return out


def write_variant(name: str, variant_func, base_path: Path = BASE_PATH) -> pd.DataFrame:
    base_df = pd.read_csv(base_path)
    result = variant_func(base_df)
    output_path = base_path.with_name(name)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    variants = {
        "dataset_variant_sparse.csv": variant_sparse,
        "dataset_variant_drift.csv": variant_drift,
        "dataset_variant_propagation.csv": variant_propagation,
        "dataset_variant_noisy.csv": variant_noisy,
    }

    for name, fn in variants.items():
        df = write_variant(name, fn)
        anomaly_rate = df["anomaly_flag"].mean()
        missing_total = df.isna().sum().sum()
        print(f"{name}: rows={len(df)}, anomaly_rate={anomaly_rate:.4f}, missing_values={missing_total}")


if __name__ == "__main__":
    main()
