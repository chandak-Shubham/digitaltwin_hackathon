from __future__ import annotations

from typing import Dict, List

import pandas as pd


def build_station_history(df: pd.DataFrame) -> pd.DataFrame:
    """Build a station-order view with vehicle-level anomaly context."""
    result = df.copy()
    result["station_index"] = result.groupby("vehicle_id")["station_id"].rank(method="dense", ascending=True).astype(int)
    result["vehicle_sequence"] = result.groupby("vehicle_id")["station_id"].transform(lambda s: range(1, len(s) + 1))
    if "vehicle_id" in result.columns:
        result["is_anomalous_vehicle"] = result.groupby("vehicle_id")["anomaly_flag"].transform("max")
    return result


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def explain_vehicle_anomaly(df: pd.DataFrame, vehicle_id: str, station_id: str) -> Dict:
    """Return an explainable anomaly summary for a given vehicle and station."""
    vehicle_df = df[df["vehicle_id"] == vehicle_id].copy()
    vehicle_df = vehicle_df.sort_values("station_id")
    vehicle_df = build_station_history(vehicle_df)

    root_cause_station = vehicle_df["root_cause_station"].iloc[0]
    if root_cause_station == "NONE":
        root_cause_station = "NONE"

    station_rows = vehicle_df[vehicle_df["station_id"] == station_id].copy()
    if station_rows.empty:
        raise ValueError(f"Station {station_id} not found for vehicle {vehicle_id}")

    current = station_rows.iloc[0]
    selected_features = [
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

    station_history = []
    for _, row in vehicle_df.iterrows():
        reasoning = {
            "station_id": row["station_id"],
            "station_name": row.get("station_name", row["station_id"]),
            "anomaly_flag": int(row.get("anomaly_flag", 0)),
            "rework_flag": int(row.get("rework_flag", 0)),
            "station_index": int(row.get("station_index", 0)),
            "feature_deltas": [],
        }
        for feature in selected_features:
            value = _safe_numeric(row.get(feature, pd.NA))
            if pd.notna(value):
                reasoning["feature_deltas"].append({"feature": feature, "value": float(value)})
        station_history.append(reasoning)

    contributors: List[Dict] = []
    for _, row in vehicle_df.iterrows():
        if row["station_id"] == station_id:
            for feature in selected_features:
                value = _safe_numeric(row.get(feature, pd.NA))
                if pd.notna(value):
                    contributors.append({
                        "station_id": row["station_id"],
                        "feature": feature,
                        "value": float(value),
                        "score": 1.0 if pd.notna(value) else 0.0,
                        "direction": "current_signal",
                    })
        elif row.get("anomaly_flag", 0) == 1:
            for feature in selected_features:
                value = _safe_numeric(row.get(feature, pd.NA))
                if pd.notna(value):
                    contributors.append({
                        "station_id": row["station_id"],
                        "feature": feature,
                        "value": float(value),
                        "score": 0.75,
                        "direction": "historical_contributor",
                    })

    if not contributors:
        contributors = [{
            "station_id": station_id,
            "feature": "overall_process_health",
            "value": float(current.get("anomaly_flag", 0)),
            "score": 0.5,
            "direction": "current_signal",
        }]

    return {
        "vehicle_id": vehicle_id,
        "predicted_station": station_id,
        "root_cause_station": root_cause_station,
        "anomaly_probability": float(current.get("anomaly_flag", 0)),
        "history": station_history,
        "contributors": contributors[:12],
    }
