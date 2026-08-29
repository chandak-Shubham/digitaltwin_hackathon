from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


DATASET_PATH = Path(__file__).with_name("dataset.csv")
STATION_ORDER = {f"S{i}": i for i in range(1, 11)}
NUMERIC_COLUMNS = [
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
CATEGORICAL_COLUMNS = [
    "station_id",
    "station_name",
    "vehicle_model",
    "vehicle_variant",
    "shift",
    "production_batch",
]


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["station_order"] = df["station_id"].map(STATION_ORDER)
    df = df.sort_values(["vehicle_id", "station_order"]).reset_index(drop=True)
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        median_value = df[column].median()
        df[column] = df[column].fillna(median_value)
    return df


def add_vehicle_history_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("vehicle_id")

    for column in NUMERIC_COLUMNS:
        df[f"{column}_lag1"] = grouped[column].shift(1)
        df[f"{column}_lag2"] = grouped[column].shift(2)
        df[f"{column}_delta_lag1"] = df[column] - df[f"{column}_lag1"]
        df[f"{column}_delta_lag2"] = df[column] - df[f"{column}_lag2"]
        df[f"{column}_rolling_mean_3"] = grouped[column].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df[f"{column}_rolling_std_3"] = grouped[column].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std().fillna(0))

    df["vehicle_anomaly_so_far"] = grouped["anomaly_flag"].transform(lambda s: s.cumsum().shift(1).fillna(0))
    df["vehicle_anomaly_rate"] = grouped["anomaly_flag"].transform(lambda s: s.expanding().mean().shift(1).fillna(0))
    df["vehicle_rework_so_far"] = grouped["rework_flag"].transform(lambda s: s.cumsum().shift(1).fillna(0))

    df["queue_time_sec_lag1"] = grouped["queue_time_sec"].shift(1)
    df["cycle_time_sec_lag1"] = grouped["cycle_time_sec"].shift(1)
    df["torque_nm_lag1"] = grouped["torque_nm"].shift(1)
    df["vibration_rms_lag1"] = grouped["vibration_rms"].shift(1)

    return df


def get_anomaly_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    raw_fields = [
        "station_id",
        "station_name",
        "vehicle_model",
        "vehicle_variant",
        "shift",
        "production_batch",
        *NUMERIC_COLUMNS,
        *[col for col in df.columns if "lag" in col or "delta" in col or "rolling" in col or "vehicle_anomaly" in col or "vehicle_rework" in col],
    ]
    feature_frame = df[raw_fields].copy()
    feature_frame = pd.get_dummies(feature_frame, columns=CATEGORICAL_COLUMNS, drop_first=False)
    feature_frame = feature_frame.fillna(0)
    return feature_frame, feature_frame.columns.tolist()


def train_anomaly_model(df: pd.DataFrame) -> Dict:
    prepared = add_vehicle_history_features(fill_missing_values(df.copy()))
    X, feature_names = get_anomaly_feature_frame(prepared)
    y = prepared["anomaly_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "model": model,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "prepared_df": prepared,
        "median_values": X_train.median(),
        "metrics": {
            "auc": float(roc_auc_score(y_test, y_prob)),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        },
        "feature_importance": pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True),
    }
    return result


def train_root_cause_model(df: pd.DataFrame) -> Dict:
    prepared = add_vehicle_history_features(fill_missing_values(df.copy()))
    root_df = prepared[prepared["root_cause_station"] != "NONE"].copy()
    root_df["target"] = root_df["root_cause_station"]

    X, feature_names = get_anomaly_feature_frame(root_df)
    y = root_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="multiclass",
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "feature_names": feature_names,
        "prepared_df": root_df,
        "median_values": X_train.median(),
        "metrics": {"accuracy": float(accuracy)},
        "feature_importance": pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True),
    }


def encode_row_to_model_features(raw_row: pd.Series, feature_names: List[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(pd.DataFrame([raw_row]), columns=CATEGORICAL_COLUMNS, drop_first=False)
    for feature in feature_names:
        if feature not in encoded.columns:
            encoded[feature] = 0
    encoded = encoded[feature_names]
    return encoded


def explain_vehicle_anomaly(df: pd.DataFrame, anomaly_model: Dict, root_cause_model: Dict, vehicle_id: str, station_id: str) -> Dict:
    prepared = anomaly_model["prepared_df"]
    vehicle_rows = prepared[prepared["vehicle_id"] == vehicle_id].sort_values("station_order").copy()
    if vehicle_rows.empty:
        raise ValueError(f"Vehicle {vehicle_id} not found in dataset")

    current_station_rows = vehicle_rows[vehicle_rows["station_id"] == station_id]
    if current_station_rows.empty:
        raise ValueError(f"Station {station_id} not found for vehicle {vehicle_id}")

    current_row = current_station_rows.iloc[0].copy()
    history_features_df = add_vehicle_history_features(fill_missing_values(df.copy()))
    raw_row = history_features_df[
        (history_features_df["vehicle_id"] == vehicle_id) & (history_features_df["station_id"] == station_id)
    ].iloc[0].copy()

    encoded = encode_row_to_model_features(raw_row, anomaly_model["feature_names"])
    root_encoded = encode_row_to_model_features(raw_row, root_cause_model["feature_names"])

    current_probability = float(anomaly_model["model"].predict_proba(encoded)[0, 1])
    root_cause_prediction = root_cause_model["model"].predict(root_encoded)[0]

    median_values = anomaly_model["median_values"]
    feature_scores = []
    for feature in anomaly_model["feature_names"]:
        if feature in encoded.columns:
            value = float(encoded.iloc[0][feature])
            median = float(median_values.get(feature, 0.0))
            importance = float(
                anomaly_model["feature_importance"].set_index("feature").loc[feature, "importance"]
                if feature in anomaly_model["feature_importance"]["feature"].tolist()
                else 0.0
            )
            score = abs(value - median) * (1 + importance / max(1, anomaly_model["feature_importance"]["importance"].max()))
            feature_scores.append({
                "feature": feature,
                "current_value": value,
                "baseline_median": median,
                "score": float(score),
            })

    feature_scores = sorted(feature_scores, key=lambda item: item["score"], reverse=True)[:10]

    station_contributors = []
    for _, row in vehicle_rows.iterrows():
        if row["station_id"] in {station_id, *[s for s in vehicle_rows["station_id"].unique() if s != station_id]}:
            station_contributors.append({
                "station_id": row["station_id"],
                "station_name": row["station_name"],
                "anomaly_flag": int(row["anomaly_flag"]),
                "root_cause_station": row["root_cause_station"],
                "queue_time_sec": float(row["queue_time_sec"]),
                "torque_nm": float(row["torque_nm"] if pd.notna(row["torque_nm"]) else 0),
                "vibration_rms": float(row["vibration_rms"] if pd.notna(row["vibration_rms"]) else 0),
            })

    return {
        "vehicle_id": vehicle_id,
        "predicted_station": station_id,
        "actual_root_cause_station": current_row.get("root_cause_station", "NONE"),
        "predicted_root_cause_station": root_cause_prediction,
        "anomaly_probability": current_probability,
        "top_contributor_features": feature_scores,
        "station_history": station_contributors,
    }


def main() -> None:
    df = load_dataset()
    anomaly_model = train_anomaly_model(df)
    root_cause_model = train_root_cause_model(df)

    print("Anomaly model AUC:", round(anomaly_model["metrics"]["auc"], 4))
    print("Root-cause model accuracy:", round(root_cause_model["metrics"]["accuracy"], 4))
    print("Top anomaly features:")
    print(anomaly_model["feature_importance"].head(10).to_string(index=False))

    sample_vehicle = df["vehicle_id"].unique()[0]
    example_station = df[df["vehicle_id"] == sample_vehicle]["station_id"].iloc[0]
    explanation = explain_vehicle_anomaly(df, anomaly_model, root_cause_model, sample_vehicle, example_station)
    print("\nSample explanation:")
    print(explanation)


if __name__ == "__main__":
    main()
