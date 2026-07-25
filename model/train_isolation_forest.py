from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data/time_series/processed/TS-predictive-dataset-v1.csv"
)

MODEL_DIR = PROJECT_ROOT / "model/artifacts"
EVIDENCE_DIR = PROJECT_ROOT / "evidence/model"

MODEL_FILE = MODEL_DIR / "isolation-forest-v1.joblib"
PREDICTIONS_FILE = (
    EVIDENCE_DIR / "IF-test-predictions-v1.csv"
)
SCORE_SUMMARY_FILE = (
    EVIDENCE_DIR / "IF-score-summary-v1.csv"
)
SUMMARY_FILE = (
    EVIDENCE_DIR / "IF-evaluation-summary-v1.json"
)
REPORT_FILE = (
    EVIDENCE_DIR / "IF-classification-report-v1.txt"
)
CHECKSUM_FILE = (
    EVIDENCE_DIR / "IF-checksums-v1.csv"
)

TRAIN_EXPERIMENT = "TS-HB-01"
HEALTHY_HOLDOUT_EXPERIMENT = "TS-HB-02"
FAILURE_EXPERIMENT = "TS-MEM-01"

TARGET_COLUMN = "failure_within_60m"
CONTAMINATION = 0.05
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "cpu_millicores",
    "memory_mib",
    "ready_numeric",
    "cpu_millicores_roll_5m_mean",
    "cpu_millicores_roll_5m_std",
    "cpu_millicores_roll_5m_min",
    "cpu_millicores_roll_5m_max",
    "cpu_millicores_roll_5m_range",
    "cpu_millicores_roll_5m_delta",
    "cpu_millicores_roll_15m_mean",
    "cpu_millicores_roll_15m_std",
    "cpu_millicores_roll_15m_min",
    "cpu_millicores_roll_15m_max",
    "cpu_millicores_roll_15m_range",
    "cpu_millicores_roll_15m_delta",
    "cpu_millicores_roll_30m_mean",
    "cpu_millicores_roll_30m_std",
    "cpu_millicores_roll_30m_min",
    "cpu_millicores_roll_30m_max",
    "cpu_millicores_roll_30m_range",
    "cpu_millicores_roll_30m_delta",
    "memory_mib_roll_5m_mean",
    "memory_mib_roll_5m_std",
    "memory_mib_roll_5m_min",
    "memory_mib_roll_5m_max",
    "memory_mib_roll_5m_range",
    "memory_mib_roll_5m_delta",
    "memory_mib_roll_15m_mean",
    "memory_mib_roll_15m_std",
    "memory_mib_roll_15m_min",
    "memory_mib_roll_15m_max",
    "memory_mib_roll_15m_range",
    "memory_mib_roll_15m_delta",
    "memory_mib_roll_30m_mean",
    "memory_mib_roll_30m_std",
    "memory_mib_roll_30m_min",
    "memory_mib_roll_30m_max",
    "memory_mib_roll_30m_range",
    "memory_mib_roll_30m_delta",
    "cpu_change_1m",
    "memory_change_1m",
    "memory_growth_rate_mib_per_min",
]

EXCLUDED_COLUMNS = [
    "timestamp_utc",
    "experiment_id",
    "label",
    "namespace",
    "pod_name",
    "restart_count",
    "phase",
    "ready",
    "node",
    "collection_error",
    "elapsed_seconds",
    "failure_timestamp_utc",
    "minutes_to_failure",
    "failure_within_5m",
    "failure_within_15m",
    "failure_within_30m",
    "failure_within_60m",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest().upper()


def class_distribution(
    values: pd.Series,
) -> dict[str, int]:
    counts = (
        values.astype(int)
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    return {
        "normal": int(counts.loc[0]),
        "failure_within_60m": int(counts.loc[1]),
    }


def safe_roc_auc(
    actual: pd.Series,
    anomaly_score: pd.Series,
) -> float | None:
    if actual.nunique() < 2:
        return None

    return float(
        roc_auc_score(actual, anomaly_score)
    )


def load_and_validate_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    dataframe = pd.read_csv(DATASET_FILE)

    required_columns = [
        "timestamp_utc",
        "experiment_id",
        "label",
        "minutes_to_failure",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    dataframe["timestamp_utc"] = pd.to_datetime(
        dataframe["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe["minutes_to_failure"] = pd.to_numeric(
        dataframe["minutes_to_failure"],
        errors="coerce",
    )

    feature_missing_counts = (
        dataframe[FEATURE_COLUMNS]
        .isna()
        .sum()
    )

    if feature_missing_counts.gt(0).any():
        raise ValueError(
            "Model features contain missing values:\n"
            f"{feature_missing_counts[feature_missing_counts.gt(0)]}"
        )

    valid_targets = set(
        dataframe[TARGET_COLUMN]
        .astype(int)
        .unique()
        .tolist()
    )

    if not valid_targets.issubset({0, 1}):
        raise ValueError(
            f"Invalid target values: {valid_targets}"
        )

    if len(dataframe) != 266:
        raise ValueError(
            f"Expected 266 rows, found {len(dataframe)}."
        )

    expected_counts = {
        TRAIN_EXPERIMENT: 90,
        HEALTHY_HOLDOUT_EXPERIMENT: 90,
        FAILURE_EXPERIMENT: 86,
    }

    actual_counts = (
        dataframe.groupby("experiment_id")
        .size()
        .to_dict()
    )

    for experiment_id, expected_count in (
        expected_counts.items()
    ):
        actual_count = int(
            actual_counts.get(experiment_id, 0)
        )

        if actual_count != expected_count:
            raise ValueError(
                f"{experiment_id}: expected "
                f"{expected_count} rows, found "
                f"{actual_count}."
            )

    return (
        dataframe
        .sort_values(
            ["experiment_id", "timestamp_utc"]
        )
        .reset_index(drop=True)
    )


def add_model_outputs(
    model: IsolationForest,
    dataframe: pd.DataFrame,
    dataset_role: str,
) -> pd.DataFrame:
    result = dataframe.copy()

    raw_prediction = model.predict(
        result[FEATURE_COLUMNS]
    )

    decision_function = model.decision_function(
        result[FEATURE_COLUMNS]
    )

    result["dataset_role"] = dataset_role
    result["isolation_forest_prediction"] = (
        raw_prediction
    )

    result["predicted_anomaly"] = (
        raw_prediction == -1
    ).astype(int)

    result["decision_function"] = (
        decision_function
    )

    # Higher values indicate stronger anomalous behaviour.
    result["anomaly_score"] = -decision_function

    return result


def score_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for (
        experiment_id,
        experiment,
    ) in dataframe.groupby(
        "experiment_id",
        sort=True,
    ):
        anomaly_count = int(
            experiment["predicted_anomaly"].sum()
        )

        records.append(
            {
                "experiment_id": experiment_id,
                "dataset_role": (
                    experiment["dataset_role"].iloc[0]
                ),
                "rows": int(len(experiment)),
                "actual_positive_60m": int(
                    experiment[TARGET_COLUMN].sum()
                ),
                "predicted_anomalies": anomaly_count,
                "anomaly_rate": float(
                    anomaly_count / len(experiment)
                ),
                "anomaly_score_min": float(
                    experiment["anomaly_score"].min()
                ),
                "anomaly_score_mean": float(
                    experiment["anomaly_score"].mean()
                ),
                "anomaly_score_max": float(
                    experiment["anomaly_score"].max()
                ),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_and_validate_dataset()

    training_data = dataframe[
        dataframe["experiment_id"]
        == TRAIN_EXPERIMENT
    ].copy()

    healthy_holdout = dataframe[
        dataframe["experiment_id"]
        == HEALTHY_HOLDOUT_EXPERIMENT
    ].copy()

    failure_evaluation = dataframe[
        dataframe["experiment_id"]
        == FAILURE_EXPERIMENT
    ].copy()

    if training_data[TARGET_COLUMN].sum() != 0:
        raise ValueError(
            "Isolation Forest training data must "
            "contain healthy observations only."
        )

    if healthy_holdout[TARGET_COLUMN].sum() != 0:
        raise ValueError(
            "Healthy holdout contains positive "
            "failure labels."
        )

    model = IsolationForest(
        n_estimators=500,
        max_samples="auto",
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        training_data[FEATURE_COLUMNS]
    )

    training_results = add_model_outputs(
        model,
        training_data,
        "healthy_training",
    )

    healthy_results = add_model_outputs(
        model,
        healthy_holdout,
        "healthy_holdout",
    )

    failure_results = add_model_outputs(
        model,
        failure_evaluation,
        "failure_evaluation",
    )

    all_results = pd.concat(
        [
            training_results,
            healthy_results,
            failure_results,
        ],
        ignore_index=True,
    )

    evaluation_results = pd.concat(
        [
            healthy_results,
            failure_results,
        ],
        ignore_index=True,
    )

    actual = (
        evaluation_results[TARGET_COLUMN]
        .astype(int)
    )

    predicted = (
        evaluation_results["predicted_anomaly"]
        .astype(int)
    )

    confusion = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    true_negative = int(confusion[0, 0])
    false_positive = int(confusion[0, 1])
    false_negative = int(confusion[1, 0])
    true_positive = int(confusion[1, 1])

    metrics = {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual,
                predicted,
            )
        ),
        "precision": float(
            precision_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                actual,
                predicted,
                zero_division=0,
            )
        ),
        "roc_auc": safe_roc_auc(
            actual,
            evaluation_results["anomaly_score"],
        ),
    }

    healthy_false_positive_rate = float(
        healthy_results[
            "predicted_anomaly"
        ].mean()
    )

    failure_experiment_anomaly_rate = float(
        failure_results[
            "predicted_anomaly"
        ].mean()
    )

    failure_window = failure_results[
        failure_results[TARGET_COLUMN] == 1
    ]

    failure_window_detection_rate = float(
        failure_window[
            "predicted_anomaly"
        ].mean()
    )

    detected_failure_rows = failure_results[
        (failure_results["predicted_anomaly"] == 1)
        & (
            failure_results[
                "minutes_to_failure"
            ].notna()
        )
    ]

    first_detection_lead_minutes = None
    first_detection_timestamp = None

    if not detected_failure_rows.empty:
        first_detected_row = (
            detected_failure_rows
            .sort_values("timestamp_utc")
            .iloc[0]
        )

        first_detection_lead_minutes = float(
            first_detected_row[
                "minutes_to_failure"
            ]
        )

        first_detection_timestamp = (
            first_detected_row["timestamp_utc"]
            .isoformat()
        )

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=[
            "normal",
            "failure_within_60m",
        ],
        zero_division=0,
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    prediction_columns = [
        "timestamp_utc",
        "experiment_id",
        "label",
        "dataset_role",
        "memory_mib",
        "cpu_millicores",
        "minutes_to_failure",
        TARGET_COLUMN,
        "isolation_forest_prediction",
        "predicted_anomaly",
        "decision_function",
        "anomaly_score",
    ]

    predictions = all_results[
        prediction_columns
    ].copy()

    predictions["timestamp_utc"] = (
        predictions["timestamp_utc"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f%z"
        )
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    experiment_scores = score_summary(
        all_results
    )

    experiment_scores.to_csv(
        SCORE_SUMMARY_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    model_artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "training_experiment": TRAIN_EXPERIMENT,
        "healthy_holdout_experiment": (
            HEALTHY_HOLDOUT_EXPERIMENT
        ),
        "failure_experiment": FAILURE_EXPERIMENT,
        "contamination": CONTAMINATION,
        "random_state": RANDOM_STATE,
        "anomaly_rule": (
            "Isolation Forest prediction -1"
        ),
        "anomaly_score_definition": (
            "Negative decision_function; "
            "higher values indicate greater anomaly"
        ),
    }

    joblib.dump(
        model_artifact,
        MODEL_FILE,
    )

    summary = {
        "model_version": "1.0",
        "model_type": "IsolationForest",
        "purpose": (
            "Detect anomalous Kubernetes resource "
            "behaviour using a model trained only "
            "on healthy observations"
        ),
        "created_with": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": {
            "file": str(
                DATASET_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "sha256": sha256_file(
                DATASET_FILE
            ),
            "total_rows": int(
                len(dataframe)
            ),
        },
        "experiment_design": {
            "healthy_training": TRAIN_EXPERIMENT,
            "healthy_holdout": (
                HEALTHY_HOLDOUT_EXPERIMENT
            ),
            "failure_evaluation": (
                FAILURE_EXPERIMENT
            ),
            "training_rows": int(
                len(training_data)
            ),
            "healthy_holdout_rows": int(
                len(healthy_holdout)
            ),
            "failure_evaluation_rows": int(
                len(failure_evaluation)
            ),
        },
        "target_usage": {
            "used_for_training": False,
            "evaluation_reference": TARGET_COLUMN,
            "explanation": (
                "Failure labels and time-to-failure "
                "values are used only after inference "
                "for evaluation and are not model inputs."
            ),
        },
        "feature_count": int(
            len(FEATURE_COLUMNS)
        ),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "hyperparameters": {
            "n_estimators": 500,
            "max_samples": "auto",
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "evaluation_rows": int(
            len(evaluation_results)
        ),
        "evaluation_target_distribution": (
            class_distribution(actual)
        ),
        "metrics": metrics,
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "operational_metrics": {
            "healthy_holdout_false_positive_rate": (
                healthy_false_positive_rate
            ),
            "failure_experiment_anomaly_rate": (
                failure_experiment_anomaly_rate
            ),
            "failure_window_detection_rate": (
                failure_window_detection_rate
            ),
            "first_detection_timestamp_utc": (
                first_detection_timestamp
            ),
            "first_detection_lead_minutes": (
                first_detection_lead_minutes
            ),
        },
        "experiment_score_summary": (
            experiment_scores
            .to_dict(orient="records")
        ),
        "limitations": [
            (
                "The model was trained on only one "
                "healthy controlled experiment."
            ),
            (
                "The healthy holdout contains the same "
                "heartbeat workload type as training."
            ),
            (
                "Only one controlled memory-failure "
                "experiment is available."
            ),
            (
                "The 60-minute failure label is used "
                "only as an evaluation reference and "
                "is not a complete definition of every "
                "possible operational anomaly."
            ),
            (
                "Additional healthy workloads and "
                "independent failure runs are required "
                "before production generalization can "
                "be claimed."
            ),
        ],
        "artifacts": {
            "model": str(
                MODEL_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "predictions": str(
                PREDICTIONS_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "score_summary": str(
                SCORE_SUMMARY_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "classification_report": str(
                REPORT_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            summary,
            file_handle,
            indent=2,
            allow_nan=False,
        )

    checksum_records = []

    for path, role in [
        (DATASET_FILE, "input"),
        (MODEL_FILE, "generated"),
        (PREDICTIONS_FILE, "generated"),
        (SCORE_SUMMARY_FILE, "generated"),
        (REPORT_FILE, "generated"),
        (SUMMARY_FILE, "generated"),
    ]:
        checksum_records.append(
            {
                "Algorithm": "SHA256",
                "Hash": sha256_file(path),
                "Path": str(path.resolve()),
                "Role": role,
            }
        )

    pd.DataFrame(
        checksum_records
    ).to_csv(
        CHECKSUM_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    print(
        "Isolation Forest training completed "
        "successfully."
    )
    print(
        f"Healthy training rows: "
        f"{len(training_data)}"
    )
    print(
        f"Healthy holdout rows: "
        f"{len(healthy_holdout)}"
    )
    print(
        f"Failure evaluation rows: "
        f"{len(failure_evaluation)}"
    )

    print("\nEvaluation metrics:")

    for metric_name, metric_value in (
        metrics.items()
    ):
        if metric_value is None:
            print(
                f"  {metric_name}: not available"
            )
        else:
            print(
                f"  {metric_name}: "
                f"{metric_value:.4f}"
            )

    print("\nConfusion matrix:")
    print(
        f"  True negatives: {true_negative}"
    )
    print(
        f"  False positives: {false_positive}"
    )
    print(
        f"  False negatives: {false_negative}"
    )
    print(
        f"  True positives: {true_positive}"
    )

    print("\nOperational metrics:")
    print(
        "  Healthy holdout false-positive "
        f"rate: {healthy_false_positive_rate:.4f}"
    )
    print(
        "  Failure experiment anomaly rate: "
        f"{failure_experiment_anomaly_rate:.4f}"
    )
    print(
        "  Failure-window detection rate: "
        f"{failure_window_detection_rate:.4f}"
    )

    if first_detection_lead_minutes is None:
        print(
            "  First detection lead time: "
            "not available"
        )
    else:
        print(
            "  First detection lead time: "
            f"{first_detection_lead_minutes:.2f} "
            "minutes"
        )

    print("\nExperiment score summary:")
    print(
        experiment_scores.to_string(
            index=False
        )
    )

    print(f"\nModel: {MODEL_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(
        f"Predictions: {PREDICTIONS_FILE}"
    )
    print(
        f"Score summary: {SCORE_SUMMARY_FILE}"
    )
    print(f"Checksums: {CHECKSUM_FILE}")


if __name__ == "__main__":
    main()