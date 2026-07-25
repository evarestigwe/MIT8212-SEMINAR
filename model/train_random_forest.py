from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
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

MODEL_FILE = MODEL_DIR / "random-forest-60m-v1.joblib"
PREDICTIONS_FILE = EVIDENCE_DIR / "RF-60m-test-predictions-v1.csv"
IMPORTANCE_FILE = EVIDENCE_DIR / "RF-60m-feature-importance-v1.csv"
SUMMARY_FILE = EVIDENCE_DIR / "RF-60m-evaluation-summary-v1.json"
REPORT_FILE = EVIDENCE_DIR / "RF-60m-classification-report-v1.txt"
CHECKSUM_FILE = EVIDENCE_DIR / "RF-60m-checksums-v1.csv"

TARGET_COLUMN = "failure_within_60m"
TRAIN_FRACTION = 0.70
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


def load_and_validate_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    dataframe = pd.read_csv(DATASET_FILE)

    required_columns = (
        ["timestamp_utc", "experiment_id", TARGET_COLUMN]
        + FEATURE_COLUMNS
    )

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

    if dataframe[FEATURE_COLUMNS].isna().any().any():
        missing_counts = (
            dataframe[FEATURE_COLUMNS]
            .isna()
            .sum()
        )

        raise ValueError(
            "Model features contain missing values:\n"
            f"{missing_counts[missing_counts.gt(0)]}"
        )

    valid_targets = set(
        dataframe[TARGET_COLUMN].unique().tolist()
    )

    if not valid_targets.issubset({0, 1}):
        raise ValueError(
            f"Target contains invalid values: {valid_targets}"
        )

    if len(dataframe) != 266:
        raise ValueError(
            f"Expected 266 rows, found {len(dataframe)}."
        )

    return (
        dataframe
        .sort_values(["experiment_id", "timestamp_utc"])
        .reset_index(drop=True)
    )


def chronological_group_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_groups = []
    testing_groups = []

    for experiment_id, experiment in dataframe.groupby(
        "experiment_id",
        sort=True,
    ):
        experiment = (
            experiment
            .sort_values("timestamp_utc")
            .reset_index(drop=True)
        )

        split_position = int(
            np.floor(len(experiment) * TRAIN_FRACTION)
        )

        if split_position <= 0 or split_position >= len(experiment):
            raise ValueError(
                f"Invalid split for {experiment_id}: "
                f"{split_position} of {len(experiment)} rows."
            )

        training_groups.append(
            experiment.iloc[:split_position].copy()
        )

        testing_groups.append(
            experiment.iloc[split_position:].copy()
        )

    training_data = (
        pd.concat(training_groups, ignore_index=True)
        .sort_values(["experiment_id", "timestamp_utc"])
        .reset_index(drop=True)
    )

    testing_data = (
        pd.concat(testing_groups, ignore_index=True)
        .sort_values(["experiment_id", "timestamp_utc"])
        .reset_index(drop=True)
    )

    return training_data, testing_data


def class_distribution(
    values: pd.Series,
) -> dict[str, int]:
    counts = (
        values
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    return {
        "negative": int(counts.loc[0]),
        "positive": int(counts.loc[1]),
    }


def rows_by_experiment(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    return {
        str(experiment_id): int(count)
        for experiment_id, count in (
            dataframe
            .groupby("experiment_id")
            .size()
            .items()
        )
    }


def safe_roc_auc(
    actual: pd.Series,
    probability: np.ndarray,
) -> float | None:
    if actual.nunique() < 2:
        return None

    return float(roc_auc_score(actual, probability))


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_and_validate_dataset()

    training_data, testing_data = (
        chronological_group_split(dataframe)
    )

    x_train = training_data[FEATURE_COLUMNS]
    y_train = training_data[TARGET_COLUMN].astype(int)

    x_test = testing_data[FEATURE_COLUMNS]
    y_test = testing_data[TARGET_COLUMN].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one target class. "
            "The model cannot be trained."
        )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    predicted_class = model.predict(x_test)
    predicted_probability = model.predict_proba(x_test)[:, 1]

    confusion = confusion_matrix(
        y_test,
        predicted_class,
        labels=[0, 1],
    )

    true_negative = int(confusion[0, 0])
    false_positive = int(confusion[0, 1])
    false_negative = int(confusion[1, 0])
    true_positive = int(confusion[1, 1])

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predicted_class)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test,
                predicted_class,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predicted_class,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predicted_class,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predicted_class,
                zero_division=0,
            )
        ),
        "roc_auc": safe_roc_auc(
            y_test,
            predicted_probability,
        ),
    }

    predictions = testing_data[
        [
            "timestamp_utc",
            "experiment_id",
            "label",
            TARGET_COLUMN,
        ]
    ].copy()

    predictions["predicted_class"] = predicted_class
    predictions["predicted_probability"] = (
        predicted_probability
    )

    predictions["timestamp_utc"] = (
        predictions["timestamp_utc"]
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_importance["rank"] = (
        np.arange(1, len(feature_importance) + 1)
    )

    feature_importance = feature_importance[
        ["rank", "feature", "importance"]
    ]

    feature_importance.to_csv(
        IMPORTANCE_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    report = classification_report(
        y_test,
        predicted_class,
        labels=[0, 1],
        target_names=[
            "no_failure_within_60m",
            "failure_within_60m",
        ],
        zero_division=0,
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    model_artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "classification_threshold": 0.5,
        "random_state": RANDOM_STATE,
        "training_fraction": TRAIN_FRACTION,
    }

    joblib.dump(model_artifact, MODEL_FILE)

    summary = {
        "model_version": "1.0",
        "model_type": "RandomForestClassifier",
        "purpose": (
            "Predict Kubernetes failure within 60 minutes"
        ),
        "created_with": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": {
            "file": str(
                DATASET_FILE.relative_to(PROJECT_ROOT)
            ),
            "sha256": sha256_file(DATASET_FILE),
            "total_rows": int(len(dataframe)),
        },
        "split_strategy": (
            "Chronological 70/30 split within each experiment"
        ),
        "train_rows": int(len(training_data)),
        "test_rows": int(len(testing_data)),
        "train_rows_by_experiment": rows_by_experiment(
            training_data
        ),
        "test_rows_by_experiment": rows_by_experiment(
            testing_data
        ),
        "train_target_distribution": class_distribution(
            y_train
        ),
        "test_target_distribution": class_distribution(
            y_test
        ),
        "target_column": TARGET_COLUMN,
        "classification_threshold": 0.5,
        "feature_count": int(len(FEATURE_COLUMNS)),
        "feature_columns": FEATURE_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "hyperparameters": {
            "n_estimators": 500,
            "max_depth": 8,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "metrics": metrics,
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "top_10_features": (
            feature_importance
            .head(10)
            .to_dict(orient="records")
        ),
        "limitations": [
            (
                "The dataset contains only three controlled "
                "experiments."
            ),
            (
                "Only one experiment contains an observed "
                "memory-failure event."
            ),
            (
                "Test results are preliminary and do not establish "
                "production generalization."
            ),
            (
                "Additional independent failure runs are required "
                "for stronger external validation."
            ),
        ],
        "artifacts": {
            "model": str(
                MODEL_FILE.relative_to(PROJECT_ROOT)
            ),
            "predictions": str(
                PREDICTIONS_FILE.relative_to(PROJECT_ROOT)
            ),
            "feature_importance": str(
                IMPORTANCE_FILE.relative_to(PROJECT_ROOT)
            ),
            "classification_report": str(
                REPORT_FILE.relative_to(PROJECT_ROOT)
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
        (IMPORTANCE_FILE, "generated"),
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

    pd.DataFrame(checksum_records).to_csv(
        CHECKSUM_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    print("Random Forest training completed successfully.")
    print(f"Dataset rows: {len(dataframe)}")
    print(f"Training rows: {len(training_data)}")
    print(f"Testing rows: {len(testing_data)}")

    print(
        "Training distribution: "
        f"{class_distribution(y_train)}"
    )

    print(
        "Testing distribution: "
        f"{class_distribution(y_test)}"
    )

    print("\nEvaluation metrics:")

    for metric_name, metric_value in metrics.items():
        if metric_value is None:
            print(f"  {metric_name}: not available")
        else:
            print(f"  {metric_name}: {metric_value:.4f}")

    print("\nConfusion matrix:")
    print(f"  True negatives: {true_negative}")
    print(f"  False positives: {false_positive}")
    print(f"  False negatives: {false_negative}")
    print(f"  True positives: {true_positive}")

    print("\nTop 10 features:")
    print(feature_importance.head(10).to_string(index=False))

    print(f"\nModel: {MODEL_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")
    print(f"Feature importance: {IMPORTANCE_FILE}")
    print(f"Checksums: {CHECKSUM_FILE}")


if __name__ == "__main__":
    main()