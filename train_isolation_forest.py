from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RESULT_DIR = ROOT / "results"

for directory in (MODEL_DIR, REPORT_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


BASELINE_EXPERIMENTS = {"HB-01", "NODE-02"}
HEALTHY_TEST_EXPERIMENTS = {"HB-02"}

FEATURES = [
    "cpu_millicores",
    "memory_mib",
    "restart_count",
    "ready_numeric",
    "pod_present",
    "collection_error_present",
]


def load_data():
    frames = []

    for path in sorted(RAW_DIR.glob("*_metrics.csv")):
        if path.name == "test_metrics.csv":
            continue

        frame = pd.read_csv(path)
        experiment_id = path.stem.replace("_metrics", "")

        frame["experiment_id"] = experiment_id
        frame["source_file"] = path.name

        # NODE-02 contained 109/109 healthy observations and no outage.
        if experiment_id == "NODE-02":
            frame["label"] = "healthy"

        frames.append(frame)

    if not frames:
        raise RuntimeError("No experimental datasets were found.")

    return pd.concat(frames, ignore_index=True)


def prepare_features(frame):
    prepared = frame.copy()

    defaults = {
        "cpu_millicores": None,
        "memory_mib": None,
        "restart_count": 0,
        "ready": False,
        "pod_name": "",
        "collection_error": "",
    }

    for column, default in defaults.items():
        if column not in prepared.columns:
            prepared[column] = default

    for column in [
        "cpu_millicores",
        "memory_mib",
        "restart_count",
    ]:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared["ready_numeric"] = (
        prepared["ready"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0})
        .fillna(0)
        .astype(int)
    )

    prepared["pod_present"] = (
        prepared["pod_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    prepared["collection_error_present"] = (
        prepared["collection_error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .astype(int)
    )

    # Binary ground truth is used only for evaluation.
    prepared["actual_status"] = prepared["label"].apply(
        lambda value: "normal" if value == "healthy" else "anomaly"
    )

    return prepared


dataset = prepare_features(load_data())

baseline = dataset[
    dataset["experiment_id"].isin(BASELINE_EXPERIMENTS)
].copy()

evaluation = dataset[
    ~dataset["experiment_id"].isin(BASELINE_EXPERIMENTS)
].copy()

if baseline.empty:
    raise RuntimeError("The healthy baseline dataset is empty.")

if not (baseline["label"] == "healthy").all():
    raise RuntimeError("Baseline contains nonhealthy labels.")

pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "detector",
            IsolationForest(
                n_estimators=300,
                contamination=0.05,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Training Isolation Forest on healthy baseline...")
pipeline.fit(baseline[FEATURES])

raw_predictions = pipeline.predict(evaluation[FEATURES])

evaluation["predicted_status"] = [
    "normal" if prediction == 1 else "anomaly"
    for prediction in raw_predictions
]

# Lower Isolation Forest decision values indicate stronger anomalies.
evaluation["decision_score"] = pipeline.decision_function(
    evaluation[FEATURES]
)

evaluation["anomaly_score"] = -evaluation["decision_score"]

actual_binary = (
    evaluation["actual_status"] == "anomaly"
).astype(int)

predicted_binary = (
    evaluation["predicted_status"] == "anomaly"
).astype(int)

accuracy = accuracy_score(actual_binary, predicted_binary)
precision = precision_score(
    actual_binary,
    predicted_binary,
    zero_division=0,
)
recall = recall_score(
    actual_binary,
    predicted_binary,
    zero_division=0,
)
f1 = f1_score(
    actual_binary,
    predicted_binary,
    zero_division=0,
)
roc_auc = roc_auc_score(
    actual_binary,
    evaluation["anomaly_score"],
)

report_text = classification_report(
    evaluation["actual_status"],
    evaluation["predicted_status"],
    labels=["normal", "anomaly"],
    zero_division=0,
)

metrics = {
    "total_validated_rows": int(len(dataset)),
    "baseline_training_rows": int(len(baseline)),
    "evaluation_rows": int(len(evaluation)),
    "baseline_experiments": sorted(BASELINE_EXPERIMENTS),
    "independent_healthy_test_experiments": sorted(
        HEALTHY_TEST_EXPERIMENTS
    ),
    "contamination": 0.05,
    "accuracy": float(accuracy),
    "anomaly_precision": float(precision),
    "anomaly_recall": float(recall),
    "anomaly_f1": float(f1),
    "roc_auc": float(roc_auc),
    "methodological_note": (
        "Isolation Forest was fitted only on validated healthy observations "
        "from HB-01 and NODE-02. NODE-02 was analytically relabelled as "
        "healthy because no node outage occurred. Evaluation used unseen "
        "healthy observations and all recorded failure-condition rows."
    ),
}

with open(
    REPORT_DIR / "isolation_forest_metrics.json",
    "w",
    encoding="utf-8",
) as output:
    json.dump(metrics, output, indent=2)

with open(
    REPORT_DIR / "isolation_forest_report.txt",
    "w",
    encoding="utf-8",
) as output:
    output.write(report_text)

result_columns = [
    "timestamp_utc",
    "experiment_id",
    "source_file",
    "label",
    "actual_status",
    "predicted_status",
    "decision_score",
    "anomaly_score",
]

evaluation[result_columns].to_csv(
    RESULT_DIR / "isolation_forest_predictions.csv",
    index=False,
)

labels = ["normal", "anomaly"]
matrix = confusion_matrix(
    evaluation["actual_status"],
    evaluation["predicted_status"],
    labels=labels,
)

plt.figure(figsize=(7, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Isolation Forest Confusion Matrix")
plt.xlabel("Predicted status")
plt.ylabel("Actual status")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "isolation_forest_confusion_matrix.png",
    dpi=300,
)
plt.close()

score_summary = (
    evaluation.groupby("label")["anomaly_score"]
    .agg(["count", "mean", "median", "min", "max"])
    .reset_index()
)

score_summary.to_csv(
    REPORT_DIR / "isolation_forest_score_summary.csv",
    index=False,
)

joblib.dump(
    pipeline,
    MODEL_DIR / "isolation_forest_final.joblib",
)

print("\nIsolation Forest training completed.")
print(f"Healthy baseline rows: {len(baseline)}")
print(f"Evaluation rows: {len(evaluation)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Anomaly precision: {precision:.4f}")
print(f"Anomaly recall: {recall:.4f}")
print(f"Anomaly F1: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification report:")
print(report_text)