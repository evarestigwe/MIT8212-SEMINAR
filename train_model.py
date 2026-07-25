from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RESULT_DIR = ROOT / "results"

for directory in (PROCESSED_DIR, MODEL_DIR, REPORT_DIR, RESULT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# NODE-01 is retained in training because it is the only validated
# node-disruption experiment. NODE-02 was completely healthy.
TEST_EXPERIMENTS = {
    "CPU-03",
    "CRASH-01",
    "HB-02",
    "MEM-02",
}

NUMERIC_FEATURES = [
    "cpu_millicores",
    "memory_mib",
    "restart_count",
    "ready_numeric",
    "pod_present",
    "collection_error_present",
]

CATEGORICAL_FEATURES = [
    "phase",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_datasets():
    frames = []

    for csv_path in sorted(RAW_DIR.glob("*_metrics.csv")):
        if csv_path.name == "test_metrics.csv":
            continue

        frame = pd.read_csv(csv_path)
        frame["source_file"] = csv_path.name

        if "experiment_id" not in frame.columns:
            frame["experiment_id"] = csv_path.stem.replace("_metrics", "")

        # NODE-02 had 109/109 healthy observations and no recorded outage.
        # Correct its analytical label without modifying the raw evidence.
        if csv_path.name == "NODE-02_metrics.csv":
            frame["label"] = "healthy"

        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No experimental CSV files found in {RAW_DIR}")

    return pd.concat(frames, ignore_index=True)


def prepare_features(frame):
    prepared = frame.copy()

    required_defaults = {
        "cpu_millicores": None,
        "memory_mib": None,
        "restart_count": 0,
        "ready": False,
        "phase": "Unknown",
        "pod_name": "",
        "collection_error": "",
    }

    for column, default in required_defaults.items():
        if column not in prepared.columns:
            prepared[column] = default

    for column in ["cpu_millicores", "memory_mib", "restart_count"]:
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

    prepared["phase"] = (
        prepared["phase"]
        .fillna("Unknown")
        .astype(str)
        .replace("", "Unknown")
    )

    return prepared


dataset = prepare_features(load_datasets())

dataset.to_csv(
    PROCESSED_DIR / "consolidated_metrics.csv",
    index=False,
)

class_distribution = (
    dataset["label"]
    .value_counts()
    .rename_axis("label")
    .reset_index(name="rows")
)

class_distribution.to_csv(
    REPORT_DIR / "class_distribution.csv",
    index=False,
)

test_mask = dataset["experiment_id"].isin(TEST_EXPERIMENTS)

train_data = dataset.loc[~test_mask].copy()
test_data = dataset.loc[test_mask].copy()

if train_data.empty or test_data.empty:
    raise RuntimeError("The experiment-grouped train/test split is empty.")

test_only_classes = set(test_data["label"]) - set(train_data["label"])
if test_only_classes:
    raise RuntimeError(
        f"Test data contains classes absent from training: {test_only_classes}"
    )

X_train = train_data[ALL_FEATURES]
y_train = train_data["label"]

X_test = test_data[ALL_FEATURES]
y_test = test_data["label"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

evaluation_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)

print("Training grouped evaluation model...")
evaluation_pipeline.fit(X_train, y_train)

predictions = evaluation_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)
weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

labels = sorted(set(y_train) | set(y_test))

report_dictionary = classification_report(
    y_test,
    predictions,
    labels=labels,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    y_test,
    predictions,
    labels=labels,
    zero_division=0,
)

metrics = {
    "total_dataset_rows": int(len(dataset)),
    "training_rows": int(len(train_data)),
    "testing_rows": int(len(test_data)),
    "training_experiments": sorted(
        train_data["experiment_id"].astype(str).unique().tolist()
    ),
    "testing_experiments": sorted(
        test_data["experiment_id"].astype(str).unique().tolist()
    ),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "evaluation_note": (
        "The split is experiment-grouped to reduce temporal leakage. "
        "NODE-01 remains in training because it is the only validated "
        "node-disruption experiment; therefore node_disruption has no "
        "independent held-out evaluation."
    ),
}

with open(
    REPORT_DIR / "evaluation_metrics.json",
    "w",
    encoding="utf-8",
) as output_file:
    json.dump(metrics, output_file, indent=2)

with open(
    REPORT_DIR / "classification_report.json",
    "w",
    encoding="utf-8",
) as output_file:
    json.dump(report_dictionary, output_file, indent=2)

with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8",
) as output_file:
    output_file.write(report_text)

prediction_results = test_data[
    ["timestamp_utc", "experiment_id", "source_file", "label"]
].copy()

prediction_results["predicted_label"] = predictions
prediction_results["correct"] = (
    prediction_results["label"]
    == prediction_results["predicted_label"]
)

prediction_results.to_csv(
    RESULT_DIR / "grouped_test_predictions.csv",
    index=False,
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

plt.figure(figsize=(9, 7))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=300,
)
plt.close()

fitted_preprocessor = evaluation_pipeline.named_steps["preprocessor"]
feature_names = fitted_preprocessor.get_feature_names_out()
importance_values = evaluation_pipeline.named_steps[
    "classifier"
].feature_importances_

feature_importance = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance_values,
        }
    )
    .sort_values("importance", ascending=False)
)

feature_importance.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False,
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance.head(15),
    x="importance",
    y="feature",
    color="steelblue",
)
plt.title("Top Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "feature_importance.png",
    dpi=300,
)
plt.close()

joblib.dump(
    evaluation_pipeline,
    MODEL_DIR / "random_forest_evaluation.joblib",
)

# Retrain the deployable model on all validated experimental rows.
final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Training final model on all validated rows...")
final_pipeline.fit(dataset[ALL_FEATURES], dataset["label"])

joblib.dump(
    final_pipeline,
    MODEL_DIR / "random_forest_final.joblib",
)

print("\nTraining completed successfully.")
print(f"Consolidated rows: {len(dataset)}")
print(f"Grouped training rows: {len(train_data)}")
print(f"Grouped testing rows: {len(test_data)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print("\nClassification report:")
print(report_text)
print(f"Final model: {MODEL_DIR / 'random_forest_final.joblib'}")