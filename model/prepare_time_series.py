from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILES = [
    PROJECT_ROOT / "data/time_series/processed/TS-HB-01_processed.csv",
    PROJECT_ROOT / "data/time_series/processed/TS-HB-02_processed.csv",
    PROJECT_ROOT / "data/time_series/raw/TS-MEM-01_metrics.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "data/time_series/processed"
EVIDENCE_DIR = PROJECT_ROOT / "evidence/time_series"

OUTPUT_FILE = OUTPUT_DIR / "TS-predictive-dataset-v1.csv"
SUMMARY_FILE = EVIDENCE_DIR / "TS-preprocessing-summary-v1.json"
CHECKSUM_FILE = EVIDENCE_DIR / "TS-preprocessing-checksums-v1.csv"

EXPECTED_ROWS_PER_EXPERIMENT = 90
ROLLING_WINDOWS = [5, 15, 30]
PREDICTION_HORIZONS = [5, 15, 30, 60]

REQUIRED_COLUMNS = [
    "timestamp_utc",
    "experiment_id",
    "label",
    "namespace",
    "pod_name",
    "cpu_millicores",
    "memory_mib",
    "restart_count",
    "phase",
    "ready",
    "node",
    "collection_error",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest().upper()


def validate_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    dataframe = pd.read_csv(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: {missing_columns}"
        )

    if len(dataframe) != EXPECTED_ROWS_PER_EXPERIMENT:
        raise ValueError(
            f"{path.name} contains {len(dataframe)} rows; "
            f"expected {EXPECTED_ROWS_PER_EXPERIMENT}."
        )

    experiment_ids = dataframe["experiment_id"].dropna().unique()

    if len(experiment_ids) != 1:
        raise ValueError(
            f"{path.name} must contain exactly one experiment ID. "
            f"Found: {experiment_ids.tolist()}"
        )

    errors = (
        dataframe["collection_error"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if errors.ne("").any():
        invalid_count = int(errors.ne("").sum())
        raise ValueError(
            f"{path.name} contains {invalid_count} collection errors."
        )

    dataframe["timestamp_utc"] = pd.to_datetime(
        dataframe["timestamp_utc"],
        utc=True,
        errors="raise",
    )

    for column in [
        "cpu_millicores",
        "memory_mib",
        "restart_count",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe["ready_numeric"] = (
        dataframe["ready"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0})
    )

    if dataframe["ready_numeric"].isna().any():
        raise ValueError(
            f"{path.name} contains invalid values in the ready column."
        )

    dataframe = (
        dataframe
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    if dataframe["timestamp_utc"].duplicated().any():
        raise ValueError(
            f"{path.name} contains duplicate timestamps."
        )

    return dataframe


def locate_memory_failure(
    dataframe: pd.DataFrame,
) -> tuple[pd.Timestamp, int]:
    memory_rows = dataframe.loc[
        dataframe["experiment_id"].eq("TS-MEM-01")
    ].copy()

    if memory_rows.empty:
        raise ValueError(
            "TS-MEM-01 was not found in the input datasets."
        )

    memory_rows = memory_rows.sort_values("timestamp_utc")

    restart_change = (
        memory_rows["restart_count"]
        .diff()
        .fillna(0)
    )

    transitions = memory_rows.loc[restart_change.gt(0)]

    if len(transitions) != 1:
        raise ValueError(
            "Expected exactly one restart transition in TS-MEM-01, "
            f"but found {len(transitions)}."
        )

    failure_row = transitions.iloc[0]

    return (
        failure_row["timestamp_utc"],
        int(failure_row["restart_count"]),
    )


def first_value(series: pd.Series) -> float:
    return float(series.iloc[0])


def add_causal_features(
    experiment: pd.DataFrame,
) -> pd.DataFrame:
    experiment = experiment.sort_values("timestamp_utc").copy()
    experiment = experiment.set_index("timestamp_utc")

    experiment["elapsed_seconds"] = (
        experiment.index - experiment.index.min()
    ).total_seconds()

    for metric in ["cpu_millicores", "memory_mib"]:
        for minutes in ROLLING_WINDOWS:
            rolling = experiment[metric].rolling(
                window=f"{minutes}min",
                min_periods=1,
                closed="both",
            )

            prefix = f"{metric}_roll_{minutes}m"

            experiment[f"{prefix}_mean"] = rolling.mean()

            experiment[f"{prefix}_std"] = (
                rolling.std(ddof=0).fillna(0.0)
            )

            experiment[f"{prefix}_min"] = rolling.min()
            experiment[f"{prefix}_max"] = rolling.max()

            experiment[f"{prefix}_range"] = (
                experiment[f"{prefix}_max"]
                - experiment[f"{prefix}_min"]
            )

            experiment[f"{prefix}_delta"] = (
                experiment[metric]
                - rolling.apply(first_value, raw=False)
            )

    experiment["cpu_change_1m"] = (
        experiment["cpu_millicores"]
        .diff()
        .fillna(0.0)
    )

    experiment["memory_change_1m"] = (
        experiment["memory_mib"]
        .diff()
        .fillna(0.0)
    )

    time_difference_minutes = (
        experiment.index
        .to_series()
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    experiment["memory_growth_rate_mib_per_min"] = (
        experiment["memory_mib"].diff()
        / time_difference_minutes
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    return experiment.reset_index()


def create_predictive_labels(
    dataframe: pd.DataFrame,
    failure_time: pd.Timestamp,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["failure_timestamp_utc"] = pd.Series(
    pd.NaT,
    index=dataframe.index,
    dtype="datetime64[ns, UTC]",
)
    dataframe["minutes_to_failure"] = np.nan

    memory_mask = dataframe["experiment_id"].eq("TS-MEM-01")

    dataframe.loc[
        memory_mask,
        "failure_timestamp_utc",
    ] = failure_time

    dataframe.loc[
        memory_mask,
        "minutes_to_failure",
    ] = (
        failure_time
        - dataframe.loc[memory_mask, "timestamp_utc"]
    ).dt.total_seconds().div(60)

    for horizon in PREDICTION_HORIZONS:
        target_column = f"failure_within_{horizon}m"

        dataframe[target_column] = (
            memory_mask
            & dataframe["minutes_to_failure"].gt(0)
            & dataframe["minutes_to_failure"].le(horizon)
        ).astype(int)

    # Exclude the failure transition and post-failure recovery rows.
    training_mask = (
        ~memory_mask
        | dataframe["timestamp_utc"].lt(failure_time)
    )

    return dataframe.loc[training_mask].copy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    input_summary = []

    for input_path in INPUT_FILES:
        frame = validate_input(input_path)
        frames.append(frame)

        input_summary.append(
            {
                "file": str(
                    input_path.relative_to(PROJECT_ROOT)
                ),
                "experiment_id": str(
                    frame["experiment_id"].iloc[0]
                ),
                "rows": int(len(frame)),
                "collection_errors": 0,
                "sha256": sha256_file(input_path),
            }
        )

    combined = pd.concat(frames, ignore_index=True)

    if len(combined) != 270:
        raise ValueError(
            f"Combined input contains {len(combined)} rows; "
            "expected 270."
        )

    failure_time, restart_count_after_failure = (
        locate_memory_failure(combined)
    )

    featured_frames = []

    for _, experiment in combined.groupby(
        "experiment_id",
        sort=True,
    ):
        featured_frames.append(
            add_causal_features(experiment)
        )

    processed = pd.concat(
        featured_frames,
        ignore_index=True,
    )

    processed = create_predictive_labels(
        processed,
        failure_time,
    )

    processed = (
        processed
        .sort_values(["experiment_id", "timestamp_utc"])
        .reset_index(drop=True)
    )

    processed["timestamp_utc"] = (
        processed["timestamp_utc"]
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    )

    processed["failure_timestamp_utc"] = (
        pd.to_datetime(
            processed["failure_timestamp_utc"],
            utc=True,
            errors="coerce",
        )
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        .fillna("")
    )

    processed.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    target_distribution = {}

    for horizon in PREDICTION_HORIZONS:
        target_column = f"failure_within_{horizon}m"
        positive_count = int(processed[target_column].sum())

        target_distribution[target_column] = {
            "positive": positive_count,
            "negative": int(
                len(processed) - positive_count
            ),
        }

    experiment_rows = {
        str(experiment_id): int(count)
        for experiment_id, count in (
            processed
            .groupby("experiment_id")
            .size()
            .items()
        )
    }

    summary = {
        "pipeline_version": "1.0",
        "purpose": (
            "Leakage-safe time-series preprocessing for "
            "Kubernetes failure prediction"
        ),
        "input_rows": int(len(combined)),
        "output_rows": int(len(processed)),
        "excluded_rows": int(
            len(combined) - len(processed)
        ),
        "failure_experiment": "TS-MEM-01",
        "failure_detection_method": (
            "First positive restart_count transition"
        ),
        "failure_timestamp_utc": failure_time.isoformat(),
        "restart_count_after_failure": (
            restart_count_after_failure
        ),
        "rolling_windows_minutes": ROLLING_WINDOWS,
        "prediction_horizons_minutes": PREDICTION_HORIZONS,
        "experiment_output_rows": experiment_rows,
        "target_distribution": target_distribution,
        "leakage_controls": [
            (
                "Rolling features use current and historical "
                "observations only"
            ),
            (
                "Restart transition row excluded from "
                "model dataset"
            ),
            (
                "Post-failure recovery rows excluded from "
                "model dataset"
            ),
            "OOMKilled event is not used as an input feature",
            (
                "Future measurements are not used as "
                "input features"
            ),
            (
                "restart_count is retained for audit but "
                "should not be selected as a model feature"
            ),
            (
                "experiment_id and label are metadata, "
                "not model features"
            ),
        ],
        "inputs": input_summary,
        "output": {
            "file": str(
                OUTPUT_FILE.relative_to(PROJECT_ROOT)
            ),
            "sha256": sha256_file(OUTPUT_FILE),
        },
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(summary, file_handle, indent=2)

    checksum_records = []

    for input_path in INPUT_FILES:
        checksum_records.append(
            {
                "Algorithm": "SHA256",
                "Hash": sha256_file(input_path),
                "Path": str(input_path.resolve()),
                "Role": "input",
            }
        )

    for generated_path in [
        OUTPUT_FILE,
        SUMMARY_FILE,
    ]:
        checksum_records.append(
            {
                "Algorithm": "SHA256",
                "Hash": sha256_file(generated_path),
                "Path": str(generated_path.resolve()),
                "Role": "generated",
            }
        )

    pd.DataFrame(checksum_records).to_csv(
        CHECKSUM_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    print("Preprocessing completed successfully.")
    print(f"Input rows: {len(combined)}")
    print(f"Output rows: {len(processed)}")

    print(
        "Excluded failure/recovery rows: "
        f"{len(combined) - len(processed)}"
    )

    print(
        f"Detected failure: {failure_time.isoformat()}"
    )

    print("\nOutput rows by experiment:")

    print(
        processed
        .groupby("experiment_id")
        .size()
        .to_string()
    )

    print("\nPredictive target distribution:")

    for target, distribution in target_distribution.items():
        print(
            f"  {target}: "
            f"positive={distribution['positive']}, "
            f"negative={distribution['negative']}"
        )

    print(f"\nDataset: {OUTPUT_FILE}")
    print(f"Summary: {SUMMARY_FILE}")
    print(f"Checksums: {CHECKSUM_FILE}")


if __name__ == "__main__":
    main()