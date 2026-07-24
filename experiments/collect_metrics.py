import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def run_kubectl(arguments):
    command = ["kubectl", *arguments]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout.strip()


def cpu_to_millicores(value):
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000

    if value.endswith("u"):
        return float(value[:-1]) / 1_000

    if value.endswith("m"):
        return float(value[:-1])

    return float(value) * 1000


def memory_to_mib(value):
    units = {
        "Ki": 1 / 1024,
        "Mi": 1,
        "Gi": 1024,
        "Ti": 1024 * 1024,
        "K": 1000 / (1024 * 1024),
        "M": 1_000_000 / (1024 * 1024),
        "G": 1_000_000_000 / (1024 * 1024),
    }

    for unit, multiplier in units.items():
        if value.endswith(unit):
            number = float(value[: -len(unit)])
            return number * multiplier

    return float(value) / (1024 * 1024)


def get_pod_metrics(namespace, selector):
    output = run_kubectl(
        [
            "get",
            "--raw",
            f"/apis/metrics.k8s.io/v1beta1/"
            f"namespaces/{namespace}/pods",
        ]
    )

    metrics_data = json.loads(output)

    pod_output = run_kubectl(
        [
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            selector,
            "-o",
            "json",
        ]
    )

    pod_data = json.loads(pod_output)

    pod_details = {}

    for pod in pod_data.get("items", []):
        pod_name = pod["metadata"]["name"]
        container_statuses = pod.get("status", {}).get(
            "containerStatuses", []
        )

        restarts = sum(
            status.get("restartCount", 0)
            for status in container_statuses
        )

        ready = bool(container_statuses) and all(
            status.get("ready", False)
            for status in container_statuses
        )

        pod_details[pod_name] = {
            "phase": pod.get("status", {}).get("phase", "Unknown"),
            "ready": ready,
            "restarts": restarts,
            "node": pod.get("spec", {}).get("nodeName", ""),
        }

    rows = []

    for pod_metric in metrics_data.get("items", []):
        pod_name = pod_metric["metadata"]["name"]

        if pod_name not in pod_details:
            continue

        total_cpu = 0.0
        total_memory = 0.0

        for container in pod_metric.get("containers", []):
            usage = container.get("usage", {})

            total_cpu += cpu_to_millicores(
                usage.get("cpu", "0")
            )

            total_memory += memory_to_mib(
                usage.get("memory", "0")
            )

        details = pod_details[pod_name]

        rows.append(
            {
                "pod_name": pod_name,
                "cpu_millicores": round(total_cpu, 3),
                "memory_mib": round(total_memory, 3),
                "restart_count": details["restarts"],
                "phase": details["phase"],
                "ready": details["ready"],
                "node": details["node"],
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Collect Kubernetes pod metrics into CSV."
    )

    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment identifier, for example HB-01",
    )

    parser.add_argument(
        "--label",
        default="healthy",
        help="Dataset label, for example healthy or cpu_stress",
    )

    parser.add_argument(
        "--namespace",
        default="seminar",
    )

    parser.add_argument(
        "--selector",
        default="app=predictive-app",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5,
        help="Seconds between samples",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Total collection duration in seconds",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Destination CSV file",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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

    file_exists = output_path.exists()
    deadline = time.monotonic() + args.duration
    sample_number = 0

    print(
        f"Collecting {args.experiment_id} metrics for "
        f"{args.duration} seconds..."
    )
    print(f"Output: {output_path}")

    with output_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        if not file_exists or output_path.stat().st_size == 0:
            writer.writeheader()

        while time.monotonic() < deadline:
            timestamp = datetime.now(
                timezone.utc
            ).isoformat()

            try:
                rows = get_pod_metrics(
                    args.namespace,
                    args.selector,
                )

                if not rows:
                    raise RuntimeError(
                        "No matching pod metrics were returned"
                    )

                for row in rows:
                    writer.writerow(
                        {
                            "timestamp_utc": timestamp,
                            "experiment_id":
                                args.experiment_id,
                            "label": args.label,
                            "namespace": args.namespace,
                            **row,
                            "collection_error": "",
                        }
                    )

                sample_number += 1

                for row in rows:
                    print(
                        f"Sample {sample_number}: "
                        f"pod={row['pod_name']} "
                        f"cpu={row['cpu_millicores']}m "
                        f"memory={row['memory_mib']}Mi "
                        f"restarts={row['restart_count']} "
                        f"phase={row['phase']}"
                    )

            except Exception as error:
                writer.writerow(
                    {
                        "timestamp_utc": timestamp,
                        "experiment_id":
                            args.experiment_id,
                        "label": args.label,
                        "namespace": args.namespace,
                        "pod_name": "",
                        "cpu_millicores": "",
                        "memory_mib": "",
                        "restart_count": "",
                        "phase": "",
                        "ready": "",
                        "node": "",
                        "collection_error": str(error),
                    }
                )

                print(f"Collection error: {error}")

            csv_file.flush()

            remaining = deadline - time.monotonic()

            if remaining > 0:
                time.sleep(min(args.interval, remaining))

    print(
        f"Collection complete. "
        f"{sample_number} samples collected."
    )


if __name__ == "__main__":
    main()