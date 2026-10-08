"""Benchmark YOLO validation accuracy and speed across image resolutions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.evaluate_detector import (
    evaluate_detector,
)
from scripts.train_yolov8 import (
    select_device,
)


def load_tuning_config(
    config_path: Path,
) -> dict:
    """Load and validate resolution tuning configuration."""

    config_path = (
        config_path
        .expanduser()
        .resolve()
    )

    if not config_path.is_file():
        raise ValueError(
            f"Tuning configuration does not exist: "
            f"{config_path}"
        )

    try:
        config = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid tuning configuration JSON."
        ) from error

    resolutions = config.get(
        "resolutions"
    )

    if (
        not isinstance(
            resolutions,
            list,
        )
        or not resolutions
    ):
        raise ValueError(
            "resolutions must be a non-empty list."
        )

    for resolution in resolutions:

        if (
            not isinstance(
                resolution,
                int,
            )
            or isinstance(
                resolution,
                bool,
            )
            or resolution <= 0
        ):
            raise ValueError(
                "Every resolution must be "
                "a positive integer."
            )

        if (
            resolution % 32
            != 0
        ):
            raise ValueError(
                f"Resolution {resolution} must "
                "be divisible by 32."
            )

    batch = config.get(
        "batch",
        4,
    )

    if (
        not isinstance(
            batch,
            int,
        )
        or isinstance(
            batch,
            bool,
        )
        or batch <= 0
    ):
        raise ValueError(
            "batch must be a positive integer."
        )

    tolerance = config.get(
        "map_tolerance",
        0.01,
    )

    if (
        not isinstance(
            tolerance,
            (
                int,
                float,
            ),
        )
        or tolerance < 0
    ):
        raise ValueError(
            "map_tolerance must be "
            "a non-negative number."
        )

    split = config.get(
        "split",
        "val",
    )

    if split != "val":
        raise ValueError(
            "Resolution tuning must use "
            "the validation split only."
        )

    return config


def extract_inference_ms(
    report: dict,
) -> float:
    """Extract inference milliseconds per image."""

    speed = report.get(
        "speed_ms_per_image",
        {},
    )

    inference = speed.get(
        "inference"
    )

    if inference is None:
        raise ValueError(
            "Evaluation report does not contain "
            "inference timing."
        )

    inference = float(
        inference
    )

    if inference <= 0:
        raise ValueError(
            "Inference time must be positive."
        )

    return inference


def add_runtime_metrics(
    report: dict,
    resolution: int,
) -> dict:
    """Convert evaluation output into one tuning record."""

    metrics = report[
        "metrics"
    ]

    inference_ms = (
        extract_inference_ms(
            report
        )
    )

    fps = (
        1000.0
        / inference_ms
    )

    return {
        "imgsz": resolution,
        "precision": float(
            metrics[
                "precision"
            ]
        ),
        "recall": float(
            metrics[
                "recall"
            ]
        ),
        "f1": float(
            metrics[
                "f1_derived"
            ]
        ),
        "map50": float(
            metrics[
                "map50"
            ]
        ),
        "map75": float(
            metrics[
                "map75"
            ]
        ),
        "map50_95": float(
            metrics[
                "map50_95"
            ]
        ),
        "inference_ms": (
            inference_ms
        ),
        "fps": fps,
    }


def select_balanced_resolution(
    results: list[dict],
    map_tolerance: float,
) -> dict:
    """Choose fastest resolution close to maximum mAP50-95."""

    if not results:
        raise ValueError(
            "No tuning results were provided."
        )

    best_map = max(
        result[
            "map50_95"
        ]
        for result in results
    )

    minimum_acceptable_map = (
        best_map
        - map_tolerance
    )

    candidates = [
        result
        for result in results
        if result[
            "map50_95"
        ]
        >= minimum_acceptable_map
    ]

    selected = min(
        candidates,
        key=lambda result: (
            result[
                "inference_ms"
            ],
            -result[
                "map50_95"
            ],
        ),
    )

    return {
        "best_map50_95": (
            best_map
        ),
        "minimum_acceptable_map50_95": (
            minimum_acceptable_map
        ),
        "selected": (
            selected
        ),
    }


def write_csv(
    results: list[dict],
    path: Path,
) -> Path:
    """Write tuning results as CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "imgsz",
        "precision",
        "recall",
        "f1",
        "map50",
        "map75",
        "map50_95",
        "inference_ms",
        "fps",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    return path


def run_resolution_tuning(
    model_path: Path,
    data_yaml: Path,
    config_path: Path,
    output_directory: Path,
    device: str | None = None,
) -> dict:
    """Evaluate one trained model at multiple validation resolutions."""

    model_path = (
        model_path
        .expanduser()
        .resolve()
    )

    data_yaml = (
        data_yaml
        .expanduser()
        .resolve()
    )

    output_directory = (
        output_directory
        .expanduser()
        .resolve()
    )

    if not model_path.is_file():
        raise ValueError(
            f"Model does not exist: "
            f"{model_path}"
        )

    if not data_yaml.is_file():
        raise ValueError(
            f"Dataset YAML does not exist: "
            f"{data_yaml}"
        )

    config = load_tuning_config(
        config_path
    )

    selected_device = (
        select_device(
            device
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for resolution in config[
        "resolutions"
    ]:

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Evaluating imgsz={resolution}"
        )

        print(
            f"Device: {selected_device}"
        )

        print(
            "=" * 60
        )

        metrics_file = (
            output_directory
            / (
                f"resolution_"
                f"{resolution}_metrics.json"
            )
        )

        report = (
            evaluate_detector(
                model_path=str(
                    model_path
                ),
                data_yaml=(
                    data_yaml
                ),
                split="val",
                imgsz=(
                    resolution
                ),
                batch=int(
                    config[
                        "batch"
                    ]
                ),
                device=(
                    selected_device
                ),
                output_path=(
                    metrics_file
                ),
            )
        )

        record = (
            add_runtime_metrics(
                report,
                resolution,
            )
        )

        results.append(
            record
        )

    selection = (
        select_balanced_resolution(
            results=results,
            map_tolerance=float(
                config[
                    "map_tolerance"
                ]
            ),
        )
    )

    summary = {
        "status": "PASSED",
        "experiment": (
            "YOLOv8 validation resolution tuning"
        ),
        "model": str(
            model_path
        ),
        "data": str(
            data_yaml
        ),
        "split": "val",
        "device": (
            selected_device
        ),
        "map_tolerance": float(
            config[
                "map_tolerance"
            ]
        ),
        "results": (
            results
        ),
        "best_map50_95": (
            selection[
                "best_map50_95"
            ]
        ),
        "minimum_acceptable_map50_95": (
            selection[
                "minimum_acceptable_map50_95"
            ]
        ),
        "selected_resolution": (
            selection[
                "selected"
            ][
                "imgsz"
            ]
        ),
        "selected_result": (
            selection[
                "selected"
            ]
        ),
        "test_set_evaluated": False,
    }

    json_path = (
        output_directory
        / "resolution_tuning_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = (
        output_directory
        / "resolution_tuning_results.csv"
    )

    write_csv(
        results,
        csv_path,
    )

    summary[
        "summary_file"
    ] = str(
        json_path
    )

    summary[
        "csv_file"
    ] = str(
        csv_path
    )

    return summary


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "runs/detect/"
            "neu_yolov8n_baseline/"
            "weights/best.pt"
        ),
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo_augmented/"
            "data.yaml"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/"
            "resolution_tuning.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/tuning/"
            "resolution"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    args = parser.parse_args()

    try:

        summary = (
            run_resolution_tuning(
                model_path=(
                    args.model
                ),
                data_yaml=(
                    args.data
                ),
                config_path=(
                    args.config
                ),
                output_directory=(
                    args.output
                ),
                device=(
                    args.device
                ),
            )
        )

    except (
        OSError,
        RuntimeError,
        ValueError,
    ) as error:

        parser.exit(
            1,
            f"Resolution tuning failed: "
            f"{error}\n",
        )

    print(
        "\nResolution tuning: PASSED"
    )

    print(
        "\nResults:"
    )

    for result in summary[
        "results"
    ]:

        print(
            (
                f"imgsz={result['imgsz']}: "
                f"mAP50-95="
                f"{result['map50_95']:.4f}, "
                f"inference="
                f"{result['inference_ms']:.2f} ms, "
                f"FPS="
                f"{result['fps']:.2f}"
            )
        )

    selected = (
        summary[
            "selected_result"
        ]
    )

    print(
        "\nSelected operating point:"
    )

    print(
        f"Image size: "
        f"{selected['imgsz']}"
    )

    print(
        f"mAP50-95: "
        f"{selected['map50_95']:.4f}"
    )

    print(
        f"Inference: "
        f"{selected['inference_ms']:.2f} ms"
    )

    print(
        f"FPS: "
        f"{selected['fps']:.2f}"
    )

    print(
        "\nTest set evaluated: False"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )