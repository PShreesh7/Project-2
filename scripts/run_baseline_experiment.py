"""Train and evaluate the first YOLOv8n NEU-DET baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluate_detector import (
    evaluate_detector,
)
from scripts.train_yolov8 import (
    train,
)


def run_baseline_experiment(
    dataset_root: Path,
    config_path: Path,
    device: str | None = None,
) -> dict:
    """Train baseline model and evaluate best checkpoint on validation data."""

    dataset_root = (
        dataset_root
        .expanduser()
        .resolve()
    )

    config_path = (
        config_path
        .expanduser()
        .resolve()
    )

    training_result = train(
        dataset_root=dataset_root,
        config_path=config_path,
        requested_device=device,
        smoke=False,
    )

    run_directory = Path(
        training_result[
            "run_directory"
        ]
    )

    best_checkpoint = (
        run_directory
        / "weights"
        / "best.pt"
    )

    if not best_checkpoint.is_file():
        raise ValueError(
            "Baseline training completed but "
            f"best.pt was not found: {best_checkpoint}"
        )

    data_yaml = (
        dataset_root
        / "data.yaml"
    )

    if not data_yaml.is_file():
        raise ValueError(
            f"Dataset YAML is missing: {data_yaml}"
        )

    validation_output = (
        run_directory
        / "baseline_val_metrics.json"
    )

    validation_report = (
        evaluate_detector(
            model_path=str(
                best_checkpoint
            ),
            data_yaml=data_yaml,
            split="val",
            imgsz=416,
            batch=4,
            device=device,
            output_path=(
                validation_output
            ),
        )
    )

    summary = {
        "status": "PASSED",
        "experiment": (
            "YOLOv8n baseline"
        ),
        "dataset_root": str(
            dataset_root
        ),
        "config": str(
            config_path
        ),
        "run_directory": str(
            run_directory
        ),
        "best_checkpoint": str(
            best_checkpoint
        ),
        "validation_metrics_file": str(
            validation_output
        ),
        "validation_metrics": (
            validation_report[
                "metrics"
            ]
        ),
        "per_class": (
            validation_report[
                "per_class"
            ]
        ),
        "speed_ms_per_image": (
            validation_report[
                "speed_ms_per_image"
            ]
        ),
        "test_set_evaluated": False,
    }

    summary_path = (
        run_directory
        / "baseline_experiment_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo_augmented"
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/"
            "yolov8n_baseline.json"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional device override, "
            "for example cpu or cuda:0."
        ),
    )

    args = parser.parse_args()

    try:

        summary = (
            run_baseline_experiment(
                dataset_root=(
                    args.dataset_root
                ),
                config_path=(
                    args.config
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
            f"Baseline experiment failed: "
            f"{error}\n",
        )

    metrics = (
        summary[
            "validation_metrics"
        ]
    )

    print(
        "\nBaseline experiment: PASSED"
    )

    print(
        "Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        "Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        "F1: "
        f"{metrics['f1_derived']:.4f}"
    )

    print(
        "mAP50: "
        f"{metrics['map50']:.4f}"
    )

    print(
        "mAP75: "
        f"{metrics['map75']:.4f}"
    )

    print(
        "mAP50-95: "
        f"{metrics['map50_95']:.4f}"
    )

    print(
        "\nBest checkpoint: "
        f"{summary['best_checkpoint']}"
    )

    print(
        "Test set evaluated: "
        f"{summary['test_set_evaluated']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )