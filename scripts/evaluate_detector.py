"""Evaluate a trained YOLO detector and save reproducible metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_class_names(
    names: Any,
) -> dict[int, str]:
    """Normalize Ultralytics class names to an integer-keyed mapping."""

    if isinstance(
        names,
        dict,
    ):
        return {
            int(
                key
            ): str(
                value
            )
            for key, value
            in names.items()
        }

    if isinstance(
        names,
        (
            list,
            tuple,
        ),
    ):
        return {
            index: str(
                value
            )
            for index, value
            in enumerate(
                names
            )
        }

    return {}


def build_metric_report(
    metrics: Any,
    class_names: Any,
    model: str,
    data: str,
    split: str,
) -> dict:
    """Convert Ultralytics validation metrics to JSON-safe data."""

    names = normalize_class_names(
        class_names
    )

    box = metrics.box

    per_class_maps = [
        float(
            value
        )
        for value in box.maps
    ]

    per_class = {}

    for class_id, value in enumerate(
        per_class_maps
    ):
        class_name = names.get(
            class_id,
            f"class_{class_id}",
        )

        per_class[
            class_name
        ] = {
            "class_id": class_id,
            "map50_95": value,
        }

    speed = {}

    for key, value in getattr(
        metrics,
        "speed",
        {},
    ).items():
        try:
            speed[
                str(
                    key
                )
            ] = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

    precision = float(
        box.mp
    )

    recall = float(
        box.mr
    )

    if (
        precision
        + recall
        > 0
    ):
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1 = 0.0

    return {
        "model": model,
        "data": data,
        "split": split,
        "primary_metric": (
            "mAP50-95"
        ),
        "metrics": {
            "precision": (
                precision
            ),
            "recall": (
                recall
            ),
            "f1_derived": (
                f1
            ),
            "map50": float(
                box.map50
            ),
            "map75": float(
                box.map75
            ),
            "map50_95": float(
                box.map
            ),
        },
        "per_class": per_class,
        "speed_ms_per_image": (
            speed
        ),
    }


def evaluate_detector(
    model_path: str,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str | None,
    output_path: Path,
) -> dict:
    """Run Ultralytics validation and persist the results."""

    from ultralytics import YOLO

    data_yaml = (
        data_yaml
        .expanduser()
        .resolve()
    )

    if not data_yaml.is_file():
        raise ValueError(
            f"Dataset YAML does not exist: "
            f"{data_yaml}"
        )

    if split not in (
        "val",
        "test",
    ):
        raise ValueError(
            "Evaluation split must be "
            "'val' or 'test'."
        )

    if imgsz <= 0:
        raise ValueError(
            "imgsz must be positive."
        )

    if batch <= 0:
        raise ValueError(
            "batch must be positive."
        )

    model = YOLO(
        model_path
    )

    validation_args = {
        "data": str(
            data_yaml
        ),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "plots": True,
        "verbose": True,
        "project": (
            "runs/evaluation"
        ),
        "name": (
            f"{Path(model_path).stem}_"
            f"{split}"
        ),
    }

    if device is not None:
        validation_args[
            "device"
        ] = device

    metrics = model.val(
        **validation_args
    )

    report = build_metric_report(
        metrics=metrics,
        class_names=(
            model.names
        ),
        model=model_path,
        data=str(
            data_yaml
        ),
        split=split,
    )

    output_path = (
        output_path
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Trained YOLO checkpoint, "
            "for example best.pt."
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
        "--split",
        choices=(
            "val",
            "test",
        ),
        default="val",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Examples: cpu, cuda:0. "
            "Omit for Ultralytics automatic selection."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/evaluation/"
            "metrics.json"
        ),
    )

    args = parser.parse_args()

    try:
        report = evaluate_detector(
            model_path=(
                args.model
            ),
            data_yaml=(
                args.data
            ),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            output_path=(
                args.output
            ),
        )

    except (
        OSError,
        ValueError,
    ) as error:
        parser.exit(
            1,
            f"Evaluation failed: "
            f"{error}\n",
        )

    metrics = report[
        "metrics"
    ]

    print(
        "\nEvaluation complete"
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
        "Results: "
        f"{args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )