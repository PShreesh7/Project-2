"""Run controlled YOLOv8 training hyperparameter experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_detector import (
    evaluate_detector,
)
from scripts.train_yolov8 import (
    train,
)
from scripts.training_preflight import (
    load_training_config,
)


def load_json(
    path: Path,
) -> dict:
    """Load a JSON object from disk."""

    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise ValueError(
            f"Required JSON file does not exist: "
            f"{path}"
        )

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON file: {path}"
        ) from error

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"Expected a JSON object: {path}"
        )

    return value


def validate_search_config(
    config: dict,
) -> list[str]:
    """Validate controlled tuning configuration."""

    errors: list[str] = []

    epochs = config.get(
        "search_epochs"
    )

    if (
        not isinstance(
            epochs,
            int,
        )
        or isinstance(
            epochs,
            bool,
        )
        or epochs <= 0
    ):
        errors.append(
            "search_epochs must be "
            "a positive integer."
        )

    batch = config.get(
        "batch"
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
        errors.append(
            "batch must be "
            "a positive integer."
        )

    tolerance = config.get(
        "map_tolerance",
        0.005,
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
        errors.append(
            "map_tolerance must be "
            "non-negative."
        )

    experiments = config.get(
        "experiments"
    )

    if (
        not isinstance(
            experiments,
            list,
        )
        or not experiments
    ):
        errors.append(
            "experiments must be "
            "a non-empty list."
        )

        return errors

    names = set()

    for index, experiment in enumerate(
        experiments,
        start=1,
    ):

        if not isinstance(
            experiment,
            dict,
        ):
            errors.append(
                f"Experiment {index} must "
                "be an object."
            )

            continue

        name = experiment.get(
            "name"
        )

        if (
            not isinstance(
                name,
                str,
            )
            or not name.strip()
        ):
            errors.append(
                f"Experiment {index} needs "
                "a non-empty name."
            )

        elif name in names:

            errors.append(
                f"Duplicate experiment name: "
                f"{name}"
            )

        else:
            names.add(
                name
            )

        optimizer = experiment.get(
            "optimizer"
        )

        if optimizer not in (
            "AdamW",
            "SGD",
        ):
            errors.append(
                f"{name}: optimizer must be "
                "AdamW or SGD."
            )

        lr0 = experiment.get(
            "lr0"
        )

        if (
            not isinstance(
                lr0,
                (
                    int,
                    float,
                ),
            )
            or lr0 <= 0
        ):
            errors.append(
                f"{name}: lr0 must be positive."
            )

    return errors


def selected_resolution_from_report(
    report: dict,
) -> int:
    """Read Commit 11's selected validation resolution."""

    if (
        report.get(
            "status"
        )
        != "PASSED"
    ):
        raise ValueError(
            "Commit 11 resolution tuning "
            "did not pass."
        )

    if report.get(
        "test_set_evaluated"
    ) is not False:
        raise ValueError(
            "Resolution tuning report does not "
            "confirm that the test set remained "
            "untouched."
        )

    resolution = report.get(
        "selected_resolution"
    )

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
        or resolution % 32 != 0
    ):
        raise ValueError(
            "Commit 11 selected_resolution "
            "is invalid."
        )

    return resolution


def build_candidate_config(
    baseline: dict,
    search: dict,
    experiment: dict,
    resolution: int,
) -> dict:
    """Build one controlled training configuration."""

    candidate = dict(
        baseline
    )

    candidate[
        "model"
    ] = "yolov8n.pt"

    candidate[
        "task"
    ] = "detect"

    candidate[
        "epochs"
    ] = int(
        search[
            "search_epochs"
        ]
    )

    candidate[
        "imgsz"
    ] = resolution

    candidate[
        "batch"
    ] = int(
        search[
            "batch"
        ]
    )

    candidate[
        "workers"
    ] = int(
        search.get(
            "workers",
            0,
        )
    )

    candidate[
        "patience"
    ] = int(
        search.get(
            "patience",
            4,
        )
    )

    candidate[
        "seed"
    ] = int(
        search.get(
            "seed",
            42,
        )
    )

    candidate[
        "deterministic"
    ] = True

    candidate[
        "project"
    ] = (
        "runs/tuning/"
        "hyperparameters"
    )

    candidate[
        "name"
    ] = (
        "neu_yolov8n_"
        + experiment[
            "name"
        ]
    )

    # Full dataset must be used for tuning.
    candidate.pop(
        "fraction",
        None,
    )

    for key in (
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
    ):

        if key in experiment:

            candidate[
                key
            ] = experiment[
                key
            ]

        else:

            candidate.pop(
                key,
                None,
            )

    # Preserve the controlled offline
    # augmentation policy.
    candidate[
        "offline_augmentation_only"
    ] = True

    candidate[
        "training_augmentation"
    ] = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
    }

    return candidate


def inference_ms_from_report(
    report: dict,
) -> float:
    """Extract measured inference latency."""

    value = (
        report.get(
            "speed_ms_per_image",
            {},
        ).get(
            "inference"
        )
    )

    if value is None:
        raise ValueError(
            "Evaluation did not report "
            "inference latency."
        )

    value = float(
        value
    )

    if value <= 0:
        raise ValueError(
            "Inference latency must be positive."
        )

    return value


def result_record(
    experiment: dict,
    evaluation: dict,
    run_directory: Path,
    best_checkpoint: Path,
) -> dict:
    """Build one comparable experiment result."""

    metrics = evaluation[
        "metrics"
    ]

    inference_ms = (
        inference_ms_from_report(
            evaluation
        )
    )

    return {
        "name": (
            experiment[
                "name"
            ]
        ),
        "optimizer": (
            experiment[
                "optimizer"
            ]
        ),
        "lr0": float(
            experiment[
                "lr0"
            ]
        ),
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
        "fps": (
            1000.0
            / inference_ms
        ),
        "run_directory": str(
            run_directory
        ),
        "best_checkpoint": str(
            best_checkpoint
        ),
    }


def select_candidate(
    results: list[dict],
    map_tolerance: float,
) -> dict:
    """Select the fastest candidate close to best validation mAP."""

    if not results:
        raise ValueError(
            "No tuning results exist."
        )

    best_map = max(
        result[
            "map50_95"
        ]
        for result in results
    )

    threshold = (
        best_map
        - map_tolerance
    )

    candidates = [
        result
        for result in results
        if result[
            "map50_95"
        ]
        >= threshold
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
        "acceptable_map50_95": (
            threshold
        ),
        "selected": (
            selected
        ),
    }


def write_results_csv(
    results: list[dict],
    output_path: Path,
) -> Path:
    """Write compact tuning comparison CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "name",
        "optimizer",
        "lr0",
        "precision",
        "recall",
        "f1",
        "map50",
        "map75",
        "map50_95",
        "inference_ms",
        "fps",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    return output_path


def run_hyperparameter_tuning(
    dataset_root: Path,
    baseline_config_path: Path,
    search_config_path: Path,
    resolution_report_path: Path,
    output_directory: Path,
    device: str | None = None,
) -> dict:
    """Train and compare controlled optimization candidates."""

    dataset_root = (
        dataset_root
        .expanduser()
        .resolve()
    )

    output_directory = (
        output_directory
        .expanduser()
        .resolve()
    )

    if not dataset_root.is_dir():
        raise ValueError(
            f"Dataset does not exist: "
            f"{dataset_root}"
        )

    baseline = (
        load_training_config(
            baseline_config_path
        )
    )

    search = (
        load_json(
            search_config_path
        )
    )

    errors = (
        validate_search_config(
            search
        )
    )

    if errors:
        raise ValueError(
            "Invalid search configuration: "
            + " | ".join(
                errors
            )
        )

    resolution_report = (
        load_json(
            resolution_report_path
        )
    )

    selected_resolution = (
        selected_resolution_from_report(
            resolution_report
        )
    )

    reference_baseline = (
        resolution_report.get(
            "selected_result"
        )
    )

    if not isinstance(
        reference_baseline,
        dict,
    ):
        raise ValueError(
            "Commit 11 selected baseline "
            "result is missing."
        )

    data_yaml = (
        dataset_root
        / "data.yaml"
    )

    if not data_yaml.is_file():
        raise ValueError(
            f"Dataset YAML does not exist: "
            f"{data_yaml}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_configs = (
        output_directory
        / "generated_configs"
    )

    generated_configs.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for experiment in search[
        "experiments"
    ]:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Hyperparameter experiment: "
            f"{experiment['name']}"
        )

        print(
            f"Resolution: "
            f"{selected_resolution}"
        )

        print(
            f"Optimizer: "
            f"{experiment['optimizer']}"
        )

        print(
            f"lr0: "
            f"{experiment['lr0']}"
        )

        print(
            "=" * 70
        )

        candidate = (
            build_candidate_config(
                baseline=baseline,
                search=search,
                experiment=experiment,
                resolution=(
                    selected_resolution
                ),
            )
        )

        candidate_config_path = (
            generated_configs
            / (
                experiment[
                    "name"
                ]
                + ".json"
            )
        )

        candidate_config_path.write_text(
            json.dumps(
                candidate,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        training = train(
            dataset_root=(
                dataset_root
            ),
            config_path=(
                candidate_config_path
            ),
            requested_device=(
                device
            ),
            smoke=False,
        )

        run_directory = Path(
            training[
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
                f"{experiment['name']}: "
                "best.pt was not produced."
            )

        metric_path = (
            output_directory
            / (
                experiment[
                    "name"
                ]
                + "_val_metrics.json"
            )
        )

        evaluation = (
            evaluate_detector(
                model_path=str(
                    best_checkpoint
                ),
                data_yaml=(
                    data_yaml
                ),
                split="val",
                imgsz=(
                    selected_resolution
                ),
                batch=int(
                    search[
                        "batch"
                    ]
                ),
                device=device,
                output_path=(
                    metric_path
                ),
            )
        )

        results.append(
            result_record(
                experiment=(
                    experiment
                ),
                evaluation=(
                    evaluation
                ),
                run_directory=(
                    run_directory
                ),
                best_checkpoint=(
                    best_checkpoint
                ),
            )
        )

    selection = (
        select_candidate(
            results,
            map_tolerance=float(
                search.get(
                    "map_tolerance",
                    0.005,
                )
            ),
        )
    )

    summary = {
        "status": "PASSED",
        "experiment": (
            "YOLOv8 hyperparameter screening"
        ),
        "model": "yolov8n.pt",
        "selected_resolution": (
            selected_resolution
        ),
        "search_epochs": (
            search[
                "search_epochs"
            ]
        ),
        "validation_only": True,
        "test_set_evaluated": False,
        "reference_baseline": (
            reference_baseline
        ),
        "results": (
            results
        ),
        "best_map50_95": (
            selection[
                "best_map50_95"
            ]
        ),
        "acceptable_map50_95": (
            selection[
                "acceptable_map50_95"
            ]
        ),
        "selected_experiment": (
            selection[
                "selected"
            ][
                "name"
            ]
        ),
        "selected_result": (
            selection[
                "selected"
            ]
        ),
    }

    summary_path = (
        output_directory
        / "hyperparameter_tuning_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = (
        output_directory
        / "hyperparameter_tuning_results.csv"
    )

    write_results_csv(
        results,
        csv_path,
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
        "--baseline-config",
        type=Path,
        default=Path(
            "configs/"
            "yolov8n_baseline.json"
        ),
    )

    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path(
            "configs/"
            "hyperparameter_tuning.json"
        ),
    )

    parser.add_argument(
        "--resolution-report",
        type=Path,
        default=Path(
            "runs/tuning/"
            "resolution/"
            "resolution_tuning_summary.json"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/tuning/"
            "hyperparameters"
        ),
    )

    parser.add_argument(
        "--device",
        default=None,
    )

    args = parser.parse_args()

    try:

        summary = (
            run_hyperparameter_tuning(
                dataset_root=(
                    args.dataset_root
                ),
                baseline_config_path=(
                    args.baseline_config
                ),
                search_config_path=(
                    args.search_config
                ),
                resolution_report_path=(
                    args.resolution_report
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
            f"Hyperparameter tuning failed: "
            f"{error}\n",
        )

    print(
        "\nHyperparameter tuning: PASSED"
    )

    print(
        "Resolution: "
        f"{summary['selected_resolution']}"
    )

    print(
        "\nExperiment results:"
    )

    for result in summary[
        "results"
    ]:

        print(
            (
                f"{result['name']}: "
                f"mAP50-95="
                f"{result['map50_95']:.4f}, "
                f"inference="
                f"{result['inference_ms']:.2f} ms, "
                f"FPS="
                f"{result['fps']:.2f}"
            )
        )

    print(
        "\nSelected experiment: "
        f"{summary['selected_experiment']}"
    )

    selected = (
        summary[
            "selected_result"
        ]
    )

    print(
        "Selected mAP50-95: "
        f"{selected['map50_95']:.4f}"
    )

    print(
        "Selected inference: "
        f"{selected['inference_ms']:.2f} ms"
    )

    print(
        "Test set evaluated: False"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )