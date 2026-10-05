"""Run final checks before YOLOv8 model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_dataset import CLASSES
from scripts.prepare_yolo_dataset import (
    create_ultralytics_yaml,
    run_preflight,
)


REQUIRED_CONFIG_KEYS = {
    "model",
    "task",
    "epochs",
    "imgsz",
    "batch",
    "seed",
    "expected_dataset",
}


def load_training_config(
    config_path: Path,
) -> dict:
    """Read the repository training configuration."""

    config_path = (
        config_path
        .expanduser()
        .resolve()
    )

    if not config_path.is_file():
        raise ValueError(
            f"Training configuration does not exist: "
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
            f"Invalid JSON training configuration: "
            f"{error}"
        ) from error

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "Training configuration must be "
            "a JSON object."
        )

    return config


def validate_training_config(
    config: dict,
) -> list[str]:
    """Return configuration validation errors."""

    errors: list[str] = []

    missing = sorted(
        REQUIRED_CONFIG_KEYS
        - set(
            config
        )
    )

    if missing:
        errors.append(
            "Missing training configuration keys: "
            + ", ".join(
                missing
            )
        )

        return errors

    model = config.get(
        "model"
    )

    if (
        not isinstance(
            model,
            str,
        )
        or not model.startswith(
            "yolov8"
        )
        or not model.endswith(
            ".pt"
        )
    ):
        errors.append(
            "Baseline model must be a YOLOv8 "
            "PyTorch checkpoint."
        )

    if (
        config.get(
            "task"
        )
        != "detect"
    ):
        errors.append(
            "Task must be 'detect'."
        )

    for key in (
        "epochs",
        "imgsz",
        "batch",
    ):
        value = config.get(
            key
        )

        if (
            not isinstance(
                value,
                int,
            )
            or isinstance(
                value,
                bool,
            )
            or value <= 0
        ):
            errors.append(
                f"{key} must be a positive integer."
            )

    imgsz = config.get(
        "imgsz"
    )

    if (
        isinstance(
            imgsz,
            int,
        )
        and imgsz > 0
        and imgsz % 32 != 0
    ):
        errors.append(
            "imgsz must be divisible by 32 "
            "for this project's baseline."
        )

    seed = config.get(
        "seed"
    )

    if (
        not isinstance(
            seed,
            int,
        )
        or isinstance(
            seed,
            bool,
        )
    ):
        errors.append(
            "seed must be an integer."
        )

    expected = config.get(
        "expected_dataset"
    )

    if not isinstance(
        expected,
        dict,
    ):
        errors.append(
            "expected_dataset must be an object."
        )

        return errors

    for key in (
        "train",
        "val",
        "test",
        "nc",
    ):
        value = expected.get(
            key
        )

        if (
            not isinstance(
                value,
                int,
            )
            or isinstance(
                value,
                bool,
            )
            or value <= 0
        ):
            errors.append(
                "expected_dataset."
                f"{key} must be "
                "a positive integer."
            )

    if (
        expected.get(
            "nc"
        )
        != len(
            CLASSES
        )
    ):
        errors.append(
            "Expected class count does not match "
            f"project classes ({len(CLASSES)})."
        )

    return errors


def read_augmentation_report(
    dataset_root: Path,
) -> dict:
    """Load Commit 7 augmentation quality evidence."""

    report_path = (
        dataset_root
        / "augmentation_quality_report.json"
    )

    if not report_path.is_file():
        raise ValueError(
            "augmentation_quality_report.json "
            "is missing. Run Commit 7 first."
        )

    try:
        report = json.loads(
            report_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "Invalid augmentation quality report."
        ) from error

    return report


def detect_runtime() -> dict:
    """Inspect the installed ML runtime."""

    try:
        import torch

    except ImportError as error:
        raise ValueError(
            "PyTorch is not installed."
        ) from error

    try:
        import ultralytics

    except ImportError as error:
        raise ValueError(
            "Ultralytics is not installed. "
            "Install requirements-ml.txt."
        ) from error

    cuda_available = (
        torch.cuda.is_available()
    )

    runtime = {
        "torch_version": (
            torch.__version__
        ),
        "ultralytics_version": (
            ultralytics.__version__
        ),
        "cuda_available": (
            cuda_available
        ),
        "selected_device": (
            "cuda:0"
            if cuda_available
            else "cpu"
        ),
    }

    if cuda_available:
        runtime[
            "cuda_device_name"
        ] = torch.cuda.get_device_name(
            0
        )

        runtime[
            "cuda_device_count"
        ] = torch.cuda.device_count()

    return runtime


def check_dataset_counts(
    preflight: dict,
    expected: dict,
) -> list[str]:
    """Compare actual dataset sizes with the training plan."""

    errors: list[str] = []

    splits = preflight.get(
        "splits",
        {},
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        actual = (
            splits.get(
                split,
                {},
            ).get(
                "image_count"
            )
        )

        target = expected.get(
            split
        )

        if actual != target:
            errors.append(
                f"{split} image count mismatch: "
                f"expected {target}, "
                f"found {actual}."
            )

    return errors


def run_training_preflight(
    dataset_root: Path,
    config_path: Path,
    check_runtime: bool = True,
) -> dict:
    """Run all checks required before training."""

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

    report = {
        "status": "READY",
        "dataset_root": str(
            dataset_root
        ),
        "config_path": str(
            config_path
        ),
        "errors": [],
        "warnings": [],
    }

    if not dataset_root.is_dir():
        report[
            "errors"
        ].append(
            f"Dataset does not exist: "
            f"{dataset_root}"
        )

        report[
            "status"
        ] = "NOT_READY"

        return report

    try:
        config = load_training_config(
            config_path
        )

        report[
            "config"
        ] = config

        report[
            "errors"
        ].extend(
            validate_training_config(
                config
            )
        )

    except ValueError as error:
        report[
            "errors"
        ].append(
            str(
                error
            )
        )

        report[
            "status"
        ] = "NOT_READY"

        return report

    dataset_preflight = (
        run_preflight(
            dataset_root
        )
    )

    report[
        "dataset_preflight_status"
    ] = dataset_preflight.get(
        "status"
    )

    report[
        "dataset"
    ] = {
        "total_samples": (
            dataset_preflight.get(
                "total_samples"
            )
        ),
        "total_objects": (
            dataset_preflight.get(
                "total_objects"
            )
        ),
        "splits": (
            dataset_preflight.get(
                "splits",
                {},
            )
        ),
    }

    if (
        dataset_preflight.get(
            "status"
        )
        != "PASSED"
    ):
        report[
            "errors"
        ].append(
            "Augmented YOLO dataset failed "
            "structural preflight."
        )

        report[
            "errors"
        ].extend(
            dataset_preflight.get(
                "errors",
                [],
            )
        )

    report[
        "errors"
    ].extend(
        check_dataset_counts(
            dataset_preflight,
            config[
                "expected_dataset"
            ],
        )
    )

    try:
        augmentation_report = (
            read_augmentation_report(
                dataset_root
            )
        )

        report[
            "augmentation_quality_status"
        ] = augmentation_report.get(
            "status"
        )

        if (
            augmentation_report.get(
                "status"
            )
            != "PASSED"
        ):
            report[
                "errors"
            ].append(
                "Commit 7 augmentation quality "
                "report is not PASSED."
            )

        report[
            "augmentation_warnings"
        ] = augmentation_report.get(
            "warnings",
            [],
        )

    except ValueError as error:
        report[
            "errors"
        ].append(
            str(
                error
            )
        )

    try:
        data_yaml = (
            create_ultralytics_yaml(
                dataset_root
            )
        )

        report[
            "data_yaml"
        ] = str(
            data_yaml
        )

    except (
        OSError,
        ValueError,
    ) as error:
        report[
            "errors"
        ].append(
            "Unable to generate Ultralytics "
            f"data.yaml: {error}"
        )

    if check_runtime:
        try:
            report[
                "runtime"
            ] = detect_runtime()

        except ValueError as error:
            report[
                "errors"
            ].append(
                str(
                    error
                )
            )

    if report[
        "errors"
    ]:
        report[
            "status"
        ] = "NOT_READY"

    return report


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

    args = parser.parse_args()

    report = run_training_preflight(
        dataset_root=(
            args.dataset_root
        ),
        config_path=(
            args.config
        ),
        check_runtime=True,
    )

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    if dataset_root.is_dir():
        report_path = (
            dataset_root
            / "training_preflight_report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    else:
        report_path = None

    print(
        "YOLO training preflight: "
        f"{report['status']}"
    )

    dataset = report.get(
        "dataset",
        {}
    )

    splits = dataset.get(
        "splits",
        {},
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        details = splits.get(
            split
        )

        if details:
            print(
                f"{split}: "
                f"{details['image_count']} images, "
                f"{details['object_count']} objects"
            )

    runtime = report.get(
        "runtime"
    )

    if runtime:
        print(
            "Ultralytics: "
            f"{runtime['ultralytics_version']}"
        )

        print(
            "PyTorch: "
            f"{runtime['torch_version']}"
        )

        print(
            "CUDA available: "
            f"{runtime['cuda_available']}"
        )

        print(
            "Selected device: "
            f"{runtime['selected_device']}"
        )

        if runtime.get(
            "cuda_device_name"
        ):
            print(
                "CUDA device: "
                f"{runtime['cuda_device_name']}"
            )

    if report.get(
        "data_yaml"
    ):
        print(
            "Dataset YAML: "
            f"{report['data_yaml']}"
        )

    if report[
        "warnings"
    ]:
        print(
            "\nWarnings:"
        )

        for warning in report[
            "warnings"
        ]:
            print(
                f"- {warning}"
            )

    if report[
        "errors"
    ]:
        print(
            "\nErrors:"
        )

        for error in report[
            "errors"
        ]:
            print(
                f"- {error}"
            )

    if report_path:
        print(
            "\nReport: "
            f"{report_path}"
        )

    return (
        0
        if report[
            "status"
        ]
        == "READY"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )