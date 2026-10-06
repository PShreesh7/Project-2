"""Train YOLOv8 using a validated repository JSON configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.training_preflight import (
    detect_runtime,
    load_training_config,
    run_training_preflight,
    validate_training_config,
)


def select_device(
    requested_device: str | None = None,
) -> str:
    """Select an explicit training device."""

    if requested_device:
        return requested_device

    runtime = detect_runtime()

    if runtime[
        "cuda_available"
    ]:
        return "cuda:0"

    return "cpu"


def build_training_arguments(
    config: dict,
    data_yaml: Path,
    device: str,
) -> dict[str, Any]:
    """Translate repository config into Ultralytics train arguments."""

    augmentation = config.get(
        "training_augmentation",
        {},
    )

    arguments: dict[
        str,
        Any,
    ] = {
        "data": str(
            data_yaml
            .expanduser()
            .resolve()
        ),
        "epochs": int(
            config["epochs"]
        ),
        "imgsz": int(
            config["imgsz"]
        ),
        "batch": int(
            config["batch"]
        ),
        "patience": int(
            config.get(
                "patience",
                10,
            )
        ),
        "workers": int(
            config.get(
                "workers",
                0,
            )
        ),
        "seed": int(
            config["seed"]
        ),
        "deterministic": bool(
            config.get(
                "deterministic",
                True,
            )
        ),
        "optimizer": config.get(
            "optimizer",
            "auto",
        ),
        "cache": config.get(
            "cache",
            False,
        ),
        "project": config.get(
            "project",
            "runs/detect",
        ),
        "name": config.get(
            "name",
            "neu_yolov8_run",
        ),
        "device": device,
        "val": bool(
            config.get(
                "val",
                True,
            )
        ),
        "save": bool(
            config.get(
                "save",
                True,
            )
        ),
        "plots": bool(
            config.get(
                "plots",
                True,
            )
        ),
        "exist_ok": True,
    }

    if "fraction" in config:
        arguments[
            "fraction"
        ] = config[
            "fraction"
        ]

    # On CPU, mixed precision does not provide
    # the CUDA acceleration benefit we want.
    arguments[
        "amp"
    ] = (
        device != "cpu"
    )

    for key in (
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
    ):
        if key in augmentation:
            arguments[
                key
            ] = augmentation[
                key
            ]

    return arguments


def validate_smoke_configuration(
    config: dict,
) -> list[str]:
    """Apply extra safety rules to a smoke-training configuration."""

    errors = (
        validate_training_config(
            config
        )
    )

    if errors:
        return errors

    epochs = config.get(
        "epochs"
    )

    if epochs != 1:
        errors.append(
            "Smoke training must use exactly "
            "1 epoch."
        )

    fraction = config.get(
        "fraction"
    )

    if not isinstance(
        fraction,
        (
            int,
            float,
        ),
    ):
        errors.append(
            "Smoke configuration requires "
            "a numeric fraction."
        )

    elif not (
        0 < fraction < 1
    ):
        errors.append(
            "Smoke fraction must be "
            "greater than 0 and less than 1."
        )

    return errors


def expected_run_directory(
    config: dict,
) -> Path:
    """Return the deterministic Ultralytics output directory."""

    return (
        Path(
            config.get(
                "project",
                "runs/detect",
            )
        )
        / config.get(
            "name",
            "neu_yolov8_run",
        )
    ).resolve()


def verify_training_artifacts(
    run_directory: Path,
) -> dict:
    """Verify critical Ultralytics training artifacts."""

    run_directory = (
        run_directory
        .expanduser()
        .resolve()
    )

    weights_directory = (
        run_directory
        / "weights"
    )

    expected = {
        "run_directory": (
            run_directory
        ),
        "results_csv": (
            run_directory
            / "results.csv"
        ),
        "last_checkpoint": (
            weights_directory
            / "last.pt"
        ),
        "best_checkpoint": (
            weights_directory
            / "best.pt"
        ),
    }

    artifact_status = {
        key: (
            str(path)
            if key == "run_directory"
            else path.is_file()
        )
        for key, path
        in expected.items()
    }

    artifact_status[
        "run_directory_exists"
    ] = (
        run_directory.is_dir()
    )

    required_ok = (
        artifact_status[
            "run_directory_exists"
        ]
        and artifact_status[
            "results_csv"
        ]
        and artifact_status[
            "last_checkpoint"
        ]
    )

    artifact_status[
        "required_artifacts_present"
    ] = required_ok

    return artifact_status


def write_training_report(
    output_path: Path,
    config_path: Path,
    data_yaml: Path,
    device: str,
    artifact_status: dict,
) -> Path:
    """Save smoke-run completion evidence."""

    output_path = (
        output_path
        .expanduser()
        .resolve()
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime = detect_runtime()

    report = {
        "status": (
            "PASSED"
            if artifact_status[
                "required_artifacts_present"
            ]
            else "FAILED"
        ),
        "config": str(
            config_path.resolve()
        ),
        "data_yaml": str(
            data_yaml.resolve()
        ),
        "device": device,
        "runtime": runtime,
        "artifacts": (
            artifact_status
        ),
    }

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def train(
    dataset_root: Path,
    config_path: Path,
    requested_device: str | None = None,
    smoke: bool = False,
) -> dict:
    """Run a fully preflighted YOLO training experiment."""

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

    config = load_training_config(
        config_path
    )

    if smoke:
        errors = (
            validate_smoke_configuration(
                config
            )
        )
    else:
        errors = (
            validate_training_config(
                config
            )
        )

    if errors:
        raise ValueError(
            "Invalid training configuration: "
            + " | ".join(
                errors
            )
        )

    preflight = (
        run_training_preflight(
            dataset_root=(
                dataset_root
            ),
            config_path=(
                config_path
            ),
            check_runtime=True,
        )
    )

    if (
        preflight[
            "status"
        ]
        != "READY"
    ):
        raise ValueError(
            "Training preflight is NOT_READY: "
            + " | ".join(
                preflight[
                    "errors"
                ]
            )
        )

    data_yaml = Path(
        preflight[
            "data_yaml"
        ]
    )

    device = select_device(
        requested_device
    )

    train_args = (
        build_training_arguments(
            config=config,
            data_yaml=data_yaml,
            device=device,
        )
    )

    print(
        "Starting YOLOv8 training"
    )

    print(
        f"Model: {config['model']}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Epochs: {train_args['epochs']}"
    )

    print(
        f"Image size: "
        f"{train_args['imgsz']}"
    )

    print(
        f"Batch size: "
        f"{train_args['batch']}"
    )

    if "fraction" in train_args:
        print(
            "Training fraction: "
            f"{train_args['fraction']}"
        )

    print(
        f"Dataset: {data_yaml}"
    )

    # Keep the heavyweight dependency inside
    # the execution path so unit tests do not
    # need to instantiate a YOLO model.
    from ultralytics import YOLO

    model = YOLO(
        config[
            "model"
        ]
    )

    model.train(
        **train_args
    )

    run_directory = (
        expected_run_directory(
            config
        )
    )

    artifact_status = (
        verify_training_artifacts(
            run_directory
        )
    )

    report_path = (
        run_directory
        / "smoke_run_report.json"
        if smoke
        else run_directory
        / "training_run_report.json"
    )

    report_path = (
        write_training_report(
            output_path=(
                report_path
            ),
            config_path=(
                config_path
            ),
            data_yaml=(
                data_yaml
            ),
            device=device,
            artifact_status=(
                artifact_status
            ),
        )
    )

    if not artifact_status[
        "required_artifacts_present"
    ]:
        raise ValueError(
            "Training returned, but required "
            "artifacts were not produced."
        )

    return {
        "run_directory": str(
            run_directory
        ),
        "report": str(
            report_path
        ),
        "device": device,
        "artifacts": (
            artifact_status
        ),
    }


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
            "Optional explicit device, "
            "for example cpu or cuda:0."
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Require smoke-run safety "
            "settings such as one epoch "
            "and a reduced training fraction."
        ),
    )

    args = parser.parse_args()

    try:
        result = train(
            dataset_root=(
                args.dataset_root
            ),
            config_path=(
                args.config
            ),
            requested_device=(
                args.device
            ),
            smoke=args.smoke,
        )

    except (
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        parser.exit(
            1,
            f"Training failed: "
            f"{error}\n",
        )

    print(
        "\nYOLO training completed successfully"
    )

    print(
        "Run directory: "
        f"{result['run_directory']}"
    )

    print(
        "Run report: "
        f"{result['report']}"
    )

    print(
        "Checkpoint: "
        f"{result['artifacts']['last_checkpoint']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )