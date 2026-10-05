"""Tests for the final YOLO training preflight."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.training_preflight import (
    run_training_preflight,
    validate_training_config,
)


class TrainingPreflightTests(
    unittest.TestCase
):

    def setUp(self):

        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.addCleanup(
            self.temp_directory.cleanup
        )

        self.root = Path(
            self.temp_directory.name
        )

        self.dataset = (
            self.root
            / "dataset"
        )

        self.config_path = (
            self.root
            / "config.json"
        )

        for split in (
            "train",
            "val",
            "test",
        ):
            (
                self.dataset
                / "images"
                / split
            ).mkdir(
                parents=True
            )

            (
                self.dataset
                / "labels"
                / split
            ).mkdir(
                parents=True
            )

    def add_sample(
        self,
        split: str,
        stem: str,
        class_id: int,
    ) -> None:

        Image.new(
            "RGB",
            (200, 200),
            color=100,
        ).save(
            self.dataset
            / "images"
            / split
            / f"{stem}.png"
        )

        (
            self.dataset
            / "labels"
            / split
            / f"{stem}.txt"
        ).write_text(
            (
                f"{class_id} "
                "0.500000 "
                "0.500000 "
                "0.250000 "
                "0.250000\n"
            ),
            encoding="utf-8",
        )

    def create_dataset(
        self,
    ) -> None:

        # 12 training samples:
        # one original and one augmented
        # example for all six classes.
        for class_id in range(
            6
        ):
            self.add_sample(
                "train",
                f"train_{class_id}",
                class_id,
            )

            self.add_sample(
                "train",
                (
                    f"train_{class_id}"
                    "__aug"
                ),
                class_id,
            )

            self.add_sample(
                "val",
                f"val_{class_id}",
                class_id,
            )

            self.add_sample(
                "test",
                f"test_{class_id}",
                class_id,
            )

        quality_report = {
            "status": "PASSED",
            "errors": [],
            "warnings": [],
        }

        (
            self.dataset
            / "augmentation_quality_report.json"
        ).write_text(
            json.dumps(
                quality_report
            ),
            encoding="utf-8",
        )

    def write_config(
        self,
        train: int = 12,
        val: int = 6,
        test: int = 6,
    ) -> None:

        config = {
            "model": "yolov8n.pt",
            "task": "detect",
            "epochs": 30,
            "imgsz": 416,
            "batch": 4,
            "seed": 42,
            "expected_dataset": {
                "train": train,
                "val": val,
                "test": test,
                "nc": 6,
            },
        }

        self.config_path.write_text(
            json.dumps(
                config
            ),
            encoding="utf-8",
        )

    def test_valid_preflight_is_ready(
        self,
    ):

        self.create_dataset()
        self.write_config()

        report = (
            run_training_preflight(
                self.dataset,
                self.config_path,
                check_runtime=False,
            )
        )

        self.assertEqual(
            report["status"],
            "READY",
        )

        self.assertEqual(
            report["errors"],
            [],
        )

    def test_data_yaml_is_generated(
        self,
    ):

        self.create_dataset()
        self.write_config()

        report = (
            run_training_preflight(
                self.dataset,
                self.config_path,
                check_runtime=False,
            )
        )

        self.assertEqual(
            report["status"],
            "READY",
        )

        self.assertTrue(
            (
                self.dataset
                / "data.yaml"
            ).is_file()
        )

    def test_wrong_dataset_count_fails(
        self,
    ):

        self.create_dataset()

        self.write_config(
            train=2520,
            val=360,
            test=180,
        )

        report = (
            run_training_preflight(
                self.dataset,
                self.config_path,
                check_runtime=False,
            )
        )

        self.assertEqual(
            report["status"],
            "NOT_READY",
        )

        self.assertTrue(
            any(
                "count mismatch"
                in error.lower()
                for error
                in report["errors"]
            )
        )

    def test_failed_augmentation_report_fails(
        self,
    ):

        self.create_dataset()
        self.write_config()

        (
            self.dataset
            / "augmentation_quality_report.json"
        ).write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "errors": [
                        "quality problem"
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = (
            run_training_preflight(
                self.dataset,
                self.config_path,
                check_runtime=False,
            )
        )

        self.assertEqual(
            report["status"],
            "NOT_READY",
        )

    def test_non_yolov8_model_is_rejected(
        self,
    ):

        config = {
            "model": "other.pt",
            "task": "detect",
            "epochs": 30,
            "imgsz": 416,
            "batch": 4,
            "seed": 42,
            "expected_dataset": {
                "train": 12,
                "val": 6,
                "test": 6,
                "nc": 6,
            },
        }

        errors = (
            validate_training_config(
                config
            )
        )

        self.assertTrue(
            any(
                "yolov8"
                in error.lower()
                for error
                in errors
            )
        )

    def test_invalid_image_size_is_rejected(
        self,
    ):

        config = {
            "model": "yolov8n.pt",
            "task": "detect",
            "epochs": 30,
            "imgsz": 415,
            "batch": 4,
            "seed": 42,
            "expected_dataset": {
                "train": 12,
                "val": 6,
                "test": 6,
                "nc": 6,
            },
        }

        errors = (
            validate_training_config(
                config
            )
        )

        self.assertTrue(
            any(
                "divisible by 32"
                in error.lower()
                for error
                in errors
            )
        )


if __name__ == "__main__":
    unittest.main()