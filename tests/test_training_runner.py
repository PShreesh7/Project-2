"""Tests for the YOLO training runner."""

import tempfile
import unittest
from pathlib import Path

from scripts.train_yolov8 import (
    build_training_arguments,
    validate_smoke_configuration,
    verify_training_artifacts,
)


class TrainingRunnerTests(
    unittest.TestCase
):

    def valid_config(
        self,
    ) -> dict:

        return {
            "model": "yolov8n.pt",
            "task": "detect",
            "epochs": 1,
            "imgsz": 320,
            "batch": 2,
            "workers": 0,
            "seed": 42,
            "fraction": 0.05,
            "project": "runs/detect",
            "name": "neu_yolov8n_smoke",
            "expected_dataset": {
                "train": 2520,
                "val": 360,
                "test": 180,
                "nc": 6,
            },
            "training_augmentation": {
                "degrees": 0.0,
                "mosaic": 0.0,
                "mixup": 0.0,
            },
        }

    def test_valid_smoke_config(
        self,
    ):

        errors = (
            validate_smoke_configuration(
                self.valid_config()
            )
        )

        self.assertEqual(
            errors,
            [],
        )

    def test_smoke_requires_one_epoch(
        self,
    ):

        config = (
            self.valid_config()
        )

        config[
            "epochs"
        ] = 2

        errors = (
            validate_smoke_configuration(
                config
            )
        )

        self.assertTrue(
            any(
                "exactly 1 epoch"
                in error
                for error in errors
            )
        )

    def test_smoke_requires_reduced_fraction(
        self,
    ):

        config = (
            self.valid_config()
        )

        config[
            "fraction"
        ] = 1.0

        errors = (
            validate_smoke_configuration(
                config
            )
        )

        self.assertTrue(
            any(
                "less than 1"
                in error
                for error in errors
            )
        )

    def test_cpu_disables_amp(
        self,
    ):

        config = (
            self.valid_config()
        )

        args = (
            build_training_arguments(
                config,
                Path(
                    "data.yaml"
                ),
                device="cpu",
            )
        )

        self.assertFalse(
            args[
                "amp"
            ]
        )

        self.assertEqual(
            args[
                "device"
            ],
            "cpu",
        )

    def test_cuda_enables_amp(
        self,
    ):

        config = (
            self.valid_config()
        )

        args = (
            build_training_arguments(
                config,
                Path(
                    "data.yaml"
                ),
                device="cuda:0",
            )
        )

        self.assertTrue(
            args[
                "amp"
            ]
        )

    def test_fraction_is_forwarded(
        self,
    ):

        config = (
            self.valid_config()
        )

        args = (
            build_training_arguments(
                config,
                Path(
                    "data.yaml"
                ),
                device="cpu",
            )
        )

        self.assertEqual(
            args[
                "fraction"
            ],
            0.05,
        )

    def test_artifacts_are_verified(
        self,
    ):

        with (
            tempfile.TemporaryDirectory()
        ) as temporary_directory:

            run_directory = Path(
                temporary_directory
            )

            (
                run_directory
                / "weights"
            ).mkdir()

            (
                run_directory
                / "results.csv"
            ).write_text(
                "epoch\n0\n",
                encoding="utf-8",
            )

            (
                run_directory
                / "weights"
                / "last.pt"
            ).write_bytes(
                b"test"
            )

            status = (
                verify_training_artifacts(
                    run_directory
                )
            )

            self.assertTrue(
                status[
                    "required_artifacts_present"
                ]
            )

    def test_missing_checkpoint_fails_artifact_check(
        self,
    ):

        with (
            tempfile.TemporaryDirectory()
        ) as temporary_directory:

            run_directory = Path(
                temporary_directory
            )

            run_directory.mkdir(
                exist_ok=True
            )

            (
                run_directory
                / "results.csv"
            ).write_text(
                "epoch\n0\n",
                encoding="utf-8",
            )

            status = (
                verify_training_artifacts(
                    run_directory
                )
            )

            self.assertFalse(
                status[
                    "required_artifacts_present"
                ]
            )


if __name__ == "__main__":
    unittest.main()