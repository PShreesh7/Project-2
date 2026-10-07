"""Tests for baseline experiment orchestration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_baseline_experiment import (
    run_baseline_experiment,
)


class BaselineExperimentTests(
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

        self.dataset.mkdir()

        (
            self.dataset
            / "data.yaml"
        ).write_text(
            (
                "path: dataset\n"
                "train: images/train\n"
                "val: images/val\n"
                "test: images/test\n"
            ),
            encoding="utf-8",
        )

        self.config = (
            self.root
            / "config.json"
        )

        self.config.write_text(
            "{}",
            encoding="utf-8",
        )

        self.run_directory = (
            self.root
            / "run"
        )

        (
            self.run_directory
            / "weights"
        ).mkdir(
            parents=True
        )

        (
            self.run_directory
            / "weights"
            / "best.pt"
        ).write_bytes(
            b"checkpoint"
        )

    @patch(
        "scripts.run_baseline_experiment."
        "evaluate_detector"
    )
    @patch(
        "scripts.run_baseline_experiment."
        "train"
    )
    def test_baseline_uses_best_checkpoint(
        self,
        mock_train,
        mock_evaluate,
    ):

        mock_train.return_value = {
            "run_directory": str(
                self.run_directory
            ),
        }

        mock_evaluate.return_value = {
            "metrics": {
                "precision": 0.8,
                "recall": 0.7,
                "f1_derived": 0.7467,
                "map50": 0.85,
                "map75": 0.65,
                "map50_95": 0.60,
            },
            "per_class": {},
            "speed_ms_per_image": {},
        }

        result = (
            run_baseline_experiment(
                self.dataset,
                self.config,
                device="cpu",
            )
        )

        self.assertEqual(
            result[
                "status"
            ],
            "PASSED",
        )

        self.assertIn(
            "best.pt",
            result[
                "best_checkpoint"
            ],
        )

    @patch(
        "scripts.run_baseline_experiment."
        "evaluate_detector"
    )
    @patch(
        "scripts.run_baseline_experiment."
        "train"
    )
    def test_only_validation_is_evaluated(
        self,
        mock_train,
        mock_evaluate,
    ):

        mock_train.return_value = {
            "run_directory": str(
                self.run_directory
            ),
        }

        mock_evaluate.return_value = {
            "metrics": {
                "precision": 0.8,
                "recall": 0.7,
                "f1_derived": 0.7467,
                "map50": 0.85,
                "map75": 0.65,
                "map50_95": 0.60,
            },
            "per_class": {},
            "speed_ms_per_image": {},
        }

        result = (
            run_baseline_experiment(
                self.dataset,
                self.config,
                device="cpu",
            )
        )

        self.assertFalse(
            result[
                "test_set_evaluated"
            ]
        )

        self.assertEqual(
            mock_evaluate.call_args.kwargs[
                "split"
            ],
            "val",
        )

    @patch(
        "scripts.run_baseline_experiment."
        "train"
    )
    def test_missing_best_checkpoint_fails(
        self,
        mock_train,
    ):

        empty_run = (
            self.root
            / "empty_run"
        )

        empty_run.mkdir()

        mock_train.return_value = {
            "run_directory": str(
                empty_run
            ),
        }

        with self.assertRaisesRegex(
            ValueError,
            "best.pt",
        ):
            run_baseline_experiment(
                self.dataset,
                self.config,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()