"""Tests for controlled YOLO hyperparameter tuning."""

import unittest

from scripts.tune_hyperparameters import (
    build_candidate_config,
    select_candidate,
    selected_resolution_from_report,
    validate_search_config,
)


class HyperparameterTuningTests(
    unittest.TestCase
):

    def baseline(
        self,
    ) -> dict:

        return {
            "model": "yolov8n.pt",
            "task": "detect",
            "epochs": 30,
            "imgsz": 416,
            "batch": 4,
            "seed": 42,
            "expected_dataset": {
                "train": 2520,
                "val": 360,
                "test": 180,
                "nc": 6,
            },
        }

    def search(
        self,
    ) -> dict:

        return {
            "search_epochs": 12,
            "batch": 4,
            "workers": 0,
            "patience": 4,
            "seed": 42,
            "map_tolerance": 0.005,
            "experiments": [
                {
                    "name": "adamw",
                    "optimizer": "AdamW",
                    "lr0": 0.001,
                    "lrf": 0.01,
                    "weight_decay": 0.0005,
                }
            ],
        }

    def test_selected_resolution_is_loaded(
        self,
    ):

        report = {
            "status": "PASSED",
            "selected_resolution": 416,
            "test_set_evaluated": False,
        }

        self.assertEqual(
            selected_resolution_from_report(
                report
            ),
            416,
        )

    def test_failed_resolution_report_is_rejected(
        self,
    ):

        report = {
            "status": "FAILED",
            "selected_resolution": 416,
            "test_set_evaluated": False,
        }

        with self.assertRaisesRegex(
            ValueError,
            "did not pass",
        ):
            selected_resolution_from_report(
                report
            )

    def test_candidate_uses_selected_resolution(
        self,
    ):

        experiment = (
            self.search()[
                "experiments"
            ][0]
        )

        candidate = (
            build_candidate_config(
                baseline=(
                    self.baseline()
                ),
                search=(
                    self.search()
                ),
                experiment=(
                    experiment
                ),
                resolution=320,
            )
        )

        self.assertEqual(
            candidate[
                "imgsz"
            ],
            320,
        )

        self.assertEqual(
            candidate[
                "optimizer"
            ],
            "AdamW",
        )

        self.assertEqual(
            candidate[
                "lr0"
            ],
            0.001,
        )

    def test_candidate_uses_search_epochs(
        self,
    ):

        experiment = (
            self.search()[
                "experiments"
            ][0]
        )

        candidate = (
            build_candidate_config(
                self.baseline(),
                self.search(),
                experiment,
                416,
            )
        )

        self.assertEqual(
            candidate[
                "epochs"
            ],
            12,
        )

    def test_balanced_selection(
        self,
    ):

        results = [
            {
                "name": "a",
                "map50_95": 0.620,
                "inference_ms": 20.0,
            },
            {
                "name": "b",
                "map50_95": 0.623,
                "inference_ms": 25.0,
            },
            {
                "name": "c",
                "map50_95": 0.600,
                "inference_ms": 15.0,
            },
        ]

        selection = (
            select_candidate(
                results,
                map_tolerance=0.005,
            )
        )

        self.assertEqual(
            selection[
                "selected"
            ][
                "name"
            ],
            "a",
        )

    def test_invalid_learning_rate_fails(
        self,
    ):

        config = (
            self.search()
        )

        config[
            "experiments"
        ][0][
            "lr0"
        ] = 0

        errors = (
            validate_search_config(
                config
            )
        )

        self.assertTrue(
            any(
                "lr0"
                in error
                for error in errors
            )
        )


if __name__ == "__main__":
    unittest.main()