"""Tests for validation resolution tuning."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.tune_resolution import (
    add_runtime_metrics,
    load_tuning_config,
    select_balanced_resolution,
    write_csv,
)


class ResolutionTuningTests(
    unittest.TestCase
):

    def test_balanced_selection_prefers_fast_candidate(
        self,
    ):

        results = [
            {
                "imgsz": 320,
                "map50_95": 0.600,
                "inference_ms": 10.0,
            },
            {
                "imgsz": 416,
                "map50_95": 0.620,
                "inference_ms": 18.0,
            },
            {
                "imgsz": 512,
                "map50_95": 0.625,
                "inference_ms": 30.0,
            },
        ]

        selection = (
            select_balanced_resolution(
                results,
                map_tolerance=0.01,
            )
        )

        self.assertEqual(
            selection[
                "selected"
            ][
                "imgsz"
            ],
            416,
        )

    def test_best_accuracy_selected_when_needed(
        self,
    ):

        results = [
            {
                "imgsz": 320,
                "map50_95": 0.55,
                "inference_ms": 10.0,
            },
            {
                "imgsz": 416,
                "map50_95": 0.60,
                "inference_ms": 20.0,
            },
        ]

        selection = (
            select_balanced_resolution(
                results,
                map_tolerance=0.01,
            )
        )

        self.assertEqual(
            selection[
                "selected"
            ][
                "imgsz"
            ],
            416,
        )

    def test_runtime_metrics_include_fps(
        self,
    ):

        report = {
            "metrics": {
                "precision": 0.80,
                "recall": 0.70,
                "f1_derived": 0.7467,
                "map50": 0.85,
                "map75": 0.65,
                "map50_95": 0.60,
            },
            "speed_ms_per_image": {
                "inference": 20.0,
            },
        }

        result = (
            add_runtime_metrics(
                report,
                resolution=416,
            )
        )

        self.assertAlmostEqual(
            result[
                "fps"
            ],
            50.0,
        )

    def test_invalid_resolution_rejected(
        self,
    ):

        with (
            tempfile.TemporaryDirectory()
        ) as directory:

            path = (
                Path(
                    directory
                )
                / "config.json"
            )

            path.write_text(
                json.dumps(
                    {
                        "resolutions": [
                            415
                        ],
                        "batch": 4,
                        "map_tolerance": 0.01,
                        "split": "val",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "divisible by 32",
            ):
                load_tuning_config(
                    path
                )

    def test_test_split_is_rejected(
        self,
    ):

        with (
            tempfile.TemporaryDirectory()
        ) as directory:

            path = (
                Path(
                    directory
                )
                / "config.json"
            )

            path.write_text(
                json.dumps(
                    {
                        "resolutions": [
                            320,
                            416,
                        ],
                        "batch": 4,
                        "map_tolerance": 0.01,
                        "split": "test",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "validation split",
            ):
                load_tuning_config(
                    path
                )

    def test_csv_is_written(
        self,
    ):

        with (
            tempfile.TemporaryDirectory()
        ) as directory:

            output = (
                Path(
                    directory
                )
                / "results.csv"
            )

            results = [
                {
                    "imgsz": 416,
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.7467,
                    "map50": 0.85,
                    "map75": 0.65,
                    "map50_95": 0.60,
                    "inference_ms": 20.0,
                    "fps": 50.0,
                }
            ]

            write_csv(
                results,
                output,
            )

            self.assertTrue(
                output.is_file()
            )


if __name__ == "__main__":
    unittest.main()