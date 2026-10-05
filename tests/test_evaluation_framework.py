"""Tests for YOLO evaluation metric serialization."""

import unittest

from scripts.evaluate_detector import (
    build_metric_report,
    normalize_class_names,
)


class FakeBoxMetrics:

    mp = 0.80
    mr = 0.70
    map50 = 0.85
    map75 = 0.65
    map = 0.60

    maps = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
    ]


class FakeMetrics:

    box = FakeBoxMetrics()

    speed = {
        "preprocess": 1.0,
        "inference": 10.0,
        "loss": 0.0,
        "postprocess": 2.0,
    }


class EvaluationFrameworkTests(
    unittest.TestCase
):

    def test_class_name_dictionary(
        self,
    ):

        names = (
            normalize_class_names(
                {
                    0: "crazing",
                    1: "inclusion",
                }
            )
        )

        self.assertEqual(
            names[0],
            "crazing",
        )

    def test_class_name_list(
        self,
    ):

        names = (
            normalize_class_names(
                [
                    "crazing",
                    "inclusion",
                ]
            )
        )

        self.assertEqual(
            names[1],
            "inclusion",
        )

    def test_metric_report_contains_map(
        self,
    ):

        names = {
            0: "crazing",
            1: "inclusion",
            2: "patches",
            3: "pitted_surface",
            4: "rolled-in_scale",
            5: "scratches",
        }

        report = (
            build_metric_report(
                metrics=FakeMetrics(),
                class_names=names,
                model="best.pt",
                data="data.yaml",
                split="val",
            )
        )

        self.assertAlmostEqual(
            report[
                "metrics"
            ][
                "map50"
            ],
            0.85,
        )

        self.assertAlmostEqual(
            report[
                "metrics"
            ][
                "map50_95"
            ],
            0.60,
        )

        self.assertAlmostEqual(
            report[
                "metrics"
            ][
                "precision"
            ],
            0.80,
        )

        self.assertAlmostEqual(
            report[
                "metrics"
            ][
                "recall"
            ],
            0.70,
        )

    def test_per_class_map_is_saved(
        self,
    ):

        names = [
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled-in_scale",
            "scratches",
        ]

        report = (
            build_metric_report(
                metrics=FakeMetrics(),
                class_names=names,
                model="best.pt",
                data="data.yaml",
                split="val",
            )
        )

        self.assertAlmostEqual(
            report[
                "per_class"
            ][
                "scratches"
            ][
                "map50_95"
            ],
            0.75,
        )

    def test_speed_is_serialized(
        self,
    ):

        report = (
            build_metric_report(
                metrics=FakeMetrics(),
                class_names=[],
                model="best.pt",
                data="data.yaml",
                split="val",
            )
        )

        self.assertEqual(
            report[
                "speed_ms_per_image"
            ][
                "inference"
            ],
            10.0,
        )


if __name__ == "__main__":
    unittest.main()