"""Tests for YOLO dataset preflight validation."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_yolo_dataset import (
    create_ultralytics_yaml,
    run_preflight,
)


class YoloPreflightTests(
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

        for split in (
            "train",
            "val",
            "test",
        ):

            (
                self.root
                / "images"
                / split
            ).mkdir(
                parents=True
            )

            (
                self.root
                / "labels"
                / split
            ).mkdir(
                parents=True
            )

    def add_sample(
        self,
        split: str,
        stem: str,
        class_id: int = 0,
        label_line: str | None = None,
    ) -> None:

        Image.new(
            "RGB",
            (200, 200),
            color=100,
        ).save(
            self.root
            / "images"
            / split
            / f"{stem}.jpg"
        )

        if label_line is None:
            label_line = (
                f"{class_id} "
                "0.500000 "
                "0.500000 "
                "0.250000 "
                "0.250000"
            )

        (
            self.root
            / "labels"
            / split
            / f"{stem}.txt"
        ).write_text(
            label_line + "\n",
            encoding="utf-8",
        )

    def create_valid_dataset(
        self,
    ) -> None:

        # Ensure every class occurs
        # in the training split.
        for class_id in range(
            6
        ):
            self.add_sample(
                "train",
                f"train_{class_id}",
                class_id,
            )

        for class_id in range(
            6
        ):
            self.add_sample(
                "val",
                f"val_{class_id}",
                class_id,
            )

        for class_id in range(
            6
        ):
            self.add_sample(
                "test",
                f"test_{class_id}",
                class_id,
            )

    def test_valid_dataset_passes(
        self,
    ):

        self.create_valid_dataset()

        report = run_preflight(
            self.root,
            expected_total=18,
        )

        self.assertEqual(
            report["status"],
            "PASSED",
        )

        self.assertEqual(
            report["total_samples"],
            18,
        )

        self.assertEqual(
            report["errors"],
            [],
        )

    def test_missing_label_fails(
        self,
    ):

        self.create_valid_dataset()

        (
            self.root
            / "labels"
            / "train"
            / "train_0.txt"
        ).unlink()

        report = run_preflight(
            self.root
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

        self.assertTrue(
            any(
                "images without labels"
                in error.lower()
                for error
                in report["errors"]
            )
        )

    def test_invalid_class_id_fails(
        self,
    ):

        self.create_valid_dataset()

        (
            self.root
            / "labels"
            / "train"
            / "train_0.txt"
        ).write_text(
            (
                "99 "
                "0.5 "
                "0.5 "
                "0.2 "
                "0.2\n"
            ),
            encoding="utf-8",
        )

        report = run_preflight(
            self.root
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

    def test_out_of_bounds_box_fails(
        self,
    ):

        self.create_valid_dataset()

        (
            self.root
            / "labels"
            / "train"
            / "train_0.txt"
        ).write_text(
            (
                "0 "
                "0.950000 "
                "0.500000 "
                "0.300000 "
                "0.200000\n"
            ),
            encoding="utf-8",
        )

        report = run_preflight(
            self.root
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

    def test_expected_total_mismatch_fails(
        self,
    ):

        self.create_valid_dataset()

        report = run_preflight(
            self.root,
            expected_total=1800,
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

        self.assertTrue(
            any(
                "unexpected dataset size"
                in error.lower()
                for error
                in report["errors"]
            )
        )

    def test_split_leakage_is_detected(
        self,
    ):

        self.create_valid_dataset()

        self.add_sample(
            "train",
            "duplicate_sample",
            0,
        )

        self.add_sample(
            "val",
            "duplicate_sample",
            0,
        )

        report = run_preflight(
            self.root
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

        self.assertTrue(
            any(
                "data leakage"
                in error.lower()
                for error
                in report["errors"]
            )
        )

    def test_yaml_is_generated(
        self,
    ):

        self.create_valid_dataset()

        report = run_preflight(
            self.root
        )

        self.assertEqual(
            report["status"],
            "PASSED",
        )

        yaml_path = (
            create_ultralytics_yaml(
                self.root
            )
        )

        content = (
            yaml_path.read_text(
                encoding="utf-8"
            )
        )

        self.assertIn(
            "train: images/train",
            content,
        )

        self.assertIn(
            "val: images/val",
            content,
        )

        self.assertIn(
            "test: images/test",
            content,
        )

        self.assertIn(
            '0: "crazing"',
            content,
        )

        self.assertIn(
            '5: "scratches"',
            content,
        )


if __name__ == "__main__":
    unittest.main()