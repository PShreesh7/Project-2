"""Tests for augmented-dataset quality validation."""

import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.validate_augmentation_quality import (
    create_preview,
    validate_augmentation_quality,
)


class AugmentationQualityTests(
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

        self.source = (
            self.root
            / "source"
        )

        self.augmented = (
            self.root
            / "augmented"
        )

        for split in (
            "train",
            "val",
            "test",
        ):
            (
                self.source
                / "images"
                / split
            ).mkdir(
                parents=True
            )

            (
                self.source
                / "labels"
                / split
            ).mkdir(
                parents=True
            )

    def add_sample(
        self,
        root: Path,
        split: str,
        stem: str,
        class_id: int,
    ) -> None:

        Image.new(
            "RGB",
            (200, 200),
            color=(
                80 + class_id,
                80 + class_id,
                80 + class_id,
            ),
        ).save(
            root
            / "images"
            / split
            / f"{stem}.png"
        )

        (
            root
            / "labels"
            / split
            / f"{stem}.txt"
        ).write_text(
            (
                f"{class_id} "
                "0.500000 "
                "0.500000 "
                "0.300000 "
                "0.300000\n"
            ),
            encoding="utf-8",
        )

    def create_dataset(
        self,
    ) -> None:

        for class_id in range(
            6
        ):
            for split in (
                "train",
                "val",
                "test",
            ):
                self.add_sample(
                    self.source,
                    split,
                    (
                        f"{split}_"
                        f"{class_id}"
                    ),
                    class_id,
                )

        shutil.copytree(
            self.source,
            self.augmented,
        )

        for class_id in range(
            6
        ):
            source_image = (
                self.source
                / "images"
                / "train"
                / f"train_{class_id}.png"
            )

            augmented_image = (
                self.augmented
                / "images"
                / "train"
                / (
                    f"train_{class_id}"
                    "__aug.png"
                )
            )

            shutil.copy2(
                source_image,
                augmented_image,
            )

            source_label = (
                self.source
                / "labels"
                / "train"
                / f"train_{class_id}.txt"
            )

            augmented_label = (
                self.augmented
                / "labels"
                / "train"
                / (
                    f"train_{class_id}"
                    "__aug.txt"
                )
            )

            shutil.copy2(
                source_label,
                augmented_label,
            )

    def test_valid_augmented_dataset_passes(
        self,
    ):

        self.create_dataset()

        report = (
            validate_augmentation_quality(
                self.source,
                self.augmented,
            )
        )

        self.assertEqual(
            report["status"],
            "PASSED",
        )

        self.assertEqual(
            report[
                "bounding_boxes"
            ][
                "duplicate_augmented_boxes"
            ],
            0,
        )

    def test_duplicate_box_fails(
        self,
    ):

        self.create_dataset()

        label_path = (
            self.augmented
            / "labels"
            / "train"
            / "train_0__aug.txt"
        )

        line = (
            label_path.read_text(
                encoding="utf-8"
            ).strip()
        )

        label_path.write_text(
            line
            + "\n"
            + line
            + "\n",
            encoding="utf-8",
        )

        report = (
            validate_augmentation_quality(
                self.source,
                self.augmented,
            )
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

        self.assertGreater(
            report[
                "bounding_boxes"
            ][
                "duplicate_augmented_boxes"
            ],
            0,
        )

    def test_validation_modification_fails(
        self,
    ):

        self.create_dataset()

        Image.new(
            "RGB",
            (200, 200),
            color=255,
        ).save(
            self.augmented
            / "images"
            / "val"
            / "val_0.png"
        )

        report = (
            validate_augmentation_quality(
                self.source,
                self.augmented,
            )
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

        self.assertTrue(
            any(
                "original image"
                in error.lower()
                and "modified"
                in error.lower()
                for error
                in report["errors"]
            )
        )

    def test_augmented_file_in_validation_fails(
        self,
    ):

        self.create_dataset()

        shutil.copy2(
            self.augmented
            / "images"
            / "val"
            / "val_0.png",
            self.augmented
            / "images"
            / "val"
            / "val_0__aug.png",
        )

        shutil.copy2(
            self.augmented
            / "labels"
            / "val"
            / "val_0.txt",
            self.augmented
            / "labels"
            / "val"
            / "val_0__aug.txt",
        )

        report = (
            validate_augmentation_quality(
                self.source,
                self.augmented,
            )
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

        self.assertTrue(
            any(
                "outside training"
                in error.lower()
                for error
                in report["errors"]
            )
        )

    def test_missing_augmented_label_fails(
        self,
    ):

        self.create_dataset()

        (
            self.augmented
            / "labels"
            / "train"
            / "train_0__aug.txt"
        ).unlink()

        report = (
            validate_augmentation_quality(
                self.source,
                self.augmented,
            )
        )

        self.assertEqual(
            report["status"],
            "FAILED",
        )

    def test_low_box_retention_creates_warning(
        self,
    ):

        self.create_dataset()

        # Keep only one object's augmentation
        # by deleting five complete augmented pairs.
        for class_id in range(
            1,
            6,
        ):
            (
                self.augmented
                / "images"
                / "train"
                / (
                    f"train_{class_id}"
                    "__aug.png"
                )
            ).unlink()

            (
                self.augmented
                / "labels"
                / "train"
                / (
                    f"train_{class_id}"
                    "__aug.txt"
                )
            ).unlink()

        report = (
            validate_augmentation_quality(
                self.source,
                self.augmented,
            )
        )

        # Membership failure is expected here,
        # and quality evidence must still report it.
        self.assertEqual(
            report["status"],
            "FAILED",
        )

        self.assertTrue(
            report["errors"]
        )

    def test_preview_is_created(
        self,
    ):

        self.create_dataset()

        preview = (
            self.root
            / "preview.jpg"
        )

        result = create_preview(
            self.source,
            self.augmented,
            preview,
            sample_count=3,
        )

        self.assertTrue(
            result.exists()
        )

        with Image.open(
            result
        ) as image:
            self.assertGreater(
                image.width,
                0,
            )

            self.assertGreater(
                image.height,
                0,
            )


if __name__ == "__main__":
    unittest.main()