"""Tests for training-only NEU-DET augmentation."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.augment_training_data import (
    create_augmented_dataset,
    load_yolo_label,
)


class TrainingAugmentationTests(
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

        self.output = (
            self.root
            / "augmented"
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
            color=(
                80 + class_id,
                80 + class_id,
                80 + class_id,
            ),
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
                "0.300000 "
                "0.300000\n"
            ),
            encoding="utf-8",
        )

    def create_valid_dataset(
        self,
    ) -> None:

        for class_id in range(
            6
        ):
            self.add_sample(
                "train",
                f"train_{class_id}",
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

    def test_load_yolo_label(
        self,
    ):

        self.add_sample(
            "train",
            "sample",
            3,
        )

        (
            bboxes,
            labels,
        ) = load_yolo_label(
            self.dataset
            / "labels"
            / "train"
            / "sample.txt"
        )

        self.assertEqual(
            labels,
            [3],
        )

        self.assertEqual(
            len(
                bboxes
            ),
            1,
        )

        self.assertEqual(
            bboxes[0],
            [
                0.5,
                0.5,
                0.3,
                0.3,
            ],
        )

    def test_one_augmented_copy_per_train_image(
        self,
    ):

        self.create_valid_dataset()

        summary = (
            create_augmented_dataset(
                self.dataset,
                self.output,
                seed=42,
                expected_train=6,
            )
        )

        self.assertEqual(
            summary[
                "augmented_train_images"
            ],
            6,
        )

        self.assertEqual(
            summary[
                "final_train_images"
            ],
            12,
        )

        augmented_images = list(
            (
                self.output
                / "images"
                / "train"
            ).glob(
                "*__aug.png"
            )
        )

        augmented_labels = list(
            (
                self.output
                / "labels"
                / "train"
            ).glob(
                "*__aug.txt"
            )
        )

        self.assertEqual(
            len(
                augmented_images
            ),
            6,
        )

        self.assertEqual(
            len(
                augmented_labels
            ),
            6,
        )

    def test_validation_is_not_augmented(
        self,
    ):

        self.create_valid_dataset()

        create_augmented_dataset(
            self.dataset,
            self.output,
            seed=42,
        )

        augmented_validation = list(
            (
                self.output
                / "images"
                / "val"
            ).glob(
                "*__aug*"
            )
        )

        self.assertEqual(
            augmented_validation,
            [],
        )

        self.assertEqual(
            len(
                list(
                    (
                        self.output
                        / "images"
                        / "val"
                    ).glob("*")
                )
            ),
            6,
        )

    def test_test_split_is_not_augmented(
        self,
    ):

        self.create_valid_dataset()

        create_augmented_dataset(
            self.dataset,
            self.output,
            seed=42,
        )

        augmented_test = list(
            (
                self.output
                / "images"
                / "test"
            ).glob(
                "*__aug*"
            )
        )

        self.assertEqual(
            augmented_test,
            [],
        )

        self.assertEqual(
            len(
                list(
                    (
                        self.output
                        / "images"
                        / "test"
                    ).glob("*")
                )
            ),
            6,
        )

    def test_augmented_labels_remain_valid(
        self,
    ):

        self.create_valid_dataset()

        create_augmented_dataset(
            self.dataset,
            self.output,
            seed=42,
        )

        for label_path in (
            self.output
            / "labels"
            / "train"
        ).glob(
            "*__aug.txt"
        ):

            (
                bboxes,
                class_ids,
            ) = load_yolo_label(
                label_path
            )

            self.assertGreater(
                len(
                    bboxes
                ),
                0,
            )

            self.assertEqual(
                len(
                    bboxes
                ),
                len(
                    class_ids
                ),
            )

    def test_expected_train_count_is_enforced(
        self,
    ):

        self.create_valid_dataset()

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected training image count",
        ):
            create_augmented_dataset(
                self.dataset,
                self.output,
                expected_train=1260,
            )

    def test_source_dataset_is_unchanged(
        self,
    ):

        self.create_valid_dataset()

        original_train_images = {
            path.name
            for path in (
                self.dataset
                / "images"
                / "train"
            ).glob("*")
        }

        create_augmented_dataset(
            self.dataset,
            self.output,
            seed=42,
        )

        after_train_images = {
            path.name
            for path in (
                self.dataset
                / "images"
                / "train"
            ).glob("*")
        }

        self.assertEqual(
            original_train_images,
            after_train_images,
        )


if __name__ == "__main__":
    unittest.main()