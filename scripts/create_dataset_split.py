"""Tests for deterministic duplicate-safe YOLO dataset splitting."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.audit_dataset import CLASSES
from scripts.create_dataset_split import create_dataset_split


class DatasetSplitTests(unittest.TestCase):
    """Test duplicate-safe deterministic dataset splitting."""

    def setUp(self) -> None:
        """Create a temporary synthetic VOC + YOLO dataset."""

        self.temp_directory = (
            tempfile.TemporaryDirectory()
        )

        self.addCleanup(
            self.temp_directory.cleanup
        )

        self.root = Path(
            self.temp_directory.name
        )

        self.raw = (
            self.root
            / "raw"
        )

        self.labels = (
            self.root
            / "labels"
        )

        self.output = (
            self.root
            / "output"
        )

        self.raw.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.labels.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Ten samples allow an exact 70/20/10 split:
        #
        # train = 7
        # val   = 2
        # test  = 1
        #
        # sample_00 and sample_01 deliberately contain
        # identical decoded image pixels even though their
        # filenames are different.
        colors = [
            20,
            20,
            40,
            60,
            80,
            100,
            120,
            140,
            160,
            180,
        ]

        for index, color in enumerate(
            colors
        ):
            stem = (
                f"sample_{index:02d}"
            )

            class_id = (
                index
                % len(CLASSES)
            )

            self.add_sample(
                stem=stem,
                color=color,
                class_id=class_id,
            )

    def add_sample(
        self,
        stem: str,
        color: int,
        class_id: int,
    ) -> None:
        """Create one image, Pascal VOC XML, and YOLO label."""

        image_path = (
            self.raw
            / f"{stem}.png"
        )

        xml_path = (
            self.raw
            / f"{stem}.xml"
        )

        label_path = (
            self.labels
            / f"{stem}.txt"
        )

        # PNG is intentionally used so the synthetic pixel
        # values remain lossless and deterministic.
        Image.new(
            "RGB",
            (
                200,
                200,
            ),
            color=(
                color,
                color,
                color,
            ),
        ).save(
            image_path
        )

        class_name = (
            CLASSES[
                class_id
            ]
        )

        xml_content = f"""\
<annotation>
    <folder>raw</folder>
    <filename>{image_path.name}</filename>
    <size>
        <width>200</width>
        <height>200</height>
        <depth>3</depth>
    </size>
    <object>
        <name>{class_name}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>50</xmin>
            <ymin>50</ymin>
            <xmax>150</xmax>
            <ymax>150</ymax>
        </bndbox>
    </object>
</annotation>
"""

        xml_path.write_text(
            xml_content,
            encoding="utf-8",
        )

        label_path.write_text(
            (
                f"{class_id} "
                "0.500000 "
                "0.500000 "
                "0.500000 "
                "0.500000\n"
            ),
            encoding="utf-8",
        )

    @staticmethod
    def read_manifest(
        manifest_path: Path,
    ) -> dict[str, str]:
        """Read stem-to-split assignments from a split manifest."""

        assignments: dict[
            str,
            str,
        ] = {}

        with manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(
                file
            )

            for row in reader:
                assignments[
                    row["stem"]
                ] = row["split"]

        return assignments

    def test_exact_split_sizes(
        self,
    ) -> None:
        """The splitter must preserve exact 70/20/10 counts."""

        summary = create_dataset_split(
            data_root=self.raw,
            labels_root=self.labels,
            output_root=self.output,
            seed=42,
        )

        self.assertEqual(
            summary[
                "total_samples"
            ],
            10,
        )

        self.assertEqual(
            summary[
                "splits"
            ][
                "train"
            ][
                "images"
            ],
            7,
        )

        self.assertEqual(
            summary[
                "splits"
            ][
                "val"
            ][
                "images"
            ],
            2,
        )

        self.assertEqual(
            summary[
                "splits"
            ][
                "test"
            ][
                "images"
            ],
            1,
        )

    def test_duplicate_images_stay_together(
        self,
    ) -> None:
        """Identical decoded images must never cross split boundaries."""

        create_dataset_split(
            data_root=self.raw,
            labels_root=self.labels,
            output_root=self.output,
            seed=42,
        )

        assignments = self.read_manifest(
            self.output
            / "split_manifest.csv"
        )

        self.assertIn(
            "sample_00",
            assignments,
        )

        self.assertIn(
            "sample_01",
            assignments,
        )

        self.assertEqual(
            assignments[
                "sample_00"
            ],
            assignments[
                "sample_01"
            ],
            (
                "Identical decoded images were "
                "placed in different dataset splits."
            ),
        )

    def test_duplicate_group_recorded_in_summary(
        self,
    ) -> None:
        """The split summary must record duplicate groups kept together."""

        summary = create_dataset_split(
            data_root=self.raw,
            labels_root=self.labels,
            output_root=self.output,
            seed=42,
        )

        groups = summary[
            "duplicate_image_groups_kept_together"
        ]

        duplicate_stem_groups = [
            {
                Path(path).stem.casefold()
                for path in group
            }
            for group in groups
        ]

        self.assertIn(
            {
                "sample_00",
                "sample_01",
            },
            duplicate_stem_groups,
        )

    def test_same_seed_is_deterministic(
        self,
    ) -> None:
        """The same seed must generate the same split assignments."""

        first_output = (
            self.root
            / "output_first"
        )

        second_output = (
            self.root
            / "output_second"
        )

        create_dataset_split(
            data_root=self.raw,
            labels_root=self.labels,
            output_root=first_output,
            seed=42,
        )

        create_dataset_split(
            data_root=self.raw,
            labels_root=self.labels,
            output_root=second_output,
            seed=42,
        )

        first_assignments = (
            self.read_manifest(
                first_output
                / "split_manifest.csv"
            )
        )

        second_assignments = (
            self.read_manifest(
                second_output
                / "split_manifest.csv"
            )
        )

        self.assertEqual(
            first_assignments,
            second_assignments,
        )

    def test_split_files_are_copied_correctly(
        self,
    ) -> None:
        """Every split must contain matching image and label files."""

        create_dataset_split(
            data_root=self.raw,
            labels_root=self.labels,
            output_root=self.output,
            seed=42,
        )

        expected_counts = {
            "train": 7,
            "val": 2,
            "test": 1,
        }

        total_images = 0
        total_labels = 0

        for split, expected_count in (
            expected_counts.items()
        ):
            image_directory = (
                self.output
                / "images"
                / split
            )

            label_directory = (
                self.output
                / "labels"
                / split
            )

            image_files = sorted(
                image_directory.glob(
                    "*.png"
                )
            )

            label_files = sorted(
                label_directory.glob(
                    "*.txt"
                )
            )

            self.assertEqual(
                len(
                    image_files
                ),
                expected_count,
            )

            self.assertEqual(
                len(
                    label_files
                ),
                expected_count,
            )

            image_stems = {
                path.stem
                for path in image_files
            }

            label_stems = {
                path.stem
                for path in label_files
            }

            self.assertEqual(
                image_stems,
                label_stems,
            )

            total_images += len(
                image_files
            )

            total_labels += len(
                label_files
            )

        self.assertEqual(
            total_images,
            10,
        )

        self.assertEqual(
            total_labels,
            10,
        )


if __name__ == "__main__":
    unittest.main()