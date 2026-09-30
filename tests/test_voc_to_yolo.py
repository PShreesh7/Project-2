"""Tests for Pascal VOC to YOLO annotation conversion."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.convert_voc_to_yolo import (
    convert_dataset,
    voc_box_to_yolo,
)


class VocToYoloTests(unittest.TestCase):

    def setUp(self):
        self.source_directory = (
            tempfile.TemporaryDirectory()
        )

        self.output_directory = (
            tempfile.TemporaryDirectory()
        )

        self.addCleanup(
            self.source_directory.cleanup
        )

        self.addCleanup(
            self.output_directory.cleanup
        )

        self.source = Path(
            self.source_directory.name
        )

        self.output = (
            Path(
                self.output_directory.name
            )
            / "converted"
        )

    def add_sample(
        self,
        stem: str = "sample_1",
        objects: tuple[
            tuple[
                str,
                tuple[
                    int,
                    int,
                    int,
                    int,
                ],
            ],
            ...,
        ] = (
            (
                "scratches",
                (10, 20, 110, 120),
            ),
        ),
    ) -> None:

        Image.new(
            "L",
            (200, 200),
            color=80,
        ).save(
            self.source
            / f"{stem}.png"
        )

        object_xml = []

        for (
            class_name,
            (
                xmin,
                ymin,
                xmax,
                ymax,
            ),
        ) in objects:

            object_xml.append(
                f"""<object>
                  <name>{class_name}</name>
                  <bndbox>
                    <xmin>{xmin}</xmin>
                    <ymin>{ymin}</ymin>
                    <xmax>{xmax}</xmax>
                    <ymax>{ymax}</ymax>
                  </bndbox>
                </object>"""
            )

        xml = f"""<annotation>
          <filename>{stem}.png</filename>
          <size>
            <width>200</width>
            <height>200</height>
          </size>
          {''.join(object_xml)}
        </annotation>"""

        (
            self.source
            / f"{stem}.xml"
        ).write_text(
            xml,
            encoding="utf-8",
        )

    def test_box_conversion(self):

        result = voc_box_to_yolo(
            10,
            20,
            110,
            120,
            200,
            200,
        )

        expected = (
            0.30,
            0.35,
            0.50,
            0.50,
        )

        for actual, target in zip(
            result,
            expected,
        ):
            self.assertAlmostEqual(
                actual,
                target,
            )

    def test_dataset_conversion_writes_yolo_label(
        self,
    ):
        self.add_sample()

        summary = convert_dataset(
            self.source,
            self.output,
        )

        label = (
            self.output
            / "labels"
            / "sample_1.txt"
        ).read_text(
            encoding="utf-8"
        ).strip()

        self.assertEqual(
            label,
            (
                "5 "
                "0.300000 "
                "0.350000 "
                "0.500000 "
                "0.500000"
            ),
        )

        self.assertEqual(
            summary["image_count"],
            1,
        )

        self.assertEqual(
            summary["object_count"],
            1,
        )

        self.assertEqual(
            summary[
                "boxes_by_class"
            ]["scratches"],
            1,
        )

    def test_multiple_objects_keep_expected_class_ids(
        self,
    ):
        self.add_sample(
            objects=(
                (
                    "crazing",
                    (0, 0, 50, 50),
                ),
                (
                    "inclusion",
                    (50, 50, 150, 150),
                ),
            )
        )

        convert_dataset(
            self.source,
            self.output,
        )

        lines = (
            self.output
            / "labels"
            / "sample_1.txt"
        ).read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertEqual(
            lines[0].split()[0],
            "0",
        )

        self.assertEqual(
            lines[1].split()[0],
            "1",
        )

    def test_class_metadata_is_written(
        self,
    ):
        self.add_sample()

        convert_dataset(
            self.source,
            self.output,
        )

        classes = (
            self.output
            / "classes.txt"
        ).read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertEqual(
            classes,
            [
                "crazing",
                "inclusion",
                "patches",
                "pitted_surface",
                "rolled-in_scale",
                "scratches",
            ],
        )

        summary = json.loads(
            (
                self.output
                / "conversion_summary.json"
            ).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            summary[
                "class_to_id"
            ]["rolled-in_scale"],
            4,
        )

    def test_conversion_stops_if_audit_fails(
        self,
    ):
        Image.new(
            "L",
            (200, 200),
            color=80,
        ).save(
            self.source
            / "missing_xml.png"
        )

        with self.assertRaisesRegex(
            ValueError,
            "Dataset audit failed",
        ):
            convert_dataset(
                self.source,
                self.output,
            )

    def test_output_cannot_be_inside_source_dataset(
        self,
    ):
        self.add_sample()

        with self.assertRaisesRegex(
            ValueError,
            "outside the source dataset",
        ):
            convert_dataset(
                self.source,
                self.source
                / "generated",
            )


if __name__ == "__main__":
    unittest.main()