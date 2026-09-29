"""Small synthetic fixtures test the auditor, not the real dataset."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image
from scripts.audit_dataset import audit_dataset


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def sample(self, name="scratches_1", box=(10, 20, 80, 100)):
        Image.new("L", (200, 200), color=30).save(
            self.root / f"{name}.png"
        )
        xmin, ymin, xmax, ymax = box
        xml = f"""<annotation>
          <filename>{name}.png</filename>
          <size><width>200</width><height>200</height></size>
          <object><name>scratches</name><bndbox>
            <xmin>{xmin}</xmin><ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax><ymax>{ymax}</ymax>
          </bndbox></object>
        </annotation>"""
        path = self.root / f"{name}.xml"
        path.write_text(xml, encoding="utf-8")
        return path

    def test_valid_pair(self):
        self.sample()
        report = audit_dataset(self.root)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["valid_box_count"], 1)
        self.assertEqual(report["boxes_by_class"]["scratches"], 1)

    def test_missing_annotation(self):
        self.sample().unlink()
        self.assertEqual(audit_dataset(self.root)["status"], "FAIL")

    def test_invalid_boxes(self):
        cases = (
            (80, 20, 10, 100),
            (10, 20, 201, 100),
            (10, 20, "nan", 100),
        )
        for box in cases:
            with self.subTest(box=box):
                self.sample(box=box)
                report = audit_dataset(self.root)
                self.assertEqual(report["valid_box_count"], 0)
                self.assertTrue(
                    any(
                        "invalid bounding box" in error
                        for error in report["errors"]
                    )
                )

    def test_corrupt_image(self):
        self.sample()
        (self.root / "scratches_1.png").write_bytes(b"not an image")
        report = audit_dataset(self.root)
        self.assertTrue(
            any("unreadable image" in error for error in report["errors"])
        )

    def test_malformed_xml(self):
        self.sample().write_text("<broken", encoding="utf-8")
        self.assertEqual(audit_dataset(self.root)["status"], "FAIL")

    def test_duplicates(self):
        self.sample("scratches_1")
        self.sample("scratches_2")
        report = audit_dataset(self.root)
        self.assertEqual(len(report["duplicate_image_groups"]), 1)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["status"], "REVIEW")

    def test_empty_dataset(self):
        self.assertEqual(audit_dataset(self.root)["status"], "FAIL")

    def test_unknown_class(self):
        path = self.sample()
        path.write_text(
            path.read_text().replace(
                "<name>scratches</name>",
                "<name>unknown</name>",
            )
        )
        report = audit_dataset(self.root)
        self.assertEqual(report["valid_box_count"], 0)
        self.assertTrue(
            any("unknown class" in error for error in report["errors"])
        )


if __name__ == "__main__":
    unittest.main()