"""Convert validated NEU-DET Pascal VOC annotations to YOLO label files."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from scripts.audit_dataset import CLASSES, IMAGE_SUFFIXES, audit_dataset


CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASSES)
}


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert one Pascal VOC box into normalized YOLO coordinates."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "image dimensions must be positive"
        )

    if not (
        0 <= xmin < xmax <= image_width
    ):
        raise ValueError(
            "x coordinates are outside the image or inverted"
        )

    if not (
        0 <= ymin < ymax <= image_height
    ):
        raise ValueError(
            "y coordinates are outside the image or inverted"
        )

    x_center = (
        (xmin + xmax) / 2.0
    ) / image_width

    y_center = (
        (ymin + ymax) / 2.0
    ) / image_height

    box_width = (
        xmax - xmin
    ) / image_width

    box_height = (
        ymax - ymin
    ) / image_height

    return (
        x_center,
        y_center,
        box_width,
        box_height,
    )


def _index_dataset(
    root: Path,
) -> tuple[
    dict[str, Path],
    dict[str, Path],
]:
    """Index one image and one XML annotation per filename stem."""

    images: dict[str, list[Path]] = defaultdict(list)

    annotations: dict[str, list[Path]] = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() in IMAGE_SUFFIXES:
            images[
                path.stem.casefold()
            ].append(path)

        elif path.suffix.lower() == ".xml":
            annotations[
                path.stem.casefold()
            ].append(path)

    image_index: dict[str, Path] = {}
    annotation_index: dict[str, Path] = {}

    stems = sorted(
        images.keys()
        | annotations.keys()
    )

    for stem in stems:
        image_paths = images.get(
            stem,
            [],
        )

        xml_paths = annotations.get(
            stem,
            [],
        )

        if (
            len(image_paths) != 1
            or len(xml_paths) != 1
        ):
            raise ValueError(
                f"{stem}: expected exactly one "
                "image and one XML annotation"
            )

        image_index[stem] = image_paths[0]
        annotation_index[stem] = xml_paths[0]

    return (
        image_index,
        annotation_index,
    )


def _parse_annotation(
    xml_path: Path,
    image_path: Path,
) -> tuple[list[str], Counter]:
    """Convert one XML annotation into YOLO label lines."""

    with Image.open(image_path) as image:
        image.load()
        width, height = image.size

    root = ET.parse(
        xml_path
    ).getroot()

    if root.tag != "annotation":
        raise ValueError(
            f"{xml_path}: expected Pascal VOC "
            "<annotation> root"
        )

    lines: list[str] = []
    class_counts: Counter = Counter()

    for index, obj in enumerate(
        root.findall("object"),
        start=1,
    ):
        class_name = obj.findtext(
            "name",
            "",
        ).strip()

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"{xml_path}, object {index}: "
                f"unknown class {class_name!r}"
            )

        try:
            xmin = float(
                obj.findtext(
                    "bndbox/xmin",
                    "",
                )
            )

            ymin = float(
                obj.findtext(
                    "bndbox/ymin",
                    "",
                )
            )

            xmax = float(
                obj.findtext(
                    "bndbox/xmax",
                    "",
                )
            )

            ymax = float(
                obj.findtext(
                    "bndbox/ymax",
                    "",
                )
            )

        except ValueError as error:
            raise ValueError(
                f"{xml_path}, object {index}: "
                "non-numeric bounding box"
            ) from error

        (
            x_center,
            y_center,
            box_width,
            box_height,
        ) = voc_box_to_yolo(
            xmin,
            ymin,
            xmax,
            ymax,
            width,
            height,
        )

        class_id = CLASS_TO_ID[
            class_name
        ]

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        class_counts[
            class_name
        ] += 1

    if not lines:
        raise ValueError(
            f"{xml_path}: annotation contains "
            "no defect objects"
        )

    return (
        lines,
        class_counts,
    )


def convert_dataset(
    source_root: Path,
    output_root: Path,
) -> dict:
    """Validate and convert NEU-DET XML annotations to YOLO labels."""

    source_root = (
        source_root
        .expanduser()
        .resolve()
    )

    output_root = (
        output_root
        .expanduser()
        .resolve()
    )

    if not source_root.is_dir():
        raise ValueError(
            "Dataset directory does not exist: "
            f"{source_root}"
        )

    if (
        output_root == source_root
        or output_root.is_relative_to(
            source_root
        )
    ):
        raise ValueError(
            "Output directory must be outside "
            "the source dataset"
        )

    # Reuse the validation implemented in Commit 2.
    audit = audit_dataset(
        source_root
    )

    if audit["errors"]:
        raise ValueError(
            "Dataset audit failed. Resolve "
            "Commit 2 audit errors before conversion."
        )

    (
        image_index,
        annotation_index,
    ) = _index_dataset(
        source_root
    )

    labels_dir = (
        output_root
        / "labels"
    )

    # Only remove generated labels.
    # The original NEU-DET dataset is never modified.
    if labels_dir.exists():
        shutil.rmtree(
            labels_dir
        )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_objects = 0

    class_counts: Counter = Counter(
        {
            name: 0
            for name in CLASSES
        }
    )

    for stem in sorted(
        annotation_index
    ):
        lines, counts = _parse_annotation(
            annotation_index[stem],
            image_index[stem],
        )

        original_stem = (
            image_index[stem].stem
        )

        label_path = (
            labels_dir
            / f"{original_stem}.txt"
        )

        label_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        total_objects += len(lines)

        class_counts.update(
            counts
        )

    classes_path = (
        output_root
        / "classes.txt"
    )

    classes_path.write_text(
        "\n".join(CLASSES) + "\n",
        encoding="utf-8",
    )

    summary = {
        "source_root": str(
            source_root
        ),
        "output_root": str(
            output_root
        ),
        "class_names": list(
            CLASSES
        ),
        "class_to_id": CLASS_TO_ID,
        "image_count": len(
            image_index
        ),
        "label_file_count": len(
            annotation_index
        ),
        "object_count": total_objects,
        "boxes_by_class": dict(
            class_counts
        ),
    }

    summary_path = (
        output_root
        / "conversion_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Root of the original "
            "NEU-DET dataset."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo_stage"
        ),
        help=(
            "Generated YOLO labels "
            "and metadata directory."
        ),
    )

    args = parser.parse_args()

    try:
        summary = convert_dataset(
            args.data_root,
            args.output_root,
        )

    except (
        OSError,
        ET.ParseError,
        ValueError,
    ) as error:
        parser.exit(
            1,
            f"Conversion failed: {error}\n",
        )

    print(
        "NEU-DET Pascal VOC -> "
        "YOLO conversion complete"
    )

    print(
        f"Images: "
        f"{summary['image_count']}"
    )

    print(
        f"Label files: "
        f"{summary['label_file_count']}"
    )

    print(
        f"Objects: "
        f"{summary['object_count']}"
    )

    print(
        f"Output: "
        f"{summary['output_root']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )