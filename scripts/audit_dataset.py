"""Read-only image and Pascal VOC annotation audit for NEU-DET."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def decoded_image_fingerprint(path: Path) -> tuple[int, int, str]:
    """Return decoded image dimensions and a content hash.

    The hash is based on RGB pixel content plus dimensions, so differently
    named or differently encoded files with identical decoded pixels receive
    the same fingerprint.
    """
    with Image.open(path) as image:
        image.load()

        width, height = image.size
        pixels = image.convert("RGB").tobytes()

    digest = hashlib.sha256(
        f"{width}x{height}:".encode() + pixels
    ).hexdigest()

    return width, height, digest


def audit_dataset(root: Path) -> dict:
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise ValueError(
            f"Dataset directory does not exist: {root}"
        )

    images = defaultdict(list)
    annotations = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if path.is_file():
            if path.suffix.lower() in IMAGE_SUFFIXES:
                images[path.stem.casefold()].append(path)

            elif path.suffix.lower() == ".xml":
                annotations[path.stem.casefold()].append(path)

    errors = []
    warnings = []

    class_counts = Counter(
        {
            name: 0
            for name in CLASSES
        }
    )

    sizes = Counter()
    hashes = defaultdict(list)
    dimensions = {}

    relative = lambda path: path.relative_to(root).as_posix()

    if not images:
        errors.append(
            "No supported images found."
        )

    if not annotations:
        errors.append(
            "No XML annotations found; use the NEU-DET archive."
        )

    for stem in sorted(
        images.keys() | annotations.keys()
    ):
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
            errors.append(
                f"{stem}: expected one image and one XML; "
                f"found {len(image_paths)} image(s), "
                f"{len(xml_paths)} XML(s)."
            )

    for paths in images.values():
        for path in paths:
            try:
                (
                    width,
                    height,
                    digest,
                ) = decoded_image_fingerprint(
                    path
                )

                dimensions[path] = (
                    width,
                    height,
                )

                sizes[
                    f"{width}x{height}"
                ] += 1

                hashes[
                    digest
                ].append(
                    relative(path)
                )

            except (
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ) as error:
                errors.append(
                    f"{relative(path)}: "
                    f"unreadable image: {error}"
                )

    paired_annotations = 0

    for stem, xml_paths in annotations.items():
        if (
            len(xml_paths) != 1
            or len(images.get(stem, [])) != 1
        ):
            continue

        image_path = images[stem][0]
        xml_path = xml_paths[0]

        if image_path not in dimensions:
            continue

        label = relative(xml_path)

        width, height = dimensions[
            image_path
        ]

        try:
            annotation = ET.parse(
                xml_path
            ).getroot()

            if annotation.tag != "annotation":
                raise ValueError(
                    "expected Pascal VOC <annotation> root"
                )

            xml_size = (
                int(
                    annotation.findtext(
                        "size/width",
                        "",
                    )
                ),
                int(
                    annotation.findtext(
                        "size/height",
                        "",
                    )
                ),
            )

            if xml_size != (
                width,
                height,
            ):
                errors.append(
                    f"{label}: XML size {xml_size} "
                    f"!= image size {(width, height)}."
                )

            filename = annotation.findtext(
                "filename",
                "",
            ).strip()

            if (
                filename
                and Path(
                    filename.replace(
                        "\\",
                        "/",
                    )
                ).stem.casefold()
                != stem
            ):
                errors.append(
                    f"{label}: filename does not match "
                    "paired image."
                )

            objects = annotation.findall(
                "object"
            )

            if not objects:
                errors.append(
                    f"{label}: no defect objects found."
                )

            paired_annotations += 1

            for index, obj in enumerate(
                objects,
                start=1,
            ):
                name = obj.findtext(
                    "name",
                    "",
                ).strip()

                if name not in CLASSES:
                    errors.append(
                        f"{label}, object {index}: "
                        f"unknown class {name!r}."
                    )
                    continue

                try:
                    (
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                    ) = [
                        float(
                            obj.findtext(
                                f"bndbox/{key}",
                                "",
                            )
                        )
                        for key in (
                            "xmin",
                            "ymin",
                            "xmax",
                            "ymax",
                        )
                    ]

                    valid = (
                        all(
                            math.isfinite(v)
                            for v in (
                                xmin,
                                ymin,
                                xmax,
                                ymax,
                            )
                        )
                        and 0 <= xmin < xmax <= width
                        and 0 <= ymin < ymax <= height
                    )

                    if not valid:
                        raise ValueError(
                            "box is inverted, empty, "
                            "non-finite, or outside the image"
                        )

                except ValueError as error:
                    errors.append(
                        f"{label}, object {index}: "
                        f"invalid bounding box: {error}."
                    )
                    continue

                class_counts[
                    name
                ] += 1

        except (
            ET.ParseError,
            OSError,
            ValueError,
        ) as error:
            errors.append(
                f"{label}: invalid annotation: {error}"
            )

    duplicates = [
        paths
        for paths in hashes.values()
        if len(paths) > 1
    ]

    if duplicates:
        warnings.append(
            "Identical decoded images found; "
            "keep each group in one split."
        )

    missing_classes = [
        name
        for name, count in class_counts.items()
        if count == 0
    ]

    if missing_classes:
        warnings.append(
            "No valid boxes counted for: "
            f"{', '.join(missing_classes)}."
        )

    if any(
        size != "200x200"
        for size in sizes
    ):
        warnings.append(
            "Images with dimensions other than 200x200 "
            "found; verify dataset source."
        )

    return {
        "dataset_root": str(root),
        "status": (
            "FAIL"
            if errors
            else "REVIEW"
            if warnings
            else "PASS"
        ),
        "image_count": sum(
            map(
                len,
                images.values(),
            )
        ),
        "annotation_count": sum(
            map(
                len,
                annotations.values(),
            )
        ),
        "paired_annotations_parsed": paired_annotations,
        "valid_box_count": sum(
            class_counts.values()
        ),
        "boxes_by_class": dict(
            class_counts
        ),
        "image_sizes": dict(
            sizes
        ),
        "duplicate_image_groups": duplicates,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/dataset_audit.json"
        ),
    )

    args = parser.parse_args()

    output = args.output.resolve()

    root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    if (
        output.suffix.lower() != ".json"
        or output.is_relative_to(root)
    ):
        parser.error(
            "Write a .json report outside "
            "the dataset directory."
        )

    try:
        report = audit_dataset(
            root
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                report,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    except (
        OSError,
        ValueError,
    ) as error:
        parser.exit(
            1,
            f"Audit could not complete: {error}\n",
        )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        f"Report saved: {output}"
    )

    return (
        1
        if report["errors"]
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )