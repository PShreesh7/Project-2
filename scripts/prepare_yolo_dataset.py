"""Validate the prepared NEU-DET YOLO dataset and create data.yaml."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image

from scripts.audit_dataset import (
    CLASSES,
    IMAGE_SUFFIXES,
    decoded_image_fingerprint,
)


SPLITS = (
    "train",
    "val",
    "test",
)

BOUNDARY_TOLERANCE = 1e-5


def index_files(
    directory: Path,
    allowed_suffixes: set[str],
) -> dict[str, Path]:
    """Index files by case-insensitive filename stem."""

    if not directory.is_dir():
        raise ValueError(
            f"Required directory does not exist: "
            f"{directory}"
        )

    groups: dict[
        str,
        list[Path],
    ] = {}

    for path in sorted(
        directory.iterdir()
    ):
        if (
            path.is_file()
            and path.suffix.lower()
            in allowed_suffixes
        ):
            key = path.stem.casefold()

            groups.setdefault(
                key,
                [],
            ).append(
                path
            )

    indexed: dict[
        str,
        Path,
    ] = {}

    for stem, paths in groups.items():
        if len(paths) != 1:
            raise ValueError(
                f"{directory}: duplicate files "
                f"for stem {stem!r}"
            )

        indexed[
            stem
        ] = paths[0]

    return indexed


def validate_image(
    image_path: Path,
) -> tuple[int, int]:
    """Confirm an image can be decoded and has valid dimensions."""

    try:
        with Image.open(
            image_path
        ) as image:
            image.load()

            width, height = (
                image.size
            )

    except Exception as error:
        raise ValueError(
            f"Unreadable image: "
            f"{image_path}"
        ) from error

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions: "
            f"{image_path}"
        )

    return (
        width,
        height,
    )


def validate_label(
    label_path: Path,
) -> Counter:
    """Validate one YOLO label file."""

    lines = (
        label_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    if not lines:
        raise ValueError(
            f"Empty label file: "
            f"{label_path}"
        )

    counts: Counter = Counter()

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):
        line = raw_line.strip()

        if not line:
            raise ValueError(
                f"{label_path}, line "
                f"{line_number}: empty line"
            )

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path}, line "
                f"{line_number}: expected "
                "5 YOLO values"
            )

        try:
            class_id = int(
                parts[0]
            )

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            width = float(
                parts[3]
            )

            height = float(
                parts[4]
            )

        except ValueError as error:
            raise ValueError(
                f"{label_path}, line "
                f"{line_number}: invalid "
                "numeric value"
            ) from error

        values = (
            x_center,
            y_center,
            width,
            height,
        )

        if not all(
            math.isfinite(
                value
            )
            for value in values
        ):
            raise ValueError(
                f"{label_path}, line "
                f"{line_number}: non-finite "
                "coordinate"
            )

        if not (
            0 <= class_id
            < len(CLASSES)
        ):
            raise ValueError(
                f"{label_path}, line "
                f"{line_number}: invalid "
                f"class id {class_id}"
            )

        if not (
            0.0 <= x_center <= 1.0
            and
            0.0 <= y_center <= 1.0
        ):
            raise ValueError(
                f"{label_path}, line "
                f"{line_number}: box center "
                "must be normalized"
            )

        if not (
            0.0 < width <= 1.0
            and
            0.0 < height <= 1.0
        ):
            raise ValueError(
                f"{label_path}, line "
                f"{line_number}: width and "
                "height must be in (0, 1]"
            )

        left = (
            x_center
            - width / 2
        )

        right = (
            x_center
            + width / 2
        )

        top = (
            y_center
            - height / 2
        )

        bottom = (
            y_center
            + height / 2
        )

        if (
            left
            < -BOUNDARY_TOLERANCE
            or top
            < -BOUNDARY_TOLERANCE
            or right
            > 1.0
            + BOUNDARY_TOLERANCE
            or bottom
            > 1.0
            + BOUNDARY_TOLERANCE
        ):
            raise ValueError(
                f"{label_path}, line "
                f"{line_number}: bounding box "
                "extends outside image bounds"
            )

        counts[
            class_id
        ] += 1

    return counts


def validate_split(
    dataset_root: Path,
    split: str,
) -> dict:
    """Validate one train/val/test split."""

    image_directory = (
        dataset_root
        / "images"
        / split
    )

    label_directory = (
        dataset_root
        / "labels"
        / split
    )

    images = index_files(
        image_directory,
        set(
            IMAGE_SUFFIXES
        ),
    )

    labels = index_files(
        label_directory,
        {".txt"},
    )

    image_stems = set(
        images
    )

    label_stems = set(
        labels
    )

    missing_labels = sorted(
        image_stems
        - label_stems
    )

    missing_images = sorted(
        label_stems
        - image_stems
    )

    if missing_labels:
        raise ValueError(
            f"{split}: images without "
            "labels: "
            + ", ".join(
                missing_labels[:10]
            )
        )

    if missing_images:
        raise ValueError(
            f"{split}: labels without "
            "images: "
            + ", ".join(
                missing_images[:10]
            )
        )

    if not image_stems:
        raise ValueError(
            f"{split}: split contains "
            "no images"
        )

    class_objects: Counter = Counter()

    image_class_presence: Counter = (
        Counter()
    )

    content_hashes: dict[
        str,
        list[str],
    ] = {}

    total_objects = 0

    for stem in sorted(
        image_stems
    ):
        validate_image(
            images[
                stem
            ]
        )

        # Generate a decoded-image fingerprint.
        #
        # This is independent of the filename, so two images
        # with different names but identical decoded pixels
        # receive the same SHA-256 fingerprint.
        try:
            (
                _,
                _,
                digest,
            ) = decoded_image_fingerprint(
                images[
                    stem
                ]
            )

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as error:
            raise ValueError(
                "Unable to fingerprint image "
                f"{images[stem]}: {error}"
            ) from error

        content_hashes.setdefault(
            digest,
            [],
        ).append(
            stem
        )

        counts = validate_label(
            labels[
                stem
            ]
        )

        total_objects += sum(
            counts.values()
        )

        class_objects.update(
            counts
        )

        for class_id in counts:
            image_class_presence[
                class_id
            ] += 1

    return {
        "stems": image_stems,
        "content_hashes": content_hashes,
        "image_count": len(
            image_stems
        ),
        "label_count": len(
            label_stems
        ),
        "object_count": total_objects,
        "objects_by_class": {
            CLASSES[
                class_id
            ]: class_objects.get(
                class_id,
                0,
            )
            for class_id
            in range(
                len(CLASSES)
            )
        },
        "images_by_class": {
            CLASSES[
                class_id
            ]: image_class_presence.get(
                class_id,
                0,
            )
            for class_id
            in range(
                len(CLASSES)
            )
        },
    }


def run_preflight(
    dataset_root: Path,
    expected_total: int | None = None,
) -> dict:
    """Run complete dataset preflight validation."""

    dataset_root = (
        dataset_root
        .expanduser()
        .resolve()
    )

    report = {
        "dataset_root": str(
            dataset_root
        ),
        "status": "PASSED",
        "class_names": list(
            CLASSES
        ),
        "splits": {},
        "content_leakage": {
            "train_val": [],
            "train_test": [],
            "val_test": [],
        },
        "errors": [],
        "warnings": [],
    }

    if not dataset_root.is_dir():
        report[
            "status"
        ] = "FAILED"

        report[
            "errors"
        ].append(
            "Dataset root does not exist."
        )

        return report

    raw_split_results = {}

    for split in SPLITS:
        try:
            result = validate_split(
                dataset_root,
                split,
            )

            raw_split_results[
                split
            ] = result

            # Do not put internal stem sets or the complete
            # SHA-256 mapping into the normal split report.
            report[
                "splits"
            ][
                split
            ] = {
                key: value
                for key, value
                in result.items()
                if key not in {
                    "stems",
                    "content_hashes",
                }
            }

        except ValueError as error:
            report[
                "errors"
            ].append(
                str(
                    error
                )
            )

    if len(
        raw_split_results
    ) == len(
        SPLITS
    ):
        train_stems = (
            raw_split_results[
                "train"
            ][
                "stems"
            ]
        )

        val_stems = (
            raw_split_results[
                "val"
            ][
                "stems"
            ]
        )

        test_stems = (
            raw_split_results[
                "test"
            ][
                "stems"
            ]
        )

        # ---------------------------------------------------------
        # Filename/stem leakage check
        # ---------------------------------------------------------

        overlap_train_val = (
            train_stems
            & val_stems
        )

        overlap_train_test = (
            train_stems
            & test_stems
        )

        overlap_val_test = (
            val_stems
            & test_stems
        )

        if overlap_train_val:
            report[
                "errors"
            ].append(
                "Data leakage detected "
                "between train and val."
            )

        if overlap_train_test:
            report[
                "errors"
            ].append(
                "Data leakage detected "
                "between train and test."
            )

        if overlap_val_test:
            report[
                "errors"
            ].append(
                "Data leakage detected "
                "between val and test."
            )

        # ---------------------------------------------------------
        # Decoded-image content leakage check
        # ---------------------------------------------------------
        #
        # Filename checking alone is not enough because the same
        # image can exist under a different filename.
        #
        # Here we compare SHA-256 hashes created from decoded RGB
        # pixels plus image dimensions.

        for first, second in (
            (
                "train",
                "val",
            ),
            (
                "train",
                "test",
            ),
            (
                "val",
                "test",
            ),
        ):
            first_hashes = (
                raw_split_results[
                    first
                ][
                    "content_hashes"
                ]
            )

            second_hashes = (
                raw_split_results[
                    second
                ][
                    "content_hashes"
                ]
            )

            shared_hashes = sorted(
                set(
                    first_hashes
                )
                & set(
                    second_hashes
                )
            )

            pair_key = (
                f"{first}_{second}"
            )

            leakage_details = []

            for digest in shared_hashes:
                leakage_details.append(
                    {
                        "sha256": digest,
                        first: first_hashes[
                            digest
                        ],
                        second: second_hashes[
                            digest
                        ],
                    }
                )

            report[
                "content_leakage"
            ][
                pair_key
            ] = leakage_details

            if shared_hashes:
                report[
                    "errors"
                ].append(
                    "Content leakage detected "
                    f"between {first} and {second}: "
                    f"{len(shared_hashes)} identical "
                    "decoded image fingerprint(s)."
                )

        # ---------------------------------------------------------
        # Dataset totals
        # ---------------------------------------------------------

        total_samples = sum(
            raw_split_results[
                split
            ][
                "image_count"
            ]
            for split in SPLITS
        )

        report[
            "total_samples"
        ] = total_samples

        report[
            "total_objects"
        ] = sum(
            raw_split_results[
                split
            ][
                "object_count"
            ]
            for split in SPLITS
        )

        if (
            expected_total
            is not None
            and total_samples
            != expected_total
        ):
            report[
                "errors"
            ].append(
                "Unexpected dataset size: "
                f"expected {expected_total}, "
                f"found {total_samples}."
            )

        # ---------------------------------------------------------
        # Class coverage
        # ---------------------------------------------------------

        train_classes = (
            raw_split_results[
                "train"
            ][
                "objects_by_class"
            ]
        )

        for class_name in CLASSES:
            if (
                train_classes[
                    class_name
                ]
                == 0
            ):
                report[
                    "errors"
                ].append(
                    "Training split has "
                    "no examples for class "
                    f"{class_name!r}."
                )

        for split in (
            "val",
            "test",
        ):
            distribution = (
                raw_split_results[
                    split
                ][
                    "objects_by_class"
                ]
            )

            for class_name in CLASSES:
                if (
                    distribution[
                        class_name
                    ]
                    == 0
                ):
                    report[
                        "warnings"
                    ].append(
                        f"{split} split has "
                        "no objects for class "
                        f"{class_name!r}."
                    )

    if report[
        "errors"
    ]:
        report[
            "status"
        ] = "FAILED"

    return report


def create_ultralytics_yaml(
    dataset_root: Path,
) -> Path:
    """Create a portable Ultralytics dataset YAML."""

    dataset_root = (
        dataset_root
        .expanduser()
        .resolve()
    )

    yaml_path = (
        dataset_root
        / "data.yaml"
    )

    # JSON-style quoted strings are valid YAML
    # and safely handle Windows paths.
    root_value = json.dumps(
        dataset_root.as_posix()
    )

    lines = [
        "# Auto-generated by "
        "scripts.prepare_yolo_dataset",
        "",
        f"path: {root_value}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASSES)}",
        "",
        "names:",
    ]

    for class_id, class_name in enumerate(
        CLASSES
    ):
        lines.append(
            f"  {class_id}: "
            f"{json.dumps(class_name)}"
        )

    yaml_path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    return yaml_path


def write_report(
    dataset_root: Path,
    report: dict,
) -> Path:
    """Save the preflight report as JSON."""

    report_path = (
        dataset_root
        / "preflight_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return report_path


def main() -> int:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo"
        ),
        help=(
            "Prepared train/val/test "
            "YOLO dataset root."
        ),
    )

    parser.add_argument(
        "--expected-total",
        type=int,
        default=None,
        help=(
            "Optional expected number "
            "of unique images."
        ),
    )

    args = parser.parse_args()

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )

    report = run_preflight(
        dataset_root,
        expected_total=(
            args.expected_total
        ),
    )

    if dataset_root.is_dir():
        report_path = write_report(
            dataset_root,
            report,
        )
    else:
        report_path = None

    if report[
        "status"
    ] != "PASSED":
        print(
            "YOLO dataset preflight: FAILED"
        )

        for error in report[
            "errors"
        ]:
            print(
                f"ERROR: {error}"
            )

        if report_path:
            print(
                f"Report: {report_path}"
            )

        return 1

    yaml_path = (
        create_ultralytics_yaml(
            dataset_root
        )
    )

    print(
        "YOLO dataset preflight: PASSED"
    )

    print(
        f"Total images: "
        f"{report['total_samples']}"
    )

    print(
        f"Total objects: "
        f"{report['total_objects']}"
    )

    for split in SPLITS:
        details = (
            report[
                "splits"
            ][
                split
            ]
        )

        print(
            f"{split}: "
            f"{details['image_count']} images, "
            f"{details['object_count']} objects"
        )

    if report[
        "warnings"
    ]:
        print(
            "\nWarnings:"
        )

        for warning in report[
            "warnings"
        ]:
            print(
                f"- {warning}"
            )

    print(
        f"\nUltralytics YAML: "
        f"{yaml_path}"
    )

    print(
        f"Preflight report: "
        f"{report_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )