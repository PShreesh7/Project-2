"""Validate quality and integrity of the augmented NEU-DET dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.audit_dataset import (
    CLASSES,
    IMAGE_SUFFIXES,
)
from scripts.prepare_yolo_dataset import (
    run_preflight,
    validate_label,
)


AUGMENTED_SUFFIX = "__aug"

DEFAULT_MAX_CLASS_DRIFT = 0.05
DEFAULT_MIN_BOX_RETENTION = 0.95


def sha256_file(
    path: Path,
) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def index_files(
    directory: Path,
    suffixes: set[str],
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
            in suffixes
        ):
            stem = (
                path.stem.casefold()
            )

            groups.setdefault(
                stem,
                [],
            ).append(path)

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


def get_split_indexes(
    dataset_root: Path,
    split: str,
) -> tuple[
    dict[str, Path],
    dict[str, Path],
]:
    """Return image and label indexes for one split."""

    images = index_files(
        dataset_root
        / "images"
        / split,
        set(
            IMAGE_SUFFIXES
        ),
    )

    labels = index_files(
        dataset_root
        / "labels"
        / split,
        {".txt"},
    )

    return (
        images,
        labels,
    )


def read_label_rows(
    label_path: Path,
) -> list[
    tuple[
        int,
        float,
        float,
        float,
        float,
    ]
]:
    """Validate and parse one YOLO label."""

    validate_label(
        label_path
    )

    rows = []

    for line in (
        label_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    ):
        parts = line.split()

        rows.append(
            (
                int(
                    parts[0]
                ),
                float(
                    parts[1]
                ),
                float(
                    parts[2]
                ),
                float(
                    parts[3]
                ),
                float(
                    parts[4]
                ),
            )
        )

    return rows


def canonical_box(
    row: tuple[
        int,
        float,
        float,
        float,
        float,
    ],
) -> tuple:
    """Create a stable representation for duplicate detection."""

    return (
        row[0],
        round(
            row[1],
            6,
        ),
        round(
            row[2],
            6,
        ),
        round(
            row[3],
            6,
        ),
        round(
            row[4],
            6,
        ),
    )


def count_duplicate_boxes(
    label_path: Path,
) -> int:
    """Count duplicate annotation rows inside one label file."""

    rows = [
        canonical_box(
            row
        )
        for row in read_label_rows(
            label_path
        )
    ]

    counts = Counter(
        rows
    )

    return sum(
        count - 1
        for count in counts.values()
        if count > 1
    )


def count_objects_by_class(
    labels: dict[
        str,
        Path,
    ],
) -> Counter:
    """Count all objects per class."""

    counts: Counter = Counter()

    for label_path in labels.values():

        for row in read_label_rows(
            label_path
        ):
            counts[
                row[0]
            ] += 1

    return counts


def class_distribution(
    counts: Counter,
) -> dict[
    int,
    float,
]:
    """Convert object counts into class proportions."""

    total = sum(
        counts.values()
    )

    if total == 0:
        return {
            class_id: 0.0
            for class_id in range(
                len(CLASSES)
            )
        }

    return {
        class_id: (
            counts.get(
                class_id,
                0,
            )
            / total
        )
        for class_id in range(
            len(CLASSES)
        )
    }


def compare_original_split(
    source_root: Path,
    augmented_root: Path,
    split: str,
) -> list[str]:
    """Verify that original files were copied without modification."""

    errors: list[str] = []

    (
        source_images,
        source_labels,
    ) = get_split_indexes(
        source_root,
        split,
    )

    (
        output_images,
        output_labels,
    ) = get_split_indexes(
        augmented_root,
        split,
    )

    source_stems = set(
        source_images
    )

    if split == "train":
        output_original_stems = {
            stem
            for stem in output_images
            if not stem.endswith(
                AUGMENTED_SUFFIX
            )
        }
    else:
        output_original_stems = set(
            output_images
        )

    if (
        source_stems
        != output_original_stems
    ):
        errors.append(
            f"{split}: original image membership changed."
        )

    source_label_stems = set(
        source_labels
    )

    if split == "train":
        output_original_label_stems = {
            stem
            for stem in output_labels
            if not stem.endswith(
                AUGMENTED_SUFFIX
            )
        }
    else:
        output_original_label_stems = set(
            output_labels
        )

    if (
        source_label_stems
        != output_original_label_stems
    ):
        errors.append(
            f"{split}: original label membership changed."
        )

    common_image_stems = (
        source_stems
        & output_original_stems
    )

    for stem in sorted(
        common_image_stems
    ):
        if (
            sha256_file(
                source_images[
                    stem
                ]
            )
            != sha256_file(
                output_images[
                    stem
                ]
            )
        ):
            errors.append(
                f"{split}: original image "
                f"{stem!r} was modified."
            )

    common_label_stems = (
        source_label_stems
        & output_original_label_stems
    )

    for stem in sorted(
        common_label_stems
    ):
        if (
            sha256_file(
                source_labels[
                    stem
                ]
            )
            != sha256_file(
                output_labels[
                    stem
                ]
            )
        ):
            errors.append(
                f"{split}: original label "
                f"{stem!r} was modified."
            )

    return errors


def check_augmented_membership(
    source_root: Path,
    augmented_root: Path,
) -> tuple[
    dict,
    list[str],
]:
    """Verify one augmented image/label pair for every train sample."""

    errors: list[str] = []

    (
        source_images,
        source_labels,
    ) = get_split_indexes(
        source_root,
        "train",
    )

    (
        output_images,
        output_labels,
    ) = get_split_indexes(
        augmented_root,
        "train",
    )

    source_stems = set(
        source_images
    )

    expected_augmented = {
        f"{stem}{AUGMENTED_SUFFIX}"
        for stem in source_stems
    }

    augmented_image_stems = {
        stem
        for stem in output_images
        if stem.endswith(
            AUGMENTED_SUFFIX
        )
    }

    augmented_label_stems = {
        stem
        for stem in output_labels
        if stem.endswith(
            AUGMENTED_SUFFIX
        )
    }

    if (
        augmented_image_stems
        != expected_augmented
    ):
        missing = sorted(
            expected_augmented
            - augmented_image_stems
        )

        extra = sorted(
            augmented_image_stems
            - expected_augmented
        )

        errors.append(
            "Augmented image membership mismatch. "
            f"Missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )

    if (
        augmented_label_stems
        != expected_augmented
    ):
        missing = sorted(
            expected_augmented
            - augmented_label_stems
        )

        extra = sorted(
            augmented_label_stems
            - expected_augmented
        )

        errors.append(
            "Augmented label membership mismatch. "
            f"Missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )

    if (
        set(
            source_labels
        )
        != source_stems
    ):
        errors.append(
            "Source training image/label membership differs."
        )

    return (
        {
            "source_train_images": len(
                source_stems
            ),
            "expected_augmented_images": len(
                expected_augmented
            ),
            "actual_augmented_images": len(
                augmented_image_stems
            ),
            "actual_augmented_labels": len(
                augmented_label_stems
            ),
        },
        errors,
    )


def check_no_augmentation_leakage(
    augmented_root: Path,
) -> list[str]:
    """Ensure augmented files occur only in training."""

    errors: list[str] = []

    for split in (
        "val",
        "test",
    ):
        (
            images,
            labels,
        ) = get_split_indexes(
            augmented_root,
            split,
        )

        leaked_images = [
            stem
            for stem in images
            if stem.endswith(
                AUGMENTED_SUFFIX
            )
        ]

        leaked_labels = [
            stem
            for stem in labels
            if stem.endswith(
                AUGMENTED_SUFFIX
            )
        ]

        if leaked_images:
            errors.append(
                f"{split}: augmented images "
                "were found outside training."
            )

        if leaked_labels:
            errors.append(
                f"{split}: augmented labels "
                "were found outside training."
            )

    return errors


def analyze_augmented_boxes(
    augmented_root: Path,
) -> tuple[
    Counter,
    int,
]:
    """Count augmented class objects and duplicate boxes."""

    (
        _,
        train_labels,
    ) = get_split_indexes(
        augmented_root,
        "train",
    )

    augmented_labels = {
        stem: path
        for stem, path
        in train_labels.items()
        if stem.endswith(
            AUGMENTED_SUFFIX
        )
    }

    duplicate_boxes = 0

    for label_path in (
        augmented_labels.values()
    ):
        duplicate_boxes += (
            count_duplicate_boxes(
                label_path
            )
        )

    class_counts = (
        count_objects_by_class(
            augmented_labels
        )
    )

    return (
        class_counts,
        duplicate_boxes,
    )


def analyze_source_boxes(
    source_root: Path,
) -> Counter:
    """Count source training objects."""

    (
        _,
        labels,
    ) = get_split_indexes(
        source_root,
        "train",
    )

    return count_objects_by_class(
        labels
    )


def calculate_class_drift(
    source_counts: Counter,
    augmented_counts: Counter,
) -> dict:
    """Compare source and augmented class proportions."""

    source_distribution = (
        class_distribution(
            source_counts
        )
    )

    augmented_distribution = (
        class_distribution(
            augmented_counts
        )
    )

    details = {}

    maximum_drift = 0.0

    for class_id, class_name in enumerate(
        CLASSES
    ):
        source_ratio = (
            source_distribution[
                class_id
            ]
        )

        augmented_ratio = (
            augmented_distribution[
                class_id
            ]
        )

        drift = abs(
            source_ratio
            - augmented_ratio
        )

        maximum_drift = max(
            maximum_drift,
            drift,
        )

        details[
            class_name
        ] = {
            "source_objects": (
                source_counts.get(
                    class_id,
                    0,
                )
            ),
            "augmented_objects": (
                augmented_counts.get(
                    class_id,
                    0,
                )
            ),
            "source_ratio": round(
                source_ratio,
                6,
            ),
            "augmented_ratio": round(
                augmented_ratio,
                6,
            ),
            "absolute_drift": round(
                drift,
                6,
            ),
        }

    return {
        "maximum_absolute_drift": round(
            maximum_drift,
            6,
        ),
        "classes": details,
    }


def draw_boxes(
    image_path: Path,
    label_path: Path,
) -> Image.Image:
    """Draw YOLO bounding boxes onto one image."""

    with Image.open(
        image_path
    ) as source:
        image = source.convert(
            "RGB"
        )

    draw = ImageDraw.Draw(
        image
    )

    width, height = (
        image.size
    )

    for (
        class_id,
        x_center,
        y_center,
        box_width,
        box_height,
    ) in read_label_rows(
        label_path
    ):

        x1 = (
            x_center
            - box_width / 2
        ) * width

        y1 = (
            y_center
            - box_height / 2
        ) * height

        x2 = (
            x_center
            + box_width / 2
        ) * width

        y2 = (
            y_center
            + box_height / 2
        ) * height

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline="red",
            width=2,
        )

        draw.text(
            (
                max(
                    0,
                    x1,
                ),
                max(
                    0,
                    y1 - 12,
                ),
            ),
            CLASSES[
                class_id
            ],
            fill="red",
        )

    return image


def create_preview(
    source_root: Path,
    augmented_root: Path,
    output_path: Path,
    sample_count: int = 6,
) -> Path:
    """Create side-by-side original/augmented annotation preview."""

    (
        source_images,
        source_labels,
    ) = get_split_indexes(
        source_root,
        "train",
    )

    (
        augmented_images,
        augmented_labels,
    ) = get_split_indexes(
        augmented_root,
        "train",
    )

    selected_stems = sorted(
        source_images
    )[
        :sample_count
    ]

    rows = []

    for stem in selected_stems:

        augmented_stem = (
            f"{stem}"
            f"{AUGMENTED_SUFFIX}"
        )

        if (
            augmented_stem
            not in augmented_images
            or augmented_stem
            not in augmented_labels
        ):
            continue

        original = draw_boxes(
            source_images[
                stem
            ],
            source_labels[
                stem
            ],
        )

        augmented = draw_boxes(
            augmented_images[
                augmented_stem
            ],
            augmented_labels[
                augmented_stem
            ],
        )

        target_height = max(
            original.height,
            augmented.height,
        )

        canvas = Image.new(
            "RGB",
            (
                original.width
                + augmented.width,
                target_height + 24,
            ),
            "white",
        )

        canvas.paste(
            original,
            (
                0,
                24,
            ),
        )

        canvas.paste(
            augmented,
            (
                original.width,
                24,
            ),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        draw.text(
            (
                5,
                5,
            ),
            f"ORIGINAL: {stem}",
            fill="black",
        )

        draw.text(
            (
                original.width + 5,
                5,
            ),
            "AUGMENTED",
            fill="black",
        )

        rows.append(
            canvas
        )

    if not rows:
        raise ValueError(
            "No image pairs were available "
            "for preview generation."
        )

    preview_width = max(
        row.width
        for row in rows
    )

    preview_height = sum(
        row.height
        for row in rows
    )

    preview = Image.new(
        "RGB",
        (
            preview_width,
            preview_height,
        ),
        "white",
    )

    y_offset = 0

    for row in rows:
        preview.paste(
            row,
            (
                0,
                y_offset,
            ),
        )

        y_offset += (
            row.height
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview.save(
        output_path,
        quality=92,
    )

    return output_path


def validate_augmentation_quality(
    source_root: Path,
    augmented_root: Path,
    max_class_drift: float = (
        DEFAULT_MAX_CLASS_DRIFT
    ),
    min_box_retention: float = (
        DEFAULT_MIN_BOX_RETENTION
    ),
) -> dict:
    """Run the complete augmentation quality audit."""

    source_root = (
        source_root
        .expanduser()
        .resolve()
    )

    augmented_root = (
        augmented_root
        .expanduser()
        .resolve()
    )

    report = {
        "status": "PASSED",
        "source_dataset": str(
            source_root
        ),
        "augmented_dataset": str(
            augmented_root
        ),
        "errors": [],
        "warnings": [],
    }

    if not source_root.is_dir():
        report[
            "status"
        ] = "FAILED"

        report[
            "errors"
        ].append(
            "Source dataset does not exist."
        )

        return report

    if not augmented_root.is_dir():
        report[
            "status"
        ] = "FAILED"

        report[
            "errors"
        ].append(
            "Augmented dataset does not exist."
        )

        return report

    source_preflight = (
        run_preflight(
            source_root
        )
    )

    report[
        "source_preflight"
    ] = (
        source_preflight[
            "status"
        ]
    )

    if (
        source_preflight[
            "status"
        ]
        != "PASSED"
    ):
        report[
            "errors"
        ].append(
            "Source dataset failed preflight."
        )

    augmented_preflight = (
        run_preflight(
            augmented_root
        )
    )

    report[
        "augmented_preflight"
    ] = (
        augmented_preflight[
            "status"
        ]
    )

    if (
        augmented_preflight[
            "status"
        ]
        != "PASSED"
    ):
        report[
            "errors"
        ].append(
            "Augmented dataset failed preflight."
        )

        report[
            "errors"
        ].extend(
            augmented_preflight[
                "errors"
            ]
        )

    for split in (
        "train",
        "val",
        "test",
    ):
        try:
            report[
                "errors"
            ].extend(
                compare_original_split(
                    source_root,
                    augmented_root,
                    split,
                )
            )

        except ValueError as error:
            report[
                "errors"
            ].append(
                str(
                    error
                )
            )

    try:
        (
            membership,
            membership_errors,
        ) = check_augmented_membership(
            source_root,
            augmented_root,
        )

        report[
            "augmentation_membership"
        ] = membership

        report[
            "errors"
        ].extend(
            membership_errors
        )

    except ValueError as error:
        report[
            "errors"
        ].append(
            str(
                error
            )
        )

    try:
        report[
            "errors"
        ].extend(
            check_no_augmentation_leakage(
                augmented_root
            )
        )

    except ValueError as error:
        report[
            "errors"
        ].append(
            str(
                error
            )
        )

    try:
        source_counts = (
            analyze_source_boxes(
                source_root
            )
        )

        (
            augmented_counts,
            duplicate_boxes,
        ) = analyze_augmented_boxes(
            augmented_root
        )

        source_objects = sum(
            source_counts.values()
        )

        augmented_objects = sum(
            augmented_counts.values()
        )

        if source_objects > 0:
            retention_ratio = (
                augmented_objects
                / source_objects
            )
        else:
            retention_ratio = 0.0

        report[
            "bounding_boxes"
        ] = {
            "source_objects": (
                source_objects
            ),
            "augmented_objects": (
                augmented_objects
            ),
            "removed_objects": (
                source_objects
                - augmented_objects
            ),
            "retention_ratio": round(
                retention_ratio,
                6,
            ),
            "duplicate_augmented_boxes": (
                duplicate_boxes
            ),
        }

        if duplicate_boxes > 0:
            report[
                "errors"
            ].append(
                "Duplicate augmented bounding "
                f"boxes detected: {duplicate_boxes}."
            )

        if (
            retention_ratio
            < min_box_retention
        ):
            report[
                "warnings"
            ].append(
                "Bounding-box retention is lower "
                f"than {min_box_retention:.2%}: "
                f"{retention_ratio:.2%}."
            )

        drift = (
            calculate_class_drift(
                source_counts,
                augmented_counts,
            )
        )

        report[
            "class_distribution"
        ] = drift

        if (
            drift[
                "maximum_absolute_drift"
            ]
            > max_class_drift
        ):
            report[
                "warnings"
            ].append(
                "Augmented class-distribution "
                "drift exceeds threshold: "
                f"{drift['maximum_absolute_drift']:.2%} "
                f"> {max_class_drift:.2%}."
            )

        for class_id, class_name in enumerate(
            CLASSES
        ):
            if (
                augmented_counts.get(
                    class_id,
                    0,
                )
                == 0
            ):
                report[
                    "errors"
                ].append(
                    "Augmentation produced no "
                    f"objects for class {class_name!r}."
                )

    except ValueError as error:
        report[
            "errors"
        ].append(
            str(
                error
            )
        )

    if report[
        "errors"
    ]:
        report[
            "status"
        ] = "FAILED"

    return report


def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo"
        ),
    )

    parser.add_argument(
        "--augmented-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo_augmented"
        ),
    )

    parser.add_argument(
        "--max-class-drift",
        type=float,
        default=(
            DEFAULT_MAX_CLASS_DRIFT
        ),
    )

    parser.add_argument(
        "--min-box-retention",
        type=float,
        default=(
            DEFAULT_MIN_BOX_RETENTION
        ),
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    report = (
        validate_augmentation_quality(
            source_root=(
                args.source_root
            ),
            augmented_root=(
                args.augmented_root
            ),
            max_class_drift=(
                args.max_class_drift
            ),
            min_box_retention=(
                args.min_box_retention
            ),
        )
    )

    augmented_root = (
        args.augmented_root
        .expanduser()
        .resolve()
    )

    report_path = (
        augmented_root
        / "augmentation_quality_report.json"
    )

    if augmented_root.is_dir():
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    preview_path = None

    if (
        report[
            "status"
        ]
        == "PASSED"
    ):
        try:
            preview_path = create_preview(
                source_root=(
                    args.source_root
                ),
                augmented_root=(
                    args.augmented_root
                ),
                output_path=(
                    augmented_root
                    / "augmentation_preview.jpg"
                ),
                sample_count=(
                    args.preview_count
                ),
            )

        except (
            OSError,
            ValueError,
        ) as error:
            report[
                "warnings"
            ].append(
                "Preview generation failed: "
                f"{error}"
            )

            report_path.write_text(
                json.dumps(
                    report,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    print(
        "Augmentation quality: "
        f"{report['status']}"
    )

    membership = report.get(
        "augmentation_membership",
        {},
    )

    if membership:
        print(
            "Source training images: "
            f"{membership['source_train_images']}"
        )

        print(
            "Augmented training images: "
            f"{membership['actual_augmented_images']}"
        )

    boxes = report.get(
        "bounding_boxes",
        {},
    )

    if boxes:
        print(
            "Source training objects: "
            f"{boxes['source_objects']}"
        )

        print(
            "Augmented training objects: "
            f"{boxes['augmented_objects']}"
        )

        print(
            "Box retention: "
            f"{boxes['retention_ratio']:.2%}"
        )

        print(
            "Duplicate augmented boxes: "
            f"{boxes['duplicate_augmented_boxes']}"
        )

    drift = report.get(
        "class_distribution",
        {},
    )

    if drift:
        print(
            "Maximum class-distribution drift: "
            f"{drift['maximum_absolute_drift']:.2%}"
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

    if report[
        "errors"
    ]:
        print(
            "\nErrors:"
        )

        for error in report[
            "errors"
        ]:
            print(
                f"- {error}"
            )

    print(
        f"\nReport: {report_path}"
    )

    if preview_path is not None:
        print(
            f"Preview: {preview_path}"
        )

    if (
        report[
            "status"
        ]
        == "FAILED"
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )