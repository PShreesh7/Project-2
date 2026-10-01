"""Create a deterministic YOLO train/validation/test dataset split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from scripts.audit_dataset import CLASSES, IMAGE_SUFFIXES


DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.20


def calculate_split_sizes(
    total: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> tuple[int, int, int]:
    """Calculate exact train, validation, and test counts."""

    if total <= 0:
        raise ValueError(
            "Dataset must contain at least one sample."
        )

    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError(
            "Train and validation ratios must be greater than zero."
        )

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "Train ratio + validation ratio must be less than 1."
        )

    train_count = int(
        total * train_ratio
    )

    val_count = int(
        total * val_ratio
    )

    test_count = (
        total
        - train_count
        - val_count
    )

    if (
        train_count == 0
        or val_count == 0
        or test_count == 0
    ):
        raise ValueError(
            "Dataset is too small for the requested split ratios."
        )

    return (
        train_count,
        val_count,
        test_count,
    )


def index_images(
    data_root: Path,
) -> dict[str, Path]:
    """Create a filename-stem index of source images."""

    image_groups: dict[
        str,
        list[Path],
    ] = {}

    for path in sorted(
        data_root.rglob("*")
    ):
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        ):
            key = path.stem.casefold()

            image_groups.setdefault(
                key,
                [],
            ).append(path)

    image_index: dict[
        str,
        Path,
    ] = {}

    for stem, paths in image_groups.items():
        if len(paths) != 1:
            raise ValueError(
                f"{stem}: expected exactly one image, "
                f"found {len(paths)}"
            )

        image_index[stem] = paths[0]

    if not image_index:
        raise ValueError(
            f"No images found in {data_root}"
        )

    return image_index


def index_labels(
    labels_root: Path,
) -> dict[str, Path]:
    """Create a filename-stem index of YOLO label files."""

    label_index: dict[
        str,
        Path,
    ] = {}

    for path in sorted(
        labels_root.glob("*.txt")
    ):
        key = path.stem.casefold()

        if key in label_index:
            raise ValueError(
                f"Duplicate YOLO label stem: {path.stem}"
            )

        label_index[key] = path

    if not label_index:
        raise ValueError(
            f"No YOLO label files found in {labels_root}"
        )

    return label_index


def validate_yolo_label(
    label_path: Path,
) -> tuple[
    Counter,
    set[int],
]:
    """Validate a YOLO label and return object/class information."""

    object_counts: Counter = Counter()
    present_classes: set[int] = set()

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    if not lines:
        raise ValueError(
            f"{label_path}: label file is empty"
        )

    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "expected 5 YOLO values"
            )

        try:
            class_id = int(
                parts[0]
            )

            coordinates = [
                float(value)
                for value in parts[1:]
            ]

        except ValueError as error:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "invalid numeric value"
            ) from error

        if not (
            0 <= class_id < len(CLASSES)
        ):
            raise ValueError(
                f"{label_path}, line {line_number}: "
                f"invalid class id {class_id}"
            )

        if not all(
            0.0 <= value <= 1.0
            for value in coordinates
        ):
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "YOLO coordinates must be between 0 and 1"
            )

        x_center, y_center, width, height = (
            coordinates
        )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{label_path}, line {line_number}: "
                "box width and height must be positive"
            )

        object_counts[
            class_id
        ] += 1

        present_classes.add(
            class_id
        )

    return (
        object_counts,
        present_classes,
    )


def collect_samples(
    data_root: Path,
    labels_root: Path,
) -> list[dict]:
    """Match source images with their generated YOLO labels."""

    image_index = index_images(
        data_root
    )

    label_index = index_labels(
        labels_root
    )

    image_stems = set(
        image_index
    )

    label_stems = set(
        label_index
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
            "Images without YOLO labels: "
            + ", ".join(
                missing_labels[:10]
            )
        )

    if missing_images:
        raise ValueError(
            "YOLO labels without images: "
            + ", ".join(
                missing_images[:10]
            )
        )

    samples = []

    for stem in sorted(
        image_stems
    ):
        label_path = (
            label_index[stem]
        )

        (
            object_counts,
            present_classes,
        ) = validate_yolo_label(
            label_path
        )

        samples.append(
            {
                "stem": stem,
                "image": image_index[
                    stem
                ],
                "label": label_path,
                "object_counts": object_counts,
                "present_classes": present_classes,
            }
        )

    return samples


def deterministic_split(
    samples: list[dict],
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> dict[
    str,
    list[dict],
]:
    """Shuffle deterministically and create exact dataset splits."""

    samples = list(
        samples
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        samples
    )

    (
        train_count,
        val_count,
        test_count,
    ) = calculate_split_sizes(
        len(samples),
        train_ratio,
        val_ratio,
    )

    train_end = (
        train_count
    )

    val_end = (
        train_count
        + val_count
    )

    splits = {
        "train": samples[
            :train_end
        ],
        "val": samples[
            train_end:val_end
        ],
        "test": samples[
            val_end:
        ],
    }

    assert (
        len(splits["test"])
        == test_count
    )

    return splits


def prepare_output_directories(
    output_root: Path,
) -> None:
    """Create clean YOLO split directories."""

    for folder_name in (
        "images",
        "labels",
    ):
        folder = (
            output_root
            / folder_name
        )

        if folder.exists():
            shutil.rmtree(
                folder
            )

    for split in (
        "train",
        "val",
        "test",
    ):
        (
            output_root
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_root
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def copy_split_files(
    splits: dict[
        str,
        list[dict],
    ],
    output_root: Path,
) -> None:
    """Copy matched image/label pairs into YOLO directories."""

    for split_name, samples in splits.items():

        image_destination = (
            output_root
            / "images"
            / split_name
        )

        label_destination = (
            output_root
            / "labels"
            / split_name
        )

        for sample in samples:

            shutil.copy2(
                sample["image"],
                image_destination
                / sample["image"].name,
            )

            shutil.copy2(
                sample["label"],
                label_destination
                / sample["label"].name,
            )


def create_split_summary(
    splits: dict[
        str,
        list[dict],
    ],
    seed: int,
) -> dict:
    """Build split statistics for review and reproducibility."""

    summary = {
        "seed": seed,
        "total_samples": sum(
            len(samples)
            for samples
            in splits.values()
        ),
        "splits": {},
    }

    for split_name, samples in splits.items():

        object_counts: Counter = Counter()
        image_presence: Counter = Counter()

        for sample in samples:

            object_counts.update(
                sample[
                    "object_counts"
                ]
            )

            for class_id in sample[
                "present_classes"
            ]:
                image_presence[
                    class_id
                ] += 1

        summary[
            "splits"
        ][split_name] = {
            "images": len(
                samples
            ),
            "objects": sum(
                object_counts.values()
            ),
            "images_by_class": {
                CLASSES[class_id]:
                image_presence.get(
                    class_id,
                    0,
                )
                for class_id
                in range(
                    len(CLASSES)
                )
            },
            "objects_by_class": {
                CLASSES[class_id]:
                object_counts.get(
                    class_id,
                    0,
                )
                for class_id
                in range(
                    len(CLASSES)
                )
            },
        }

    return summary


def write_manifest(
    splits: dict[
        str,
        list[dict],
    ],
    output_root: Path,
) -> None:
    """Write a CSV manifest showing which split owns each sample."""

    manifest_path = (
        output_root
        / "split_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "stem",
                "split",
                "image_filename",
                "label_filename",
                "source_image",
            ]
        )

        for split_name in (
            "train",
            "val",
            "test",
        ):
            for sample in sorted(
                splits[
                    split_name
                ],
                key=lambda item:
                item["stem"],
            ):
                writer.writerow(
                    [
                        sample[
                            "stem"
                        ],
                        split_name,
                        sample[
                            "image"
                        ].name,
                        sample[
                            "label"
                        ].name,
                        str(
                            sample[
                                "image"
                            ]
                        ),
                    ]
                )


def create_dataset_split(
    data_root: Path,
    labels_root: Path,
    output_root: Path,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> dict:
    """Create a deterministic YOLO dataset split."""

    data_root = (
        data_root
        .expanduser()
        .resolve()
    )

    labels_root = (
        labels_root
        .expanduser()
        .resolve()
    )

    output_root = (
        output_root
        .expanduser()
        .resolve()
    )

    if not data_root.is_dir():
        raise ValueError(
            f"Dataset directory does not exist: {data_root}"
        )

    if not labels_root.is_dir():
        raise ValueError(
            f"Labels directory does not exist: {labels_root}"
        )

    if (
        output_root == data_root
        or output_root.is_relative_to(
            data_root
        )
    ):
        raise ValueError(
            "Output directory must not be inside "
            "the original dataset."
        )

    samples = collect_samples(
        data_root,
        labels_root,
    )

    splits = deterministic_split(
        samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    prepare_output_directories(
        output_root
    )

    copy_split_files(
        splits,
        output_root,
    )

    write_manifest(
        splits,
        output_root,
    )

    summary = create_split_summary(
        splits,
        seed,
    )

    summary[
        "ratios"
    ] = {
        "train": train_ratio,
        "val": val_ratio,
        "test": (
            1
            - train_ratio
            - val_ratio
        ),
    }

    summary_path = (
        output_root
        / "split_summary.json"
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
            "Root directory of the original NEU-DET dataset."
        ),
    )

    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo_stage/"
            "labels"
        ),
        help=(
            "Directory containing YOLO labels "
            "generated by Commit 3."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo"
        ),
        help=(
            "Destination for train/val/test dataset."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
    )

    args = parser.parse_args()

    try:
        summary = create_dataset_split(
            data_root=args.data_root,
            labels_root=args.labels_root,
            output_root=args.output_root,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    except (
        OSError,
        ValueError,
    ) as error:
        parser.exit(
            1,
            f"Dataset split failed: {error}\n",
        )

    print(
        "NEU-DET dataset split complete"
    )

    print(
        f"Seed: {summary['seed']}"
    )

    print(
        f"Total: {summary['total_samples']}"
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        print(
            f"{split}: "
            f"{summary['splits'][split]['images']} images"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )