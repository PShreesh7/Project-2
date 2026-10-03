"""Create a reproducible training-only augmented NEU-DET YOLO dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image

from scripts.audit_dataset import (
    CLASSES,
    IMAGE_SUFFIXES,
)
from scripts.prepare_yolo_dataset import (
    run_preflight,
    validate_label,
)


DEFAULT_SEED = 42
AUGMENTED_SUFFIX = "__aug"


def index_files(
    directory: Path,
    suffixes: set[str],
) -> dict[str, Path]:
    """Index files using case-insensitive filename stems."""

    if not directory.is_dir():
        raise ValueError(
            f"Directory does not exist: {directory}"
        )

    groups: dict[str, list[Path]] = {}

    for path in sorted(directory.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() in suffixes
        ):
            key = path.stem.casefold()

            groups.setdefault(
                key,
                [],
            ).append(path)

    indexed: dict[str, Path] = {}

    for stem, paths in groups.items():
        if len(paths) != 1:
            raise ValueError(
                f"{directory}: expected one file "
                f"for stem {stem!r}, found {len(paths)}"
            )

        indexed[stem] = paths[0]

    return indexed


def load_yolo_label(
    label_path: Path,
) -> tuple[list[list[float]], list[int]]:
    """Read YOLO boxes and corresponding class IDs."""

    # Reuse Commit 5 validation before parsing.
    validate_label(
        label_path
    )

    bboxes: list[list[float]] = []
    class_labels: list[int] = []

    for raw_line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = raw_line.split()

        class_id = int(
            parts[0]
        )

        bbox = [
            float(value)
            for value in parts[1:]
        ]

        class_labels.append(
            class_id
        )

        bboxes.append(
            bbox
        )

    return (
        bboxes,
        class_labels,
    )


def write_yolo_label(
    label_path: Path,
    bboxes: list,
    class_labels: list,
) -> None:
    """Write transformed bounding boxes in YOLO format."""

    if len(bboxes) != len(
        class_labels
    ):
        raise ValueError(
            "Bounding-box and class-label counts do not match."
        )

    if not bboxes:
        raise ValueError(
            "Augmented sample contains no bounding boxes."
        )

    lines = []

    for class_id, bbox in zip(
        class_labels,
        bboxes,
    ):
        if len(bbox) < 4:
            raise ValueError(
                "Invalid augmented bounding box."
            )

        x_center = float(
            bbox[0]
        )
        y_center = float(
            bbox[1]
        )
        width = float(
            bbox[2]
        )
        height = float(
            bbox[3]
        )

        lines.append(
            f"{int(class_id)} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    # Validate our own generated output.
    validate_label(
        label_path
    )


def build_transform(
    seed: int,
) -> A.Compose:
    """Build one deterministic Albumentations pipeline."""

    return A.Compose(
        [
            # Geometric variation.
            A.Rotate(
                limit=(-12, 12),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),

            # Sensor / acquisition noise.
            A.GaussNoise(
                std_range=(0.01, 0.03),
                mean_range=(0.0, 0.0),
                p=0.5,
            ),

            # Illumination variation.
            A.RandomBrightnessContrast(
                brightness_limit=(-0.15, 0.15),
                contrast_limit=(-0.15, 0.15),
                p=0.7,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=[
                "class_labels"
            ],
            min_visibility=0.20,
            clip=True,
            filter_invalid_bboxes=True,
        ),
        seed=seed,
    )


def load_image(
    image_path: Path,
) -> np.ndarray:
    """Load an image into a NumPy array."""

    try:
        with Image.open(
            image_path
        ) as image:
            image.load()

            if image.mode not in (
                "L",
                "RGB",
            ):
                image = image.convert(
                    "RGB"
                )

            array = np.array(
                image
            )

    except Exception as error:
        raise ValueError(
            f"Unable to read image: {image_path}"
        ) from error

    if array.size == 0:
        raise ValueError(
            f"Empty image: {image_path}"
        )

    return array


def save_image(
    image: np.ndarray,
    destination: Path,
) -> None:
    """Save an augmented image losslessly as PNG."""

    image = np.asarray(
        image
    )

    image = np.clip(
        image,
        0,
        255,
    ).astype(
        np.uint8
    )

    Image.fromarray(
        image
    ).save(
        destination,
        format="PNG",
    )


def validate_source_pairs(
    dataset_root: Path,
    split: str,
) -> list[
    tuple[str, Path, Path]
]:
    """Return matched image/label pairs for one split."""

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

    image_stems = set(
        images
    )

    label_stems = set(
        labels
    )

    if image_stems != label_stems:
        missing_labels = sorted(
            image_stems
            - label_stems
        )

        missing_images = sorted(
            label_stems
            - image_stems
        )

        raise ValueError(
            f"{split}: image/label mismatch. "
            f"Missing labels={missing_labels[:5]}, "
            f"missing images={missing_images[:5]}"
        )

    return [
        (
            stem,
            images[stem],
            labels[stem],
        )
        for stem in sorted(
            image_stems
        )
    ]


def validate_output_location(
    dataset_root: Path,
    output_root: Path,
) -> None:
    """Prevent accidental deletion of the input dataset."""

    if output_root == dataset_root:
        raise ValueError(
            "Output dataset cannot overwrite "
            "the source dataset."
        )

    try:
        output_root.relative_to(
            dataset_root
        )
        raise ValueError(
            "Output dataset cannot be inside "
            "the source dataset."
        )
    except ValueError as error:
        if (
            str(error)
            == "Output dataset cannot be inside "
            "the source dataset."
        ):
            raise

    try:
        dataset_root.relative_to(
            output_root
        )
        raise ValueError(
            "Output dataset cannot contain "
            "the source dataset."
        )
    except ValueError as error:
        if (
            str(error)
            == "Output dataset cannot contain "
            "the source dataset."
        ):
            raise


def prepare_output(
    output_root: Path,
) -> None:
    """Create a fresh generated dataset directory."""

    if output_root.exists():
        shutil.rmtree(
            output_root
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


def copy_original_split(
    dataset_root: Path,
    output_root: Path,
    split: str,
) -> int:
    """Copy an original split without changing its data."""

    pairs = validate_source_pairs(
        dataset_root,
        split,
    )

    for _, image_path, label_path in pairs:

        shutil.copy2(
            image_path,
            output_root
            / "images"
            / split
            / image_path.name,
        )

        shutil.copy2(
            label_path,
            output_root
            / "labels"
            / split
            / label_path.name,
        )

    return len(
        pairs
    )


def augment_one_sample(
    image_path: Path,
    label_path: Path,
    output_image_path: Path,
    output_label_path: Path,
    seed: int,
    max_attempts: int = 5,
) -> tuple[int, int]:
    """Create one augmentation while retaining at least one object."""

    image = load_image(
        image_path
    )

    (
        source_bboxes,
        source_classes,
    ) = load_yolo_label(
        label_path
    )

    source_object_count = len(
        source_bboxes
    )

    for attempt in range(
        max_attempts
    ):
        transform = build_transform(
            seed
            + attempt
        )

        result = transform(
            image=image,
            bboxes=source_bboxes,
            class_labels=source_classes,
        )

        augmented_bboxes = list(
            result["bboxes"]
        )

        augmented_classes = list(
            result[
                "class_labels"
            ]
        )

        if augmented_bboxes:
            save_image(
                result["image"],
                output_image_path,
            )

            write_yolo_label(
                output_label_path,
                augmented_bboxes,
                augmented_classes,
            )

            return (
                source_object_count,
                len(
                    augmented_bboxes
                ),
            )

    raise ValueError(
        "Unable to create a valid augmentation "
        f"for {image_path.name} after "
        f"{max_attempts} attempts."
    )


def create_augmented_dataset(
    dataset_root: Path,
    output_root: Path,
    seed: int = DEFAULT_SEED,
    expected_train: int | None = None,
) -> dict:
    """Build a complete YOLO dataset with augmented training samples."""

    dataset_root = (
        dataset_root
        .expanduser()
        .resolve()
    )

    output_root = (
        output_root
        .expanduser()
        .resolve()
    )

    if not dataset_root.is_dir():
        raise ValueError(
            f"Dataset does not exist: "
            f"{dataset_root}"
        )

    validate_output_location(
        dataset_root,
        output_root,
    )

    # Commit 5 becomes a hard prerequisite.
    source_preflight = run_preflight(
        dataset_root
    )

    if (
        source_preflight[
            "status"
        ]
        != "PASSED"
    ):
        raise ValueError(
            "Source dataset failed Commit 5 "
            "preflight validation."
        )

    train_pairs = (
        validate_source_pairs(
            dataset_root,
            "train",
        )
    )

    if (
        expected_train
        is not None
        and len(
            train_pairs
        )
        != expected_train
    ):
        raise ValueError(
            "Unexpected training image count: "
            f"expected {expected_train}, "
            f"found {len(train_pairs)}."
        )

    prepare_output(
        output_root
    )

    # Preserve all original data.
    original_counts = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        original_counts[
            split
        ] = copy_original_split(
            dataset_root,
            output_root,
            split,
        )

    source_objects = 0
    augmented_objects = 0

    source_class_counts: Counter = (
        Counter()
    )

    augmented_class_counts: Counter = (
        Counter()
    )

    for index, (
        stem,
        image_path,
        label_path,
    ) in enumerate(
        train_pairs
    ):

        (
            _,
            source_classes,
        ) = load_yolo_label(
            label_path
        )

        source_class_counts.update(
            source_classes
        )

        augmented_image_path = (
            output_root
            / "images"
            / "train"
            / f"{stem}{AUGMENTED_SUFFIX}.png"
        )

        augmented_label_path = (
            output_root
            / "labels"
            / "train"
            / f"{stem}{AUGMENTED_SUFFIX}.txt"
        )

        (
            original_box_count,
            augmented_box_count,
        ) = augment_one_sample(
            image_path=image_path,
            label_path=label_path,
            output_image_path=(
                augmented_image_path
            ),
            output_label_path=(
                augmented_label_path
            ),
            seed=(
                seed
                + index * 100
            ),
        )

        source_objects += (
            original_box_count
        )

        augmented_objects += (
            augmented_box_count
        )

        (
            _,
            generated_classes,
        ) = load_yolo_label(
            augmented_label_path
        )

        augmented_class_counts.update(
            generated_classes
        )

    summary = {
        "seed": seed,
        "source_dataset": str(
            dataset_root
        ),
        "output_dataset": str(
            output_root
        ),
        "augmentation_policy": {
            "rotation_degrees": [
                -12,
                12,
            ],
            "rotation_probability": 1.0,
            "gaussian_noise_std_range": [
                0.01,
                0.03,
            ],
            "gaussian_noise_probability": 0.5,
            "brightness_limit": [
                -0.15,
                0.15,
            ],
            "contrast_limit": [
                -0.15,
                0.15,
            ],
            "lighting_probability": 0.7,
            "minimum_bbox_visibility": 0.20,
        },
        "original_images": {
            "train": original_counts[
                "train"
            ],
            "val": original_counts[
                "val"
            ],
            "test": original_counts[
                "test"
            ],
        },
        "augmented_train_images": len(
            train_pairs
        ),
        "final_train_images": (
            original_counts[
                "train"
            ]
            + len(
                train_pairs
            )
        ),
        "source_train_objects": (
            source_objects
        ),
        "augmented_train_objects": (
            augmented_objects
        ),
        "boxes_removed_during_augmentation": (
            source_objects
            - augmented_objects
        ),
        "source_objects_by_class": {
            CLASSES[class_id]:
            source_class_counts.get(
                class_id,
                0,
            )
            for class_id in range(
                len(CLASSES)
            )
        },
        "augmented_objects_by_class": {
            CLASSES[class_id]:
            augmented_class_counts.get(
                class_id,
                0,
            )
            for class_id in range(
                len(CLASSES)
            )
        },
    }

    summary_path = (
        output_root
        / "augmentation_summary.json"
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
        "--dataset-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/processed/"
            "neu_det_yolo_augmented"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--expected-train",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    try:
        summary = (
            create_augmented_dataset(
                dataset_root=(
                    args.dataset_root
                ),
                output_root=(
                    args.output_root
                ),
                seed=args.seed,
                expected_train=(
                    args.expected_train
                ),
            )
        )

    except (
        OSError,
        ValueError,
    ) as error:
        parser.exit(
            1,
            f"Augmentation failed: "
            f"{error}\n",
        )

    print(
        "NEU-DET augmentation complete"
    )

    print(
        "Original training images: "
        f"{summary['original_images']['train']}"
    )

    print(
        "Augmented training images: "
        f"{summary['augmented_train_images']}"
    )

    print(
        "Final training images: "
        f"{summary['final_train_images']}"
    )

    print(
        "Validation images: "
        f"{summary['original_images']['val']}"
    )

    print(
        "Test images: "
        f"{summary['original_images']['test']}"
    )

    print(
        "Original training objects: "
        f"{summary['source_train_objects']}"
    )

    print(
        "Augmented training objects: "
        f"{summary['augmented_train_objects']}"
    )

    print(
        "Boxes removed after transforms: "
        f"{summary['boxes_removed_during_augmentation']}"
    )

    print(
        "Summary: "
        f"{Path(summary['output_dataset']) / 'augmentation_summary.json'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )