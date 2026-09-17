from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


@dataclass(frozen=True)
class SemanticClass:
    index: int
    name: str
    color: tuple[int, int, int]


def load_classes(path: Path) -> list[SemanticClass]:
    classes: list[SemanticClass] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        columns = [column.strip() for column in line.split("|")]
        if len(columns) < 10 or not columns[1].isdigit():
            continue

        color_values = columns[6].strip("()").split(",")
        if len(color_values) != 3 or not all(value.strip().isdigit() for value in color_values):
            continue

        classes.append(
            SemanticClass(
                index=int(columns[1]),
                name=columns[8].split(";")[0],
                color=tuple(int(value.strip()) for value in color_values),
            )
        )

    if not classes:
        raise ValueError(f"Nenhuma classe foi lida de {path}.")
    return classes


def load_class_ids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "class_idx" not in archive.files:
            raise ValueError(f"The 'class_idx' key was not found in {path}.")
        class_ids = np.asarray(archive["class_idx"])

    if class_ids.ndim != 2:
        raise ValueError(
            f"The 'class_ids' key must contain a 2D array; received shape: {class_ids.shape}."
        )
    return class_ids


def class_ids_to_semantic_image(
    class_ids: np.ndarray, classes: list[SemanticClass]
) -> Image.Image:
    semantic_image = np.zeros((*class_ids.shape, 3), dtype=np.uint8)
    for semantic_class in classes:
        semantic_image[class_ids == semantic_class.index] = semantic_class.color
    return Image.fromarray(semantic_image, mode="RGB")


def save_labeled_semantic_segmentation(
    image: Image.Image,
    class_ids: np.ndarray,
    classes: list[SemanticClass],
    output: Path,
) -> None:
    semantic_image = class_ids_to_semantic_image(class_ids, classes)
    labeled_image = Image.blend(image.convert("RGB"), semantic_image, alpha=0.5)
    label_layer = Image.new("RGBA", labeled_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(label_layer)
    font = ImageFont.load_default(size=22)

    for semantic_class in classes:
        rows, columns = np.where(class_ids == semantic_class.index)
        if not rows.size:
            continue

        label = semantic_class.name
        label_x = int(np.median(columns))
        label_y = int(np.median(rows))
        left, top, right, bottom = draw.textbbox(
            (label_x, label_y), label, font=font
        )
        draw.rectangle(
            (left - 8, top - 4, right + 8, bottom + 4), fill=(0, 0, 0, 180)
        )
        draw.text(
            (label_x, label_y), label, font=font, fill=(255, 255, 255, 255)
        )

    labeled_image = Image.alpha_composite(labeled_image.convert("RGBA"), label_layer)
    labeled_image.convert("RGB").save(output)


def keep_main_road(road_mask: np.ndarray) -> np.ndarray:
    """Remove thin lines while preserving disconnected, wider road regions."""
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_opening(road_mask, structure=structure)


def create_road_obstacle_mask(
    road_mask: np.ndarray, obstacle_mask: np.ndarray
) -> Image.Image:
    combined_mask = np.zeros((*road_mask.shape, 3), dtype=np.uint8)
    combined_mask[road_mask] = (255, 255, 255)
    combined_mask[obstacle_mask] = (255, 102, 0)
    return Image.fromarray(combined_mask, mode="RGB")


def extract_obstacles(
    class_ids: np.ndarray, road_index: int, min_area: int, max_area: int
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    road_mask = keep_main_road(class_ids == road_index)
    filled_road = ndimage.binary_fill_holes(road_mask)
    object_mask = filled_road & ~road_mask
    labeled_objects, number_of_objects = ndimage.label(object_mask)
    obstacle_mask = np.zeros_like(object_mask, dtype=bool)
    kept_components: list[tuple[int, np.ndarray]] = []

    for component_id in range(1, number_of_objects + 1):
        component = labeled_objects == component_id
        area = int(component.sum())
        if min_area <= area <= max_area:
            obstacle_mask |= component
            kept_components.append((component_id, component))

    return obstacle_mask, kept_components


def describe_components(
    components: list[tuple[int, np.ndarray]],
    class_ids: np.ndarray,
    class_by_index: dict[int, SemanticClass],
) -> list[dict[str, int | str | list[int]]]:
    descriptions: list[dict[str, int | str | list[int]]] = []

    for component_id, component in components:
        component_class_ids = class_ids[component]
        valid_class_ids = component_class_ids[component_class_ids >= 0]
        if valid_class_ids.size:
            values, counts = np.unique(valid_class_ids, return_counts=True)
            dominant_class_id = int(values[np.argmax(counts)])
            class_name = class_by_index[dominant_class_id].name
        else:
            dominant_class_id = -1
            class_name = "unknown"

        descriptions.append(
            {
                "component_id": component_id,
                "class_id": dominant_class_id,
                "class_name": class_name,
                "area_pixels": int(component.sum()),
            }
        )
    return descriptions


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Extracts objects inside the region segmented as road."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "datasets" / "RoadObstacle21" / "Mask2Former_DINOv2_inferences" / "validation_3.npz",
        help="NPZ file containing the class IDs under the 'class_ids' key.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=project_root / "docs" / "ade20k_classes.md",
        help="ADE20K class table.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=project_root / "data" / "validation_3.jpg",
        help="Reference image used to generate full-resolution visual outputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "output")
    parser.add_argument("--road-index", type=int, default=7)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--max-area", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.min_area > arguments.max_area:
        raise ValueError("--min-area must be less than or equal to --max-area.")

    classes = load_classes(arguments.classes)
    class_by_index = {
        semantic_class.index: semantic_class for semantic_class in classes
    }
    if arguments.road_index not in class_by_index:
        raise ValueError(f"Road class {arguments.road_index} was not found.")

    class_ids = load_class_ids(arguments.input)
    road_mask = keep_main_road(class_ids == arguments.road_index)
    road_pixels = int(np.count_nonzero(road_mask))
    if road_pixels == 0:
        raise ValueError("No road pixels were found in the class ID matrix.")

    obstacle_mask, components = extract_obstacles(
        class_ids, arguments.road_index, arguments.min_area, arguments.max_area
    )
    obstacles = describe_components(components, class_ids, class_by_index)
    reference_image = Image.open(arguments.image).convert("RGB")
    if reference_image.size != (class_ids.shape[1], class_ids.shape[0]):
        raise ValueError(
            "The class ID matrix and reference image must have the same resolution: "
            f"class_ids={class_ids.shape[1]}x{class_ids.shape[0]}, "
            f"image={reference_image.width}x{reference_image.height}."
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    create_road_obstacle_mask(road_mask, obstacle_mask).save(
        arguments.output_dir / "road_obstacle_mask.png"
    )
    save_labeled_semantic_segmentation(
        reference_image,
        class_ids,
        classes,
        arguments.output_dir / "semantic_segmentation_labeled.png",
    )
    report = {
        "input": str(arguments.input),
        "road_class": class_by_index[arguments.road_index].name,
        "road_pixels": road_pixels,
        "obstacle_count": len(obstacles),
        "obstacles": obstacles,
    }
    (arguments.output_dir / "obstacles.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Road: {road_pixels} pixels")
    print(f"Detected obstacles: {len(obstacles)}")
    for obstacle in obstacles:
        print(
            f"- {obstacle['class_name']} (id={obstacle['class_id']}, "
            f"area={obstacle['area_pixels']} px)"
        )


if __name__ == "__main__":
    main()
