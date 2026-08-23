from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


@dataclass(frozen=True)
class SemanticClass:
    index: int
    name: str
    color: tuple[int, int, int]


def load_classes(path: Path) -> list[SemanticClass]:
    classes: list[SemanticClass] = []
    color_pattern = re.compile(r"\((\d+),\s*(\d+),\s*(\d+)\)")

    for line in path.read_text(encoding="utf-8").splitlines():
        columns = [column.strip() for column in line.split("|")]
        if len(columns) < 10 or not columns[1].isdigit():
            continue

        color_match = color_pattern.fullmatch(columns[6])
        if color_match is None:
            continue

        classes.append(
            SemanticClass(
                index=int(columns[1]),
                name=columns[8].split(";")[0],
                color=tuple(map(int, color_match.groups())),
            )
        )

    if not classes:
        raise ValueError(f"Nenhuma classe foi lida de {path}.")
    return classes


def rgb_to_class_ids(
    image: np.ndarray, classes: list[SemanticClass], color_tolerance: float
) -> np.ndarray:
    colors = np.array(
        [semantic_class.color for semantic_class in classes], dtype=np.int16
    )
    class_indices = np.array(
        [semantic_class.index for semantic_class in classes], dtype=np.int32
    )
    flat_image = image.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat_image, axis=0, return_inverse=True)
    distances = np.sum(
        (
            unique_colors[:, np.newaxis, :].astype(np.int32)
            - colors[np.newaxis, :, :].astype(np.int32)
        )
        ** 2,
        axis=2,
        dtype=np.int32,
    )
    nearest_positions = np.argmin(distances, axis=1)
    nearest_distances = np.sqrt(
        distances[np.arange(len(unique_colors)), nearest_positions]
    )
    unique_class_ids = class_indices[nearest_positions]
    unique_class_ids[nearest_distances > color_tolerance] = -1
    return unique_class_ids[inverse].reshape(image.shape[:2])


def keep_main_road(road_mask: np.ndarray) -> np.ndarray:
    """Remove thin lines while preserving disconnected, wider road regions."""
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_opening(road_mask, structure=structure)


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
        rows, columns = np.where(component)
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
                "bbox_xyxy": [
                    int(columns.min()),
                    int(rows.min()),
                    int(columns.max()),
                    int(rows.max()),
                ],
            }
        )
    return descriptions


def save_overlay(
    image: Image.Image, obstacles: list[dict[str, int | str | list[int]]], output: Path
) -> None:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    for obstacle in obstacles:
        left, top, right, bottom = obstacle["bbox_xyxy"]
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0, 255), width=4)
        draw.text(
            (left, max(0, top - 16)), str(obstacle["class_name"]), fill=(255, 0, 0, 255)
        )
    overlay.convert("RGB").save(output)


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Extracts objects inside the region segmented as road."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "05_Schafgasse.jpg",
        help="RGB image with the color-coded semantic mask.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=project_root / "src" / "classes.md",
        help="ADE20K class table.",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "output")
    parser.add_argument("--road-index", type=int, default=7)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--max-area", type=int, default=50_000)
    parser.add_argument("--color-tolerance", type=float, default=40.0)
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

    input_image = Image.open(arguments.input).convert("RGB")
    image_array = np.asarray(input_image)
    class_ids = rgb_to_class_ids(image_array, classes, arguments.color_tolerance)
    road_mask = keep_main_road(class_ids == arguments.road_index)
    road_pixels = int(np.count_nonzero(road_mask))
    if road_pixels == 0:
        raise ValueError("No road pixels were found in the image.")

    obstacle_mask, components = extract_obstacles(
        class_ids, arguments.road_index, arguments.min_area, arguments.max_area
    )
    obstacles = describe_components(components, class_ids, class_by_index)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(road_mask.astype(np.uint8) * 255, mode="L").save(
        arguments.output_dir / "road_mask.png"
    )
    Image.fromarray(obstacle_mask.astype(np.uint8) * 255, mode="L").save(
        arguments.output_dir / "obstacle_mask.png"
    )
    save_overlay(input_image, obstacles, arguments.output_dir / "obstacles_overlay.jpg")
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
