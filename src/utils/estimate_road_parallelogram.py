from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def load_class_ids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "class_idx" not in archive.files:
            raise ValueError(f"The 'class_idx' key was not found in {path}.")
        class_ids = np.asarray(archive["class_idx"])

    if class_ids.ndim != 2:
        raise ValueError(f"Expected a 2D class-index array, got {class_ids.shape}.")
    return class_ids


def boundary_points(
    road_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    rows = np.flatnonzero(np.any(road_mask, axis=1))
    if rows.size == 0:
        raise ValueError("The road mask is empty.")

    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    for y in rows:
        columns = np.flatnonzero(road_mask[y])
        left_points.append((float(y), float(columns[0])))
        right_points.append((float(y), float(columns[-1])))

    return (
        np.asarray(left_points, dtype=np.float64),
        np.asarray(right_points, dtype=np.float64),
        int(rows[0]),
        int(rows[-1]),
    )


def fit_line_ransac(
    points: np.ndarray,
    image_width: int,
    residual_threshold: float,
    iterations: int,
    seed: int,
    side: str,
) -> tuple[float, float, int]:
    """Fit the closest one-sided support line using every valid edge point."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")

    usable = points[(points[:, 1] > 0) & (points[:, 1] < image_width - 1)]
    if usable.shape[0] < 2:
        usable = points
    if usable.shape[0] < 2:
        raise ValueError("At least two road-boundary points are required.")

    # Search slopes around the least-squares trend. For each slope, place the
    # line as close as possible while keeping every edge point on the road side.
    y_values = usable[:, 0]
    x_values = usable[:, 1]
    initial_slope = float(np.polyfit(y_values, x_values, deg=1)[0])
    slope_candidates = np.linspace(initial_slope - 1.5, initial_slope + 1.5, 4_001)
    best_model: tuple[float, float, float] | None = None

    for slope in slope_candidates:
        supports = x_values - slope * y_values
        intercept = float(np.min(supports) if side == "left" else np.max(supports))
        clearance = (
            x_values - (slope * y_values + intercept)
            if side == "left"
            else (slope * y_values + intercept) - x_values
        )
        score = float(np.mean(clearance))
        if best_model is None or score < best_model[2]:
            best_model = (float(slope), intercept, score)

    if best_model is None:
        raise ValueError("Could not fit a road boundary.")
    return best_model[0], best_model[1], int(usable.shape[0])


def include_enclosed_obstacles(
    road_mask: np.ndarray,
    left_slope: float,
    left_intercept: float,
    right_slope: float,
    right_intercept: float,
    minimum_area: int = 20,
) -> tuple[float, float, list[dict[str, int]]]:
    """Expand a side only when an enclosed non-road component would be excluded."""
    nonroad = (~road_mask).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        nonroad, connectivity=8
    )
    height, width = road_mask.shape
    adjustments: list[dict[str, int]] = []

    for component_id in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_id]
        if (
            area < minimum_area
            or x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        ):
            continue

        rows, columns = np.where(labels == component_id)
        left_distance = (left_slope * rows + left_intercept) - columns
        right_distance = columns - (right_slope * rows + right_intercept)
        left_shift = float(np.max(left_distance))
        right_shift = float(np.max(right_distance))

        if left_shift > 0 and np.mean(columns) < width * 0.5:
            left_intercept -= left_shift
            adjustments.append(
                {
                    "component_id": int(component_id),
                    "left_shift": int(np.ceil(left_shift)),
                }
            )
        elif right_shift > 0 and np.mean(columns) >= width * 0.5:
            right_intercept += right_shift
            adjustments.append(
                {
                    "component_id": int(component_id),
                    "right_shift": int(np.ceil(right_shift)),
                }
            )

    return left_intercept, right_intercept, adjustments


def estimate_parallelogram(
    road_mask: np.ndarray,
    residual_threshold: float = 8.0,
    iterations: int = 1_000,
    seed: int = 42,
    boundary_quantile: float = 5.0,
    boundary_margin: float = 0.0,
    inclusion_margin: float = 0.0,
    bottom_extension: int = 1_080,
) -> dict[str, object]:
    if not 0 < boundary_quantile < 50:
        raise ValueError("boundary_quantile must be between 0 and 50.")
    if boundary_margin < 0:
        raise ValueError("boundary_margin must be non-negative.")
    if inclusion_margin < 0:
        raise ValueError("inclusion_margin must be non-negative.")
    if bottom_extension < 0:
        raise ValueError("bottom_extension must be non-negative.")

    left_points, right_points, y_top, y_bottom = boundary_points(road_mask)
    image_height, image_width = road_mask.shape
    left_slope, left_intercept, left_inliers = fit_line_ransac(
        left_points, image_width, residual_threshold, iterations, seed, "left"
    )
    right_slope, right_intercept, right_inliers = fit_line_ransac(
        right_points, image_width, residual_threshold, iterations, seed + 1, "right"
    )

    total_margin = boundary_margin + inclusion_margin
    left_intercept -= total_margin
    right_intercept += total_margin
    obstacle_adjustments: list[dict[str, int]] = []
    if inclusion_margin > 0:
        left_intercept, right_intercept, obstacle_adjustments = (
            include_enclosed_obstacles(
                road_mask,
                left_slope,
                left_intercept,
                right_slope,
                right_intercept,
            )
        )

    polygon_bottom = image_height + bottom_extension

    def point(slope: float, intercept: float, y: int, clip_x: bool) -> list[int]:
        x = round(slope * y + intercept)
        if clip_x:
            x = int(np.clip(x, 0, image_width - 1))
        return [x, y]

    polygon = [
        point(left_slope, left_intercept, y_top, clip_x=True),
        point(right_slope, right_intercept, y_top, clip_x=True),
        point(right_slope, right_intercept, polygon_bottom, clip_x=False),
        point(left_slope, left_intercept, polygon_bottom, clip_x=False),
    ]
    return {
        "image_size": [image_width, image_height],
        "road_y_top": y_top,
        "road_y_bottom": y_bottom,
        "polygon_y_bottom": polygon_bottom,
        "bottom_extension_pixels": bottom_extension,
        "left_line": {
            "slope": left_slope,
            "intercept": left_intercept,
            "inliers": left_inliers,
        },
        "right_line": {
            "slope": right_slope,
            "intercept": right_intercept,
            "inliers": right_inliers,
        },
        "boundary_fit": {
            "quantile": boundary_quantile,
            "margin_pixels": total_margin,
            "boundary_margin_pixels": boundary_margin,
            "inclusion_margin_pixels": inclusion_margin,
            "obstacle_adjustments": obstacle_adjustments,
        },
        "polygon": polygon,
    }


def save_visualization(
    image_path: Path,
    road_mask: np.ndarray,
    polygon: list[list[int]],
    output_path: Path,
) -> None:
    image = np.asarray(Image.open(image_path).convert("RGBA"))
    if image.shape[:2] != road_mask.shape:
        raise ValueError(
            f"Image and mask resolutions differ: image={image.shape[1]}x{image.shape[0]}, "
            f"mask={road_mask.shape[1]}x{road_mask.shape[0]}."
        )

    overlay = image.copy()
    overlay[road_mask] = (40, 180, 120, 100)
    result = Image.alpha_composite(Image.fromarray(image), Image.fromarray(overlay))
    polygon_layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(polygon_layer)
    polygon_points = [tuple(vertex) for vertex in polygon]
    draw.polygon(
        polygon_points, fill=(0, 180, 255, 55), outline=(255, 70, 30, 255), width=6
    )
    draw.line(polygon_points + [polygon_points[0]], fill=(255, 255, 255, 255), width=3)
    result = Image.alpha_composite(result, polygon_layer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output_path)


def find_image(image_dir: Path, stem: str) -> Path:
    for extension in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{stem}{extension}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"No RGB image found for {stem} in {image_dir}.")


def process_file(
    input_path: Path,
    image_dir: Path,
    output_dir: Path,
    road_index: int,
    boundary_quantile: float,
    boundary_margin: float,
    inclusion_margin: float,
    bottom_extension: int,
) -> None:
    class_ids = load_class_ids(input_path)
    result = estimate_parallelogram(
        class_ids == road_index,
        boundary_quantile=boundary_quantile,
        boundary_margin=boundary_margin,
        inclusion_margin=inclusion_margin,
        bottom_extension=bottom_extension,
    )
    target_dir = output_dir / input_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "road_parallelogram.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    save_visualization(
        find_image(image_dir, input_path.stem),
        class_ids == road_index,
        result["polygon"],
        target_dir / "road_parallelogram.png",
    )
    print(f"{input_path.name}: polygon={result['polygon']}")


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Estimate road-side lines with RANSAC and draw a road parallelogram."
    )
    parser.add_argument("--input", type=Path, help="One NPZ file to process.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root
        / "data"
        / "datasets"
        / "RoadObstacle21"
        / "Mask2Former_DINOv2_inferences",
        help="Directory containing NPZ files with class IDs.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=project_root / "data" / "datasets" / "RoadObstacle21" / "rgb",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "output")
    parser.add_argument("--road-index", type=int, default=7)
    parser.add_argument(
        "--boundary-quantile",
        type=float,
        default=5.0,
        help="Deprecated compatibility option; the edge fit uses all valid boundary points.",
    )
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.0,
        help="Extra outward margin in pixels after fitting the mask envelope.",
    )
    parser.add_argument(
        "--inclusion-margin",
        type=float,
        default=0.0,
        help="Extra outward margin that allows nearby non-road pixels inside the polygon.",
    )
    parser.add_argument(
        "--bottom-extension",
        type=int,
        default=1_080,
        help="Pixels below the image where the side lines are extended before closing the polygon.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    input_paths = (
        [arguments.input]
        if arguments.input
        else sorted(arguments.input_dir.glob("*.npz"))
    )
    if not input_paths:
        raise ValueError("No NPZ files were found.")
    for input_path in input_paths:
        process_file(
            input_path,
            arguments.image_dir,
            arguments.output_dir,
            arguments.road_index,
            arguments.boundary_quantile,
            arguments.boundary_margin,
            arguments.inclusion_margin,
            arguments.bottom_extension,
        )


if __name__ == "__main__":
    main()
