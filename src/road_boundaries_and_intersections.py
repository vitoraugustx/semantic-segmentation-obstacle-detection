from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from .utils import config
except ImportError:
    from utils import config


@dataclass(frozen=True)
class PolynomialFit:
    coefficients: np.ndarray
    y_min: float
    y_max: float
    inliers: int

    def x_at(self, y: np.ndarray | float) -> np.ndarray | float:
        normalized_y = (
            2.0 * (np.asarray(y) - self.y_min) / (self.y_max - self.y_min) - 1.0
        )
        values = np.polynomial.polynomial.polyval(normalized_y, self.coefficients)
        return float(values) if np.ndim(values) == 0 else values


def load_class_ids(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "class_idx" not in archive.files:
            raise ValueError(f"The 'class_idx' key was not found in {path}.")
        class_ids = np.asarray(archive["class_idx"])

    if class_ids.ndim != 2:
        raise ValueError(
            f"Expected a 2D class index array, received {class_ids.shape}."
        )
    return class_ids


def load_road_mask(path: Path, road_index: int) -> np.ndarray:
    road_mask = load_class_ids(path) == road_index
    if not np.any(road_mask):
        raise ValueError(f"No pixels for road class {road_index} were found in {path}.")
    return road_mask


def select_intersecting_classes(
    class_ids: np.ndarray,
    region_mask: np.ndarray,
    road_index: int,
    threshold: float,
) -> tuple[dict[int, np.ndarray], list[dict[str, float | int]]]:
    selected_masks: dict[int, np.ndarray] = {}
    statistics: list[dict[str, float | int]] = []
    for class_id in np.unique(class_ids):
        class_id = int(class_id)
        if class_id == road_index:
            continue
        class_mask = class_ids == class_id
        class_pixels = int(class_mask.sum())
        intersection_pixels = int(np.count_nonzero(class_mask & region_mask))
        intersection_percent = 100.0 * intersection_pixels / class_pixels
        if intersection_pixels and intersection_percent >= threshold:
            selected_masks[class_id] = class_mask & region_mask
            statistics.append(
                {
                    "class_id": class_id,
                    "class_pixels": class_pixels,
                    "intersection_pixels": intersection_pixels,
                    "intersection_percent": intersection_percent,
                }
            )
    return selected_masks, statistics


def save_segmentation_mask(
    road_mask: np.ndarray,
    selected_masks: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    pixels = np.zeros((*road_mask.shape, 3), dtype=np.uint8)
    pixels[road_mask] = config.ROAD_MASK_COLOR
    for class_mask in selected_masks.values():
        pixels[class_mask] = config.INTERSECTING_MASK_COLOR
    Image.fromarray(pixels, mode="RGB").save(output_path)


def extract_boundaries(
    road_mask: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    rows = np.flatnonzero(np.any(road_mask, axis=1))
    road_y_top, road_y_bottom = int(rows[0]), int(rows[-1])
    fit_rows = rows
    left_x = np.array([np.flatnonzero(road_mask[y])[0] for y in fit_rows], dtype=float)
    right_x = np.array(
        [np.flatnonzero(road_mask[y])[-1] for y in fit_rows], dtype=float
    )
    return (
        road_y_top,
        road_y_bottom,
        np.column_stack((fit_rows, left_x)),
        np.column_stack((fit_rows, right_x)),
    )


def fit_polynomial(
    points: np.ndarray,
    degree: int,
    residual_threshold: float,
    iterations: int,
) -> PolynomialFit:
    if len(points) <= degree:
        raise ValueError(f"At least {degree + 1} boundary points are required.")

    y_values = points[:, 0].astype(float)
    x_values = points[:, 1].astype(float)
    y_min, y_max = float(y_values.min()), float(y_values.max())
    normalized_y = 2.0 * (y_values - y_min) / (y_max - y_min) - 1.0
    inliers = np.ones(len(points), dtype=bool)

    for _ in range(iterations):
        coefficients = np.polynomial.polynomial.polyfit(
            normalized_y[inliers], x_values[inliers], deg=degree
        )
        residuals = np.abs(
            x_values - np.polynomial.polynomial.polyval(normalized_y, coefficients)
        )
        new_inliers = residuals <= residual_threshold
        if np.array_equal(new_inliers, inliers):
            break
        if new_inliers.sum() <= degree:
            break
        inliers = new_inliers

    coefficients = np.polynomial.polynomial.polyfit(
        normalized_y[inliers], x_values[inliers], deg=degree
    )
    return PolynomialFit(coefficients, y_min, y_max, int(inliers.sum()))


def make_curved_region(
    road_mask: np.ndarray,
    rows: np.ndarray,
    left_curve: PolynomialFit,
    right_curve: PolynomialFit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    _, width = road_mask.shape
    fitted_left = np.asarray(left_curve.x_at(rows), dtype=float)
    fitted_right = np.asarray(right_curve.x_at(rows), dtype=float)
    crossing_count = int(np.count_nonzero(fitted_left >= fitted_right))
    left_boundary = np.minimum(fitted_left, fitted_right)
    right_boundary = np.maximum(fitted_left, fitted_right)
    region_mask = np.zeros_like(road_mask, dtype=bool)
    for row, left, right in zip(rows, left_boundary, right_boundary):
        start = max(0, round(left))
        end = min(width - 1, round(right))
        if start <= end:
            region_mask[row, start : end + 1] = True
    return region_mask, left_boundary, right_boundary, crossing_count


def save_overlay(
    image_path: Path,
    road_mask: np.ndarray,
    region_mask: np.ndarray,
    rows: np.ndarray,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    if image.size != (road_mask.shape[1], road_mask.shape[0]):
        raise ValueError(
            f"Image and NPZ have different sizes: {image.size} vs {road_mask.shape[::-1]}."
        )

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_pixels = np.asarray(overlay).copy()
    overlay_pixels[region_mask] = (255, 165, 0, 80)
    overlay = Image.fromarray(overlay_pixels, mode="RGBA")
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)
    draw.line(
        list(zip(left_boundary.tolist(), rows.tolist())),
        fill=(255, 80, 0, 255),
        width=4,
    )
    draw.line(
        list(zip(right_boundary.tolist(), rows.tolist())),
        fill=(255, 80, 0, 255),
        width=4,
    )
    image.save(output_path)


def process_file(
    input_path: Path, output_dir: Path, arguments: argparse.Namespace
) -> None:
    class_ids = load_class_ids(input_path)
    road_mask = class_ids == arguments.road_index
    if not np.any(road_mask):
        raise ValueError(
            f"No pixels for road class {arguments.road_index} were found in {input_path}."
        )
    road_y_top, road_y_bottom, left_points, right_points = extract_boundaries(road_mask)
    left_curve = fit_polynomial(
        left_points,
        arguments.degree,
        arguments.residual_threshold,
        arguments.fit_iterations,
    )
    right_curve = fit_polynomial(
        right_points,
        arguments.degree,
        arguments.residual_threshold,
        arguments.fit_iterations,
    )
    region_mask, left_boundary, right_boundary, crossing_count = make_curved_region(
        road_mask, left_points[:, 0].astype(int), left_curve, right_curve
    )
    selected_masks, intersection_statistics = select_intersecting_classes(
        class_ids,
        region_mask,
        arguments.road_index,
        arguments.intersection_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / input_path.stem
    save_segmentation_mask(
        road_mask,
        selected_masks,
        output_stem.with_name(f"{output_stem.name}_road_segmentation_mask.png"),
    )
    report = {
        "input": str(input_path),
        "image_size": [road_mask.shape[1], road_mask.shape[0]],
        "road_y_top": road_y_top,
        "road_y_bottom": road_y_bottom,
        "method": "robust_polynomial_boundary",
        "intersection_threshold_percent": arguments.intersection_threshold,
        "intersecting_classes": intersection_statistics,
        "road_mask_color": config.ROAD_MASK_COLOR,
        "intersecting_mask_color": config.INTERSECTING_MASK_COLOR,
        "polynomial_degree": arguments.degree,
        "left_curve": left_curve.coefficients.tolist(),
        "right_curve": right_curve.coefficients.tolist(),
        "left_inliers": left_curve.inliers,
        "right_inliers": right_curve.inliers,
        "curve_crossings_before_ordering": crossing_count,
    }
    output_stem.with_name(f"{output_stem.name}_road_curve.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    if arguments.image_dir is not None:
        image_path = (
            arguments.image_dir / f"{input_path.stem}{arguments.image_extension}"
        )
        if not image_path.is_file():
            raise FileNotFoundError(f"RGB image not found: {image_path}")
        save_overlay(
            image_path,
            road_mask,
            region_mask,
            left_points[:, 0].astype(int),
            left_boundary,
            right_boundary,
            output_stem.with_name(f"{output_stem.name}_road_curve_overlay.png"),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate curved road boundaries from semantic segmentation masks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="One NPZ file. If omitted, process all NPZ files in --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=config.INPUT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=config.IMAGE_DIR,
        help="RGB directory used to generate overlay images.",
    )
    parser.add_argument("--image-extension", default=config.IMAGE_EXTENSION)
    parser.add_argument("--road-index", type=int, default=config.ROAD_INDEX)
    parser.add_argument(
        "--intersection-threshold",
        type=float,
        default=config.INTERSECTION_THRESHOLD,
        help="Minimum percentage of a class mask that must intersect the road region.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=config.POLYNOMIAL_DEGREE,
        help="Polynomial degree for each road boundary.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=config.RESIDUAL_THRESHOLD,
        help="Boundary residual threshold in pixels.",
    )
    parser.add_argument(
        "--fit-iterations",
        type=int,
        default=config.FIT_ITERATIONS,
        help="Number of robust outlier rejection iterations.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    input_paths = (
        [arguments.input]
        if arguments.input is not None
        else sorted(arguments.input_dir.glob("*.npz"))
    )
    if not input_paths:
        raise ValueError("No NPZ files were found.")
    for input_path in input_paths:
        process_file(input_path, arguments.output_dir, arguments)
        print(f"Processed {input_path.name}")


if __name__ == "__main__":
    main()
