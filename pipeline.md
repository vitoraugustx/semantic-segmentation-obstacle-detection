# Semantic Segmentation Obstacle Detection Pipeline

This document describes the obstacle detection algorithm implemented in
`src/semantic_obstacle_detection.py`.

## Input

The ADE20K inference step is performed before this pipeline. The program reads
an `.npz` file containing a `class_ids` array:

- shape: `H x W`;
- one integer class ID per pixel;
- road class ID: `7` by default;
- reference image: used to generate the overlay and must have the same
  resolution as `class_ids`.

The default input file is `data/segmentation_prediction.npz`.

## Algorithm

### 1. Load the semantic prediction

`class_ids` is an `H x W` matrix containing the predicted semantic class for
each pixel. The program also loads the ADE20K class table to map class IDs to
class names and colors.

### 2. Create and clean the road mask

The initial road mask selects pixels whose class ID matches the configured road
class:

```python
road_mask = class_ids == road_index
```

Before detecting holes, the mask is processed with a `3 x 3` binary opening:

```python
road_mask = ndimage.binary_opening(road_mask, structure=np.ones((3, 3)))
```

This removes thin isolated lines while preserving wider road regions,
including disconnected regions.

### 3. Fill holes in the road

Binary hole filling identifies enclosed regions surrounded by the cleaned road
mask:

```python
filled_road = ndimage.binary_fill_holes(road_mask)
```

### 4. Find candidate obstacles

Pixels inside the filled road that are not part of the original road mask are
treated as candidate obstacles:

```python
object_mask = filled_road & ~road_mask
```

This approach detects regions classified as something other than road when
they are enclosed by the road region. It does not independently verify that a
region is a physical obstacle.

### 5. Label connected components

The candidate mask is split into connected components with
`scipy.ndimage.label`. Each component is measured in pixels.

### 6. Filter components by area

Only components whose area is within the inclusive range are retained:

```python
min_area <= area <= max_area
```

The defaults are:

- `min_area = 200`: removes small regions treated as noise;
- `max_area = 50_000`: removes large regions unlikely to be the small
  obstacles of interest.

These values can be changed with `--min-area` and `--max-area`.

### 7. Assign a class to each obstacle

For every retained component, the program reads the original class IDs at the
component pixels. The dominant class is the class ID that occurs most often.
Pixels with negative IDs are ignored. If no valid class ID remains, the class
is reported as `unknown` with ID `-1`.

The report also includes the component area.

## Pipeline flow

```text
Segmentation inference
        |
        v
class_ids (H x W) + reference image
        |
        v
Road class mask (road_index = 7)
        |
        v
3 x 3 binary opening
        |
        v
Binary hole filling
        |
        v
Candidate regions inside the road
        |
        v
Connected-component labeling
        |
        v
Inclusive area filter
        |
        v
Dominant class + obstacle mask
```

## Outputs

The output directory is `output/` by default. The program writes:

- `road_obstacle_mask.png`: RGB mask where the road is `(255, 255, 255)` and
        retained obstacles are `(255, 102, 0)`; pixels outside these regions are
        black;
- `semantic_segmentation.png`: colorized semantic class map;
- `semantic_segmentation_labeled.png`: original image blended with the class map
        and annotated with the ADE20K class names;
- `obstacles.json`: component metadata and summary counts.

## Class masks inside the road parallelogram

The utility `src/utils/estimate_road_parallelogram.py` also evaluates every
non-road class against the estimated parallelogram. A class is selected when
the fraction of its pixels that falls inside the parallelogram is greater than
or equal to `--intersection-threshold` (default `0.1`, or 10%). The complete
mask of each selected class is then shown in orange in
`road_parallelogram.png`, and the selected IDs are stored under
`intersection_filter.selected_class_ids` in `road_parallelogram.json`.

For example:

```text
--intersection-threshold 0.1
```

## Command-line options

```text
--input PATH          NPZ file containing the class_ids array
--classes PATH        ADE20K class table
--image PATH          Reference image for visual outputs
--output-dir PATH     Directory for generated outputs
--road-index INTEGER  Road class ID (default: 7)
--min-area INTEGER    Minimum component area (default: 200)
--max-area INTEGER    Maximum component area (default: 50000)
--intersection-threshold FLOAT
                      Minimum fraction of a class mask inside the road
                      parallelogram (default: 0.1)
```