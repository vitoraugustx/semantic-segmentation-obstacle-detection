# Implementation plan

## 1. Obtain the semantic prediction

`pred` is an `H × W` matrix containing the class ID for each pixel.

## 2. Create the road mask

```python
road_mask = pred == ROAD_ID
```

## 3. Fill road holes

```python
filled_road = binary_fill_holes(road_mask)
```

## 4. Extract objects inside the road

```python
object_mask = filled_road & ~road_mask
```

This operation obtains pixels classified as something else but located inside
the road region.

## 5. Filter by size

- Remove very small regions, which are considered noise.
- Remove very large regions, which probably do not correspond to the small
  obstacles of interest.

## 6. Retrieve the object class

```python
object_labels = pred[object_mask]
```

## Pipeline flow

```text
Image
  ↓
ADE20K
  ↓
pred (H × W)
  ↓
road_mask
  ↓
Hole filling
  ↓
Objects inside the road
  ↓
Size filter
  ↓
Class + object mask
```