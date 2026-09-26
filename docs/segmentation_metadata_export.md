# Segmentation Metadata Export Plan

1. **Semantic segmentation inference**

- Run the ADE20K model as usual -> `150 logits/classes`.
- Apply `softmax` to obtain the probability distribution for the 150 classes at each pixel.

2. **Calculate per-pixel metadata**
```
P_road          -> (H, W) float16 -> probability of the class at index `7`.
P_non_drivable  -> (H, W) float16 -> 1 - P_road
entropy         -> (H, W) float16 -> entropy of the 150-class probability distribution normalized to [0, 1].
class_idx       -> (H, W) int16   -> index of the class with the highest probability.
confidence      -> (H, W) float16 -> probability of the most likely class.
```

```
np.savez_compressed(
    "segmentation_metadata.npz",
    P_road=P_road.astype(np.float16),
    P_non_drivable=P_non_drivable.astype(np.float16),
    entropy=entropy.astype(np.float16),
    class_idx=class_idx.astype(np.uint8),
    confidence=confidence.astype(np.float16),
)
```

## Segmentation script CLI

```text
--input-dir PATH      Directory containing NPZ files with the class_idx array
--classes PATH        ADE20K class table
--image-dir PATH      Directory containing matching RGB images
--output-dir PATH     Directory for generated outputs, grouped by input stem
```