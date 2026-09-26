from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories
INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "RoadObstacle21"
    / "Mask2Former_DINOv2_inferences"
)
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGE_DIR = PROJECT_ROOT / "data" / "datasets" / "RoadObstacle21" / "rgb"
IMAGE_EXTENSION = ".jpg"

# Segmentation
ROAD_INDEX = 7
INTERSECTION_THRESHOLD = 1.0
ROAD_MASK_COLOR = (128, 64, 128)
INTERSECTING_MASK_COLOR = (0, 0, 142)

# Polynomial boundary fitting
POLYNOMIAL_DEGREE = 9
RESIDUAL_THRESHOLD = 20.0
FIT_ITERATIONS = 5
