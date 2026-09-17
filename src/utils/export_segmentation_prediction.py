import argparse
import urllib.request
from pathlib import Path

import mmcv
import numpy as np
import torch
from tqdm import tqdm
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor


from PIL import Image
import dinov2.eval.segmentation_m2f.models

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

IMAGES_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def export_segmentation_prediction(
    model,
    image_path: str,
    output_path: str,
):
    """Run mmseg inference and optionally save auxiliary probabilities."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    image_bgr = image_rgb[:, :, ::-1]

    segmentation_logits = inference_segmentor(model, image_bgr)[0]

    archive = {
        "class_idx": segmentation_logits + 1,  # Convert to 1-based indexing
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)
    return segmentation_logits


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Export segmentation metadata for every image in a directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "output",
        help="Directory where one NPZ file per image will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    input_dir = arguments.input_dir
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    image_paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGES_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in: {input_dir}")

    output_paths = [arguments.output_dir / f"{path.stem}.npz" for path in image_paths]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Input images contain duplicate stems for NPZ output names.")

    cfg_str = load_config_from_url(CONFIG_URL)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    # Change the image scale to 1920x1080 to input the model; default is 2048x512
    # cfg.data.test.pipeline[1].img_scale = (1920, 1080)

    model = init_segmentor(cfg)
    load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
    model.cuda()
    model.eval()

    for image_path, output_path in tqdm(
        zip(image_paths, output_paths),
        total=len(image_paths),
        desc="Exporting predictions",
    ):
        export_segmentation_prediction(
            model=model,
            image_path=str(image_path),
            output_path=str(output_path),
        )


if __name__ == "__main__":
    main()
