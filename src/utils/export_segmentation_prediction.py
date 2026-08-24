import urllib.request
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose, LoadImage
from PIL import Image
import dinov2.eval.segmentation_m2f.models

DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def export_segmentation_prediction(
    model,
    cfg,
    image_path: str,
    output_path: str,
    include_probabilities: bool = False,
):
    """Run mmseg inference and optionally save auxiliary probabilities."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
    image_bgr = image_rgb[:, :, ::-1]

    test_pipeline = Compose([LoadImage()] + cfg.data.test.pipeline[1:])
    data = test_pipeline(dict(img=image_bgr))
    data = collate([data], samples_per_gpu=1)

    device = next(model.parameters()).device
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        data["img_metas"] = [item.data[0] for item in data["img_metas"]]

    model.eval()
    with torch.no_grad():
        # This is the original mmseg inference path and is authoritative for IDs.
        result = model(return_loss=False, rescale=True, **data)
        class_ids_array = np.asarray(result[0], dtype=np.uint8) + 1

        # The original API does not expose probabilities, so derive them from the
        # same batch only after obtaining the authoritative segmentation result.
        img_tensor = data["img"][0]
        img_metas = data["img_metas"]
        features = model.extract_feat(img_tensor)
        all_cls_scores, all_mask_preds = model.decode_head(features, img_metas)
        class_scores = F.softmax(all_cls_scores[-1], dim=-1)[:, :, :150]
        mask_scores = F.interpolate(
            all_mask_preds[-1],
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        probabilities = torch.einsum(
            "bqc,bqhw->bchw", class_scores, mask_scores
        )
        probabilities = probabilities / (
            probabilities.sum(dim=1, keepdim=True) + 1e-8
        )
        probabilities = F.interpolate(
            probabilities,
            size=class_ids_array.shape,
            mode="bilinear",
            align_corners=False,
        )
        probabilities = probabilities / (
            probabilities.sum(dim=1, keepdim=True) + 1e-8
        )
        confidence_array = (
            probabilities.max(dim=1).values.squeeze(0).cpu().numpy().astype(np.float16)
        )

    archive = {
        "class_ids": class_ids_array,
        "confidence": confidence_array,
    }
    if include_probabilities:
        archive["probabilities"] = (
            probabilities.squeeze(0).cpu().numpy().astype(np.float16)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)
    print(f"Prediction saved to: {output_path}")
    print(f"Fields: {', '.join(archive)}")
    return class_ids_array


cfg_str = load_config_from_url(CONFIG_URL)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = init_segmentor(cfg)
load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
model.cuda()
model.eval()

class_ids = export_segmentation_prediction(
    model=model,
    cfg=cfg,
    image_path="images/05_Schafgasse.png",
    output_path="output/segmentation_prediction.npz",
    include_probabilities=False,
)
