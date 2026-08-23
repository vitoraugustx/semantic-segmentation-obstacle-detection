import urllib.request
from pathlib import Path
from typing import Tuple

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
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
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs Mask2Former inference and saves per-pixel IDs and confidences.

    Saved IDs use the ADE20K convention from 1 to 150. When requested,
    the file also includes ``probabilities`` in the [150, H, W] format.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(dict(img_info=dict(filename=str(image_path)), img_prefix=None))
    img_tensor = data["img"][0].unsqueeze(0).to(device)
    img_metas = [data["img_metas"][0].data]

    model.eval()
    with torch.no_grad():
        features = model.extract_feat(img_tensor)
        all_cls_scores, all_mask_preds = model.decode_head(features, img_metas)
        cls_score = all_cls_scores[-1]
        mask_pred = F.interpolate(
            all_mask_preds[-1],
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        class_probabilities = F.softmax(cls_score, dim=-1)[:, :, :150]
        mask_probabilities = mask_pred.sigmoid()
        probabilities = torch.einsum(
            "bqc,bqhw->bchw", class_probabilities, mask_probabilities
        )
        probabilities = probabilities / (probabilities.sum(dim=1, keepdim=True) + 1e-8)

        class_ids = probabilities.argmax(dim=1).squeeze(0) + 1
        confidence = probabilities.max(dim=1).values.squeeze(0)

    class_ids_array = class_ids.cpu().numpy().astype(np.uint8)
    confidence_array = confidence.cpu().numpy().astype(np.float16)
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
    return class_ids_array, confidence_array


cfg_str = load_config_from_url(CONFIG_URL)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

model = init_segmentor(cfg)
load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
model.cuda()
model.eval()

class_ids, confidence = export_segmentation_prediction(
    model=model,
    cfg=cfg,
    image_path="images/05_Schafgasse.png",
    output_path="output/segmentation_prediction.npz",
    include_probabilities=False,
)
