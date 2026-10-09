from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def groundingdino_python_paths(config_path: str | Path) -> list[str]:
    groundingdino_root = Path(config_path).resolve().parents[1]
    build_paths = sorted((groundingdino_root / "build").glob("lib.*"))
    return [str(path) for path in build_paths] + [str(groundingdino_root.parent)]


class TravelUAVDinoMonitor:
    def __init__(
        self,
        *,
        groundingdino_config: str | Path,
        groundingdino_model_path: str | Path | None,
        device: str | int = "cuda",
    ):
        self.groundingdino_config = Path(groundingdino_config)
        self.groundingdino_model_path = Path(groundingdino_model_path) if groundingdino_model_path else None
        self.device = device
        self.dino_model = None

    def _init_model(self) -> None:
        if self.dino_model is not None:
            return
        if self.groundingdino_model_path is None:
            raise FileNotFoundError("groundingdino_model_path is required when TravelUAV DINO stop is enabled")
        import sys

        import torch

        for path in reversed(groundingdino_python_paths(self.groundingdino_config)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from groundingdino.util.inference import load_model, predict

        device = torch.device(self.device)
        model = load_model(str(self.groundingdino_config), str(self.groundingdino_model_path), device=device)
        model.to(device=device)
        self.dino_model = partial(predict, model=model, device=device)

    def get_dino_results(self, observation: dict[str, Any], obj_info: str) -> bool:
        images = observation.get("rgb_record") or []
        depths = observation.get("depth_record") or []
        for image, depth in zip(images, depths):
            boxes, logits = self.detect(image, obj_info)
            for index, box in enumerate(boxes):
                point = list(map(int, box))
                center = (int((point[0] + point[2]) / 2), int((point[1] + point[3]) / 2))
                depth_data = int(depth[center[1], center[0]] / 2.55)
                if depth_data < 18:
                    _ = logits[index]
                    return True
        return False

    def detect(self, img: Any, prompt: str):
        self._init_model()
        import groundingdino.datasets.transforms as T
        import torch
        from groundingdino.util import box_ops

        img_src = np.asarray(img)
        pil_img = Image.fromarray(img_src)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(pil_img, None)
        boxes, logits, _phrases = self.dino_model(
            image=image_transformed,
            caption=prompt,
            box_threshold=0.6,
            text_threshold=0.40,
        )
        logits = logits.detach().cpu().numpy()
        height, width, _ = img_src.shape
        boxes_xyxy = (box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([width, height, width, height])).cpu().numpy()
        keep = ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) <= width * 0.6) & (
            (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) <= height * 0.5
        )
        return boxes_xyxy[keep], logits[keep]
