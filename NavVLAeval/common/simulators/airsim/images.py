from __future__ import annotations

from typing import Any

import numpy as np


def decode_scene_response(response: Any) -> np.ndarray:
    return np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)


def decode_depth_response(airsim_module: Any, response: Any) -> np.ndarray:
    depth_img_in_meters = airsim_module.list_to_2d_float_array(
        response.image_data_float,
        response.width,
        response.height,
    )
    return (np.clip(depth_img_in_meters, 0, 100) / 100 * 255).astype(np.uint8)
