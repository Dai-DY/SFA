"""SAM/Grounded-SAM wrapper placeholder.

This module exposes `segment_target(image_path, target_word)` which should return a binary
mask (numpy array) and the bounding box. Currently provides a placeholder mask when heavy
models are not installed. Guidance included for replacing with Grounded-SAM or MobileSAM.
"""
from typing import Tuple
import numpy as np
from PIL import Image
import os

def placeholder_mask(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mask = np.zeros((h, w), dtype=np.uint8)
    # crude center circle as placeholder
    cy, cx = h // 2, w // 2
    rr = min(h, w) // 6
    y, x = np.ogrid[:h, :w]
    mask[((y - cy) ** 2 + (x - cx) ** 2) <= rr * rr] = 1
    return mask

def segment_target(image_path: str, target_word: str) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """Return (mask, bbox) where bbox is (x1,y1,x2,y2).

    Replace implementation to call Grounded-SAM or MobileSAM.
    """
    # Check for environment variable to decide whether to call real model
    if os.environ.get("USE_GROUNDED_SAM"):
        # Here you'd load the grounded-sam model and run it.
        # For now, raise NotImplementedError to indicate replacement point.
        raise NotImplementedError("Grounded-SAM integration not implemented in this placeholder.")

    mask = placeholder_mask(image_path)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return mask, (0,0,0,0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return mask, (x1, y1, x2, y2)
