"""Smoke test for segment_target using test image.

By default runs placeholder path (USE_GROUNDED_SAM unset). Set USE_GROUNDED_SAM=1 to
exercise real Grounded-SAM if dependencies are available.
"""
import os
from pathlib import Path
from PIL import Image
import numpy as np

from src.segmenter import segment_target

TEST_IMAGE = Path(__file__).with_name("test.png")


def test_segment_target_runs_placeholder():
    os.environ.pop("USE_GROUNDED_SAM", None)
    mask, bbox = segment_target(str(TEST_IMAGE), "cup", save_visuals=False)
    w, h = Image.open(TEST_IMAGE).size
    assert mask.shape == (h, w)
    assert len(bbox) == 4


def test_segment_target_runs_real_if_enabled():
    if not os.environ.get("USE_GROUNDED_SAM"):
        return  # skip unless explicitly requested
    mask, bbox = segment_target(str(TEST_IMAGE), "cup", save_visuals=False)
    w, h = Image.open(TEST_IMAGE).size
    assert mask.shape == (h, w)
    assert len(bbox) == 4
    assert np.any(mask)


if __name__ == "__main__":
    os.environ.setdefault("USE_GROUNDED_SAM", "1")
    mask, bbox = segment_target(str(TEST_IMAGE), "cup", save_visuals=True)
    print(f"Result - Mask shape: {mask.shape}, BBox: {bbox}")
