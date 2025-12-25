"""Segmentation helpers: Grounded-SAM with placeholder fallback."""

from pathlib import Path
from typing import Tuple
import os
import numpy as np
from PIL import Image
import cv2

try:
    import torch
    from groundeddino_vl import load_model, predict
    from segment_anything import sam_model_registry, SamPredictor
    GROUNDING_DINO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GroundingDINO or Segment-Anything not available: {e}")
    GROUNDING_DINO_AVAILABLE = False
    torch = None


# Model paths (relative to repo root)
GROUNDING_DINO_CONFIG_PATH = "tools/GroundedDINO-VL/groundeddino_vl/models/configs/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "tools/weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT_PATH = "tools/weights/sam_vit_h_4b8939.pth"

# Global model instances
grounding_dino_model = None
sam_predictor = None


def _with_suffix(path: str, suffix: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))


def _ensure_models() -> bool:
    """Lazy-load GroundingDINO + SAM once."""
    global grounding_dino_model, sam_predictor
    if not GROUNDING_DINO_AVAILABLE:
        return False
    if grounding_dino_model is None or sam_predictor is None:
        print("Initializing GroundingDINO and SAM models...")
        try:
            grounding_dino_model = load_model(
                GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH
            )
            print("DINO load done")
            sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH)
            sam_predictor = SamPredictor(sam)
            print("Models initialized successfully.")
        except Exception as e:  # pragma: no cover - initialization errors handled by caller
            print(f"Failed to initialize models: {e}")
            return False
    return True


def _save_dino_box(image_path: str, image_bgr: np.ndarray, box_xyxy: np.ndarray, target_word: str):
    x1, y1, x2, y2 = map(int, box_xyxy.tolist())
    vis = image_bgr.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, target_word, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out_path = _with_suffix(image_path, "_dino")
    cv2.imwrite(out_path, vis)
    print(f"DINO detection box saved to {out_path}")


def _save_mask_overlay(image_path: str, image_bgr: np.ndarray, mask: np.ndarray, bbox, target_word: str):
    x1, y1, x2, y2 = bbox
    overlay = image_bgr.copy()
    color_layer = np.zeros_like(overlay)
    color_layer[mask.astype(bool)] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.4, 0)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 128, 255), 2)
    cv2.putText(overlay, target_word, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
    out_path = _with_suffix(image_path, "_mask")
    cv2.imwrite(out_path, overlay)
    print(f"Mask overlay saved to {out_path}")


def _segment_with_grounded_sam(
    image_path: str,
    target_word: str,
    save_visuals: bool,
    box_threshold: float,
    text_threshold: float,
):
    if not _ensure_models():
        return _placeholder_segmentation(image_path, save_visuals)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read image from {image_path}")
        return _placeholder_segmentation(image_path, save_visuals)

    original_bgr = image_bgr.copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print(f"Running GroundingDINO for '{target_word}'...")
    result = predict(
        model=grounding_dino_model,
        image=image_rgb,
        text_prompt=target_word,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    boxes_xyxy = result.to_xyxy(denormalize=True)
    if boxes_xyxy.size(0) == 0:
        print(f"GroundingDINO found no objects matching '{target_word}'. Falling back to placeholder.")
        return _placeholder_segmentation(image_path, save_visuals)

    best_idx = int(torch.argmax(result.scores).item()) if len(result.scores) > 0 else 0
    best_box = boxes_xyxy[best_idx].unsqueeze(0)

    if save_visuals:
        _save_dino_box(image_path, original_bgr, best_box[0].cpu().numpy(), target_word)

    print("Running SAM for segmentation...")
    sam_predictor.set_image(image_rgb)
    h, w, _ = image_rgb.shape
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(best_box, (h, w))
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    mask = masks[0, 0].cpu().numpy().astype(np.uint8)
    ys, xs = np.where(mask)
    if ys.size == 0:
        print("SAM returned an empty mask. Falling back to placeholder.")
        return _placeholder_segmentation(image_path, save_visuals)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = (x1, y1, x2, y2)

    if save_visuals:
        _save_mask_overlay(image_path, original_bgr, mask, bbox, target_word)

    print(f"Segmentation successful. Mask shape: {mask.shape}, BBox: {bbox}")
    return mask, bbox


def _placeholder_segmentation(image_path: str, save_visuals: bool) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    mask = placeholder_mask(image_path)
    ys, xs = np.where(mask)
    if ys.size == 0:
        if save_visuals:
            try:
                original_image_bgr = cv2.imread(image_path)
                if original_image_bgr is not None:
                    cv2.imwrite(_with_suffix(image_path, "_mask"), original_image_bgr)
            except Exception:
                pass
        return mask, (0, 0, 0, 0)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    bbox = (x1, y1, x2, y2)

    if save_visuals:
        try:
            original_image_bgr = cv2.imread(image_path)
            if original_image_bgr is not None:
                _save_mask_overlay(image_path, original_image_bgr, mask, bbox, target_word="placeholder")
        except Exception:
            pass

    return mask, bbox


def placeholder_mask(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mask = np.zeros((h, w), dtype=np.uint8)
    cy, cx = h // 2, w // 2
    rr = min(h, w) // 6
    y, x = np.ogrid[:h, :w]
    mask[((y - cy) ** 2 + (x - cx) ** 2) <= rr * rr] = 1
    return mask


def segment_target(
    image_path: str,
    target_word: str,
    *,
    save_visuals: bool = True,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return (mask, bbox) where bbox is (x1,y1,x2,y2).

    If USE_GROUNDED_SAM is set and dependencies are present, runs GroundingDINO+SAM;
    otherwise falls back to a placeholder circular mask.
    """
    if os.environ.get("USE_GROUNDED_SAM") and GROUNDING_DINO_AVAILABLE:
        try:
            return _segment_with_grounded_sam(
                image_path,
                target_word,
                save_visuals=save_visuals,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except Exception as e:  # pragma: no cover - runtime errors fall back
            print(f"Real segmentation failed: {e}. Falling back to placeholder.")
            return _placeholder_segmentation(image_path, save_visuals)

    print("Using placeholder segmentation.")
    return _placeholder_segmentation(image_path, save_visuals)
