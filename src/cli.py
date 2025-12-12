"""Simple CLI to run the step: parse instruction, segment target, compute 2D center."""
import sys
import numpy as np
from PIL import Image
from .parser import extract_target_from_instruction
from .segmenter import segment_target

def compute_affordance_center(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    cy = ys.mean()
    cx = xs.mean()
    return int(cx), int(cy)

def main(argv):
    if len(argv) < 3:
        print("Usage: cli.py <image_path> <instruction>")
        return 2
    image_path = argv[1]
    instruction = " ".join(argv[2:])
    target = extract_target_from_instruction(instruction)
    print(f"Parsed target: {target}")
    mask, bbox = segment_target(image_path, target)
    center = compute_affordance_center(mask)
    print(f"BBox: {bbox}")
    print(f"Affordance center (x,y): {center}")
    # Optionally save mask visualization
    try:
        img = Image.open(image_path).convert("RGBA")
        arr = np.array(img)
        alpha = (mask * 120).astype(np.uint8)
        overlay = np.zeros_like(arr)
        overlay[..., 0] = 255
        overlay[..., 3] = alpha
        out = Image.alpha_composite(img, Image.fromarray(overlay))
        out.save("mask_overlay.png")
        print("Saved overlay to mask_overlay.png")
    except Exception as e:
        print("Could not save overlay:", e)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
