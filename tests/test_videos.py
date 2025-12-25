"""Sample frames from two view folders under a root and run parser+segmenter on them.

Usage: set ROOT variable to dataset root and run within SFA env:
    python scripts/test_videos.py
"""
import os
import cv2
import numpy as np
from pathlib import Path
from src.get_target import extract_action_objects
from src.segmenter import segment_target
from PIL import Image


ROOT = '/root/gpufree-data/datasets/gello-pick-cup-fps15/videos/chunk-000'
OUTDIR = '/root/Project/SFA/outputs'
INSTR = 'Please pick up the green cup and insert it into the pink cup.'
FRAME_STEP = 30  # sample every 30 frames (~2s at 15fps)
MAX_PER_VIDEO = 3


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_overlay(image_bgr, mask, out_path):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
    alpha = (mask * 120).astype(np.uint8)
    overlay = np.zeros_like(image)
    overlay[..., 0] = 255
    overlay[..., 3] = alpha
    out = Image.alpha_composite(Image.fromarray(image), Image.fromarray(overlay))
    out.save(out_path)


def process_video(video_path, view_name):
    # Transcode to H.264 into a temp file to avoid AV1 decode issues
    tmp_out = f"/tmp/{Path(video_path).stem}_h264.mp4"
    if os.path.exists(tmp_out):
        try:
            os.remove(tmp_out)
        except Exception:
            pass
    cmd = f"ffmpeg -y -i \"{video_path}\" -c:v libx264 -preset veryfast -crf 23 -an \"{tmp_out}\" > /dev/null 2>&1"
    rc = os.system(cmd)
    if rc != 0 or not os.path.exists(tmp_out):
        print(f"Transcode failed for {video_path}, attempting to open original file")
        vid = cv2.VideoCapture(str(video_path))
    else:
        vid = cv2.VideoCapture(str(tmp_out))
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    saved = 0
    idx = 0
    name = Path(video_path).stem
    while True and saved < MAX_PER_VIDEO:
        ret, frame = vid.read()
        if not ret:
            break
        if idx % FRAME_STEP == 0:
            img_path = f"/tmp/{name}_frame{idx}.png"
            cv2.imwrite(img_path, frame)
            try:
                result = extract_action_objects(INSTR)
                source_object = result['source_object']
                target = result['destination_object']
                action = result['actions']
                mask, bbox = segment_target(img_path, target)
                out_dir = os.path.join(OUTDIR, view_name)
                ensure_dir(out_dir)
                out_path = os.path.join(out_dir, f"{name}_frame{idx}.png")
                save_overlay(frame, mask, out_path)
                print(f"Saved {out_path} (bbox={bbox}, center approx)")
                saved += 1
            except NotImplementedError as e:
                print("Model integration not implemented; placeholder not available:", e)
                break
            except Exception as e:
                print("Error processing frame:", e)
        idx += 1
    vid.release()
    # cleanup temp file
    try:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    except Exception:
        pass


def main():
    root = Path(ROOT)
    if not root.exists():
        print("Root path not found:", ROOT)
        return 2
    # look for subdirs that contain main and wrist views
    for child in sorted(root.iterdir()):
        if child.is_dir():
            view_name = child.name
            # find video files
            videos = list(child.glob("*.mp4")) + list(child.glob("*.avi"))
            if not videos:
                print(f"No videos in {child}")
                continue
            for v in videos:
                print(f"Processing {v}")
                process_video(v, view_name)

    print("Done")


if __name__ == '__main__':
    main()
