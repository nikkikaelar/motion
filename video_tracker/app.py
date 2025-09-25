"""
MVP: server-side video processor
- Detector: Ultralytics YOLO (fast/quality preset)
- Tracker: ByteTrack (via custom YAML)
- Output: MP4 with colored, per-ID boxes
- Draw only moving objects (speed threshold)
- "Static camera" mode: background subtraction filter
- Fallback: detect-only when tracking collapses
"""

import argparse
import os
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Tuple, Deque, Optional, Set
from tqdm import tqdm
from ultralytics import YOLO

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="YOLO + ByteTrack video annotator (MVP)")
    p.add_argument("--input", required=True, help="Path to input video")
    p.add_argument("--output", required=True, help="Path to output MP4")
    p.add_argument("--only-moving", action="store_true", help="Draw boxes only for moving tracks")
    p.add_argument("--static-camera", action="store_true", help="Enable background subtraction filter")
    p.add_argument("--resolution-cap", type=int, default=1280, help="Max width (px); height auto-scales")
    p.add_argument("--preset", choices=["fast", "quality"], default="fast", help="Model size vs speed")
    p.add_argument("--detect-only", action="store_true", help="Skip tracking and run detections only (faster)")
    p.add_argument("--min-detect-conf", type=float, default=0.5,
                   help="Minimum detector confidence to accept a box (applied before tracking filters)")
    p.add_argument("--class-allowlist", nargs="+", default=None,
                   help="Optional list of class names to keep (e.g., horse person). Others are filtered out.")
    p.add_argument("--speed-threshold", type=float, default=10.0,
                   help="Min avg speed (pixels/frame) to count as 'moving'")
    p.add_argument("--motion-overlap", type=float, default=0.05,
                   help="Min fraction of bbox overlapping motion mask (static-camera mode)")
    p.add_argument("--min-track-conf", type=float, default=0.1,
                   help="Drop boxes with mean score below this")
    p.add_argument("--fallback-patience", type=int, default=30,
                   help="# consecutive frames with no tracks before fallback to detect-only")
    p.add_argument("--tracker-yaml", default="trackers/bytetrack.yaml",
                   help="Path to ByteTrack YAML")
    return p.parse_args()

# ------------------------- Video I/O -------------------------

def ensure_video_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap

def open_video_writer(out_path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # broad compatibility
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {out_path}")
    return writer

def compute_resize(w: int, h: int, cap_w: int) -> Tuple[int, int, float]:
    if w <= cap_w:
        return w, h, 1.0
    scale = cap_w / float(w)
    return int(round(w * scale)), int(round(h * scale)), scale

# ------------------------- Drawing utils -------------------------

def hash_id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Stable, bright color per ID (HSV→BGR)."""
    h = (track_id * 47) % 180  # 0..179 in OpenCV
    s, v = 200, 255
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_box(img, xyxy, color, label):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

# ------------------------- Motion & speed -------------------------

def bbox_centroid(xyxy) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def avg_speed_px_per_frame(history: Deque[Tuple[float,float]], max_pairs: int = 5) -> float:
    """Average step distance across latest pairs to smooth jitter."""
    if len(history) < 2:
        return 0.0
    dists = []
    pairs = min(max_pairs, len(history) - 1)
    for i in range(1, pairs + 1):
        (x2,y2) = history[-i]
        (x1,y1) = history[-i-1]
        dists.append(((x2-x1)**2 + (y2-y1)**2) ** 0.5)
    return float(np.mean(dists)) if dists else 0.0

def motion_overlap_fraction(mask: np.ndarray, xyxy) -> float:
    """Fraction of bbox area marked 'motion' in mask (mask is 0/255)."""
    x1, y1, x2, y2 = [int(max(0, v)) for v in xyxy]
    x2 = min(mask.shape[1]-1, x2)
    y2 = min(mask.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi)) / float(roi.size)

# ------------------------- Main processing -------------------------

def main():
    args = parse_args()

    # Select model by preset
    model_name = "yolov8n.pt" if args.preset == "fast" else "yolov8x.pt"
    model = YOLO(model_name)

    # Open video
    cap = ensure_video_capture(args.input)
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Resize plan
    out_w, out_h, scale = compute_resize(src_w, src_h, args.resolution_cap)
    writer = open_video_writer(args.output, fps, out_w, out_h)

    # Background subtractor (for static camera)
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False) if args.static_camera else None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    # Track state
    # Track ID -> centroid history (for speed calc)
    centroids: Dict[int, Deque[Tuple[float,float]]] = defaultdict(lambda: deque(maxlen=12))
    # Track ID -> running confidence mean
    track_conf: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=15))

    # Fallback logic / mode selection
    consecutive_frames_no_tracks = 0
    use_detect_only = bool(args.detect_only)

    # Normalize class allowlist if provided
    allow: Optional[Set[str]] = None
    if args.class_allowlist:
        allow = {s.lower() for s in args.class_allowlist}

    progress = tqdm(total=frames if frames > 0 else None, desc="Processing", unit="f")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Resize if needed
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        draw_frame = frame.copy()

        # Motion mask (only used as a filter when static camera is on)
        motion_mask = None
        if bg_sub is not None:
            raw = bg_sub.apply(frame)
            motion_mask = cv2.medianBlur(raw, 5)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        # Decide mode
        if not use_detect_only:
            # TRACK MODE
            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker_yaml,
                verbose=False
            )
            r = results[0] if results else None

            if r is not None and r.boxes is not None and r.boxes.id is not None:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                scores     = r.boxes.conf.cpu().numpy()
                ids        = r.boxes.id.cpu().numpy().astype(int)
                classes    = r.boxes.cls.cpu().numpy().astype(int) if getattr(r.boxes, 'cls', None) is not None else None
                names      = r.names
                n_tracks = len(ids)

                if classes is None:
                    classes_iter = [None] * len(ids)
                else:
                    classes_iter = classes

                for xyxy, conf, tid, cls in zip(boxes_xyxy, scores, ids, classes_iter):
                    # Detector confidence gate
                    if float(conf) < args.min_detect_conf:
                        continue

                    # Optional class filtering
                    if allow is not None and cls is not None:
                        cname = names.get(int(cls), str(int(cls))) if isinstance(names, dict) else str(int(cls))
                        if cname.lower() not in allow:
                            continue

                    # Confidence gate (helps with jitter)
                    track_conf[tid].append(float(conf))
                    mean_conf = float(np.mean(track_conf[tid]))
                    if mean_conf < args.min_track_conf:
                        continue

                    # Speed check
                    c = bbox_centroid(xyxy)
                    centroids[tid].append(c)
                    speed = avg_speed_px_per_frame(centroids[tid])

                    is_moving = speed >= args.speed_threshold
                    if args.only_moving and not is_moving:
                        # Optional extra gate in static-cam mode: require motion overlap if present
                        if motion_mask is not None:
                            if motion_overlap_fraction(motion_mask, xyxy) < args.motion_overlap:
                                continue
                        else:
                            continue

                    # Static-camera motion filter (even when not only-moving)
                    if motion_mask is not None:
                        if motion_overlap_fraction(motion_mask, xyxy) < args.motion_overlap and args.only_moving:
                            continue

                    color = hash_id_to_color(tid)
                    if allow is not None and cls is not None:
                        label_cls = names.get(int(cls), "obj") if isinstance(names, dict) else "obj"
                    else:
                        label_cls = "id"
                    label = f"{label_cls} {tid} • {int(speed):d}px/f"
                    draw_box(draw_frame, xyxy, color, label)

            # Collapse criterion → fallback to detect-only
            if n_tracks == 0 and not args.detect_only:
                consecutive_frames_no_tracks += 1
                if consecutive_frames_no_tracks >= args.fallback_patience:
                    use_detect_only = True
            else:
                consecutive_frames_no_tracks = 0

        else:
            # DETECT-ONLY MODE (fallback)
            r = model.predict(frame, verbose=False)[0]
            n_boxes = 0
            if r.boxes is not None:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                scores     = r.boxes.conf.cpu().numpy()
                classes    = r.boxes.cls.cpu().numpy().astype(int)
                names      = r.names
                for xyxy, conf, cls in zip(boxes_xyxy, scores, classes):
                    # Detector confidence gate
                    if float(conf) < args.min_detect_conf:
                        continue
                    n_boxes += 1

                    # Optional class filtering
                    if allow is not None:
                        cname = names.get(int(cls), "obj") if isinstance(names, dict) else str(int(cls))
                        if cname.lower() not in allow:
                            continue

                    # If static camera or only-moving: use motion mask as proxy for "moving"
                    if args.only_moving and bg_sub is not None:
                        if motion_overlap_fraction(motion_mask, xyxy) < args.motion_overlap:
                            continue

                    color = (0, 255, 255)  # one color in fallback
                    cname = names.get(int(cls), "obj") if isinstance(names, dict) else str(int(cls))
                    label = f"{cname} {conf:.2f}"
                    draw_box(draw_frame, xyxy, color, label)

            # Try to recover into track mode if detections stable again
            if n_boxes > 0:
                consecutive_frames_no_tracks = 0
            else:
                consecutive_frames_no_tracks += 1
            if consecutive_frames_no_tracks == 0:
                pass  # already stable
            elif consecutive_frames_no_tracks > args.fallback_patience * 2:
                # Give tracking another chance
                use_detect_only = False
                consecutive_frames_no_tracks = 0

        writer.write(draw_frame)
        progress.update(1)

    progress.close()
    writer.release()
    cap.release()

    print(f"Done. Wrote: {args.output}")

if __name__ == "__main__":
    main()
