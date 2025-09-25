"""
WEB GUI: DOT-MATRIX / MONO / MINIMAL / PERFORATION / CARBON-COPY SHADOW / SCANLINE / GRID-INDEX
TONE: ALL-CAPS, DRY, PRECISE, INDUSTRIAL
TYPOGRAPHY: MONOSPACE ONLY, TIGHT LEADING, NO LIGATURES
"""

import os
import io
import tempfile
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Tuple, Deque, Optional, Set
from ultralytics import YOLO
from simple_ai import SimpleThreatDetector

app = Flask(__name__)
app.secret_key = 'motion_tracker_secret_key_2024'

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Global processing state
processing_status = {}
processing_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size(file):
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    return size

# Import tracking functions from original app
def get_class_based_color_and_label(cls_name: str, conf: float) -> Tuple[Tuple[int, int, int], str]:
    """Get color and label based on detected class and context."""
    cls_name_lower = cls_name.lower()
    
    # Threat detection - red colors
    if any(weapon in cls_name_lower for weapon in ['gun', 'pistol', 'rifle', 'knife', 'weapon']):
        return (0, 0, 255), "ROBBERY"  # Red for threats
    elif cls_name_lower in ['person'] and conf > 0.7:
        # Check if person might be a victim (lower confidence, different context)
        return (0, 255, 0), "VICTIM"  # Green for victims
    elif cls_name_lower in ['car', 'truck', 'bus']:
        return (255, 255, 0), f"VEHICLE"  # Yellow for vehicles
    elif cls_name_lower in ['dog', 'cat']:
        return (255, 0, 255), f"ANIMAL"  # Magenta for animals
    else:
        # Default color based on class
        return (0, 255, 255), cls_name.upper()  # Cyan for others

def hash_id_to_color(track_id: int) -> Tuple[int, int, int]:
    """Stable, bright color per ID (HSV→BGR)."""
    h = (track_id * 47) % 180
    s, v = 200, 255
    bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_box(img, xyxy, color, label):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

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

def process_video_async(session_id, input_path, output_path, options):
    """Process video in background thread"""
    try:
        processing_status[session_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'INITIALIZING AI MODELS...'
        }
        
        # Select model
        model_name = "yolov8n.pt" if options.get('preset', 'fast') == 'fast' else "yolov8x.pt"
        model = YOLO(model_name)
        
        # Initialize AI system (always on)
        processing_status[session_id]['message'] = 'LOADING AI INTELLIGENCE...'
        ai_detector = SimpleThreatDetector()
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"FAILED TO OPEN VIDEO: {input_path}")
        
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        
        # Resize if needed
        resolution_cap = options.get('resolution_cap', 1280)
        if src_w <= resolution_cap:
            out_w, out_h, scale = src_w, src_h, 1.0
        else:
            scale = resolution_cap / float(src_w)
            out_w, out_h = int(round(src_w * scale)), int(round(src_h * scale))
        
        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"FAILED TO OPEN WRITER: {output_path}")
        
        # Background subtractor
        bg_sub = None
        if options.get('static_camera', False):
            bg_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        
        # Track state
        centroids: Dict[int, Deque[Tuple[float,float]]] = defaultdict(lambda: deque(maxlen=12))
        track_conf: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=15))
        
        # Processing parameters
        min_detect_conf = options.get('min_detect_conf', 0.5)
        speed_threshold = options.get('speed_threshold', 10.0)
        only_moving = options.get('only_moving', False)
        threat_detection = options.get('threat_detection', True)
        motion_overlap = options.get('motion_overlap', 0.05)
        min_track_conf = options.get('min_track_conf', 0.1)
        
        processing_status[session_id]['message'] = 'PROCESSING FRAMES...'
        
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Resize if needed
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            
            draw_frame = frame.copy()
            
            # Motion mask
            motion_mask = None
            if bg_sub is not None:
                raw = bg_sub.apply(frame)
                motion_mask = cv2.medianBlur(raw, 5)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            
            # Track objects
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker="trackers/bytetrack.yaml",
                    verbose=False
                )
            except AttributeError as e:
                if "fuse_score" in str(e):
                    # Fallback to detect-only mode if tracker fails
                    results = model.predict(frame, verbose=False)
                else:
                    raise e
            
            r = results[0] if results else None
            
            if r is not None and r.boxes is not None and r.boxes.id is not None:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int)
                classes = r.boxes.cls.cpu().numpy().astype(int) if getattr(r.boxes, 'cls', None) is not None else None
                names = r.names
                
                if classes is None:
                    classes_iter = [None] * len(ids)
                else:
                    classes_iter = classes
                
                for xyxy, conf, tid, cls in zip(boxes_xyxy, scores, ids, classes_iter):
                    if float(conf) < min_detect_conf:
                        continue
                    
                    # Confidence tracking
                    track_conf[tid].append(float(conf))
                    mean_conf = float(np.mean(track_conf[tid]))
                    if mean_conf < min_track_conf:
                        continue
                    
                    # Speed calculation
                    c = bbox_centroid(xyxy)
                    centroids[tid].append(c)
                    speed = avg_speed_px_per_frame(centroids[tid])
                    
                    is_moving = speed >= speed_threshold
                    if only_moving and not is_moving:
                        if motion_mask is not None:
                            # Motion overlap check
                            x1, y1, x2, y2 = [int(max(0, v)) for v in xyxy]
                            x2 = min(motion_mask.shape[1]-1, x2)
                            y2 = min(motion_mask.shape[0]-1, y2)
                            if x2 > x1 and y2 > y1:
                                roi = motion_mask[y1:y2, x1:x2]
                                if roi.size > 0:
                                    overlap = float(np.count_nonzero(roi)) / float(roi.size)
                                    if overlap < motion_overlap:
                                        continue
                        else:
                            continue
                    
                    # Get AI analysis (always on)
                    if cls is not None and ai_detector:
                        cls_name = names.get(int(cls), "obj") if isinstance(names, dict) else "obj"
                        
                        # Use simple AI for threat assessment
                        ai_analysis = ai_detector.analyze_visual_threats(
                            frame, xyxy, tid, cls_name, float(conf)
                        )
                        
                        color = ai_analysis["color"]
                        threat_label = ai_analysis["label"]
                        
                        # Add AI confidence to label
                        if only_moving:
                            label = f"{threat_label} {tid} • {int(speed):d}px/f • AI:{ai_analysis['threat_score']:.2f}"
                        else:
                            label = f"{threat_label} {tid} • AI:{ai_analysis['threat_score']:.2f}"
                            
                    else:
                        # Fallback to basic detection if AI fails
                        color = hash_id_to_color(tid)
                        if cls is not None:
                            cls_name = names.get(int(cls), "obj") if isinstance(names, dict) else "obj"
                            if only_moving:
                                label = f"{cls_name.upper()} {tid} • {int(speed):d}px/f"
                            else:
                                label = f"{cls_name.upper()} {tid}"
                        else:
                            label = f"ID {tid} • {int(speed):d}px/f"
                    
                    draw_box(draw_frame, xyxy, color, label)
            
            writer.write(draw_frame)
            frame_count += 1
            
            # Update progress
            if frames > 0:
                progress = int((frame_count / frames) * 100)
                processing_status[session_id]['progress'] = progress
                processing_status[session_id]['message'] = f'PROCESSING... {frame_count}/{frames} FRAMES'
        
        writer.release()
        cap.release()
        
        # Clean up input file (with retry for Windows)
        try:
            os.unlink(input_path)
        except PermissionError:
            # Windows sometimes locks files, try again after a delay
            import time
            time.sleep(1)
            try:
                os.unlink(input_path)
            except PermissionError:
                pass  # Give up if still locked
        
        processing_status[session_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'PROCESSING COMPLETE',
            'output_file': output_path
        }
        
    except Exception as e:
        processing_status[session_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'ERROR: {str(e)}'
        }
        # Clean up files on error (with retry for Windows)
        if os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except PermissionError:
                pass  # Give up if locked
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except PermissionError:
                pass  # Give up if locked

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'NO FILE PROVIDED'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'NO FILE SELECTED'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'INVALID FILE TYPE'}), 400
    
    # Check file size
    file_size = get_file_size(file)
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': f'FILE TOO LARGE: {file_size//1024//1024}MB > 100MB'}), 400
    
    # Generate session ID
    session_id = f"session_{int(time.time() * 1000)}"
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_input_{filename}")
    file.save(input_path)
    
    # Get processing options
    options = {
        'preset': request.form.get('preset', 'fast'),
        'only_moving': request.form.get('only_moving') == 'on',
        'static_camera': request.form.get('static_camera') == 'on',
        'threat_detection': request.form.get('threat_detection') == 'on',
        'resolution_cap': int(request.form.get('resolution_cap', 1280)),
        'min_detect_conf': float(request.form.get('min_detect_conf', 0.5)),
        'speed_threshold': float(request.form.get('speed_threshold', 10.0)),
        'motion_overlap': float(request.form.get('motion_overlap', 0.05)),
        'min_track_conf': float(request.form.get('min_track_conf', 0.1))
    }
    
    # Create output path
    output_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_output.mp4")
    
    # Start processing in background
    thread = threading.Thread(target=process_video_async, args=(session_id, input_path, output_path, options))
    thread.daemon = True
    thread.start()
    
    return jsonify({'session_id': session_id, 'message': 'UPLOAD SUCCESSFUL'})

@app.route('/status/<session_id>')
def get_status(session_id):
    if session_id not in processing_status:
        return jsonify({'error': 'INVALID SESSION'}), 404
    
    return jsonify(processing_status[session_id])

@app.route('/download/<session_id>')
def download_file(session_id):
    if session_id not in processing_status:
        return jsonify({'error': 'INVALID SESSION'}), 404
    
    status = processing_status[session_id]
    if status['status'] != 'completed':
        return jsonify({'error': 'PROCESSING NOT COMPLETE'}), 400
    
    output_file = status['output_file']
    if not os.path.exists(output_file):
        return jsonify({'error': 'OUTPUT FILE NOT FOUND'}), 404
    
    return send_file(output_file, as_attachment=True, download_name=f"tracked_{session_id}.mp4")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
