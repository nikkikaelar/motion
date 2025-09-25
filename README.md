# Entropy: Video Tracker MVP

Server-side video processor that annotates videos with colored, per-ID boxes for moving objects.

- Detector: Ultralytics YOLO (fast/quality preset)
- Tracker: ByteTrack (via custom YAML)
- Output: MP4 with per-ID boxes and speed labels
- Options:
  - Only draw moving objects (speed threshold)
  - Static-camera background subtraction filter
  - Detect-only mode for maximum speed
  - Class allowlist and minimum detection confidence

## Directory structure
- `video_tracker/app.py` — main application
- `video_tracker/requirements.txt` — Python dependencies
- `video_tracker/trackers/bytetrack.yaml` — ByteTrack config
- `video_tracker/run.ps1` — Windows PowerShell runner
- `video_tracker/run.sh` — Bash runner (WSL/git-bash)

## Quick start (Windows, PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r video_tracker\requirements.txt

# Fast run (detect-only)
python video_tracker\app.py \ 
  --input "C:\\path\\to\\input.mp4" \ 
  --output "C:\\path\\to\\out.mp4" \ 
  --preset fast --resolution-cap 640 --detect-only --min-detect-conf 0.6 --class-allowlist horse

# Higher accuracy run
python video_tracker\app.py \ 
  --input "C:\\path\\to\\input.mp4" \ 
  --output "C:\\path\\to\\out.mp4" \ 
  --preset quality --resolution-cap 960 --min-detect-conf 0.6 --class-allowlist horse
```

## Notes
- On first run, YOLOv8 model weights will be downloaded automatically.
- Tracking requires extra dependencies; detect-only mode avoids tracker dependencies and is fastest on CPU.
- Use `--only-moving` and `--static-camera` when appropriate for your footage.
