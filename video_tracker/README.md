# Video Tracker MVP

Upload a video; get an MP4 with colored, per-ID boxes on moving objects.  
Uses YOLO (fast/quality) + ByteTrack. Static-camera mode uses background subtraction to cut false motion.  
Falls back to detect-only if tracking collapses.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r video_tracker/requirements.txt
chmod +x video_tracker/run.sh
./video_tracker/run.sh input.mp4 out.mp4 --only-moving --preset fast --resolution-cap 1280
```
