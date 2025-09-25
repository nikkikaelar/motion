python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r video_tracker\requirements.txt


# Higher accuracy run
python video_tracker\app.py \ 
  --input "video_tracker\robbery.mp4" \ 
  --output "video_tracker\robbery_out.mp4" \ 
  --preset quality --resolution-cap 960 --min-detect-conf 0.6 --class-allowlist robbers