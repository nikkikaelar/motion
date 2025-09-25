param(
    [string]$InputPath = "input.mp4",
    [string]$OutputPath = "out.mp4"
)

$ErrorActionPreference = "Stop"

# Usage examples:
#   ./video_tracker/run.ps1 -InputPath input.mp4 -OutputPath out.mp4
#   ./video_tracker/run.ps1 -InputPath input.mp4 -OutputPath out.mp4 -- --only-moving --static-camera --preset quality --resolution-cap 1280 --speed-threshold 25
# Note: extra args after "--" are forwarded to the Python script.

# Forward any remaining arguments to Python
$extra = $args

python video_tracker/app.py --input $InputPath --output $OutputPath @extra
