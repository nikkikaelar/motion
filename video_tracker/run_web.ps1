# MOTION TRACKER WEB GUI
# DOT-MATRIX / MONO / MINIMAL / PERFORATION / CARBON-COPY SHADOW / SCANLINE / GRID-INDEX

Write-Host "============================================================" -ForegroundColor Green
Write-Host "MOTION TRACKER WEB GUI" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "AESTHETIC: DOT-MATRIX / MONO / MINIMAL" -ForegroundColor Green
Write-Host "TONE: ALL-CAPS, DRY, PRECISE, INDUSTRIAL" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "STARTING SERVER..." -ForegroundColor Green
Write-Host "ACCESS: http://localhost:5000" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

# Change to script directory
Set-Location $PSScriptRoot

# Run the web application
python run_web.py
