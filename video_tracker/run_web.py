#!/usr/bin/env python3
"""
MOTION TRACKER WEB GUI
DOT-MATRIX / MONO / MINIMAL / PERFORATION / CARBON-COPY SHADOW / SCANLINE / GRID-INDEX
"""

import os
import sys
from web_app import app

if __name__ == '__main__':
    print("=" * 60)
    print("MOTION TRACKER WEB GUI")
    print("=" * 60)
    print("AESTHETIC: DOT-MATRIX / MONO / MINIMAL")
    print("TONE: ALL-CAPS, DRY, PRECISE, INDUSTRIAL")
    print("=" * 60)
    print("STARTING SERVER...")
    print("ACCESS: http://localhost:5000")
    print("=" * 60)
    
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
