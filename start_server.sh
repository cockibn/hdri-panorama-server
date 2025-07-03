#!/bin/bash

# HDRi 360 Studio - Start OpenCV Panorama Processing Server

echo "🌍 Starting HDRi 360 Studio OpenCV Processing Server..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Create necessary directories
mkdir -p uploads outputs temp

# Start the server
echo "🚀 Server starting on http://localhost:5001"
echo "📸 Ready to process 360° panoramas from your iOS app!"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"

python3 app.py