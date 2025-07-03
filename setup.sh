#!/bin/bash

# HDRi 360 Studio - OpenCV Panorama Processing Server Setup

echo "🌍 HDRi 360 Studio - OpenCV Panorama Processing Server Setup"
echo "============================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the server:"
echo "   ./start_server.sh"
echo ""
echo "🔗 Server will be available at:"
echo "   http://localhost:5000"
echo ""
echo "🏥 Health check:"
echo "   curl http://localhost:5000/health"
echo ""