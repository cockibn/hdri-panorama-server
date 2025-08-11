#!/bin/bash
# HDRi 360 Studio Panorama Server Deployment Script

set -e  # Exit on any error

echo "🚀 Deploying HDRi 360 Studio Panorama Server..."

# Check if API_KEY is set
if [ -z "$API_KEY" ]; then
    echo "❌ ERROR: API_KEY environment variable is required for deployment"
    echo "Set it with: export API_KEY='your-secure-api-key'"
    exit 1
fi

# Set production defaults if not specified
export FLASK_ENV=${FLASK_ENV:-production}
export PORT=${PORT:-5001}
export HOST=${HOST:-0.0.0.0}
export PANORAMA_RESOLUTION=${PANORAMA_RESOLUTION:-6K}

echo "✅ Configuration:"
echo "   API Key: ${API_KEY:0:8}... (${#API_KEY} chars)"
echo "   Environment: $FLASK_ENV"
echo "   Port: $PORT"
echo "   Resolution: $PANORAMA_RESOLUTION"

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads outputs

# Set proper permissions
echo "🔐 Setting permissions..."
chmod 755 uploads outputs
chmod +x app.py

echo "🎯 Starting panorama server..."
python app.py