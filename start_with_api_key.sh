#!/bin/bash
# Start HDRi 360 Studio Server with matching API key

export API_KEY="HDRi360Studio2024SecureKey"
export FLASK_ENV="production"
export PORT="${PORT:-5001}"

echo "🚀 Starting HDRi 360 Studio Panorama Server"
echo "   API Key: ${API_KEY:0:15}... (configured)"
echo "   Environment: $FLASK_ENV"
echo "   Port: $PORT"
echo ""

python app.py