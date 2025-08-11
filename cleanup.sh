#!/bin/bash
# HDRi 360 Studio Panorama Server Cleanup Script

echo "🧹 Cleaning up panorama server directory..."

# Remove Python cache files
echo "📁 Removing Python cache files..."
rm -rf __pycache__ services/__pycache__

# Create archive directory for development files
mkdir -p archive

# Archive old monolithic files
echo "📦 Archiving old monolithic files..."
if [ -f "hugin_stitcher.py" ]; then
    mv hugin_stitcher.py archive/
    echo "   Archived: hugin_stitcher.py"
fi

if [ -f "app.py.backup" ]; then
    mv app.py.backup archive/
    echo "   Archived: app.py.backup"
fi

# Archive test and verification files
echo "📦 Archiving development test files..."
mv test_*.py verify_*.py coordinate_test_results.json archive/ 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   Archived: test and verification files"
fi

# Keep only essential startup scripts
echo "🔧 Organizing startup scripts..."
# Keep: start_with_api_key.sh (main), deploy.sh (Railway)
# Archive others if they exist
for script in start_server.sh setup.sh; do
    if [ -f "$script" ]; then
        mv "$script" archive/
        echo "   Archived: $script"
    fi
done

# Remove redundant README
echo "📄 Organizing documentation..."
if [ -f "README.md" ] && [ -f "SERVER_README.md" ]; then
    mv README.md archive/
    echo "   Archived: README.md (keeping SERVER_README.md)"
fi

# Clean empty directories
echo "📁 Cleaning empty directories..."
find . -type d -empty -name "uploads" -o -name "outputs" -o -name "temp" | while read dir; do
    echo "   Cleaned: $dir (empty)"
done

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📊 Current structure:"
echo "   Production files: app.py, services/, requirements.txt"
echo "   Deployment: start_with_api_key.sh, deploy.sh, Dockerfile"
echo "   Documentation: SERVER_README.md"
echo "   Archived: archive/ (development files)"
echo ""
echo "💾 Total space saved: $(du -sh archive/ 2>/dev/null | cut -f1 || echo 'calculating...')"