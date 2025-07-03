#!/bin/bash
# Install Hugin panorama stitching tools
# This script installs Hugin on various Linux distributions

set -e

echo "🌍 Installing Hugin panorama tools..."

# Detect the Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VERSION=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    OS=$(lsb_release -si)
    VERSION=$(lsb_release -sr)
elif [ -f /etc/redhat-release ]; then
    OS="Red Hat"
    VERSION=$(cat /etc/redhat-release | awk '{print $4}')
else
    echo "❌ Cannot detect Linux distribution"
    exit 1
fi

echo "📋 Detected OS: $OS $VERSION"

# Install Hugin based on the distribution
case "$OS" in
    "Ubuntu"|"Debian"*)
        echo "🔧 Installing Hugin on Ubuntu/Debian..."
        apt-get update
        apt-get install -y hugin hugin-tools enblend enfuse
        ;;
    "CentOS"*|"Red Hat"*|"Rocky"*|"AlmaLinux"*)
        echo "🔧 Installing Hugin on CentOS/RHEL..."
        yum update -y
        yum install -y epel-release
        yum install -y hugin hugin-tools enblend enfuse
        ;;
    "Fedora"*)
        echo "🔧 Installing Hugin on Fedora..."
        dnf update -y
        dnf install -y hugin hugin-tools enblend enfuse
        ;;
    "Alpine"*)
        echo "🔧 Installing Hugin on Alpine..."
        apk update
        apk add hugin hugin-tools enblend enfuse
        ;;
    *)
        echo "❌ Unsupported Linux distribution: $OS"
        echo "Please install Hugin manually:"
        echo "  - Ubuntu/Debian: sudo apt-get install hugin hugin-tools enblend enfuse"
        echo "  - CentOS/RHEL: sudo yum install hugin hugin-tools enblend enfuse"
        echo "  - Fedora: sudo dnf install hugin hugin-tools enblend enfuse"
        echo "  - Alpine: sudo apk add hugin hugin-tools enblend enfuse"
        exit 1
        ;;
esac

echo "✅ Verifying Hugin installation..."

# Verify required tools are available
REQUIRED_TOOLS=("pto_gen" "cpfind" "autooptimiser" "pano_modify" "nona" "enblend")
MISSING_TOOLS=()

for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        MISSING_TOOLS+=("$tool")
    else
        echo "✅ $tool: $(which $tool)"
    fi
done

if [ ${#MISSING_TOOLS[@]} -eq 0 ]; then
    echo "🎉 Hugin installation successful!"
    echo "📋 Available tools:"
    for tool in "${REQUIRED_TOOLS[@]}"; do
        echo "  - $tool"
    done
else
    echo "❌ Installation incomplete. Missing tools:"
    for tool in "${MISSING_TOOLS[@]}"; do
        echo "  - $tool"
    done
    exit 1
fi

echo "🌍 Hugin is ready for 360° panorama processing!"