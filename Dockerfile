# HDRi 360 Studio - Hugin-based Panorama Processing Server
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Hugin
RUN apt-get update && apt-get install -y \
    # Basic utilities
    wget curl ca-certificates \
    # Build tools
    build-essential \
    # Image processing libraries
    libjpeg-dev libpng-dev libtiff-dev \
    libwebp-dev libopenjp2-7-dev \
    # Hugin and related tools
    hugin hugin-tools enblend enfuse \
    # Additional dependencies
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads outputs temp

# Verify Hugin installation
RUN echo "Verifying Hugin installation..." && \
    pto_gen --help > /dev/null && \
    cpfind --help > /dev/null && \
    autooptimiser --help > /dev/null && \
    pano_modify --help > /dev/null && \
    nona --help > /dev/null && \
    enblend --help > /dev/null && \
    echo "✅ Hugin tools verified successfully"

# Set permissions
RUN chmod +x install_hugin.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5001}/health || exit 1

# Expose port
EXPOSE ${PORT:-5001}

# Run the application
CMD ["python", "app.py"]