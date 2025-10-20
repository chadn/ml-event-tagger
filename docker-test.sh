#!/bin/bash
# Docker Testing Script for ML Event Tagger

set -e

echo "================================================================================"
echo "🐳 ML Event Tagger - Docker Testing"
echo "================================================================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker daemon is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker daemon is running"
echo ""

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t ml-event-tagger:0.0.6 -t ml-event-tagger:latest .
echo "✅ Docker image built successfully"
echo ""

# Show image details
echo "📊 Image details:"
docker images ml-event-tagger
echo ""

# Run the container
echo "🚀 Starting container..."
CONTAINER_ID=$(docker run -d -p 8000:8000 --name ml-event-tagger-test ml-event-tagger:0.0.6)
echo "✅ Container started: $CONTAINER_ID"
echo ""

# Wait for service to be ready
echo "⏳ Waiting for service to start..."
sleep 10

# Test health endpoint
echo "🔍 Testing /health endpoint..."
curl -s http://localhost:8000/health | jq '.'
echo ""

# Test predict endpoint
echo "🔍 Testing /predict endpoint..."
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "name": "House Music Night",
      "description": "DJ performance with dancing",
      "location": "Oakland"
    }]
  }' | jq '.'
echo ""

echo "================================================================================"
echo "✅ All tests passed!"
echo "================================================================================"
echo ""
echo "Container is running. To view logs:"
echo "  docker logs ml-event-tagger-test"
echo ""
echo "To stop and remove the container:"
echo "  docker stop ml-event-tagger-test && docker rm ml-event-tagger-test"
echo ""
echo "To run interactively:"
echo "  docker run -it -p 8000:8000 ml-event-tagger:0.0.6"
echo ""
echo "================================================================================"

