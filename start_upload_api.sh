#!/bin/bash
# Start the lightweight upload API (no ML dependencies)

echo "🚀 Starting POV Bounties Upload API..."
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Using local storage."
    echo "   To enable S3 uploads, create .env with AWS credentials."
    echo ""
fi

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Loaded environment variables from .env"
    echo ""
fi

# Run the upload API
python upload_api.py

