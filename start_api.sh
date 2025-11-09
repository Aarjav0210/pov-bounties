#!/bin/bash

# Video Validation API Startup Script

echo "🚀 Starting Video Validation API..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if CUDA is available
echo "🔍 Checking for CUDA..."
python3 -c "import torch; print('✅ CUDA available' if torch.cuda.is_available() else '⚠️  CUDA not available, will use CPU (slow!)')"
echo ""

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📥 Installing dependencies..."
    pip3 install -r requirements.txt
fi
echo ""

# Check if config.py exists, if not inform user (optional)
if [ ! -f "config.py" ]; then
    echo "ℹ️  No config.py found - using defaults"
    echo "   To customize: cp config.example.py config.py"
    echo ""
else
    echo "✅ Using config.py"
    echo ""
fi

# Start the server
echo "🌐 Starting FastAPI server..."
echo "📡 API will be available at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run with uvicorn - uses host/port from config.py or defaults
python3 api.py

