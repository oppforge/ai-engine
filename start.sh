#!/bin/bash

# OppForge AI Engine Startup Script

echo "🤖 Starting OppForge AI Engine..."

# Navigate to ai-engine directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if Groq API key is set
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  Warning: GROQ_API_KEY not set. Checking backend .env..."
    if [ -f "../backend/.env" ]; then
        source ../backend/.env
        echo "✅ Loaded environment from backend/.env"
    else
        echo "❌ Error: No GROQ_API_KEY found. Please set it in backend/.env"
        exit 1
    fi
fi

# Run the server
echo "🚀 Starting AI Engine on port 8001..."
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
