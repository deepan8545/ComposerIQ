#!/bin/bash

echo ""
echo " ============================================"
echo "  ComposerIQ — Visibility Engine"
echo " ============================================"
echo ""

# Kill any existing process on port 8000
PID=$(netstat -ano 2>/dev/null | grep ':8000' | grep 'LISTENING' | awk '{print $5}' | head -1)
if [ -n "$PID" ]; then
    echo " Killing existing process on port 8000 (PID: $PID)..."
    taskkill //F //PID "$PID" 2>/dev/null
    sleep 1
    echo " Done."
    echo ""
fi

# Check Python
if ! command -v python &> /dev/null; then
    echo " ERROR: Python not found. Install from python.org"
    exit 1
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    echo " Creating virtual environment..."
    python -m venv venv
fi

# Activate venv (Git Bash on Windows uses Scripts/)
source venv/Scripts/activate

# Install if needed
if ! pip show fastapi &> /dev/null; then
    echo " Installing dependencies... (this takes 2-3 mins first time)"
    pip install -r requirements.txt
fi

# Create .env if missing
if [ ! -f ".env" ]; then
    echo ""
    echo " No .env found. Copying .env.example..."
    cp .env.example .env
    echo "DEMO_MODE=true" >> .env
    echo " Running in DEMO MODE. Add API keys to .env for full AI mode."
fi

echo ""
echo " Starting ComposerIQ..."
echo " Open browser at: http://localhost:8000"
echo " Press Ctrl+C to stop."
echo ""

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
