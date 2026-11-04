#!/bin/bash

# Run script for RAG Knowledge Base

echo "🚀 Starting AI-Powered Knowledge Base..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the application
echo "🌐 Starting server on http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
echo ""
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

