#!/bin/bash

# Setup script for RAG Knowledge Base

echo "🚀 Setting up AI-Powered Knowledge Base..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your OpenAI API key!"
    echo "   Open .env and set: OPENAI_API_KEY=your_actual_api_key"
    echo ""
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/chroma_db
mkdir -p data/uploads

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the application: python -m uvicorn app.main:app --reload"
echo "4. Open http://localhost:8000 in your browser"
echo ""

