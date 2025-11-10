#!/bin/bash

echo "🚀 Setting up Customer Analysis System..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Setup ML Pipeline
echo "📊 Setting up ML Pipeline..."
cd ml_pipeline
pip install -r requirements.txt
python pipeline.py
cd ..

# Setup Backend
echo "🔧 Setting up Backend..."
cd backend
pip install -r requirements.txt
cd ..

# Setup Frontend
echo "⚛️  Setting up Frontend..."
cd frontend
npm install
echo "REACT_APP_API_URL=http://localhost:5000" > .env
cd ..

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "1. Backend: cd backend && python app.py"
echo "2. Frontend: cd frontend && npm start"

