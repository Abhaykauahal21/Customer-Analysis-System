@echo off
echo 🚀 Setting up Customer Analysis System...

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 16+
    exit /b 1
)

echo ✅ Prerequisites check passed

REM Setup ML Pipeline
echo 📊 Setting up ML Pipeline...
cd ml_pipeline
pip install -r requirements.txt
python pipeline.py
cd ..

REM Setup Backend
echo 🔧 Setting up Backend...
cd backend
pip install -r requirements.txt
cd ..

REM Setup Frontend
echo ⚛️  Setting up Frontend...
cd frontend
call npm install
echo REACT_APP_API_URL=http://localhost:5000 > .env
cd ..

echo ✅ Setup complete!
echo.
echo To start the application:
echo 1. Backend: cd backend ^&^& python app.py
echo 2. Frontend: cd frontend ^&^& npm start

pause

