# Quick Start Guide

Get the Customer Analysis System up and running in 5 minutes!

## Prerequisites Check

```bash
# Check Python version (need 3.8+)
python --version

# Check Node.js version (need 16+)
node --version

# Check npm
npm --version
```

## Windows Quick Start

```bash
# Run the setup script
setup.bat

# Then start backend (in new terminal)
cd backend
python app.py

# Then start frontend (in new terminal)
cd frontend
npm start
```

## Linux/Mac Quick Start

```bash
# Make setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh

# Then start backend (in new terminal)
cd backend
python app.py

# Then start frontend (in new terminal)
cd frontend
npm start
```

## Manual Setup (Step by Step)

### 1. ML Pipeline (One-time setup)

```bash
cd ml_pipeline
pip install -r requirements.txt
python pipeline.py
```

**Expected output:**
- ✅ Models saved to `ml_pipeline/models/`
- ✅ Visualization data saved to `ml_pipeline/results/`

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Expected output:**
```
✅ All models loaded successfully
 * Running on http://0.0.0.0:5000
```

### 3. Frontend

```bash
cd frontend
npm install
echo "REACT_APP_API_URL=http://localhost:5000" > .env
npm start
```

**Expected output:**
- Browser opens at `http://localhost:3000`

## Verify Installation

1. **Backend Health Check:**
   - Open browser: `http://localhost:5000/api/health`
   - Should return: `{"status": "healthy", "models_loaded": true}`

2. **Frontend:**
   - Should see dashboard at `http://localhost:3000`
   - Navigate to different pages using sidebar

3. **Test Prediction:**
   - Go to "Prediction" page
   - Fill in sample data
   - Click "Predict Purchase Likelihood"
   - Should see results page

## Troubleshooting

### Issue: Models not loading
**Solution:** Run `python ml_pipeline/pipeline.py` first

### Issue: CORS errors
**Solution:** Make sure Flask-CORS is installed: `pip install Flask-CORS`

### Issue: Frontend can't connect to backend
**Solution:** 
- Check backend is running on port 5000
- Verify `.env` file has correct API URL
- Check browser console for errors

### Issue: Port already in use
**Solution:**
- Backend: Change port in `backend/app.py` or set `PORT` environment variable
- Frontend: React will prompt to use different port

## Next Steps

1. Explore the Dashboard to see model performance
2. Check Customer Segmentation visualization
3. Try making predictions with different customer profiles
4. Review the documentation in `docs/` folder

## Production Deployment

See [Deployment Guide](docs/DEPLOYMENT.md) for deploying to:
- Render/PythonAnywhere (Backend)
- Netlify/Vercel (Frontend)

---

**Need help?** Check the main [README.md](README.md) or open an issue.

