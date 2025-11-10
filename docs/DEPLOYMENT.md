# Deployment Guide

## Backend Deployment (Render/PythonAnywhere)

### Option 1: Render Deployment

1. **Create a Render account** at https://render.com

2. **Create a new Web Service**:
   - Connect your GitHub repository
   - Select the `backend` directory as root
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`

3. **Environment Variables**:
   ```
   FLASK_ENV=production
   PORT=5000
   SECRET_KEY=your-secret-key-here
   ```

4. **Important**: Make sure to run the ML pipeline first and commit the `models/` directory to your repository, or upload models separately.

### Option 2: PythonAnywhere Deployment

1. **Create a PythonAnywhere account** at https://www.pythonanywhere.com

2. **Upload your code**:
   - Use the Files tab to upload your backend code
   - Upload the ML models from `ml_pipeline/models/`

3. **Create a Web App**:
   - Go to Web tab
   - Click "Add a new web app"
   - Select Flask and Python 3.8+
   - Set source code directory to your backend folder

4. **Configure WSGI file**:
   ```python
   import sys
   path = '/home/yourusername/backend'
   if path not in sys.path:
       sys.path.append(path)
   
   from app import app as application
   ```

5. **Set environment variables** in the Web app configuration

## Frontend Deployment (Netlify/Vercel)

### Option 1: Netlify Deployment

1. **Create a Netlify account** at https://netlify.com

2. **Deploy from Git**:
   - Connect your GitHub repository
   - Base directory: `frontend`
   - Build command: `npm install && npm run build`
   - Publish directory: `frontend/build`

3. **Environment Variables**:
   ```
   REACT_APP_API_URL=https://your-backend-url.onrender.com
   ```

4. **Deploy** and get your frontend URL

### Option 2: Vercel Deployment

1. **Create a Vercel account** at https://vercel.com

2. **Import Project**:
   - Connect your GitHub repository
   - Root directory: `frontend`
   - Framework preset: Create React App

3. **Environment Variables**:
   ```
   REACT_APP_API_URL=https://your-backend-url.onrender.com
   ```

4. **Deploy** and get your frontend URL

## Post-Deployment Checklist

- [ ] Backend API is accessible
- [ ] Frontend can connect to backend (check CORS)
- [ ] ML models are loaded correctly
- [ ] All API endpoints are working
- [ ] Environment variables are set correctly

