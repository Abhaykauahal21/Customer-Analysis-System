# System Architecture

## Overview

The Customer Behavior and Sales Analysis System is a full-stack application that combines machine learning, backend API, and frontend visualization to analyze customer behavior and predict purchase likelihood.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (React.js)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │Dashboard │  │Segmentation│ │Prediction│  │ Results  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│         │            │             │              │         │
│         └────────────┴─────────────┴──────────────┘         │
│                            │                                 │
│                    Axios HTTP Client                         │
└────────────────────────────┼─────────────────────────────────┘
                             │
                             │ REST API
                             │
┌────────────────────────────┼─────────────────────────────────┐
│                    BACKEND (Flask)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ /predict │  │/clusters │  │ /models  │  │ /features│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│         │            │             │              │         │
│         └────────────┴─────────────┴──────────────┘         │
│                            │                                 │
│                    Model Service Layer                       │
└────────────────────────────┼─────────────────────────────────┘
                             │
                             │ Load Models
                             │
┌────────────────────────────┼─────────────────────────────────┐
│              ML PIPELINE (Python)                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   PCA    │  │ K-Means  │  │Logistic  │  │Decision  │  │
│  │          │  │          │  │Regression│  │  Tree    │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│  ┌──────────┐  ┌──────────┐                                │
│  │  Naive   │  │   KNN    │                                │
│  │  Bayes   │  │          │                                │
│  └──────────┘  └──────────┘                                │
│         │            │             │              │         │
│         └────────────┴─────────────┴──────────────┘         │
│                            │                                 │
│                    Trained Models (.pkl)                     │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Training Phase
```
Raw Data → Data Cleaning → Feature Engineering → 
PCA → K-Means → Model Training → Model Evaluation → 
Save Models → Generate Visualization Data
```

### 2. Prediction Phase
```
User Input → API Request → Feature Scaling → 
PCA Transform → K-Means Assignment → 
Model Prediction → Return Results
```

### 3. Visualization Phase
```
API Request → Load Visualization Data → 
Return JSON → Frontend Rendering → Charts
```

## Technology Stack

### Frontend
- **React.js 18** - UI framework
- **React Router** - Navigation
- **TailwindCSS** - Styling
- **Recharts** - Data visualization
- **Axios** - HTTP client

### Backend
- **Flask** - Web framework
- **Flask-CORS** - Cross-origin resource sharing
- **scikit-learn** - Machine learning
- **joblib** - Model serialization
- **gunicorn** - Production server

### ML Pipeline
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - ML algorithms
- **matplotlib/seaborn** - Visualization

## File Structure

```
customer-analysis-system/
├── backend/
│   ├── app.py                 # Flask application
│   ├── requirements.txt       # Python dependencies
│   └── Procfile              # Deployment config
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   ├── services/         # API services
│   │   └── App.js            # Main app component
│   ├── public/
│   └── package.json
├── ml_pipeline/
│   ├── pipeline.py           # ML pipeline script
│   ├── data/                 # Dataset directory
│   ├── models/               # Trained models (.pkl)
│   └── results/              # Visualization data
└── docs/                     # Documentation
```

## Model Pipeline

### 1. Data Processing
- Load customer and retail datasets
- Clean missing values and duplicates
- Merge on CustomerID and Country
- Feature encoding and scaling

### 2. Dimensionality Reduction
- **PCA**: Reduce to 2 principal components
- Explained variance tracking

### 3. Clustering
- **K-Means**: Optimal K via elbow method
- Cluster assignment and centers

### 4. Classification
- **Logistic Regression**: Linear classifier
- **Decision Tree**: Non-linear classifier
- **Naive Bayes**: Probabilistic classifier
- **KNN**: Instance-based classifier

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC AUC Score
- Confusion Matrix
- Best model selection

## Security Considerations

- CORS enabled for frontend-backend communication
- Environment variables for sensitive data
- Input validation on API endpoints
- Error handling and logging

## Scalability

- Stateless API design
- Model caching in memory
- Efficient data structures
- Optimized ML pipeline

## Deployment

- **Backend**: Render/PythonAnywhere
- **Frontend**: Netlify/Vercel
- **Models**: Version controlled or cloud storage
- **Database**: Optional for production (currently file-based)

