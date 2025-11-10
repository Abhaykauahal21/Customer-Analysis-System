"""
Flask API for Customer Behavior Analysis System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# Configure CORS with explicit settings
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / 'ml_pipeline' / 'models'
RESULTS_DIR = BASE_DIR.parent / 'ml_pipeline' / 'results'

# Load models
scaler = None
pca = None
kmeans = None
best_model = None
models = {}

def load_models():
    """Load all trained models"""
    global scaler, pca, kmeans, best_model, models
    
    try:
        # Check if models directory exists
        if not MODELS_DIR.exists():
            print(f"❌ Models directory not found: {MODELS_DIR}")
            return False
        
        # Load required models
        required_models = {
            'scaler': 'scaler.pkl',
            'pca': 'pca.pkl',
            'kmeans': 'kmeans.pkl',
            'best_model': 'best_model.pkl'
        }
        
        loaded_models = {}
        for model_name, filename in required_models.items():
            filepath = MODELS_DIR / filename
            if not filepath.exists():
                print(f"❌ Required model file not found: {filepath}")
                return False
            try:
                loaded_models[model_name] = joblib.load(filepath)
                print(f"✅ Loaded {model_name}")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {str(e)}")
                return False
        
        scaler = loaded_models['scaler']
        pca = loaded_models['pca']
        kmeans = loaded_models['kmeans']
        best_model = loaded_models['best_model']
        
        # Load optional individual models
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'Naive Bayes': 'naive_bayes.pkl',
            'KNN': 'knn.pkl',
            'Random Forest': 'random_forest.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl',
            'XGBoost': 'xgboost.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = MODELS_DIR / filename
            if filepath.exists():
                try:
                    models[name] = joblib.load(filepath)
                    print(f"✅ Loaded {name}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not load {name}: {str(e)}")
            else:
                print(f"⚠️  Warning: Model file not found: {filename}")
        
        print("✅ All required models loaded successfully")
        return True
    except Exception as e:
        import traceback
        print(f"❌ Error loading models: {str(e)}")
        print(traceback.format_exc())
        return False

# Load models on startup
if not load_models():
    print("⚠️  Warning: Models not loaded. Run ml_pipeline/pipeline.py first.")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': best_model is not None
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict purchase likelihood for a customer"""
    try:
        if best_model is None or scaler is None or pca is None or kmeans is None:
            return jsonify({
                'error': 'Models not loaded. Please ensure ML pipeline has been run and models exist.'
            }), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided. Please send JSON data.'}), 400
        
        # Expected features (must match pipeline feature_columns)
        feature_columns = [
            'Age', 'Income', 'Kidhome', 'Teenhome', 'Recency',
            'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
            'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
            'NumWebVisitsMonth', 'Education_Encoded', 'Marital_Status_Encoded',
            'Country_Encoded', 'TotalSpending', 'TotalPurchases', 'SpendingScore',
            'TotalSpent', 'TotalQuantity', 'TransactionCount'
        ]
        
        # Validate input
        missing_features = [f for f in feature_columns if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {missing_features}',
                'required_features': feature_columns
            }), 400
        
        # Validate data types and ranges
        try:
            # Create feature array with proper type conversion
            features = np.array([[float(data.get(f, 0)) for f in feature_columns]], dtype=np.float64)
            
            # Check for NaN or Inf values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return jsonify({
                    'error': 'Invalid feature values detected (NaN or Inf). Please check your input data.'
                }), 400
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': f'Invalid data type in features: {str(e)}'
            }), 400
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get cluster assignment (same for all models)
        features_pca = pca.transform(features_scaled)
        cluster = int(kmeans.predict(features_pca)[0])
        cluster_center = kmeans.cluster_centers_[cluster]
        distance = float(np.linalg.norm(features_pca[0] - cluster_center))
        
        # Get predictions from ALL models
        CONFIDENCE_THRESHOLD = 0.6
        all_predictions = {}
        
        # Get prediction from best model (primary prediction)
        probability = None
        if hasattr(best_model, 'predict_proba'):
            try:
                probability = best_model.predict_proba(features_scaled)[0]
            except Exception as e:
                print(f"Warning: Could not get probabilities from best model: {e}")
        
        if probability is not None and len(probability) >= 2:
            purchase_prob = float(probability[1])
            no_purchase_prob = float(probability[0])
            
            if purchase_prob >= CONFIDENCE_THRESHOLD:
                prediction = 1  # Purchase
            else:
                prediction = 0  # No Purchase
        else:
            prediction = int(best_model.predict(features_scaled)[0])
            if probability is None:
                probability = np.array([0.5, 0.5]) if prediction == 0 else np.array([0.3, 0.7])
        
        if probability is None or len(probability) < 2:
            probability = np.array([0.5, 0.5])
        
        confidence = float(probability[1]) if prediction == 1 else float(probability[0])
        
        # Store best model prediction
        all_predictions['Best Model'] = {
            'prediction': int(prediction),
            'probability': {
                'no_purchase': float(probability[0]),
                'purchase': float(probability[1])
            },
            'confidence': float(confidence)
        }
        
        # Get predictions from all other loaded models
        for model_name, model in models.items():
            try:
                model_prediction = None
                model_probability = None
                
                # Try to get probabilities
                if hasattr(model, 'predict_proba'):
                    try:
                        model_probability = model.predict_proba(features_scaled)[0]
                    except Exception as e:
                        print(f"Warning: Could not get probabilities from {model_name}: {e}")
                
                # Get prediction
                if model_probability is not None and len(model_probability) >= 2:
                    model_purchase_prob = float(model_probability[1])
                    model_no_purchase_prob = float(model_probability[0])
                    
                    # Apply confidence threshold
                    if model_purchase_prob >= CONFIDENCE_THRESHOLD:
                        model_prediction = 1
                    else:
                        model_prediction = 0
                else:
                    model_prediction = int(model.predict(features_scaled)[0])
                    if model_probability is None:
                        model_probability = np.array([0.5, 0.5]) if model_prediction == 0 else np.array([0.3, 0.7])
                
                if model_probability is None or len(model_probability) < 2:
                    model_probability = np.array([0.5, 0.5])
                
                model_confidence = float(model_probability[1]) if model_prediction == 1 else float(model_probability[0])
                
                all_predictions[model_name] = {
                    'prediction': int(model_prediction),
                    'probability': {
                        'no_purchase': float(model_probability[0]),
                        'purchase': float(model_probability[1])
                    },
                    'confidence': float(model_confidence)
                }
            except Exception as e:
                print(f"Warning: Error getting prediction from {model_name}: {str(e)}")
                continue
        
        # Calculate consensus (majority vote)
        predictions_list = [pred['prediction'] for pred in all_predictions.values()]
        consensus_prediction = 1 if sum(predictions_list) > len(predictions_list) / 2 else 0
        consensus_count = sum(predictions_list)
        total_models = len(all_predictions)
        
        response = {
            'prediction': int(prediction),  # Best model prediction (primary)
            'probability': {
                'no_purchase': float(probability[0]),
                'purchase': float(probability[1])
            },
            'confidence': float(confidence),
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'cluster': cluster,
            'cluster_center_distance': distance,
            'pca_coordinates': {
                'x': float(features_pca[0, 0]),
                'y': float(features_pca[0, 1])
            },
            'all_predictions': all_predictions,  # Predictions from all models
            'consensus': {
                'prediction': int(consensus_prediction),
                'vote_count': consensus_count,
                'total_models': total_models,
                'agreement': f"{consensus_count}/{total_models} models predict {'Purchase' if consensus_prediction == 1 else 'No Purchase'}"
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /api/predict: {error_trace}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'type': type(e).__name__
        }), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters():
    """Get PCA and K-Means visualization data"""
    try:
        results_file = RESULTS_DIR / 'visualization_data.json'
        
        if not results_file.exists():
            return jsonify({
                'error': 'Visualization data not found. Run ML pipeline first.',
                'path': str(results_file)
            }), 404
        
        import json
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'clusters' not in data:
            return jsonify({
                'error': 'Invalid visualization data format. Missing clusters key.'
            }), 500
        
        return jsonify(data['clusters']), 200
        
    except json.JSONDecodeError as e:
        return jsonify({
            'error': f'Failed to parse visualization data: {str(e)}'
        }), 500
    except Exception as e:
        import traceback
        print(f"Error in /api/clusters: {traceback.format_exc()}")
        return jsonify({
            'error': f'Failed to load cluster data: {str(e)}'
        }), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get model performance metrics"""
    try:
        results_file = RESULTS_DIR / 'visualization_data.json'
        
        if not results_file.exists():
            return jsonify({
                'error': 'Model metrics not found. Run ML pipeline first.',
                'path': str(results_file)
            }), 404
        
        import json
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'models' not in data or 'model_metrics' not in data:
            return jsonify({
                'error': 'Invalid visualization data format. Missing models or model_metrics keys.'
            }), 500
        
        return jsonify({
            'models': data.get('models', []),
            'detailed_metrics': data.get('model_metrics', {}),
            'summary': data.get('summary', {})
        }), 200
        
    except json.JSONDecodeError as e:
        return jsonify({
            'error': f'Failed to parse model metrics: {str(e)}'
        }), 500
    except Exception as e:
        import traceback
        print(f"Error in /api/models: {traceback.format_exc()}")
        return jsonify({
            'error': f'Failed to load model metrics: {str(e)}'
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get feature information for prediction form"""
    return jsonify({
        'features': [
            {'name': 'Age', 'type': 'number', 'min': 18, 'max': 100},
            {'name': 'Income', 'type': 'number', 'min': 0, 'max': 200000},
            {'name': 'Kidhome', 'type': 'number', 'min': 0, 'max': 3},
            {'name': 'Teenhome', 'type': 'number', 'min': 0, 'max': 3},
            {'name': 'Recency', 'type': 'number', 'min': 0, 'max': 365},
            {'name': 'MntWines', 'type': 'number', 'min': 0, 'max': 2000},
            {'name': 'MntFruits', 'type': 'number', 'min': 0, 'max': 500},
            {'name': 'MntMeatProducts', 'type': 'number', 'min': 0, 'max': 2000},
            {'name': 'MntFishProducts', 'type': 'number', 'min': 0, 'max': 500},
            {'name': 'MntSweetProducts', 'type': 'number', 'min': 0, 'max': 500},
            {'name': 'MntGoldProds', 'type': 'number', 'min': 0, 'max': 1000},
            {'name': 'NumDealsPurchases', 'type': 'number', 'min': 0, 'max': 20},
            {'name': 'NumWebPurchases', 'type': 'number', 'min': 0, 'max': 30},
            {'name': 'NumCatalogPurchases', 'type': 'number', 'min': 0, 'max': 30},
            {'name': 'NumStorePurchases', 'type': 'number', 'min': 0, 'max': 30},
            {'name': 'NumWebVisitsMonth', 'type': 'number', 'min': 0, 'max': 30},
            {'name': 'Education_Encoded', 'type': 'number', 'min': 0, 'max': 4, 'description': '0=Basic, 1=2n Cycle, 2=Graduation, 3=Master, 4=PhD'},
            {'name': 'Marital_Status_Encoded', 'type': 'number', 'min': 0, 'max': 4, 'description': '0=Divorced, 1=Married, 2=Single, 3=Together, 4=Widow'},
            {'name': 'Country_Encoded', 'type': 'number', 'min': 0, 'max': 5, 'description': '0=Canada, 1=France, 2=Germany, 3=Spain, 4=UK, 5=USA'},
            {'name': 'TotalSpending', 'type': 'number', 'min': 0, 'max': 10000},
            {'name': 'TotalPurchases', 'type': 'number', 'min': 0, 'max': 100},
            {'name': 'SpendingScore', 'type': 'number', 'min': 0, 'max': 10},
            {'name': 'TotalSpent', 'type': 'number', 'min': 0, 'max': 50000},
            {'name': 'TotalQuantity', 'type': 'number', 'min': 0, 'max': 10000},
            {'name': 'TransactionCount', 'type': 'number', 'min': 0, 'max': 1000}
        ]
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)

