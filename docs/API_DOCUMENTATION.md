# API Documentation

## Base URL
```
http://localhost:5000 (Development)
https://your-backend-url.onrender.com (Production)
```

## Endpoints

### 1. Health Check
**GET** `/api/health`

Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

---

### 2. Predict Purchase Likelihood
**POST** `/api/predict`

Predict purchase likelihood for a customer.

**Request Body:**
```json
{
  "Age": 45,
  "Income": 50000,
  "Kidhome": 1,
  "Teenhome": 0,
  "Recency": 30,
  "MntWines": 200,
  "MntFruits": 50,
  "MntMeatProducts": 150,
  "MntFishProducts": 40,
  "MntSweetProducts": 30,
  "MntGoldProds": 60,
  "NumDealsPurchases": 3,
  "NumWebPurchases": 5,
  "NumCatalogPurchases": 3,
  "NumStorePurchases": 8,
  "NumWebVisitsMonth": 6,
  "Education_Encoded": 2,
  "Marital_Status_Encoded": 1,
  "Country_Encoded": 5,
  "TotalSpending": 530,
  "TotalPurchases": 19,
  "SpendingScore": 5.3,
  "TotalSpent": 5000,
  "TotalQuantity": 100,
  "TransactionCount": 20
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": {
    "no_purchase": 0.25,
    "purchase": 0.75
  },
  "cluster": 2,
  "cluster_center_distance": 1.234,
  "pca_coordinates": {
    "x": 0.123,
    "y": -0.456
  }
}
```

---

### 3. Get Cluster Data
**GET** `/api/clusters`

Get PCA and K-Means visualization data.

**Response:**
```json
{
  "pca_data": [
    {
      "x": 0.123,
      "y": -0.456,
      "cluster": 0
    },
    ...
  ],
  "cluster_centers": [
    {
      "x": 0.5,
      "y": 0.3,
      "cluster": 0
    },
    ...
  ],
  "explained_variance": [0.45, 0.30]
}
```

---

### 4. Get Model Metrics
**GET** `/api/models`

Get performance metrics for all trained models.

**Response:**
```json
{
  "models": [
    {
      "name": "Logistic Regression",
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.88,
      "f1_score": 0.85,
      "roc_auc": 0.91
    },
    ...
  ],
  "detailed_metrics": {
    "Logistic Regression": {
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.88,
      "f1_score": 0.85,
      "roc_auc": 0.91,
      "confusion_matrix": [[100, 20], [15, 65]]
    },
    ...
  }
}
```

---

### 5. Get Feature Information
**GET** `/api/features`

Get feature definitions for the prediction form.

**Response:**
```json
{
  "features": [
    {
      "name": "Age",
      "type": "number",
      "min": 18,
      "max": 100
    },
    ...
  ]
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error message here"
}
```

**Status Codes:**
- `200` - Success
- `400` - Bad Request (missing/invalid parameters)
- `404` - Not Found (resource doesn't exist)
- `500` - Internal Server Error

