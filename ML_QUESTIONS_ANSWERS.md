# 🤖 Machine Learning Questions & Answers

## 📋 Table of Contents
1. [Models Overview](#models-overview)
2. [Data Splitting](#data-splitting)
3. [Feature Engineering](#feature-engineering)
4. [Dimensionality Reduction](#dimensionality-reduction)
5. [Clustering](#clustering)
6. [Model Training](#model-training)
7. [Model Evaluation](#model-evaluation)
8. [Hyperparameter Tuning](#hyperparameter-tuning)
9. [Model Selection](#model-selection)
10. [Data Preprocessing](#data-preprocessing)

---

## 🎯 Models Overview

### Q1: Which machine learning models are used in this project?

**Answer:**
The project uses **4 classification models** to predict customer purchase likelihood:

1. **Logistic Regression**
   - Type: Linear classifier
   - Algorithm: Maximum likelihood estimation
   - Use case: Baseline linear model for binary classification

2. **Decision Tree**
   - Type: Non-linear tree-based classifier
   - Max depth: 10
   - Use case: Captures non-linear relationships and feature interactions

3. **Naive Bayes (Gaussian)**
   - Type: Probabilistic classifier
   - Assumption: Features are conditionally independent
   - Use case: Fast training and good for probabilistic predictions

4. **K-Nearest Neighbors (KNN)**
   - Type: Instance-based/lazy learning classifier
   - Optimal K: Found via GridSearchCV (typically 3-21)
   - Use case: Non-parametric classification based on similarity

**Additional Models:**
- **PCA (Principal Component Analysis)**: For dimensionality reduction
- **K-Means Clustering**: For customer segmentation

---

## 📊 Data Splitting

### Q2: What is the training and testing data ratio?

**Answer:**
- **Training Data**: **80%** of the dataset
- **Testing Data**: **20%** of the dataset
- **Split Method**: `train_test_split` with `test_size=0.2`
- **Random State**: `42` (for reproducibility)
- **Stratification**: Yes, using `stratify=y` to maintain class distribution

**Example:**
- If you have 2,000 customers:
  - Training set: 1,600 customers (80%)
  - Testing set: 400 customers (20%)

**Code Reference:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Q3: Why is stratification used in train-test split?

**Answer:**
Stratification (`stratify=y`) ensures that both training and testing sets have the same proportion of classes (purchase vs. no purchase). This prevents:
- Imbalanced class distribution in test set
- Biased model evaluation
- Poor generalization estimates

---

## 🔧 Feature Engineering

### Q4: How many features are used for model training?

**Answer:**
The model uses **25 features** in total:

**Demographic Features:**
- Age (calculated from Year_Birth)
- Income
- Kidhome
- Teenhome
- Recency

**Spending Features:**
- MntWines
- MntFruits
- MntMeatProducts
- MntFishProducts
- MntSweetProducts
- MntGoldProds

**Purchase Behavior Features:**
- NumDealsPurchases
- NumWebPurchases
- NumCatalogPurchases
- NumStorePurchases
- NumWebVisitsMonth

**Encoded Categorical Features:**
- Education_Encoded (Label Encoded)
- Marital_Status_Encoded (Label Encoded)
- Country_Encoded (Label Encoded)

**Derived Features:**
- TotalSpending (sum of all spending categories)
- TotalPurchases (sum of all purchase counts)
- SpendingScore (TotalSpending / 1000)
- TotalSpent (from retail data)
- TotalQuantity (from retail data)
- TransactionCount (from retail data)

### Q5: How are categorical features handled?

**Answer:**
Categorical features are encoded using **Label Encoding**:
- **Education**: Encoded to numeric values (0, 1, 2, 3, 4)
- **Marital_Status**: Encoded to numeric values
- **Country**: Encoded to numeric values

**Label Encoder** assigns a unique integer to each category, making them suitable for machine learning algorithms.

### Q6: Is feature scaling applied?

**Answer:**
Yes! **StandardScaler** is used for feature scaling:
- **Method**: Standardization (Z-score normalization)
- **Formula**: `(x - mean) / std`
- **Applied to**: All numeric features before training
- **Purpose**: Ensures all features are on the same scale, improving model performance

**When Applied:**
- Before PCA transformation
- Before training all classification models
- Before making predictions

---

## 📉 Dimensionality Reduction

### Q7: What is PCA and how is it used?

**Answer:**
**PCA (Principal Component Analysis)** is used for dimensionality reduction:

- **Components**: 2 principal components (for 2D visualization)
- **Purpose**: Reduce high-dimensional feature space to 2 dimensions
- **Variance Explained**: Typically 20-30% (varies by dataset)
- **Random State**: 42 (for reproducibility)

**Process:**
1. Features are first standardized using StandardScaler
2. PCA transforms 25 features → 2 principal components
3. Used for visualization and K-Means clustering

**Code:**
```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
```

### Q8: Why use only 2 PCA components?

**Answer:**
- **Visualization**: 2D scatter plots are easy to visualize and interpret
- **Clustering**: K-Means works well in 2D space
- **Trade-off**: Some information loss is acceptable for visualization purposes
- **Alternative**: Could use more components (e.g., 3-5) for better variance retention

---

## 🎯 Clustering

### Q9: How does K-Means clustering work?

**Answer:**
**K-Means** is used for customer segmentation:

- **Number of Clusters (K)**: **4 clusters**
- **Method**: Elbow method to determine optimal K
- **Random State**: 42
- **Initializations**: 10 (n_init=10)
- **Input**: PCA-transformed data (2D)

**Process:**
1. Elbow method evaluates K from 2 to 10
2. Optimal K = 4 (based on inertia/within-cluster sum of squares)
3. K-Means assigns each customer to one of 4 clusters
4. Cluster centers are calculated

**Output:**
- 4 customer segments
- Cluster labels for each customer
- Cluster centers (for visualization)

### Q10: How is the optimal number of clusters determined?

**Answer:**
The **Elbow Method** is used:
- Tests K values from 2 to 10
- Calculates inertia (within-cluster sum of squares) for each K
- Optimal K = 4 (point where adding more clusters doesn't significantly reduce inertia)
- Visual inspection of the "elbow" in the inertia plot

---

## 🚀 Model Training

### Q11: How are the models trained?

**Answer:**
Models are trained using the following process:

1. **Data Preparation**:
   - Features are standardized using StandardScaler
   - Training on 80% of data (X_train, y_train)

2. **Model Training**:
   - Each model is fit on training data
   - Models are trained sequentially

3. **Hyperparameter Tuning** (for KNN):
   - GridSearchCV with 5-fold cross-validation
   - Tests K values: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21
   - Selects best K based on accuracy

4. **Training Time**: Varies by model complexity

**Code Flow:**
```python
# Scale training data
X_train_scaled = scaler.transform(X_train)

# Train each model
for model in models:
    model.fit(X_train_scaled, y_train)
```

### Q12: What are the hyperparameters for each model?

**Answer:**

**1. Logistic Regression:**
- `random_state=42`: For reproducibility
- `max_iter=1000`: Maximum iterations for convergence
- Default solver: 'lbfgs'

**2. Decision Tree:**
- `random_state=42`: For reproducibility
- `max_depth=10`: Maximum tree depth (prevents overfitting)
- Default criterion: 'gini'

**3. Naive Bayes (Gaussian):**
- No hyperparameters set (uses defaults)
- Assumes Gaussian distribution of features

**4. K-Nearest Neighbors:**
- `n_neighbors`: Optimized via GridSearchCV (typically 3-21)
- `cv=5`: 5-fold cross-validation for hyperparameter search
- `scoring='accuracy'`: Metric for selecting best K

**5. K-Means:**
- `n_clusters=4`: Number of clusters
- `random_state=42`: For reproducibility
- `n_init=10`: Number of initializations

---

## 📈 Model Evaluation

### Q13: What evaluation metrics are used?

**Answer:**
The project uses **6 evaluation metrics**:

1. **Accuracy**
   - Formula: `(TP + TN) / (TP + TN + FP + FN)`
   - Meaning: Overall correctness of predictions

2. **Precision**
   - Formula: `TP / (TP + FP)`
   - Meaning: Of predicted positives, how many are actually positive

3. **Recall (Sensitivity)**
   - Formula: `TP / (TP + FN)`
   - Meaning: Of actual positives, how many are correctly identified

4. **F1-Score**
   - Formula: `2 * (Precision * Recall) / (Precision + Recall)`
   - Meaning: Harmonic mean of precision and recall
   - **Used for model selection** (best model has highest F1)

5. **ROC-AUC Score**
   - Formula: Area under ROC curve
   - Meaning: Model's ability to distinguish between classes
   - Range: 0 to 1 (higher is better)

6. **Confusion Matrix**
   - Shows: TP, TN, FP, FN
   - Meaning: Detailed breakdown of predictions

### Q14: How is the best model selected?

**Answer:**
The **best model is selected based on F1-Score**:

- **Selection Criteria**: Highest F1-Score
- **Reason**: F1-Score balances precision and recall
- **Process**: 
  1. All models are evaluated on test set
  2. F1-Scores are compared
  3. Model with highest F1-Score is selected as "best_model"
  4. Best model is saved as `best_model.pkl`

**Code:**
```python
best_model_name = max(model_metrics.keys(), 
                     key=lambda x: model_metrics[x]['f1_score'])
best_model = models[best_model_name]
```

---

## 🔍 Hyperparameter Tuning

### Q15: How is hyperparameter tuning performed?

**Answer:**
**GridSearchCV** is used for KNN hyperparameter tuning:

- **Parameter**: `n_neighbors` (K value)
- **Search Space**: Range from 3 to 21 (step of 2)
- **Cross-Validation**: 5-fold CV
- **Scoring Metric**: Accuracy
- **Process**:
  1. Tests all K values: [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
  2. For each K, performs 5-fold cross-validation
  3. Selects K with highest average accuracy
  4. Trains final model with optimal K

**Why only KNN?**
- KNN is sensitive to K value
- Other models have fewer critical hyperparameters
- Decision Tree max_depth is set to prevent overfitting

---

## 🧹 Data Preprocessing

### Q16: What data preprocessing steps are applied?

**Answer:**
Comprehensive preprocessing pipeline:

1. **Data Loading**:
   - Loads customer personality dataset
   - Loads/creates retail transaction dataset

2. **Data Cleaning**:
   - **Missing Values**: 
     - Numeric: Filled with median
     - Categorical: Filled with mode
   - **Duplicates**: Removed
   - **Invalid Values**: Filtered (negative quantities, prices)

3. **Data Merging**:
   - Merges customer and retail data on CustomerID
   - Aggregates retail transactions per customer

4. **Feature Engineering**:
   - Calculates Age from Year_Birth
   - Calculates TotalSpending
   - Calculates TotalPurchases
   - Creates SpendingScore

5. **Encoding**:
   - Label encoding for categorical features

6. **Scaling**:
   - StandardScaler for all numeric features

7. **Target Creation**:
   - Uses 'Response' column if available
   - Otherwise creates based on spending behavior

---

## 📊 Dataset Information

### Q17: What is the dataset size?

**Answer:**
- **Source**: Kaggle Customer Personality Analysis dataset
- **Original Size**: ~2,240 customers (varies)
- **After Cleaning**: Typically 2,000+ customers
- **Features**: 25 features per customer
- **Target**: Binary (Purchase Likelihood: 0 or 1)

### Q18: What is the class distribution?

**Answer:**
Class distribution varies but typically:
- **Class 0 (No Purchase)**: ~60-65%
- **Class 1 (Purchase)**: ~35-40%

**Note**: Stratified split ensures this distribution is maintained in train/test sets.

---

## 🎓 Model Performance

### Q19: What are typical model performance metrics?

**Answer:**
Performance varies by dataset, but typical ranges:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.60-0.65 | 0.40-0.45 | 0.15-0.25 | 0.20-0.30 | 0.55-0.60 |
| Decision Tree | 0.55-0.60 | 0.40-0.45 | 0.25-0.35 | 0.30-0.40 | 0.50-0.55 |
| Naive Bayes | 0.58-0.62 | 0.38-0.42 | 0.10-0.15 | 0.15-0.20 | 0.50-0.55 |
| KNN | 0.54-0.58 | 0.25-0.30 | 0.10-0.15 | 0.14-0.20 | 0.49-0.52 |

**Note**: Actual performance depends on the dataset quality and size.

### Q20: Why might model performance be moderate?

**Answer:**
Several factors contribute:

1. **Imbalanced Dataset**: More "no purchase" than "purchase" cases
2. **Complex Problem**: Purchase behavior is influenced by many factors
3. **Limited Features**: May not capture all purchase drivers
4. **Data Quality**: Real-world data often has noise
5. **Non-linear Relationships**: Some patterns may be hard to capture

**Improvement Strategies:**
- Collect more features
- Use ensemble methods (Random Forest, XGBoost)
- Handle class imbalance (SMOTE, class weights)
- Feature selection
- More data

---

## 🔄 Model Persistence

### Q21: How are models saved and loaded?

**Answer:**
Models are saved using **Joblib**:

- **Format**: `.pkl` files (Python pickle format)
- **Location**: `ml_pipeline/models/`
- **Saved Models**:
  - `scaler.pkl` - StandardScaler
  - `pca.pkl` - PCA transformer
  - `kmeans.pkl` - K-Means model
  - `best_model.pkl` - Best performing classifier
  - `logistic_regression.pkl`
  - `decision_tree.pkl`
  - `naive_bayes.pkl`
  - `knn.pkl`

**Loading:**
- Models are loaded at backend startup
- Used for real-time predictions via API

---

## 🎯 Prediction Process

### Q22: How does the prediction workflow work?

**Answer:**
Complete prediction pipeline:

1. **Input**: Customer features (25 features)
2. **Scaling**: Apply StandardScaler transformation
3. **PCA**: Transform to 2D (for clustering)
4. **Clustering**: Assign customer to cluster using K-Means
5. **Prediction**: Use best model to predict purchase likelihood
6. **Probability**: Get probability scores for both classes
7. **Output**: 
   - Prediction (0 or 1)
   - Probabilities
   - Cluster assignment
   - PCA coordinates

**Code Flow:**
```python
# Scale features
features_scaled = scaler.transform(features)

# Predict
prediction = best_model.predict(features_scaled)

# Get cluster
features_pca = pca.transform(features_scaled)
cluster = kmeans.predict(features_pca)
```

---

## 📚 Additional Questions

### Q23: What is the random_state parameter and why is it used?

**Answer:**
- **Value**: `random_state=42`
- **Purpose**: Ensures reproducibility
- **Effect**: Same random seed produces same results
- **Used in**: 
  - Train-test split
  - Model initialization
  - K-Means clustering
  - Data generation

**Why 42?**: Common convention in data science (reference to "Hitchhiker's Guide to the Galaxy")

### Q24: What is cross-validation and how is it used?

**Answer:**
**5-Fold Cross-Validation** is used for KNN hyperparameter tuning:

- **Process**: 
  1. Split data into 5 folds
  2. Train on 4 folds, test on 1 fold
  3. Repeat 5 times (each fold used as test once)
  4. Average the results
- **Purpose**: More robust hyperparameter selection
- **Benefit**: Reduces overfitting to a single train-test split

### Q25: What libraries and frameworks are used?

**Answer:**
**Core ML Libraries:**
- `scikit-learn`: All ML models and preprocessing
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `joblib`: Model serialization

**Visualization:**
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization

**Data Processing:**
- `LabelEncoder`: Categorical encoding
- `StandardScaler`: Feature scaling
- `PCA`: Dimensionality reduction
- `KMeans`: Clustering

---

## 📝 Summary

### Key Takeaways:

1. **4 Classification Models**: Logistic Regression, Decision Tree, Naive Bayes, KNN
2. **Train/Test Split**: 80/20 with stratification
3. **25 Features**: Demographic, spending, purchase behavior, and derived features
4. **Feature Scaling**: StandardScaler applied to all features
5. **Dimensionality Reduction**: PCA with 2 components
6. **Clustering**: K-Means with 4 clusters
7. **Hyperparameter Tuning**: GridSearchCV for KNN (K=3 to 21)
8. **Model Selection**: Based on F1-Score
9. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
10. **Reproducibility**: Random state = 42 throughout

---

## 🔗 Related Documentation

- [README.md](../README.md) - Project overview
- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) - System architecture
- [API_DOCUMENTATION.md](../docs/API_DOCUMENTATION.md) - API endpoints
- [DEBUGGING_REPORT.md](../DEBUGGING_REPORT.md) - Debugging details

---

**Last Updated**: After ML pipeline implementation  
**Questions?**: Refer to the code in `ml_pipeline/pipeline.py` for implementation details

