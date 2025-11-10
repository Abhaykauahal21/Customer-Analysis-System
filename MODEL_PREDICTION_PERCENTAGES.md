# 📊 Model Prediction Performance Percentages

## 🎯 Current Model Performance

Based on the trained models in your project, here are the **actual prediction percentages**:

---

## 📈 Model Performance Summary

### 🏆 **Best Model: Decision Tree**
- **Selected Based On**: Highest F1-Score (34.09%)
- **Used For**: Production predictions via API

| Metric | Percentage |
|--------|------------|
| **Accuracy** | **56.50%** |
| **Precision** | **41.28%** |
| **Recall** | **29.03%** |
| **F1-Score** | **34.09%** |
| **ROC-AUC** | **52.19%** |

---

## 📊 All Models Performance Comparison

### 1. **Decision Tree** (Best Model) 🏆
- **Accuracy**: **56.50%**
- **Precision**: **41.28%**
- **Recall**: **29.03%**
- **F1-Score**: **34.09%**
- **ROC-AUC**: **52.19%**

### 2. **Naive Bayes**
- **Accuracy**: **59.00%** ⬆️ (Highest)
- **Precision**: **39.53%**
- **Recall**: **10.97%**
- **F1-Score**: **17.17%**
- **ROC-AUC**: **52.16%**

### 3. **Logistic Regression**
- **Accuracy**: **61.00%** ⬆️⬆️ (Highest)
- **Precision**: **42.86%** ⬆️ (Highest)
- **Recall**: **1.94%** ⬇️ (Lowest)
- **F1-Score**: **3.70%** ⬇️ (Lowest)
- **ROC-AUC**: **55.54%** ⬆️ (Highest)

### 4. **KNN (K-Nearest Neighbors)**
- **Accuracy**: **54.00%**
- **Precision**: **26.23%**
- **Recall**: **10.32%**
- **F1-Score**: **14.81%**
- **ROC-AUC**: **49.47%**

---

## 📝 What These Percentages Mean

### **Accuracy (56.50% for Best Model)**
- **Meaning**: Out of 100 predictions, **56.5 are correct**
- **Interpretation**: The model correctly predicts purchase likelihood 56.5% of the time
- **Note**: This is the overall correctness rate

### **Precision (41.28% for Best Model)**
- **Meaning**: When the model predicts "Purchase", it's correct **41.28% of the time**
- **Interpretation**: Out of 100 "Purchase" predictions, about 41 are actually purchases
- **Business Impact**: Helps avoid false positives (predicting purchase when customer won't buy)

### **Recall (29.03% for Best Model)**
- **Meaning**: The model identifies **29.03% of actual purchasers**
- **Interpretation**: Out of 100 actual purchasers, the model finds 29
- **Business Impact**: Helps identify potential customers who will actually purchase

### **F1-Score (34.09% for Best Model)**
- **Meaning**: Balanced measure of precision and recall
- **Interpretation**: Harmonic mean of precision and recall
- **Why Used**: Best metric for imbalanced datasets (more "no purchase" than "purchase")

### **ROC-AUC (52.19% for Best Model)**
- **Meaning**: Model's ability to distinguish between classes
- **Interpretation**: 52.19% better than random guessing (50%)
- **Range**: 0.5 (random) to 1.0 (perfect)

---

## 🎯 Prediction Confidence

When the model makes a prediction, it also provides **probability scores**:

- **Purchase Probability**: 0% to 100%
- **No Purchase Probability**: 0% to 100%
- **Total**: Always equals 100%

**Example:**
- If model predicts "Purchase" with 75% confidence:
  - Purchase: 75%
  - No Purchase: 25%

---

## 📊 Performance Analysis

### **Why These Percentages?**

1. **Moderate Accuracy (56-61%)**:
   - Purchase prediction is a complex problem
   - Many factors influence customer behavior
   - Real-world data has noise and uncertainty

2. **Lower Recall (1.94-29.03%)**:
   - Model is conservative in predicting purchases
   - Better at avoiding false positives
   - May miss some actual purchasers

3. **Moderate Precision (26-43%)**:
   - When model predicts purchase, it's correct about 1/3 to 2/5 of the time
   - Better than random guessing (which would be ~35-40% based on class distribution)

4. **F1-Score Selection**:
   - Decision Tree selected because it balances precision and recall
   - Better for imbalanced datasets than accuracy alone

---

## 🔄 Training vs Testing Performance

- **Training Data**: 80% of dataset
- **Testing Data**: 20% of dataset
- **Performance Shown**: Based on **testing data** (unseen during training)
- **This ensures**: Real-world performance estimate

---

## 📈 How to Interpret Predictions

### **High Confidence Predictions (>70%)**
- **Purchase**: Strong likelihood customer will purchase
- **No Purchase**: Strong likelihood customer won't purchase

### **Medium Confidence (50-70%)**
- **Uncertain**: Model is less confident
- **Recommendation**: Use additional factors for decision

### **Low Confidence (<50%)**
- **Very Uncertain**: Model has low confidence
- **Recommendation**: Treat as neutral/unknown

---

## 🎓 Model Selection Rationale

**Decision Tree was selected as best model because:**
- ✅ Highest F1-Score (34.09%)
- ✅ Good balance between precision and recall
- ✅ Better at identifying purchasers (29.03% recall) than other models
- ✅ Reasonable precision (41.28%)

**Note**: While Logistic Regression has higher accuracy (61%), it has very low recall (1.94%), meaning it misses most actual purchasers.

---

## 📊 Performance by Model Type

| Model | Best Metric | Value |
|-------|-------------|-------|
| **Decision Tree** | F1-Score | **34.09%** 🏆 |
| **Logistic Regression** | Accuracy | **61.00%** |
| **Logistic Regression** | Precision | **42.86%** |
| **Logistic Regression** | ROC-AUC | **55.54%** |
| **Naive Bayes** | Accuracy | **59.00%** |

---

## 🔍 Understanding the Metrics

### **Accuracy vs F1-Score**
- **Accuracy**: Overall correctness (can be misleading with imbalanced data)
- **F1-Score**: Better for imbalanced datasets (considers both precision and recall)

### **Why F1-Score Matters More**
- Dataset has more "No Purchase" (60-65%) than "Purchase" (35-40%)
- A model could achieve high accuracy by always predicting "No Purchase"
- F1-Score ensures the model actually finds purchasers

---

## 📝 Summary

**Current Prediction Performance:**
- **Best Model Accuracy**: **56.50%**
- **Best Model F1-Score**: **34.09%**
- **Best Model Precision**: **41.28%**
- **Best Model Recall**: **29.03%**

**Model Used for Predictions**: **Decision Tree**

**Prediction Confidence**: Provided as probability percentages (0-100%)

---

## 🔄 Improving Performance

To improve prediction percentages, consider:
1. **More Data**: Collect more customer records
2. **Better Features**: Add more relevant customer attributes
3. **Feature Engineering**: Create better derived features
4. **Ensemble Methods**: Combine multiple models (Random Forest, XGBoost)
5. **Class Balancing**: Handle imbalanced dataset (SMOTE, class weights)
6. **Hyperparameter Tuning**: Optimize all model parameters

---

**Last Updated**: Based on current trained models  
**Data Source**: Customer Personality Analysis dataset  
**Test Set Size**: 20% of total dataset

