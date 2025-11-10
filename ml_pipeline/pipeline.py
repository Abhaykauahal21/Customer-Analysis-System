"""
Customer Behavior and Sales Analysis - ML Pipeline
Performs data processing, PCA, K-Means clustering, and classification
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class CustomerAnalysisPipeline:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        self.best_model = None
        self.models = {}
        self.model_metrics = {}
        
    def load_data(self):
        """Load and inspect datasets"""
        print("📊 Loading datasets...")
        
        # Try to load Kaggle dataset (Customer Personality Analysis)
        # Check for common filenames from Kaggle
        possible_files = [
            'marketing_campaign.csv',
            'customer_personality.csv',
            'marketing_campaign.xlsx',
            'customer_personality.xlsx'
        ]
        
        customer_file = None
        for filename in possible_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                customer_file = filepath
                print(f"   Found dataset: {filename}")
                break
        
        # Also check for any CSV file in data directory
        if customer_file is None:
            csv_files = list(self.data_dir.glob('*.csv'))
            if csv_files:
                customer_file = csv_files[0]
                print(f"   Found dataset: {customer_file.name}")
        
        # Load customer dataset
        if customer_file and customer_file.exists():
            print(f"   Loading: {customer_file.name}")
            try:
                if customer_file.suffix == '.xlsx':
                    customer_df = pd.read_excel(customer_file)
                else:
                    customer_df = pd.read_csv(customer_file)
                
                print(f"✅ Customer dataset loaded: {customer_df.shape}")
                print(f"   Columns: {list(customer_df.columns[:5])}... ({len(customer_df.columns)} total)")
                
                # Check if we need to create a retail dataset or use existing
                retail_file = self.data_dir / 'online_retail.csv'
                if retail_file.exists():
                    retail_df = pd.read_csv(retail_file)
                    print(f"✅ Retail dataset loaded: {retail_df.shape}")
                else:
                    # Create a synthetic retail dataset based on customer data
                    print("   Creating retail transaction data from customer data...")
                    retail_df = self._create_retail_from_customer(customer_df)
                    retail_df.to_csv(retail_file, index=False)
                    print(f"✅ Retail dataset created: {retail_df.shape}")
                
                return customer_df, retail_df
                
            except Exception as e:
                print(f"⚠️  Error loading {customer_file.name}: {str(e)}")
                print("   Falling back to synthetic data generation...")
        
        # If no dataset found, create synthetic data
        print("⚠️  Kaggle dataset not found. Generating synthetic data...")
        print("   To use real data, download from:")
        print("   https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis")
        print("   Or run: python download_kaggle_dataset.py")
        self._generate_synthetic_data()
        customer_file = self.data_dir / 'customer_personality.csv'
        retail_file = self.data_dir / 'online_retail.csv'
        
        # Load synthetic datasets
        customer_df = pd.read_csv(customer_file)
        retail_df = pd.read_csv(retail_file)
        
        print(f"✅ Customer dataset: {customer_df.shape}")
        print(f"✅ Retail dataset: {retail_df.shape}")
        
        return customer_df, retail_df
    
    def _generate_synthetic_data(self):
        """Generate synthetic datasets for demonstration"""
        np.random.seed(42)
        n_customers = 2000
        
        # Customer Personality Dataset
        customer_data = {
            'CustomerID': range(1, n_customers + 1),
            'Year_Birth': np.random.randint(1950, 2000, n_customers),
            'Education': np.random.choice(['Graduation', 'PhD', 'Master', '2n Cycle', 'Basic'], n_customers),
            'Marital_Status': np.random.choice(['Single', 'Together', 'Married', 'Divorced', 'Widow'], n_customers),
            'Income': np.random.normal(50000, 20000, n_customers).clip(0),
            'Kidhome': np.random.randint(0, 3, n_customers),
            'Teenhome': np.random.randint(0, 3, n_customers),
            'Dt_Customer': pd.date_range('2010-01-01', periods=n_customers, freq='D'),
            'Recency': np.random.randint(0, 100, n_customers),
            'MntWines': np.random.poisson(200, n_customers),
            'MntFruits': np.random.poisson(50, n_customers),
            'MntMeatProducts': np.random.poisson(150, n_customers),
            'MntFishProducts': np.random.poisson(40, n_customers),
            'MntSweetProducts': np.random.poisson(30, n_customers),
            'MntGoldProds': np.random.poisson(60, n_customers),
            'NumDealsPurchases': np.random.poisson(3, n_customers),
            'NumWebPurchases': np.random.poisson(5, n_customers),
            'NumCatalogPurchases': np.random.poisson(3, n_customers),
            'NumStorePurchases': np.random.poisson(8, n_customers),
            'NumWebVisitsMonth': np.random.poisson(6, n_customers),
            'AcceptedCmp1': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
            'AcceptedCmp2': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
            'AcceptedCmp3': np.random.choice([0, 1], n_customers, p=[0.75, 0.25]),
            'AcceptedCmp4': np.random.choice([0, 1], n_customers, p=[0.85, 0.15]),
            'AcceptedCmp5': np.random.choice([0, 1], n_customers, p=[0.9, 0.1]),
            'Response': np.random.choice([0, 1], n_customers, p=[0.6, 0.4]),
            'Complain': np.random.choice([0, 1], n_customers, p=[0.95, 0.05]),
            'Country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Spain', 'Canada'], n_customers)
        }
        
        customer_df = pd.DataFrame(customer_data)
        customer_df['SpendingScore'] = (
            customer_df['MntWines'] + customer_df['MntFruits'] + 
            customer_df['MntMeatProducts'] + customer_df['MntFishProducts'] + 
            customer_df['MntSweetProducts'] + customer_df['MntGoldProds']
        ) / 1000
        
        # Online Retail Dataset
        retail_data = {
            'InvoiceNo': [f'INV{i:06d}' for i in range(1, n_customers + 1)],
            'CustomerID': range(1, n_customers + 1),
            'StockCode': [f'STOCK{np.random.randint(1000, 9999)}' for _ in range(n_customers)],
            'Description': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], n_customers),
            'Quantity': np.random.poisson(5, n_customers).clip(1),
            'InvoiceDate': pd.date_range('2020-01-01', periods=n_customers, freq='H'),
            'UnitPrice': np.random.normal(10, 5, n_customers).clip(0.1),
            'Country': customer_df['Country'].values
        }
        
        retail_df = pd.DataFrame(retail_data)
        retail_df['TotalSpent'] = retail_df['Quantity'] * retail_df['UnitPrice']
        
        # Save synthetic data
        customer_df.to_csv(self.data_dir / 'customer_personality.csv', index=False)
        retail_df.to_csv(self.data_dir / 'online_retail.csv', index=False)
        
        print("✅ Synthetic datasets generated and saved")
    
    def _create_retail_from_customer(self, customer_df):
        """Create retail transaction data from customer personality data"""
        np.random.seed(42)
        retail_records = []
        
        # Generate transactions for each customer
        for idx, row in customer_df.iterrows():
            customer_id = row.get('ID', row.get('CustomerID', idx + 1))
            
            # Calculate number of transactions based on purchase behavior
            total_purchases = (
                row.get('NumDealsPurchases', 0) + 
                row.get('NumWebPurchases', 0) + 
                row.get('NumCatalogPurchases', 0) + 
                row.get('NumStorePurchases', 0)
            )
            
            # Generate transactions
            n_transactions = max(1, int(total_purchases * np.random.uniform(0.8, 1.2)))
            
            for i in range(n_transactions):
                retail_records.append({
                    'InvoiceNo': f'INV{int(customer_id):06d}{i:03d}',
                    'CustomerID': customer_id,
                    'StockCode': f'STOCK{np.random.randint(1000, 9999)}',
                    'Description': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D']),
                    'Quantity': np.random.poisson(5).clip(1, 20),
                    'InvoiceDate': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
                    'UnitPrice': np.random.normal(10, 5).clip(0.1, 100),
                    'Country': row.get('Country', 'Unknown')
                })
        
        retail_df = pd.DataFrame(retail_records)
        retail_df['TotalSpent'] = retail_df['Quantity'] * retail_df['UnitPrice']
        
        return retail_df
    
    def clean_data(self, customer_df, retail_df):
        """Clean and preprocess datasets"""
        print("\n🧹 Cleaning datasets...")
        
        # Clean customer dataset
        customer_df = customer_df.copy()
        
        # Handle missing values - fill numeric columns with median, categorical with mode
        numeric_cols = customer_df.select_dtypes(include=[np.number]).columns
        categorical_cols = customer_df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if customer_df[col].isna().sum() > 0:
                customer_df[col].fillna(customer_df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if customer_df[col].isna().sum() > 0:
                customer_df[col].fillna(customer_df[col].mode()[0] if len(customer_df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # Remove duplicates
        customer_df = customer_df.drop_duplicates()
        
        # Clean retail dataset
        retail_df = retail_df.copy()
        
        # Remove rows with missing critical fields
        retail_df = retail_df.dropna(subset=['Quantity', 'UnitPrice'])
        
        # Filter invalid values
        retail_df = retail_df[retail_df['Quantity'] > 0]
        retail_df = retail_df[retail_df['UnitPrice'] > 0]
        
        # Remove duplicates
        retail_df = retail_df.drop_duplicates()
        
        print(f"✅ Customer data cleaned: {customer_df.shape}")
        print(f"✅ Retail data cleaned: {retail_df.shape}")
        
        return customer_df, retail_df
    
    def merge_datasets(self, customer_df, retail_df):
        """Merge datasets on CustomerID and Country"""
        print("\n🔗 Merging datasets...")
        
        # Identify customer ID column (could be ID, CustomerID, etc.)
        customer_id_col = None
        for col in ['ID', 'CustomerID', 'customer_id', 'CUSTOMER_ID']:
            if col in customer_df.columns:
                customer_id_col = col
                break
        
        if customer_id_col is None:
            # If no ID column found, create one
            customer_df['ID'] = range(1, len(customer_df) + 1)
            customer_id_col = 'ID'
            print(f"   Created ID column: {customer_id_col}")
        else:
            print(f"   Using ID column: {customer_id_col}")
        
        # Aggregate retail data by customer
        if 'TotalSpent' not in retail_df.columns:
            retail_df['TotalSpent'] = retail_df.get('Quantity', 0) * retail_df.get('UnitPrice', 0)
        
        retail_agg = retail_df.groupby('CustomerID').agg({
            'TotalSpent': 'sum',
            'Quantity': 'sum',
            'InvoiceNo': 'count'
        }).reset_index()
        retail_agg.columns = ['CustomerID', 'TotalSpent', 'TotalQuantity', 'TransactionCount']
        
        # Merge on CustomerID (map customer_id_col to CustomerID if different)
        if customer_id_col != 'CustomerID':
            # Create mapping
            customer_df_temp = customer_df.copy()
            customer_df_temp['CustomerID'] = customer_df_temp[customer_id_col]
        else:
            customer_df_temp = customer_df.copy()
        
        merged_df = customer_df_temp.merge(retail_agg, on='CustomerID', how='left')
        
        # Fill missing values from merge
        merged_df['TotalSpent'] = merged_df['TotalSpent'].fillna(0)
        merged_df['TotalQuantity'] = merged_df['TotalQuantity'].fillna(0)
        merged_df['TransactionCount'] = merged_df['TransactionCount'].fillna(0)
        
        print(f"✅ Merged dataset: {merged_df.shape}")
        
        return merged_df
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        print("\n🔧 Preparing features...")
        
        df = df.copy()
        
        # Encode categorical features (handle missing columns gracefully)
        le_education = LabelEncoder()
        le_marital = LabelEncoder()
        le_country = LabelEncoder()
        
        if 'Education' in df.columns:
            df['Education_Encoded'] = le_education.fit_transform(df['Education'].astype(str))
        else:
            df['Education_Encoded'] = 0
            print("   ⚠️  Education column not found, using default")
        
        if 'Marital_Status' in df.columns:
            df['Marital_Status_Encoded'] = le_marital.fit_transform(df['Marital_Status'].astype(str))
        else:
            df['Marital_Status_Encoded'] = 0
            print("   ⚠️  Marital_Status column not found, using default")
        
        if 'Country' in df.columns:
            df['Country_Encoded'] = le_country.fit_transform(df['Country'].astype(str))
        else:
            df['Country_Encoded'] = 0
            print("   ⚠️  Country column not found, using default")
        
        # Calculate age
        if 'Year_Birth' in df.columns:
            df['Age'] = 2024 - df['Year_Birth']
        else:
            df['Age'] = 45  # Default age
            print("   ⚠️  Year_Birth column not found, using default age")
        
        # Calculate total spending (handle missing columns)
        spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                        'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        available_spending = [col for col in spending_cols if col in df.columns]
        
        if available_spending:
            df['TotalSpending'] = df[available_spending].sum(axis=1)
        else:
            df['TotalSpending'] = df.get('TotalSpent', 0)
            print("   ⚠️  Spending columns not found, using TotalSpent")
        
        # Calculate total purchases
        purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 
                        'NumCatalogPurchases', 'NumStorePurchases']
        available_purchases = [col for col in purchase_cols if col in df.columns]
        
        if available_purchases:
            df['TotalPurchases'] = df[available_purchases].sum(axis=1)
        else:
            df['TotalPurchases'] = df.get('TransactionCount', 0)
            print("   ⚠️  Purchase columns not found, using TransactionCount")
        
        # Calculate SpendingScore if not present
        if 'SpendingScore' not in df.columns:
            df['SpendingScore'] = df['TotalSpending'] / 1000
        
        # Build feature list with fallbacks
        feature_mapping = {
            'Age': 'Age',
            'Income': 'Income',
            'Kidhome': 'Kidhome',
            'Teenhome': 'Teenhome',
            'Recency': 'Recency',
            'MntWines': 'MntWines',
            'MntFruits': 'MntFruits',
            'MntMeatProducts': 'MntMeatProducts',
            'MntFishProducts': 'MntFishProducts',
            'MntSweetProducts': 'MntSweetProducts',
            'MntGoldProds': 'MntGoldProds',
            'NumDealsPurchases': 'NumDealsPurchases',
            'NumWebPurchases': 'NumWebPurchases',
            'NumCatalogPurchases': 'NumCatalogPurchases',
            'NumStorePurchases': 'NumStorePurchases',
            'NumWebVisitsMonth': 'NumWebVisitsMonth',
        }
        
        # Select available features
        feature_columns = []
        for feature in [
            'Age', 'Income', 'Kidhome', 'Teenhome', 'Recency',
            'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
            'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
            'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
            'NumWebVisitsMonth', 'Education_Encoded', 'Marital_Status_Encoded',
            'Country_Encoded', 'TotalSpending', 'TotalPurchases', 'SpendingScore',
            'TotalSpent', 'TotalQuantity', 'TransactionCount'
        ]:
            if feature in df.columns:
                feature_columns.append(feature)
            else:
                # Use default value for missing features
                df[feature] = 0
                feature_columns.append(feature)
                if feature not in ['Education_Encoded', 'Marital_Status_Encoded', 'Country_Encoded']:
                    print(f"   ⚠️  {feature} not found, using default (0)")
        
        # Create target variable (Purchase Likelihood)
        if 'Response' in df.columns:
            df['PurchaseLikelihood'] = df['Response'].astype(int)
        elif 'AcceptedCmp3' in df.columns:
            # Use campaign acceptance as proxy
            df['PurchaseLikelihood'] = df['AcceptedCmp3'].astype(int)
        else:
            # Create target based on spending behavior
            median_spending = df['TotalSpending'].median()
            df['PurchaseLikelihood'] = (df['TotalSpending'] > median_spending).astype(int)
            print("   Created PurchaseLikelihood based on spending behavior")
        
        X = df[feature_columns].copy()
        y = df['PurchaseLikelihood'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.median())
        
        print(f"✅ Features prepared: {X.shape}")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, df, feature_columns
    
    def apply_pca(self, X, n_components=2):
        """Apply PCA for dimensionality reduction"""
        print("\n📉 Applying PCA...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        
        explained_variance = self.pca.explained_variance_ratio_
        print(f"✅ PCA completed")
        print(f"   Explained variance: {explained_variance}")
        print(f"   Total variance explained: {sum(explained_variance):.2%}")
        
        return X_pca, X_scaled
    
    def apply_kmeans(self, X_pca, max_k=10):
        """Apply K-Means clustering with elbow method"""
        print("\n🎯 Applying K-Means clustering...")
        
        # Elbow method
        inertias = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X_pca)
            inertias.append(kmeans_temp.inertia_)
        
        # Find optimal K (simplified - using elbow at k=4 or 5)
        optimal_k = 4  # Can be improved with automated elbow detection
        
        # Fit final K-Means
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_pca)
        
        print(f"✅ K-Means completed with k={optimal_k}")
        print(f"   Cluster distribution: {np.bincount(cluster_labels)}")
        
        return cluster_labels, optimal_k, inertias
    
    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train multiple classification models with improved hyperparameters"""
        print("\n🤖 Training classification models...")
        
        # Calculate class weights for imbalanced data
        from collections import Counter
        class_counts = Counter(y_train)
        total = sum(class_counts.values())
        class_weights = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=2000,
                class_weight='balanced',
                C=1.0,
                solver='lbfgs'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42, 
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced'
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5
            ),
            'KNN': None  # Will be set after finding optimal k
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        # Find optimal K for KNN with better search
        print("   Finding optimal K for KNN...")
        knn_params = {'n_neighbors': range(3, 31, 2), 'weights': ['uniform', 'distance']}
        knn_base = KNeighborsClassifier()
        knn_grid = GridSearchCV(knn_base, knn_params, cv=5, scoring='f1', n_jobs=-1)
        knn_grid.fit(X_train, y_train)
        optimal_knn_k = knn_grid.best_params_['n_neighbors']
        optimal_knn_weights = knn_grid.best_params_['weights']
        models['KNN'] = KNeighborsClassifier(
            n_neighbors=optimal_knn_k,
            weights=optimal_knn_weights
        )
        print(f"   Optimal K for KNN: {optimal_knn_k}, weights: {optimal_knn_weights}")
        
        # Train all models
        for name, model in models.items():
            if model is None:
                continue
            print(f"   Training {name}...")
            try:
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred).tolist()
                
                roc_auc = None
                if y_pred_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_test, y_pred_proba)
                    except:
                        pass
                
                self.model_metrics[name] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'roc_auc': float(roc_auc) if roc_auc else None,
                    'confusion_matrix': cm
                }
                
                self.models[name] = model
                print(f"      ✅ {name} - Accuracy: {accuracy*100:.2f}%, F1: {f1*100:.2f}%")
            except Exception as e:
                print(f"      ❌ Error training {name}: {str(e)}")
                continue
        
        # Find best model based on F1 score
        if self.model_metrics:
            best_model_name = max(self.model_metrics.keys(), 
                                 key=lambda x: self.model_metrics[x]['f1_score'])
            self.best_model = self.models[best_model_name]
            
            # Calculate total accuracy
            total_accuracy = sum(metrics['accuracy'] for metrics in self.model_metrics.values())
            avg_accuracy = total_accuracy / len(self.model_metrics)
            
            print(f"\n✅ All models trained")
            print(f"🏆 Best model: {best_model_name} (F1: {self.model_metrics[best_model_name]['f1_score']:.4f})")
            print(f"📊 Total Accuracy (Sum): {total_accuracy*100:.2f}%")
            print(f"📊 Average Accuracy: {avg_accuracy*100:.2f}%")
            print(f"📊 Number of Models: {len(self.model_metrics)}")
        else:
            print("❌ No models were successfully trained")
        
        return self.model_metrics
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving models...")
        
        # Save scaler
        joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
        
        # Save PCA
        joblib.dump(self.pca, self.models_dir / 'pca.pkl')
        
        # Save K-Means
        joblib.dump(self.kmeans, self.models_dir / 'kmeans.pkl')
        
        # Save all models
        for name, model in self.models.items():
            joblib.dump(model, self.models_dir / f'{name.lower().replace(" ", "_")}.pkl')
        
        # Save best model
        joblib.dump(self.best_model, self.models_dir / 'best_model.pkl')
        
        print("✅ Models saved")
    
    def generate_visualization_data(self, X_pca, cluster_labels):
        """Generate JSON data for frontend visualization"""
        print("\n📊 Generating visualization data...")
        
        # PCA + Cluster data
        cluster_data = {
            'pca_data': [
                {
                    'x': float(X_pca[i, 0]),
                    'y': float(X_pca[i, 1]),
                    'cluster': int(cluster_labels[i])
                }
                for i in range(len(X_pca))
            ],
            'cluster_centers': [
                {
                    'x': float(center[0]),
                    'y': float(center[1]),
                    'cluster': int(i)
                }
                for i, center in enumerate(self.kmeans.cluster_centers_)
            ],
            'explained_variance': [float(v) for v in self.pca.explained_variance_ratio_]
        }
        
        # Model comparison data
        model_comparison = []
        total_accuracy = 0
        for name, metrics in sorted(self.model_metrics.items(), 
                                   key=lambda x: x[1]['f1_score'], reverse=True):
            model_comparison.append({
                'name': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            })
            total_accuracy += metrics['accuracy']
        
        # Calculate summary statistics
        avg_accuracy = total_accuracy / len(model_comparison) if model_comparison else 0
        max_accuracy = max([m['accuracy'] for m in model_comparison]) if model_comparison else 0
        min_accuracy = min([m['accuracy'] for m in model_comparison]) if model_comparison else 0
        
        # Save visualization data
        viz_data = {
            'clusters': cluster_data,
            'models': model_comparison,
            'model_metrics': self.model_metrics,
            'summary': {
                'total_accuracy': float(total_accuracy),
                'average_accuracy': float(avg_accuracy),
                'max_accuracy': float(max_accuracy),
                'min_accuracy': float(min_accuracy),
                'num_models': len(model_comparison)
            }
        }
        
        with open(self.results_dir / 'visualization_data.json', 'w') as f:
            json.dump(viz_data, f, indent=2)
        
        print("✅ Visualization data saved")
        
        return viz_data
    
    def print_model_comparison(self):
        """Print model comparison table"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Sort by F1 score
        sorted_models = sorted(self.model_metrics.items(), 
                              key=lambda x: x[1]['f1_score'], reverse=True)
        
        print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
        print("-"*80)
        
        for name, metrics in sorted_models:
            roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "N/A"
            print(f"{name:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f} {roc_auc_str:<12}")
        
        print("="*80)
    
    def run_pipeline(self):
        """Run the complete ML pipeline"""
        print("\n" + "="*80)
        print("CUSTOMER BEHAVIOR ANALYSIS - ML PIPELINE")
        print("="*80)
        
        # Load data
        customer_df, retail_df = self.load_data()
        
        # Clean data
        customer_df, retail_df = self.clean_data(customer_df, retail_df)
        
        # Merge datasets
        merged_df = self.merge_datasets(customer_df, retail_df)
        
        # Prepare features
        X, y, df, feature_columns = self.prepare_features(merged_df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply PCA
        X_pca, X_scaled = self.apply_pca(X, n_components=2)
        
        # Apply K-Means
        cluster_labels, optimal_k, inertias = self.apply_kmeans(X_pca)
        
        # Train classification models
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.train_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Print comparison
        self.print_model_comparison()
        
        # Save models
        self.save_models()
        
        # Generate visualization data
        viz_data = self.generate_visualization_data(X_pca, cluster_labels)
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return viz_data, self.model_metrics

if __name__ == '__main__':
    pipeline = CustomerAnalysisPipeline()
    viz_data, metrics = pipeline.run_pipeline()

