# Entity Relationship Diagram

## Data Model

```
┌─────────────────────────────────────┐
│     Customer Personality Data      │
├─────────────────────────────────────┤
│ CustomerID (PK)                     │
│ Year_Birth                          │
│ Education                           │
│ Marital_Status                      │
│ Income                              │
│ Kidhome                             │
│ Teenhome                            │
│ Dt_Customer                         │
│ Recency                             │
│ MntWines                            │
│ MntFruits                           │
│ MntMeatProducts                     │
│ MntFishProducts                     │
│ MntSweetProducts                    │
│ MntGoldProds                        │
│ NumDealsPurchases                   │
│ NumWebPurchases                     │
│ NumCatalogPurchases                 │
│ NumStorePurchases                   │
│ NumWebVisitsMonth                   │
│ AcceptedCmp1-5                      │
│ Response                            │
│ Complain                            │
│ Country                             │
│ SpendingScore                       │
└─────────────────────────────────────┘
            │
            │ 1:1
            │
            ▼
┌─────────────────────────────────────┐
│      Merged Customer Dataset         │
├─────────────────────────────────────┤
│ CustomerID (PK)                      │
│ [All Customer Personality Fields]    │
│ TotalSpent                           │
│ TotalQuantity                        │
│ TransactionCount                     │
│ Age (derived)                        │
│ TotalSpending (derived)              │
│ TotalPurchases (derived)             │
│ PurchaseLikelihood (target)          │
└─────────────────────────────────────┘
            │
            │ 1:N
            │
            ▼
┌─────────────────────────────────────┐
│      Online Retail Data             │
├─────────────────────────────────────┤
│ InvoiceNo                            │
│ CustomerID (FK)                      │
│ StockCode                            │
│ Description                          │
│ Quantity                             │
│ InvoiceDate                          │
│ UnitPrice                            │
│ Country                              │
│ TotalSpent (derived)                 │
└─────────────────────────────────────┘
```

## ML Model Relationships

```
┌─────────────────────────────────────┐
│      Feature Engineering             │
├─────────────────────────────────────┤
│ Input: Merged Dataset                │
│ Output: Feature Matrix (X)           │
│         Target Vector (y)             │
└─────────────────────────────────────┘
            │
            ├─────────────────┐
            │                 │
            ▼                 ▼
┌──────────────────┐  ┌──────────────────┐
│   PCA Transform  │  │  StandardScaler  │
├──────────────────┤  ├──────────────────┤
│ Input: X_scaled  │  │ Input: X         │
│ Output: X_pca    │  │ Output: X_scaled │
│ (2 components)   │  └──────────────────┘
└──────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│      K-Means Clustering              │
├─────────────────────────────────────┤
│ Input: X_pca                         │
│ Output: Cluster Labels               │
│         Cluster Centers              │
└─────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│   Classification Models              │
├─────────────────────────────────────┤
│ • Logistic Regression                │
│ • Decision Tree                      │
│ • Naive Bayes                        │
│ • KNN                                │
│                                      │
│ Input: X_scaled, y                   │
│ Output: Predictions                  │
│         Probabilities                │
│         Performance Metrics          │
└─────────────────────────────────────┘
```

## Data Flow Diagram

```
┌──────────────┐
│  Raw Data    │
│  (CSV Files) │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Data Cleaning   │
│  & Merging       │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Feature          │
│ Engineering      │
└──────┬───────────┘
       │
       ├──────────────┐
       │              │
       ▼              ▼
┌──────────┐   ┌──────────┐
│   PCA    │   │ Scaling   │
└────┬─────┘   └─────┬─────┘
     │              │
     └──────┬───────┘
            │
            ▼
     ┌──────────┐
     │ K-Means  │
     └────┬─────┘
          │
          ▼
     ┌──────────┐
     │ Models   │
     └────┬─────┘
          │
          ▼
     ┌──────────┐
     │ Results  │
     └──────────┘
```

