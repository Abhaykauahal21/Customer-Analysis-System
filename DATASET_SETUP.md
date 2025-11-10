# 📊 Dataset Setup Guide

## Downloading the Kaggle Dataset

### Option 1: Automatic Download (Recommended)

1. **Install Kaggle API** (if not already installed):
   ```bash
   pip install kaggle
   ```

2. **Setup Kaggle API Credentials**:
   - Go to https://www.kaggle.com/account
   - Scroll down to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`
   - Place it in:
     - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     - **Linux/Mac**: `~/.kaggle/kaggle.json`

3. **Run the download script**:
   ```bash
   cd ml_pipeline
   python download_kaggle_dataset.py
   ```

### Option 2: Manual Download

1. **Visit the dataset page**:
   - https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

2. **Download the dataset**:
   - Click the "Download" button (requires Kaggle account)
   - Extract the ZIP file

3. **Place the CSV file**:
   - Copy the CSV file to: `ml_pipeline/data/`
   - Rename it to: `marketing_campaign.csv` or `customer_personality.csv`

## Dataset Information

**Dataset Name**: Customer Personality Analysis  
**Source**: Kaggle  
**URL**: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

### Expected Columns

The dataset should contain columns like:
- `ID` or `CustomerID` - Customer identifier
- `Year_Birth` - Birth year
- `Education` - Education level
- `Marital_Status` - Marital status
- `Income` - Annual income
- `Kidhome` - Number of children
- `Teenhome` - Number of teenagers
- `Dt_Customer` - Customer enrollment date
- `Recency` - Days since last purchase
- `MntWines`, `MntFruits`, `MntMeatProducts`, etc. - Spending amounts
- `NumDealsPurchases`, `NumWebPurchases`, etc. - Purchase counts
- `AcceptedCmp1-5` - Campaign acceptance
- `Response` - Last campaign response
- `Complain` - Customer complaints
- `Country` - Country

## Running the Pipeline

Once the dataset is downloaded:

```bash
cd ml_pipeline
python pipeline.py
```

The pipeline will:
1. ✅ Automatically detect the Kaggle dataset
2. ✅ Load and clean the data
3. ✅ Create retail transaction data if needed
4. ✅ Train all ML models
5. ✅ Generate visualizations

## Troubleshooting

### Issue: "Dataset not found"
**Solution**: 
- Ensure the CSV file is in `ml_pipeline/data/`
- Check the filename matches expected names
- Run `python download_kaggle_dataset.py` again

### Issue: "Column not found" warnings
**Solution**: 
- The pipeline will use default values for missing columns
- This is normal if the dataset structure differs slightly
- The pipeline is designed to be flexible

### Issue: Kaggle API authentication error
**Solution**:
- Verify `kaggle.json` is in the correct location
- Check that your Kaggle account has access to the dataset
- Use manual download instead

## Next Steps

After downloading and running the pipeline:
1. ✅ Models will be saved to `ml_pipeline/models/`
2. ✅ Visualization data will be saved to `ml_pipeline/results/`
3. ✅ Restart your backend to use the new models
4. ✅ Your frontend will display real data!

---

**Note**: The pipeline is designed to work with both the real Kaggle dataset and synthetic data, so it will gracefully handle missing columns or files.

