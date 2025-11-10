# 🚀 Quick Dataset Download Guide

## Download the Real Kaggle Dataset

### Step 1: Install Kaggle API
```bash
cd ml_pipeline
pip install kaggle
```

### Step 2: Setup Kaggle Credentials

1. Go to: https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`

5. **Place the file**:
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Create the `.kaggle` folder if it doesn't exist

### Step 3: Download the Dataset

**Option A: Using the script**
```bash
cd ml_pipeline
python download_kaggle_dataset.py
```

**Option B: Manual download**
1. Visit: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
2. Click "Download" (requires Kaggle account)
3. Extract the ZIP file
4. Copy the CSV file to: `ml_pipeline/data/marketing_campaign.csv`

### Step 4: Run the Pipeline

```bash
cd ml_pipeline
python pipeline.py
```

The pipeline will automatically:
- ✅ Detect the Kaggle dataset
- ✅ Load and process it
- ✅ Train models with real data
- ✅ Generate visualizations

---

## What to Expect

The real Kaggle dataset contains:
- **~2,240 customers** (vs 2,000 synthetic)
- **Real customer behavior patterns**
- **Actual spending and purchase data**
- **Better model accuracy**

---

## Troubleshooting

**"Kaggle API authentication error"**
- Verify `kaggle.json` is in the correct location
- Check your Kaggle account has access

**"Dataset not found"**
- Ensure file is in `ml_pipeline/data/`
- Check filename matches: `marketing_campaign.csv` or `customer_personality.csv`

**"Column warnings"**
- Normal if dataset structure differs slightly
- Pipeline will use defaults for missing columns

---

**Ready to use real data!** 🎉

