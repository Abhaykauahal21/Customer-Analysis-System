"""
Script to download Customer Personality Analysis dataset from Kaggle
Requires: pip install kaggle
Setup: Place your kaggle.json API credentials in ~/.kaggle/kaggle.json
"""

import os
import zipfile
from pathlib import Path
import subprocess
import sys

def check_kaggle_installed():
    """Check if kaggle package is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        return False

def download_dataset():
    """Download the Customer Personality Analysis dataset from Kaggle"""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Dataset information
    dataset_name = 'imakash3011/customer-personality-analysis'
    
    print("📥 Downloading Customer Personality Analysis dataset from Kaggle...")
    print(f"   Dataset: {dataset_name}")
    
    # Check if kaggle is installed
    if not check_kaggle_installed():
        print("\n❌ Kaggle package not found!")
        print("   Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("   ✅ Kaggle package installed")
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        print("   ✅ Kaggle API authenticated")
        
        # Download dataset
        print("   ⬇️  Downloading dataset files...")
        api.dataset_download_files(dataset_name, path=str(data_dir), unzip=True)
        
        print("   ✅ Dataset downloaded successfully!")
        
        # List downloaded files
        files = list(data_dir.glob('*'))
        print(f"\n   📁 Downloaded files:")
        for f in files:
            if f.is_file():
                print(f"      - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        
        # Check for the main dataset file
        csv_files = list(data_dir.glob('*.csv'))
        if csv_files:
            main_file = csv_files[0]
            print(f"\n   ✅ Main dataset file: {main_file.name}")
            
            # Rename to standard name if needed
            if main_file.name != 'marketing_campaign.csv' and main_file.name != 'customer_personality.csv':
                # Try to rename to a standard name
                standard_name = data_dir / 'marketing_campaign.csv'
                if not standard_name.exists():
                    main_file.rename(standard_name)
                    print(f"   📝 Renamed to: {standard_name.name}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("\n📋 Manual Download Instructions:")
        print("   1. Go to: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis")
        print("   2. Click 'Download' button")
        print("   3. Extract the ZIP file")
        print("   4. Copy the CSV file(s) to: ml_pipeline/data/")
        print("   5. Rename the main file to: marketing_campaign.csv or customer_personality.csv")
        return False

if __name__ == '__main__':
    print("="*80)
    print("KAGGLE DATASET DOWNLOADER")
    print("="*80)
    print()
    
    success = download_dataset()
    
    if success:
        print("\n" + "="*80)
        print("✅ DOWNLOAD COMPLETE")
        print("="*80)
        print("\nNext steps:")
        print("1. Run the ML pipeline: python pipeline.py")
        print("2. The pipeline will automatically detect and use the downloaded dataset")
    else:
        print("\n" + "="*80)
        print("⚠️  DOWNLOAD FAILED - Please download manually")
        print("="*80)

