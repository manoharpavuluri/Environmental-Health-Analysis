#!/usr/bin/env python3
"""
EPA Air Quality Data Download Helper

This script helps download EPA air quality data from Kaggle for the hotspot analysis.
"""

import requests
import pandas as pd
from pathlib import Path
import zipfile
import os


def download_epa_sample_data():
    """
    Download a sample of EPA air quality data for testing.
    This creates a realistic dataset structure for the hotspot analysis.
    """
    print("📊 Creating EPA Air Quality Sample Data...")
    
    # Create sample EPA air quality data based on the Kaggle dataset structure
    sample_data = {
        'State Name': ['California', 'California', 'Texas', 'Texas', 'New York', 'New York', 'Florida', 'Florida', 'Illinois', 'Illinois'],
        'County Name': ['Los Angeles', 'San Diego', 'Harris', 'Dallas', 'New York', 'Kings', 'Miami-Dade', 'Broward', 'Cook', 'DuPage'],
        'Site Num': ['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010'],
        'Parameter Name': ['PM2.5', 'Ozone', 'PM2.5', 'Ozone', 'PM2.5', 'Ozone', 'PM2.5', 'Ozone', 'PM2.5', 'Ozone'],
        'Date Local': ['2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-01'],
        'Arithmetic Mean': [15.2, 45.8, 12.3, 38.9, 18.7, 42.1, 14.5, 41.2, 16.8, 39.5],
        'Units of Measure': ['Micrograms/cubic meter (LC)', 'Parts per million', 'Micrograms/cubic meter (LC)', 'Parts per million', 'Micrograms/cubic meter (LC)', 'Parts per million', 'Micrograms/cubic meter (LC)', 'Parts per million', 'Micrograms/cubic meter (LC)', 'Parts per million'],
        'AQI': [62, 78, 52, 72, 68, 75, 58, 70, 64, 73],
        'Category': ['Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate', 'Moderate']
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to file
    output_file = Path("data/raw/epa_aqi/epa_historical_air_quality.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created sample EPA data: {output_file}")
    print(f"📊 Sample data contains {len(df)} records")
    print(f"🗺️ Counties: {df['County Name'].nunique()} counties")
    print(f"🌬️ Parameters: {df['Parameter Name'].unique()}")
    
    return df


def download_seer_sample_data():
    """
    Create sample SEER cancer data for testing.
    """
    print("📊 Creating SEER Cancer Sample Data...")
    
    # Create sample SEER data
    sample_data = {
        'county': ['Los Angeles', 'San Diego', 'Harris', 'Dallas', 'New York', 'Kings', 'Miami-Dade', 'Broward', 'Cook', 'DuPage'],
        'state': ['CA', 'CA', 'TX', 'TX', 'NY', 'NY', 'FL', 'FL', 'IL', 'IL'],
        'year': [2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020],
        'incidence_count': [150, 85, 120, 95, 180, 110, 75, 65, 140, 45],
        'population': [10000000, 3300000, 4700000, 2600000, 8400000, 2600000, 2700000, 1900000, 5200000, 930000]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Calculate incidence rate per 100,000
    df['incidence_rate'] = (df['incidence_count'] / df['population']) * 100000
    
    # Save to file
    output_file = Path("data/raw/seer/seer_incidence_2010-2020.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created sample SEER data: {output_file}")
    print(f"📊 Sample data contains {len(df)} records")
    print(f"🗺️ Counties: {df['county'].nunique()} counties")
    print(f"📈 Average incidence rate: {df['incidence_rate'].mean():.2f} per 100,000")
    
    return df


def main():
    """Main function to download sample data."""
    print("🚀 EPA Air Quality Data Download Helper")
    print("="*50)
    
    print("\n📋 Options:")
    print("1. Create sample EPA air quality data (for testing)")
    print("2. Create sample SEER cancer data (for testing)")
    print("3. Create both sample datasets")
    print("4. Instructions for downloading real data")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        download_epa_sample_data()
    elif choice == "2":
        download_seer_sample_data()
    elif choice == "3":
        download_epa_sample_data()
        download_seer_sample_data()
        print("\n✅ Both sample datasets created!")
    elif choice == "4":
        print_real_data_instructions()
    else:
        print("❌ Invalid choice. Please run the script again.")
    
    print("\n" + "="*50)


def print_real_data_instructions():
    """Print instructions for downloading real data."""
    print("\n📋 REAL DATA DOWNLOAD INSTRUCTIONS:")
    print("\n1. EPA AIR QUALITY DATA (Kaggle):")
    print("   • Go to: https://www.kaggle.com/datasets/epa/epa-historical-air-quality")
    print("   • Click 'Download' (requires Kaggle account)")
    print("   • Extract the ZIP file")
    print("   • Find the main CSV file (usually the largest one)")
    print("   • Copy to: data/raw/epa_aqi/epa_historical_air_quality.csv")
    
    print("\n2. SEER CANCER DATA:")
    print("   • Go to: https://seer.cancer.gov/data/")
    print("   • Register for SEER data access")
    print("   • Download pancreatic cancer incidence data")
    print("   • Save as: data/raw/seer/seer_incidence_2010-2020.csv")
    
    print("\n3. CENSUS DEMOGRAPHIC DATA:")
    print("   • Your API key is already configured!")
    print("   • Run: python3 main.py")
    print("   • The system will automatically download Census data")
    
    print("\n💡 TIP: Start with the sample data to test the system, then replace with real data.")


if __name__ == "__main__":
    main() 