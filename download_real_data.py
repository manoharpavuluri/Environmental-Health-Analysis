#!/usr/bin/env python3
"""
Real Data Download Helper

This script helps users download real data from various sources for the
environmental health analysis system.
"""

import sys
import argparse
from pathlib import Path
import webbrowser
import subprocess
from loguru import logger


def print_data_requirements():
    """Print data requirements and download instructions."""
    print("\n" + "="*80)
    print("📊 REAL DATA REQUIREMENTS FOR ENVIRONMENTAL HEALTH ANALYSIS")
    print("="*80)
    
    print("\n🎯 REQUIRED DATA SOURCES:")
    print("\n1. SEER CANCER DATA")
    print("   • Source: https://seer.cancer.gov/data/")
    print("   • Required: Pancreatic cancer incidence data")
    print("   • Format: CSV with county, state, incidence count, population")
    print("   • Save to: data/raw/seer/seer_incidence_2010-2020.csv")
    
    print("\n2. EPA AIR QUALITY DATA")
    print("   • Source: https://www.kaggle.com/datasets/epa/epa-historical-air-quality")
    print("   • Required: Historical air quality monitoring data")
    print("   • Format: CSV with county, state, AQI, pollutant levels")
    print("   • Save to: data/raw/epa_aqi/epa_historical_air_quality.csv")
    
    print("\n3. EPA WATER QUALITY DATA (Optional)")
    print("   • Source: https://www.epa.gov/ground-water-and-drinking-water")
    print("   • Required: Drinking water contaminant data")
    print("   • Format: CSV with county, state, contaminant levels")
    print("   • Save to: data/raw/epa_water/epa_water_quality.csv")
    
    print("\n4. CENSUS DEMOGRAPHIC DATA (Optional)")
    print("   • Source: https://api.census.gov/data/key_signup.html")
    print("   • Required: County-level demographic information")
    print("   • Format: CSV with county, state, demographics")
    print("   • Save to: data/raw/census/census_demographics_2020.csv")


def open_data_sources():
    """Open data source websites in browser."""
    print("\n🌐 OPENING DATA SOURCE WEBSITES...")
    
    sources = [
        ("SEER Cancer Data", "https://seer.cancer.gov/data/"),
        ("EPA Air Quality (Kaggle)", "https://www.kaggle.com/datasets/epa/epa-historical-air-quality"),
        ("EPA Water Quality", "https://www.epa.gov/ground-water-and-drinking-water"),
        ("Census API", "https://api.census.gov/data/key_signup.html")
    ]
    
    for name, url in sources:
        print(f"Opening {name}...")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open {name}: {e}")


def check_data_directories():
    """Check if data directories exist and create them if needed."""
    print("\n📁 CHECKING DATA DIRECTORIES...")
    
    directories = [
        "data/raw/seer",
        "data/raw/epa_aqi", 
        "data/raw/epa_water",
        "data/raw/census",
        "data/processed",
        "data/results"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")


def check_existing_data():
    """Check for existing data files."""
    print("\n🔍 CHECKING FOR EXISTING DATA...")
    
    data_files = {
        "SEER Cancer Data": "data/raw/seer/seer_incidence_2010-2020.csv",
        "EPA Air Quality Data": "data/raw/epa_aqi/epa_historical_air_quality.csv",
        "EPA Water Quality Data": "data/raw/epa_water/epa_water_quality.csv",
        "Census Demographic Data": "data/raw/census/census_demographics_2020.csv"
    }
    
    found_data = []
    missing_data = []
    
    for name, filepath in data_files.items():
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size
            print(f"✅ {name}: {filepath} ({size:,} bytes)")
            found_data.append(name)
        else:
            print(f"❌ {name}: {filepath} (MISSING)")
            missing_data.append(name)
    
    return found_data, missing_data


def print_download_instructions():
    """Print step-by-step download instructions."""
    print("\n📋 STEP-BY-STEP DOWNLOAD INSTRUCTIONS:")
    print("\n1. SEER CANCER DATA:")
    print("   a) Go to https://seer.cancer.gov/data/")
    print("   b) Register for SEER data access")
    print("   c) Download pancreatic cancer incidence data (2010-2020)")
    print("   d) Save as: data/raw/seer/seer_incidence_2010-2020.csv")
    
    print("\n2. EPA AIR QUALITY DATA:")
    print("   a) Go to https://www.kaggle.com/datasets/epa/epa-historical-air-quality")
    print("   b) Download the dataset (requires Kaggle account)")
    print("   c) Extract the CSV file")
    print("   d) Save as: data/raw/epa_aqi/epa_historical_air_quality.csv")
    
    print("\n3. EPA WATER QUALITY DATA (Optional):")
    print("   a) Go to https://www.epa.gov/ground-water-and-drinking-water")
    print("   b) Navigate to drinking water data")
    print("   c) Download county-level contaminant data")
    print("   d) Save as: data/raw/epa_water/epa_water_quality.csv")
    
    print("\n4. CENSUS DEMOGRAPHIC DATA (Optional):")
    print("   a) Go to https://api.census.gov/data/key_signup.html")
    print("   b) Sign up for free API key")
    print("   c) Use the API or download pre-made datasets")
    print("   d) Save as: data/raw/census/census_demographics_2020.csv")


def print_data_format_requirements():
    """Print data format requirements."""
    print("\n📊 DATA FORMAT REQUIREMENTS:")
    
    print("\nSEER Cancer Data Format:")
    print("   Required columns: county, state, year, incidence_count, population")
    print("   Example:")
    print("   county,state,year,incidence_count,population")
    print("   Los Angeles,CA,2020,150,10000000")
    
    print("\nEPA Air Quality Data Format:")
    print("   Required columns: State Name, County Name, Date Local, Arithmetic Mean, AQI")
    print("   Example:")
    print("   State Name,County Name,Date Local,Arithmetic Mean,AQI")
    print("   California,Los Angeles,2020-01-01,12.5,52")
    
    print("\nEPA Water Quality Data Format:")
    print("   Required columns: State, County, Contaminant, Level, Date")
    print("   Example:")
    print("   State,County,Contaminant,Level,Date")
    print("   California,Los Angeles,Arsenic,0.005,2020-01-01")
    
    print("\nCensus Demographic Data Format:")
    print("   Required columns: State, County, Population, Median_Income, Education_Level")
    print("   Example:")
    print("   State,County,Population,Median_Income,Education_Level")
    print("   California,Los Angeles,10000000,75000,Bachelors")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Real Data Download Helper")
    parser.add_argument("--open-sources", action="store_true", help="Open data source websites")
    parser.add_argument("--check-data", action="store_true", help="Check for existing data")
    parser.add_argument("--instructions", action="store_true", help="Show download instructions")
    parser.add_argument("--format", action="store_true", help="Show data format requirements")
    
    args = parser.parse_args()
    
    print_data_requirements()
    
    if args.open_sources:
        open_data_sources()
    
    if args.check_data:
        check_data_directories()
        found_data, missing_data = check_existing_data()
        
        if found_data:
            print(f"\n✅ Found {len(found_data)} data sources")
        if missing_data:
            print(f"\n❌ Missing {len(missing_data)} data sources")
    
    if args.instructions:
        print_download_instructions()
    
    if args.format:
        print_data_format_requirements()
    
    if not any([args.open_sources, args.check_data, args.instructions, args.format]):
        # Default behavior
        check_data_directories()
        found_data, missing_data = check_existing_data()
        
        if missing_data:
            print(f"\n⚠️  You need to download {len(missing_data)} data sources to run the analysis.")
            print("Run with --instructions to see download steps.")
            print("Run with --open-sources to open data websites.")
        else:
            print(f"\n🎉 All required data sources are available!")
            print("You can now run the analysis with: python main.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main() 