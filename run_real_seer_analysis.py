#!/usr/bin/env python3
"""
Real SEER Data Analysis

This script downloads real SEER pancreatic cancer data and runs the top quartile analysis.
"""

import sys
from pathlib import Path
from loguru import logger
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_acquisition.seer_downloader import SEERDownloader
from analysis.seer_zipcode_analysis import SEERZipcodeAnalyzer
from utils.api_keys import setup_api_keys, get_api_key


def download_real_seer_data():
    """Download real SEER pancreatic cancer data."""
    print("🔍 Downloading Real SEER Data...")
    print("=" * 50)
    
    # Setup API keys
    api_keys = setup_api_keys()
    seer_api_key = get_api_key('seer', api_keys)
    
    if not seer_api_key or seer_api_key == 'your_seer_api_key_here':
        print("❌ No valid SEER API key found.")
        print("💡 Run: python3 add_seer_api_key.py to add your API key")
        return None
    
    # Initialize downloader
    downloader = SEERDownloader(api_key=seer_api_key)
    
    # Download zip code level data
    print("📥 Downloading SEER pancreatic cancer data...")
    data = downloader.download_zipcode_data()
    
    if data.empty:
        print("❌ No data downloaded. Check your API key and SEER access.")
        return None
    
    # Show data summary
    summary = downloader.get_data_summary(data)
    print("\n📊 DOWNLOADED DATA SUMMARY:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    return data


def run_real_analysis(data):
    """Run top quartile analysis on real SEER data."""
    print("\n🎯 Running Real SEER Analysis...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SEERZipcodeAnalyzer()
    
    # Run complete USA analysis
    results = analyzer.run_complete_usa_analysis()
    
    if not results:
        print("❌ No results generated.")
        return
    
    # Display results
    print("\n" + "=" * 50)
    print("📈 REAL SEER ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total zip codes analyzed: {results['total_zipcodes']:,}")
    print(f"   Zip codes in top quartile: {results['top_quartile_zipcodes']:,}")
    print(f"   Top quartile threshold: {results['top_quartile_threshold']:.2f} per 100,000")
    
    if 'top_quartile_data' in results and not results['top_quartile_data'].empty:
        print(f"\n🏆 TOP 5 ZIP CODES:")
        top_5 = results['top_quartile_data'].head(5)
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"   {i}. Zip {row['zip_code']} ({row['county']}, {row['state']}): {row['incidence_rate']:.2f} per 100,000")
    
    print(f"\n📁 GENERATED FILES:")
    print(f"   📊 Chart: {results['plot_file']}")
    print(f"   🗺️ Map: {results['map_file']}")
    print(f"   📄 Report: {results['report_file']}")
    
    print("\n" + "=" * 50)
    print("✅ Real SEER analysis completed successfully!")
    print("📁 Check the results directory for detailed reports and visualizations.")
    print("🌐 Open the HTML files in your browser to view interactive maps.")


def main():
    """Main function to run real SEER analysis."""
    parser = argparse.ArgumentParser(description="Real SEER Data Analysis")
    parser.add_argument("--download-only", action="store_true", 
                       help="Only download data, don't run analysis")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Only run analysis on existing data")
    
    args = parser.parse_args()
    
    print("🎯 REAL SEER PANCREATIC CANCER ANALYSIS")
    print("=" * 60)
    
    if args.analysis_only:
        # Run analysis on existing data
        print("📊 Running analysis on existing SEER data...")
        analyzer = SEERZipcodeAnalyzer()
        data = analyzer.load_seer_data()
        
        if data.empty:
            print("❌ No existing SEER data found.")
            print("💡 Run without --analysis-only to download data first")
            return
        
        run_real_analysis(data)
        
    elif args.download_only:
        # Only download data
        data = download_real_seer_data()
        if data is not None:
            print("\n✅ Data download completed!")
            print("💡 Run without --download-only to perform analysis")
        
    else:
        # Download data and run analysis
        data = download_real_seer_data()
        if data is not None:
            run_real_analysis(data)


if __name__ == "__main__":
    main() 