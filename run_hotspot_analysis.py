#!/usr/bin/env python3
"""
Hotspot Analysis Runner for Pancreatic Cancer and Air Quality.

This script specifically runs the hotspot analysis to identify areas with high
pancreatic cancer incidence that may be associated with poor air quality.
"""

import sys
import argparse
from pathlib import Path
import yaml
from loguru import logger
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_acquisition.seer_downloader import SEERDownloader
from data_acquisition.kaggle_epa_downloader import KaggleEPADownloader
from analysis.hotspot_analysis import HotspotAnalyzer
from utils.api_keys import setup_api_keys, get_api_key


def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/hotspot_analysis.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )


def load_data():
    """Load SEER and AQI data for hotspot analysis."""
    logger.info("Loading data for hotspot analysis...")
    
    # Setup API keys
    api_keys = setup_api_keys()
    
    # Initialize data downloaders
    seer_downloader = SEERDownloader(api_key=get_api_key('seer', api_keys))
    aqi_downloader = KaggleEPADownloader()
    
    # Load data
    seer_data = seer_downloader.get_all_data()
    aqi_data = aqi_downloader.get_all_data()
    
    return seer_data, aqi_data


def run_hotspot_analysis(seer_data, aqi_data):
    """Run the complete hotspot analysis."""
    logger.info("Starting hotspot analysis...")
    
    # Initialize hotspot analyzer
    hotspot_analyzer = HotspotAnalyzer()
    
    # Run analysis
    results = hotspot_analyzer.run_complete_hotspot_analysis(seer_data, aqi_data)
    
    return results


def print_results(results):
    """Print analysis results summary."""
    print("\n" + "="*60)
    print("🎯 PANCREATIC CANCER HOTSPOT ANALYSIS RESULTS")
    print("="*60)
    
    if not results:
        print("❌ No results available - insufficient data")
        return
    
    # Print hotspot summary
    hotspot_results = results.get('hotspot_results', {})
    if hotspot_results:
        print(f"\n📊 Hotspots Identified: {len(hotspot_results)} types")
        for hotspot_type, data in hotspot_results.items():
            print(f"  • {hotspot_type.replace('_', ' ').title()}: {len(data)} areas")
    
    # Print map files
    map_files = results.get('map_files', {})
    if map_files:
        print(f"\n🗺️ Interactive Maps Created: {len(map_files)}")
        for map_type, map_file in map_files.items():
            print(f"  • {map_type.replace('_', ' ').title()}: {map_file}")
    
    # Print report file
    report_file = results.get('report_file', '')
    if report_file:
        print(f"\n📄 Analysis Report: {report_file}")
    
    # Print data summary
    cancer_summary = results.get('cancer_data_summary', {})
    aqi_summary = results.get('air_quality_summary', {})
    
    if cancer_summary:
        print(f"\n📈 Cancer Data Summary:")
        print(f"  • Total records: {cancer_summary.get('total_records', 0)}")
        print(f"  • Unique counties: {cancer_summary.get('unique_counties', 0)}")
    
    if aqi_summary:
        print(f"\n🌬️ Air Quality Data Summary:")
        print(f"  • Total records: {aqi_summary.get('total_records', 0)}")
        print(f"  • Unique counties: {aqi_summary.get('unique_counties', 0)}")
    
    print("\n" + "="*60)


def main():
    """Main function to run hotspot analysis."""
    parser = argparse.ArgumentParser(description="Run pancreatic cancer hotspot analysis")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of data")
    parser.add_argument("--analysis-only", action="store_true", help="Run analysis only (use existing data)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        if args.analysis_only:
            # Load existing data
            logger.info("Loading existing data for analysis...")
            seer_data = {}
            aqi_data = {}
            
            # Load SEER data
            seer_files = list(Path("data/raw/seer").glob("*.csv"))
            if seer_files:
                for file in seer_files:
                    data_type = file.stem.split('_')[1]
                    seer_data[data_type] = pd.read_csv(file)
            
            # Load AQI data
            aqi_files = list(Path("data/raw/epa_aqi").glob("*.csv"))
            if aqi_files:
                for file in aqi_files:
                    data_type = file.stem.split('_')[1]
                    aqi_data[data_type] = pd.read_csv(file)
        else:
            # Download fresh data
            seer_data, aqi_data = load_data()
        
        # Check if we have sufficient data
        if not seer_data or not aqi_data:
            logger.error("Insufficient data for hotspot analysis")
            print("\n❌ Error: Insufficient data for analysis")
            print("Please ensure you have:")
            print("  • SEER pancreatic cancer data in data/raw/seer/")
            print("  • EPA air quality data in data/raw/epa_aqi/")
            print("\nTo download data, run without --analysis-only flag")
            return False
        
        # Run hotspot analysis
        results = run_hotspot_analysis(seer_data, aqi_data)
        
        # Print results
        print_results(results)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during hotspot analysis: {e}")
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 