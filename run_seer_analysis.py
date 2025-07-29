#!/usr/bin/env python3
"""
SEER Zip Code Analysis Runner

This script runs the SEER zip code analysis to identify and visualize
zip codes with the highest pancreatic cancer incidence in New York and Texas.
"""

import sys
from pathlib import Path
from loguru import logger
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from analysis.seer_zipcode_analysis import SEERZipcodeAnalyzer


def main():
    """Main function to run SEER zip code analysis."""
    parser = argparse.ArgumentParser(description="SEER Zip Code Analysis")
    parser.add_argument("--states", nargs="+", default=["NY", "TX"], 
                       help="States to analyze (default: NY TX)")
    parser.add_argument("--output-dir", default="data/results/seer_zipcode",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🎯 SEER Zip Code Analysis for Pancreatic Cancer")
    print("=" * 60)
    print(f"📊 Analyzing states: {', '.join(args.states)}")
    print(f"📁 Output directory: {args.output_dir}")
    print()
    
    # Initialize analyzer
    analyzer = SEERZipcodeAnalyzer()
    
    # Run analysis
    results = analyzer.run_complete_analysis(args.states)
    
    if not results:
        print("❌ No results generated. Check if SEER data is available.")
        return
    
    # Display results
    print("\n" + "=" * 60)
    print("📈 ANALYSIS RESULTS")
    print("=" * 60)
    
    for state, result in results.items():
        print(f"\n🏛️ {state} ANALYSIS:")
        print(f"   📊 Counties analyzed: {result['total_counties']}")
        print(f"   📈 Mean incidence rate: {result['mean_incidence_rate']:.2f} per 100,000")
        print(f"   🔥 Max incidence rate: {result['max_incidence_rate']:.2f} per 100,000")
        
        if not result['high_incidence_zipcodes'].empty:
            print(f"   🏆 Top 3 counties with highest incidence:")
            top_3 = result['high_incidence_zipcodes'].head(3)
            for i, (idx, row) in enumerate(top_3.iterrows(), 1):
                print(f"      {i}. {row['county']}: {row['incidence_rate']:.2f} per 100,000")
        
        print(f"   📄 Report: {result['report_file']}")
        print(f"   📊 Chart: {result['plot_file']}")
        print(f"   🗺️ Map: {result['map_file']}")
    
    print("\n" + "=" * 60)
    print("✅ Analysis completed successfully!")
    print("📁 Check the results directory for detailed reports and visualizations.")
    print("🌐 Open the HTML files in your browser to view interactive maps.")


if __name__ == "__main__":
    main() 