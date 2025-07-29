#!/usr/bin/env python3
"""
Display USA SEER Analysis Results

This script displays the results of the USA SEER zip code analysis in a comprehensive format.
"""

import pandas as pd
from pathlib import Path
import webbrowser
import os


def display_usa_results():
    """Display the USA SEER analysis results."""
    print("🎯 USA PANCREATIC CANCER TOP QUARTILE ANALYSIS")
    print("=" * 70)
    
    # Load the data to show detailed results
    seer_file = Path("data/raw/seer/seer_incidence_2010-2020.csv")
    if seer_file.exists():
        data = pd.read_csv(seer_file)
        
        # Calculate top quartile threshold
        top_quartile_threshold = data['incidence_rate'].quantile(0.75)
        top_quartile_data = data[data['incidence_rate'] >= top_quartile_threshold].copy()
        
        print("\n📊 COMPREHENSIVE DATA SUMMARY:")
        print(f"   Total zip codes analyzed: {len(data):,}")
        print(f"   Zip codes in top quartile: {len(top_quartile_data):,}")
        print(f"   Top quartile threshold: {top_quartile_threshold:.2f} per 100,000")
        print(f"   States represented: {data['state'].nunique()}")
        print(f"   Incidence rate range: {data['incidence_rate'].min():.2f} - {data['incidence_rate'].max():.2f} per 100,000")
        
        print("\n🏆 TOP 10 ZIP CODES BY INCIDENCE RATE:")
        top_10 = top_quartile_data.nlargest(10, 'incidence_rate')
        for i, (idx, row) in enumerate(top_10.iterrows(), 1):
            print(f"   {i:2d}. Zip {row['zip_code']:5d} ({row['county']:20s}, {row['state']}): {row['incidence_rate']:5.2f} per 100,000")
        
        print("\n📈 STATE DISTRIBUTION OF TOP QUARTILE ZIP CODES:")
        state_counts = top_quartile_data['state'].value_counts()
        for state, count in state_counts.head(10).items():
            print(f"   {state}: {count:3d} zip codes")
        
        print("\n🔍 STATISTICAL INSIGHTS:")
        print(f"   Mean incidence rate (all): {data['incidence_rate'].mean():.2f} per 100,000")
        print(f"   Median incidence rate (all): {data['incidence_rate'].median():.2f} per 100,000")
        print(f"   Mean incidence rate (top quartile): {top_quartile_data['incidence_rate'].mean():.2f} per 100,000")
        print(f"   Standard deviation: {data['incidence_rate'].std():.2f} per 100,000")
        
        # Show states with highest average incidence
        state_avg = data.groupby('state')['incidence_rate'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        print(f"\n🏛️ STATES WITH HIGHEST AVERAGE INCIDENCE RATES:")
        for state in state_avg.head(5).index:
            avg_rate = state_avg.loc[state, 'mean']
            count = state_avg.loc[state, 'count']
            print(f"   {state}: {avg_rate:.2f} per 100,000 ({count} zip codes)")
    
    print("\n📁 GENERATED FILES:")
    results_dir = Path("data/results/seer_usa_analysis")
    if results_dir.exists():
        files = list(results_dir.glob("*"))
        for file in files:
            file_type = "📊 Chart" if file.suffix == ".png" else "🗺️ Map" if file.suffix == ".html" and "map" in file.name else "📄 Report"
            print(f"   {file_type}: {file.name}")
    
    print("\n🌐 OPENING INTERACTIVE MAP...")
    
    # Open the HTML map file in browser
    map_file = results_dir / "usa_top_quartile_map.html"
    if map_file.exists():
        print(f"   Opening: {map_file.name}")
        webbrowser.open(f"file://{map_file.absolute()}")
    
    print("\n" + "=" * 70)
    print("✅ USA Top Quartile Analysis Complete!")
    print("📊 The PNG file contains comprehensive charts showing:")
    print("   • Distribution of incidence rates with top quartile threshold")
    print("   • Top 20 zip codes by incidence rate")
    print("   • State distribution of top quartile zip codes")
    print("   • Box plots showing incidence rate variation by state")
    print("\n🗺️ The HTML map shows interactive markers for all top quartile zip codes")
    print("📄 The HTML report contains detailed statistics and rankings")
    print("\n💡 To use real SEER data:")
    print("   1. Download real SEER data from https://seer.cancer.gov/data/")
    print("   2. Replace the sample data file with real data")
    print("   3. Run the analysis again")


if __name__ == "__main__":
    display_usa_results() 