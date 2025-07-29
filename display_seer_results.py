#!/usr/bin/env python3
"""
Display SEER Analysis Results

This script displays the results of the SEER zip code analysis in a user-friendly format.
"""

import pandas as pd
from pathlib import Path
import webbrowser
import os


def display_results():
    """Display the SEER analysis results."""
    print("🎯 PANCREATIC CANCER INCIDENCE ANALYSIS RESULTS")
    print("=" * 60)
    
    # Load the sample data to show the actual values
    seer_file = Path("data/raw/seer/seer_incidence_2010-2020.csv")
    if seer_file.exists():
        data = pd.read_csv(seer_file)
        
        # Filter for NY and TX
        ny_tx_data = data[data['state'].isin(['NY', 'TX'])].copy()
        
        print("\n📊 DATA SUMMARY:")
        print(f"   Total counties analyzed: {len(ny_tx_data)}")
        print(f"   New York counties: {len(ny_tx_data[ny_tx_data['state'] == 'NY'])}")
        print(f"   Texas counties: {len(ny_tx_data[ny_tx_data['state'] == 'TX'])}")
        
        print("\n🏆 TOP COUNTIES BY INCIDENCE RATE:")
        top_counties = ny_tx_data.nlargest(5, 'incidence_rate')
        for i, (idx, row) in enumerate(top_counties.iterrows(), 1):
            print(f"   {i}. {row['county']}, {row['state']}: {row['incidence_rate']:.2f} per 100,000")
        
        print("\n📈 STATE COMPARISON:")
        for state in ['NY', 'TX']:
            state_data = ny_tx_data[ny_tx_data['state'] == state]
            if not state_data.empty:
                mean_rate = state_data['incidence_rate'].mean()
                max_rate = state_data['incidence_rate'].max()
                max_county = state_data.loc[state_data['incidence_rate'].idxmax(), 'county']
                print(f"   {state}: Mean {mean_rate:.2f}, Max {max_rate:.2f} ({max_county})")
    
    print("\n📁 GENERATED FILES:")
    results_dir = Path("data/results/seer_zipcode")
    if results_dir.exists():
        files = list(results_dir.glob("*"))
        for file in files:
            file_type = "📊 Chart" if file.suffix == ".png" else "🗺️ Map" if file.suffix == ".html" and "map" in file.name else "📄 Report"
            print(f"   {file_type}: {file.name}")
    
    print("\n🌐 OPENING INTERACTIVE MAPS...")
    
    # Open the HTML files in browser
    map_files = list(results_dir.glob("*map*.html"))
    for map_file in map_files:
        print(f"   Opening: {map_file.name}")
        webbrowser.open(f"file://{map_file.absolute()}")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete! Check your browser for interactive maps.")
    print("📊 The PNG files contain detailed charts of incidence rates.")
    print("📄 The HTML report files contain comprehensive analysis summaries.")


if __name__ == "__main__":
    display_results() 