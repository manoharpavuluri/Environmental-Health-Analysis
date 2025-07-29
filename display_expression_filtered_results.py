#!/usr/bin/env python3
"""
Display Expression-Filtered Analysis Results
Show summary statistics and open the 3D visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import webbrowser
import os

def display_results():
    """Display summary of expression-filtered analysis results"""
    print("🎯 Expression-Filtered 3D Visualization Results")
    print("=" * 60)
    
    # Load the data
    data_file = Path("xenium_expression_filtered_analysis/expression_filtered_transcripts.csv")
    
    if not data_file.exists():
        print("❌ Results file not found. Please run the analysis first.")
        return
    
    df = pd.read_csv(data_file)
    
    print(f"📊 Dataset Summary:")
    print(f"   - Total transcript locations: {len(df):,}")
    print(f"   - Genes analyzed: {df['gene'].nunique()}")
    print(f"   - Z levels with data: {df['z_level'].nunique()}")
    
    print(f"\n🧬 Gene-Specific Statistics:")
    for gene in df['gene'].unique():
        gene_data = df[df['gene'] == gene]
        print(f"   {gene}:")
        print(f"     - Transcript count: {len(gene_data):,}")
        print(f"     - Mean expression intensity: {gene_data['expression_intensity'].mean():.3f}")
        print(f"     - Max expression intensity: {gene_data['expression_intensity'].max():.3f}")
        print(f"     - Z levels present: {gene_data['z_level'].nunique()}")
    
    print(f"\n📍 Spatial Distribution:")
    print(f"   - X range: {df['x'].min():.1f} - {df['x'].max():.1f} μm")
    print(f"   - Y range: {df['y'].min():.1f} - {df['y'].max():.1f} μm")
    print(f"   - Z range: {df['z'].min():.1f} - {df['z'].max():.1f} μm")
    
    print(f"\n📈 Expression Intensity Statistics:")
    print(f"   - Overall mean intensity: {df['expression_intensity'].mean():.3f}")
    print(f"   - Overall max intensity: {df['expression_intensity'].max():.3f}")
    print(f"   - Intensity std dev: {df['expression_intensity'].std():.3f}")
    
    # Show top expression areas
    print(f"\n🔥 Top Expression Areas (by gene):")
    for gene in df['gene'].unique():
        gene_data = df[df['gene'] == gene]
        top_areas = gene_data.nlargest(5, 'expression_intensity')
        print(f"   {gene} (top 5):")
        for _, row in top_areas.iterrows():
            print(f"     - Intensity {row['expression_intensity']:.3f} at ({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f})")
    
    # Check for visualization file
    html_file = Path("xenium_expression_filtered_analysis/expression_filtered_3d_visualization.html")
    if html_file.exists():
        print(f"\n📁 Generated Files:")
        print(f"   - 3D Visualization: {html_file}")
        print(f"   - Raw Data: {data_file}")
        
        # Open the visualization
        print(f"\n🌐 Opening 3D visualization in browser...")
        webbrowser.open(f'file://{html_file.absolute()}')
        
        print(f"\n💡 Visualization Features:")
        print(f"   - Epcam: Blue dots with intensity-based brightness")
        print(f"   - Chl1: Yellow dots with intensity-based brightness") 
        print(f"   - Retnla: Green dots with intensity-based brightness")
        print(f"   - Black background with white axes")
        print(f"   - Interactive 3D rotation and zoom")
        print(f"   - Hover for transcript details")
        print(f"   - Filtered for areas with ≥10 adjacent transcripts")
    else:
        print(f"❌ Visualization file not found: {html_file}")

if __name__ == "__main__":
    display_results() 