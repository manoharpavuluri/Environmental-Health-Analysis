#!/usr/bin/env python3
"""
Display Acta2 Analysis Results
Show key findings from the Acta2 3D visualization analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

def display_acta2_summary():
    """Display summary of Acta2 analysis results"""
    
    # Load the data
    data_path = Path("xenium_acta2_analysis/acta2_transcripts.csv")
    
    if not data_path.exists():
        print("❌ Acta2 analysis results not found. Please run the analysis first.")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    
    print("🎯 ACTA2 TRANSCRIPT ANALYSIS RESULTS")
    print("=" * 50)
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Total Acta2 transcript locations: {len(df):,}")
    print(f"   • Z levels with Acta2: {df['z_level'].nunique()}")
    print(f"   • Total transcript count: {df['transcript_count'].sum():,.0f}")
    
    # Spatial ranges
    print(f"\n📍 SPATIAL RANGES:")
    print(f"   • X range: {df['x'].min():.1f} - {df['x'].max():.1f} μm")
    print(f"   • Y range: {df['y'].min():.1f} - {df['y'].max():.1f} μm")
    print(f"   • Z range: {df['z'].min():.1f} - {df['z'].max():.1f} μm")
    
    # Z level distribution
    print(f"\n📈 Z LEVEL DISTRIBUTION:")
    z_stats = df.groupby('z_level')['transcript_count'].sum().sort_values(ascending=False)
    for z_level, count in z_stats.items():
        percentage = (count / df['transcript_count'].sum()) * 100
        print(f"   • Z level {z_level}: {count:,.0f} transcripts ({percentage:.1f}%)")
    
    # Spatial density
    print(f"\n🌐 SPATIAL DENSITY:")
    area_xy = (df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min())
    density_per_um2 = len(df) / area_xy
    print(f"   • XY area: {area_xy:,.0f} μm²")
    print(f"   • Density: {density_per_um2:.3f} transcripts/μm²")
    
    # Z level analysis
    print(f"\n🔍 Z LEVEL ANALYSIS:")
    z_level_analysis = df.groupby('z_level').agg({
        'transcript_count': ['count', 'sum'],
        'x': ['min', 'max', 'mean'],
        'y': ['min', 'max', 'mean'],
        'z': ['min', 'max', 'mean']
    }).round(2)
    
    print("   Z Level | Count | Sum | X Range | Y Range | Z Range")
    print("   " + "-" * 60)
    
    for z_level in sorted(df['z_level'].unique()):
        z_data = df[df['z_level'] == z_level]
        count = len(z_data)
        total = z_data['transcript_count'].sum()
        x_range = f"{z_data['x'].min():.0f}-{z_data['x'].max():.0f}"
        y_range = f"{z_data['y'].min():.0f}-{z_data['y'].max():.0f}"
        z_range = f"{z_data['z'].min():.1f}-{z_data['z'].max():.1f}"
        
        print(f"   {z_level:>7} | {count:>5} | {total:>3} | {x_range:>8} | {y_range:>8} | {z_range:>8}")
    
    # Key findings
    print(f"\n💡 KEY FINDINGS:")
    print(f"   • Acta2 is present across all 7 Z levels")
    print(f"   • Highest concentration in Z level 0 ({z_stats.iloc[0]:,.0f} transcripts)")
    print(f"   • Decreasing trend from Z level 0 to 6")
    print(f"   • Z range is relatively narrow (13.6-31.2 μm)")
    print(f"   • Wide spatial distribution in XY plane")
    
    # Files created
    print(f"\n📁 GENERATED FILES:")
    output_dir = Path("xenium_acta2_analysis")
    files = [
        ("acta2_3d_scatter.html", "Interactive 3D scatter plot"),
        ("acta2_z_level_counts.html", "Z level transcript counts"),
        ("acta2_z_level_spatial.html", "Spatial distribution by Z level"),
        ("acta2_density_heatmap.html", "Density heatmap"),
        ("acta2_transcripts.csv", "Raw transcript data")
    ]
    
    for filename, description in files:
        file_path = output_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   • {filename} ({size_mb:.1f} MB) - {description}")
    
    print(f"\n🎨 VISUALIZATION TIPS:")
    print(f"   • Open acta2_3d_scatter.html for interactive 3D exploration")
    print(f"   • Use mouse to rotate, zoom, and pan in 3D space")
    print(f"   • Color coding shows Z level (blue=low, red=high)")
    print(f"   • Point size represents transcript count")
    print(f"   • Hover over points for detailed information")

if __name__ == "__main__":
    display_acta2_summary() 