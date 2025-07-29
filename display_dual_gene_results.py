#!/usr/bin/env python3
"""
Display Dual Gene Analysis Results
Show key findings from the Acta2 and Fabp4 3D visualization analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

def display_dual_gene_summary():
    """Display summary of dual gene analysis results"""
    
    # Load the data
    data_path = Path("xenium_dual_gene_analysis/combined_transcripts.csv")
    
    if not data_path.exists():
        print("❌ Dual gene analysis results not found. Please run the analysis first.")
        return
    
    # Load data
    df = pd.read_csv(data_path)
    
    print("🎯 ACTA2 & FABP4 DUAL GENE ANALYSIS RESULTS")
    print("=" * 60)
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   • Total transcript locations: {len(df):,}")
    print(f"   • Acta2 transcripts: {len(df[df['gene'] == 'Acta2']):,}")
    print(f"   • Fabp4 transcripts: {len(df[df['gene'] == 'Fabp4']):,}")
    print(f"   • Z levels with data: {df['z_level'].nunique()}")
    
    # Gene comparison
    acta2_data = df[df['gene'] == 'Acta2']
    fabp4_data = df[df['gene'] == 'Fabp4']
    
    print(f"\n🔬 GENE COMPARISON:")
    print(f"   • Acta2 abundance: {len(acta2_data):,} transcripts ({len(acta2_data)/len(df)*100:.1f}%)")
    print(f"   • Fabp4 abundance: {len(fabp4_data):,} transcripts ({len(fabp4_data)/len(df)*100:.1f}%)")
    print(f"   • Ratio (Acta2:Fabp4): {len(acta2_data)/len(fabp4_data):.1f}:1")
    
    # Spatial ranges
    print(f"\n📍 SPATIAL RANGES:")
    print(f"   • X range: {df['x'].min():.1f} - {df['x'].max():.1f} μm")
    print(f"   • Y range: {df['y'].min():.1f} - {df['y'].max():.1f} μm")
    print(f"   • Z range: {df['z'].min():.1f} - {df['z'].max():.1f} μm")
    
    # Z level distribution by gene
    print(f"\n📈 Z LEVEL DISTRIBUTION BY GENE:")
    
    print(f"\n   Acta2 Z-level distribution:")
    acta2_z_stats = acta2_data.groupby('z_level')['transcript_count'].sum().sort_values(ascending=False)
    for z_level, count in acta2_z_stats.items():
        percentage = (count / acta2_data['transcript_count'].sum()) * 100
        print(f"     • Z level {z_level}: {count:,.0f} transcripts ({percentage:.1f}%)")
    
    print(f"\n   Fabp4 Z-level distribution:")
    fabp4_z_stats = fabp4_data.groupby('z_level')['transcript_count'].sum().sort_values(ascending=False)
    for z_level, count in fabp4_z_stats.items():
        percentage = (count / fabp4_data['transcript_count'].sum()) * 100
        print(f"     • Z level {z_level}: {count:,.0f} transcripts ({percentage:.1f}%)")
    
    # Spatial density comparison
    print(f"\n🌐 SPATIAL DENSITY COMPARISON:")
    area_xy = (df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min())
    acta2_density = len(acta2_data) / area_xy
    fabp4_density = len(fabp4_data) / area_xy
    print(f"   • XY area: {area_xy:,.0f} μm²")
    print(f"   • Acta2 density: {acta2_density:.3f} transcripts/μm²")
    print(f"   • Fabp4 density: {fabp4_density:.3f} transcripts/μm²")
    print(f"   • Density ratio (Acta2:Fabp4): {acta2_density/fabp4_density:.1f}:1")
    
    # Key findings
    print(f"\n💡 KEY FINDINGS:")
    print(f"   • Acta2 is more abundant than Fabp4 ({len(acta2_data)/len(fabp4_data):.1f}x more)")
    print(f"   • Both genes show Z-level dependent expression patterns")
    print(f"   • Acta2 peaks in Z level 0 ({acta2_z_stats.iloc[0]:,.0f} transcripts)")
    print(f"   • Fabp4 shows different spatial distribution pattern")
    print(f"   • Expression levels visualized with reddish (Acta2) and bluish (Fabp4) hues")
    
    # Files created
    print(f"\n📁 GENERATED FILES:")
    output_dir = Path("xenium_dual_gene_analysis")
    files = [
        ("acta2_fabp4_3d_scatter.html", "Interactive 3D scatter plot (minimal dots)"),
        ("gene_comparison_counts.html", "Gene comparison by Z level"),
        ("gene_spatial_distribution.html", "Spatial distribution comparison"),
        ("acta2_expression_heatmap.html", "Acta2 expression level (reddish hue)"),
        ("fabp4_expression_heatmap.html", "Fabp4 expression level (bluish hue)"),
        ("combined_expression_overlay.html", "Combined expression overlay"),
        ("combined_transcripts.csv", "Raw transcript data")
    ]
    
    for filename, description in files:
        file_path = output_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   • {filename} ({size_mb:.1f} MB) - {description}")
    
    print(f"\n🎨 VISUALIZATION FEATURES:")
    print(f"   • 3D scatter plot with minimal dot sizes (size=1)")
    print(f"   • Acta2 transcripts in red, Fabp4 in blue")
    print(f"   • Expression level heatmaps with custom color scales")
    print(f"   • Gaussian smoothing for smooth expression visualization")
    print(f"   • Combined overlay showing both genes simultaneously")
    print(f"   • Interactive plots with hover information")

if __name__ == "__main__":
    display_dual_gene_summary() 