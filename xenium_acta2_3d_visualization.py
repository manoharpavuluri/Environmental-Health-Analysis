#!/usr/bin/env python3
"""
Xenium Acta2 3D Visualization
Extract and visualize Acta2 transcripts across Z levels in 3D space
"""

import zarr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class XeniumActa2Visualizer:
    """Visualize Acta2 transcripts in 3D space across Z levels"""
    
    def __init__(self, data_path: str):
        """Initialize with path to Xenium data"""
        self.data_path = data_path
        self.zarr_data = zarr.open(data_path)
        self.gene_panel = self._load_gene_panel()
        self.acta2_codeword = self._find_acta2_codeword()
        
    def _load_gene_panel(self) -> Dict:
        """Load gene panel information"""
        panel_path = Path(self.data_path).parent / "gene_panel.json"
        with open(panel_path, 'r') as f:
            return json.load(f)
    
    def _find_acta2_codeword(self) -> int:
        """Find the codeword for Acta2 gene"""
        targets = self.gene_panel.get('payload', {}).get('targets', [])
        for target in targets:
            if target.get('type', {}).get('data', {}).get('name') == 'Acta2':
                return target['codewords'][0]  # First codeword
        raise ValueError("Acta2 not found in gene panel")
    
    def extract_acta2_transcripts(self) -> pd.DataFrame:
        """Extract all Acta2 transcripts with their 3D coordinates"""
        print(f"🔍 Extracting Acta2 transcripts (codeword: {self.acta2_codeword})...")
        
        all_transcripts = []
        
        # Iterate through all Z levels and chunks
        grid_keys = list(self.zarr_data['grids'].keys())
        for z_level_str in grid_keys:
            z_level = int(z_level_str)
            print(f"  Processing Z level {z_level}...")
            
            # Get all chunks for this Z level
            z_chunks = self.zarr_data['grids'][z_level_str]
            
            for chunk_name in z_chunks.keys():
                chunk = z_chunks[chunk_name]
                
                # Get transcript data
                locations = chunk['location'][:]  # (N, 3) - x, y, z coordinates
                gene_ids = chunk['gene_identity'][:]  # (N, 1) - gene codewords
                
                # Find Acta2 transcripts
                acta2_mask = gene_ids.flatten() == self.acta2_codeword
                
                if np.any(acta2_mask):
                    acta2_locations = locations[acta2_mask]
                    
                    # Create DataFrame for this chunk
                    chunk_data = pd.DataFrame({
                        'x': acta2_locations[:, 0],
                        'y': acta2_locations[:, 1], 
                        'z': acta2_locations[:, 2],
                        'z_level': z_level,
                        'transcript_count': np.ones(len(acta2_locations)),  # Each location = 1 transcript
                        'chunk': chunk_name
                    })
                    
                    all_transcripts.append(chunk_data)
        
        if not all_transcripts:
            print("⚠️  No Acta2 transcripts found!")
            return pd.DataFrame()
        
        # Combine all chunks
        acta2_df = pd.concat(all_transcripts, ignore_index=True)
        print(f"✅ Found {len(acta2_df)} Acta2 transcript locations")
        
        return acta2_df
    
    def create_3d_scatter_plot(self, acta2_df: pd.DataFrame) -> go.Figure:
        """Create interactive 3D scatter plot of Acta2 transcripts"""
        if acta2_df.empty:
            print("⚠️  No data to plot")
            return go.Figure()
        
        # Color by Z level
        fig = px.scatter_3d(
            acta2_df, 
            x='x', y='y', z='z',
            color='z_level',
            size='transcript_count',
            title='Acta2 Transcripts in 3D Space',
            labels={'x': 'X (μm)', 'y': 'Y (μm)', 'z': 'Z (μm)', 'z_level': 'Z Level'},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)", 
                zaxis_title="Z (μm)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title_x=0.5,
            height=800
        )
        
        return fig
    
    def create_z_level_analysis(self, acta2_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
        """Create analysis plots for Z level distribution"""
        if acta2_df.empty:
            return go.Figure(), go.Figure()
        
        # Z level statistics
        z_stats = acta2_df.groupby('z_level').agg({
            'transcript_count': ['count', 'sum', 'mean'],
            'x': ['min', 'max'],
            'y': ['min', 'max'],
            'z': ['min', 'max']
        }).round(2)
        
        print("\n📊 Z Level Statistics:")
        print(z_stats)
        
        # 1. Transcript count by Z level
        fig1 = px.bar(
            acta2_df.groupby('z_level')['transcript_count'].sum().reset_index(),
            x='z_level', y='transcript_count',
            title='Total Acta2 Transcript Count by Z Level',
            labels={'z_level': 'Z Level', 'transcript_count': 'Total Transcript Count'}
        )
        
        # 2. Spatial distribution by Z level
        fig2 = px.scatter(
            acta2_df, x='x', y='y', color='z_level',
            size='transcript_count',
            title='Acta2 Spatial Distribution by Z Level (XY projection)',
            labels={'x': 'X (μm)', 'y': 'Y (μm)', 'z_level': 'Z Level'},
            color_continuous_scale='viridis'
        )
        
        return fig1, fig2
    
    def create_density_heatmap(self, acta2_df: pd.DataFrame) -> go.Figure:
        """Create density heatmap of Acta2 transcripts"""
        if acta2_df.empty:
            return go.Figure()
        
        # Create 2D histogram
        x_bins = np.linspace(acta2_df['x'].min(), acta2_df['x'].max(), 50)
        y_bins = np.linspace(acta2_df['y'].min(), acta2_df['y'].max(), 50)
        
        H, xedges, yedges = np.histogram2d(
            acta2_df['x'], acta2_df['y'], 
            bins=[x_bins, y_bins],
            weights=acta2_df['transcript_count']
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=H.T,
            x=xedges[:-1],
            y=yedges[:-1],
            colorscale='viridis',
            name='Acta2 Density'
        ))
        
        fig.update_layout(
            title='Acta2 Transcript Density Heatmap (XY projection)',
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            height=600
        )
        
        return fig
    
    def run_analysis(self) -> Dict:
        """Run complete Acta2 analysis"""
        print("🚀 Starting Acta2 3D Visualization Analysis")
        print("=" * 50)
        
        # Extract Acta2 transcripts
        acta2_df = self.extract_acta2_transcripts()
        
        if acta2_df.empty:
            print("❌ No Acta2 transcripts found. Analysis complete.")
            return {}
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        
        # 3D scatter plot
        fig_3d = self.create_3d_scatter_plot(acta2_df)
        
        # Z level analysis
        fig_z_count, fig_z_spatial = self.create_z_level_analysis(acta2_df)
        
        # Density heatmap
        fig_density = self.create_density_heatmap(acta2_df)
        
        # Save plots
        output_dir = Path("xenium_acta2_analysis")
        output_dir.mkdir(exist_ok=True)
        
        fig_3d.write_html(output_dir / "acta2_3d_scatter.html")
        fig_z_count.write_html(output_dir / "acta2_z_level_counts.html")
        fig_z_spatial.write_html(output_dir / "acta2_z_level_spatial.html")
        fig_density.write_html(output_dir / "acta2_density_heatmap.html")
        
        # Save data
        acta2_df.to_csv(output_dir / "acta2_transcripts.csv", index=False)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
        print(f"📁 Files created:")
        print(f"   - acta2_3d_scatter.html (Interactive 3D plot)")
        print(f"   - acta2_z_level_counts.html (Z level statistics)")
        print(f"   - acta2_z_level_spatial.html (Spatial distribution)")
        print(f"   - acta2_density_heatmap.html (Density heatmap)")
        print(f"   - acta2_transcripts.csv (Raw data)")
        
        return {
            'data': acta2_df,
            'figures': {
                '3d_scatter': fig_3d,
                'z_count': fig_z_count,
                'z_spatial': fig_z_spatial,
                'density': fig_density
            },
            'output_dir': output_dir
        }

def main():
    """Main function to run Acta2 analysis"""
    data_path = "/Volumes/T7 Shield/Xenium data/output-XETG00234__0056769__Region_1__20250124__224947/transcripts.zarr"
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"❌ Data path not found: {data_path}")
        return
    
    # Create visualizer and run analysis
    visualizer = XeniumActa2Visualizer(data_path)
    results = visualizer.run_analysis()
    
    if results:
        print(f"\n🎯 Summary:")
        print(f"   - Total Acta2 transcript locations: {len(results['data'])}")
        print(f"   - Z levels with Acta2: {results['data']['z_level'].nunique()}")
        print(f"   - Total transcript count: {results['data']['transcript_count'].sum()}")
        print(f"   - Spatial range X: {results['data']['x'].min():.1f} - {results['data']['x'].max():.1f} μm")
        print(f"   - Spatial range Y: {results['data']['y'].min():.1f} - {results['data']['y'].max():.1f} μm")
        print(f"   - Spatial range Z: {results['data']['z'].min():.1f} - {results['data']['z'].max():.1f} μm")

if __name__ == "__main__":
    main() 