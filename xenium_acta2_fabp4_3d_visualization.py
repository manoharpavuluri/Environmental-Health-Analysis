#!/usr/bin/env python3
"""
Xenium Acta2 and Fabp4 3D Visualization
Extract and visualize Acta2 (red) and Fabp4 (blue) transcripts across Z levels in 3D space
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

class XeniumDualGeneVisualizer:
    """Visualize Acta2 and Fabp4 transcripts in 3D space across Z levels"""
    
    def __init__(self, data_path: str):
        """Initialize with path to Xenium data"""
        self.data_path = data_path
        self.zarr_data = zarr.open(data_path)
        self.gene_panel = self._load_gene_panel()
        self.acta2_codeword = self._find_gene_codeword('Acta2')
        self.fabp4_codeword = self._find_gene_codeword('Fabp4')
        
    def _load_gene_panel(self) -> Dict:
        """Load gene panel information"""
        panel_path = Path(self.data_path).parent / "gene_panel.json"
        with open(panel_path, 'r') as f:
            return json.load(f)
    
    def _find_gene_codeword(self, gene_name: str) -> int:
        """Find the codeword for a specific gene"""
        targets = self.gene_panel.get('payload', {}).get('targets', [])
        for target in targets:
            if target.get('type', {}).get('data', {}).get('name') == gene_name:
                return target['codewords'][0]  # First codeword
        raise ValueError(f"{gene_name} not found in gene panel")
    
    def extract_gene_transcripts(self, gene_name: str, codeword: int) -> pd.DataFrame:
        """Extract all transcripts for a specific gene with their 3D coordinates"""
        print(f"🔍 Extracting {gene_name} transcripts (codeword: {codeword})...")
        
        all_transcripts = []
        
        # Iterate through all Z levels and chunks
        grid_keys = list(self.zarr_data['grids'].keys())
        for z_level_str in grid_keys:
            z_level = int(z_level_str)
            print(f"  Processing Z level {z_level} for {gene_name}...")
            
            # Get all chunks for this Z level
            z_chunks = self.zarr_data['grids'][z_level_str]
            
            for chunk_name in z_chunks.keys():
                chunk = z_chunks[chunk_name]
                
                # Get transcript data
                locations = chunk['location'][:]  # (N, 3) - x, y, z coordinates
                gene_ids = chunk['gene_identity'][:]  # (N, 1) - gene codewords
                
                # Find transcripts for this gene
                gene_mask = gene_ids.flatten() == codeword
                
                if np.any(gene_mask):
                    gene_locations = locations[gene_mask]
                    
                    # Create DataFrame for this chunk
                    chunk_data = pd.DataFrame({
                        'x': gene_locations[:, 0],
                        'y': gene_locations[:, 1], 
                        'z': gene_locations[:, 2],
                        'z_level': z_level,
                        'transcript_count': np.ones(len(gene_locations)),  # Each location = 1 transcript
                        'chunk': chunk_name,
                        'gene': gene_name
                    })
                    
                    all_transcripts.append(chunk_data)
        
        if not all_transcripts:
            print(f"⚠️  No {gene_name} transcripts found!")
            return pd.DataFrame()
        
        # Combine all chunks
        gene_df = pd.concat(all_transcripts, ignore_index=True)
        print(f"✅ Found {len(gene_df)} {gene_name} transcript locations")
        
        return gene_df
    
    def extract_both_genes(self) -> pd.DataFrame:
        """Extract both Acta2 and Fabp4 transcripts"""
        print("🚀 Extracting both genes...")
        
        # Extract Acta2 transcripts
        acta2_df = self.extract_gene_transcripts('Acta2', self.acta2_codeword)
        
        # Extract Fabp4 transcripts
        fabp4_df = self.extract_gene_transcripts('Fabp4', self.fabp4_codeword)
        
        # Combine both datasets
        if not acta2_df.empty and not fabp4_df.empty:
            combined_df = pd.concat([acta2_df, fabp4_df], ignore_index=True)
            print(f"✅ Combined dataset: {len(combined_df)} total transcript locations")
            return combined_df
        elif not acta2_df.empty:
            print("⚠️  Only Acta2 data available")
            return acta2_df
        elif not fabp4_df.empty:
            print("⚠️  Only Fabp4 data available")
            return fabp4_df
        else:
            print("❌ No transcript data found for either gene")
            return pd.DataFrame()
    
    def create_3d_scatter_plot(self, combined_df: pd.DataFrame) -> go.Figure:
        """Create interactive 3D scatter plot of both genes with minimal dot sizes"""
        if combined_df.empty:
            print("⚠️  No data to plot")
            return go.Figure()
        
        # Create figure
        fig = go.Figure()
        
        # Plot Acta2 transcripts in red
        acta2_data = combined_df[combined_df['gene'] == 'Acta2']
        if not acta2_data.empty:
            fig.add_trace(go.Scatter3d(
                x=acta2_data['x'],
                y=acta2_data['y'],
                z=acta2_data['z'],
                mode='markers',
                marker=dict(
                    size=1,  # Minimal dot size
                    color='red',
                    opacity=0.8
                ),
                name='Acta2',
                hovertemplate='<b>Acta2</b><br>' +
                            'X: %{x:.1f} μm<br>' +
                            'Y: %{y:.1f} μm<br>' +
                            'Z: %{z:.1f} μm<br>' +
                            'Z Level: %{customdata}<extra></extra>',
                customdata=acta2_data['z_level']
            ))
        
        # Plot Fabp4 transcripts in blue
        fabp4_data = combined_df[combined_df['gene'] == 'Fabp4']
        if not fabp4_data.empty:
            fig.add_trace(go.Scatter3d(
                x=fabp4_data['x'],
                y=fabp4_data['y'],
                z=fabp4_data['z'],
                mode='markers',
                marker=dict(
                    size=1,  # Minimal dot size
                    color='blue',
                    opacity=0.8
                ),
                name='Fabp4',
                hovertemplate='<b>Fabp4</b><br>' +
                            'X: %{x:.1f} μm<br>' +
                            'Y: %{y:.1f} μm<br>' +
                            'Z: %{z:.1f} μm<br>' +
                            'Z Level: %{customdata}<extra></extra>',
                customdata=fabp4_data['z_level']
            ))
        
        # Update layout
        fig.update_layout(
            title='Acta2 (Red) and Fabp4 (Blue) Transcripts in 3D Space',
            scene=dict(
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)", 
                zaxis_title="Z (μm)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_gene_comparison_plots(self, combined_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
        """Create comparison plots for both genes"""
        if combined_df.empty:
            return go.Figure(), go.Figure()
        
        # 1. Transcript count comparison by Z level
        z_comparison = combined_df.groupby(['z_level', 'gene'])['transcript_count'].sum().reset_index()
        
        fig1 = px.bar(
            z_comparison, 
            x='z_level', 
            y='transcript_count',
            color='gene',
            title='Transcript Count by Z Level and Gene',
            labels={'z_level': 'Z Level', 'transcript_count': 'Transcript Count', 'gene': 'Gene'},
            color_discrete_map={'Acta2': 'red', 'Fabp4': 'blue'}
        )
        
        # 2. Spatial distribution comparison
        fig2 = px.scatter(
            combined_df, 
            x='x', 
            y='y', 
            color='gene',
            title='Spatial Distribution by Gene (XY projection)',
            labels={'x': 'X (μm)', 'y': 'Y (μm)', 'gene': 'Gene'},
            color_discrete_map={'Acta2': 'red', 'Fabp4': 'blue'}
        )
        
        return fig1, fig2
    
    def create_density_heatmaps(self, combined_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
        """Create density heatmaps for each gene with expression level visualization"""
        if combined_df.empty:
            return go.Figure(), go.Figure()
        
        # Acta2 density heatmap with reddish hue
        acta2_data = combined_df[combined_df['gene'] == 'Acta2']
        if not acta2_data.empty:
            # Create fine-grained bins for better resolution
            x_bins = np.linspace(acta2_data['x'].min(), acta2_data['x'].max(), 100)
            y_bins = np.linspace(acta2_data['y'].min(), acta2_data['y'].max(), 100)
            
            H_acta2, xedges, yedges = np.histogram2d(
                acta2_data['x'], acta2_data['y'], 
                bins=[x_bins, y_bins],
                weights=acta2_data['transcript_count']
            )
            
            # Apply Gaussian smoothing for better visualization
            from scipy.ndimage import gaussian_filter
            H_acta2_smooth = gaussian_filter(H_acta2, sigma=1.0)
            
            fig1 = go.Figure(data=go.Heatmap(
                z=H_acta2_smooth.T,
                x=xedges[:-1],
                y=yedges[:-1],
                colorscale=[
                    [0, 'rgba(255,255,255,0)'],  # Transparent white
                    [0.1, 'rgba(255,200,200,0.3)'],  # Light red
                    [0.3, 'rgba(255,150,150,0.6)'],  # Medium red
                    [0.6, 'rgba(255,100,100,0.8)'],  # Darker red
                    [1, 'rgba(255,50,50,1)']  # Bright red
                ],
                name='Acta2 Expression',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Acta2 Expression Level"),
                    thickness=15,
                    len=0.5
                )
            ))
            
            fig1.update_layout(
                title='Acta2 Expression Level Heatmap (XY projection)',
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                height=600,
                width=800
            )
        else:
            fig1 = go.Figure()
        
        # Fabp4 density heatmap with bluish hue
        fabp4_data = combined_df[combined_df['gene'] == 'Fabp4']
        if not fabp4_data.empty:
            # Create fine-grained bins for better resolution
            x_bins = np.linspace(fabp4_data['x'].min(), fabp4_data['x'].max(), 100)
            y_bins = np.linspace(fabp4_data['y'].min(), fabp4_data['y'].max(), 100)
            
            H_fabp4, xedges, yedges = np.histogram2d(
                fabp4_data['x'], fabp4_data['y'], 
                bins=[x_bins, y_bins],
                weights=fabp4_data['transcript_count']
            )
            
            # Apply Gaussian smoothing for better visualization
            from scipy.ndimage import gaussian_filter
            H_fabp4_smooth = gaussian_filter(H_fabp4, sigma=1.0)
            
            fig2 = go.Figure(data=go.Heatmap(
                z=H_fabp4_smooth.T,
                x=xedges[:-1],
                y=yedges[:-1],
                colorscale=[
                    [0, 'rgba(255,255,255,0)'],  # Transparent white
                    [0.1, 'rgba(200,200,255,0.3)'],  # Light blue
                    [0.3, 'rgba(150,150,255,0.6)'],  # Medium blue
                    [0.6, 'rgba(100,100,255,0.8)'],  # Darker blue
                    [1, 'rgba(50,50,255,1)']  # Bright blue
                ],
                name='Fabp4 Expression',
                showscale=True,
                colorbar=dict(
                    title=dict(text="Fabp4 Expression Level"),
                    thickness=15,
                    len=0.5
                )
            ))
            
            fig2.update_layout(
                title='Fabp4 Expression Level Heatmap (XY projection)',
                xaxis_title='X (μm)',
                yaxis_title='Y (μm)',
                height=600,
                width=800
            )
        else:
            fig2 = go.Figure()
        
        return fig1, fig2
    
    def create_combined_expression_overlay(self, combined_df: pd.DataFrame) -> go.Figure:
        """Create a combined expression overlay showing both genes"""
        if combined_df.empty:
            return go.Figure()
        
        # Get common spatial range
        x_min, x_max = combined_df['x'].min(), combined_df['x'].max()
        y_min, y_max = combined_df['y'].min(), combined_df['y'].max()
        
        # Create fine-grained bins
        x_bins = np.linspace(x_min, x_max, 100)
        y_bins = np.linspace(y_min, y_max, 100)
        
        # Calculate expression for each gene
        acta2_data = combined_df[combined_df['gene'] == 'Acta2']
        fabp4_data = combined_df[combined_df['gene'] == 'Fabp4']
        
        fig = go.Figure()
        
        # Add Acta2 expression (reddish)
        if not acta2_data.empty:
            H_acta2, _, _ = np.histogram2d(
                acta2_data['x'], acta2_data['y'], 
                bins=[x_bins, y_bins],
                weights=acta2_data['transcript_count']
            )
            
            from scipy.ndimage import gaussian_filter
            H_acta2_smooth = gaussian_filter(H_acta2, sigma=1.0)
            
            fig.add_trace(go.Heatmap(
                z=H_acta2_smooth.T,
                x=x_bins[:-1],
                y=y_bins[:-1],
                colorscale=[
                    [0, 'rgba(255,255,255,0)'],
                    [0.3, 'rgba(255,200,200,0.4)'],
                    [0.7, 'rgba(255,100,100,0.7)'],
                    [1, 'rgba(255,50,50,0.9)']
                ],
                name='Acta2 Expression',
                showscale=False
            ))
        
        # Add Fabp4 expression (bluish)
        if not fabp4_data.empty:
            H_fabp4, _, _ = np.histogram2d(
                fabp4_data['x'], fabp4_data['y'], 
                bins=[x_bins, y_bins],
                weights=fabp4_data['transcript_count']
            )
            
            from scipy.ndimage import gaussian_filter
            H_fabp4_smooth = gaussian_filter(H_fabp4, sigma=1.0)
            
            fig.add_trace(go.Heatmap(
                z=H_fabp4_smooth.T,
                x=x_bins[:-1],
                y=y_bins[:-1],
                colorscale=[
                    [0, 'rgba(255,255,255,0)'],
                    [0.3, 'rgba(200,200,255,0.4)'],
                    [0.7, 'rgba(100,100,255,0.7)'],
                    [1, 'rgba(50,50,255,0.9)']
                ],
                name='Fabp4 Expression',
                showscale=False
            ))
        
        fig.update_layout(
            title='Combined Expression Overlay: Acta2 (Reddish) + Fabp4 (Bluish)',
            xaxis_title='X (μm)',
            yaxis_title='Y (μm)',
            height=600,
            width=800,
            showlegend=True
        )
        
        return fig
    
    def run_analysis(self) -> Dict:
        """Run complete dual gene analysis"""
        print("🚀 Starting Acta2 and Fabp4 3D Visualization Analysis")
        print("=" * 60)
        
        # Extract both genes
        combined_df = self.extract_both_genes()
        
        if combined_df.empty:
            print("❌ No transcript data found. Analysis complete.")
            return {}
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        
        # 3D scatter plot
        fig_3d = self.create_3d_scatter_plot(combined_df)
        
        # Gene comparison plots
        fig_comparison, fig_spatial = self.create_gene_comparison_plots(combined_df)
        
        # Density heatmaps
        fig_acta2_density, fig_fabp4_density = self.create_density_heatmaps(combined_df)
        
        # Create combined expression overlay
        fig_combined_expression = self.create_combined_expression_overlay(combined_df)
        
        # Save plots
        output_dir = Path("xenium_dual_gene_analysis")
        output_dir.mkdir(exist_ok=True)
        
        fig_3d.write_html(output_dir / "acta2_fabp4_3d_scatter.html")
        fig_comparison.write_html(output_dir / "gene_comparison_counts.html")
        fig_spatial.write_html(output_dir / "gene_spatial_distribution.html")
        fig_acta2_density.write_html(output_dir / "acta2_expression_heatmap.html")
        fig_fabp4_density.write_html(output_dir / "fabp4_expression_heatmap.html")
        fig_combined_expression.write_html(output_dir / "combined_expression_overlay.html")
        
        # Save data
        combined_df.to_csv(output_dir / "combined_transcripts.csv", index=False)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
        print(f"📁 Files created:")
        print(f"   - acta2_fabp4_3d_scatter.html (Interactive 3D plot)")
        print(f"   - gene_comparison_counts.html (Gene comparison)")
        print(f"   - gene_spatial_distribution.html (Spatial distribution)")
        print(f"   - acta2_expression_heatmap.html (Acta2 expression level)")
        print(f"   - fabp4_expression_heatmap.html (Fabp4 expression level)")
        print(f"   - combined_expression_overlay.html (Combined expression)")
        print(f"   - combined_transcripts.csv (Raw data)")
        
        return {
            'data': combined_df,
            'figures': {
                '3d_scatter': fig_3d,
                'comparison': fig_comparison,
                'spatial': fig_spatial,
                'acta2_density': fig_acta2_density,
                'fabp4_density': fig_fabp4_density
            },
            'output_dir': output_dir
        }

def main():
    """Main function to run dual gene analysis"""
    data_path = "/Volumes/T7 Shield/Xenium data/output-XETG00234__0056769__Region_1__20250124__224947/transcripts.zarr"
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"❌ Data path not found: {data_path}")
        return
    
    # Create visualizer and run analysis
    visualizer = XeniumDualGeneVisualizer(data_path)
    results = visualizer.run_analysis()
    
    if results:
        df = results['data']
        print(f"\n🎯 Summary:")
        print(f"   - Total transcript locations: {len(df)}")
        print(f"   - Acta2 transcripts: {len(df[df['gene'] == 'Acta2'])}")
        print(f"   - Fabp4 transcripts: {len(df[df['gene'] == 'Fabp4'])}")
        print(f"   - Z levels with data: {df['z_level'].nunique()}")
        print(f"   - Spatial range X: {df['x'].min():.1f} - {df['x'].max():.1f} μm")
        print(f"   - Spatial range Y: {df['y'].min():.1f} - {df['y'].max():.1f} μm")
        print(f"   - Spatial range Z: {df['z'].min():.1f} - {df['z'].max():.1f} μm")

if __name__ == "__main__":
    main() 