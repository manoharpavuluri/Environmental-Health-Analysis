#!/usr/bin/env python3
"""
Xenium Expression-Filtered 3D Visualization
Visualize Epcam (blue), Chl1 (yellow), and Retnla (green) with expression-based color intensity
Filter for areas with at least 10 adjacent pixels
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
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class XeniumAllTranscriptsVisualizer:
    """Visualize multiple genes with expression-based color intensity for all transcripts"""
    
    def __init__(self, data_path: str):
        """Initialize with path to Xenium data"""
        self.data_path = data_path
        self.zarr_data = zarr.open(data_path)
        self.gene_panel = self._load_gene_panel()
        self.epcam_codeword = self._find_gene_codeword('Epcam')
        self.chl1_codeword = self._find_gene_codeword('Chl1')
        self.retnla_codeword = self._find_gene_codeword('Retnla')
        
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
    
    def filter_by_adjacency(self, df: pd.DataFrame, min_adjacent: int = 10, radius_um: float = 50.0) -> pd.DataFrame:
        """Filter transcripts to only include areas with at least min_adjacent nearby transcripts"""
        print(f"🔍 Filtering for areas with at least {min_adjacent} adjacent transcripts (radius: {radius_um} μm)...")
        
        if df.empty:
            return df
        
        # Group by gene for separate filtering
        filtered_dfs = []
        
        for gene in df['gene'].unique():
            gene_data = df[df['gene'] == gene].copy()
            print(f"  Processing {gene}: {len(gene_data)} transcripts")
            
            if len(gene_data) < min_adjacent:
                print(f"    Skipping {gene} - insufficient transcripts")
                continue
            
            # Calculate pairwise distances
            coords = gene_data[['x', 'y', 'z']].values
            
            # Use a more efficient approach for large datasets
            if len(coords) > 10000:
                # For large datasets, use a grid-based approach
                filtered_coords = self._filter_large_dataset(coords, min_adjacent, radius_um)
            else:
                # For smaller datasets, calculate all pairwise distances
                distances = cdist(coords, coords)
                # Count neighbors within radius for each point
                neighbor_counts = np.sum(distances < radius_um, axis=1) - 1  # Subtract self
                # Keep points with sufficient neighbors
                high_density_mask = neighbor_counts >= min_adjacent
                filtered_coords = coords[high_density_mask]
            
            if len(filtered_coords) > 0:
                # Create filtered DataFrame
                filtered_gene_data = gene_data.iloc[:len(filtered_coords)].copy()
                filtered_gene_data[['x', 'y', 'z']] = filtered_coords
                filtered_dfs.append(filtered_gene_data)
                print(f"    Kept {len(filtered_coords)} transcripts after filtering")
            else:
                print(f"    No transcripts met adjacency criteria for {gene}")
        
        if filtered_dfs:
            result = pd.concat(filtered_dfs, ignore_index=True)
            print(f"✅ Filtered dataset: {len(result)} total transcript locations")
            return result
        else:
            print("⚠️  No transcripts met adjacency criteria")
            return pd.DataFrame()
    
    def _filter_large_dataset(self, coords: np.ndarray, min_adjacent: int, radius_um: float) -> np.ndarray:
        """Efficient filtering for large datasets using grid-based approach"""
        # Create a grid for spatial indexing
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()
        
        # Grid cell size based on radius
        cell_size = radius_um / 2
        
        # Create grid indices
        grid_x = ((coords[:, 0] - x_min) / cell_size).astype(int)
        grid_y = ((coords[:, 1] - y_min) / cell_size).astype(int)
        grid_z = ((coords[:, 2] - z_min) / cell_size).astype(int)
        
        # Create grid dictionary
        grid = {}
        for i, (gx, gy, gz) in enumerate(zip(grid_x, grid_y, grid_z)):
            key = (gx, gy, gz)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        
        # Find high-density areas
        high_density_indices = []
        for cell_indices in grid.values():
            if len(cell_indices) >= min_adjacent:
                # Check distances within this cell and neighboring cells
                cell_coords = coords[cell_indices]
                
                # Check distances to all other points in nearby cells
                for idx in cell_indices:
                    point = coords[idx]
                    distances = np.linalg.norm(coords - point, axis=1)
                    neighbor_count = np.sum(distances < radius_um) - 1  # Subtract self
                    
                    if neighbor_count >= min_adjacent:
                        high_density_indices.append(idx)
        
        return coords[high_density_indices]
    
    def calculate_expression_intensity(self, df: pd.DataFrame, radius_um: float = 100.0) -> pd.DataFrame:
        """Calculate expression intensity based on local transcript density"""
        print(f"📊 Calculating expression intensity (radius: {radius_um} μm)...")
        
        if df.empty:
            return df
        
        df = df.copy()
        df['expression_intensity'] = 0.0
        
        # Group by gene for separate calculation
        for gene in df['gene'].unique():
            gene_data = df[df['gene'] == gene].copy()
            coords = gene_data[['x', 'y', 'z']].values
            
            # Calculate local density for each point
            intensities = []
            for i, point in enumerate(coords):
                distances = np.linalg.norm(coords - point, axis=1)
                local_density = np.sum(distances < radius_um)
                intensities.append(local_density)
            
            # Normalize intensities to 0-1 range
            intensities = np.array(intensities)
            if intensities.max() > 0:
                intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
            
            # Update the dataframe
            gene_indices = df[df['gene'] == gene].index
            df.loc[gene_indices, 'expression_intensity'] = intensities
        
        return df
    
    def create_3d_expression_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create 3D scatter plot with expression-based color intensity"""
        if df.empty:
            print("⚠️  No data to plot")
            return go.Figure()
        
        # Create figure
        fig = go.Figure()
        
        # Color schemes for each gene
        gene_colors = {
            'Epcam': 'blue',
            'Chl1': 'yellow', 
            'Retnla': 'green'
        }
        
        # Plot each gene with expression-based intensity
        for gene in df['gene'].unique():
            gene_data = df[df['gene'] == gene].copy()
            
            if not gene_data.empty:
                # Create color array based on expression intensity
                intensities = gene_data['expression_intensity'].values
                
                # Create color scale based on gene
                base_color = gene_colors[gene]
                
                if base_color == 'blue':
                    colors = [f'rgba(0, 0, {int(255 * intensity)}, 0.8)' for intensity in intensities]
                elif base_color == 'yellow':
                    colors = [f'rgba({int(255 * intensity)}, {int(255 * intensity)}, 0, 0.8)' for intensity in intensities]
                elif base_color == 'green':
                    colors = [f'rgba(0, {int(255 * intensity)}, 0, 0.8)' for intensity in intensities]
                else:
                    colors = [f'rgba(128, 128, 128, 0.8)' for _ in intensities]
                
                fig.add_trace(go.Scatter3d(
                    x=gene_data['x'],
                    y=gene_data['y'],
                    z=gene_data['z'],
                    mode='markers',
                    marker=dict(
                        size=2,  # Slightly larger for visibility
                        color=colors,
                        opacity=0.8
                    ),
                    name=f'{gene} (Expression)',
                    hovertemplate=f'<b>{gene}</b><br>' +
                                'X: %{x:.1f} μm<br>' +
                                'Y: %{y:.1f} μm<br>' +
                                'Z: %{z:.1f} μm<br>' +
                                'Expression: %{customdata:.2f}<br>' +
                                'Z Level: %{text}<extra></extra>',
                    customdata=gene_data['expression_intensity'],
                    text=gene_data['z_level']
                ))
        
        # Update layout with black background and white axes
        fig.update_layout(
            title='Epcam (Blue), Chl1 (Yellow), Retnla (Green) - All Transcripts 3D Visualization',
            scene=dict(
                xaxis_title="X (μm)",
                yaxis_title="Y (μm)", 
                zaxis_title="Z (μm)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                # Black background
                bgcolor='black',
                # White axes
                xaxis=dict(
                    gridcolor='white',
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    showline=True,
                    linecolor='white',
                    title=dict(text="X (μm)", font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                yaxis=dict(
                    gridcolor='white',
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    showline=True,
                    linecolor='white',
                    title=dict(text="Y (μm)", font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                zaxis=dict(
                    gridcolor='white',
                    zerolinecolor='white',
                    showgrid=True,
                    gridwidth=1,
                    showline=True,
                    linecolor='white',
                    title=dict(text="Z (μm)", font=dict(color='white')),
                    tickfont=dict(color='white')
                )
            ),
            title_x=0.5,
            height=800,
            showlegend=True,
            # Black background for the entire plot
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white')
        )
        
        return fig
    
    def run_analysis(self) -> Dict:
        """Run complete all-transcripts 3D visualization analysis"""
        print("🚀 Starting All Transcripts 3D Visualization Analysis")
        print("=" * 70)
        
        # Extract all genes
        print("📊 Extracting gene transcripts...")
        epcam_df = self.extract_gene_transcripts('Epcam', self.epcam_codeword)
        chl1_df = self.extract_gene_transcripts('Chl1', self.chl1_codeword)
        retnla_df = self.extract_gene_transcripts('Retnla', self.retnla_codeword)
        
        # Combine all datasets
        all_dfs = [df for df in [epcam_df, chl1_df, retnla_df] if not df.empty]
        
        if not all_dfs:
            print("❌ No transcript data found. Analysis complete.")
            return {}
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"✅ Combined dataset: {len(combined_df)} total transcript locations")
        
        # Use all transcripts (no filtering)
        all_transcripts_df = combined_df.copy()
        print(f"✅ Using all {len(all_transcripts_df)} transcript locations (no filtering)")
        
        # Calculate expression intensity
        expression_df = self.calculate_expression_intensity(all_transcripts_df, radius_um=100.0)
        
        # Create visualization
        print("\n📊 Creating 3D expression visualization...")
        fig_3d = self.create_3d_expression_plot(expression_df)
        
        # Save results
        output_dir = Path("xenium_all_transcripts_analysis")
        output_dir.mkdir(exist_ok=True)
        
        fig_3d.write_html(output_dir / "all_transcripts_3d_visualization.html")
        expression_df.to_csv(output_dir / "all_transcripts_data.csv", index=False)
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
        print(f"📁 Files created:")
        print(f"   - all_transcripts_3d_visualization.html (Interactive 3D plot)")
        print(f"   - all_transcripts_data.csv (Raw data)")
        
        return {
            'data': expression_df,
            'figures': {
                '3d_expression': fig_3d
            },
            'output_dir': output_dir
        }

def main():
    """Main function to run all-transcripts 3D visualization analysis"""
    data_path = "/Volumes/T7 Shield/Xenium data/output-XETG00234__0056769__Region_1__20250124__224947/transcripts.zarr"
    
    # Check if data exists
    if not Path(data_path).exists():
        print(f"❌ Data path not found: {data_path}")
        return
    
    # Create visualizer and run analysis
    visualizer = XeniumAllTranscriptsVisualizer(data_path)
    results = visualizer.run_analysis()
    
    if results:
        df = results['data']
        print(f"\n🎯 Summary:")
        print(f"   - Total transcript locations: {len(df)}")
        for gene in df['gene'].unique():
            gene_count = len(df[df['gene'] == gene])
            print(f"   - {gene} transcripts: {gene_count}")
        print(f"   - Z levels with data: {df['z_level'].nunique()}")
        print(f"   - Spatial range X: {df['x'].min():.1f} - {df['x'].max():.1f} μm")
        print(f"   - Spatial range Y: {df['y'].min():.1f} - {df['y'].max():.1f} μm")
        print(f"   - Spatial range Z: {df['z'].min():.1f} - {df['z'].max():.1f} μm")

if __name__ == "__main__":
    main() 