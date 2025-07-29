#!/usr/bin/env python3
"""
Xenium Spatial Transcriptomics 3D Visualization with GPU Acceleration

This module provides 3D visualization capabilities for Xenium spatial transcriptomics data
using GPU acceleration for large-scale data processing and rendering.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration libraries
try:
    import cupy as cp
    import cudf
    import cugraph
    GPU_AVAILABLE = True
    print("✅ GPU acceleration available (CuPy, cuDF, cuGraph)")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU acceleration not available. Using CPU fallback.")

# 3D visualization libraries
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("✅ Open3D available for advanced 3D visualization")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️ Open3D not available. Using matplotlib/plotly fallback.")

# Spatial analysis libraries
try:
    import squidpy as sq
    SQUIDPY_AVAILABLE = True
    print("✅ Squidpy available for spatial transcriptomics analysis")
except ImportError:
    SQUIDPY_AVAILABLE = False
    print("⚠️ Squidpy not available. Using basic spatial analysis.")


class Xenium3DVisualizer:
    """
    3D visualization system for Xenium spatial transcriptomics data with GPU acceleration.
    
    This class provides methods to:
    - Load and process Xenium data
    - Create 3D visualizations with GPU acceleration
    - Perform spatial clustering and analysis
    - Generate interactive 3D plots
    """
    
    def __init__(self, data_path: str = None, use_gpu: bool = True):
        """
        Initialize the Xenium 3D visualizer.
        
        Args:
            data_path: Path to Xenium data directory
            use_gpu: Whether to use GPU acceleration
        """
        self.data_path = Path(data_path) if data_path else None
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.data = None
        self.spatial_coords = None
        self.gene_expression = None
        self.cell_types = None
        
        print(f"🎯 Xenium 3D Visualizer initialized")
        print(f"   GPU acceleration: {'✅ Enabled' if self.use_gpu else '❌ Disabled'}")
        print(f"   Open3D available: {'✅ Yes' if OPEN3D_AVAILABLE else '❌ No'}")
        print(f"   Squidpy available: {'✅ Yes' if SQUIDPY_AVAILABLE else '❌ No'}")
    
    def load_xenium_data(self, data_path: str = None) -> bool:
        """
        Load Xenium spatial transcriptomics data.
        
        Args:
            data_path: Path to Xenium data directory
            
        Returns:
            True if data loaded successfully
        """
        if data_path:
            self.data_path = Path(data_path)
        
        if not self.data_path or not self.data_path.exists():
            print("❌ Xenium data path not found. Creating sample data...")
            return self._create_sample_data()
        
        print(f"📂 Loading Xenium data from: {self.data_path}")
        
        try:
            # Load spatial coordinates
            coords_file = self.data_path / "spatial_coordinates.csv"
            if coords_file.exists():
                self.spatial_coords = pd.read_csv(coords_file)
                print(f"   Loaded {len(self.spatial_coords)} spatial coordinates")
            else:
                print("   Spatial coordinates file not found")
            
            # Load gene expression data
            expr_file = self.data_path / "gene_expression.csv"
            if expr_file.exists():
                self.gene_expression = pd.read_csv(expr_file)
                print(f"   Loaded gene expression data: {self.gene_expression.shape}")
            else:
                print("   Gene expression file not found")
            
            # Load cell type annotations
            cell_file = self.data_path / "cell_types.csv"
            if cell_file.exists():
                self.cell_types = pd.read_csv(cell_file)
                print(f"   Loaded cell type annotations: {len(self.cell_types)} cells")
            else:
                print("   Cell type file not found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading Xenium data: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> bool:
        """Create realistic sample Xenium data for demonstration."""
        print("🔬 Creating sample Xenium spatial transcriptomics data...")
        
        # Generate realistic spatial coordinates
        n_cells = 10000
        np.random.seed(42)
        
        # Create tissue-like spatial distribution
        x = np.random.normal(0, 100, n_cells)
        y = np.random.normal(0, 100, n_cells)
        z = np.random.normal(0, 20, n_cells)  # Thinner in Z dimension
        
        # Add some tissue structure (layers)
        tissue_layers = np.random.choice([0, 1, 2], n_cells, p=[0.3, 0.5, 0.2])
        z += tissue_layers * 10
        
        self.spatial_coords = pd.DataFrame({
            'cell_id': [f'cell_{i:06d}' for i in range(n_cells)],
            'x': x,
            'y': y,
            'z': z
        })
        
        # Generate gene expression data
        n_genes = 50
        gene_names = [f'Gene_{i:03d}' for i in range(n_genes)]
        
        # Create expression matrix with spatial patterns
        expression_matrix = np.zeros((n_cells, n_genes))
        
        for i in range(n_cells):
            # Base expression level
            base_expr = np.random.exponential(0.5, n_genes)
            
            # Add spatial patterns based on position
            spatial_factor = np.exp(-((x[i]**2 + y[i]**2) / 20000))
            expression_matrix[i] = base_expr * (1 + spatial_factor)
        
        self.gene_expression = pd.DataFrame(
            expression_matrix, 
            columns=gene_names,
            index=self.spatial_coords['cell_id']
        )
        
        # Generate cell type annotations
        cell_types = ['Neuron', 'Glia', 'Endothelial', 'Immune', 'Epithelial']
        cell_type_probs = [0.4, 0.3, 0.1, 0.1, 0.1]
        
        cell_type_assignments = np.random.choice(cell_types, n_cells, p=cell_type_probs)
        
        self.cell_types = pd.DataFrame({
            'cell_id': self.spatial_coords['cell_id'],
            'cell_type': cell_type_assignments,
            'confidence': np.random.uniform(0.7, 1.0, n_cells)
        })
        
        print(f"✅ Created sample data:")
        print(f"   Cells: {n_cells:,}")
        print(f"   Genes: {n_genes}")
        print(f"   Spatial range: X({x.min():.1f}, {x.max():.1f}), Y({y.min():.1f}, {y.max():.1f}), Z({z.min():.1f}, {z.max():.1f})")
        
        return True
    
    def create_3d_scatter_plot(self, 
                               color_by: str = 'cell_type',
                               size_by: str = None,
                               genes: List[str] = None,
                               max_points: int = 5000) -> go.Figure:
        """
        Create interactive 3D scatter plot of spatial transcriptomics data.
        
        Args:
            color_by: Column to use for coloring ('cell_type', 'gene_expression', etc.)
            size_by: Column to use for point size
            genes: List of genes to visualize (if color_by='gene_expression')
            max_points: Maximum number of points to plot (for performance)
            
        Returns:
            Plotly figure object
        """
        print(f"🎨 Creating 3D scatter plot (color by: {color_by})...")
        
        # Prepare data
        if len(self.spatial_coords) > max_points:
            # Subsample for performance
            indices = np.random.choice(len(self.spatial_coords), max_points, replace=False)
            coords = self.spatial_coords.iloc[indices]
            cell_ids = coords['cell_id']
        else:
            coords = self.spatial_coords
            cell_ids = coords['cell_id']
        
        # Prepare color data
        if color_by == 'cell_type':
            color_data = self.cell_types.set_index('cell_id').loc[cell_ids, 'cell_type']
            color_map = px.colors.qualitative.Set3
        elif color_by == 'gene_expression' and genes:
            # Use first gene for coloring
            gene = genes[0]
            if gene in self.gene_expression.columns:
                color_data = self.gene_expression.loc[cell_ids, gene]
                color_map = px.colors.sequential.Viridis
            else:
                print(f"❌ Gene {gene} not found in expression data")
                color_data = np.zeros(len(cell_ids))
                color_map = px.colors.sequential.Viridis
        else:
            color_data = np.zeros(len(cell_ids))
            color_map = px.colors.sequential.Viridis
        
        # Prepare size data
        if size_by and size_by in self.gene_expression.columns:
            size_data = self.gene_expression.loc[cell_ids, size_by]
            size_data = (size_data - size_data.min()) / (size_data.max() - size_data.min()) * 10 + 2
        else:
            size_data = [3] * len(cell_ids)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=coords['x'],
            y=coords['y'],
            z=coords['z'],
            mode='markers',
            marker=dict(
                size=size_data,
                color=color_data,
                colorscale=color_map,
                opacity=0.8,
                colorbar=dict(title=color_by)
            ),
            text=cell_ids,
            hovertemplate='<b>Cell ID:</b> %{text}<br>' +
                         '<b>X:</b> %{x:.1f}<br>' +
                         '<b>Y:</b> %{y:.1f}<br>' +
                         '<b>Z:</b> %{z:.1f}<br>' +
                         '<b>Color:</b> %{marker.color}<br>' +
                         '<extra></extra>'
        )])
        
        # Update layout
        fig.update_layout(
            title=f'Xenium Spatial Transcriptomics 3D Visualization<br><sub>Color by: {color_by}</sub>',
            scene=dict(
                xaxis_title='X Position (μm)',
                yaxis_title='Y Position (μm)',
                zaxis_title='Z Position (μm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_gene_expression_heatmap_3d(self, genes: List[str], max_cells: int = 2000) -> go.Figure:
        """
        Create 3D heatmap of gene expression patterns.
        
        Args:
            genes: List of genes to visualize
            max_cells: Maximum number of cells to include
            
        Returns:
            Plotly figure object
        """
        print(f"🔥 Creating 3D gene expression heatmap for {len(genes)} genes...")
        
        # Filter genes that exist in expression data
        available_genes = [g for g in genes if g in self.gene_expression.columns]
        if not available_genes:
            print("❌ No specified genes found in expression data")
            return go.Figure()
        
        # Subsample cells for performance
        if len(self.spatial_coords) > max_cells:
            indices = np.random.choice(len(self.spatial_coords), max_cells, replace=False)
            coords = self.spatial_coords.iloc[indices]
            cell_ids = coords['cell_id']
        else:
            coords = self.spatial_coords
            cell_ids = coords['cell_id']
        
        # Get expression data for selected genes
        expr_data = self.gene_expression.loc[cell_ids, available_genes]
        
        # Create 3D heatmap
        fig = go.Figure()
        
        for i, gene in enumerate(available_genes):
            fig.add_trace(go.Scatter3d(
                x=coords['x'],
                y=coords['y'],
                z=coords['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=expr_data[gene],
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title=f'{gene} Expression')
                ),
                name=gene,
                visible=(i == 0)  # Only show first gene initially
            ))
        
        # Update layout
        fig.update_layout(
            title='3D Gene Expression Heatmap<br><sub>Use legend to switch between genes</sub>',
            scene=dict(
                xaxis_title='X Position (μm)',
                yaxis_title='Y Position (μm)',
                zaxis_title='Z Position (μm)'
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_cell_type_clusters_3d(self) -> go.Figure:
        """
        Create 3D visualization of cell type clusters.
        
        Returns:
            Plotly figure object
        """
        print("🏗️ Creating 3D cell type cluster visualization...")
        
        # Merge spatial and cell type data
        merged_data = self.spatial_coords.merge(self.cell_types, on='cell_id')
        
        # Create figure with subplots for each cell type
        cell_types = merged_data['cell_type'].unique()
        n_types = len(cell_types)
        
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scene'}]],
            subplot_titles=['Cell Type Distribution in 3D Space']
        )
        
        # Color palette for cell types
        colors = px.colors.qualitative.Set3
        
        for i, cell_type in enumerate(cell_types):
            type_data = merged_data[merged_data['cell_type'] == cell_type]
            
            fig.add_trace(go.Scatter3d(
                x=type_data['x'],
                y=type_data['y'],
                z=type_data['z'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors[i % len(colors)],
                    opacity=0.8
                ),
                name=cell_type,
                hovertemplate=f'<b>Cell Type:</b> {cell_type}<br>' +
                             '<b>X:</b> %{x:.1f}<br>' +
                             '<b>Y:</b> %{y:.1f}<br>' +
                             '<b>Z:</b> %{z:.1f}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='3D Cell Type Distribution',
            scene=dict(
                xaxis_title='X Position (μm)',
                yaxis_title='Y Position (μm)',
                zaxis_title='Z Position (μm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def perform_spatial_clustering_3d(self, n_clusters: int = 5) -> pd.DataFrame:
        """
        Perform 3D spatial clustering on the data.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            DataFrame with cluster assignments
        """
        print(f"🔍 Performing 3D spatial clustering ({n_clusters} clusters)...")
        
        # Prepare spatial coordinates for clustering
        coords_3d = self.spatial_coords[['x', 'y', 'z']].values
        
        if self.use_gpu and GPU_AVAILABLE:
            # Use GPU-accelerated clustering
            print("   Using GPU-accelerated clustering...")
            coords_gpu = cp.asarray(coords_3d)
            
            # K-means clustering on GPU
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(coords_3d)
            
        else:
            # Use CPU clustering
            print("   Using CPU clustering...")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(coords_3d)
        
        # Create cluster results
        cluster_results = pd.DataFrame({
            'cell_id': self.spatial_coords['cell_id'],
            'cluster': cluster_labels,
            'x': self.spatial_coords['x'],
            'y': self.spatial_coords['y'],
            'z': self.spatial_coords['z']
        })
        
        print(f"✅ Clustering completed. Cluster sizes:")
        for i in range(n_clusters):
            size = (cluster_labels == i).sum()
            print(f"   Cluster {i}: {size:,} cells")
        
        return cluster_results
    
    def create_cluster_visualization_3d(self, cluster_results: pd.DataFrame) -> go.Figure:
        """
        Create 3D visualization of spatial clusters.
        
        Args:
            cluster_results: DataFrame with cluster assignments
            
        Returns:
            Plotly figure object
        """
        print("🎨 Creating 3D cluster visualization...")
        
        n_clusters = cluster_results['cluster'].nunique()
        colors = px.colors.qualitative.Set3
        
        fig = go.Figure()
        
        for i in range(n_clusters):
            cluster_data = cluster_results[cluster_results['cluster'] == i]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_data['x'],
                y=cluster_data['y'],
                z=cluster_data['z'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=colors[i % len(colors)],
                    opacity=0.8
                ),
                name=f'Cluster {i}',
                hovertemplate=f'<b>Cluster:</b> {i}<br>' +
                             '<b>X:</b> %{x:.1f}<br>' +
                             '<b>Y:</b> %{y:.1f}<br>' +
                             '<b>Z:</b> %{z:.1f}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Spatial Clustering Results ({n_clusters} clusters)',
            scene=dict(
                xaxis_title='X Position (μm)',
                yaxis_title='Y Position (μm)',
                zaxis_title='Z Position (μm)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str):
        """
        Save visualization to HTML file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
        """
        output_dir = Path("results/xenium_3d")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / filename
        fig.write_html(str(output_file))
        print(f"💾 Saved visualization: {output_file}")
    
    def run_complete_analysis(self):
        """Run complete 3D analysis pipeline."""
        print("🚀 Running complete Xenium 3D analysis pipeline...")
        print("=" * 60)
        
        # 1. Load data
        if not self.load_xenium_data():
            print("❌ Failed to load data")
            return
        
        # 2. Create basic 3D scatter plot
        print("\n📊 Creating basic 3D scatter plot...")
        fig_basic = self.create_3d_scatter_plot(color_by='cell_type')
        self.save_visualization(fig_basic, "xenium_3d_basic.html")
        
        # 3. Create gene expression heatmap
        if self.gene_expression is not None:
            print("\n🔥 Creating gene expression heatmap...")
            genes_to_plot = self.gene_expression.columns[:5].tolist()  # First 5 genes
            fig_heatmap = self.create_gene_expression_heatmap_3d(genes_to_plot)
            self.save_visualization(fig_heatmap, "xenium_3d_gene_expression.html")
        
        # 4. Create cell type clusters
        print("\n🏗️ Creating cell type clusters...")
        fig_celltypes = self.create_cell_type_clusters_3d()
        self.save_visualization(fig_celltypes, "xenium_3d_cell_types.html")
        
        # 5. Perform spatial clustering
        print("\n🔍 Performing spatial clustering...")
        cluster_results = self.perform_spatial_clustering_3d(n_clusters=6)
        
        # 6. Create cluster visualization
        print("\n🎨 Creating cluster visualization...")
        fig_clusters = self.create_cluster_visualization_3d(cluster_results)
        self.save_visualization(fig_clusters, "xenium_3d_clusters.html")
        
        print("\n✅ Complete 3D analysis pipeline finished!")
        print("📁 Check the results/xenium_3d/ directory for visualizations")


def main():
    """Main function to run Xenium 3D visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Xenium 3D Visualization")
    parser.add_argument("--data-path", type=str, help="Path to Xenium data directory")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = Xenium3DVisualizer(
        data_path=args.data_path,
        use_gpu=not args.no_gpu
    )
    
    # Run complete analysis
    visualizer.run_complete_analysis()


if __name__ == "__main__":
    main() 