#!/usr/bin/env python3
"""
GPU-Accelerated Spatial Analysis for Xenium Data

This module provides GPU-accelerated spatial analysis capabilities for
Xenium spatial transcriptomics data, including:
- Spatial clustering
- Gene expression pattern analysis
- Cell type distribution analysis
- Spatial correlation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# GPU acceleration
try:
    import cupy as cp
    import cudf
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import UMAP as cuUMAP
    GPU_AVAILABLE = True
    print("✅ GPU spatial analysis available (CuPy, cuML)")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU spatial analysis not available. Using CPU fallback.")


class GPUSpatialAnalyzer:
    """
    GPU-accelerated spatial analysis for Xenium spatial transcriptomics data.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize GPU spatial analyzer.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.spatial_coords = None
        self.gene_expression = None
        self.cell_types = None
        
        print(f"🔬 GPU Spatial Analyzer initialized")
        print(f"   GPU acceleration: {'✅ Enabled' if self.use_gpu else '❌ Disabled'}")
    
    def load_data(self, spatial_coords: pd.DataFrame, 
                  gene_expression: pd.DataFrame = None,
                  cell_types: pd.DataFrame = None):
        """
        Load data for spatial analysis.
        
        Args:
            spatial_coords: DataFrame with x, y, z coordinates
            gene_expression: DataFrame with gene expression data
            cell_types: DataFrame with cell type annotations
        """
        self.spatial_coords = spatial_coords
        self.gene_expression = gene_expression
        self.cell_types = cell_types
        
        print(f"📊 Loaded data:")
        print(f"   Spatial coordinates: {len(spatial_coords):,} cells")
        if gene_expression is not None:
            print(f"   Gene expression: {gene_expression.shape[1]} genes")
        if cell_types is not None:
            print(f"   Cell types: {cell_types['cell_type'].nunique()} types")
    
    def compute_spatial_distances(self, max_points: int = 5000) -> np.ndarray:
        """
        Compute pairwise spatial distances between cells.
        
        Args:
            max_points: Maximum number of points to compute distances for
            
        Returns:
            Distance matrix
        """
        print(f"📏 Computing spatial distances...")
        
        if len(self.spatial_coords) > max_points:
            # Subsample for performance
            indices = np.random.choice(len(self.spatial_coords), max_points, replace=False)
            coords = self.spatial_coords.iloc[indices][['x', 'y', 'z']].values
        else:
            coords = self.spatial_coords[['x', 'y', 'z']].values
        
        if self.use_gpu:
            # GPU-accelerated distance computation
            coords_gpu = cp.asarray(coords)
            distances_gpu = cp.zeros((len(coords), len(coords)))
            
            for i in range(len(coords)):
                diff = coords_gpu - coords_gpu[i]
                distances_gpu[i] = cp.sqrt(cp.sum(diff**2, axis=1))
            
            distances = cp.asnumpy(distances_gpu)
        else:
            # CPU distance computation
            distances = squareform(pdist(coords))
        
        print(f"✅ Computed distance matrix: {distances.shape}")
        return distances
    
    def spatial_clustering_gpu(self, method: str = 'kmeans', n_clusters: int = 5) -> np.ndarray:
        """
        Perform GPU-accelerated spatial clustering.
        
        Args:
            method: Clustering method ('kmeans', 'dbscan')
            n_clusters: Number of clusters for k-means
            
        Returns:
            Cluster labels
        """
        print(f"🔍 Performing GPU spatial clustering ({method})...")
        
        coords = self.spatial_coords[['x', 'y', 'z']].values
        
        if self.use_gpu:
            coords_gpu = cp.asarray(coords)
            
            if method == 'kmeans':
                kmeans = cuKMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(coords_gpu)
                labels = cp.asnumpy(labels)
            elif method == 'dbscan':
                # DBSCAN on GPU
                from cuml.cluster import DBSCAN as cuDBSCAN
                dbscan = cuDBSCAN(eps=50, min_samples=5)
                labels = dbscan.fit_predict(coords_gpu)
                labels = cp.asnumpy(labels)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        else:
            # CPU fallback
            if method == 'kmeans':
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(coords)
            elif method == 'dbscan':
                dbscan = DBSCAN(eps=50, min_samples=5)
                labels = dbscan.fit_predict(coords)
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        
        print(f"✅ Clustering completed. Found {len(np.unique(labels))} clusters")
        return labels
    
    def analyze_gene_expression_patterns(self, genes: List[str] = None) -> Dict:
        """
        Analyze spatial patterns in gene expression.
        
        Args:
            genes: List of genes to analyze (if None, use all genes)
            
        Returns:
            Dictionary with analysis results
        """
        print(f"🧬 Analyzing gene expression patterns...")
        
        if self.gene_expression is None:
            print("❌ No gene expression data available")
            return {}
        
        if genes is None:
            genes = self.gene_expression.columns.tolist()
        
        # Filter genes that exist in data
        available_genes = [g for g in genes if g in self.gene_expression.columns]
        
        results = {
            'genes_analyzed': available_genes,
            'spatial_correlations': {},
            'expression_stats': {},
            'spatial_hotspots': {}
        }
        
        coords = self.spatial_coords[['x', 'y', 'z']].values
        
        for gene in available_genes:
            expression = self.gene_expression[gene].values
            
            # Compute spatial correlation
            if self.use_gpu:
                # GPU-accelerated correlation
                coords_gpu = cp.asarray(coords)
                expr_gpu = cp.asarray(expression)
                
                # Normalize coordinates and expression
                coords_norm = (coords_gpu - cp.mean(coords_gpu, axis=0)) / cp.std(coords_gpu, axis=0)
                expr_norm = (expr_gpu - cp.mean(expr_gpu)) / cp.std(expr_gpu)
                
                # Compute correlation
                correlation = cp.corrcoef(coords_norm.T, expr_norm)[:3, 3]
                correlation = cp.asnumpy(correlation)
            else:
                # CPU correlation
                correlation = []
                for i in range(3):  # x, y, z
                    corr, _ = pearsonr(coords[:, i], expression)
                    correlation.append(corr)
                correlation = np.array(correlation)
            
            results['spatial_correlations'][gene] = correlation
            
            # Expression statistics
            results['expression_stats'][gene] = {
                'mean': np.mean(expression),
                'std': np.std(expression),
                'max': np.max(expression),
                'min': np.min(expression)
            }
            
            # Identify spatial hotspots (high expression areas)
            threshold = np.percentile(expression, 90)
            hotspots = expression > threshold
            results['spatial_hotspots'][gene] = {
                'n_hotspots': np.sum(hotspots),
                'hotspot_coords': coords[hotspots]
            }
        
        print(f"✅ Analyzed {len(available_genes)} genes")
        return results
    
    def create_spatial_heatmap(self, gene: str, resolution: int = 50) -> np.ndarray:
        """
        Create spatial heatmap for a specific gene.
        
        Args:
            gene: Gene name
            resolution: Grid resolution for heatmap
            
        Returns:
            2D heatmap array
        """
        print(f"🔥 Creating spatial heatmap for {gene}...")
        
        if gene not in self.gene_expression.columns:
            print(f"❌ Gene {gene} not found in expression data")
            return np.zeros((resolution, resolution))
        
        coords = self.spatial_coords[['x', 'y']].values
        expression = self.gene_expression[gene].values
        
        # Create grid
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        
        # Create heatmap
        heatmap = np.zeros((resolution, resolution))
        
        for i in range(len(coords)):
            x_idx = int((coords[i, 0] - x_min) / (x_max - x_min) * (resolution - 1))
            y_idx = int((coords[i, 1] - y_min) / (y_max - y_min) * (resolution - 1))
            
            if 0 <= x_idx < resolution and 0 <= y_idx < resolution:
                heatmap[y_idx, x_idx] += expression[i]
        
        print(f"✅ Created heatmap: {heatmap.shape}")
        return heatmap
    
    def analyze_cell_type_distribution(self) -> Dict:
        """
        Analyze spatial distribution of cell types.
        
        Returns:
            Dictionary with analysis results
        """
        print(f"🏗️ Analyzing cell type distribution...")
        
        if self.cell_types is None:
            print("❌ No cell type data available")
            return {}
        
        # Merge spatial and cell type data
        merged_data = self.spatial_coords.merge(self.cell_types, on='cell_id')
        
        results = {
            'cell_type_counts': merged_data['cell_type'].value_counts().to_dict(),
            'spatial_distributions': {},
            'nearest_neighbor_analysis': {}
        }
        
        # Analyze spatial distribution for each cell type
        for cell_type in merged_data['cell_type'].unique():
            type_data = merged_data[merged_data['cell_type'] == cell_type]
            coords = type_data[['x', 'y', 'z']].values
            
            # Spatial statistics
            results['spatial_distributions'][cell_type] = {
                'count': len(type_data),
                'center': np.mean(coords, axis=0),
                'spread': np.std(coords, axis=0),
                'volume': np.prod(np.std(coords, axis=0))  # Approximate volume
            }
            
            # Nearest neighbor analysis
            if len(coords) > 1:
                distances = pdist(coords)
                results['nearest_neighbor_analysis'][cell_type] = {
                    'mean_distance': np.mean(distances),
                    'std_distance': np.std(distances),
                    'min_distance': np.min(distances),
                    'max_distance': np.max(distances)
                }
        
        print(f"✅ Analyzed {len(results['cell_type_counts'])} cell types")
        return results
    
    def create_spatial_network(self, max_distance: float = 100) -> Dict:
        """
        Create spatial network of cell interactions.
        
        Args:
            max_distance: Maximum distance for connections
            
        Returns:
            Dictionary with network information
        """
        print(f"🌐 Creating spatial network (max distance: {max_distance}μm)...")
        
        coords = self.spatial_coords[['x', 'y', 'z']].values
        
        if self.use_gpu:
            # GPU-accelerated network creation
            coords_gpu = cp.asarray(coords)
            distances_gpu = cp.zeros((len(coords), len(coords)))
            
            for i in range(len(coords)):
                diff = coords_gpu - coords_gpu[i]
                distances_gpu[i] = cp.sqrt(cp.sum(diff**2, axis=1))
            
            # Find connections within max_distance
            connections = cp.asnumpy(distances_gpu < max_distance)
            np.fill_diagonal(connections, False)  # Remove self-connections
            
        else:
            # CPU network creation
            distances = squareform(pdist(coords))
            connections = distances < max_distance
            np.fill_diagonal(connections, False)
        
        # Network statistics
        n_connections = np.sum(connections)
        n_cells = len(coords)
        
        network_stats = {
            'n_cells': n_cells,
            'n_connections': n_connections,
            'connection_density': n_connections / (n_cells * (n_cells - 1)),
            'avg_connections_per_cell': n_connections / n_cells,
            'max_distance': max_distance
        }
        
        print(f"✅ Created network: {n_connections:,} connections between {n_cells:,} cells")
        return network_stats
    
    def visualize_spatial_analysis(self, analysis_results: Dict) -> None:
        """
        Create visualizations for spatial analysis results.
        
        Args:
            analysis_results: Results from spatial analysis
        """
        print(f"🎨 Creating spatial analysis visualizations...")
        
        # Create output directory
        output_dir = Path("results/spatial_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Gene expression pattern visualization
        if 'spatial_correlations' in analysis_results:
            genes = list(analysis_results['spatial_correlations'].keys())
            correlations = np.array([analysis_results['spatial_correlations'][g] for g in genes])
            
            plt.figure(figsize=(12, 8))
            plt.imshow(correlations.T, cmap='RdBu_r', aspect='auto')
            plt.colorbar(label='Spatial Correlation')
            plt.xlabel('Genes')
            plt.ylabel('Spatial Dimension (X, Y, Z)')
            plt.title('Spatial Correlation Patterns')
            plt.xticks(range(len(genes)), genes, rotation=45)
            plt.yticks(range(3), ['X', 'Y', 'Z'])
            plt.tight_layout()
            plt.savefig(output_dir / "spatial_correlations.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Cell type distribution visualization
        if 'cell_type_counts' in analysis_results:
            cell_types = list(analysis_results['cell_type_counts'].keys())
            counts = list(analysis_results['cell_type_counts'].values())
            
            plt.figure(figsize=(10, 6))
            plt.bar(cell_types, counts)
            plt.xlabel('Cell Type')
            plt.ylabel('Count')
            plt.title('Cell Type Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / "cell_type_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"💾 Saved visualizations to {output_dir}")


def main():
    """Main function to run GPU spatial analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Spatial Analysis")
    parser.add_argument("--data-path", type=str, required=True, help="Path to Xenium data")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = GPUSpatialAnalyzer(use_gpu=not args.no_gpu)
    
    # Load data (you would implement this based on your data format)
    print("📂 Loading Xenium data...")
    # This is a placeholder - implement based on your data format
    # spatial_coords = pd.read_csv(f"{args.data_path}/spatial_coords.csv")
    # gene_expression = pd.read_csv(f"{args.data_path}/gene_expression.csv")
    # cell_types = pd.read_csv(f"{args.data_path}/cell_types.csv")
    
    # analyzer.load_data(spatial_coords, gene_expression, cell_types)
    
    # Run analysis
    print("🔍 Running spatial analysis...")
    # analysis_results = analyzer.analyze_gene_expression_patterns()
    # analyzer.visualize_spatial_analysis(analysis_results)
    
    print("✅ Spatial analysis completed!")


if __name__ == "__main__":
    main() 