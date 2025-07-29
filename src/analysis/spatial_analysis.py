"""
Spatial analysis module for environmental health research.

This module performs geographic analysis of environmental factors and
health outcomes, including spatial clustering and hotspot detection.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import yaml
from loguru import logger
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Spatial analysis libraries
try:
    import pysal
    from pysal.explore.esda import moran, geary, join_counts
    from pysal.viz.splot.esda import moran_scatterplot, lisa_cluster
    from pysal.lib.weights import W, WSP
    PYSAL_AVAILABLE = True
except ImportError:
    logger.warning("PySAL not available, spatial statistics will be limited")
    PYSAL_AVAILABLE = False
    # Create dummy classes for compatibility
    class W:
        def __init__(self):
            pass
    class WSP:
        def __init__(self):
            pass


class SpatialAnalyzer:
    """
    Performs spatial analysis of environmental health data.
    
    This class analyzes geographic patterns in environmental exposures
    and health outcomes using spatial statistics and visualization.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the spatial analyzer."""
        self.config = self._load_config(config_path)
        self.spatial_config = self.config['analysis']['spatial']
        self.resolution = self.spatial_config['resolution']
        self.coordinate_system = self.spatial_config['coordinate_system']
        
        # Create results directory
        self.results_dir = Path("data/results/spatial")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Spatial Analyzer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def analyze_spatial_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Analyze spatial patterns in environmental health data.
        
        Args:
            data: DataFrame with geographic and health/environmental data
            
        Returns:
            Dictionary with spatial analysis results
        """
        logger.info("Analyzing spatial patterns...")
        
        results = {
            'spatial_autocorrelation': {},
            'hotspots': {},
            'clustering': {},
            'visualizations': {}
        }
        
        try:
            # Convert to GeoDataFrame if coordinates are available
            gdf = self._prepare_spatial_data(data)
            
            if gdf is not None and len(gdf) > 0:
                # Spatial autocorrelation analysis
                results['spatial_autocorrelation'] = self._analyze_spatial_autocorrelation(gdf)
                
                # Hotspot analysis
                results['hotspots'] = self._detect_hotspots(gdf)
                
                # Spatial clustering
                results['clustering'] = self._perform_spatial_clustering(gdf)
                
                # Create spatial visualizations
                results['visualizations'] = self._create_spatial_visualizations(gdf)
                
                # Save results
                self._save_spatial_results(results)
                
            else:
                logger.warning("No valid spatial data for analysis")
                
        except Exception as e:
            logger.error(f"Error in spatial analysis: {e}")
        
        return results
    
    def _prepare_spatial_data(self, data: pd.DataFrame) -> Optional[gpd.GeoDataFrame]:
        """Prepare data for spatial analysis."""
        try:
            # Check if we have geographic data
            if 'county' not in data.columns:
                logger.warning("No county information in data")
                return None
            
            # For demonstration, create sample coordinates
            # In practice, you would load actual county boundaries
            sample_coords = {
                'Los Angeles, CA': (-118.2437, 34.0522),
                'Cook, IL': (-87.6298, 41.8781),
                'Harris, TX': (-95.3698, 29.7604),
                'Maricopa, AZ': (-112.0740, 33.4484),
                'San Diego, CA': (-117.1611, 32.7157),
                'Orange, CA': (-117.8531, 33.7175),
                'Miami-Dade, FL': (-80.1918, 25.7617),
                'Dallas, TX': (-96.7970, 32.7767),
                'King, WA': (-122.3321, 47.6062),
                'Riverside, CA': (-117.3962, 33.9533)
            }
            
            # Add coordinates to data
            data_with_coords = data.copy()
            data_with_coords['longitude'] = data_with_coords['county'].map(
                lambda x: sample_coords.get(x, (0, 0))[0]
            )
            data_with_coords['latitude'] = data_with_coords['county'].map(
                lambda x: sample_coords.get(x, (0, 0))[1]
            )
            
            # Create GeoDataFrame
            from shapely.geometry import Point
            geometry = [Point(xy) for xy in zip(data_with_coords['longitude'], 
                                               data_with_coords['latitude'])]
            
            gdf = gpd.GeoDataFrame(data_with_coords, geometry=geometry, 
                                  crs=self.coordinate_system)
            
            logger.info(f"Created GeoDataFrame with {len(gdf)} records")
            return gdf
            
        except Exception as e:
            logger.error(f"Error preparing spatial data: {e}")
            return None
    
    def _analyze_spatial_autocorrelation(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Analyze spatial autocorrelation in the data."""
        results = {}
        
        try:
            # Create spatial weights matrix
            weights = self._create_spatial_weights(gdf)
            
            if weights is None:
                return results
            
            # Analyze key variables
            variables = ['incidence_rate', 'mortality_rate', 'aqi', 'arsenic']
            
            for var in variables:
                if var in gdf.columns:
                    # Moran's I
                    moran_result = self._calculate_morans_i(gdf[var], weights)
                    if moran_result:
                        results[f'{var}_moran'] = moran_result
                    
                    # Geary's C
                    geary_result = self._calculate_gearys_c(gdf[var], weights)
                    if geary_result:
                        results[f'{var}_geary'] = geary_result
            
        except Exception as e:
            logger.error(f"Error in spatial autocorrelation analysis: {e}")
        
        return results
    
    def _create_spatial_weights(self, gdf: gpd.GeoDataFrame) -> Optional[W]:
        """Create spatial weights matrix."""
        try:
            # Use k-nearest neighbors (k=3)
            from pysal.lib.weights import KNN
            weights = KNN.from_dataframe(gdf, k=3)
            return weights
        except Exception as e:
            logger.error(f"Error creating spatial weights: {e}")
            return None
    
    def _calculate_morans_i(self, data: pd.Series, weights: W) -> Optional[Dict]:
        """Calculate Moran's I statistic."""
        try:
            from pysal.explore.esda import moran
            moran_result = moran(data, weights)
            
            return {
                'moran_i': moran_result.I,
                'p_value': moran_result.p_norm,
                'z_score': moran_result.z_norm,
                'significant': moran_result.p_norm < 0.05
            }
        except Exception as e:
            logger.error(f"Error calculating Moran's I: {e}")
            return None
    
    def _calculate_gearys_c(self, data: pd.Series, weights: W) -> Optional[Dict]:
        """Calculate Geary's C statistic."""
        try:
            from pysal.explore.esda import geary
            geary_result = geary(data, weights)
            
            return {
                'geary_c': geary_result.C,
                'p_value': geary_result.p_norm,
                'z_score': geary_result.z_norm,
                'significant': geary_result.p_norm < 0.05
            }
        except Exception as e:
            logger.error(f"Error calculating Geary's C: {e}")
            return None
    
    def _detect_hotspots(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Detect spatial hotspots in the data."""
        results = {}
        
        try:
            # Local Moran's I for hotspot detection
            variables = ['incidence_rate', 'mortality_rate', 'aqi']
            
            for var in variables:
                if var in gdf.columns:
                    lisa_result = self._calculate_local_morans_i(gdf[var], gdf)
                    if lisa_result:
                        results[f'{var}_hotspots'] = lisa_result
            
        except Exception as e:
            logger.error(f"Error in hotspot detection: {e}")
        
        return results
    
    def _calculate_local_morans_i(self, data: pd.Series, gdf: gpd.GeoDataFrame) -> Optional[Dict]:
        """Calculate Local Moran's I for hotspot detection."""
        try:
            from pysal.explore.esda import moran
            from pysal.lib.weights import KNN
            
            weights = KNN.from_dataframe(gdf, k=3)
            lisa = moran.Moran_Local(data, weights)
            
            return {
                'local_moran_i': lisa.Is,
                'p_values': lisa.p_sim,
                'hotspots': np.where(lisa.Is > 0, 1, 0),
                'coldspots': np.where(lisa.Is < 0, 1, 0)
            }
        except Exception as e:
            logger.error(f"Error calculating Local Moran's I: {e}")
            return None
    
    def _perform_spatial_clustering(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Perform spatial clustering analysis."""
        results = {}
        
        try:
            # K-means clustering with spatial constraints
            from sklearn.cluster import KMeans
            
            # Prepare features for clustering
            features = ['incidence_rate', 'mortality_rate', 'aqi']
            feature_cols = [col for col in features if col in gdf.columns]
            
            if len(feature_cols) > 0:
                X = gdf[feature_cols].fillna(0)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                results['spatial_clusters'] = {
                    'cluster_labels': clusters,
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'inertia': kmeans.inertia_,
                    'n_clusters': 3
                }
                
                # Add cluster information to GeoDataFrame
                gdf['cluster'] = clusters
                
        except Exception as e:
            logger.error(f"Error in spatial clustering: {e}")
        
        return results
    
    def _create_spatial_visualizations(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Create spatial visualizations."""
        results = {}
        
        try:
            # 1. Choropleth map of cancer rates
            if 'incidence_rate' in gdf.columns:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                gdf.plot(column='incidence_rate', cmap='Reds', 
                        legend=True, ax=ax, missing_kwds={'color': 'lightgrey'})
                ax.set_title('Pancreatic Cancer Incidence Rate by County')
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'cancer_incidence_map.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                results['cancer_incidence_map'] = 'cancer_incidence_map.png'
            
            # 2. Choropleth map of AQI
            if 'aqi' in gdf.columns:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                gdf.plot(column='aqi', cmap='Blues', 
                        legend=True, ax=ax, missing_kwds={'color': 'lightgrey'})
                ax.set_title('Air Quality Index by County')
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'aqi_map.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                results['aqi_map'] = 'aqi_map.png'
            
            # 3. Scatter plot of AQI vs Cancer Rate
            if 'incidence_rate' in gdf.columns and 'aqi' in gdf.columns:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.scatter(gdf['aqi'], gdf['incidence_rate'], alpha=0.6)
                ax.set_xlabel('Air Quality Index')
                ax.set_ylabel('Cancer Incidence Rate (per 100,000)')
                ax.set_title('AQI vs Pancreatic Cancer Incidence')
                
                # Add trend line
                z = np.polyfit(gdf['aqi'].dropna(), gdf['incidence_rate'].dropna(), 1)
                p = np.poly1d(z)
                ax.plot(gdf['aqi'], p(gdf['aqi']), "r--", alpha=0.8)
                
                plt.tight_layout()
                plt.savefig(self.results_dir / 'aqi_cancer_scatter.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                results['aqi_cancer_scatter'] = 'aqi_cancer_scatter.png'
            
            # 4. Cluster map
            if 'cluster' in gdf.columns:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                gdf.plot(column='cluster', cmap='Set3', 
                        legend=True, ax=ax, categorical=True)
                ax.set_title('Spatial Clusters of Environmental-Health Patterns')
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'spatial_clusters.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                results['spatial_clusters'] = 'spatial_clusters.png'
            
        except Exception as e:
            logger.error(f"Error creating spatial visualizations: {e}")
        
        return results
    
    def _save_spatial_results(self, results: Dict):
        """Save spatial analysis results."""
        import json
        
        try:
            # Save results to JSON
            with open(self.results_dir / 'spatial_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Spatial analysis results saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving spatial results: {e}")


if __name__ == "__main__":
    # Example usage
    analyzer = SpatialAnalyzer()
    
    # Sample data
    sample_data = pd.DataFrame({
        'county': ['Los Angeles, CA', 'Cook, IL', 'Harris, TX', 'Maricopa, AZ'],
        'incidence_rate': [12.5, 11.8, 13.2, 10.9],
        'mortality_rate': [11.0, 10.5, 12.1, 9.8],
        'aqi': [75, 65, 85, 70]
    })
    
    results = analyzer.analyze_spatial_patterns(sample_data)
    print("Spatial analysis completed") 