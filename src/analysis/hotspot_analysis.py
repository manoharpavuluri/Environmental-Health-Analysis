"""
Hotspot Analysis for Pancreatic Cancer and Air Quality.

This module performs spatial analysis to identify hotspots of pancreatic cancer
incidence that may be associated with air quality factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from loguru import logger
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')


class HotspotAnalyzer:
    """
    Analyzes pancreatic cancer hotspots in relation to air quality factors.
    
    This class performs comprehensive spatial analysis to identify areas with
    high pancreatic cancer incidence that may be associated with poor air quality.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the hotspot analyzer."""
        self.config = self._load_config(config_path)
        self.results_dir = Path("data/results/hotspots")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Hotspot Analyzer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_data(self, seer_data: Dict[str, pd.DataFrame], aqi_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare SEER and AQI data for analysis.
        
        Args:
            seer_data: Dictionary containing SEER cancer data
            aqi_data: Dictionary containing EPA air quality data
            
        Returns:
            Tuple of (cancer_data, air_quality_data) DataFrames
        """
        logger.info("Loading data for hotspot analysis...")
        
        # Load cancer incidence data
        cancer_data = seer_data.get('incidence', pd.DataFrame())
        if cancer_data.empty:
            # Try to load from file directly
            seer_file = Path("data/raw/seer/seer_incidence_2010-2020.csv")
            if seer_file.exists():
                cancer_data = pd.read_csv(seer_file)
                logger.info(f"Loaded cancer data from file: {len(cancer_data)} records")
        
        if cancer_data.empty:
            logger.error("No SEER incidence data available")
            return pd.DataFrame(), pd.DataFrame()
        
        # Load air quality data
        air_quality_data = aqi_data.get('county_level', pd.DataFrame())
        if air_quality_data.empty:
            # Try raw data if county level not available
            air_quality_data = aqi_data.get('raw', pd.DataFrame())
        
        # If still empty, try to load from file directly
        if air_quality_data.empty:
            aqi_file = Path("data/raw/epa_aqi/epa_historical_air_quality.csv")
            if aqi_file.exists():
                air_quality_data = pd.read_csv(aqi_file)
                logger.info(f"Loaded air quality data from file: {len(air_quality_data)} records")
        
        if air_quality_data.empty:
            logger.error("No air quality data available")
            return cancer_data, pd.DataFrame()
        
        logger.info(f"Loaded {len(cancer_data)} cancer records and {len(air_quality_data)} air quality records")
        return cancer_data, air_quality_data
    
    def identify_hotspots(self, cancer_data: pd.DataFrame, air_quality_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Identify pancreatic cancer hotspots using multiple methods.
        
        Args:
            cancer_data: DataFrame with cancer incidence data
            air_quality_data: DataFrame with air quality data
            
        Returns:
            Dictionary containing hotspot analysis results
        """
        logger.info("Identifying pancreatic cancer hotspots...")
        
        results = {}
        
        # Method 1: High incidence areas
        high_incidence = self._find_high_incidence_areas(cancer_data)
        if not high_incidence.empty:
            results['high_incidence'] = high_incidence
        
        # Method 2: Areas with poor air quality
        poor_air_quality = self._find_poor_air_quality_areas(air_quality_data)
        if not poor_air_quality.empty:
            results['poor_air_quality'] = poor_air_quality
        
        # Method 3: Correlation-based hotspots
        correlation_hotspots = self._find_correlation_hotspots(cancer_data, air_quality_data)
        if not correlation_hotspots.empty:
            results['correlation_hotspots'] = correlation_hotspots
        
        # Method 4: Spatial clustering
        cluster_hotspots = self._find_cluster_hotspots(cancer_data, air_quality_data)
        if not cluster_hotspots.empty:
            results['cluster_hotspots'] = cluster_hotspots
        
        logger.info(f"Identified {len(results)} types of hotspots")
        return results
    
    def _find_high_incidence_areas(self, cancer_data: pd.DataFrame) -> pd.DataFrame:
        """Find areas with high pancreatic cancer incidence rates."""
        if cancer_data.empty:
            return pd.DataFrame()
        
        # Calculate incidence rates by county/state
        if 'county' in cancer_data.columns and 'state' in cancer_data.columns:
            # Use existing incidence rate if available
            if 'incidence_rate' in cancer_data.columns:
                incidence_by_area = cancer_data[['county', 'state', 'incidence_rate']].copy()
            else:
                # Calculate incidence rate per 100,000
                cancer_data['incidence_rate'] = (cancer_data['incidence_count'] / cancer_data['population']) * 100000
                incidence_by_area = cancer_data[['county', 'state', 'incidence_rate']].copy()
            
            # Rename columns to match expected format
            incidence_by_area = incidence_by_area.rename(columns={
                'county': 'County Name',
                'state': 'State Name'
            })
            
            # Identify high incidence areas (top 10%)
            threshold = incidence_by_area['incidence_rate'].quantile(0.9)
            high_incidence = incidence_by_area[incidence_by_area['incidence_rate'] >= threshold]
            
            return high_incidence
        
        return pd.DataFrame()
    
    def _find_poor_air_quality_areas(self, air_quality_data: pd.DataFrame) -> pd.DataFrame:
        """Find areas with poor air quality indicators."""
        if air_quality_data.empty:
            return pd.DataFrame()
        
        poor_air_quality = pd.DataFrame()
        
        # Check for different air quality parameters
        if 'Mean_AQI' in air_quality_data.columns:
            # High AQI areas
            threshold = air_quality_data['Mean_AQI'].quantile(0.8)
            poor_air_quality = air_quality_data[air_quality_data['Mean_AQI'] >= threshold]
        
        elif 'Arithmetic Mean' in air_quality_data.columns:
            # High pollutant levels
            threshold = air_quality_data['Arithmetic Mean'].quantile(0.8)
            poor_air_quality = air_quality_data[air_quality_data['Arithmetic Mean'] >= threshold]
        
        return poor_air_quality
    
    def _find_correlation_hotspots(self, cancer_data: pd.DataFrame, air_quality_data: pd.DataFrame) -> pd.DataFrame:
        """Find areas where cancer incidence correlates with poor air quality."""
        if cancer_data.empty or air_quality_data.empty:
            return pd.DataFrame()
        
        # Merge data by geographic area
        merged_data = self._merge_geographic_data(cancer_data, air_quality_data)
        
        if merged_data.empty:
            return pd.DataFrame()
        
        # Calculate correlations
        correlations = {}
        for col in merged_data.columns:
            if col in ['incidence_rate', 'Mean_AQI', 'Arithmetic Mean', 'PM2.5', 'Ozone']:
                corr = merged_data[['incidence_rate', col]].corr().iloc[0, 1]
                correlations[col] = corr
        
        # Find areas with high correlation
        correlation_threshold = 0.3
        high_correlation_areas = merged_data[
            (merged_data['incidence_rate'] > merged_data['incidence_rate'].median()) &
            (merged_data['Mean_AQI'] > merged_data['Mean_AQI'].median())
        ]
        
        return high_correlation_areas
    
    def _find_cluster_hotspots(self, cancer_data: pd.DataFrame, air_quality_data: pd.DataFrame) -> pd.DataFrame:
        """Find spatial clusters of high cancer incidence and poor air quality."""
        if cancer_data.empty or air_quality_data.empty:
            return pd.DataFrame()
        
        # Prepare data for clustering
        cluster_data = self._prepare_cluster_data(cancer_data, air_quality_data)
        
        if cluster_data.empty:
            return pd.DataFrame()
        
        # Perform DBSCAN clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data[['incidence_rate', 'aqi_score']])
        
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        cluster_data['cluster'] = dbscan.fit_predict(scaled_data)
        
        # Identify hotspot clusters (high incidence + poor air quality)
        hotspot_clusters = cluster_data[
            (cluster_data['cluster'] != -1) &
            (cluster_data['incidence_rate'] > cluster_data['incidence_rate'].median()) &
            (cluster_data['aqi_score'] > cluster_data['aqi_score'].median())
        ]
        
        return hotspot_clusters
    
    def _merge_geographic_data(self, cancer_data: pd.DataFrame, air_quality_data: pd.DataFrame) -> pd.DataFrame:
        """Merge cancer and air quality data by geographic area."""
        # Standardize geographic identifiers
        if 'County Name' in cancer_data.columns and 'County Name' in air_quality_data.columns:
            # Merge by county
            merged = pd.merge(
                cancer_data, 
                air_quality_data, 
                on=['State Name', 'County Name'], 
                how='inner'
            )
            return merged
        
        return pd.DataFrame()
    
    def _prepare_cluster_data(self, cancer_data: pd.DataFrame, air_quality_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for spatial clustering."""
        merged_data = self._merge_geographic_data(cancer_data, air_quality_data)
        
        if merged_data.empty:
            return pd.DataFrame()
        
        # Create composite air quality score
        if 'Mean_AQI' in merged_data.columns:
            merged_data['aqi_score'] = merged_data['Mean_AQI']
        elif 'Arithmetic Mean' in merged_data.columns:
            merged_data['aqi_score'] = merged_data['Arithmetic Mean']
        else:
            merged_data['aqi_score'] = 0
        
        # Select relevant columns for clustering
        cluster_cols = ['incidence_rate', 'aqi_score']
        available_cols = [col for col in cluster_cols if col in merged_data.columns]
        
        if len(available_cols) < 2:
            return pd.DataFrame()
        
        return merged_data[available_cols].dropna()
    
    def create_hotspot_maps(self, hotspot_results: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Create interactive maps showing hotspots.
        
        Args:
            hotspot_results: Dictionary containing hotspot analysis results
            
        Returns:
            Dictionary mapping hotspot types to map file paths
        """
        logger.info("Creating hotspot maps...")
        
        map_files = {}
        
        for hotspot_type, data in hotspot_results.items():
            if data.empty:
                continue
            
            map_file = self._create_hotspot_map(data, hotspot_type)
            if map_file:
                map_files[hotspot_type] = map_file
        
        return map_files
    
    def _create_hotspot_map(self, data: pd.DataFrame, hotspot_type: str) -> Optional[str]:
        """Create an interactive map for a specific hotspot type."""
        try:
            # Create base map centered on US
            m = folium.Map(
                location=[39.8283, -98.5795],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
            # Add markers for hotspots
            for idx, row in data.iterrows():
                # Use approximate coordinates (you'd need real coordinates for production)
                lat = 39.8283 + np.random.normal(0, 5)  # Approximate US center
                lon = -98.5795 + np.random.normal(0, 5)
                
                popup_text = f"""
                <b>{hotspot_type.replace('_', ' ').title()}</b><br>
                State: {row.get('State Name', 'N/A')}<br>
                County: {row.get('County Name', 'N/A')}<br>
                Incidence Rate: {row.get('incidence_rate', 'N/A'):.2f}<br>
                AQI: {row.get('Mean_AQI', 'N/A')}
                """
                
                folium.Marker(
                    [lat, lon],
                    popup=popup_text,
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
            
            # Save map
            map_filename = f"hotspot_map_{hotspot_type}.html"
            map_filepath = self.results_dir / map_filename
            m.save(str(map_filepath))
            
            logger.info(f"Created hotspot map: {map_filepath}")
            return str(map_filepath)
            
        except Exception as e:
            logger.error(f"Error creating map for {hotspot_type}: {e}")
            return None
    
    def generate_hotspot_report(self, hotspot_results: Dict[str, pd.DataFrame], map_files: Dict[str, str]) -> str:
        """
        Generate a comprehensive hotspot analysis report.
        
        Args:
            hotspot_results: Dictionary containing hotspot analysis results
            map_files: Dictionary mapping hotspot types to map file paths
            
        Returns:
            Path to the generated report file
        """
        logger.info("Generating hotspot analysis report...")
        
        report_file = self.results_dir / "hotspot_analysis_report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pancreatic Cancer Hotspot Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .hotspot {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
                .map-link {{ color: blue; text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>Pancreatic Cancer Hotspot Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report identifies areas with high pancreatic cancer incidence that may be associated with poor air quality.</p>
                <p>Total hotspots identified: {len(hotspot_results)}</p>
            </div>
            
            <div class="section">
                <h2>Hotspot Analysis Results</h2>
        """
        
        for hotspot_type, data in hotspot_results.items():
            html_content += f"""
                <div class="hotspot">
                    <h3>{hotspot_type.replace('_', ' ').title()}</h3>
                    <p>Number of hotspots: {len(data)}</p>
                    <p>Average incidence rate: {data.get('incidence_rate', pd.Series()).mean():.2f}</p>
                    <p>Average AQI: {data.get('Mean_AQI', pd.Series()).mean():.2f}</p>
                    {f'<p><a href="{map_files.get(hotspot_type, "")}" class="map-link">View Interactive Map</a></p>' if hotspot_type in map_files else ''}
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li>Further investigate identified hotspots for environmental factors</li>
                    <li>Conduct detailed air quality monitoring in hotspot areas</li>
                    <li>Implement targeted public health interventions</li>
                    <li>Consider policy changes to improve air quality in affected areas</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated hotspot report: {report_file}")
        return str(report_file)
    
    def run_complete_hotspot_analysis(self, seer_data: Dict[str, pd.DataFrame], aqi_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run complete hotspot analysis pipeline.
        
        Args:
            seer_data: Dictionary containing SEER cancer data
            aqi_data: Dictionary containing EPA air quality data
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Running complete hotspot analysis...")
        
        # Load and prepare data
        cancer_data, air_quality_data = self.load_data(seer_data, aqi_data)
        
        if cancer_data.empty or air_quality_data.empty:
            logger.error("Insufficient data for hotspot analysis")
            return {}
        
        # Identify hotspots
        hotspot_results = self.identify_hotspots(cancer_data, air_quality_data)
        
        # Create maps
        map_files = self.create_hotspot_maps(hotspot_results)
        
        # Generate report
        report_file = self.generate_hotspot_report(hotspot_results, map_files)
        
        # Compile results
        results = {
            'hotspot_results': hotspot_results,
            'map_files': map_files,
            'report_file': report_file,
            'cancer_data_summary': {
                'total_records': len(cancer_data),
                'unique_counties': cancer_data['County Name'].nunique() if 'County Name' in cancer_data.columns else 0
            },
            'air_quality_summary': {
                'total_records': len(air_quality_data),
                'unique_counties': air_quality_data['County Name'].nunique() if 'County Name' in air_quality_data.columns else 0
            }
        }
        
        logger.info("Hotspot analysis completed successfully")
        return results 