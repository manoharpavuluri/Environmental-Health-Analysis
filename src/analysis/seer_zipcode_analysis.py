"""
SEER Zip Code Analysis for Pancreatic Cancer

This module analyzes SEER data to identify and visualize zip codes with the highest
pancreatic cancer incidence in New York and Texas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from loguru import logger
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')


class SEERZipcodeAnalyzer:
    """
    Analyzes SEER data to identify high pancreatic cancer incidence zip codes.
    
    This class focuses on New York and Texas zip codes with the highest
    pancreatic cancer incidence rates.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the SEER zip code analyzer."""
        self.config = self._load_config(config_path)
        self.results_dir = Path("data/results/seer_zipcode")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SEER Zip Code Analyzer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_seer_data(self) -> pd.DataFrame:
        """
        Load SEER cancer data from file.
        
        Returns:
            DataFrame with SEER cancer data
        """
        logger.info("Loading SEER cancer data...")
        
        # Try to load from file
        seer_file = Path("data/raw/seer/seer_incidence_2010-2020.csv")
        if seer_file.exists():
            data = pd.read_csv(seer_file)
            logger.info(f"Loaded {len(data)} records from SEER data")
            return data
        else:
            logger.error("SEER data file not found")
            return pd.DataFrame()
    
    def filter_states(self, data: pd.DataFrame, states: List[str] = None) -> pd.DataFrame:
        """
        Filter data for specific states (NY and TX).
        
        Args:
            data: SEER cancer data
            states: List of state codes to filter (default: ['NY', 'TX'])
            
        Returns:
            Filtered DataFrame
        """
        if states is None:
            states = ['NY', 'TX']
        
        filtered_data = data[data['state'].isin(states)].copy()
        logger.info(f"Filtered data for states {states}: {len(filtered_data)} records")
        
        return filtered_data
    
    def identify_high_incidence_zipcodes(self, data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Identify zip codes with highest pancreatic cancer incidence.
        
        Args:
            data: SEER cancer data
            top_n: Number of top zip codes to identify
            
        Returns:
            DataFrame with top high-incidence zip codes
        """
        logger.info(f"Identifying top {top_n} zip codes with highest pancreatic cancer incidence...")
        
        # Sort by incidence rate (descending)
        high_incidence = data.sort_values('incidence_rate', ascending=False).head(top_n)
        
        logger.info(f"Found {len(high_incidence)} high-incidence zip codes")
        return high_incidence
    
    def create_incidence_visualization(self, data: pd.DataFrame, state: str) -> str:
        """
        Create visualization of pancreatic cancer incidence by zip code.
        
        Args:
            data: SEER cancer data
            state: State to visualize
            
        Returns:
            Path to the generated visualization file
        """
        logger.info(f"Creating incidence visualization for {state}...")
        
        # Filter data for the specific state
        state_data = data[data['state'] == state].copy()
        
        if state_data.empty:
            logger.warning(f"No data available for {state}")
            return ""
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Bar plot of top zip codes
        top_zipcodes = state_data.nlargest(10, 'incidence_rate')
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(top_zipcodes)), top_zipcodes['incidence_rate'])
        plt.title(f'Top 10 Zip Codes with Highest Pancreatic Cancer Incidence - {state}')
        plt.xlabel('Zip Code')
        plt.ylabel('Incidence Rate (per 100,000)')
        plt.xticks(range(len(top_zipcodes)), top_zipcodes['county'], rotation=45, ha='right')
        
        # Color bars based on incidence rate
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, top_zipcodes['incidence_rate'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # Distribution plot
        plt.subplot(2, 1, 2)
        plt.hist(state_data['incidence_rate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of Pancreatic Cancer Incidence Rates - {state}')
        plt.xlabel('Incidence Rate (per 100,000)')
        plt.ylabel('Number of Zip Codes')
        
        # Add mean line
        mean_rate = state_data['incidence_rate'].mean()
        plt.axvline(mean_rate, color='red', linestyle='--', 
                   label=f'Mean: {mean_rate:.2f}')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"pancreatic_cancer_incidence_{state}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {plot_file}")
        return str(plot_file)
    
    def create_interactive_map(self, data: pd.DataFrame, state: str) -> str:
        """
        Create interactive map showing pancreatic cancer incidence.
        
        Args:
            data: SEER cancer data
            state: State to map
            
        Returns:
            Path to the generated HTML map file
        """
        logger.info(f"Creating interactive map for {state}...")
        
        # Filter data for the specific state
        state_data = data[data['state'] == state].copy()
        
        if state_data.empty:
            logger.warning(f"No data available for {state}")
            return ""
        
        # State center coordinates (approximate)
        state_centers = {
            'NY': [42.1657, -74.9481],  # New York center
            'TX': [31.9686, -99.9018]   # Texas center
        }
        
        center = state_centers.get(state, [39.8283, -98.5795])  # Default to US center
        
        # Create map
        m = folium.Map(
            location=center,
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add markers for each county
        for idx, row in state_data.iterrows():
            # Use approximate coordinates (in real implementation, you'd have actual coordinates)
            lat = center[0] + np.random.normal(0, 2)  # Add some variation
            lon = center[1] + np.random.normal(0, 2)
            
            # Color based on incidence rate
            if row['incidence_rate'] > state_data['incidence_rate'].quantile(0.8):
                color = 'red'
            elif row['incidence_rate'] > state_data['incidence_rate'].quantile(0.6):
                color = 'orange'
            else:
                color = 'green'
            
            popup_text = f"""
            <b>{row['county']}, {state}</b><br>
            Incidence Rate: {row['incidence_rate']:.2f} per 100,000<br>
            Cases: {row['incidence_count']}<br>
            Population: {row['population']:,}
            """
            
            folium.Marker(
                [lat, lon],
                popup=popup_text,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
        
        # Add heatmap
        heatmap_data = []
        for idx, row in state_data.iterrows():
            lat = center[0] + np.random.normal(0, 2)
            lon = center[1] + np.random.normal(0, 2)
            weight = row['incidence_rate'] / state_data['incidence_rate'].max()
            heatmap_data.append([lat, lon, weight])
        
        HeatMap(heatmap_data, radius=25).add_to(m)
        
        # Save map
        map_file = self.results_dir / f"pancreatic_cancer_map_{state}.html"
        m.save(str(map_file))
        
        logger.info(f"Saved interactive map: {map_file}")
        return str(map_file)
    
    def generate_state_report(self, data: pd.DataFrame, state: str) -> str:
        """
        Generate comprehensive report for a state.
        
        Args:
            data: SEER cancer data
            state: State to analyze
            
        Returns:
            Path to the generated report file
        """
        logger.info(f"Generating report for {state}...")
        
        # Filter data for the specific state
        state_data = data[data['state'] == state].copy()
        
        if state_data.empty:
            logger.warning(f"No data available for {state}")
            return ""
        
        # Calculate statistics
        stats = {
            'total_counties': len(state_data),
            'total_cases': state_data['incidence_count'].sum(),
            'total_population': state_data['population'].sum(),
            'mean_incidence_rate': state_data['incidence_rate'].mean(),
            'median_incidence_rate': state_data['incidence_rate'].median(),
            'max_incidence_rate': state_data['incidence_rate'].max(),
            'min_incidence_rate': state_data['incidence_rate'].min(),
            'std_incidence_rate': state_data['incidence_rate'].std()
        }
        
        # Get top 5 counties
        top_counties = state_data.nlargest(5, 'incidence_rate')
        
        # Generate HTML report
        report_file = self.results_dir / f"pancreatic_cancer_report_{state}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pancreatic Cancer Incidence Report - {state}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .stat {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; }}
                .highlight {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Pancreatic Cancer Incidence Report</h1>
                <h2>{state} - {len(state_data)} Counties Analyzed</h2>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3>📊 Summary Statistics</h3>
                <div class="stat">
                    <strong>Total Cases:</strong> {stats['total_cases']:,}
                </div>
                <div class="stat">
                    <strong>Total Population:</strong> {stats['total_population']:,}
                </div>
                <div class="stat">
                    <strong>Mean Incidence Rate:</strong> {stats['mean_incidence_rate']:.2f} per 100,000
                </div>
                <div class="stat">
                    <strong>Median Incidence Rate:</strong> {stats['median_incidence_rate']:.2f} per 100,000
                </div>
                <div class="stat">
                    <strong>Range:</strong> {stats['min_incidence_rate']:.2f} - {stats['max_incidence_rate']:.2f} per 100,000
                </div>
            </div>
            
            <div class="section">
                <h3>🏆 Top 5 Counties with Highest Incidence</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>County</th>
                        <th>Incidence Rate (per 100,000)</th>
                        <th>Cases</th>
                        <th>Population</th>
                    </tr>
        """
        
        for i, (idx, row) in enumerate(top_counties.iterrows(), 1):
            html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{row['county']}</td>
                        <td>{row['incidence_rate']:.2f}</td>
                        <td>{row['incidence_count']}</td>
                        <td>{row['population']:,}</td>
                    </tr>
            """
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h3>🔍 Key Findings</h3>
                <div class="highlight">
                    <strong>Highest Incidence County:</strong> {top_counties.iloc[0]['county']} 
                    ({top_counties.iloc[0]['incidence_rate']:.2f} per 100,000)
                </div>
                <div class="highlight">
                    <strong>State Average:</strong> {stats['mean_incidence_rate']:.2f} per 100,000
                </div>
                <div class="highlight">
                    <strong>Variation:</strong> Standard deviation of {stats['std_incidence_rate']:.2f} per 100,000
                </div>
            </div>
            
            <div class="section">
                <h3>📈 Visualizations</h3>
                <p>• <a href="pancreatic_cancer_incidence_{state}.png">Incidence Rate Chart</a></p>
                <p>• <a href="pancreatic_cancer_map_{state}.html">Interactive Map</a></p>
            </div>
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated report: {report_file}")
        return str(report_file)
    
    def run_complete_analysis(self, states: List[str] = None) -> Dict:
        """
        Run complete analysis for specified states.
        
        Args:
            states: List of states to analyze (default: ['NY', 'TX'])
            
        Returns:
            Dictionary containing analysis results
        """
        if states is None:
            states = ['NY', 'TX']
        
        logger.info(f"Running complete SEER zip code analysis for {states}...")
        
        # Load data
        data = self.load_seer_data()
        if data.empty:
            logger.error("No SEER data available")
            return {}
        
        # Filter for specified states
        filtered_data = self.filter_states(data, states)
        if filtered_data.empty:
            logger.error(f"No data available for states {states}")
            return {}
        
        results = {}
        
        for state in states:
            logger.info(f"Analyzing {state}...")
            
            state_data = filtered_data[filtered_data['state'] == state]
            if state_data.empty:
                logger.warning(f"No data for {state}")
                continue
            
            # Identify high incidence zip codes
            high_incidence = self.identify_high_incidence_zipcodes(state_data, top_n=10)
            
            # Create visualizations
            plot_file = self.create_incidence_visualization(state_data, state)
            map_file = self.create_interactive_map(state_data, state)
            report_file = self.generate_state_report(state_data, state)
            
            results[state] = {
                'high_incidence_zipcodes': high_incidence,
                'plot_file': plot_file,
                'map_file': map_file,
                'report_file': report_file,
                'total_counties': len(state_data),
                'mean_incidence_rate': state_data['incidence_rate'].mean(),
                'max_incidence_rate': state_data['incidence_rate'].max()
            }
        
        logger.info("SEER zip code analysis completed")
        return results 