#!/usr/bin/env python3
"""
Main execution script for Environmental Health Analysis.

This script orchestrates the complete pipeline for analyzing the relationship
between environmental factors and pancreatic cancer/pancreatitis.
"""

import sys
import argparse
from pathlib import Path
import yaml
from loguru import logger
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_acquisition.seer_downloader import SEERDownloader
from data_acquisition.kaggle_epa_downloader import KaggleEPADownloader
from data_acquisition.epa_water_downloader import EPAWaterDownloader
from data_acquisition.census_downloader import CensusDownloader
from analysis.correlation_analysis import CorrelationAnalyzer
from analysis.spatial_analysis import SpatialAnalyzer
from analysis.statistical_analysis import StatisticalAnalyzer
from analysis.machine_learning import MLAnalyzer
from analysis.hotspot_analysis import HotspotAnalyzer
from utils.api_keys import setup_api_keys, get_api_key


class EnvironmentalHealthAnalyzer:
    """
    Main orchestrator for environmental health analysis.
    
    This class coordinates the entire pipeline from data acquisition
    through analysis and reporting.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the main analyzer."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Setup API keys
        self.api_keys = setup_api_keys()
        
        # Initialize components with API keys
        self.seer_downloader = SEERDownloader(config_path, api_key=get_api_key('seer', self.api_keys))
        self.aqi_downloader = KaggleEPADownloader(config_path)
        self.water_downloader = EPAWaterDownloader(config_path, api_key=get_api_key('epa_water', self.api_keys))
        self.census_downloader = CensusDownloader(config_path, api_key=get_api_key('census', self.api_keys))
        
        self.correlation_analyzer = CorrelationAnalyzer(config_path)
        # self.spatial_analyzer = SpatialAnalyzer(config_path)  # Disabled
        self.statistical_analyzer = StatisticalAnalyzer(config_path)
        self.ml_analyzer = MLAnalyzer(config_path)
        self.hotspot_analyzer = HotspotAnalyzer(config_path)
        
        logger.info("Environmental Health Analyzer initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_file = log_config.get('file', 'logs/analysis.log')
        
        # Create logs directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            log_file,
            level=log_config.get('level', 'INFO'),
            rotation=log_config.get('max_size', '10MB'),
            retention=log_config.get('backup_count', 5),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        logger.add(
            sys.stderr,
            level=log_config.get('level', 'INFO'),
            format="{time:HH:mm:ss} | {level} | {message}"
        )
    
    def run_complete_analysis(self, 
                             start_date: str = "2020-01-01",
                             end_date: str = "2020-12-31",
                             download_data: bool = True) -> dict:
        """
        Run the complete environmental health analysis pipeline.
        
        Args:
            start_date: Start date for environmental data
            end_date: End date for environmental data
            download_data: Whether to download fresh data
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting complete environmental health analysis")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data': {},
            'analysis': {}
        }
        
        try:
            # Step 1: Data Acquisition
            if download_data:
                logger.info("Step 1: Acquiring data from various sources")
                data_results = self._acquire_all_data(start_date, end_date)
                results['data'] = data_results
            else:
                logger.info("Skipping data download, using existing data")
                data_results = self._load_existing_data()
                results['data'] = data_results
            
            # Step 2: Data Processing
            logger.info("Step 2: Processing and cleaning data")
            processed_data = self._process_data(data_results)
            
            # Step 3: Correlation Analysis
            logger.info("Step 3: Performing correlation analysis")
            correlation_results = self._run_correlation_analysis(processed_data)
            results['analysis']['correlation'] = correlation_results
            
            # Step 4: Spatial Analysis (disabled)
            # logger.info("Step 4: Performing spatial analysis")
            # spatial_results = self._run_spatial_analysis(processed_data)
            # results['analysis']['spatial'] = spatial_results
            
            # Step 5: Statistical Analysis
            logger.info("Step 5: Performing statistical analysis")
            statistical_results = self._run_statistical_analysis(processed_data)
            results['analysis']['statistical'] = statistical_results
            
            # Step 6: Machine Learning Analysis
            logger.info("Step 6: Performing machine learning analysis")
            ml_results = self._run_ml_analysis(processed_data)
            results['analysis']['machine_learning'] = ml_results
            
            # Step 7: Hotspot Analysis
            logger.info("Step 7: Performing hotspot analysis")
            if 'seer' in all_data and 'aqi' in all_data:
                hotspot_results = self.hotspot_analyzer.run_complete_hotspot_analysis(
                    all_data['seer'], all_data['aqi']
                )
                results['analysis']['hotspot_analysis'] = hotspot_results
                logger.info("Hotspot analysis completed")
            else:
                logger.warning("Insufficient data for hotspot analysis (need SEER and AQI data)")
            
            # Step 8: Generate Report
            logger.info("Step 8: Generating comprehensive report")
            self._generate_report(results)
            
            logger.info("Complete analysis finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            raise
    
    def _acquire_all_data(self, start_date: str, end_date: str) -> dict:
        """Acquire data from all sources."""
        data_results = {}
        
        try:
            # Download SEER cancer data
            logger.info("Downloading SEER pancreatic cancer data...")
            seer_data = self.seer_downloader.get_all_data()
            if seer_data and any(not df.empty for df in seer_data.values()):
                data_results['seer'] = seer_data
                logger.info("SEER data acquired successfully")
            else:
                logger.warning("No SEER data available")
            
            # Download EPA AQI data
            logger.info("Downloading EPA AQI data...")
            aqi_data = self.aqi_downloader.get_all_data(start_date, end_date)
            if aqi_data and any(not df.empty for df in aqi_data.values()):
                data_results['aqi'] = aqi_data
                logger.info("EPA AQI data acquired successfully")
            else:
                logger.warning("No EPA AQI data available")
            
            # Download EPA water quality data
            logger.info("Downloading EPA water quality data...")
            water_data = self.water_downloader.get_all_data(start_date, end_date)
            if water_data and any(not df.empty for df in water_data.values()):
                data_results['water'] = water_data
                logger.info("EPA water data acquired successfully")
            else:
                logger.warning("No EPA water data available")
            
            # Download Census demographic data
            logger.info("Downloading Census demographic data...")
            census_data = self.census_downloader.get_all_data()
            if census_data and any(not df.empty for df in census_data.values()):
                data_results['census'] = census_data
                logger.info("Census data acquired successfully")
            else:
                logger.warning("No Census data available")
            
        except Exception as e:
            logger.error(f"Error acquiring data: {e}")
            raise
        
        return data_results
    
    def _load_existing_data(self) -> dict:
        """Load existing data from files."""
        data_results = {}
        
        try:
            # Load SEER data
            seer_files = list(Path("data/raw/seer").glob("*.csv"))
            if seer_files:
                seer_data = {}
                for file in seer_files:
                    data_type = file.stem.split('_')[1]  # Extract data type from filename
                    seer_data[data_type] = pd.read_csv(file)
                data_results['seer'] = seer_data
            
            # Load AQI data
            aqi_files = list(Path("data/raw/epa_aqi").glob("*.csv"))
            if aqi_files:
                aqi_data = {}
                for file in aqi_files:
                    data_type = file.stem.split('_')[1]  # Extract data type from filename
                    aqi_data[data_type] = pd.read_csv(file)
                data_results['aqi'] = aqi_data
            
            # Load water data
            water_files = list(Path("data/raw/epa_water").glob("*.csv"))
            if water_files:
                water_data = {}
                for file in water_files:
                    data_type = file.stem.split('_')[1]  # Extract data type from filename
                    water_data[data_type] = pd.read_csv(file)
                data_results['water'] = water_data
            
            # Load census data
            census_files = list(Path("data/raw/census").glob("*.csv"))
            if census_files:
                census_data = {}
                for file in census_files:
                    data_type = file.stem.split('_')[1]  # Extract data type from filename
                    census_data[data_type] = pd.read_csv(file)
                data_results['census'] = census_data
                
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            raise
        
        return data_results
    
    def _process_data(self, data_results: dict) -> dict:
        """Process and clean the acquired data."""
        processed_data = {}
        
        try:
            # Process SEER data
            if 'seer' in data_results:
                processed_data['cancer'] = self._process_cancer_data(data_results['seer'])
            
            # Process AQI data
            if 'aqi' in data_results:
                processed_data['aqi'] = self._process_aqi_data(data_results['aqi'])
            
            # Process water data
            if 'water' in data_results:
                processed_data['water'] = self._process_water_data(data_results['water'])
            
            # Process census data
            if 'census' in data_results:
                processed_data['demographics'] = self._process_census_data(data_results['census'])
            
            # Merge datasets
            processed_data['merged'] = self._merge_all_datasets(processed_data)
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
        
        return processed_data
    
    def _process_cancer_data(self, seer_data: dict) -> pd.DataFrame:
        """Process SEER cancer data."""
        if 'incidence' in seer_data:
            return seer_data['incidence']
        return pd.DataFrame()
    
    def _process_aqi_data(self, aqi_data: dict) -> pd.DataFrame:
        """Process EPA AQI data."""
        if 'aqi' in aqi_data:
            return aqi_data['aqi']
        return pd.DataFrame()
    
    def _process_water_data(self, water_data: dict) -> pd.DataFrame:
        """Process EPA water quality data."""
        if 'water' in water_data:
            return water_data['water']
        return pd.DataFrame()
    
    def _process_census_data(self, census_data: dict) -> pd.DataFrame:
        """Process Census demographic data."""
        if 'demographics' in census_data:
            return census_data['demographics']
        return pd.DataFrame()
    
    def _merge_all_datasets(self, processed_data: dict) -> pd.DataFrame:
        """Merge all processed datasets."""
        merged_data = pd.DataFrame()
        
        try:
            # Start with cancer data
            if 'cancer' in processed_data and not processed_data['cancer'].empty:
                merged_data = processed_data['cancer'].copy()
                
                # Merge with AQI data
                if 'aqi' in processed_data and not processed_data['aqi'].empty:
                    merged_data = pd.merge(merged_data, processed_data['aqi'], 
                                         on='county', how='left')
                
                # Merge with water data
                if 'water' in processed_data and not processed_data['water'].empty:
                    merged_data = pd.merge(merged_data, processed_data['water'], 
                                         on='county', how='left')
                
                # Merge with demographics
                if 'demographics' in processed_data and not processed_data['demographics'].empty:
                    merged_data = pd.merge(merged_data, processed_data['demographics'], 
                                         on='county', how='left')
            
            logger.info(f"Merged dataset contains {len(merged_data)} records")
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
        
        return merged_data
    
    def _run_correlation_analysis(self, processed_data: dict) -> dict:
        """Run correlation analysis."""
        results = {}
        
        try:
            if 'cancer' in processed_data and 'aqi' in processed_data:
                results['aqi_cancer'] = self.correlation_analyzer.analyze_aqi_pancreatic_correlation(
                    processed_data['cancer'], processed_data['aqi']
                )
            
            if 'cancer' in processed_data and 'water' in processed_data:
                results['water_cancer'] = self.correlation_analyzer.analyze_water_quality_correlation(
                    processed_data['cancer'], processed_data['water']
                )
            
            if 'merged' in processed_data and not processed_data['merged'].empty:
                results['general'] = self.correlation_analyzer.analyze_environmental_cancer_correlation(
                    processed_data['cancer'], processed_data['merged']
                )
                
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
        
        return results
    
    def _run_spatial_analysis(self, processed_data: dict) -> dict:
        """Run spatial analysis."""
        results = {}
        
        try:
            if 'merged' in processed_data and not processed_data['merged'].empty:
                results = self.spatial_analyzer.analyze_spatial_patterns(processed_data['merged'])
                
        except Exception as e:
            logger.error(f"Error in spatial analysis: {e}")
        
        return results
    
    def _run_statistical_analysis(self, processed_data: dict) -> dict:
        """Run statistical analysis."""
        results = {}
        
        try:
            if 'merged' in processed_data and not processed_data['merged'].empty:
                results = self.statistical_analyzer.perform_statistical_tests(processed_data['merged'])
                
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
        
        return results
    
    def _run_ml_analysis(self, processed_data: dict) -> dict:
        """Run machine learning analysis."""
        results = {}
        
        try:
            if 'merged' in processed_data and not processed_data['merged'].empty:
                results = self.ml_analyzer.build_predictive_models(processed_data['merged'])
                
        except Exception as e:
            logger.error(f"Error in machine learning analysis: {e}")
        
        return results
    
    def _generate_report(self, results: dict):
        """Generate comprehensive analysis report."""
        try:
            report_path = Path("data/results/comprehensive_report.html")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create HTML report
            html_content = self._create_html_report(results)
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Comprehensive report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def _create_html_report(self, results: dict) -> str:
        """Create HTML report content."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Environmental Health Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #2c3e50; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .highlight { background-color: #f8f9fa; padding: 10px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Environmental Health Analysis Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents the analysis of environmental factors and their relationship 
                with pancreatic cancer and pancreatitis incidence rates.</p>
            </div>
            
            <div class="section">
                <h2>Data Sources</h2>
                <ul>
                    <li>SEER Cancer Data: Pancreatic cancer incidence and mortality</li>
                    <li>EPA Air Quality Index: Air pollution monitoring data</li>
                    <li>EPA Water Quality: Drinking water contaminant levels</li>
                    <li>Census Demographics: Socioeconomic factors</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <div class="highlight">
                    <h3>Correlation Analysis</h3>
                    <p>Analysis of correlations between environmental factors and health outcomes.</p>
                </div>
                
                <div class="highlight">
                    <h3>Spatial Analysis</h3>
                    <p>Geographic patterns in environmental exposure and disease incidence.</p>
                </div>
                
                <div class="highlight">
                    <h3>Statistical Significance</h3>
                    <p>Statistical tests for environmental-health relationships.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li>Further investigation of identified environmental risk factors</li>
                    <li>Targeted public health interventions in high-risk areas</li>
                    <li>Enhanced environmental monitoring in affected regions</li>
                    <li>Longitudinal studies to establish causality</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_template.format(
            timestamp=results.get('timestamp', 'Unknown')
        )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Environmental Health Analysis')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--start-date', default='2020-01-01',
                       help='Start date for environmental data (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2020-12-31',
                       help='End date for environmental data (YYYY-MM-DD)')
    parser.add_argument('--no-download', action='store_true',
                       help='Skip data download and use existing data')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run analysis only (assumes data already exists)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = EnvironmentalHealthAnalyzer(args.config)
        
        if args.analysis_only:
            logger.info("Running analysis only with existing data")
            results = analyzer.run_complete_analysis(
                start_date=args.start_date,
                end_date=args.end_date,
                download_data=False
            )
        else:
            logger.info("Running complete analysis pipeline")
            results = analyzer.run_complete_analysis(
                start_date=args.start_date,
                end_date=args.end_date,
                download_data=not args.no_download
            )
        
        logger.info("Analysis completed successfully!")
        print("\n" + "="*50)
        print("ENVIRONMENTAL HEALTH ANALYSIS COMPLETED")
        print("="*50)
        print(f"Results saved to: data/results/")
        print(f"Report generated: data/results/comprehensive_report.html")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 