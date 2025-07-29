"""
SEER Data Downloader

This module downloads real SEER pancreatic cancer data using the SEER API.
"""

import pandas as pd
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from loguru import logger
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SEERDownloader:
    """
    Downloads pancreatic cancer data from SEER using the API.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", api_key: Optional[str] = None):
        """Initialize the SEER downloader."""
        self.config = self._load_config(config_path)
        self.api_key = api_key
        self.base_url = "https://api.seer.cancer.gov"
        self.data_dir = Path("data/raw/seer")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("SEER Downloader initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}
    
    def download_pancreatic_cancer_data(self, 
                                      start_year: int = 2010,
                                      end_year: int = 2020,
                                      geography: str = "zip") -> pd.DataFrame:
        """
        Download pancreatic cancer incidence data from SEER.
        
        Args:
            start_year: Start year for data (default: 2010)
            end_year: End year for data (default: 2020)
            geography: Geographic level (zip, county, state)
            
        Returns:
            DataFrame with pancreatic cancer incidence data
        """
        logger.info(f"Downloading SEER pancreatic cancer data for {start_year}-{end_year}...")
        
        if not self.api_key:
            logger.error("SEER API key required for data access")
            logger.info("Please add your SEER API key to config/api_keys.yaml")
            return pd.DataFrame()
        
        try:
            # SEER API endpoint for pancreatic cancer data
            endpoint = f"{self.base_url}/data/incidence"
            
            # API parameters
            params = {
                'api_key': self.api_key,
                'cancer_type': 'pancreas',
                'start_year': start_year,
                'end_year': end_year,
                'geography': geography,
                'format': 'json'
            }
            
            logger.info("Making SEER API request...")
            response = requests.get(endpoint, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully downloaded SEER data")
                return self._process_seer_data(data)
            else:
                logger.error(f"SEER API request failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error downloading SEER data: {e}")
            return pd.DataFrame()
    
    def _process_seer_data(self, data: Dict) -> pd.DataFrame:
        """
        Process raw SEER data into a standardized format.
        
        Args:
            data: Raw SEER API response
            
        Returns:
            Processed DataFrame
        """
        logger.info("Processing SEER data...")
        
        try:
            # Extract the data array from SEER response
            if 'data' in data:
                records = data['data']
            else:
                records = data
            
            processed_data = []
            
            for record in records:
                # Map SEER fields to our standard format
                processed_record = {
                    'zip_code': record.get('zip_code', record.get('geography_code')),
                    'county': record.get('county_name', ''),
                    'state': record.get('state_code', ''),
                    'year': record.get('year', 2020),
                    'incidence_count': record.get('count', 0),
                    'population': record.get('population', 0),
                    'incidence_rate': record.get('rate', 0)
                }
                
                # Calculate incidence rate if not provided
                if processed_record['incidence_rate'] == 0 and processed_record['population'] > 0:
                    processed_record['incidence_rate'] = (
                        processed_record['incidence_count'] / processed_record['population']
                    ) * 100000
                
                processed_data.append(processed_record)
            
            df = pd.DataFrame(processed_data)
            
            # Remove records with zero population or invalid data
            df = df[df['population'] > 0]
            df = df[df['incidence_rate'] >= 0]
            
            logger.info(f"Processed {len(df)} valid records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing SEER data: {e}")
            return pd.DataFrame()
    
    def download_by_state(self, states: List[str] = None) -> pd.DataFrame:
        """
        Download pancreatic cancer data for specific states.
        
        Args:
            states: List of state codes to download
            
        Returns:
            DataFrame with state-specific data
        """
        if states is None:
            states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
        
        logger.info(f"Downloading SEER data for states: {states}")
        
        all_data = []
        
        for state in states:
            logger.info(f"Downloading data for {state}...")
            
            try:
                # Download data for specific state
                state_data = self.download_pancreatic_cancer_data(geography="county")
                
                # Filter for the specific state
                if not state_data.empty:
                    state_data = state_data[state_data['state'] == state]
                    all_data.append(state_data)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error downloading data for {state}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self._save_data(combined_df, f"seer_pancreatic_cancer_{datetime.now().strftime('%Y%m%d')}.csv")
            return combined_df
        else:
            logger.warning("No SEER data downloaded")
            return pd.DataFrame()
    
    def download_zipcode_data(self) -> pd.DataFrame:
        """
        Download pancreatic cancer data at zip code level.
        
        Returns:
            DataFrame with zip code level data
        """
        logger.info("Downloading SEER zip code level data...")
        
        try:
            # Try to download zip code level data
            zip_data = self.download_pancreatic_cancer_data(geography="zip")
            
            if not zip_data.empty:
                self._save_data(zip_data, f"seer_zipcode_pancreatic_cancer_{datetime.now().strftime('%Y%m%d')}.csv")
                return zip_data
            else:
                logger.warning("No zip code level data available, trying county level...")
                return self.download_by_state()
                
        except Exception as e:
            logger.error(f"Error downloading zip code data: {e}")
            return pd.DataFrame()
    
    def _save_data(self, df: pd.DataFrame, filename: str):
        """
        Save downloaded data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        if df.empty:
            logger.warning("No data to save")
            return
        
        output_file = self.data_dir / filename
        df.to_csv(output_file, index=False)
        logger.info(f"Saved SEER data: {output_file}")
        logger.info(f"Records: {len(df)}")
        logger.info(f"States: {df['state'].nunique()}")
        logger.info(f"Zip codes: {df['zip_code'].nunique()}")
        logger.info(f"Incidence rate range: {df['incidence_rate'].min():.2f} - {df['incidence_rate'].max():.2f}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for downloaded data.
        
        Args:
            df: SEER data DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        summary = {
            'total_records': len(df),
            'states_represented': df['state'].nunique(),
            'zip_codes': df['zip_code'].nunique(),
            'years_covered': df['year'].nunique(),
            'total_cases': df['incidence_count'].sum(),
            'total_population': df['population'].sum(),
            'mean_incidence_rate': df['incidence_rate'].mean(),
            'median_incidence_rate': df['incidence_rate'].median(),
            'max_incidence_rate': df['incidence_rate'].max(),
            'min_incidence_rate': df['incidence_rate'].min(),
            'std_incidence_rate': df['incidence_rate'].std()
        }
        
        return summary


def main():
    """Main function to test SEER data download."""
    from src.utils.api_keys import setup_api_keys, get_api_key
    
    # Setup API keys
    api_keys = setup_api_keys()
    seer_api_key = get_api_key('seer', api_keys)
    
    if not seer_api_key:
        print("❌ No SEER API key found. Please add your SEER API key to config/api_keys.yaml")
        return
    
    # Initialize downloader
    downloader = SEERDownloader(api_key=seer_api_key)
    
    # Download data
    print("🔍 Testing SEER API connection...")
    data = downloader.download_zipcode_data()
    
    if not data.empty:
        summary = downloader.get_data_summary(data)
        print("\n📊 SEER DATA SUMMARY:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
    else:
        print("❌ No data downloaded. Check your API key and SEER access.")


if __name__ == "__main__":
    main() 