"""
Census data downloader for demographic information.

This module handles downloading demographic data from the US Census Bureau
API, including socioeconomic factors that may influence health outcomes.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from loguru import logger
import time
from datetime import datetime
import json


class CensusDownloader:
    """
    Downloads and processes US Census data for environmental health analysis.
    
    Census data provides demographic and socioeconomic information that can
    be used as confounding variables in environmental health studies.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", api_key: Optional[str] = None):
        """Initialize the Census downloader with configuration."""
        self.config = self._load_config(config_path)
        self.census_config = self.config['data_sources']['census']
        self.base_url = self.census_config['base_url']
        self.api_key_required = self.census_config['api_key_required']
        self.api_key = api_key
        
        # Create data directory
        self.data_dir = Path("data/raw/census")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Census Downloader initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def download_demographic_data(self, 
                                year: int = 2020,
                                variables: Optional[List[str]] = None,
                                states: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Download demographic data from US Census API.
        
        Args:
            year: Census year (default: 2020)
            variables: List of Census variables to download (default: config variables)
            states: List of state FIPS codes (default: all states)
            
        Returns:
            DataFrame with demographic data
        """
        logger.info("Downloading Census demographic data...")
        
        if variables is None:
            variables = self.census_config['variables']
        
        if states is None:
            states = self._get_all_states()
        
        all_data = []
        
        for state in states:
            logger.info(f"Downloading demographic data for state {state}")
            
            try:
                # Census API parameters
                params = {
                    'get': ','.join(variables),
                    'for': 'county:*',
                    'in': f'state:{state}',
                    'key': self.api_key if self.api_key else 'demo'
                }
                
                response = self._make_api_request(params)
                
                if response is not None:
                    df = pd.DataFrame(response)
                    df['state_fips'] = state
                    df['year'] = year
                    all_data.append(df)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error downloading data for state {state}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self._save_data(combined_df, f"census_demographics_{year}.csv")
            return combined_df
        else:
            logger.warning("No demographic data downloaded")
            return pd.DataFrame()
    
    def download_income_data(self, year: int = 2020) -> pd.DataFrame:
        """
        Download household income data.
        
        Args:
            year: Census year (default: 2020)
            
        Returns:
            DataFrame with income data
        """
        logger.info("Downloading Census income data...")
        
        return self.download_demographic_data(
            year=year,
            variables=['B19013_001E']  # Median household income
        )
    
    def download_education_data(self, year: int = 2020) -> pd.DataFrame:
        """
        Download education level data.
        
        Args:
            year: Census year (default: 2020)
            
        Returns:
            DataFrame with education data
        """
        logger.info("Downloading Census education data...")
        
        return self.download_demographic_data(
            year=year,
            variables=['B15003_022E']  # Bachelor's degree
        )
    
    def download_employment_data(self, year: int = 2020) -> pd.DataFrame:
        """
        Download employment data.
        
        Args:
            year: Census year (default: 2020)
            
        Returns:
            DataFrame with employment data
        """
        logger.info("Downloading Census employment data...")
        
        return self.download_demographic_data(
            year=year,
            variables=['B23025_002E']  # Employment status
        )
    
    def _make_api_request(self, params: Dict) -> Optional[List]:
        """
        Make API request to US Census Bureau.
        
        Note: Census API requires registration for production use.
        This implementation uses the API key if available.
        """
        if not self.api_key or self.api_key == 'demo':
            logger.error("Census API key required for data access")
            logger.info("Please obtain Census API key from: https://api.census.gov/data/key_signup.html")
            return None
        
        # Add API key to parameters
        params['key'] = self.api_key
        logger.info("Making Census API request with provided key")
        
        try:
            # Construct the API URL for ACS 5-year estimates
            url = f"{self.base_url}/2020/acs/acs5"
            
            # Make the API request
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully retrieved {len(data)} records from Census API")
                return data
            else:
                logger.error(f"Census API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making Census API request: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Census API request: {e}")
            return None
    
    def _generate_sample_census_data(self, params: Dict) -> List[Dict]:
        """Generate sample Census data for demonstration purposes (DEPRECATED)."""
        logger.warning("Sample data generation is deprecated. Use real API calls instead.")
        return []
    
    def _get_all_states(self) -> List[str]:
        """Get list of all state FIPS codes."""
        return ['06', '12', '13', '17', '25', '36', '48']  # CA, FL, GA, IL, MA, NY, TX
    
    def _save_data(self, df: pd.DataFrame, filename: str):
        """Save data to CSV file."""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def get_all_data(self, year: int = 2020) -> Dict[str, pd.DataFrame]:
        """
        Download all available Census demographic data.
        
        Args:
            year: Census year (default: 2020)
            
        Returns:
            Dictionary containing different types of demographic data
        """
        logger.info("Downloading all Census demographic data...")
        
        data = {
            'demographics': self.download_demographic_data(year),
            'income': self.download_income_data(year),
            'education': self.download_education_data(year),
            'employment': self.download_employment_data(year)
        }
        
        logger.info("Census demographic data download completed")
        return data


if __name__ == "__main__":
    # Example usage
    downloader = CensusDownloader()
    data = downloader.get_all_data()
    
    print("Downloaded Census data summary:")
    for data_type, df in data.items():
        print(f"{data_type}: {len(df)} records") 