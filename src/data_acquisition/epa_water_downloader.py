"""
EPA Water Quality data downloader.

This module handles downloading water quality data from EPA's Safe Drinking
Water Information System (SDWIS) and other water quality databases.
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from loguru import logger
import time
from datetime import datetime, timedelta
import json


class EPAWaterDownloader:
    """
    Downloads and processes EPA water quality data for environmental health analysis.
    
    EPA SDWIS provides drinking water quality data including contaminant levels
    from public water systems across the United States.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", api_key: Optional[str] = None):
        """Initialize the EPA water downloader with configuration."""
        self.config = self._load_config(config_path)
        self.water_config = self.config['data_sources']['epa_water']
        self.base_url = self.water_config['base_url']
        self.api_endpoint = self.water_config['api_endpoint']
        self.api_key = api_key
        
        # Create data directory
        self.data_dir = Path("data/raw/epa_water")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("EPA Water Downloader initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def download_water_quality_data(self, 
                                  start_date: str = "2020-01-01",
                                  end_date: str = "2020-12-31",
                                  contaminants: Optional[List[str]] = None,
                                  states: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Download water quality data from EPA SDWIS.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            contaminants: List of contaminants to download (default: config contaminants)
            states: List of state FIPS codes (default: all states)
            
        Returns:
            DataFrame with water quality data
        """
        logger.info("Downloading EPA water quality data...")
        
        if contaminants is None:
            contaminants = self.water_config['contaminants']
        
        if states is None:
            states = self._get_all_states()
        
        all_data = []
        
        for contaminant in contaminants:
            logger.info(f"Downloading data for contaminant {contaminant}")
            
            for state in states:
                try:
                    # EPA SDWIS API parameters
                    params = {
                        'email': 'your-email@example.com',  # Required for EPA API
                        'key': self.api_key if self.api_key else 'demo',
                        'contaminant': contaminant,
                        'bdate': start_date,
                        'edate': end_date,
                        'state': state,
                        'format': 'json'
                    }
                    
                    response = self._make_api_request(params)
                    
                    if response is not None:
                        df = pd.DataFrame(response)
                        df['contaminant'] = contaminant
                        all_data.append(df)
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error downloading data for contaminant {contaminant}, state {state}: {e}")
                    continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self._save_data(combined_df, f"epa_water_{start_date}_{end_date}.csv")
            return combined_df
        else:
            logger.warning("No water quality data downloaded")
            return pd.DataFrame()
    
    def download_arsenic_data(self, 
                            start_date: str = "2020-01-01",
                            end_date: str = "2020-12-31") -> pd.DataFrame:
        """
        Download arsenic concentration data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with arsenic data
        """
        logger.info("Downloading arsenic data...")
        
        return self.download_water_quality_data(
            start_date=start_date,
            end_date=end_date,
            contaminants=['arsenic']
        )
    
    def download_lead_data(self, 
                          start_date: str = "2020-01-01",
                          end_date: str = "2020-12-31") -> pd.DataFrame:
        """
        Download lead concentration data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with lead data
        """
        logger.info("Downloading lead data...")
        
        return self.download_water_quality_data(
            start_date=start_date,
            end_date=end_date,
            contaminants=['lead']
        )
    
    def download_nitrate_data(self, 
                             start_date: str = "2020-01-01",
                             end_date: str = "2020-12-31") -> pd.DataFrame:
        """
        Download nitrate concentration data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with nitrate data
        """
        logger.info("Downloading nitrate data...")
        
        return self.download_water_quality_data(
            start_date=start_date,
            end_date=end_date,
            contaminants=['nitrate']
        )
    
    def _make_api_request(self, params: Dict) -> Optional[List]:
        """
        Make API request to EPA SDWIS.
        
        Note: EPA SDWIS API requires registration for production use.
        This implementation uses the API key if available.
        """
        if not self.api_key or self.api_key == 'demo':
            logger.error("EPA SDWIS API key required for data access")
            logger.info("Please obtain EPA water quality data from: https://www.epa.gov/ground-water-and-drinking-water")
            return None
        
        # Add API key to parameters
        params['key'] = self.api_key
        logger.info("Making EPA SDWIS API request with provided key")
        
        try:
            # Construct the API URL
            url = f"{self.api_endpoint}/data"
            
            # Make the API request
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully retrieved {len(data)} records from EPA SDWIS API")
                return data
            else:
                logger.error(f"EPA SDWIS API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making EPA SDWIS API request: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in EPA SDWIS API request: {e}")
            return None
    
    def _generate_sample_water_data(self, params: Dict) -> List[Dict]:
        """Generate sample EPA water quality data for demonstration purposes (DEPRECATED)."""
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
    
    def get_all_data(self, 
                    start_date: str = "2020-01-01",
                    end_date: str = "2020-12-31") -> Dict[str, pd.DataFrame]:
        """
        Download all available EPA water quality data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing different types of water quality data
        """
        logger.info("Downloading all EPA water quality data...")
        
        data = {
            'water_quality': self.download_water_quality_data(start_date, end_date),
            'arsenic': self.download_arsenic_data(start_date, end_date),
            'lead': self.download_lead_data(start_date, end_date),
            'nitrate': self.download_nitrate_data(start_date, end_date)
        }
        
        logger.info("EPA water quality data download completed")
        return data


if __name__ == "__main__":
    # Example usage
    downloader = EPAWaterDownloader()
    data = downloader.get_all_data()
    
    print("Downloaded EPA water quality data summary:")
    for data_type, df in data.items():
        print(f"{data_type}: {len(df)} records") 