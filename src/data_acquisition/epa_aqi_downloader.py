"""
EPA Air Quality Index (AQI) data downloader.

This module handles downloading air quality data from EPA's Air Quality System (AQS)
API, including PM2.5, ozone, and other air pollutants.
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


class EPAAQIDownloader:
    """
    Downloads and processes EPA air quality data for environmental health analysis.
    
    EPA AQS provides air quality monitoring data from thousands of monitoring
    stations across the United States.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", api_key: Optional[str] = None):
        """Initialize the EPA AQI downloader with configuration."""
        self.config = self._load_config(config_path)
        self.aqi_config = self.config['data_sources']['epa_aqi']
        self.base_url = self.aqi_config['base_url']
        self.api_endpoint = self.aqi_config['api_endpoint']
        self.api_key = api_key
        
        # Create data directory
        self.data_dir = Path("data/raw/epa_aqi")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("EPA AQI Downloader initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def download_aqi_data(self, 
                         start_date: str = "2020-01-01",
                         end_date: str = "2020-12-31",
                         parameters: Optional[List[str]] = None,
                         states: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Download AQI data from EPA AQS API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            parameters: List of parameter codes (default: config parameters)
            states: List of state FIPS codes (default: all states)
            
        Returns:
            DataFrame with AQI data
        """
        logger.info("Downloading EPA AQI data...")
        
        if parameters is None:
            parameters = self.aqi_config['parameters']
        
        if states is None:
            states = self._get_all_states()
        
        all_data = []
        
        for parameter in parameters:
            logger.info(f"Downloading data for parameter {parameter}")
            
            for state in states:
                try:
                    # EPA AQS API parameters
                    params = {
                        'email': 'your-email@example.com',  # Required for EPA API
                        'key': self.api_key if self.api_key else 'demo',
                        'param': parameter,
                        'bdate': start_date,
                        'edate': end_date,
                        'state': state,
                        'format': 'json'
                    }
                    
                    response = self._make_api_request(params)
                    
                    if response is not None:
                        df = pd.DataFrame(response)
                        df['parameter_code'] = parameter
                        all_data.append(df)
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error downloading data for parameter {parameter}, state {state}: {e}")
                    continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self._save_data(combined_df, f"epa_aqi_{start_date}_{end_date}.csv")
            return combined_df
        else:
            logger.warning("No AQI data downloaded")
            return pd.DataFrame()
    
    def download_daily_aqi(self, 
                          start_date: str = "2020-01-01",
                          end_date: str = "2020-12-31",
                          counties: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Download daily AQI summary data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            counties: List of county FIPS codes (default: major counties)
            
        Returns:
            DataFrame with daily AQI data
        """
        logger.info("Downloading daily AQI data...")
        
        if counties is None:
            counties = self._get_major_counties()
        
        all_data = []
        
        for county in counties:
            logger.info(f"Downloading daily AQI for county {county}")
            
            try:
                params = {
                    'email': 'your-email@example.com',
                    'key': self.api_key if self.api_key else 'demo',
                    'param': '88101',  # PM2.5
                    'bdate': start_date,
                    'edate': end_date,
                    'county': county,
                    'format': 'json'
                }
                
                response = self._make_api_request(params)
                
                if response is not None:
                    df = pd.DataFrame(response)
                    df['county_fips'] = county
                    all_data.append(df)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error downloading daily AQI for county {county}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self._save_data(combined_df, f"epa_daily_aqi_{start_date}_{end_date}.csv")
            return combined_df
        else:
            logger.warning("No daily AQI data downloaded")
            return pd.DataFrame()
    
    def download_ozone_data(self, 
                           start_date: str = "2020-01-01",
                           end_date: str = "2020-12-31") -> pd.DataFrame:
        """
        Download ozone concentration data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with ozone data
        """
        logger.info("Downloading ozone data...")
        
        return self.download_aqi_data(
            start_date=start_date,
            end_date=end_date,
            parameters=['44201']  # Ozone parameter code
        )
    
    def download_pm25_data(self, 
                          start_date: str = "2020-01-01",
                          end_date: str = "2020-12-31") -> pd.DataFrame:
        """
        Download PM2.5 concentration data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with PM2.5 data
        """
        logger.info("Downloading PM2.5 data...")
        
        return self.download_aqi_data(
            start_date=start_date,
            end_date=end_date,
            parameters=['88101']  # PM2.5 parameter code
        )
    
    def _make_api_request(self, params: Dict) -> Optional[List]:
        """
        Make API request to EPA AQS.
        
        Note: EPA AQS API requires registration for production use.
        This implementation uses the API key if available.
        """
        if not self.api_key or self.api_key == 'demo':
            logger.error("EPA AQS API key required for data access")
            return None
        
        # Add API key to parameters
        params['key'] = self.api_key
        logger.info("Making EPA AQS API request with provided key")
        
        try:
            # Construct the API URL
            url = f"{self.api_endpoint}/data"
            
            # Make the API request
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully retrieved {len(data)} records from EPA AQS API")
                return data
            else:
                logger.error(f"EPA AQS API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making EPA AQS API request: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in EPA AQS API request: {e}")
            return None
    
    def _generate_sample_aqi_data(self, params: Dict) -> List[Dict]:
        """Generate sample EPA AQI data for demonstration purposes (DEPRECATED)."""
        logger.warning("Sample data generation is deprecated. Use real API calls instead.")
        return []
    
    def _get_aqi_category(self, aqi: float) -> str:
        """Convert AQI value to category."""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def _get_all_states(self) -> List[str]:
        """Get list of all state FIPS codes."""
        return ['06', '12', '13', '17', '25', '36', '48']  # CA, FL, GA, IL, MA, NY, TX
    
    def _get_major_counties(self) -> List[str]:
        """Get list of major county FIPS codes."""
        return ['06037', '12086', '13121', '17031', '25025', '36061', '48113']  # LA, Miami-Dade, Fulton, Cook, Suffolk, New York, Harris
    
    def _save_data(self, df: pd.DataFrame, filename: str):
        """Save data to CSV file."""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def get_all_data(self, 
                    start_date: str = "2020-01-01",
                    end_date: str = "2020-12-31") -> Dict[str, pd.DataFrame]:
        """
        Download all available EPA AQI data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing different types of AQI data
        """
        logger.info("Downloading all EPA AQI data...")
        
        data = {
            'aqi': self.download_aqi_data(start_date, end_date),
            'daily_aqi': self.download_daily_aqi(start_date, end_date),
            'ozone': self.download_ozone_data(start_date, end_date),
            'pm25': self.download_pm25_data(start_date, end_date)
        }
        
        logger.info("EPA AQI data download completed")
        return data


if __name__ == "__main__":
    # Example usage
    downloader = EPAAQIDownloader()
    data = downloader.get_all_data()
    
    print("Downloaded EPA AQI data summary:")
    for data_type, df in data.items():
        print(f"{data_type}: {len(df)} records") 