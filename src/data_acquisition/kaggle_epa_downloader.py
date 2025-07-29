"""
Kaggle EPA Air Quality data downloader.

This module handles downloading EPA air quality data from Kaggle's EPA Historical Air Quality dataset.
This provides a more reliable and comprehensive source than the EPA API.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from loguru import logger
import time
from datetime import datetime, timedelta
import zipfile
import requests
import os


class KaggleEPADownloader:
    """
    Downloads and processes EPA air quality data from Kaggle for environmental health analysis.
    
    Uses the EPA Historical Air Quality dataset from Kaggle:
    https://www.kaggle.com/datasets/epa/epa-historical-air-quality
    
    This dataset contains comprehensive air quality monitoring data from EPA's Air Quality System (AQS).
    """
    
    def __init__(self, config_path: str = "config/config.yaml", kaggle_username: Optional[str] = None, kaggle_key: Optional[str] = None):
        """Initialize the Kaggle EPA downloader with configuration."""
        self.config = self._load_config(config_path)
        self.kaggle_username = kaggle_username
        self.kaggle_key = kaggle_key
        
        # Create data directory
        self.data_dir = Path("data/raw/epa_aqi")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Kaggle EPA Downloader initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def download_epa_dataset(self, force_download: bool = False) -> pd.DataFrame:
        """
        Download EPA Historical Air Quality dataset from Kaggle.
        
        Args:
            force_download: Force re-download even if file exists
            
        Returns:
            DataFrame with EPA air quality data
        """
        logger.info("Downloading EPA Historical Air Quality dataset from Kaggle...")
        
        # Dataset URL and filename
        dataset_url = "https://www.kaggle.com/datasets/epa/epa-historical-air-quality"
        filename = "epa_historical_air_quality.csv"
        filepath = self.data_dir / filename
        
        # Check if file already exists
        if filepath.exists() and not force_download:
            logger.info(f"Dataset already exists at {filepath}")
            return pd.read_csv(filepath)
        
        try:
            # For Kaggle datasets, we need to use kaggle CLI or download manually
            # Since we can't use kaggle CLI here, we'll provide instructions
            logger.info("Kaggle dataset download requires manual steps:")
            logger.info("1. Go to: https://www.kaggle.com/datasets/epa/epa-historical-air-quality")
            logger.info("2. Download the dataset")
            logger.info("3. Extract and place the CSV file in data/raw/epa_aqi/")
            logger.info("4. Rename to 'epa_historical_air_quality.csv'")
            
            logger.warning("No EPA air quality data found. Please download the dataset manually.")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error downloading EPA dataset: {e}")
            return pd.DataFrame()
    
    def _create_sample_structure(self) -> pd.DataFrame:
        """Create sample data structure based on EPA Historical Air Quality dataset."""
        logger.warning("Sample data generation is deprecated. Use real Kaggle dataset instead.")
        return pd.DataFrame()
    
    def process_aqi_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Process the EPA air quality data into different categories.
        
        Args:
            df: Raw EPA air quality DataFrame
            
        Returns:
            Dictionary containing processed data by category
        """
        logger.info("Processing EPA air quality data...")
        
        if df.empty:
            logger.warning("No data to process")
            return {}
        
        processed_data = {}
        
        # Filter by parameter types
        if 'Parameter Name' in df.columns:
            # PM2.5 data
            pm25_data = df[df['Parameter Name'].str.contains('PM2.5', na=False)]
            if not pm25_data.empty:
                processed_data['pm25'] = pm25_data
            
            # Ozone data
            ozone_data = df[df['Parameter Name'].str.contains('Ozone', na=False)]
            if not ozone_data.empty:
                processed_data['ozone'] = ozone_data
            
            # All AQI data
            aqi_data = df[df['Parameter Name'].isin(['PM2.5', 'Ozone', 'PM10', 'NO2', 'SO2', 'CO'])]
            if not aqi_data.empty:
                processed_data['aqi'] = aqi_data
        
        # Save processed data
        for category, data in processed_data.items():
            filename = f"epa_{category}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = self.data_dir / filename
            data.to_csv(filepath, index=False)
            logger.info(f"Saved {category} data to {filepath}")
        
        return processed_data
    
    def get_daily_aqi_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create daily AQI summary from the EPA data.
        
        Args:
            df: Raw EPA air quality DataFrame
            
        Returns:
            DataFrame with daily AQI summaries
        """
        logger.info("Creating daily AQI summary...")
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by date and calculate daily statistics
        if 'Date Local' in df.columns and 'Arithmetic Mean' in df.columns:
            daily_summary = df.groupby('Date Local').agg({
                'Arithmetic Mean': ['mean', 'max', 'min'],
                'AQI': ['mean', 'max', 'min'],
                'Category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
            }).reset_index()
            
            # Flatten column names
            daily_summary.columns = ['Date', 'Mean_Value', 'Max_Value', 'Min_Value', 'Mean_AQI', 'Max_AQI', 'Min_AQI', 'Category']
            
            return daily_summary
        
        return pd.DataFrame()
    
    def get_county_level_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data to county level for spatial analysis.
        
        Args:
            df: Raw EPA air quality DataFrame
            
        Returns:
            DataFrame with county-level air quality data
        """
        logger.info("Creating county-level air quality data...")
        
        if df.empty:
            return pd.DataFrame()
        
        # Group by county and calculate statistics
        if 'County Name' in df.columns and 'Arithmetic Mean' in df.columns:
            county_data = df.groupby(['State Name', 'County Name']).agg({
                'Arithmetic Mean': ['mean', 'max', 'min', 'std'],
                'AQI': ['mean', 'max', 'min'],
                'Category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
            }).reset_index()
            
            # Flatten column names
            county_data.columns = ['State', 'County', 'Mean_Value', 'Max_Value', 'Min_Value', 'Std_Value', 'Mean_AQI', 'Max_AQI', 'Min_AQI', 'Category']
            
            return county_data
        
        return pd.DataFrame()
    
    def get_all_data(self, 
                    start_date: str = "2020-01-01",
                    end_date: str = "2020-12-31",
                    force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download and process all EPA air quality data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_download: Force re-download of dataset
            
        Returns:
            Dictionary containing different types of air quality data
        """
        logger.info("Downloading and processing all EPA air quality data...")
        
        # Download the main dataset
        raw_data = self.download_epa_dataset(force_download)
        
        if raw_data.empty:
            logger.warning("No EPA data available")
            return {}
        
        # Process the data
        processed_data = self.process_aqi_data(raw_data)
        
        # Add additional processed datasets
        daily_summary = self.get_daily_aqi_summary(raw_data)
        county_data = self.get_county_level_data(raw_data)
        
        if not daily_summary.empty:
            processed_data['daily_summary'] = daily_summary
        
        if not county_data.empty:
            processed_data['county_level'] = county_data
        
        # Add raw data
        processed_data['raw'] = raw_data
        
        logger.info("EPA air quality data processing completed")
        return processed_data 