"""
API Key Management Utility

This module handles loading and managing API keys for various data sources
in a secure manner.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional
from loguru import logger


def load_api_keys(config_path: str = "config/api_keys.yaml") -> Dict[str, str]:
    """
    Load API keys from configuration file or environment variables.
    
    Args:
        config_path: Path to API keys configuration file
        
    Returns:
        Dictionary containing API keys
    """
    api_keys = {}
    
    # Try to load from config file first
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
                if config and 'api_keys' in config:
                    api_keys = config['api_keys']
                    logger.info("API keys loaded from configuration file")
        except Exception as e:
            logger.warning(f"Could not load API keys from config file: {e}")
    
    # Override with environment variables if available
    env_keys = {
        'epa_aqs_api_key': os.getenv('EPA_AQS_API_KEY'),
        'epa_water_api_key': os.getenv('EPA_WATER_API_KEY'),
        'census_api_key': os.getenv('CENSUS_API_KEY'),
        'seer_api_key': os.getenv('SEER_API_KEY'),
    }
    
    for key, value in env_keys.items():
        if value:
            api_keys[key] = value
    
    # Check if we have any API keys
    if not api_keys:
        logger.warning("No API keys found. Using demo/placeholder data.")
    
    return api_keys


def get_api_key(service: str, api_keys: Dict[str, str]) -> Optional[str]:
    """
    Get API key for a specific service.
    
    Args:
        service: Service name (epa, census, seer)
        api_keys: Dictionary of API keys
        
    Returns:
        API key if available, None otherwise
    """
    key_mapping = {
        'epa_aqs': 'epa_aqs_api_key',
        'epa_water': 'epa_water_api_key', 
        'census': 'census_api_key',
        'seer': 'seer_api_key'
    }
    
    key_name = key_mapping.get(service)
    if key_name and key_name in api_keys:
        return api_keys[key_name]
    
    return None


def validate_api_key(api_key: Optional[str], service: str) -> bool:
    """
    Validate that an API key is present and not a placeholder.
    
    Args:
        api_key: API key to validate
        service: Service name for logging
        
    Returns:
        True if API key is valid, False otherwise
    """
    if not api_key:
        logger.warning(f"No API key provided for {service}")
        return False
    
    if api_key in ['demo', 'placeholder', 'your_api_key_here']:
        logger.warning(f"Using placeholder API key for {service}")
        return False
    
    return True


def setup_api_keys() -> Dict[str, str]:
    """
    Setup API keys for the entire system.
    
    Returns:
        Dictionary of API keys
    """
    logger.info("Setting up API keys...")
    
    # Load API keys
    api_keys = load_api_keys()
    
    # Validate keys
    services = ['epa_aqs', 'epa_water', 'census', 'seer']
    for service in services:
        key = get_api_key(service, api_keys)
        if validate_api_key(key, service):
            logger.info(f"Valid API key found for {service}")
        else:
            logger.warning(f"No valid API key for {service} - using demo data")
    
    return api_keys 