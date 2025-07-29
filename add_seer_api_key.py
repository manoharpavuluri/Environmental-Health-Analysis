#!/usr/bin/env python3
"""
Add SEER API Key

This script helps you add your SEER API key to the configuration file.
"""

import yaml
from pathlib import Path


def add_seer_api_key():
    """Add SEER API key to configuration."""
    print("🔑 SEER API Key Configuration")
    print("=" * 40)
    
    # Get API key from user
    api_key = input("Please enter your SEER API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided.")
        return
    
    # Load current configuration
    config_file = Path("config/api_keys.yaml")
    
    if config_file.exists():
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        config = {
            'api_keys': {
                'epa_aqs_api_key': 'your_epa_aqs_api_key_here',
                'epa_water_api_key': 'your_epa_water_api_key_here',
                'census_api_key': 'b70de9a5aeed1c67c3ae065cd36629933cd2fb18',
                'seer_api_key': 'your_seer_api_key_here'
            }
        }
    
    # Update SEER API key
    config['api_keys']['seer_api_key'] = api_key
    
    # Save updated configuration
    with open(config_file, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    print(f"✅ SEER API key added to {config_file}")
    print("🔍 You can now run the SEER data downloader.")
    
    # Test the configuration
    print("\n🧪 Testing configuration...")
    try:
        from src.utils.api_keys import setup_api_keys, get_api_key
        api_keys = setup_api_keys()
        seer_key = get_api_key('seer', api_keys)
        
        if seer_key and seer_key != 'your_seer_api_key_here':
            print("✅ SEER API key loaded successfully!")
        else:
            print("❌ SEER API key not found in configuration.")
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")


if __name__ == "__main__":
    add_seer_api_key() 