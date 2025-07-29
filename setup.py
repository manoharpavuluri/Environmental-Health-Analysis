#!/usr/bin/env python3
"""
Setup script for Environmental Health Analysis System.

This script helps users set up the analysis environment and configure
the system for their specific needs.
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data/raw/seer",
        "data/raw/epa_aqi", 
        "data/raw/epa_water",
        "data/raw/census",
        "data/processed",
        "data/results/correlation",
        "data/results/spatial",
        "data/results/statistical",
        "data/results/machine_learning",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    return True

def setup_configuration():
    """Set up configuration files."""
    print("\n⚙️ Setting up configuration...")
    
    # Check if config file exists
    config_file = Path("config/config.yaml")
    if not config_file.exists():
        print("❌ Configuration file not found. Please ensure config/config.yaml exists.")
        return False
    
    print("✅ Configuration file found")
    return True

def run_tests():
    """Run system tests."""
    print("\n🧪 Running system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def create_sample_data():
    """Create sample data for testing."""
    print("\n📊 Creating sample data...")
    
    try:
        # Run the main script to generate sample data
        result = subprocess.run([sys.executable, "main.py", "--analysis-only"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample data created successfully")
            return True
        else:
            print("⚠️ Sample data creation had issues (this is normal for demo)")
            return True  # Don't fail setup for this
    except Exception as e:
        print(f"⚠️ Could not create sample data: {e}")
        return True  # Don't fail setup for this

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Environmental Health Analysis System Setup Complete!")
    print("="*60)
    
    print("\n📝 Next Steps:")
    print("1. Explore the Jupyter notebook:")
    print("   jupyter notebook notebooks/example_analysis.ipynb")
    print("\n2. Run the complete analysis pipeline:")
    print("   python main.py")
    print("\n3. Run with specific parameters:")
    print("   python main.py --start-date 2020-01-01 --end-date 2020-12-31")
    print("\n4. Run analysis only (using existing data):")
    print("   python main.py --analysis-only")
    print("\n5. Test the system:")
    print("   python test_system.py")
    
    print("\n📚 Documentation:")
    print("- README.md: Project overview and usage instructions")
    print("- config/config.yaml: Configuration options")
    print("- src/: Source code modules")
    print("- data/: Data storage and results")
    
    print("\n🔧 Customization:")
    print("- Modify config/config.yaml for your specific needs")
    print("- Add your own data sources in src/data_acquisition/")
    print("- Extend analysis methods in src/analysis/")
    
    print("\n⚠️ Important Notes:")
    print("- This demo uses placeholder data. For real analysis, you'll need:")
    print("  * SEER API access for cancer data")
    print("  * EPA API keys for environmental data")
    print("  * Census API key for demographic data")
    print("- Get data from:")
print("  * SEER: https://seer.cancer.gov/")
print("  * EPA AQI (Kaggle): https://www.kaggle.com/datasets/epa/epa-historical-air-quality")
print("  * Census: https://api.census.gov/")

def main():
    """Main setup function."""
    print("🚀 Environmental Health Analysis System - Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Setup configuration
    if not setup_configuration():
        return False
    
    # Run tests
    if not run_tests():
        print("⚠️ Tests failed, but setup can continue...")
    
    # Create sample data
    create_sample_data()
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 