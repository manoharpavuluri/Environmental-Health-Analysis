#!/usr/bin/env python3
"""
Test script for Environmental Health Analysis System.

This script tests the basic functionality of the analysis pipeline
to ensure all components work correctly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_data_acquisition():
    """Test data acquisition modules."""
    print("🧪 Testing Data Acquisition Modules...")
    
    try:
        from data_acquisition.seer_downloader import SEERDownloader
        from data_acquisition.epa_aqi_downloader import EPAAQIDownloader
        from data_acquisition.epa_water_downloader import EPAWaterDownloader
        from data_acquisition.census_downloader import CensusDownloader
        
        # Test SEER downloader
        seer_downloader = SEERDownloader()
        seer_data = seer_downloader.get_all_data()
        print(f"   ✅ SEER Downloader: {len(seer_data['incidence'])} records")
        
        # Test AQI downloader
        aqi_downloader = EPAAQIDownloader()
        aqi_data = aqi_downloader.get_all_data()
        print(f"   ✅ EPA AQI Downloader: {len(aqi_data['aqi'])} records")
        
        # Test water downloader
        water_downloader = EPAWaterDownloader()
        water_data = water_downloader.get_all_data()
        print(f"   ✅ EPA Water Downloader: {len(water_data['water_quality'])} records")
        
        # Test census downloader
        census_downloader = CensusDownloader()
        census_data = census_downloader.get_all_data()
        print(f"   ✅ Census Downloader: {len(census_data['demographics'])} records")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data Acquisition Test Failed: {e}")
        return False

def test_analysis_modules():
    """Test analysis modules."""
    print("\n🧪 Testing Analysis Modules...")
    
    try:
        from analysis.correlation_analysis import CorrelationAnalyzer
        from analysis.spatial_analysis import SpatialAnalyzer
        from analysis.statistical_analysis import StatisticalAnalyzer
        from analysis.machine_learning import MLAnalyzer
        
        # Create sample data
        sample_data = pd.DataFrame({
            'county': ['Los Angeles, CA', 'Cook, IL', 'Harris, TX', 'Maricopa, AZ'],
            'incidence_rate': [12.5, 11.8, 13.2, 10.9],
            'mortality_rate': [11.0, 10.5, 12.1, 9.8],
            'aqi': [75, 65, 85, 70],
            'arsenic': [5.2, 3.1, 7.8, 2.9],
            'lead': [2.1, 1.8, 3.2, 1.5]
        })
        
        # Test correlation analyzer
        correlation_analyzer = CorrelationAnalyzer()
        correlation_results = correlation_analyzer.analyze_aqi_pancreatic_correlation(
            sample_data, sample_data
        )
        print(f"   ✅ Correlation Analyzer: {len(correlation_results.get('correlations', {}))} correlations")
        
        # Test spatial analyzer (disabled)
        # spatial_analyzer = SpatialAnalyzer()
        # spatial_results = spatial_analyzer.analyze_spatial_patterns(sample_data)
        # print(f"   ✅ Spatial Analyzer: {len(spatial_results.get('spatial_autocorrelation', {}))} spatial tests")
        
        # Test statistical analyzer
        statistical_analyzer = StatisticalAnalyzer()
        statistical_results = statistical_analyzer.perform_statistical_tests(sample_data)
        print(f"   ✅ Statistical Analyzer: {len(statistical_results.get('hypothesis_tests', {}))} hypothesis tests")
        
        # Test machine learning analyzer
        ml_analyzer = MLAnalyzer()
        ml_results = ml_analyzer.build_predictive_models(sample_data)
        print(f"   ✅ Machine Learning Analyzer: {len(ml_results.get('regression_models', {}))} regression models")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Analysis Modules Test Failed: {e}")
        return False

def test_main_pipeline():
    """Test the main analysis pipeline."""
    print("\n🧪 Testing Main Analysis Pipeline...")
    
    try:
        from main import EnvironmentalHealthAnalyzer
        
        # Initialize analyzer
        analyzer = EnvironmentalHealthAnalyzer()
        
        # Run analysis with sample data
        results = analyzer.run_complete_analysis(
            start_date="2020-01-01",
            end_date="2020-12-31",
            download_data=False  # Use existing data
        )
        
        print(f"   ✅ Main Pipeline: Analysis completed successfully")
        print(f"   📊 Results summary:")
        print(f"      - Data sources: {len(results.get('data', {}))}")
        print(f"      - Analysis types: {len(results.get('analysis', {}))}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Main Pipeline Test Failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing Configuration...")
    
    try:
        import yaml
        
        # Test config file loading
        with open("config/config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        
        required_sections = ['data_sources', 'analysis', 'visualization', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
        
        print(f"   ✅ Configuration: All required sections present")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration Test Failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Environmental Health Analysis System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Acquisition", test_data_acquisition),
        ("Analysis Modules", test_analysis_modules),
        ("Main Pipeline", test_main_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"   ✅ {test_name} test PASSED")
            else:
                print(f"   ❌ {test_name} test FAILED")
        except Exception as e:
            print(f"   ❌ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run the main analysis: python main.py")
        print("   3. Explore the Jupyter notebook: notebooks/example_analysis.ipynb")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 