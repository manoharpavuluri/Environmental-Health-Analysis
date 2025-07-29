"""
Data acquisition module for environmental health analysis.

This module handles downloading and collecting data from various sources:
- SEER cancer data
- EPA air quality data
- EPA water quality data
- Census demographic data
"""

from .seer_downloader import SEERDownloader
from .epa_aqi_downloader import EPAAQIDownloader
from .epa_water_downloader import EPAWaterDownloader
from .census_downloader import CensusDownloader

__all__ = [
    'SEERDownloader',
    'EPAAQIDownloader',
    'EPAWaterDownloader',
    'CensusDownloader'
] 