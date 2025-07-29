# Pancreatic Cancer Hotspot Analysis Guide

## Overview

This system identifies geographic hotspots where pancreatic cancer incidence is high and may be associated with poor air quality. The analysis uses SEER cancer data and EPA air quality data to identify areas that warrant further investigation.

## 🎯 Analysis Methods

### 1. High Incidence Areas
- Identifies counties with pancreatic cancer incidence rates in the top 10%
- Calculates incidence rates per 100,000 population
- Maps geographic distribution of high-incidence areas

### 2. Poor Air Quality Areas
- Identifies counties with air quality index (AQI) in the top 20%
- Analyzes multiple pollutants (PM2.5, Ozone, PM10, NO2, SO2, CO)
- Maps areas with consistently poor air quality

### 3. Correlation Hotspots
- Finds areas where high cancer incidence correlates with poor air quality
- Uses statistical correlation analysis
- Identifies areas with both high incidence and poor air quality

### 4. Spatial Clustering
- Uses DBSCAN clustering algorithm to identify spatial clusters
- Combines cancer incidence and air quality data
- Identifies geographic clusters of concern

## 📊 Data Requirements

### SEER Cancer Data
- **Source**: SEER (Surveillance, Epidemiology, and End Results)
- **Required Fields**: County, State, Incidence Count, Population
- **Time Period**: 2010-2020 (configurable)
- **Cancer Type**: Pancreatic cancer

### EPA Air Quality Data
- **Source**: Kaggle EPA Historical Air Quality Dataset
- **URL**: https://www.kaggle.com/datasets/epa/epa-historical-air-quality
- **Required Fields**: County, State, AQI, Pollutant Levels, Date
- **Pollutants**: PM2.5, Ozone, PM10, NO2, SO2, CO

## 🚀 How to Run

### Option 1: Complete Analysis (Download + Analyze)
```bash
python run_hotspot_analysis.py
```

### Option 2: Analysis Only (Use Existing Data)
```bash
python run_hotspot_analysis.py --analysis-only
```

### Option 3: Force Re-download
```bash
python run_hotspot_analysis.py --force-download
```

### Option 4: Full Pipeline
```bash
python main.py
```

## 📁 Output Files

### Interactive Maps
- `data/results/hotspots/hotspot_map_high_incidence.html`
- `data/results/hotspots/hotspot_map_poor_air_quality.html`
- `data/results/hotspots/hotspot_map_correlation_hotspots.html`
- `data/results/hotspots/hotspot_map_cluster_hotspots.html`

### Analysis Report
- `data/results/hotspots/hotspot_analysis_report.html`

### Data Files
- `data/raw/seer/seer_incidence_2010-2020.csv`
- `data/raw/epa_aqi/epa_historical_air_quality.csv`

## 🔍 Understanding Results

### High Incidence Hotspots
- Areas with pancreatic cancer rates > 90th percentile
- May indicate genetic, environmental, or lifestyle factors
- Requires further investigation of local risk factors

### Poor Air Quality Hotspots
- Areas with AQI > 80th percentile
- May indicate environmental pollution sources
- Important for public health interventions

### Correlation Hotspots
- Areas with both high cancer incidence AND poor air quality
- Strongest evidence for environmental association
- Priority areas for intervention

### Cluster Hotspots
- Spatially clustered areas of concern
- May indicate regional environmental factors
- Useful for policy and intervention planning

## 📈 Statistical Methods

### Incidence Rate Calculation
```
Incidence Rate = (Cancer Cases / Population) × 100,000
```

### Correlation Analysis
- Pearson correlation between cancer incidence and AQI
- Threshold: r > 0.3 for significance
- Controls for population size and demographics

### Spatial Clustering
- DBSCAN algorithm with eps=0.5, min_samples=3
- Standardized features (incidence rate, AQI score)
- Identifies geographic clusters

## 🛠️ Configuration

### Edit `config/config.yaml`
```yaml
analysis:
  hotspot:
    incidence_threshold: 0.9  # Top 10% for high incidence
    aqi_threshold: 0.8        # Top 20% for poor air quality
    correlation_threshold: 0.3 # Minimum correlation
    cluster_eps: 0.5          # DBSCAN epsilon
    cluster_min_samples: 3     # DBSCAN min samples
```

### Data Sources
```yaml
data_sources:
  seer:
    years: [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    cancer_types: ["pancreas", "pancreatic"]
  
  epa_aqi:
    source: "kaggle"
    dataset_url: "https://www.kaggle.com/datasets/epa/epa-historical-air-quality"
```

## 🔧 Troubleshooting

### No Data Available
```
❌ Error: Insufficient data for analysis
```
**Solution**: Download the required datasets
1. Get SEER data from: https://seer.cancer.gov/data/
2. Get EPA data from: https://www.kaggle.com/datasets/epa/epa-historical-air-quality

### Missing Dependencies
```
ModuleNotFoundError: No module named 'folium'
```
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### API Key Issues
```
SEER API key required for data access
```
**Solution**: Configure API keys in `config/api_keys.yaml`

## 📋 Next Steps

### For Researchers
1. Validate hotspot results with local health data
2. Conduct detailed environmental assessments
3. Investigate specific pollutants in hotspot areas
4. Consider temporal analysis (seasonal patterns)

### For Public Health Officials
1. Prioritize intervention in correlation hotspots
2. Implement air quality monitoring in identified areas
3. Develop targeted public health campaigns
4. Consider policy changes for air quality improvement

### For Policy Makers
1. Use cluster analysis for regional policy planning
2. Allocate resources based on hotspot severity
3. Consider environmental justice implications
4. Develop long-term monitoring programs

## 📚 References

- SEER Cancer Data: https://seer.cancer.gov/
- EPA Air Quality Data: https://www.kaggle.com/datasets/epa/epa-historical-air-quality
- Spatial Analysis Methods: https://pysal.org/
- Statistical Analysis: https://scipy.org/

## 🤝 Contributing

To improve the hotspot analysis:

1. Add new analysis methods to `src/analysis/hotspot_analysis.py`
2. Enhance visualization in the mapping functions
3. Add temporal analysis capabilities
4. Include additional environmental factors
5. Improve statistical methods

## 📞 Support

For issues or questions:
1. Check the logs in `logs/hotspot_analysis.log`
2. Review the configuration in `config/config.yaml`
3. Ensure all dependencies are installed
4. Verify data sources are accessible 