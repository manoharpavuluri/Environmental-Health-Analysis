# Environmental Health Analysis System - Setup Guide

## Quick Start

1. **Install dependencies** (already done):
   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. **Run with demo data**:
   ```bash
   python3 main.py --analysis-only
   ```

## Setting Up API Keys (Optional)

For real data analysis, you'll need API keys from various sources:

### 1. EPA Air Quality Data

**Air Quality Data (Kaggle Dataset):**
- Go to: https://www.kaggle.com/datasets/epa/epa-historical-air-quality
- Download the EPA Historical Air Quality dataset
- Place the CSV file in `data/raw/epa_aqi/` directory
- No API key required - it's a public dataset

**Water Quality Data:**
- EPA water data is publicly available
- No API key required for basic access

### 2. Census API Key

**Demographic Data:**
- Go to: https://api.census.gov/data/key_signup.html
- Sign up for a free API key
- You'll receive your key immediately

### 3. SEER Data Access

**Cancer Data:**
- Go to: https://seer.cancer.gov/data/
- Register for SEER data access
- Follow the registration process

## Configuring API Keys

### Option 1: Configuration File (Recommended)

1. Copy the template:
   ```bash
   cp config/api_keys_template.yaml config/api_keys.yaml
   ```

2. Edit `config/api_keys.yaml` and add your API keys:
   ```yaml
   api_keys:
     epa:
       aqs_api_key: "your_actual_epa_key_here"
       water_api_key: "your_water_key_if_needed"
     census:
       api_key: "your_census_api_key_here"
     seer:
       api_key: "your_seer_api_key_here"
   ```

### Option 2: Environment Variables

Set environment variables:
```bash
export EPA_AQS_API_KEY="your_epa_key"
export CENSUS_API_KEY="your_census_key"
export SEER_API_KEY="your_seer_key"
```

### Option 3: .env File

Create a `.env` file in the project root:
```
EPA_AQS_API_KEY=your_epa_key_here
CENSUS_API_KEY=your_census_key_here
SEER_API_KEY=your_seer_key_here
```

## Running the System

### Demo Mode (No API Keys Required)
```bash
python3 main.py --analysis-only
```

### Full Analysis with Real Data
```bash
python3 main.py --start-date 2020-01-01 --end-date 2020-12-31
```

### Custom Date Range
```bash
python3 main.py --start-date 2019-01-01 --end-date 2021-12-31
```

### Using Existing Data
```bash
python3 main.py --no-download
```

## Security Notes

- **Never commit API keys** to version control
- The `.gitignore` file protects sensitive files
- API keys are loaded securely from environment or config files
- The system falls back to demo data if no valid keys are found

## Troubleshooting

### No API Keys Available
- The system will use demo/placeholder data
- All analysis will work with sample data
- Results will be marked as demo data

### API Key Errors
- Check that your API keys are valid
- Ensure you have proper access permissions
- Check API rate limits and quotas

### Data Download Issues
- Verify internet connection
- Check API service status
- Review logs in `logs/analysis.log`

## Next Steps

1. **Explore the Jupyter notebook**:
   ```bash
   jupyter notebook notebooks/example_analysis.ipynb
   ```

2. **Run tests**:
   ```bash
   python3 test_system.py
   ```

3. **Customize configuration**:
   - Edit `config/config.yaml` for analysis parameters
   - Modify data sources in `src/data_acquisition/`
   - Add new analysis methods in `src/analysis/`

## Support

- Check the logs in `logs/analysis.log` for detailed error messages
- Review the README.md for project overview
- Examine the source code in `src/` for implementation details 