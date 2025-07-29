"""
Correlation analysis module for environmental health research.

This module analyzes correlations between environmental factors (AQI, water quality)
and pancreatic cancer/pancreatitis incidence rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import yaml
from loguru import logger
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    """
    Analyzes correlations between environmental factors and health outcomes.
    
    This class performs correlation analysis between environmental exposures
    (air quality, water quality) and pancreatic cancer/pancreatitis rates.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the correlation analyzer."""
        self.config = self._load_config(config_path)
        self.analysis_config = self.config['analysis']['statistical']
        self.confidence_level = self.analysis_config['confidence_level']
        self.significance_threshold = self.analysis_config['significance_threshold']
        self.correlation_method = self.analysis_config['correlation_method']
        
        # Create results directory
        self.results_dir = Path("data/results/correlation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Correlation Analyzer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def analyze_environmental_cancer_correlation(self, 
                                              cancer_data: pd.DataFrame,
                                              environmental_data: pd.DataFrame,
                                              merge_key: str = 'county') -> Dict:
        """
        Analyze correlation between environmental factors and cancer rates.
        
        Args:
            cancer_data: DataFrame with cancer incidence/mortality data
            environmental_data: DataFrame with environmental exposure data
            merge_key: Column name to merge datasets
            
        Returns:
            Dictionary with correlation results
        """
        logger.info("Analyzing environmental-cancer correlations...")
        
        # Merge datasets
        merged_data = self._merge_datasets(cancer_data, environmental_data, merge_key)
        
        if merged_data.empty:
            logger.error("No data after merging")
            return {}
        
        # Calculate correlations
        correlations = self._calculate_correlations(merged_data)
        
        # Perform statistical tests
        statistical_tests = self._perform_statistical_tests(merged_data)
        
        # Generate visualizations
        self._create_correlation_plots(merged_data)
        
        results = {
            'correlations': correlations,
            'statistical_tests': statistical_tests,
            'merged_data': merged_data
        }
        
        self._save_results(results)
        return results
    
    def analyze_aqi_pancreatic_correlation(self, 
                                         cancer_data: pd.DataFrame,
                                         aqi_data: pd.DataFrame) -> Dict:
        """
        Analyze correlation between AQI and pancreatic cancer rates.
        
        Args:
            cancer_data: DataFrame with pancreatic cancer data
            aqi_data: DataFrame with AQI data
            
        Returns:
            Dictionary with AQI-cancer correlation results
        """
        logger.info("Analyzing AQI-pancreatic cancer correlations...")
        
        # Prepare data
        cancer_summary = self._summarize_cancer_data(cancer_data)
        aqi_summary = self._summarize_aqi_data(aqi_data)
        
        # Merge datasets
        merged_data = pd.merge(cancer_summary, aqi_summary, 
                              on='county', how='inner')
        
        if merged_data.empty:
            logger.error("No data after merging AQI and cancer data")
            return {}
        
        # Calculate correlations
        correlations = {}
        
        # AQI vs cancer rates
        for cancer_col in ['incidence_rate', 'mortality_rate']:
            for aqi_col in ['avg_aqi', 'max_aqi', 'unhealthy_days']:
                if cancer_col in merged_data.columns and aqi_col in merged_data.columns:
                    corr, p_value = self._calculate_correlation(
                        merged_data[cancer_col], merged_data[aqi_col]
                    )
                    correlations[f"{cancer_col}_vs_{aqi_col}"] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < self.significance_threshold
                    }
        
        # Create visualizations
        self._create_aqi_cancer_plots(merged_data)
        
        results = {
            'correlations': correlations,
            'merged_data': merged_data,
            'summary_stats': self._calculate_summary_stats(merged_data)
        }
        
        self._save_aqi_results(results)
        return results
    
    def analyze_water_quality_correlation(self, 
                                        cancer_data: pd.DataFrame,
                                        water_data: pd.DataFrame) -> Dict:
        """
        Analyze correlation between water quality and pancreatic cancer rates.
        
        Args:
            cancer_data: DataFrame with pancreatic cancer data
            water_data: DataFrame with water quality data
            
        Returns:
            Dictionary with water quality-cancer correlation results
        """
        logger.info("Analyzing water quality-pancreatic cancer correlations...")
        
        # Prepare data
        cancer_summary = self._summarize_cancer_data(cancer_data)
        water_summary = self._summarize_water_data(water_data)
        
        # Merge datasets
        merged_data = pd.merge(cancer_summary, water_summary, 
                              on='county', how='inner')
        
        if merged_data.empty:
            logger.error("No data after merging water quality and cancer data")
            return {}
        
        # Calculate correlations
        correlations = {}
        
        # Water contaminants vs cancer rates
        for cancer_col in ['incidence_rate', 'mortality_rate']:
            for contaminant_col in ['arsenic', 'lead', 'chromium', 'nitrate']:
                if cancer_col in merged_data.columns and contaminant_col in merged_data.columns:
                    corr, p_value = self._calculate_correlation(
                        merged_data[cancer_col], merged_data[contaminant_col]
                    )
                    correlations[f"{cancer_col}_vs_{contaminant_col}"] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < self.significance_threshold
                    }
        
        # Create visualizations
        self._create_water_cancer_plots(merged_data)
        
        results = {
            'correlations': correlations,
            'merged_data': merged_data,
            'summary_stats': self._calculate_summary_stats(merged_data)
        }
        
        self._save_water_results(results)
        return results
    
    def _merge_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                       merge_key: str) -> pd.DataFrame:
        """Merge two datasets on specified key."""
        try:
            merged = pd.merge(df1, df2, on=merge_key, how='inner')
            logger.info(f"Merged datasets: {len(merged)} records")
            return merged
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            return pd.DataFrame()
    
    def _calculate_correlations(self, data: pd.DataFrame) -> Dict:
        """Calculate correlations between all numeric columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr, p_value = self._calculate_correlation(data[col1], data[col2])
                correlations[f"{col1}_vs_{col2}"] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < self.significance_threshold
                }
        
        return correlations
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """Calculate correlation coefficient and p-value."""
        # Remove NaN values
        mask = ~(x.isna() | y.isna())
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return np.nan, np.nan
        
        if self.correlation_method == 'pearson':
            corr, p_value = stats.pearsonr(x_clean, y_clean)
        elif self.correlation_method == 'spearman':
            corr, p_value = stats.spearmanr(x_clean, y_clean)
        else:
            corr, p_value = stats.kendalltau(x_clean, y_clean)
        
        return corr, p_value
    
    def _perform_statistical_tests(self, data: pd.DataFrame) -> Dict:
        """Perform additional statistical tests."""
        tests = {}
        
        # T-test for high vs low exposure groups
        if 'aqi' in data.columns and 'incidence_rate' in data.columns:
            high_aqi = data[data['aqi'] > data['aqi'].median()]
            low_aqi = data[data['aqi'] <= data['aqi'].median()]
            
            if len(high_aqi) > 0 and len(low_aqi) > 0:
                t_stat, p_value = stats.ttest_ind(
                    high_aqi['incidence_rate'], low_aqi['incidence_rate']
                )
                tests['aqi_high_vs_low'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.significance_threshold
                }
        
        return tests
    
    def _summarize_cancer_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Summarize cancer data by county."""
        if 'county' not in data.columns:
            logger.error("County column not found in cancer data")
            return pd.DataFrame()
        
        summary = data.groupby('county').agg({
            'incidence_rate': ['mean', 'std', 'count'],
            'mortality_rate': ['mean', 'std', 'count'],
            'year': 'count'
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['county'] + [f"{col[0]}_{col[1]}" for col in summary.columns[1:]]
        
        return summary
    
    def _summarize_aqi_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Summarize AQI data by county."""
        if 'county' not in data.columns:
            logger.error("County column not found in AQI data")
            return pd.DataFrame()
        
        summary = data.groupby('county').agg({
            'aqi': ['mean', 'max', 'std'],
            'concentration': 'mean',
            'date': 'count'
        }).reset_index()
        
        # Flatten column names
        summary.columns = ['county'] + [f"{col[0]}_{col[1]}" for col in summary.columns[1:]]
        
        # Calculate unhealthy days
        if 'aqi' in data.columns:
            unhealthy_days = data[data['aqi'] > 100].groupby('county').size()
            summary = summary.merge(unhealthy_days.reset_index(name='unhealthy_days'), 
                                  on='county', how='left')
        
        return summary
    
    def _summarize_water_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Summarize water quality data by county."""
        if 'county' not in data.columns:
            logger.error("County column not found in water data")
            return pd.DataFrame()
        
        # Assuming water data has contaminant columns
        contaminant_cols = [col for col in data.columns if col in 
                           ['arsenic', 'lead', 'chromium', 'nitrate', 'cadmium']]
        
        if not contaminant_cols:
            logger.warning("No contaminant columns found in water data")
            return pd.DataFrame()
        
        summary = data.groupby('county')[contaminant_cols].agg(['mean', 'max']).reset_index()
        
        # Flatten column names
        summary.columns = ['county'] + [f"{col[0]}_{col[1]}" for col in summary.columns[1:]]
        
        return summary
    
    def _calculate_summary_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the dataset."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'count': data[col].count()
            }
        
        return stats
    
    def _create_correlation_plots(self, data: pd.DataFrame):
        """Create correlation plots."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation plots")
            return
        
        # Correlation matrix heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix of Environmental and Health Variables')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Scatter plots for key variables
        key_pairs = [
            ('incidence_rate', 'aqi'),
            ('mortality_rate', 'aqi'),
            ('incidence_rate', 'arsenic'),
            ('mortality_rate', 'arsenic')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (x_col, y_col) in enumerate(key_pairs):
            if x_col in data.columns and y_col in data.columns:
                axes[i].scatter(data[x_col], data[y_col], alpha=0.6)
                axes[i].set_xlabel(x_col.replace('_', ' ').title())
                axes[i].set_ylabel(y_col.replace('_', ' ').title())
                axes[i].set_title(f'{x_col} vs {y_col}')
                
                # Add trend line
                z = np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1)
                p = np.poly1d(z)
                axes[i].plot(data[x_col], p(data[x_col]), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_aqi_cancer_plots(self, data: pd.DataFrame):
        """Create AQI-cancer specific plots."""
        if 'incidence_rate' not in data.columns or 'aqi_mean' not in data.columns:
            logger.warning("Required columns not found for AQI-cancer plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AQI vs Incidence Rate
        axes[0, 0].scatter(data['aqi_mean'], data['incidence_rate'], alpha=0.6)
        axes[0, 0].set_xlabel('Average AQI')
        axes[0, 0].set_ylabel('Incidence Rate (per 100,000)')
        axes[0, 0].set_title('AQI vs Pancreatic Cancer Incidence')
        
        # AQI vs Mortality Rate
        if 'mortality_rate' in data.columns:
            axes[0, 1].scatter(data['aqi_mean'], data['mortality_rate'], alpha=0.6)
            axes[0, 1].set_xlabel('Average AQI')
            axes[0, 1].set_ylabel('Mortality Rate (per 100,000)')
            axes[0, 1].set_title('AQI vs Pancreatic Cancer Mortality')
        
        # Unhealthy Days vs Incidence
        if 'unhealthy_days' in data.columns:
            axes[1, 0].scatter(data['unhealthy_days'], data['incidence_rate'], alpha=0.6)
            axes[1, 0].set_xlabel('Number of Unhealthy AQI Days')
            axes[1, 0].set_ylabel('Incidence Rate (per 100,000)')
            axes[1, 0].set_title('Unhealthy Days vs Incidence Rate')
        
        # AQI Distribution
        axes[1, 1].hist(data['aqi_mean'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Average AQI')
        axes[1, 1].set_ylabel('Number of Counties')
        axes[1, 1].set_title('Distribution of Average AQI by County')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'aqi_cancer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_water_cancer_plots(self, data: pd.DataFrame):
        """Create water quality-cancer specific plots."""
        contaminant_cols = [col for col in data.columns if any(cont in col for cont in 
                           ['arsenic', 'lead', 'chromium', 'nitrate'])]
        
        if not contaminant_cols or 'incidence_rate' not in data.columns:
            logger.warning("Required columns not found for water-cancer plots")
            return
        
        n_contaminants = min(len(contaminant_cols), 4)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, contaminant_col in enumerate(contaminant_cols[:n_contaminants]):
            if contaminant_col in data.columns:
                axes[i].scatter(data[contaminant_col], data['incidence_rate'], alpha=0.6)
                axes[i].set_xlabel(contaminant_col.replace('_', ' ').title())
                axes[i].set_ylabel('Incidence Rate (per 100,000)')
                axes[i].set_title(f'{contaminant_col.replace("_", " ").title()} vs Incidence Rate')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'water_cancer_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, results: Dict):
        """Save correlation results to file."""
        import json
        
        # Save correlations
        with open(self.results_dir / 'correlation_results.json', 'w') as f:
            json.dump(results['correlations'], f, indent=2, default=str)
        
        # Save statistical tests
        with open(self.results_dir / 'statistical_tests.json', 'w') as f:
            json.dump(results['statistical_tests'], f, indent=2, default=str)
        
        # Save merged data
        if 'merged_data' in results:
            results['merged_data'].to_csv(self.results_dir / 'merged_data.csv', index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def _save_aqi_results(self, results: Dict):
        """Save AQI-specific results."""
        import json
        
        with open(self.results_dir / 'aqi_correlation_results.json', 'w') as f:
            json.dump(results['correlations'], f, indent=2, default=str)
        
        if 'merged_data' in results:
            results['merged_data'].to_csv(self.results_dir / 'aqi_merged_data.csv', index=False)
    
    def _save_water_results(self, results: Dict):
        """Save water quality-specific results."""
        import json
        
        with open(self.results_dir / 'water_correlation_results.json', 'w') as f:
            json.dump(results['correlations'], f, indent=2, default=str)
        
        if 'merged_data' in results:
            results['merged_data'].to_csv(self.results_dir / 'water_merged_data.csv', index=False)


if __name__ == "__main__":
    # Example usage
    analyzer = CorrelationAnalyzer()
    
    # Load sample data (you would load your actual data here)
    cancer_data = pd.DataFrame({
        'county': ['Los Angeles, CA', 'Cook, IL', 'Harris, TX'],
        'incidence_rate': [12.5, 11.8, 13.2],
        'mortality_rate': [11.0, 10.5, 12.1]
    })
    
    aqi_data = pd.DataFrame({
        'county': ['Los Angeles, CA', 'Cook, IL', 'Harris, TX'],
        'aqi': [75, 65, 85],
        'concentration': [9.0, 7.8, 10.2]
    })
    
    results = analyzer.analyze_aqi_pancreatic_correlation(cancer_data, aqi_data)
    print("Correlation analysis completed") 