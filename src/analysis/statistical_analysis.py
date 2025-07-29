"""
Statistical analysis module for environmental health research.

This module performs statistical tests and analysis on environmental factors
and health outcomes, including regression analysis and hypothesis testing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, List, Optional, Tuple
import yaml
from loguru import logger
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    Performs statistical analysis of environmental health data.
    
    This class conducts various statistical tests to examine relationships
    between environmental factors and health outcomes.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the statistical analyzer."""
        self.config = self._load_config(config_path)
        self.stats_config = self.config['analysis']['statistical']
        self.confidence_level = self.stats_config['confidence_level']
        self.significance_threshold = self.stats_config['significance_threshold']
        
        # Create results directory
        self.results_dir = Path("data/results/statistical")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Statistical Analyzer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def perform_statistical_tests(self, data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            data: DataFrame with environmental and health data
            
        Returns:
            Dictionary with statistical test results
        """
        logger.info("Performing statistical analysis...")
        
        results = {
            'descriptive_stats': {},
            'hypothesis_tests': {},
            'regression_analysis': {},
            'anova_results': {},
            'diagnostic_tests': {}
        }
        
        try:
            # Descriptive statistics
            results['descriptive_stats'] = self._calculate_descriptive_stats(data)
            
            # Hypothesis tests
            results['hypothesis_tests'] = self._perform_hypothesis_tests(data)
            
            # Regression analysis
            results['regression_analysis'] = self._perform_regression_analysis(data)
            
            # ANOVA tests
            results['anova_results'] = self._perform_anova_tests(data)
            
            # Diagnostic tests
            results['diagnostic_tests'] = self._perform_diagnostic_tests(data)
            
            # Create visualizations
            self._create_statistical_plots(data)
            
            # Save results
            self._save_statistical_results(results)
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
        
        return results
    
    def _calculate_descriptive_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate descriptive statistics for all numeric variables."""
        stats_dict = {}
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                stats_dict[col] = {
                    'count': data[col].count(),
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'q25': data[col].quantile(0.25),
                    'q75': data[col].quantile(0.75),
                    'skewness': data[col].skew(),
                    'kurtosis': data[col].kurtosis()
                }
                
        except Exception as e:
            logger.error(f"Error calculating descriptive statistics: {e}")
        
        return stats_dict
    
    def _perform_hypothesis_tests(self, data: pd.DataFrame) -> Dict:
        """Perform hypothesis tests on the data."""
        tests = {}
        
        try:
            # T-test for high vs low AQI groups
            if 'aqi' in data.columns and 'incidence_rate' in data.columns:
                high_aqi = data[data['aqi'] > data['aqi'].median()]
                low_aqi = data[data['aqi'] <= data['aqi'].median()]
                
                if len(high_aqi) > 0 and len(low_aqi) > 0:
                    t_stat, p_value = stats.ttest_ind(
                        high_aqi['incidence_rate'], low_aqi['incidence_rate']
                    )
                    tests['aqi_high_vs_low_ttest'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.significance_threshold,
                        'high_aqi_mean': high_aqi['incidence_rate'].mean(),
                        'low_aqi_mean': low_aqi['incidence_rate'].mean()
                    }
            
            # Mann-Whitney U test for non-parametric comparison
            if 'aqi' in data.columns and 'incidence_rate' in data.columns:
                high_aqi = data[data['aqi'] > data['aqi'].median()]
                low_aqi = data[data['aqi'] <= data['aqi'].median()]
                
                if len(high_aqi) > 0 and len(low_aqi) > 0:
                    u_stat, p_value = stats.mannwhitneyu(
                        high_aqi['incidence_rate'], low_aqi['incidence_rate'],
                        alternative='two-sided'
                    )
                    tests['aqi_high_vs_low_mannwhitney'] = {
                        'u_statistic': u_stat,
                        'p_value': p_value,
                        'significant': p_value < self.significance_threshold
                    }
            
            # Chi-square test for categorical variables
            if 'aqi_category' in data.columns and 'county' in data.columns:
                contingency_table = pd.crosstab(data['aqi_category'], data['county'])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                tests['aqi_category_chi2'] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'significant': p_value < self.significance_threshold
                }
            
            # Correlation tests
            if 'aqi' in data.columns and 'incidence_rate' in data.columns:
                # Pearson correlation
                pearson_corr, pearson_p = stats.pearsonr(
                    data['aqi'].dropna(), data['incidence_rate'].dropna()
                )
                tests['aqi_incidence_pearson'] = {
                    'correlation': pearson_corr,
                    'p_value': pearson_p,
                    'significant': pearson_p < self.significance_threshold
                }
                
                # Spearman correlation
                spearman_corr, spearman_p = stats.spearmanr(
                    data['aqi'].dropna(), data['incidence_rate'].dropna()
                )
                tests['aqi_incidence_spearman'] = {
                    'correlation': spearman_corr,
                    'p_value': spearman_p,
                    'significant': spearman_p < self.significance_threshold
                }
                
        except Exception as e:
            logger.error(f"Error performing hypothesis tests: {e}")
        
        return tests
    
    def _perform_regression_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform regression analysis."""
        regression_results = {}
        
        try:
            # Simple linear regression: AQI vs Incidence Rate
            if 'aqi' in data.columns and 'incidence_rate' in data.columns:
                from scipy import stats
                
                # Remove NaN values
                clean_data = data[['aqi', 'incidence_rate']].dropna()
                
                if len(clean_data) > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        clean_data['aqi'], clean_data['incidence_rate']
                    )
                    
                    regression_results['aqi_vs_incidence'] = {
                        'slope': slope,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_error': std_err,
                        'significant': p_value < self.significance_threshold
                    }
            
            # Multiple regression with multiple environmental factors
            if all(col in data.columns for col in ['aqi', 'arsenic', 'incidence_rate']):
                from sklearn.linear_model import LinearRegression
                from sklearn.preprocessing import StandardScaler
                
                # Prepare features
                features = ['aqi', 'arsenic']
                X = data[features].fillna(data[features].mean())
                y = data['incidence_rate'].fillna(data['incidence_rate'].mean())
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Fit model
                model = LinearRegression()
                model.fit(X_scaled, y)
                
                # Calculate R-squared
                r_squared = model.score(X_scaled, y)
                
                regression_results['multiple_regression'] = {
                    'coefficients': model.coef_.tolist(),
                    'intercept': model.intercept_,
                    'r_squared': r_squared,
                    'feature_names': features
                }
                
        except Exception as e:
            logger.error(f"Error performing regression analysis: {e}")
        
        return regression_results
    
    def _perform_anova_tests(self, data: pd.DataFrame) -> Dict:
        """Perform ANOVA tests."""
        anova_results = {}
        
        try:
            # One-way ANOVA for AQI categories
            if 'aqi_category' in data.columns and 'incidence_rate' in data.columns:
                categories = data['aqi_category'].unique()
                groups = [data[data['aqi_category'] == cat]['incidence_rate'].dropna() 
                         for cat in categories if len(data[data['aqi_category'] == cat]) > 0]
                
                if len(groups) > 1:
                    f_stat, p_value = stats.f_oneway(*groups)
                    anova_results['aqi_category_anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < self.significance_threshold,
                        'categories': categories.tolist()
                    }
                    
                    # Post-hoc Tukey test
                    if p_value < self.significance_threshold:
                        tukey_result = pairwise_tukeyhsd(
                            data['incidence_rate'].dropna(),
                            data['aqi_category'].dropna()
                        )
                        anova_results['aqi_category_tukey'] = {
                            'significant_pairs': tukey_result.pvalues < self.significance_threshold,
                            'p_values': tukey_result.pvalues.tolist()
                        }
            
        except Exception as e:
            logger.error(f"Error performing ANOVA tests: {e}")
        
        return anova_results
    
    def _perform_diagnostic_tests(self, data: pd.DataFrame) -> Dict:
        """Perform diagnostic tests for regression assumptions."""
        diagnostic_results = {}
        
        try:
            # Normality test for residuals (if regression was performed)
            if 'aqi' in data.columns and 'incidence_rate' in data.columns:
                clean_data = data[['aqi', 'incidence_rate']].dropna()
                
                if len(clean_data) > 2:
                    # Fit simple linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        clean_data['aqi'], clean_data['incidence_rate']
                    )
                    
                    # Calculate residuals
                    predicted = slope * clean_data['aqi'] + intercept
                    residuals = clean_data['incidence_rate'] - predicted
                    
                    # Shapiro-Wilk test for normality
                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    diagnostic_results['residuals_normality'] = {
                        'shapiro_statistic': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'normal': shapiro_p > 0.05
                    }
                    
                    # Breusch-Pagan test for homoscedasticity
                    try:
                        from statsmodels.stats.diagnostic import het_breuschpagan
                        bp_stat, bp_p_value, bp_f_stat, bp_f_p_value = het_breuschpagan(
                            residuals, clean_data[['aqi']]
                        )
                        diagnostic_results['homoscedasticity'] = {
                            'breusch_pagan_statistic': bp_stat,
                            'breusch_pagan_p_value': bp_p_value,
                            'homoscedastic': bp_p_value > 0.05
                        }
                    except:
                        logger.warning("Could not perform Breusch-Pagan test")
            
        except Exception as e:
            logger.error(f"Error performing diagnostic tests: {e}")
        
        return diagnostic_results
    
    def _create_statistical_plots(self, data: pd.DataFrame):
        """Create statistical visualization plots."""
        try:
            # 1. Distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            if 'incidence_rate' in data.columns:
                axes[0, 0].hist(data['incidence_rate'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[0, 0].set_title('Distribution of Cancer Incidence Rate')
                axes[0, 0].set_xlabel('Incidence Rate (per 100,000)')
                axes[0, 0].set_ylabel('Frequency')
            
            if 'aqi' in data.columns:
                axes[0, 1].hist(data['aqi'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[0, 1].set_title('Distribution of Air Quality Index')
                axes[0, 1].set_xlabel('AQI')
                axes[0, 1].set_ylabel('Frequency')
            
            if 'mortality_rate' in data.columns:
                axes[1, 0].hist(data['mortality_rate'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Distribution of Cancer Mortality Rate')
                axes[1, 0].set_xlabel('Mortality Rate (per 100,000)')
                axes[1, 0].set_ylabel('Frequency')
            
            if 'arsenic' in data.columns:
                axes[1, 1].hist(data['arsenic'].dropna(), bins=20, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Distribution of Arsenic Levels')
                axes[1, 1].set_xlabel('Arsenic (ppb)')
                axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Box plots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            if 'aqi_category' in data.columns and 'incidence_rate' in data.columns:
                data.boxplot(column='incidence_rate', by='aqi_category', ax=axes[0])
                axes[0].set_title('Cancer Incidence Rate by AQI Category')
                axes[0].set_xlabel('AQI Category')
                axes[0].set_ylabel('Incidence Rate (per 100,000)')
            
            if 'aqi_category' in data.columns and 'mortality_rate' in data.columns:
                data.boxplot(column='mortality_rate', by='aqi_category', ax=axes[1])
                axes[1].set_title('Cancer Mortality Rate by AQI Category')
                axes[1].set_xlabel('AQI Category')
                axes[1].set_ylabel('Mortality Rate (per 100,000)')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'boxplots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Q-Q plots for normality
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            if 'incidence_rate' in data.columns:
                stats.probplot(data['incidence_rate'].dropna(), dist="norm", plot=axes[0, 0])
                axes[0, 0].set_title('Q-Q Plot: Cancer Incidence Rate')
            
            if 'aqi' in data.columns:
                stats.probplot(data['aqi'].dropna(), dist="norm", plot=axes[0, 1])
                axes[0, 1].set_title('Q-Q Plot: Air Quality Index')
            
            if 'mortality_rate' in data.columns:
                stats.probplot(data['mortality_rate'].dropna(), dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot: Cancer Mortality Rate')
            
            if 'arsenic' in data.columns:
                stats.probplot(data['arsenic'].dropna(), dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot: Arsenic Levels')
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'qq_plots.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating statistical plots: {e}")
    
    def _save_statistical_results(self, results: Dict):
        """Save statistical analysis results."""
        import json
        
        try:
            # Save results to JSON
            with open(self.results_dir / 'statistical_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Statistical analysis results saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving statistical results: {e}")


if __name__ == "__main__":
    # Example usage
    analyzer = StatisticalAnalyzer()
    
    # Sample data
    sample_data = pd.DataFrame({
        'county': ['Los Angeles, CA', 'Cook, IL', 'Harris, TX', 'Maricopa, AZ'],
        'incidence_rate': [12.5, 11.8, 13.2, 10.9],
        'mortality_rate': [11.0, 10.5, 12.1, 9.8],
        'aqi': [75, 65, 85, 70],
        'aqi_category': ['Moderate', 'Moderate', 'Unhealthy for Sensitive Groups', 'Moderate']
    })
    
    results = analyzer.perform_statistical_tests(sample_data)
    print("Statistical analysis completed") 