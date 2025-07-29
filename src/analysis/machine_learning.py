"""
Machine learning module for environmental health research.

This module builds predictive models to analyze relationships between
environmental factors and health outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import yaml
from loguru import logger
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class MLAnalyzer:
    """
    Performs machine learning analysis on environmental health data.
    
    This class builds predictive models to understand relationships
    between environmental factors and health outcomes.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the machine learning analyzer."""
        self.config = self._load_config(config_path)
        self.ml_config = self.config['analysis']['machine_learning']
        self.test_size = self.ml_config['test_size']
        self.random_state = self.ml_config['random_state']
        self.cv_folds = self.ml_config['cv_folds']
        
        # Create results directory
        self.results_dir = Path("data/results/machine_learning")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Machine Learning Analyzer initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def build_predictive_models(self, data: pd.DataFrame) -> Dict:
        """
        Build predictive models for environmental health analysis.
        
        Args:
            data: DataFrame with environmental and health data
            
        Returns:
            Dictionary with model results and performance metrics
        """
        logger.info("Building predictive models...")
        
        results = {
            'regression_models': {},
            'classification_models': {},
            'clustering_results': {},
            'feature_importance': {},
            'model_comparison': {}
        }
        
        try:
            # Prepare data
            X, y_regression, y_classification = self._prepare_data(data)
            
            if X is not None and len(X) > 0:
                # Regression models
                results['regression_models'] = self._build_regression_models(X, y_regression)
                
                # Classification models
                results['classification_models'] = self._build_classification_models(X, y_classification)
                
                # Clustering analysis
                results['clustering_results'] = self._perform_clustering(X)
                
                # Feature importance analysis
                results['feature_importance'] = self._analyze_feature_importance(X, y_regression)
                
                # Model comparison
                results['model_comparison'] = self._compare_models(X, y_regression, y_classification)
                
                # Create visualizations
                self._create_ml_visualizations(results)
                
                # Save results
                self._save_ml_results(results)
                
            else:
                logger.warning("No valid data for machine learning analysis")
                
        except Exception as e:
            logger.error(f"Error in machine learning analysis: {e}")
        
        return results
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:
        """Prepare data for machine learning models."""
        try:
            # Select features and target variables
            feature_cols = ['aqi', 'arsenic', 'lead', 'chromium', 'nitrate']
            available_features = [col for col in feature_cols if col in data.columns]
            
            if len(available_features) == 0:
                logger.warning("No environmental features found in data")
                return None, None, None
            
            # Prepare features
            X = data[available_features].fillna(data[available_features].mean())
            
            # Prepare regression target (cancer incidence rate)
            y_regression = None
            if 'incidence_rate' in data.columns:
                y_regression = data['incidence_rate'].fillna(data['incidence_rate'].mean())
            
            # Prepare classification target (high/low risk)
            y_classification = None
            if 'incidence_rate' in data.columns:
                median_incidence = data['incidence_rate'].median()
                y_classification = (data['incidence_rate'] > median_incidence).astype(int)
            
            logger.info(f"Prepared data with {len(X)} samples and {len(available_features)} features")
            return X, y_regression, y_classification
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None, None
    
    def _build_regression_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Build regression models to predict health outcomes."""
        models = {}
        
        try:
            if y is None:
                logger.warning("No target variable for regression")
                return models
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 1. Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            
            models['linear_regression'] = {
                'model': lr_model,
                'r2_score': r2_score(y_test, lr_pred),
                'mse': mean_squared_error(y_test, lr_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
                'coefficients': dict(zip(X.columns, lr_model.coef_)),
                'intercept': lr_model.intercept_
            }
            
            # 2. Random Forest Regression
            rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            
            models['random_forest_regression'] = {
                'model': rf_model,
                'r2_score': r2_score(y_test, rf_pred),
                'mse': mean_squared_error(y_test, rf_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'feature_importance': dict(zip(X.columns, rf_model.feature_importances_))
            }
            
            # Cross-validation
            cv_scores = cross_val_score(rf_model, X, y, cv=self.cv_folds, scoring='r2')
            models['random_forest_regression']['cv_scores'] = cv_scores.tolist()
            models['random_forest_regression']['cv_mean'] = cv_scores.mean()
            models['random_forest_regression']['cv_std'] = cv_scores.std()
            
        except Exception as e:
            logger.error(f"Error building regression models: {e}")
        
        return models
    
    def _build_classification_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Build classification models to predict risk categories."""
        models = {}
        
        try:
            if y is None:
                logger.warning("No target variable for classification")
                return models
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 1. Logistic Regression
            lr_classifier = LogisticRegression(random_state=self.random_state)
            lr_classifier.fit(X_train_scaled, y_train)
            lr_pred = lr_classifier.predict(X_test_scaled)
            
            models['logistic_regression'] = {
                'model': lr_classifier,
                'accuracy': (lr_pred == y_test).mean(),
                'classification_report': classification_report(y_test, lr_pred, output_dict=True),
                'coefficients': dict(zip(X.columns, lr_classifier.coef_[0])),
                'intercept': lr_classifier.intercept_[0]
            }
            
            # 2. Random Forest Classification
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf_classifier.fit(X_train, y_train)
            rf_pred = rf_classifier.predict(X_test)
            
            models['random_forest_classification'] = {
                'model': rf_classifier,
                'accuracy': (rf_pred == y_test).mean(),
                'classification_report': classification_report(y_test, rf_pred, output_dict=True),
                'feature_importance': dict(zip(X.columns, rf_classifier.feature_importances_))
            }
            
            # Cross-validation
            cv_scores = cross_val_score(rf_classifier, X, y, cv=self.cv_folds, scoring='accuracy')
            models['random_forest_classification']['cv_scores'] = cv_scores.tolist()
            models['random_forest_classification']['cv_mean'] = cv_scores.mean()
            models['random_forest_classification']['cv_std'] = cv_scores.std()
            
        except Exception as e:
            logger.error(f"Error building classification models: {e}")
        
        return models
    
    def _perform_clustering(self, X: pd.DataFrame) -> Dict:
        """Perform clustering analysis on environmental data."""
        clustering_results = {}
        
        try:
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            clustering_results['kmeans'] = {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'inertia': kmeans.inertia_,
                'n_clusters': 3
            }
            
            # PCA for dimensionality reduction and visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            clustering_results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'transformed_data': X_pca.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error performing clustering: {e}")
        
        return clustering_results
    
    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze feature importance across different models."""
        feature_importance = {}
        
        try:
            if y is None:
                return feature_importance
            
            # Random Forest feature importance
            rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf_model.fit(X, y)
            
            feature_importance['random_forest'] = dict(zip(X.columns, rf_model.feature_importances_))
            
            # Correlation-based importance
            correlations = {}
            for col in X.columns:
                corr = X[col].corr(y)
                correlations[col] = abs(corr)
            
            feature_importance['correlation'] = correlations
            
            # Sort features by importance
            rf_importance = sorted(feature_importance['random_forest'].items(), 
                                 key=lambda x: x[1], reverse=True)
            feature_importance['top_features'] = [feature for feature, importance in rf_importance[:3]]
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
        
        return feature_importance
    
    def _compare_models(self, X: pd.DataFrame, y_regression: pd.Series, y_classification: pd.Series) -> Dict:
        """Compare performance of different models."""
        comparison = {}
        
        try:
            if y_regression is not None:
                # Compare regression models
                regression_models = self._build_regression_models(X, y_regression)
                
                comparison['regression'] = {
                    'linear_regression_r2': regression_models.get('linear_regression', {}).get('r2_score', 0),
                    'random_forest_r2': regression_models.get('random_forest_regression', {}).get('r2_score', 0),
                    'best_regression_model': max(
                        regression_models.get('linear_regression', {}).get('r2_score', 0),
                        regression_models.get('random_forest_regression', {}).get('r2_score', 0)
                    )
                }
            
            if y_classification is not None:
                # Compare classification models
                classification_models = self._build_classification_models(X, y_classification)
                
                comparison['classification'] = {
                    'logistic_regression_accuracy': classification_models.get('logistic_regression', {}).get('accuracy', 0),
                    'random_forest_accuracy': classification_models.get('random_forest_classification', {}).get('accuracy', 0),
                    'best_classification_model': max(
                        classification_models.get('logistic_regression', {}).get('accuracy', 0),
                        classification_models.get('random_forest_classification', {}).get('accuracy', 0)
                    )
                }
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
        
        return comparison
    
    def _create_ml_visualizations(self, results: Dict):
        """Create machine learning visualization plots."""
        try:
            # 1. Feature importance plot
            if 'feature_importance' in results and 'random_forest' in results['feature_importance']:
                importance = results['feature_importance']['random_forest']
                features = list(importance.keys())
                scores = list(importance.values())
                
                plt.figure(figsize=(10, 6))
                plt.barh(features, scores)
                plt.xlabel('Feature Importance')
                plt.title('Random Forest Feature Importance')
                plt.tight_layout()
                plt.savefig(self.results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Model comparison plot
            if 'model_comparison' in results:
                comparison = results['model_comparison']
                
                if 'regression' in comparison:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Regression comparison
                    reg_models = ['Linear Regression', 'Random Forest']
                    reg_scores = [
                        comparison['regression']['linear_regression_r2'],
                        comparison['regression']['random_forest_r2']
                    ]
                    
                    ax1.bar(reg_models, reg_scores)
                    ax1.set_ylabel('R² Score')
                    ax1.set_title('Regression Model Comparison')
                    ax1.set_ylim(0, 1)
                    
                    # Classification comparison
                    if 'classification' in comparison:
                        class_models = ['Logistic Regression', 'Random Forest']
                        class_scores = [
                            comparison['classification']['logistic_regression_accuracy'],
                            comparison['classification']['random_forest_accuracy']
                        ]
                        
                        ax2.bar(class_models, class_scores)
                        ax2.set_ylabel('Accuracy')
                        ax2.set_title('Classification Model Comparison')
                        ax2.set_ylim(0, 1)
                    
                    plt.tight_layout()
                    plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 3. Clustering visualization
            if 'clustering_results' in results and 'pca' in results['clustering_results']:
                pca_data = results['clustering_results']['pca']['transformed_data']
                cluster_labels = results['clustering_results']['kmeans']['cluster_labels']
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter([point[0] for point in pca_data], 
                                    [point[1] for point in pca_data], 
                                    c=cluster_labels, cmap='viridis')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.title('Environmental Data Clustering (PCA)')
                plt.colorbar(scatter)
                plt.tight_layout()
                plt.savefig(self.results_dir / 'clustering_pca.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"Error creating ML visualizations: {e}")
    
    def _save_ml_results(self, results: Dict):
        """Save machine learning results."""
        import json
        
        try:
            # Save results to JSON
            with open(self.results_dir / 'ml_analysis_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Machine learning results saved to {self.results_dir}")
            
        except Exception as e:
            logger.error(f"Error saving ML results: {e}")


if __name__ == "__main__":
    # Example usage
    analyzer = MLAnalyzer()
    
    # Sample data
    sample_data = pd.DataFrame({
        'county': ['Los Angeles, CA', 'Cook, IL', 'Harris, TX', 'Maricopa, AZ'],
        'incidence_rate': [12.5, 11.8, 13.2, 10.9],
        'aqi': [75, 65, 85, 70],
        'arsenic': [5.2, 3.1, 7.8, 2.9],
        'lead': [2.1, 1.8, 3.2, 1.5]
    })
    
    results = analyzer.build_predictive_models(sample_data)
    print("Machine learning analysis completed") 