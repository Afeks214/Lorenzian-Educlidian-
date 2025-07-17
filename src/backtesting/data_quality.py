"""
Data Quality Assurance Module
============================

Comprehensive data quality validation and monitoring for institutional backtesting:
- Missing data detection and handling
- Outlier identification and analysis
- Data consistency checks
- Quality scoring and reporting
- Data integrity validation
- Statistical anomaly detection
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataQualityAssurance:
    """Comprehensive data quality assurance for backtesting"""
    
    def __init__(self, 
                 outlier_threshold: float = 3.0,
                 missing_data_threshold: float = 0.05,
                 min_data_points: int = 100):
        """
        Initialize data quality assurance
        
        Args:
            outlier_threshold: Z-score threshold for outlier detection
            missing_data_threshold: Maximum acceptable missing data ratio
            min_data_points: Minimum required data points
        """
        self.outlier_threshold = outlier_threshold
        self.missing_data_threshold = missing_data_threshold
        self.min_data_points = min_data_points
        
        # Quality tracking
        self.quality_reports = {}
        self.outliers_detected = {}
        self.data_issues = []
        
        print("✅ Data Quality Assurance initialized")
        print(f"   📊 Outlier Threshold: {outlier_threshold} standard deviations")
        print(f"   📊 Missing Data Threshold: {missing_data_threshold:.1%}")
        print(f"   📊 Minimum Data Points: {min_data_points}")
    
    def comprehensive_quality_check(self, data: pd.DataFrame, 
                                  price_columns: List[str] = None,
                                  volume_columns: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment
        
        Args:
            data: DataFrame to analyze
            price_columns: List of price column names
            volume_columns: List of volume column names
            
        Returns:
            Comprehensive quality report
        """
        if price_columns is None:
            price_columns = ['Open', 'High', 'Low', 'Close']
        if volume_columns is None:
            volume_columns = ['Volume']
        
        quality_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'data_overview': self._analyze_data_overview(data),
            'missing_data_analysis': self._analyze_missing_data(data),
            'outlier_analysis': self._detect_outliers(data, price_columns + volume_columns),
            'consistency_checks': self._perform_consistency_checks(data, price_columns),
            'statistical_anomalies': self._detect_statistical_anomalies(data, price_columns),
            'data_integrity': self._validate_data_integrity(data, price_columns, volume_columns),
            'quality_score': 0,
            'recommendations': [],
            'data_issues': []
        }
        
        # Calculate overall quality score
        quality_report['quality_score'] = self._calculate_quality_score(quality_report)
        
        # Generate recommendations
        quality_report['recommendations'] = self._generate_data_recommendations(quality_report)
        
        # Store report
        self.quality_reports[datetime.now().isoformat()] = quality_report
        
        return quality_report
    
    def _analyze_data_overview(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic data characteristics"""
        try:
            # Basic statistics
            total_rows = len(data)
            total_columns = len(data.columns)
            
            # Date range analysis
            if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                date_range = {
                    'start_date': str(data.index.min()),
                    'end_date': str(data.index.max()),
                    'total_days': (data.index.max() - data.index.min()).days if len(data) > 0 else 0
                }
            else:
                date_range = {
                    'start_date': None,
                    'end_date': None,
                    'total_days': 0
                }
            
            # Data types
            data_types = data.dtypes.astype(str).to_dict()
            
            # Memory usage
            memory_usage = data.memory_usage(deep=True).sum() / (1024**2)  # MB
            
            return {
                'total_rows': total_rows,
                'total_columns': total_columns,
                'date_range': date_range,
                'data_types': data_types,
                'memory_usage_mb': round(memory_usage, 2),
                'sufficient_data': total_rows >= self.min_data_points
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        try:
            total_cells = data.size
            missing_counts = data.isnull().sum()
            missing_percentages = (missing_counts / len(data)) * 100
            
            # Overall missing data statistics
            total_missing = data.isnull().sum().sum()
            missing_ratio = total_missing / total_cells
            
            # Missing data by column
            column_analysis = {}
            for col in data.columns:
                missing_count = missing_counts[col]
                missing_pct = missing_percentages[col]
                
                column_analysis[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_pct),
                    'acceptable': missing_pct <= (self.missing_data_threshold * 100)
                }
            
            # Missing data patterns
            patterns = self._identify_missing_patterns(data)
            
            return {
                'total_missing_cells': int(total_missing),
                'missing_ratio': float(missing_ratio),
                'acceptable_overall': missing_ratio <= self.missing_data_threshold,
                'column_analysis': column_analysis,
                'missing_patterns': patterns,
                'columns_with_issues': [col for col, analysis in column_analysis.items() 
                                      if not analysis['acceptable']]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _identify_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns in missing data"""
        try:
            patterns = {}
            
            # Check for consecutive missing values
            for col in data.columns:
                if data[col].isnull().any():
                    null_series = data[col].isnull()
                    consecutive_nulls = []
                    current_streak = 0
                    
                    for is_null in null_series:
                        if is_null:
                            current_streak += 1
                        else:
                            if current_streak > 0:
                                consecutive_nulls.append(current_streak)
                            current_streak = 0
                    
                    if current_streak > 0:
                        consecutive_nulls.append(current_streak)
                    
                    patterns[col] = {
                        'max_consecutive_missing': max(consecutive_nulls) if consecutive_nulls else 0,
                        'avg_consecutive_missing': np.mean(consecutive_nulls) if consecutive_nulls else 0,
                        'total_gaps': len(consecutive_nulls)
                    }
            
            return patterns
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_outliers(self, data: pd.DataFrame, 
                        numeric_columns: List[str]) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        try:
            outlier_analysis = {}
            
            for col in numeric_columns:
                if col not in data.columns:
                    continue
                    
                column_data = data[col].dropna()
                if len(column_data) == 0:
                    continue
                
                # Z-score method
                z_scores = np.abs((column_data - column_data.mean()) / column_data.std())
                z_outliers = column_data[z_scores > self.outlier_threshold]
                
                # IQR method
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
                
                # Modified Z-score (using median)
                median = column_data.median()
                mad = np.median(np.abs(column_data - median))
                modified_z_scores = 0.6745 * (column_data - median) / mad if mad != 0 else np.zeros_like(column_data)
                modified_z_outliers = column_data[np.abs(modified_z_scores) > 3.5]
                
                outlier_analysis[col] = {
                    'z_score_outliers': {
                        'count': len(z_outliers),
                        'percentage': (len(z_outliers) / len(column_data)) * 100,
                        'values': z_outliers.tolist()[:10]  # Top 10 outliers
                    },
                    'iqr_outliers': {
                        'count': len(iqr_outliers),
                        'percentage': (len(iqr_outliers) / len(column_data)) * 100,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    },
                    'modified_z_outliers': {
                        'count': len(modified_z_outliers),
                        'percentage': (len(modified_z_outliers) / len(column_data)) * 100
                    },
                    'statistics': {
                        'mean': float(column_data.mean()),
                        'median': float(column_data.median()),
                        'std': float(column_data.std()),
                        'min': float(column_data.min()),
                        'max': float(column_data.max())
                    }
                }
            
            return outlier_analysis
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_consistency_checks(self, data: pd.DataFrame, 
                                  price_columns: List[str]) -> Dict[str, Any]:
        """Perform data consistency checks"""
        try:
            consistency_issues = []
            consistency_results = {}
            
            # Price consistency checks
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                # High should be >= Open, Low, Close
                high_violations = ((data['High'] < data['Open']) | 
                                 (data['High'] < data['Low']) | 
                                 (data['High'] < data['Close'])).sum()
                
                # Low should be <= Open, High, Close
                low_violations = ((data['Low'] > data['Open']) | 
                                (data['Low'] > data['High']) | 
                                (data['Low'] > data['Close'])).sum()
                
                consistency_results['price_consistency'] = {
                    'high_violations': int(high_violations),
                    'low_violations': int(low_violations),
                    'total_violations': int(high_violations + low_violations),
                    'violation_percentage': ((high_violations + low_violations) / len(data)) * 100
                }
                
                if high_violations > 0:
                    consistency_issues.append(f"High price violations: {high_violations}")
                if low_violations > 0:
                    consistency_issues.append(f"Low price violations: {low_violations}")
            
            # Volume consistency (should be non-negative)
            if 'Volume' in data.columns:
                negative_volume = (data['Volume'] < 0).sum()
                zero_volume = (data['Volume'] == 0).sum()
                
                consistency_results['volume_consistency'] = {
                    'negative_volume_count': int(negative_volume),
                    'zero_volume_count': int(zero_volume),
                    'negative_volume_percentage': (negative_volume / len(data)) * 100,
                    'zero_volume_percentage': (zero_volume / len(data)) * 100
                }
                
                if negative_volume > 0:
                    consistency_issues.append(f"Negative volume entries: {negative_volume}")
            
            # Date consistency (chronological order)
            if hasattr(data.index, 'to_series'):
                date_series = data.index.to_series()
                non_chronological = (date_series.diff() < timedelta(0)).sum()
                
                consistency_results['date_consistency'] = {
                    'non_chronological_count': int(non_chronological),
                    'non_chronological_percentage': (non_chronological / len(data)) * 100
                }
                
                if non_chronological > 0:
                    consistency_issues.append(f"Non-chronological dates: {non_chronological}")
            
            # Duplicate detection
            duplicate_rows = data.duplicated().sum()
            duplicate_index = data.index.duplicated().sum() if hasattr(data.index, 'duplicated') else 0
            
            consistency_results['duplicates'] = {
                'duplicate_rows': int(duplicate_rows),
                'duplicate_index': int(duplicate_index),
                'duplicate_percentage': (duplicate_rows / len(data)) * 100
            }
            
            if duplicate_rows > 0:
                consistency_issues.append(f"Duplicate rows: {duplicate_rows}")
            if duplicate_index > 0:
                consistency_issues.append(f"Duplicate index values: {duplicate_index}")
            
            consistency_results['issues'] = consistency_issues
            consistency_results['has_issues'] = len(consistency_issues) > 0
            
            return consistency_results
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_statistical_anomalies(self, data: pd.DataFrame, 
                                     price_columns: List[str]) -> Dict[str, Any]:
        """Detect statistical anomalies in the data"""
        try:
            anomaly_analysis = {}
            
            for col in price_columns:
                if col not in data.columns:
                    continue
                    
                column_data = data[col].dropna()
                if len(column_data) < 10:
                    continue
                
                # Calculate returns for price columns
                returns = column_data.pct_change().dropna()
                
                # Statistical tests
                anomalies = {}
                
                # Extreme returns (> 20% in one period)
                extreme_returns = returns[abs(returns) > 0.20]
                anomalies['extreme_returns'] = {
                    'count': len(extreme_returns),
                    'percentage': (len(extreme_returns) / len(returns)) * 100,
                    'max_positive': float(returns.max()) if len(returns) > 0 else 0,
                    'max_negative': float(returns.min()) if len(returns) > 0 else 0
                }
                
                # Consecutive extreme moves
                extreme_mask = abs(returns) > 0.10  # 10% moves
                consecutive_extremes = self._find_consecutive_occurrences(extreme_mask)
                anomalies['consecutive_extremes'] = {
                    'max_consecutive': max(consecutive_extremes) if consecutive_extremes else 0,
                    'occurrences': len(consecutive_extremes)
                }
                
                # Volatility clustering detection
                rolling_vol = returns.rolling(window=20).std()
                vol_spikes = rolling_vol > (rolling_vol.mean() + 2 * rolling_vol.std())
                anomalies['volatility_spikes'] = {
                    'count': int(vol_spikes.sum()),
                    'percentage': (vol_spikes.sum() / len(rolling_vol)) * 100
                }
                
                # Skewness and kurtosis
                if len(returns) > 30:
                    skewness = returns.skew()
                    kurtosis = returns.kurtosis()
                    
                    anomalies['distribution_characteristics'] = {
                        'skewness': float(skewness),
                        'kurtosis': float(kurtosis),
                        'is_highly_skewed': abs(skewness) > 2,
                        'has_fat_tails': kurtosis > 3
                    }
                
                anomaly_analysis[col] = anomalies
            
            return anomaly_analysis
        except Exception as e:
            return {'error': str(e)}
    
    def _find_consecutive_occurrences(self, boolean_series: pd.Series) -> List[int]:
        """Find lengths of consecutive True occurrences"""
        consecutive_lengths = []
        current_length = 0
        
        for value in boolean_series:
            if value:
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                current_length = 0
        
        if current_length > 0:
            consecutive_lengths.append(current_length)
        
        return consecutive_lengths
    
    def _validate_data_integrity(self, data: pd.DataFrame, 
                                price_columns: List[str],
                                volume_columns: List[str]) -> Dict[str, Any]:
        """Validate overall data integrity"""
        try:
            integrity_results = {
                'completeness_score': 0,
                'consistency_score': 0,
                'accuracy_score': 0,
                'overall_integrity_score': 0,
                'critical_issues': [],
                'warnings': []
            }
            
            # Completeness score (based on missing data)
            total_expected_values = data.size
            total_missing = data.isnull().sum().sum()
            completeness_score = ((total_expected_values - total_missing) / total_expected_values) * 100
            integrity_results['completeness_score'] = float(completeness_score)
            
            if completeness_score < 95:
                integrity_results['critical_issues'].append(f"Low data completeness: {completeness_score:.1f}%")
            
            # Consistency score (based on logical constraints)
            consistency_violations = 0
            total_checks = 0
            
            # Price consistency
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                high_violations = ((data['High'] < data['Open']) | 
                                 (data['High'] < data['Low']) | 
                                 (data['High'] < data['Close'])).sum()
                low_violations = ((data['Low'] > data['Open']) | 
                                (data['Low'] > data['High']) | 
                                (data['Low'] > data['Close'])).sum()
                consistency_violations += high_violations + low_violations
                total_checks += len(data) * 2
            
            # Volume consistency
            if 'Volume' in data.columns:
                negative_volume = (data['Volume'] < 0).sum()
                consistency_violations += negative_volume
                total_checks += len(data)
            
            if total_checks > 0:
                consistency_score = ((total_checks - consistency_violations) / total_checks) * 100
                integrity_results['consistency_score'] = float(consistency_score)
                
                if consistency_score < 99:
                    integrity_results['critical_issues'].append(f"Data consistency issues: {consistency_violations} violations")
            else:
                integrity_results['consistency_score'] = 100.0
            
            # Accuracy score (based on outliers and anomalies)
            total_outliers = 0
            total_data_points = 0
            
            for col in price_columns + volume_columns:
                if col in data.columns:
                    column_data = data[col].dropna()
                    if len(column_data) > 0:
                        z_scores = np.abs((column_data - column_data.mean()) / column_data.std())
                        outliers = (z_scores > self.outlier_threshold).sum()
                        total_outliers += outliers
                        total_data_points += len(column_data)
            
            if total_data_points > 0:
                accuracy_score = ((total_data_points - total_outliers) / total_data_points) * 100
                integrity_results['accuracy_score'] = float(accuracy_score)
                
                if accuracy_score < 95:
                    integrity_results['warnings'].append(f"High number of outliers detected: {total_outliers}")
            else:
                integrity_results['accuracy_score'] = 100.0
            
            # Overall integrity score
            scores = [integrity_results['completeness_score'], 
                     integrity_results['consistency_score'], 
                     integrity_results['accuracy_score']]
            integrity_results['overall_integrity_score'] = float(np.mean([s for s in scores if s > 0]))
            
            return integrity_results
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        try:
            scores = []
            weights = []
            
            # Data completeness (30% weight)
            missing_analysis = quality_report.get('missing_data_analysis', {})
            if 'missing_ratio' in missing_analysis:
                completeness_score = (1 - missing_analysis['missing_ratio']) * 100
                scores.append(completeness_score)
                weights.append(0.3)
            
            # Data consistency (25% weight)
            consistency_analysis = quality_report.get('consistency_checks', {})
            if not consistency_analysis.get('has_issues', True):
                consistency_score = 100
            else:
                # Reduce score based on number of issues
                issues_count = len(consistency_analysis.get('issues', []))
                consistency_score = max(0, 100 - (issues_count * 10))
            scores.append(consistency_score)
            weights.append(0.25)
            
            # Outlier analysis (20% weight)
            outlier_analysis = quality_report.get('outlier_analysis', {})
            total_outliers = 0
            total_points = 0
            for col_analysis in outlier_analysis.values():
                if isinstance(col_analysis, dict) and 'z_score_outliers' in col_analysis:
                    total_outliers += col_analysis['z_score_outliers']['count']
                    # Estimate total points (this is approximate)
                    total_points += 1000  # Assume 1000 points per column
            
            if total_points > 0:
                outlier_score = max(0, 100 - (total_outliers / total_points * 1000))
            else:
                outlier_score = 100
            scores.append(outlier_score)
            weights.append(0.2)
            
            # Data integrity (25% weight)
            integrity_analysis = quality_report.get('data_integrity', {})
            integrity_score = integrity_analysis.get('overall_integrity_score', 100)
            scores.append(integrity_score)
            weights.append(0.25)
            
            # Calculate weighted average
            if scores and weights:
                total_weight = sum(weights)
                weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
                return round(weighted_score, 1)
            else:
                return 50.0  # Default neutral score
                
        except Exception as e:
            return 50.0  # Default neutral score on error
    
    def _generate_data_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        try:
            # Missing data recommendations
            missing_analysis = quality_report.get('missing_data_analysis', {})
            if missing_analysis.get('missing_ratio', 0) > self.missing_data_threshold:
                recommendations.append(f"HIGH MISSING DATA: {missing_analysis['missing_ratio']:.1%} of data is missing - consider data imputation or source improvement")
            
            columns_with_issues = missing_analysis.get('columns_with_issues', [])
            if columns_with_issues:
                recommendations.append(f"COLUMNS WITH MISSING DATA: {', '.join(columns_with_issues)} - review data collection for these fields")
            
            # Consistency recommendations
            consistency_analysis = quality_report.get('consistency_checks', {})
            if consistency_analysis.get('has_issues', False):
                issues = consistency_analysis.get('issues', [])
                for issue in issues:
                    recommendations.append(f"CONSISTENCY ISSUE: {issue}")
            
            # Outlier recommendations
            outlier_analysis = quality_report.get('outlier_analysis', {})
            for col, analysis in outlier_analysis.items():
                if isinstance(analysis, dict) and 'z_score_outliers' in analysis:
                    outlier_pct = analysis['z_score_outliers']['percentage']
                    if outlier_pct > 5:  # More than 5% outliers
                        recommendations.append(f"HIGH OUTLIERS IN {col}: {outlier_pct:.1f}% of values are outliers - investigate data source")
            
            # Integrity recommendations
            integrity_analysis = quality_report.get('data_integrity', {})
            overall_score = integrity_analysis.get('overall_integrity_score', 100)
            if overall_score < 80:
                recommendations.append(f"LOW DATA INTEGRITY: Overall score {overall_score:.1f}% - comprehensive data review required")
            
            critical_issues = integrity_analysis.get('critical_issues', [])
            for issue in critical_issues:
                recommendations.append(f"CRITICAL: {issue}")
            
            # General recommendations
            quality_score = quality_report.get('quality_score', 100)
            if quality_score < 70:
                recommendations.append("OVERALL DATA QUALITY POOR: Consider alternative data sources or extensive preprocessing")
            elif quality_score < 85:
                recommendations.append("MODERATE DATA QUALITY: Implement data validation and monitoring procedures")
            elif quality_score >= 95:
                recommendations.append("EXCELLENT DATA QUALITY: Data is suitable for institutional-grade analysis")
            
            if not recommendations:
                recommendations.append("DATA QUALITY ACCEPTABLE: No major issues detected")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def generate_quality_report_text(self, quality_report: Dict[str, Any]) -> str:
        """Generate text summary of quality report"""
        report = []
        report.append("=" * 80)
        report.append("DATA QUALITY ASSURANCE REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {quality_report.get('analysis_timestamp', 'Unknown')}")
        report.append(f"Overall Quality Score: {quality_report.get('quality_score', 0):.1f}/100")
        report.append("")
        
        # Data Overview
        overview = quality_report.get('data_overview', {})
        report.append("DATA OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total Rows: {overview.get('total_rows', 'Unknown')}")
        report.append(f"Total Columns: {overview.get('total_columns', 'Unknown')}")
        report.append(f"Memory Usage: {overview.get('memory_usage_mb', 0):.2f} MB")
        report.append(f"Sufficient Data: {'Yes' if overview.get('sufficient_data', False) else 'No'}")
        report.append("")
        
        # Missing Data Analysis
        missing = quality_report.get('missing_data_analysis', {})
        report.append("MISSING DATA ANALYSIS")
        report.append("-" * 40)
        report.append(f"Missing Ratio: {missing.get('missing_ratio', 0):.2%}")
        report.append(f"Acceptable: {'Yes' if missing.get('acceptable_overall', False) else 'No'}")
        
        columns_with_issues = missing.get('columns_with_issues', [])
        if columns_with_issues:
            report.append(f"Problematic Columns: {', '.join(columns_with_issues)}")
        report.append("")
        
        # Consistency Checks
        consistency = quality_report.get('consistency_checks', {})
        report.append("CONSISTENCY CHECKS")
        report.append("-" * 40)
        report.append(f"Has Issues: {'Yes' if consistency.get('has_issues', False) else 'No'}")
        
        issues = consistency.get('issues', [])
        if issues:
            for issue in issues:
                report.append(f"  • {issue}")
        report.append("")
        
        # Data Integrity
        integrity = quality_report.get('data_integrity', {})
        report.append("DATA INTEGRITY")
        report.append("-" * 40)
        report.append(f"Completeness Score: {integrity.get('completeness_score', 0):.1f}%")
        report.append(f"Consistency Score: {integrity.get('consistency_score', 0):.1f}%")
        report.append(f"Accuracy Score: {integrity.get('accuracy_score', 0):.1f}%")
        report.append("")
        
        # Recommendations
        recommendations = quality_report.get('recommendations', [])
        if recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)