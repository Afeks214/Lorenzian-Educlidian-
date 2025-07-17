"""
Performance Report Comparison and Trending System
Advanced analysis of test performance trends and comparative reporting
"""

import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import statistics
import logging
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from .advanced_test_reporting import TestResult, TestSuite, TestStatus
from .coverage_analyzer import CoverageReport


class TrendDirection(Enum):
    """Trend directions"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


class PerformanceMetric(Enum):
    """Performance metrics to track"""
    EXECUTION_TIME = "execution_time"
    SUCCESS_RATE = "success_rate"
    COVERAGE = "coverage"
    TEST_COUNT = "test_count"
    FAILURE_RATE = "failure_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"


@dataclass
class PerformanceBenchmark:
    """Performance benchmark data"""
    metric_name: str
    baseline_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    improvement_goal: float
    measurement_unit: str
    higher_is_better: bool = True


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    trend_direction: TrendDirection
    slope: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    forecast_values: List[float]
    forecast_dates: List[datetime]
    anomalies: List[Dict[str, Any]]
    volatility_score: float
    improvement_rate: float


@dataclass
class PerformanceComparison:
    """Performance comparison results"""
    baseline_period: Tuple[datetime, datetime]
    comparison_period: Tuple[datetime, datetime]
    metrics_comparison: Dict[str, Dict[str, Any]]
    statistical_significance: Dict[str, Dict[str, Any]]
    performance_regression: List[Dict[str, Any]]
    performance_improvement: List[Dict[str, Any]]
    overall_assessment: str


class PerformanceTrendAnalyzer:
    """Advanced performance trend analysis and comparison"""
    
    def __init__(self, db_path: str = "performance_trends.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
        # Default benchmarks
        self.benchmarks = self._create_default_benchmarks()
        
        # Analysis configuration
        self.config = {
            'min_data_points': 5,
            'forecast_periods': 10,
            'anomaly_threshold': 2.0,  # Standard deviations
            'volatility_window': 10,
            'significance_level': 0.05,
            'trend_window': 30  # days
        }
    
    def _init_database(self):
        """Initialize database for performance tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suite_name TEXT,
                execution_date TIMESTAMP,
                metric_name TEXT,
                metric_value REAL,
                metric_unit TEXT,
                context_data TEXT,
                environment_info TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT UNIQUE,
                baseline_value REAL,
                target_value REAL,
                threshold_warning REAL,
                threshold_critical REAL,
                improvement_goal REAL,
                measurement_unit TEXT,
                higher_is_better BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trend_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT UNIQUE,
                suite_name TEXT,
                metric_name TEXT,
                analysis_period_start TIMESTAMP,
                analysis_period_end TIMESTAMP,
                trend_direction TEXT,
                slope REAL,
                r_squared REAL,
                p_value REAL,
                confidence_interval TEXT,
                forecast_data TEXT,
                anomalies TEXT,
                volatility_score REAL,
                improvement_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comparison_id TEXT UNIQUE,
                suite_name TEXT,
                baseline_start TIMESTAMP,
                baseline_end TIMESTAMP,
                comparison_start TIMESTAMP,
                comparison_end TIMESTAMP,
                metrics_comparison TEXT,
                statistical_significance TEXT,
                performance_regression TEXT,
                performance_improvement TEXT,
                overall_assessment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_suite_date ON performance_history(suite_name, execution_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_metric ON performance_history(metric_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trend_suite ON trend_analyses(suite_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_comparison_suite ON performance_comparisons(suite_name)')
        
        conn.commit()
        conn.close()
    
    def _create_default_benchmarks(self) -> Dict[str, PerformanceBenchmark]:
        """Create default performance benchmarks"""
        
        benchmarks = {
            'execution_time': PerformanceBenchmark(
                metric_name='execution_time',
                baseline_value=120.0,  # 2 minutes
                target_value=60.0,     # 1 minute
                threshold_warning=180.0,  # 3 minutes
                threshold_critical=300.0,  # 5 minutes
                improvement_goal=0.1,   # 10% improvement
                measurement_unit='seconds',
                higher_is_better=False
            ),
            'success_rate': PerformanceBenchmark(
                metric_name='success_rate',
                baseline_value=85.0,
                target_value=95.0,
                threshold_warning=80.0,
                threshold_critical=70.0,
                improvement_goal=0.05,  # 5% improvement
                measurement_unit='percentage',
                higher_is_better=True
            ),
            'coverage': PerformanceBenchmark(
                metric_name='coverage',
                baseline_value=70.0,
                target_value=85.0,
                threshold_warning=65.0,
                threshold_critical=50.0,
                improvement_goal=0.1,   # 10% improvement
                measurement_unit='percentage',
                higher_is_better=True
            ),
            'test_count': PerformanceBenchmark(
                metric_name='test_count',
                baseline_value=100,
                target_value=200,
                threshold_warning=50,
                threshold_critical=25,
                improvement_goal=0.2,   # 20% improvement
                measurement_unit='count',
                higher_is_better=True
            ),
            'failure_rate': PerformanceBenchmark(
                metric_name='failure_rate',
                baseline_value=15.0,
                target_value=5.0,
                threshold_warning=20.0,
                threshold_critical=30.0,
                improvement_goal=0.1,   # 10% improvement
                measurement_unit='percentage',
                higher_is_better=False
            )
        }
        
        return benchmarks
    
    async def record_performance_metrics(self, 
                                       suite: TestSuite,
                                       coverage_report: Optional[CoverageReport] = None,
                                       environment_info: Optional[Dict[str, Any]] = None):
        """Record performance metrics from test execution"""
        
        execution_date = suite.end_time
        
        # Extract metrics from test suite
        metrics = {
            'execution_time': suite.total_duration,
            'success_rate': suite.success_rate,
            'test_count': suite.total_tests,
            'failure_rate': suite.failure_rate,
            'average_test_duration': statistics.mean(r.duration for r in suite.results) if suite.results else 0,
            'max_test_duration': max(r.duration for r in suite.results) if suite.results else 0,
            'throughput': suite.total_tests / suite.total_duration if suite.total_duration > 0 else 0
        }
        
        # Add coverage metrics
        if coverage_report:
            metrics.update({
                'coverage': coverage_report.overall_coverage,
                'line_coverage': coverage_report.line_coverage,
                'branch_coverage': coverage_report.branch_coverage,
                'function_coverage': coverage_report.function_coverage
            })
        
        # Add memory and CPU metrics if available
        if suite.results:
            memory_usage = [r.memory_usage for r in suite.results if r.memory_usage]
            cpu_usage = [r.cpu_usage for r in suite.results if r.cpu_usage]
            
            if memory_usage:
                metrics.update({
                    'avg_memory_usage': statistics.mean(memory_usage),
                    'max_memory_usage': max(memory_usage)
                })
            
            if cpu_usage:
                metrics.update({
                    'avg_cpu_usage': statistics.mean(cpu_usage),
                    'max_cpu_usage': max(cpu_usage)
                })
        
        # Store metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                context_data = {
                    'passed': suite.passed,
                    'failed': suite.failed,
                    'skipped': suite.skipped,
                    'errors': suite.errors
                }
                
                cursor.execute('''
                    INSERT INTO performance_history 
                    (suite_name, execution_date, metric_name, metric_value, 
                     metric_unit, context_data, environment_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    suite.suite_name,
                    execution_date,
                    metric_name,
                    metric_value,
                    self._get_metric_unit(metric_name),
                    json.dumps(context_data),
                    json.dumps(environment_info or {})
                ))
        
        conn.commit()
        conn.close()
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric"""
        unit_map = {
            'execution_time': 'seconds',
            'success_rate': 'percentage',
            'coverage': 'percentage',
            'test_count': 'count',
            'failure_rate': 'percentage',
            'average_test_duration': 'seconds',
            'max_test_duration': 'seconds',
            'throughput': 'tests/second',
            'avg_memory_usage': 'MB',
            'max_memory_usage': 'MB',
            'avg_cpu_usage': 'percentage',
            'max_cpu_usage': 'percentage'
        }
        
        return unit_map.get(metric_name, 'units')
    
    async def analyze_trends(self, 
                           suite_name: str,
                           metric_names: List[str],
                           days: int = 30) -> Dict[str, TrendAnalysis]:
        """Analyze trends for specified metrics"""
        
        conn = sqlite3.connect(self.db_path)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        trend_analyses = {}
        
        for metric_name in metric_names:
            # Get historical data
            query = '''
                SELECT execution_date, metric_value
                FROM performance_history
                WHERE suite_name = ? AND metric_name = ?
                AND execution_date BETWEEN ? AND ?
                ORDER BY execution_date
            '''
            
            df = pd.read_sql_query(query, conn, params=(suite_name, metric_name, start_date, end_date))
            
            if len(df) < self.config['min_data_points']:
                self.logger.warning(f"Insufficient data for trend analysis of {metric_name}: {len(df)} points")
                continue
            
            # Perform trend analysis
            trend_analysis = await self._perform_trend_analysis(df, metric_name, start_date, end_date)
            trend_analyses[metric_name] = trend_analysis
            
            # Store analysis results
            await self._store_trend_analysis(suite_name, trend_analysis)
        
        conn.close()
        
        return trend_analyses
    
    async def _perform_trend_analysis(self, 
                                    df: pd.DataFrame,
                                    metric_name: str,
                                    start_date: datetime,
                                    end_date: datetime) -> TrendAnalysis:
        """Perform detailed trend analysis"""
        
        # Convert dates to numeric for regression
        df['date_numeric'] = pd.to_datetime(df['execution_date']).astype(int) // 10**9
        
        X = df[['date_numeric']].values
        y = df['metric_value'].values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        slope = model.coef_[0]
        r_squared = r2_score(y, y_pred)
        
        # Statistical significance test
        n = len(df)
        se = np.sqrt(mean_squared_error(y, y_pred) / (n - 2))
        t_stat = slope / (se / np.sqrt(np.sum((X[:, 0] - np.mean(X[:, 0]))**2)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Confidence interval
        confidence_interval = self._calculate_confidence_interval(slope, se, n)
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(slope, p_value, r_squared)
        
        # Generate forecast
        forecast_values, forecast_dates = self._generate_forecast(
            model, df, self.config['forecast_periods']
        )
        
        # Detect anomalies
        anomalies = self._detect_anomalies(df, y_pred)
        
        # Calculate volatility
        volatility_score = self._calculate_volatility(df['metric_value'].values)
        
        # Calculate improvement rate
        improvement_rate = self._calculate_improvement_rate(
            df['metric_value'].values, metric_name
        )
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            slope=slope,
            r_squared=r_squared,
            p_value=p_value,
            confidence_interval=confidence_interval,
            forecast_values=forecast_values,
            forecast_dates=forecast_dates,
            anomalies=anomalies,
            volatility_score=volatility_score,
            improvement_rate=improvement_rate
        )
    
    def _calculate_confidence_interval(self, slope: float, se: float, n: int) -> Tuple[float, float]:
        """Calculate confidence interval for trend slope"""
        
        t_critical = stats.t.ppf(1 - self.config['significance_level'] / 2, n - 2)
        margin_error = t_critical * se
        
        return (slope - margin_error, slope + margin_error)
    
    def _determine_trend_direction(self, slope: float, p_value: float, r_squared: float) -> TrendDirection:
        """Determine trend direction based on statistical analysis"""
        
        if p_value > self.config['significance_level'] or r_squared < 0.1:
            return TrendDirection.STABLE
        
        if r_squared < 0.3:  # Low correlation indicates volatility
            return TrendDirection.VOLATILE
        
        if slope > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DECLINING
    
    def _generate_forecast(self, 
                          model: LinearRegression,
                          df: pd.DataFrame,
                          periods: int) -> Tuple[List[float], List[datetime]]:
        """Generate forecast values"""
        
        last_date = pd.to_datetime(df['execution_date'].iloc[-1])
        
        # Generate future dates (assuming daily frequency)
        future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
        future_numeric = [date.timestamp() for date in future_dates]
        
        # Predict future values
        X_future = np.array(future_numeric).reshape(-1, 1)
        forecast_values = model.predict(X_future).tolist()
        
        return forecast_values, future_dates
    
    def _detect_anomalies(self, df: pd.DataFrame, y_pred: np.ndarray) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        
        residuals = df['metric_value'].values - y_pred
        threshold = self.config['anomaly_threshold'] * np.std(residuals)
        
        anomalies = []
        
        for i, residual in enumerate(residuals):
            if abs(residual) > threshold:
                anomalies.append({
                    'date': df.iloc[i]['execution_date'],
                    'actual_value': df.iloc[i]['metric_value'],
                    'predicted_value': y_pred[i],
                    'residual': residual,
                    'severity': 'high' if abs(residual) > threshold * 1.5 else 'medium'
                })
        
        return anomalies
    
    def _calculate_volatility(self, values: np.ndarray) -> float:
        """Calculate volatility score"""
        
        if len(values) < 2:
            return 0.0
        
        # Calculate rolling standard deviation
        if len(values) > self.config['volatility_window']:
            window_size = self.config['volatility_window']
            rolling_std = pd.Series(values).rolling(window=window_size).std()
            volatility = rolling_std.mean()
        else:
            volatility = np.std(values)
        
        # Normalize by mean to get coefficient of variation
        mean_value = np.mean(values)
        if mean_value != 0:
            return volatility / abs(mean_value)
        else:
            return 0.0
    
    def _calculate_improvement_rate(self, values: np.ndarray, metric_name: str) -> float:
        """Calculate improvement rate"""
        
        if len(values) < 2:
            return 0.0
        
        # Get benchmark
        benchmark = self.benchmarks.get(metric_name)
        if not benchmark:
            return 0.0
        
        # Calculate improvement rate
        first_value = values[0]
        last_value = values[-1]
        
        if benchmark.higher_is_better:
            improvement = (last_value - first_value) / first_value if first_value != 0 else 0
        else:
            improvement = (first_value - last_value) / first_value if first_value != 0 else 0
        
        return improvement
    
    async def _store_trend_analysis(self, suite_name: str, analysis: TrendAnalysis):
        """Store trend analysis results"""
        
        analysis_id = f"TREND_{suite_name}_{analysis.metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trend_analyses 
            (analysis_id, suite_name, metric_name, analysis_period_start, analysis_period_end,
             trend_direction, slope, r_squared, p_value, confidence_interval,
             forecast_data, anomalies, volatility_score, improvement_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id,
            suite_name,
            analysis.metric_name,
            datetime.now() - timedelta(days=30),  # Assuming 30-day analysis
            datetime.now(),
            analysis.trend_direction.value,
            analysis.slope,
            analysis.r_squared,
            analysis.p_value,
            json.dumps(analysis.confidence_interval),
            json.dumps({
                'values': analysis.forecast_values,
                'dates': [d.isoformat() for d in analysis.forecast_dates]
            }),
            json.dumps(analysis.anomalies, default=str),
            analysis.volatility_score,
            analysis.improvement_rate
        ))
        
        conn.commit()
        conn.close()
    
    async def compare_performance_periods(self,
                                        suite_name: str,
                                        baseline_start: datetime,
                                        baseline_end: datetime,
                                        comparison_start: datetime,
                                        comparison_end: datetime) -> PerformanceComparison:
        """Compare performance between two time periods"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get baseline metrics
        baseline_query = '''
            SELECT metric_name, AVG(metric_value) as avg_value, 
                   STDDEV(metric_value) as std_value, COUNT(*) as count
            FROM performance_history
            WHERE suite_name = ? AND execution_date BETWEEN ? AND ?
            GROUP BY metric_name
        '''
        
        baseline_df = pd.read_sql_query(
            baseline_query, conn, 
            params=(suite_name, baseline_start, baseline_end)
        )
        
        # Get comparison metrics
        comparison_df = pd.read_sql_query(
            baseline_query, conn,
            params=(suite_name, comparison_start, comparison_end)
        )
        
        conn.close()
        
        # Perform comparison analysis
        metrics_comparison = {}
        statistical_significance = {}
        performance_regression = []
        performance_improvement = []
        
        for metric_name in set(baseline_df['metric_name']).intersection(set(comparison_df['metric_name'])):
            baseline_data = baseline_df[baseline_df['metric_name'] == metric_name].iloc[0]
            comparison_data = comparison_df[comparison_df['metric_name'] == metric_name].iloc[0]
            
            # Calculate comparison metrics
            comparison_result = self._compare_metric_values(
                baseline_data, comparison_data, metric_name
            )
            
            metrics_comparison[metric_name] = comparison_result
            
            # Statistical significance test
            significance_result = await self._test_statistical_significance(
                suite_name, metric_name, baseline_start, baseline_end,
                comparison_start, comparison_end
            )
            
            statistical_significance[metric_name] = significance_result
            
            # Identify regressions and improvements
            if comparison_result['change_percent'] < -5:  # 5% threshold
                performance_regression.append({
                    'metric_name': metric_name,
                    'change_percent': comparison_result['change_percent'],
                    'significance': significance_result.get('p_value', 1.0)
                })
            elif comparison_result['change_percent'] > 5:
                performance_improvement.append({
                    'metric_name': metric_name,
                    'change_percent': comparison_result['change_percent'],
                    'significance': significance_result.get('p_value', 1.0)
                })
        
        # Overall assessment
        overall_assessment = self._assess_overall_performance(
            performance_regression, performance_improvement
        )
        
        # Create and store comparison
        comparison = PerformanceComparison(
            baseline_period=(baseline_start, baseline_end),
            comparison_period=(comparison_start, comparison_end),
            metrics_comparison=metrics_comparison,
            statistical_significance=statistical_significance,
            performance_regression=performance_regression,
            performance_improvement=performance_improvement,
            overall_assessment=overall_assessment
        )
        
        await self._store_performance_comparison(suite_name, comparison)
        
        return comparison
    
    def _compare_metric_values(self, 
                              baseline_data: pd.Series,
                              comparison_data: pd.Series,
                              metric_name: str) -> Dict[str, Any]:
        """Compare two metric values"""
        
        baseline_value = baseline_data['avg_value']
        comparison_value = comparison_data['avg_value']
        
        # Calculate change
        if baseline_value != 0:
            change_percent = ((comparison_value - baseline_value) / baseline_value) * 100
        else:
            change_percent = 0.0
        
        # Determine if change is positive or negative based on metric type
        benchmark = self.benchmarks.get(metric_name)
        if benchmark and not benchmark.higher_is_better:
            change_percent = -change_percent  # Reverse for metrics where lower is better
        
        return {
            'baseline_value': baseline_value,
            'comparison_value': comparison_value,
            'change_absolute': comparison_value - baseline_value,
            'change_percent': change_percent,
            'baseline_std': baseline_data['std_value'],
            'comparison_std': comparison_data['std_value'],
            'baseline_count': baseline_data['count'],
            'comparison_count': comparison_data['count'],
            'assessment': self._assess_change(change_percent, metric_name)
        }
    
    def _assess_change(self, change_percent: float, metric_name: str) -> str:
        """Assess the significance of a change"""
        
        if abs(change_percent) < 2:
            return "minimal"
        elif abs(change_percent) < 5:
            return "small"
        elif abs(change_percent) < 15:
            return "moderate"
        elif abs(change_percent) < 30:
            return "large"
        else:
            return "very_large"
    
    async def _test_statistical_significance(self,
                                           suite_name: str,
                                           metric_name: str,
                                           baseline_start: datetime,
                                           baseline_end: datetime,
                                           comparison_start: datetime,
                                           comparison_end: datetime) -> Dict[str, Any]:
        """Test statistical significance of difference between periods"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get raw data for both periods
        baseline_query = '''
            SELECT metric_value FROM performance_history
            WHERE suite_name = ? AND metric_name = ? AND execution_date BETWEEN ? AND ?
        '''
        
        baseline_values = pd.read_sql_query(
            baseline_query, conn,
            params=(suite_name, metric_name, baseline_start, baseline_end)
        )['metric_value'].values
        
        comparison_values = pd.read_sql_query(
            baseline_query, conn,
            params=(suite_name, metric_name, comparison_start, comparison_end)
        )['metric_value'].values
        
        conn.close()
        
        if len(baseline_values) < 2 or len(comparison_values) < 2:
            return {"test": "insufficient_data", "p_value": 1.0}
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(baseline_values, comparison_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(baseline_values) - 1) * np.var(baseline_values) + 
                                 (len(comparison_values) - 1) * np.var(comparison_values)) / 
                                (len(baseline_values) + len(comparison_values) - 2))
            
            if pooled_std != 0:
                effect_size = (np.mean(comparison_values) - np.mean(baseline_values)) / pooled_std
            else:
                effect_size = 0.0
            
            return {
                "test": "t_test",
                "t_statistic": t_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "significant": p_value < self.config['significance_level'],
                "interpretation": self._interpret_effect_size(effect_size)
            }
            
        except Exception as e:
            self.logger.error(f"Error in statistical test: {e}")
            return {"test": "error", "p_value": 1.0, "error": str(e)}
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size"""
        
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _assess_overall_performance(self,
                                  regressions: List[Dict[str, Any]],
                                  improvements: List[Dict[str, Any]]) -> str:
        """Assess overall performance change"""
        
        significant_regressions = [r for r in regressions if r['significance'] < 0.05]
        significant_improvements = [i for i in improvements if i['significance'] < 0.05]
        
        if len(significant_regressions) > len(significant_improvements):
            return "regression"
        elif len(significant_improvements) > len(significant_regressions):
            return "improvement"
        elif len(significant_regressions) == 0 and len(significant_improvements) == 0:
            return "stable"
        else:
            return "mixed"
    
    async def _store_performance_comparison(self, suite_name: str, comparison: PerformanceComparison):
        """Store performance comparison results"""
        
        comparison_id = f"COMP_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_comparisons 
            (comparison_id, suite_name, baseline_start, baseline_end,
             comparison_start, comparison_end, metrics_comparison,
             statistical_significance, performance_regression,
             performance_improvement, overall_assessment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            comparison_id,
            suite_name,
            comparison.baseline_period[0],
            comparison.baseline_period[1],
            comparison.comparison_period[0],
            comparison.comparison_period[1],
            json.dumps(comparison.metrics_comparison, default=str),
            json.dumps(comparison.statistical_significance, default=str),
            json.dumps(comparison.performance_regression, default=str),
            json.dumps(comparison.performance_improvement, default=str),
            comparison.overall_assessment
        ))
        
        conn.commit()
        conn.close()
    
    async def generate_performance_dashboard(self, suite_name: str) -> Dict[str, Any]:
        """Generate comprehensive performance dashboard"""
        
        # Get recent trend analyses
        recent_trends = await self.get_recent_trend_analyses(suite_name, days=30)
        
        # Get performance comparisons
        recent_comparisons = await self.get_recent_comparisons(suite_name, days=30)
        
        # Generate performance charts
        charts = await self._generate_performance_charts(suite_name)
        
        # Calculate performance scores
        performance_scores = await self._calculate_performance_scores(suite_name)
        
        # Generate recommendations
        recommendations = await self._generate_performance_recommendations(
            suite_name, recent_trends, recent_comparisons
        )
        
        return {
            'suite_name': suite_name,
            'generated_at': datetime.now().isoformat(),
            'trend_analyses': recent_trends,
            'performance_comparisons': recent_comparisons,
            'charts': charts,
            'performance_scores': performance_scores,
            'recommendations': recommendations,
            'summary': {
                'total_metrics_tracked': len(recent_trends),
                'trending_up': len([t for t in recent_trends.values() if t['trend_direction'] == 'improving']),
                'trending_down': len([t for t in recent_trends.values() if t['trend_direction'] == 'declining']),
                'volatile_metrics': len([t for t in recent_trends.values() if t['trend_direction'] == 'volatile'])
            }
        }
    
    async def get_recent_trend_analyses(self, suite_name: str, days: int = 30) -> Dict[str, Any]:
        """Get recent trend analyses"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trend_analyses
            WHERE suite_name = ? AND created_at > datetime('now', '-{} days')
            ORDER BY created_at DESC
        '''.format(days), (suite_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to dictionary
        columns = [desc[0] for desc in cursor.description]
        analyses = {}
        
        for row in results:
            analysis_dict = dict(zip(columns, row))
            metric_name = analysis_dict['metric_name']
            
            # Parse JSON fields
            analysis_dict['confidence_interval'] = json.loads(analysis_dict['confidence_interval'])
            analysis_dict['forecast_data'] = json.loads(analysis_dict['forecast_data'])
            analysis_dict['anomalies'] = json.loads(analysis_dict['anomalies'])
            
            analyses[metric_name] = analysis_dict
        
        return analyses
    
    async def get_recent_comparisons(self, suite_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent performance comparisons"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM performance_comparisons
            WHERE suite_name = ? AND created_at > datetime('now', '-{} days')
            ORDER BY created_at DESC
        '''.format(days), (suite_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to dictionary
        columns = [desc[0] for desc in cursor.description]
        comparisons = []
        
        for row in results:
            comparison_dict = dict(zip(columns, row))
            
            # Parse JSON fields
            comparison_dict['metrics_comparison'] = json.loads(comparison_dict['metrics_comparison'])
            comparison_dict['statistical_significance'] = json.loads(comparison_dict['statistical_significance'])
            comparison_dict['performance_regression'] = json.loads(comparison_dict['performance_regression'])
            comparison_dict['performance_improvement'] = json.loads(comparison_dict['performance_improvement'])
            
            comparisons.append(comparison_dict)
        
        return comparisons
    
    async def _generate_performance_charts(self, suite_name: str) -> Dict[str, str]:
        """Generate performance visualization charts"""
        
        charts = {}
        chart_dir = Path("performance_charts")
        chart_dir.mkdir(exist_ok=True)
        
        # Get historical data
        conn = sqlite3.connect(self.db_path)
        
        # Key metrics to visualize
        key_metrics = ['execution_time', 'success_rate', 'coverage', 'test_count']
        
        for metric_name in key_metrics:
            query = '''
                SELECT execution_date, metric_value
                FROM performance_history
                WHERE suite_name = ? AND metric_name = ?
                AND execution_date > datetime('now', '-90 days')
                ORDER BY execution_date
            '''
            
            df = pd.read_sql_query(query, conn, params=(suite_name, metric_name))
            
            if not df.empty:
                # Create trend chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['execution_date'],
                    y=df['metric_value'],
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(width=2)
                ))
                
                # Add benchmark lines if available
                benchmark = self.benchmarks.get(metric_name)
                if benchmark:
                    fig.add_hline(
                        y=benchmark.target_value,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Target"
                    )
                    
                    fig.add_hline(
                        y=benchmark.threshold_warning,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="Warning"
                    )
                
                fig.update_layout(
                    title=f'{metric_name.replace("_", " ").title()} Trend',
                    xaxis_title='Date',
                    yaxis_title=self._get_metric_unit(metric_name),
                    height=400
                )
                
                chart_path = chart_dir / f"{suite_name}_{metric_name}_trend.html"
                fig.write_html(chart_path)
                charts[metric_name] = str(chart_path)
        
        conn.close()
        
        return charts
    
    async def _calculate_performance_scores(self, suite_name: str) -> Dict[str, Any]:
        """Calculate performance scores"""
        
        conn = sqlite3.connect(self.db_path)
        
        scores = {}
        
        for metric_name, benchmark in self.benchmarks.items():
            # Get latest value
            query = '''
                SELECT metric_value FROM performance_history
                WHERE suite_name = ? AND metric_name = ?
                ORDER BY execution_date DESC
                LIMIT 1
            '''
            
            cursor = conn.cursor()
            cursor.execute(query, (suite_name, metric_name))
            result = cursor.fetchone()
            
            if result:
                current_value = result[0]
                
                # Calculate score (0-100)
                if benchmark.higher_is_better:
                    if current_value >= benchmark.target_value:
                        score = 100
                    elif current_value <= benchmark.threshold_critical:
                        score = 0
                    else:
                        score = ((current_value - benchmark.threshold_critical) / 
                                (benchmark.target_value - benchmark.threshold_critical)) * 100
                else:
                    if current_value <= benchmark.target_value:
                        score = 100
                    elif current_value >= benchmark.threshold_critical:
                        score = 0
                    else:
                        score = ((benchmark.threshold_critical - current_value) / 
                                (benchmark.threshold_critical - benchmark.target_value)) * 100
                
                scores[metric_name] = {
                    'current_value': current_value,
                    'target_value': benchmark.target_value,
                    'score': max(0, min(100, score)),
                    'status': self._get_performance_status(score)
                }
        
        conn.close()
        
        # Calculate overall score
        if scores:
            overall_score = statistics.mean(score_data['score'] for score_data in scores.values())
            scores['overall'] = {
                'score': overall_score,
                'status': self._get_performance_status(overall_score)
            }
        
        return scores
    
    def _get_performance_status(self, score: float) -> str:
        """Get performance status based on score"""
        
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "fair"
        elif score >= 60:
            return "poor"
        else:
            return "critical"
    
    async def _generate_performance_recommendations(self,
                                                  suite_name: str,
                                                  trends: Dict[str, Any],
                                                  comparisons: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations"""
        
        recommendations = []
        
        # Analyze trends
        for metric_name, trend_data in trends.items():
            if trend_data['trend_direction'] == 'declining':
                recommendations.append(
                    f"⚠️ {metric_name.replace('_', ' ').title()} is declining. "
                    f"Consider investigating root causes and implementing improvements."
                )
            
            if trend_data['volatility_score'] > 0.5:
                recommendations.append(
                    f"📊 {metric_name.replace('_', ' ').title()} shows high volatility. "
                    f"Consider stabilizing the testing environment or process."
                )
            
            if trend_data['anomalies']:
                recommendations.append(
                    f"🔍 {len(trend_data['anomalies'])} anomalies detected in {metric_name}. "
                    f"Review these outliers for potential issues."
                )
        
        # Analyze comparisons
        if comparisons:
            latest_comparison = comparisons[0]
            
            if latest_comparison['performance_regression']:
                recommendations.append(
                    f"🚨 Performance regression detected in {len(latest_comparison['performance_regression'])} metrics. "
                    f"Immediate action required."
                )
            
            if latest_comparison['overall_assessment'] == 'regression':
                recommendations.append(
                    "📉 Overall performance has regressed. "
                    "Conduct thorough analysis and implement corrective measures."
                )
        
        # General recommendations
        recommendations.append(
            "📈 Regularly monitor performance trends and set up alerts for critical metrics."
        )
        
        recommendations.append(
            "🎯 Establish performance benchmarks and targets for all key metrics."
        )
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    async def export_performance_data(self, 
                                    suite_name: str,
                                    format: str = "csv",
                                    days: int = 90) -> str:
        """Export performance data in various formats"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM performance_history
            WHERE suite_name = ? AND execution_date > datetime('now', '-{} days')
            ORDER BY execution_date DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(suite_name,))
        conn.close()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'csv':
            export_path = f"performance_data_{suite_name}_{timestamp}.csv"
            df.to_csv(export_path, index=False)
        elif format.lower() == 'json':
            export_path = f"performance_data_{suite_name}_{timestamp}.json"
            df.to_json(export_path, orient='records', date_format='iso')
        elif format.lower() == 'excel':
            export_path = f"performance_data_{suite_name}_{timestamp}.xlsx"
            df.to_excel(export_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return export_path