"""
ML-based Test Optimization Recommendations System
Agent 5 Mission: Real-time Test Monitoring & Analytics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import structlog
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logger = structlog.get_logger()


@dataclass
class TestOptimization:
    """Test optimization recommendation"""
    test_name: str
    optimization_type: str  # performance, resource, stability, maintenance
    priority: str  # low, medium, high, critical
    description: str
    impact_estimate: float  # Expected improvement percentage
    implementation_effort: str  # low, medium, high
    suggested_actions: List[str]
    confidence_score: float  # 0-1
    evidence: Dict[str, Any]
    timestamp: datetime


@dataclass
class FlakyTestInfo:
    """Flaky test information"""
    test_name: str
    flakiness_score: float  # 0-1
    failure_patterns: List[str]
    success_rate: float
    typical_failure_reasons: List[str]
    recommended_fixes: List[str]
    last_updated: datetime


class TestPerformancePredictor:
    """ML model for predicting test performance"""
    
    def __init__(self, model_path: str = "test_performance_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'test_complexity', 'historical_avg_duration', 'memory_usage',
            'cpu_usage', 'io_operations', 'network_calls', 'database_queries',
            'day_of_week', 'hour_of_day', 'system_load'
        ]
        self.is_trained = False
    
    def extract_features(self, test_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from test data"""
        features = []
        
        # Test complexity (estimated based on various factors)
        complexity = self._calculate_test_complexity(test_data)
        features.append(complexity)
        
        # Historical average duration
        historical_avg = test_data.get('historical_avg_duration', 5000)
        features.append(historical_avg)
        
        # Resource usage
        features.append(test_data.get('memory_usage', 0))
        features.append(test_data.get('cpu_usage', 0))
        features.append(test_data.get('io_operations', 0))
        features.append(test_data.get('network_calls', 0))
        features.append(test_data.get('database_queries', 0))
        
        # Temporal features
        now = datetime.now()
        features.append(now.weekday())  # Day of week
        features.append(now.hour)  # Hour of day
        
        # System load (mock - in real system would get actual load)
        features.append(test_data.get('system_load', 0.5))
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_test_complexity(self, test_data: Dict[str, Any]) -> float:
        """Calculate test complexity score"""
        complexity = 0
        
        # Base complexity from test type
        test_type = test_data.get('test_type', 'unit')
        complexity_map = {
            'unit': 1,
            'integration': 3,
            'system': 5,
            'performance': 7,
            'e2e': 9
        }
        complexity += complexity_map.get(test_type, 1)
        
        # Add complexity based on dependencies
        dependencies = test_data.get('dependencies', [])
        complexity += len(dependencies) * 0.5
        
        # Add complexity based on data size
        data_size = test_data.get('data_size_mb', 0)
        complexity += data_size * 0.1
        
        return complexity
    
    def train(self, training_data: pd.DataFrame):
        """Train the performance prediction model"""
        if len(training_data) < 10:
            logger.warning("Insufficient training data for performance model")
            return
        
        # Extract features and target
        X = []
        y = []
        
        for _, row in training_data.iterrows():
            features = self.extract_features(row.to_dict())
            X.append(features.flatten())
            y.append(row.get('duration_ms', 0))
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Save model
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, self.model_path)
        
        self.is_trained = True
        logger.info("Performance prediction model trained successfully")
    
    def predict(self, test_data: Dict[str, Any]) -> Tuple[float, float]:
        """Predict test execution time"""
        if not self.is_trained:
            return 5000.0, 0.0  # Default prediction with low confidence
        
        features = self.extract_features(test_data)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence based on feature importance and prediction variance
        confidence = self._calculate_prediction_confidence(features_scaled)
        
        return prediction, confidence
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in prediction"""
        if not hasattr(self.model, 'estimators_'):
            return 0.5
        
        # Use variance of individual tree predictions as confidence metric
        predictions = np.array([tree.predict(features)[0] for tree in self.model.estimators_])
        variance = np.var(predictions)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + variance / 1000)  # Normalize by expected range
        return min(max(confidence, 0.1), 0.9)  # Clamp between 0.1 and 0.9


class FlakyTestDetector:
    """ML-based flaky test detection"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.flaky_tests = {}
        self.test_history = defaultdict(list)
    
    def analyze_test_stability(self, test_name: str, test_results: List[Dict[str, Any]]) -> FlakyTestInfo:
        """Analyze test stability and detect flakiness"""
        if len(test_results) < 5:
            return None  # Need minimum history
        
        # Calculate basic stability metrics
        total_runs = len(test_results)
        successful_runs = sum(1 for r in test_results if r.get('status') == 'passed')
        success_rate = successful_runs / total_runs
        
        # Analyze duration variance
        durations = [r.get('duration_ms', 0) for r in test_results if r.get('duration_ms')]
        duration_variance = np.var(durations) if durations else 0
        
        # Analyze failure patterns
        failures = [r for r in test_results if r.get('status') == 'failed']
        failure_patterns = self._extract_failure_patterns(failures)
        
        # Calculate flakiness score
        flakiness_score = self._calculate_flakiness_score(
            success_rate, duration_variance, failure_patterns
        )
        
        # Generate recommended fixes
        recommended_fixes = self._generate_stability_fixes(
            success_rate, duration_variance, failure_patterns
        )
        
        flaky_info = FlakyTestInfo(
            test_name=test_name,
            flakiness_score=flakiness_score,
            failure_patterns=[p['pattern'] for p in failure_patterns],
            success_rate=success_rate,
            typical_failure_reasons=[p['reason'] for p in failure_patterns],
            recommended_fixes=recommended_fixes,
            last_updated=datetime.now()
        )
        
        self.flaky_tests[test_name] = flaky_info
        return flaky_info
    
    def _extract_failure_patterns(self, failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract patterns from test failures"""
        patterns = []
        
        # Group failures by error message
        error_groups = defaultdict(list)
        for failure in failures:
            error_msg = failure.get('error_message', 'Unknown error')
            # Normalize error message for pattern matching
            normalized_error = self._normalize_error_message(error_msg)
            error_groups[normalized_error].append(failure)
        
        for pattern, occurrences in error_groups.items():
            patterns.append({
                'pattern': pattern,
                'count': len(occurrences),
                'reason': self._classify_failure_reason(pattern),
                'frequency': len(occurrences) / len(failures)
            })
        
        return sorted(patterns, key=lambda p: p['count'], reverse=True)
    
    def _normalize_error_message(self, error_msg: str) -> str:
        """Normalize error message for pattern matching"""
        # Remove timestamps, line numbers, and other variable parts
        import re
        
        # Remove timestamps
        error_msg = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '', error_msg)
        
        # Remove line numbers
        error_msg = re.sub(r'line \d+', 'line X', error_msg)
        
        # Remove file paths
        error_msg = re.sub(r'/[^\s]+\.py', 'file.py', error_msg)
        
        # Remove memory addresses
        error_msg = re.sub(r'0x[0-9a-fA-F]+', '0xXXXX', error_msg)
        
        return error_msg.strip()
    
    def _classify_failure_reason(self, pattern: str) -> str:
        """Classify failure reason from pattern"""
        pattern_lower = pattern.lower()
        
        if 'timeout' in pattern_lower:
            return 'timeout'
        elif 'connection' in pattern_lower:
            return 'network_issue'
        elif 'memory' in pattern_lower or 'oom' in pattern_lower:
            return 'memory_issue'
        elif 'assertion' in pattern_lower:
            return 'assertion_error'
        elif 'import' in pattern_lower:
            return 'dependency_issue'
        elif 'permission' in pattern_lower:
            return 'permission_error'
        else:
            return 'unknown'
    
    def _calculate_flakiness_score(self, success_rate: float, duration_variance: float, 
                                  failure_patterns: List[Dict[str, Any]]) -> float:
        """Calculate flakiness score (0-1)"""
        score = 0
        
        # Success rate component (lower success rate = higher flakiness)
        if success_rate < 0.95:
            score += (1 - success_rate) * 0.5
        
        # Duration variance component
        if duration_variance > 1000000:  # High variance in duration
            score += 0.3
        
        # Failure pattern diversity component
        if len(failure_patterns) > 3:  # Multiple different failure patterns
            score += 0.2
        
        # Intermittent failure component
        if failure_patterns:
            max_frequency = max(p['frequency'] for p in failure_patterns)
            if max_frequency < 0.8:  # No single dominant failure pattern
                score += 0.2
        
        return min(score, 1.0)
    
    def _generate_stability_fixes(self, success_rate: float, duration_variance: float,
                                 failure_patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommended fixes for stability issues"""
        fixes = []
        
        if success_rate < 0.9:
            fixes.append("Investigate and fix frequent failures")
        
        if duration_variance > 1000000:
            fixes.append("Add timeouts and optimize performance for consistent execution")
        
        # Pattern-specific fixes
        for pattern in failure_patterns:
            reason = pattern['reason']
            if reason == 'timeout':
                fixes.append("Increase timeout values or optimize test performance")
            elif reason == 'network_issue':
                fixes.append("Add network retry logic and connection pooling")
            elif reason == 'memory_issue':
                fixes.append("Optimize memory usage and add cleanup")
            elif reason == 'assertion_error':
                fixes.append("Review test assertions and expected behavior")
            elif reason == 'dependency_issue':
                fixes.append("Ensure all dependencies are properly configured")
        
        return list(set(fixes))  # Remove duplicates


class TestMaintenancePredictor:
    """Predict test maintenance needs"""
    
    def __init__(self):
        self.maintenance_history = deque(maxlen=1000)
        self.complexity_thresholds = {
            'low': 5,
            'medium': 15,
            'high': 30
        }
    
    def predict_maintenance_needs(self, test_data: Dict[str, Any]) -> List[TestOptimization]:
        """Predict maintenance needs for tests"""
        optimizations = []
        
        # Check for outdated tests
        last_modified = test_data.get('last_modified')
        if last_modified:
            days_since_modified = (datetime.now() - last_modified).days
            if days_since_modified > 180:  # 6 months
                optimizations.append(TestOptimization(
                    test_name=test_data['test_name'],
                    optimization_type='maintenance',
                    priority='medium',
                    description='Test has not been updated in 6+ months',
                    impact_estimate=0.1,
                    implementation_effort='low',
                    suggested_actions=['Review test relevance', 'Update test data', 'Refactor if needed'],
                    confidence_score=0.7,
                    evidence={'days_since_modified': days_since_modified},
                    timestamp=datetime.now()
                ))
        
        # Check for slow tests
        avg_duration = test_data.get('avg_duration_ms', 0)
        if avg_duration > 30000:  # 30 seconds
            optimizations.append(TestOptimization(
                test_name=test_data['test_name'],
                optimization_type='performance',
                priority='high',
                description='Test execution time exceeds 30 seconds',
                impact_estimate=0.5,
                implementation_effort='medium',
                suggested_actions=['Profile test execution', 'Optimize slow operations', 'Consider test splitting'],
                confidence_score=0.8,
                evidence={'avg_duration_ms': avg_duration},
                timestamp=datetime.now()
            ))
        
        # Check for high resource usage
        memory_usage = test_data.get('avg_memory_mb', 0)
        if memory_usage > 1000:  # 1GB
            optimizations.append(TestOptimization(
                test_name=test_data['test_name'],
                optimization_type='resource',
                priority='medium',
                description='Test uses excessive memory',
                impact_estimate=0.3,
                implementation_effort='medium',
                suggested_actions=['Optimize data structures', 'Add memory cleanup', 'Reduce test data size'],
                confidence_score=0.7,
                evidence={'avg_memory_mb': memory_usage},
                timestamp=datetime.now()
            ))
        
        # Check for test complexity
        complexity = self._calculate_test_complexity(test_data)
        if complexity > self.complexity_thresholds['high']:
            optimizations.append(TestOptimization(
                test_name=test_data['test_name'],
                optimization_type='maintenance',
                priority='medium',
                description='Test has high complexity',
                impact_estimate=0.4,
                implementation_effort='high',
                suggested_actions=['Split test into smaller tests', 'Simplify test logic', 'Extract helper functions'],
                confidence_score=0.6,
                evidence={'complexity_score': complexity},
                timestamp=datetime.now()
            ))
        
        return optimizations
    
    def _calculate_test_complexity(self, test_data: Dict[str, Any]) -> float:
        """Calculate test complexity score"""
        complexity = 0
        
        # Lines of code
        loc = test_data.get('lines_of_code', 0)
        complexity += loc * 0.1
        
        # Number of assertions
        assertions = test_data.get('assertion_count', 0)
        complexity += assertions * 0.5
        
        # Number of dependencies
        dependencies = test_data.get('dependency_count', 0)
        complexity += dependencies * 1.0
        
        # Cyclomatic complexity
        cyclomatic = test_data.get('cyclomatic_complexity', 0)
        complexity += cyclomatic * 2.0
        
        return complexity


class TestOptimizationEngine:
    """Main ML-based test optimization engine"""
    
    def __init__(self, db_path: str = "test_optimization.db"):
        self.db_path = db_path
        self.performance_predictor = TestPerformancePredictor()
        self.flaky_detector = FlakyTestDetector()
        self.maintenance_predictor = TestMaintenancePredictor()
        self.optimization_history = deque(maxlen=1000)
        
        self.setup_database()
    
    def setup_database(self):
        """Setup database for optimization tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_optimizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                optimization_type TEXT,
                priority TEXT,
                description TEXT,
                impact_estimate REAL,
                implementation_effort TEXT,
                suggested_actions TEXT,
                confidence_score REAL,
                evidence TEXT,
                timestamp DATETIME,
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                timestamp DATETIME,
                duration_ms REAL,
                memory_mb REAL,
                cpu_usage REAL,
                status TEXT,
                error_message TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_test_suite(self, test_suite_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entire test suite and generate optimization recommendations"""
        all_optimizations = []
        flaky_tests = []
        
        # Analyze each test
        for test_data in test_suite_data:
            test_name = test_data['test_name']
            
            # Performance optimization
            performance_opts = self._analyze_performance_optimization(test_data)
            all_optimizations.extend(performance_opts)
            
            # Flaky test detection
            test_results = test_data.get('test_results', [])
            if test_results:
                flaky_info = self.flaky_detector.analyze_test_stability(test_name, test_results)
                if flaky_info and flaky_info.flakiness_score > 0.3:
                    flaky_tests.append(flaky_info)
            
            # Maintenance predictions
            maintenance_opts = self.maintenance_predictor.predict_maintenance_needs(test_data)
            all_optimizations.extend(maintenance_opts)
        
        # Store optimizations
        self._store_optimizations(all_optimizations)
        
        # Generate summary report
        return self._generate_optimization_report(all_optimizations, flaky_tests)
    
    def _analyze_performance_optimization(self, test_data: Dict[str, Any]) -> List[TestOptimization]:
        """Analyze performance optimization opportunities"""
        optimizations = []
        
        # Predict optimal execution time
        predicted_time, confidence = self.performance_predictor.predict(test_data)
        actual_time = test_data.get('avg_duration_ms', 0)
        
        if actual_time > predicted_time * 1.5:  # 50% slower than predicted
            optimizations.append(TestOptimization(
                test_name=test_data['test_name'],
                optimization_type='performance',
                priority='high',
                description=f'Test is {((actual_time/predicted_time - 1) * 100):.1f}% slower than predicted',
                impact_estimate=(actual_time - predicted_time) / actual_time,
                implementation_effort='medium',
                suggested_actions=[
                    'Profile test execution',
                    'Identify performance bottlenecks',
                    'Optimize slow operations'
                ],
                confidence_score=confidence,
                evidence={
                    'actual_time_ms': actual_time,
                    'predicted_time_ms': predicted_time,
                    'slowdown_percentage': ((actual_time/predicted_time - 1) * 100)
                },
                timestamp=datetime.now()
            ))
        
        return optimizations
    
    def _store_optimizations(self, optimizations: List[TestOptimization]):
        """Store optimization recommendations in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for opt in optimizations:
            cursor.execute('''
                INSERT INTO test_optimizations 
                (test_name, optimization_type, priority, description, impact_estimate,
                 implementation_effort, suggested_actions, confidence_score, evidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                opt.test_name,
                opt.optimization_type,
                opt.priority,
                opt.description,
                opt.impact_estimate,
                opt.implementation_effort,
                json.dumps(opt.suggested_actions),
                opt.confidence_score,
                json.dumps(opt.evidence),
                opt.timestamp
            ))
        
        conn.commit()
        conn.close()
    
    def _generate_optimization_report(self, optimizations: List[TestOptimization], 
                                    flaky_tests: List[FlakyTestInfo]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        # Categorize optimizations
        by_priority = defaultdict(list)
        by_type = defaultdict(list)
        
        for opt in optimizations:
            by_priority[opt.priority].append(opt)
            by_type[opt.optimization_type].append(opt)
        
        # Calculate potential impact
        total_impact = sum(opt.impact_estimate for opt in optimizations)
        
        # Generate recommendations
        recommendations = self._generate_actionable_recommendations(optimizations, flaky_tests)
        
        return {
            'summary': {
                'total_optimizations': len(optimizations),
                'critical_optimizations': len(by_priority['critical']),
                'high_priority_optimizations': len(by_priority['high']),
                'flaky_tests': len(flaky_tests),
                'estimated_total_impact': total_impact,
                'top_optimization_types': sorted(by_type.keys(), key=lambda x: len(by_type[x]), reverse=True)
            },
            'optimizations_by_priority': {
                priority: [asdict(opt) for opt in opts]
                for priority, opts in by_priority.items()
            },
            'optimizations_by_type': {
                opt_type: [asdict(opt) for opt in opts]
                for opt_type, opts in by_type.items()
            },
            'flaky_tests': [asdict(flaky) for flaky in flaky_tests],
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
    
    def _generate_actionable_recommendations(self, optimizations: List[TestOptimization], 
                                           flaky_tests: List[FlakyTestInfo]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Priority-based recommendations
        critical_opts = [opt for opt in optimizations if opt.priority == 'critical']
        if critical_opts:
            recommendations.append({
                'type': 'urgent_action',
                'title': 'Critical optimizations required',
                'description': f'{len(critical_opts)} tests need immediate attention',
                'actions': [opt.test_name for opt in critical_opts[:5]]  # Top 5
            })
        
        # Performance recommendations
        slow_tests = [opt for opt in optimizations if opt.optimization_type == 'performance']
        if len(slow_tests) > 5:
            recommendations.append({
                'type': 'performance_improvement',
                'title': 'Performance optimization opportunity',
                'description': f'{len(slow_tests)} tests can be optimized for better performance',
                'actions': ['Profile slow tests', 'Implement caching', 'Optimize algorithms']
            })
        
        # Flaky test recommendations
        if len(flaky_tests) > 3:
            recommendations.append({
                'type': 'stability_improvement',
                'title': 'Test stability issues detected',
                'description': f'{len(flaky_tests)} flaky tests need attention',
                'actions': ['Investigate flaky tests', 'Add retry logic', 'Fix root causes']
            })
        
        # Maintenance recommendations
        maintenance_opts = [opt for opt in optimizations if opt.optimization_type == 'maintenance']
        if len(maintenance_opts) > 10:
            recommendations.append({
                'type': 'maintenance_cleanup',
                'title': 'Test maintenance required',
                'description': f'{len(maintenance_opts)} tests need maintenance',
                'actions': ['Update outdated tests', 'Refactor complex tests', 'Remove obsolete tests']
            })
        
        return recommendations
    
    def get_optimization_status(self, days: int = 30) -> Dict[str, Any]:
        """Get optimization status over time period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT optimization_type, priority, status, COUNT(*) as count
            FROM test_optimizations
            WHERE timestamp > ?
            GROUP BY optimization_type, priority, status
        ''', (cutoff_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        status_summary = {
            'pending': 0,
            'in_progress': 0,
            'completed': 0,
            'by_type': defaultdict(int),
            'by_priority': defaultdict(int)
        }
        
        for opt_type, priority, status, count in results:
            status_summary[status] += count
            status_summary['by_type'][opt_type] += count
            status_summary['by_priority'][priority] += count
        
        return {
            'period_days': days,
            'status_summary': dict(status_summary),
            'timestamp': datetime.now()
        }
    
    def train_models(self, training_data: pd.DataFrame):
        """Train ML models with historical data"""
        logger.info("Training optimization models...")
        
        # Train performance predictor
        self.performance_predictor.train(training_data)
        
        logger.info("Model training completed")