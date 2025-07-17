"""
Historical RTO Trend Analysis System for comprehensive performance analytics.

This module provides:
- Historical trend analysis and forecasting
- Performance pattern detection
- Anomaly detection in RTO metrics
- Capacity planning insights
- Comparative analysis across time periods
- Statistical analysis and reporting
"""

import asyncio
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.monitoring.rto_monitor import RTOMetric, RTOStatus, RTOTarget

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"

class AnomalyType(Enum):
    """Anomaly types."""
    SPIKE = "spike"
    DIP = "dip"
    GRADUAL_INCREASE = "gradual_increase"
    GRADUAL_DECREASE = "gradual_decrease"
    PATTERN_BREAK = "pattern_break"

@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    component: str
    period_days: int
    direction: TrendDirection
    slope: float
    r_squared: float
    prediction_7d: float
    prediction_30d: float
    confidence_interval: Tuple[float, float]
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "period_days": self.period_days,
            "direction": self.direction.value,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "prediction_7d": self.prediction_7d,
            "prediction_30d": self.prediction_30d,
            "confidence_interval": self.confidence_interval,
            "details": self.details
        }

@dataclass
class AnomalyDetection:
    """Anomaly detection results."""
    component: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: float
    rto_value: float
    expected_range: Tuple[float, float]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "rto_value": self.rto_value,
            "expected_range": self.expected_range,
            "description": self.description,
            "metadata": self.metadata
        }

@dataclass
class PerformancePattern:
    """Performance pattern analysis."""
    component: str
    pattern_type: str
    description: str
    frequency: str
    impact: float
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "impact": self.impact,
            "examples": self.examples
        }

@dataclass
class CapacityInsight:
    """Capacity planning insights."""
    component: str
    current_capacity: float
    projected_capacity_7d: float
    projected_capacity_30d: float
    scaling_recommendation: str
    risk_level: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "current_capacity": self.current_capacity,
            "projected_capacity_7d": self.projected_capacity_7d,
            "projected_capacity_30d": self.projected_capacity_30d,
            "scaling_recommendation": self.scaling_recommendation,
            "risk_level": self.risk_level,
            "details": self.details
        }

class RTOTrendAnalyzer:
    """RTO trend analysis engine."""
    
    def __init__(self, db_path: str = "rto_metrics.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
    
    def _get_time_series_data(self, component: str, days: int) -> List[Tuple[datetime, float]]:
        """Get time series data for analysis."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT timestamp, actual_seconds FROM rto_metrics 
                WHERE component = ? AND timestamp >= ?
                ORDER BY timestamp
            """, (component, cutoff.isoformat()))
            
            data = []
            for row in cursor.fetchall():
                timestamp = datetime.fromisoformat(row['timestamp'])
                rto_value = row['actual_seconds']
                data.append((timestamp, rto_value))
            
            return data
    
    def analyze_trend(self, component: str, days: int = 30) -> TrendAnalysis:
        """Analyze RTO trend for a component."""
        data = self._get_time_series_data(component, days)
        
        if len(data) < 10:
            return TrendAnalysis(
                component=component,
                period_days=days,
                direction=TrendDirection.STABLE,
                slope=0.0,
                r_squared=0.0,
                prediction_7d=0.0,
                prediction_30d=0.0,
                confidence_interval=(0.0, 0.0),
                details={"error": "Insufficient data for trend analysis"}
            )
        
        # Prepare data for regression
        timestamps = [t.timestamp() for t, _ in data]
        rto_values = [v for _, v in data]
        
        # Normalize timestamps
        base_time = min(timestamps)
        x_values = [(t - base_time) / 3600 for t in timestamps]  # Hours from start
        
        # Perform linear regression
        X = np.array(x_values).reshape(-1, 1)
        y = np.array(rto_values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        
        # Calculate slope (change per hour)
        slope = model.coef_[0]
        
        # Determine trend direction
        if abs(slope) < 0.001:  # Very small slope
            direction = TrendDirection.STABLE
        elif slope > 0:
            direction = TrendDirection.DEGRADING
        else:
            direction = TrendDirection.IMPROVING
        
        # Check for volatility
        residuals = y - y_pred
        volatility = np.std(residuals)
        mean_value = np.mean(y)
        
        if volatility > mean_value * 0.5:  # High volatility
            direction = TrendDirection.VOLATILE
        
        # Make predictions
        current_time = max(x_values)
        pred_7d = model.predict([[current_time + 7 * 24]])[0]  # 7 days ahead
        pred_30d = model.predict([[current_time + 30 * 24]])[0]  # 30 days ahead
        
        # Calculate confidence interval
        residual_std = np.std(residuals)
        confidence_interval = (
            pred_7d - 1.96 * residual_std,
            pred_7d + 1.96 * residual_std
        )
        
        return TrendAnalysis(
            component=component,
            period_days=days,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            prediction_7d=max(0, pred_7d),
            prediction_30d=max(0, pred_30d),
            confidence_interval=confidence_interval,
            details={
                "data_points": len(data),
                "mean_rto": mean_value,
                "volatility": volatility,
                "trend_strength": abs(slope) / volatility if volatility > 0 else 0
            }
        )
    
    def detect_anomalies(self, component: str, days: int = 7) -> List[AnomalyDetection]:
        """Detect anomalies in RTO data."""
        data = self._get_time_series_data(component, days)
        
        if len(data) < 20:
            return []
        
        timestamps = [t for t, _ in data]
        rto_values = [v for _, v in data]
        
        # Calculate statistical thresholds
        mean_rto = statistics.mean(rto_values)
        std_rto = statistics.stdev(rto_values)
        
        # Z-score based anomaly detection
        z_threshold = 2.5
        anomalies = []
        
        for i, (timestamp, rto_value) in enumerate(data):
            z_score = abs(rto_value - mean_rto) / std_rto if std_rto > 0 else 0
            
            if z_score > z_threshold:
                # Determine anomaly type
                if rto_value > mean_rto + z_threshold * std_rto:
                    anomaly_type = AnomalyType.SPIKE
                else:
                    anomaly_type = AnomalyType.DIP
                
                anomaly = AnomalyDetection(
                    component=component,
                    timestamp=timestamp,
                    anomaly_type=anomaly_type,
                    severity=z_score,
                    rto_value=rto_value,
                    expected_range=(
                        mean_rto - 2 * std_rto,
                        mean_rto + 2 * std_rto
                    ),
                    description=f"RTO value {rto_value:.2f}s is {z_score:.1f} standard deviations from mean ({mean_rto:.2f}s)",
                    metadata={
                        "z_score": z_score,
                        "mean_rto": mean_rto,
                        "std_rto": std_rto
                    }
                )
                anomalies.append(anomaly)
        
        # Detect gradual changes using sliding window
        window_size = min(10, len(data) // 4)
        if window_size >= 5:
            for i in range(window_size, len(data) - window_size):
                before_window = rto_values[i-window_size:i]
                after_window = rto_values[i:i+window_size]
                
                before_mean = statistics.mean(before_window)
                after_mean = statistics.mean(after_window)
                
                # Check for significant change
                if abs(after_mean - before_mean) > std_rto:
                    if after_mean > before_mean:
                        anomaly_type = AnomalyType.GRADUAL_INCREASE
                    else:
                        anomaly_type = AnomalyType.GRADUAL_DECREASE
                    
                    anomaly = AnomalyDetection(
                        component=component,
                        timestamp=timestamps[i],
                        anomaly_type=anomaly_type,
                        severity=abs(after_mean - before_mean) / std_rto,
                        rto_value=after_mean,
                        expected_range=(before_mean - std_rto, before_mean + std_rto),
                        description=f"Gradual change detected: {before_mean:.2f}s → {after_mean:.2f}s",
                        metadata={
                            "before_mean": before_mean,
                            "after_mean": after_mean,
                            "window_size": window_size
                        }
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_patterns(self, component: str, days: int = 30) -> List[PerformancePattern]:
        """Detect performance patterns."""
        data = self._get_time_series_data(component, days)
        
        if len(data) < 50:
            return []
        
        patterns = []
        timestamps = [t for t, _ in data]
        rto_values = [v for _, v in data]
        
        # Daily pattern analysis
        daily_patterns = self._analyze_daily_patterns(timestamps, rto_values)
        if daily_patterns:
            patterns.extend(daily_patterns)
        
        # Weekly pattern analysis
        weekly_patterns = self._analyze_weekly_patterns(timestamps, rto_values)
        if weekly_patterns:
            patterns.extend(weekly_patterns)
        
        # Clustering-based pattern detection
        cluster_patterns = self._analyze_cluster_patterns(component, timestamps, rto_values)
        if cluster_patterns:
            patterns.extend(cluster_patterns)
        
        return patterns
    
    def _analyze_daily_patterns(self, timestamps: List[datetime], rto_values: List[float]) -> List[PerformancePattern]:
        """Analyze daily patterns."""
        patterns = []
        
        # Group by hour of day
        hourly_data = {}
        for timestamp, rto_value in zip(timestamps, rto_values):
            hour = timestamp.hour
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(rto_value)
        
        # Find hours with consistently high/low RTO
        overall_mean = statistics.mean(rto_values)
        overall_std = statistics.stdev(rto_values)
        
        for hour, values in hourly_data.items():
            if len(values) < 5:
                continue
            
            hour_mean = statistics.mean(values)
            
            if hour_mean > overall_mean + overall_std:
                pattern = PerformancePattern(
                    component=timestamps[0].strftime('%Y-%m-%d'),  # Use first timestamp's date as component context
                    pattern_type="daily_peak",
                    description=f"Consistently high RTO at {hour:02d}:00 ({hour_mean:.2f}s avg)",
                    frequency="daily",
                    impact=(hour_mean - overall_mean) / overall_mean,
                    examples=[{
                        "hour": hour,
                        "avg_rto": hour_mean,
                        "sample_count": len(values)
                    }]
                )
                patterns.append(pattern)
            
            elif hour_mean < overall_mean - overall_std:
                pattern = PerformancePattern(
                    component=timestamps[0].strftime('%Y-%m-%d'),
                    pattern_type="daily_optimal",
                    description=f"Consistently low RTO at {hour:02d}:00 ({hour_mean:.2f}s avg)",
                    frequency="daily",
                    impact=(overall_mean - hour_mean) / overall_mean,
                    examples=[{
                        "hour": hour,
                        "avg_rto": hour_mean,
                        "sample_count": len(values)
                    }]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_weekly_patterns(self, timestamps: List[datetime], rto_values: List[float]) -> List[PerformancePattern]:
        """Analyze weekly patterns."""
        patterns = []
        
        # Group by day of week
        weekly_data = {}
        for timestamp, rto_value in zip(timestamps, rto_values):
            day = timestamp.weekday()  # 0=Monday, 6=Sunday
            if day not in weekly_data:
                weekly_data[day] = []
            weekly_data[day].append(rto_value)
        
        # Find days with unusual patterns
        overall_mean = statistics.mean(rto_values)
        overall_std = statistics.stdev(rto_values)
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day, values in weekly_data.items():
            if len(values) < 5:
                continue
            
            day_mean = statistics.mean(values)
            day_std = statistics.stdev(values) if len(values) > 1 else 0
            
            if day_mean > overall_mean + overall_std:
                pattern = PerformancePattern(
                    component=timestamps[0].strftime('%Y-%m-%d'),
                    pattern_type="weekly_peak",
                    description=f"Higher RTO on {day_names[day]} ({day_mean:.2f}s avg)",
                    frequency="weekly",
                    impact=(day_mean - overall_mean) / overall_mean,
                    examples=[{
                        "day": day_names[day],
                        "avg_rto": day_mean,
                        "std_rto": day_std,
                        "sample_count": len(values)
                    }]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_cluster_patterns(self, component: str, timestamps: List[datetime], rto_values: List[float]) -> List[PerformancePattern]:
        """Analyze patterns using clustering."""
        patterns = []
        
        if len(rto_values) < 20:
            return patterns
        
        try:
            # Prepare features for clustering
            features = []
            for i, (timestamp, rto_value) in enumerate(zip(timestamps, rto_values)):
                features.append([
                    timestamp.hour,
                    timestamp.weekday(),
                    rto_value,
                    i  # Position in sequence
                ])
            
            # Normalize features
            features_array = np.array(features)
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Apply DBSCAN clustering
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(features_scaled)
            
            # Analyze clusters
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                cluster_mask = cluster_labels == label
                cluster_rto_values = [rto_values[i] for i in range(len(rto_values)) if cluster_mask[i]]
                cluster_timestamps = [timestamps[i] for i in range(len(timestamps)) if cluster_mask[i]]
                
                if len(cluster_rto_values) < 5:
                    continue
                
                cluster_mean = statistics.mean(cluster_rto_values)
                overall_mean = statistics.mean(rto_values)
                
                if abs(cluster_mean - overall_mean) > statistics.stdev(rto_values):
                    pattern_type = "cluster_high" if cluster_mean > overall_mean else "cluster_low"
                    
                    pattern = PerformancePattern(
                        component=component,
                        pattern_type=pattern_type,
                        description=f"Cluster of {len(cluster_rto_values)} measurements with {pattern_type.split('_')[1]} RTO ({cluster_mean:.2f}s avg)",
                        frequency="irregular",
                        impact=abs(cluster_mean - overall_mean) / overall_mean,
                        examples=[{
                            "cluster_id": label,
                            "avg_rto": cluster_mean,
                            "sample_count": len(cluster_rto_values),
                            "time_range": f"{min(cluster_timestamps).strftime('%Y-%m-%d %H:%M')} - {max(cluster_timestamps).strftime('%Y-%m-%d %H:%M')}"
                        }]
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            logger.warning(f"Error in cluster analysis: {e}")
        
        return patterns
    
    def generate_capacity_insights(self, component: str, days: int = 30) -> CapacityInsight:
        """Generate capacity planning insights."""
        data = self._get_time_series_data(component, days)
        
        if len(data) < 10:
            return CapacityInsight(
                component=component,
                current_capacity=0.0,
                projected_capacity_7d=0.0,
                projected_capacity_30d=0.0,
                scaling_recommendation="Insufficient data",
                risk_level="unknown"
            )
        
        # Get trend analysis
        trend = self.analyze_trend(component, days)
        
        # Current capacity utilization (inverse of RTO performance)
        rto_values = [v for _, v in data]
        target_rto = RTOTarget.DATABASE.value if component == "database" else RTOTarget.TRADING_ENGINE.value
        
        current_performance = statistics.mean(rto_values[-10:])  # Last 10 measurements
        current_capacity = max(0, min(100, (target_rto / current_performance) * 100))
        
        # Projected capacity
        projected_capacity_7d = max(0, min(100, (target_rto / trend.prediction_7d) * 100)) if trend.prediction_7d > 0 else 0
        projected_capacity_30d = max(0, min(100, (target_rto / trend.prediction_30d) * 100)) if trend.prediction_30d > 0 else 0
        
        # Risk assessment
        if current_capacity < 50:
            risk_level = "critical"
        elif current_capacity < 70:
            risk_level = "high"
        elif current_capacity < 85:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Scaling recommendation
        if trend.direction == TrendDirection.DEGRADING:
            if current_capacity < 70:
                scaling_recommendation = "Immediate scaling required"
            else:
                scaling_recommendation = "Consider scaling in next 7 days"
        elif trend.direction == TrendDirection.STABLE:
            scaling_recommendation = "Monitor performance trends"
        else:
            scaling_recommendation = "Performance improving, no scaling needed"
        
        return CapacityInsight(
            component=component,
            current_capacity=current_capacity,
            projected_capacity_7d=projected_capacity_7d,
            projected_capacity_30d=projected_capacity_30d,
            scaling_recommendation=scaling_recommendation,
            risk_level=risk_level,
            details={
                "current_rto": current_performance,
                "target_rto": target_rto,
                "trend_direction": trend.direction.value,
                "trend_strength": trend.details.get("trend_strength", 0)
            }
        )

class RTOAnalyticsSystem:
    """Comprehensive RTO analytics system."""
    
    def __init__(self, db_path: str = "rto_metrics.db"):
        self.db_path = db_path
        self.analyzer = RTOTrendAnalyzer(db_path)
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key."""
        return f"{method}_{hash(frozenset(kwargs.items()))}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is valid."""
        if key not in self._cache:
            return False
        
        entry_time = self._cache[key]["timestamp"]
        return (datetime.utcnow() - entry_time).total_seconds() < self._cache_ttl
    
    def get_comprehensive_analysis(self, component: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analysis for a component."""
        cache_key = self._get_cache_key("comprehensive", component=component, days=days)
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]["data"]
        
        # Perform analysis
        trend = self.analyzer.analyze_trend(component, days)
        anomalies = self.analyzer.detect_anomalies(component, min(days, 7))
        patterns = self.analyzer.detect_patterns(component, days)
        capacity = self.analyzer.generate_capacity_insights(component, days)
        
        analysis = {
            "component": component,
            "analysis_period": days,
            "generated_at": datetime.utcnow().isoformat(),
            "trend_analysis": trend.to_dict(),
            "anomaly_detection": [a.to_dict() for a in anomalies],
            "performance_patterns": [p.to_dict() for p in patterns],
            "capacity_insights": capacity.to_dict(),
            "summary": {
                "overall_health": self._assess_overall_health(trend, anomalies, capacity),
                "key_findings": self._generate_key_findings(trend, anomalies, patterns, capacity),
                "recommendations": self._generate_recommendations(trend, anomalies, patterns, capacity)
            }
        }
        
        # Cache result
        self._cache[cache_key] = {
            "data": analysis,
            "timestamp": datetime.utcnow()
        }
        
        return analysis
    
    def _assess_overall_health(self, trend: TrendAnalysis, anomalies: List[AnomalyDetection], capacity: CapacityInsight) -> str:
        """Assess overall health."""
        if capacity.risk_level == "critical":
            return "critical"
        elif trend.direction == TrendDirection.DEGRADING and len(anomalies) > 5:
            return "poor"
        elif trend.direction == TrendDirection.VOLATILE or capacity.risk_level == "high":
            return "warning"
        elif trend.direction == TrendDirection.IMPROVING:
            return "good"
        else:
            return "stable"
    
    def _generate_key_findings(self, trend: TrendAnalysis, anomalies: List[AnomalyDetection], 
                              patterns: List[PerformancePattern], capacity: CapacityInsight) -> List[str]:
        """Generate key findings."""
        findings = []
        
        # Trend findings
        if trend.direction == TrendDirection.DEGRADING:
            findings.append(f"Performance degrading with slope {trend.slope:.4f}/hour")
        elif trend.direction == TrendDirection.IMPROVING:
            findings.append(f"Performance improving with slope {trend.slope:.4f}/hour")
        
        # Anomaly findings
        if len(anomalies) > 0:
            spike_count = sum(1 for a in anomalies if a.anomaly_type == AnomalyType.SPIKE)
            if spike_count > 0:
                findings.append(f"Detected {spike_count} RTO spikes in recent period")
        
        # Pattern findings
        for pattern in patterns:
            if pattern.impact > 0.2:  # Significant impact
                findings.append(f"Found {pattern.pattern_type} pattern with {pattern.impact:.1%} impact")
        
        # Capacity findings
        if capacity.risk_level in ["critical", "high"]:
            findings.append(f"Capacity at {capacity.current_capacity:.1f}% - {capacity.risk_level} risk level")
        
        return findings
    
    def _generate_recommendations(self, trend: TrendAnalysis, anomalies: List[AnomalyDetection], 
                                 patterns: List[PerformancePattern], capacity: CapacityInsight) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        
        # Trend-based recommendations
        if trend.direction == TrendDirection.DEGRADING:
            recommendations.append("Investigate root cause of performance degradation")
            recommendations.append("Consider implementing performance optimizations")
        
        # Anomaly-based recommendations
        high_severity_anomalies = [a for a in anomalies if a.severity > 3.0]
        if len(high_severity_anomalies) > 0:
            recommendations.append("Investigate high-severity anomalies for system issues")
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.pattern_type == "daily_peak":
                recommendations.append(f"Optimize resources during peak hours: {pattern.description}")
            elif pattern.pattern_type == "weekly_peak":
                recommendations.append(f"Plan capacity for weekly patterns: {pattern.description}")
        
        # Capacity-based recommendations
        recommendations.append(capacity.scaling_recommendation)
        
        return recommendations
    
    def get_comparative_analysis(self, components: List[str], days: int = 30) -> Dict[str, Any]:
        """Get comparative analysis across components."""
        analysis = {
            "components": components,
            "comparison_period": days,
            "generated_at": datetime.utcnow().isoformat(),
            "component_analysis": {},
            "comparative_metrics": {}
        }
        
        # Analyze each component
        for component in components:
            analysis["component_analysis"][component] = self.get_comprehensive_analysis(component, days)
        
        # Comparative metrics
        trends = [analysis["component_analysis"][c]["trend_analysis"] for c in components]
        capacities = [analysis["component_analysis"][c]["capacity_insights"] for c in components]
        
        analysis["comparative_metrics"] = {
            "performance_ranking": self._rank_components_by_performance(components, trends, capacities),
            "stability_comparison": self._compare_stability(components, trends),
            "capacity_comparison": self._compare_capacity(components, capacities),
            "risk_assessment": self._assess_comparative_risk(components, analysis["component_analysis"])
        }
        
        return analysis
    
    def _rank_components_by_performance(self, components: List[str], trends: List[Dict], capacities: List[Dict]) -> List[Dict]:
        """Rank components by performance."""
        rankings = []
        
        for i, component in enumerate(components):
            trend = trends[i]
            capacity = capacities[i]
            
            # Calculate performance score
            capacity_score = capacity["current_capacity"]
            trend_score = 50 + (trend["slope"] * -1000)  # Negative slope is better
            stability_score = max(0, 100 - trend["details"]["volatility"] * 100)
            
            overall_score = (capacity_score + trend_score + stability_score) / 3
            
            rankings.append({
                "component": component,
                "overall_score": overall_score,
                "capacity_score": capacity_score,
                "trend_score": trend_score,
                "stability_score": stability_score,
                "rank": 0  # Will be set after sorting
            })
        
        # Sort by overall score
        rankings.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Set ranks
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
        
        return rankings
    
    def _compare_stability(self, components: List[str], trends: List[Dict]) -> Dict[str, Any]:
        """Compare stability across components."""
        stability_data = {}
        
        for i, component in enumerate(components):
            trend = trends[i]
            volatility = trend["details"]["volatility"]
            direction = trend["direction"]
            
            stability_data[component] = {
                "volatility": volatility,
                "direction": direction,
                "stability_score": max(0, 100 - volatility * 100)
            }
        
        # Find most/least stable
        most_stable = min(stability_data.items(), key=lambda x: x[1]["volatility"])
        least_stable = max(stability_data.items(), key=lambda x: x[1]["volatility"])
        
        return {
            "stability_by_component": stability_data,
            "most_stable": {"component": most_stable[0], "volatility": most_stable[1]["volatility"]},
            "least_stable": {"component": least_stable[0], "volatility": least_stable[1]["volatility"]}
        }
    
    def _compare_capacity(self, components: List[str], capacities: List[Dict]) -> Dict[str, Any]:
        """Compare capacity across components."""
        capacity_data = {}
        
        for i, component in enumerate(components):
            capacity = capacities[i]
            capacity_data[component] = {
                "current_capacity": capacity["current_capacity"],
                "risk_level": capacity["risk_level"],
                "scaling_recommendation": capacity["scaling_recommendation"]
            }
        
        # Find highest/lowest capacity
        highest_capacity = max(capacity_data.items(), key=lambda x: x[1]["current_capacity"])
        lowest_capacity = min(capacity_data.items(), key=lambda x: x[1]["current_capacity"])
        
        return {
            "capacity_by_component": capacity_data,
            "highest_capacity": {"component": highest_capacity[0], "capacity": highest_capacity[1]["current_capacity"]},
            "lowest_capacity": {"component": lowest_capacity[0], "capacity": lowest_capacity[1]["current_capacity"]}
        }
    
    def _assess_comparative_risk(self, components: List[str], component_analysis: Dict) -> Dict[str, Any]:
        """Assess comparative risk."""
        risk_levels = []
        
        for component in components:
            analysis = component_analysis[component]
            capacity_risk = analysis["capacity_insights"]["risk_level"]
            trend_direction = analysis["trend_analysis"]["direction"]
            anomaly_count = len(analysis["anomaly_detection"])
            
            # Calculate risk score
            risk_score = 0
            if capacity_risk == "critical":
                risk_score += 40
            elif capacity_risk == "high":
                risk_score += 30
            elif capacity_risk == "medium":
                risk_score += 20
            
            if trend_direction == "degrading":
                risk_score += 20
            elif trend_direction == "volatile":
                risk_score += 15
            
            risk_score += min(anomaly_count * 5, 30)
            
            risk_levels.append({
                "component": component,
                "risk_score": risk_score,
                "risk_level": capacity_risk,
                "contributing_factors": {
                    "capacity_risk": capacity_risk,
                    "trend_direction": trend_direction,
                    "anomaly_count": anomaly_count
                }
            })
        
        # Sort by risk score
        risk_levels.sort(key=lambda x: x["risk_score"], reverse=True)
        
        return {
            "risk_by_component": risk_levels,
            "highest_risk": risk_levels[0] if risk_levels else None,
            "overall_risk_level": self._calculate_overall_risk(risk_levels)
        }
    
    def _calculate_overall_risk(self, risk_levels: List[Dict]) -> str:
        """Calculate overall risk level."""
        if not risk_levels:
            return "unknown"
        
        avg_risk_score = sum(r["risk_score"] for r in risk_levels) / len(risk_levels)
        
        if avg_risk_score >= 60:
            return "critical"
        elif avg_risk_score >= 40:
            return "high"
        elif avg_risk_score >= 20:
            return "medium"
        else:
            return "low"
    
    def generate_report(self, components: List[str], days: int = 30) -> str:
        """Generate comprehensive analysis report."""
        analysis = self.get_comparative_analysis(components, days)
        
        report = f"""
# RTO Analytics Report
Generated: {analysis['generated_at']}
Analysis Period: {days} days
Components: {', '.join(components)}

## Executive Summary
Overall Risk Level: {analysis['comparative_metrics']['risk_assessment']['overall_risk_level'].upper()}

### Performance Ranking
"""
        
        for ranking in analysis['comparative_metrics']['performance_ranking']:
            report += f"{ranking['rank']}. {ranking['component']}: {ranking['overall_score']:.1f}/100\n"
        
        report += "\n## Component Analysis\n"
        
        for component in components:
            comp_analysis = analysis['component_analysis'][component]
            report += f"\n### {component.upper()}\n"
            report += f"Health: {comp_analysis['summary']['overall_health']}\n"
            report += f"Capacity: {comp_analysis['capacity_insights']['current_capacity']:.1f}%\n"
            report += f"Trend: {comp_analysis['trend_analysis']['direction']}\n"
            report += f"Anomalies: {len(comp_analysis['anomaly_detection'])}\n"
            
            if comp_analysis['summary']['key_findings']:
                report += "\nKey Findings:\n"
                for finding in comp_analysis['summary']['key_findings']:
                    report += f"- {finding}\n"
            
            if comp_analysis['summary']['recommendations']:
                report += "\nRecommendations:\n"
                for rec in comp_analysis['summary']['recommendations']:
                    report += f"- {rec}\n"
        
        return report

# Global analytics instance
analytics_system = None

def initialize_analytics_system(db_path: str = "rto_metrics.db") -> RTOAnalyticsSystem:
    """Initialize global analytics system."""
    global analytics_system
    analytics_system = RTOAnalyticsSystem(db_path)
    return analytics_system

def get_analytics_system() -> Optional[RTOAnalyticsSystem]:
    """Get global analytics system instance."""
    return analytics_system