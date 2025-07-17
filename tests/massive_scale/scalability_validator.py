"""
Scalability Validator - Phase 3A Implementation
Agent Epsilon: Massive Scale Testing Architecture

Advanced scalability validation with:
- Auto-scaling capability testing
- Resource efficiency analysis
- Performance degradation detection
- System breaking point identification
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import psutil
import matplotlib.pyplot as plt
from pathlib import Path

from .million_tps_simulator import MillionTPSSimulator, MillionTPSTestConfig

logger = logging.getLogger(__name__)

@dataclass
class ScalabilityTestPoint:
    """Single scalability test point"""
    target_tps: int
    actual_tps: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    cpu_usage: float
    memory_usage: float
    error_rate: float
    active_nodes: int
    timestamp: float

@dataclass
class ScalabilityConfig:
    """Scalability test configuration"""
    min_tps: int = 1000
    max_tps: int = 2000000
    step_size: int = 100000
    test_duration_seconds: int = 300
    stability_threshold: float = 0.95
    latency_threshold_ms: int = 100
    error_rate_threshold: float = 0.001

class ScalabilityValidator:
    """
    Scalability Validator
    
    Validates system scalability by testing performance across
    different load levels and identifying breaking points.
    """
    
    def __init__(self, config: ScalabilityConfig):
        self.config = config
        self.test_id = str(uuid.uuid4())
        
        # Test data
        self.test_points: List[ScalabilityTestPoint] = []
        self.breaking_point = None
        self.optimal_point = None
        
        # Analysis results
        self.scalability_coefficient = None
        self.efficiency_curve = None
        self.recommendation_report = None
        
    async def run_scalability_validation(self) -> Dict[str, Any]:
        """Run complete scalability validation"""
        logger.info("🔍 Starting Scalability Validation")
        
        try:
            # Generate test points
            test_loads = self._generate_test_loads()
            
            # Execute tests for each load level
            for target_tps in test_loads:
                await self._test_load_level(target_tps)
                
            # Analyze results
            analysis = await self._analyze_scalability()
            
            # Generate recommendations
            recommendations = self._generate_recommendations()
            
            # Create final report
            report = self._create_scalability_report(analysis, recommendations)
            
            return report
            
        except Exception as e:
            logger.error(f"Scalability validation failed: {e}")
            raise
            
    def _generate_test_loads(self) -> List[int]:
        """Generate test load levels"""
        loads = []
        
        # Start with small increments
        current_load = self.config.min_tps
        
        while current_load <= self.config.max_tps:
            loads.append(current_load)
            
            # Adaptive step size - smaller steps near breaking point
            if current_load < 100000:
                step = 10000
            elif current_load < 500000:
                step = 50000
            else:
                step = self.config.step_size
                
            current_load += step
            
        return loads
        
    async def _test_load_level(self, target_tps: int):
        """Test a specific load level"""
        logger.info(f"📊 Testing load level: {target_tps:,} TPS")
        
        # Create test configuration
        test_config = MillionTPSTestConfig(
            target_tps=target_tps,
            duration_minutes=self.config.test_duration_seconds // 60,
            ramp_up_minutes=1,
            ramp_down_minutes=1
        )
        
        # Run test
        simulator = MillionTPSSimulator(test_config)
        
        try:
            await simulator.initialize()
            
            # Run shorter test for scalability validation
            test_result = await simulator.run_million_tps_test()
            
            # Extract metrics
            performance = test_result["performance_results"]
            
            # Create test point
            test_point = ScalabilityTestPoint(
                target_tps=target_tps,
                actual_tps=performance["max_tps_achieved"],
                latency_p50=performance["latency_statistics"].get("p50_avg", 0),
                latency_p95=performance["latency_statistics"].get("p95_avg", 0),
                latency_p99=performance["latency_statistics"].get("p99_avg", 0),
                cpu_usage=performance.get("peak_cpu_usage", 0),
                memory_usage=performance.get("peak_memory_usage", 0),
                error_rate=1 - performance["success_rate"],
                active_nodes=performance["active_nodes"],
                timestamp=time.time()
            )
            
            self.test_points.append(test_point)
            
            # Check if we've hit breaking point
            if self._is_breaking_point(test_point):
                logger.warning(f"🚨 Breaking point detected at {target_tps:,} TPS")
                self.breaking_point = test_point
                
            # Check for optimal point
            if self._is_optimal_point(test_point):
                self.optimal_point = test_point
                
            logger.info(f"✅ Test completed - Actual: {test_point.actual_tps:,.0f} TPS, "
                       f"P99: {test_point.latency_p99:.2f}ms, "
                       f"Error Rate: {test_point.error_rate:.4f}")
                       
        except Exception as e:
            logger.error(f"Load test failed at {target_tps:,} TPS: {e}")
            
            # Create failed test point
            failed_point = ScalabilityTestPoint(
                target_tps=target_tps,
                actual_tps=0,
                latency_p50=0,
                latency_p95=0, 
                latency_p99=0,
                cpu_usage=0,
                memory_usage=0,
                error_rate=1.0,
                active_nodes=0,
                timestamp=time.time()
            )
            
            self.test_points.append(failed_point)
            self.breaking_point = failed_point
            
        finally:
            await simulator.cleanup()
            
    def _is_breaking_point(self, test_point: ScalabilityTestPoint) -> bool:
        """Check if test point represents a breaking point"""
        return (
            test_point.actual_tps < test_point.target_tps * self.config.stability_threshold or
            test_point.latency_p99 > self.config.latency_threshold_ms or
            test_point.error_rate > self.config.error_rate_threshold
        )
        
    def _is_optimal_point(self, test_point: ScalabilityTestPoint) -> bool:
        """Check if test point represents optimal performance"""
        if not self.test_points:
            return False
            
        # Calculate efficiency (TPS per CPU%)
        if test_point.cpu_usage > 0:
            efficiency = test_point.actual_tps / test_point.cpu_usage
            
            # Check if this is the most efficient point so far
            if self.optimal_point is None:
                return True
                
            optimal_efficiency = self.optimal_point.actual_tps / self.optimal_point.cpu_usage if self.optimal_point.cpu_usage > 0 else 0
            
            return efficiency > optimal_efficiency
            
        return False
        
    async def _analyze_scalability(self) -> Dict[str, Any]:
        """Analyze scalability test results"""
        logger.info("📈 Analyzing scalability results")
        
        if not self.test_points:
            raise ValueError("No test points available for analysis")
            
        # Calculate scalability coefficient
        self.scalability_coefficient = self._calculate_scalability_coefficient()
        
        # Generate efficiency curve
        self.efficiency_curve = self._generate_efficiency_curve()
        
        # Identify performance characteristics
        performance_analysis = self._analyze_performance_characteristics()
        
        # Resource utilization analysis
        resource_analysis = self._analyze_resource_utilization()
        
        # Bottleneck identification
        bottleneck_analysis = self._identify_bottlenecks()
        
        return {
            "scalability_coefficient": self.scalability_coefficient,
            "efficiency_curve": self.efficiency_curve,
            "performance_analysis": performance_analysis,
            "resource_analysis": resource_analysis,
            "bottleneck_analysis": bottleneck_analysis,
            "breaking_point": self.breaking_point.__dict__ if self.breaking_point else None,
            "optimal_point": self.optimal_point.__dict__ if self.optimal_point else None
        }
        
    def _calculate_scalability_coefficient(self) -> float:
        """Calculate scalability coefficient (0-1, higher is better)"""
        if len(self.test_points) < 2:
            return 0.0
            
        # Calculate ideal vs actual scaling
        valid_points = [p for p in self.test_points if p.actual_tps > 0]
        
        if not valid_points:
            return 0.0
            
        # Linear regression to find scaling trend
        x = np.array([p.target_tps for p in valid_points])
        y = np.array([p.actual_tps for p in valid_points])
        
        # Perfect scaling would be y = x
        # Calculate R-squared for how close we are to perfect scaling
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - x) ** 2)
        
        if ss_tot == 0:
            return 1.0
            
        r_squared = 1 - (ss_res / ss_tot)
        
        # Adjust for efficiency (actual vs target)
        efficiency_factor = np.mean(y / x)
        
        return max(0.0, min(1.0, r_squared * efficiency_factor))
        
    def _generate_efficiency_curve(self) -> Dict[str, List[float]]:
        """Generate efficiency curve data"""
        curves = {
            "target_tps": [],
            "actual_tps": [],
            "efficiency": [],
            "latency_p99": [],
            "cpu_usage": [],
            "memory_usage": []
        }
        
        for point in self.test_points:
            curves["target_tps"].append(point.target_tps)
            curves["actual_tps"].append(point.actual_tps)
            curves["efficiency"].append(point.actual_tps / point.target_tps if point.target_tps > 0 else 0)
            curves["latency_p99"].append(point.latency_p99)
            curves["cpu_usage"].append(point.cpu_usage)
            curves["memory_usage"].append(point.memory_usage)
            
        return curves
        
    def _analyze_performance_characteristics(self) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        if not self.test_points:
            return {}
            
        # Throughput analysis
        max_throughput = max(p.actual_tps for p in self.test_points)
        avg_throughput = np.mean([p.actual_tps for p in self.test_points])
        
        # Latency analysis
        latencies = [p.latency_p99 for p in self.test_points if p.latency_p99 > 0]
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        avg_latency = np.mean(latencies) if latencies else 0
        
        # Error rate analysis
        error_rates = [p.error_rate for p in self.test_points]
        max_error_rate = max(error_rates)
        avg_error_rate = np.mean(error_rates)
        
        return {
            "throughput": {
                "max_tps": max_throughput,
                "avg_tps": avg_throughput,
                "throughput_range": max_throughput - min([p.actual_tps for p in self.test_points])
            },
            "latency": {
                "min_p99_ms": min_latency,
                "max_p99_ms": max_latency,
                "avg_p99_ms": avg_latency,
                "latency_increase": max_latency - min_latency
            },
            "reliability": {
                "max_error_rate": max_error_rate,
                "avg_error_rate": avg_error_rate,
                "error_rate_increase": max_error_rate - min(error_rates)
            }
        }
        
    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze resource utilization patterns"""
        if not self.test_points:
            return {}
            
        # CPU utilization analysis
        cpu_usage = [p.cpu_usage for p in self.test_points if p.cpu_usage > 0]
        max_cpu = max(cpu_usage) if cpu_usage else 0
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0
        
        # Memory utilization analysis
        memory_usage = [p.memory_usage for p in self.test_points if p.memory_usage > 0]
        max_memory = max(memory_usage) if memory_usage else 0
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        
        # Efficiency analysis
        efficiency_points = []
        for point in self.test_points:
            if point.cpu_usage > 0 and point.actual_tps > 0:
                efficiency_points.append(point.actual_tps / point.cpu_usage)
                
        max_efficiency = max(efficiency_points) if efficiency_points else 0
        avg_efficiency = np.mean(efficiency_points) if efficiency_points else 0
        
        return {
            "cpu": {
                "max_usage": max_cpu,
                "avg_usage": avg_cpu,
                "utilization_trend": "increasing" if cpu_usage and cpu_usage[-1] > cpu_usage[0] else "stable"
            },
            "memory": {
                "max_usage": max_memory,
                "avg_usage": avg_memory,
                "utilization_trend": "increasing" if memory_usage and memory_usage[-1] > memory_usage[0] else "stable"
            },
            "efficiency": {
                "max_tps_per_cpu": max_efficiency,
                "avg_tps_per_cpu": avg_efficiency,
                "efficiency_trend": "decreasing" if efficiency_points and efficiency_points[-1] < efficiency_points[0] else "stable"
            }
        }
        
    def _identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify system bottlenecks"""
        bottlenecks = {
            "cpu_bottleneck": False,
            "memory_bottleneck": False,
            "latency_bottleneck": False,
            "error_rate_bottleneck": False,
            "primary_bottleneck": None
        }
        
        if not self.test_points:
            return bottlenecks
            
        # Check for CPU bottleneck
        high_cpu_points = [p for p in self.test_points if p.cpu_usage > 80]
        if len(high_cpu_points) / len(self.test_points) > 0.3:
            bottlenecks["cpu_bottleneck"] = True
            
        # Check for memory bottleneck
        high_memory_points = [p for p in self.test_points if p.memory_usage > 80]
        if len(high_memory_points) / len(self.test_points) > 0.3:
            bottlenecks["memory_bottleneck"] = True
            
        # Check for latency bottleneck
        high_latency_points = [p for p in self.test_points if p.latency_p99 > self.config.latency_threshold_ms]
        if len(high_latency_points) / len(self.test_points) > 0.3:
            bottlenecks["latency_bottleneck"] = True
            
        # Check for error rate bottleneck
        high_error_points = [p for p in self.test_points if p.error_rate > self.config.error_rate_threshold]
        if len(high_error_points) / len(self.test_points) > 0.3:
            bottlenecks["error_rate_bottleneck"] = True
            
        # Identify primary bottleneck
        if bottlenecks["cpu_bottleneck"]:
            bottlenecks["primary_bottleneck"] = "cpu"
        elif bottlenecks["memory_bottleneck"]:
            bottlenecks["primary_bottleneck"] = "memory"
        elif bottlenecks["latency_bottleneck"]:
            bottlenecks["primary_bottleneck"] = "latency"
        elif bottlenecks["error_rate_bottleneck"]:
            bottlenecks["primary_bottleneck"] = "error_rate"
            
        return bottlenecks
        
    def _generate_recommendations(self) -> List[str]:
        """Generate scalability recommendations"""
        recommendations = []
        
        if not self.test_points:
            return ["Unable to generate recommendations - no test data available"]
            
        # Analyze scalability coefficient
        if self.scalability_coefficient < 0.7:
            recommendations.append(
                "Poor scalability detected. Consider horizontal scaling, "
                "load balancing optimization, or application architecture review."
            )
        elif self.scalability_coefficient > 0.9:
            recommendations.append(
                "Excellent scalability achieved. System scales well with load."
            )
            
        # Breaking point analysis
        if self.breaking_point:
            recommendations.append(
                f"Breaking point identified at {self.breaking_point.target_tps:,} TPS. "
                f"Consider this as maximum safe operating capacity."
            )
            
        # Optimal point analysis
        if self.optimal_point:
            recommendations.append(
                f"Optimal efficiency point at {self.optimal_point.target_tps:,} TPS. "
                f"Best price/performance ratio achieved at this load level."
            )
            
        # Resource utilization recommendations
        max_cpu = max(p.cpu_usage for p in self.test_points if p.cpu_usage > 0)
        max_memory = max(p.memory_usage for p in self.test_points if p.memory_usage > 0)
        
        if max_cpu > 90:
            recommendations.append(
                "High CPU utilization detected. Consider CPU optimization or scaling."
            )
            
        if max_memory > 90:
            recommendations.append(
                "High memory utilization detected. Consider memory optimization or scaling."
            )
            
        # Latency recommendations
        max_latency = max(p.latency_p99 for p in self.test_points if p.latency_p99 > 0)
        
        if max_latency > self.config.latency_threshold_ms:
            recommendations.append(
                f"High latency detected ({max_latency:.2f}ms). "
                f"Optimize application performance or increase resources."
            )
            
        return recommendations
        
    def _create_scalability_report(self, analysis: Dict[str, Any], 
                                 recommendations: List[str]) -> Dict[str, Any]:
        """Create comprehensive scalability report"""
        
        report = {
            "test_metadata": {
                "test_id": self.test_id,
                "timestamp": datetime.utcnow().isoformat(),
                "test_duration": sum(self.config.test_duration_seconds for _ in self.test_points),
                "total_test_points": len(self.test_points),
                "test_range": {
                    "min_tps": self.config.min_tps,
                    "max_tps": self.config.max_tps,
                    "step_size": self.config.step_size
                }
            },
            "scalability_analysis": analysis,
            "test_results": [point.__dict__ for point in self.test_points],
            "recommendations": recommendations,
            "summary": {
                "scalability_grade": self._calculate_scalability_grade(),
                "max_sustainable_tps": self._calculate_max_sustainable_tps(),
                "optimal_operating_point": self.optimal_point.__dict__ if self.optimal_point else None,
                "system_breaking_point": self.breaking_point.__dict__ if self.breaking_point else None
            }
        }
        
        # Log summary
        logger.info("=" * 80)
        logger.info("🔍 SCALABILITY VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Scalability Grade: {report['summary']['scalability_grade']}")
        logger.info(f"Max Sustainable TPS: {report['summary']['max_sustainable_tps']:,}")
        logger.info(f"Scalability Coefficient: {self.scalability_coefficient:.3f}")
        
        if self.optimal_point:
            logger.info(f"Optimal Point: {self.optimal_point.target_tps:,} TPS")
            
        if self.breaking_point:
            logger.info(f"Breaking Point: {self.breaking_point.target_tps:,} TPS")
            
        logger.info("=" * 80)
        
        return report
        
    def _calculate_scalability_grade(self) -> str:
        """Calculate scalability grade"""
        if self.scalability_coefficient >= 0.9:
            return "A+ (Excellent)"
        elif self.scalability_coefficient >= 0.8:
            return "A (Very Good)"
        elif self.scalability_coefficient >= 0.7:
            return "B (Good)"
        elif self.scalability_coefficient >= 0.6:
            return "C (Fair)"
        elif self.scalability_coefficient >= 0.5:
            return "D (Poor)"
        else:
            return "F (Failing)"
            
    def _calculate_max_sustainable_tps(self) -> int:
        """Calculate maximum sustainable TPS"""
        if not self.test_points:
            return 0
            
        # Find highest TPS that meets all thresholds
        sustainable_points = []
        
        for point in self.test_points:
            if (point.latency_p99 <= self.config.latency_threshold_ms and
                point.error_rate <= self.config.error_rate_threshold and
                point.actual_tps >= point.target_tps * self.config.stability_threshold):
                sustainable_points.append(point)
                
        if sustainable_points:
            return int(max(p.actual_tps for p in sustainable_points))
        else:
            return 0
            
    async def generate_scalability_visualizations(self, output_dir: str = "/tmp/scalability_charts"):
        """Generate scalability visualization charts"""
        logger.info("📊 Generating scalability visualizations")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Throughput vs Target chart
        await self._create_throughput_chart(output_dir)
        
        # Latency vs Load chart
        await self._create_latency_chart(output_dir)
        
        # Resource utilization chart
        await self._create_resource_chart(output_dir)
        
        # Efficiency curve chart
        await self._create_efficiency_chart(output_dir)
        
        logger.info(f"📈 Visualizations saved to {output_dir}")
        
    async def _create_throughput_chart(self, output_dir: str):
        """Create throughput vs target chart"""
        if not self.test_points:
            return
            
        target_tps = [p.target_tps for p in self.test_points]
        actual_tps = [p.actual_tps for p in self.test_points]
        
        plt.figure(figsize=(12, 8))
        plt.plot(target_tps, actual_tps, 'b-', linewidth=2, label='Actual TPS')
        plt.plot(target_tps, target_tps, 'r--', linewidth=2, label='Target TPS (Perfect Scaling)')
        
        plt.xlabel('Target TPS')
        plt.ylabel('Actual TPS')
        plt.title('Throughput Scalability Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark breaking point
        if self.breaking_point:
            plt.axvline(x=self.breaking_point.target_tps, color='red', linestyle=':', 
                       label=f'Breaking Point ({self.breaking_point.target_tps:,} TPS)')
            
        # Mark optimal point
        if self.optimal_point:
            plt.axvline(x=self.optimal_point.target_tps, color='green', linestyle=':', 
                       label=f'Optimal Point ({self.optimal_point.target_tps:,} TPS)')
            
        plt.legend()
        plt.savefig(f"{output_dir}/throughput_scalability.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    async def _create_latency_chart(self, output_dir: str):
        """Create latency vs load chart"""
        if not self.test_points:
            return
            
        target_tps = [p.target_tps for p in self.test_points]
        p50_latency = [p.latency_p50 for p in self.test_points]
        p95_latency = [p.latency_p95 for p in self.test_points]
        p99_latency = [p.latency_p99 for p in self.test_points]
        
        plt.figure(figsize=(12, 8))
        plt.plot(target_tps, p50_latency, 'g-', linewidth=2, label='P50 Latency')
        plt.plot(target_tps, p95_latency, 'b-', linewidth=2, label='P95 Latency')
        plt.plot(target_tps, p99_latency, 'r-', linewidth=2, label='P99 Latency')
        
        plt.axhline(y=self.config.latency_threshold_ms, color='red', linestyle='--', 
                   label=f'Latency Threshold ({self.config.latency_threshold_ms}ms)')
        
        plt.xlabel('Target TPS')
        plt.ylabel('Latency (ms)')
        plt.title('Latency vs Load Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{output_dir}/latency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    async def _create_resource_chart(self, output_dir: str):
        """Create resource utilization chart"""
        if not self.test_points:
            return
            
        target_tps = [p.target_tps for p in self.test_points]
        cpu_usage = [p.cpu_usage for p in self.test_points]
        memory_usage = [p.memory_usage for p in self.test_points]
        
        plt.figure(figsize=(12, 8))
        plt.plot(target_tps, cpu_usage, 'r-', linewidth=2, label='CPU Usage (%)')
        plt.plot(target_tps, memory_usage, 'b-', linewidth=2, label='Memory Usage (%)')
        
        plt.axhline(y=80, color='orange', linestyle='--', label='80% Threshold')
        plt.axhline(y=90, color='red', linestyle='--', label='90% Threshold')
        
        plt.xlabel('Target TPS')
        plt.ylabel('Resource Usage (%)')
        plt.title('Resource Utilization vs Load')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{output_dir}/resource_utilization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    async def _create_efficiency_chart(self, output_dir: str):
        """Create efficiency curve chart"""
        if not self.test_points:
            return
            
        target_tps = [p.target_tps for p in self.test_points]
        efficiency = [p.actual_tps / p.target_tps if p.target_tps > 0 else 0 for p in self.test_points]
        
        plt.figure(figsize=(12, 8))
        plt.plot(target_tps, efficiency, 'g-', linewidth=2, label='Efficiency Ratio')
        plt.axhline(y=self.config.stability_threshold, color='red', linestyle='--', 
                   label=f'Stability Threshold ({self.config.stability_threshold})')
        
        plt.xlabel('Target TPS')
        plt.ylabel('Efficiency Ratio (Actual/Target)')
        plt.title('System Efficiency Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{output_dir}/efficiency_curve.png", dpi=300, bbox_inches='tight')
        plt.close()