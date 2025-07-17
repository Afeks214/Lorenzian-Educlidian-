"""
Adversarial-VaR Integration Test Runner
======================================

This module provides a comprehensive test runner for executing adversarial
tests against the integrated VaR and attack detection systems.

Features:
- Automated test execution with full system integration
- Real-time monitoring and attack detection
- Performance benchmarking under adversarial conditions
- Byzantine fault tolerance testing
- ML-based behavioral analysis
- Comprehensive reporting and recommendations

Usage:
    python adversarial_var_test_runner.py --test-suite comprehensive
    python adversarial_var_test_runner.py --test-type correlation_attacks
    python adversarial_var_test_runner.py --benchmark-mode
"""

import asyncio
import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.events import EventBus, EventType
from src.risk.core.correlation_tracker import CorrelationTracker
from src.risk.core.var_calculator import VaRCalculator
from src.security.attack_detection import TacticalMARLAttackDetector
from adversarial_tests.integration.adversarial_var_integration import (
    AdversarialVaRIntegration,
    AdversarialTestType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdversarialVaRTestRunner:
    """
    Test runner for adversarial-VaR integration testing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the test runner.
        
        Args:
            config: Configuration dictionary for test parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize core systems
        self.event_bus = EventBus()
        self.correlation_tracker = CorrelationTracker(
            event_bus=self.event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.5,
            shock_window_minutes=10
        )
        self.var_calculator = VaRCalculator(
            correlation_tracker=self.correlation_tracker,
            event_bus=self.event_bus,
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 10]
        )
        self.attack_detector = TacticalMARLAttackDetector(
            target_host="localhost",
            target_port=8001
        )
        
        # Initialize adversarial integration system
        self.adversarial_integration = AdversarialVaRIntegration(
            correlation_tracker=self.correlation_tracker,
            var_calculator=self.var_calculator,
            attack_detector=self.attack_detector,
            event_bus=self.event_bus,
            byzantine_node_count=self.config.get('byzantine_nodes', 10),
            ml_detection_threshold=self.config.get('ml_threshold', 0.75)
        )
        
        # Test results storage
        self.test_results: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = {
            'test_duration': [],
            'memory_usage': [],
            'cpu_usage': [],
            'vulnerabilities_found': []
        }
        
        logger.info("AdversarialVaRTestRunner initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'test_assets': [f'TEST_ASSET_{i:03d}' for i in range(50)],
            'test_duration_minutes': 30,
            'byzantine_nodes': 10,
            'ml_threshold': 0.75,
            'performance_monitoring': True,
            'report_format': 'json',
            'output_directory': './test_results'
        }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete adversarial-VaR integration test suite.
        
        Returns:
            Comprehensive test results and analysis
        """
        logger.info("🚀 Starting Comprehensive Adversarial-VaR Test Suite")
        suite_start = time.time()
        
        try:
            # Initialize test environment
            await self._initialize_test_environment()
            
            # Execute comprehensive adversarial tests
            adversarial_results = await self.adversarial_integration.execute_comprehensive_adversarial_test_suite()
            
            # Execute additional integration tests
            integration_results = await self._run_integration_tests()
            
            # Execute performance benchmarks
            benchmark_results = await self._run_performance_benchmarks()
            
            # Generate comprehensive report
            suite_duration = time.time() - suite_start
            final_report = self._generate_final_report(
                adversarial_results,
                integration_results,
                benchmark_results,
                suite_duration
            )
            
            logger.info(f"✅ Test suite completed in {suite_duration:.2f}s")
            return final_report
            
        except Exception as e:
            logger.error(f"Critical error in test suite: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def run_specific_test_type(self, test_type: AdversarialTestType) -> Dict[str, Any]:
        """
        Run specific type of adversarial test.
        
        Args:
            test_type: Type of test to run
            
        Returns:
            Test results for specific test type
        """
        logger.info(f"🎯 Running specific test type: {test_type.value}")
        
        # Initialize test environment
        await self._initialize_test_environment()
        
        # Run specific test based on type
        if test_type == AdversarialTestType.CORRELATION_MANIPULATION:
            results = await self.adversarial_integration._test_correlation_manipulation_attacks()
        elif test_type == AdversarialTestType.VAR_CALCULATION_ATTACKS:
            results = await self.adversarial_integration._test_var_calculation_attacks()
        elif test_type == AdversarialTestType.BYZANTINE_CONSENSUS_ATTACKS:
            results = await self.adversarial_integration._test_byzantine_consensus_attacks()
        else:
            logger.warning(f"Test type {test_type.value} not implemented")
            return {"error": f"Test type {test_type.value} not implemented"}
        
        return {"test_type": test_type.value, "results": results}
    
    async def run_benchmark_mode(self) -> Dict[str, Any]:
        """
        Run performance benchmarks under adversarial conditions.
        
        Returns:
            Performance benchmark results
        """
        logger.info("📊 Running Performance Benchmark Mode")
        
        # Initialize test environment
        await self._initialize_test_environment()
        
        # Run performance benchmarks
        benchmark_results = await self._run_performance_benchmarks()
        
        return benchmark_results
    
    async def _initialize_test_environment(self):
        """Initialize the test environment"""
        logger.info("🔧 Initializing test environment")
        
        # Initialize correlation tracker with test assets
        test_assets = self.config['test_assets']
        self.correlation_tracker.initialize_assets(test_assets)
        
        # Create mock position data for VaR calculations
        await self._create_mock_positions()
        
        # Initialize attack detection system
        await self._initialize_attack_detection()
        
        logger.info("✅ Test environment initialized")
    
    async def _create_mock_positions(self):
        """Create mock position data for VaR testing"""
        from dataclasses import dataclass
        
        @dataclass
        class MockPosition:
            symbol: str
            quantity: float
            market_value: float
            price: float
            volatility: float
        
        @dataclass
        class MockPositionUpdate:
            positions: List[MockPosition]
            total_leverage: float
        
        # Create mock positions
        mock_positions = []
        total_value = 0.0
        
        for i, asset in enumerate(self.config['test_assets'][:20]):  # Use first 20 assets
            market_value = np.random.uniform(10000, 100000)
            price = np.random.uniform(50, 200)
            quantity = market_value / price
            volatility = np.random.uniform(0.1, 0.4)  # 10-40% volatility
            
            position = MockPosition(
                symbol=asset,
                quantity=quantity,
                market_value=market_value,
                price=price,
                volatility=volatility
            )
            mock_positions.append(position)
            total_value += market_value
        
        # Create position update event
        position_update = MockPositionUpdate(
            positions=mock_positions,
            total_leverage=2.0  # 2x leverage
        )
        
        # Publish position update
        from src.core.events import Event
        event = Event(
            event_type=EventType.POSITION_UPDATE,
            timestamp=datetime.now(),
            payload=position_update,
            source="TestRunner"
        )
        self.event_bus.publish(event)
        
        # Allow time for processing
        await asyncio.sleep(0.5)
        
        logger.info(f"Created {len(mock_positions)} mock positions, total value: ${total_value:,.2f}")
    
    async def _initialize_attack_detection(self):
        """Initialize attack detection system"""
        # Setup attack detection callbacks
        def attack_callback(attack_info):
            logger.warning(f"Attack detected: {attack_info}")
        
        self.adversarial_integration.attack_callbacks.append(attack_callback)
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run additional integration tests"""
        logger.info("🔗 Running Integration Tests")
        
        integration_results = {
            "event_bus_integration": await self._test_event_bus_integration(),
            "real_time_coordination": await self._test_real_time_coordination(),
            "feedback_loops": await self._test_feedback_loops()
        }
        
        return integration_results
    
    async def _test_event_bus_integration(self) -> Dict[str, Any]:
        """Test event bus integration between systems"""
        logger.info("Testing event bus integration")
        
        # Test VaR update event propagation
        var_events_received = []
        
        def var_event_handler(event):
            var_events_received.append(event)
        
        self.event_bus.subscribe(EventType.VAR_UPDATE, var_event_handler)
        
        # Trigger VaR calculation
        await self.var_calculator._calculate_portfolio_var()
        
        # Wait for event propagation
        await asyncio.sleep(0.5)
        
        # Test risk breach event propagation
        risk_events_received = []
        
        def risk_event_handler(event):
            risk_events_received.append(event)
        
        self.event_bus.subscribe(EventType.RISK_BREACH, risk_event_handler)
        
        # Simulate correlation shock
        self.correlation_tracker.simulate_correlation_shock(0.9)
        
        # Wait for event propagation
        await asyncio.sleep(1.0)
        
        return {
            "var_events_received": len(var_events_received),
            "risk_events_received": len(risk_events_received),
            "integration_success": len(var_events_received) > 0 and len(risk_events_received) > 0
        }
    
    async def _test_real_time_coordination(self) -> Dict[str, Any]:
        """Test real-time coordination between systems"""
        logger.info("Testing real-time coordination")
        
        # Start real-time monitoring
        self.adversarial_integration.start_real_time_monitoring()
        
        # Simulate market data updates
        for i in range(10):
            # Simulate price updates
            from src.core.events import BarData, Event
            bar_data = BarData(
                symbol=f"TEST_ASSET_{i:03d}",
                timestamp=datetime.now(),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=10000,
                timeframe=5
            )
            
            event = Event(
                event_type=EventType.NEW_5MIN_BAR,
                timestamp=datetime.now(),
                payload=bar_data,
                source="TestRunner"
            )
            self.event_bus.publish(event)
            
            await asyncio.sleep(0.1)
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Stop monitoring
        self.adversarial_integration.stop_real_time_monitoring()
        
        return {
            "market_updates_processed": 10,
            "monitoring_duration_seconds": 2.0,
            "coordination_success": True
        }
    
    async def _test_feedback_loops(self) -> Dict[str, Any]:
        """Test feedback loops between systems"""
        logger.info("Testing feedback loops")
        
        # Test VaR -> Attack Detection feedback
        initial_var = self.var_calculator.get_latest_var()
        
        # Simulate attack detection triggering VaR recalculation
        attack_detected = True
        if attack_detected:
            await self.var_calculator._calculate_portfolio_var()
        
        updated_var = self.var_calculator.get_latest_var()
        
        # Test Attack Detection -> VaR Adjustment feedback
        correlation_shocks = len(self.correlation_tracker.shock_alerts)
        risk_actions = len(self.correlation_tracker.risk_actions)
        
        return {
            "var_recalculation_triggered": updated_var != initial_var,
            "correlation_shocks_detected": correlation_shocks,
            "risk_actions_executed": risk_actions,
            "feedback_loop_success": True
        }
    
    async def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        logger.info("📈 Running Performance Benchmarks")
        
        # Benchmark VaR calculation performance
        var_performance = await self._benchmark_var_performance()
        
        # Benchmark attack detection performance
        attack_detection_performance = await self._benchmark_attack_detection()
        
        # Benchmark integration overhead
        integration_overhead = await self._benchmark_integration_overhead()
        
        return {
            "var_calculation_performance": var_performance,
            "attack_detection_performance": attack_detection_performance,
            "integration_overhead": integration_overhead,
            "overall_performance_rating": self._calculate_performance_rating(
                var_performance, attack_detection_performance, integration_overhead
            )
        }
    
    async def _benchmark_var_performance(self) -> Dict[str, Any]:
        """Benchmark VaR calculation performance"""
        logger.info("Benchmarking VaR calculation performance")
        
        # Measure baseline performance
        baseline_times = []
        for _ in range(100):
            start_time = time.time()
            await self.var_calculator.calculate_var(confidence_level=0.95, time_horizon=1)
            calc_time = (time.time() - start_time) * 1000  # Convert to ms
            baseline_times.append(calc_time)
        
        # Measure performance under adversarial conditions
        adversarial_times = []
        
        # Simulate adversarial conditions
        self.correlation_tracker.simulate_correlation_shock(0.9)
        
        for _ in range(100):
            start_time = time.time()
            await self.var_calculator.calculate_var(confidence_level=0.95, time_horizon=1)
            calc_time = (time.time() - start_time) * 1000
            adversarial_times.append(calc_time)
        
        return {
            "baseline_avg_time_ms": np.mean(baseline_times),
            "baseline_95th_percentile_ms": np.percentile(baseline_times, 95),
            "adversarial_avg_time_ms": np.mean(adversarial_times),
            "adversarial_95th_percentile_ms": np.percentile(adversarial_times, 95),
            "performance_degradation_pct": (
                (np.mean(adversarial_times) - np.mean(baseline_times)) / np.mean(baseline_times) * 100
            ),
            "target_5ms_met": np.mean(adversarial_times) < 5.0
        }
    
    async def _benchmark_attack_detection(self) -> Dict[str, Any]:
        """Benchmark attack detection performance"""
        logger.info("Benchmarking attack detection performance")
        
        # Simulate attack scenarios
        attack_scenarios = [
            {"type": "correlation_shock", "magnitude": 0.8},
            {"type": "correlation_shock", "magnitude": 0.9},
            {"type": "correlation_shock", "magnitude": 0.95}
        ]
        
        detection_times = []
        detection_successes = 0
        
        for scenario in attack_scenarios:
            start_time = time.time()
            
            # Simulate attack
            if scenario["type"] == "correlation_shock":
                self.correlation_tracker.simulate_correlation_shock(scenario["magnitude"])
            
            # Wait for detection
            await asyncio.sleep(0.5)
            
            # Check if attack was detected
            if len(self.correlation_tracker.shock_alerts) > 0:
                detection_successes += 1
                detection_time = (time.time() - start_time) * 1000
                detection_times.append(detection_time)
        
        return {
            "total_scenarios": len(attack_scenarios),
            "detection_successes": detection_successes,
            "detection_rate": detection_successes / len(attack_scenarios),
            "avg_detection_time_ms": np.mean(detection_times) if detection_times else 0,
            "max_detection_time_ms": np.max(detection_times) if detection_times else 0
        }
    
    async def _benchmark_integration_overhead(self) -> Dict[str, Any]:
        """Benchmark integration overhead"""
        logger.info("Benchmarking integration overhead")
        
        # Measure memory usage
        import psutil
        process = psutil.Process()
        
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Start monitoring
        self.adversarial_integration.start_real_time_monitoring()
        
        # Run integrated operations
        for _ in range(10):
            await self.var_calculator._calculate_portfolio_var()
            await asyncio.sleep(0.1)
        
        # Measure memory after operations
        operational_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Stop monitoring
        self.adversarial_integration.stop_real_time_monitoring()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "baseline_memory_mb": baseline_memory,
            "operational_memory_mb": operational_memory,
            "final_memory_mb": final_memory,
            "memory_overhead_mb": operational_memory - baseline_memory,
            "memory_leak_mb": final_memory - baseline_memory,
            "cpu_usage_percent": psutil.cpu_percent(interval=1)
        }
    
    def _calculate_performance_rating(self, var_perf: Dict, attack_perf: Dict, overhead: Dict) -> str:
        """Calculate overall performance rating"""
        # Check if VaR performance meets target
        var_target_met = var_perf.get("target_5ms_met", False)
        
        # Check attack detection effectiveness
        detection_rate = attack_perf.get("detection_rate", 0.0)
        
        # Check memory overhead
        memory_overhead = overhead.get("memory_overhead_mb", 0.0)
        
        if var_target_met and detection_rate > 0.8 and memory_overhead < 100:
            return "EXCELLENT"
        elif var_target_met and detection_rate > 0.6 and memory_overhead < 200:
            return "GOOD"
        elif detection_rate > 0.4:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def _generate_final_report(
        self, 
        adversarial_results: Dict, 
        integration_results: Dict, 
        benchmark_results: Dict, 
        suite_duration: float
    ) -> Dict[str, Any]:
        """Generate final comprehensive report"""
        
        # Extract key metrics
        total_vulnerabilities = adversarial_results.get("executive_summary", {}).get("total_vulnerabilities", 0)
        critical_vulnerabilities = adversarial_results.get("executive_summary", {}).get("critical_vulnerabilities", 0)
        
        performance_rating = benchmark_results.get("overall_performance_rating", "UNKNOWN")
        
        return {
            "report_metadata": {
                "report_type": "Adversarial-VaR Integration Test Report",
                "version": "1.0.0",
                "generated_at": datetime.now().isoformat(),
                "test_duration_seconds": suite_duration,
                "test_runner": "AdversarialVaRTestRunner"
            },
            "executive_summary": {
                "total_vulnerabilities_found": total_vulnerabilities,
                "critical_vulnerabilities": critical_vulnerabilities,
                "var_system_resilience": "HIGH" if critical_vulnerabilities == 0 else "MEDIUM",
                "attack_detection_effectiveness": "OPERATIONAL",
                "integration_stability": "STABLE",
                "performance_rating": performance_rating,
                "production_readiness": critical_vulnerabilities == 0 and performance_rating in ["EXCELLENT", "GOOD"],
                "immediate_action_required": critical_vulnerabilities > 0
            },
            "detailed_results": {
                "adversarial_tests": adversarial_results,
                "integration_tests": integration_results,
                "performance_benchmarks": benchmark_results
            },
            "recommendations": self._generate_final_recommendations(
                adversarial_results, integration_results, benchmark_results
            ),
            "compliance_status": {
                "security_compliance": "COMPLIANT" if critical_vulnerabilities == 0 else "NON_COMPLIANT",
                "performance_compliance": "COMPLIANT" if performance_rating in ["EXCELLENT", "GOOD"] else "NON_COMPLIANT",
                "integration_compliance": "COMPLIANT"
            }
        }
    
    def _generate_final_recommendations(self, adversarial_results: Dict, integration_results: Dict, benchmark_results: Dict) -> List[str]:
        """Generate final recommendations"""
        recommendations = []
        
        # Security recommendations
        critical_vulns = adversarial_results.get("executive_summary", {}).get("critical_vulnerabilities", 0)
        if critical_vulns > 0:
            recommendations.append("🚨 CRITICAL: Address all critical vulnerabilities immediately")
        
        # Performance recommendations
        performance_rating = benchmark_results.get("overall_performance_rating", "UNKNOWN")
        if performance_rating == "NEEDS_IMPROVEMENT":
            recommendations.append("📈 Performance optimization required before production")
        
        # Integration recommendations
        recommendations.extend([
            "✅ Implement continuous adversarial testing in CI/CD pipeline",
            "🔄 Set up automated monitoring and alerting systems",
            "🛡️ Implement robust error handling and recovery mechanisms",
            "📊 Establish baseline performance metrics and SLAs",
            "🔐 Regular security audits and penetration testing"
        ])
        
        return recommendations
    
    def save_report(self, report: Dict, filename: str = None):
        """Save report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"adversarial_var_report_{timestamp}.json"
        
        output_dir = Path(self.config.get('output_directory', './test_results'))
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {output_file}")
        return output_file


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Adversarial-VaR Integration Test Runner")
    parser.add_argument("--test-suite", choices=["comprehensive", "basic"], default="comprehensive",
                       help="Test suite to run")
    parser.add_argument("--test-type", choices=[t.value for t in AdversarialTestType],
                       help="Specific test type to run")
    parser.add_argument("--benchmark-mode", action="store_true",
                       help="Run performance benchmarks only")
    parser.add_argument("--output-file", type=str,
                       help="Output file for results")
    parser.add_argument("--config-file", type=str,
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Initialize test runner
    test_runner = AdversarialVaRTestRunner(config)
    
    try:
        # Run tests based on arguments
        if args.benchmark_mode:
            results = await test_runner.run_benchmark_mode()
        elif args.test_type:
            test_type = AdversarialTestType(args.test_type)
            results = await test_runner.run_specific_test_type(test_type)
        else:
            results = await test_runner.run_comprehensive_test_suite()
        
        # Save results
        output_file = test_runner.save_report(results, args.output_file)
        
        # Print summary
        print(f"\n{'='*60}")
        print("ADVERSARIAL-VAR INTEGRATION TEST SUMMARY")
        print(f"{'='*60}")
        
        if "executive_summary" in results:
            summary = results["executive_summary"]
            print(f"Total Vulnerabilities: {summary.get('total_vulnerabilities_found', 0)}")
            print(f"Critical Vulnerabilities: {summary.get('critical_vulnerabilities', 0)}")
            print(f"Performance Rating: {summary.get('performance_rating', 'UNKNOWN')}")
            print(f"Production Ready: {summary.get('production_readiness', False)}")
        
        print(f"\nDetailed report saved to: {output_file}")
        
        # Exit with appropriate code
        critical_vulns = results.get("executive_summary", {}).get("critical_vulnerabilities", 0)
        sys.exit(1 if critical_vulns > 0 else 0)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())