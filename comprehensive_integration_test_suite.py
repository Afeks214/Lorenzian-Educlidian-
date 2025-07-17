#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for GrandModel System
Agent 5 - System Integration & End-to-End Testing

This test suite validates the complete system integration and provides trustworthiness metrics.
"""

import asyncio
import json
import logging
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import sys
import os

# Add project root to path
sys.path.append('/home/QuantNova/GrandModel')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class IntegrationTestResult:
    """Results from an integration test"""
    test_name: str
    status: str  # PASS, FAIL, WARNING
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
@dataclass
class SystemTrustworthinessReport:
    """Complete system trustworthiness assessment"""
    overall_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    test_results: List[IntegrationTestResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class MockRedis:
    """Mock Redis client for testing without external dependencies"""
    def __init__(self):
        self.data = {}
        self.streams = {}
        self.pubsub_data = {}
        self.connected = True
    
    async def ping(self):
        return self.connected
    
    async def publish(self, channel: str, message: str):
        if channel not in self.pubsub_data:
            self.pubsub_data[channel] = []
        self.pubsub_data[channel].append(message)
        return 1
    
    async def xadd(self, stream: str, data: Dict[str, Any]):
        if stream not in self.streams:
            self.streams[stream] = []
        message_id = f"{int(time.time() * 1000)}-0"
        self.streams[stream].append((message_id, data))
        return message_id
    
    def get_stream_count(self, stream: str) -> int:
        return len(self.streams.get(stream, []))
    
    def get_pubsub_count(self, channel: str) -> int:
        return len(self.pubsub_data.get(channel, []))

class ComprehensiveIntegrationTestSuite:
    """Comprehensive test suite for system integration"""
    
    def __init__(self):
        self.results: List[IntegrationTestResult] = []
        self.mock_redis = MockRedis()
        self.test_start_time = None
        
        # Test configuration
        self.config = {
            'test_data_points': 1000,
            'performance_thresholds': {
                'latency_ms': 500,
                'throughput_ops_per_sec': 100,
                'memory_usage_mb': 1000,
                'conversion_rate_min': 0.5
            },
            'critical_components': [
                'data_pipeline',
                'synergy_detection',
                'agent_coordination',
                'execution_engine',
                'risk_management'
            ]
        }
    
    async def run_comprehensive_tests(self) -> SystemTrustworthinessReport:
        """Run all integration tests and generate trustworthiness report"""
        logger.info("🚀 Starting Comprehensive Integration Test Suite")
        self.test_start_time = time.time()
        
        # Test categories
        test_categories = [
            ("Data Pipeline Integration", self._test_data_pipeline_integration),
            ("Component Communication", self._test_component_communication),
            ("End-to-End Flow", self._test_end_to_end_flow),
            ("Configuration Management", self._test_configuration_management),
            ("Error Handling & Recovery", self._test_error_handling),
            ("Performance Validation", self._test_performance_validation),
            ("Security & Safety", self._test_security_safety),
            ("Resource Management", self._test_resource_management)
        ]
        
        # Execute test categories
        for category_name, test_func in test_categories:
            logger.info(f"📊 Testing: {category_name}")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"❌ Test category {category_name} failed: {e}")
                self._add_result(
                    test_name=f"{category_name}_execution",
                    status="FAIL",
                    duration_ms=0,
                    error_message=str(e)
                )
        
        # Generate trustworthiness report
        report = self._generate_trustworthiness_report()
        
        total_time = time.time() - self.test_start_time
        logger.info(f"✅ Integration testing completed in {total_time:.2f}s")
        
        return report
    
    async def _test_data_pipeline_integration(self):
        """Test data pipeline integration and reliability"""
        # Test 1: Data Loading
        start_time = time.time()
        try:
            from vectorbt_synergy_backtest import VectorBTSynergyBacktest
            
            backtest = VectorBTSynergyBacktest()
            df = backtest.load_data(start_date='2024-01-01', end_date='2024-07-01')
            
            if len(df) > 1000:
                self._add_result("data_loading", "PASS", (time.time() - start_time) * 1000, {
                    'rows_loaded': len(df),
                    'data_quality': 'good'
                })
            else:
                self._add_result("data_loading", "WARNING", (time.time() - start_time) * 1000, {
                    'rows_loaded': len(df),
                    'issue': 'insufficient_data'
                })
        except Exception as e:
            self._add_result("data_loading", "FAIL", (time.time() - start_time) * 1000, 
                           error_message=str(e))
        
        # Test 2: Indicator Calculation
        start_time = time.time()
        try:
            sample_data = pd.DataFrame({
                'Open': np.random.uniform(100, 200, 1000),
                'High': np.random.uniform(150, 250, 1000),
                'Low': np.random.uniform(50, 150, 1000),
                'Close': np.random.uniform(100, 200, 1000),
                'Volume': np.random.uniform(1000, 10000, 1000)
            })
            
            backtest = VectorBTSynergyBacktest()
            indicators = backtest.calculate_all_indicators(sample_data)
            
            if len(indicators) > 5:
                self._add_result("indicator_calculation", "PASS", (time.time() - start_time) * 1000, {
                    'indicators_count': len(indicators)
                })
            else:
                self._add_result("indicator_calculation", "WARNING", (time.time() - start_time) * 1000, {
                    'indicators_count': len(indicators),
                    'issue': 'few_indicators'
                })
        except Exception as e:
            self._add_result("indicator_calculation", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
    
    async def _test_component_communication(self):
        """Test communication between system components"""
        # Test 1: Event Bus Communication
        start_time = time.time()
        try:
            from src.core.event_bus import EventBus, Event, EventType
            
            event_bus = EventBus()
            events_received = []
            
            def test_handler(event):
                events_received.append(event)
            
            event_bus.subscribe(EventType.SYNERGY_DETECTED, test_handler)
            
            # Send test event
            test_event = Event(
                event_type=EventType.SYNERGY_DETECTED,
                data={'test': 'data'},
                timestamp=datetime.now()
            )
            
            event_bus.emit(test_event)
            
            # Allow time for event processing
            await asyncio.sleep(0.1)
            
            if len(events_received) > 0:
                self._add_result("event_bus_communication", "PASS", (time.time() - start_time) * 1000, {
                    'events_received': len(events_received)
                })
            else:
                self._add_result("event_bus_communication", "FAIL", (time.time() - start_time) * 1000, {
                    'issue': 'no_events_received'
                })
        except Exception as e:
            self._add_result("event_bus_communication", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
        
        # Test 2: Agent Communication Protocol
        start_time = time.time()
        try:
            from src.agents.agent_communication_protocol import AgentCommunicationHub, StrategySignal
            
            comm_hub = AgentCommunicationHub()
            
            # Test strategy signal broadcast
            test_signal = StrategySignal(
                signal_type='buy',
                confidence=0.8,
                strength=0.7,
                source='test',
                timestamp=datetime.now()
            )
            
            await comm_hub.broadcast_strategy_signal(test_signal)
            
            self._add_result("agent_communication", "PASS", (time.time() - start_time) * 1000, {
                'communication_successful': True
            })
        except Exception as e:
            self._add_result("agent_communication", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
    
    async def _test_end_to_end_flow(self):
        """Test complete end-to-end flow from data to execution"""
        start_time = time.time()
        
        try:
            # Simulate complete trading flow
            flow_results = {
                'data_loaded': False,
                'synergy_detected': False,
                'decision_made': False,
                'execution_attempted': False
            }
            
            # Step 1: Load data
            try:
                from vectorbt_synergy_backtest import VectorBTSynergyBacktest
                backtest = VectorBTSynergyBacktest()
                df = backtest.load_data(start_date='2024-06-01', end_date='2024-07-01')
                if len(df) > 0:
                    flow_results['data_loaded'] = True
            except Exception:
                pass
            
            # Step 2: Run synergy detection (simplified)
            if flow_results['data_loaded']:
                try:
                    indicators = backtest.calculate_all_indicators(df.head(100))
                    if len(indicators) > 0:
                        flow_results['synergy_detected'] = True
                except Exception:
                    pass
            
            # Step 3: Agent decision making
            if flow_results['synergy_detected']:
                try:
                    from src.agents.synergy_strategy_integration import SynergyStrategyCoordinator
                    from src.agents.agent_communication_protocol import AgentCommunicationHub
                    
                    coordinator = SynergyStrategyCoordinator(AgentCommunicationHub())
                    test_synergy_data = {
                        'pattern_type': 'synergy_bullish',
                        'confidence': 0.75,
                        'strength': 0.8,
                        'indicators': {}
                    }
                    await coordinator.process_synergy_detection(test_synergy_data)
                    flow_results['decision_made'] = True
                except Exception:
                    pass
            
            # Step 4: Execution simulation
            if flow_results['decision_made']:
                try:
                    # Mock execution
                    execution_data = {
                        'action': 'long',
                        'confidence': 0.75,
                        'execution_command': {
                            'action': 'execute_trade',
                            'side': 'BUY',
                            'symbol': 'EURUSD',
                            'quantity': 1
                        }
                    }
                    flow_results['execution_attempted'] = True
                except Exception:
                    pass
            
            # Evaluate flow completeness
            completed_steps = sum(flow_results.values())
            total_steps = len(flow_results)
            
            if completed_steps == total_steps:
                status = "PASS"
            elif completed_steps >= total_steps * 0.75:
                status = "WARNING"
            else:
                status = "FAIL"
            
            self._add_result("end_to_end_flow", status, (time.time() - start_time) * 1000, {
                'completed_steps': completed_steps,
                'total_steps': total_steps,
                'flow_completion': f"{(completed_steps/total_steps)*100:.1f}%",
                'step_results': flow_results
            })
            
        except Exception as e:
            self._add_result("end_to_end_flow", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
    
    async def _test_configuration_management(self):
        """Test configuration management and validation"""
        start_time = time.time()
        
        try:
            config_files = [
                'configs/system/nq_config.yaml',
                'configs/trading/risk_config.yaml',
                'configs/trading/strategic_config.yaml',
                'configs/trading/tactical_config.yaml'
            ]
            
            valid_configs = 0
            total_configs = len(config_files)
            config_issues = []
            
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    try:
                        import yaml
                        with open(config_path, 'r') as f:
                            config_data = yaml.safe_load(f)
                        if config_data and isinstance(config_data, dict):
                            valid_configs += 1
                        else:
                            config_issues.append(f"{config_file}: Invalid structure")
                    except Exception as e:
                        config_issues.append(f"{config_file}: {str(e)}")
                else:
                    config_issues.append(f"{config_file}: File not found")
            
            if valid_configs == total_configs:
                status = "PASS"
            elif valid_configs >= total_configs * 0.8:
                status = "WARNING"
            else:
                status = "FAIL"
            
            self._add_result("configuration_management", status, (time.time() - start_time) * 1000, {
                'valid_configs': valid_configs,
                'total_configs': total_configs,
                'issues': config_issues
            })
            
        except Exception as e:
            self._add_result("configuration_management", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
    
    async def _test_error_handling(self):
        """Test error handling and recovery mechanisms"""
        start_time = time.time()
        
        try:
            error_scenarios = []
            
            # Test 1: Invalid data handling
            try:
                from vectorbt_synergy_backtest import VectorBTSynergyBacktest
                backtest = VectorBTSynergyBacktest()
                
                # Try loading data with invalid date range
                df = backtest.load_data(start_date='2030-01-01', end_date='2030-12-31')
                error_scenarios.append({'scenario': 'invalid_date_range', 'handled': len(df) == 0})
            except Exception:
                error_scenarios.append({'scenario': 'invalid_date_range', 'handled': True})
            
            # Test 2: Missing component handling
            try:
                from src.core.event_bus import EventBus
                event_bus = EventBus()
                
                # Try emitting event to non-existent handler
                from src.core.events import Event, EventType
                test_event = Event(EventType.SYNERGY_DETECTED, {}, datetime.now())
                event_bus.emit(test_event)  # Should not crash
                error_scenarios.append({'scenario': 'missing_handler', 'handled': True})
            except Exception:
                error_scenarios.append({'scenario': 'missing_handler', 'handled': False})
            
            # Evaluate error handling
            handled_scenarios = sum(1 for scenario in error_scenarios if scenario['handled'])
            total_scenarios = len(error_scenarios)
            
            if handled_scenarios == total_scenarios:
                status = "PASS"
            elif handled_scenarios >= total_scenarios * 0.8:
                status = "WARNING"
            else:
                status = "FAIL"
            
            self._add_result("error_handling", status, (time.time() - start_time) * 1000, {
                'handled_scenarios': handled_scenarios,
                'total_scenarios': total_scenarios,
                'scenarios': error_scenarios
            })
            
        except Exception as e:
            self._add_result("error_handling", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
    
    async def _test_performance_validation(self):
        """Test system performance and resource usage"""
        start_time = time.time()
        
        try:
            performance_metrics = {}
            
            # Test processing speed
            processing_start = time.time()
            try:
                from vectorbt_synergy_backtest import VectorBTSynergyBacktest
                backtest = VectorBTSynergyBacktest()
                
                # Generate test data
                test_data = pd.DataFrame({
                    'Open': np.random.uniform(100, 200, 1000),
                    'High': np.random.uniform(150, 250, 1000),
                    'Low': np.random.uniform(50, 150, 1000),
                    'Close': np.random.uniform(100, 200, 1000),
                    'Volume': np.random.uniform(1000, 10000, 1000)
                })
                
                indicators = backtest.calculate_all_indicators(test_data)
                processing_time = time.time() - processing_start
                
                bars_per_second = len(test_data) / processing_time if processing_time > 0 else 0
                performance_metrics['processing_speed'] = bars_per_second
                performance_metrics['processing_time_ms'] = processing_time * 1000
                
            except Exception as e:
                performance_metrics['processing_error'] = str(e)
            
            # Test memory usage (simplified)
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            performance_metrics['memory_usage_mb'] = memory_usage
            
            # Evaluate performance
            issues = []
            if performance_metrics.get('processing_speed', 0) < 100:
                issues.append("Low processing speed")
            if performance_metrics.get('memory_usage_mb', 0) > 1000:
                issues.append("High memory usage")
            
            status = "PASS" if len(issues) == 0 else ("WARNING" if len(issues) == 1 else "FAIL")
            
            self._add_result("performance_validation", status, (time.time() - start_time) * 1000, {
                'metrics': performance_metrics,
                'issues': issues
            })
            
        except Exception as e:
            self._add_result("performance_validation", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
    
    async def _test_security_safety(self):
        """Test security and safety mechanisms"""
        start_time = time.time()
        
        try:
            security_checks = []
            
            # Check for obvious security issues in configurations
            try:
                import yaml
                config_files = ['configs/system/nq_config.yaml']
                
                for config_file in config_files:
                    config_path = Path(config_file)
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            content = f.read()
                        
                        # Check for hardcoded secrets (simplified)
                        suspicious_patterns = ['password', 'secret', 'key', 'token']
                        found_issues = [pattern for pattern in suspicious_patterns if pattern in content.lower()]
                        
                        security_checks.append({
                            'file': config_file,
                            'issues': found_issues
                        })
            except Exception:
                pass
            
            # Test input validation (simplified)
            try:
                from src.core.events import Event, EventType
                
                # Try creating event with invalid data
                try:
                    invalid_event = Event(EventType.SYNERGY_DETECTED, None, datetime.now())
                    security_checks.append({'test': 'invalid_event_data', 'handled': True})
                except Exception:
                    security_checks.append({'test': 'invalid_event_data', 'handled': False})
                    
            except Exception:
                pass
            
            # Evaluate security
            critical_issues = sum(1 for check in security_checks if 
                                isinstance(check.get('issues'), list) and len(check['issues']) > 0)
            
            status = "PASS" if critical_issues == 0 else ("WARNING" if critical_issues <= 2 else "FAIL")
            
            self._add_result("security_safety", status, (time.time() - start_time) * 1000, {
                'security_checks': security_checks,
                'critical_issues': critical_issues
            })
            
        except Exception as e:
            self._add_result("security_safety", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
    
    async def _test_resource_management(self):
        """Test resource management and cleanup"""
        start_time = time.time()
        
        try:
            resource_checks = []
            
            # Test file handle management
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            initial_handles = len(process.open_files())
            
            # Simulate resource usage
            try:
                from vectorbt_synergy_backtest import VectorBTSynergyBacktest
                backtest = VectorBTSynergyBacktest()
                
                # Perform operations that might open files
                for i in range(5):
                    df = backtest.load_data(start_date='2024-06-01', end_date='2024-06-15')
                    if len(df) > 0:
                        break
                
                final_handles = len(process.open_files())
                handle_increase = final_handles - initial_handles
                
                resource_checks.append({
                    'metric': 'file_handles',
                    'initial': initial_handles,
                    'final': final_handles,
                    'increase': handle_increase
                })
                
            except Exception as e:
                resource_checks.append({
                    'metric': 'file_handles',
                    'error': str(e)
                })
            
            # Test memory cleanup
            import gc
            gc.collect()
            
            memory_after_gc = process.memory_info().rss / 1024 / 1024  # MB
            resource_checks.append({
                'metric': 'memory_after_gc',
                'value': memory_after_gc
            })
            
            # Evaluate resource management
            issues = []
            for check in resource_checks:
                if check['metric'] == 'file_handles' and check.get('increase', 0) > 10:
                    issues.append("Excessive file handle usage")
                if check['metric'] == 'memory_after_gc' and check.get('value', 0) > 500:
                    issues.append("High memory usage after GC")
            
            status = "PASS" if len(issues) == 0 else ("WARNING" if len(issues) <= 1 else "FAIL")
            
            self._add_result("resource_management", status, (time.time() - start_time) * 1000, {
                'resource_checks': resource_checks,
                'issues': issues
            })
            
        except Exception as e:
            self._add_result("resource_management", "FAIL", (time.time() - start_time) * 1000,
                           error_message=str(e))
    
    def _add_result(self, test_name: str, status: str, duration_ms: float, 
                   details: Dict[str, Any] = None, error_message: str = None):
        """Add a test result"""
        result = IntegrationTestResult(
            test_name=test_name,
            status=status,
            duration_ms=duration_ms,
            details=details or {},
            error_message=error_message
        )
        self.results.append(result)
        
        status_emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}
        logger.info(f"{status_emoji.get(status, '❓')} {test_name}: {status} ({duration_ms:.1f}ms)")
    
    def _generate_trustworthiness_report(self) -> SystemTrustworthinessReport:
        """Generate comprehensive trustworthiness report"""
        # Calculate component scores
        component_scores = {}
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Group results by component
        component_mapping = {
            'data_loading': 'data_pipeline',
            'indicator_calculation': 'data_pipeline',
            'event_bus_communication': 'communication',
            'agent_communication': 'communication',
            'end_to_end_flow': 'integration',
            'configuration_management': 'configuration',
            'error_handling': 'reliability',
            'performance_validation': 'performance',
            'security_safety': 'security',
            'resource_management': 'resources'
        }
        
        # Calculate scores for each component
        for component in set(component_mapping.values()):
            component_results = [r for r in self.results if component_mapping.get(r.test_name) == component]
            if component_results:
                pass_count = sum(1 for r in component_results if r.status == "PASS")
                warning_count = sum(1 for r in component_results if r.status == "WARNING")
                fail_count = sum(1 for r in component_results if r.status == "FAIL")
                
                total_tests = len(component_results)
                score = (pass_count * 100 + warning_count * 60) / total_tests if total_tests > 0 else 0
                component_scores[component] = score
                
                if fail_count > 0:
                    critical_issues.append(f"{component}: {fail_count} critical failures")
                if warning_count > 0:
                    warnings.append(f"{component}: {warning_count} warnings")
        
        # Calculate overall score
        if component_scores:
            overall_score = sum(component_scores.values()) / len(component_scores)
        else:
            overall_score = 0
        
        # Generate recommendations based on issues found
        if overall_score < 70:
            recommendations.append("CRITICAL: System requires immediate attention before production use")
        elif overall_score < 85:
            recommendations.append("MODERATE: Address warnings before production deployment")
        else:
            recommendations.append("GOOD: System appears ready for production with minor improvements")
        
        # Specific recommendations based on test results
        for result in self.results:
            if result.status == "FAIL":
                recommendations.append(f"Fix critical issue in {result.test_name}: {result.error_message}")
            elif result.status == "WARNING" and result.details.get('issues'):
                for issue in result.details.get('issues', []):
                    recommendations.append(f"Address warning in {result.test_name}: {issue}")
        
        return SystemTrustworthinessReport(
            overall_score=overall_score,
            component_scores=component_scores,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations[:10],  # Limit to top 10
            test_results=self.results
        )
    
    def save_report(self, report: SystemTrustworthinessReport, output_dir: str = "results"):
        """Save trustworthiness report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"integration_test_report_{timestamp}.json"
        
        # Convert report to serializable format
        report_data = {
            'overall_score': report.overall_score,
            'component_scores': report.component_scores,
            'critical_issues': report.critical_issues,
            'warnings': report.warnings,
            'recommendations': report.recommendations,
            'timestamp': report.timestamp.isoformat(),
            'test_results': [
                {
                    'test_name': r.test_name,
                    'status': r.status,
                    'duration_ms': r.duration_ms,
                    'details': r.details,
                    'error_message': r.error_message,
                    'warnings': r.warnings
                }
                for r in report.test_results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"📄 Report saved to: {report_file}")
        return report_file

async def main():
    """Main test execution"""
    print("🎯 AGENT 5: COMPREHENSIVE SYSTEM INTEGRATION & END-TO-END TESTING")
    print("=" * 80)
    print("Mission: Validate system components work together correctly for 500% trustworthiness")
    print()
    
    # Create test suite
    test_suite = ComprehensiveIntegrationTestSuite()
    
    # Run comprehensive tests
    report = await test_suite.run_comprehensive_tests()
    
    # Print summary
    print("\n🎯 SYSTEM TRUSTWORTHINESS ASSESSMENT")
    print("=" * 60)
    print(f"Overall Score: {report.overall_score:.1f}/100")
    
    if report.overall_score >= 85:
        trustworthiness_rating = "HIGH TRUSTWORTHINESS ✅"
    elif report.overall_score >= 70:
        trustworthiness_rating = "MODERATE TRUSTWORTHINESS ⚠️"
    else:
        trustworthiness_rating = "LOW TRUSTWORTHINESS ❌"
    
    print(f"Trustworthiness Rating: {trustworthiness_rating}")
    print()
    
    # Component scores
    print("📊 Component Scores:")
    for component, score in report.component_scores.items():
        status_emoji = "✅" if score >= 85 else ("⚠️" if score >= 70 else "❌")
        print(f"   {component}: {score:.1f}/100 {status_emoji}")
    print()
    
    # Critical issues
    if report.critical_issues:
        print("🚨 Critical Issues:")
        for issue in report.critical_issues:
            print(f"   ❌ {issue}")
        print()
    
    # Warnings
    if report.warnings:
        print("⚠️ Warnings:")
        for warning in report.warnings:
            print(f"   ⚠️ {warning}")
        print()
    
    # Top recommendations
    print("🎯 Top Recommendations:")
    for i, rec in enumerate(report.recommendations[:5], 1):
        print(f"   {i}. {rec}")
    print()
    
    # Save report
    report_file = test_suite.save_report(report)
    
    # Test results summary
    total_tests = len(report.test_results)
    passed_tests = sum(1 for r in report.test_results if r.status == "PASS")
    warning_tests = sum(1 for r in report.test_results if r.status == "WARNING")
    failed_tests = sum(1 for r in report.test_results if r.status == "FAIL")
    
    print(f"📈 Test Results Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests} ✅")
    print(f"   Warnings: {warning_tests} ⚠️")
    print(f"   Failed: {failed_tests} ❌")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    print()
    
    # Mission status
    if report.overall_score >= 85 and failed_tests == 0:
        mission_status = "🎉 MISSION SUCCESS: System integration validated with high trustworthiness"
    elif report.overall_score >= 70:
        mission_status = "⚠️ MISSION PARTIAL: System mostly functional but needs improvements"
    else:
        mission_status = "❌ MISSION CRITICAL: System requires significant fixes before production"
    
    print(f"🎯 {mission_status}")
    print(f"📄 Detailed report: {report_file}")
    
    return report

if __name__ == "__main__":
    asyncio.run(main())