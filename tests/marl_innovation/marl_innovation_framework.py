"""
MARL Innovation Framework - Core Testing Infrastructure
======================================================

This module provides the core framework for advanced MARL testing with comprehensive
multi-agent validation, emergent behavior detection, and coordination testing.

Key Features:
- Unified testing interface for all MARL components
- Advanced multi-agent interaction validation
- Real-time emergent behavior detection
- Comprehensive coordination testing
- Performance benchmarking and reporting
- Integration with existing test infrastructure

Author: Agent Delta - MARL Testing Innovation Specialist
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import pytest

# Core testing imports
from .multi_agent_interaction_validator import MultiAgentInteractionValidator
from .emergent_behavior_detector import EmergentBehaviorDetector
from .agent_coordination_tester import AgentCoordinationTester
from .adversarial_marl_tester import AdversarialMARLTester
from .config.marl_test_config import MARLTestConfig
from .utils.marl_test_utils import MARLTestUtils

# System imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

logger = logging.getLogger(__name__)


class TestPhase(Enum):
    """Test execution phases."""
    INITIALIZATION = "initialization"
    INTERACTION_VALIDATION = "interaction_validation"
    EMERGENT_BEHAVIOR = "emergent_behavior"
    COORDINATION_TESTING = "coordination_testing"
    ADVERSARIAL_TESTING = "adversarial_testing"
    PERFORMANCE_VALIDATION = "performance_validation"
    REPORTING = "reporting"


@dataclass
class MARLTestResult:
    """Comprehensive MARL test result structure."""
    test_id: str
    test_name: str
    phase: TestPhase
    timestamp: datetime
    duration_ms: float
    success: bool
    score: float
    metrics: Dict[str, Any]
    details: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class MARLTestSession:
    """Test session management and tracking."""
    session_id: str
    start_time: datetime
    config: MARLTestConfig
    results: List[MARLTestResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    success_rate: float = 0.0
    overall_score: float = 0.0
    status: str = "initialized"


class MARLInnovationFramework:
    """
    Advanced MARL Testing Innovation Framework.
    
    This framework provides comprehensive testing capabilities for multi-agent
    reinforcement learning systems with focus on:
    - Complex multi-agent interactions
    - Emergent behavior patterns
    - Coordination protocol validation
    - Adversarial robustness testing
    """
    
    def __init__(self, config: Optional[MARLTestConfig] = None):
        """Initialize the MARL Innovation Framework."""
        self.config = config or MARLTestConfig()
        self.session = None
        self.validators = {}
        self.utils = MARLTestUtils()
        
        # Initialize core components
        self._initialize_components()
        
        # Performance tracking
        self.performance_metrics = {
            'total_tests_run': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'average_test_duration_ms': 0.0,
            'total_agents_tested': 0,
            'interaction_patterns_detected': 0,
            'emergent_behaviors_found': 0,
            'coordination_issues_identified': 0,
            'adversarial_vulnerabilities_found': 0
        }
        
        logger.info("MARL Innovation Framework initialized successfully")
    
    def _initialize_components(self):
        """Initialize core testing components."""
        try:
            # Multi-agent interaction validator
            self.validators['interaction'] = MultiAgentInteractionValidator(
                config=self.config.interaction_config
            )
            
            # Emergent behavior detector
            self.validators['emergent'] = EmergentBehaviorDetector(
                config=self.config.emergent_config
            )
            
            # Agent coordination tester
            self.validators['coordination'] = AgentCoordinationTester(
                config=self.config.coordination_config
            )
            
            # Adversarial MARL tester
            self.validators['adversarial'] = AdversarialMARLTester(
                config=self.config.adversarial_config
            )
            
            logger.info("All MARL testing components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MARL components: {str(e)}")
            raise
    
    async def start_test_session(self, session_name: str = None) -> str:
        """Start a new MARL test session."""
        session_id = f"marl_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if session_name:
            session_id = f"{session_name}_{session_id}"
        
        self.session = MARLTestSession(
            session_id=session_id,
            start_time=datetime.now(),
            config=self.config
        )
        
        logger.info(f"Started MARL test session: {session_id}")
        return session_id
    
    async def run_comprehensive_test_suite(self, 
                                         agent_system: Any, 
                                         test_data: Dict[str, Any],
                                         custom_tests: List[str] = None) -> MARLTestSession:
        """
        Run comprehensive MARL test suite.
        
        Args:
            agent_system: The MARL system to test
            test_data: Test data and scenarios
            custom_tests: Optional list of specific tests to run
            
        Returns:
            Complete test session with results
        """
        if not self.session:
            await self.start_test_session()
        
        self.session.status = "running"
        start_time = time.time()
        
        try:
            # Define test sequence
            test_sequence = custom_tests or [
                'interaction_validation',
                'emergent_behavior_detection',
                'coordination_testing',
                'adversarial_testing',
                'performance_validation'
            ]
            
            # Execute test phases
            for test_name in test_sequence:
                await self._execute_test_phase(test_name, agent_system, test_data)
            
            # Calculate final metrics
            self._calculate_session_metrics()
            
            self.session.status = "completed"
            self.session.total_duration_ms = (time.time() - start_time) * 1000
            
            logger.info(f"MARL test session completed: {self.session.session_id}")
            logger.info(f"Success rate: {self.session.success_rate:.2%}")
            logger.info(f"Overall score: {self.session.overall_score:.3f}")
            
        except Exception as e:
            self.session.status = "failed"
            logger.error(f"Test session failed: {str(e)}")
            raise
        
        return self.session
    
    async def _execute_test_phase(self, 
                                 test_name: str, 
                                 agent_system: Any, 
                                 test_data: Dict[str, Any]) -> MARLTestResult:
        """Execute a specific test phase."""
        phase_start = time.time()
        test_id = f"{test_name}_{len(self.session.results)}"
        
        logger.info(f"Starting test phase: {test_name}")
        
        try:
            # Route to appropriate validator
            if test_name == 'interaction_validation':
                result = await self._test_interaction_validation(test_id, agent_system, test_data)
            elif test_name == 'emergent_behavior_detection':
                result = await self._test_emergent_behavior_detection(test_id, agent_system, test_data)
            elif test_name == 'coordination_testing':
                result = await self._test_coordination(test_id, agent_system, test_data)
            elif test_name == 'adversarial_testing':
                result = await self._test_adversarial_robustness(test_id, agent_system, test_data)
            elif test_name == 'performance_validation':
                result = await self._test_performance_validation(test_id, agent_system, test_data)
            else:
                raise ValueError(f"Unknown test phase: {test_name}")
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Add to session results
            self.session.results.append(result)
            
            logger.info(f"Test phase {test_name} completed: {'SUCCESS' if result.success else 'FAILED'}")
            
        except Exception as e:
            # Create error result
            result = MARLTestResult(
                test_id=test_id,
                test_name=test_name,
                phase=TestPhase(test_name.lower()),
                timestamp=datetime.now(),
                duration_ms=(time.time() - phase_start) * 1000,
                success=False,
                score=0.0,
                metrics={},
                details={'error': str(e)},
                errors=[str(e)]
            )
            
            self.session.results.append(result)
            logger.error(f"Test phase {test_name} failed: {str(e)}")
        
        return result
    
    async def _test_interaction_validation(self, 
                                         test_id: str, 
                                         agent_system: Any, 
                                         test_data: Dict[str, Any]) -> MARLTestResult:
        """Test multi-agent interaction validation."""
        start_time = time.time()
        
        try:
            validator = self.validators['interaction']
            
            # Run interaction validation
            validation_result = await validator.validate_agent_interactions(
                agent_system=agent_system,
                test_scenarios=test_data.get('interaction_scenarios', []),
                validation_config=self.config.interaction_config
            )
            
            # Calculate metrics
            metrics = {
                'interaction_coverage': validation_result.get('coverage', 0.0),
                'interaction_quality_score': validation_result.get('quality_score', 0.0),
                'protocol_compliance': validation_result.get('protocol_compliance', 0.0),
                'communication_efficiency': validation_result.get('communication_efficiency', 0.0),
                'deadlock_incidents': validation_result.get('deadlock_incidents', 0),
                'message_loss_rate': validation_result.get('message_loss_rate', 0.0)
            }
            
            # Determine success and score
            success = (
                metrics['interaction_coverage'] >= 0.9 and
                metrics['interaction_quality_score'] >= 0.8 and
                metrics['deadlock_incidents'] == 0
            )
            
            score = (
                metrics['interaction_coverage'] * 0.3 +
                metrics['interaction_quality_score'] * 0.3 +
                metrics['protocol_compliance'] * 0.2 +
                metrics['communication_efficiency'] * 0.2
            )
            
            # Generate recommendations
            recommendations = []
            if metrics['interaction_coverage'] < 0.9:
                recommendations.append("Improve interaction coverage by testing more agent combinations")
            if metrics['deadlock_incidents'] > 0:
                recommendations.append("Implement deadlock detection and resolution mechanisms")
            if metrics['message_loss_rate'] > 0.01:
                recommendations.append("Improve message reliability and error handling")
            
            result = MARLTestResult(
                test_id=test_id,
                test_name="Multi-Agent Interaction Validation",
                phase=TestPhase.INTERACTION_VALIDATION,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                success=success,
                score=score,
                metrics=metrics,
                details=validation_result,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Interaction validation failed: {str(e)}")
            raise
    
    async def _test_emergent_behavior_detection(self, 
                                              test_id: str, 
                                              agent_system: Any, 
                                              test_data: Dict[str, Any]) -> MARLTestResult:
        """Test emergent behavior detection."""
        start_time = time.time()
        
        try:
            detector = self.validators['emergent']
            
            # Run emergent behavior detection
            detection_result = await detector.detect_emergent_behaviors(
                agent_system=agent_system,
                observation_window=test_data.get('observation_window', 1000),
                behavior_patterns=test_data.get('known_patterns', []),
                detection_config=self.config.emergent_config
            )
            
            # Calculate metrics
            metrics = {
                'behaviors_detected': detection_result.get('behaviors_detected', 0),
                'novel_behaviors': detection_result.get('novel_behaviors', 0),
                'pattern_recognition_accuracy': detection_result.get('pattern_accuracy', 0.0),
                'behavioral_diversity': detection_result.get('behavioral_diversity', 0.0),
                'convergence_analysis': detection_result.get('convergence_analysis', {}),
                'stability_score': detection_result.get('stability_score', 0.0)
            }
            
            # Determine success and score
            success = (
                metrics['behaviors_detected'] > 0 and
                metrics['pattern_recognition_accuracy'] >= 0.8 and
                metrics['stability_score'] >= 0.7
            )
            
            score = (
                min(metrics['behaviors_detected'] / 5, 1.0) * 0.3 +
                metrics['pattern_recognition_accuracy'] * 0.3 +
                metrics['behavioral_diversity'] * 0.2 +
                metrics['stability_score'] * 0.2
            )
            
            # Generate recommendations
            recommendations = []
            if metrics['novel_behaviors'] > 0:
                recommendations.append("Investigate novel behaviors for potential system improvements")
            if metrics['stability_score'] < 0.7:
                recommendations.append("Improve system stability to reduce behavioral variance")
            if metrics['behavioral_diversity'] < 0.5:
                recommendations.append("Encourage behavioral diversity through exploration mechanisms")
            
            result = MARLTestResult(
                test_id=test_id,
                test_name="Emergent Behavior Detection",
                phase=TestPhase.EMERGENT_BEHAVIOR,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                success=success,
                score=score,
                metrics=metrics,
                details=detection_result,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Emergent behavior detection failed: {str(e)}")
            raise
    
    async def _test_coordination(self, 
                               test_id: str, 
                               agent_system: Any, 
                               test_data: Dict[str, Any]) -> MARLTestResult:
        """Test agent coordination mechanisms."""
        start_time = time.time()
        
        try:
            tester = self.validators['coordination']
            
            # Run coordination testing
            coordination_result = await tester.test_coordination_mechanisms(
                agent_system=agent_system,
                coordination_scenarios=test_data.get('coordination_scenarios', []),
                performance_metrics=test_data.get('performance_metrics', {}),
                coordination_config=self.config.coordination_config
            )
            
            # Calculate metrics
            metrics = {
                'coordination_efficiency': coordination_result.get('efficiency', 0.0),
                'consensus_achievement_rate': coordination_result.get('consensus_rate', 0.0),
                'coordination_latency_ms': coordination_result.get('latency_ms', 0.0),
                'resource_utilization': coordination_result.get('resource_utilization', 0.0),
                'conflict_resolution_success': coordination_result.get('conflict_resolution', 0.0),
                'scalability_score': coordination_result.get('scalability_score', 0.0)
            }
            
            # Determine success and score
            success = (
                metrics['coordination_efficiency'] >= 0.8 and
                metrics['consensus_achievement_rate'] >= 0.9 and
                metrics['coordination_latency_ms'] <= 100.0
            )
            
            score = (
                metrics['coordination_efficiency'] * 0.3 +
                metrics['consensus_achievement_rate'] * 0.3 +
                (1.0 - min(metrics['coordination_latency_ms'] / 100.0, 1.0)) * 0.2 +
                metrics['conflict_resolution_success'] * 0.2
            )
            
            # Generate recommendations
            recommendations = []
            if metrics['coordination_latency_ms'] > 100.0:
                recommendations.append("Optimize coordination protocols to reduce latency")
            if metrics['consensus_achievement_rate'] < 0.9:
                recommendations.append("Improve consensus mechanisms and fault tolerance")
            if metrics['scalability_score'] < 0.7:
                recommendations.append("Enhance system scalability for larger agent populations")
            
            result = MARLTestResult(
                test_id=test_id,
                test_name="Agent Coordination Testing",
                phase=TestPhase.COORDINATION_TESTING,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                success=success,
                score=score,
                metrics=metrics,
                details=coordination_result,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Coordination testing failed: {str(e)}")
            raise
    
    async def _test_adversarial_robustness(self, 
                                         test_id: str, 
                                         agent_system: Any, 
                                         test_data: Dict[str, Any]) -> MARLTestResult:
        """Test adversarial robustness of MARL system."""
        start_time = time.time()
        
        try:
            tester = self.validators['adversarial']
            
            # Run adversarial testing
            adversarial_result = await tester.test_adversarial_robustness(
                agent_system=agent_system,
                attack_scenarios=test_data.get('attack_scenarios', []),
                defense_mechanisms=test_data.get('defense_mechanisms', []),
                adversarial_config=self.config.adversarial_config
            )
            
            # Calculate metrics
            metrics = {
                'robustness_score': adversarial_result.get('robustness_score', 0.0),
                'attacks_defended': adversarial_result.get('attacks_defended', 0),
                'attacks_successful': adversarial_result.get('attacks_successful', 0),
                'defense_effectiveness': adversarial_result.get('defense_effectiveness', 0.0),
                'recovery_time_ms': adversarial_result.get('recovery_time_ms', 0.0),
                'false_positive_rate': adversarial_result.get('false_positive_rate', 0.0)
            }
            
            # Determine success and score
            total_attacks = metrics['attacks_defended'] + metrics['attacks_successful']
            defense_rate = metrics['attacks_defended'] / max(total_attacks, 1)
            
            success = (
                defense_rate >= 0.9 and
                metrics['robustness_score'] >= 0.8 and
                metrics['false_positive_rate'] <= 0.05
            )
            
            score = (
                metrics['robustness_score'] * 0.4 +
                defense_rate * 0.3 +
                metrics['defense_effectiveness'] * 0.2 +
                (1.0 - metrics['false_positive_rate']) * 0.1
            )
            
            # Generate recommendations
            recommendations = []
            if defense_rate < 0.9:
                recommendations.append("Strengthen defense mechanisms against adversarial attacks")
            if metrics['recovery_time_ms'] > 1000:
                recommendations.append("Improve system recovery time after attacks")
            if metrics['false_positive_rate'] > 0.05:
                recommendations.append("Reduce false positive rate in attack detection")
            
            result = MARLTestResult(
                test_id=test_id,
                test_name="Adversarial Robustness Testing",
                phase=TestPhase.ADVERSARIAL_TESTING,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                success=success,
                score=score,
                metrics=metrics,
                details=adversarial_result,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Adversarial testing failed: {str(e)}")
            raise
    
    async def _test_performance_validation(self, 
                                         test_id: str, 
                                         agent_system: Any, 
                                         test_data: Dict[str, Any]) -> MARLTestResult:
        """Test performance validation of MARL system."""
        start_time = time.time()
        
        try:
            # Performance benchmarking
            performance_result = await self._benchmark_performance(
                agent_system=agent_system,
                benchmark_scenarios=test_data.get('benchmark_scenarios', []),
                performance_targets=test_data.get('performance_targets', {})
            )
            
            # Calculate metrics
            metrics = {
                'throughput_ops_per_sec': performance_result.get('throughput', 0.0),
                'latency_p99_ms': performance_result.get('latency_p99', 0.0),
                'memory_usage_mb': performance_result.get('memory_usage', 0.0),
                'cpu_utilization_percent': performance_result.get('cpu_utilization', 0.0),
                'scalability_factor': performance_result.get('scalability_factor', 1.0),
                'error_rate': performance_result.get('error_rate', 0.0)
            }
            
            # Determine success and score based on performance targets
            targets = test_data.get('performance_targets', {})
            success = (
                metrics['throughput_ops_per_sec'] >= targets.get('min_throughput', 100) and
                metrics['latency_p99_ms'] <= targets.get('max_latency_ms', 100) and
                metrics['error_rate'] <= targets.get('max_error_rate', 0.01)
            )
            
            # Calculate performance score
            throughput_score = min(metrics['throughput_ops_per_sec'] / targets.get('min_throughput', 100), 1.0)
            latency_score = max(0, 1.0 - metrics['latency_p99_ms'] / targets.get('max_latency_ms', 100))
            error_score = max(0, 1.0 - metrics['error_rate'] / targets.get('max_error_rate', 0.01))
            
            score = (throughput_score * 0.4 + latency_score * 0.4 + error_score * 0.2)
            
            # Generate recommendations
            recommendations = []
            if metrics['throughput_ops_per_sec'] < targets.get('min_throughput', 100):
                recommendations.append("Optimize system throughput through parallelization")
            if metrics['latency_p99_ms'] > targets.get('max_latency_ms', 100):
                recommendations.append("Reduce system latency through caching and optimization")
            if metrics['memory_usage_mb'] > targets.get('max_memory_mb', 1000):
                recommendations.append("Optimize memory usage and implement garbage collection")
            
            result = MARLTestResult(
                test_id=test_id,
                test_name="Performance Validation",
                phase=TestPhase.PERFORMANCE_VALIDATION,
                timestamp=datetime.now(),
                duration_ms=(time.time() - start_time) * 1000,
                success=success,
                score=score,
                metrics=metrics,
                details=performance_result,
                recommendations=recommendations
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Performance validation failed: {str(e)}")
            raise
    
    async def _benchmark_performance(self, 
                                   agent_system: Any, 
                                   benchmark_scenarios: List[Dict], 
                                   performance_targets: Dict) -> Dict[str, Any]:
        """Run performance benchmarks on MARL system."""
        # Placeholder for actual performance benchmarking
        # In production, this would run actual performance tests
        
        results = {
            'throughput': np.random.uniform(50, 200),
            'latency_p99': np.random.uniform(10, 150),
            'memory_usage': np.random.uniform(200, 800),
            'cpu_utilization': np.random.uniform(40, 90),
            'scalability_factor': np.random.uniform(1.0, 3.0),
            'error_rate': np.random.uniform(0.001, 0.02)
        }
        
        return results
    
    def _update_performance_metrics(self, result: MARLTestResult):
        """Update framework performance metrics."""
        self.performance_metrics['total_tests_run'] += 1
        
        if result.success:
            self.performance_metrics['successful_tests'] += 1
        else:
            self.performance_metrics['failed_tests'] += 1
        
        # Update specific metrics based on test type
        if result.phase == TestPhase.INTERACTION_VALIDATION:
            self.performance_metrics['interaction_patterns_detected'] += result.metrics.get('interaction_coverage', 0)
        elif result.phase == TestPhase.EMERGENT_BEHAVIOR:
            self.performance_metrics['emergent_behaviors_found'] += result.metrics.get('behaviors_detected', 0)
        elif result.phase == TestPhase.COORDINATION_TESTING:
            if result.metrics.get('consensus_achievement_rate', 0) < 0.9:
                self.performance_metrics['coordination_issues_identified'] += 1
        elif result.phase == TestPhase.ADVERSARIAL_TESTING:
            self.performance_metrics['adversarial_vulnerabilities_found'] += result.metrics.get('attacks_successful', 0)
        
        # Update average test duration
        total_duration = self.performance_metrics['average_test_duration_ms'] * (self.performance_metrics['total_tests_run'] - 1)
        self.performance_metrics['average_test_duration_ms'] = (total_duration + result.duration_ms) / self.performance_metrics['total_tests_run']
    
    def _calculate_session_metrics(self):
        """Calculate overall session metrics."""
        if not self.session.results:
            return
        
        # Calculate success rate
        successful_tests = sum(1 for r in self.session.results if r.success)
        self.session.success_rate = successful_tests / len(self.session.results)
        
        # Calculate overall score
        total_score = sum(r.score for r in self.session.results)
        self.session.overall_score = total_score / len(self.session.results)
    
    def generate_comprehensive_report(self, 
                                    output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.session:
            raise ValueError("No active test session")
        
        report = {
            'session_info': {
                'session_id': self.session.session_id,
                'start_time': self.session.start_time.isoformat(),
                'duration_ms': self.session.total_duration_ms,
                'status': self.session.status,
                'success_rate': self.session.success_rate,
                'overall_score': self.session.overall_score
            },
            'test_results': [],
            'performance_metrics': self.performance_metrics,
            'summary': {
                'total_tests': len(self.session.results),
                'successful_tests': sum(1 for r in self.session.results if r.success),
                'failed_tests': sum(1 for r in self.session.results if not r.success),
                'average_score': self.session.overall_score,
                'key_findings': [],
                'recommendations': []
            }
        }
        
        # Add detailed test results
        for result in self.session.results:
            report['test_results'].append({
                'test_id': result.test_id,
                'test_name': result.test_name,
                'phase': result.phase.value,
                'success': result.success,
                'score': result.score,
                'duration_ms': result.duration_ms,
                'metrics': result.metrics,
                'recommendations': result.recommendations,
                'warnings': result.warnings,
                'errors': result.errors
            })
        
        # Generate key findings and recommendations
        report['summary']['key_findings'] = self._generate_key_findings()
        report['summary']['recommendations'] = self._generate_recommendations()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Comprehensive report saved to: {output_path}")
        
        return report
    
    def _generate_key_findings(self) -> List[str]:
        """Generate key findings from test results."""
        findings = []
        
        # Analyze test results for key findings
        for result in self.session.results:
            if result.phase == TestPhase.INTERACTION_VALIDATION:
                if result.metrics.get('deadlock_incidents', 0) > 0:
                    findings.append(f"Deadlock incidents detected: {result.metrics['deadlock_incidents']}")
            elif result.phase == TestPhase.EMERGENT_BEHAVIOR:
                if result.metrics.get('novel_behaviors', 0) > 0:
                    findings.append(f"Novel emergent behaviors discovered: {result.metrics['novel_behaviors']}")
            elif result.phase == TestPhase.ADVERSARIAL_TESTING:
                if result.metrics.get('attacks_successful', 0) > 0:
                    findings.append(f"Successful adversarial attacks: {result.metrics['attacks_successful']}")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate consolidated recommendations."""
        all_recommendations = []
        
        for result in self.session.results:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        return unique_recommendations[:10]  # Top 10 recommendations
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get current framework status."""
        return {
            'framework_version': '1.0.0',
            'active_session': self.session.session_id if self.session else None,
            'performance_metrics': self.performance_metrics,
            'validators_status': {
                'interaction': self.validators['interaction'].is_initialized(),
                'emergent': self.validators['emergent'].is_initialized(),
                'coordination': self.validators['coordination'].is_initialized(),
                'adversarial': self.validators['adversarial'].is_initialized()
            },
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }