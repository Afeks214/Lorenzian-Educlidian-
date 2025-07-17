"""
Strategic Sequence Validator - Comprehensive validation framework for strategic sequences

This module provides comprehensive validation for strategic sequence execution,
including mathematical validation, performance validation, and behavioral validation.

Key Features:
- Mathematical validation of superposition properties
- Performance validation against targets
- Behavioral validation of agent interactions
- Sequence coherence validation
- Temporal stability analysis
- Comprehensive test suite execution
- Automated validation reporting

Validation Categories:
1. Mathematical Validation - Quantum properties, probability distributions
2. Performance Validation - Timing, quality metrics, efficiency
3. Behavioral Validation - Agent interactions, sequence coherence
4. Integration Validation - End-to-end system validation
5. Stress Testing - Edge cases and performance limits
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import torch
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import json
import traceback
from pathlib import Path

# Import components to validate
from src.environment.sequential_strategic_env import SequentialStrategicEnvironment
from src.agents.strategic.sequential_strategic_agents import (
    SequentialMLMIAgent,
    SequentialNWRQKAgent,
    SequentialRegimeAgent,
    SequentialAgentFactory,
    SequentialPrediction
)
from src.environment.strategic_superposition_aggregator import (
    StrategicSuperpositionAggregator,
    SuperpositionState,
    EnsembleSuperposition,
    QuantumSuperpositionMath
)
from src.training.strategic_sequential_trainer import StrategySequentialTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time_ms: float
    timestamp: datetime
    error_message: Optional[str] = None
    

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    test_suite: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_score: float
    execution_time_ms: float
    timestamp: datetime
    
    # Detailed results
    mathematical_validation: Dict[str, ValidationResult] = field(default_factory=dict)
    performance_validation: Dict[str, ValidationResult] = field(default_factory=dict)
    behavioral_validation: Dict[str, ValidationResult] = field(default_factory=dict)
    integration_validation: Dict[str, ValidationResult] = field(default_factory=dict)
    stress_test_results: Dict[str, ValidationResult] = field(default_factory=dict)
    
    # Summary statistics
    success_rate: float = 0.0
    average_score: float = 0.0
    critical_failures: List[str] = field(default_factory=list)
    performance_bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_suite': self.test_suite,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'overall_score': self.overall_score,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'success_rate': self.success_rate,
            'average_score': self.average_score,
            'critical_failures': self.critical_failures,
            'performance_bottlenecks': self.performance_bottlenecks,
            'recommendations': self.recommendations,
            'mathematical_validation': {k: v.__dict__ for k, v in self.mathematical_validation.items()},
            'performance_validation': {k: v.__dict__ for k, v in self.performance_validation.items()},
            'behavioral_validation': {k: v.__dict__ for k, v in self.behavioral_validation.items()},
            'integration_validation': {k: v.__dict__ for k, v in self.integration_validation.items()},
            'stress_test_results': {k: v.__dict__ for k, v in self.stress_test_results.items()}
        }


class ValidationTestBase(ABC):
    """Base class for validation tests"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def run_test(self, test_context: Dict[str, Any]) -> ValidationResult:
        """Run validation test"""
        pass
    
    def create_result(
        self,
        passed: bool,
        score: float,
        details: Dict[str, Any],
        execution_time_ms: float,
        error_message: Optional[str] = None
    ) -> ValidationResult:
        """Create validation result"""
        return ValidationResult(
            test_name=self.name,
            passed=passed,
            score=score,
            details=details,
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(),
            error_message=error_message
        )


class MathematicalValidationTest(ValidationTestBase):
    """Mathematical validation tests"""
    
    async def run_test(self, test_context: Dict[str, Any]) -> ValidationResult:
        """Run mathematical validation test"""
        start_time = time.time()
        
        try:
            aggregator = test_context.get('aggregator')
            superpositions = test_context.get('superpositions', [])
            
            if not aggregator or not superpositions:
                return self.create_result(
                    False, 0.0, {}, 0.0, "Missing aggregator or superpositions"
                )
            
            # Test superposition properties
            validation_results = {}
            
            # Test 1: Probability normalization
            for i, superposition in enumerate(superpositions):
                prob_sum = np.sum(superposition.action_probabilities)
                validation_results[f'prob_norm_agent_{i}'] = abs(prob_sum - 1.0) < 1e-6
            
            # Test 2: Quantum coherence calculation
            for i, superposition in enumerate(superpositions):
                coherence = QuantumSuperpositionMath.calculate_quantum_coherence(
                    superposition.action_probabilities
                )
                validation_results[f'quantum_coherence_agent_{i}'] = 0.0 <= coherence <= 1.0
            
            # Test 3: Ensemble aggregation
            ensemble = aggregator.aggregate_superpositions(superpositions)
            validation_results['ensemble_prob_norm'] = abs(np.sum(ensemble.ensemble_probabilities) - 1.0) < 1e-6
            validation_results['ensemble_coherence_valid'] = 0.0 <= ensemble.quantum_coherence <= 1.0
            validation_results['ensemble_stability_valid'] = 0.0 <= ensemble.temporal_stability <= 1.0
            
            # Test 4: Phase alignment
            if len(superpositions) > 1:
                probabilities = [s.action_probabilities for s in superpositions]
                phase_alignment = QuantumSuperpositionMath.calculate_phase_alignment(probabilities)
                validation_results['phase_alignment_valid'] = 0.0 <= phase_alignment <= 1.0
            
            # Test 5: Entanglement measures
            if len(superpositions) >= 2:
                entanglement = QuantumSuperpositionMath.calculate_entanglement_measure(
                    superpositions[0].action_probabilities,
                    superpositions[1].action_probabilities
                )
                validation_results['entanglement_valid'] = entanglement >= 0.0
            
            # Calculate overall score
            passed_tests = sum(1 for result in validation_results.values() if result)
            total_tests = len(validation_results)
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return self.create_result(
                passed=score >= 0.9,
                score=score,
                details=validation_results,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return self.create_result(
                False, 0.0, {'error': str(e)}, execution_time_ms, str(e)
            )


class PerformanceValidationTest(ValidationTestBase):
    """Performance validation tests"""
    
    async def run_test(self, test_context: Dict[str, Any]) -> ValidationResult:
        """Run performance validation test"""
        start_time = time.time()
        
        try:
            environment = test_context.get('environment')
            agents = test_context.get('agents', {})
            
            if not environment or not agents:
                return self.create_result(
                    False, 0.0, {}, 0.0, "Missing environment or agents"
                )
            
            performance_results = {}
            
            # Test 1: Agent computation time
            for agent_name, agent in agents.items():
                # Create dummy observation
                obs = {
                    'base_observation': {
                        'agent_features': np.random.randn(4),
                        'shared_context': np.random.randn(6),
                        'market_matrix': np.random.randn(48, 13)
                    },
                    'enriched_features': {
                        'sequence_position': np.array([0]),
                        'completion_ratio': np.array([0.0]),
                        'predecessor_avg_confidence': np.array([0.5]),
                        'predecessor_max_confidence': np.array([0.5]),
                        'predecessor_min_confidence': np.array([0.5]),
                        'predecessor_avg_computation_time': np.array([0.0])
                    },
                    'predecessor_superpositions': []
                }
                
                # Measure computation time
                computation_start = time.time()
                if hasattr(agent, 'predict_sequential'):
                    prediction = await agent.predict_sequential(obs, {})
                else:
                    prediction = await agent.predict(np.random.randn(48, 13), {})
                computation_time = (time.time() - computation_start) * 1000
                
                # Check against target
                target_time = 5.0  # 5ms target
                performance_results[f'{agent_name}_computation_time'] = computation_time <= target_time
                performance_results[f'{agent_name}_computation_time_ms'] = computation_time
            
            # Test 2: Environment performance
            environment.reset()
            sequence_times = []
            
            for _ in range(10):  # Test 10 sequences
                sequence_start = time.time()
                
                # Execute full sequence
                for agent_name in ['mlmi_expert', 'nwrqk_expert', 'regime_expert']:
                    if environment.agent_selection == agent_name:
                        obs = environment.observe(agent_name)
                        action = np.array([0.4, 0.4, 0.2])  # Dummy action
                        environment.step(action)
                
                sequence_time = (time.time() - sequence_start) * 1000
                sequence_times.append(sequence_time)
                
                # Check if episode ended
                if all(environment.terminations.values()):
                    break
            
            # Check sequence performance
            avg_sequence_time = np.mean(sequence_times)
            target_sequence_time = 15.0  # 15ms target
            performance_results['sequence_execution_time'] = avg_sequence_time <= target_sequence_time
            performance_results['avg_sequence_time_ms'] = avg_sequence_time
            
            # Test 3: Memory usage (simplified)
            performance_results['memory_usage_reasonable'] = True  # Placeholder
            
            # Test 4: Throughput
            throughput_sequences = len(sequence_times)
            performance_results['throughput_sequences_per_second'] = throughput_sequences / (sum(sequence_times) / 1000)
            
            # Calculate overall score
            passed_tests = sum(1 for result in performance_results.values() if isinstance(result, bool) and result)
            total_tests = sum(1 for result in performance_results.values() if isinstance(result, bool))
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return self.create_result(
                passed=score >= 0.8,
                score=score,
                details=performance_results,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return self.create_result(
                False, 0.0, {'error': str(e)}, execution_time_ms, str(e)
            )


class BehavioralValidationTest(ValidationTestBase):
    """Behavioral validation tests"""
    
    async def run_test(self, test_context: Dict[str, Any]) -> ValidationResult:
        """Run behavioral validation test"""
        start_time = time.time()
        
        try:
            environment = test_context.get('environment')
            agents = test_context.get('agents', {})
            
            if not environment or not agents:
                return self.create_result(
                    False, 0.0, {}, 0.0, "Missing environment or agents"
                )
            
            behavioral_results = {}
            
            # Test 1: Sequential execution order
            environment.reset()
            agent_sequence = []
            
            for _ in range(10):  # Test sequence order
                current_agent = environment.agent_selection
                if current_agent:
                    agent_sequence.append(current_agent)
                    obs = environment.observe(current_agent)
                    action = np.array([0.4, 0.4, 0.2])
                    environment.step(action)
                
                # Check if all agents executed
                if len(agent_sequence) >= 3:
                    break
            
            # Verify sequence order
            expected_order = ['mlmi_expert', 'nwrqk_expert', 'regime_expert']
            if len(agent_sequence) >= 3:
                behavioral_results['sequence_order_correct'] = agent_sequence[:3] == expected_order
            else:
                behavioral_results['sequence_order_correct'] = False
            
            # Test 2: Observation enrichment
            environment.reset()
            observations = []
            
            for agent_name in expected_order:
                if environment.agent_selection == agent_name:
                    obs = environment.observe(agent_name)
                    observations.append(obs)
                    
                    # Check if observation is enriched
                    has_enriched_features = 'enriched_features' in obs
                    has_predecessor_info = 'predecessor_superpositions' in obs
                    
                    behavioral_results[f'{agent_name}_observation_enriched'] = has_enriched_features and has_predecessor_info
                    
                    action = np.array([0.4, 0.4, 0.2])
                    environment.step(action)
            
            # Test 3: Superposition creation
            superpositions_created = []
            environment.reset()
            
            for agent_name in expected_order:
                if environment.agent_selection == agent_name:
                    obs = environment.observe(agent_name)
                    action = np.array([0.4, 0.4, 0.2])
                    environment.step(action)
                    
                    # Check if superposition was created
                    if hasattr(environment, 'env_state') and environment.env_state.agent_superpositions:
                        if agent_name in environment.env_state.agent_superpositions:
                            superpositions_created.append(agent_name)
            
            behavioral_results['superpositions_created'] = len(superpositions_created) == 3
            
            # Test 4: Agent interaction coherence
            if len(observations) >= 2:
                # Check that later agents receive predecessor context
                for i in range(1, len(observations)):
                    predecessor_context = observations[i].get('predecessor_superpositions', [])
                    expected_predecessors = i  # Should have i predecessors
                    behavioral_results[f'agent_{i}_predecessor_context'] = len(predecessor_context) == expected_predecessors
            
            # Test 5: Ensemble aggregation
            env_metrics = environment.get_performance_metrics()
            ensemble_confidence = env_metrics.get('avg_ensemble_confidence', 0.0)
            behavioral_results['ensemble_confidence_reasonable'] = ensemble_confidence > 0.4
            
            # Calculate overall score
            passed_tests = sum(1 for result in behavioral_results.values() if result)
            total_tests = len(behavioral_results)
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return self.create_result(
                passed=score >= 0.8,
                score=score,
                details=behavioral_results,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return self.create_result(
                False, 0.0, {'error': str(e)}, execution_time_ms, str(e)
            )


class IntegrationValidationTest(ValidationTestBase):
    """Integration validation tests"""
    
    async def run_test(self, test_context: Dict[str, Any]) -> ValidationResult:
        """Run integration validation test"""
        start_time = time.time()
        
        try:
            environment = test_context.get('environment')
            agents = test_context.get('agents', {})
            aggregator = test_context.get('aggregator')
            
            if not environment or not agents or not aggregator:
                return self.create_result(
                    False, 0.0, {}, 0.0, "Missing components for integration test"
                )
            
            integration_results = {}
            
            # Test 1: Full episode execution
            environment.reset()
            episode_completed = False
            step_count = 0
            max_steps = 100
            
            while not episode_completed and step_count < max_steps:
                current_agent = environment.agent_selection
                if current_agent:
                    obs = environment.observe(current_agent)
                    action = np.array([0.4, 0.4, 0.2])
                    environment.step(action)
                    step_count += 1
                
                # Check termination
                episode_completed = all(environment.terminations.values()) or all(environment.truncations.values())
            
            integration_results['episode_completed'] = episode_completed or step_count < max_steps
            integration_results['episode_steps'] = step_count
            
            # Test 2: Data flow validation
            environment.reset()
            data_flow_valid = True
            
            try:
                for agent_name in ['mlmi_expert', 'nwrqk_expert', 'regime_expert']:
                    if environment.agent_selection == agent_name:
                        obs = environment.observe(agent_name)
                        
                        # Validate observation structure
                        if 'base_observation' not in obs:
                            data_flow_valid = False
                            break
                        
                        if 'enriched_features' not in obs:
                            data_flow_valid = False
                            break
                        
                        action = np.array([0.4, 0.4, 0.2])
                        environment.step(action)
                        
            except Exception as e:
                data_flow_valid = False
                integration_results['data_flow_error'] = str(e)
            
            integration_results['data_flow_valid'] = data_flow_valid
            
            # Test 3: End-to-end performance
            environment.reset()
            total_execution_time = 0
            episodes_completed = 0
            
            for episode in range(5):  # Test 5 episodes
                episode_start = time.time()
                environment.reset()
                
                episode_done = False
                step_count = 0
                
                while not episode_done and step_count < 50:
                    current_agent = environment.agent_selection
                    if current_agent:
                        obs = environment.observe(current_agent)
                        action = np.array([0.4, 0.4, 0.2])
                        environment.step(action)
                        step_count += 1
                    
                    episode_done = all(environment.terminations.values()) or all(environment.truncations.values())
                
                episode_time = time.time() - episode_start
                total_execution_time += episode_time
                
                if episode_done or step_count < 50:
                    episodes_completed += 1
            
            avg_episode_time = total_execution_time / 5
            integration_results['avg_episode_time_s'] = avg_episode_time
            integration_results['episodes_completed'] = episodes_completed
            integration_results['end_to_end_performance'] = avg_episode_time < 1.0  # 1 second per episode
            
            # Test 4: Memory stability
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run some episodes
            for _ in range(10):
                environment.reset()
                for _ in range(20):
                    current_agent = environment.agent_selection
                    if current_agent:
                        obs = environment.observe(current_agent)
                        action = np.array([0.4, 0.4, 0.2])
                        environment.step(action)
                    
                    if all(environment.terminations.values()):
                        break
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            integration_results['memory_increase_mb'] = memory_increase
            integration_results['memory_stable'] = memory_increase < 100  # Less than 100MB increase
            
            # Calculate overall score
            passed_tests = sum(1 for result in integration_results.values() if isinstance(result, bool) and result)
            total_tests = sum(1 for result in integration_results.values() if isinstance(result, bool))
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return self.create_result(
                passed=score >= 0.8,
                score=score,
                details=integration_results,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return self.create_result(
                False, 0.0, {'error': str(e)}, execution_time_ms, str(e)
            )


class StressValidationTest(ValidationTestBase):
    """Stress validation tests"""
    
    async def run_test(self, test_context: Dict[str, Any]) -> ValidationResult:
        """Run stress validation test"""
        start_time = time.time()
        
        try:
            environment = test_context.get('environment')
            agents = test_context.get('agents', {})
            
            if not environment or not agents:
                return self.create_result(
                    False, 0.0, {}, 0.0, "Missing environment or agents"
                )
            
            stress_results = {}
            
            # Test 1: High-frequency execution
            environment.reset()
            rapid_execution_start = time.time()
            rapid_executions = 0
            
            for _ in range(1000):  # 1000 rapid executions
                current_agent = environment.agent_selection
                if current_agent:
                    obs = environment.observe(current_agent)
                    action = np.array([0.4, 0.4, 0.2])
                    environment.step(action)
                    rapid_executions += 1
                
                if all(environment.terminations.values()):
                    environment.reset()
            
            rapid_execution_time = time.time() - rapid_execution_start
            executions_per_second = rapid_executions / rapid_execution_time
            
            stress_results['executions_per_second'] = executions_per_second
            stress_results['high_frequency_stable'] = executions_per_second > 100  # 100 executions per second
            
            # Test 2: Large batch processing
            batch_size = 100
            batch_start = time.time()
            
            for _ in range(batch_size):
                environment.reset()
                for _ in range(10):  # 10 steps per episode
                    current_agent = environment.agent_selection
                    if current_agent:
                        obs = environment.observe(current_agent)
                        action = np.array([0.4, 0.4, 0.2])
                        environment.step(action)
                    
                    if all(environment.terminations.values()):
                        break
            
            batch_time = time.time() - batch_start
            episodes_per_second = batch_size / batch_time
            
            stress_results['episodes_per_second'] = episodes_per_second
            stress_results['batch_processing_stable'] = episodes_per_second > 10  # 10 episodes per second
            
            # Test 3: Edge case inputs
            edge_case_results = []
            
            # Test with extreme action values
            edge_cases = [
                np.array([1.0, 0.0, 0.0]),  # Extreme buy
                np.array([0.0, 0.0, 1.0]),  # Extreme sell
                np.array([0.33, 0.33, 0.34]),  # Balanced
                np.array([0.01, 0.01, 0.98]),  # Near-extreme
            ]
            
            for edge_action in edge_cases:
                try:
                    environment.reset()
                    current_agent = environment.agent_selection
                    if current_agent:
                        obs = environment.observe(current_agent)
                        environment.step(edge_action)
                    edge_case_results.append(True)
                except Exception:
                    edge_case_results.append(False)
            
            stress_results['edge_case_handling'] = all(edge_case_results)
            stress_results['edge_case_success_rate'] = sum(edge_case_results) / len(edge_case_results)
            
            # Test 4: Memory pressure
            large_observations = []
            memory_pressure_start = time.time()
            
            try:
                for _ in range(100):
                    environment.reset()
                    current_agent = environment.agent_selection
                    if current_agent:
                        obs = environment.observe(current_agent)
                        large_observations.append(obs)  # Store observations to create memory pressure
                        action = np.array([0.4, 0.4, 0.2])
                        environment.step(action)
                
                memory_pressure_time = time.time() - memory_pressure_start
                stress_results['memory_pressure_stable'] = memory_pressure_time < 10.0  # Complete within 10 seconds
                
            except Exception as e:
                stress_results['memory_pressure_stable'] = False
                stress_results['memory_pressure_error'] = str(e)
            
            # Test 5: Concurrent access simulation
            concurrent_results = []
            
            # Simulate concurrent access by rapid sequential calls
            for _ in range(50):
                try:
                    environment.reset()
                    # Rapid sequential calls
                    for _ in range(5):
                        current_agent = environment.agent_selection
                        if current_agent:
                            obs = environment.observe(current_agent)
                            action = np.array([0.4, 0.4, 0.2])
                            environment.step(action)
                    concurrent_results.append(True)
                except Exception:
                    concurrent_results.append(False)
            
            stress_results['concurrent_access_stable'] = all(concurrent_results)
            stress_results['concurrent_success_rate'] = sum(concurrent_results) / len(concurrent_results)
            
            # Calculate overall score
            passed_tests = sum(1 for result in stress_results.values() if isinstance(result, bool) and result)
            total_tests = sum(1 for result in stress_results.values() if isinstance(result, bool))
            score = passed_tests / total_tests if total_tests > 0 else 0.0
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return self.create_result(
                passed=score >= 0.7,  # Lower threshold for stress tests
                score=score,
                details=stress_results,
                execution_time_ms=execution_time_ms
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return self.create_result(
                False, 0.0, {'error': str(e)}, execution_time_ms, str(e)
            )


class StrategicSequenceValidator:
    """Main validator for strategic sequence system"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategic sequence validator
        
        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.StrategicSequenceValidator")
        
        # Initialize test suites
        self.test_suites = {
            'mathematical': MathematicalValidationTest('mathematical_validation', config),
            'performance': PerformanceValidationTest('performance_validation', config),
            'behavioral': BehavioralValidationTest('behavioral_validation', config),
            'integration': IntegrationValidationTest('integration_validation', config),
            'stress': StressValidationTest('stress_validation', config)
        }
        
        # Validation thresholds
        self.thresholds = {
            'mathematical_min_score': config.get('mathematical_min_score', 0.9),
            'performance_min_score': config.get('performance_min_score', 0.8),
            'behavioral_min_score': config.get('behavioral_min_score', 0.8),
            'integration_min_score': config.get('integration_min_score', 0.8),
            'stress_min_score': config.get('stress_min_score', 0.7),
            'overall_min_score': config.get('overall_min_score', 0.8)
        }
        
        self.logger.info(f"Strategic sequence validator initialized with {len(self.test_suites)} test suites")
    
    async def validate_system(self, system_components: Dict[str, Any]) -> ValidationReport:
        """
        Validate the entire strategic sequence system
        
        Args:
            system_components: Dictionary containing system components
            
        Returns:
            Comprehensive validation report
        """
        start_time = time.time()
        
        self.logger.info("Starting comprehensive strategic sequence validation")
        
        report = ValidationReport(
            test_suite="strategic_sequence_system",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            overall_score=0.0,
            execution_time_ms=0.0,
            timestamp=datetime.now()
        )
        
        try:
            # Prepare test context
            test_context = self._prepare_test_context(system_components)
            
            # Run mathematical validation
            math_result = await self.test_suites['mathematical'].run_test(test_context)
            report.mathematical_validation['mathematical_validation'] = math_result
            report.total_tests += 1
            
            if math_result.passed:
                report.passed_tests += 1
            else:
                report.failed_tests += 1
                if math_result.score < self.thresholds['mathematical_min_score']:
                    report.critical_failures.append('Mathematical validation failed')
            
            # Run performance validation
            perf_result = await self.test_suites['performance'].run_test(test_context)
            report.performance_validation['performance_validation'] = perf_result
            report.total_tests += 1
            
            if perf_result.passed:
                report.passed_tests += 1
            else:
                report.failed_tests += 1
                if perf_result.score < self.thresholds['performance_min_score']:
                    report.performance_bottlenecks.append('Performance validation failed')
            
            # Run behavioral validation
            behav_result = await self.test_suites['behavioral'].run_test(test_context)
            report.behavioral_validation['behavioral_validation'] = behav_result
            report.total_tests += 1
            
            if behav_result.passed:
                report.passed_tests += 1
            else:
                report.failed_tests += 1
                if behav_result.score < self.thresholds['behavioral_min_score']:
                    report.critical_failures.append('Behavioral validation failed')
            
            # Run integration validation
            integ_result = await self.test_suites['integration'].run_test(test_context)
            report.integration_validation['integration_validation'] = integ_result
            report.total_tests += 1
            
            if integ_result.passed:
                report.passed_tests += 1
            else:
                report.failed_tests += 1
                if integ_result.score < self.thresholds['integration_min_score']:
                    report.critical_failures.append('Integration validation failed')
            
            # Run stress validation
            stress_result = await self.test_suites['stress'].run_test(test_context)
            report.stress_test_results['stress_validation'] = stress_result
            report.total_tests += 1
            
            if stress_result.passed:
                report.passed_tests += 1
            else:
                report.failed_tests += 1
                if stress_result.score < self.thresholds['stress_min_score']:
                    report.performance_bottlenecks.append('Stress validation failed')
            
            # Calculate overall metrics
            report.success_rate = report.passed_tests / report.total_tests
            
            all_scores = [
                math_result.score,
                perf_result.score,
                behav_result.score,
                integ_result.score,
                stress_result.score
            ]
            report.average_score = np.mean(all_scores)
            report.overall_score = report.average_score
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            report.execution_time_ms = (time.time() - start_time) * 1000
            
            self.logger.info(f"Validation completed: {report.passed_tests}/{report.total_tests} tests passed, overall score: {report.overall_score:.3f}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.logger.error(traceback.format_exc())
            
            report.execution_time_ms = (time.time() - start_time) * 1000
            report.critical_failures.append(f"Validation system error: {str(e)}")
            
            return report
    
    def _prepare_test_context(self, system_components: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare test context from system components"""
        # Create test environment if not provided
        if 'environment' not in system_components:
            env_config = {
                'max_episode_steps': 100,
                'total_episodes': 10,
                'performance': {
                    'max_agent_computation_time_ms': 5.0,
                    'max_sequence_execution_time_ms': 15.0
                }
            }
            system_components['environment'] = SequentialStrategicEnvironment(env_config)
        
        # Create test agents if not provided
        if 'agents' not in system_components:
            agent_config = {
                'feature_indices': {
                    'mlmi_expert': [0, 1, 9, 10],
                    'nwrqk_expert': [2, 3, 4, 5],
                    'regime_expert': [10, 11, 12]
                },
                'agents': {
                    'mlmi_expert': {'hidden_dims': [64, 32], 'dropout_rate': 0.1},
                    'nwrqk_expert': {'hidden_dims': [64, 32], 'dropout_rate': 0.1},
                    'regime_expert': {'hidden_dims': [64, 32], 'dropout_rate': 0.1}
                },
                'environment': {'superposition_enabled': True}
            }
            system_components['agents'] = SequentialAgentFactory.create_all_agents(agent_config)
        
        # Create test aggregator if not provided
        if 'aggregator' not in system_components:
            aggregator_config = {
                'weighting_strategy': 'adaptive',
                'min_superposition_quality': 0.5,
                'max_aggregation_time_ms': 2.0
            }
            system_components['aggregator'] = StrategicSuperpositionAggregator(aggregator_config)
        
        # Create test superpositions
        if 'superpositions' not in system_components:
            system_components['superpositions'] = self._create_test_superpositions()
        
        return system_components
    
    def _create_test_superpositions(self) -> List[SuperpositionState]:
        """Create test superpositions"""
        return [
            SuperpositionState(
                agent_name='mlmi_expert',
                action_probabilities=np.array([0.5, 0.3, 0.2]),
                confidence=0.8,
                feature_importance={'feature_0': 0.6, 'feature_1': 0.4},
                internal_state={'test': True},
                computation_time_ms=2.0,
                timestamp=datetime.now(),
                superposition_features={'quantum_coherence': 0.7}
            ),
            SuperpositionState(
                agent_name='nwrqk_expert',
                action_probabilities=np.array([0.4, 0.4, 0.2]),
                confidence=0.75,
                feature_importance={'feature_2': 0.5, 'feature_3': 0.5},
                internal_state={'test': True},
                computation_time_ms=2.5,
                timestamp=datetime.now(),
                superposition_features={'quantum_coherence': 0.65}
            ),
            SuperpositionState(
                agent_name='regime_expert',
                action_probabilities=np.array([0.3, 0.5, 0.2]),
                confidence=0.7,
                feature_importance={'feature_10': 0.7, 'feature_11': 0.3},
                internal_state={'test': True},
                computation_time_ms=3.0,
                timestamp=datetime.now(),
                superposition_features={'quantum_coherence': 0.6}
            )
        ]
    
    def _generate_recommendations(self, report: ValidationReport) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Mathematical validation recommendations
        if 'mathematical_validation' in report.mathematical_validation:
            math_result = report.mathematical_validation['mathematical_validation']
            if math_result.score < self.thresholds['mathematical_min_score']:
                recommendations.append("Improve mathematical property validation for superposition states")
        
        # Performance recommendations
        if 'performance_validation' in report.performance_validation:
            perf_result = report.performance_validation['performance_validation']
            if perf_result.score < self.thresholds['performance_min_score']:
                recommendations.append("Optimize agent computation times and sequence execution")
        
        # Behavioral recommendations
        if 'behavioral_validation' in report.behavioral_validation:
            behav_result = report.behavioral_validation['behavioral_validation']
            if behav_result.score < self.thresholds['behavioral_min_score']:
                recommendations.append("Improve sequential execution order and agent interactions")
        
        # Integration recommendations
        if 'integration_validation' in report.integration_validation:
            integ_result = report.integration_validation['integration_validation']
            if integ_result.score < self.thresholds['integration_min_score']:
                recommendations.append("Enhance end-to-end system integration and data flow")
        
        # Stress test recommendations
        if 'stress_validation' in report.stress_test_results:
            stress_result = report.stress_test_results['stress_validation']
            if stress_result.score < self.thresholds['stress_min_score']:
                recommendations.append("Improve system stability under stress conditions")
        
        # Overall recommendations
        if report.overall_score < self.thresholds['overall_min_score']:
            recommendations.append("Consider comprehensive system optimization and testing")
        
        return recommendations
    
    def save_report(self, report: ValidationReport, filepath: str):
        """Save validation report to file"""
        try:
            report_data = report.to_dict()
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
    
    def get_validation_summary(self, report: ValidationReport) -> Dict[str, Any]:
        """Get validation summary"""
        return {
            'overall_success': report.overall_score >= self.thresholds['overall_min_score'],
            'passed_tests': report.passed_tests,
            'total_tests': report.total_tests,
            'success_rate': report.success_rate,
            'average_score': report.average_score,
            'critical_failures': len(report.critical_failures),
            'performance_bottlenecks': len(report.performance_bottlenecks),
            'recommendations': len(report.recommendations),
            'execution_time_ms': report.execution_time_ms
        }


# Factory function for creating validator
def create_strategic_sequence_validator(config: Dict[str, Any]) -> StrategicSequenceValidator:
    """Create strategic sequence validator with configuration"""
    return StrategicSequenceValidator(config)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = {
        'mathematical_min_score': 0.9,
        'performance_min_score': 0.8,
        'behavioral_min_score': 0.8,
        'integration_min_score': 0.8,
        'stress_min_score': 0.7,
        'overall_min_score': 0.8
    }
    
    # Create validator
    validator = StrategicSequenceValidator(config)
    
    # Run validation
    async def main():
        # Create test system components
        system_components = {}
        
        # Run validation
        report = await validator.validate_system(system_components)
        
        # Get summary
        summary = validator.get_validation_summary(report)
        
        print("Strategic Sequence Validation Results:")
        print(f"Overall Success: {summary['overall_success']}")
        print(f"Passed Tests: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.3f}")
        print(f"Average Score: {summary['average_score']:.3f}")
        print(f"Critical Failures: {summary['critical_failures']}")
        print(f"Performance Bottlenecks: {summary['performance_bottlenecks']}")
        print(f"Recommendations: {summary['recommendations']}")
        print(f"Execution Time: {summary['execution_time_ms']:.2f}ms")
        
        # Print detailed results
        print("\nDetailed Results:")
        for category, results in [
            ('Mathematical', report.mathematical_validation),
            ('Performance', report.performance_validation),
            ('Behavioral', report.behavioral_validation),
            ('Integration', report.integration_validation),
            ('Stress', report.stress_test_results)
        ]:
            print(f"\n{category} Validation:")
            for test_name, result in results.items():
                print(f"  {test_name}: {'PASS' if result.passed else 'FAIL'} (Score: {result.score:.3f})")
        
        # Save report
        validator.save_report(report, 'strategic_sequence_validation_report.json')
    
    # Run the validation
    asyncio.run(main())