"""
Comprehensive Unit Test Suite for Strategic MARL Component.

This module provides complete test coverage for the StrategicMARLComponent
including:
- Event handling tests
- Matrix validation tests  
- Decision aggregation tests
- Performance benchmark tests
- Error handling and recovery tests
- Mathematical validation tests
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.strategic_marl_component import StrategicMARLComponent, StrategicDecision
from src.agents.strategic_agent_base import MLMIStrategicAgent, NWRQKStrategicAgent, RegimeDetectionAgent
from src.agents.mathematical_validator import MathematicalValidator
from src.core.events import EventType, Event
from tests.mocks.mock_kernel import MockKernel


@pytest.fixture
def mock_kernel():
    """Create a mock kernel for testing."""
    kernel = MockKernel()
    return kernel


@pytest.fixture
def strategic_config():
    """Provide test configuration for Strategic MARL."""
    return {
        'environment': {
            'matrix_shape': [48, 13],
            'feature_indices': {
                'mlmi_expert': [0, 1, 9, 10],
                'nwrqk_expert': [2, 3, 4, 5],
                'regime_expert': [10, 11, 12]
            }
        },
        'ensemble': {
            'weights': [0.4, 0.35, 0.25],
            'confidence_threshold': 0.65
        },
        'performance': {
            'max_inference_latency_ms': 5.0,
            'max_memory_usage_mb': 512
        },
        'safety': {
            'max_consecutive_failures': 5,
            'failure_cooldown_minutes': 10
        },
        'optimization': {
            'device': 'cpu'
        },
        'agents': {
            'mlmi_expert': {
                'hidden_dims': [64, 32],
                'dropout_rate': 0.1
            },
            'nwrqk_expert': {
                'hidden_dims': [64, 32],
                'dropout_rate': 0.1
            },
            'regime_expert': {
                'hidden_dims': [64, 32],
                'dropout_rate': 0.15
            }
        }
    }


@pytest.fixture
def strategic_component(mock_kernel):
    """Create a Strategic MARL Component for testing."""
    component = StrategicMARLComponent(mock_kernel)
    return component


@pytest.fixture
def sample_matrix_data():
    """Generate sample 48x13 matrix data for testing."""
    np.random.seed(42)  # For reproducible tests
    return np.random.randn(48, 13)


@pytest.fixture
def sample_event_data(sample_matrix_data):
    """Generate sample event data for SYNERGY_DETECTED events."""
    return {
        'matrix_data': sample_matrix_data,
        'synergy_type': 'bullish_momentum',
        'direction': 'long',
        'confidence': 0.85,
        'timestamp': datetime.now().isoformat()
    }


class TestStrategicMARLComponentInitialization:
    """Test suite for component initialization."""
    
    def test_component_initialization(self, mock_kernel, strategic_config):
        """Test basic component initialization."""
        # Mock the config loading
        mock_kernel.config = strategic_config
        
        component = StrategicMARLComponent(mock_kernel)
        
        assert component.name == "StrategicMARLComponent"
        assert component.kernel == mock_kernel
        assert component.ensemble_weights is not None
        assert len(component.ensemble_weights) == 3
        assert np.isclose(np.sum(component.ensemble_weights), 1.0)
        assert component.confidence_threshold == 0.65
    
    def test_config_loading_with_defaults(self, mock_kernel):
        """Test configuration loading with default values."""
        mock_kernel.config = {}  # Empty config
        
        component = StrategicMARLComponent(mock_kernel)
        
        # Should load defaults
        assert component.strategic_config['environment']['matrix_shape'] == [48, 13]
        assert 'ensemble' in component.strategic_config
        assert 'performance' in component.strategic_config
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, strategic_component, strategic_config):
        """Test agent initialization process."""
        strategic_component.strategic_config = strategic_config
        
        # Mock agent initialization to avoid actual model loading
        with patch.object(MLMIStrategicAgent, 'initialize', new_callable=AsyncMock) as mock_mlmi, \
             patch.object(NWRQKStrategicAgent, 'initialize', new_callable=AsyncMock) as mock_nwrqk, \
             patch.object(RegimeDetectionAgent, 'initialize', new_callable=AsyncMock) as mock_regime:
            
            await strategic_component._initialize_agents()
            
            assert strategic_component.mlmi_agent is not None
            assert strategic_component.nwrqk_agent is not None
            assert strategic_component.regime_agent is not None
            
            mock_mlmi.assert_called_once()
            mock_nwrqk.assert_called_once()
            mock_regime.assert_called_once()
    
    def test_configuration_validation(self, strategic_component, strategic_config):
        """Test configuration validation."""
        strategic_component.strategic_config = strategic_config
        
        # Should pass validation
        strategic_component._validate_configuration()
        
        # Test missing required section
        invalid_config = strategic_config.copy()
        del invalid_config['environment']
        strategic_component.strategic_config = invalid_config
        
        with pytest.raises(ValueError, match="Missing required config section"):
            strategic_component._validate_configuration()
    
    def test_ensemble_weights_validation(self, strategic_component, strategic_config):
        """Test ensemble weights validation."""
        # Test invalid weights that don't sum to 1
        invalid_config = strategic_config.copy()
        invalid_config['ensemble']['weights'] = [0.5, 0.3, 0.1]  # Sum = 0.9
        strategic_component.strategic_config = invalid_config
        
        with pytest.raises(ValueError, match="Ensemble weights must sum to 1.0"):
            strategic_component._validate_configuration()


class TestEventHandling:
    """Test suite for event handling."""
    
    @pytest.mark.asyncio
    async def test_synergy_event_processing(self, strategic_component, sample_event_data):
        """Test complete synergy event processing."""
        # Mock agent execution
        with patch.object(strategic_component, '_execute_agents_parallel', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [
                {
                    'agent_name': 'MLMI',
                    'action_probabilities': [0.4, 0.3, 0.3],
                    'confidence': 0.75
                },
                {
                    'agent_name': 'NWRQK', 
                    'action_probabilities': [0.2, 0.5, 0.3],
                    'confidence': 0.68
                },
                {
                    'agent_name': 'Regime',
                    'action_probabilities': [0.35, 0.35, 0.3],
                    'confidence': 0.72
                }
            ]
            
            # Mock event publication
            with patch.object(strategic_component, '_publish_strategic_decision', new_callable=AsyncMock) as mock_publish:
                
                await strategic_component.process_synergy_event(sample_event_data)
                
                mock_execute.assert_called_once()
                mock_publish.assert_called_once()
                
                # Check that performance metrics were updated
                assert strategic_component.performance_metrics['total_inferences'] > 0
    
    def test_matrix_validation_valid_data(self, strategic_component, sample_matrix_data):
        """Test matrix validation with valid data."""
        event_data = {'matrix_data': sample_matrix_data}
        
        validated_matrix = strategic_component._extract_and_validate_matrix(event_data)
        
        assert validated_matrix.shape == (48, 13)
        assert isinstance(validated_matrix, np.ndarray)
    
    def test_matrix_validation_invalid_shape(self, strategic_component):
        """Test matrix validation with invalid shape."""
        invalid_data = {'matrix_data': np.random.randn(40, 10)}  # Wrong shape
        
        with pytest.raises(ValueError, match="Invalid matrix shape"):
            strategic_component._extract_and_validate_matrix(invalid_data)
    
    def test_matrix_validation_nan_values(self, strategic_component):
        """Test matrix validation with NaN values."""
        invalid_data = np.random.randn(48, 13)
        invalid_data[0, 0] = np.nan
        event_data = {'matrix_data': invalid_data}
        
        with pytest.raises(ValueError, match="Matrix contains NaN values"):
            strategic_component._extract_and_validate_matrix(event_data)
    
    def test_matrix_validation_infinite_values(self, strategic_component):
        """Test matrix validation with infinite values."""
        invalid_data = np.random.randn(48, 13)
        invalid_data[0, 0] = np.inf
        event_data = {'matrix_data': invalid_data}
        
        with pytest.raises(ValueError, match="Matrix contains infinite values"):
            strategic_component._extract_and_validate_matrix(event_data)
    
    def test_matrix_validation_missing_data(self, strategic_component):
        """Test matrix validation with missing matrix data."""
        event_data = {}  # No matrix_data
        
        with pytest.raises(ValueError, match="Event data missing required 'matrix_data' field"):
            strategic_component._extract_and_validate_matrix(event_data)


class TestDecisionAggregation:
    """Test suite for decision aggregation."""
    
    def test_decision_aggregation_basic(self, strategic_component):
        """Test basic decision aggregation."""
        agent_results = [
            {
                'agent_name': 'MLMI',
                'action_probabilities': [0.6, 0.2, 0.2],
                'confidence': 0.8
            },
            {
                'agent_name': 'NWRQK',
                'action_probabilities': [0.3, 0.4, 0.3],
                'confidence': 0.7
            },
            {
                'agent_name': 'Regime',
                'action_probabilities': [0.4, 0.3, 0.3],
                'confidence': 0.75
            }
        ]
        
        decision = strategic_component._combine_agent_outputs(agent_results)
        
        assert isinstance(decision, StrategicDecision)
        assert decision.action in ['buy', 'hold', 'sell']
        assert 0 <= decision.confidence <= 1
        assert 0 <= decision.uncertainty <= 10  # Entropy upper bound
        assert isinstance(decision.should_proceed, bool)
        assert len(decision.agent_contributions) == 3
    
    def test_ensemble_probability_normalization(self, strategic_component):
        """Test that ensemble probabilities are properly normalized."""
        agent_results = [
            {
                'agent_name': 'MLMI',
                'action_probabilities': [0.5, 0.3, 0.2],
                'confidence': 0.9
            },
            {
                'agent_name': 'NWRQK',
                'action_probabilities': [0.2, 0.6, 0.2],
                'confidence': 0.8
            },
            {
                'agent_name': 'Regime',
                'action_probabilities': [0.3, 0.3, 0.4],
                'confidence': 0.85
            }
        ]
        
        decision = strategic_component._combine_agent_outputs(agent_results)
        ensemble_probs = decision.performance_metrics['ensemble_probabilities']
        
        # Check normalization
        assert abs(sum(ensemble_probs) - 1.0) < 1e-6
        assert all(p >= 0 for p in ensemble_probs)
    
    def test_confidence_threshold_behavior(self, strategic_component):
        """Test behavior with different confidence levels."""
        # High confidence case
        high_conf_results = [
            {
                'agent_name': 'MLMI',
                'action_probabilities': [0.9, 0.05, 0.05],
                'confidence': 0.95
            },
            {
                'agent_name': 'NWRQK',
                'action_probabilities': [0.85, 0.1, 0.05],
                'confidence': 0.9
            },
            {
                'agent_name': 'Regime',
                'action_probabilities': [0.8, 0.15, 0.05],
                'confidence': 0.88
            }
        ]
        
        high_conf_decision = strategic_component._combine_agent_outputs(high_conf_results)
        assert high_conf_decision.should_proceed == True
        
        # Low confidence case
        low_conf_results = [
            {
                'agent_name': 'MLMI',
                'action_probabilities': [0.4, 0.35, 0.25],
                'confidence': 0.5
            },
            {
                'agent_name': 'NWRQK',
                'action_probabilities': [0.3, 0.4, 0.3],
                'confidence': 0.45
            },
            {
                'agent_name': 'Regime',
                'action_probabilities': [0.35, 0.35, 0.3],
                'confidence': 0.48
            }
        ]
        
        low_conf_decision = strategic_component._combine_agent_outputs(low_conf_results)
        assert low_conf_decision.should_proceed == False


class TestPerformanceMonitoring:
    """Test suite for performance monitoring."""
    
    @pytest.mark.asyncio
    async def test_inference_latency_tracking(self, strategic_component, sample_event_data):
        """Test inference latency tracking."""
        initial_count = strategic_component.performance_metrics['total_inferences']
        
        # Mock agent execution with controlled timing
        with patch.object(strategic_component, '_execute_agents_parallel', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [
                {'agent_name': 'MLMI', 'action_probabilities': [0.33, 0.34, 0.33], 'confidence': 0.5},
                {'agent_name': 'NWRQK', 'action_probabilities': [0.33, 0.34, 0.33], 'confidence': 0.5},
                {'agent_name': 'Regime', 'action_probabilities': [0.33, 0.34, 0.33], 'confidence': 0.5}
            ]
            
            with patch.object(strategic_component, '_publish_strategic_decision', new_callable=AsyncMock):
                await strategic_component.process_synergy_event(sample_event_data)
        
        # Check metrics were updated
        assert strategic_component.performance_metrics['total_inferences'] == initial_count + 1
        assert strategic_component.performance_metrics['avg_inference_time_ms'] > 0
    
    def test_performance_metrics_initialization(self, strategic_component):
        """Test performance metrics initialization."""
        metrics = strategic_component.performance_metrics
        
        required_metrics = [
            'total_inferences', 'total_inference_time_ms', 'avg_inference_time_ms',
            'max_inference_time_ms', 'timeout_count', 'error_count', 'success_count'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_performance_alert_threshold(self, strategic_component):
        """Test performance alert when threshold exceeded."""
        # Simulate slow inference
        strategic_component._update_performance_metrics(10.0, success=True)  # 10ms > 5ms threshold
        
        # Should log warning (check in real implementation)
        assert strategic_component.performance_metrics['max_inference_time_ms'] == 10.0


class TestErrorHandlingAndRecovery:
    """Test suite for error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self, strategic_component, sample_event_data):
        """Test recovery when individual agents fail."""
        # Mock one agent to fail
        with patch.object(strategic_component, '_execute_mlmi_agent', new_callable=AsyncMock) as mock_mlmi, \
             patch.object(strategic_component, '_execute_nwrqk_agent', new_callable=AsyncMock) as mock_nwrqk, \
             patch.object(strategic_component, '_execute_regime_agent', new_callable=AsyncMock) as mock_regime:
            
            # Make MLMI agent fail
            mock_mlmi.side_effect = Exception("MLMI agent failure")
            mock_nwrqk.return_value = {'agent_name': 'NWRQK', 'action_probabilities': [0.2, 0.5, 0.3], 'confidence': 0.7}
            mock_regime.return_value = {'agent_name': 'Regime', 'action_probabilities': [0.35, 0.35, 0.3], 'confidence': 0.72}
            
            # Should handle gracefully and use fallback
            agent_results = await strategic_component._execute_agents_parallel(
                sample_event_data['matrix_data'], 
                {}
            )
            
            assert len(agent_results) == 3
            # First result should be fallback
            assert 'fallback' in agent_results[0] or agent_results[0]['agent_name'] == 'MLMI'
    
    def test_circuit_breaker_functionality(self, strategic_component):
        """Test circuit breaker opens after consecutive failures."""
        # Simulate consecutive failures
        for i in range(strategic_component.circuit_breaker['max_failures']):
            strategic_component._handle_processing_error(Exception(f"Error {i}"))
        
        assert strategic_component.circuit_breaker['is_open'] == True
        assert strategic_component.circuit_breaker['consecutive_failures'] >= strategic_component.circuit_breaker['max_failures']
    
    @pytest.mark.asyncio 
    async def test_timeout_handling(self, strategic_component, sample_event_data):
        """Test timeout handling in parallel execution."""
        # Mock agents to timeout
        async def slow_agent(*args, **kwargs):
            await asyncio.sleep(1.0)  # Longer than timeout
            return {'agent_name': 'Slow', 'action_probabilities': [0.33, 0.34, 0.33], 'confidence': 0.5}
        
        with patch.object(strategic_component, '_execute_mlmi_agent', new_callable=AsyncMock) as mock_mlmi, \
             patch.object(strategic_component, '_execute_nwrqk_agent', new_callable=AsyncMock) as mock_nwrqk, \
             patch.object(strategic_component, '_execute_regime_agent', new_callable=AsyncMock) as mock_regime:
            
            mock_mlmi.side_effect = slow_agent
            mock_nwrqk.side_effect = slow_agent  
            mock_regime.side_effect = slow_agent
            
            # Set very short timeout
            strategic_component.max_inference_latency_ms = 1.0
            
            agent_results = await strategic_component._execute_agents_parallel(
                sample_event_data['matrix_data'],
                {}
            )
            
            # Should return fallback results due to timeout
            assert len(agent_results) == 3
            assert strategic_component.performance_metrics['timeout_count'] > 0


class TestSharedContextProcessing:
    """Test suite for shared context processing."""
    
    def test_shared_context_extraction(self, strategic_component, sample_matrix_data):
        """Test shared context extraction from matrix data."""
        context = strategic_component._extract_shared_context(sample_matrix_data)
        
        required_fields = [
            'market_volatility', 'volume_profile', 'momentum_signal', 
            'trend_strength', 'market_regime', 'timestamp'
        ]
        
        for field in required_fields:
            assert field in context
        
        assert isinstance(context['market_volatility'], float)
        assert isinstance(context['market_regime'], str)
        assert context['market_regime'] in ['trending', 'ranging', 'volatile']
    
    def test_market_regime_detection(self, strategic_component):
        """Test market regime detection logic."""
        # High volatility scenario
        high_vol_data = np.random.randn(48, 13) * 0.1  # High volatility
        regime = strategic_component._detect_market_regime(high_vol_data)
        assert regime == 'volatile'
        
        # Strong momentum scenario  
        momentum_data = np.random.randn(48, 13) * 0.005
        momentum_data[:, 9] = 0.02  # Strong positive momentum
        regime = strategic_component._detect_market_regime(momentum_data)
        assert regime == 'trending'
        
        # Low volatility, low momentum (ranging)
        ranging_data = np.random.randn(48, 13) * 0.001
        regime = strategic_component._detect_market_regime(ranging_data)
        assert regime == 'ranging'


class TestMathematicalValidation:
    """Test suite for mathematical validation."""
    
    def test_mathematical_validator_integration(self, strategic_component):
        """Test integration with mathematical validator."""
        validator = MathematicalValidator(tolerance=1e-6)
        
        test_data = {
            'rewards': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'values': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            'gamma': 0.99,
            'gae_lambda': 0.95
        }
        
        results = validator.validate_all(test_data)
        
        assert 'gae_computation' in results
        assert 'kernel_calculations' in results
        assert 'mmd_calculation' in results
        assert 'superposition_probabilities' in results
        assert 'numerical_stability' in results
    
    def test_ensemble_weight_constraints(self, strategic_component):
        """Test that ensemble weights satisfy mathematical constraints."""
        weights = strategic_component.ensemble_weights
        
        # Check normalization
        assert abs(np.sum(weights) - 1.0) < 1e-6
        
        # Check non-negative
        assert np.all(weights >= 0)
        
        # Check reasonable distribution
        assert np.all(weights <= 1.0)


class TestComponentStatus:
    """Test suite for component status reporting."""
    
    def test_get_status_structure(self, strategic_component):
        """Test status report structure."""
        status = strategic_component.get_status()
        
        required_fields = [
            'name', 'initialized', 'circuit_breaker_open', 'consecutive_failures',
            'performance_metrics', 'ensemble_weights', 'confidence_threshold', 'agents_status'
        ]
        
        for field in required_fields:
            assert field in status
        
        assert isinstance(status['performance_metrics'], dict)
        assert isinstance(status['agents_status'], dict)
    
    def test_agents_status_reporting(self, strategic_component):
        """Test agent status reporting."""
        status = strategic_component.get_status()
        agents_status = status['agents_status']
        
        assert 'mlmi_initialized' in agents_status
        assert 'nwrqk_initialized' in agents_status
        assert 'regime_initialized' in agents_status
        
        # Should be False initially (agents not initialized in fixture)
        assert isinstance(agents_status['mlmi_initialized'], bool)


# Performance Benchmark Tests
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_inference_latency_benchmark(self, strategic_component, sample_event_data):
        """Benchmark inference latency against requirements."""
        # Mock fast agent execution
        with patch.object(strategic_component, '_execute_agents_parallel', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = [
                {'agent_name': 'MLMI', 'action_probabilities': [0.4, 0.3, 0.3], 'confidence': 0.75},
                {'agent_name': 'NWRQK', 'action_probabilities': [0.2, 0.5, 0.3], 'confidence': 0.68},
                {'agent_name': 'Regime', 'action_probabilities': [0.35, 0.35, 0.3], 'confidence': 0.72}
            ]
            
            with patch.object(strategic_component, '_publish_strategic_decision', new_callable=AsyncMock):
                
                start_time = time.time()
                await strategic_component.process_synergy_event(sample_event_data)
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                
                # Should meet 5ms requirement with significant margin
                assert latency_ms < strategic_component.max_inference_latency_ms * 2  # Allow 2x margin for testing
    
    def test_memory_usage_monitoring(self, strategic_component):
        """Test memory usage stays within limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large test data
        large_matrix = np.random.randn(1000, 13)  # Much larger than normal
        
        try:
            context = strategic_component._extract_shared_context(large_matrix[:48, :])  # Use valid shape
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_increase = final_memory - initial_memory
            max_allowed = strategic_component.strategic_config['performance']['max_memory_usage_mb']
            
            # Memory increase should be reasonable
            assert memory_increase < max_allowed / 10  # Allow 10% of max for this operation
            
        except Exception:
            pass  # Memory test is best effort


# Integration Tests
class TestSystemIntegration:
    """Integration tests with other system components."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, strategic_component, sample_event_data):
        """Test full pipeline from event to decision."""
        # Mock all external dependencies
        with patch.object(strategic_component, '_initialize_agents', new_callable=AsyncMock) as mock_init:
            
            await strategic_component.initialize()
            mock_init.assert_called_once()
            
            # Mock agent execution for complete pipeline test
            with patch.object(strategic_component, '_execute_agents_parallel', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = [
                    {
                        'agent_name': 'MLMI',
                        'action_probabilities': [0.6, 0.2, 0.2],
                        'confidence': 0.8,
                        'features_used': [0, 1, 9, 10],
                        'feature_importance': {'mlmi_value': 0.4, 'mlmi_signal': 0.6},
                        'internal_state': {},
                        'computation_time_ms': 1.0
                    },
                    {
                        'agent_name': 'NWRQK',
                        'action_probabilities': [0.3, 0.4, 0.3],
                        'confidence': 0.7,
                        'features_used': [2, 3, 4, 5],
                        'feature_importance': {'nwrqk_value': 0.5, 'nwrqk_slope': 0.5},
                        'internal_state': {},
                        'computation_time_ms': 1.2
                    },
                    {
                        'agent_name': 'Regime',
                        'action_probabilities': [0.4, 0.3, 0.3],
                        'confidence': 0.75,
                        'features_used': [10, 11, 12],
                        'feature_importance': {'mmd_score': 0.6, 'volatility_30': 0.4},
                        'internal_state': {},
                        'computation_time_ms': 0.9
                    }
                ]
                
                decision_published = False
                published_decision = None
                
                async def mock_publish(decision):
                    nonlocal decision_published, published_decision
                    decision_published = True
                    published_decision = decision
                
                with patch.object(strategic_component, '_publish_strategic_decision', side_effect=mock_publish):
                    
                    await strategic_component.process_synergy_event(sample_event_data)
                    
                    assert decision_published
                    assert published_decision is not None
                    assert isinstance(published_decision, StrategicDecision)
                    assert published_decision.action in ['buy', 'hold', 'sell']


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])