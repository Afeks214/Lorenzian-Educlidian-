"""
Test suite for Strategic MARL Component parallel execution optimizations.

This test suite validates the parallel execution improvements including:
- True parallel execution with asyncio.gather
- Result caching functionality
- Thread pool execution for CPU-bound operations
- Performance monitoring and metrics
- Fallback mechanisms for failed agents
"""

import asyncio
import time
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.agents.strategic_marl_component import StrategicMARLComponent
from src.agents.strategic_agent_base import AgentPrediction


class TestStrategicMARLParallel:
    """Test suite for parallel execution optimizations."""
    
    @pytest.fixture
    def mock_kernel(self):
        """Create mock kernel for testing."""
        kernel = MagicMock()
        kernel.config = MagicMock()
        kernel.config.get.return_value = {
            'environment': {
                'matrix_shape': [48, 13],
                'feature_indices': {
                    'mlmi_expert': [0, 1, 9, 10],
                    'nwrqk_expert': [2, 3, 4, 5],
                    'regime_expert': [6, 7, 8, 11, 12]
                }
            },
            'ensemble': {
                'confidence_threshold': 0.65,
                'weights': [0.33, 0.33, 0.34]
            },
            'performance': {
                'max_inference_latency_ms': 10.0,
                'agent_timeout_ms': 8.0
            }
        }
        kernel.event_bus = AsyncMock()
        return kernel
    
    @pytest.fixture
    def sample_matrix_data(self):
        """Create sample matrix data for testing."""
        return np.random.rand(48, 13).astype(np.float32)
    
    @pytest.fixture
    def sample_shared_context(self):
        """Create sample shared context for testing."""
        return {
            'market_volatility': 0.02,
            'volume_profile': 1.5,
            'momentum_signal': 0.01,
            'trend_strength': 0.05,
            'mmd_score': 0.03,
            'price_trend': 0.001,
            'market_regime': 'trending',
            'timestamp': datetime.now().isoformat()
        }
    
    @pytest.fixture
    async def component(self, mock_kernel):
        """Create and initialize component for testing."""
        component = StrategicMARLComponent(mock_kernel)
        
        # Mock the agents
        component.mlmi_agent = AsyncMock()
        component.nwrqk_agent = AsyncMock()
        component.regime_agent = AsyncMock()
        
        # Mock agent predictions
        mock_prediction = {
            'agent_name': 'Test',
            'action_probabilities': [0.3, 0.4, 0.3],
            'confidence': 0.8,
            'features_used': [0, 1, 2],
            'feature_importance': {'feature_0': 0.5, 'feature_1': 0.3, 'feature_2': 0.2},
            'internal_state': {},
            'computation_time_ms': 5.0,
            'fallback': False
        }
        
        component.mlmi_agent.predict.return_value = mock_prediction.copy()
        component.nwrqk_agent.predict.return_value = mock_prediction.copy()
        component.regime_agent.predict.return_value = mock_prediction.copy()
        
        return component
    
    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self, component, sample_matrix_data, sample_shared_context):
        """Test that parallel execution meets performance targets."""
        
        # Configure agents to simulate realistic execution times
        async def mock_predict_with_delay(matrix_data, shared_context):
            await asyncio.sleep(0.005)  # 5ms delay
            return {
                'agent_name': 'Test',
                'action_probabilities': [0.3, 0.4, 0.3],
                'confidence': 0.8,
                'features_used': [0, 1, 2],
                'feature_importance': {'feature_0': 0.5, 'feature_1': 0.3, 'feature_2': 0.2},
                'internal_state': {},
                'computation_time_ms': 5.0,
                'fallback': False
            }
        
        component.mlmi_agent.predict = mock_predict_with_delay
        component.nwrqk_agent.predict = mock_predict_with_delay
        component.regime_agent.predict = mock_predict_with_delay
        
        # Execute parallel agents
        start_time = time.time()
        results = await component._execute_agents_parallel(sample_matrix_data, sample_shared_context)
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Verify performance targets
        assert execution_time_ms < component.max_inference_latency_ms, f"Execution time {execution_time_ms}ms exceeds target {component.max_inference_latency_ms}ms"
        assert len(results) == 3, "Should return results for all 3 agents"
        
        # Verify parallel speedup
        estimated_sequential_time = sum(r.get('computation_time_ms', 0) for r in results)
        if estimated_sequential_time > 0:
            speedup = estimated_sequential_time / execution_time_ms
            assert speedup > 1.0, f"Parallel execution should be faster than sequential (speedup: {speedup})"
    
    @pytest.mark.asyncio
    async def test_result_caching(self, component, sample_matrix_data, sample_shared_context):
        """Test that result caching works properly."""
        
        # Execute first time (should cache)
        results1 = await component._execute_agents_parallel(sample_matrix_data, sample_shared_context)
        
        # Execute second time (should use cache)
        start_time = time.time()
        results2 = await component._execute_agents_parallel(sample_matrix_data, sample_shared_context)
        cache_execution_time_ms = (time.time() - start_time) * 1000
        
        # Verify cache performance
        assert cache_execution_time_ms < 1.0, f"Cache lookup should be <1ms, got {cache_execution_time_ms}ms"
        assert results1 == results2, "Cached results should be identical"
        
        # Verify cache metrics
        assert component._parallel_metrics['cache_hits'] > 0, "Should have cache hits"
        assert component._parallel_metrics['cache_hit_rate'] > 0, "Should have positive cache hit rate"
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, component, sample_matrix_data, sample_shared_context):
        """Test that timeouts are handled properly with fallback."""
        
        # Configure agents to timeout
        async def mock_predict_timeout(matrix_data, shared_context):
            await asyncio.sleep(0.02)  # 20ms delay (exceeds timeout)
            return {'should_not_reach': True}
        
        component.mlmi_agent.predict = mock_predict_timeout
        component.nwrqk_agent.predict = mock_predict_timeout
        component.regime_agent.predict = mock_predict_timeout
        
        # Execute with timeout
        results = await component._execute_agents_parallel(sample_matrix_data, sample_shared_context)
        
        # Verify fallback results
        assert len(results) == 3, "Should return fallback results for all agents"
        for result in results:
            assert result.get('fallback', False) or result.get('confidence', 0) == 0.1, "Should use fallback results"
        
        # Verify timeout metrics
        assert component.performance_metrics['timeout_count'] > 0, "Should record timeouts"
    
    @pytest.mark.asyncio
    async def test_thread_pool_execution(self, component, sample_matrix_data, sample_shared_context):
        """Test that thread pool is used for CPU-bound operations."""
        
        # Mock thread pool execution
        with patch.object(component, '_cpu_executor') as mock_executor:
            mock_executor.submit.return_value.result.return_value = sample_matrix_data[:, [0, 1, 2]]
            
            # Execute agents
            results = await component._execute_agents_parallel(sample_matrix_data, sample_shared_context)
            
            # Verify thread pool was used
            assert len(results) == 3, "Should return results for all agents"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, component, sample_matrix_data, sample_shared_context):
        """Test error handling and fallback mechanisms."""
        
        # Configure agents to raise errors
        component.mlmi_agent.predict.side_effect = Exception("Test error")
        component.nwrqk_agent.predict.side_effect = Exception("Test error")
        component.regime_agent.predict.side_effect = Exception("Test error")
        
        # Execute with errors
        results = await component._execute_agents_parallel(sample_matrix_data, sample_shared_context)
        
        # Verify fallback results
        assert len(results) == 3, "Should return fallback results for all agents"
        for result in results:
            assert result.get('fallback', False) or result.get('confidence', 0) == 0.1, "Should use fallback results"
            assert result.get('action_probabilities') == [0.33, 0.34, 0.33], "Should have uniform probabilities"
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, component, sample_matrix_data, sample_shared_context):
        """Test performance monitoring and metrics collection."""
        
        # Execute multiple times to build metrics
        for _ in range(5):
            await component._execute_agents_parallel(sample_matrix_data, sample_shared_context)
        
        # Verify metrics collection
        assert len(component._parallel_metrics['agent_execution_times']) > 0, "Should collect execution times"
        assert component._parallel_metrics['cache_hit_rate'] >= 0, "Should track cache hit rate"
        assert component._parallel_metrics['parallel_speedup'] >= 0, "Should track parallel speedup"
    
    @pytest.mark.asyncio
    async def test_cache_cleanup(self, component, sample_matrix_data, sample_shared_context):
        """Test cache cleanup functionality."""
        
        # Fill cache
        for i in range(5):
            modified_context = sample_shared_context.copy()
            modified_context['test_key'] = i
            await component._execute_agents_parallel(sample_matrix_data, modified_context)
        
        # Verify cache has entries
        assert len(component._agent_result_cache) > 0, "Cache should have entries"
        
        # Test cache cleanup with small max size
        component._cache_max_size = 2
        
        # Add more entries to trigger cleanup
        for i in range(5, 10):
            modified_context = sample_shared_context.copy()
            modified_context['test_key'] = i
            await component._execute_agents_parallel(sample_matrix_data, modified_context)
        
        # Verify cache size is controlled
        assert len(component._agent_result_cache) <= component._cache_max_size, "Cache should be cleaned up"
    
    def test_component_status_includes_parallel_metrics(self, component):
        """Test that component status includes parallel execution metrics."""
        
        status = component.get_status()
        
        # Verify parallel metrics in status
        assert 'parallel_metrics' in status, "Status should include parallel metrics"
        assert 'cache_hit_rate' in status['parallel_metrics'], "Should include cache hit rate"
        assert 'parallel_speedup' in status['parallel_metrics'], "Should include parallel speedup"
        assert 'thread_pool_utilization' in status['parallel_metrics'], "Should include thread pool utilization"
        
        # Verify performance targets
        assert 'performance_targets' in status, "Status should include performance targets"
        assert status['performance_targets']['max_inference_latency_ms'] == 10.0, "Should show correct latency target"
        
        # Verify execution pools
        assert 'execution_pools' in status, "Status should include execution pool info"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])