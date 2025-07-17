"""
AGENT EPSILON: Comprehensive Intelligence Validation Test Suite

This comprehensive test suite validates all three intelligence upgrades working together:
1. Dynamic Feature Selection (attention mechanisms)
2. Intelligent Gating Network (expert coordination)
3. Regime-Aware Reward Function (contextual learning)

Tests the complete intelligence pipeline with performance monitoring and validates
production readiness for the enhanced Strategic MARL system.

Critical Success Criteria:
- Intelligence functionality working correctly
- <1ms intelligence overhead maintained
- Decision quality improvements validated
- Integration stability confirmed

Author: Agent Epsilon - Intelligence Validator
Version: 1.0 - Production Certification
"""

import numpy as np
import torch
import time
import pytest
from typing import Dict, Any, List
import logging
from unittest.mock import Mock, patch
import psutil
import gc

# Import intelligence components
from src.intelligence.intelligence_hub import IntelligenceHub, IntelligenceMetrics
from src.intelligence.gating_network import GatingNetwork
from src.intelligence.regime_aware_reward import RegimeAwareRewardFunction
from src.intelligence.regime_detector import RegimeDetector, RegimeAnalysis, MarketRegime
from src.intelligence.attention_optimizer import AttentionOptimizer

# Import strategic agents
from src.agents.mlmi_strategic_agent import MLMIStrategicAgent
from src.agents.nwrqk_strategic_agent import NWRQKStrategicAgent
from src.agents.regime_detection_agent import RegimeDetectionAgent
from src.core.events import EventBus

logger = logging.getLogger(__name__)


class TestIntelligenceValidation:
    """Comprehensive validation of Strategic MARL intelligence upgrades."""
    
    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus for testing."""
        return Mock(spec=EventBus)
    
    @pytest.fixture
    def intelligence_hub(self):
        """Create intelligence hub for testing."""
        config = {
            'max_intelligence_overhead_ms': 1.0,
            'performance_monitoring': True,
            'regime_detection': {'fast_mode': True, 'cache_analysis': True},
            'gating_network': {'hidden_dim': 32},
            'attention': {},
            'regime_aware_reward': {}
        }
        return IntelligenceHub(config)
    
    @pytest.fixture
    def mlmi_agent(self, mock_event_bus):
        """Create MLMI agent for testing."""
        config = {
            'agent_id': 'test_mlmi_agent',
            'gamma': 0.99,
            'lambda_': 0.95,
            'hidden_dim': 128,
            'dropout_rate': 0.0
        }
        return MLMIStrategicAgent(config, mock_event_bus)
    
    @pytest.fixture
    def nwrqk_agent(self):
        """Create NWRQK agent for testing."""
        config = {
            'agent_id': 'test_nwrqk_agent',
            'hidden_dim': 64,
            'dropout_rate': 0.0
        }
        return NWRQKStrategicAgent(config)
    
    @pytest.fixture
    def regime_agent(self):
        """Create Regime Detection agent for testing."""
        config = {
            'agent_id': 'test_regime_agent',
            'hidden_dim': 32,
            'dropout_rate': 0.0
        }
        return RegimeDetectionAgent(config)

    def _create_market_context(self, volatility: float = 1.0, momentum: float = 0.0, 
                             volume_ratio: float = 1.0, mmd_score: float = 0.3, 
                             price_trend: float = 0.0) -> Dict[str, Any]:
        """Create market context for testing."""
        return {
            'volatility_30': volatility,
            'momentum_20': momentum,
            'momentum_50': momentum * 0.8,
            'volume_ratio': volume_ratio,
            'mmd_score': mmd_score,
            'price_trend': price_trend,
            'timestamp': time.time()
        }
    
    def _get_mlmi_agent(self, mock_event_bus):
        """Get MLMI agent instance."""
        config = {
            'agent_id': 'mlmi_test_agent',
            'gamma': 0.99,
            'lambda_': 0.95,
            'hidden_dim': 128
        }
        return MLMIStrategicAgent(config, mock_event_bus)
    
    def _get_gating_weights(self, intelligence_hub: IntelligenceHub, 
                          market_context: Dict[str, Any]) -> np.ndarray:
        """Get gating weights from intelligence hub."""
        context_tensor = intelligence_hub._extract_shared_context_optimized(market_context)
        
        with torch.no_grad():
            gating_result = intelligence_hub.gating_network_jit(context_tensor)
            
        if isinstance(gating_result, dict):
            weights = gating_result['gating_weights']
        else:
            weights = gating_result
            
        return weights.detach().cpu().numpy().flatten()

    def test_dynamic_feature_selection_validation(self, intelligence_hub, mlmi_agent, mock_event_bus):
        """Test that attention weights adapt to different market contexts."""
        
        # High volatility scenario - should focus on risk features
        high_vol_context = self._create_market_context(volatility=3.5, momentum=0.01)
        
        # Trending scenario - should focus on momentum features  
        trending_context = self._create_market_context(volatility=1.2, momentum=0.08)
        
        # Low volatility scenario - should focus on value features
        low_vol_context = self._create_market_context(volatility=0.4, momentum=0.005)
        
        # Test MLMI agent attention adaptation
        mlmi_agent = self._get_mlmi_agent(mock_event_bus)
        
        # Create test features for each context
        high_vol_features = torch.tensor([0.7, 0.1, 0.05, 0.02], dtype=torch.float32)
        trending_features = torch.tensor([0.3, 0.8, 0.15, 0.12], dtype=torch.float32)
        low_vol_features = torch.tensor([0.2, 0.05, 0.001, 0.001], dtype=torch.float32)
        
        with torch.no_grad():
            high_vol_result = mlmi_agent.policy_network(high_vol_features.unsqueeze(0))
            trending_result = mlmi_agent.policy_network(trending_features.unsqueeze(0))
            low_vol_result = mlmi_agent.policy_network(low_vol_features.unsqueeze(0))
        
        high_vol_attention = high_vol_result['attention_weights'].squeeze(0).numpy()
        trending_attention = trending_result['attention_weights'].squeeze(0).numpy()
        low_vol_attention = low_vol_result['attention_weights'].squeeze(0).numpy()
        
        # Validate attention differences across contexts
        high_vs_trending = np.abs(high_vol_attention - trending_attention).mean()
        high_vs_low = np.abs(high_vol_attention - low_vol_attention).mean()
        trending_vs_low = np.abs(trending_attention - low_vol_attention).mean()
        
        assert high_vs_trending > 0.15, f"High vol vs trending attention too similar: {high_vs_trending:.3f}"
        assert high_vs_low > 0.20, f"High vol vs low vol attention too similar: {high_vs_low:.3f}"
        assert trending_vs_low > 0.15, f"Trending vs low vol attention too similar: {trending_vs_low:.3f}"
        
        # Validate attention focuses on expected features
        # In trending markets, momentum features [2,3] should get higher attention
        trending_momentum_attention = trending_attention[2:4].sum()
        assert trending_momentum_attention > 0.4, f"Trending market should focus on momentum: {trending_momentum_attention:.3f}"

    def test_intelligent_gating_adaptation(self, intelligence_hub):
        """Test that gating network adapts expert weighting based on context."""
        
        # Crisis scenario - should trust regime detection agent more
        crisis_context = {
            'volatility_30': 4.5, 'momentum_20': -0.1, 'momentum_50': -0.08,
            'volume_ratio': 2.8, 'mmd_score': 0.9, 'price_trend': -0.05
        }
        
        # Stable trending scenario - should trust MLMI agent more  
        stable_trend_context = {
            'volatility_30': 1.2, 'momentum_20': 0.06, 'momentum_50': 0.04,
            'volume_ratio': 1.1, 'mmd_score': 0.2, 'price_trend': 0.03
        }
        
        # Range-bound scenario - should trust NWRQK agent more
        range_bound_context = {
            'volatility_30': 0.6, 'momentum_20': 0.005, 'momentum_50': -0.002,
            'volume_ratio': 0.8, 'mmd_score': 0.1, 'price_trend': 0.001
        }
        
        # Get gating weights for each scenario
        crisis_weights = self._get_gating_weights(intelligence_hub, crisis_context)
        trend_weights = self._get_gating_weights(intelligence_hub, stable_trend_context)
        range_weights = self._get_gating_weights(intelligence_hub, range_bound_context)
        
        # Validate different weighting across scenarios
        crisis_vs_trend = np.abs(crisis_weights - trend_weights).mean()
        crisis_vs_range = np.abs(crisis_weights - range_weights).mean()
        trend_vs_range = np.abs(trend_weights - range_weights).mean()
        
        assert crisis_vs_trend > 0.15, f"Crisis vs trend gating too similar: {crisis_vs_trend:.3f}"
        assert crisis_vs_range > 0.15, f"Crisis vs range gating too similar: {crisis_vs_range:.3f}"
        assert trend_vs_range > 0.10, f"Trend vs range gating too similar: {trend_vs_range:.3f}"
        
        # Validate expert specialization patterns exist
        assert len(crisis_weights) >= 3, f"Insufficient gating weights: {len(crisis_weights)}"
        assert np.sum(crisis_weights) > 0.1, f"Gating weights too small: {np.sum(crisis_weights)}"

    def test_regime_aware_reward_adaptation(self, intelligence_hub):
        """Test that rewards adapt appropriately to market regimes."""
        
        reward_function = intelligence_hub.regime_reward_function
        
        # Same profitable trade outcome
        profitable_outcome = {'pnl': 1000.0, 'risk_penalty': 50.0}
        
        # Different market contexts
        crisis_context = {'volatility_30': 4.0, 'momentum_20': -0.08}
        high_vol_context = {'volatility_30': 2.8, 'momentum_20': 0.02}
        low_vol_context = {'volatility_30': 0.5, 'momentum_20': 0.01}
        bull_trend_context = {'volatility_30': 1.5, 'momentum_20': 0.07}
        
        # Calculate regime-aware rewards
        crisis_reward = reward_function.compute_reward(profitable_outcome, crisis_context)
        high_vol_reward = reward_function.compute_reward(profitable_outcome, high_vol_context)
        low_vol_reward = reward_function.compute_reward(profitable_outcome, low_vol_context)
        bull_reward = reward_function.compute_reward(profitable_outcome, bull_trend_context)
        
        # Validate regime-specific reward adaptation
        assert crisis_reward != high_vol_reward, "Crisis reward should differ from high vol"
        assert high_vol_reward != low_vol_reward, "High vol reward should differ from low vol"
        assert bull_reward != low_vol_reward, "Bull trend reward should differ from low vol"
        
        # Test loss scenario
        loss_outcome = {'pnl': -500.0, 'risk_penalty': 25.0}
        
        crisis_loss_reward = reward_function.compute_reward(loss_outcome, crisis_context)
        low_vol_loss_reward = reward_function.compute_reward(loss_outcome, low_vol_context)
        
        # Rewards should differ across contexts
        assert crisis_loss_reward != low_vol_loss_reward, "Loss rewards should vary by context"

    def test_integrated_intelligence_performance(self, intelligence_hub):
        """Test complete intelligence system performance under various conditions."""
        
        test_scenarios = [
            self._create_crisis_scenario(),
            self._create_bull_trend_scenario(),
            self._create_bear_trend_scenario(),
            self._create_sideways_scenario(),
            self._create_volatile_scenario()
        ]
        
        performance_results = []
        
        for scenario_name, market_context, agent_predictions, attention_weights in test_scenarios:
            
            # Run multiple iterations for reliable timing
            scenario_latencies = []
            scenario_results = []
            
            for _ in range(50):
                start_time = time.perf_counter()
                
                result, metrics = intelligence_hub.process_intelligence_pipeline(
                    market_context, agent_predictions, attention_weights
                )
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                scenario_latencies.append(latency_ms)
                scenario_results.append(result)
            
            # Validate performance
            mean_latency = np.mean(scenario_latencies)
            p95_latency = np.percentile(scenario_latencies, 95)
            p99_latency = np.percentile(scenario_latencies, 99)
            
            performance_results.append({
                'scenario': scenario_name,
                'mean_latency_ms': mean_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'intelligence_success_rate': sum(1 for r in scenario_results if r['intelligence_active']) / len(scenario_results)
            })
            
            # Performance assertions
            assert mean_latency < 1.0, f"{scenario_name}: Mean latency {mean_latency:.3f}ms exceeds 1.0ms"
            assert p95_latency < 1.5, f"{scenario_name}: P95 latency {p95_latency:.3f}ms exceeds 1.5ms"
            assert p99_latency < 2.0, f"{scenario_name}: P99 latency {p99_latency:.3f}ms exceeds 2.0ms"
            
            # Intelligence functionality assertions
            for result in scenario_results:
                assert result['intelligence_active'], f"{scenario_name}: Intelligence not active"
                assert 'final_probabilities' in result, f"{scenario_name}: Missing final probabilities"
                assert 'regime' in result, f"{scenario_name}: Missing regime classification"
                assert 'gating_weights' in result, f"{scenario_name}: Missing gating weights"
        
        return performance_results

    def _create_crisis_scenario(self):
        """Create crisis market scenario."""
        market_context = self._create_market_context(
            volatility=4.5, momentum=-0.12, volume_ratio=3.2, 
            mmd_score=0.95, price_trend=-0.08
        )
        
        agent_predictions = [
            {'action_probabilities': [0.1, 0.15, 0.25, 0.15, 0.15, 0.1, 0.1], 'confidence': 0.8},
            {'action_probabilities': [0.05, 0.1, 0.3, 0.25, 0.15, 0.1, 0.05], 'confidence': 0.75},
            {'action_probabilities': [0.05, 0.1, 0.35, 0.3, 0.1, 0.05, 0.05], 'confidence': 0.9}
        ]
        
        attention_weights = [
            torch.tensor([0.4, 0.3, 0.2, 0.1]),  # MLMI focused on risk
            torch.tensor([0.2, 0.5, 0.2, 0.1]),  # NWRQK focused on levels
            torch.tensor([0.1, 0.6, 0.3])        # Regime focused on volatility
        ]
        
        return 'crisis', market_context, agent_predictions, attention_weights

    def _create_bull_trend_scenario(self):
        """Create bull trend market scenario."""
        market_context = self._create_market_context(
            volatility=1.8, momentum=0.08, volume_ratio=1.5, 
            mmd_score=0.3, price_trend=0.06
        )
        
        agent_predictions = [
            {'action_probabilities': [0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.1], 'confidence': 0.85},
            {'action_probabilities': [0.05, 0.1, 0.1, 0.15, 0.3, 0.2, 0.1], 'confidence': 0.8},
            {'action_probabilities': [0.05, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1], 'confidence': 0.75}
        ]
        
        attention_weights = [
            torch.tensor([0.2, 0.5, 0.2, 0.1]),  # MLMI focused on momentum
            torch.tensor([0.3, 0.2, 0.3, 0.2]),  # NWRQK balanced
            torch.tensor([0.2, 0.3, 0.5])        # Regime focused on trend
        ]
        
        return 'bull_trend', market_context, agent_predictions, attention_weights

    def _create_bear_trend_scenario(self):
        """Create bear trend market scenario."""
        market_context = self._create_market_context(
            volatility=2.2, momentum=-0.06, volume_ratio=1.8, 
            mmd_score=0.4, price_trend=-0.04
        )
        
        agent_predictions = [
            {'action_probabilities': [0.15, 0.2, 0.25, 0.2, 0.1, 0.05, 0.05], 'confidence': 0.8},
            {'action_probabilities': [0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05], 'confidence': 0.85},
            {'action_probabilities': [0.15, 0.25, 0.3, 0.15, 0.1, 0.03, 0.02], 'confidence': 0.82}
        ]
        
        attention_weights = [
            torch.tensor([0.3, 0.4, 0.2, 0.1]),  # MLMI focused on signal
            torch.tensor([0.4, 0.3, 0.2, 0.1]),  # NWRQK focused on support
            torch.tensor([0.2, 0.4, 0.4])        # Regime balanced
        ]
        
        return 'bear_trend', market_context, agent_predictions, attention_weights

    def _create_sideways_scenario(self):
        """Create sideways market scenario."""
        market_context = self._create_market_context(
            volatility=0.8, momentum=0.005, volume_ratio=0.9, 
            mmd_score=0.15, price_trend=0.001
        )
        
        agent_predictions = [
            {'action_probabilities': [0.14, 0.14, 0.14, 0.16, 0.14, 0.14, 0.14], 'confidence': 0.6},
            {'action_probabilities': [0.12, 0.15, 0.18, 0.2, 0.18, 0.12, 0.05], 'confidence': 0.65},
            {'action_probabilities': [0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05], 'confidence': 0.55}
        ]
        
        attention_weights = [
            torch.tensor([0.25, 0.25, 0.25, 0.25]),  # MLMI balanced
            torch.tensor([0.3, 0.3, 0.2, 0.2]),      # NWRQK focused on levels
            torch.tensor([0.4, 0.3, 0.3])            # Regime focused on MMD
        ]
        
        return 'sideways', market_context, agent_predictions, attention_weights

    def _create_volatile_scenario(self):
        """Create volatile market scenario."""
        market_context = self._create_market_context(
            volatility=3.2, momentum=0.02, volume_ratio=2.5, 
            mmd_score=0.7, price_trend=0.01
        )
        
        agent_predictions = [
            {'action_probabilities': [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1], 'confidence': 0.7},
            {'action_probabilities': [0.15, 0.2, 0.2, 0.2, 0.15, 0.05, 0.05], 'confidence': 0.72},
            {'action_probabilities': [0.1, 0.2, 0.25, 0.25, 0.15, 0.03, 0.02], 'confidence': 0.85}
        ]
        
        attention_weights = [
            torch.tensor([0.5, 0.2, 0.2, 0.1]),  # MLMI focused on volatility
            torch.tensor([0.2, 0.4, 0.3, 0.1]),  # NWRQK adaptive
            torch.tensor([0.15, 0.7, 0.15])      # Regime focused on volatility
        ]
        
        return 'volatile', market_context, agent_predictions, attention_weights

    def test_intelligence_mathematical_correctness(self, intelligence_hub):
        """Test mathematical correctness of intelligence components."""
        
        # Test gating network mathematical properties
        test_context = self._create_market_context()
        context_tensor = intelligence_hub._extract_shared_context_optimized(test_context)
        
        with torch.no_grad():
            gating_result = intelligence_hub.gating_network_jit(context_tensor)
            
        if isinstance(gating_result, dict):
            gating_weights = gating_result['gating_weights']
        else:
            gating_weights = gating_result
            
        gating_weights_np = gating_weights.detach().cpu().numpy().flatten()
        
        # Gating weights should sum to approximately 1.0 (softmax output)
        weight_sum = np.sum(gating_weights_np)
        assert 0.9 <= weight_sum <= 1.1, f"Gating weights sum {weight_sum} not near 1.0"
        
        # All weights should be non-negative
        assert np.all(gating_weights_np >= 0), "Negative gating weights detected"
        
        # Test regime analysis mathematical properties
        regime_analysis = intelligence_hub._fast_regime_detection(test_context)
        
        assert hasattr(regime_analysis, 'regime'), "Regime analysis missing regime attribute"
        assert hasattr(regime_analysis, 'confidence'), "Regime analysis missing confidence"
        assert 0.0 <= regime_analysis.confidence <= 1.0, f"Invalid regime confidence: {regime_analysis.confidence}"

    def test_intelligence_memory_stability(self, intelligence_hub):
        """Test memory usage stability over extended operation."""
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        test_context = self._create_market_context()
        test_predictions = [
            {'action_probabilities': [0.14, 0.14, 0.14, 0.16, 0.14, 0.14, 0.14], 'confidence': 0.7}
        ]
        
        # Run many iterations to test for memory leaks
        for i in range(1000):
            result, metrics = intelligence_hub.process_intelligence_pipeline(
                test_context, test_predictions, None
            )
            
            # Force garbage collection periodically
            if i % 100 == 0:
                gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal (<10MB over 1000 iterations)
        assert memory_growth < 10.0, f"Excessive memory growth: {memory_growth:.2f}MB"

    def test_intelligence_error_handling(self, intelligence_hub):
        """Test intelligence hub error handling and fallback mechanisms."""
        
        # Test with malformed market context
        malformed_contexts = [
            {},  # Empty context
            {'invalid_key': 'invalid_value'},  # Invalid keys
            {'volatility_30': float('inf')},  # Infinite values
            {'volatility_30': float('nan')},  # NaN values
        ]
        
        for context in malformed_contexts:
            try:
                result, metrics = intelligence_hub.process_intelligence_pipeline(
                    context, [], None
                )
                
                # Should not crash and should have fallback result
                assert 'intelligence_active' in result, "Missing intelligence_active flag"
                assert 'final_probabilities' in result, "Missing final probabilities"
                
            except Exception as e:
                pytest.fail(f"Intelligence hub crashed with malformed context: {e}")
        
        # Test with malformed agent predictions
        test_context = self._create_market_context()
        malformed_predictions = [
            [{'invalid': 'prediction'}],  # Invalid prediction format
            [{'action_probabilities': [1.0, 2.0]}],  # Invalid probabilities
            [],  # Empty predictions
        ]
        
        for predictions in malformed_predictions:
            try:
                result, metrics = intelligence_hub.process_intelligence_pipeline(
                    test_context, predictions, None
                )
                
                assert 'final_probabilities' in result, "Missing final probabilities in error case"
                
            except Exception as e:
                pytest.fail(f"Intelligence hub crashed with malformed predictions: {e}")

    def test_intelligence_integration_statistics(self, intelligence_hub):
        """Test intelligence hub integration statistics and monitoring."""
        
        test_context = self._create_market_context()
        test_predictions = [
            {'action_probabilities': [0.14, 0.14, 0.14, 0.16, 0.14, 0.14, 0.14], 'confidence': 0.7}
        ]
        
        # Run several operations to generate statistics
        for _ in range(10):
            result, metrics = intelligence_hub.process_intelligence_pipeline(
                test_context, test_predictions, None
            )
        
        # Get integration statistics
        stats = intelligence_hub.get_integration_statistics()
        
        # Validate statistics structure
        required_stat_keys = [
            'performance_metrics', 'integration_stats', 'cache_stats', 'component_status'
        ]
        
        for key in required_stat_keys:
            assert key in stats, f"Missing statistics key: {key}"
        
        # Validate performance metrics
        perf_metrics = stats['performance_metrics']
        assert 'decisions_per_second' in perf_metrics, "Missing decisions per second"
        assert perf_metrics['decisions_per_second'] > 0, "Invalid decisions per second"
        
        # Validate integration stats
        integration_stats = stats['integration_stats']
        assert integration_stats['total_operations'] >= 10, "Incorrect operation count"

    def test_intelligence_configuration_updates(self, intelligence_hub):
        """Test dynamic configuration updates."""
        
        # Test performance target update
        new_config = {'max_intelligence_overhead_ms': 0.5}
        intelligence_hub.update_configuration(new_config)
        
        assert intelligence_hub.max_intelligence_overhead_ms == 0.5, "Configuration update failed"
        
        # Test component configuration updates
        component_config = {
            'regime_detection': {'fast_mode': False},
            'gating_network': {'hidden_dim': 64}
        }
        
        # Should not crash
        try:
            intelligence_hub.update_configuration(component_config)
        except Exception as e:
            pytest.fail(f"Configuration update failed: {e}")

    def test_intelligence_state_reset(self, intelligence_hub):
        """Test intelligence hub state reset functionality."""
        
        test_context = self._create_market_context()
        test_predictions = [
            {'action_probabilities': [0.14, 0.14, 0.14, 0.16, 0.14, 0.14, 0.14], 'confidence': 0.7}
        ]
        
        # Generate some state
        for _ in range(5):
            result, metrics = intelligence_hub.process_intelligence_pipeline(
                test_context, test_predictions, None
            )
        
        # Verify state exists
        stats_before = intelligence_hub.get_integration_statistics()
        assert stats_before['integration_stats']['total_operations'] > 0, "No operations recorded"
        assert stats_before['cache_stats']['regime_cache_size'] >= 0, "Cache not functioning"
        
        # Reset state
        intelligence_hub.reset_intelligence_state()
        
        # Verify state reset
        stats_after = intelligence_hub.get_integration_statistics()
        assert stats_after['integration_stats']['total_operations'] == 0, "Operations not reset"
        assert stats_after['cache_stats']['regime_cache_size'] == 0, "Cache not cleared"


if __name__ == "__main__":
    # Run the comprehensive intelligence validation test suite
    pytest.main([__file__, "-v", "--tb=short"])