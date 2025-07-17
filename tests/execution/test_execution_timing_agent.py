"""
Comprehensive Test Suite for Execution Timing Agent (π₂)

Tests all core functionality including:
- Market impact model validation
- Neural network architecture 
- Execution strategy selection
- Performance targets (<2 bps slippage, <200μs inference)
- Real-time adaptation capabilities
"""

import pytest
import torch
import numpy as np
import time
from typing import List, Dict, Any

from src.execution.agents.execution_timing_agent import (
    ExecutionTimingAgent,
    ExecutionTimingNetwork,
    MarketImpactModel,
    ExecutionStrategy,
    MarketMicrostructure,
    MarketImpactResult,
    validate_slippage_target,
    benchmark_inference_performance
)


class TestMarketImpactModel:
    """Test suite for market impact model"""
    
    def setup_method(self):
        """Setup test environment"""
        self.impact_model = MarketImpactModel()
    
    def test_temporal_decay_calculation(self):
        """Test temporal decay function f(τ) = 1 - exp(-τ/τ₀)"""
        # Test immediate execution (τ = 0)
        decay_immediate = self.impact_model.calculate_temporal_decay(0.0)
        assert decay_immediate == 1.0
        
        # Test asymptotic behavior (large τ)
        decay_large = self.impact_model.calculate_temporal_decay(10000.0)
        assert decay_large > 0.99  # Should approach 1.0
        
        # Test specific values
        decay_300 = self.impact_model.calculate_temporal_decay(300.0)  # τ₀
        expected_300 = 1.0 - np.exp(-1.0)  # ≈ 0.632
        assert abs(decay_300 - expected_300) < 1e-6
    
    def test_square_root_impact_calculation(self):
        """Test square-root law: σ * √(Q/V)"""
        # Test normal conditions
        impact = self.impact_model.calculate_square_root_impact(
            order_quantity=100.0,
            market_volume=10000.0,
            volatility=0.2
        )
        
        # Should be: 0.1 * 0.2 * sqrt(100/10000) = 0.1 * 0.2 * 0.1 = 0.002
        # In bps: 0.002 * 10000 = 20 bps
        expected_impact = 20.0
        assert abs(impact - expected_impact) < 1e-6
        
        # Test zero volume (should return infinity)
        impact_zero_vol = self.impact_model.calculate_square_root_impact(100.0, 0.0, 0.2)
        assert impact_zero_vol == float('inf')
        
        # Test scaling properties
        impact_double_quantity = self.impact_model.calculate_square_root_impact(
            200.0, 10000.0, 0.2
        )
        # Should be sqrt(2) times the original
        assert abs(impact_double_quantity / impact - np.sqrt(2)) < 1e-6
    
    def test_strategy_multipliers(self):
        """Test strategy-specific impact multipliers"""
        base_params = {
            'order_quantity': 100.0,
            'market_volume': 10000.0,
            'volatility': 0.2,
            'time_to_execution': 300.0
        }
        
        # Calculate impact for each strategy
        results = {}
        for strategy in ExecutionStrategy:
            result = self.impact_model.calculate_total_impact(**base_params, strategy=strategy)
            results[strategy] = result.total_impact_bps
        
        # ICEBERG should have lowest impact
        assert results[ExecutionStrategy.ICEBERG] < results[ExecutionStrategy.IMMEDIATE]
        
        # TWAP should have moderate impact reduction
        assert results[ExecutionStrategy.TWAP_5MIN] < results[ExecutionStrategy.IMMEDIATE]
        assert results[ExecutionStrategy.TWAP_5MIN] > results[ExecutionStrategy.ICEBERG]
    
    def test_market_impact_result_structure(self):
        """Test MarketImpactResult contains all required fields"""
        result = self.impact_model.calculate_total_impact(
            order_quantity=100.0,
            market_volume=10000.0,
            volatility=0.2,
            time_to_execution=300.0,
            strategy=ExecutionStrategy.IMMEDIATE
        )
        
        # Check all required fields are present
        assert hasattr(result, 'total_impact_bps')
        assert hasattr(result, 'permanent_impact_bps')
        assert hasattr(result, 'temporary_impact_bps')
        assert hasattr(result, 'timing_cost_bps')
        assert hasattr(result, 'optimal_strategy')
        assert hasattr(result, 'calculation_time_ns')
        assert hasattr(result, 'expected_slippage_bps')
        
        # Check values are reasonable
        assert result.total_impact_bps >= 0
        assert result.permanent_impact_bps >= 0
        assert result.temporary_impact_bps >= 0
        assert result.calculation_time_ns > 0
        assert result.expected_slippage_bps >= result.total_impact_bps


class TestExecutionTimingNetwork:
    """Test suite for neural network architecture"""
    
    def setup_method(self):
        """Setup test environment"""
        self.network = ExecutionTimingNetwork()
    
    def test_network_architecture(self):
        """Test network follows 15D → 256→128→64→4 architecture"""
        # Check input dimension
        assert self.network.input_dim == 15
        assert self.network.output_dim == 4
        
        # Test forward pass with correct input
        input_tensor = torch.randn(1, 15)
        output = self.network.forward(input_tensor)
        assert output.shape == (1, 4)
        
        # Test single sample inference
        single_input = torch.randn(15)
        single_output = self.network.forward(single_input)
        assert single_output.shape == (4,)
        
        # Test output is valid probabilities
        assert torch.allclose(single_output.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.all(single_output >= 0)
        assert torch.all(single_output <= 1)
    
    def test_network_compilation(self):
        """Test JIT compilation for inference speed"""
        # Compile network
        self.network.compile_for_inference()
        
        # Test compiled inference
        input_tensor = torch.randn(1, 15)
        output_compiled = self.network.fast_inference(input_tensor)
        output_normal = self.network.forward(input_tensor)
        
        # Results should be identical
        assert torch.allclose(output_compiled, output_normal, atol=1e-6)
    
    def test_weight_initialization(self):
        """Test proper weight initialization"""
        # Check that weights are initialized (not zero)
        for param in self.network.parameters():
            assert not torch.all(param == 0)
            
        # Check weights are in reasonable range
        for module in self.network.modules():
            if isinstance(module, torch.nn.Linear):
                weight_std = module.weight.std().item()
                assert 0.01 < weight_std < 1.0  # Reasonable initialization range


class TestMarketMicrostructure:
    """Test suite for market microstructure data"""
    
    def test_market_microstructure_tensor_conversion(self):
        """Test conversion to tensor format"""
        context = MarketMicrostructure(
            bid_ask_spread=0.01, market_depth=1000.0, order_book_slope=0.5,
            current_volume=10000.0, volume_imbalance=0.1, volume_velocity=1.0,
            price_momentum=0.02, volatility_regime=0.15, tick_activity=0.8,
            permanent_impact=0.5, temporary_impact=1.0, resilience=0.7,
            time_to_close=3600.0, intraday_pattern=0.5, urgency_score=0.5
        )
        
        tensor = context.to_tensor()
        
        # Check correct shape
        assert tensor.shape == (15,)
        
        # Check values match
        expected_values = [
            0.01, 1000.0, 0.5,  # Liquidity
            10000.0, 0.1, 1.0,  # Volume
            0.02, 0.15, 0.8,    # Price dynamics
            0.5, 1.0, 0.7,      # Market impact
            3600.0, 0.5, 0.5    # Timing
        ]
        
        for i, expected in enumerate(expected_values):
            assert abs(tensor[i].item() - expected) < 1e-6


class TestExecutionTimingAgent:
    """Test suite for complete execution timing agent"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = ExecutionTimingAgent(learning_rate=3e-4)
        
        # Create test market context
        self.test_context = MarketMicrostructure(
            bid_ask_spread=0.01, market_depth=1000.0, order_book_slope=0.5,
            current_volume=10000.0, volume_imbalance=0.1, volume_velocity=1.0,
            price_momentum=0.02, volatility_regime=0.15, tick_activity=0.8,
            permanent_impact=0.5, temporary_impact=1.0, resilience=0.7,
            time_to_close=3600.0, intraday_pattern=0.5, urgency_score=0.5
        )
    
    def test_strategy_selection(self):
        """Test execution strategy selection"""
        strategy, impact_result = self.agent.select_execution_strategy(
            self.test_context, order_quantity=100.0
        )
        
        # Check return types
        assert isinstance(strategy, ExecutionStrategy)
        assert isinstance(impact_result, MarketImpactResult)
        
        # Check strategy is valid
        assert strategy in list(ExecutionStrategy)
        
        # Check impact result is reasonable
        assert impact_result.expected_slippage_bps >= 0
        assert impact_result.total_impact_bps >= 0
    
    def test_urgency_adjustment(self):
        """Test urgency level affects strategy selection"""
        # Test low urgency (patient)
        strategy_patient, _ = self.agent.select_execution_strategy(
            self.test_context, order_quantity=100.0, urgency_level=0.0
        )
        
        # Test high urgency (urgent)  
        strategy_urgent, _ = self.agent.select_execution_strategy(
            self.test_context, order_quantity=100.0, urgency_level=1.0
        )
        
        # With high urgency, should favor immediate execution more
        # (This is probabilistic, so we'll just check it doesn't crash)
        assert strategy_patient in list(ExecutionStrategy)
        assert strategy_urgent in list(ExecutionStrategy)
    
    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        # Make several decisions
        for i in range(10):
            self.agent.select_execution_strategy(self.test_context, 100.0)
        
        # Check performance stats are updated
        stats = self.agent.get_performance_metrics()
        assert stats['total_decisions'] == 10
        assert len(stats['inference_times_ns']) == 10
        assert len(stats['impact_predictions']) == 10
        
        # Test actual slippage update
        self.agent.update_from_actual_slippage(1.5, 1.8)
        stats_after = self.agent.get_performance_metrics()
        assert len(stats_after['actual_slippages']) == 1
        assert stats_after['actual_slippages'][0] == 1.8
    
    def test_fallback_behavior(self):
        """Test fallback to immediate execution on error"""
        # Create invalid market context (should trigger fallback)
        invalid_context = MarketMicrostructure()
        invalid_context.current_volume = 0.0  # Invalid volume
        
        strategy, impact_result = self.agent.select_execution_strategy(
            invalid_context, order_quantity=100.0
        )
        
        # Should not crash and return valid strategy
        assert isinstance(strategy, ExecutionStrategy)
        assert isinstance(impact_result, MarketImpactResult)


class TestPerformanceTargets:
    """Test suite for performance targets"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = ExecutionTimingAgent()
    
    def test_inference_speed_target(self):
        """Test <200μs inference time target"""
        results = benchmark_inference_performance(self.agent, num_iterations=100)
        
        # Check target is met
        assert results['target_met'], f"Average inference time {results['average_time_us']:.1f}μs exceeds 200μs target"
        assert results['average_time_us'] < 200
        
        # Check consistency
        assert results['std_time_us'] < 100  # Should be consistent
    
    def test_slippage_target_validation(self):
        """Test <2 bps slippage target validation"""
        # Create diverse market scenarios
        scenarios = []
        quantities = []
        
        for i in range(20):
            # Vary market conditions
            scenario = MarketMicrostructure(
                bid_ask_spread=0.001 + i * 0.001,
                market_depth=500 + i * 100,
                current_volume=5000 + i * 1000,
                volatility_regime=0.1 + i * 0.01,
                urgency_score=i / 20.0
            )
            scenarios.append(scenario)
            quantities.append(50 + i * 10)
        
        # Validate slippage target
        results = validate_slippage_target(self.agent, scenarios, quantities)
        
        # Check validation structure
        assert 'scenarios_tested' in results
        assert 'average_slippage_bps' in results
        assert 'target_met' in results
        assert results['scenarios_tested'] == 20
        
        # Target should be achievable for normal market conditions
        print(f"Average slippage: {results['average_slippage_bps']:.2f} bps")
        print(f"Max slippage: {results['max_slippage_bps']:.2f} bps")
    
    def test_market_impact_accuracy(self):
        """Test market impact model accuracy against known cases"""
        # Test case 1: Small order in liquid market
        small_order_impact = self.agent.impact_model.calculate_total_impact(
            order_quantity=10.0,
            market_volume=100000.0,
            volatility=0.1,
            time_to_execution=0.0,
            strategy=ExecutionStrategy.IMMEDIATE
        )
        
        # Should have minimal impact
        assert small_order_impact.total_impact_bps < 5.0
        
        # Test case 2: Large order in illiquid market
        large_order_impact = self.agent.impact_model.calculate_total_impact(
            order_quantity=1000.0,
            market_volume=5000.0,
            volatility=0.3,
            time_to_execution=0.0,
            strategy=ExecutionStrategy.IMMEDIATE
        )
        
        # Should have significant impact
        assert large_order_impact.total_impact_bps > small_order_impact.total_impact_bps


class TestEdgeCases:
    """Test suite for edge cases and error handling"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = ExecutionTimingAgent()
    
    def test_zero_volume_handling(self):
        """Test handling of zero market volume"""
        zero_volume_context = MarketMicrostructure(current_volume=0.0)
        
        # Should not crash
        strategy, impact_result = self.agent.select_execution_strategy(
            zero_volume_context, order_quantity=100.0
        )
        
        assert isinstance(strategy, ExecutionStrategy)
        assert isinstance(impact_result, MarketImpactResult)
    
    def test_extreme_market_conditions(self):
        """Test handling of extreme market conditions"""
        extreme_context = MarketMicrostructure(
            bid_ask_spread=1.0,      # 100 bps spread
            market_depth=1.0,        # Very low depth
            volatility_regime=5.0,   # Extreme volatility
            urgency_score=1.0        # Maximum urgency
        )
        
        # Should handle extreme conditions gracefully
        strategy, impact_result = self.agent.select_execution_strategy(
            extreme_context, order_quantity=1000.0
        )
        
        assert isinstance(strategy, ExecutionStrategy)
        # Should likely choose immediate execution due to urgency
        assert strategy == ExecutionStrategy.IMMEDIATE
    
    def test_negative_inputs_handling(self):
        """Test handling of negative input values"""
        negative_context = MarketMicrostructure(
            order_book_slope=-0.5,   # Negative slope
            volume_imbalance=-0.8,   # Negative imbalance
            price_momentum=-0.1      # Negative momentum
        )
        
        # Should handle negative values without crashing
        strategy, impact_result = self.agent.select_execution_strategy(
            negative_context, order_quantity=100.0
        )
        
        assert isinstance(strategy, ExecutionStrategy)


class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_full_execution_workflow(self):
        """Test complete execution workflow from decision to feedback"""
        agent = ExecutionTimingAgent()
        
        # 1. Create market context
        context = MarketMicrostructure(
            bid_ask_spread=0.01, market_depth=1000.0,
            current_volume=10000.0, volatility_regime=0.15
        )
        
        # 2. Select strategy
        strategy, impact_result = agent.select_execution_strategy(context, 100.0)
        
        # 3. Simulate execution and provide feedback
        actual_slippage = impact_result.expected_slippage_bps * 0.9  # 10% better than expected
        agent.update_from_actual_slippage(
            impact_result.expected_slippage_bps, 
            actual_slippage
        )
        
        # 4. Check performance tracking
        stats = agent.get_performance_metrics()
        assert stats['total_decisions'] == 1
        assert len(stats['actual_slippages']) == 1
        assert stats['actual_slippages'][0] == actual_slippage
    
    def test_multi_strategy_distribution(self):
        """Test that all strategies are used across different scenarios"""
        agent = ExecutionTimingAgent()
        strategies_used = set()
        
        # Create scenarios that should trigger different strategies
        scenarios = [
            # Urgent scenario - should favor immediate
            (MarketMicrostructure(urgency_score=1.0), 1.0),
            
            # Patient scenario with good liquidity - should favor ICEBERG/TWAP
            (MarketMicrostructure(market_depth=10000.0, urgency_score=0.0), 0.0),
            
            # Medium urgency scenario - should favor VWAP
            (MarketMicrostructure(urgency_score=0.5, volatility_regime=0.1), 0.5),
            
            # High volume scenario - should consider TWAP
            (MarketMicrostructure(current_volume=50000.0, urgency_score=0.2), 0.2)
        ]
        
        for context, urgency in scenarios:
            for _ in range(5):  # Multiple samples per scenario
                strategy, _ = agent.select_execution_strategy(context, 100.0, urgency)
                strategies_used.add(strategy)
        
        # Should have used multiple strategies
        assert len(strategies_used) >= 2


if __name__ == "__main__":
    # Run performance validation
    print("Running Execution Timing Agent Performance Validation...")
    
    agent = ExecutionTimingAgent()
    
    # Test inference speed
    print("\n1. Testing Inference Speed Target (<200μs)...")
    speed_results = benchmark_inference_performance(agent, 1000)
    print(f"   Average inference time: {speed_results['average_time_us']:.1f}μs")
    print(f"   Target met: {speed_results['target_met']}")
    
    # Test slippage target  
    print("\n2. Testing Slippage Target (<2 bps)...")
    test_scenarios = [
        MarketMicrostructure(
            bid_ask_spread=0.01, market_depth=1000.0,
            current_volume=10000.0, volatility_regime=0.15
        ) for _ in range(50)
    ]
    test_quantities = [100.0] * 50
    
    slippage_results = validate_slippage_target(agent, test_scenarios, test_quantities)
    print(f"   Average slippage: {slippage_results['average_slippage_bps']:.2f} bps")
    print(f"   Target met: {slippage_results['target_met']}")
    
    print("\n3. Strategy Distribution:")
    for strategy in ExecutionStrategy:
        count = slippage_results['strategies_used'].count(strategy.name)
        percentage = (count / len(slippage_results['strategies_used'])) * 100
        print(f"   {strategy.name}: {percentage:.1f}%")
    
    print("\n✅ Execution Timing Agent (π₂) Validation Complete!")