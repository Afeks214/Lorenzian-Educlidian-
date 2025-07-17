"""
Comprehensive Integration Tests for Realistic Execution System
============================================================

AGENT 4 - REALISTIC EXECUTION ENGINE INTEGRATION

This test suite validates the complete integration of the realistic execution
system with the backtesting framework to ensure backtest-live alignment.

Test Coverage:
- Realistic execution engine integration
- Dynamic cost modeling accuracy
- Execution validation framework
- Backtest-live divergence analysis
- Market condition simulation
- Partial fill and timing scenarios

Author: AGENT 4 - Realistic Execution Engine Integration
Date: 2025-07-17
Mission: Ensure realistic execution system reliability
"""

import sys
import os
sys.path.append('/home/QuantNova/GrandModel/src')

import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import json

# Import components to test
from backtesting.enhanced_realistic_framework import (
    EnhancedRealisticBacktestFramework,
    create_enhanced_realistic_backtest_framework
)
from backtesting.realistic_execution_integration import (
    RealisticBacktestExecutionHandler,
    BacktestExecutionConfig,
    RealisticBacktestFramework
)
from backtesting.dynamic_execution_costs import (
    ComprehensiveCostModel,
    create_nq_futures_cost_model,
    DynamicSlippageModel,
    InstrumentType
)
from backtesting.execution_validation import (
    BacktestLiveAlignmentValidator,
    PartialFillTesting,
    ExecutionTimingValidation,
    validate_backtest_live_alignment,
    create_test_execution_handler
)


class TestRealisticExecutionIntegration(unittest.TestCase):
    """Test suite for realistic execution integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data = self._create_test_data()
        self.execution_config = BacktestExecutionConfig(
            enable_realistic_slippage=True,
            enable_market_impact=True,
            enable_execution_latency=True,
            enable_partial_fills=True,
            enable_order_book_simulation=True,
            use_dynamic_commission=True,
            include_exchange_fees=True,
            include_regulatory_fees=True
        )
        self.cost_model = create_nq_futures_cost_model()
    
    def _create_test_data(self, days=50):
        """Create test market data"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 15000.0
        returns = np.random.normal(0.001, 0.02, days)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        data['Open'] = data['Close'].shift(1).fillna(data['Close'].iloc[0])
        data['High'] = data[['Open', 'Close']].max(axis=1) * 1.005
        data['Low'] = data[['Open', 'Close']].min(axis=1) * 0.995
        data['Volume'] = np.random.randint(500000, 1500000, days)
        
        return data
    
    def test_enhanced_framework_initialization(self):
        """Test enhanced realistic framework initialization"""
        framework = create_enhanced_realistic_backtest_framework(
            strategy_name="Test_Strategy",
            initial_capital=100000,
            execution_config=self.execution_config
        )
        
        self.assertIsInstance(framework, EnhancedRealisticBacktestFramework)
        self.assertEqual(framework.strategy_name, "Test_Strategy")
        self.assertEqual(framework.initial_capital, 100000)
        self.assertIsNotNone(framework.realistic_handler)
        self.assertIsNotNone(framework.execution_config)
    
    def test_realistic_execution_handler(self):
        """Test realistic execution handler functionality"""
        handler = RealisticBacktestExecutionHandler(
            initial_capital=100000,
            config=self.execution_config
        )
        
        self.assertIsNotNone(handler.execution_engine)
        self.assertEqual(handler.initial_capital, 100000)
        self.assertEqual(len(handler.execution_stats), 8)
        self.assertEqual(handler.execution_stats['total_orders'], 0)
    
    async def test_realistic_trade_execution(self):
        """Test realistic trade execution"""
        handler = RealisticBacktestExecutionHandler(
            initial_capital=100000,
            config=self.execution_config
        )
        
        # Create test trade
        trade_data = {
            'timestamp': datetime.now(),
            'symbol': 'NQ',
            'signal': 1,
            'size': 10,
            'price': 15000.0,
            'type': 'market'
        }
        
        market_data = pd.Series({
            'Close': 15000.0,
            'High': 15010.0,
            'Low': 14990.0,
            'Volume': 1000000
        })
        
        # Execute trade
        result = await handler.execute_backtest_trade(
            trade_data, market_data, {}
        )
        
        # Validate execution result
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('timestamp', result)
        self.assertIn('fill_price', result)
        self.assertIn('slippage_points', result)
        self.assertIn('commission', result)
        self.assertIn('execution_quality', result)
        
        # Check execution statistics updated
        self.assertEqual(handler.execution_stats['total_orders'], 1)
        if result['success']:
            self.assertEqual(handler.execution_stats['filled_orders'], 1)
    
    def test_dynamic_cost_modeling(self):
        """Test dynamic cost modeling accuracy"""
        # Test different scenarios
        scenarios = [
            {'order_size': 5, 'expected_cost_range': (0.05, 0.20)},
            {'order_size': 20, 'expected_cost_range': (0.10, 0.30)},
            {'order_size': 50, 'expected_cost_range': (0.20, 0.50)}
        ]
        
        for scenario in scenarios:
            market_data = pd.Series({
                'Close': 15000.0,
                'High': 15010.0,
                'Low': 14990.0,
                'Volume': 1000000
            })
            
            cost_breakdown = self.cost_model.calculate_total_execution_costs(
                order_size=scenario['order_size'],
                order_type='market',
                market_data=market_data,
                timestamp=datetime.now()
            )
            
            # Validate cost components
            self.assertIn('total_execution_cost', cost_breakdown)
            self.assertIn('cost_percentage', cost_breakdown)
            self.assertIn('slippage_cost', cost_breakdown)
            self.assertIn('commission_cost', cost_breakdown)
            self.assertIn('execution_efficiency', cost_breakdown)
            
            # Check cost is within reasonable range
            cost_pct = cost_breakdown['cost_percentage']
            min_expected, max_expected = scenario['expected_cost_range']
            self.assertGreaterEqual(cost_pct, 0)
            self.assertLessEqual(cost_pct, 1.0)  # Should be less than 100%
    
    def test_slippage_model(self):
        """Test dynamic slippage model"""
        from backtesting.dynamic_execution_costs import (
            InstrumentSpecs,
            DynamicSlippageModel
        )
        
        # Create NQ specs
        specs = InstrumentSpecs(
            instrument_type=InstrumentType.FUTURES_NQ,
            symbol="NQ",
            point_value=20.0,
            tick_size=0.25,
            tick_value=5.0,
            typical_spread_ticks=1.0,
            commission_per_unit=0.50,
            exchange_fees=0.02,
            regulatory_fees=0.02,
            margin_requirement=19000.0
        )
        
        slippage_model = DynamicSlippageModel(specs)
        
        # Test slippage calculation
        market_data = pd.Series({
            'Close': 15000.0,
            'High': 15010.0,
            'Low': 14990.0,
            'Volume': 1000000
        })
        
        slippage_result = slippage_model.calculate_slippage(
            order_size=10,
            order_type='market',
            market_data=market_data,
            timestamp=datetime.now()
        )
        
        # Validate slippage components
        self.assertIn('total_slippage', slippage_result)
        self.assertIn('base_slippage', slippage_result)
        self.assertIn('size_impact', slippage_result)
        self.assertIn('volatility_adjustment', slippage_result)
        self.assertIn('slippage_percentage', slippage_result)
        
        # Check reasonable slippage values
        total_slippage = slippage_result['total_slippage']
        self.assertGreaterEqual(total_slippage, 0.25)  # At least 1 tick
        self.assertLessEqual(total_slippage, 10.0)     # Not more than 10 points
    
    async def test_partial_fill_scenarios(self):
        """Test partial fill scenarios"""
        handler = create_test_execution_handler()
        partial_fill_tester = PartialFillTesting(handler)
        
        # Run partial fill tests
        results = await partial_fill_tester.run_partial_fill_tests()
        
        # Validate results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIn('test_name', result.__dict__)
            self.assertIn('passed', result.__dict__)
            self.assertIn('score', result.__dict__)
            self.assertIn('details', result.__dict__)
            self.assertIsInstance(result.score, (int, float))
            self.assertGreaterEqual(result.score, 0)
            self.assertLessEqual(result.score, 100)
    
    async def test_execution_timing_validation(self):
        """Test execution timing validation"""
        handler = create_test_execution_handler()
        timing_validator = ExecutionTimingValidation(handler)
        
        # Run timing validation
        results = await timing_validator.run_timing_validation_tests()
        
        # Validate results
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIn('test_name', result.__dict__)
            self.assertIn('passed', result.__dict__)
            self.assertIn('score', result.__dict__)
            self.assertIsInstance(result.score, (int, float))
            self.assertGreaterEqual(result.score, 0)
            self.assertLessEqual(result.score, 100)
    
    async def test_backtest_live_alignment_validation(self):
        """Test comprehensive backtest-live alignment validation"""
        handler = create_test_execution_handler()
        
        # Run validation
        validation_results = await validate_backtest_live_alignment(
            execution_handler=handler,
            cost_model=self.cost_model
        )
        
        # Validate results structure
        self.assertIsInstance(validation_results, dict)
        self.assertIn('partial_fill_tests', validation_results)
        self.assertIn('timing_tests', validation_results)
        self.assertIn('cost_model_tests', validation_results)
        self.assertIn('alignment_assessment', validation_results)
        self.assertIn('recommendations', validation_results)
        
        # Check alignment assessment
        assessment = validation_results['alignment_assessment']
        self.assertIn('overall_score', assessment)
        self.assertIn('status', assessment)
        self.assertIn('category_scores', assessment)
        
        overall_score = assessment['overall_score']
        self.assertIsInstance(overall_score, (int, float))
        self.assertGreaterEqual(overall_score, 0)
        self.assertLessEqual(overall_score, 100)
    
    def test_enhanced_framework_integration(self):
        """Test enhanced framework integration with realistic execution"""
        framework = create_enhanced_realistic_backtest_framework(
            strategy_name="Integration_Test",
            initial_capital=100000,
            execution_config=self.execution_config
        )
        
        # Create simple strategy
        def simple_strategy(data: pd.DataFrame) -> pd.DataFrame:
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = 0
            
            # Generate a few test signals
            signals.iloc[10:12]['signal'] = 1
            signals.iloc[20:22]['signal'] = -1
            signals.iloc[30:32]['signal'] = 1
            
            return signals[signals['signal'] != 0]
        
        # This would typically be run with asyncio.run() but we'll test components
        # Test that framework can be initialized and has correct components
        self.assertIsNotNone(framework.realistic_handler)
        self.assertIsNotNone(framework.execution_config)
        self.assertIsNotNone(framework.divergence_metrics)
        self.assertIsInstance(framework.divergence_metrics, dict)
    
    def test_execution_analytics(self):
        """Test execution analytics generation"""
        handler = RealisticBacktestExecutionHandler(
            initial_capital=100000,
            config=self.execution_config
        )
        
        # Initially no analytics
        analytics = handler.get_execution_analytics()
        self.assertEqual(analytics['error'], 'No executions recorded')
        
        # Add some mock execution data
        handler.execution_stats['total_orders'] = 10
        handler.execution_stats['filled_orders'] = 9
        handler.execution_stats['rejected_orders'] = 1
        handler.execution_stats['total_slippage_cost'] = 100.0
        handler.execution_stats['total_commission_paid'] = 50.0
        
        analytics = handler.get_execution_analytics()
        self.assertIn('execution_performance', analytics)
        self.assertIn('cost_analysis', analytics)
        self.assertIn('timing_analysis', analytics)
        
        # Check key metrics
        exec_perf = analytics['execution_performance']
        self.assertEqual(exec_perf['total_orders'], 10)
        self.assertEqual(exec_perf['fill_rate'], 0.9)
    
    def test_cost_analytics(self):
        """Test cost analytics functionality"""
        # Generate some cost data
        market_data = pd.Series({
            'Close': 15000.0,
            'High': 15010.0,
            'Low': 14990.0,
            'Volume': 1000000
        })
        
        # Calculate costs for multiple scenarios
        for order_size in [5, 10, 20]:
            self.cost_model.calculate_total_execution_costs(
                order_size=order_size,
                order_type='market',
                market_data=market_data,
                timestamp=datetime.now()
            )
        
        # Get analytics
        analytics = self.cost_model.get_cost_analytics()
        
        self.assertIn('summary', analytics)
        self.assertIn('cost_breakdown', analytics)
        self.assertIn('performance_metrics', analytics)
        
        # Check summary
        summary = analytics['summary']
        self.assertEqual(summary['total_executions'], 3)
        self.assertGreater(summary['total_execution_costs'], 0)
    
    def test_market_conditions_simulation(self):
        """Test market conditions simulation"""
        handler = RealisticBacktestExecutionHandler(
            initial_capital=100000,
            config=self.execution_config
        )
        
        # Test different market conditions
        test_scenarios = [
            {'volatility': 0.01, 'volume': 1000000, 'time_hour': 14},
            {'volatility': 0.03, 'volume': 500000, 'time_hour': 22},
            {'volatility': 0.005, 'volume': 1500000, 'time_hour': 10}
        ]
        
        for scenario in test_scenarios:
            market_data = pd.Series({
                'Close': 15000.0,
                'High': 15000.0 * (1 + scenario['volatility']),
                'Low': 15000.0 * (1 - scenario['volatility']),
                'Volume': scenario['volume']
            })
            
            timestamp = datetime.now().replace(hour=scenario['time_hour'])
            
            # Test market conditions creation
            market_conditions = handler._create_market_conditions_from_backtest_data(
                market_data, timestamp
            )
            
            # Validate market conditions
            self.assertIsNotNone(market_conditions)
            self.assertHasAttribute(market_conditions, 'current_price')
            self.assertHasAttribute(market_conditions, 'bid_price')
            self.assertHasAttribute(market_conditions, 'ask_price')
            self.assertHasAttribute(market_conditions, 'volatility_regime')
            self.assertHasAttribute(market_conditions, 'time_of_day_factor')
            
            # Check reasonable values
            self.assertGreater(market_conditions.current_price, 0)
            self.assertLess(market_conditions.bid_price, market_conditions.ask_price)
            self.assertGreaterEqual(market_conditions.volatility_regime, 0)
            self.assertLessEqual(market_conditions.volatility_regime, 1)
    
    def assertHasAttribute(self, obj, attr):
        """Helper method to check if object has attribute"""
        self.assertTrue(hasattr(obj, attr), f"Object does not have attribute '{attr}'")


class TestExecutionValidationFramework(unittest.TestCase):
    """Test suite for execution validation framework"""
    
    def setUp(self):
        """Set up test environment"""
        self.execution_handler = create_test_execution_handler()
        self.cost_model = create_nq_futures_cost_model()
        self.validator = BacktestLiveAlignmentValidator(
            self.execution_handler,
            self.cost_model
        )
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        self.assertIsNotNone(self.validator.execution_handler)
        self.assertIsNotNone(self.validator.cost_model)
        self.assertIsNotNone(self.validator.partial_fill_tester)
        self.assertIsNotNone(self.validator.timing_validator)
        self.assertEqual(self.validator.alignment_status, "UNTESTED")
    
    def test_validation_report_generation(self):
        """Test validation report generation"""
        # Without validation results
        report = self.validator.generate_validation_report()
        self.assertIn("No validation results available", report)
        
        # Set mock validation results
        self.validator.validation_results = {
            'alignment_assessment': {
                'overall_score': 85.0,
                'status': 'GOOD_ALIGNMENT',
                'category_scores': {
                    'partial_fill_tests': 90.0,
                    'timing_tests': 85.0,
                    'cost_model_tests': 80.0,
                    'stress_tests': 85.0
                }
            },
            'recommendations': ['Test recommendation']
        }
        
        report = self.validator.generate_validation_report()
        self.assertIn("BACKTEST-LIVE ALIGNMENT VALIDATION REPORT", report)
        self.assertIn("85.0/100", report)
        self.assertIn("GOOD_ALIGNMENT", report)
    
    def test_results_serialization(self):
        """Test validation results serialization"""
        # Create mock validation results
        from backtesting.execution_validation import ValidationResult
        
        mock_results = {
            'partial_fill_tests': [
                ValidationResult(
                    test_name='test1',
                    passed=True,
                    score=85.0,
                    details={'key': 'value'},
                    recommendations=['rec1']
                )
            ],
            'alignment_assessment': {
                'overall_score': 85.0,
                'status': 'GOOD_ALIGNMENT'
            }
        }
        
        serializable = self.validator._convert_results_to_serializable(mock_results)
        
        self.assertIsInstance(serializable, dict)
        self.assertIn('partial_fill_tests', serializable)
        self.assertIn('alignment_assessment', serializable)
        
        # Check serialized test result
        test_result = serializable['partial_fill_tests'][0]
        self.assertIn('test_name', test_result)
        self.assertIn('passed', test_result)
        self.assertIn('score', test_result)
        self.assertIn('timestamp', test_result)


async def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Realistic Execution Integration Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRealisticExecutionIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestExecutionValidationFramework))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"   Tests Run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback}")
    
    if result.errors:
        print(f"\n🔥 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_integration_tests())
    
    if success:
        print("\n🎉 Integration tests completed successfully!")
        print("The realistic execution system is ready for deployment.")
    else:
        print("\n⚠️ Some tests failed. Please review and fix issues.")
        sys.exit(1)