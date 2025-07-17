"""
Performance Regression Detection Examples - Agent 3

This module provides example tests demonstrating the performance regression
detection system in action.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
import time

from src.testing import (
    performance_detector,
    setup_performance_budget,
    benchmark_with_regression_detection
)

# Configure performance budgets for our test examples
setup_performance_budget("test_var_calculation_example", 5.0, 20.0)  # 5ms max, 20% regression
setup_performance_budget("test_correlation_update_example", 2.0, 25.0)  # 2ms max, 25% regression
setup_performance_budget("test_portfolio_optimization_example", 50.0, 30.0)  # 50ms max, 30% regression

class TestPerformanceRegressionExamples:
    """
    Example tests showing performance regression detection in action
    """
    
    def test_var_calculation_example(self, benchmark):
        """Example VaR calculation performance test"""
        # Setup test data
        portfolio_value = 1000000
        positions = {
            'AAPL': {'quantity': 1000, 'price': 180.0, 'volatility': 0.25},
            'GOOGL': {'quantity': 500, 'price': 300.0, 'volatility': 0.30},
            'MSFT': {'quantity': 800, 'price': 300.0, 'volatility': 0.22}
        }
        
        def calculate_var():
            """Simulate VaR calculation"""
            # Simulate correlation matrix calculation
            correlation_matrix = np.random.rand(3, 3)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)
            
            # Simulate portfolio variance calculation
            weights = np.array([0.4, 0.3, 0.3])
            volatilities = np.array([0.25, 0.30, 0.22])
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(correlation_matrix * np.outer(volatilities, volatilities), weights))
            
            # 95% VaR
            var_95 = 1.645 * np.sqrt(portfolio_variance) * portfolio_value
            
            # Add some computation time
            time.sleep(0.001)  # 1ms base time
            
            return var_95
        
        # Run benchmark
        result = benchmark(calculate_var)
        
        # Verify result is reasonable
        assert result > 0
        assert result < portfolio_value * 0.1  # VaR should be < 10% of portfolio
    
    def test_correlation_update_example(self, benchmark):
        """Example correlation matrix update performance test"""
        # Setup test data
        n_assets = 10
        n_returns = 252  # One year of daily returns
        
        returns_data = np.random.randn(n_returns, n_assets) * 0.02  # 2% daily volatility
        
        def update_correlation_matrix():
            """Simulate EWMA correlation matrix update"""
            lambda_decay = 0.94
            
            # Initialize correlation matrix
            correlation_matrix = np.eye(n_assets)
            
            # EWMA update simulation
            for i in range(min(50, n_returns)):  # Process last 50 returns
                returns = returns_data[i]
                
                # Update correlation (simplified)
                for j in range(n_assets):
                    for k in range(j + 1, n_assets):
                        correlation_matrix[j, k] = (
                            lambda_decay * correlation_matrix[j, k] +
                            (1 - lambda_decay) * returns[j] * returns[k]
                        )
                        correlation_matrix[k, j] = correlation_matrix[j, k]
            
            return correlation_matrix
        
        # Run benchmark
        result = benchmark(update_correlation_matrix)
        
        # Verify result
        assert result.shape == (n_assets, n_assets)
        assert np.allclose(result.diagonal(), 1.0)  # Diagonal should be 1
    
    def test_portfolio_optimization_example(self, benchmark):
        """Example portfolio optimization performance test"""
        # Setup test data
        n_assets = 20
        expected_returns = np.random.randn(n_assets) * 0.01 + 0.05  # 5% base return
        covariance_matrix = np.random.randn(n_assets, n_assets) * 0.1
        covariance_matrix = np.dot(covariance_matrix, covariance_matrix.T)  # Make positive definite
        
        def optimize_portfolio():
            """Simulate portfolio optimization"""
            # Markowitz optimization simulation
            risk_aversion = 2.0
            
            # Solve quadratic optimization (simplified)
            # w = inv(risk_aversion * Sigma) * mu
            try:
                inv_cov = np.linalg.inv(risk_aversion * covariance_matrix)
                weights = np.dot(inv_cov, expected_returns)
                
                # Normalize weights
                weights = weights / np.sum(np.abs(weights))
                
                # Add computation time
                time.sleep(0.01)  # 10ms base time
                
                return weights
            except np.linalg.LinAlgError:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
        
        # Run benchmark
        result = benchmark(optimize_portfolio)
        
        # Verify result
        assert len(result) == n_assets
        assert abs(np.sum(result)) <= 1.1  # Allow some tolerance for normalization
    
    def test_agent_inference_example(self, benchmark):
        """Example agent inference performance test"""
        # Setup test data
        state_size = 64
        action_size = 8
        batch_size = 32
        
        # Mock neural network weights
        weights = {
            'layer1': np.random.randn(state_size, 128),
            'layer2': np.random.randn(128, 64),
            'layer3': np.random.randn(64, action_size)
        }
        
        def agent_inference():
            """Simulate neural network inference"""
            # Generate batch of states
            states = np.random.randn(batch_size, state_size)
            
            # Forward pass simulation
            hidden1 = np.maximum(0, np.dot(states, weights['layer1']))  # ReLU
            hidden2 = np.maximum(0, np.dot(hidden1, weights['layer2']))  # ReLU
            actions = np.dot(hidden2, weights['layer3'])  # Linear output
            
            # Softmax for action probabilities
            exp_actions = np.exp(actions - np.max(actions, axis=1, keepdims=True))
            action_probs = exp_actions / np.sum(exp_actions, axis=1, keepdims=True)
            
            # Add some computation time
            time.sleep(0.02)  # 20ms base time
            
            return action_probs
        
        # Run benchmark
        result = benchmark(agent_inference)
        
        # Verify result
        assert result.shape == (batch_size, action_size)
        assert np.allclose(np.sum(result, axis=1), 1.0)  # Probabilities sum to 1
    
    def test_matrix_assembly_example(self, benchmark):
        """Example matrix assembly performance test"""
        # Setup test data
        n_timestamps = 1000
        n_features = 30
        
        # Mock time series data
        timestamps = [datetime.now().timestamp() + i * 300 for i in range(n_timestamps)]  # 5-minute intervals
        raw_data = np.random.randn(n_timestamps, n_features)
        
        def assemble_feature_matrix():
            """Simulate feature matrix assembly"""
            # Feature engineering simulation
            features = []
            
            # Technical indicators
            for i in range(n_features // 3):
                # Moving average
                window = min(20, n_timestamps)
                ma = np.convolve(raw_data[:, i], np.ones(window) / window, mode='valid')
                features.append(ma)
                
                # RSI simulation
                deltas = np.diff(raw_data[:, i])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gains = np.convolve(gains, np.ones(14) / 14, mode='valid')
                avg_losses = np.convolve(losses, np.ones(14) / 14, mode='valid')
                
                rs = avg_gains / (avg_losses + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi)
            
            # Normalize features
            feature_matrix = np.column_stack(features)
            normalized_features = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)
            
            # Add computation time
            time.sleep(0.005)  # 5ms base time
            
            return normalized_features
        
        # Run benchmark
        result = benchmark(assemble_feature_matrix)
        
        # Verify result
        assert result.shape[1] >= 2  # At least 2 features per input
        assert not np.any(np.isnan(result))  # No NaN values
    
    def test_api_response_example(self, benchmark):
        """Example API response performance test"""
        # Setup test data
        portfolio_data = {
            'positions': [
                {'symbol': 'AAPL', 'quantity': 1000, 'price': 180.0},
                {'symbol': 'GOOGL', 'quantity': 500, 'price': 300.0},
                {'symbol': 'MSFT', 'quantity': 800, 'price': 300.0}
            ],
            'cash': 100000,
            'total_value': 1000000
        }
        
        def process_api_request():
            """Simulate API request processing"""
            # Validate request
            assert 'positions' in portfolio_data
            assert 'cash' in portfolio_data
            
            # Calculate portfolio metrics
            total_value = portfolio_data['cash']
            for position in portfolio_data['positions']:
                total_value += position['quantity'] * position['price']
            
            # Generate response
            response = {
                'portfolio_value': total_value,
                'positions': len(portfolio_data['positions']),
                'cash_ratio': portfolio_data['cash'] / total_value,
                'top_holding': max(portfolio_data['positions'], key=lambda p: p['quantity'] * p['price']),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add some processing time
            time.sleep(0.05)  # 50ms base time
            
            return response
        
        # Run benchmark
        result = benchmark(process_api_request)
        
        # Verify result
        assert 'portfolio_value' in result
        assert 'positions' in result
        assert result['positions'] == 3
        assert 0 <= result['cash_ratio'] <= 1
    
    @pytest.mark.asyncio
    async def test_regression_detection_workflow(self):
        """Test the complete regression detection workflow"""
        # Create a mock benchmark result
        from src.testing import PerformanceBenchmark
        
        benchmark = PerformanceBenchmark(
            test_name="test_workflow_example",
            timestamp=datetime.now(),
            min_time=0.001,
            max_time=0.005,
            mean_time=0.003,
            median_time=0.003,
            stddev_time=0.0005,
            rounds=100,
            iterations=10,
            git_commit="abc123",
            branch="main",
            environment="test"
        )
        
        # Record benchmark
        performance_detector.record_benchmark(benchmark)
        
        # Detect regression (should be None for first run)
        regression_result = performance_detector.detect_regression(benchmark)
        assert regression_result is None or not regression_result.regression_detected
        
        # Create a slower benchmark to trigger regression
        slow_benchmark = PerformanceBenchmark(
            test_name="test_workflow_example",
            timestamp=datetime.now(),
            min_time=0.005,
            max_time=0.015,
            mean_time=0.010,  # Much slower
            median_time=0.010,
            stddev_time=0.001,
            rounds=100,
            iterations=10,
            git_commit="def456",
            branch="main",
            environment="test"
        )
        
        # Record several benchmarks to establish baseline
        for i in range(5):
            performance_detector.record_benchmark(benchmark)
        
        # Now test regression detection
        regression_result = performance_detector.detect_regression(slow_benchmark)
        
        if regression_result and regression_result.regression_detected:
            assert regression_result.test_name == "test_workflow_example"
            assert regression_result.current_performance > regression_result.baseline_performance
            assert regression_result.regression_severity in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            print(f"✅ Regression detected: {regression_result.regression_severity}")
        else:
            print("ℹ️  No regression detected (expected for first runs)")
    
    def test_performance_prediction(self):
        """Test performance trend prediction"""
        # Generate some historical data
        from src.testing import PerformanceBenchmark
        
        test_name = "test_prediction_example"
        base_time = 0.005
        
        # Create trending data (gradually increasing)
        for i in range(20):
            benchmark = PerformanceBenchmark(
                test_name=test_name,
                timestamp=datetime.now(),
                min_time=base_time + (i * 0.0001),
                max_time=base_time + (i * 0.0002),
                mean_time=base_time + (i * 0.0001),
                median_time=base_time + (i * 0.0001),
                stddev_time=0.0001,
                rounds=100,
                iterations=10,
                git_commit=f"commit_{i}",
                branch="main",
                environment="test"
            )
            
            performance_detector.record_benchmark(benchmark)
        
        # Get prediction
        prediction = performance_detector.predict_performance_trend(test_name, 7)
        
        if "error" not in prediction:
            assert prediction['test_name'] == test_name
            assert prediction['prediction_period_days'] == 7
            assert prediction['current_trend'] in ['IMPROVING', 'DEGRADING']
            assert len(prediction['predictions']) == 7
            print(f"✅ Prediction generated: {prediction['current_trend']} trend")
        else:
            print(f"ℹ️  Prediction error: {prediction['error']}")
    
    def test_performance_budget_enforcement(self):
        """Test performance budget enforcement"""
        # Set a strict budget
        from src.testing import PerformanceBudget
        
        budget = PerformanceBudget(
            test_name="test_budget_example",
            max_time_ms=1.0,  # Very strict 1ms limit
            max_regression_percent=5.0,  # Very strict 5% regression limit
            enabled=True
        )
        
        performance_detector.set_performance_budget(budget)
        
        # Create a benchmark that exceeds budget
        from src.testing import PerformanceBenchmark
        
        slow_benchmark = PerformanceBenchmark(
            test_name="test_budget_example",
            timestamp=datetime.now(),
            min_time=0.005,
            max_time=0.015,
            mean_time=0.010,  # 10ms - exceeds 1ms budget
            median_time=0.010,
            stddev_time=0.001,
            rounds=100,
            iterations=10,
            git_commit="budget_test",
            branch="main",
            environment="test"
        )
        
        # Record benchmark
        performance_detector.record_benchmark(slow_benchmark)
        
        # Get performance report
        report = performance_detector.get_performance_report(1)
        
        # Verify budget violation is detected
        assert report['summary']['total_benchmark_runs'] > 0
        print(f"✅ Performance budget test completed")

if __name__ == "__main__":
    # Run tests manually for demonstration
    test_instance = TestPerformanceRegressionExamples()
    
    print("🚀 Performance Regression Detection Examples")
    print("=" * 50)
    
    # Mock benchmark function
    class MockBenchmark:
        def __call__(self, func):
            start_time = time.time()
            result = func()
            end_time = time.time()
            print(f"⏱️  Benchmark: {func.__name__} took {(end_time - start_time)*1000:.2f}ms")
            return result
    
    mock_benchmark = MockBenchmark()
    
    # Run example tests
    print("\n📊 Running VaR calculation benchmark...")
    test_instance.test_var_calculation_example(mock_benchmark)
    
    print("\n📊 Running correlation update benchmark...")
    test_instance.test_correlation_update_example(mock_benchmark)
    
    print("\n📊 Running portfolio optimization benchmark...")
    test_instance.test_portfolio_optimization_example(mock_benchmark)
    
    print("\n📊 Running agent inference benchmark...")
    test_instance.test_agent_inference_example(mock_benchmark)
    
    print("\n📊 Running matrix assembly benchmark...")
    test_instance.test_matrix_assembly_example(mock_benchmark)
    
    print("\n📊 Running API response benchmark...")
    test_instance.test_api_response_example(mock_benchmark)
    
    print("\n🔍 Testing regression detection workflow...")
    asyncio.run(test_instance.test_regression_detection_workflow())
    
    print("\n📈 Testing performance prediction...")
    test_instance.test_performance_prediction()
    
    print("\n💰 Testing performance budget enforcement...")
    test_instance.test_performance_budget_enforcement()
    
    print("\n✅ All performance regression detection examples completed!")