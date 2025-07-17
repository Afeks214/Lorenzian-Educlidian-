"""
Performance Benchmarking Script

Tests and validates the performance improvements in baseline strategies.
Provides benchmarks for various agent types and technical indicators.
"""

import numpy as np
import time
from typing import Dict, List, Any, Tuple
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Import all baseline agents
from .random_agent import RandomAgent, BiasedRandomAgent
from .rule_based_agent import RuleBasedAgent, TechnicalRuleBasedAgent, EnhancedRuleBasedAgent
from .rule_based_agent import AdvancedMomentumAgent, AdvancedMeanReversionAgent
from .momentum_strategies import MACDCrossoverAgent, RSIAgent, DualMomentumAgent, BreakoutAgent
from .benchmark_agents import BuyAndHoldAgent, EqualWeightAgent, MarketCapWeightedAgent, SectorRotationAgent, RiskParityAgent
from .technical_indicators import TechnicalIndicators, AdvancedTechnicalIndicators, PerformanceOptimizer


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite
    """
    
    def __init__(self, num_agents: int = 10, num_steps: int = 1000):
        """
        Initialize benchmark
        
        Args:
            num_agents: Number of agents per type to test
            num_steps: Number of simulation steps
        """
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.results = {}
        
        # Generate synthetic price data
        self.price_data = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self) -> Dict[str, np.ndarray]:
        """Generate synthetic market data for testing"""
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic price series
        returns = np.random.normal(0.0005, 0.02, self.num_steps)  # Daily returns
        prices = np.exp(np.cumsum(returns)) * 100  # Start at $100
        
        # Generate OHLC data
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, self.num_steps)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, self.num_steps)))
        volume = np.random.lognormal(10, 0.5, self.num_steps)
        
        # Generate volatility
        volatility = np.random.lognormal(-2, 0.3, self.num_steps)
        
        return {
            'prices': prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume,
            'volatility': volatility
        }
    
    def _create_observation(self, step: int) -> Dict[str, Any]:
        """Create observation for step"""
        return {
            'features': np.array([
                self.price_data['prices'][step],
                self.price_data['high'][step],
                self.price_data['low'][step],
                self.price_data['close'][step],
                self.price_data['volume'][step]
            ]),
            'shared_context': np.array([
                self.price_data['prices'][step],
                returns_if_available := (self.price_data['prices'][step] / self.price_data['prices'][max(0, step-1)] - 1) if step > 0 else 0,
                np.log(self.price_data['volatility'][step])
            ]),
            'synergy_active': np.random.randint(0, 2),
            'synergy_info': {
                'direction': np.random.choice([-1, 0, 1]),
                'confidence': np.random.uniform(0.3, 0.9),
                'type': np.random.choice(['TYPE_1', 'TYPE_2', 'TYPE_3', 'TYPE_4'])
            }
        }
    
    def benchmark_agent_type(self, agent_class, agent_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Benchmark a specific agent type"""
        if agent_config is None:
            agent_config = {}
        
        agents = [agent_class(agent_config) for _ in range(self.num_agents)]
        
        # Timing
        start_time = time.time()
        
        # Run simulation
        for step in range(self.num_steps):
            observation = self._create_observation(step)
            
            for agent in agents:
                action = agent.get_action(observation)
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        actions_per_second = (self.num_agents * self.num_steps) / total_time
        
        # Get agent statistics
        stats = [agent.get_statistics() for agent in agents]
        
        return {
            'total_time': total_time,
            'actions_per_second': actions_per_second,
            'avg_time_per_action': total_time / (self.num_agents * self.num_steps),
            'agent_stats': stats
        }
    
    def benchmark_technical_indicators(self) -> Dict[str, Any]:
        """Benchmark technical indicator calculations"""
        prices = self.price_data['prices']
        high = self.price_data['high']
        low = self.price_data['low']
        close = self.price_data['close']
        
        results = {}
        
        # Test basic indicators
        basic_indicators = [
            ('SMA_20', lambda: TechnicalIndicators.sma(prices, 20)),
            ('EMA_20', lambda: TechnicalIndicators.ema(prices, 20)),
            ('RSI_14', lambda: TechnicalIndicators.rsi(prices, 14)),
            ('MACD', lambda: TechnicalIndicators.macd(prices)),
            ('Bollinger_Bands', lambda: TechnicalIndicators.bollinger_bands(prices)),
            ('ATR', lambda: TechnicalIndicators.atr(high, low, close)),
            ('Stochastic', lambda: TechnicalIndicators.stochastic(high, low, close))
        ]
        
        for name, func in basic_indicators:
            # Time single calculation
            start_time = time.time()
            result = func()
            end_time = time.time()
            
            results[name] = {
                'calculation_time': end_time - start_time,
                'result_shape': result.shape if hasattr(result, 'shape') else 'tuple'
            }
        
        # Test advanced indicators
        advanced_indicators = [
            ('Adaptive_EMA', lambda: AdvancedTechnicalIndicators.adaptive_ema(prices)),
            ('Keltner_Channels', lambda: AdvancedTechnicalIndicators.keltner_channels(high, low, close)),
            ('CCI', lambda: AdvancedTechnicalIndicators.commodity_channel_index(high, low, close)),
            ('Parabolic_SAR', lambda: AdvancedTechnicalIndicators.parabolic_sar(high, low)),
            ('Vortex_Indicator', lambda: AdvancedTechnicalIndicators.vortex_indicator(high, low, close))
        ]
        
        for name, func in advanced_indicators:
            start_time = time.time()
            result = func()
            end_time = time.time()
            
            results[f"Advanced_{name}"] = {
                'calculation_time': end_time - start_time,
                'result_shape': result.shape if hasattr(result, 'shape') else 'tuple'
            }
        
        return results
    
    def benchmark_batch_processing(self) -> Dict[str, Any]:
        """Benchmark batch processing capabilities"""
        # Create multiple price series
        batch_size = 20
        price_batches = []
        
        for i in range(batch_size):
            # Create variations of the base price data
            noise = np.random.normal(0, 0.1, len(self.price_data['prices']))
            varied_prices = self.price_data['prices'] * (1 + noise)
            price_batches.append(varied_prices)
        
        price_batch = np.array(price_batches)
        
        # Test batch processing
        start_time = time.time()
        batch_results = PerformanceOptimizer.vectorized_indicator_batch(
            price_batch, TechnicalIndicators.sma, period=20
        )
        batch_time = time.time() - start_time
        
        # Test individual processing
        start_time = time.time()
        individual_results = []
        for prices in price_batches:
            result = TechnicalIndicators.sma(prices, 20)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        speedup = individual_time / batch_time if batch_time > 0 else 0
        
        return {
            'batch_time': batch_time,
            'individual_time': individual_time,
            'speedup': speedup,
            'batch_size': batch_size
        }
    
    def benchmark_parallel_processing(self) -> Dict[str, Any]:
        """Benchmark parallel processing capabilities"""
        # Create multiple price series
        num_series = 100
        price_series_list = []
        
        for i in range(num_series):
            noise = np.random.normal(0, 0.05, len(self.price_data['prices']))
            varied_prices = self.price_data['prices'] * (1 + noise)
            price_series_list.append(varied_prices)
        
        # Define indicator configurations
        indicator_configs = [
            {'name': 'SMA_20', 'function': 'sma', 'params': {'period': 20}},
            {'name': 'EMA_20', 'function': 'ema', 'params': {'period': 20}},
            {'name': 'RSI_14', 'function': 'rsi', 'params': {'period': 14}},
        ]
        
        # Test parallel processing
        start_time = time.time()
        parallel_results = PerformanceOptimizer.parallel_indicator_calculation(
            price_series_list, indicator_configs
        )
        parallel_time = time.time() - start_time
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = {}
        for config in indicator_configs:
            indicator_name = config['name']
            indicator_func = getattr(TechnicalIndicators, config['function'])
            params = config.get('params', {})
            
            results = []
            for prices in price_series_list:
                result = indicator_func(prices, **params)
                results.append(result)
            sequential_results[indicator_name] = np.array(results)
        sequential_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        return {
            'parallel_time': parallel_time,
            'sequential_time': sequential_time,
            'speedup': speedup,
            'num_series': num_series,
            'num_indicators': len(indicator_configs)
        }
    
    def benchmark_caching(self) -> Dict[str, Any]:
        """Benchmark caching performance"""
        prices = self.price_data['prices']
        cache_dict = {}
        
        # Test with caching
        start_time = time.time()
        for i in range(50):  # Repeat calculations
            result = PerformanceOptimizer.cached_indicator_calculation(
                prices, TechnicalIndicators.sma, cache_dict, 'sma_20', period=20
            )
        cached_time = time.time() - start_time
        
        # Test without caching
        start_time = time.time()
        for i in range(50):
            result = TechnicalIndicators.sma(prices, 20)
        uncached_time = time.time() - start_time
        
        speedup = uncached_time / cached_time if cached_time > 0 else 0
        
        return {
            'cached_time': cached_time,
            'uncached_time': uncached_time,
            'speedup': speedup,
            'cache_size': len(cache_dict)
        }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        results = {}
        
        print("Running comprehensive performance benchmark...")
        
        # Benchmark different agent types
        agent_types = [
            ('RandomAgent', RandomAgent),
            ('RuleBasedAgent', RuleBasedAgent),
            ('TechnicalRuleBasedAgent', TechnicalRuleBasedAgent),
            ('AdvancedMomentumAgent', AdvancedMomentumAgent),
            ('AdvancedMeanReversionAgent', AdvancedMeanReversionAgent),
            ('MACDCrossoverAgent', MACDCrossoverAgent),
            ('RSIAgent', RSIAgent),
            ('BuyAndHoldAgent', BuyAndHoldAgent),
            ('EqualWeightAgent', EqualWeightAgent),
            ('MarketCapWeightedAgent', MarketCapWeightedAgent),
            ('SectorRotationAgent', SectorRotationAgent),
            ('RiskParityAgent', RiskParityAgent)
        ]
        
        print(f"Benchmarking {len(agent_types)} agent types...")
        for name, agent_class in agent_types:
            print(f"  Benchmarking {name}...")
            try:
                results[f'agent_{name}'] = self.benchmark_agent_type(agent_class)
            except Exception as e:
                print(f"    Error benchmarking {name}: {e}")
                results[f'agent_{name}'] = {'error': str(e)}
        
        # Benchmark technical indicators
        print("Benchmarking technical indicators...")
        results['technical_indicators'] = self.benchmark_technical_indicators()
        
        # Benchmark batch processing
        print("Benchmarking batch processing...")
        results['batch_processing'] = self.benchmark_batch_processing()
        
        # Benchmark parallel processing
        print("Benchmarking parallel processing...")
        results['parallel_processing'] = self.benchmark_parallel_processing()
        
        # Benchmark caching
        print("Benchmarking caching...")
        results['caching'] = self.benchmark_caching()
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate formatted performance report"""
        report = []
        report.append("=" * 80)
        report.append("BASELINE STRATEGIES PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Agent performance summary
        report.append("AGENT PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        agent_results = {k: v for k, v in results.items() if k.startswith('agent_')}
        
        # Sort by performance
        sorted_agents = sorted(
            agent_results.items(),
            key=lambda x: x[1].get('actions_per_second', 0),
            reverse=True
        )
        
        for name, data in sorted_agents:
            if 'error' in data:
                report.append(f"{name:25}: ERROR - {data['error']}")
            else:
                report.append(f"{name:25}: {data['actions_per_second']:8.0f} actions/sec")
        
        report.append("")
        
        # Technical indicators performance
        report.append("TECHNICAL INDICATORS PERFORMANCE")
        report.append("-" * 40)
        
        ti_results = results.get('technical_indicators', {})
        sorted_indicators = sorted(
            ti_results.items(),
            key=lambda x: x[1]['calculation_time']
        )
        
        for name, data in sorted_indicators:
            report.append(f"{name:20}: {data['calculation_time']*1000:8.2f} ms")
        
        report.append("")
        
        # Optimization performance
        report.append("OPTIMIZATION PERFORMANCE")
        report.append("-" * 40)
        
        batch_results = results.get('batch_processing', {})
        if batch_results:
            report.append(f"Batch Processing Speedup: {batch_results['speedup']:.2f}x")
        
        parallel_results = results.get('parallel_processing', {})
        if parallel_results:
            report.append(f"Parallel Processing Speedup: {parallel_results['speedup']:.2f}x")
        
        caching_results = results.get('caching', {})
        if caching_results:
            report.append(f"Caching Speedup: {caching_results['speedup']:.2f}x")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def run_performance_benchmark():
    """Run performance benchmark and display results"""
    benchmark = PerformanceBenchmark(num_agents=5, num_steps=500)
    results = benchmark.run_comprehensive_benchmark()
    report = benchmark.generate_performance_report(results)
    
    print(report)
    
    # Save results to file
    with open('/home/QuantNova/GrandModel/baselines/performance_results.txt', 'w') as f:
        f.write(report)
    
    return results


if __name__ == "__main__":
    run_performance_benchmark()