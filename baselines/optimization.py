"""
Baseline Strategy Optimization Module
Agent 5 - System Integration and Production Architecture

Vectorized operations for all baseline strategies, caching layer for expensive computations,
parallel execution framework, and performance monitoring.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum
import threading
from contextlib import contextmanager
import functools
import psutil
import tracemalloc
from pathlib import Path
import json
import hashlib

from .random_agent import RandomAgent, BiasedRandomAgent, ContextualRandomAgent
from .rule_based_agent import RuleBasedAgent, EnhancedRuleBasedAgent
from .technical_indicators import TechnicalIndicators
from ..analysis.performance_optimizer import (
    PerformanceOptimizer, 
    OptimizationMode,
    PerformanceMetrics,
    cached,
    optimized
)


class BaselineStrategy(Enum):
    """Baseline strategy types"""
    RANDOM = "random"
    BIASED_RANDOM = "biased_random"
    CONTEXTUAL_RANDOM = "contextual_random"
    RULE_BASED = "rule_based"
    ENHANCED_RULE_BASED = "enhanced_rule_based"
    TECHNICAL_INDICATORS = "technical_indicators"
    BUY_AND_HOLD = "buy_and_hold"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class BaselineResult:
    """Result from baseline strategy execution"""
    strategy_name: str
    actions: np.ndarray
    performance_metrics: Dict[str, Any]
    execution_time: float
    memory_usage: int
    cache_stats: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorizedParams:
    """Parameters for vectorized operations"""
    batch_size: int = 1000
    chunk_size: int = 10000
    parallel_workers: int = 4
    use_gpu: bool = False
    memory_limit_mb: int = 1000


class BaselineOptimizer:
    """
    Optimized baseline strategy execution with vectorization and caching
    
    Features:
    - Vectorized operations for all baseline strategies
    - Multi-level caching for expensive computations
    - Parallel execution framework
    - Memory usage monitoring
    - Performance profiling
    """
    
    def __init__(self, 
                 performance_optimizer: Optional[PerformanceOptimizer] = None,
                 vectorized_params: Optional[VectorizedParams] = None):
        
        self.performance_optimizer = performance_optimizer or PerformanceOptimizer()
        self.vectorized_params = vectorized_params or VectorizedParams()
        self.logger = logging.getLogger(__name__)
        
        # Strategy registry
        self._strategy_registry = {
            BaselineStrategy.RANDOM: self._execute_random_vectorized,
            BaselineStrategy.BIASED_RANDOM: self._execute_biased_random_vectorized,
            BaselineStrategy.CONTEXTUAL_RANDOM: self._execute_contextual_random_vectorized,
            BaselineStrategy.RULE_BASED: self._execute_rule_based_vectorized,
            BaselineStrategy.ENHANCED_RULE_BASED: self._execute_enhanced_rule_based_vectorized,
            BaselineStrategy.TECHNICAL_INDICATORS: self._execute_technical_indicators_vectorized,
            BaselineStrategy.BUY_AND_HOLD: self._execute_buy_and_hold_vectorized,
            BaselineStrategy.MOMENTUM: self._execute_momentum_vectorized,
            BaselineStrategy.MEAN_REVERSION: self._execute_mean_reversion_vectorized
        }
        
        # Performance tracking
        self._execution_stats = {}
        self._memory_stats = {}
        
        # Initialize agents with optimization
        self._initialize_optimized_agents()
        
    def _initialize_optimized_agents(self) -> None:
        """Initialize optimized agent instances"""
        self.agents = {}
        
        # Random agents
        self.agents[BaselineStrategy.RANDOM] = RandomAgent({'random_seed': 42})
        self.agents[BaselineStrategy.BIASED_RANDOM] = BiasedRandomAgent({
            'random_seed': 42,
            'bias_type': 'bullish',
            'bias_strength': 0.2
        })
        self.agents[BaselineStrategy.CONTEXTUAL_RANDOM] = ContextualRandomAgent({
            'random_seed': 42,
            'volatility_sensitivity': 1.0
        })
        
        # Rule-based agents
        self.agents[BaselineStrategy.RULE_BASED] = RuleBasedAgent({
            'trend_threshold': 0.02,
            'volatility_threshold': 0.05
        })
        self.agents[BaselineStrategy.ENHANCED_RULE_BASED] = EnhancedRuleBasedAgent({
            'trend_threshold': 0.02,
            'volatility_threshold': 0.05,
            'regime_detection': True
        })
        
        # Technical indicators
        self.agents[BaselineStrategy.TECHNICAL_INDICATORS] = TechnicalIndicators({
            'sma_window': 20,
            'rsi_window': 14,
            'macd_params': {'fast': 12, 'slow': 26, 'signal': 9}
        })
        
    @optimized(cache_ttl=3600, parallel=True, memory_optimize=True, profile=True)
    def execute_strategy(self, 
                        strategy: BaselineStrategy,
                        observations: np.ndarray,
                        config: Optional[Dict[str, Any]] = None) -> BaselineResult:
        """
        Execute baseline strategy with optimization
        
        Args:
            strategy: Strategy type to execute
            observations: Observation data
            config: Strategy configuration
            
        Returns:
            BaselineResult with performance metrics
        """
        config = config or {}
        
        # Track execution start
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss
        
        # Memory tracking
        tracemalloc.start()
        
        try:
            # Get strategy executor
            strategy_executor = self._strategy_registry.get(strategy)
            if not strategy_executor:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            # Execute strategy
            actions = strategy_executor(observations, config)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(actions, observations)
            
            # Get execution stats
            end_time = time.time()
            end_memory = process.memory_info().rss
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Memory tracking
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Cache statistics
            cache_stats = self.performance_optimizer.cache.stats()
            
            # Store execution statistics
            self._execution_stats[strategy.value] = {
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'cache_hits': cache_stats.get('multi_level_stats', {}).get('l1_hits', 0),
                'cache_misses': cache_stats.get('multi_level_stats', {}).get('misses', 0)
            }
            
            # Create result
            result = BaselineResult(
                strategy_name=strategy.value,
                actions=actions,
                performance_metrics=performance_metrics,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cache_stats=cache_stats,
                metadata={
                    'config': config,
                    'observations_shape': observations.shape,
                    'peak_memory': peak_memory,
                    'current_memory': current_memory
                }
            )
            
            self.logger.info(f"Strategy {strategy.value} executed in {execution_time:.4f}s")
            
            return result
            
        except Exception as e:
            tracemalloc.stop()
            self.logger.error(f"Strategy execution failed: {e}")
            raise
            
    def _execute_random_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute random strategy with vectorization"""
        batch_size = len(observations)
        action_dim = config.get('action_dim', 3)
        
        # Vectorized Dirichlet sampling
        alpha = np.ones(action_dim) * config.get('dirichlet_alpha', 1.0)
        actions = np.random.dirichlet(alpha, size=batch_size)
        
        return actions
        
    def _execute_biased_random_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute biased random strategy with vectorization"""
        batch_size = len(observations)
        action_dim = config.get('action_dim', 3)
        
        # Base random actions
        alpha = np.ones(action_dim) * config.get('dirichlet_alpha', 1.0)
        base_actions = np.random.dirichlet(alpha, size=batch_size)
        
        # Bias configuration
        bias_type = config.get('bias_type', 'bullish')
        bias_strength = config.get('bias_strength', 0.2)
        
        # Vectorized bias application
        if bias_type == 'bullish':
            bias_vector = np.array([0.1, 0.2, 0.7])
        elif bias_type == 'bearish':
            bias_vector = np.array([0.7, 0.2, 0.1])
        elif bias_type == 'neutral':
            bias_vector = np.array([0.1, 0.8, 0.1])
        else:
            bias_vector = np.array([1.0, 1.0, 1.0]) / 3.0
            
        # Broadcast bias to all samples
        bias_matrix = np.tile(bias_vector, (batch_size, 1))
        
        # Blend actions with bias
        biased_actions = (1 - bias_strength) * base_actions + bias_strength * bias_matrix
        
        # Normalize to ensure valid distributions
        biased_actions = biased_actions / biased_actions.sum(axis=1, keepdims=True)
        
        return biased_actions
        
    def _execute_contextual_random_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute contextual random strategy with vectorization"""
        batch_size = len(observations)
        action_dim = config.get('action_dim', 3)
        
        # Extract volatility context (assume in position 2)
        volatilities = np.exp(observations[:, 2]) if observations.shape[1] > 2 else np.ones(batch_size)
        
        # Vectorized alpha adjustment
        base_alpha = config.get('dirichlet_alpha', 1.0)
        volatility_sensitivity = config.get('volatility_sensitivity', 1.0)
        
        alphas = base_alpha / (1.0 + volatility_sensitivity * volatilities)
        alphas = np.clip(alphas, 0.1, None)  # Minimum alpha
        
        # Generate actions with varying randomness
        actions = np.zeros((batch_size, action_dim))
        for i in range(batch_size):
            alpha_vec = np.ones(action_dim) * alphas[i]
            actions[i] = np.random.dirichlet(alpha_vec)
            
        return actions
        
    def _execute_rule_based_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute rule-based strategy with vectorization"""
        batch_size = len(observations)
        action_dim = config.get('action_dim', 3)
        
        # Extract features (assuming standardized format)
        prices = observations[:, 0] if observations.shape[1] > 0 else np.zeros(batch_size)
        returns = observations[:, 1] if observations.shape[1] > 1 else np.zeros(batch_size)
        volatilities = observations[:, 2] if observations.shape[1] > 2 else np.ones(batch_size)
        
        # Vectorized rule logic
        trend_threshold = config.get('trend_threshold', 0.02)
        volatility_threshold = config.get('volatility_threshold', 0.05)
        
        # Bullish conditions
        bullish_mask = (returns > trend_threshold) & (volatilities < volatility_threshold)
        
        # Bearish conditions
        bearish_mask = (returns < -trend_threshold) & (volatilities < volatility_threshold)
        
        # Neutral conditions (default)
        neutral_mask = ~(bullish_mask | bearish_mask)
        
        # Initialize actions
        actions = np.zeros((batch_size, action_dim))
        
        # Apply rules vectorized
        actions[bullish_mask] = np.array([0.1, 0.2, 0.7])  # Bullish
        actions[bearish_mask] = np.array([0.7, 0.2, 0.1])  # Bearish
        actions[neutral_mask] = np.array([0.2, 0.6, 0.2])  # Neutral
        
        return actions
        
    def _execute_enhanced_rule_based_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute enhanced rule-based strategy with vectorization"""
        # Start with base rule-based
        base_actions = self._execute_rule_based_vectorized(observations, config)
        
        # Add regime detection enhancement
        if config.get('regime_detection', True):
            # Simple regime detection using rolling volatility
            window = min(20, len(observations))
            if len(observations) >= window:
                vol_rolling = np.std(observations[-window:, 1])  # Rolling volatility of returns
                
                if vol_rolling > 0.1:  # High volatility regime
                    # More conservative actions
                    base_actions = base_actions * 0.8
                    base_actions[:, 1] += 0.2  # Increase neutral weight
                    
                # Renormalize
                base_actions = base_actions / base_actions.sum(axis=1, keepdims=True)
        
        return base_actions
        
    def _execute_technical_indicators_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute technical indicators strategy with vectorization"""
        batch_size = len(observations)
        action_dim = config.get('action_dim', 3)
        
        # Extract price data
        prices = observations[:, 0] if observations.shape[1] > 0 else np.arange(batch_size)
        
        # Vectorized technical indicators
        sma_window = config.get('sma_window', 20)
        rsi_window = config.get('rsi_window', 14)
        
        # Simple Moving Average
        sma = self._calculate_sma_vectorized(prices, sma_window)
        
        # RSI
        rsi = self._calculate_rsi_vectorized(prices, rsi_window)
        
        # MACD
        macd, signal = self._calculate_macd_vectorized(prices, config.get('macd_params', {}))
        
        # Combine signals
        actions = np.zeros((batch_size, action_dim))
        
        # Price above SMA and RSI < 70 -> Bullish
        bullish_mask = (prices > sma) & (rsi < 70) & (macd > signal)
        
        # Price below SMA and RSI > 30 -> Bearish
        bearish_mask = (prices < sma) & (rsi > 30) & (macd < signal)
        
        # Default neutral
        neutral_mask = ~(bullish_mask | bearish_mask)
        
        actions[bullish_mask] = np.array([0.1, 0.2, 0.7])
        actions[bearish_mask] = np.array([0.7, 0.2, 0.1])
        actions[neutral_mask] = np.array([0.2, 0.6, 0.2])
        
        return actions
        
    def _execute_buy_and_hold_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute buy and hold strategy with vectorization"""
        batch_size = len(observations)
        action_dim = config.get('action_dim', 3)
        
        # Buy and hold = always long
        buy_and_hold_action = np.array([0.0, 0.0, 1.0])
        actions = np.tile(buy_and_hold_action, (batch_size, 1))
        
        return actions
        
    def _execute_momentum_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute momentum strategy with vectorization"""
        batch_size = len(observations)
        action_dim = config.get('action_dim', 3)
        
        # Extract returns
        returns = observations[:, 1] if observations.shape[1] > 1 else np.zeros(batch_size)
        
        # Momentum lookback
        lookback = config.get('momentum_lookback', 10)
        
        # Calculate momentum signal
        momentum_signal = np.zeros(batch_size)
        for i in range(lookback, batch_size):
            momentum_signal[i] = np.mean(returns[i-lookback:i])
            
        # Convert to actions
        actions = np.zeros((batch_size, action_dim))
        
        # Positive momentum -> Long
        long_mask = momentum_signal > 0
        actions[long_mask] = np.array([0.1, 0.2, 0.7])
        
        # Negative momentum -> Short
        short_mask = momentum_signal < 0
        actions[short_mask] = np.array([0.7, 0.2, 0.1])
        
        # Neutral for zero momentum
        neutral_mask = momentum_signal == 0
        actions[neutral_mask] = np.array([0.2, 0.6, 0.2])
        
        return actions
        
    def _execute_mean_reversion_vectorized(self, observations: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Execute mean reversion strategy with vectorization"""
        batch_size = len(observations)
        action_dim = config.get('action_dim', 3)
        
        # Extract prices
        prices = observations[:, 0] if observations.shape[1] > 0 else np.arange(batch_size)
        
        # Calculate mean reversion signal
        lookback = config.get('mean_reversion_lookback', 20)
        threshold = config.get('mean_reversion_threshold', 2.0)
        
        # Rolling mean and std
        rolling_mean = np.zeros(batch_size)
        rolling_std = np.zeros(batch_size)
        
        for i in range(lookback, batch_size):
            window = prices[i-lookback:i]
            rolling_mean[i] = np.mean(window)
            rolling_std[i] = np.std(window)
            
        # Z-score
        z_score = np.zeros(batch_size)
        non_zero_std = rolling_std != 0
        z_score[non_zero_std] = (prices[non_zero_std] - rolling_mean[non_zero_std]) / rolling_std[non_zero_std]
        
        # Mean reversion actions
        actions = np.zeros((batch_size, action_dim))
        
        # Price too high -> Short
        short_mask = z_score > threshold
        actions[short_mask] = np.array([0.7, 0.2, 0.1])
        
        # Price too low -> Long
        long_mask = z_score < -threshold
        actions[long_mask] = np.array([0.1, 0.2, 0.7])
        
        # Neutral
        neutral_mask = np.abs(z_score) <= threshold
        actions[neutral_mask] = np.array([0.2, 0.6, 0.2])
        
        return actions
        
    @cached(ttl=1800)
    def _calculate_sma_vectorized(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Simple Moving Average with vectorization"""
        sma = np.zeros_like(prices)
        
        for i in range(window, len(prices)):
            sma[i] = np.mean(prices[i-window:i])
            
        # Fill initial values with first valid SMA
        if len(prices) >= window:
            sma[:window] = sma[window]
            
        return sma
        
    @cached(ttl=1800)
    def _calculate_rsi_vectorized(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate RSI with vectorization"""
        if len(prices) < window + 1:
            return np.full_like(prices, 50.0)
            
        # Calculate price changes
        price_changes = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        # Calculate RSI
        rsi = np.zeros(len(prices))
        
        for i in range(window, len(gains)):
            avg_gain = np.mean(gains[i-window:i])
            avg_loss = np.mean(losses[i-window:i])
            
            if avg_loss == 0:
                rsi[i+1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i+1] = 100.0 - (100.0 / (1.0 + rs))
                
        # Fill initial values
        rsi[:window+1] = 50.0
        
        return rsi
        
    @cached(ttl=1800)
    def _calculate_macd_vectorized(self, prices: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD with vectorization"""
        fast_period = params.get('fast', 12)
        slow_period = params.get('slow', 26)
        signal_period = params.get('signal', 9)
        
        # Calculate EMAs
        ema_fast = self._calculate_ema_vectorized(prices, fast_period)
        ema_slow = self._calculate_ema_vectorized(prices, slow_period)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = self._calculate_ema_vectorized(macd_line, signal_period)
        
        return macd_line, signal_line
        
    def _calculate_ema_vectorized(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate Exponential Moving Average with vectorization"""
        alpha = 2.0 / (window + 1.0)
        ema = np.zeros_like(prices)
        
        if len(prices) > 0:
            ema[0] = prices[0]
            
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
                
        return ema
        
    def _calculate_performance_metrics(self, actions: np.ndarray, observations: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics for strategy results"""
        
        # Basic statistics
        action_stats = {
            'mean_actions': np.mean(actions, axis=0).tolist(),
            'std_actions': np.std(actions, axis=0).tolist(),
            'min_actions': np.min(actions, axis=0).tolist(),
            'max_actions': np.max(actions, axis=0).tolist()
        }
        
        # Entropy (measure of randomness)
        entropies = []
        for action in actions:
            safe_action = np.clip(action, 1e-10, 1.0)
            entropy = -np.sum(safe_action * np.log(safe_action))
            entropies.append(entropy)
            
        entropy_stats = {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies)
        }
        
        # Action distribution
        action_distribution = {
            'short_percentage': np.mean(actions[:, 0]),
            'neutral_percentage': np.mean(actions[:, 1]),
            'long_percentage': np.mean(actions[:, 2])
        }
        
        return {
            'action_stats': action_stats,
            'entropy_stats': entropy_stats,
            'action_distribution': action_distribution,
            'total_actions': len(actions),
            'action_shape': actions.shape
        }
        
    def execute_batch_strategies(self, 
                               strategies: List[BaselineStrategy],
                               observations: np.ndarray,
                               configs: Optional[Dict[BaselineStrategy, Dict[str, Any]]] = None) -> Dict[BaselineStrategy, BaselineResult]:
        """Execute multiple strategies in parallel"""
        configs = configs or {}
        
        # Use parallel processing
        def execute_single_strategy(strategy):
            config = configs.get(strategy, {})
            return strategy, self.execute_strategy(strategy, observations, config)
            
        results = {}
        
        if len(strategies) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.vectorized_params.parallel_workers) as executor:
                future_to_strategy = {
                    executor.submit(execute_single_strategy, strategy): strategy
                    for strategy in strategies
                }
                
                for future in as_completed(future_to_strategy):
                    strategy, result = future.result()
                    results[strategy] = result
        else:
            # Sequential execution for single strategy
            for strategy in strategies:
                config = configs.get(strategy, {})
                results[strategy] = self.execute_strategy(strategy, observations, config)
                
        return results
        
    def benchmark_strategies(self, 
                           strategies: List[BaselineStrategy],
                           observations: np.ndarray,
                           iterations: int = 10) -> Dict[str, Any]:
        """Benchmark multiple strategies"""
        benchmark_results = {}
        
        for strategy in strategies:
            times = []
            memory_usage = []
            
            for _ in range(iterations):
                start_time = time.time()
                process = psutil.Process()
                start_memory = process.memory_info().rss
                
                # Execute strategy
                result = self.execute_strategy(strategy, observations)
                
                end_time = time.time()
                end_memory = process.memory_info().rss
                
                times.append(end_time - start_time)
                memory_usage.append(end_memory - start_memory)
                
            benchmark_results[strategy.value] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'mean_memory': np.mean(memory_usage),
                'std_memory': np.std(memory_usage),
                'iterations': iterations
            }
            
        return benchmark_results
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'execution_stats': self._execution_stats,
            'memory_stats': self._memory_stats,
            'cache_stats': self.performance_optimizer.cache.stats(),
            'system_stats': self.performance_optimizer.get_system_stats()
        }
        
    def clear_caches(self) -> None:
        """Clear all caches"""
        self.performance_optimizer.clear_all_caches()
        self._execution_stats.clear()
        self._memory_stats.clear()
        
    @contextmanager
    def optimization_context(self, mode: OptimizationMode = OptimizationMode.BALANCED):
        """Context manager for optimization mode"""
        original_mode = self.performance_optimizer.mode
        
        try:
            self.performance_optimizer.optimize_for_mode(mode)
            yield self
        finally:
            self.performance_optimizer.optimize_for_mode(original_mode)


# Global baseline optimizer instance
_baseline_optimizer: Optional[BaselineOptimizer] = None


def get_baseline_optimizer(performance_optimizer: Optional[PerformanceOptimizer] = None) -> BaselineOptimizer:
    """Get global baseline optimizer instance"""
    global _baseline_optimizer
    if _baseline_optimizer is None:
        _baseline_optimizer = BaselineOptimizer(performance_optimizer)
    return _baseline_optimizer


def execute_baseline_strategy(strategy: BaselineStrategy, 
                            observations: np.ndarray,
                            config: Optional[Dict[str, Any]] = None) -> BaselineResult:
    """Convenience function to execute baseline strategy"""
    return get_baseline_optimizer().execute_strategy(strategy, observations, config)


def benchmark_baseline_strategies(strategies: List[BaselineStrategy],
                                observations: np.ndarray,
                                iterations: int = 10) -> Dict[str, Any]:
    """Convenience function to benchmark baseline strategies"""
    return get_baseline_optimizer().benchmark_strategies(strategies, observations, iterations)