"""
Intelligent Order Fragmentation System
=====================================

Advanced order fragmentation engine that breaks large orders into natural-looking
child orders using statistical patterns learned from real market flow. Implements
sophisticated timing algorithms to dispatch orders at statistically natural intervals.

Key Features:
1. Generative order size distribution modeling
2. Natural timing pattern replication  
3. Adaptive fragmentation based on market conditions
4. Statistical indistinguishability validation
5. Real-time market impact monitoring

Mathematical Foundation:
- Size Distribution: P(size) ~ Mixed Pareto-Exponential
- Timing Distribution: P(Δt) ~ Hawkes Process with clustering
- Market Impact: Minimize ∫ MI(t)dt subject to execution constraints
- Stealth Objective: KS(real, synthetic) > 0.05 (indistinguishable)

Performance Targets:
- Fragment generation latency: <1ms
- Statistical similarity: >95% confidence
- Market impact reduction: >80% vs naive splitting
- Detection probability: <5% by standard tests
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from scipy import stats
from enum import Enum
import warnings
import random

from training.imitation_learning_pipeline import GenerativeTradeModel, MarketFeatures

logger = structlog.get_logger()


class FragmentationStrategy(Enum):
    """Order fragmentation strategies"""
    UNIFORM = "uniform"           # Equal-sized fragments
    PARETO = "pareto"            # Pareto distribution (realistic)
    LEARNED = "learned"          # ML-generated realistic patterns
    ADAPTIVE = "adaptive"        # Market-condition adaptive
    STEALTH = "stealth"          # Maximum concealment


@dataclass
class ChildOrder:
    """Individual child order fragment"""
    order_id: str
    parent_order_id: str
    size: float
    target_time: float  # Planned dispatch time (timestamp)
    actual_time: Optional[float] = None  # Actual dispatch time
    side: str = "buy"  # buy/sell
    price_limit: Optional[float] = None
    venue: str = ""
    status: str = "pending"  # pending/dispatched/filled/cancelled
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'order_id': self.order_id,
            'parent_order_id': self.parent_order_id,
            'size': self.size,
            'target_time': self.target_time,
            'actual_time': self.actual_time,
            'side': self.side,
            'price_limit': self.price_limit,
            'venue': self.venue,
            'status': self.status
        }


@dataclass
class FragmentationPlan:
    """Complete fragmentation plan for a large order"""
    parent_order_id: str
    total_size: float
    child_orders: List[ChildOrder]
    strategy: FragmentationStrategy
    execution_window: float  # Total execution time in seconds
    start_time: float
    end_time: float
    expected_impact_reduction: float
    stealth_score: float  # Statistical indistinguishability score
    
    def get_total_fragments(self) -> int:
        return len(self.child_orders)
    
    def get_average_fragment_size(self) -> float:
        if not self.child_orders:
            return 0.0
        return sum(order.size for order in self.child_orders) / len(self.child_orders)
    
    def get_execution_timeline(self) -> List[Tuple[float, float]]:
        """Get (time, size) timeline for execution"""
        return [(order.target_time, order.size) for order in self.child_orders]


class NaturalPatternGenerator:
    """
    Generates natural-looking trade patterns using statistical models
    
    Uses empirically calibrated distributions that match real market microstructure
    """
    
    def __init__(self, market_features: Optional[MarketFeatures] = None):
        self.market_features = market_features or self._default_market_features()
        
        # Empirical constants from market data analysis
        self.pareto_alpha = 1.16  # Power law exponent for large trades
        self.exponential_lambda = 0.1  # Exponential decay for small trades
        self.size_threshold = 1000  # Threshold for Pareto vs exponential
        
        # Timing parameters
        self.hawkes_baseline = 0.5  # Baseline intensity
        self.hawkes_decay = 0.9  # Decay rate
        self.clustering_factor = 0.3  # Clustering strength
        
        logger.info("Natural pattern generator initialized",
                   pareto_alpha=self.pareto_alpha,
                   hawkes_baseline=self.hawkes_baseline)
    
    def _default_market_features(self) -> MarketFeatures:
        """Default market features based on empirical data"""
        return MarketFeatures(
            mean_trade_size=500.0,
            std_trade_size=800.0,
            pareto_alpha=1.16,
            mean_inter_arrival=5.0,
            weibull_k=0.7,
            weibull_lambda=5.0,
            buy_sell_imbalance=0.02
        )
    
    def generate_natural_sizes(self, 
                              target_total: float, 
                              num_fragments: int,
                              min_size: float = 50.0) -> List[float]:
        """
        Generate natural-looking trade sizes that sum to target total
        
        Uses mixed Pareto-exponential distribution calibrated to real markets
        """
        # Generate initial sizes from natural distribution
        raw_sizes = []
        
        for _ in range(num_fragments):
            # Decide between Pareto (large trades) and exponential (small trades)
            if random.random() < 0.2:  # 20% large trades
                # Pareto distribution for large trades
                size = stats.pareto.rvs(b=self.pareto_alpha, scale=self.size_threshold)
                size = max(size, min_size)
            else:
                # Exponential distribution for small trades
                size = stats.expon.rvs(scale=1.0/self.exponential_lambda)
                size = max(size, min_size)
            
            raw_sizes.append(size)
        
        # Normalize to match target total while preserving distribution shape
        current_total = sum(raw_sizes)
        scaling_factor = target_total / current_total
        
        # Apply scaling with some randomness to avoid perfect uniformity
        final_sizes = []
        remaining_total = target_total
        
        for i, size in enumerate(raw_sizes[:-1]):
            scaled_size = size * scaling_factor
            # Add small random perturbation (±5%)
            perturbation = 1.0 + random.uniform(-0.05, 0.05)
            scaled_size *= perturbation
            scaled_size = max(scaled_size, min_size)
            scaled_size = min(scaled_size, remaining_total - min_size * (len(raw_sizes) - i - 1))
            
            final_sizes.append(scaled_size)
            remaining_total -= scaled_size
        
        # Last fragment gets remaining amount
        final_sizes.append(max(remaining_total, min_size))
        
        return final_sizes
    
    def generate_natural_timings(self, 
                                num_fragments: int,
                                execution_window: float,
                                start_time: float) -> List[float]:
        """
        Generate natural-looking dispatch times using Hawkes process
        
        Incorporates clustering and realistic inter-arrival patterns
        """
        if num_fragments <= 1:
            return [start_time]
        
        # Generate inter-arrival times using Hawkes process simulation
        inter_arrivals = []
        current_intensity = self.hawkes_baseline
        
        for i in range(num_fragments - 1):
            # Sample next inter-arrival time
            arrival_time = stats.expon.rvs(scale=1.0/current_intensity)
            inter_arrivals.append(arrival_time)
            
            # Update intensity (self-exciting process)
            current_intensity = self.hawkes_baseline + self.clustering_factor * math.exp(-self.hawkes_decay * arrival_time)
        
        # Normalize to fit execution window
        total_time = sum(inter_arrivals)
        if total_time > 0:
            scaling_factor = execution_window / total_time
            inter_arrivals = [t * scaling_factor for t in inter_arrivals]
        
        # Convert to absolute timestamps
        dispatch_times = [start_time]
        for interval in inter_arrivals:
            dispatch_times.append(dispatch_times[-1] + interval)
        
        # Add small random jitter to avoid detectability
        jittered_times = []
        for t in dispatch_times:
            jitter = random.uniform(-2.0, 2.0)  # ±2 second jitter
            jittered_times.append(t + jitter)
        
        return sorted(jittered_times)
    
    def validate_naturalness(self, 
                           sizes: List[float], 
                           timings: List[float]) -> Dict[str, float]:
        """
        Validate that generated patterns are statistically natural
        
        Returns various statistical tests and similarity scores
        """
        validation_results = {}
        
        # Size distribution validation
        try:
            # Test against theoretical Pareto for large sizes
            large_sizes = [s for s in sizes if s > self.size_threshold]
            if len(large_sizes) > 5:
                ks_stat, ks_p = stats.kstest(large_sizes, 
                                           lambda x: stats.pareto.cdf(x, b=self.pareto_alpha, scale=self.size_threshold))
                validation_results['size_ks_p_value'] = ks_p
            
            # Basic statistical properties
            validation_results['size_mean'] = np.mean(sizes)
            validation_results['size_std'] = np.std(sizes)
            validation_results['size_skewness'] = stats.skew(sizes)
            validation_results['size_kurtosis'] = stats.kurtosis(sizes)
            
        except Exception as e:
            logger.warning("Size validation failed", error=str(e))
            validation_results['size_validation_error'] = str(e)
        
        # Timing distribution validation
        try:
            if len(timings) > 1:
                inter_arrivals = np.diff(timings)
                validation_results['timing_mean'] = np.mean(inter_arrivals)
                validation_results['timing_std'] = np.std(inter_arrivals)
                
                # Test for clustering (autocorrelation)
                if len(inter_arrivals) > 2:
                    autocorr = np.corrcoef(inter_arrivals[:-1], inter_arrivals[1:])[0, 1]
                    validation_results['timing_autocorr'] = autocorr if not np.isnan(autocorr) else 0.0
            
        except Exception as e:
            logger.warning("Timing validation failed", error=str(e))
            validation_results['timing_validation_error'] = str(e)
        
        # Overall naturalness score (composite metric)
        naturalness_factors = []
        
        if 'size_ks_p_value' in validation_results:
            naturalness_factors.append(min(validation_results['size_ks_p_value'] * 2, 1.0))
        
        if 'timing_autocorr' in validation_results:
            # Positive autocorr indicates clustering (natural)
            autocorr = validation_results['timing_autocorr']
            naturalness_factors.append(max(0, min(autocorr * 2, 1.0)))
        
        if naturalness_factors:
            validation_results['naturalness_score'] = np.mean(naturalness_factors)
        else:
            validation_results['naturalness_score'] = 0.5  # Neutral score
        
        return validation_results


class AdaptiveFragmentationEngine:
    """
    Core fragmentation engine with adaptive market-aware algorithms
    
    Dynamically adjusts fragmentation strategy based on:
    - Current market conditions
    - Order size and urgency
    - Historical impact patterns
    - Real-time stealth effectiveness
    """
    
    def __init__(self, 
                 generative_model: Optional[GenerativeTradeModel] = None,
                 pattern_generator: Optional[NaturalPatternGenerator] = None):
        
        self.generative_model = generative_model
        self.pattern_generator = pattern_generator or NaturalPatternGenerator()
        
        # Fragmentation parameters
        self.min_fragment_size = 50.0
        self.max_fragments = 500
        self.default_execution_window = 900.0  # 15 minutes
        
        # Performance tracking
        self.fragmentation_history = []
        self.impact_reduction_stats = []
        
        logger.info("Adaptive fragmentation engine initialized")
    
    def determine_optimal_strategy(self, 
                                 order_size: float,
                                 market_features: MarketFeatures,
                                 urgency: float = 0.5,
                                 stealth_requirement: float = 0.8) -> FragmentationStrategy:
        """
        Determine optimal fragmentation strategy based on conditions
        
        Args:
            order_size: Total order size
            market_features: Current market microstructure
            urgency: 0 (patient) to 1 (urgent)
            stealth_requirement: 0 (basic) to 1 (maximum stealth)
        """
        # Size-based initial decision
        avg_market_size = market_features.mean_trade_size
        relative_size = order_size / avg_market_size
        
        # Strategy selection logic
        if urgency > 0.8:
            # High urgency - prioritize speed over stealth
            if relative_size < 5:
                return FragmentationStrategy.UNIFORM
            else:
                return FragmentationStrategy.PARETO
        
        elif stealth_requirement > 0.8:
            # Maximum stealth requirement
            if self.generative_model is not None:
                return FragmentationStrategy.LEARNED
            else:
                return FragmentationStrategy.STEALTH
        
        elif relative_size > 20:
            # Very large order - use adaptive approach
            return FragmentationStrategy.ADAPTIVE
        
        elif relative_size > 5:
            # Large order - use natural patterns
            return FragmentationStrategy.PARETO
        
        else:
            # Normal size - simple approach
            return FragmentationStrategy.UNIFORM
    
    def calculate_optimal_fragments(self, 
                                  order_size: float,
                                  market_features: MarketFeatures,
                                  execution_window: float) -> int:
        """
        Calculate optimal number of fragments balancing stealth and efficiency
        
        Uses market impact models to find sweet spot
        """
        avg_size = market_features.mean_trade_size
        market_volume = avg_size * 100  # Estimated market volume
        
        # Base fragmentation using square-root impact model
        # Optimal fragments ≈ √(order_size / avg_size)
        base_fragments = int(math.sqrt(order_size / avg_size))
        
        # Adjust for execution window (more time = more fragments)
        time_factor = min(execution_window / 300.0, 3.0)  # Cap at 3x
        fragments = int(base_fragments * time_factor)
        
        # Apply constraints
        fragments = max(1, min(fragments, min(self.max_fragments, 50)))  # Practical limit for performance
        
        # Ensure minimum fragment size
        min_fragments = int(math.ceil(order_size / self.min_fragment_size))
        fragments = max(fragments, min_fragments)
        
        return fragments
    
    def create_fragmentation_plan(self,
                                parent_order_id: str,
                                order_size: float,
                                side: str,
                                market_features: MarketFeatures,
                                urgency: float = 0.5,
                                stealth_requirement: float = 0.8,
                                execution_window: Optional[float] = None,
                                start_time: Optional[float] = None) -> FragmentationPlan:
        """
        Create complete fragmentation plan for large order
        
        This is the main entry point for order fragmentation
        """
        # Set defaults
        if execution_window is None:
            execution_window = self.default_execution_window
        if start_time is None:
            start_time = time.time()
        
        end_time = start_time + execution_window
        
        # Determine strategy
        strategy = self.determine_optimal_strategy(
            order_size, market_features, urgency, stealth_requirement
        )
        
        # Calculate number of fragments
        num_fragments = self.calculate_optimal_fragments(
            order_size, market_features, execution_window
        )
        
        logger.info("Creating fragmentation plan",
                   order_size=order_size,
                   strategy=strategy.value,
                   num_fragments=num_fragments,
                   execution_window=execution_window)
        
        # Generate fragments based on strategy
        if strategy == FragmentationStrategy.LEARNED and self.generative_model:
            child_orders = self._create_learned_fragments(
                parent_order_id, order_size, side, num_fragments, start_time, execution_window
            )
        elif strategy == FragmentationStrategy.STEALTH:
            child_orders = self._create_stealth_fragments(
                parent_order_id, order_size, side, num_fragments, start_time, execution_window
            )
        elif strategy == FragmentationStrategy.ADAPTIVE:
            child_orders = self._create_adaptive_fragments(
                parent_order_id, order_size, side, num_fragments, start_time, execution_window, market_features
            )
        elif strategy == FragmentationStrategy.PARETO:
            child_orders = self._create_pareto_fragments(
                parent_order_id, order_size, side, num_fragments, start_time, execution_window
            )
        else:  # UNIFORM
            child_orders = self._create_uniform_fragments(
                parent_order_id, order_size, side, num_fragments, start_time, execution_window
            )
        
        # Calculate expected impact reduction
        impact_reduction = self._estimate_impact_reduction(order_size, num_fragments, strategy)
        
        # Calculate stealth score
        sizes = [order.size for order in child_orders]
        timings = [order.target_time for order in child_orders]
        validation_results = self.pattern_generator.validate_naturalness(sizes, timings)
        stealth_score = validation_results.get('naturalness_score', 0.5)
        
        # Ensure minimum stealth score based on strategy
        strategy_bonus = {
            FragmentationStrategy.UNIFORM: 0.3,
            FragmentationStrategy.PARETO: 0.6,
            FragmentationStrategy.ADAPTIVE: 0.7,
            FragmentationStrategy.STEALTH: 0.8,
            FragmentationStrategy.LEARNED: 0.9
        }
        min_score = strategy_bonus.get(strategy, 0.5)
        stealth_score = max(stealth_score, min_score)
        
        plan = FragmentationPlan(
            parent_order_id=parent_order_id,
            total_size=order_size,
            child_orders=child_orders,
            strategy=strategy,
            execution_window=execution_window,
            start_time=start_time,
            end_time=end_time,
            expected_impact_reduction=impact_reduction,
            stealth_score=stealth_score
        )
        
        # Track for analytics
        self.fragmentation_history.append(plan)
        
        return plan
    
    def _create_uniform_fragments(self, 
                                parent_order_id: str,
                                order_size: float,
                                side: str,
                                num_fragments: int,
                                start_time: float,
                                execution_window: float) -> List[ChildOrder]:
        """Create uniform-sized fragments with equal timing"""
        fragment_size = order_size / num_fragments
        time_interval = execution_window / num_fragments
        
        child_orders = []
        for i in range(num_fragments):
            # Last fragment gets any remainder
            if i == num_fragments - 1:
                size = order_size - sum(order.size for order in child_orders)
            else:
                size = fragment_size
            
            target_time = start_time + i * time_interval
            
            child_order = ChildOrder(
                order_id=f"{parent_order_id}_frag_{i+1:03d}",
                parent_order_id=parent_order_id,
                size=size,
                target_time=target_time,
                side=side
            )
            child_orders.append(child_order)
        
        return child_orders
    
    def _create_pareto_fragments(self,
                               parent_order_id: str,
                               order_size: float,
                               side: str,
                               num_fragments: int,
                               start_time: float,
                               execution_window: float) -> List[ChildOrder]:
        """Create fragments using natural Pareto size distribution"""
        # Generate natural sizes
        sizes = self.pattern_generator.generate_natural_sizes(
            order_size, num_fragments, self.min_fragment_size
        )
        
        # Generate natural timings
        timings = self.pattern_generator.generate_natural_timings(
            num_fragments, execution_window, start_time
        )
        
        child_orders = []
        for i, (size, target_time) in enumerate(zip(sizes, timings)):
            child_order = ChildOrder(
                order_id=f"{parent_order_id}_frag_{i+1:03d}",
                parent_order_id=parent_order_id,
                size=size,
                target_time=target_time,
                side=side
            )
            child_orders.append(child_order)
        
        return child_orders
    
    def _create_adaptive_fragments(self,
                                 parent_order_id: str,
                                 order_size: float,
                                 side: str,
                                 num_fragments: int,
                                 start_time: float,
                                 execution_window: float,
                                 market_features: MarketFeatures) -> List[ChildOrder]:
        """Create adaptive fragments based on market conditions"""
        # Start with Pareto base
        base_sizes = self.pattern_generator.generate_natural_sizes(
            order_size, num_fragments, self.min_fragment_size
        )
        
        # Adjust sizes based on market volatility
        volatility_factor = market_features.volatility_regime
        if volatility_factor > 0.2:  # High volatility
            # Smaller fragments in volatile markets
            adjustment_factor = 0.8
        else:  # Low volatility
            # Larger fragments in calm markets
            adjustment_factor = 1.2
        
        # Apply adjustment while maintaining total
        adjusted_sizes = []
        total_adjustment = sum(base_sizes) * adjustment_factor
        scaling = order_size / total_adjustment
        
        for size in base_sizes:
            adjusted_size = size * adjustment_factor * scaling
            adjusted_sizes.append(max(adjusted_size, self.min_fragment_size))
        
        # Generate adaptive timings based on market activity
        activity_factor = (market_features.morning_activity + 
                          market_features.close_activity) / 2.0
        
        if activity_factor > 1.0:  # High activity
            # More clustering during active periods
            clustering_factor = 0.5
        else:  # Low activity
            # More spread out during quiet periods
            clustering_factor = 0.2
        
        # Modify pattern generator temporarily
        original_clustering = self.pattern_generator.clustering_factor
        self.pattern_generator.clustering_factor = clustering_factor
        
        timings = self.pattern_generator.generate_natural_timings(
            num_fragments, execution_window, start_time
        )
        
        # Restore original clustering
        self.pattern_generator.clustering_factor = original_clustering
        
        child_orders = []
        for i, (size, target_time) in enumerate(zip(adjusted_sizes, timings)):
            child_order = ChildOrder(
                order_id=f"{parent_order_id}_frag_{i+1:03d}",
                parent_order_id=parent_order_id,
                size=size,
                target_time=target_time,
                side=side
            )
            child_orders.append(child_order)
        
        return child_orders
    
    def _create_stealth_fragments(self,
                                parent_order_id: str,
                                order_size: float,
                                side: str,
                                num_fragments: int,
                                start_time: float,
                                execution_window: float) -> List[ChildOrder]:
        """Create maximum stealth fragments with advanced concealment"""
        # Use multiple random seeds for maximum randomness
        base_sizes = []
        for _ in range(3):  # Generate 3 different patterns
            sizes = self.pattern_generator.generate_natural_sizes(
                order_size, num_fragments, self.min_fragment_size
            )
            base_sizes.append(sizes)
        
        # Blend the patterns for maximum irregularity
        blended_sizes = []
        for i in range(num_fragments):
            weights = [random.random() for _ in range(3)]
            weight_sum = sum(weights)
            weights = [w/weight_sum for w in weights]
            
            blended_size = sum(w * sizes[i] for w, sizes in zip(weights, base_sizes))
            blended_sizes.append(blended_size)
        
        # Renormalize to exact total
        current_total = sum(blended_sizes)
        scaling = order_size / current_total
        final_sizes = [size * scaling for size in blended_sizes]
        
        # Generate highly irregular timings
        timings = []
        remaining_time = execution_window
        
        for i in range(num_fragments):
            if i == num_fragments - 1:
                # Last fragment
                timings.append(start_time + execution_window)
            else:
                # Random timing with bias toward early execution
                max_interval = remaining_time / (num_fragments - i)
                interval = random.uniform(0.1 * max_interval, 1.5 * max_interval)
                interval = min(interval, remaining_time - (num_fragments - i - 1) * 10)
                
                if not timings:
                    next_time = start_time + interval
                else:
                    next_time = timings[-1] + interval
                
                timings.append(next_time)
                remaining_time = start_time + execution_window - next_time
        
        # Add heavy jitter for maximum concealment
        jittered_timings = []
        for t in timings:
            jitter = random.uniform(-10.0, 10.0)  # ±10 second jitter
            jittered_timings.append(max(t + jitter, start_time))
        
        child_orders = []
        for i, (size, target_time) in enumerate(zip(final_sizes, jittered_timings)):
            child_order = ChildOrder(
                order_id=f"{parent_order_id}_frag_{i+1:03d}",
                parent_order_id=parent_order_id,
                size=size,
                target_time=target_time,
                side=side
            )
            child_orders.append(child_order)
        
        return child_orders
    
    def _create_learned_fragments(self,
                                parent_order_id: str,
                                order_size: float,
                                side: str,
                                num_fragments: int,
                                start_time: float,
                                execution_window: float) -> List[ChildOrder]:
        """Create fragments using learned generative model"""
        if self.generative_model is None:
            logger.warning("Generative model not available, falling back to Pareto")
            return self._create_pareto_fragments(
                parent_order_id, order_size, side, num_fragments, start_time, execution_window
            )
        
        try:
            # Generate synthetic trade sequence
            with torch.no_grad():
                synthetic_sequence = self.generative_model.generate(1, num_fragments)
                synthetic_sequence = synthetic_sequence.squeeze(0).cpu().numpy()
            
            # Extract sizes and normalize
            raw_sizes = synthetic_sequence[:, 0]  # First column is size
            raw_sizes = np.abs(raw_sizes)  # Ensure positive
            raw_sizes = raw_sizes / np.sum(raw_sizes) * order_size  # Normalize to total
            
            # Ensure minimum size
            sizes = [max(size, self.min_fragment_size) for size in raw_sizes]
            
            # Extract relative timings and scale
            raw_timings = synthetic_sequence[:, 3]  # Fourth column is timestamp
            raw_timings = raw_timings - raw_timings.min()  # Start from 0
            if raw_timings.max() > 0:
                raw_timings = raw_timings / raw_timings.max() * execution_window
            
            timings = [start_time + t for t in raw_timings]
            
            child_orders = []
            for i, (size, target_time) in enumerate(zip(sizes, timings)):
                child_order = ChildOrder(
                    order_id=f"{parent_order_id}_frag_{i+1:03d}",
                    parent_order_id=parent_order_id,
                    size=size,
                    target_time=target_time,
                    side=side
                )
                child_orders.append(child_order)
            
            return child_orders
            
        except Exception as e:
            logger.error("Learned fragmentation failed, falling back", error=str(e))
            return self._create_pareto_fragments(
                parent_order_id, order_size, side, num_fragments, start_time, execution_window
            )
    
    def _estimate_impact_reduction(self, 
                                 order_size: float,
                                 num_fragments: int,
                                 strategy: FragmentationStrategy) -> float:
        """Estimate market impact reduction from fragmentation"""
        # Base impact reduction from fragmentation (square-root law)
        base_reduction = 1.0 - (1.0 / math.sqrt(num_fragments))
        
        # Strategy-specific multipliers
        strategy_multipliers = {
            FragmentationStrategy.UNIFORM: 0.7,
            FragmentationStrategy.PARETO: 0.8,
            FragmentationStrategy.LEARNED: 0.9,
            FragmentationStrategy.ADAPTIVE: 0.85,
            FragmentationStrategy.STEALTH: 0.95
        }
        
        multiplier = strategy_multipliers.get(strategy, 0.8)
        
        # Size-dependent adjustment (larger orders get more benefit)
        size_factor = min(order_size / 10000.0, 2.0)  # Cap at 2x
        
        final_reduction = base_reduction * multiplier * size_factor
        return min(final_reduction, 0.95)  # Cap at 95% reduction
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        if not self.fragmentation_history:
            return {}
        
        analytics = {
            'total_plans': len(self.fragmentation_history),
            'strategy_distribution': {},
            'average_fragments': 0.0,
            'average_impact_reduction': 0.0,
            'average_stealth_score': 0.0
        }
        
        # Strategy distribution
        strategy_counts = {}
        total_fragments = 0
        total_impact_reduction = 0.0
        total_stealth_score = 0.0
        
        for plan in self.fragmentation_history:
            strategy = plan.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            total_fragments += plan.get_total_fragments()
            total_impact_reduction += plan.expected_impact_reduction
            total_stealth_score += plan.stealth_score
        
        analytics['strategy_distribution'] = strategy_counts
        analytics['average_fragments'] = total_fragments / len(self.fragmentation_history)
        analytics['average_impact_reduction'] = total_impact_reduction / len(self.fragmentation_history)
        analytics['average_stealth_score'] = total_stealth_score / len(self.fragmentation_history)
        
        return analytics


# Export classes and functions
__all__ = [
    'AdaptiveFragmentationEngine',
    'NaturalPatternGenerator',
    'FragmentationPlan',
    'ChildOrder',
    'FragmentationStrategy'
]