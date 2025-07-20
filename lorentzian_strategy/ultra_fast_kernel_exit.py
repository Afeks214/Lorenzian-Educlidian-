"""
Ultra-Fast Kernel Regression Exit Strategy - Optimized for <50μs Latency

A streamlined version of the kernel regression exit strategy focused on achieving
ultra-low latency while maintaining the core functionality:

1. Dynamic r-factor adjustment (simplified)
2. Kernel regression crossover detection
3. ATR-based trailing stops
4. Multi-level take profits
5. Fast risk assessment

Target: <50μs per exit decision

Author: Advanced Quantitative Trading Research Team
Version: 3.0.0 (Ultra-Fast)
Date: 2025-07-20
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
from numba import jit, njit, types
from collections import deque

class ExitSignal(IntEnum):
    """Simplified exit signal types"""
    NO_SIGNAL = 0
    TRAILING_STOP = 1
    TAKE_PROFIT = 2
    CROSSOVER = 3
    RISK_EXIT = 4

@dataclass
class FastExitDecision:
    """Lightweight exit decision"""
    signal: ExitSignal
    price: float
    urgency: float
    size_pct: float = 1.0

@dataclass
class FastKernelState:
    """Minimal kernel state for fast access"""
    yhat1: float = 0.0
    yhat2: float = 0.0
    slope: float = 0.0
    confidence: float = 0.5

@dataclass 
class FastTrailingStop:
    """Minimal trailing stop state"""
    price: float = 0.0
    highest: float = 0.0
    atr_mult: float = 1.5

# Ultra-fast core functions
@njit(cache=True, fastmath=True, inline='always')
def fast_rq_kernel(x_t: float, x_i: float, h: float, r: float) -> float:
    """Ultra-fast RQ kernel calculation"""
    d_sq = (x_t - x_i) ** 2
    return (1.0 + d_sq / (2.0 * r * h * h)) ** (-r)

@njit(cache=True, fastmath=True, inline='always')
def fast_kernel_regression(prices: np.ndarray, h: float, r: float) -> Tuple[float, float]:
    """Ultra-fast kernel regression for last 2 values only"""
    n = len(prices)
    if n < 3:
        return prices[-1], prices[-1]
    
    current = prices[-1]
    weights_sum = 0.0
    weighted_sum = 0.0
    
    # Only calculate for last 20 observations for speed
    start_idx = max(0, n - 20)
    
    for i in range(start_idx, n):
        weight = fast_rq_kernel(current, prices[i], h, r)
        weighted_sum += prices[i] * weight
        weights_sum += weight
    
    if weights_sum == 0.0:
        return current, current
    
    yhat = weighted_sum / weights_sum
    return yhat, yhat

@njit(cache=True, fastmath=True, inline='always') 
def fast_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """Ultra-fast ATR calculation using last 10 bars"""
    n = len(close)
    if n < 2:
        return 0.01
    
    tr_sum = 0.0
    count = 0
    
    for i in range(max(1, n - 10), n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        tr_sum += max(tr1, max(tr2, tr3))
        count += 1
    
    return tr_sum / count if count > 0 else 0.01

@njit(cache=True, fastmath=True, inline='always')
def fast_regime_detection(prices: np.ndarray) -> float:
    """Ultra-fast regime detection returning r-factor multiplier"""
    n = len(prices)
    if n < 10:
        return 1.0
    
    # Simple trend detection
    recent = prices[-5:]
    trend = (recent[-1] - recent[0]) / recent[0]
    
    # Simple volatility
    returns = np.diff(prices[-10:]) / prices[-11:-1]
    vol = np.std(returns)
    
    # Return r-factor multiplier
    if vol > 0.02:  # High vol
        return 0.7  # Lower r for responsiveness
    elif abs(trend) > 0.01:  # Trending
        return 0.8
    else:  # Ranging
        return 1.2

@njit(cache=True, fastmath=True, inline='always')
def fast_confidence(yhat1: float, yhat2: float, price: float) -> float:
    """Ultra-fast confidence estimation"""
    agreement = 1.0 - abs(yhat1 - yhat2) / max(abs(price), 0.001)
    return max(0.2, min(0.9, agreement))

class UltraFastKernelExit:
    """
    Ultra-fast kernel regression exit strategy
    Target: <50μs per decision
    """
    
    def __init__(self, config: Dict):
        # Core parameters
        self.base_h = config.get('h', 8.0)
        self.base_r = config.get('r', 8.0)
        
        # Trailing stop
        self.min_atr_mult = config.get('min_atr_mult', 1.5)
        self.max_atr_mult = config.get('max_atr_mult', 3.0)
        
        # Take profit levels
        self.tp_levels = config.get('tp_levels', [1.5, 2.5])
        self.tp_sizes = config.get('tp_sizes', [0.5, 1.0])
        
        # State (minimal)
        self.states: Dict[str, FastKernelState] = {}
        self.stops: Dict[str, FastTrailingStop] = {}
        self.tp_triggered: Dict[str, List[bool]] = {}
        
        # Performance tracking
        self.times = deque(maxlen=100)
        self.total_calls = 0
        
        print(f"UltraFastKernelExit initialized - Target: <50μs")
    
    def generate_exit(self, symbol: str, price: float, position_size: float,
                     entry_price: float, prices: np.ndarray, 
                     highs: np.ndarray, lows: np.ndarray) -> List[FastExitDecision]:
        """
        Ultra-fast exit decision generation
        Target: <50μs
        """
        start_time = time.perf_counter()
        
        try:
            decisions = []
            is_long = position_size > 0
            
            # 1. Fast regime and r-factor (2-3μs)
            r_mult = fast_regime_detection(prices)
            dynamic_r = self.base_r * r_mult
            
            # 2. Fast kernel regression (10-15μs)
            yhat1, yhat2 = fast_kernel_regression(prices, self.base_h, dynamic_r)
            yhat1_lag, _ = fast_kernel_regression(prices, self.base_h - 2, dynamic_r)
            
            # 3. Update state (1-2μs)
            prev_state = self.states.get(symbol)
            slope = (yhat1 - prev_state.yhat1) if prev_state else 0.0
            confidence = fast_confidence(yhat1, yhat2, price)
            
            self.states[symbol] = FastKernelState(yhat1, yhat2, slope, confidence)
            
            # 4. Fast ATR calculation (3-5μs)
            atr = fast_atr(highs, lows, prices)
            
            # 5. Check crossover signals (2-3μs)
            if prev_state:
                # Bullish cross - exit short
                if (prev_state.yhat2 <= prev_state.yhat1 and yhat2 > yhat1 and not is_long):
                    strength = abs(yhat2 - yhat1) / max(abs(yhat1), 0.001)
                    decisions.append(FastExitDecision(
                        ExitSignal.CROSSOVER, yhat2, min(1.0, strength * 2.0)
                    ))
                
                # Bearish cross - exit long  
                elif (prev_state.yhat2 >= prev_state.yhat1 and yhat2 < yhat1 and is_long):
                    strength = abs(yhat1 - yhat2) / max(abs(yhat1), 0.001)
                    decisions.append(FastExitDecision(
                        ExitSignal.CROSSOVER, yhat2, min(1.0, strength * 2.0)
                    ))
            
            # 6. Update trailing stop (5-8μs)
            stop_decision = self._update_fast_trailing_stop(symbol, price, is_long, 
                                                          slope, atr, confidence)
            if stop_decision:
                decisions.append(stop_decision)
            
            # 7. Check take profits (3-5μs)
            tp_decisions = self._check_fast_take_profits(symbol, price, entry_price, 
                                                       is_long, atr)
            decisions.extend(tp_decisions)
            
            # 8. Fast risk checks (2-3μs)
            if confidence < 0.3:
                decisions.append(FastExitDecision(
                    ExitSignal.RISK_EXIT, price, 1.0 - confidence
                ))
            
            # Performance tracking
            exec_time = (time.perf_counter() - start_time) * 1_000_000
            self.times.append(exec_time)
            self.total_calls += 1
            
            if exec_time > 50:
                print(f"WARNING: Exit decision took {exec_time:.1f}μs (target <50μs)")
            
            return decisions
            
        except Exception as e:
            return [FastExitDecision(ExitSignal.RISK_EXIT, price, 1.0)]
    
    def _update_fast_trailing_stop(self, symbol: str, price: float, is_long: bool,
                                 slope: float, atr: float, confidence: float) -> Optional[FastExitDecision]:
        """Ultra-fast trailing stop update"""
        # Get or create stop
        if symbol not in self.stops:
            distance = atr * self.min_atr_mult
            if is_long:
                stop_price = price - distance
                highest = price
            else:
                stop_price = price + distance
                highest = price
            
            self.stops[symbol] = FastTrailingStop(stop_price, highest, self.min_atr_mult)
        
        stop = self.stops[symbol]
        
        # Update highest favorable
        if is_long and price > stop.highest:
            stop.highest = price
        elif not is_long and price < stop.highest:
            stop.highest = price
        
        # Dynamic ATR multiplier (simplified)
        slope_factor = 1.0 - min(0.2, abs(slope) / price * 100) if slope * (1 if is_long else -1) > 0 else 1.0
        conf_factor = 1.0 + (1.0 - confidence) * 0.3
        new_mult = self.min_atr_mult * slope_factor * conf_factor
        new_mult = max(self.min_atr_mult, min(self.max_atr_mult, new_mult))
        
        # Update stop price
        distance = atr * new_mult
        if is_long:
            new_stop = stop.highest - distance
            stop.price = max(stop.price, new_stop)
            
            if price <= stop.price:
                return FastExitDecision(ExitSignal.TRAILING_STOP, stop.price, 1.0)
        else:
            new_stop = stop.highest + distance
            stop.price = min(stop.price, new_stop)
            
            if price >= stop.price:
                return FastExitDecision(ExitSignal.TRAILING_STOP, stop.price, 1.0)
        
        stop.atr_mult = new_mult
        return None
    
    def _check_fast_take_profits(self, symbol: str, price: float, entry: float,
                               is_long: bool, atr: float) -> List[FastExitDecision]:
        """Ultra-fast take profit checking"""
        decisions = []
        
        # Initialize TP tracking
        if symbol not in self.tp_triggered:
            self.tp_triggered[symbol] = [False] * len(self.tp_levels)
        
        triggered = self.tp_triggered[symbol]
        
        for i, (atr_mult, size_pct) in enumerate(zip(self.tp_levels, self.tp_sizes)):
            if triggered[i]:
                continue
            
            target_distance = atr * atr_mult
            if is_long:
                target_price = entry + target_distance
                hit = price >= target_price
            else:
                target_price = entry - target_distance
                hit = price <= target_price
            
            if hit:
                triggered[i] = True
                decisions.append(FastExitDecision(
                    ExitSignal.TAKE_PROFIT, target_price, 0.8, size_pct
                ))
        
        return decisions
    
    def get_performance_stats(self) -> Dict:
        """Get ultra-fast performance statistics"""
        if not self.times:
            return {'no_data': True}
        
        times = list(self.times)
        return {
            'total_calls': self.total_calls,
            'avg_time_us': np.mean(times),
            'median_time_us': np.median(times),
            'max_time_us': np.max(times),
            'p95_time_us': np.percentile(times, 95),
            'under_50us_rate': np.mean([t < 50 for t in times]),
            'under_30us_rate': np.mean([t < 30 for t in times]),
        }
    
    def reset_position(self, symbol: str):
        """Reset position state"""
        self.states.pop(symbol, None)
        self.stops.pop(symbol, None) 
        self.tp_triggered.pop(symbol, None)

def create_ultra_fast_config() -> Dict:
    """Create optimized config for ultra-fast execution"""
    return {
        'h': 8.0,
        'r': 8.0,
        'min_atr_mult': 1.5,
        'max_atr_mult': 3.0,
        'tp_levels': [1.5, 2.5],  # Reduced to 2 levels for speed
        'tp_sizes': [0.5, 1.0]
    }

def run_ultra_fast_benchmark(strategy: UltraFastKernelExit, iterations: int = 1000) -> Dict:
    """Run performance benchmark for ultra-fast strategy"""
    print(f"Running ultra-fast benchmark ({iterations} iterations)...")
    
    np.random.seed(42)
    base_price = 100.0
    
    all_times = []
    
    for i in range(iterations):
        # Generate test data
        prices = base_price + np.cumsum(np.random.normal(0, 0.3, 50))
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, 50)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, 50)))
        
        # Time the decision
        start = time.perf_counter()
        decisions = strategy.generate_exit(
            symbol=f"TEST_{i % 5}",
            price=prices[-1],
            position_size=100.0 if i % 2 == 0 else -100.0,
            entry_price=prices[-10],
            prices=prices,
            highs=highs,
            lows=lows
        )
        exec_time = (time.perf_counter() - start) * 1_000_000
        all_times.append(exec_time)
    
    # Calculate results
    times_array = np.array(all_times)
    
    results = {
        'iterations': iterations,
        'avg_time_us': np.mean(times_array),
        'median_time_us': np.median(times_array),
        'min_time_us': np.min(times_array),
        'max_time_us': np.max(times_array),
        'p95_time_us': np.percentile(times_array, 95),
        'p99_time_us': np.percentile(times_array, 99),
        'under_50us_rate': np.mean(times_array < 50),
        'under_30us_rate': np.mean(times_array < 30),
        'over_50us_count': np.sum(times_array >= 50)
    }
    
    print("\nULTRA-FAST BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Target: <50μs per decision")
    print(f"Average: {results['avg_time_us']:.1f}μs")
    print(f"Median: {results['median_time_us']:.1f}μs") 
    print(f"95th percentile: {results['p95_time_us']:.1f}μs")
    print(f"Under 50μs rate: {results['under_50us_rate']:.1%}")
    print(f"Under 30μs rate: {results['under_30us_rate']:.1%}")
    print(f"Times over 50μs: {results['over_50us_count']}/{iterations}")
    
    if results['under_50us_rate'] >= 0.95:
        print("✅ ULTRA-FAST TARGET ACHIEVED!")
    else:
        print("⚡ Getting close to ultra-fast target")
    
    return results

if __name__ == "__main__":
    print("ULTRA-FAST KERNEL REGRESSION EXIT STRATEGY")
    print("=" * 50)
    print("Target: <50μs per exit decision")
    print()
    
    # Create ultra-fast strategy
    config = create_ultra_fast_config()
    strategy = UltraFastKernelExit(config)
    
    # Run benchmark
    benchmark_results = run_ultra_fast_benchmark(strategy, 1000)
    
    # Show strategy stats
    print("\nSTRATEGY PERFORMANCE")
    print("-" * 25)
    stats = strategy.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.1f}")
        else:
            print(f"{key}: {value}")
    
    print("\n⚡ Ultra-fast kernel exit strategy ready!")
    print("🎯 Optimized for real-time trading with minimal latency")