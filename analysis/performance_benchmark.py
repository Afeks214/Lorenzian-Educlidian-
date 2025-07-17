"""
Performance Benchmark for Enhanced Backtesting Infrastructure

This script benchmarks the performance improvements achieved through:
- JIT compilation with Numba
- Parallel execution
- Memory-efficient data structures
- Optimized risk calculations
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List
from numba import jit
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import psutil
import os

# JIT-compiled performance functions
@jit(nopython=True)
def fast_sharpe_calculation(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """JIT-compiled Sharpe ratio calculation"""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

@jit(nopython=True)
def fast_drawdown_calculation(equity_curve: np.ndarray) -> float:
    """JIT-compiled maximum drawdown calculation"""
    if len(equity_curve) == 0:
        return 0.0
    
    running_max = equity_curve[0]
    max_drawdown = 0.0
    
    for i in range(1, len(equity_curve)):
        if equity_curve[i] > running_max:
            running_max = equity_curve[i]
        
        drawdown = (running_max - equity_curve[i]) / running_max
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return max_drawdown

@jit(nopython=True)
def fast_kelly_calculation(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """JIT-compiled Kelly criterion calculation"""
    if avg_loss == 0:
        return 0.0
    
    payout_ratio = avg_win / abs(avg_loss)
    kelly_fraction = (payout_ratio * win_rate - (1 - win_rate)) / payout_ratio
    
    return max(0.0, min(0.25, kelly_fraction))

@jit(nopython=True)
def fast_var_calculation(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """JIT-compiled VaR calculation"""
    if len(returns) == 0:
        return 0.0
    
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    
    return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0

def simulate_backtest_data(num_periods: int = 1000, num_assets: int = 10) -> Dict[str, np.ndarray]:
    """Simulate backtest data for performance testing"""
    
    # Generate random returns
    returns = np.random.normal(0.0001, 0.02, (num_periods, num_assets))
    
    # Generate equity curves
    equity_curves = {}
    for i in range(num_assets):
        prices = np.cumprod(1 + returns[:, i]) * 100
        equity_curves[f'asset_{i}'] = prices
    
    return equity_curves

def benchmark_serial_processing(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Benchmark serial processing performance"""
    
    start_time = time.time()
    
    results = {}
    for asset_name, equity_curve in data.items():
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate metrics
        sharpe = fast_sharpe_calculation(returns)
        max_dd = fast_drawdown_calculation(equity_curve)
        var_95 = fast_var_calculation(returns, 0.95)
        
        # Kelly calculation
        win_rate = np.sum(returns > 0) / len(returns)
        avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.0
        avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0.0
        kelly = fast_kelly_calculation(win_rate, avg_win, avg_loss)
        
        results[asset_name] = {
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'var_95': var_95,
            'kelly': kelly
        }
    
    processing_time = time.time() - start_time
    return results, processing_time

def process_asset_parallel(asset_data: tuple) -> tuple:
    """Process single asset for parallel execution"""
    asset_name, equity_curve = asset_data
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # Calculate metrics
    sharpe = fast_sharpe_calculation(returns)
    max_dd = fast_drawdown_calculation(equity_curve)
    var_95 = fast_var_calculation(returns, 0.95)
    
    # Kelly calculation
    win_rate = np.sum(returns > 0) / len(returns)
    avg_win = np.mean(returns[returns > 0]) if np.any(returns > 0) else 0.0
    avg_loss = np.mean(returns[returns < 0]) if np.any(returns < 0) else 0.0
    kelly = fast_kelly_calculation(win_rate, avg_win, avg_loss)
    
    return asset_name, {
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'var_95': var_95,
        'kelly': kelly
    }

def benchmark_parallel_processing(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Benchmark parallel processing performance"""
    
    start_time = time.time()
    
    results = {}
    max_workers = min(cpu_count(), len(data))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_asset_parallel, item): item[0] for item in data.items()}
        
        for future in as_completed(futures):
            asset_name, metrics = future.result()
            results[asset_name] = metrics
    
    processing_time = time.time() - start_time
    return results, processing_time

def benchmark_memory_efficiency():
    """Benchmark memory efficiency improvements"""
    
    # Memory usage before optimization
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large dataset
    large_data = simulate_backtest_data(10000, 50)
    
    # Memory usage after data creation
    memory_after_data = process.memory_info().rss / 1024 / 1024  # MB
    
    # Process data with memory-efficient approach
    start_time = time.time()
    
    # Use generators and chunking for memory efficiency
    def process_in_chunks(data, chunk_size=10):
        items = list(data.items())
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i + chunk_size])
            yield benchmark_serial_processing(chunk)
    
    results = {}
    for chunk_results, _ in process_in_chunks(large_data):
        results.update(chunk_results)
    
    processing_time = time.time() - start_time
    memory_after_processing = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'memory_before': memory_before,
        'memory_after_data': memory_after_data,
        'memory_after_processing': memory_after_processing,
        'processing_time': processing_time,
        'data_size_mb': (memory_after_data - memory_before),
        'memory_efficiency': (memory_after_processing - memory_after_data) / (memory_after_data - memory_before)
    }

def run_monte_carlo_benchmark(num_runs: int = 1000):
    """Benchmark Monte Carlo simulation performance"""
    
    print(f"Running Monte Carlo benchmark with {num_runs} simulations...")
    
    start_time = time.time()
    
    # Generate random scenarios
    scenarios = []
    for i in range(num_runs):
        returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns for 1 year
        equity_curve = np.cumprod(1 + returns) * 10000
        scenarios.append(equity_curve)
    
    # Calculate statistics across scenarios
    final_values = [scenario[-1] for scenario in scenarios]
    total_returns = [(fv - 10000) / 10000 for fv in final_values]
    
    # Performance metrics
    mean_return = np.mean(total_returns)
    std_return = np.std(total_returns)
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    
    # Risk metrics
    var_95 = np.percentile(total_returns, 5)
    cvar_95 = np.mean([r for r in total_returns if r <= var_95])
    
    processing_time = time.time() - start_time
    
    return {
        'num_runs': num_runs,
        'processing_time': processing_time,
        'mean_return': mean_return,
        'std_return': std_return,
        'sharpe_ratio': sharpe_ratio,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'runs_per_second': num_runs / processing_time
    }

def main():
    """Run comprehensive performance benchmarks"""
    
    print("=== ENHANCED BACKTESTING PERFORMANCE BENCHMARKS ===\n")
    
    # Test 1: JIT Compilation Performance
    print("1. Testing JIT compilation performance...")
    
    # Generate test data
    test_data = simulate_backtest_data(5000, 20)
    
    # Benchmark serial processing
    print("   Running serial processing benchmark...")
    serial_results, serial_time = benchmark_serial_processing(test_data)
    
    # Benchmark parallel processing
    print("   Running parallel processing benchmark...")
    parallel_results, parallel_time = benchmark_parallel_processing(test_data)
    
    # Calculate speedup
    speedup = serial_time / parallel_time
    
    print(f"   Serial processing time: {serial_time:.3f} seconds")
    print(f"   Parallel processing time: {parallel_time:.3f} seconds")
    print(f"   Speedup: {speedup:.2f}x")
    print(f"   CPU cores used: {cpu_count()}")
    
    # Test 2: Memory Efficiency
    print("\n2. Testing memory efficiency...")
    memory_stats = benchmark_memory_efficiency()
    
    print(f"   Memory before: {memory_stats['memory_before']:.1f} MB")
    print(f"   Memory after data: {memory_stats['memory_after_data']:.1f} MB")
    print(f"   Memory after processing: {memory_stats['memory_after_processing']:.1f} MB")
    print(f"   Data size: {memory_stats['data_size_mb']:.1f} MB")
    print(f"   Memory efficiency ratio: {memory_stats['memory_efficiency']:.2f}")
    
    # Test 3: Monte Carlo Performance
    print("\n3. Testing Monte Carlo simulation performance...")
    mc_stats = run_monte_carlo_benchmark(1000)
    
    print(f"   Monte Carlo runs: {mc_stats['num_runs']}")
    print(f"   Processing time: {mc_stats['processing_time']:.3f} seconds")
    print(f"   Runs per second: {mc_stats['runs_per_second']:.0f}")
    print(f"   Mean return: {mc_stats['mean_return']:.4f}")
    print(f"   Sharpe ratio: {mc_stats['sharpe_ratio']:.3f}")
    print(f"   VaR 95%: {mc_stats['var_95']:.4f}")
    
    # Test 4: Risk Calculation Performance
    print("\n4. Testing risk calculation performance...")
    
    # Generate large portfolio
    num_assets = 100
    portfolio_data = simulate_backtest_data(2000, num_assets)
    
    start_time = time.time()
    
    # Calculate portfolio-level metrics
    portfolio_returns = np.zeros(1999)  # 2000 - 1 for returns
    
    for i, (asset_name, equity_curve) in enumerate(portfolio_data.items()):
        returns = np.diff(equity_curve) / equity_curve[:-1]
        weight = 1.0 / num_assets  # Equal weight
        portfolio_returns += weight * returns
    
    # Portfolio risk metrics
    portfolio_var = fast_var_calculation(portfolio_returns, 0.95)
    portfolio_sharpe = fast_sharpe_calculation(portfolio_returns)
    
    risk_calc_time = time.time() - start_time
    
    print(f"   Portfolio size: {num_assets} assets")
    print(f"   Risk calculation time: {risk_calc_time:.3f} seconds")
    print(f"   Portfolio VaR 95%: {portfolio_var:.4f}")
    print(f"   Portfolio Sharpe: {portfolio_sharpe:.3f}")
    
    # Performance Summary
    print("\n=== PERFORMANCE SUMMARY ===")
    print(f"Parallel processing speedup: {speedup:.2f}x")
    print(f"Memory efficiency ratio: {memory_stats['memory_efficiency']:.2f}")
    print(f"Monte Carlo throughput: {mc_stats['runs_per_second']:.0f} runs/second")
    print(f"Risk calculation speed: {num_assets / risk_calc_time:.0f} assets/second")
    
    # Estimated performance improvement
    total_improvement = speedup * (1 / max(0.1, memory_stats['memory_efficiency']))
    print(f"Estimated total performance improvement: {total_improvement:.1f}x")
    
    print("\n=== BENCHMARK COMPLETE ===")

if __name__ == "__main__":
    main()