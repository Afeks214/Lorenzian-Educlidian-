"""
Usage Examples for Lorentzian Feature Engineering Pipeline
=========================================================

Comprehensive examples demonstrating how to use the Lorentzian Classification
feature engineering system in various scenarios.

Examples:
1. Basic Feature Engineering
2. Real-time Streaming Processing
3. Multi-timeframe Analysis
4. Custom Configuration
5. Integration with Trading Systems
6. Performance Optimization

Author: Claude Code Agent
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any
import warnings

from .feature_engineering import (
    LorentzianFeatureEngine,
    LorentzianConfig,
    KernelConfig,
    FeatureConfig,
    FeatureType,
    NormalizationMethod,
    create_production_config
)

warnings.filterwarnings('ignore')


def example_1_basic_usage():
    """Example 1: Basic feature engineering usage"""
    
    print("=" * 60)
    print("EXAMPLE 1: Basic Feature Engineering Usage")
    print("=" * 60)
    
    # Create feature engine with default configuration
    config = create_production_config()
    engine = LorentzianFeatureEngine(config)
    
    print(f"✅ Created feature engine with {len(config.feature_configs)} features")
    
    # Generate sample OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    # Simulate realistic price movement
    base_price = 100.0
    returns = np.random.normal(0.0002, 0.015, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    noise = 0.005
    high = prices * (1 + np.abs(np.random.normal(0, noise, 100)))
    low = prices * (1 - np.abs(np.random.normal(0, noise, 100)))
    open_prices = np.roll(prices, 1)
    volume = np.random.lognormal(9, 0.5, 100)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low, 
        'close': prices,
        'volume': volume
    }, index=dates)
    
    print(f"📊 Generated {len(df)} bars of sample data")
    
    # Process features
    print("🔄 Processing features...")
    start_time = time.time()
    
    feature_df = engine.process_dataframe(df)
    
    processing_time = time.time() - start_time
    print(f"✅ Processing completed in {processing_time:.3f} seconds")
    
    # Display results
    print(f"\n📈 RESULTS:")
    print(f"   Input shape: {df.shape}")
    print(f"   Output shape: {feature_df.shape}")
    print(f"   Features: {list(feature_df.columns[:5])}")
    print(f"   Processing time per bar: {processing_time/len(df)*1000:.2f}ms")
    
    # Show sample features
    print(f"\n🔍 SAMPLE FEATURES (last 3 rows):")
    print(feature_df.iloc[-3:, :5].round(4))
    
    # Performance stats
    perf_stats = engine.get_performance_stats()
    print(f"\n⚡ PERFORMANCE:")
    print(f"   Average calculation time: {perf_stats.get('avg_calculation_time_ms', 0):.2f}ms")
    print(f"   Average feature quality: {perf_stats.get('avg_feature_quality', 0):.3f}")
    
    return engine, feature_df


def example_2_streaming_processing():
    """Example 2: Real-time streaming feature processing"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Real-time Streaming Processing")
    print("=" * 60)
    
    # Create engine optimized for streaming
    config = create_production_config()
    config.streaming_mode = True
    config.enable_caching = True
    
    engine = LorentzianFeatureEngine(config)
    print(f"✅ Created streaming-optimized engine")
    
    # Simulate real-time data feed
    print("🔄 Simulating real-time data stream...")
    
    base_price = 100.0
    current_time = pd.Timestamp.now()
    
    streaming_results = []
    
    for i in range(20):
        # Simulate new bar arrival
        time.sleep(0.01)  # Simulate real-time delay
        
        # Generate new OHLCV bar
        price_change = np.random.normal(0, 0.001)
        new_price = base_price * (1 + price_change)
        
        high = new_price * (1 + abs(np.random.normal(0, 0.005)))
        low = new_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = base_price  # Previous close
        volume = np.random.lognormal(9, 0.2)
        
        # Process single bar
        result = engine.process_bar(
            high=high,
            low=low,
            open_price=open_price,
            close=new_price,
            volume=volume,
            timestamp=current_time + pd.Timedelta(minutes=i)
        )
        
        streaming_results.append(result)
        base_price = new_price
        
        # Print every 5th result
        if (i + 1) % 5 == 0:
            features = result['normalized_features']
            print(f"   Bar {i+1:2d}: [{', '.join(f'{f:.3f}' for f in features[:3])}...] "
                  f"({result['calculation_time_ms']:.1f}ms, Q:{result['quality_score']:.2f})")
    
    print(f"✅ Processed {len(streaming_results)} bars in real-time")
    
    # Analyze streaming performance
    calc_times = [r['calculation_time_ms'] for r in streaming_results]
    quality_scores = [r['quality_score'] for r in streaming_results]
    
    print(f"\n📊 STREAMING PERFORMANCE:")
    print(f"   Average processing time: {np.mean(calc_times):.2f}ms")
    print(f"   Max processing time: {np.max(calc_times):.2f}ms")
    print(f"   Average quality score: {np.mean(quality_scores):.3f}")
    print(f"   Real-time capable: {'✅' if np.max(calc_times) < 50 else '❌'} (< 50ms)")
    
    return streaming_results


def example_3_multi_timeframe_analysis():
    """Example 3: Multi-timeframe feature analysis"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multi-timeframe Analysis")
    print("=" * 60)
    
    # Create engine
    engine = LorentzianFeatureEngine(create_production_config())
    
    # Generate high-frequency data (1-minute bars)
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1T')
    
    # Generate realistic intraday data
    base_price = 100.0
    intraday_pattern = np.sin(np.linspace(0, 2*np.pi, n_bars)) * 0.002  # Daily pattern
    noise = np.random.normal(0, 0.008, n_bars)
    returns = intraday_pattern + noise
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    high = prices * (1 + np.abs(np.random.normal(0, 0.003, n_bars)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.003, n_bars)))
    open_prices = np.roll(prices, 1)
    volume = np.random.lognormal(8.5, 0.3, n_bars)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)
    
    print(f"📊 Generated {len(df)} minute bars")
    
    # Process multiple timeframes
    timeframes = ['5T', '15T', '1H']
    print(f"🔄 Processing {len(timeframes)} timeframes: {timeframes}")
    
    start_time = time.time()
    mtf_features = engine.get_multi_timeframe_features(df, timeframes)
    processing_time = time.time() - start_time
    
    print(f"✅ Multi-timeframe processing completed in {processing_time:.2f}s")
    
    # Analyze results
    print(f"\n📈 MULTI-TIMEFRAME RESULTS:")
    print(f"   Original data shape: {df.shape}")
    print(f"   MTF features shape: {mtf_features.shape}")
    print(f"   Feature columns: {len(mtf_features.columns)}")
    
    # Show timeframe breakdown
    for tf in timeframes:
        tf_cols = [col for col in mtf_features.columns if tf in col]
        print(f"   {tf} timeframe: {len(tf_cols)} features")
    
    # Sample correlation analysis between timeframes
    print(f"\n🔗 TIMEFRAME CORRELATIONS:")
    base_feature = 'rsi'
    correlations = {}
    
    for i, tf1 in enumerate(timeframes):
        for tf2 in timeframes[i+1:]:
            col1 = f"{base_feature}_{tf1}"
            col2 = f"{base_feature}_{tf2}"
            
            if col1 in mtf_features.columns and col2 in mtf_features.columns:
                corr = mtf_features[[col1, col2]].corr().iloc[0, 1]
                correlations[f"{tf1}-{tf2}"] = corr
                print(f"   {tf1} vs {tf2}: {corr:.3f}")
    
    # Show sample multi-timeframe features
    print(f"\n🔍 SAMPLE MTF FEATURES (last 3 rows, first 6 columns):")
    print(mtf_features.iloc[-3:, :6].round(4))
    
    return mtf_features


def example_4_custom_configuration():
    """Example 4: Custom feature configuration"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 60)
    
    # Create custom feature configurations
    custom_features = [
        FeatureConfig("rsi", FeatureType.MOMENTUM, 21, NormalizationMethod.MIN_MAX, 0.3),
        FeatureConfig("wt1", FeatureType.MOMENTUM, 15, NormalizationMethod.MIN_MAX, 0.4),
        FeatureConfig("adx", FeatureType.TREND, 20, NormalizationMethod.PERCENTILE, 0.2),
        FeatureConfig("momentum", FeatureType.MOMENTUM, 10, NormalizationMethod.Z_SCORE, 0.1),
    ]
    
    # Create custom kernel configuration
    custom_kernel = KernelConfig(
        kernel_type="gaussian",
        lookback_window=12,
        relative_weighting=5.0,
        regression_level=30.0,
        adaptive_params=True
    )
    
    # Create custom overall configuration
    custom_config = LorentzianConfig(
        lookback_window=12,
        k_neighbors=10,
        feature_count=4,
        kernel_config=custom_kernel,
        feature_configs=custom_features,
        enable_numba=True,
        enable_caching=True,
        streaming_mode=True,
        use_volatility_filter=True,
        volatility_threshold=0.12,
        use_adx_filter=True,
        adx_threshold=20.0
    )
    
    print(f"⚙️  Created custom configuration:")
    print(f"   Features: {[f.name for f in custom_features]}")
    print(f"   Kernel: {custom_kernel.kernel_type}")
    print(f"   Lookback: {custom_config.lookback_window}")
    print(f"   K-neighbors: {custom_config.k_neighbors}")
    
    # Create engine with custom config
    engine = LorentzianFeatureEngine(custom_config)
    
    # Generate test data
    np.random.seed(42)
    test_df = pd.DataFrame({
        'open': [100, 101, 99, 102, 98, 101, 103, 99],
        'high': [102, 103, 101, 104, 100, 103, 105, 101],
        'low': [99, 100, 98, 101, 97, 100, 102, 98],
        'close': [101, 99, 102, 98, 101, 103, 99, 100],
        'volume': [1000, 1100, 900, 1200, 800, 1050, 1300, 950]
    }, index=pd.date_range('2024-01-01', periods=8, freq='1H'))
    
    print(f"📊 Processing test data with custom configuration...")
    
    # Process features
    feature_df = engine.process_dataframe(test_df)
    
    print(f"✅ Custom processing completed")
    print(f"   Output shape: {feature_df.shape}")
    print(f"   Feature names: {list(feature_df.columns[:-2])}")  # Exclude metadata
    
    # Compare with default configuration
    default_engine = LorentzianFeatureEngine(create_production_config())
    default_features = default_engine.process_dataframe(test_df)
    
    print(f"\n🔄 COMPARISON WITH DEFAULT:")
    print(f"   Custom features: {feature_df.shape[1] - 2}")  # Exclude metadata
    print(f"   Default features: {default_features.shape[1] - 2}")
    
    # Show feature importance differences
    custom_importance = engine.get_feature_importance()
    default_importance = default_engine.get_feature_importance()
    
    print(f"\n⚖️  FEATURE IMPORTANCE COMPARISON:")
    all_features = set(custom_importance.keys()) | set(default_importance.keys())
    for feature in sorted(all_features):
        custom_weight = custom_importance.get(feature, 0.0)
        default_weight = default_importance.get(feature, 0.0)
        print(f"   {feature:10s}: Custom={custom_weight:.3f}, Default={default_weight:.3f}")
    
    return engine, feature_df


def example_5_trading_integration():
    """Example 5: Integration with trading system"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Trading System Integration")
    print("=" * 60)
    
    # Create optimized engine for trading
    config = create_production_config()
    config.streaming_mode = True
    config.enable_caching = True
    
    engine = LorentzianFeatureEngine(config)
    print(f"🎯 Created trading-optimized engine")
    
    # Trading system components
    class SimpleLorentzianStrategy:
        def __init__(self, feature_engine):
            self.engine = feature_engine
            self.feature_history = []
            self.signals = []
            self.positions = []
            self.pnl = []
            
        def process_bar(self, bar_data):
            """Process new bar and generate trading signal"""
            # Extract OHLCV
            result = self.engine.process_bar(
                high=bar_data['high'],
                low=bar_data['low'],
                open_price=bar_data['open'],
                close=bar_data['close'],
                volume=bar_data['volume'],
                timestamp=bar_data['timestamp']
            )
            
            features = result['normalized_features']
            self.feature_history.append(features)
            
            # Simple signal generation (placeholder for actual Lorentzian classification)
            if len(self.feature_history) >= 2:
                # Look for feature pattern changes
                prev_features = self.feature_history[-2]
                curr_features = features
                
                # Simple momentum-based signal
                feature_momentum = np.mean(curr_features) - np.mean(prev_features)
                
                if feature_momentum > 0.01 and result['quality_score'] > 0.7:
                    signal = 1  # Buy
                elif feature_momentum < -0.01 and result['quality_score'] > 0.7:
                    signal = -1  # Sell
                else:
                    signal = 0  # Hold
            else:
                signal = 0
            
            self.signals.append(signal)
            
            # Position management
            current_position = self.positions[-1] if self.positions else 0
            
            if signal == 1 and current_position <= 0:
                new_position = 1  # Go long
            elif signal == -1 and current_position >= 0:
                new_position = -1  # Go short
            else:
                new_position = current_position  # Hold position
            
            self.positions.append(new_position)
            
            # Calculate PnL (simplified)
            if len(self.positions) >= 2:
                price_change = bar_data['close'] - (bar_data['open'] if len(self.positions) == 2 
                                                  else self.last_price)
                position_pnl = self.positions[-2] * price_change
                self.pnl.append(position_pnl)
            else:
                self.pnl.append(0.0)
            
            self.last_price = bar_data['close']
            
            return {
                'signal': signal,
                'position': new_position,
                'features': features,
                'quality': result['quality_score'],
                'processing_time': result['calculation_time_ms']
            }
    
    # Create strategy
    strategy = SimpleLorentzianStrategy(engine)
    print(f"📈 Created simple Lorentzian strategy")
    
    # Generate trading data
    np.random.seed(42)
    n_bars = 50
    
    base_price = 100.0
    trading_results = []
    
    print(f"🔄 Running strategy on {n_bars} bars...")
    
    for i in range(n_bars):
        # Generate bar data
        price_change = np.random.normal(0.0002, 0.01)
        new_price = base_price * (1 + price_change)
        
        bar_data = {
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i),
            'open': base_price,
            'high': max(base_price, new_price) * (1 + abs(np.random.normal(0, 0.002))),
            'low': min(base_price, new_price) * (1 - abs(np.random.normal(0, 0.002))),
            'close': new_price,
            'volume': np.random.lognormal(9, 0.3)
        }
        
        # Process through strategy
        result = strategy.process_bar(bar_data)
        trading_results.append(result)
        
        base_price = new_price
        
        # Print every 10th result
        if (i + 1) % 10 == 0:
            print(f"   Bar {i+1:2d}: Signal={result['signal']:2d}, "
                  f"Position={result['position']:2d}, Quality={result['quality']:.2f}")
    
    # Analyze trading results
    signals = [r['signal'] for r in trading_results]
    positions = [r['position'] for r in trading_results]
    processing_times = [r['processing_time'] for r in trading_results]
    
    total_pnl = sum(strategy.pnl)
    signal_count = len([s for s in signals if s != 0])
    avg_processing_time = np.mean(processing_times)
    
    print(f"\n📊 TRADING RESULTS:")
    print(f"   Total bars processed: {len(trading_results)}")
    print(f"   Signals generated: {signal_count}")
    print(f"   Buy signals: {len([s for s in signals if s == 1])}")
    print(f"   Sell signals: {len([s for s in signals if s == -1])}")
    print(f"   Total PnL: {total_pnl:.2f}")
    print(f"   Average processing time: {avg_processing_time:.2f}ms")
    
    return strategy, trading_results


def example_6_performance_optimization():
    """Example 6: Performance optimization techniques"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Performance Optimization")
    print("=" * 60)
    
    # Test different configuration options
    configurations = [
        ("Default", create_production_config()),
        ("No Numba", create_production_config()),
        ("No Caching", create_production_config()),
        ("Minimal Features", create_production_config())
    ]
    
    # Modify configurations
    configurations[1][1].enable_numba = False
    configurations[2][1].enable_caching = False
    configurations[3][1].feature_configs = configurations[3][1].feature_configs[:3]  # Only 3 features
    
    # Generate test data
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='5T')
    
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_bars)))
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_prices = np.roll(prices, 1)
    volume = np.random.lognormal(9, 0.4, n_bars)
    
    test_df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)
    
    print(f"📊 Testing performance on {len(test_df)} bars")
    
    results = {}
    
    for name, config in configurations:
        print(f"\n🔄 Testing {name} configuration...")
        
        # Create engine
        engine = LorentzianFeatureEngine(config)
        
        # Measure processing time
        start_time = time.time()
        feature_df = engine.process_dataframe(test_df)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(test_df) / processing_time
        avg_time_per_bar = processing_time / len(test_df) * 1000  # ms
        
        # Get performance stats
        perf_stats = engine.get_performance_stats()
        
        results[name] = {
            'total_time': processing_time,
            'throughput': throughput,
            'time_per_bar': avg_time_per_bar,
            'avg_calc_time': perf_stats.get('avg_calculation_time_ms', 0),
            'feature_count': len(config.feature_configs),
            'cache_enabled': config.enable_caching,
            'numba_enabled': config.enable_numba
        }
        
        print(f"   ✅ {name}: {processing_time:.3f}s total, {throughput:.1f} bars/sec")
    
    # Performance comparison
    print(f"\n⚡ PERFORMANCE COMPARISON:")
    print(f"{'Configuration':<15} {'Time(s)':<8} {'Bars/sec':<10} {'ms/bar':<8} {'Features':<8}")
    print("-" * 55)
    
    baseline_time = results['Default']['total_time']
    
    for name, result in results.items():
        speedup = baseline_time / result['total_time']
        speedup_str = f"({speedup:.1f}x)" if name != 'Default' else ""
        
        print(f"{name:<15} {result['total_time']:<8.3f} {result['throughput']:<10.1f} "
              f"{result['time_per_bar']:<8.1f} {result['feature_count']:<8} {speedup_str}")
    
    # Recommendations
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    best_config = min(results.items(), key=lambda x: x[1]['total_time'])
    fastest_throughput = max(results.items(), key=lambda x: x[1]['throughput'])
    
    print(f"   Fastest overall: {best_config[0]} ({best_config[1]['total_time']:.3f}s)")
    print(f"   Highest throughput: {fastest_throughput[0]} ({fastest_throughput[1]['throughput']:.1f} bars/sec)")
    
    if results['No Numba']['total_time'] > results['Default']['total_time'] * 1.5:
        print(f"   ✅ Numba provides significant speedup - keep enabled")
    
    if results['No Caching']['total_time'] > results['Default']['total_time'] * 1.2:
        print(f"   ✅ Caching provides good speedup - keep enabled")
    
    return results


def run_all_examples():
    """Run all examples in sequence"""
    
    print("🚀 LORENTZIAN FEATURE ENGINEERING EXAMPLES")
    print("=" * 80)
    print("Running comprehensive examples demonstrating the feature engineering pipeline...")
    
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Streaming Processing", example_2_streaming_processing),
        ("Multi-timeframe Analysis", example_3_multi_timeframe_analysis),
        ("Custom Configuration", example_4_custom_configuration),
        ("Trading Integration", example_5_trading_integration),
        ("Performance Optimization", example_6_performance_optimization)
    ]
    
    results = {}
    total_start_time = time.time()
    
    for name, example_func in examples:
        try:
            print(f"\n🎯 Running {name}...")
            start_time = time.time()
            
            result = example_func()
            
            execution_time = time.time() - start_time
            results[name] = {
                'status': 'success',
                'execution_time': execution_time,
                'result': result
            }
            
            print(f"✅ {name} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            results[name] = {
                'status': 'error',
                'execution_time': execution_time,
                'error': str(e)
            }
            
            print(f"❌ {name} failed: {e}")
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n" + "=" * 80)
    print("EXAMPLES SUMMARY")
    print("=" * 80)
    
    successful = len([r for r in results.values() if r['status'] == 'success'])
    total = len(results)
    
    print(f"Total examples: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Total execution time: {total_time:.2f}s")
    
    print(f"\nINDIVIDUAL RESULTS:")
    for name, result in results.items():
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"  {status_icon} {name}: {result['execution_time']:.2f}s")
        if result['status'] == 'error':
            print(f"     Error: {result['error']}")
    
    print(f"\n🎉 Examples demonstration completed!")
    print(f"💡 The Lorentzian Feature Engineering system is ready for production use.")
    
    return results


if __name__ == "__main__":
    # Run all examples
    example_results = run_all_examples()