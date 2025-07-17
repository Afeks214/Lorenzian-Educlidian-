"""
Comprehensive Benchmarks for Indicators System
Academic accuracy validation, real-world data testing, and production readiness verification
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import json
import os

from src.indicators.engine import IndicatorEngine
from src.indicators.custom.mlmi import MLMICalculator
from src.indicators.custom.nwrqk import NWRQKCalculator
from src.indicators.custom.fvg import FVGDetector
from src.indicators.custom.lvn import LVNAnalyzer
from src.indicators.custom.mmd import MMDFeatureExtractor
from src.core.events import EventType, Event, BarData
from tests.mocks.mock_kernel import MockKernel
from tests.mocks.mock_event_bus import MockEventBus


@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    indicator_name: str
    test_name: str
    accuracy_score: float
    performance_ms: float
    memory_mb: float
    passed: bool
    details: Dict[str, Any]


class AcademicBenchmarks:
    """Academic benchmark implementations for validation"""
    
    @staticmethod
    def reference_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Reference ATR implementation for validation"""
        true_ranges = []
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
            
        # Calculate ATR using simple moving average
        atr = []
        for i in range(period-1, len(true_ranges)):
            atr_val = np.mean(true_ranges[i-period+1:i+1])
            atr.append(atr_val)
            
        return np.array(atr)
    
    @staticmethod
    def reference_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Reference RSI implementation for validation"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi = []
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
                
        return np.array(rsi)
    
    @staticmethod
    def reference_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Reference Simple Moving Average for validation"""
        sma = []
        for i in range(period-1, len(prices)):
            sma.append(np.mean(prices[i-period+1:i+1]))
        return np.array(sma)
    
    @staticmethod
    def reference_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference Bollinger Bands for validation"""
        sma = AcademicBenchmarks.reference_sma(prices, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(period-1, len(prices)):
            window = prices[i-period+1:i+1]
            std = np.std(window)
            upper_band.append(sma[i-period+1] + (std_dev * std))
            lower_band.append(sma[i-period+1] - (std_dev * std))
            
        return np.array(upper_band), sma, np.array(lower_band)


class RealWorldDataGenerator:
    """Generate realistic market data for testing"""
    
    @staticmethod
    def generate_trending_market(length: int = 1000, trend_strength: float = 0.1) -> List[BarData]:
        """Generate trending market data"""
        bars = []
        base_price = 100.0
        
        for i in range(length):
            # Add trend
            trend_component = i * trend_strength
            # Add noise
            noise = np.random.normal(0, 0.5)
            # Add some mean reversion
            mean_reversion = -0.1 * (base_price + trend_component - 100.0)
            
            price = base_price + trend_component + noise + mean_reversion
            
            # Generate OHLC
            high = price + np.random.uniform(0.1, 1.0)
            low = price - np.random.uniform(0.1, 1.0)
            open_price = price + np.random.uniform(-0.5, 0.5)
            
            volume = int(np.random.uniform(800, 1200))
            timestamp = datetime.now() + timedelta(minutes=i*30)
            
            bar = BarData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=price,
                volume=volume,
                timeframe=30
            )
            bars.append(bar)
            
        return bars
    
    @staticmethod
    def generate_volatile_market(length: int = 1000, volatility: float = 2.0) -> List[BarData]:
        """Generate volatile market data"""
        bars = []
        base_price = 100.0
        
        for i in range(length):
            # High volatility with clustering
            if i > 0:
                prev_change = bars[i-1].close - base_price
                volatility_factor = 1.0 + abs(prev_change) * 0.1
            else:
                volatility_factor = 1.0
                
            change = np.random.normal(0, volatility * volatility_factor)
            price = base_price + change
            
            # Generate OHLC with high ranges
            range_size = np.random.uniform(1.0, 5.0)
            high = price + range_size * np.random.uniform(0.3, 0.7)
            low = price - range_size * np.random.uniform(0.3, 0.7)
            open_price = price + np.random.uniform(-range_size/2, range_size/2)
            
            volume = int(np.random.uniform(500, 2000))
            timestamp = datetime.now() + timedelta(minutes=i*30)
            
            bar = BarData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=price,
                volume=volume,
                timeframe=30
            )
            bars.append(bar)
            
        return bars
    
    @staticmethod
    def generate_ranging_market(length: int = 1000, range_width: float = 10.0) -> List[BarData]:
        """Generate ranging market data"""
        bars = []
        base_price = 100.0
        
        for i in range(length):
            # Oscillate within range
            cycle_position = (i / length) * 4 * np.pi  # 4 full cycles
            range_position = np.sin(cycle_position) * (range_width / 2)
            noise = np.random.normal(0, 0.2)
            
            price = base_price + range_position + noise
            
            # Generate OHLC
            high = price + np.random.uniform(0.1, 0.5)
            low = price - np.random.uniform(0.1, 0.5)
            open_price = price + np.random.uniform(-0.3, 0.3)
            
            volume = int(np.random.uniform(900, 1100))
            timestamp = datetime.now() + timedelta(minutes=i*30)
            
            bar = BarData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=price,
                volume=volume,
                timeframe=30
            )
            bars.append(bar)
            
        return bars


class TestComprehensiveBenchmarks:
    """Comprehensive benchmark test suite"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_kernel = MockKernel()
        self.mock_event_bus = MockEventBus()
        self.benchmarks = AcademicBenchmarks()
        self.data_generator = RealWorldDataGenerator()
        self.results = []
        
    def calculate_accuracy_score(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Calculate accuracy score between predicted and actual values"""
        if len(predicted) == 0 or len(actual) == 0:
            return 0.0
            
        # Handle different lengths
        min_len = min(len(predicted), len(actual))
        pred_aligned = predicted[-min_len:] if len(predicted) > min_len else predicted
        actual_aligned = actual[-min_len:] if len(actual) > min_len else actual
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(pred_aligned, actual_aligned)[0, 1]
        
        # Calculate RMSE normalized by standard deviation
        rmse = np.sqrt(np.mean((pred_aligned - actual_aligned) ** 2))
        std_dev = np.std(actual_aligned)
        normalized_rmse = rmse / std_dev if std_dev > 0 else 1.0
        
        # Combine metrics (higher is better)
        accuracy = max(0, correlation) * (1 - min(normalized_rmse, 1.0))
        
        return accuracy
    
    def test_mlmi_accuracy_benchmark(self):
        """Test MLMI accuracy against known patterns"""
        # Generate trending data
        bars = self.data_generator.generate_trending_market(500, 0.2)
        
        mlmi = MLMICalculator({'num_neighbors': 200}, self.mock_event_bus)
        
        start_time = time.time()
        mlmi_values = []
        
        for bar in bars:
            result = mlmi.calculate_30m(bar)
            mlmi_values.append(result['mlmi_value'])
            
        calc_time = (time.time() - start_time) * 1000
        
        # MLMI should detect the trend
        valid_values = [v for v in mlmi_values if v != 0.0]
        
        # Test performance
        avg_time_per_calc = calc_time / len(bars)
        
        # Generate prices for comparison
        prices = [bar.close for bar in bars]
        trend_indicator = np.diff(prices)  # Simple trend measure
        
        # Calculate accuracy
        if len(valid_values) > 10:
            accuracy = self.calculate_accuracy_score(
                np.array(valid_values[-len(trend_indicator):]), 
                trend_indicator
            )
        else:
            accuracy = 0.0
            
        result = BenchmarkResult(
            indicator_name="MLMI",
            test_name="trending_market_accuracy",
            accuracy_score=accuracy,
            performance_ms=avg_time_per_calc,
            memory_mb=0.0,  # TODO: Add memory measurement
            passed=accuracy > 0.3 and avg_time_per_calc < 1.0,
            details={
                'total_calculations': len(bars),
                'valid_values': len(valid_values),
                'trend_detected': len([v for v in valid_values if v > 0]) > len(valid_values) * 0.6
            }
        )
        
        self.results.append(result)
        assert result.passed, f"MLMI benchmark failed: accuracy={accuracy:.3f}, time={avg_time_per_calc:.3f}ms"
        
    def test_nwrqk_accuracy_benchmark(self):
        """Test NWRQK accuracy against kernel regression theory"""
        # Generate smooth data suitable for kernel regression
        bars = self.data_generator.generate_ranging_market(200, 5.0)
        
        nwrqk = NWRQKCalculator({'h': 8.0, 'r': 8.0}, self.mock_event_bus)
        
        start_time = time.time()
        nwrqk_values = []
        
        for bar in bars:
            result = nwrqk.calculate_30m(bar)
            nwrqk_values.append(result['nwrqk_value'])
            
        calc_time = (time.time() - start_time) * 1000
        avg_time_per_calc = calc_time / len(bars)
        
        # NWRQK should smooth the data
        valid_values = [v for v in nwrqk_values if v != 0.0]
        
        if len(valid_values) > 10:
            # Test smoothness (should be less volatile than original)
            prices = [bar.close for bar in bars[-len(valid_values):]]
            price_volatility = np.std(np.diff(prices))
            nwrqk_volatility = np.std(np.diff(valid_values))
            
            smoothness_ratio = nwrqk_volatility / price_volatility if price_volatility > 0 else 1.0
            accuracy = max(0, 1.0 - smoothness_ratio)  # Lower volatility = better smoothing
        else:
            accuracy = 0.0
            
        result = BenchmarkResult(
            indicator_name="NWRQK",
            test_name="smoothing_accuracy",
            accuracy_score=accuracy,
            performance_ms=avg_time_per_calc,
            memory_mb=0.0,
            passed=accuracy > 0.2 and avg_time_per_calc < 1.0,
            details={
                'total_calculations': len(bars),
                'valid_values': len(valid_values),
                'smoothness_achieved': accuracy > 0.2
            }
        )
        
        self.results.append(result)
        assert result.passed, f"NWRQK benchmark failed: accuracy={accuracy:.3f}, time={avg_time_per_calc:.3f}ms"
        
    def test_fvg_detection_benchmark(self):
        """Test FVG detection accuracy"""
        # Generate data with known gaps
        bars = []
        base_price = 100.0
        
        # Create pattern with gaps
        for i in range(100):
            if i == 30:  # Create bullish gap
                high = base_price + 5
                low = base_price + 3  # Gap: low > previous high
                price = base_price + 4
            elif i == 60:  # Create bearish gap
                high = base_price - 3
                low = base_price - 5  # Gap: high < previous low
                price = base_price - 4
            else:
                high = base_price + np.random.uniform(0.1, 1.0)
                low = base_price - np.random.uniform(0.1, 1.0)
                price = base_price + np.random.uniform(-0.5, 0.5)
                
            timestamp = datetime.now() + timedelta(minutes=i*5)
            bar = BarData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=price,
                high=high,
                low=low,
                close=price,
                volume=1000,
                timeframe=5
            )
            bars.append(bar)
            
        fvg = FVGDetector({}, self.mock_event_bus)
        
        start_time = time.time()
        gap_detections = []
        
        for bar in bars:
            result = fvg.calculate_5m(bar)
            gap_detections.append(result)
            
        calc_time = (time.time() - start_time) * 1000
        avg_time_per_calc = calc_time / len(bars)
        
        # Check if gaps were detected
        bullish_detected = any(r['fvg_bullish_active'] for r in gap_detections[31:40])
        bearish_detected = any(r['fvg_bearish_active'] for r in gap_detections[61:70])
        
        accuracy = (bullish_detected + bearish_detected) / 2.0
        
        result = BenchmarkResult(
            indicator_name="FVG",
            test_name="gap_detection_accuracy",
            accuracy_score=accuracy,
            performance_ms=avg_time_per_calc,
            memory_mb=0.0,
            passed=accuracy >= 0.5 and avg_time_per_calc < 1.0,
            details={
                'total_calculations': len(bars),
                'bullish_detected': bullish_detected,
                'bearish_detected': bearish_detected,
                'gaps_created': 2
            }
        )
        
        self.results.append(result)
        assert result.passed, f"FVG benchmark failed: accuracy={accuracy:.3f}, time={avg_time_per_calc:.3f}ms"
        
    def test_lvn_identification_benchmark(self):
        """Test LVN identification accuracy"""
        # Generate data with known volume patterns
        bars = []
        base_price = 100.0
        
        for i in range(100):
            if 40 <= i <= 50:  # Low volume area
                volume = int(np.random.uniform(100, 300))
                price = base_price + np.random.uniform(-0.2, 0.2)
            elif 20 <= i <= 30:  # High volume area (POC)
                volume = int(np.random.uniform(2000, 5000))
                price = base_price + np.random.uniform(-0.5, 0.5)
            else:
                volume = int(np.random.uniform(800, 1200))
                price = base_price + np.random.uniform(-1.0, 1.0)
                
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = BarData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=price,
                high=price + np.random.uniform(0.1, 0.5),
                low=price - np.random.uniform(0.1, 0.5),
                close=price,
                volume=volume,
                timeframe=30
            )
            bars.append(bar)
            
        lvn = LVNAnalyzer({}, self.mock_event_bus)
        
        start_time = time.time()
        lvn_results = []
        
        for bar in bars:
            result = lvn.calculate_30m(bar)
            lvn_results.append(result)
            
        calc_time = (time.time() - start_time) * 1000
        avg_time_per_calc = calc_time / len(bars)
        
        # Check if LVN was identified in the low volume area
        valid_results = [r for r in lvn_results if r['nearest_lvn_strength'] > 0]
        
        if valid_results:
            avg_strength = np.mean([r['nearest_lvn_strength'] for r in valid_results])
            accuracy = avg_strength
        else:
            accuracy = 0.0
            
        result = BenchmarkResult(
            indicator_name="LVN",
            test_name="volume_profile_accuracy",
            accuracy_score=accuracy,
            performance_ms=avg_time_per_calc,
            memory_mb=0.0,
            passed=accuracy > 0.1 and avg_time_per_calc < 1.0,
            details={
                'total_calculations': len(bars),
                'valid_results': len(valid_results),
                'avg_strength': accuracy
            }
        )
        
        self.results.append(result)
        assert result.passed, f"LVN benchmark failed: accuracy={accuracy:.3f}, time={avg_time_per_calc:.3f}ms"
        
    def test_mmd_regime_detection_benchmark(self):
        """Test MMD regime detection accuracy"""
        # Generate data with distinct regimes
        trending_bars = self.data_generator.generate_trending_market(200, 0.3)
        volatile_bars = self.data_generator.generate_volatile_market(200, 3.0)
        
        mmd = MMDFeatureExtractor({}, self.mock_event_bus)
        
        # Test trending regime
        start_time = time.time()
        trending_results = []
        
        for bar in trending_bars:
            result = mmd.calculate_30m(bar)
            trending_results.append(result)
            
        trending_time = (time.time() - start_time) * 1000
        
        # Reset for volatile regime
        mmd.reset()
        
        start_time = time.time()
        volatile_results = []
        
        for bar in volatile_bars:
            result = mmd.calculate_30m(bar)
            volatile_results.append(result)
            
        volatile_time = (time.time() - start_time) * 1000
        
        # Compare MMD scores
        trending_scores = [r['mmd_features'][12] for r in trending_results if r['mmd_features'][12] > 0]
        volatile_scores = [r['mmd_features'][12] for r in volatile_results if r['mmd_features'][12] > 0]
        
        if trending_scores and volatile_scores:
            avg_trending = np.mean(trending_scores)
            avg_volatile = np.mean(volatile_scores)
            
            # MMD should distinguish between regimes
            regime_separation = abs(avg_trending - avg_volatile) / max(avg_trending, avg_volatile)
            accuracy = min(regime_separation, 1.0)
        else:
            accuracy = 0.0
            
        avg_time_per_calc = (trending_time + volatile_time) / (len(trending_bars) + len(volatile_bars))
        
        result = BenchmarkResult(
            indicator_name="MMD",
            test_name="regime_detection_accuracy",
            accuracy_score=accuracy,
            performance_ms=avg_time_per_calc,
            memory_mb=0.0,
            passed=accuracy > 0.1 and avg_time_per_calc < 10.0,
            details={
                'trending_calculations': len(trending_bars),
                'volatile_calculations': len(volatile_bars),
                'trending_scores': len(trending_scores),
                'volatile_scores': len(volatile_scores),
                'regime_separation': accuracy
            }
        )
        
        self.results.append(result)
        assert result.passed, f"MMD benchmark failed: accuracy={accuracy:.3f}, time={avg_time_per_calc:.3f}ms"
        
    def test_engine_integration_benchmark(self):
        """Test complete engine integration benchmark"""
        engine = IndicatorEngine("benchmark_test", self.mock_kernel)
        
        # Generate comprehensive test data
        bars = self.data_generator.generate_trending_market(500, 0.1)
        
        start_time = time.time()
        
        # Process all bars
        for bar in bars:
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            
        total_time = (time.time() - start_time) * 1000
        avg_time_per_bar = total_time / len(bars)
        
        # Check final state
        features = engine.get_current_features()
        feature_summary = engine.get_feature_summary()
        
        # Test completeness
        expected_features = [
            'mlmi_value', 'nwrqk_value', 'fvg_bullish_active', 
            'lvn_nearest_price', 'mmd_features'
        ]
        
        completeness = sum(1 for f in expected_features if f in features) / len(expected_features)
        
        result = BenchmarkResult(
            indicator_name="ENGINE",
            test_name="integration_benchmark",
            accuracy_score=completeness,
            performance_ms=avg_time_per_bar,
            memory_mb=0.0,
            passed=completeness >= 0.8 and avg_time_per_bar < 5.0,
            details={
                'total_bars': len(bars),
                'features_generated': len(features),
                'calculations_5min': feature_summary['calculations_5min'],
                'calculations_30min': feature_summary['calculations_30min'],
                'events_emitted': feature_summary['events_emitted']
            }
        )
        
        self.results.append(result)
        assert result.passed, f"Engine benchmark failed: completeness={completeness:.3f}, time={avg_time_per_bar:.3f}ms"
        
    def test_production_readiness_benchmark(self):
        """Test production readiness with realistic load"""
        engine = IndicatorEngine("production_test", self.mock_kernel)
        
        # Simulate 24 hours of 30-minute bars
        bars = self.data_generator.generate_trending_market(48, 0.05)
        
        latencies = []
        memory_usage = []
        
        import psutil
        process = psutil.Process()
        
        for i, bar in enumerate(bars):
            # Measure individual processing time
            start_time = time.perf_counter()
            engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000
            latencies.append(latency)
            
            # Measure memory usage
            if i % 10 == 0:
                memory_usage.append(process.memory_info().rss / 1024 / 1024)
                
        # Calculate statistics
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        memory_growth = max(memory_usage) - min(memory_usage) if memory_usage else 0
        
        # Production requirements
        production_ready = (
            avg_latency < 10.0 and      # < 10ms average
            p95_latency < 50.0 and      # < 50ms 95th percentile
            p99_latency < 100.0 and     # < 100ms 99th percentile
            memory_growth < 50.0        # < 50MB memory growth
        )
        
        result = BenchmarkResult(
            indicator_name="PRODUCTION",
            test_name="production_readiness",
            accuracy_score=1.0 if production_ready else 0.0,
            performance_ms=avg_latency,
            memory_mb=memory_growth,
            passed=production_ready,
            details={
                'avg_latency_ms': avg_latency,
                'max_latency_ms': max_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'memory_growth_mb': memory_growth,
                'bars_processed': len(bars)
            }
        )
        
        self.results.append(result)
        assert result.passed, f"Production readiness failed: avg_latency={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, memory={memory_growth:.2f}MB"
        
    def test_stress_testing_benchmark(self):
        """Test system under stress conditions"""
        engine = IndicatorEngine("stress_test", self.mock_kernel)
        
        # Generate stress test data
        stress_bars = []
        
        # Mixed regime data
        stress_bars.extend(self.data_generator.generate_trending_market(200, 0.5))
        stress_bars.extend(self.data_generator.generate_volatile_market(200, 5.0))
        stress_bars.extend(self.data_generator.generate_ranging_market(200, 20.0))
        
        # Add some extreme values
        for i in range(50):
            extreme_price = 100.0 + np.random.uniform(-90, 90)
            extreme_volume = int(np.random.uniform(1, 100000))
            
            timestamp = datetime.now() + timedelta(minutes=i*30)
            bar = BarData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=extreme_price,
                high=extreme_price + np.random.uniform(0, 50),
                low=extreme_price - np.random.uniform(0, 50),
                close=extreme_price,
                volume=extreme_volume,
                timeframe=30
            )
            stress_bars.append(bar)
            
        # Process under stress
        start_time = time.time()
        errors = 0
        
        for bar in stress_bars:
            try:
                engine._on_30min_bar(Event(EventType.NEW_30MIN_BAR, bar))
            except Exception as e:
                errors += 1
                
        total_time = (time.time() - start_time) * 1000
        avg_time_per_bar = total_time / len(stress_bars)
        
        # Check system stability
        error_rate = errors / len(stress_bars)
        stability = 1.0 - error_rate
        
        result = BenchmarkResult(
            indicator_name="STRESS",
            test_name="stress_testing",
            accuracy_score=stability,
            performance_ms=avg_time_per_bar,
            memory_mb=0.0,
            passed=stability > 0.99 and avg_time_per_bar < 10.0,
            details={
                'total_bars': len(stress_bars),
                'errors': errors,
                'error_rate': error_rate,
                'stability': stability
            }
        )
        
        self.results.append(result)
        assert result.passed, f"Stress test failed: stability={stability:.3f}, avg_time={avg_time_per_bar:.2f}ms"
        
    def test_generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        # Run all previous tests to populate results
        if not self.results:
            # Run essential benchmarks
            self.test_mlmi_accuracy_benchmark()
            self.test_nwrqk_accuracy_benchmark()
            self.test_fvg_detection_benchmark()
            self.test_lvn_identification_benchmark()
            self.test_mmd_regime_detection_benchmark()
            self.test_engine_integration_benchmark()
            self.test_production_readiness_benchmark()
            self.test_stress_testing_benchmark()
            
        # Generate report
        report = {
            'benchmark_timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'passed_tests': sum(1 for r in self.results if r.passed),
            'failed_tests': sum(1 for r in self.results if not r.passed),
            'overall_success_rate': sum(1 for r in self.results if r.passed) / len(self.results) if self.results else 0,
            'performance_summary': {
                'avg_performance_ms': np.mean([r.performance_ms for r in self.results]),
                'max_performance_ms': np.max([r.performance_ms for r in self.results]),
                'min_performance_ms': np.min([r.performance_ms for r in self.results]),
            },
            'accuracy_summary': {
                'avg_accuracy': np.mean([r.accuracy_score for r in self.results]),
                'max_accuracy': np.max([r.accuracy_score for r in self.results]),
                'min_accuracy': np.min([r.accuracy_score for r in self.results]),
            },
            'detailed_results': [
                {
                    'indicator': r.indicator_name,
                    'test': r.test_name,
                    'accuracy': r.accuracy_score,
                    'performance_ms': r.performance_ms,
                    'memory_mb': r.memory_mb,
                    'passed': r.passed,
                    'details': r.details
                }
                for r in self.results
            ]
        }
        
        # Save report
        report_path = '/tmp/indicators_benchmark_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        print("\n" + "="*60)
        print("INDICATORS COMPREHENSIVE BENCHMARK REPORT")
        print("="*60)
        print(f"Total tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Success rate: {report['overall_success_rate']:.1%}")
        print(f"Average performance: {report['performance_summary']['avg_performance_ms']:.2f}ms")
        print(f"Average accuracy: {report['accuracy_summary']['avg_accuracy']:.3f}")
        print("\nDetailed results:")
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            print(f"  {result.indicator_name:8} {result.test_name:25} {status:4} "
                  f"acc={result.accuracy_score:.3f} perf={result.performance_ms:.2f}ms")
                  
        print("="*60)
        print(f"Report saved to: {report_path}")
        
        # Verify overall success
        assert report['overall_success_rate'] > 0.8, f"Overall benchmark success rate too low: {report['overall_success_rate']:.1%}"
        assert report['performance_summary']['avg_performance_ms'] < 5.0, f"Average performance too slow: {report['performance_summary']['avg_performance_ms']:.2f}ms"
        
    def teardown_method(self):
        """Clean up after tests"""
        # Clear results for next test
        self.results.clear()