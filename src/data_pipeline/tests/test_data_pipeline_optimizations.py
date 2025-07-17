"""
Comprehensive test suite for data pipeline optimizations

This module contains comprehensive tests for all data pipeline optimizations
including performance benchmarks, correctness validation, and stress testing.
"""

import pytest
import numpy as np
import pandas as pd
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Import all optimization modules
from ..streaming.realtime_streamer import RealtimeDataStreamer, MarketDataStreamer, BackpressureStrategy
from ..validation.realtime_validator import RealtimeDataValidator, MarketDataValidator
from ..transformation.optimized_transformers import OptimizedTransformer, StreamingTransformer
from ..storage.optimized_storage import CompressedDataStore, OptimizedFileStorage, CompressionType
from ..storage.lifecycle_manager import DataLifecycleManager, DataTier
from ..indicators.parallel_indicators import ParallelIndicatorCalculator, ProcessingMode
from ..caching.intelligent_cache import IntelligentCache, CacheStrategy
from ..incremental.incremental_calculator import StreamingDataProcessor
from ..quality.quality_monitor import DataQualityMonitor
from ..lineage.lineage_tracker import DataLineageTracker

class TestDataPipelineOptimizations:
    """Comprehensive test suite for data pipeline optimizations"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        size = 1000
        
        # Generate realistic price data
        price_base = 100
        price_changes = np.random.normal(0, 0.02, size)
        prices = price_base * np.exp(np.cumsum(price_changes))
        
        # Generate OHLCV data
        data = {
            'open': prices * (1 + np.random.normal(0, 0.001, size)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, size))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, size))),
            'close': prices,
            'volume': np.random.randint(1000, 10000, size)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_tick_data(self):
        """Generate sample tick data for testing"""
        np.random.seed(42)
        size = 10000
        
        base_price = 100
        tick_changes = np.random.normal(0, 0.01, size)
        prices = base_price + np.cumsum(tick_changes)
        
        return {
            'prices': prices,
            'volumes': np.random.randint(1, 1000, size),
            'timestamps': [time.time() + i * 0.001 for i in range(size)]  # 1ms intervals
        }

class TestRealtimeStreaming:
    """Test real-time streaming capabilities"""
    
    def test_basic_streaming_functionality(self, temp_dir):
        """Test basic streaming functionality"""
        streamer = RealtimeDataStreamer(
            buffer_size=1000,
            max_latency_us=1000,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST
        )
        
        # Start streaming
        streamer.start()
        
        # Send test messages
        test_data = [{'price': 100.0 + i, 'volume': 1000} for i in range(100)]
        
        results = []
        def data_callback(message):
            results.append(message.data)
        
        streamer.register_data_callback(data_callback)
        
        # Send messages
        for i, data in enumerate(test_data):
            success = streamer.send_message(data, source='test', message_type='tick')
            assert success, f"Failed to send message {i}"
        
        # Wait for processing
        time.sleep(0.1)
        
        # Check results
        assert len(results) > 0, "No messages processed"
        
        # Get metrics
        metrics = streamer.get_metrics()
        assert metrics.throughput_msgs_per_sec > 0
        assert metrics.avg_latency_us < 1000  # Should be under 1ms
        
        streamer.stop()
    
    def test_backpressure_handling(self):
        """Test backpressure handling strategies"""
        streamer = RealtimeDataStreamer(
            buffer_size=10,  # Small buffer to trigger backpressure
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST
        )
        
        streamer.start()
        
        # Send more messages than buffer can hold
        for i in range(20):
            streamer.send_message({'data': i}, source='test')
        
        # Check that backpressure events were recorded
        metrics = streamer.get_metrics()
        assert metrics.backpressure_events > 0
        
        streamer.stop()
    
    def test_market_data_streaming(self):
        """Test specialized market data streaming"""
        streamer = MarketDataStreamer()
        
        # Add filters
        streamer.add_symbol_filter('AAPL')
        streamer.add_data_type_filter('tick')
        
        # Add validators
        streamer.add_price_validator(lambda price: 0 < price < 1000)
        
        streamer.start()
        
        # Send valid tick data
        success = streamer.send_tick_data('AAPL', 150.0, 100)
        assert success
        
        # Send invalid tick data (filtered out)
        success = streamer.send_tick_data('GOOGL', 2000.0, 100)  # Wrong symbol
        assert success  # Message sent but should be filtered
        
        streamer.stop()
    
    def test_streaming_performance(self):
        """Test streaming performance under load"""
        streamer = RealtimeDataStreamer(
            buffer_size=10000,
            max_latency_us=500  # 500 microseconds
        )
        
        streamer.start()
        
        # Send high-frequency messages
        start_time = time.time()
        message_count = 10000
        
        for i in range(message_count):
            streamer.send_message({'tick': i}, source='perf_test')
        
        # Wait for processing
        time.sleep(0.5)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check throughput
        metrics = streamer.get_metrics()
        throughput = message_count / processing_time
        
        assert throughput > 1000, f"Throughput too low: {throughput} msg/s"
        assert metrics.avg_latency_us < 1000, f"Latency too high: {metrics.avg_latency_us}us"
        
        streamer.stop()

class TestDataValidation:
    """Test data validation capabilities"""
    
    def test_basic_validation(self):
        """Test basic data validation functionality"""
        validator = RealtimeDataValidator()
        
        # Add validation rules
        from ..validation.realtime_validator import NotNullValidationRule, RangeValidationRule
        validator.add_field_rule('price', NotNullValidationRule('price_not_null'))
        validator.add_field_rule('price', RangeValidationRule('price_range', 0, 1000))
        
        # Test valid data
        valid_record = {'price': 100.0, 'volume': 1000}
        issues = validator.validate_record(valid_record)
        assert len(issues) == 0, f"Valid record failed validation: {issues}"
        
        # Test invalid data
        invalid_record = {'price': -50.0, 'volume': 1000}  # Negative price
        issues = validator.validate_record(invalid_record)
        assert len(issues) > 0, "Invalid record passed validation"
        
        # Test null data
        null_record = {'price': None, 'volume': 1000}
        issues = validator.validate_record(null_record)
        assert len(issues) > 0, "Null record passed validation"
    
    def test_market_data_validation(self):
        """Test market data specific validation"""
        validator = MarketDataValidator()
        
        # Test valid market data
        valid_tick = {'symbol': 'AAPL', 'price': 150.0, 'volume': 100, 'timestamp': time.time()}
        issues = validator.validate_record(valid_tick)
        assert len(issues) == 0, f"Valid tick failed validation: {issues}"
        
        # Test invalid market data
        invalid_tick = {'symbol': 'AAPL', 'price': -150.0, 'volume': 100, 'timestamp': time.time()}
        issues = validator.validate_record(invalid_tick)
        assert len(issues) > 0, "Invalid tick passed validation"
    
    def test_dataframe_validation(self, sample_ohlcv_data):
        """Test DataFrame validation"""
        validator = RealtimeDataValidator()
        
        # Test clean data
        report = validator.validate_dataframe(sample_ohlcv_data)
        assert report.overall_score > 90, f"Clean data quality score too low: {report.overall_score}"
        
        # Test data with issues
        dirty_data = sample_ohlcv_data.copy()
        dirty_data.loc[0:10, 'close'] = np.nan  # Add null values
        dirty_data.loc[20:30, 'volume'] = -1  # Add invalid values
        
        report = validator.validate_dataframe(dirty_data)
        assert report.overall_score < 95, "Dirty data quality score too high"
        assert len(report.issues) > 0, "No issues detected in dirty data"
    
    def test_validation_performance(self):
        """Test validation performance"""
        validator = RealtimeDataValidator(max_workers=4, enable_parallel=True)
        
        # Add multiple validation rules
        from ..validation.realtime_validator import NotNullValidationRule, RangeValidationRule
        validator.add_field_rule('price', NotNullValidationRule('price_not_null'))
        validator.add_field_rule('price', RangeValidationRule('price_range', 0, 1000))
        validator.add_field_rule('volume', NotNullValidationRule('volume_not_null'))
        validator.add_field_rule('volume', RangeValidationRule('volume_range', 1, 1000000))
        
        # Generate test data
        test_records = [{'price': 100.0 + i, 'volume': 1000 + i} for i in range(1000)]
        
        # Measure performance
        start_time = time.time()
        
        for record in test_records:
            validator.validate_record(record)
        
        end_time = time.time()
        
        # Check performance
        metrics = validator.get_metrics()
        total_time = end_time - start_time
        throughput = len(test_records) / total_time
        
        assert throughput > 1000, f"Validation throughput too low: {throughput} records/s"
        assert metrics.avg_validation_time_us < 1000, f"Average validation time too high: {metrics.avg_validation_time_us}us"

class TestOptimizedTransformations:
    """Test optimized transformation capabilities"""
    
    def test_single_indicator_calculation(self, sample_ohlcv_data):
        """Test single indicator calculation"""
        transformer = OptimizedTransformer()
        
        # Test SMA calculation
        close_prices = sample_ohlcv_data['close'].values
        sma_result = transformer.transform_single(close_prices, 'sma', window=20)
        
        assert len(sma_result) == len(close_prices)
        assert not np.isnan(sma_result[-1])  # Last value should be valid
        
        # Test EMA calculation
        ema_result = transformer.transform_single(close_prices, 'ema', alpha=0.1)
        assert len(ema_result) == len(close_prices)
        assert not np.isnan(ema_result[-1])
    
    def test_multiple_indicators(self, sample_ohlcv_data):
        """Test multiple indicator calculations"""
        transformer = OptimizedTransformer(enable_parallel=True)
        
        # Convert to OHLCV format
        ohlcv_data = {
            'open': sample_ohlcv_data['open'].values,
            'high': sample_ohlcv_data['high'].values,
            'low': sample_ohlcv_data['low'].values,
            'close': sample_ohlcv_data['close'].values,
            'volume': sample_ohlcv_data['volume'].values
        }
        
        # Calculate multiple indicators
        indicators = ['sma', 'ema', 'rsi', 'bollinger_bands', 'macd']
        results = transformer.transform_ohlcv(ohlcv_data, indicators)
        
        assert len(results) >= len(indicators)
        assert 'sma_20' in results
        assert 'ema_20' in results
        assert 'rsi_14' in results
    
    def test_streaming_transformer(self):
        """Test streaming transformer"""
        transformer = StreamingTransformer(buffer_size=100)
        
        # Add tick data
        results = []
        for i in range(50):
            price = 100.0 + np.random.normal(0, 1)
            volume = 1000 + np.random.randint(-100, 100)
            
            result = transformer.add_tick(price, volume, time.time())
            results.append(result)
        
        # Check that indicators are calculated
        latest_result = results[-1]
        assert 'sma_20' in latest_result
        assert 'ema_20' in latest_result
        
        # Check performance
        stats = transformer.get_performance_stats()
        assert stats['avg_latency_us'] < 1000  # Should be under 1ms
    
    def test_transformation_performance(self, sample_ohlcv_data):
        """Test transformation performance"""
        transformer = OptimizedTransformer(enable_parallel=True, max_workers=4)
        
        # Prepare data
        ohlcv_data = {
            'close': sample_ohlcv_data['close'].values,
            'high': sample_ohlcv_data['high'].values,
            'low': sample_ohlcv_data['low'].values,
            'volume': sample_ohlcv_data['volume'].values
        }
        
        # Benchmark multiple calculations
        start_time = time.time()
        
        for _ in range(100):
            transformer.transform_ohlcv(ohlcv_data, ['sma', 'ema', 'rsi'])
        
        end_time = time.time()
        
        # Check performance
        metrics = transformer.get_metrics()
        total_time = end_time - start_time
        throughput = 100 / total_time
        
        assert throughput > 10, f"Transformation throughput too low: {throughput} ops/s"
        assert metrics.avg_latency_us < 10000, f"Average latency too high: {metrics.avg_latency_us}us"

class TestOptimizedStorage:
    """Test optimized storage capabilities"""
    
    def test_compressed_storage(self, temp_dir, sample_ohlcv_data):
        """Test compressed data storage"""
        storage = CompressedDataStore(
            storage_path=temp_dir,
            compression_type=CompressionType.ZSTD,
            compression_level=3
        )
        
        # Store data
        test_data = sample_ohlcv_data.to_dict()
        block_id = storage.store_data('test_data', test_data)
        
        assert block_id is not None
        assert 'test_data' in storage.list_keys()
        
        # Retrieve data
        retrieved_data, metadata = storage.retrieve_data('test_data')
        assert retrieved_data is not None
        assert metadata is not None
        
        # Check compression effectiveness
        metrics = storage.get_metrics()
        assert metrics.compression_ratio < 1.0  # Should be compressed
    
    def test_file_storage_formats(self, temp_dir, sample_ohlcv_data):
        """Test different file storage formats"""
        storage = OptimizedFileStorage(temp_dir)
        
        # Test Parquet storage
        parquet_path = storage.store_dataframe_parquet(sample_ohlcv_data, 'test_parquet')
        assert Path(parquet_path).exists()
        
        loaded_parquet = storage.load_dataframe_parquet('test_parquet')
        pd.testing.assert_frame_equal(sample_ohlcv_data, loaded_parquet)
        
        # Test Feather storage
        feather_path = storage.store_dataframe_feather(sample_ohlcv_data, 'test_feather')
        assert Path(feather_path).exists()
        
        loaded_feather = storage.load_dataframe_feather('test_feather')
        pd.testing.assert_frame_equal(sample_ohlcv_data, loaded_feather)
        
        # Test numpy storage
        test_array = sample_ohlcv_data['close'].values
        numpy_path = storage.store_array_numpy(test_array, 'test_numpy')
        assert Path(numpy_path).exists()
        
        loaded_numpy = storage.load_array_numpy('test_numpy')
        np.testing.assert_array_equal(test_array, loaded_numpy)
    
    def test_storage_performance(self, temp_dir, sample_ohlcv_data):
        """Test storage performance"""
        storage = CompressedDataStore(
            storage_path=temp_dir,
            compression_type=CompressionType.LZ4  # Fast compression
        )
        
        # Measure write performance
        start_time = time.time()
        
        for i in range(100):
            test_data = {'iteration': i, 'data': sample_ohlcv_data.to_dict()}
            storage.store_data(f'test_{i}', test_data)
        
        write_time = time.time() - start_time
        
        # Measure read performance
        start_time = time.time()
        
        for i in range(100):
            storage.retrieve_data(f'test_{i}')
        
        read_time = time.time() - start_time
        
        # Check performance
        metrics = storage.get_metrics()
        
        write_throughput = 100 / write_time
        read_throughput = 100 / read_time
        
        assert write_throughput > 10, f"Write throughput too low: {write_throughput} ops/s"
        assert read_throughput > 50, f"Read throughput too low: {read_throughput} ops/s"
        assert metrics.avg_write_latency_us < 100000, f"Write latency too high: {metrics.avg_write_latency_us}us"
        assert metrics.avg_read_latency_us < 50000, f"Read latency too high: {metrics.avg_read_latency_us}us"

class TestParallelIndicators:
    """Test parallel indicator calculations"""
    
    def test_single_indicator_parallel(self, sample_ohlcv_data):
        """Test single indicator with parallel processing"""
        calculator = ParallelIndicatorCalculator(max_workers=4)
        
        # Test SMA calculation
        result = calculator.calculate_single_indicator(
            sample_ohlcv_data['close'].values, 'sma_20'
        )
        
        assert result.name == 'sma_20'
        assert len(result.values) == len(sample_ohlcv_data)
        assert result.calculation_time_us > 0
    
    def test_multiple_indicators_parallel(self, sample_ohlcv_data):
        """Test multiple indicators with parallel processing"""
        calculator = ParallelIndicatorCalculator(max_workers=4)
        
        # Prepare OHLCV data
        ohlcv_data = {
            'open': sample_ohlcv_data['open'].values,
            'high': sample_ohlcv_data['high'].values,
            'low': sample_ohlcv_data['low'].values,
            'close': sample_ohlcv_data['close'].values,
            'volume': sample_ohlcv_data['volume'].values
        }
        
        # Calculate multiple indicators
        indicators = ['sma_20', 'ema_20', 'rsi_14', 'bollinger_bands', 'macd']
        results = calculator.calculate_multiple_indicators(
            ohlcv_data, indicators, ProcessingMode.MULTI_THREAD
        )
        
        assert len(results) == len(indicators)
        for indicator in indicators:
            assert indicator in results
            assert results[indicator].calculation_time_us > 0
    
    def test_indicator_caching(self, sample_ohlcv_data):
        """Test indicator result caching"""
        calculator = ParallelIndicatorCalculator(enable_cache=True)
        
        # Calculate indicator twice
        data = sample_ohlcv_data['close'].values
        
        # First calculation
        result1 = calculator.calculate_single_indicator(data, 'sma_20')
        assert not result1.cache_hit
        
        # Second calculation (should be cached)
        result2 = calculator.calculate_single_indicator(data, 'sma_20')
        assert result2.cache_hit
        
        # Check that results are identical
        np.testing.assert_array_equal(result1.values, result2.values)
    
    def test_parallel_performance(self, sample_ohlcv_data):
        """Test parallel calculation performance"""
        calculator = ParallelIndicatorCalculator(max_workers=4)
        
        # Prepare data
        ohlcv_data = {
            'close': sample_ohlcv_data['close'].values,
            'high': sample_ohlcv_data['high'].values,
            'low': sample_ohlcv_data['low'].values,
            'volume': sample_ohlcv_data['volume'].values
        }
        
        # Benchmark parallel vs sequential
        indicators = ['sma_20', 'ema_20', 'rsi_14', 'bollinger_bands']
        
        # Sequential
        start_time = time.time()
        seq_results = calculator.calculate_multiple_indicators(
            ohlcv_data, indicators, ProcessingMode.SINGLE_THREAD
        )
        seq_time = time.time() - start_time
        
        # Parallel
        start_time = time.time()
        par_results = calculator.calculate_multiple_indicators(
            ohlcv_data, indicators, ProcessingMode.MULTI_THREAD
        )
        par_time = time.time() - start_time
        
        # Check performance improvement
        speedup = seq_time / par_time
        assert speedup > 1.0, f"No performance improvement: {speedup}x"
        
        # Check correctness
        for indicator in indicators:
            np.testing.assert_array_almost_equal(
                seq_results[indicator].values, 
                par_results[indicator].values
            )

class TestIntelligentCaching:
    """Test intelligent caching system"""
    
    def test_basic_cache_operations(self, temp_dir):
        """Test basic cache operations"""
        cache = IntelligentCache(
            max_size=100,
            max_memory_mb=10,
            cache_strategy=CacheStrategy.LRU,
            persistence_path=temp_dir
        )
        
        # Test put and get
        cache.put('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Test cache hit/miss
        assert cache.contains('key1')
        assert not cache.contains('nonexistent')
        
        # Test deletion
        cache.delete('key1')
        assert not cache.contains('key1')
    
    def test_cache_eviction(self):
        """Test cache eviction strategies"""
        cache = IntelligentCache(
            max_size=3,  # Small cache to trigger eviction
            cache_strategy=CacheStrategy.LRU
        )
        
        # Fill cache
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')
        
        # Add one more item to trigger eviction
        cache.put('key4', 'value4')
        
        # Check that oldest item was evicted
        assert not cache.contains('key1')
        assert cache.contains('key4')
    
    def test_cache_performance(self):
        """Test cache performance"""
        cache = IntelligentCache(
            max_size=10000,
            cache_strategy=CacheStrategy.ADAPTIVE
        )
        
        # Measure put performance
        start_time = time.time()
        
        for i in range(1000):
            cache.put(f'key_{i}', f'value_{i}')
        
        put_time = time.time() - start_time
        
        # Measure get performance
        start_time = time.time()
        
        for i in range(1000):
            cache.get(f'key_{i}')
        
        get_time = time.time() - start_time
        
        # Check performance
        metrics = cache.get_metrics()
        
        put_throughput = 1000 / put_time
        get_throughput = 1000 / get_time
        
        assert put_throughput > 1000, f"Put throughput too low: {put_throughput} ops/s"
        assert get_throughput > 10000, f"Get throughput too low: {get_throughput} ops/s"
        assert metrics.avg_response_time_us < 1000, f"Average response time too high: {metrics.avg_response_time_us}us"
    
    def test_cache_persistence(self, temp_dir):
        """Test cache persistence"""
        # Create cache with persistence
        cache1 = IntelligentCache(
            max_size=100,
            persistence_path=temp_dir,
            enable_persistence=True
        )
        
        # Add some data
        cache1.put('persistent_key', 'persistent_value')
        
        # Close cache
        cache1._cleanup()
        
        # Create new cache instance
        cache2 = IntelligentCache(
            max_size=100,
            persistence_path=temp_dir,
            enable_persistence=True
        )
        
        # Check that data persisted
        assert cache2.contains('persistent_key')
        assert cache2.get('persistent_key') == 'persistent_value'

class TestIncrementalCalculations:
    """Test incremental calculation capabilities"""
    
    def test_incremental_sma(self):
        """Test incremental SMA calculation"""
        from ..incremental.incremental_calculator import IncrementalSMA, UpdateEvent, UpdateType
        
        sma = IncrementalSMA(window_size=5)
        
        # Add data points
        values = [10, 20, 30, 40, 50]
        for value in values:
            event = UpdateEvent(UpdateType.APPEND, new_value=value)
            result = sma.update(event)
        
        # Check result
        expected_sma = sum(values) / len(values)
        assert abs(result - expected_sma) < 0.001
    
    def test_incremental_ema(self):
        """Test incremental EMA calculation"""
        from ..incremental.incremental_calculator import IncrementalEMA, UpdateEvent, UpdateType
        
        ema = IncrementalEMA(window_size=5)
        
        # Add data points
        values = [10, 20, 30, 40, 50]
        for value in values:
            event = UpdateEvent(UpdateType.APPEND, new_value=value)
            result = ema.update(event)
        
        # Check that result is reasonable
        assert result > 0
        assert result != sum(values) / len(values)  # Should be different from SMA
    
    def test_streaming_processor(self):
        """Test streaming data processor"""
        processor = StreamingDataProcessor({
            'sma_20': {'type': 'sma', 'window_size': 20},
            'ema_20': {'type': 'ema', 'window_size': 20},
            'rsi_14': {'type': 'rsi', 'window_size': 14}
        })
        
        # Process tick data
        for i in range(50):
            price = 100.0 + np.random.normal(0, 1)
            volume = 1000 + np.random.randint(-100, 100)
            
            results = processor.process_tick(price, volume)
        
        # Check results
        assert 'sma_20' in results
        assert 'ema_20' in results
        assert 'rsi_14' in results
        
        # Check performance
        stats = processor.get_performance_stats()
        assert stats['avg_processing_time_us'] < 1000
    
    def test_incremental_performance(self):
        """Test incremental calculation performance"""
        processor = StreamingDataProcessor({
            'sma_20': {'type': 'sma', 'window_size': 20},
            'ema_20': {'type': 'ema', 'window_size': 20}
        })
        
        # Process many ticks
        start_time = time.time()
        
        for i in range(10000):
            price = 100.0 + np.random.normal(0, 1)
            processor.process_tick(price, 1000)
        
        total_time = time.time() - start_time
        throughput = 10000 / total_time
        
        # Check performance
        assert throughput > 1000, f"Incremental calculation throughput too low: {throughput} ticks/s"
        
        stats = processor.get_performance_stats()
        assert stats['avg_processing_time_us'] < 1000

class TestQualityMonitoring:
    """Test data quality monitoring"""
    
    def test_basic_quality_monitoring(self):
        """Test basic quality monitoring"""
        monitor = DataQualityMonitor(enable_persistence=False)
        
        # Monitor good data
        for i in range(100):
            alerts = monitor.monitor_data_point('price', 100.0 + i, 'numeric')
            assert len(alerts) == 0  # No alerts for good data
        
        # Monitor bad data
        alerts = monitor.monitor_data_point('price', None, 'numeric')
        assert len(alerts) > 0  # Should detect null value
        
        alerts = monitor.monitor_data_point('price', float('inf'), 'numeric')
        assert len(alerts) > 0  # Should detect infinite value
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        monitor = DataQualityMonitor(enable_persistence=False)
        
        # Add normal data
        for i in range(1000):
            value = 100.0 + np.random.normal(0, 5)
            monitor.monitor_data_point('price', value, 'numeric')
        
        # Add anomalous data
        alerts = monitor.monitor_data_point('price', 500.0, 'numeric')  # Obvious outlier
        assert len(alerts) > 0  # Should detect anomaly
    
    def test_dataframe_quality_monitoring(self, sample_ohlcv_data):
        """Test DataFrame quality monitoring"""
        monitor = DataQualityMonitor(enable_persistence=False)
        
        # Monitor clean data
        report = monitor.monitor_dataframe(sample_ohlcv_data)
        assert report.overall_score > 90
        
        # Monitor dirty data
        dirty_data = sample_ohlcv_data.copy()
        dirty_data.loc[0:10, 'close'] = np.nan
        dirty_data.loc[20:30, 'volume'] = -1
        
        report = monitor.monitor_dataframe(dirty_data)
        assert report.overall_score < 95
        assert len(report.anomalies) > 0
    
    def test_quality_monitoring_performance(self):
        """Test quality monitoring performance"""
        monitor = DataQualityMonitor(enable_persistence=False)
        
        # Generate test data
        test_data = [(f'field_{i}', 100.0 + i, 'numeric') for i in range(10000)]
        
        # Measure performance
        start_time = time.time()
        
        for field, value, data_type in test_data:
            monitor.monitor_data_point(field, value, data_type)
        
        total_time = time.time() - start_time
        throughput = len(test_data) / total_time
        
        # Check performance
        assert throughput > 1000, f"Quality monitoring throughput too low: {throughput} records/s"
        
        summary = monitor.get_quality_summary()
        assert summary['avg_processing_time_us'] < 1000

class TestDataLineage:
    """Test data lineage tracking"""
    
    def test_basic_lineage_tracking(self, temp_dir):
        """Test basic lineage tracking"""
        from ..lineage.lineage_tracker import create_data_asset, create_transformation
        
        tracker = DataLineageTracker(persistence_path=temp_dir)
        
        # Create assets
        source_asset = create_data_asset('source_data', 'dataset')
        target_asset = create_data_asset('processed_data', 'dataset')
        
        # Register assets
        tracker.register_asset(source_asset)
        tracker.register_asset(target_asset)
        
        # Create transformation
        transformation = create_transformation(
            'data_processing',
            'filter',
            [source_asset.asset_id],
            [target_asset.asset_id]
        )
        
        # Register transformation
        tracker.register_transformation(transformation)
        
        # Track data flow
        tracker.track_data_flow(
            source_asset.asset_id,
            target_asset.asset_id,
            transformation.transformation_id
        )
        
        # Check lineage
        lineage = tracker.get_asset_lineage(target_asset.asset_id)
        assert len(lineage['upstream']) > 0
        assert source_asset.asset_id in [asset['id'] for asset in lineage['upstream']]
    
    def test_impact_analysis(self, temp_dir):
        """Test impact analysis"""
        from ..lineage.lineage_tracker import create_data_asset, create_transformation
        
        tracker = DataLineageTracker(persistence_path=temp_dir)
        
        # Create a chain of assets
        asset1 = create_data_asset('asset1', 'dataset')
        asset2 = create_data_asset('asset2', 'dataset')
        asset3 = create_data_asset('asset3', 'dataset')
        
        for asset in [asset1, asset2, asset3]:
            tracker.register_asset(asset)
        
        # Create transformations
        transform1 = create_transformation('transform1', 'filter', [asset1.asset_id], [asset2.asset_id])
        transform2 = create_transformation('transform2', 'aggregate', [asset2.asset_id], [asset3.asset_id])
        
        tracker.register_transformation(transform1)
        tracker.register_transformation(transform2)
        
        # Track flows
        tracker.track_data_flow(asset1.asset_id, asset2.asset_id, transform1.transformation_id)
        tracker.track_data_flow(asset2.asset_id, asset3.asset_id, transform2.transformation_id)
        
        # Analyze impact
        impact = tracker.get_impact_analysis(asset1.asset_id)
        assert impact['total_impacted_assets'] >= 2
        assert len(impact['impacted_assets']) >= 2
    
    def test_validation_rules(self, temp_dir):
        """Test validation rules"""
        from ..lineage.lineage_tracker import create_data_asset, create_validation_rule
        
        tracker = DataLineageTracker(persistence_path=temp_dir)
        
        # Create asset
        asset = create_data_asset('test_asset', 'dataset', schema={'field1': 'string', 'field2': 'number'})
        tracker.register_asset(asset)
        
        # Create validation rule
        rule = create_validation_rule(
            'schema_validation',
            'schema',
            [asset.asset_id],
            {'required_fields': ['field1', 'field2']}
        )
        
        # Add rule
        tracker.add_validation_rule(rule)
        
        # Validate asset
        results = tracker.validate_asset(asset.asset_id)
        assert len(results) > 0
        assert results[0].passed  # Should pass with correct schema
    
    def test_lineage_performance(self, temp_dir):
        """Test lineage tracking performance"""
        tracker = DataLineageTracker(persistence_path=temp_dir)
        
        # Create many assets
        start_time = time.time()
        
        for i in range(1000):
            from ..lineage.lineage_tracker import create_data_asset
            asset = create_data_asset(f'asset_{i}', 'dataset')
            tracker.register_asset(asset)
        
        registration_time = time.time() - start_time
        
        # Check performance
        assert registration_time < 10.0, f"Asset registration too slow: {registration_time}s"
        
        summary = tracker.get_lineage_summary()
        assert summary['total_assets'] == 1000
        assert summary['avg_event_processing_time_us'] < 1000

class TestIntegrationScenarios:
    """Test integration scenarios combining multiple optimizations"""
    
    def test_end_to_end_pipeline(self, temp_dir, sample_tick_data):
        """Test complete end-to-end pipeline"""
        # Setup components
        streamer = RealtimeDataStreamer(buffer_size=10000)
        validator = RealtimeDataValidator()
        processor = StreamingDataProcessor({
            'sma_20': {'type': 'sma', 'window_size': 20},
            'ema_20': {'type': 'ema', 'window_size': 20}
        })
        cache = IntelligentCache(max_size=1000)
        storage = CompressedDataStore(temp_dir)
        
        # Setup validation rules
        from ..validation.realtime_validator import RangeValidationRule
        validator.add_field_rule('price', RangeValidationRule('price_range', 0, 1000))
        
        # Process data pipeline
        results = []
        
        def process_tick(message):
            # Validate
            data = message.data
            issues = validator.validate_record(data)
            
            if len(issues) == 0:  # Only process valid data
                # Calculate indicators
                indicators = processor.process_tick(data['price'], data.get('volume', 0))
                
                # Cache results
                cache.put(f"tick_{message.timestamp}", indicators)
                
                # Store to persistent storage
                storage.store_data(f"tick_{message.timestamp}", {
                    'data': data,
                    'indicators': indicators
                })
                
                results.append(indicators)
        
        # Start streaming
        streamer.start()
        streamer.register_data_callback(process_tick)
        
        # Send test data
        for i in range(100):
            price = sample_tick_data['prices'][i]
            volume = sample_tick_data['volumes'][i]
            
            streamer.send_message({
                'price': price,
                'volume': volume,
                'timestamp': time.time()
            })
        
        # Wait for processing
        time.sleep(0.5)
        
        # Check results
        assert len(results) > 0
        assert 'sma_20' in results[-1]
        assert 'ema_20' in results[-1]
        
        # Check component performance
        stream_metrics = streamer.get_metrics()
        validator_metrics = validator.get_metrics()
        processor_stats = processor.get_performance_stats()
        cache_metrics = cache.get_metrics()
        storage_metrics = storage.get_metrics()
        
        assert stream_metrics.avg_latency_us < 1000
        assert validator_metrics.avg_validation_time_us < 1000
        assert processor_stats['avg_processing_time_us'] < 1000
        assert cache_metrics.avg_response_time_us < 1000
        assert storage_metrics.avg_write_latency_us < 100000
        
        streamer.stop()
    
    def test_stress_testing(self, temp_dir):
        """Test system under stress conditions"""
        # Create high-load scenario
        streamer = RealtimeDataStreamer(
            buffer_size=50000,
            max_latency_us=1000,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST
        )
        
        processor = StreamingDataProcessor({
            'sma_5': {'type': 'sma', 'window_size': 5},
            'sma_20': {'type': 'sma', 'window_size': 20},
            'ema_5': {'type': 'ema', 'window_size': 5},
            'ema_20': {'type': 'ema', 'window_size': 20}
        })
        
        # Performance counters
        processed_count = 0
        error_count = 0
        
        def stress_processor(message):
            nonlocal processed_count, error_count
            try:
                data = message.data
                processor.process_tick(data['price'], data.get('volume', 0))
                processed_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"Processing error: {str(e)}")
        
        streamer.start()
        streamer.register_data_callback(stress_processor)
        
        # Send high-frequency data
        message_count = 100000
        start_time = time.time()
        
        for i in range(message_count):
            price = 100.0 + np.random.normal(0, 5)
            volume = np.random.randint(100, 1000)
            
            streamer.send_message({
                'price': price,
                'volume': volume,
                'timestamp': time.time()
            })
        
        # Wait for processing
        time.sleep(2.0)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check stress test results
        throughput = processed_count / total_time
        error_rate = error_count / message_count * 100
        
        assert throughput > 10000, f"Stress test throughput too low: {throughput} msg/s"
        assert error_rate < 5.0, f"Stress test error rate too high: {error_rate}%"
        
        # Check component metrics
        stream_metrics = streamer.get_metrics()
        processor_stats = processor.get_performance_stats()
        
        assert stream_metrics.avg_latency_us < 2000  # Allow higher latency under stress
        assert processor_stats['avg_processing_time_us'] < 2000
        
        streamer.stop()
    
    def test_memory_usage(self, temp_dir):
        """Test memory usage under various conditions"""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create components
        streamer = RealtimeDataStreamer(buffer_size=10000)
        cache = IntelligentCache(max_size=10000, max_memory_mb=100)
        storage = CompressedDataStore(temp_dir)
        
        # Fill with data
        for i in range(10000):
            data = {'iteration': i, 'data': np.random.randn(100).tolist()}
            cache.put(f'key_{i}', data)
            if i % 100 == 0:
                storage.store_data(f'batch_{i}', data)
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500, f"Memory usage too high: {memory_increase}MB"
        
        # Check component-specific memory usage
        cache_metrics = cache.get_metrics()
        storage_metrics = storage.get_metrics()
        
        assert cache_metrics.memory_usage_mb < 120  # Should respect limit
        assert storage_metrics.storage_utilization_mb > 0

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def test_latency_benchmarks(self, sample_tick_data):
        """Test latency benchmarks for all components"""
        results = {}
        
        # Streaming latency
        streamer = RealtimeDataStreamer(buffer_size=10000)
        streamer.start()
        
        processed_count = 0
        def count_processor(message):
            nonlocal processed_count
            processed_count += 1
        
        streamer.register_data_callback(count_processor)
        
        start_time = time.time()
        for i in range(1000):
            streamer.send_message({'price': sample_tick_data['prices'][i % len(sample_tick_data['prices'])]})
        
        while processed_count < 1000:
            time.sleep(0.001)
        
        end_time = time.time()
        
        stream_metrics = streamer.get_metrics()
        results['streaming_latency_us'] = stream_metrics.avg_latency_us
        results['streaming_throughput'] = stream_metrics.throughput_msgs_per_sec
        
        streamer.stop()
        
        # Validation latency
        validator = RealtimeDataValidator()
        start_time = time.time_ns()
        
        for i in range(1000):
            validator.validate_record({'price': sample_tick_data['prices'][i % len(sample_tick_data['prices'])]})
        
        end_time = time.time_ns()
        validation_time_us = (end_time - start_time) / 1000 / 1000  # per record
        results['validation_latency_us'] = validation_time_us
        
        # Indicator calculation latency
        calculator = ParallelIndicatorCalculator()
        start_time = time.time_ns()
        
        for i in range(100):
            calculator.calculate_single_indicator(sample_tick_data['prices'][:100], 'sma_20')
        
        end_time = time.time_ns()
        indicator_time_us = (end_time - start_time) / 1000 / 100  # per calculation
        results['indicator_latency_us'] = indicator_time_us
        
        # Cache latency
        cache = IntelligentCache(max_size=10000)
        start_time = time.time_ns()
        
        for i in range(1000):
            cache.put(f'key_{i}', f'value_{i}')
        
        end_time = time.time_ns()
        cache_put_time_us = (end_time - start_time) / 1000 / 1000  # per operation
        
        start_time = time.time_ns()
        
        for i in range(1000):
            cache.get(f'key_{i}')
        
        end_time = time.time_ns()
        cache_get_time_us = (end_time - start_time) / 1000 / 1000  # per operation
        
        results['cache_put_latency_us'] = cache_put_time_us
        results['cache_get_latency_us'] = cache_get_time_us
        
        # Print benchmark results
        print("\n=== PERFORMANCE BENCHMARK RESULTS ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}")
        
        # Assert performance requirements
        assert results['streaming_latency_us'] < 1000, "Streaming latency too high"
        assert results['validation_latency_us'] < 1000, "Validation latency too high"
        assert results['indicator_latency_us'] < 10000, "Indicator calculation latency too high"
        assert results['cache_put_latency_us'] < 100, "Cache put latency too high"
        assert results['cache_get_latency_us'] < 50, "Cache get latency too high"
        
        return results
    
    def test_throughput_benchmarks(self, sample_ohlcv_data):
        """Test throughput benchmarks for all components"""
        results = {}
        
        # Streaming throughput (tested above)
        
        # Validation throughput
        validator = RealtimeDataValidator(max_workers=4)
        test_records = [{'price': 100.0 + i} for i in range(10000)]
        
        start_time = time.time()
        for record in test_records:
            validator.validate_record(record)
        end_time = time.time()
        
        validation_throughput = len(test_records) / (end_time - start_time)
        results['validation_throughput'] = validation_throughput
        
        # Indicator calculation throughput
        calculator = ParallelIndicatorCalculator(max_workers=4)
        ohlcv_data = {
            'close': sample_ohlcv_data['close'].values,
            'high': sample_ohlcv_data['high'].values,
            'low': sample_ohlcv_data['low'].values,
            'volume': sample_ohlcv_data['volume'].values
        }
        
        start_time = time.time()
        for i in range(100):
            calculator.calculate_multiple_indicators(ohlcv_data, ['sma_20', 'ema_20', 'rsi_14'])
        end_time = time.time()
        
        indicator_throughput = 100 / (end_time - start_time)
        results['indicator_throughput'] = indicator_throughput
        
        # Cache throughput
        cache = IntelligentCache(max_size=50000)
        
        start_time = time.time()
        for i in range(10000):
            cache.put(f'key_{i}', f'value_{i}')
        end_time = time.time()
        
        cache_put_throughput = 10000 / (end_time - start_time)
        results['cache_put_throughput'] = cache_put_throughput
        
        # Print benchmark results
        print("\n=== THROUGHPUT BENCHMARK RESULTS ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f} ops/s")
        
        # Assert performance requirements
        assert results['validation_throughput'] > 1000, "Validation throughput too low"
        assert results['indicator_throughput'] > 10, "Indicator throughput too low"
        assert results['cache_put_throughput'] > 10000, "Cache put throughput too low"
        
        return results

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
