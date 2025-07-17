#!/usr/bin/env python3
"""
Comprehensive demonstration of the scalable data pipeline for 5-year datasets
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
import json
from datetime import datetime, timedelta
import random

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.core.data_loader import ScalableDataLoader
from data_pipeline.core.config import DataPipelineConfig
from data_pipeline.streaming.data_streamer import DataStreamer
from data_pipeline.preprocessing.data_processor import DataProcessor, PreprocessingConfig
from data_pipeline.parallel.parallel_processor import ParallelProcessor, ParallelConfig
from data_pipeline.caching.cache_manager import CacheManager, CacheConfig
from data_pipeline.validation.data_validator import DataValidator, ValidationConfig
from data_pipeline.validation.data_validator import create_financial_data_rules
from data_pipeline.performance.performance_monitor import PerformanceMonitor, PerformanceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipelineDemo:
    """
    Comprehensive demonstration of the scalable data pipeline
    """
    
    def __init__(self):
        self.demo_data_dir = Path("/tmp/pipeline_demo_data")
        self.demo_data_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline components
        self.pipeline_config = DataPipelineConfig.for_production()
        self.data_loader = ScalableDataLoader(self.pipeline_config)
        self.data_streamer = DataStreamer(self.pipeline_config)
        self.data_processor = DataProcessor(pipeline_config=self.pipeline_config)
        self.parallel_processor = ParallelProcessor(pipeline_config=self.pipeline_config)
        self.cache_manager = CacheManager()
        self.data_validator = DataValidator(pipeline_config=self.pipeline_config)
        self.performance_monitor = PerformanceMonitor()
        
        # Demo statistics
        self.demo_stats = {
            'files_generated': 0,
            'total_rows': 0,
            'total_size_mb': 0,
            'processing_times': {},
            'validation_results': {},
            'cache_performance': {}
        }
    
    def generate_sample_data(self, num_files: int = 10, rows_per_file: int = 100000) -> List[str]:
        """
        Generate sample high-frequency trading data files
        """
        logger.info(f"Generating {num_files} sample data files with {rows_per_file} rows each")
        
        file_paths = []
        start_date = datetime.now() - timedelta(days=5*365)  # 5 years ago
        
        for file_idx in range(num_files):
            file_path = self.demo_data_dir / f"trading_data_{file_idx:03d}.csv"
            
            # Generate time series data
            timestamps = []
            current_time = start_date + timedelta(days=file_idx * 30)  # Each file = 1 month
            
            for i in range(rows_per_file):
                timestamps.append(current_time + timedelta(minutes=i))
            
            # Generate OHLCV data
            base_price = 100.0 + random.uniform(-20, 20)
            data = []
            
            for i, ts in enumerate(timestamps):
                # Generate realistic price movements
                price_change = random.uniform(-0.02, 0.02)  # ±2% change
                
                if i == 0:
                    open_price = base_price
                else:
                    open_price = data[i-1]['close']
                
                close_price = open_price * (1 + price_change)
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
                volume = random.randint(1000, 100000)
                
                data.append({
                    'timestamp': ts.isoformat(),
                    'open': round(open_price, 2),
                    'high': round(high_price, 2),
                    'low': round(low_price, 2),
                    'close': round(close_price, 2),
                    'volume': volume
                })
            
            # Write to CSV
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            file_paths.append(str(file_path))
            
            # Update statistics
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            self.demo_stats['files_generated'] += 1
            self.demo_stats['total_rows'] += rows_per_file
            self.demo_stats['total_size_mb'] += file_size
            
            logger.info(f"Generated {file_path} ({file_size:.1f} MB)")
        
        logger.info(f"Generated {len(file_paths)} files totaling {self.demo_stats['total_size_mb']:.1f} MB")
        return file_paths
    
    def demo_data_loading(self, file_paths: List[str]):
        """
        Demonstrate efficient data loading
        """
        logger.info("=== Data Loading Demo ===")
        
        start_time = time.time()
        
        # Test chunk loading
        total_chunks = 0
        total_rows = 0
        
        for file_path in file_paths[:3]:  # Test first 3 files
            logger.info(f"Loading chunks from {file_path}")
            
            # Get file metadata
            metadata = self.data_loader.get_file_metadata(file_path)
            logger.info(f"File metadata: {metadata.row_count} rows, {metadata.file_size_mb:.1f} MB")
            
            # Load in chunks
            for chunk in self.data_loader.load_chunks(file_path, chunk_size=5000):
                total_chunks += 1
                total_rows += len(chunk.data)
                
                # Process chunk (simulate work)
                time.sleep(0.001)  # Simulate processing time
        
        elapsed = time.time() - start_time
        throughput = total_rows / elapsed
        
        self.demo_stats['processing_times']['data_loading'] = elapsed
        
        logger.info(f"Loaded {total_chunks} chunks ({total_rows} rows) in {elapsed:.2f}s")
        logger.info(f"Throughput: {throughput:.0f} rows/sec")
        
        # Test performance stats
        stats = self.data_loader.get_performance_stats()
        logger.info(f"Data loader stats: {stats}")
    
    def demo_data_streaming(self, file_paths: List[str]):
        """
        Demonstrate data streaming with minimal memory footprint
        """
        logger.info("=== Data Streaming Demo ===")
        
        start_time = time.time()
        
        # Define transformation function
        def transform_data(df: pd.DataFrame) -> pd.DataFrame:
            """Add some computed columns"""
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100
            df['volume_ma'] = df['volume'].rolling(window=5, min_periods=1).mean()
            return df
        
        # Define filter function
        def filter_high_volume(df: pd.DataFrame) -> bool:
            """Filter for high volume periods"""
            return df['volume'].mean() > 10000
        
        # Stream data with transformation and filtering
        total_chunks = 0
        total_rows = 0
        
        for chunk in self.data_streamer.stream_multiple_files(
            file_paths[:2],  # Test first 2 files
            transform_func=transform_data,
            filter_func=filter_high_volume,
            parallel=True
        ):
            total_chunks += 1
            total_rows += len(chunk.data)
            
            # Cache frequently accessed data
            cache_key = f"chunk_{chunk.chunk_id}_{chunk.file_path}"
            self.cache_manager.put(cache_key, chunk.data)
        
        elapsed = time.time() - start_time
        throughput = total_rows / elapsed
        
        self.demo_stats['processing_times']['data_streaming'] = elapsed
        
        logger.info(f"Streamed {total_chunks} chunks ({total_rows} rows) in {elapsed:.2f}s")
        logger.info(f"Throughput: {throughput:.0f} rows/sec")
        
        # Test streaming stats
        stats = self.data_streamer.get_streaming_stats()
        logger.info(f"Streaming stats: {stats}")
    
    def demo_data_preprocessing(self, file_paths: List[str]):
        """
        Demonstrate data preprocessing pipeline
        """
        logger.info("=== Data Preprocessing Demo ===")
        
        start_time = time.time()
        
        # Configure preprocessing
        preprocessing_config = PreprocessingConfig(
            enable_parallel_processing=True,
            remove_duplicates=True,
            handle_missing_values="interpolate",
            outlier_detection=True,
            normalize_features=True,
            create_technical_indicators=True,
            create_lag_features=True,
            lag_periods=[1, 5, 10]
        )
        
        processor = DataProcessor(preprocessing_config, self.pipeline_config)
        
        # Process files
        total_chunks = 0
        total_rows = 0
        
        preprocessing_steps = [
            "clean_data",
            "handle_missing_values",
            "normalize_features",
            "engineer_features"
        ]
        
        for chunk in processor.process_multiple_files(
            file_paths[:2],  # Test first 2 files
            preprocessing_steps=preprocessing_steps,
            parallel=True
        ):
            total_chunks += 1
            total_rows += len(chunk.data)
            
            # Show sample of processed data
            if total_chunks == 1:
                logger.info(f"Sample processed data columns: {list(chunk.data.columns)}")
                logger.info(f"Sample processed data shape: {chunk.data.shape}")
        
        elapsed = time.time() - start_time
        throughput = total_rows / elapsed
        
        self.demo_stats['processing_times']['data_preprocessing'] = elapsed
        
        logger.info(f"Preprocessed {total_chunks} chunks ({total_rows} rows) in {elapsed:.2f}s")
        logger.info(f"Throughput: {throughput:.0f} rows/sec")
        
        # Test preprocessing stats
        stats = processor.get_processing_stats()
        logger.info(f"Preprocessing stats: {stats}")
    
    def demo_parallel_processing(self, file_paths: List[str]):
        """
        Demonstrate parallel data processing
        """
        logger.info("=== Parallel Processing Demo ===")
        
        start_time = time.time()
        
        # Define processing function
        def process_chunk(chunk):
            """Process a chunk of data"""
            data = chunk.data.copy()
            
            # Add some computations
            data['price_range'] = data['high'] - data['low']
            data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
            data['volume_price'] = data['volume'] * data['close']
            
            # Simulate some work
            time.sleep(0.01)
            
            return chunk
        
        # Process files in parallel
        total_chunks = 0
        total_rows = 0
        
        for chunk in self.parallel_processor.process_files_parallel(
            file_paths[:2],  # Test first 2 files
            process_chunk
        ):
            total_chunks += 1
            total_rows += len(chunk.data)
        
        elapsed = time.time() - start_time
        throughput = total_rows / elapsed
        
        self.demo_stats['processing_times']['parallel_processing'] = elapsed
        
        logger.info(f"Parallel processed {total_chunks} chunks ({total_rows} rows) in {elapsed:.2f}s")
        logger.info(f"Throughput: {throughput:.0f} rows/sec")
        
        # Test parallel processing stats
        stats = self.parallel_processor.get_stats()
        logger.info(f"Parallel processing stats: {stats}")
    
    def demo_caching(self, file_paths: List[str]):
        """
        Demonstrate caching strategies
        """
        logger.info("=== Caching Demo ===")
        
        start_time = time.time()
        
        # Cache some data
        cache_keys = []
        
        for i, file_path in enumerate(file_paths[:3]):
            for j, chunk in enumerate(self.data_loader.load_chunks(file_path, chunk_size=1000)):
                cache_key = f"file_{i}_chunk_{j}"
                self.cache_manager.put(cache_key, chunk.data)
                cache_keys.append(cache_key)
                
                if len(cache_keys) >= 10:  # Cache first 10 chunks
                    break
        
        # Test cache retrieval
        hits = 0
        misses = 0
        
        for key in cache_keys:
            data = self.cache_manager.get(key)
            if data is not None:
                hits += 1
            else:
                misses += 1
        
        # Test cache statistics
        cache_stats = self.cache_manager.get_stats()
        
        elapsed = time.time() - start_time
        self.demo_stats['processing_times']['caching'] = elapsed
        self.demo_stats['cache_performance'] = cache_stats
        
        logger.info(f"Cached {len(cache_keys)} items in {elapsed:.2f}s")
        logger.info(f"Cache hits: {hits}, misses: {misses}")
        logger.info(f"Cache stats: {cache_stats}")
    
    def demo_data_validation(self, file_paths: List[str]):
        """
        Demonstrate data validation
        """
        logger.info("=== Data Validation Demo ===")
        
        start_time = time.time()
        
        # Add financial data validation rules
        validation_rules = create_financial_data_rules()
        self.data_validator.add_rules(validation_rules)
        
        # Validate files
        all_results = []
        
        for file_path in file_paths[:2]:  # Test first 2 files
            results = self.data_validator.validate_file(file_path)
            all_results.extend(results)
            
            logger.info(f"Validated {file_path}: {len(results)} issues found")
        
        # Get validation summary
        summary = self.data_validator.get_validation_summary()
        
        elapsed = time.time() - start_time
        self.demo_stats['processing_times']['data_validation'] = elapsed
        self.demo_stats['validation_results'] = summary
        
        logger.info(f"Validation completed in {elapsed:.2f}s")
        logger.info(f"Validation summary: {summary}")
        
        # Generate validation report
        report_path = self.data_validator.generate_report()
        logger.info(f"Validation report saved to: {report_path}")
    
    def demo_performance_monitoring(self, file_paths: List[str]):
        """
        Demonstrate performance monitoring
        """
        logger.info("=== Performance Monitoring Demo ===")
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Simulate some work
        time.sleep(2)
        
        # Process some data while monitoring
        for chunk in self.data_loader.load_chunks(file_paths[0], chunk_size=1000):
            # Simulate processing
            _ = chunk.data.describe()
            time.sleep(0.1)
        
        # Wait a bit more
        time.sleep(2)
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
        
        # Get performance summary
        summary = self.performance_monitor.get_performance_summary(duration_seconds=60)
        logger.info(f"Performance summary: {summary}")
        
        # Get alerts
        alerts = self.performance_monitor.get_alerts()
        if alerts:
            logger.info(f"Performance alerts: {alerts}")
        
        # Export metrics
        metrics_path = self.demo_data_dir / "performance_metrics.json"
        self.performance_monitor.export_metrics(str(metrics_path))
        logger.info(f"Performance metrics exported to: {metrics_path}")
    
    def run_comprehensive_demo(self):
        """
        Run comprehensive demonstration of all pipeline components
        """
        logger.info("=" * 60)
        logger.info("SCALABLE DATA PIPELINE DEMONSTRATION")
        logger.info("5-Year High-Frequency Trading Data Processing")
        logger.info("=" * 60)
        
        try:
            # Generate sample data
            file_paths = self.generate_sample_data(num_files=5, rows_per_file=50000)
            
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            # Run all demos
            self.demo_data_loading(file_paths)
            self.demo_data_streaming(file_paths)
            self.demo_data_preprocessing(file_paths)
            self.demo_parallel_processing(file_paths)
            self.demo_caching(file_paths)
            self.demo_data_validation(file_paths)
            self.demo_performance_monitoring(file_paths)
            
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
            
            # Generate final report
            self.generate_final_report()
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise
        finally:
            # Cleanup
            self.cleanup()
    
    def generate_final_report(self):
        """
        Generate final demonstration report
        """
        logger.info("=== Final Report ===")
        
        # Calculate overall statistics
        total_processing_time = sum(self.demo_stats['processing_times'].values())
        avg_throughput = self.demo_stats['total_rows'] / total_processing_time if total_processing_time > 0 else 0
        
        report = {
            'demonstration_summary': {
                'total_files_processed': self.demo_stats['files_generated'],
                'total_rows_processed': self.demo_stats['total_rows'],
                'total_data_size_mb': self.demo_stats['total_size_mb'],
                'total_processing_time_seconds': total_processing_time,
                'average_throughput_rows_per_second': avg_throughput
            },
            'component_performance': self.demo_stats['processing_times'],
            'cache_performance': self.demo_stats['cache_performance'],
            'validation_results': self.demo_stats['validation_results'],
            'performance_monitoring': self.performance_monitor.get_performance_summary(),
            'system_info': self.performance_monitor.get_system_info()
        }
        
        # Save report
        report_path = self.demo_data_dir / "pipeline_demo_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info(f"Demonstration completed successfully!")
        logger.info(f"Total files processed: {report['demonstration_summary']['total_files_processed']}")
        logger.info(f"Total rows processed: {report['demonstration_summary']['total_rows_processed']:,}")
        logger.info(f"Total data size: {report['demonstration_summary']['total_data_size_mb']:.1f} MB")
        logger.info(f"Total processing time: {report['demonstration_summary']['total_processing_time_seconds']:.2f} seconds")
        logger.info(f"Average throughput: {report['demonstration_summary']['average_throughput_rows_per_second']:.0f} rows/second")
        logger.info(f"Detailed report saved to: {report_path}")
    
    def cleanup(self):
        """
        Cleanup demo resources
        """
        logger.info("Cleaning up demo resources...")
        
        # Shutdown components
        self.cache_manager.shutdown()
        self.parallel_processor.shutdown()
        
        # Clean up temporary files if needed
        # (keeping them for inspection in this demo)
        logger.info("Cleanup completed")

def main():
    """
    Main demonstration function
    """
    try:
        demo = DataPipelineDemo()
        demo.run_comprehensive_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()