#!/usr/bin/env python3
"""
Integration Test for Unified Data Pipeline System

This script validates the complete data pipeline system functionality
to ensure all components work together correctly.
"""

import sys
import os
import tempfile
import shutil
import time
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from unified_data_loader import UnifiedDataLoader
from memory_manager import MemoryManager
from data_flow_coordinator import DataFlowCoordinator, create_notebook_client, DataStreamType
from performance_monitor import PerformanceMonitor, PerformanceTimer
from scalability_manager import ScalabilityManager, ScalingConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Comprehensive integration test suite for the unified data pipeline"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        self.setup_test_environment()
    
    def setup_test_environment(self):
        """Set up test environment with sample data"""
        logger.info("Setting up test environment...")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temp directory: {self.temp_dir}")
        
        # Create test data directory
        self.test_data_dir = Path(self.temp_dir) / "data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Generate sample NQ data
        self.create_sample_data()
        
        # Initialize components
        self.initialize_components()
        
        logger.info("Test environment setup complete")
    
    def create_sample_data(self):
        """Create sample NQ data files for testing"""
        logger.info("Creating sample NQ data files...")
        
        # Generate sample 30-minute data
        dates = pd.date_range('2024-01-01', periods=1000, freq='30min')
        
        data_30min = pd.DataFrame({
            'timestamp': dates,
            'open': 15000 + np.random.randn(1000) * 100,
            'high': 15000 + np.random.randn(1000) * 100 + 50,
            'low': 15000 + np.random.randn(1000) * 100 - 50,
            'close': 15000 + np.random.randn(1000) * 100,
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        # Ensure OHLC consistency
        data_30min['high'] = np.maximum(data_30min['high'], 
                                       np.maximum(data_30min['open'], data_30min['close']))
        data_30min['low'] = np.minimum(data_30min['low'], 
                                      np.minimum(data_30min['open'], data_30min['close']))
        
        # Save 30-minute data
        data_30min.to_csv(self.test_data_dir / "NQ - 30 min - ETH.csv", index=False)
        
        # Generate sample 5-minute data (more rows)
        dates_5min = pd.date_range('2024-01-01', periods=6000, freq='5min')
        
        data_5min = pd.DataFrame({
            'timestamp': dates_5min,
            'open': 15000 + np.random.randn(6000) * 100,
            'high': 15000 + np.random.randn(6000) * 100 + 50,
            'low': 15000 + np.random.randn(6000) * 100 - 50,
            'close': 15000 + np.random.randn(6000) * 100,
            'volume': np.random.randint(1000, 10000, 6000)
        })
        
        # Ensure OHLC consistency
        data_5min['high'] = np.maximum(data_5min['high'], 
                                      np.maximum(data_5min['open'], data_5min['close']))
        data_5min['low'] = np.minimum(data_5min['low'], 
                                     np.minimum(data_5min['open'], data_5min['close']))
        
        # Save 5-minute data
        data_5min.to_csv(self.test_data_dir / "NQ - 5 min - ETH.csv", index=False)
        
        logger.info(f"Created sample data files:")
        logger.info(f"  - 30min data: {len(data_30min)} rows")
        logger.info(f"  - 5min data: {len(data_5min)} rows")
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        # Initialize data loader
        self.data_loader = UnifiedDataLoader(
            data_dir=str(self.test_data_dir),
            chunk_size=1000,
            cache_enabled=True,
            validation_enabled=True,
            preprocessing_enabled=True
        )
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(
            shared_pool_size_gb=1.0,  # Smaller for testing
            enable_monitoring=True,
            monitoring_interval=1.0
        )
        
        # Initialize coordinator
        coord_dir = Path(self.temp_dir) / "coordination"
        self.coordinator = DataFlowCoordinator(
            coordination_dir=str(coord_dir),
            enable_persistence=True
        )
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(enable_dashboard=False)
        
        # Initialize scalability manager
        scaling_config = ScalingConfiguration(
            max_workers=4,  # Smaller for testing
            enable_gpu_acceleration=torch.cuda.is_available(),
            auto_scaling_enabled=False  # Disable for testing
        )
        self.scalability_manager = ScalabilityManager(scaling_config)
        
        logger.info("Components initialized successfully")
    
    def test_data_loading(self) -> Dict[str, Any]:
        """Test unified data loading functionality"""
        logger.info("Testing data loading...")
        
        test_result = {
            'name': 'data_loading',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # Test loading 30-minute data
            with PerformanceTimer(self.performance_monitor, 'test_load_30min'):
                data_30min = self.data_loader.load_data('30min')
            
            test_result['details']['30min_rows'] = len(data_30min)
            test_result['details']['30min_columns'] = len(data_30min.columns)
            
            # Test loading 5-minute data
            with PerformanceTimer(self.performance_monitor, 'test_load_5min'):
                data_5min = self.data_loader.load_data('5min')
            
            test_result['details']['5min_rows'] = len(data_5min)
            test_result['details']['5min_columns'] = len(data_5min.columns)
            
            # Test chunked loading
            chunks = list(self.data_loader.load_chunked_data('30min', chunk_size=100))
            test_result['details']['chunk_count'] = len(chunks)
            
            # Test data statistics
            stats = self.data_loader.get_data_statistics('30min')
            test_result['details']['has_statistics'] = bool(stats)
            
            # Validate data integrity
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data_30min.columns:
                    test_result['errors'].append(f"Missing column: {col}")
                    test_result['passed'] = False
            
            logger.info("Data loading test completed successfully")
            
        except Exception as e:
            test_result['passed'] = False
            test_result['errors'].append(str(e))
            logger.error(f"Data loading test failed: {e}")
        
        return test_result
    
    def test_memory_management(self) -> Dict[str, Any]:
        """Test memory management functionality"""
        logger.info("Testing memory management...")
        
        test_result = {
            'name': 'memory_management',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # Load test data
            data_30min = self.data_loader.load_data('30min')
            
            # Test storing data in shared pool
            success = self.memory_manager.store_data('test_data_30min', data_30min)
            test_result['details']['store_success'] = success
            
            if not success:
                test_result['errors'].append("Failed to store data in shared pool")
                test_result['passed'] = False
            
            # Test retrieving data from shared pool
            retrieved_data = self.memory_manager.retrieve_data('test_data_30min')
            test_result['details']['retrieve_success'] = retrieved_data is not None
            
            if retrieved_data is None:
                test_result['errors'].append("Failed to retrieve data from shared pool")
                test_result['passed'] = False
            
            # Test memory report
            memory_report = self.memory_manager.get_memory_report()
            test_result['details']['memory_report_available'] = bool(memory_report)
            
            # Test memory optimization
            self.memory_manager.optimize_memory()
            test_result['details']['optimization_completed'] = True
            
            logger.info("Memory management test completed successfully")
            
        except Exception as e:
            test_result['passed'] = False
            test_result['errors'].append(str(e))
            logger.error(f"Memory management test failed: {e}")
        
        return test_result
    
    def test_data_flow_coordination(self) -> Dict[str, Any]:
        """Test data flow coordination functionality"""
        logger.info("Testing data flow coordination...")
        
        test_result = {
            'name': 'data_flow_coordination',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # Create notebook clients
            execution_client = create_notebook_client('test_execution', 'execution', self.coordinator)
            risk_client = create_notebook_client('test_risk', 'risk', self.coordinator)
            
            # Create data stream
            stream = execution_client.create_data_stream(
                'test_stream',
                DataStreamType.MARKET_DATA,
                ['test_risk']
            )
            
            test_result['details']['stream_created'] = stream is not None
            
            # Test data publishing
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
                'price': np.random.randn(10) * 100 + 15000
            })
            
            publish_success = stream.publish(test_data, {'test': True})
            test_result['details']['publish_success'] = publish_success
            
            # Test data consumption
            messages = stream.get_messages(max_messages=1)
            test_result['details']['messages_received'] = len(messages)
            
            if len(messages) == 0:
                test_result['errors'].append("No messages received from stream")
                test_result['passed'] = False
            
            # Test data synchronization
            sync_success = execution_client.sync_data(
                'test_risk',
                'test_sync_data',
                test_data
            )
            test_result['details']['sync_success'] = sync_success
            
            # Get coordination status
            coord_status = self.coordinator.get_coordination_status()
            test_result['details']['coordination_status'] = coord_status
            
            # Cleanup
            execution_client.cleanup()
            risk_client.cleanup()
            
            logger.info("Data flow coordination test completed successfully")
            
        except Exception as e:
            test_result['passed'] = False
            test_result['errors'].append(str(e))
            logger.error(f"Data flow coordination test failed: {e}")
        
        return test_result
    
    def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring functionality"""
        logger.info("Testing performance monitoring...")
        
        test_result = {
            'name': 'performance_monitoring',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # Record some test metrics
            self.performance_monitor.record_metric('test_metric', 1.5)
            self.performance_monitor.record_metric('test_metric', 2.0)
            self.performance_monitor.record_metric('test_metric', 1.8)
            
            # Test performance timer
            with PerformanceTimer(self.performance_monitor, 'test_timer'):
                time.sleep(0.1)  # Simulate some work
            
            # Get performance summary
            summary = self.performance_monitor.get_performance_summary()
            test_result['details']['summary_available'] = bool(summary)
            
            # Test benchmark suite
            benchmark_suite = self.performance_monitor.create_benchmark_suite(self.data_loader)
            test_result['details']['benchmark_suite_created'] = benchmark_suite is not None
            
            # Run a simple benchmark
            loading_results = benchmark_suite.benchmark_loading_performance(['30min'], iterations=2)
            test_result['details']['benchmark_completed'] = bool(loading_results)
            
            logger.info("Performance monitoring test completed successfully")
            
        except Exception as e:
            test_result['passed'] = False
            test_result['errors'].append(str(e))
            logger.error(f"Performance monitoring test failed: {e}")
        
        return test_result
    
    def test_scalability_features(self) -> Dict[str, Any]:
        """Test scalability features"""
        logger.info("Testing scalability features...")
        
        test_result = {
            'name': 'scalability_features',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # Initialize scalability system
            self.scalability_manager.initialize_system()
            
            # Get system capabilities
            capabilities = self.scalability_manager.get_system_capabilities()
            test_result['details']['capabilities'] = capabilities
            
            # Test processing with sample data
            sample_data = torch.randn(1000, 10)
            
            def sample_processing_func(data):
                return torch.mean(data, dim=1, keepdim=True)
            
            # Process data
            start_time = time.time()
            result = self.scalability_manager.process_large_dataset(
                sample_data,
                sample_processing_func,
                batch_size=100
            )
            processing_time = time.time() - start_time
            
            test_result['details']['processing_completed'] = result is not None
            test_result['details']['processing_time'] = processing_time
            test_result['details']['input_shape'] = list(sample_data.shape)
            test_result['details']['output_shape'] = list(result.shape) if result is not None else None
            
            # Get performance statistics
            perf_stats = self.scalability_manager.get_performance_statistics()
            test_result['details']['performance_stats'] = perf_stats
            
            logger.info("Scalability features test completed successfully")
            
        except Exception as e:
            test_result['passed'] = False
            test_result['errors'].append(str(e))
            logger.error(f"Scalability features test failed: {e}")
        
        return test_result
    
    def test_integration_workflow(self) -> Dict[str, Any]:
        """Test complete integration workflow"""
        logger.info("Testing complete integration workflow...")
        
        test_result = {
            'name': 'integration_workflow',
            'passed': True,
            'details': {},
            'errors': []
        }
        
        try:
            # Step 1: Load data
            data_30min = self.data_loader.load_data('30min')
            test_result['details']['data_loaded'] = len(data_30min) > 0
            
            # Step 2: Store in shared memory
            self.memory_manager.store_data('workflow_data', data_30min)
            
            # Step 3: Create coordination
            execution_client = create_notebook_client('workflow_execution', 'execution', self.coordinator)
            risk_client = create_notebook_client('workflow_risk', 'risk', self.coordinator)
            
            # Step 4: Create data stream
            stream = execution_client.create_data_stream(
                'workflow_stream',
                DataStreamType.MARKET_DATA,
                ['workflow_risk']
            )
            
            # Step 5: Process data with monitoring
            with PerformanceTimer(self.performance_monitor, 'workflow_processing'):
                # Simulate processing
                processed_data = data_30min.copy()
                processed_data['processed'] = True
                
                # Publish to stream
                stream.publish(processed_data.head(10), {'workflow_step': 'processing'})
            
            # Step 6: Scale processing if needed
            if torch.cuda.is_available():
                tensor_data = torch.randn(500, 20)
                
                def simple_processing(data):
                    return torch.nn.functional.relu(data).mean(dim=1, keepdim=True)
                
                scaled_result = self.scalability_manager.process_large_dataset(
                    tensor_data,
                    simple_processing,
                    batch_size=50
                )
                
                test_result['details']['scaled_processing_completed'] = scaled_result is not None
            
            # Step 7: Validate results
            messages = stream.get_messages(max_messages=1)
            test_result['details']['workflow_messages'] = len(messages)
            
            # Step 8: Get final statistics
            final_stats = {
                'data_loader': self.data_loader.get_performance_metrics(),
                'memory_manager': self.memory_manager.get_memory_report(),
                'coordinator': self.coordinator.get_coordination_status(),
                'performance_monitor': self.performance_monitor.get_performance_summary()
            }
            
            test_result['details']['final_statistics'] = final_stats
            
            # Cleanup
            execution_client.cleanup()
            risk_client.cleanup()
            
            logger.info("Integration workflow test completed successfully")
            
        except Exception as e:
            test_result['passed'] = False
            test_result['errors'].append(str(e))
            logger.error(f"Integration workflow test failed: {e}")
        
        return test_result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting comprehensive integration test suite...")
        
        test_methods = [
            self.test_data_loading,
            self.test_memory_management,
            self.test_data_flow_coordination,
            self.test_performance_monitoring,
            self.test_scalability_features,
            self.test_integration_workflow
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                result = test_method()
                results[result['name']] = result
                
                if result['passed']:
                    passed_tests += 1
                    logger.info(f"✅ {result['name']} - PASSED")
                else:
                    logger.error(f"❌ {result['name']} - FAILED")
                    for error in result['errors']:
                        logger.error(f"   Error: {error}")
                
            except Exception as e:
                logger.error(f"❌ {test_method.__name__} - CRASHED: {e}")
                results[test_method.__name__] = {
                    'name': test_method.__name__,
                    'passed': False,
                    'errors': [str(e)],
                    'details': {}
                }
        
        # Summary
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'test_results': results
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"INTEGRATION TEST SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        
        if summary['success_rate'] == 1.0:
            logger.info("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        else:
            logger.warning("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
        
        return summary
    
    def cleanup(self):
        """Clean up test environment"""
        logger.info("Cleaning up test environment...")
        
        try:
            # Cleanup components
            if hasattr(self, 'memory_manager'):
                self.memory_manager.cleanup()
            
            if hasattr(self, 'coordinator'):
                self.coordinator.cleanup()
            
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.cleanup()
            
            if hasattr(self, 'scalability_manager'):
                self.scalability_manager.cleanup()
            
            # Remove temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Removed temp directory: {self.temp_dir}")
        
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        logger.info("Test environment cleanup completed")

def main():
    """Main test execution"""
    print("🚀 Starting Unified Data Pipeline Integration Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = IntegrationTestSuite()
    
    try:
        # Run all tests
        results = test_suite.run_all_tests()
        
        # Save results
        import json
        with open('integration_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📊 Test results saved to: integration_test_results.json")
        
        # Exit with appropriate code
        exit_code = 0 if results['success_rate'] == 1.0 else 1
        
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        exit_code = 2
    
    finally:
        # Cleanup
        test_suite.cleanup()
    
    print("\n" + "=" * 60)
    print("🏁 Integration tests completed")
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)