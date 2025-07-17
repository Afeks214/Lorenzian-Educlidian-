#!/usr/bin/env python3
"""
Comprehensive Test Suite for Batch Processing Framework

This test suite validates the batch processing capabilities for large dataset training
including memory efficiency, sliding window data loading, and checkpoint management.
"""

import os
import sys
import time
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import torch
import json

# Add project path
sys.path.append('/home/QuantNova/GrandModel')
from colab.utils.batch_processor import (
    BatchProcessor, BatchConfig, MemoryMonitor, SlidingWindowDataLoader,
    DataStreamer, CheckpointManager, calculate_optimal_batch_size,
    create_large_dataset_simulation
)


class TestBatchProcessing(unittest.TestCase):
    """Test suite for batch processing framework"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, 'test_data.csv')
        self.checkpoint_dir = os.path.join(self.temp_dir, 'checkpoints')
        
        # Create test data
        create_large_dataset_simulation(
            output_path=self.test_data_path,
            num_rows=1000,
            features=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        # Test configuration
        self.config = BatchConfig(
            batch_size=8,
            sequence_length=50,
            overlap=10,
            prefetch_batches=2,
            max_memory_percent=70.0,
            checkpoint_frequency=10,
            enable_caching=True,
            cache_size=100,
            num_workers=1
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_batch_config_creation(self):
        """Test batch configuration creation"""
        config = BatchConfig(batch_size=32, sequence_length=100)
        
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.sequence_length, 100)
        self.assertEqual(config.overlap, 20)  # Default value
        self.assertEqual(config.prefetch_batches, 2)  # Default value
        self.assertTrue(config.enable_caching)  # Default value
    
    def test_memory_monitor(self):
        """Test memory monitoring functionality"""
        monitor = MemoryMonitor(max_memory_percent=80.0)
        
        # Test memory usage retrieval
        usage = monitor.get_memory_usage()
        self.assertIn('system_used_gb', usage)
        self.assertIn('system_total_gb', usage)
        self.assertIn('system_percent', usage)
        self.assertIn('process_percent', usage)
        
        # Test batch size optimization
        current_batch_size = 32
        optimized_size = monitor.optimize_batch_size(current_batch_size, 70.0)
        self.assertIsInstance(optimized_size, int)
        self.assertGreater(optimized_size, 0)
    
    def test_sliding_window_data_loader(self):
        """Test sliding window data loader"""
        loader = SlidingWindowDataLoader(
            data_path=self.test_data_path,
            config=self.config,
            chunksize=100
        )
        
        # Test dataset size calculation
        dataset_size = loader.get_dataset_size()
        self.assertEqual(dataset_size, 1000)
        
        # Test sliding window creation
        windows = list(loader.create_sliding_windows(0, 200))
        self.assertGreater(len(windows), 0)
        
        # Verify window properties
        for window in windows[:3]:  # Test first 3 windows
            self.assertEqual(len(window), self.config.sequence_length)
            self.assertIn('Date', window.columns)
            self.assertIn('Close', window.columns)
    
    def test_batch_creation(self):
        """Test batch creation from sliding windows"""
        loader = SlidingWindowDataLoader(
            data_path=self.test_data_path,
            config=self.config,
            chunksize=100
        )
        
        # Test batch creation
        batches = list(loader.create_batches(0, 300))
        self.assertGreater(len(batches), 0)
        
        # Verify batch properties
        for batch in batches[:2]:  # Test first 2 batches
            self.assertLessEqual(len(batch), self.config.batch_size)
            for window in batch:
                self.assertEqual(len(window), self.config.sequence_length)
    
    def test_data_streamer(self):
        """Test data streaming functionality"""
        loader = SlidingWindowDataLoader(
            data_path=self.test_data_path,
            config=self.config,
            chunksize=100
        )
        
        streamer = DataStreamer(loader, prefetch_size=2)
        
        # Test streaming start
        streamer.start_streaming(0, 300)
        self.assertTrue(streamer.streaming)
        
        # Test batch retrieval
        batch = streamer.get_next_batch()
        self.assertIsNotNone(batch)
        self.assertLessEqual(len(batch), self.config.batch_size)
        
        # Test streaming stop
        streamer.stop_streaming()
        self.assertFalse(streamer.streaming)
    
    def test_checkpoint_manager(self):
        """Test checkpoint management"""
        manager = CheckpointManager(self.checkpoint_dir, max_checkpoints=3)
        
        # Test checkpoint saving
        model_state = {'model_param': torch.randn(10, 10)}
        optimizer_state = {'optimizer_param': 'test'}
        metrics = {'loss': 0.5, 'accuracy': 0.8}
        
        manager.save_checkpoint(
            model_state=model_state,
            optimizer_state=optimizer_state,
            epoch=1,
            batch_idx=10,
            data_position=100,
            metrics=metrics
        )
        
        # Test checkpoint loading
        loaded_checkpoint = manager.load_checkpoint()
        self.assertIsNotNone(loaded_checkpoint)
        self.assertEqual(loaded_checkpoint['epoch'], 1)
        self.assertEqual(loaded_checkpoint['batch_idx'], 10)
        self.assertEqual(loaded_checkpoint['metrics']['loss'], 0.5)
        
        # Test checkpoint info
        checkpoint_info = manager.get_latest_checkpoint_info()
        self.assertIsNotNone(checkpoint_info)
        self.assertEqual(checkpoint_info['epoch'], 1)
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization"""
        processor = BatchProcessor(
            data_path=self.test_data_path,
            config=self.config,
            checkpoint_dir=self.checkpoint_dir
        )
        
        self.assertEqual(processor.data_path, self.test_data_path)
        self.assertEqual(processor.config, self.config)
        self.assertIsNotNone(processor.checkpoint_manager)
        self.assertIsNotNone(processor.data_loader)
        self.assertIsNotNone(processor.streamer)
    
    def test_batch_processing_with_mock_trainer(self):
        """Test batch processing with mock trainer"""
        # Create mock trainer
        mock_trainer = Mock()
        mock_trainer.get_action.return_value = np.random.randint(0, 5)
        mock_trainer.store_transition.return_value = None
        mock_trainer.update.return_value = 0.5
        
        processor = BatchProcessor(
            data_path=self.test_data_path,
            config=self.config,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Test batch processing
        batch_count = 0
        for batch_result in processor.process_batches(mock_trainer, end_idx=200):
            batch_count += 1
            
            # Verify batch result structure
            self.assertIn('batch_idx', batch_result)
            self.assertIn('batch_size', batch_result)
            self.assertIn('batch_time', batch_result)
            self.assertIn('memory_usage', batch_result)
            self.assertIn('metrics', batch_result)
            
            # Verify metrics
            self.assertIn('avg_reward', batch_result['metrics'])
            self.assertIn('avg_loss', batch_result['metrics'])
            self.assertIn('num_episodes', batch_result['metrics'])
            
            if batch_count >= 3:  # Test with limited batches
                break
        
        # Verify mock trainer was called
        self.assertGreater(mock_trainer.get_action.call_count, 0)
        self.assertGreater(mock_trainer.store_transition.call_count, 0)
    
    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation"""
        # Test with different dataset sizes
        test_cases = [
            (500, 2.0, 50, 8),    # Small dataset
            (5000, 4.0, 100, 16),  # Medium dataset
            (50000, 8.0, 200, 32), # Large dataset
        ]
        
        for data_size, memory_limit, sequence_length, expected_max in test_cases:
            batch_size = calculate_optimal_batch_size(
                data_size=data_size,
                memory_limit_gb=memory_limit,
                sequence_length=sequence_length
            )
            
            self.assertIsInstance(batch_size, int)
            self.assertGreater(batch_size, 0)
            self.assertLessEqual(batch_size, expected_max)
    
    def test_large_dataset_simulation(self):
        """Test large dataset simulation creation"""
        sim_path = os.path.join(self.temp_dir, 'simulated_large.csv')
        
        created_path = create_large_dataset_simulation(
            output_path=sim_path,
            num_rows=5000,
            features=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        self.assertEqual(created_path, sim_path)
        self.assertTrue(os.path.exists(sim_path))
        
        # Verify simulated data
        df = pd.read_csv(sim_path)
        self.assertEqual(len(df), 5000)
        self.assertIn('Date', df.columns)
        self.assertIn('Close', df.columns)
        self.assertIn('Volume', df.columns)
    
    def test_memory_optimization_during_processing(self):
        """Test memory optimization during batch processing"""
        config = BatchConfig(
            batch_size=4,
            sequence_length=30,
            overlap=5,
            max_memory_percent=60.0,  # Low memory limit
            checkpoint_frequency=5
        )
        
        processor = BatchProcessor(
            data_path=self.test_data_path,
            config=config,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Mock trainer with memory tracking
        mock_trainer = Mock()
        mock_trainer.get_action.return_value = np.random.randint(0, 5)
        mock_trainer.store_transition.return_value = None
        mock_trainer.update.return_value = 0.5
        
        # Track memory usage during processing
        memory_usages = []
        batch_count = 0
        
        for batch_result in processor.process_batches(mock_trainer, end_idx=150):
            memory_usages.append(batch_result['memory_usage']['system_percent'])
            batch_count += 1
            
            if batch_count >= 5:  # Test with limited batches
                break
        
        # Verify memory monitoring
        self.assertGreater(len(memory_usages), 0)
        for usage in memory_usages:
            self.assertIsInstance(usage, (int, float))
            self.assertGreater(usage, 0)
    
    def test_checkpoint_resumption(self):
        """Test training resumption from checkpoint"""
        config = BatchConfig(
            batch_size=4,
            sequence_length=20,
            checkpoint_frequency=2
        )
        
        processor = BatchProcessor(
            data_path=self.test_data_path,
            config=config,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Mock trainer with state methods
        mock_trainer = Mock()
        mock_trainer.get_action.return_value = np.random.randint(0, 5)
        mock_trainer.store_transition.return_value = None
        mock_trainer.update.return_value = 0.5
        mock_trainer.get_model_state.return_value = {'param': torch.randn(5, 5)}
        mock_trainer.get_optimizer_state.return_value = {'lr': 0.001}
        
        # Run initial processing to create checkpoint
        batch_count = 0
        for batch_result in processor.process_batches(mock_trainer, end_idx=100):
            batch_count += 1
            if batch_count >= 3:  # Create checkpoint
                break
        
        # Verify checkpoint was created
        checkpoint_info = processor.checkpoint_manager.get_latest_checkpoint_info()
        self.assertIsNotNone(checkpoint_info)
        
        # Test resumption
        new_processor = BatchProcessor(
            data_path=self.test_data_path,
            config=config,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Resume from checkpoint
        batch_count = 0
        for batch_result in new_processor.process_batches(
            mock_trainer, 
            resume_from_checkpoint=True, 
            end_idx=150
        ):
            batch_count += 1
            if batch_count >= 2:  # Test resumption
                break
        
        # Verify resumption worked
        self.assertGreater(batch_count, 0)


class TestBatchProcessingIntegration(unittest.TestCase):
    """Integration tests for batch processing with notebook components"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, 'integration_test_data.csv')
        
        # Create larger test dataset
        create_large_dataset_simulation(
            output_path=self.test_data_path,
            num_rows=2000,
            features=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        )
        
        self.config = BatchConfig(
            batch_size=16,
            sequence_length=48,
            overlap=12,
            prefetch_batches=2,
            max_memory_percent=75.0,
            checkpoint_frequency=20,
            enable_caching=True
        )
    
    def tearDown(self):
        """Clean up integration test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_strategic_batch_processing_simulation(self):
        """Test batch processing with strategic MAPPO simulation"""
        # Mock strategic components
        class MockStrategicMatrixProcessor:
            def __init__(self):
                self.feature_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13']
            
            def process_batch(self, data_batch):
                return np.random.randn(len(data_batch), 48, 13)
        
        class MockUncertaintyQuantifier:
            def quantify_uncertainty(self, matrix):
                return {
                    'overall_confidence': np.random.uniform(0.3, 0.9),
                    'confidence_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH'])
                }
        
        class MockRegimeAgent:
            def detect_regime(self, matrix):
                return {
                    'current_regime': np.random.randint(0, 4),
                    'regime_confidence': np.random.uniform(0.5, 0.95)
                }
        
        class MockVectorDB:
            def __init__(self):
                self.decisions = []
            
            def add_decision(self, matrix, decision_data):
                self.decisions.append(decision_data)
        
        # Create strategic trainer mock
        class MockStrategicTrainer:
            def __init__(self):
                self.matrix_processor = MockStrategicMatrixProcessor()
                self.uncertainty_quantifier = MockUncertaintyQuantifier()
                self.regime_agent = MockRegimeAgent()
                self.vector_db = MockVectorDB()
                self.batch_results = []
            
            def process_batch(self, data_batch):
                batch_matrices = self.matrix_processor.process_batch(data_batch)
                
                batch_rewards = []
                for matrix in batch_matrices:
                    uncertainty = self.uncertainty_quantifier.quantify_uncertainty(matrix)
                    regime = self.regime_agent.detect_regime(matrix)
                    reward = uncertainty['overall_confidence'] * regime['regime_confidence']
                    batch_rewards.append(reward)
                    
                    self.vector_db.add_decision(matrix, {
                        'uncertainty': uncertainty,
                        'regime': regime,
                        'reward': reward
                    })
                
                return {
                    'avg_reward': np.mean(batch_rewards),
                    'avg_loss': np.random.uniform(0.1, 0.5),
                    'num_episodes': len(data_batch)
                }
        
        # Test strategic batch processing
        processor = BatchProcessor(
            data_path=self.test_data_path,
            config=self.config,
            checkpoint_dir=os.path.join(self.temp_dir, 'strategic_checkpoints')
        )
        
        strategic_trainer = MockStrategicTrainer()
        
        # Process batches
        batch_count = 0
        total_rewards = []
        
        for batch_result in processor.process_batches(strategic_trainer, end_idx=500):
            batch_count += 1
            total_rewards.append(batch_result['metrics']['avg_reward'])
            
            # Verify strategic-specific metrics
            self.assertIn('avg_reward', batch_result['metrics'])
            self.assertIn('num_episodes', batch_result['metrics'])
            
            if batch_count >= 5:
                break
        
        # Verify strategic processing worked
        self.assertGreater(batch_count, 0)
        self.assertGreater(len(total_rewards), 0)
        self.assertGreater(len(strategic_trainer.vector_db.decisions), 0)
        
        print(f"Strategic batch processing test: {batch_count} batches, "
              f"avg reward: {np.mean(total_rewards):.3f}")
    
    def test_tactical_batch_processing_simulation(self):
        """Test batch processing with tactical MAPPO simulation"""
        # Mock tactical trainer
        class MockTacticalTrainer:
            def __init__(self):
                self.state_dim = 7
                self.action_dim = 5
                self.batch_results = []
            
            def get_action(self, state, deterministic=False):
                return np.random.randint(0, self.action_dim)
            
            def store_transition(self, state, action, reward, next_state, done):
                pass
            
            def update(self):
                return np.random.uniform(0.01, 0.1)
        
        # Test tactical batch processing
        tactical_config = BatchConfig(
            batch_size=32,
            sequence_length=60,
            overlap=15,
            max_memory_percent=80.0,
            checkpoint_frequency=50
        )
        
        processor = BatchProcessor(
            data_path=self.test_data_path,
            config=tactical_config,
            checkpoint_dir=os.path.join(self.temp_dir, 'tactical_checkpoints')
        )
        
        tactical_trainer = MockTacticalTrainer()
        
        # Process batches
        batch_count = 0
        processing_times = []
        
        for batch_result in processor.process_batches(tactical_trainer, end_idx=800):
            batch_count += 1
            processing_times.append(batch_result['batch_time'])
            
            # Verify tactical-specific metrics
            self.assertIn('avg_reward', batch_result['metrics'])
            self.assertIn('avg_loss', batch_result['metrics'])
            self.assertLess(batch_result['batch_time'], 10.0)  # Should be fast
            
            if batch_count >= 10:
                break
        
        # Verify tactical processing performance
        self.assertGreater(batch_count, 0)
        self.assertLess(np.mean(processing_times), 5.0)  # Should be fast
        
        print(f"Tactical batch processing test: {batch_count} batches, "
              f"avg processing time: {np.mean(processing_times):.3f}s")


def run_batch_processing_tests():
    """Run all batch processing tests"""
    print("🧪 Running Batch Processing Test Suite...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add basic tests
    test_suite.addTest(unittest.makeSuite(TestBatchProcessing))
    
    # Add integration tests
    test_suite.addTest(unittest.makeSuite(TestBatchProcessingIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ All batch processing tests passed!")
    else:
        print("❌ Some tests failed:")
        for failure in result.failures:
            print(f"   FAIL: {failure[0]}")
        for error in result.errors:
            print(f"   ERROR: {error[0]}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_batch_processing_tests()
    exit(0 if success else 1)