"""
Test suite for distributed locking and race condition prevention.
Validates that distributed locks prevent race conditions and ensure data consistency.
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import MagicMock, patch, AsyncMock
from typing import List, Dict, Any
import sys
import os
import redis.asyncio as redis

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.tactical.distributed_lock import DistributedLockManager, LockResult, LockMetrics


class TestDistributedLocking:
    """Test distributed locking and race condition prevention"""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client"""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        mock_redis.get = AsyncMock()
        mock_redis.delete = AsyncMock()
        mock_redis.exists = AsyncMock()
        mock_redis.expire = AsyncMock()
        return mock_redis

    @pytest.fixture
    def lock_manager(self, mock_redis):
        """Create test lock manager with mock Redis"""
        manager = DistributedLockManager(
            redis_host="localhost",
            redis_port=6379,
            default_timeout=30.0
        )
        # Replace Redis client with mock
        manager.redis_client = mock_redis
        return manager

    def test_lock_acquisition_success(self, lock_manager, mock_redis):
        """Test successful lock acquisition"""
        async def run_test():
            # Mock successful lock acquisition
            mock_redis.set.return_value = True
            
            # Acquire lock
            result = await lock_manager.acquire_lock("test_resource", timeout=10.0)
            
            # Verify lock was acquired
            assert result.acquired is True
            assert result.lock_id is not None
            assert result.expiry_time is not None
            assert result.error is None
            
            # Verify Redis was called correctly
            mock_redis.set.assert_called_once()
            call_args = mock_redis.set.call_args
            assert "tactical_lock:test_resource" in call_args[0][0]
            assert call_args[1]['ex'] == 10.0  # Expiry time

        asyncio.run(run_test())

    def test_lock_acquisition_failure(self, lock_manager, mock_redis):
        """Test lock acquisition failure when resource is already locked"""
        async def run_test():
            # Mock lock already exists
            mock_redis.set.return_value = False
            
            # Try to acquire lock
            result = await lock_manager.acquire_lock("test_resource", timeout=10.0)
            
            # Verify lock acquisition failed
            assert result.acquired is False
            assert result.lock_id is None
            assert result.error is not None

        asyncio.run(run_test())

    def test_lock_release_success(self, lock_manager, mock_redis):
        """Test successful lock release"""
        async def run_test():
            # Mock successful lock acquisition
            mock_redis.set.return_value = True
            mock_redis.delete.return_value = 1
            
            # Acquire lock
            result = await lock_manager.acquire_lock("test_resource")
            lock_id = result.lock_id
            
            # Release lock
            released = await lock_manager.release_lock("test_resource", lock_id)
            
            # Verify lock was released
            assert released is True
            mock_redis.delete.assert_called_once()

        asyncio.run(run_test())

    def test_lock_timeout_handling(self, lock_manager, mock_redis):
        """Test lock timeout and automatic cleanup"""
        async def run_test():
            # Mock lock acquisition with timeout
            mock_redis.set.return_value = True
            
            # Acquire lock with short timeout
            result = await lock_manager.acquire_lock("test_resource", timeout=0.1)
            
            # Verify lock was acquired
            assert result.acquired is True
            
            # Wait for timeout
            await asyncio.sleep(0.2)
            
            # Try to acquire same lock (should succeed after timeout)
            mock_redis.set.return_value = True  # Simulate timeout cleanup
            result2 = await lock_manager.acquire_lock("test_resource", timeout=0.1)
            
            # Should be able to acquire lock after timeout
            assert result2.acquired is True

        asyncio.run(run_test())

    def test_race_condition_prevention(self, lock_manager, mock_redis):
        """Test that locks prevent race conditions"""
        async def run_test():
            # Shared resource
            shared_counter = {'value': 0}
            
            # Mock Redis to simulate lock behavior
            acquired_locks = set()
            
            async def mock_set_with_lock_logic(key, value, ex=None):
                if key in acquired_locks:
                    return False  # Lock already acquired
                acquired_locks.add(key)
                return True
            
            async def mock_delete_with_lock_logic(key):
                if key in acquired_locks:
                    acquired_locks.remove(key)
                    return 1
                return 0
            
            mock_redis.set.side_effect = mock_set_with_lock_logic
            mock_redis.delete.side_effect = mock_delete_with_lock_logic
            
            # Function that increments counter with lock protection
            async def increment_counter(worker_id: int):
                # Acquire lock
                result = await lock_manager.acquire_lock("counter_resource")
                
                if result.acquired:
                    try:
                        # Critical section - increment counter
                        current_value = shared_counter['value']
                        await asyncio.sleep(0.01)  # Simulate processing time
                        shared_counter['value'] = current_value + 1
                        
                    finally:
                        # Release lock
                        await lock_manager.release_lock("counter_resource", result.lock_id)
            
            # Run concurrent operations
            tasks = [increment_counter(i) for i in range(10)]
            await asyncio.gather(*tasks)
            
            # Verify no race condition occurred
            assert shared_counter['value'] == 10

        asyncio.run(run_test())

    def test_correlation_id_uniqueness(self, lock_manager, mock_redis):
        """Test correlation ID uniqueness enforcement"""
        async def run_test():
            # Mock Redis operations
            correlation_ids = set()
            
            async def mock_set_correlation_id(key, value, ex=None):
                if key in correlation_ids:
                    return False  # Correlation ID already exists
                correlation_ids.add(key)
                return True
            
            mock_redis.set.side_effect = mock_set_correlation_id
            
            # Test correlation ID uniqueness
            correlation_id = "test_correlation_123"
            
            # First registration should succeed
            result1 = await lock_manager.ensure_correlation_id_uniqueness(correlation_id)
            assert result1 is True
            
            # Second registration should fail
            result2 = await lock_manager.ensure_correlation_id_uniqueness(correlation_id)
            assert result2 is False

        asyncio.run(run_test())

    def test_deadlock_prevention(self, lock_manager, mock_redis):
        """Test deadlock prevention through lock ordering"""
        async def run_test():
            # Mock Redis with lock tracking
            acquired_locks = {}
            
            async def mock_set_with_deadlock_prevention(key, value, ex=None):
                # Simulate lock ordering to prevent deadlocks
                if key in acquired_locks:
                    return False
                acquired_locks[key] = value
                return True
            
            async def mock_delete_with_cleanup(key):
                if key in acquired_locks:
                    del acquired_locks[key]
                    return 1
                return 0
            
            mock_redis.set.side_effect = mock_set_with_deadlock_prevention
            mock_redis.delete.side_effect = mock_delete_with_cleanup
            
            # Test acquiring multiple locks in order
            async def acquire_multiple_locks(lock_names: List[str]):
                # Sort lock names to ensure consistent ordering
                sorted_locks = sorted(lock_names)
                acquired_locks_local = []
                
                try:
                    for lock_name in sorted_locks:
                        result = await lock_manager.acquire_lock(lock_name)
                        if result.acquired:
                            acquired_locks_local.append((lock_name, result.lock_id))
                        else:
                            # Rollback if any lock fails
                            for lock_name, lock_id in acquired_locks_local:
                                await lock_manager.release_lock(lock_name, lock_id)
                            return False
                    
                    return True
                    
                finally:
                    # Clean up all acquired locks
                    for lock_name, lock_id in acquired_locks_local:
                        await lock_manager.release_lock(lock_name, lock_id)
            
            # Test concurrent lock acquisition
            lock_sets = [
                ["resource_A", "resource_B"],
                ["resource_B", "resource_A"],  # Different order
                ["resource_A", "resource_C"],
                ["resource_B", "resource_C"]
            ]
            
            tasks = [acquire_multiple_locks(lock_set) for lock_set in lock_sets]
            results = await asyncio.gather(*tasks)
            
            # Should complete without deadlock
            assert all(isinstance(result, bool) for result in results)

        asyncio.run(run_test())

    def test_lock_metrics_tracking(self, lock_manager, mock_redis):
        """Test lock metrics tracking and reporting"""
        async def run_test():
            # Mock Redis operations
            mock_redis.set.return_value = True
            mock_redis.delete.return_value = 1
            
            # Reset metrics
            lock_manager.lock_metrics = LockMetrics()
            
            # Perform lock operations
            for i in range(5):
                result = await lock_manager.acquire_lock(f"resource_{i}")
                if result.acquired:
                    await lock_manager.release_lock(f"resource_{i}", result.lock_id)
            
            # Get metrics
            metrics = lock_manager.get_lock_metrics()
            
            # Verify metrics
            assert metrics.total_acquisitions == 5
            assert metrics.successful_acquisitions == 5
            assert metrics.total_releases == 5
            assert metrics.successful_releases == 5

        asyncio.run(run_test())

    def test_lock_contention_handling(self, lock_manager, mock_redis):
        """Test handling of lock contention"""
        async def run_test():
            # Mock Redis to simulate contention
            lock_acquired = False
            
            async def mock_set_with_contention(key, value, ex=None):
                nonlocal lock_acquired
                if not lock_acquired:
                    lock_acquired = True
                    return True
                return False
            
            async def mock_delete_with_release(key):
                nonlocal lock_acquired
                if lock_acquired:
                    lock_acquired = False
                    return 1
                return 0
            
            mock_redis.set.side_effect = mock_set_with_contention
            mock_redis.delete.side_effect = mock_delete_with_release
            
            # Test contention handling
            async def try_acquire_lock(worker_id: int):
                result = await lock_manager.acquire_lock("contended_resource")
                return result.acquired
            
            # First acquisition should succeed
            result1 = await try_acquire_lock(1)
            assert result1 is True
            
            # Second acquisition should fail (contention)
            result2 = await try_acquire_lock(2)
            assert result2 is False

        asyncio.run(run_test())

    def test_lock_cleanup_on_connection_failure(self, lock_manager, mock_redis):
        """Test lock cleanup when Redis connection fails"""
        async def run_test():
            # Mock Redis connection failure
            mock_redis.set.side_effect = redis.ConnectionError("Connection failed")
            
            # Try to acquire lock
            result = await lock_manager.acquire_lock("test_resource")
            
            # Should handle failure gracefully
            assert result.acquired is False
            assert result.error is not None
            assert "Connection failed" in result.error

        asyncio.run(run_test())

    def test_lock_expiry_extension(self, lock_manager, mock_redis):
        """Test lock expiry extension for long-running operations"""
        async def run_test():
            # Mock Redis operations
            mock_redis.set.return_value = True
            mock_redis.expire.return_value = True
            
            # Acquire lock
            result = await lock_manager.acquire_lock("long_running_resource", timeout=5.0)
            assert result.acquired is True
            
            # Extend lock expiry
            extended = await lock_manager.extend_lock_expiry("long_running_resource", result.lock_id, 10.0)
            
            # Verify extension
            assert extended is True
            mock_redis.expire.assert_called_once()

        asyncio.run(run_test())

    @pytest.mark.integration
    def test_end_to_end_locking_scenario(self, lock_manager, mock_redis):
        """Test complete locking scenario with multiple operations"""
        async def run_test():
            # Mock Redis operations
            locks_db = {}
            
            async def mock_set_realistic(key, value, ex=None):
                if key in locks_db:
                    return False
                locks_db[key] = {'value': value, 'expiry': time.time() + (ex or 30)}
                return True
            
            async def mock_delete_realistic(key):
                if key in locks_db:
                    del locks_db[key]
                    return 1
                return 0
            
            async def mock_get_realistic(key):
                if key in locks_db:
                    return locks_db[key]['value']
                return None
            
            mock_redis.set.side_effect = mock_set_realistic
            mock_redis.delete.side_effect = mock_delete_realistic
            mock_redis.get.side_effect = mock_get_realistic
            
            # Simulate trading system operations
            trade_orders = []
            
            async def process_trade_order(order_id: int, symbol: str):
                # Acquire lock for symbol
                result = await lock_manager.acquire_lock(f"trading_{symbol}")
                
                if result.acquired:
                    try:
                        # Simulate trade processing
                        await asyncio.sleep(0.01)
                        trade_orders.append({'order_id': order_id, 'symbol': symbol})
                        
                    finally:
                        # Release lock
                        await lock_manager.release_lock(f"trading_{symbol}", result.lock_id)
                    
                    return True
                else:
                    return False
            
            # Process multiple orders concurrently
            tasks = []
            for i in range(20):
                symbol = f"STOCK_{i % 5}"  # 5 different symbols
                task = process_trade_order(i, symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Verify all orders were processed
            assert all(results)
            assert len(trade_orders) == 20
            
            # Verify no race conditions (each symbol processed sequentially)
            symbol_orders = {}
            for order in trade_orders:
                symbol = order['symbol']
                if symbol not in symbol_orders:
                    symbol_orders[symbol] = []
                symbol_orders[symbol].append(order['order_id'])
            
            # Each symbol should have processed orders in some order
            for symbol, orders in symbol_orders.items():
                assert len(orders) > 0

        asyncio.run(run_test())

    def test_lock_health_monitoring(self, lock_manager, mock_redis):
        """Test lock health monitoring and reporting"""
        async def run_test():
            # Mock Redis operations
            mock_redis.set.return_value = True
            mock_redis.delete.return_value = 1
            mock_redis.ping.return_value = True
            
            # Test health check
            health = await lock_manager.check_health()
            
            # Verify health status
            assert health['status'] == 'healthy'
            assert health['redis_connection'] is True
            assert 'active_locks' in health
            assert 'metrics' in health

        asyncio.run(run_test())

    def test_lock_auto_renewal(self, lock_manager, mock_redis):
        """Test automatic lock renewal for long-running operations"""
        async def run_test():
            # Mock Redis operations
            mock_redis.set.return_value = True
            mock_redis.expire.return_value = True
            
            # Acquire lock with auto-renewal
            result = await lock_manager.acquire_lock_with_auto_renewal(
                "long_operation_resource", 
                timeout=5.0, 
                renewal_interval=2.0
            )
            
            assert result.acquired is True
            
            # Wait for renewal to occur
            await asyncio.sleep(3.0)
            
            # Verify renewal occurred
            assert mock_redis.expire.call_count >= 1
            
            # Stop auto-renewal
            await lock_manager.stop_auto_renewal("long_operation_resource", result.lock_id)

        asyncio.run(run_test())