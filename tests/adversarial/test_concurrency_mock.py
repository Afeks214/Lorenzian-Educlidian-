#!/usr/bin/env python3
"""
AGENT 4 RED TEAM CERTIFIER: Concurrency Protection Test (Mock Version)
Mission: Aegis - Tactical MARL Final Security Validation

This test validates that the tactical MARL system properly handles
concurrent identical events using mocked Redis for testing.

🎯 OBJECTIVE: Spawn 100 simultaneous identical events, verify only 1 processes

SECURITY REQUIREMENTS:
- Only ONE event with same correlation_id should be processed
- Remaining 99 events should be detected as duplicates and discarded
- No race conditions should occur
- Processing must be atomic
- System should remain stable under high concurrency
"""

import asyncio
import time
import uuid
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock
from collections import defaultdict
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SynergyEvent:
    """Structure for SYNERGY_DETECTED events."""
    synergy_type: str
    direction: int  # 1 for long, -1 for short
    confidence: float
    signal_sequence: List[Dict[str, Any]]
    market_context: Dict[str, Any]
    correlation_id: str
    timestamp: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SynergyEvent':
        """Create SynergyEvent from dictionary."""
        return cls(
            synergy_type=data.get('synergy_type', ''),
            direction=data.get('direction', 0),
            confidence=data.get('confidence', 0.0),
            signal_sequence=data.get('signal_sequence', []),
            market_context=data.get('market_context', {}),
            correlation_id=data.get('correlation_id', ''),
            timestamp=data.get('timestamp', time.time())
        )

@dataclass
class TacticalDecision:
    """Structure for tactical decisions."""
    action: str  # "long", "short", "hold"
    confidence: float
    agent_votes: List[Dict[str, Any]]
    consensus_breakdown: Dict[str, float]
    execution_command: Dict[str, Any]
    timing: Dict[str, float]
    correlation_id: str
    timestamp: float

class MockRedisLock:
    """Mock Redis distributed lock for testing."""
    
    # Class-level lock state (simulates distributed state)
    _locks = {}
    _lock_mutex = threading.Lock()
    
    def __init__(self, key: str, timeout: int = 10, sleep: float = 0.01, blocking_timeout: float = 0.1):
        self.key = key
        self.timeout = timeout
        self.sleep = sleep
        self.blocking_timeout = blocking_timeout
        self.acquired = False
        self.acquisition_time = None
    
    async def acquire(self, blocking: bool = True) -> bool:
        """Acquire the lock."""
        start_time = time.time()
        
        while True:
            with MockRedisLock._lock_mutex:
                # Check if lock is available
                if self.key not in MockRedisLock._locks:
                    # Acquire lock
                    MockRedisLock._locks[self.key] = {
                        'owner': id(self),
                        'acquired_at': time.time(),
                        'timeout': self.timeout
                    }
                    self.acquired = True
                    self.acquisition_time = time.time()
                    return True
                
                # Check if existing lock has expired
                existing_lock = MockRedisLock._locks[self.key]
                if time.time() - existing_lock['acquired_at'] > existing_lock['timeout']:
                    # Lock expired, acquire it
                    MockRedisLock._locks[self.key] = {
                        'owner': id(self),
                        'acquired_at': time.time(),
                        'timeout': self.timeout
                    }
                    self.acquired = True
                    self.acquisition_time = time.time()
                    return True
            
            # If not blocking, return False immediately
            if not blocking:
                return False
            
            # Check timeout
            if time.time() - start_time > self.blocking_timeout:
                return False
            
            # Wait before retry
            await asyncio.sleep(self.sleep)
    
    async def release(self):
        """Release the lock."""
        with MockRedisLock._lock_mutex:
            if self.key in MockRedisLock._locks and MockRedisLock._locks[self.key]['owner'] == id(self):
                del MockRedisLock._locks[self.key]
                self.acquired = False

class MockRedisClient:
    """Mock Redis client for testing."""
    
    def __init__(self):
        self.data = {}
        self.mutex = threading.Lock()
    
    async def ping(self):
        """Mock ping."""
        return True
    
    async def close(self):
        """Mock close."""
        pass
    
    def lock(self, key: str, timeout: int = 10, sleep: float = 0.01, blocking_timeout: float = 0.1):
        """Create a mock lock."""
        return MockRedisLock(key, timeout, sleep, blocking_timeout)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self.mutex:
            return key in self.data
    
    async def setex(self, key: str, timeout: int, value: str):
        """Set key with expiration."""
        with self.mutex:
            self.data[key] = {
                'value': value,
                'expires_at': time.time() + timeout
            }
    
    async def delete(self, key: str):
        """Delete key."""
        with self.mutex:
            if key in self.data:
                del self.data[key]
    
    def _cleanup_expired(self):
        """Remove expired keys."""
        current_time = time.time()
        expired_keys = []
        for key, data in self.data.items():
            if 'expires_at' in data and current_time > data['expires_at']:
                expired_keys.append(key)
        for key in expired_keys:
            del self.data[key]

class MockTacticalController:
    """Mock tactical controller for testing concurrency behavior."""
    
    def __init__(self, controller_id: str):
        self.controller_id = controller_id
        self.redis_client = MockRedisClient()
        self.decisions_processed = 0
        self.processing_times = []
        
        # Components (mocked)
        self.tactical_env = AsyncMock()
        self.decision_aggregator = AsyncMock()
        self.matrix_assembler = AsyncMock()
        
        # Mock aggregator to return predictable results
        self.decision_aggregator.aggregate_decisions = AsyncMock(return_value={
            "action": "long",
            "confidence": 0.8,
            "consensus_breakdown": {"long": 0.8, "hold": 0.2}
        })
    
    async def initialize(self):
        """Initialize controller."""
        await self.redis_client.ping()
    
    async def cleanup(self):
        """Cleanup controller."""
        await self.redis_client.close()
    
    async def on_synergy_detected(self, synergy_event: SynergyEvent) -> TacticalDecision:
        """
        Process SYNERGY_DETECTED event with concurrency protection.
        
        This method implements the same concurrency protection logic
        as the real controller but with mocked Redis.
        """
        timing = {}
        lock = None
        
        # Step 0: Acquire distributed lock to prevent race conditions
        lock_start = time.perf_counter()
        
        if not synergy_event.correlation_id:
            logger.error("Event missing correlation_id - cannot acquire lock")
            return self._create_error_response(synergy_event, "missing_correlation_id")
        
        lock_key = f"tactical:event_lock:{synergy_event.correlation_id}"
        lock = self.redis_client.lock(
            lock_key,
            timeout=10,  # 10 second timeout
            sleep=0.001,  # 1ms retry interval for faster testing
            blocking_timeout=0.05  # Don't block for more than 50ms
        )
        
        try:
            # Try to acquire lock with short timeout to maintain low latency
            if not await lock.acquire(blocking=False):
                logger.warning(
                    f"Event {synergy_event.correlation_id} is already being processed by another instance. Discarding."
                )
                return self._create_duplicate_event_response(synergy_event)
                
            timing['lock_acquisition_ms'] = (time.perf_counter() - lock_start) * 1000
            
            # Step 1: Idempotency check (now protected by lock)
            idempotency_start = time.perf_counter()
            if await self._is_duplicate_event(synergy_event):
                logger.warning(
                    f"Duplicate event detected, skipping processing: {synergy_event.correlation_id}"
                )
                return self._create_duplicate_event_response(synergy_event)
            
            # Mark event as being processed (atomic with lock)
            await self._mark_event_processing(synergy_event)
            timing['idempotency_check_ms'] = (time.perf_counter() - idempotency_start) * 1000
            
            # Step 2: Simulate processing (matrix fetch, agent decisions, etc.)
            processing_start = time.perf_counter()
            
            # Simulate some processing time
            await asyncio.sleep(0.001 + (hash(synergy_event.correlation_id) % 10) * 0.0001)  # 1-2ms
            
            timing['processing_ms'] = (time.perf_counter() - processing_start) * 1000
            
            # Create tactical decision
            decision = TacticalDecision(
                action="long",
                confidence=0.8,
                agent_votes=[],
                consensus_breakdown={"long": 0.8, "hold": 0.2},
                execution_command={
                    "action": "execute_trade",
                    "side": "BUY",
                    "correlation_id": synergy_event.correlation_id
                },
                timing=timing,
                correlation_id=synergy_event.correlation_id,
                timestamp=time.time()
            )
            
            # Mark event as completed
            await self._mark_event_completed(synergy_event, decision)
            
            # Update metrics
            self.decisions_processed += 1
            total_time = time.perf_counter() - lock_start
            self.processing_times.append(total_time)
            
            logger.info(
                f"✅ Controller {self.controller_id}: Event {synergy_event.correlation_id} processed successfully "
                f"in {total_time*1000:.2f}ms"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error processing synergy event {synergy_event.correlation_id}: {e}")
            return self._create_error_response(synergy_event, str(e))
            
        finally:
            # Always release the lock
            if lock and lock.acquired:
                try:
                    await lock.release()
                    timing['lock_total_ms'] = (time.perf_counter() - lock_start) * 1000
                    logger.debug(
                        f"Controller {self.controller_id}: Released lock for event {synergy_event.correlation_id} "
                        f"after {timing.get('lock_total_ms', 0):.2f}ms"
                    )
                except Exception as lock_error:
                    logger.error(
                        f"Controller {self.controller_id}: Failed to release lock for event "
                        f"{synergy_event.correlation_id}: {lock_error}"
                    )
    
    async def _is_duplicate_event(self, synergy_event: SynergyEvent) -> bool:
        """Check if this event has already been processed."""
        if not synergy_event.correlation_id:
            return False
        
        try:
            # Clean up expired keys first
            self.redis_client._cleanup_expired()
            
            # Check if event is already being processed
            processing_key = f"tactical:processing:{synergy_event.correlation_id}"
            is_processing = await self.redis_client.exists(processing_key)
            
            if is_processing:
                return True
            
            # Check if event has already been completed
            completed_key = f"tactical:completed:{synergy_event.correlation_id}"
            is_completed = await self.redis_client.exists(completed_key)
            
            return is_completed
            
        except Exception as e:
            logger.error(f"Failed to check event duplication: {e}")
            return False
    
    async def _mark_event_processing(self, synergy_event: SynergyEvent):
        """Mark event as being processed."""
        if not synergy_event.correlation_id:
            return
        
        try:
            processing_key = f"tactical:processing:{synergy_event.correlation_id}"
            await self.redis_client.setex(processing_key, 300, "1")  # 5 minute TTL
            
        except Exception as e:
            logger.error(f"Failed to mark event as processing: {e}")
    
    async def _mark_event_completed(self, synergy_event: SynergyEvent, decision: TacticalDecision):
        """Mark event as completed and remove from processing."""
        if not synergy_event.correlation_id:
            return
        
        try:
            processing_key = f"tactical:processing:{synergy_event.correlation_id}"
            completed_key = f"tactical:completed:{synergy_event.correlation_id}"
            
            # Remove from processing
            await self.redis_client.delete(processing_key)
            
            # Mark as completed with 1 hour TTL
            completion_data = {
                "action": decision.action,
                "confidence": decision.confidence,
                "completion_time": time.time(),
                "controller_id": self.controller_id
            }
            
            await self.redis_client.setex(completed_key, 3600, json.dumps(completion_data))
            
        except Exception as e:
            logger.error(f"Failed to mark event as completed: {e}")
    
    def _create_duplicate_event_response(self, synergy_event: SynergyEvent) -> TacticalDecision:
        """Create a response for duplicate events."""
        return TacticalDecision(
            action="hold",
            confidence=0.0,
            agent_votes=[],
            consensus_breakdown={"duplicate_event": 1.0},
            execution_command={
                "action": "hold",
                "reason": "duplicate_event",
                "correlation_id": synergy_event.correlation_id
            },
            timing={"duplicate_check_ms": 0.0},
            correlation_id=synergy_event.correlation_id,
            timestamp=time.time()
        )
    
    def _create_error_response(self, synergy_event: SynergyEvent, error: str) -> TacticalDecision:
        """Create an error response."""
        return TacticalDecision(
            action="hold",
            confidence=0.0,
            agent_votes=[],
            consensus_breakdown={"error": 1.0},
            execution_command={
                "action": "hold",
                "reason": f"error: {error}",
                "correlation_id": synergy_event.correlation_id
            },
            timing={"error_ms": 0.0},
            correlation_id=synergy_event.correlation_id,
            timestamp=time.time()
        )

@dataclass
class ConcurrencyTestResult:
    """Results from concurrency testing."""
    total_events_sent: int
    unique_events_processed: int
    duplicate_events_detected: int
    race_conditions_detected: int
    processing_times: List[float]
    errors: List[str]
    lock_conflicts: int
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate (should be exactly 1 event processed per correlation_id)."""
        if self.total_events_sent == 0:
            return 0.0
        return (1.0 if self.unique_events_processed == 1 and 
                self.duplicate_events_detected == self.total_events_sent - 1 else 0.0)

async def test_concurrent_identical_events(num_concurrent: int = 100) -> ConcurrencyTestResult:
    """
    🚨 CRITICAL TEST: Concurrent Identical Events Protection
    
    Spawns multiple identical events simultaneously to test:
    1. Distributed locking effectiveness
    2. Idempotency protection 
    3. Race condition prevention
    4. System stability under load
    """
    
    # Generate unique correlation ID for this test
    correlation_id = f"test_concurrent_{uuid.uuid4().hex}"
    
    # Create identical synergy event
    base_event = SynergyEvent(
        synergy_type="test_bullish_breakout",
        direction=1,
        confidence=0.85,
        signal_sequence=[],
        market_context={"test": True},
        correlation_id=correlation_id,
        timestamp=time.time()
    )
    
    # Track results
    processed_events = []
    duplicate_events = []
    errors = []
    processing_times = []
    lock_conflicts = 0
    
    # Create multiple controllers (simulating multiple instances)
    controllers = []
    for i in range(min(num_concurrent, 10)):  # Limit actual controllers to 10
        controller = MockTacticalController(f"controller_{i}")
        await controller.initialize()
        controllers.append(controller)
    
    async def process_event(controller_idx: int, event_idx: int):
        """Process a single event and track results."""
        start_time = time.perf_counter()
        controller = controllers[controller_idx % len(controllers)]
        
        try:
            # Create copy of event to avoid shared state
            event_copy = SynergyEvent(
                synergy_type=base_event.synergy_type,
                direction=base_event.direction,
                confidence=base_event.confidence,
                signal_sequence=base_event.signal_sequence.copy(),
                market_context=base_event.market_context.copy(),
                correlation_id=base_event.correlation_id,  # Same correlation_id!
                timestamp=base_event.timestamp
            )
            
            # Process the event
            decision = await controller.on_synergy_detected(event_copy)
            
            processing_time = time.perf_counter() - start_time
            processing_times.append(processing_time)
            
            # Check if this was actually processed or duplicate
            if decision.action != "hold" or "duplicate" not in decision.execution_command.get("reason", ""):
                processed_events.append({
                    "controller_idx": controller_idx,
                    "event_idx": event_idx,
                    "decision": decision,
                    "processing_time": processing_time
                })
                logger.info(f"✅ Event {event_idx} processed by controller {controller_idx}")
            else:
                duplicate_events.append({
                    "controller_idx": controller_idx,
                    "event_idx": event_idx,
                    "decision": decision,
                    "processing_time": processing_time
                })
                logger.info(f"🔄 Event {event_idx} detected as duplicate by controller {controller_idx}")
                
        except Exception as e:
            error_msg = f"Event {event_idx} on controller {controller_idx}: {str(e)}"
            errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            
            # Check if error is due to lock conflict
            if "lock" in str(e).lower() or "busy" in str(e).lower():
                nonlocal lock_conflicts
                lock_conflicts += 1
    
    # Launch all events simultaneously
    logger.info(f"🚀 Launching {num_concurrent} concurrent identical events with correlation_id: {correlation_id}")
    
    tasks = []
    for i in range(num_concurrent):
        controller_idx = i % len(controllers)
        task = asyncio.create_task(process_event(controller_idx, i))
        tasks.append(task)
    
    # Wait for all events to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Cleanup controllers
    for controller in controllers:
        await controller.cleanup()
    
    # Analyze results
    result = ConcurrencyTestResult(
        total_events_sent=num_concurrent,
        unique_events_processed=len(processed_events),
        duplicate_events_detected=len(duplicate_events),
        race_conditions_detected=max(0, len(processed_events) - 1),  # Should be 0
        processing_times=processing_times,
        errors=errors,
        lock_conflicts=lock_conflicts
    )
    
    # Log detailed results
    logger.info(f"📊 CONCURRENCY TEST RESULTS:")
    logger.info(f"   Events sent: {result.total_events_sent}")
    logger.info(f"   Unique processed: {result.unique_events_processed}")
    logger.info(f"   Duplicates detected: {result.duplicate_events_detected}")
    logger.info(f"   Race conditions: {result.race_conditions_detected}")
    logger.info(f"   Lock conflicts: {result.lock_conflicts}")
    logger.info(f"   Errors: {len(result.errors)}")
    logger.info(f"   Success rate: {result.success_rate * 100:.1f}%")
    
    if result.processing_times:
        avg_time = sum(result.processing_times) / len(result.processing_times)
        max_time = max(result.processing_times)
        logger.info(f"   Avg processing time: {avg_time*1000:.2f}ms")
        logger.info(f"   Max processing time: {max_time*1000:.2f}ms")
    
    return result

async def test_rapid_fire_different_events(num_events: int = 50) -> Dict[str, Any]:
    """
    🔥 RAPID FIRE TEST: Different Events Processing
    
    Tests system ability to handle many different events rapidly.
    """
    
    controller = MockTacticalController("rapid_fire_controller")
    await controller.initialize()
    
    processing_times = []
    errors = []
    
    async def process_unique_event(event_idx: int):
        """Process a unique event."""
        start_time = time.perf_counter()
        
        try:
            event = SynergyEvent(
                synergy_type=f"test_event_type_{event_idx % 5}",
                direction=1 if event_idx % 2 == 0 else -1,
                confidence=0.7 + (event_idx % 10) * 0.02,
                signal_sequence=[],
                market_context={"event_idx": event_idx},
                correlation_id=f"unique_event_{event_idx}_{uuid.uuid4().hex[:8]}",
                timestamp=time.time()
            )
            
            decision = await controller.on_synergy_detected(event)
            processing_time = time.perf_counter() - start_time
            processing_times.append(processing_time)
            
            logger.info(f"✅ Unique event {event_idx} processed in {processing_time*1000:.2f}ms")
            
        except Exception as e:
            errors.append(f"Event {event_idx}: {str(e)}")
            logger.error(f"❌ Event {event_idx} failed: {e}")
    
    # Launch all unique events
    logger.info(f"🔥 Launching {num_events} rapid-fire unique events")
    
    tasks = [asyncio.create_task(process_unique_event(i)) for i in range(num_events)]
    await asyncio.gather(*tasks, return_exceptions=True)
    
    await controller.cleanup()
    
    # Calculate statistics
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    success_rate = (len(processing_times) / num_events) if num_events > 0 else 0
    
    result = {
        "total_events": num_events,
        "successful_events": len(processing_times),
        "failed_events": len(errors),
        "success_rate": success_rate,
        "avg_processing_time_ms": avg_time * 1000,
        "max_processing_time_ms": max_time * 1000,
        "errors": errors
    }
    
    logger.info(f"📊 RAPID FIRE TEST RESULTS:")
    logger.info(f"   Success rate: {success_rate * 100:.1f}%")
    logger.info(f"   Avg time: {avg_time*1000:.2f}ms")
    logger.info(f"   Max time: {max_time*1000:.2f}ms")
    
    return result

async def run_comprehensive_concurrency_tests():
    """Run all concurrency tests."""
    
    logger.info("🚨 STARTING COMPREHENSIVE CONCURRENCY TESTING")
    logger.info("=" * 80)
    
    # Test 1: 100 concurrent identical events
    logger.info("🧪 TEST 1: 100 Concurrent Identical Events")
    result1 = await test_concurrent_identical_events(100)
    
    # Test 2: Rapid fire different events
    logger.info("\n🧪 TEST 2: 50 Rapid Fire Different Events")
    result2 = await test_rapid_fire_different_events(50)
    
    # Compile final results
    logger.info("\n" + "="*80)
    logger.info("🏆 FINAL CONCURRENCY TEST RESULTS")
    logger.info("="*80)
    
    # Determine overall pass/fail
    test1_pass = result1.success_rate == 1.0 and result1.race_conditions_detected == 0
    test2_pass = result2["success_rate"] >= 0.95 and result2["avg_processing_time_ms"] < 100
    
    overall_pass = test1_pass and test2_pass
    
    logger.info(f"✅ Test 1 (Identical Events): {'PASS' if test1_pass else 'FAIL'}")
    logger.info(f"✅ Test 2 (Rapid Fire): {'PASS' if test2_pass else 'FAIL'}")
    logger.info(f"🎯 OVERALL RESULT: {'PASS' if overall_pass else 'FAIL'}")
    
    if overall_pass:
        logger.info("🛡️ CONCURRENCY PROTECTION: SYSTEM IS BULLETPROOF")
    else:
        logger.error("🚨 CONCURRENCY PROTECTION: VULNERABILITIES DETECTED")
    
    return {
        "test_1_identical_events": result1,
        "test_2_rapid_fire": result2,
        "overall_pass": overall_pass,
        "test_1_pass": test1_pass,
        "test_2_pass": test2_pass
    }

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_comprehensive_concurrency_tests())