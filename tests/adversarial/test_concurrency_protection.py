#!/usr/bin/env python3
"""
AGENT 4 RED TEAM CERTIFIER: Concurrency Protection Test
Mission: Aegis - Tactical MARL Final Security Validation

This test validates that the tactical MARL system properly handles
concurrent identical events using distributed locking mechanisms.

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
from typing import List, Dict, Any
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.asyncio as redis

# Import tactical controller
from src.tactical.controller import TacticalMARLController, SynergyEvent, TacticalDecision

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class TacticalConcurrencyTester:
    """Advanced concurrency testing for tactical MARL system."""
    
    def __init__(self):
        self.redis_client = None
        self.test_results = []
        
    async def setup(self):
        """Setup test environment."""
        try:
            # Connect to Redis test database
            self.redis_client = redis.from_url("redis://localhost:6379/3")  # Use test DB
            await self.redis_client.ping()
            
            # Clear test database
            await self.redis_client.flushdb()
            
            logger.info("🔧 Concurrency test environment initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            raise
    
    async def teardown(self):
        """Cleanup test environment."""
        if self.redis_client:
            await self.redis_client.flushdb()
            await self.redis_client.close()
    
    async def test_concurrent_identical_events(self, num_concurrent: int = 100) -> ConcurrencyTestResult:
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
            controller = TacticalMARLController(redis_url="redis://localhost:6379/3")
            # Mock the heavy components to focus on concurrency logic
            controller.tactical_env = AsyncMock()
            controller.decision_aggregator = AsyncMock()
            controller.matrix_assembler = AsyncMock()
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
                
                # Mock the decision aggregator to return predictable results
                controller.decision_aggregator.aggregate_decisions = AsyncMock(return_value={
                    "action": "long",
                    "confidence": 0.8,
                    "consensus_breakdown": {"long": 0.8, "hold": 0.2}
                })
                
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
    
    async def test_rapid_fire_different_events(self, num_events: int = 50) -> Dict[str, Any]:
        """
        🔥 RAPID FIRE TEST: Different Events Processing
        
        Tests system ability to handle many different events rapidly.
        """
        
        controller = TacticalMARLController(redis_url="redis://localhost:6379/3")
        controller.tactical_env = AsyncMock()
        controller.decision_aggregator = AsyncMock()
        controller.matrix_assembler = AsyncMock()
        await controller.initialize()
        
        # Mock decision aggregator
        controller.decision_aggregator.aggregate_decisions = AsyncMock(return_value={
            "action": "long",
            "confidence": 0.8,
            "consensus_breakdown": {"long": 0.8, "hold": 0.2}
        })
        
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
    
    async def test_lock_timeout_behavior(self) -> Dict[str, Any]:
        """
        ⏱️ LOCK TIMEOUT TEST: Lock Timeout Behavior
        
        Tests behavior when locks are held too long.
        """
        
        controller1 = TacticalMARLController(redis_url="redis://localhost:6379/3")
        controller2 = TacticalMARLController(redis_url="redis://localhost:6379/3")
        
        for controller in [controller1, controller2]:
            controller.tactical_env = AsyncMock()
            controller.decision_aggregator = AsyncMock()
            controller.matrix_assembler = AsyncMock()
            await controller.initialize()
        
        correlation_id = f"lock_timeout_test_{uuid.uuid4().hex}"
        
        # Mock a slow decision aggregator for controller1
        async def slow_aggregate(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate slow processing
            return {
                "action": "long",
                "confidence": 0.8,
                "consensus_breakdown": {"long": 0.8, "hold": 0.2}
            }
        
        controller1.decision_aggregator.aggregate_decisions = slow_aggregate
        controller2.decision_aggregator.aggregate_decisions = AsyncMock(return_value={
            "action": "short",
            "confidence": 0.7,
            "consensus_breakdown": {"short": 0.7, "hold": 0.3}
        })
        
        event = SynergyEvent(
            synergy_type="lock_timeout_test",
            direction=1,
            confidence=0.8,
            signal_sequence=[],
            market_context={},
            correlation_id=correlation_id,
            timestamp=time.time()
        )
        
        # Start both controllers processing the same event
        task1 = asyncio.create_task(controller1.on_synergy_detected(event))
        
        # Small delay to ensure controller1 gets the lock first
        await asyncio.sleep(0.1)
        
        task2 = asyncio.create_task(controller2.on_synergy_detected(event))
        
        # Wait for both to complete
        results = await asyncio.gather(task1, task2, return_exceptions=True)
        
        await controller1.cleanup()
        await controller2.cleanup()
        
        # Analyze results
        processed_count = 0
        duplicate_count = 0
        error_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
            elif isinstance(result, TacticalDecision):
                if result.action != "hold" or "duplicate" not in result.execution_command.get("reason", ""):
                    processed_count += 1
                else:
                    duplicate_count += 1
        
        test_result = {
            "processed_events": processed_count,
            "duplicate_events": duplicate_count,
            "errors": error_count,
            "lock_behavior": "correct" if processed_count == 1 and duplicate_count >= 1 else "incorrect"
        }
        
        logger.info(f"📊 LOCK TIMEOUT TEST RESULTS:")
        logger.info(f"   Processed: {processed_count}")
        logger.info(f"   Duplicates: {duplicate_count}")
        logger.info(f"   Errors: {error_count}")
        logger.info(f"   Lock behavior: {test_result['lock_behavior']}")
        
        return test_result

async def run_comprehensive_concurrency_tests():
    """Run all concurrency tests."""
    
    tester = TacticalConcurrencyTester()
    await tester.setup()
    
    try:
        logger.info("🚨 STARTING COMPREHENSIVE CONCURRENCY TESTING")
        logger.info("=" * 80)
        
        # Test 1: 100 concurrent identical events
        logger.info("🧪 TEST 1: 100 Concurrent Identical Events")
        result1 = await tester.test_concurrent_identical_events(100)
        
        # Test 2: Rapid fire different events
        logger.info("\n🧪 TEST 2: 50 Rapid Fire Different Events")
        result2 = await tester.test_rapid_fire_different_events(50)
        
        # Test 3: Lock timeout behavior
        logger.info("\n🧪 TEST 3: Lock Timeout Behavior")
        result3 = await tester.test_lock_timeout_behavior()
        
        # Compile final results
        logger.info("\n" + "="*80)
        logger.info("🏆 FINAL CONCURRENCY TEST RESULTS")
        logger.info("="*80)
        
        # Determine overall pass/fail
        test1_pass = result1.success_rate == 1.0 and result1.race_conditions_detected == 0
        test2_pass = result2["success_rate"] >= 0.95 and result2["avg_processing_time_ms"] < 100
        test3_pass = result3["lock_behavior"] == "correct"
        
        overall_pass = test1_pass and test2_pass and test3_pass
        
        logger.info(f"✅ Test 1 (Identical Events): {'PASS' if test1_pass else 'FAIL'}")
        logger.info(f"✅ Test 2 (Rapid Fire): {'PASS' if test2_pass else 'FAIL'}")
        logger.info(f"✅ Test 3 (Lock Timeout): {'PASS' if test3_pass else 'FAIL'}")
        logger.info(f"🎯 OVERALL RESULT: {'PASS' if overall_pass else 'FAIL'}")
        
        if overall_pass:
            logger.info("🛡️ CONCURRENCY PROTECTION: SYSTEM IS BULLETPROOF")
        else:
            logger.error("🚨 CONCURRENCY PROTECTION: VULNERABILITIES DETECTED")
        
        return {
            "test_1_identical_events": result1,
            "test_2_rapid_fire": result2,
            "test_3_lock_timeout": result3,
            "overall_pass": overall_pass,
            "test_1_pass": test1_pass,
            "test_2_pass": test2_pass,
            "test_3_pass": test3_pass
        }
        
    finally:
        await tester.teardown()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_comprehensive_concurrency_tests())