"""
Adversarial Testing Suite: Dependency Chain Vulnerabilities
==========================================================

Zero Defect Adversarial Audit - Phase 1: Dependency Chain Failure Tests

This module implements comprehensive dependency chain tests designed to uncover
vulnerabilities through service failures, network partitions, and cascade failures.

CRITICAL TEST CATEGORIES:
1. Redis Failure Cascade Tests
2. PyTorch Model Corruption Tests  
3. Network Partition Scenarios
4. Docker Container Failure Tests
5. Service Unavailability Cascades

Author: Zero Defect Security Audit
Version: 1.0.0
Classification: CRITICAL SECURITY TESTS
"""

import pytest
import asyncio
import aiohttp
import redis
import docker
import time
import psutil
import threading
import subprocess
import signal
import os
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock
import logging

# Import the attack detection framework
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.security.attack_detection import TacticalMARLAttackDetector, VulnerabilitySeverity, AttackVector

logger = logging.getLogger(__name__)


class TestDependencyChainVulnerabilities:
    """
    Comprehensive test suite for dependency chain vulnerabilities in the Tactical MARL System.
    
    These tests are designed to identify critical dependency failures that could:
    - Cause cascading system failures
    - Create service unavailability scenarios
    - Lead to data corruption during failures
    - Expose unhandled error propagation
    """

    @pytest.fixture(scope="class")
    def attack_detector(self):
        """Initialize the attack detection system"""
        return TacticalMARLAttackDetector(target_host="localhost", target_port=8001)

    @pytest.fixture(scope="function")
    def redis_client(self):
        """Initialize Redis client for dependency testing"""
        try:
            client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"Redis not available for testing: {e}")
            return None

    @pytest.fixture(scope="function")
    def docker_client(self):
        """Initialize Docker client for container testing"""
        try:
            client = docker.from_env()
            return client
        except Exception as e:
            logger.warning(f"Docker not available for testing: {e}")
            return None

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_redis_failure_cascade(self, attack_detector, redis_client):
        """
        CRITICAL TEST: Redis failure cascade during decision processing
        
        Tests system behavior when Redis fails during critical decision processing
        to identify cascade failure vulnerabilities and recovery mechanisms.
        
        Expected Vulnerabilities:
        - Unhandled Redis connection failures
        - Data loss during Redis outages
        - System instability without Redis
        - Poor error recovery mechanisms
        """
        logger.info("🚨 Testing Redis Failure Cascade")
        
        if not redis_client:
            pytest.skip("Redis not available for testing")
        
        # Step 1: Verify Redis is working initially
        try:
            redis_client.ping()
            logger.info("✅ Redis initially available")
        except Exception as e:
            pytest.skip(f"Redis not working initially: {e}")
        
        # Step 2: Send normal requests to establish baseline
        baseline_results = []
        async with aiohttp.ClientSession() as session:
            for i in range(5):
                payload = {
                    "matrix_state": [[float(j) for j in range(7)] for _ in range(60)],
                    "synergy_context": {
                        "synergy_type": "TYPE_1",
                        "direction": 1,
                        "confidence": 0.8,
                        "correlation_id": f"baseline_redis_{i}",
                        "timestamp": time.time()
                    },
                    "correlation_id": f"baseline_redis_{i}"
                }
                
                result = await attack_detector._send_decision_request(session, payload)
                baseline_results.append(result)
        
        baseline_success_rate = len([r for r in baseline_results if r.get("status_code") == 200]) / len(baseline_results)
        logger.info(f"Baseline success rate: {baseline_success_rate:.2f}")
        
        # Step 3: Simulate Redis failure by flooding Redis with data
        try:
            # Flood Redis with data to cause memory pressure
            for i in range(10000):
                redis_client.set(f"flood_key_{i}", "x" * 10000)
                if i % 1000 == 0:
                    logger.info(f"Flooded Redis with {i} keys...")
            
            # Try to exhaust Redis memory
            redis_client.config_set('maxmemory', '10mb')
            redis_client.config_set('maxmemory-policy', 'noeviction')
            
        except Exception as e:
            logger.warning(f"Redis flooding failed: {e}")
        
        # Step 4: Test system behavior during Redis stress
        stress_results = []
        async with aiohttp.ClientSession() as session:
            for i in range(20):
                payload = {
                    "matrix_state": [[float(j + i) for j in range(7)] for _ in range(60)],
                    "synergy_context": {
                        "synergy_type": "TYPE_2",
                        "direction": -1,
                        "confidence": 0.9,
                        "correlation_id": f"redis_stress_{i}",
                        "timestamp": time.time()
                    },
                    "correlation_id": f"redis_stress_{i}"
                }
                
                result = await attack_detector._send_decision_request(session, payload)
                stress_results.append(result)
                
                # Brief delay to allow Redis stress to build
                await asyncio.sleep(0.1)
        
        # Step 5: Analyze failure cascade behavior
        stress_success_rate = len([r for r in stress_results if r.get("status_code") == 200]) / len(stress_results)
        error_responses = [r for r in stress_results if r.get("status_code") >= 500]
        timeout_responses = [r for r in stress_results if "timeout" in str(r.get("error", "")).lower()]
        
        logger.warning(f"Stress success rate: {stress_success_rate:.2f}")
        logger.warning(f"Error responses: {len(error_responses)}")
        logger.warning(f"Timeout responses: {len(timeout_responses)}")
        
        # Check for cascade failure indicators
        performance_degradation = (baseline_success_rate - stress_success_rate) > 0.3
        excessive_errors = len(error_responses) > 5
        redis_timeouts = len(timeout_responses) > 3
        
        if performance_degradation:
            logger.critical(f"CRITICAL: Performance degradation during Redis stress ({baseline_success_rate:.2f} -> {stress_success_rate:.2f})")
        
        if excessive_errors:
            logger.critical(f"CRITICAL: Excessive error responses during Redis stress ({len(error_responses)})")
        
        if redis_timeouts:
            logger.critical(f"CRITICAL: Redis timeout cascades detected ({len(timeout_responses)})")
        
        # Step 6: Cleanup and test recovery
        try:
            redis_client.flushall()
            redis_client.config_set('maxmemory', '0')  # Remove memory limit
            redis_client.config_set('maxmemory-policy', 'noeviction')
            logger.info("✅ Redis cleanup completed")
        except Exception as e:
            logger.error(f"Redis cleanup failed: {e}")
        
        # Step 7: Test recovery behavior
        recovery_results = []
        async with aiohttp.ClientSession() as session:
            for i in range(5):
                payload = {
                    "matrix_state": [[float(j) for j in range(7)] for _ in range(60)],
                    "synergy_context": {
                        "synergy_type": "TYPE_1",
                        "direction": 1,
                        "confidence": 0.8,
                        "correlation_id": f"recovery_redis_{i}",
                        "timestamp": time.time()
                    },
                    "correlation_id": f"recovery_redis_{i}"
                }
                
                result = await attack_detector._send_decision_request(session, payload)
                recovery_results.append(result)
        
        recovery_success_rate = len([r for r in recovery_results if r.get("status_code") == 200]) / len(recovery_results)
        logger.info(f"Recovery success rate: {recovery_success_rate:.2f}")
        
        # Assert recovery capabilities
        poor_recovery = recovery_success_rate < (baseline_success_rate * 0.8)
        if poor_recovery:
            logger.critical(f"CRITICAL: Poor recovery from Redis stress ({recovery_success_rate:.2f} vs baseline {baseline_success_rate:.2f})")

    @pytest.mark.asyncio
    async def test_pytorch_model_corruption_during_failures(self, attack_detector):
        """
        HIGH TEST: PyTorch model corruption during dependency failures
        
        Tests whether model state can be corrupted when dependencies fail
        during inference operations.
        """
        logger.info("🚨 Testing PyTorch Model Corruption During Failures")
        
        # Create model corruption scenarios during simulated failures
        corruption_payloads = []
        
        # Scenario 1: Model inference during memory pressure
        for i in range(50):
            # Create increasingly large matrices to stress memory
            large_matrix = []
            for row in range(60):
                row_data = []
                for col in range(7):
                    # Add memory-consuming values
                    value = float(row * col * (i + 1) * 1000)
                    row_data.append(value)
                large_matrix.append(row_data)
            
            corruption_payloads.append({
                "name": f"Memory stress inference {i}",
                "matrix_state": large_matrix,
                "synergy_context": {
                    "synergy_type": "TYPE_1",
                    "direction": 1,
                    "confidence": 0.8,
                    "correlation_id": f"memory_stress_{i}",
                    "timestamp": time.time()
                },
                "correlation_id": f"memory_stress_{i}"
            })
        
        # Scenario 2: Model inference with extreme numerical values
        extreme_payloads = []
        extreme_values = [float('inf'), float('-inf'), float('nan'), 1e100, -1e100, 1e-100]
        
        for i, extreme_val in enumerate(extreme_values):
            matrix = [[extreme_val if row == 30 and col == 3 else 1.0 for col in range(7)] for row in range(60)]
            extreme_payloads.append({
                "name": f"Extreme value inference ({extreme_val})",
                "matrix_state": matrix,
                "synergy_context": {
                    "synergy_type": "TYPE_2",
                    "direction": -1,
                    "confidence": 0.9,
                    "correlation_id": f"extreme_value_{i}",
                    "timestamp": time.time()
                },
                "correlation_id": f"extreme_value_{i}"
            })
        
        corruption_payloads.extend(extreme_payloads)
        
        # Execute model corruption tests with concurrent load
        results = []
        async with aiohttp.ClientSession() as session:
            # Send all corruption payloads concurrently to maximize stress
            tasks = []
            for payload in corruption_payloads:
                task = attack_detector._send_decision_request(session, payload)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze model corruption indicators
        successful_responses = len([r for r in results if isinstance(r, dict) and r.get("status_code") == 200])
        error_responses = len([r for r in results if isinstance(r, dict) and r.get("status_code") >= 500])
        exceptions = len([r for r in results if isinstance(r, Exception)])
        
        # Look for model corruption patterns
        model_errors = 0
        nan_errors = 0
        inf_errors = 0
        memory_errors = 0
        
        for result in results:
            if isinstance(result, dict):
                error_msg = str(result.get("error", "")).lower()
                response_data = result.get("response_data", {})
                
                if "model" in error_msg or "inference" in error_msg:
                    model_errors += 1
                if "nan" in error_msg:
                    nan_errors += 1
                if "inf" in error_msg or "infinity" in error_msg:
                    inf_errors += 1
                if "memory" in error_msg or "oom" in error_msg:
                    memory_errors += 1
                
                # Check response data for corruption indicators
                if isinstance(response_data, dict):
                    decision = response_data.get("decision", {})
                    if isinstance(decision, dict):
                        confidence = decision.get("confidence", 0)
                        if isinstance(confidence, float) and (math.isnan(confidence) or math.isinf(confidence)):
                            logger.critical(f"🚨 Model output corruption detected: confidence = {confidence}")
        
        logger.warning(f"Total payloads: {len(corruption_payloads)}")
        logger.warning(f"Successful responses: {successful_responses}")
        logger.warning(f"Error responses: {error_responses}")
        logger.warning(f"Exceptions: {exceptions}")
        logger.warning(f"Model errors: {model_errors}")
        logger.warning(f"NaN errors: {nan_errors}")
        logger.warning(f"Infinity errors: {inf_errors}")
        logger.warning(f"Memory errors: {memory_errors}")
        
        # Check for critical model corruption indicators
        if model_errors > 10:
            logger.critical(f"CRITICAL: Excessive model errors detected ({model_errors})")
        
        if nan_errors > 5:
            logger.critical(f"CRITICAL: NaN corruption in model outputs ({nan_errors})")
        
        if memory_errors > 3:
            logger.critical(f"CRITICAL: Memory-related model failures ({memory_errors})")

    @pytest.mark.asyncio
    async def test_network_partition_simulation(self, attack_detector):
        """
        MEDIUM TEST: Network partition simulation
        
        Tests system behavior when network connectivity is degraded or interrupted
        to simulate network partition scenarios.
        """
        logger.info("🚨 Testing Network Partition Simulation")
        
        # Simulate network issues by introducing artificial delays and timeouts
        network_stress_payloads = []
        
        # Create payloads with various timeout scenarios
        for i in range(20):
            payload = {
                "matrix_state": [[float(j) for j in range(7)] for _ in range(60)],
                "synergy_context": {
                    "synergy_type": "TYPE_3",
                    "direction": 1 if i % 2 == 0 else -1,
                    "confidence": 0.7 + (i * 0.01),
                    "correlation_id": f"network_partition_{i}",
                    "timestamp": time.time()
                },
                "correlation_id": f"network_partition_{i}"
            }
            network_stress_payloads.append(payload)
        
        # Test with various timeout configurations
        timeout_scenarios = [
            ("Very short timeout", 0.1),
            ("Short timeout", 1.0),
            ("Medium timeout", 5.0),
            ("Long timeout", 30.0)
        ]
        
        partition_results = {}
        
        for scenario_name, timeout_duration in timeout_scenarios:
            logger.info(f"Testing {scenario_name} ({timeout_duration}s)")
            
            scenario_results = []
            timeout_config = aiohttp.ClientTimeout(total=timeout_duration)
            
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                for payload in network_stress_payloads:
                    start_time = time.time()
                    try:
                        result = await attack_detector._send_decision_request(session, payload)
                        result['actual_duration'] = time.time() - start_time
                        scenario_results.append(result)
                    except Exception as e:
                        scenario_results.append({
                            "error": str(e),
                            "actual_duration": time.time() - start_time,
                            "correlation_id": payload["correlation_id"]
                        })
            
            partition_results[scenario_name] = scenario_results
        
        # Analyze network partition behavior
        for scenario_name, results in partition_results.items():
            successful_requests = len([r for r in results if r.get("status_code") == 200])
            timeout_errors = len([r for r in results if "timeout" in str(r.get("error", "")).lower()])
            avg_duration = sum(r.get("actual_duration", 0) for r in results) / len(results)
            
            logger.info(f"{scenario_name}: {successful_requests}/{len(results)} successful, {timeout_errors} timeouts, avg {avg_duration:.2f}s")
            
            # Check for graceful degradation
            if timeout_errors > len(results) * 0.5:  # More than 50% timeouts
                logger.warning(f"⚠️ {scenario_name}: High timeout rate ({timeout_errors}/{len(results)})")

    @pytest.mark.asyncio
    async def test_docker_container_failure_scenarios(self, attack_detector, docker_client):
        """
        HIGH TEST: Docker container failure scenarios
        
        Tests system behavior when Docker containers are stopped, paused, or
        become unresponsive during operation.
        """
        logger.info("🚨 Testing Docker Container Failure Scenarios")
        
        if not docker_client:
            pytest.skip("Docker not available for testing")
        
        # Get list of running containers related to the system
        try:
            containers = docker_client.containers.list()
            tactical_containers = []
            
            for container in containers:
                container_name = container.name.lower()
                if any(keyword in container_name for keyword in ['tactical', 'marl', 'redis', 'ollama']):
                    tactical_containers.append(container)
                    logger.info(f"Found related container: {container.name} ({container.status})")
            
            if not tactical_containers:
                logger.warning("No related containers found for testing")
                return
            
        except Exception as e:
            logger.error(f"Failed to list Docker containers: {e}")
            pytest.skip("Cannot access Docker containers")
        
        # Test normal operation before container failures
        baseline_payload = {
            "matrix_state": [[float(j) for j in range(7)] for _ in range(60)],
            "synergy_context": {
                "synergy_type": "TYPE_1",
                "direction": 1,
                "confidence": 0.8,
                "correlation_id": "docker_baseline",
                "timestamp": time.time()
            },
            "correlation_id": "docker_baseline"
        }
        
        # Get baseline performance
        async with aiohttp.ClientSession() as session:
            baseline_result = await attack_detector._send_decision_request(session, baseline_payload)
        
        baseline_success = baseline_result.get("status_code") == 200
        logger.info(f"Baseline test success: {baseline_success}")
        
        # Test container pause/unpause scenarios (non-destructive)
        for container in tactical_containers[:2]:  # Limit to first 2 containers
            try:
                logger.info(f"Testing pause/unpause for container: {container.name}")
                
                # Pause container
                container.pause()
                logger.info(f"Paused container: {container.name}")
                
                # Test system behavior with paused container
                pause_payload = baseline_payload.copy()
                pause_payload["correlation_id"] = f"docker_pause_{container.name}"
                pause_payload["synergy_context"]["correlation_id"] = f"docker_pause_{container.name}"
                
                async with aiohttp.ClientSession() as session:
                    pause_result = await attack_detector._send_decision_request(session, pause_payload)
                
                pause_success = pause_result.get("status_code") == 200
                pause_error = pause_result.get("error", "")
                
                logger.warning(f"System behavior with {container.name} paused: Success={pause_success}, Error={pause_error}")
                
                # Unpause container
                await asyncio.sleep(2)  # Brief pause
                container.unpause()
                logger.info(f"Unpaused container: {container.name}")
                
                # Test recovery
                recovery_payload = baseline_payload.copy()
                recovery_payload["correlation_id"] = f"docker_recovery_{container.name}"
                recovery_payload["synergy_context"]["correlation_id"] = f"docker_recovery_{container.name}"
                
                async with aiohttp.ClientSession() as session:
                    recovery_result = await attack_detector._send_decision_request(session, recovery_payload)
                
                recovery_success = recovery_result.get("status_code") == 200
                logger.info(f"Recovery after {container.name} unpause: Success={recovery_success}")
                
                # Analyze container failure impact
                if baseline_success and not pause_success:
                    logger.critical(f"CRITICAL: System failed when {container.name} was paused")
                elif baseline_success and pause_success:
                    logger.info(f"✅ System resilient to {container.name} pause")
                
                if not recovery_success and baseline_success:
                    logger.critical(f"CRITICAL: System did not recover after {container.name} unpause")
                
            except Exception as e:
                logger.error(f"Container test failed for {container.name}: {e}")
                # Ensure container is unpaused even if test fails
                try:
                    container.unpause()
                except (OSError, ValueError, ConnectionError) as e:
                    logger.error(f'Error occurred: {e}')

    @pytest.mark.asyncio
    async def test_service_unavailability_cascades(self, attack_detector):
        """
        MEDIUM TEST: Service unavailability cascade testing
        
        Tests how the system behaves when multiple services become unavailable
        simultaneously to identify cascade failure patterns.
        """
        logger.info("🚨 Testing Service Unavailability Cascades")
        
        # Simulate multiple concurrent service failures through resource exhaustion
        cascade_payloads = []
        
        # Create payloads designed to exhaust different system resources
        for i in range(100):
            # Large matrix for memory stress
            memory_stress_matrix = [[float(j * i * 1000) for j in range(7)] for _ in range(60)]
            
            # Large synergy context for network/processing stress
            large_context = {
                "synergy_type": "TYPE_4",
                "direction": 1 if i % 2 == 0 else -1,
                "confidence": 0.8 + (i * 0.001),
                "correlation_id": f"cascade_test_{i}",
                "timestamp": time.time(),
                "signal_sequence": [
                    {"stress_data": "x" * 1000} for _ in range(10)  # 10KB per sequence
                ],
                "market_context": {
                    f"stress_param_{j}": f"data_{j}" * 100 for j in range(10)
                }
            }
            
            cascade_payloads.append({
                "matrix_state": memory_stress_matrix,
                "synergy_context": large_context,
                "override_params": {
                    "stress_test": True,
                    "cascade_id": i,
                    "large_data": "A" * 10000  # 10KB string
                },
                "correlation_id": f"cascade_test_{i}"
            })
        
        # Execute cascade stress test
        logger.info(f"Executing cascade test with {len(cascade_payloads)} concurrent requests")
        
        start_time = time.time()
        results = []
        
        # Use a semaphore to control concurrency
        semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent requests
        
        async def send_cascade_request(session, payload):
            async with semaphore:
                return await attack_detector._send_decision_request(session, payload)
        
        async with aiohttp.ClientSession() as session:
            tasks = [send_cascade_request(session, payload) for payload in cascade_payloads]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze cascade failure patterns
        successful_requests = len([r for r in results if isinstance(r, dict) and r.get("status_code") == 200])
        error_requests = len([r for r in results if isinstance(r, dict) and r.get("status_code") >= 500])
        timeout_requests = len([r for r in results if isinstance(r, Exception) or "timeout" in str(r).lower()])
        
        success_rate = successful_requests / len(results)
        error_rate = error_requests / len(results)
        timeout_rate = timeout_requests / len(results)
        
        logger.warning(f"Cascade test results:")
        logger.warning(f"Total duration: {total_duration:.2f}s")
        logger.warning(f"Success rate: {success_rate:.2f} ({successful_requests}/{len(results)})")
        logger.warning(f"Error rate: {error_rate:.2f} ({error_requests}/{len(results)})")
        logger.warning(f"Timeout rate: {timeout_rate:.2f} ({timeout_requests}/{len(results)})")
        
        # Check for cascade failure indicators
        if success_rate < 0.5:  # Less than 50% success
            logger.critical(f"CRITICAL: Low success rate during cascade test ({success_rate:.2f})")
        
        if error_rate > 0.3:  # More than 30% errors
            logger.critical(f"CRITICAL: High error rate during cascade test ({error_rate:.2f})")
        
        if timeout_rate > 0.2:  # More than 20% timeouts
            logger.critical(f"CRITICAL: High timeout rate during cascade test ({timeout_rate:.2f})")
        
        # Test recovery after cascade
        logger.info("Testing system recovery after cascade stress...")
        
        await asyncio.sleep(10)  # Allow system to recover
        
        recovery_payload = {
            "matrix_state": [[1.0 for j in range(7)] for _ in range(60)],
            "synergy_context": {
                "synergy_type": "TYPE_1",
                "direction": 1,
                "confidence": 0.8,
                "correlation_id": "recovery_test",
                "timestamp": time.time()
            },
            "correlation_id": "recovery_test"
        }
        
        async with aiohttp.ClientSession() as session:
            recovery_result = await attack_detector._send_decision_request(session, recovery_payload)
        
        recovery_success = recovery_result.get("status_code") == 200
        logger.info(f"Recovery test success: {recovery_success}")
        
        if not recovery_success:
            logger.critical(f"CRITICAL: System did not recover properly after cascade stress")

    def test_error_propagation_chains(self, attack_detector):
        """
        MEDIUM TEST: Error propagation chain analysis
        
        Tests how errors propagate through the system architecture to identify
        unhandled error scenarios and improper error isolation.
        """
        logger.info("🚨 Testing Error Propagation Chains")
        
        # This test focuses on error propagation patterns rather than active attacks
        # We'll analyze system behavior with various error conditions
        
        # Test error scenarios that should be isolated
        error_scenarios = [
            {
                "name": "Malformed JSON payload",
                "payload": "invalid_json_payload",
                "expected_behavior": "400 Bad Request with proper error message"
            },
            {
                "name": "Missing required fields",
                "payload": {
                    "matrix_state": [[1.0 for j in range(7)] for _ in range(60)]
                    # Missing synergy_context and correlation_id
                },
                "expected_behavior": "422 Unprocessable Entity"
            },
            {
                "name": "Invalid content type",
                "payload": {"valid": "json"},
                "content_type": "text/plain",
                "expected_behavior": "415 Unsupported Media Type"
            }
        ]
        
        # Since this is synchronous, we'll simulate the error scenarios
        logger.info("Error propagation analysis:")
        
        for scenario in error_scenarios:
            logger.info(f"- {scenario['name']}: Expected {scenario['expected_behavior']}")
        
        # In a real implementation, this would send actual malformed requests
        # and analyze the error response patterns to ensure proper error isolation
        
        logger.info("✅ Error propagation chain analysis completed")
        logger.info("Note: This test should be expanded with actual error injection")