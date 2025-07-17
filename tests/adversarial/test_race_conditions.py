"""
Adversarial Testing Suite: Race Condition Vulnerabilities
========================================================

Zero Defect Adversarial Audit - Phase 1: Race Condition & Concurrency Tests

This module implements comprehensive race condition tests designed to uncover
vulnerabilities in the Tactical MARL System under high concurrency scenarios.

CRITICAL TEST CATEGORIES:
1. Concurrent Decision Processing Attacks
2. Model State Corruption Tests  
3. Shared Resource Contention Scenarios
4. Memory Leak Exploitation Tests
5. Response Consistency Validation

Author: Zero Defect Security Audit
Version: 1.0.0
Classification: CRITICAL SECURITY TESTS
"""

import pytest
import asyncio
import aiohttp
import time
import threading
import concurrent.futures
import numpy as np
import psutil
import gc
import weakref
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging

# Import the attack detection framework
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.security.attack_detection import TacticalMARLAttackDetector, VulnerabilitySeverity, AttackVector

logger = logging.getLogger(__name__)


class TestRaceConditionVulnerabilities:
    """
    Comprehensive test suite for race condition vulnerabilities in the Tactical MARL System.
    
    These tests are designed to identify critical race conditions that could:
    - Corrupt decision processing logic
    - Cause model state inconsistencies  
    - Create resource deadlocks
    - Lead to memory leaks and system instability
    """

    @pytest.fixture(scope="class")
    def attack_detector(self):
        """Initialize the attack detection system"""
        return TacticalMARLAttackDetector(target_host="localhost", target_port=8001)

    @pytest.fixture(scope="function")
    def baseline_metrics(self):
        """Capture baseline system metrics before each test"""
        memory = psutil.virtual_memory()
        return {
            "memory_used_mb": memory.used / (1024 * 1024),
            "memory_percent": memory.percent,
            "cpu_percent": psutil.cpu_percent(interval=1),
            "timestamp": time.time()
        }

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minute timeout
    async def test_concurrent_decision_endpoint_flooding(self, attack_detector, baseline_metrics):
        """
        CRITICAL TEST: Concurrent decision endpoint flooding
        
        Tests system behavior under extreme concurrent load (1500+ simultaneous requests)
        to identify race conditions in decision processing pipeline.
        
        Expected Vulnerabilities:
        - Request processing race conditions
        - Resource contention failures
        - Response time degradation attacks
        """
        logger.info("🚨 Testing Concurrent Decision Endpoint Flooding")
        
        # Execute the attack
        attack_result = await attack_detector._race_attack_concurrent_decisions()
        
        # Verify attack execution
        assert attack_result.attack_name == "Concurrent Decision Endpoint Flooding"
        assert attack_result.execution_time > 0
        
        # Analyze performance impact
        if attack_result.performance_impact:
            total_requests = attack_result.performance_impact.get("total_requests", 0)
            error_patterns = attack_result.performance_impact.get("error_patterns", {})
            status_codes = attack_result.performance_impact.get("status_codes", {})
            
            # Log critical findings
            logger.warning(f"Processed {total_requests} concurrent requests")
            logger.warning(f"Error patterns detected: {error_patterns}")
            logger.warning(f"Status code distribution: {status_codes}")
            
            # Check for critical race condition indicators
            timeout_errors = error_patterns.get('TimeoutError', 0)
            server_errors = status_codes.get(500, 0)
            
            # Assert critical vulnerabilities are detected
            if timeout_errors > 50:
                logger.critical(f"CRITICAL: Excessive timeout errors detected ({timeout_errors})")
                assert any(v.vulnerability_id == "RACE_001" for v in attack_result.vulnerabilities_found)
            
            if server_errors > 100:
                logger.critical(f"CRITICAL: Excessive server errors detected ({server_errors})")
                assert any(v.vulnerability_id == "RACE_002" for v in attack_result.vulnerabilities_found)
        
        # Log all discovered vulnerabilities
        for vuln in attack_result.vulnerabilities_found:
            logger.critical(f"VULNERABILITY FOUND: {vuln.vulnerability_id} - {vuln.description}")
            assert vuln.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH, VulnerabilitySeverity.MEDIUM]

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_model_state_corruption_attack(self, attack_detector, baseline_metrics):
        """
        CRITICAL TEST: Model state corruption via concurrent inference
        
        Tests for model state corruption through simultaneous inference calls
        with adversarial inputs designed to corrupt neural network states.
        
        Expected Vulnerabilities:
        - NaN injection attacks
        - Infinite value corruption
        - Gradient explosion scenarios
        - Model weight corruption
        """
        logger.info("🚨 Testing Model State Corruption via Concurrent Inference")
        
        # Execute the model corruption attack
        attack_result = await attack_detector._race_attack_model_state_corruption()
        
        # Verify attack execution
        assert attack_result.attack_name == "Model State Corruption via Concurrent Inference"
        assert attack_result.execution_time > 0
        
        # Analyze model corruption results
        if attack_result.performance_impact:
            matrices_tested = attack_result.performance_impact.get("matrices_tested", 0)
            nan_errors = attack_result.performance_impact.get("nan_errors", 0)
            inf_errors = attack_result.performance_impact.get("inf_errors", 0)
            grad_explosion_errors = attack_result.performance_impact.get("grad_explosion_errors", 0)
            
            logger.warning(f"Tested {matrices_tested} adversarial matrices")
            logger.warning(f"NaN errors detected: {nan_errors}")
            logger.warning(f"Infinity errors detected: {inf_errors}")
            logger.warning(f"Gradient explosion errors: {grad_explosion_errors}")
            
            # Assert model corruption vulnerabilities are detected
            if nan_errors > 50:
                logger.critical(f"CRITICAL: Model accepts NaN inputs ({nan_errors} errors)")
                assert any(v.vulnerability_id == "MODEL_001" for v in attack_result.vulnerabilities_found)
            
            if inf_errors > 30:
                logger.critical(f"CRITICAL: Model vulnerable to infinite value injection ({inf_errors} errors)")
                assert any(v.vulnerability_id == "MODEL_002" for v in attack_result.vulnerabilities_found)
        
        # Verify vulnerability severity levels
        for vuln in attack_result.vulnerabilities_found:
            logger.critical(f"MODEL VULNERABILITY: {vuln.vulnerability_id} - {vuln.description}")
            assert vuln.attack_vector == AttackVector.MODEL_CORRUPTION
            assert vuln.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]

    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_shared_resource_contention(self, attack_detector, baseline_metrics):
        """
        HIGH TEST: Shared resource contention through strategic timing
        
        Tests for deadlocks and resource starvation through strategic timing
        of requests designed to maximize resource contention.
        
        Expected Vulnerabilities:
        - Resource lock contention
        - Correlation ID conflicts
        - Deadlock scenarios
        - Request processing delays
        """
        logger.info("🚨 Testing Shared Resource Contention")
        
        # Execute the resource contention attack
        attack_result = await attack_detector._race_attack_shared_resource_contention()
        
        # Verify attack execution
        assert attack_result.attack_name == "Shared Resource Contention Attack"
        assert attack_result.execution_time > 0
        
        # Analyze resource contention results
        if attack_result.performance_impact:
            total_requests = attack_result.performance_impact.get("total_requests", 0)
            timeout_errors = attack_result.performance_impact.get("timeout_errors", 0)
            correlation_conflicts = attack_result.performance_impact.get("correlation_conflicts", 0)
            
            logger.warning(f"Tested {total_requests} contention requests")
            logger.warning(f"Timeout errors: {timeout_errors}")
            logger.warning(f"Correlation ID conflicts: {correlation_conflicts}")
            
            # Assert resource contention vulnerabilities
            if timeout_errors > 20:
                logger.critical(f"CRITICAL: Resource contention causing timeouts ({timeout_errors})")
                assert any(v.vulnerability_id == "RESOURCE_001" for v in attack_result.vulnerabilities_found)
            
            if correlation_conflicts > 0:
                logger.critical(f"CRITICAL: Correlation ID conflicts detected ({correlation_conflicts})")
                assert any(v.vulnerability_id == "RESOURCE_002" for v in attack_result.vulnerabilities_found)
        
        # Verify resource vulnerability characteristics
        for vuln in attack_result.vulnerabilities_found:
            logger.critical(f"RESOURCE VULNERABILITY: {vuln.vulnerability_id} - {vuln.description}")
            assert vuln.attack_vector == AttackVector.RACE_CONDITION

    @pytest.mark.asyncio
    @pytest.mark.timeout(600)  # 10 minute timeout for memory tests
    async def test_memory_leak_continuous_loops(self, attack_detector, baseline_metrics):
        """
        CRITICAL TEST: Memory leak exploitation through continuous decision loops
        
        Tests for memory leaks by creating continuous decision processing loops
        with increasingly large payloads designed to consume system memory.
        
        Expected Vulnerabilities:
        - Memory leak accumulation
        - Garbage collection failures
        - Resource cleanup issues
        - Out of memory scenarios
        """
        logger.info("🚨 Testing Memory Leak Continuous Decision Loops")
        
        # Capture pre-attack memory state
        pre_attack_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # Execute the memory leak attack
        attack_result = await attack_detector._race_attack_memory_leak_loops()
        
        # Verify attack execution
        assert attack_result.attack_name == "Memory Leak Continuous Decision Loops"
        assert attack_result.execution_time > 0
        
        # Analyze memory leak results
        if attack_result.performance_impact:
            baseline_memory = attack_result.performance_impact.get("baseline_memory_mb", 0)
            max_memory_increase = attack_result.performance_impact.get("max_memory_increase_mb", 0)
            memory_not_released = attack_result.performance_impact.get("memory_not_released_mb", 0)
            memory_measurements = attack_result.performance_impact.get("memory_measurements", [])
            
            logger.warning(f"Baseline memory: {baseline_memory:.1f} MB")
            logger.warning(f"Maximum memory increase: {max_memory_increase:.1f} MB")
            logger.warning(f"Memory not released after GC: {memory_not_released:.1f} MB")
            logger.warning(f"Memory measurements collected: {len(memory_measurements)}")
            
            # Assert memory leak vulnerabilities
            if max_memory_increase > 1000:  # 1GB increase
                logger.critical(f"CRITICAL: Severe memory leak detected ({max_memory_increase:.1f} MB increase)")
                assert any(v.vulnerability_id == "MEMORY_001" for v in attack_result.vulnerabilities_found)
            
            if memory_not_released > 200:  # 200MB not released
                logger.critical(f"CRITICAL: Memory not properly released ({memory_not_released:.1f} MB)")
                assert any(v.vulnerability_id == "MEMORY_002" for v in attack_result.vulnerabilities_found)
        
        # Verify memory vulnerability characteristics
        for vuln in attack_result.vulnerabilities_found:
            logger.critical(f"MEMORY VULNERABILITY: {vuln.vulnerability_id} - {vuln.description}")
            assert vuln.attack_vector == AttackVector.MEMORY_EXHAUSTION
            assert vuln.severity in [VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_response_consistency_under_load(self, attack_detector):
        """
        MEDIUM TEST: Response consistency validation under concurrent load
        
        Tests whether the system returns consistent responses for identical
        requests under high concurrency load.
        """
        logger.info("🚨 Testing Response Consistency Under Load")
        
        # Create identical payloads for consistency testing
        base_payload = {
            "matrix_state": [[float(j) for j in range(7)] for _ in range(60)],
            "synergy_context": {
                "synergy_type": "TYPE_1",
                "direction": 1,
                "confidence": 0.8,
                "correlation_id": "consistency_test",
                "timestamp": time.time()
            },
            "correlation_id": "consistency_test"
        }
        
        # Send multiple identical requests concurrently
        concurrent_requests = 100
        responses = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(concurrent_requests):
                # Use different correlation IDs to avoid conflicts
                payload = base_payload.copy()
                payload["correlation_id"] = f"consistency_test_{i}"
                payload["synergy_context"]["correlation_id"] = f"consistency_test_{i}"
                
                task = attack_detector._send_decision_request(session, payload)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze response consistency
        successful_responses = [r for r in responses if isinstance(r, dict) and r.get("status_code") == 200]
        decisions = [r.get("response_data", {}).get("decision", {}) for r in successful_responses]
        
        logger.info(f"Successful responses: {len(successful_responses)}/{concurrent_requests}")
        
        # Check for decision consistency (all identical inputs should produce similar decisions)
        if len(decisions) > 10:
            actions = [d.get("action", -1) for d in decisions if d]
            confidences = [d.get("confidence", 0.0) for d in decisions if d]
            
            if actions:
                action_variance = len(set(actions)) / len(actions)
                confidence_std = np.std(confidences) if confidences else 0.0
                
                logger.info(f"Action variance: {action_variance:.3f}")
                logger.info(f"Confidence std dev: {confidence_std:.3f}")
                
                # High variance indicates potential race conditions in decision logic
                if action_variance > 0.3:  # More than 30% different actions
                    logger.warning(f"HIGH VARIANCE: Inconsistent decisions detected (variance: {action_variance:.3f})")
                
                if confidence_std > 0.2:  # High confidence deviation
                    logger.warning(f"HIGH DEVIATION: Confidence inconsistency detected (std: {confidence_std:.3f})")

    @pytest.mark.asyncio
    async def test_correlation_id_collision_detection(self, attack_detector):
        """
        HIGH TEST: Correlation ID collision and conflict detection
        
        Tests system behavior when multiple requests use identical correlation IDs
        to identify response mapping vulnerabilities.
        """
        logger.info("🚨 Testing Correlation ID Collision Detection")
        
        # Create requests with deliberately identical correlation IDs
        collision_id = "collision_test_id"
        collision_requests = 50
        
        payloads = []
        for i in range(collision_requests):
            payload = {
                "matrix_state": [[float(j + i) for j in range(7)] for _ in range(60)],  # Slightly different data
                "synergy_context": {
                    "synergy_type": "TYPE_1",
                    "direction": 1 if i % 2 == 0 else -1,  # Alternating directions
                    "confidence": 0.8 + (i * 0.001),  # Slightly different confidences
                    "correlation_id": collision_id,  # IDENTICAL CORRELATION ID
                    "timestamp": time.time()
                },
                "correlation_id": collision_id  # IDENTICAL CORRELATION ID
            }
            payloads.append(payload)
        
        # Send collision requests
        responses = []
        async with aiohttp.ClientSession() as session:
            tasks = [attack_detector._send_decision_request(session, payload) for payload in payloads]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze collision responses
        successful_responses = [r for r in responses if isinstance(r, dict) and r.get("status_code") == 200]
        
        if len(successful_responses) > 1:
            # Check if responses are identical for identical correlation IDs
            decisions = [r.get("response_data", {}).get("decision", {}) for r in successful_responses]
            decision_strings = [json.dumps(d, sort_keys=True) for d in decisions if d]
            
            unique_decisions = set(decision_strings)
            
            logger.info(f"Responses for collision ID: {len(successful_responses)}")
            logger.info(f"Unique decisions: {len(unique_decisions)}")
            
            # If we get different decisions for the same correlation ID, it's a vulnerability
            if len(unique_decisions) > 1:
                logger.critical(f"CRITICAL: Correlation ID collision produces different responses!")
                logger.critical(f"Expected: 1 unique decision, Got: {len(unique_decisions)}")
                
                # This indicates a critical vulnerability in correlation ID handling
                assert len(unique_decisions) == 1, f"Correlation ID collision vulnerability: {len(unique_decisions)} different responses"

    def test_thread_safety_decision_aggregator(self):
        """
        HIGH TEST: Thread safety of decision aggregator under concurrent access
        
        Tests the TacticalDecisionAggregator for thread safety issues when
        accessed concurrently from multiple threads.
        """
        logger.info("🚨 Testing Decision Aggregator Thread Safety")
        
        # Import the decision aggregator
        from components.tactical_decision_aggregator import TacticalDecisionAggregator
        
        # Create aggregator instance
        aggregator = TacticalDecisionAggregator()
        
        # Create test data
        test_agent_outputs = {
            "fvg_agent": {
                "probabilities": np.array([0.3, 0.4, 0.3]),
                "action": 1,
                "confidence": 0.8,
                "timestamp": time.time()
            },
            "momentum_agent": {
                "probabilities": np.array([0.2, 0.3, 0.5]),
                "action": 2,
                "confidence": 0.7,
                "timestamp": time.time()
            },
            "entry_opt_agent": {
                "probabilities": np.array([0.4, 0.3, 0.3]),
                "action": 0,
                "confidence": 0.9,
                "timestamp": time.time()
            }
        }
        
        test_market_state = type('MockMarketState', (), {'price': 100.0})()
        test_synergy_context = {
            "type": "TYPE_1",
            "direction": 1,
            "confidence": 0.8
        }
        
        # Function to run aggregation in thread
        def run_aggregation(thread_id):
            results = []
            for i in range(100):  # 100 aggregations per thread
                try:
                    decision = aggregator.aggregate_decisions(
                        test_agent_outputs,
                        test_market_state,
                        test_synergy_context
                    )
                    results.append({
                        "thread_id": thread_id,
                        "iteration": i,
                        "decision": decision,
                        "success": True
                    })
                except Exception as e:
                    results.append({
                        "thread_id": thread_id,
                        "iteration": i,
                        "error": str(e),
                        "success": False
                    })
                    logger.error(f"Thread {thread_id} iteration {i} failed: {e}")
            return results
        
        # Run concurrent aggregations
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_aggregation, i) for i in range(num_threads)]
            all_results = []
            
            for future in as_completed(futures):
                thread_results = future.result()
                all_results.extend(thread_results)
        
        # Analyze thread safety results
        successful_aggregations = [r for r in all_results if r["success"]]
        failed_aggregations = [r for r in all_results if not r["success"]]
        
        logger.info(f"Successful aggregations: {len(successful_aggregations)}")
        logger.info(f"Failed aggregations: {len(failed_aggregations)}")
        
        # Check for thread safety issues
        if failed_aggregations:
            logger.critical(f"THREAD SAFETY ISSUE: {len(failed_aggregations)} aggregations failed")
            for failure in failed_aggregations[:5]:  # Log first 5 failures
                logger.critical(f"Thread {failure['thread_id']} iteration {failure['iteration']}: {failure['error']}")
        
        # Check for decision consistency across threads
        decisions = [r["decision"] for r in successful_aggregations if "decision" in r]
        if decisions:
            execute_decisions = [d.execute for d in decisions]
            actions = [d.action for d in decisions]
            confidences = [d.confidence for d in decisions]
            
            # Check variance in decisions (should be relatively consistent for identical inputs)
            action_variance = len(set(actions)) / len(actions) if actions else 0
            confidence_std = np.std(confidences) if confidences else 0
            
            logger.info(f"Decision action variance: {action_variance:.3f}")
            logger.info(f"Decision confidence std: {confidence_std:.3f}")
            
            # High variance might indicate thread safety issues
            if action_variance > 0.1:  # More than 10% variance in actions
                logger.warning(f"HIGH VARIANCE: Potential thread safety issue in decision actions")
            
            if confidence_std > 0.05:  # High confidence deviation
                logger.warning(f"HIGH DEVIATION: Potential thread safety issue in confidence calculation")
        
        # Assert no thread safety failures
        assert len(failed_aggregations) == 0, f"Thread safety vulnerability: {len(failed_aggregations)} failures"

    @pytest.mark.asyncio
    async def test_request_rate_limiting_bypass(self, attack_detector):
        """
        MEDIUM TEST: Request rate limiting bypass attempts
        
        Tests whether rate limiting mechanisms can be bypassed through
        various techniques like header manipulation or request spacing.
        """
        logger.info("🚨 Testing Request Rate Limiting Bypass")
        
        # Test 1: Rapid sequential requests
        rapid_requests = 200
        request_interval = 0.01  # 10ms between requests
        
        start_time = time.time()
        responses = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(rapid_requests):
                payload = {
                    "matrix_state": [[float(j) for j in range(7)] for _ in range(60)],
                    "synergy_context": {
                        "synergy_type": "TYPE_1", 
                        "direction": 1,
                        "confidence": 0.8,
                        "correlation_id": f"rate_limit_test_{i}",
                        "timestamp": time.time()
                    },
                    "correlation_id": f"rate_limit_test_{i}"
                }
                
                response = await attack_detector._send_decision_request(session, payload)
                responses.append(response)
                
                # Brief interval between requests
                await asyncio.sleep(request_interval)
        
        total_time = time.time() - start_time
        request_rate = rapid_requests / total_time
        
        # Analyze rate limiting behavior
        successful_requests = len([r for r in responses if isinstance(r, dict) and r.get("status_code") == 200])
        rate_limited_requests = len([r for r in responses if isinstance(r, dict) and r.get("status_code") == 429])
        
        logger.info(f"Request rate: {request_rate:.2f} req/sec")
        logger.info(f"Successful requests: {successful_requests}/{rapid_requests}")
        logger.info(f"Rate limited requests: {rate_limited_requests}")
        
        # Check if rate limiting is working
        if rate_limited_requests == 0 and request_rate > 100:  # More than 100 req/sec allowed
            logger.warning(f"POTENTIAL ISSUE: No rate limiting detected at {request_rate:.2f} req/sec")
        
        # Test 2: Header manipulation attempts
        bypass_attempts = [
            {"X-Forwarded-For": "192.168.1.100"},
            {"X-Real-IP": "10.0.0.50"},
            {"User-Agent": "BypassBot/1.0"},
            {"X-Rate-Limit-Bypass": "true"},
            {"Authorization": "Bearer fake_token"}
        ]
        
        for headers in bypass_attempts:
            payload = {
                "matrix_state": [[1.0 for j in range(7)] for _ in range(60)],
                "synergy_context": {
                    "synergy_type": "TYPE_1",
                    "direction": 1, 
                    "confidence": 0.8,
                    "correlation_id": f"bypass_test_{hash(str(headers))}",
                    "timestamp": time.time()
                },
                "correlation_id": f"bypass_test_{hash(str(headers))}"
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        f"{attack_detector.base_url}/decide",
                        json=payload,
                        headers={**headers, "Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Bypass attempt with {headers} succeeded")
                        elif response.status == 429:
                            logger.info(f"Bypass attempt with {headers} rate limited")
                        else:
                            logger.info(f"Bypass attempt with {headers} returned {response.status}")
                except Exception as e:
                    logger.info(f"Bypass attempt with {headers} failed: {e}")