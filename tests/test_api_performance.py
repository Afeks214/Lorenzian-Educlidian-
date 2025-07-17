"""
API Performance test suite with load testing.
Tests response times, throughput, and error rates under load.
"""

import logging


import pytest
import time
import statistics
import httpx
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from locust import HttpUser, task, between
import concurrent.futures
import os
import json

# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
LOAD_TEST_DURATION = 60  # seconds
CONCURRENT_USERS = 100
P95_LATENCY_THRESHOLD_MS = 100


class TestAPIPerformance:
    """Test suite for API performance validation."""
    
    @pytest.fixture
    def client(self):
        """Create HTTP client for testing."""
        return httpx.Client(base_url=API_BASE_URL, timeout=30.0)
    
    @pytest.fixture
    def valid_token(self):
        """Generate a valid JWT token for tests."""
        # In real tests, this would generate a proper JWT
        return "test-valid-token"
    
    def test_health_endpoint_performance(self, client):
        """
        Test health endpoint response time under normal conditions.
        """
        response_times = []
        
        # Warm up
        for _ in range(5):
            client.get("/health")
        
        # Measure response times
        for _ in range(100):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            if response.status_code == 200:
                response_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        
        print(f"Health endpoint performance:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  P95: {p95_time:.2f}ms")
        print(f"  P99: {p99_time:.2f}ms")
        
        # Assert performance requirements
        assert p95_time < P95_LATENCY_THRESHOLD_MS, \
            f"P95 latency {p95_time:.2f}ms exceeds {P95_LATENCY_THRESHOLD_MS}ms threshold"
    
    @pytest.mark.integration
    def test_concurrent_load_performance(self, client):
        """
        Test 2.3 & 2.4: Load test with concurrent users.
        Verify P95 < 100ms and 0% error rate.
        """
        results = {
            "response_times": [],
            "errors": 0,
            "success": 0,
            "status_codes": {}
        }
        
        def make_health_request():
            """Make a single health check request."""
            try:
                start_time = time.time()
                response = client.get("/health", timeout=5.0)
                end_time = time.time()
                
                response_time_ms = (end_time - start_time) * 1000
                
                # Record results
                results["response_times"].append(response_time_ms)
                results["status_codes"][response.status_code] = \
                    results["status_codes"].get(response.status_code, 0) + 1
                
                if response.status_code == 200:
                    results["success"] += 1
                else:
                    results["errors"] += 1
                    
                return response_time_ms
            except Exception as e:
                results["errors"] += 1
                return None
        
        print(f"Starting load test with {CONCURRENT_USERS} concurrent users...")
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_USERS) as executor:
            # Submit requests for the duration
            start_time = time.time()
            futures = []
            
            while time.time() - start_time < LOAD_TEST_DURATION:
                # Submit batch of requests
                batch_futures = [
                    executor.submit(make_health_request) 
                    for _ in range(CONCURRENT_USERS)
                ]
                futures.extend(batch_futures)
                
                # Small delay between batches
                time.sleep(0.1)
            
            # Wait for all requests to complete
            concurrent.futures.wait(futures, timeout=30)
        
        # Calculate statistics
        valid_response_times = [rt for rt in results["response_times"] if rt is not None]
        
        if valid_response_times:
            avg_time = statistics.mean(valid_response_times)
            p50_time = statistics.median(valid_response_times)
            p95_time = statistics.quantiles(valid_response_times, n=20)[18]
            p99_time = statistics.quantiles(valid_response_times, n=100)[98]
            max_time = max(valid_response_times)
            
            total_requests = results["success"] + results["errors"]
            error_rate = (results["errors"] / total_requests * 100) if total_requests > 0 else 0
            
            print(f"\nLoad test results:")
            print(f"  Total requests: {total_requests}")
            print(f"  Successful: {results['success']}")
            print(f"  Errors: {results['errors']}")
            print(f"  Error rate: {error_rate:.2f}%")
            print(f"\nResponse times:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Median (P50): {p50_time:.2f}ms")
            print(f"  P95: {p95_time:.2f}ms")
            print(f"  P99: {p99_time:.2f}ms")
            print(f"  Max: {max_time:.2f}ms")
            print(f"\nStatus codes: {results['status_codes']}")
            
            # Assert performance requirements
            assert p95_time < P95_LATENCY_THRESHOLD_MS, \
                f"P95 latency {p95_time:.2f}ms exceeds {P95_LATENCY_THRESHOLD_MS}ms threshold"
            
            assert error_rate == 0, \
                f"Error rate {error_rate:.2f}% should be 0%"
        else:
            pytest.fail("No valid response times recorded")
    
    @pytest.mark.asyncio
    async def test_decide_endpoint_performance(self, valid_token):
        """
        Test /decide endpoint performance with realistic payloads.
        """
        async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
            # Prepare realistic request payload
            request_data = {
                "market_state": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": "BTCUSDT",
                    "price": 50000.0,
                    "volume": 1000.0,
                    "volatility": 0.02,
                    "trend": "bullish"
                },
                "synergy_context": {
                    "synergy_type": "TYPE_1",
                    "strength": 0.85,
                    "confidence": 0.9,
                    "pattern_data": {
                        "indicators": ["MLMI", "NWRQK", "FVG", "LVN"],
                        "signals": {"entry": True, "strength": "high"}
                    },
                    "correlation_id": "perf-test-001"
                },
                "matrix_data": {
                    "matrix_type": "30m",
                    "shape": [48, 13],
                    "data": [[i * 0.01 for i in range(13)] for _ in range(48)],
                    "features": ["open", "high", "low", "close", "volume", 
                               "mlmi", "nwrqk", "fvg", "lvn", "mmd", 
                               "rsi", "macd", "bb"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            headers = {"Authorization": f"Bearer {valid_token}"}
            response_times = []
            
            # Warm up
            for _ in range(5):
                await client.post("/decide", json=request_data, headers=headers)
            
            # Measure response times
            for i in range(50):
                # Vary the request slightly each time
                request_data["synergy_context"]["correlation_id"] = f"perf-test-{i:03d}"
                request_data["market_state"]["price"] = 50000.0 + (i * 10)
                
                start_time = time.time()
                response = await client.post("/decide", json=request_data, headers=headers)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append((end_time - start_time) * 1000)
                    
                    # Verify inference latency from response
                    response_data = response.json()
                    if "inference_latency_ms" in response_data:
                        assert response_data["inference_latency_ms"] < 5.0, \
                            f"Inference latency {response_data['inference_latency_ms']}ms exceeds 5ms limit"
            
            # Calculate statistics
            if response_times:
                avg_time = statistics.mean(response_times)
                p95_time = statistics.quantiles(response_times, n=20)[18]
                
                print(f"\nDecide endpoint performance:")
                print(f"  Average: {avg_time:.2f}ms")
                print(f"  P95: {p95_time:.2f}ms")
                
                # The decide endpoint has a higher threshold due to ML inference
                assert p95_time < 200, \
                    f"Decide endpoint P95 {p95_time:.2f}ms exceeds 200ms threshold"
    
    def test_metrics_endpoint_performance(self, client):
        """
        Test Prometheus metrics endpoint performance.
        """
        response_times = []
        
        # Generate some metrics first
        for _ in range(10):
            client.get("/health")
        
        # Measure metrics endpoint
        for _ in range(20):
            start_time = time.time()
            response = client.get("/metrics")
            end_time = time.time()
            
            if response.status_code == 200:
                response_times.append((end_time - start_time) * 1000)
                
                # Verify metrics format
                assert response.headers.get("content-type").startswith("text/plain"), \
                    "Metrics should be in Prometheus text format"
        
        if response_times:
            avg_time = statistics.mean(response_times)
            print(f"\nMetrics endpoint performance:")
            print(f"  Average: {avg_time:.2f}ms")
            
            # Metrics endpoint should be fast
            assert avg_time < 50, \
                f"Metrics endpoint average {avg_time:.2f}ms exceeds 50ms threshold"
    
    @pytest.mark.integration
    def test_sustained_load(self, client):
        """
        Test system stability under sustained load.
        """
        print("\nTesting sustained load for system stability...")
        
        # Track performance over time
        time_buckets = []
        bucket_duration = 10  # seconds per bucket
        test_duration = 60  # total seconds
        
        start_time = time.time()
        current_bucket = []
        
        while time.time() - start_time < test_duration:
            request_start = time.time()
            try:
                response = client.get("/health", timeout=2.0)
                request_time = (time.time() - request_start) * 1000
                
                if response.status_code == 200:
                    current_bucket.append(request_time)
            except (ConnectionError, OSError, TimeoutError) as e:
                logger.error(f'Error occurred: {e}')
            
            # Check if we need to move to next bucket
            if len(current_bucket) > 0 and \
               time.time() - start_time > len(time_buckets) * bucket_duration:
                time_buckets.append(current_bucket)
                current_bucket = []
        
        # Add final bucket
        if current_bucket:
            time_buckets.append(current_bucket)
        
        # Analyze performance degradation
        bucket_p95s = []
        for i, bucket in enumerate(time_buckets):
            if bucket:
                p95 = statistics.quantiles(bucket, n=20)[18]
                bucket_p95s.append(p95)
                print(f"  Bucket {i+1} P95: {p95:.2f}ms")
        
        # Check for performance degradation
        if len(bucket_p95s) >= 2:
            first_bucket_p95 = bucket_p95s[0]
            last_bucket_p95 = bucket_p95s[-1]
            degradation = ((last_bucket_p95 - first_bucket_p95) / first_bucket_p95) * 100
            
            print(f"\nPerformance degradation: {degradation:.1f}%")
            
            # Assert no significant degradation (< 20%)
            assert degradation < 20, \
                f"Performance degraded by {degradation:.1f}% over sustained load"
    
    def test_resource_efficiency(self, client):
        """
        Test that the API efficiently uses resources.
        """
        # Make a request and check resource headers if available
        response = client.get("/health")
        
        # Check if server provides resource usage info
        if "X-Process-Memory-MB" in response.headers:
            memory_mb = float(response.headers["X-Process-Memory-MB"])
            assert memory_mb < 512, \
                f"Memory usage {memory_mb}MB exceeds 512MB limit"
        
        # Verify connection reuse
        with httpx.Client(base_url=API_BASE_URL) as persistent_client:
            # Make multiple requests on same connection
            response_times = []
            for _ in range(10):
                start = time.time()
                response = persistent_client.get("/health")
                response_times.append((time.time() - start) * 1000)
            
            # Later requests should be faster (connection reuse)
            first_half_avg = statistics.mean(response_times[:5])
            second_half_avg = statistics.mean(response_times[5:])
            
            print(f"\nConnection reuse test:")
            print(f"  First half avg: {first_half_avg:.2f}ms")
            print(f"  Second half avg: {second_half_avg:.2f}ms")
            
            # Second half should be faster or similar (not slower)
            assert second_half_avg <= first_half_avg * 1.1, \
                "Performance should improve with connection reuse"


# Locust load test configuration
class StrategicMARLUser(HttpUser):
    """Locust user for load testing."""
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Setup before tests."""
        # In real test, would authenticate and get token
        self.token = "test-token"
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def health_check(self):
        """Health check endpoint."""
        self.client.get("/health")
    
    @task(1)
    def metrics(self):
        """Metrics endpoint."""
        self.client.get("/metrics")
    
    @task(2)
    def make_decision(self):
        """Decision endpoint with realistic payload."""
        payload = {
            "market_state": {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": "BTCUSDT",
                "price": 50000.0 + (time.time() % 1000),
                "volume": 1000.0,
                "volatility": 0.02,
                "trend": "bullish"
            },
            "synergy_context": {
                "synergy_type": f"TYPE_{(int(time.time()) % 4) + 1}",
                "strength": 0.7 + (time.time() % 0.3),
                "confidence": 0.8,
                "pattern_data": {},
                "correlation_id": f"load-test-{time.time()}"
            },
            "matrix_data": {
                "matrix_type": "30m",
                "shape": [48, 13],
                "data": [[0.1] * 13 for _ in range(48)],
                "features": ["f"] * 13,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        self.client.post("/decide", json=payload, headers=self.headers)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])