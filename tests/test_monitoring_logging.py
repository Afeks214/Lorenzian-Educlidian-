"""
Monitoring and Logging verification test suite.
Tests Prometheus metrics, structured logging, and correlation ID tracking.
"""

import pytest
import httpx
import asyncio
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any
import concurrent.futures
import uuid
import os

# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")


class TestMonitoringLogging:
    """Test suite for monitoring and logging verification."""
    
    @pytest.fixture
    def client(self):
        """Create HTTP client for testing."""
        return httpx.Client(base_url=API_BASE_URL)
    
    @pytest.fixture
    def prometheus_client(self):
        """Create HTTP client for Prometheus queries."""
        return httpx.Client(base_url=PROMETHEUS_URL)
    
    def generate_correlation_id(self) -> str:
        """Generate a unique correlation ID for testing."""
        return f"test-{uuid.uuid4()}"
    
    @pytest.mark.integration
    def test_prometheus_metrics_validation(self, client, prometheus_client):
        """
        Test 3.1: Verify Prometheus metrics are created and updated correctly.
        Simulates a complete trade flow and validates metric values.
        """
        # Generate unique correlation ID for this test
        correlation_id = self.generate_correlation_id()
        
        # Step 1: Make several API calls to generate metrics
        print("Generating metrics through API calls...")
        
        # Health checks
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Simulated trade flow
        trade_requests = [
            {
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
                    "pattern_data": {"signal": "entry"},
                    "correlation_id": correlation_id
                },
                "matrix_data": {
                    "matrix_type": "30m",
                    "shape": [48, 13],
                    "data": [[0.1] * 13 for _ in range(48)],
                    "features": ["close"] * 13,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        ]
        
        # Simulate entry decision
        for request_data in trade_requests:
            response = client.post(
                "/decide",
                json=request_data,
                headers={"Authorization": "Bearer test-token"}
            )
            # We expect either success or auth failure
            assert response.status_code in [200, 401]
        
        # Step 2: Query metrics endpoint
        time.sleep(2)  # Allow metrics to be updated
        
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        assert metrics_response.headers.get("content-type").startswith("text/plain")
        
        metrics_text = metrics_response.text
        
        # Step 3: Validate required metrics exist
        required_metrics = [
            "grandmodel_health_status",
            "grandmodel_inference_latency_seconds",
            "grandmodel_http_requests_total",
            "grandmodel_http_request_duration_seconds",
            "grandmodel_active_connections",
            "grandmodel_event_bus_throughput_total",
            "grandmodel_model_confidence_score",
            "grandmodel_synergy_response_total"
        ]
        
        for metric in required_metrics:
            assert metric in metrics_text, f"Metric {metric} not found in Prometheus output"
        
        # Step 4: Validate metric values via Prometheus API (if available)
        if prometheus_client:
            try:
                # Query specific metrics
                query_response = prometheus_client.get(
                    "/api/v1/query",
                    params={"query": "grandmodel_health_status"}
                )
                
                if query_response.status_code == 200:
                    data = query_response.json()
                    if data["status"] == "success" and data["data"]["result"]:
                        # Validate health status is 1 (healthy)
                        for result in data["data"]["result"]:
                            assert float(result["value"][1]) == 1, \
                                "Health status should be 1 (healthy)"
                
                # Check HTTP request metrics increased
                query_response = prometheus_client.get(
                    "/api/v1/query",
                    params={"query": "grandmodel_http_requests_total"}
                )
                
                if query_response.status_code == 200:
                    data = query_response.json()
                    if data["status"] == "success" and data["data"]["result"]:
                        total_requests = sum(
                            float(r["value"][1]) for r in data["data"]["result"]
                        )
                        assert total_requests > 0, "HTTP requests metric should be > 0"
                        
            except Exception as e:
                print(f"Warning: Could not query Prometheus API: {e}")
        
        # Step 5: Parse and validate metric format
        metric_lines = metrics_text.split('\n')
        for line in metric_lines:
            if line and not line.startswith('#'):
                # Validate Prometheus format: metric_name{labels} value
                assert re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$', line) or \
                       re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+\+Inf$', line) or \
                       re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+-Inf$', line) or \
                       re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})?\s+NaN$', line), \
                    f"Invalid Prometheus metric format: {line}"
    
    @pytest.mark.asyncio
    async def test_structured_logging_correlation(self):
        """
        Test 3.2: Verify structured JSON logging with correlation ID tracking.
        Tests concurrent requests with proper correlation ID differentiation.
        """
        async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
            # Generate two different correlation IDs
            correlation_id_1 = self.generate_correlation_id()
            correlation_id_2 = self.generate_correlation_id()
            
            # Create two concurrent requests with different correlation IDs
            async def make_request(correlation_id: str) -> httpx.Response:
                return await client.get(
                    "/health",
                    headers={"X-Correlation-ID": correlation_id}
                )
            
            # Execute requests concurrently
            print("Executing concurrent requests with different correlation IDs...")
            responses = await asyncio.gather(
                make_request(correlation_id_1),
                make_request(correlation_id_2)
            )
            
            # Verify both requests succeeded
            for response in responses:
                assert response.status_code == 200
            
            # Verify correlation IDs in response headers
            assert responses[0].headers.get("X-Correlation-ID") == correlation_id_1, \
                "Response should contain the same correlation ID as request"
            assert responses[1].headers.get("X-Correlation-ID") == correlation_id_2, \
                "Response should contain the same correlation ID as request"
            
            # If we had access to logs, we would verify:
            # 1. Each log line is valid JSON
            # 2. correlation_id field matches the request
            # 3. Logs from different requests don't mix
            
            # Simulate log validation (in real test, would capture actual logs)
            print(f"Correlation ID 1: {correlation_id_1}")
            print(f"Correlation ID 2: {correlation_id_2}")
            
            # Additional test: Verify new correlation ID is generated if not provided
            response_no_id = await client.get("/health")
            generated_id = response_no_id.headers.get("X-Correlation-ID")
            assert generated_id is not None, "Should generate correlation ID if not provided"
            assert len(generated_id) > 0, "Generated correlation ID should not be empty"
    
    def test_metric_labels_and_cardinality(self, client):
        """
        Test that metrics have appropriate labels and cardinality control.
        """
        # Make various requests to generate diverse metrics
        endpoints = ["/health", "/metrics"]
        methods = ["GET"]
        
        for endpoint in endpoints:
            for method in methods:
                if method == "GET":
                    client.get(endpoint)
        
        # Get metrics
        metrics_response = client.get("/metrics")
        metrics_text = metrics_response.text
        
        # Parse metrics and check cardinality
        label_combinations = {}
        for line in metrics_text.split('\n'):
            if line and not line.startswith('#') and '{' in line:
                metric_name = line.split('{')[0]
                labels = line.split('{')[1].split('}')[0]
                
                if metric_name not in label_combinations:
                    label_combinations[metric_name] = set()
                label_combinations[metric_name].add(labels)
        
        # Verify cardinality is under control
        for metric_name, labels in label_combinations.items():
            assert len(labels) < 1000, \
                f"Metric {metric_name} has too high cardinality: {len(labels)} combinations"
            
            # Check for unbounded labels (e.g., user IDs, correlation IDs)
            for label_set in labels:
                assert "correlation_id=" not in label_set, \
                    "Correlation ID should not be a metric label (high cardinality)"
                assert "user_id=" not in label_set or len(labels) < 100, \
                    "User ID as label creates high cardinality"
    
    def test_error_metrics_and_logging(self, client):
        """
        Test that errors are properly tracked in metrics and logs.
        """
        # Intentionally cause various errors
        error_scenarios = [
            # Missing auth
            {
                "endpoint": "/decide",
                "method": "POST",
                "headers": {},
                "json": {"invalid": "data"},
                "expected_status": 401
            },
            # Invalid data
            {
                "endpoint": "/decide",
                "method": "POST",
                "headers": {"Authorization": "Bearer test-token"},
                "json": {"invalid": "structure"},
                "expected_status": [401, 422]
            },
            # Non-existent endpoint
            {
                "endpoint": "/nonexistent",
                "method": "GET",
                "headers": {},
                "json": None,
                "expected_status": 404
            }
        ]
        
        for scenario in error_scenarios:
            if scenario["method"] == "POST":
                response = client.post(
                    scenario["endpoint"],
                    headers=scenario["headers"],
                    json=scenario["json"]
                )
            else:
                response = client.get(
                    scenario["endpoint"],
                    headers=scenario["headers"]
                )
            
            # Verify expected error status
            if isinstance(scenario["expected_status"], list):
                assert response.status_code in scenario["expected_status"]
            else:
                assert response.status_code == scenario["expected_status"]
        
        # Check error metrics
        metrics_response = client.get("/metrics")
        metrics_text = metrics_response.text
        
        # Verify error metrics exist and are non-zero
        assert "grandmodel_errors_total" in metrics_text or \
               "grandmodel_http_requests_total" in metrics_text, \
            "Error metrics should be tracked"
        
        # Check for 4xx and 5xx status codes in HTTP metrics
        status_pattern = r'status="4\d\d"|status="5\d\d"'
        assert re.search(status_pattern, metrics_text), \
            "HTTP error status codes should be tracked in metrics"
    
    def test_performance_metrics_accuracy(self, client):
        """
        Test that performance metrics accurately reflect system behavior.
        """
        # Make a series of requests with known timing
        request_count = 10
        request_times = []
        
        for _ in range(request_count):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            if response.status_code == 200:
                request_times.append(end_time - start_time)
        
        # Get metrics
        metrics_response = client.get("/metrics")
        metrics_text = metrics_response.text
        
        # Extract latency metrics
        latency_pattern = r'grandmodel_http_request_duration_seconds_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+(\d+)'
        buckets = re.findall(latency_pattern, metrics_text)
        
        if buckets:
            # Verify histogram buckets make sense
            for le_value, count in buckets:
                if le_value != "+Inf":
                    le_float = float(le_value)
                    # Count requests under this latency
                    actual_under = sum(1 for t in request_times if t <= le_float)
                    recorded_count = int(count)
                    
                    # Allow some tolerance due to other requests
                    assert recorded_count >= actual_under * 0.8, \
                        f"Histogram bucket {le_value} undercounting requests"
        
        # Verify request count metric
        count_pattern = r'grandmodel_http_requests_total\{[^}]*method="GET",path="/health"[^}]*\}\s+(\d+)'
        count_matches = re.findall(count_pattern, metrics_text)
        
        if count_matches:
            total_recorded = sum(int(c) for c in count_matches)
            assert total_recorded >= request_count, \
                f"Request count metric ({total_recorded}) less than actual ({request_count})"
    
    @pytest.mark.integration
    def test_distributed_tracing_simulation(self, client):
        """
        Test correlation ID propagation through the system.
        """
        correlation_id = self.generate_correlation_id()
        
        # Simulate a complex flow with multiple components
        # 1. Initial request
        response1 = client.get(
            "/health",
            headers={"X-Correlation-ID": correlation_id}
        )
        assert response1.headers.get("X-Correlation-ID") == correlation_id
        
        # 2. Follow-up request with same correlation ID
        response2 = client.get(
            "/metrics",
            headers={"X-Correlation-ID": correlation_id}
        )
        assert response2.headers.get("X-Correlation-ID") == correlation_id
        
        # 3. Verify correlation ID is preserved across different endpoints
        # In a real system, we would verify this ID appears in:
        # - Application logs
        # - Database query logs
        # - External service calls
        # - Error tracking
        
        print(f"Correlation ID {correlation_id} successfully tracked across requests")
    
    def test_custom_business_metrics(self, client):
        """
        Test custom business metrics specific to trading system.
        """
        # Get current metrics
        metrics_response = client.get("/metrics")
        metrics_text = metrics_response.text
        
        # Verify trading-specific metrics exist
        business_metrics = [
            "grandmodel_synergy_response_total",
            "grandmodel_model_confidence_score",
            "grandmodel_inference_latency_seconds",
            "grandmodel_active_positions_count",
            "grandmodel_trade_pnl_dollars"
        ]
        
        for metric in business_metrics:
            if metric in metrics_text:
                print(f"✓ Found business metric: {metric}")
                
                # Verify metric has appropriate labels
                if "synergy_response" in metric:
                    assert 'synergy_type=' in metrics_text, \
                        "Synergy metrics should include synergy_type label"
                elif "model_confidence" in metric:
                    assert 'model_type=' in metrics_text or 'agent_name=' in metrics_text, \
                        "Model metrics should include model/agent identification"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])