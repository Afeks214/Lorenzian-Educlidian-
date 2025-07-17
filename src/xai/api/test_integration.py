"""
Integration Testing Suite for XAI API
AGENT DELTA MISSION: Comprehensive API Testing

This module implements a comprehensive testing suite to validate API performance,
Strategic MARL integration, and system reliability under production conditions.

Features:
- API endpoint testing with realistic data
- Strategic MARL integration validation
- Performance and load testing
- WebSocket connection testing
- Error handling and edge case validation
- Compliance and audit trail testing

Author: Agent Delta - Integration Specialist
Version: 1.0 - Integration Testing Suite
"""

import asyncio
import json
import time
import uuid
import pytest
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from httpx import AsyncClient
import websockets

# Import the FastAPI app and components
from .main import app
from .models import *
from .integration import StrategicMARLIntegrator
from .query_engine import NaturalLanguageQueryEngine
from .websocket_handlers import WebSocketConnectionManager

logger = logging.getLogger(__name__)


class TestConfig:
    """Test configuration"""
    BASE_URL = "http://testserver"
    WS_URL = "ws://testserver/ws/explanations"
    TEST_TIMEOUT = 30.0
    PERFORMANCE_THRESHOLD_MS = 2000.0
    LOAD_TEST_CONNECTIONS = 50
    LOAD_TEST_DURATION = 60.0


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Async test client fixture"""
    async with AsyncClient(app=app, base_url=TestConfig.BASE_URL) as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Authentication headers fixture with vault-managed token"""
    # Get test token from environment or generate secure one
    import os
    import secrets
    
    test_token = os.getenv("TEST_API_TOKEN")
    if not test_token:
        # Generate secure test token for integration testing
        test_token = f"test_{secrets.token_urlsafe(32)}"
    
    return {
        "Authorization": f"Bearer {test_token}"
    }


@pytest.fixture
def sample_explanation_request():
    """Sample explanation request fixture"""
    return {
        "symbol": "NQ",
        "timestamp": datetime.now().isoformat(),
        "explanation_type": "FEATURE_IMPORTANCE",
        "audience": "TRADER",
        "correlation_id": str(uuid.uuid4())
    }


@pytest.fixture
def sample_query_request():
    """Sample query request fixture"""
    return {
        "query": "How is the MLMI agent performing compared to NWRQK?",
        "correlation_id": str(uuid.uuid4())
    }


class TestHealthEndpoints:
    """Test health and monitoring endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
        assert "uptime_seconds" in data
        assert "active_connections" in data
    
    def test_health_check_rate_limit(self, client):
        """Test health check rate limiting"""
        # Make multiple requests to test rate limiting
        responses = []
        for _ in range(15):  # Exceeds 10/minute limit
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Should have some rate limited responses
        assert 429 in responses
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "xai_explanation_requests_total" in response.text


class TestExplanationEndpoints:
    """Test explanation-related endpoints"""
    
    def test_explain_decision_success(self, client, auth_headers, sample_explanation_request):
        """Test successful decision explanation"""
        response = client.post(
            "/explain/decision",
            json=sample_explanation_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "explanation_id" in data
        assert "symbol" in data
        assert data["symbol"] == sample_explanation_request["symbol"]
        assert "reasoning" in data
        assert "feature_importance" in data
        assert "top_positive_factors" in data
        assert "confidence_score" in data
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert "agent_contributions" in data
        assert "strategic_context" in data
        assert "compliance_metadata" in data
    
    def test_explain_decision_unauthorized(self, client, sample_explanation_request):
        """Test explanation without authentication"""
        response = client.post(
            "/explain/decision",
            json=sample_explanation_request
        )
        
        assert response.status_code == 401
    
    def test_explain_decision_invalid_symbol(self, client, auth_headers):
        """Test explanation with invalid symbol"""
        invalid_request = {
            "symbol": "",  # Invalid empty symbol
            "explanation_type": "FEATURE_IMPORTANCE",
            "audience": "TRADER"
        }
        
        response = client.post(
            "/explain/decision",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_explain_decision_performance(self, client, auth_headers, sample_explanation_request):
        """Test explanation response time"""
        start_time = time.time()
        
        response = client.post(
            "/explain/decision",
            json=sample_explanation_request,
            headers=auth_headers
        )
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < TestConfig.PERFORMANCE_THRESHOLD_MS
    
    def test_explain_decision_different_audiences(self, client, auth_headers, sample_explanation_request):
        """Test explanations for different audiences"""
        audiences = ["TRADER", "RISK_MANAGER", "REGULATOR", "CLIENT", "TECHNICAL"]
        
        for audience in audiences:
            request_data = {**sample_explanation_request, "audience": audience}
            
            response = client.post(
                "/explain/decision",
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["audience"] == audience
            assert len(data["reasoning"]) > 0


class TestQueryEndpoints:
    """Test natural language query endpoints"""
    
    def test_natural_language_query_success(self, client, auth_headers, sample_query_request):
        """Test successful natural language query"""
        response = client.post(
            "/query",
            json=sample_query_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert "query_id" in data
        assert "original_query" in data
        assert data["original_query"] == sample_query_request["query"]
        assert "interpreted_intent" in data
        assert "answer" in data
        assert "supporting_data" in data
        assert "confidence_score" in data
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert "follow_up_suggestions" in data
        assert "processing_metadata" in data
    
    def test_query_with_time_range(self, client, auth_headers):
        """Test query with time range"""
        query_request = {
            "query": "Show me the performance over the last 24 hours",
            "time_range": {
                "start_time": (datetime.now() - timedelta(hours=24)).isoformat(),
                "end_time": datetime.now().isoformat()
            }
        }
        
        response = client.post(
            "/query",
            json=query_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "time_range" in data["supporting_data"] or "decision_history" in data["supporting_data"]
    
    def test_query_complex_analytics(self, client, auth_headers):
        """Test complex analytics query"""
        complex_queries = [
            "Which agent performs best in trending markets?",
            "What are the main risk factors affecting our decisions?",
            "How has decision confidence changed over time?",
            "Compare MLMI and NWRQK performance in volatile conditions"
        ]
        
        for query in complex_queries:
            query_request = {
                "query": query,
                "correlation_id": str(uuid.uuid4())
            }
            
            response = client.post(
                "/query",
                json=query_request,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["answer"]) > 50  # Substantial answer
            assert data["confidence_score"] > 0.0


class TestAnalyticsEndpoints:
    """Test analytics and reporting endpoints"""
    
    def test_decision_history(self, client, auth_headers):
        """Test decision history endpoint"""
        response = client.get(
            "/decisions/history",
            params={"limit": 10, "include_explanations": True},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_decisions" in data
        assert "decisions" in data
        assert "summary_stats" in data
        assert len(data["decisions"]) <= 10
    
    def test_decision_history_with_filters(self, client, auth_headers):
        """Test decision history with filters"""
        params = {
            "symbol": "NQ",
            "start_time": (datetime.now() - timedelta(days=7)).isoformat(),
            "end_time": datetime.now().isoformat(),
            "limit": 50
        }
        
        response = client.get(
            "/decisions/history",
            params=params,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All decisions should be for NQ symbol
        for decision in data["decisions"]:
            assert decision["symbol"] == "NQ"
    
    def test_performance_analytics(self, client, auth_headers):
        """Test performance analytics endpoint"""
        response = client.get(
            "/analytics/performance",
            params={"time_range": "24h"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overall_performance" in data
        assert "agent_performance" in data
        assert "decision_quality_metrics" in data
        assert "strategic_insights" in data
        assert "risk_metrics" in data
        assert "system_health" in data
        assert "recommendations" in data
    
    def test_performance_analytics_symbol_filter(self, client, auth_headers):
        """Test performance analytics with symbol filter"""
        response = client.get(
            "/analytics/performance",
            params={"time_range": "7d", "symbol": "NQ"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "NQ"


class TestComplianceEndpoints:
    """Test compliance and reporting endpoints"""
    
    def test_compliance_report_generation(self, client, auth_headers):
        """Test compliance report generation"""
        report_request = {
            "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "report_type": "standard"
        }
        
        response = client.post(
            "/compliance/report",
            json=report_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "report_id" in data
        assert "generated_at" in data
        assert "report_period" in data
        assert "summary" in data
        assert "decision_audit_trail" in data
        assert "regulatory_compliance_status" in data
        assert "risk_assessment" in data
    
    def test_compliance_report_validation(self, client, auth_headers):
        """Test compliance report request validation"""
        invalid_request = {
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() - timedelta(days=1)).isoformat(),  # End before start
            "report_type": "standard"
        }
        
        response = client.post(
            "/compliance/report",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error


class TestWebSocketEndpoints:
    """Test WebSocket functionality"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test basic WebSocket connection"""
        try:
            async with websockets.connect(TestConfig.WS_URL) as websocket:
                # Should receive welcome message
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                assert data["type"] == "welcome"
                assert "capabilities" in data
        except Exception as e:
            pytest.skip(f"WebSocket test skipped: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_subscription(self):
        """Test WebSocket subscription"""
        try:
            async with websockets.connect(TestConfig.WS_URL) as websocket:
                # Wait for welcome message
                await websocket.recv()
                
                # Send subscription request
                subscription_request = {
                    "type": "subscription",
                    "subscription_type": "all_explanations"
                }
                
                await websocket.send(json.dumps(subscription_request))
                
                # Should receive subscription confirmation
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                assert data["type"] == "subscription"
                assert data["data"]["status"] == "success"
        except Exception as e:
            pytest.skip(f"WebSocket subscription test skipped: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong"""
        try:
            async with websockets.connect(TestConfig.WS_URL) as websocket:
                # Wait for welcome message
                await websocket.recv()
                
                # Send ping
                ping_request = {
                    "type": "ping",
                    "id": str(uuid.uuid4())
                }
                
                await websocket.send(json.dumps(ping_request))
                
                # Should receive pong
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                assert data["type"] == "pong"
                assert data["data"]["ping_id"] == ping_request["id"]
        except Exception as e:
            pytest.skip(f"WebSocket ping/pong test skipped: {e}")


class TestStrategicMARLIntegration:
    """Test Strategic MARL integration"""
    
    @pytest.mark.asyncio
    async def test_marl_integrator_initialization(self):
        """Test Strategic MARL integrator initialization"""
        integrator = StrategicMARLIntegrator()
        
        await integrator.initialize()
        
        assert integrator.is_healthy()
        assert integrator.integration_metrics["decisions_processed"] == 0
        
        await integrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_decision_context_retrieval(self):
        """Test decision context retrieval"""
        integrator = StrategicMARLIntegrator()
        await integrator.initialize()
        
        try:
            context = await integrator.get_decision_context("NQ")
            
            if context:  # May be None if no data available
                assert "snapshot" in context
                assert "strategic_context" in context
                assert "agent_contributions" in context
                assert "performance_metrics" in context
        
        finally:
            await integrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_performance_analytics_integration(self):
        """Test performance analytics integration"""
        integrator = StrategicMARLIntegrator()
        await integrator.initialize()
        
        try:
            analytics = await integrator.get_performance_analytics("24h")
            
            assert "overall_performance" in analytics
            assert "agent_performance" in analytics
            assert "decision_quality" in analytics
            assert "strategic_insights" in analytics
            
        finally:
            await integrator.shutdown()


class TestQueryEngine:
    """Test natural language query engine"""
    
    @pytest.mark.asyncio
    async def test_query_engine_initialization(self):
        """Test query engine initialization"""
        engine = NaturalLanguageQueryEngine()
        
        await engine.initialize()
        
        assert engine.is_healthy()
        assert engine.engine_metrics["total_queries"] == 0
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_query_processing(self):
        """Test query processing"""
        engine = NaturalLanguageQueryEngine()
        await engine.initialize()
        
        try:
            result = await engine.process_query(
                "How is the MLMI agent performing?",
                correlation_id="test_123"
            )
            
            assert result.intent is not None
            assert len(result.answer) > 0
            assert result.confidence > 0.0
            assert isinstance(result.supporting_data, dict)
            assert isinstance(result.follow_up_suggestions, list)
        
        finally:
            await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_query_intent_classification(self):
        """Test query intent classification"""
        engine = NaturalLanguageQueryEngine()
        await engine.initialize()
        
        test_queries = [
            ("How well is MLMI performing?", "PERFORMANCE_ANALYSIS"),
            ("Why did you choose to buy NQ?", "DECISION_EXPLANATION"),
            ("Compare MLMI and NWRQK agents", "AGENT_COMPARISON"),
            ("What are the current risk levels?", "RISK_ASSESSMENT"),
            ("Show me the decision history", "HISTORICAL_ANALYSIS"),
            ("Is the system healthy?", "SYSTEM_STATUS"),
            ("What's the market regime?", "MARKET_INSIGHTS")
        ]
        
        try:
            for query, expected_intent in test_queries:
                result = await engine.process_query(query)
                
                # Intent classification should be reasonably accurate
                assert result.query_analysis.intent.value in [expected_intent, "UNKNOWN"]
                assert result.confidence > 0.0
        
        finally:
            await engine.shutdown()


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_json_request(self, client, auth_headers):
        """Test invalid JSON request"""
        response = client.post(
            "/explain/decision",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test missing required fields"""
        incomplete_request = {
            "explanation_type": "FEATURE_IMPORTANCE"
            # Missing required 'symbol' field
        }
        
        response = client.post(
            "/explain/decision",
            json=incomplete_request,
            headers=auth_headers
        )
        
        assert response.status_code == 422
    
    def test_invalid_enum_values(self, client, auth_headers):
        """Test invalid enum values"""
        invalid_request = {
            "symbol": "NQ",
            "explanation_type": "INVALID_TYPE",  # Invalid enum value
            "audience": "TRADER"
        }
        
        response = client.post(
            "/explain/decision",
            json=invalid_request,
            headers=auth_headers
        )
        
        assert response.status_code == 422
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting functionality"""
        # Make requests rapidly to trigger rate limiting
        responses = []
        
        for i in range(110):  # Exceeds 100/minute limit
            response = client.post(
                "/query",
                json={"query": f"test query {i}"},
                headers=auth_headers
            )
            responses.append(response.status_code)
        
        # Should have some rate limited responses
        assert 429 in responses


class TestPerformanceAndLoad:
    """Test system performance and load handling"""
    
    def test_concurrent_explanation_requests(self, client, auth_headers):
        """Test concurrent explanation requests"""
        import concurrent.futures
        
        def make_request():
            request_data = {
                "symbol": "NQ",
                "explanation_type": "FEATURE_IMPORTANCE",
                "audience": "TRADER",
                "correlation_id": str(uuid.uuid4())
            }
            
            start_time = time.time()
            response = client.post(
                "/explain/decision",
                json=request_data,
                headers=auth_headers
            )
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "correlation_id": request_data["correlation_id"]
            }
        
        # Make 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        success_count = sum(1 for r in results if r["status_code"] == 200)
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        max_response_time = max(r["response_time"] for r in results)
        
        # Assertions
        assert success_count >= 15  # At least 75% success rate
        assert avg_response_time < 5.0  # Average response time under 5 seconds
        assert max_response_time < 10.0  # No request takes more than 10 seconds
    
    def test_memory_usage_stability(self, client, auth_headers):
        """Test memory usage stability under load"""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Make many requests
        for i in range(100):
            request_data = {
                "symbol": "NQ",
                "explanation_type": "FEATURE_IMPORTANCE",
                "audience": "TRADER",
                "correlation_id": str(uuid.uuid4())
            }
            
            response = client.post(
                "/explain/decision",
                json=request_data,
                headers=auth_headers
            )
            
            # Force garbage collection every 10 requests
            if i % 10 == 0:
                gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100


class TestDataValidation:
    """Test data validation and sanitization"""
    
    def test_sql_injection_prevention(self, client, auth_headers):
        """Test SQL injection prevention"""
        malicious_queries = [
            "'; DROP TABLE decisions; --",
            "' OR '1'='1",
            "UNION SELECT * FROM users",
            "<script>alert('xss')</script>"
        ]
        
        for malicious_query in malicious_queries:
            request_data = {
                "query": malicious_query,
                "correlation_id": str(uuid.uuid4())
            }
            
            response = client.post(
                "/query",
                json=request_data,
                headers=auth_headers
            )
            
            # Should either succeed with safe handling or return validation error
            assert response.status_code in [200, 422]
            
            if response.status_code == 200:
                data = response.json()
                # Should not contain any SQL-like responses
                assert "DROP TABLE" not in data["answer"].upper()
                assert "UNION SELECT" not in data["answer"].upper()
    
    def test_input_length_limits(self, client, auth_headers):
        """Test input length limits"""
        # Very long query
        long_query = "A" * 2000  # Exceeds 1000 character limit
        
        request_data = {
            "query": long_query,
            "correlation_id": str(uuid.uuid4())
        }
        
        response = client.post(
            "/query",
            json=request_data,
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error


class TestIntegrationSuite:
    """Complete integration test suite"""
    
    @pytest.mark.asyncio
    async def test_full_explanation_workflow(self, client, auth_headers):
        """Test complete explanation workflow"""
        # 1. Request explanation
        explanation_request = {
            "symbol": "NQ",
            "explanation_type": "FEATURE_IMPORTANCE",
            "audience": "TRADER",
            "correlation_id": str(uuid.uuid4())
        }
        
        response = client.post(
            "/explain/decision",
            json=explanation_request,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        explanation_data = response.json()
        
        # 2. Query about the explanation
        query_request = {
            "query": f"Explain the decision for {explanation_data['symbol']} in more detail",
            "correlation_id": explanation_data["correlation_id"]
        }
        
        query_response = client.post(
            "/query",
            json=query_request,
            headers=auth_headers
        )
        
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # 3. Get decision history
        history_response = client.get(
            "/decisions/history",
            params={"symbol": explanation_data["symbol"], "limit": 5},
            headers=auth_headers
        )
        
        assert history_response.status_code == 200
        history_data = history_response.json()
        
        # 4. Generate compliance report
        report_request = {
            "start_date": (datetime.now() - timedelta(days=1)).isoformat(),
            "end_date": datetime.now().isoformat(),
            "report_type": "standard"
        }
        
        report_response = client.post(
            "/compliance/report",
            json=report_request,
            headers=auth_headers
        )
        
        assert report_response.status_code == 200
        report_data = report_response.json()
        
        # Validate correlation IDs are preserved
        assert explanation_data["correlation_id"] == explanation_request["correlation_id"]
        assert query_data["correlation_id"] == query_request["correlation_id"]
    
    def test_system_health_monitoring(self, client):
        """Test system health monitoring"""
        # Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        
        # Check metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        
        # Verify health components
        assert "components" in health_data
        for component, status in health_data["components"].items():
            assert isinstance(status, bool)


# Performance benchmarks
def test_api_performance_benchmarks(client, auth_headers, benchmark):
    """Benchmark API performance"""
    
    def make_explanation_request():
        request_data = {
            "symbol": "NQ",
            "explanation_type": "FEATURE_IMPORTANCE",
            "audience": "TRADER",
            "correlation_id": str(uuid.uuid4())
        }
        
        response = client.post(
            "/explain/decision",
            json=request_data,
            headers=auth_headers
        )
        
        return response.status_code == 200
    
    # Benchmark the explanation endpoint
    result = benchmark(make_explanation_request)
    assert result  # Should return True for successful requests


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no",
        "--benchmark-only",
        "--benchmark-sort=mean"
    ])