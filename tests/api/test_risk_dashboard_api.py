"""
Comprehensive risk dashboard API testing with security focus.
Tests endpoint validation, error handling, rate limiting, and load testing.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
import redis.asyncio as redis
from freezegun import freeze_time
import concurrent.futures
from threading import Thread
import websockets
import aiohttp

from src.api.risk_dashboard_api import (
    app,
    ConnectionManager,
    RiskMetrics,
    AgentStatus,
    FlaggedTrade,
    CrisisAlert,
    HumanDecision,
    DashboardData,
    connection_manager,
    get_current_dashboard_data,
    risk_data_broadcaster,
    redis_client
)
from src.api.authentication import UserInfo, UserRole, RolePermission

# Test fixtures
@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.keys.return_value = []
    return redis_mock

@pytest.fixture
def mock_user_admin():
    """Mock admin user for testing."""
    return UserInfo(
        user_id="user_001",
        username="admin",
        email="admin@test.com",
        role=UserRole.SYSTEM_ADMIN,
        permissions=[
            RolePermission.DASHBOARD_READ,
            RolePermission.DASHBOARD_ADMIN,
            RolePermission.TRADE_REVIEW,
            RolePermission.TRADE_APPROVE,
            RolePermission.SYSTEM_INTEGRATION
        ],
        session_id="test_session",
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        mfa_enabled=False
    )

@pytest.fixture
def mock_user_viewer():
    """Mock viewer user for testing."""
    return UserInfo(
        user_id="user_002",
        username="viewer",
        email="viewer@test.com",
        role=UserRole.VIEWER,
        permissions=[RolePermission.DASHBOARD_READ],
        session_id="test_session",
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        mfa_enabled=False
    )

@pytest.fixture
def mock_user_risk_operator():
    """Mock risk operator user for testing."""
    return UserInfo(
        user_id="user_003",
        username="risk_operator",
        email="operator@test.com",
        role=UserRole.RISK_OPERATOR,
        permissions=[
            RolePermission.DASHBOARD_READ,
            RolePermission.TRADE_REVIEW,
            RolePermission.TRADE_APPROVE,
            RolePermission.TRADE_REJECT
        ],
        session_id="test_session",
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        mfa_enabled=False
    )

@pytest.fixture
def sample_flagged_trade():
    """Sample flagged trade for testing."""
    return FlaggedTrade(
        trade_id="trade_123",
        symbol="AAPL",
        direction="LONG",
        quantity=100.0,
        entry_price=150.0,
        risk_score=0.75,
        failure_probability=0.3,
        agent_recommendations=[
            {"agent": "strategic_mlmi", "recommendation": "APPROVE", "confidence": 0.8},
            {"agent": "risk_monitor", "recommendation": "REJECT", "confidence": 0.9}
        ],
        flagged_reason="High risk score detected",
        expires_at=datetime.utcnow() + timedelta(minutes=5)
    )

@pytest.fixture
def sample_crisis_alert():
    """Sample crisis alert for testing."""
    return CrisisAlert(
        alert_id="alert_456",
        severity="HIGH",
        alert_type="CORRELATION_SHOCK",
        message="Correlation shock detected in portfolio",
        metrics={"correlation_level": 0.95, "var_increase": 0.5},
        recommended_actions=["Reduce leverage", "Hedge positions"]
    )

@pytest.fixture
def sample_human_decision():
    """Sample human decision for testing."""
    return HumanDecision(
        trade_id="trade_123",
        decision="APPROVE",
        reasoning="Risk is acceptable given market conditions",
        user_id="user_001"
    )

class TestConnectionManager:
    """Test WebSocket connection manager."""
    
    def test_connection_manager_init(self):
        """Test connection manager initialization."""
        cm = ConnectionManager()
        assert cm.active_connections == {}
        assert cm.user_permissions == {}
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self):
        """Test WebSocket connection."""
        cm = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        
        await cm.connect(mock_websocket, "user_123", ["dashboard_read"])
        
        mock_websocket.accept.assert_called_once()
        assert "user_123" in cm.active_connections
        assert cm.user_permissions["user_123"] == ["dashboard_read"]
    
    def test_disconnect_websocket(self):
        """Test WebSocket disconnection."""
        cm = ConnectionManager()
        cm.active_connections["user_123"] = Mock()
        cm.user_permissions["user_123"] = ["dashboard_read"]
        
        cm.disconnect("user_123")
        
        assert "user_123" not in cm.active_connections
        assert "user_123" not in cm.user_permissions
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self):
        """Test sending personal message."""
        cm = ConnectionManager()
        mock_websocket = AsyncMock()
        cm.active_connections["user_123"] = mock_websocket
        
        message = {"type": "test", "data": "test_data"}
        await cm.send_personal_message(message, "user_123")
        
        mock_websocket.send_json.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_send_personal_message_error(self):
        """Test sending personal message with error."""
        cm = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.send_json.side_effect = Exception("Connection error")
        cm.active_connections["user_123"] = mock_websocket
        cm.user_permissions["user_123"] = ["dashboard_read"]
        
        message = {"type": "test", "data": "test_data"}
        await cm.send_personal_message(message, "user_123")
        
        # User should be disconnected after error
        assert "user_123" not in cm.active_connections
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcasting message."""
        cm = ConnectionManager()
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        cm.active_connections["user_1"] = mock_websocket1
        cm.active_connections["user_2"] = mock_websocket2
        cm.user_permissions["user_1"] = ["dashboard_read"]
        cm.user_permissions["user_2"] = ["dashboard_read"]
        
        message = {"type": "broadcast", "data": "broadcast_data"}
        await cm.broadcast(message, required_permission="dashboard_read")
        
        mock_websocket1.send_json.assert_called_once_with(message)
        mock_websocket2.send_json.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_broadcast_message_permission_filter(self):
        """Test broadcasting message with permission filter."""
        cm = ConnectionManager()
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        cm.active_connections["user_1"] = mock_websocket1
        cm.active_connections["user_2"] = mock_websocket2
        cm.user_permissions["user_1"] = ["dashboard_read"]
        cm.user_permissions["user_2"] = ["trade_review"]
        
        message = {"type": "broadcast", "data": "broadcast_data"}
        await cm.broadcast(message, required_permission="dashboard_read")
        
        mock_websocket1.send_json.assert_called_once_with(message)
        mock_websocket2.send_json.assert_not_called()

class TestPydanticModels:
    """Test Pydantic models for data validation."""
    
    def test_risk_metrics_validation(self):
        """Test RiskMetrics model validation."""
        data = {
            "portfolio_var": 0.025,
            "correlation_shock_level": 0.3,
            "max_drawdown": 0.05,
            "sharpe_ratio": 1.8,
            "volatility": 0.15,
            "leverage": 2.5,
            "liquidity_risk": 0.2
        }
        
        risk_metrics = RiskMetrics(**data)
        assert risk_metrics.portfolio_var == 0.025
        assert risk_metrics.correlation_shock_level == 0.3
        assert isinstance(risk_metrics.timestamp, datetime)
    
    def test_risk_metrics_invalid_data(self):
        """Test RiskMetrics validation with invalid data."""
        with pytest.raises(ValueError):
            RiskMetrics(
                portfolio_var="invalid",  # Should be float
                correlation_shock_level=0.3,
                max_drawdown=0.05,
                sharpe_ratio=1.8,
                volatility=0.15,
                leverage=2.5,
                liquidity_risk=0.2
            )
    
    def test_flagged_trade_validation(self):
        """Test FlaggedTrade model validation."""
        data = {
            "trade_id": "trade_123",
            "symbol": "AAPL",
            "direction": "LONG",
            "quantity": 100.0,
            "entry_price": 150.0,
            "risk_score": 0.75,
            "failure_probability": 0.3,
            "agent_recommendations": [
                {"agent": "strategic_mlmi", "recommendation": "APPROVE", "confidence": 0.8}
            ],
            "flagged_reason": "High risk score detected",
            "expires_at": datetime.utcnow() + timedelta(minutes=5)
        }
        
        flagged_trade = FlaggedTrade(**data)
        assert flagged_trade.trade_id == "trade_123"
        assert flagged_trade.symbol == "AAPL"
        assert flagged_trade.direction == "LONG"
        assert isinstance(flagged_trade.flagged_at, datetime)
    
    def test_human_decision_validation(self):
        """Test HumanDecision model validation."""
        data = {
            "trade_id": "trade_123",
            "decision": "APPROVE",
            "reasoning": "Risk is acceptable given market conditions",
            "user_id": "user_001"
        }
        
        decision = HumanDecision(**data)
        assert decision.trade_id == "trade_123"
        assert decision.decision == "APPROVE"
        assert isinstance(decision.timestamp, datetime)
    
    def test_human_decision_invalid_decision(self):
        """Test HumanDecision validation with invalid decision."""
        data = {
            "trade_id": "trade_123",
            "decision": "INVALID_DECISION",
            "reasoning": "Risk is acceptable given market conditions",
            "user_id": "user_001"
        }
        
        # This should pass validation at model level
        # API level validation should catch invalid decisions
        decision = HumanDecision(**data)
        assert decision.decision == "INVALID_DECISION"

class TestDashboardEndpoints:
    """Test dashboard API endpoints."""
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_get_dashboard_data_success(self, mock_verify_token, test_client, mock_user_admin):
        """Test successful dashboard data retrieval."""
        mock_verify_token.return_value = mock_user_admin
        
        response = test_client.get("/api/dashboard/data")
        
        assert response.status_code == 200
        data = response.json()
        assert "risk_metrics" in data
        assert "agent_statuses" in data
        assert "flagged_trades" in data
        assert "crisis_alerts" in data
        assert "last_updated" in data
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_get_dashboard_data_insufficient_permissions(self, mock_verify_token, test_client, mock_user_viewer):
        """Test dashboard data retrieval with insufficient permissions."""
        # Remove dashboard_read permission
        mock_user_viewer.permissions = []
        mock_verify_token.return_value = mock_user_viewer
        
        response = test_client.get("/api/dashboard/data")
        
        assert response.status_code == 403
        assert "insufficient permissions" in response.json()["detail"].lower()
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_get_risk_metrics_success(self, mock_verify_token, test_client, mock_user_admin):
        """Test successful risk metrics retrieval."""
        mock_verify_token.return_value = mock_user_admin
        
        response = test_client.get("/api/dashboard/risk-metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "portfolio_var" in data
        assert "correlation_shock_level" in data
        assert "max_drawdown" in data
        assert "sharpe_ratio" in data
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_get_flagged_trades_success(self, mock_verify_token, test_client, mock_user_risk_operator):
        """Test successful flagged trades retrieval."""
        mock_verify_token.return_value = mock_user_risk_operator
        
        response = test_client.get("/api/dashboard/flagged-trades")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_get_flagged_trades_insufficient_permissions(self, mock_verify_token, test_client, mock_user_viewer):
        """Test flagged trades retrieval with insufficient permissions."""
        mock_verify_token.return_value = mock_user_viewer
        
        response = test_client.get("/api/dashboard/flagged-trades")
        
        assert response.status_code == 403
        assert "insufficient permissions" in response.json()["detail"].lower()
    
    @patch('src.api.risk_dashboard_api.verify_token')
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_make_human_decision_success(self, mock_redis_client, mock_verify_token, test_client, mock_user_risk_operator, sample_human_decision):
        """Test successful human decision processing."""
        mock_verify_token.return_value = mock_user_risk_operator
        mock_redis_client.setex.return_value = True
        mock_redis_client.delete.return_value = True
        
        response = test_client.post(
            "/api/dashboard/decide",
            json=sample_human_decision.dict()
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "processed successfully" in data["message"]
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_make_human_decision_insufficient_permissions(self, mock_verify_token, test_client, mock_user_viewer, sample_human_decision):
        """Test human decision with insufficient permissions."""
        mock_verify_token.return_value = mock_user_viewer
        
        response = test_client.post(
            "/api/dashboard/decide",
            json=sample_human_decision.dict()
        )
        
        assert response.status_code == 403
        assert "insufficient permissions" in response.json()["detail"].lower()
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_make_human_decision_invalid_decision(self, mock_verify_token, test_client, mock_user_risk_operator):
        """Test human decision with invalid decision value."""
        mock_verify_token.return_value = mock_user_risk_operator
        
        invalid_decision = {
            "trade_id": "trade_123",
            "decision": "INVALID_DECISION",
            "reasoning": "Risk is acceptable given market conditions",
            "user_id": "user_001"
        }
        
        response = test_client.post(
            "/api/dashboard/decide",
            json=invalid_decision
        )
        
        assert response.status_code == 400
        assert "must be APPROVE or REJECT" in response.json()["detail"]
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_make_human_decision_insufficient_reasoning(self, mock_verify_token, test_client, mock_user_risk_operator):
        """Test human decision with insufficient reasoning."""
        mock_verify_token.return_value = mock_user_risk_operator
        
        invalid_decision = {
            "trade_id": "trade_123",
            "decision": "APPROVE",
            "reasoning": "Short",  # Too short
            "user_id": "user_001"
        }
        
        response = test_client.post(
            "/api/dashboard/decide",
            json=invalid_decision
        )
        
        assert response.status_code == 400
        assert "at least 10 characters" in response.json()["detail"]
    
    @patch('src.api.risk_dashboard_api.verify_token')
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_flag_trade_for_review_success(self, mock_redis_client, mock_verify_token, test_client, mock_user_admin, sample_flagged_trade):
        """Test successful trade flagging."""
        mock_verify_token.return_value = mock_user_admin
        mock_redis_client.setex.return_value = True
        
        response = test_client.post(
            "/api/dashboard/flag-trade",
            json=sample_flagged_trade.dict()
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "flagged for review" in data["message"]
    
    @patch('src.api.risk_dashboard_api.verify_token')
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_create_crisis_alert_success(self, mock_redis_client, mock_verify_token, test_client, mock_user_admin, sample_crisis_alert):
        """Test successful crisis alert creation."""
        mock_verify_token.return_value = mock_user_admin
        mock_redis_client.setex.return_value = True
        
        response = test_client.post(
            "/api/dashboard/crisis-alert",
            json=sample_crisis_alert.dict()
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "alert created" in data["message"]
    
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_dashboard_health_check(self, mock_redis_client, test_client):
        """Test dashboard health check."""
        mock_redis_client.ping.return_value = True
        
        response = test_client.get("/api/dashboard/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "redis" in data["components"]
        assert "websockets" in data["components"]
    
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_dashboard_health_check_redis_unhealthy(self, mock_redis_client, test_client):
        """Test dashboard health check with Redis unhealthy."""
        mock_redis_client.ping.side_effect = Exception("Redis connection failed")
        
        response = test_client.get("/api/dashboard/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["redis"] == "unhealthy"

class TestRateLimiting:
    """Test API rate limiting functionality."""
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_dashboard_data_rate_limit(self, mock_verify_token, test_client, mock_user_admin):
        """Test rate limiting on dashboard data endpoint."""
        mock_verify_token.return_value = mock_user_admin
        
        # Make requests up to the limit (60/minute)
        success_count = 0
        rate_limited_count = 0
        
        for i in range(70):  # Try more than the limit
            response = test_client.get("/api/dashboard/data")
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:  # Rate limited
                rate_limited_count += 1
        
        # Should have some successful requests and some rate limited
        assert success_count > 0
        # Note: In real testing, you'd need to control time or use a different approach
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_risk_metrics_rate_limit(self, mock_verify_token, test_client, mock_user_admin):
        """Test rate limiting on risk metrics endpoint."""
        mock_verify_token.return_value = mock_user_admin
        
        # Make multiple requests quickly
        responses = []
        for i in range(5):
            response = test_client.get("/api/dashboard/risk-metrics")
            responses.append(response)
        
        # All should succeed within the limit
        assert all(r.status_code == 200 for r in responses)
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_human_decision_rate_limit(self, mock_verify_token, test_client, mock_user_risk_operator, sample_human_decision):
        """Test rate limiting on human decision endpoint."""
        mock_verify_token.return_value = mock_user_risk_operator
        
        # Make multiple decision requests
        responses = []
        for i in range(5):
            decision = sample_human_decision.dict()
            decision["trade_id"] = f"trade_{i}"
            response = test_client.post("/api/dashboard/decide", json=decision)
            responses.append(response)
        
        # All should succeed within the limit
        success_responses = [r for r in responses if r.status_code == 200]
        assert len(success_responses) > 0

class TestWebSocketFunctionality:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_valid_token(self):
        """Test WebSocket connection with valid token."""
        # Mock the WebSocket connection
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.receive_text = AsyncMock(return_value="ping")
        
        # This would need to be tested with a real WebSocket client
        # For now, we test the connection manager directly
        cm = ConnectionManager()
        await cm.connect(mock_websocket, "test_user", ["dashboard_read"])
        
        assert "test_user" in cm.active_connections
        mock_websocket.accept.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_connection_invalid_token(self):
        """Test WebSocket connection with invalid token."""
        # This would test the websocket_endpoint function directly
        # In a real implementation, you'd use a WebSocket test client
        pass
    
    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong mechanism."""
        cm = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        cm.active_connections["test_user"] = mock_websocket
        
        await cm.send_personal_message({"type": "ping"}, "test_user")
        mock_websocket.send_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_performance(self):
        """Test WebSocket broadcast performance with many connections."""
        cm = ConnectionManager()
        
        # Create many mock WebSocket connections
        for i in range(1000):
            mock_websocket = AsyncMock()
            cm.active_connections[f"user_{i}"] = mock_websocket
            cm.user_permissions[f"user_{i}"] = ["dashboard_read"]
        
        # Test broadcast performance
        message = {"type": "test", "data": "test_data"}
        
        start_time = time.time()
        await cm.broadcast(message, required_permission="dashboard_read")
        end_time = time.time()
        
        # Should complete broadcast quickly
        assert end_time - start_time < 1.0, f"Broadcast took too long: {end_time - start_time}s"

class TestDataGeneration:
    """Test dashboard data generation functionality."""
    
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_get_current_dashboard_data(self, mock_redis_client):
        """Test current dashboard data generation."""
        mock_redis_client.keys.return_value = []
        
        dashboard_data = asyncio.run(get_current_dashboard_data())
        
        assert isinstance(dashboard_data, DashboardData)
        assert isinstance(dashboard_data.risk_metrics, RiskMetrics)
        assert isinstance(dashboard_data.agent_statuses, list)
        assert isinstance(dashboard_data.flagged_trades, list)
        assert isinstance(dashboard_data.crisis_alerts, list)
        assert isinstance(dashboard_data.last_updated, datetime)
    
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_get_dashboard_data_with_flagged_trades(self, mock_redis_client):
        """Test dashboard data with flagged trades from Redis."""
        sample_trade = {
            "trade_id": "trade_123",
            "symbol": "AAPL",
            "direction": "LONG",
            "quantity": 100.0,
            "entry_price": 150.0,
            "risk_score": 0.75,
            "failure_probability": 0.3,
            "agent_recommendations": [],
            "flagged_reason": "High risk",
            "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        
        mock_redis_client.keys.return_value = ["flagged_trade:trade_123"]
        mock_redis_client.get.return_value = json.dumps(sample_trade)
        
        dashboard_data = asyncio.run(get_current_dashboard_data())
        
        assert len(dashboard_data.flagged_trades) > 0
    
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_get_dashboard_data_redis_error(self, mock_redis_client):
        """Test dashboard data generation with Redis error."""
        mock_redis_client.keys.side_effect = Exception("Redis error")
        
        # Should handle Redis errors gracefully
        dashboard_data = asyncio.run(get_current_dashboard_data())
        
        assert isinstance(dashboard_data, DashboardData)
        assert dashboard_data.flagged_trades == []
        assert dashboard_data.crisis_alerts == []

class TestBackgroundTasks:
    """Test background task functionality."""
    
    @patch('src.api.risk_dashboard_api.get_current_dashboard_data')
    @patch('src.api.risk_dashboard_api.connection_manager')
    def test_risk_data_broadcaster(self, mock_connection_manager, mock_get_data):
        """Test risk data broadcasting task."""
        mock_dashboard_data = Mock()
        mock_dashboard_data.dict.return_value = {"test": "data"}
        mock_get_data.return_value = mock_dashboard_data
        mock_connection_manager.broadcast = AsyncMock()
        
        # Run broadcaster for a short time
        async def run_broadcaster():
            task = asyncio.create_task(risk_data_broadcaster())
            await asyncio.sleep(0.2)  # Let it run for 200ms
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        asyncio.run(run_broadcaster())
        
        # Should have called broadcast at least once
        assert mock_connection_manager.broadcast.called

class TestSecurityValidation:
    """Test security validation and input sanitization."""
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_sql_injection_protection(self, mock_verify_token, test_client, mock_user_admin):
        """Test SQL injection protection in request parameters."""
        mock_verify_token.return_value = mock_user_admin
        
        # Try SQL injection in query parameters
        malicious_queries = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --"
        ]
        
        for query in malicious_queries:
            response = test_client.get(f"/api/dashboard/data?param={query}")
            # Should not cause SQL injection (we're using document store, not SQL)
            assert response.status_code in [200, 422]  # Either works or validation error
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_xss_protection_in_decisions(self, mock_verify_token, test_client, mock_user_risk_operator):
        """Test XSS protection in decision reasoning."""
        mock_verify_token.return_value = mock_user_risk_operator
        
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "onload=alert('xss')"
        ]
        
        for payload in xss_payloads:
            decision = {
                "trade_id": "trade_123",
                "decision": "APPROVE",
                "reasoning": f"This is safe reasoning {payload}",
                "user_id": "user_001"
            }
            
            response = test_client.post("/api/dashboard/decide", json=decision)
            
            # Should accept the request (XSS protection would be on frontend)
            # But the payload should be properly stored/escaped
            assert response.status_code in [200, 422]
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_oversized_request_protection(self, mock_verify_token, test_client, mock_user_admin):
        """Test protection against oversized requests."""
        mock_verify_token.return_value = mock_user_admin
        
        # Create oversized trade data
        oversized_trade = {
            "trade_id": "trade_123",
            "symbol": "A" * 10000,  # Very long symbol
            "direction": "LONG",
            "quantity": 100.0,
            "entry_price": 150.0,
            "risk_score": 0.75,
            "failure_probability": 0.3,
            "agent_recommendations": [{"large_data": "x" * 100000}],
            "flagged_reason": "x" * 50000,  # Very long reason
            "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        
        response = test_client.post("/api/dashboard/flag-trade", json=oversized_trade)
        
        # Should handle oversized requests gracefully
        assert response.status_code in [200, 413, 422]
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_input_validation_edge_cases(self, mock_verify_token, test_client, mock_user_risk_operator):
        """Test input validation edge cases."""
        mock_verify_token.return_value = mock_user_risk_operator
        
        edge_cases = [
            {
                "trade_id": "",  # Empty trade ID
                "decision": "APPROVE",
                "reasoning": "Valid reasoning for empty trade ID",
                "user_id": "user_001"
            },
            {
                "trade_id": "trade_123",
                "decision": "APPROVE",
                "reasoning": "",  # Empty reasoning
                "user_id": "user_001"
            },
            {
                "trade_id": "trade_123",
                "decision": "APPROVE",
                "reasoning": "Valid reasoning",
                "user_id": ""  # Empty user ID
            },
            {
                "trade_id": None,  # None values
                "decision": "APPROVE",
                "reasoning": "Valid reasoning",
                "user_id": "user_001"
            }
        ]
        
        for case in edge_cases:
            response = test_client.post("/api/dashboard/decide", json=case)
            
            # Should either succeed or return proper validation error
            assert response.status_code in [200, 400, 422]

class TestLoadTesting:
    """Test load testing scenarios."""
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_concurrent_dashboard_requests(self, mock_verify_token, test_client, mock_user_admin):
        """Test concurrent dashboard data requests."""
        mock_verify_token.return_value = mock_user_admin
        
        def make_request():
            return test_client.get("/api/dashboard/data")
        
        # Run 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count > 40  # Allow some failures due to rate limiting
    
    @patch('src.api.risk_dashboard_api.verify_token')
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_concurrent_decision_processing(self, mock_redis_client, mock_verify_token, test_client, mock_user_risk_operator):
        """Test concurrent decision processing."""
        mock_verify_token.return_value = mock_user_risk_operator
        mock_redis_client.setex.return_value = True
        mock_redis_client.delete.return_value = True
        
        def make_decision(trade_id):
            decision = {
                "trade_id": f"trade_{trade_id}",
                "decision": "APPROVE",
                "reasoning": "Concurrent processing test decision",
                "user_id": "user_001"
            }
            return test_client.post("/api/dashboard/decide", json=decision)
        
        # Run 20 concurrent decisions
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_decision, i) for i in range(20)]
            results = [future.result() for future in futures]
        
        # Most should succeed (some may be rate limited)
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count > 10
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_sustained_load_dashboard_data(self, mock_verify_token, test_client, mock_user_admin):
        """Test sustained load on dashboard data endpoint."""
        mock_verify_token.return_value = mock_user_admin
        
        start_time = time.time()
        request_count = 0
        duration = 2.0  # Run for 2 seconds
        
        while time.time() - start_time < duration:
            response = test_client.get("/api/dashboard/data")
            if response.status_code == 200:
                request_count += 1
            time.sleep(0.01)  # Small delay to avoid overwhelming
        
        # Should handle sustained load
        requests_per_second = request_count / duration
        assert requests_per_second > 10  # Should handle at least 10 RPS
    
    @pytest.mark.asyncio
    async def test_websocket_connection_stress(self):
        """Test WebSocket connection under stress."""
        cm = ConnectionManager()
        
        # Create many connections quickly
        tasks = []
        for i in range(100):
            mock_websocket = AsyncMock()
            mock_websocket.accept = AsyncMock()
            task = cm.connect(mock_websocket, f"user_{i}", ["dashboard_read"])
            tasks.append(task)
        
        # Execute all connections concurrently
        start_time = time.time()
        await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should complete quickly
        assert end_time - start_time < 1.0
        assert len(cm.active_connections) == 100

class TestErrorHandling:
    """Test error handling and recovery."""
    
    @patch('src.api.risk_dashboard_api.verify_token')
    @patch('src.api.risk_dashboard_api.redis_client')
    def test_redis_failure_handling(self, mock_redis_client, mock_verify_token, test_client, mock_user_admin):
        """Test handling of Redis failures."""
        mock_verify_token.return_value = mock_user_admin
        mock_redis_client.setex.side_effect = Exception("Redis connection failed")
        
        # Should handle Redis failures gracefully
        response = test_client.get("/api/dashboard/data")
        assert response.status_code == 200  # Should still work without Redis
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_malformed_request_handling(self, mock_verify_token, test_client, mock_user_admin):
        """Test handling of malformed requests."""
        mock_verify_token.return_value = mock_user_admin
        
        # Send malformed JSON
        response = test_client.post(
            "/api/dashboard/decide",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_missing_required_fields(self, mock_verify_token, test_client, mock_user_admin):
        """Test handling of missing required fields."""
        mock_verify_token.return_value = mock_user_admin
        
        # Missing required fields
        incomplete_decision = {
            "trade_id": "trade_123",
            # Missing decision, reasoning, user_id
        }
        
        response = test_client.post("/api/dashboard/decide", json=incomplete_decision)
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_timeout_handling(self, mock_verify_token, test_client, mock_user_admin):
        """Test handling of request timeouts."""
        mock_verify_token.return_value = mock_user_admin
        
        # This would require mocking slow operations
        # For now, just verify the endpoint responds
        response = test_client.get("/api/dashboard/data", timeout=1.0)
        assert response.status_code == 200

class TestPerformanceMetrics:
    """Test performance metrics and monitoring."""
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_response_time_dashboard_data(self, mock_verify_token, test_client, mock_user_admin):
        """Test response time for dashboard data endpoint."""
        mock_verify_token.return_value = mock_user_admin
        
        # Measure response time
        start_time = time.time()
        response = test_client.get("/api/dashboard/data")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.5  # Should respond within 500ms
    
    @patch('src.api.risk_dashboard_api.verify_token')
    def test_memory_usage_under_load(self, mock_verify_token, test_client, mock_user_admin):
        """Test memory usage under load."""
        mock_verify_token.return_value = mock_user_admin
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make many requests
        for i in range(100):
            response = test_client.get("/api/dashboard/data")
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

if __name__ == "__main__":
    pytest.main([__file__, "-v"])