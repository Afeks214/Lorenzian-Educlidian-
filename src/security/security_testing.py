"""
Comprehensive Security Testing Framework
Automated security testing and vulnerability assessment
"""

import os
import json
import asyncio
import subprocess
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import requests
import socket
import ssl
import hashlib
import secrets
from urllib.parse import urlparse

from src.monitoring.logger_config import get_logger
from src.security.encryption import encrypt_data, decrypt_data
from src.security.audit_logger import AuditEventType, AuditSeverity

logger = get_logger(__name__)

class SecurityTestType(str, Enum):
    """Types of security tests"""
    VULNERABILITY_SCAN = "vulnerability_scan"
    PENETRATION_TEST = "penetration_test"
    AUTHENTICATION_TEST = "authentication_test"
    AUTHORIZATION_TEST = "authorization_test"
    INJECTION_TEST = "injection_test"
    XSS_TEST = "xss_test"
    CSRF_TEST = "csrf_test"
    ENCRYPTION_TEST = "encryption_test"
    NETWORK_SECURITY_TEST = "network_security_test"
    API_SECURITY_TEST = "api_security_test"
    COMPLIANCE_TEST = "compliance_test"
    SOCIAL_ENGINEERING_TEST = "social_engineering_test"

class VulnerabilitySeverity(str, Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class TestStatus(str, Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SecurityVulnerability:
    """Security vulnerability record"""
    vulnerability_id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    affected_component: Optional[str] = None
    affected_version: Optional[str] = None
    attack_vector: Optional[str] = None
    impact: Optional[str] = None
    solution: Optional[str] = None
    references: List[str] = field(default_factory=list)
    discovered_date: datetime = field(default_factory=datetime.utcnow)
    status: str = "open"  # open, fixed, accepted, false_positive
    assigned_to: Optional[str] = None
    fix_deadline: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vulnerability_id": self.vulnerability_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "cve_id": self.cve_id,
            "cvss_score": self.cvss_score,
            "affected_component": self.affected_component,
            "affected_version": self.affected_version,
            "attack_vector": self.attack_vector,
            "impact": self.impact,
            "solution": self.solution,
            "references": self.references,
            "discovered_date": self.discovered_date.isoformat(),
            "status": self.status,
            "assigned_to": self.assigned_to,
            "fix_deadline": self.fix_deadline.isoformat() if self.fix_deadline else None
        }

@dataclass
class SecurityTest:
    """Security test record"""
    test_id: str
    test_type: SecurityTestType
    test_name: str
    description: str
    target: str
    status: TestStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[int] = None  # seconds
    executed_by: Optional[str] = None
    test_parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    vulnerabilities: List[str] = field(default_factory=list)  # vulnerability IDs
    score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_type": self.test_type.value,
            "test_name": self.test_name,
            "description": self.description,
            "target": self.target,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "executed_by": self.executed_by,
            "test_parameters": self.test_parameters,
            "results": self.results,
            "vulnerabilities": self.vulnerabilities,
            "score": self.score
        }

class AuthenticationTester:
    """Authentication security testing"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    async def test_password_policy(self) -> Dict[str, Any]:
        """Test password policy enforcement"""
        test_passwords = [
            "123",  # Too short
            "password",  # Common password
            "Password123",  # Good password
            "P@ssw0rd123!",  # Strong password
            "a" * 100,  # Very long password
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection attempt
        ]
        
        results = {
            "test_name": "Password Policy Test",
            "passed": True,
            "weak_passwords_rejected": 0,
            "strong_passwords_accepted": 0,
            "vulnerabilities": []
        }
        
        for password in test_passwords:
            try:
                # Test password registration/change
                response = self.session.post(
                    f"{self.base_url}/auth/register",
                    json={
                        "username": f"testuser_{secrets.token_hex(4)}",
                        "email": f"test_{secrets.token_hex(4)}@example.com",
                        "password": password
                    }
                )
                
                # Analyze response
                if response.status_code == 400 and len(password) < 8:
                    results["weak_passwords_rejected"] += 1
                elif response.status_code == 200 and len(password) >= 8:
                    results["strong_passwords_accepted"] += 1
                elif response.status_code == 200 and len(password) < 8:
                    results["vulnerabilities"].append({
                        "type": "Weak password accepted",
                        "password": password[:10] + "..." if len(password) > 10 else password,
                        "severity": "medium"
                    })
                    results["passed"] = False
                    
            except Exception as e:
                logger.error(f"Password test error: {e}")
        
        return results
    
    async def test_brute_force_protection(self) -> Dict[str, Any]:
        """Test brute force protection"""
        results = {
            "test_name": "Brute Force Protection Test",
            "passed": True,
            "attempts_made": 0,
            "lockout_triggered": False,
            "vulnerabilities": []
        }
        
        test_username = "testuser"
        wrong_password = "wrongpassword"
        
        try:
            # Attempt multiple failed logins
            for attempt in range(10):
                response = self.session.post(
                    f"{self.base_url}/auth/login",
                    json={
                        "username": test_username,
                        "password": wrong_password
                    }
                )
                
                results["attempts_made"] += 1
                
                # Check if account is locked
                if response.status_code == 429 or "locked" in response.text.lower():
                    results["lockout_triggered"] = True
                    break
                
                await asyncio.sleep(0.1)  # Small delay
            
            if not results["lockout_triggered"]:
                results["vulnerabilities"].append({
                    "type": "No brute force protection",
                    "description": "Multiple failed login attempts not blocked",
                    "severity": "high"
                })
                results["passed"] = False
                
        except Exception as e:
            logger.error(f"Brute force test error: {e}")
        
        return results
    
    async def test_session_management(self) -> Dict[str, Any]:
        """Test session management security"""
        results = {
            "test_name": "Session Management Test",
            "passed": True,
            "vulnerabilities": []
        }
        
        try:
            # Test session timeout
            login_response = self.session.post(
                f"{self.base_url}/auth/login",
                json={
                    "username": "testuser",
                    "password": "testpassword"
                }
            )
            
            if login_response.status_code == 200:
                # Check session cookie security
                cookies = login_response.cookies
                
                for cookie in cookies:
                    if cookie.name.lower() in ['session', 'sessionid', 'jsessionid']:
                        if not cookie.secure:
                            results["vulnerabilities"].append({
                                "type": "Insecure session cookie",
                                "description": "Session cookie not marked as Secure",
                                "severity": "medium"
                            })
                            results["passed"] = False
                        
                        if not cookie.has_nonstandard_attr('HttpOnly'):
                            results["vulnerabilities"].append({
                                "type": "HttpOnly not set",
                                "description": "Session cookie not marked as HttpOnly",
                                "severity": "medium"
                            })
                            results["passed"] = False
                            
        except Exception as e:
            logger.error(f"Session management test error: {e}")
        
        return results

class InjectionTester:
    """Injection attack testing"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    async def test_sql_injection(self) -> Dict[str, Any]:
        """Test SQL injection vulnerabilities"""
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' OR '1'='1' --",
            "admin'--",
            "' OR 1=1#",
            "\" OR \"1\"=\"1",
            "1; UPDATE users SET password='hacked' WHERE username='admin'; --"
        ]
        
        results = {
            "test_name": "SQL Injection Test",
            "passed": True,
            "payloads_tested": len(sql_payloads),
            "vulnerabilities": []
        }
        
        # Test endpoints that might be vulnerable
        test_endpoints = [
            "/auth/login",
            "/api/users/search",
            "/api/trades/history",
            "/api/reports/generate"
        ]
        
        for endpoint in test_endpoints:
            for payload in sql_payloads:
                try:
                    # Test GET parameters
                    response = self.session.get(
                        f"{self.base_url}{endpoint}",
                        params={"id": payload, "search": payload}
                    )
                    
                    if self._detect_sql_injection(response):
                        results["vulnerabilities"].append({
                            "type": "SQL Injection",
                            "endpoint": endpoint,
                            "payload": payload,
                            "method": "GET",
                            "severity": "high"
                        })
                        results["passed"] = False
                    
                    # Test POST parameters
                    response = self.session.post(
                        f"{self.base_url}{endpoint}",
                        json={"username": payload, "search": payload, "id": payload}
                    )
                    
                    if self._detect_sql_injection(response):
                        results["vulnerabilities"].append({
                            "type": "SQL Injection",
                            "endpoint": endpoint,
                            "payload": payload,
                            "method": "POST",
                            "severity": "high"
                        })
                        results["passed"] = False
                        
                except Exception as e:
                    logger.error(f"SQL injection test error: {e}")
        
        return results
    
    def _detect_sql_injection(self, response: requests.Response) -> bool:
        """Detect SQL injection vulnerability indicators"""
        sql_errors = [
            "mysql_fetch_array",
            "ORA-01756",
            "Microsoft OLE DB Provider for ODBC Drivers",
            "You have an error in your SQL syntax",
            "Warning: mysql_",
            "MySQLSyntaxErrorException",
            "valid MySQL result",
            "PostgreSQL query failed",
            "Warning: pg_",
            "valid PostgreSQL result",
            "SQLite/JDBCDriver",
            "SQLite.Exception",
            "System.Data.SQLite.SQLiteException",
            "Warning: sqlite_",
            "[SQLITE_ERROR]"
        ]
        
        response_text = response.text.lower()
        
        for error in sql_errors:
            if error.lower() in response_text:
                return True
        
        # Check for unusual response patterns
        if response.status_code == 500 and "error" in response_text:
            return True
        
        return False
    
    async def test_command_injection(self) -> Dict[str, Any]:
        """Test command injection vulnerabilities"""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "; cat /etc/shadow",
            "| id",
            "; ps aux",
            "&& netstat -an",
            "; uname -a",
            "| find / -name *.conf",
            "&& cat /proc/version"
        ]
        
        results = {
            "test_name": "Command Injection Test",
            "passed": True,
            "payloads_tested": len(command_payloads),
            "vulnerabilities": []
        }
        
        # Test endpoints that might execute commands
        test_endpoints = [
            "/api/system/ping",
            "/api/files/download",
            "/api/reports/export",
            "/api/backup/create"
        ]
        
        for endpoint in test_endpoints:
            for payload in command_payloads:
                try:
                    response = self.session.post(
                        f"{self.base_url}{endpoint}",
                        json={"command": payload, "filename": payload, "path": payload}
                    )
                    
                    if self._detect_command_injection(response):
                        results["vulnerabilities"].append({
                            "type": "Command Injection",
                            "endpoint": endpoint,
                            "payload": payload,
                            "severity": "critical"
                        })
                        results["passed"] = False
                        
                except Exception as e:
                    logger.error(f"Command injection test error: {e}")
        
        return results
    
    def _detect_command_injection(self, response: requests.Response) -> bool:
        """Detect command injection vulnerability indicators"""
        command_outputs = [
            "root:",
            "uid=",
            "gid=",
            "Linux",
            "total ",
            "drwx",
            "/bin/",
            "/usr/",
            "/etc/",
            "TCP ",
            "UDP ",
            "LISTEN"
        ]
        
        response_text = response.text
        
        for output in command_outputs:
            if output in response_text:
                return True
        
        return False

class XSSTester:
    """Cross-Site Scripting (XSS) testing"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    async def test_reflected_xss(self) -> Dict[str, Any]:
        """Test reflected XSS vulnerabilities"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "<marquee onstart=alert('XSS')>"
        ]
        
        results = {
            "test_name": "Reflected XSS Test",
            "passed": True,
            "payloads_tested": len(xss_payloads),
            "vulnerabilities": []
        }
        
        # Test endpoints that might reflect input
        test_endpoints = [
            "/search",
            "/api/users/search",
            "/api/trades/filter",
            "/error",
            "/profile"
        ]
        
        for endpoint in test_endpoints:
            for payload in xss_payloads:
                try:
                    # Test GET parameters
                    response = self.session.get(
                        f"{self.base_url}{endpoint}",
                        params={"q": payload, "search": payload, "message": payload}
                    )
                    
                    if self._detect_xss_reflection(response, payload):
                        results["vulnerabilities"].append({
                            "type": "Reflected XSS",
                            "endpoint": endpoint,
                            "payload": payload,
                            "method": "GET",
                            "severity": "medium"
                        })
                        results["passed"] = False
                    
                    # Test POST parameters
                    response = self.session.post(
                        f"{self.base_url}{endpoint}",
                        json={"search": payload, "comment": payload, "message": payload}
                    )
                    
                    if self._detect_xss_reflection(response, payload):
                        results["vulnerabilities"].append({
                            "type": "Reflected XSS",
                            "endpoint": endpoint,
                            "payload": payload,
                            "method": "POST",
                            "severity": "medium"
                        })
                        results["passed"] = False
                        
                except Exception as e:
                    logger.error(f"XSS test error: {e}")
        
        return results
    
    def _detect_xss_reflection(self, response: requests.Response, payload: str) -> bool:
        """Detect XSS vulnerability by checking if payload is reflected unencoded"""
        # Check if payload appears unencoded in response
        if payload in response.text:
            return True
        
        # Check for partial reflections
        dangerous_patterns = [
            "<script",
            "javascript:",
            "onerror=",
            "onload=",
            "onfocus=",
            "onmouseover="
        ]
        
        response_text = response.text.lower()
        
        for pattern in dangerous_patterns:
            if pattern in response_text and pattern in payload.lower():
                return True
        
        return False

class NetworkSecurityTester:
    """Network security testing"""
    
    def __init__(self, target_host: str, target_port: int = 443):
        self.target_host = target_host
        self.target_port = target_port
    
    async def test_ssl_tls_configuration(self) -> Dict[str, Any]:
        """Test SSL/TLS configuration"""
        results = {
            "test_name": "SSL/TLS Configuration Test",
            "passed": True,
            "vulnerabilities": []
        }
        
        try:
            # Test SSL/TLS connection
            context = ssl.create_default_context()
            
            with socket.create_connection((self.target_host, self.target_port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=self.target_host) as ssock:
                    # Get SSL certificate info
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()
                    
                    # Check TLS version
                    if version in ['TLSv1', 'TLSv1.1']:
                        results["vulnerabilities"].append({
                            "type": "Weak TLS Version",
                            "description": f"Using deprecated TLS version: {version}",
                            "severity": "medium"
                        })
                        results["passed"] = False
                    
                    # Check cipher suite
                    if cipher:
                        cipher_name = cipher[0]
                        if any(weak in cipher_name for weak in ['RC4', 'DES', 'MD5', 'SHA1']):
                            results["vulnerabilities"].append({
                                "type": "Weak Cipher Suite",
                                "description": f"Using weak cipher: {cipher_name}",
                                "severity": "medium"
                            })
                            results["passed"] = False
                    
                    # Check certificate expiration
                    if cert:
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_until_expiry = (not_after - datetime.utcnow()).days
                        
                        if days_until_expiry < 30:
                            results["vulnerabilities"].append({
                                "type": "Certificate Expiration Warning",
                                "description": f"Certificate expires in {days_until_expiry} days",
                                "severity": "low" if days_until_expiry > 0 else "high"
                            })
                            
                            if days_until_expiry <= 0:
                                results["passed"] = False
                    
                    results["ssl_info"] = {
                        "version": version,
                        "cipher": cipher,
                        "certificate": cert
                    }
                    
        except Exception as e:
            results["vulnerabilities"].append({
                "type": "SSL/TLS Connection Error",
                "description": str(e),
                "severity": "high"
            })
            results["passed"] = False
        
        return results
    
    async def test_open_ports(self) -> Dict[str, Any]:
        """Test for open ports"""
        results = {
            "test_name": "Open Ports Test",
            "passed": True,
            "open_ports": [],
            "vulnerabilities": []
        }
        
        # Common ports to test
        test_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 3306]
        
        for port in test_ports:
            try:
                with socket.create_connection((self.target_host, port), timeout=1) as sock:
                    results["open_ports"].append(port)
                    
                    # Check for potentially dangerous open ports
                    if port in [21, 23, 25, 110, 143]:  # Unencrypted protocols
                        results["vulnerabilities"].append({
                            "type": "Unencrypted Protocol",
                            "description": f"Port {port} is open (unencrypted protocol)",
                            "severity": "medium"
                        })
                        results["passed"] = False
                    
                    if port == 3389:  # RDP
                        results["vulnerabilities"].append({
                            "type": "Remote Desktop Exposed",
                            "description": "RDP port 3389 is open to external access",
                            "severity": "high"
                        })
                        results["passed"] = False
                        
            except (socket.timeout, ConnectionRefusedError):
                pass  # Port is closed
            except Exception as e:
                logger.error(f"Port scan error for {port}: {e}")
        
        return results

class SecurityTestSuite:
    """Comprehensive security test suite"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.host = urlparse(base_url).hostname
        self.port = urlparse(base_url).port or (443 if base_url.startswith('https') else 80)
        
        self.vulnerabilities: Dict[str, SecurityVulnerability] = {}
        self.test_results: Dict[str, SecurityTest] = {}
        
        # Initialize testers
        self.auth_tester = AuthenticationTester(base_url)
        self.injection_tester = InjectionTester(base_url)
        self.xss_tester = XSSTester(base_url)
        self.network_tester = NetworkSecurityTester(self.host, self.port)
    
    async def run_comprehensive_security_test(self) -> Dict[str, Any]:
        """Run comprehensive security test suite"""
        logger.info("Starting comprehensive security test suite")
        
        test_suite_results = {
            "test_suite_id": str(secrets.token_hex(16)),
            "target": self.base_url,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "overall_status": "running",
            "tests_passed": 0,
            "tests_failed": 0,
            "vulnerabilities_found": 0,
            "security_score": 0.0,
            "test_results": {},
            "vulnerabilities": {},
            "recommendations": []
        }
        
        # Define test suite
        test_suite = [
            ("authentication_password_policy", self.auth_tester.test_password_policy),
            ("authentication_brute_force", self.auth_tester.test_brute_force_protection),
            ("authentication_session_mgmt", self.auth_tester.test_session_management),
            ("injection_sql", self.injection_tester.test_sql_injection),
            ("injection_command", self.injection_tester.test_command_injection),
            ("xss_reflected", self.xss_tester.test_reflected_xss),
            ("network_ssl_tls", self.network_tester.test_ssl_tls_configuration),
            ("network_open_ports", self.network_tester.test_open_ports)
        ]
        
        # Execute tests
        for test_name, test_func in test_suite:
            try:
                logger.info(f"Running test: {test_name}")
                
                # Create test record
                test_record = SecurityTest(
                    test_id=str(secrets.token_hex(8)),
                    test_type=SecurityTestType.VULNERABILITY_SCAN,
                    test_name=test_name,
                    description=f"Security test: {test_name}",
                    target=self.base_url,
                    status=TestStatus.RUNNING,
                    started_at=datetime.utcnow()
                )
                
                # Execute test
                start_time = datetime.utcnow()
                test_result = await test_func()
                end_time = datetime.utcnow()
                
                # Update test record
                test_record.completed_at = end_time
                test_record.duration = int((end_time - start_time).total_seconds())
                test_record.status = TestStatus.COMPLETED if test_result["passed"] else TestStatus.FAILED
                test_record.results = test_result
                
                # Store test result
                self.test_results[test_name] = test_record
                test_suite_results["test_results"][test_name] = test_record.to_dict()
                
                # Update counters
                if test_result["passed"]:
                    test_suite_results["tests_passed"] += 1
                else:
                    test_suite_results["tests_failed"] += 1
                
                # Process vulnerabilities
                if "vulnerabilities" in test_result:
                    for vuln_data in test_result["vulnerabilities"]:
                        vulnerability = self._create_vulnerability_record(vuln_data, test_name)
                        self.vulnerabilities[vulnerability.vulnerability_id] = vulnerability
                        test_suite_results["vulnerabilities"][vulnerability.vulnerability_id] = vulnerability.to_dict()
                        test_suite_results["vulnerabilities_found"] += 1
                
                logger.info(f"Test {test_name} completed: {'PASSED' if test_result['passed'] else 'FAILED'}")
                
            except Exception as e:
                logger.error(f"Test {test_name} failed with error: {e}")
                test_suite_results["tests_failed"] += 1
        
        # Calculate security score
        total_tests = test_suite_results["tests_passed"] + test_suite_results["tests_failed"]
        if total_tests > 0:
            base_score = (test_suite_results["tests_passed"] / total_tests) * 100
            
            # Apply vulnerability penalty
            critical_vulns = sum(1 for v in self.vulnerabilities.values() if v.severity == VulnerabilitySeverity.CRITICAL)
            high_vulns = sum(1 for v in self.vulnerabilities.values() if v.severity == VulnerabilitySeverity.HIGH)
            medium_vulns = sum(1 for v in self.vulnerabilities.values() if v.severity == VulnerabilitySeverity.MEDIUM)
            
            penalty = (critical_vulns * 30) + (high_vulns * 20) + (medium_vulns * 10)
            test_suite_results["security_score"] = max(0, base_score - penalty)
        
        # Generate recommendations
        test_suite_results["recommendations"] = self._generate_security_recommendations()
        
        # Finalize results
        test_suite_results["completed_at"] = datetime.utcnow().isoformat()
        test_suite_results["overall_status"] = "completed"
        
        logger.info(f"Security test suite completed. Score: {test_suite_results['security_score']:.2f}")
        
        return test_suite_results
    
    def _create_vulnerability_record(self, vuln_data: Dict[str, Any], test_name: str) -> SecurityVulnerability:
        """Create vulnerability record from test results"""
        vulnerability_id = str(secrets.token_hex(8))
        
        # Map severity
        severity_map = {
            "critical": VulnerabilitySeverity.CRITICAL,
            "high": VulnerabilitySeverity.HIGH,
            "medium": VulnerabilitySeverity.MEDIUM,
            "low": VulnerabilitySeverity.LOW,
            "info": VulnerabilitySeverity.INFO
        }
        
        severity = severity_map.get(vuln_data.get("severity", "medium"), VulnerabilitySeverity.MEDIUM)
        
        # Calculate fix deadline based on severity
        fix_deadline = None
        if severity == VulnerabilitySeverity.CRITICAL:
            fix_deadline = datetime.utcnow() + timedelta(days=1)
        elif severity == VulnerabilitySeverity.HIGH:
            fix_deadline = datetime.utcnow() + timedelta(days=7)
        elif severity == VulnerabilitySeverity.MEDIUM:
            fix_deadline = datetime.utcnow() + timedelta(days=30)
        
        return SecurityVulnerability(
            vulnerability_id=vulnerability_id,
            title=vuln_data.get("type", "Security Vulnerability"),
            description=vuln_data.get("description", ""),
            severity=severity,
            affected_component=vuln_data.get("endpoint", test_name),
            attack_vector=vuln_data.get("payload", ""),
            fix_deadline=fix_deadline
        )
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on test results"""
        recommendations = []
        
        # General recommendations
        recommendations.append("Implement regular security testing and vulnerability assessments")
        recommendations.append("Keep all software and dependencies up to date")
        recommendations.append("Implement proper input validation and sanitization")
        recommendations.append("Use HTTPS for all communications")
        recommendations.append("Implement proper authentication and authorization controls")
        
        # Specific recommendations based on vulnerabilities
        vuln_types = [v.title for v in self.vulnerabilities.values()]
        
        if "SQL Injection" in vuln_types:
            recommendations.append("Use parameterized queries and prepared statements to prevent SQL injection")
        
        if "Reflected XSS" in vuln_types:
            recommendations.append("Implement proper output encoding and Content Security Policy (CSP)")
        
        if "Weak TLS Version" in vuln_types:
            recommendations.append("Upgrade to TLS 1.2 or higher and disable weak cipher suites")
        
        if "No brute force protection" in vuln_types:
            recommendations.append("Implement account lockout and rate limiting for login attempts")
        
        if "Weak password accepted" in vuln_types:
            recommendations.append("Enforce strong password policies and consider multi-factor authentication")
        
        return recommendations
    
    async def generate_security_report(self) -> str:
        """Generate comprehensive security report"""
        report_data = {
            "report_id": str(secrets.token_hex(16)),
            "generated_at": datetime.utcnow().isoformat(),
            "target": self.base_url,
            "executive_summary": {
                "total_tests": len(self.test_results),
                "tests_passed": sum(1 for t in self.test_results.values() if t.status == TestStatus.COMPLETED),
                "tests_failed": sum(1 for t in self.test_results.values() if t.status == TestStatus.FAILED),
                "vulnerabilities_found": len(self.vulnerabilities),
                "critical_vulnerabilities": sum(1 for v in self.vulnerabilities.values() if v.severity == VulnerabilitySeverity.CRITICAL),
                "high_vulnerabilities": sum(1 for v in self.vulnerabilities.values() if v.severity == VulnerabilitySeverity.HIGH),
                "medium_vulnerabilities": sum(1 for v in self.vulnerabilities.values() if v.severity == VulnerabilitySeverity.MEDIUM),
                "low_vulnerabilities": sum(1 for v in self.vulnerabilities.values() if v.severity == VulnerabilitySeverity.LOW)
            },
            "test_results": {test_id: test.to_dict() for test_id, test in self.test_results.items()},
            "vulnerabilities": {vuln_id: vuln.to_dict() for vuln_id, vuln in self.vulnerabilities.items()},
            "recommendations": self._generate_security_recommendations()
        }
        
        # Save report to file
        reports_dir = Path("/var/log/grandmodel/security/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = reports_dir / f"security_report_{report_data['report_id']}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Security report generated: {report_file}")
        
        return str(report_file)

# Global security test suite
security_test_suite: Optional[SecurityTestSuite] = None

def get_security_test_suite(base_url: str) -> SecurityTestSuite:
    """Get or create security test suite"""
    global security_test_suite
    
    if security_test_suite is None or security_test_suite.base_url != base_url:
        security_test_suite = SecurityTestSuite(base_url)
    
    return security_test_suite

# Convenience functions
async def run_security_scan(base_url: str) -> Dict[str, Any]:
    """Run comprehensive security scan"""
    test_suite = get_security_test_suite(base_url)
    return await test_suite.run_comprehensive_security_test()

async def generate_security_report(base_url: str) -> str:
    """Generate security report"""
    test_suite = get_security_test_suite(base_url)
    return await test_suite.generate_security_report()

async def test_authentication_security(base_url: str) -> Dict[str, Any]:
    """Test authentication security"""
    auth_tester = AuthenticationTester(base_url)
    
    results = {
        "password_policy": await auth_tester.test_password_policy(),
        "brute_force_protection": await auth_tester.test_brute_force_protection(),
        "session_management": await auth_tester.test_session_management()
    }
    
    return results

async def test_injection_vulnerabilities(base_url: str) -> Dict[str, Any]:
    """Test injection vulnerabilities"""
    injection_tester = InjectionTester(base_url)
    
    results = {
        "sql_injection": await injection_tester.test_sql_injection(),
        "command_injection": await injection_tester.test_command_injection()
    }
    
    return results

async def test_xss_vulnerabilities(base_url: str) -> Dict[str, Any]:
    """Test XSS vulnerabilities"""
    xss_tester = XSSTester(base_url)
    
    results = {
        "reflected_xss": await xss_tester.test_reflected_xss()
    }
    
    return results

async def test_network_security(host: str, port: int = 443) -> Dict[str, Any]:
    """Test network security"""
    network_tester = NetworkSecurityTester(host, port)
    
    results = {
        "ssl_tls_configuration": await network_tester.test_ssl_tls_configuration(),
        "open_ports": await network_tester.test_open_ports()
    }
    
    return results
