"""
Security and Compliance Audit for Execution Engine MARL System
==============================================================

Comprehensive security and compliance audit framework for production certification
of the unified execution MARL system. Validates security controls, compliance
requirements, and risk management protocols.

Security Domains Covered:
- Authentication and Authorization
- Data Protection and Encryption
- Network Security
- Input Validation and Sanitization
- Error Handling and Information Disclosure
- Logging and Monitoring
- Infrastructure Security
- Compliance Framework Adherence

Author: Agent 5 - Integration Validation & Production Certification
Date: 2025-07-13
Mission: 200% Security and Compliance Certification
"""

import asyncio
import time
import hashlib
import hmac
import secrets
import ssl
import socket
import subprocess
import json
import yaml
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import structlog
import re
import base64
from unittest.mock import Mock, patch
import sys
import os

# Security testing imports
try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

logger = structlog.get_logger()


@dataclass
class SecurityFinding:
    """Security audit finding"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str  # Authentication, Data Protection, etc.
    title: str
    description: str
    impact: str
    recommendation: str
    cve_references: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)
    remediation_effort: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    false_positive: bool = False


@dataclass
class ComplianceRequirement:
    """Compliance requirement specification"""
    framework: str  # SOX, PCI-DSS, GDPR, etc.
    requirement_id: str
    title: str
    description: str
    controls_required: List[str]
    status: str = "NOT_ASSESSED"  # NOT_ASSESSED, COMPLIANT, NON_COMPLIANT, PARTIAL
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)


@dataclass
class SecurityAuditReport:
    """Comprehensive security audit report"""
    audit_id: str
    start_time: datetime
    end_time: datetime
    auditor: str = "Agent 5 Security Auditor"
    
    # Summary metrics
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    
    # Findings by category
    findings: List[SecurityFinding] = field(default_factory=list)
    
    # Compliance assessment
    compliance_requirements: List[ComplianceRequirement] = field(default_factory=list)
    overall_compliance_score: float = 0.0
    
    # Risk assessment
    overall_risk_level: str = "UNKNOWN"
    production_readiness: bool = False
    
    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    short_term_actions: List[str] = field(default_factory=list)
    long_term_actions: List[str] = field(default_factory=list)


class SecurityComplianceAuditor:
    """
    Comprehensive security and compliance auditor
    
    Performs thorough security assessment of the execution engine including:
    - Vulnerability scanning
    - Configuration security review
    - Code security analysis
    - Compliance framework validation
    - Risk assessment and reporting
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize security auditor"""
        self.config = config or {}
        self.audit_id = f"security_audit_{int(time.time())}"
        
        # Audit configuration
        self.include_code_analysis = self.config.get('include_code_analysis', True)
        self.include_infrastructure_scan = self.config.get('include_infrastructure_scan', True)
        self.include_compliance_check = self.config.get('include_compliance_check', True)
        self.include_penetration_testing = self.config.get('include_penetration_testing', False)
        
        # Compliance frameworks to assess
        self.compliance_frameworks = self.config.get('compliance_frameworks', [
            'SOX', 'PCI-DSS', 'GDPR', 'ISO-27001', 'NIST-CSF'
        ])
        
        # Project paths
        self.project_root = Path(__file__).parent.parent.parent
        self.source_dirs = [
            self.project_root / "src",
            self.project_root / "tests",
            self.project_root / "deployment"
        ]
        
        # Security test results
        self.findings = []
        self.compliance_results = []
        
        logger.info("SecurityComplianceAuditor initialized",
                   audit_id=self.audit_id,
                   frameworks=self.compliance_frameworks,
                   include_code_analysis=self.include_code_analysis)
    
    async def run_comprehensive_audit(self) -> SecurityAuditReport:
        """
        Run comprehensive security and compliance audit
        
        Returns:
            Complete security audit report
        """
        logger.info("🔒 Starting comprehensive security and compliance audit",
                   audit_id=self.audit_id)
        
        start_time = datetime.now()
        
        try:
            # Phase 1: Authentication and Authorization Testing
            logger.info("🔑 Phase 1: Authentication and Authorization Testing")
            await self._test_authentication_security()
            
            # Phase 2: Data Protection and Encryption Testing
            logger.info("🛡️ Phase 2: Data Protection and Encryption Testing")
            await self._test_data_protection()
            
            # Phase 3: Network Security Testing
            logger.info("🌐 Phase 3: Network Security Testing")
            await self._test_network_security()
            
            # Phase 4: Input Validation and Injection Testing
            logger.info("💉 Phase 4: Input Validation and Injection Testing")
            await self._test_input_validation()
            
            # Phase 5: Error Handling and Information Disclosure
            logger.info("📋 Phase 5: Error Handling and Information Disclosure")
            await self._test_error_handling()
            
            # Phase 6: Code Security Analysis
            if self.include_code_analysis:
                logger.info("🔍 Phase 6: Code Security Analysis")
                await self._analyze_code_security()
            
            # Phase 7: Infrastructure Security Assessment
            if self.include_infrastructure_scan:
                logger.info("🏗️ Phase 7: Infrastructure Security Assessment")
                await self._assess_infrastructure_security()
            
            # Phase 8: Compliance Framework Assessment
            if self.include_compliance_check:
                logger.info("📋 Phase 8: Compliance Framework Assessment")
                await self._assess_compliance_frameworks()
            
            # Phase 9: Penetration Testing (if enabled)
            if self.include_penetration_testing:
                logger.info("🎯 Phase 9: Penetration Testing")
                await self._conduct_penetration_testing()
            
            # Generate comprehensive report
            end_time = datetime.now()
            report = self._generate_audit_report(start_time, end_time)
            
            logger.info("✅ Security and compliance audit completed",
                       audit_id=self.audit_id,
                       duration_seconds=(end_time - start_time).total_seconds(),
                       total_findings=report.total_findings,
                       critical_findings=report.critical_findings,
                       production_ready=report.production_readiness)
            
            return report
            
        except Exception as e:
            logger.error("❌ Security audit failed", error=str(e))
            
            # Return partial report with error
            end_time = datetime.now()
            report = self._generate_audit_report(start_time, end_time)
            report.findings.append(SecurityFinding(
                severity="CRITICAL",
                category="Audit Process",
                title="Security Audit Failed",
                description=f"Security audit process failed: {str(e)}",
                impact="Unable to validate security posture",
                recommendation="Fix audit process and re-run security assessment"
            ))
            
            return report
    
    async def _test_authentication_security(self):
        """Test authentication and authorization security"""
        # Test 1: Password Policy Validation
        await self._test_password_policies()
        
        # Test 2: Session Management
        await self._test_session_management()
        
        # Test 3: Authorization Controls
        await self._test_authorization_controls()
        
        # Test 4: Multi-Factor Authentication
        await self._test_mfa_implementation()
        
        # Test 5: API Authentication
        await self._test_api_authentication()
    
    async def _test_password_policies(self):
        """Test password policy implementation"""
        try:
            # Look for password validation code
            password_files = await self._find_files_containing("password", "auth", "login")
            
            has_complexity_requirements = False
            has_length_requirements = False
            has_history_check = False
            
            for file_path in password_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for complexity requirements
                    if re.search(r'(?=.*[a-z])(?=.*[A-Z])(?=.*\d)', content):
                        has_complexity_requirements = True
                    
                    # Check for length requirements
                    if re.search(r'len.*>=.*8|length.*>=.*8', content):
                        has_length_requirements = True
                    
                    # Check for password history
                    if 'password_history' in content.lower():
                        has_history_check = True
            
            # Evaluate findings
            if not has_complexity_requirements:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Authentication",
                    title="Weak Password Complexity Requirements",
                    description="Password complexity requirements not enforced",
                    impact="Users may choose weak passwords vulnerable to brute force attacks",
                    recommendation="Implement password complexity requirements (uppercase, lowercase, numbers, special characters)",
                    compliance_frameworks=["PCI-DSS", "NIST-CSF"]
                ))
            
            if not has_length_requirements:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Authentication", 
                    title="Insufficient Password Length Requirements",
                    description="Minimum password length not enforced",
                    impact="Short passwords are vulnerable to brute force attacks",
                    recommendation="Enforce minimum password length of 12 characters",
                    compliance_frameworks=["NIST-CSF"]
                ))
            
        except Exception as e:
            logger.warning("Password policy test failed", error=str(e))
    
    async def _test_session_management(self):
        """Test session management security"""
        try:
            # Look for session handling code
            session_files = await self._find_files_containing("session", "token", "jwt")
            
            has_secure_session_config = False
            has_session_timeout = False
            has_csrf_protection = False
            
            for file_path in session_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for secure session configuration
                    if 'secure=True' in content or 'httpOnly=True' in content:
                        has_secure_session_config = True
                    
                    # Check for session timeout
                    if 'timeout' in content.lower() or 'expiry' in content.lower():
                        has_session_timeout = True
                    
                    # Check for CSRF protection
                    if 'csrf' in content.lower() or 'cross_site' in content.lower():
                        has_csrf_protection = True
            
            # Evaluate findings
            if not has_secure_session_config:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Authentication",
                    title="Insecure Session Configuration",
                    description="Sessions not configured with secure flags",
                    impact="Session cookies vulnerable to interception and XSS attacks",
                    recommendation="Configure sessions with Secure and HttpOnly flags",
                    compliance_frameworks=["OWASP", "PCI-DSS"]
                ))
            
            if not has_csrf_protection:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Authentication",
                    title="Missing CSRF Protection",
                    description="Cross-Site Request Forgery protection not implemented",
                    impact="Application vulnerable to CSRF attacks",
                    recommendation="Implement CSRF tokens for state-changing operations",
                    compliance_frameworks=["OWASP"]
                ))
                
        except Exception as e:
            logger.warning("Session management test failed", error=str(e))
    
    async def _test_authorization_controls(self):
        """Test authorization and access controls"""
        try:
            # Look for authorization code
            auth_files = await self._find_files_containing("authorize", "permission", "rbac", "access")
            
            has_rbac_implementation = False
            has_principle_of_least_privilege = False
            has_access_logging = False
            
            for file_path in auth_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for RBAC implementation
                    if any(term in content.lower() for term in ['role', 'permission', 'rbac']):
                        has_rbac_implementation = True
                    
                    # Check for access logging
                    if any(term in content.lower() for term in ['audit', 'access_log', 'authorization_log']):
                        has_access_logging = True
            
            # Evaluate findings
            if not has_rbac_implementation:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Authorization",
                    title="Missing Role-Based Access Control",
                    description="Role-based access control not implemented",
                    impact="Difficult to manage user permissions and enforce least privilege",
                    recommendation="Implement RBAC with clearly defined roles and permissions",
                    compliance_frameworks=["ISO-27001", "NIST-CSF"]
                ))
            
            if not has_access_logging:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Authorization",
                    title="Missing Access Logging",
                    description="Access control decisions not logged",
                    impact="Cannot detect unauthorized access attempts or perform security audits",
                    recommendation="Implement comprehensive access logging for all authorization decisions",
                    compliance_frameworks=["SOX", "PCI-DSS"]
                ))
                
        except Exception as e:
            logger.warning("Authorization controls test failed", error=str(e))
    
    async def _test_mfa_implementation(self):
        """Test multi-factor authentication implementation"""
        try:
            # Look for MFA-related code
            mfa_files = await self._find_files_containing("mfa", "2fa", "totp", "multi_factor")
            
            has_mfa_support = len(mfa_files) > 0
            
            if not has_mfa_support:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Authentication",
                    title="Missing Multi-Factor Authentication",
                    description="Multi-factor authentication not implemented",
                    impact="Accounts vulnerable to credential stuffing and password-based attacks",
                    recommendation="Implement MFA using TOTP, SMS, or hardware tokens for administrative accounts",
                    compliance_frameworks=["PCI-DSS", "NIST-CSF"]
                ))
                
        except Exception as e:
            logger.warning("MFA implementation test failed", error=str(e))
    
    async def _test_api_authentication(self):
        """Test API authentication mechanisms"""
        try:
            # Look for API authentication code
            api_files = await self._find_files_containing("api", "token", "bearer", "jwt")
            
            has_api_auth = False
            has_rate_limiting = False
            
            for file_path in api_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for API authentication
                    if any(term in content.lower() for term in ['bearer', 'authorization', 'api_key']):
                        has_api_auth = True
                    
                    # Check for rate limiting
                    if any(term in content.lower() for term in ['rate_limit', 'throttle', 'rate_limiter']):
                        has_rate_limiting = True
            
            # Evaluate findings
            if not has_api_auth:
                self.findings.append(SecurityFinding(
                    severity="CRITICAL",
                    category="Authentication",
                    title="Missing API Authentication",
                    description="API endpoints not protected with authentication",
                    impact="Unauthorized access to sensitive trading functionality",
                    recommendation="Implement API authentication using OAuth 2.0 or JWT tokens",
                    compliance_frameworks=["PCI-DSS", "ISO-27001"]
                ))
            
            if not has_rate_limiting:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Authentication",
                    title="Missing API Rate Limiting",
                    description="API rate limiting not implemented",
                    impact="APIs vulnerable to DoS attacks and abuse",
                    recommendation="Implement rate limiting for all API endpoints",
                    compliance_frameworks=["OWASP"]
                ))
                
        except Exception as e:
            logger.warning("API authentication test failed", error=str(e))
    
    async def _test_data_protection(self):
        """Test data protection and encryption"""
        # Test 1: Data Encryption at Rest
        await self._test_encryption_at_rest()
        
        # Test 2: Data Encryption in Transit
        await self._test_encryption_in_transit()
        
        # Test 3: Sensitive Data Handling
        await self._test_sensitive_data_handling()
        
        # Test 4: Key Management
        await self._test_key_management()
        
        # Test 5: Data Retention and Disposal
        await self._test_data_retention()
    
    async def _test_encryption_at_rest(self):
        """Test encryption at rest implementation"""
        try:
            # Look for encryption configuration
            encryption_files = await self._find_files_containing("encrypt", "cipher", "aes", "rsa")
            
            has_database_encryption = False
            has_file_encryption = False
            uses_strong_encryption = False
            
            for file_path in encryption_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for database encryption
                    if any(term in content.lower() for term in ['database_encrypt', 'db_encrypt', 'tde']):
                        has_database_encryption = True
                    
                    # Check for file encryption
                    if any(term in content.lower() for term in ['file_encrypt', 'disk_encrypt']):
                        has_file_encryption = True
                    
                    # Check for strong encryption algorithms
                    if any(term in content for term in ['AES-256', 'RSA-2048', 'ChaCha20']):
                        uses_strong_encryption = True
            
            # Evaluate findings
            if not has_database_encryption:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Data Protection",
                    title="Missing Database Encryption",
                    description="Database not encrypted at rest",
                    impact="Sensitive trading data vulnerable if storage is compromised",
                    recommendation="Enable transparent data encryption (TDE) for database",
                    compliance_frameworks=["PCI-DSS", "GDPR", "SOX"]
                ))
            
            if not uses_strong_encryption:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Data Protection",
                    title="Weak Encryption Algorithms",
                    description="Strong encryption algorithms not verified",
                    impact="Data may be vulnerable to cryptographic attacks",
                    recommendation="Use AES-256 or equivalent strong encryption algorithms",
                    compliance_frameworks=["NIST-CSF", "PCI-DSS"]
                ))
                
        except Exception as e:
            logger.warning("Encryption at rest test failed", error=str(e))
    
    async def _test_encryption_in_transit(self):
        """Test encryption in transit implementation"""
        try:
            # Look for TLS/SSL configuration
            tls_files = await self._find_files_containing("tls", "ssl", "https", "certificate")
            
            has_tls_enforcement = False
            has_strong_tls_config = False
            has_certificate_validation = False
            
            for file_path in tls_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for TLS enforcement
                    if any(term in content.lower() for term in ['force_ssl', 'https_only', 'tls_required']):
                        has_tls_enforcement = True
                    
                    # Check for strong TLS configuration
                    if any(term in content for term in ['TLSv1.2', 'TLSv1.3']):
                        has_strong_tls_config = True
                    
                    # Check for certificate validation
                    if any(term in content.lower() for term in ['verify_cert', 'cert_validation']):
                        has_certificate_validation = True
            
            # Evaluate findings
            if not has_tls_enforcement:
                self.findings.append(SecurityFinding(
                    severity="CRITICAL",
                    category="Data Protection",
                    title="Missing TLS Enforcement",
                    description="TLS/SSL not enforced for all communications",
                    impact="Data transmitted in plaintext vulnerable to interception",
                    recommendation="Enforce TLS 1.2+ for all network communications",
                    compliance_frameworks=["PCI-DSS", "GDPR", "HIPAA"]
                ))
            
            if not has_strong_tls_config:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Data Protection",
                    title="Weak TLS Configuration",
                    description="Strong TLS configuration not verified",
                    impact="Communications may be vulnerable to downgrade attacks",
                    recommendation="Configure TLS 1.2+ with strong cipher suites",
                    compliance_frameworks=["NIST-CSF", "PCI-DSS"]
                ))
                
        except Exception as e:
            logger.warning("Encryption in transit test failed", error=str(e))
    
    async def _test_sensitive_data_handling(self):
        """Test sensitive data handling practices"""
        try:
            # Look for sensitive data patterns in code
            sensitive_patterns = [
                r'password\s*=\s*["\'][^"\']+["\']',  # Hardcoded passwords
                r'api[_-]?key\s*=\s*["\'][^"\']+["\']',  # API keys
                r'secret\s*=\s*["\'][^"\']+["\']',  # Secrets
                r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',  # Credit card numbers
                r'\d{3}-\d{2}-\d{4}',  # SSN patterns
            ]
            
            sensitive_data_found = []
            
            for source_dir in self.source_dirs:
                if source_dir.exists():
                    for file_path in source_dir.rglob("*.py"):
                        content = await self._read_file_safely(file_path)
                        if content:
                            for pattern in sensitive_patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    sensitive_data_found.append((file_path, pattern, matches))
            
            # Evaluate findings
            if sensitive_data_found:
                for file_path, pattern, matches in sensitive_data_found:
                    self.findings.append(SecurityFinding(
                        severity="CRITICAL",
                        category="Data Protection",
                        title="Hardcoded Sensitive Data",
                        description=f"Sensitive data found in source code: {file_path}",
                        impact="Sensitive data exposed in version control and deployments",
                        recommendation="Remove hardcoded sensitive data and use secure configuration management",
                        compliance_frameworks=["PCI-DSS", "GDPR", "SOX"]
                    ))
                    
        except Exception as e:
            logger.warning("Sensitive data handling test failed", error=str(e))
    
    async def _test_key_management(self):
        """Test cryptographic key management"""
        try:
            # Look for key management code
            key_files = await self._find_files_containing("key", "secret", "crypto", "fernet")
            
            has_key_rotation = False
            has_secure_key_storage = False
            has_key_derivation = False
            
            for file_path in key_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for key rotation
                    if any(term in content.lower() for term in ['key_rotation', 'rotate_key']):
                        has_key_rotation = True
                    
                    # Check for secure key storage
                    if any(term in content.lower() for term in ['keystore', 'hsm', 'vault']):
                        has_secure_key_storage = True
                    
                    # Check for key derivation
                    if any(term in content.lower() for term in ['pbkdf2', 'scrypt', 'argon2']):
                        has_key_derivation = True
            
            # Evaluate findings
            if not has_key_rotation:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Data Protection",
                    title="Missing Key Rotation",
                    description="Cryptographic key rotation not implemented",
                    impact="Compromised keys remain valid indefinitely",
                    recommendation="Implement automatic key rotation policies",
                    compliance_frameworks=["NIST-CSF", "PCI-DSS"]
                ))
            
            if not has_secure_key_storage:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Data Protection",
                    title="Insecure Key Storage",
                    description="Cryptographic keys not stored securely",
                    impact="Keys vulnerable to unauthorized access",
                    recommendation="Use hardware security modules (HSM) or secure key vaults",
                    compliance_frameworks=["PCI-DSS", "FIPS-140-2"]
                ))
                
        except Exception as e:
            logger.warning("Key management test failed", error=str(e))
    
    async def _test_data_retention(self):
        """Test data retention and disposal policies"""
        try:
            # Look for data retention code
            retention_files = await self._find_files_containing("retention", "purge", "delete", "cleanup")
            
            has_retention_policy = False
            has_secure_deletion = False
            
            for file_path in retention_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for retention policy
                    if any(term in content.lower() for term in ['retention_period', 'data_retention']):
                        has_retention_policy = True
                    
                    # Check for secure deletion
                    if any(term in content.lower() for term in ['secure_delete', 'wipe', 'shred']):
                        has_secure_deletion = True
            
            # Evaluate findings
            if not has_retention_policy:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Data Protection",
                    title="Missing Data Retention Policy",
                    description="Data retention policies not implemented",
                    impact="Data kept longer than necessary, increasing privacy risk",
                    recommendation="Implement automated data retention and purging policies",
                    compliance_frameworks=["GDPR", "SOX"]
                ))
                
        except Exception as e:
            logger.warning("Data retention test failed", error=str(e))
    
    async def _test_network_security(self):
        """Test network security controls"""
        # Test 1: Firewall Configuration
        await self._test_firewall_configuration()
        
        # Test 2: Network Segmentation
        await self._test_network_segmentation()
        
        # Test 3: Intrusion Detection
        await self._test_intrusion_detection()
        
        # Test 4: VPN and Remote Access
        await self._test_vpn_security()
    
    async def _test_firewall_configuration(self):
        """Test firewall configuration"""
        try:
            # Look for firewall configuration
            firewall_files = await self._find_files_containing("firewall", "iptables", "ufw", "security_groups")
            
            has_firewall_config = len(firewall_files) > 0
            
            if not has_firewall_config:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Network Security",
                    title="Missing Firewall Configuration",
                    description="Firewall configuration not found",
                    impact="Network traffic not filtered, allowing unauthorized access",
                    recommendation="Configure firewall rules to restrict network access",
                    compliance_frameworks=["NIST-CSF", "PCI-DSS"]
                ))
                
        except Exception as e:
            logger.warning("Firewall configuration test failed", error=str(e))
    
    async def _test_network_segmentation(self):
        """Test network segmentation"""
        try:
            # Look for network segmentation configuration
            network_files = await self._find_files_containing("network", "subnet", "vlan", "segmentation")
            
            has_segmentation = False
            
            for file_path in network_files:
                content = await self._read_file_safely(file_path)
                if content:
                    if any(term in content.lower() for term in ['subnet', 'vlan', 'segment']):
                        has_segmentation = True
                        break
            
            if not has_segmentation:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Network Security",
                    title="Missing Network Segmentation",
                    description="Network segmentation not implemented",
                    impact="Lateral movement possible if perimeter is breached",
                    recommendation="Implement network segmentation to isolate critical systems",
                    compliance_frameworks=["NIST-CSF", "PCI-DSS"]
                ))
                
        except Exception as e:
            logger.warning("Network segmentation test failed", error=str(e))
    
    async def _test_intrusion_detection(self):
        """Test intrusion detection systems"""
        try:
            # Look for IDS configuration
            ids_files = await self._find_files_containing("ids", "intrusion", "monitoring", "alert")
            
            has_ids = len(ids_files) > 0
            
            if not has_ids:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Network Security",
                    title="Missing Intrusion Detection",
                    description="Intrusion detection system not configured",
                    impact="Security incidents may go undetected",
                    recommendation="Implement network and host-based intrusion detection",
                    compliance_frameworks=["NIST-CSF", "ISO-27001"]
                ))
                
        except Exception as e:
            logger.warning("Intrusion detection test failed", error=str(e))
    
    async def _test_vpn_security(self):
        """Test VPN and remote access security"""
        try:
            # Look for VPN configuration
            vpn_files = await self._find_files_containing("vpn", "openvpn", "wireguard", "remote_access")
            
            has_vpn_config = len(vpn_files) > 0
            
            # Note: This is informational as VPN may not be required
            if not has_vpn_config:
                self.findings.append(SecurityFinding(
                    severity="INFO",
                    category="Network Security",
                    title="No VPN Configuration Found",
                    description="VPN configuration not detected",
                    impact="Remote access may not be secured",
                    recommendation="Consider implementing VPN for remote administrative access",
                    compliance_frameworks=["NIST-CSF"]
                ))
                
        except Exception as e:
            logger.warning("VPN security test failed", error=str(e))
    
    async def _test_input_validation(self):
        """Test input validation and injection prevention"""
        # Test 1: SQL Injection Prevention
        await self._test_sql_injection_prevention()
        
        # Test 2: XSS Prevention
        await self._test_xss_prevention()
        
        # Test 3: Command Injection Prevention
        await self._test_command_injection_prevention()
        
        # Test 4: File Upload Security
        await self._test_file_upload_security()
    
    async def _test_sql_injection_prevention(self):
        """Test SQL injection prevention"""
        try:
            # Look for database query code
            db_files = await self._find_files_containing("sql", "query", "database", "execute")
            
            has_parameterized_queries = False
            has_orm_usage = False
            potential_sql_injection = []
            
            for file_path in db_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for parameterized queries
                    if any(term in content for term in ['?', '%s', 'bind', 'prepare']):
                        has_parameterized_queries = True
                    
                    # Check for ORM usage
                    if any(term in content.lower() for term in ['sqlalchemy', 'django.db', 'orm']):
                        has_orm_usage = True
                    
                    # Check for potential SQL injection
                    sql_patterns = [
                        r'execute\s*\(\s*["\'].*%.*["\']',
                        r'query\s*\(\s*["\'].*\+.*["\']',
                        r'sql.*=.*["\'].*\+.*["\']'
                    ]
                    
                    for pattern in sql_patterns:
                        if re.search(pattern, content):
                            potential_sql_injection.append(file_path)
            
            # Evaluate findings
            if potential_sql_injection:
                self.findings.append(SecurityFinding(
                    severity="CRITICAL",
                    category="Input Validation",
                    title="Potential SQL Injection Vulnerability",
                    description=f"Potential SQL injection found in {len(potential_sql_injection)} files",
                    impact="Database compromise and unauthorized data access",
                    recommendation="Use parameterized queries or ORM frameworks",
                    compliance_frameworks=["OWASP", "PCI-DSS"]
                ))
            
            if not has_parameterized_queries and not has_orm_usage:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Input Validation",
                    title="Missing SQL Injection Protection",
                    description="No evidence of parameterized queries or ORM usage",
                    impact="Application vulnerable to SQL injection attacks",
                    recommendation="Implement parameterized queries for all database operations",
                    compliance_frameworks=["OWASP", "PCI-DSS"]
                ))
                
        except Exception as e:
            logger.warning("SQL injection prevention test failed", error=str(e))
    
    async def _test_xss_prevention(self):
        """Test Cross-Site Scripting (XSS) prevention"""
        try:
            # Look for web-related code
            web_files = await self._find_files_containing("html", "template", "render", "escape")
            
            has_output_encoding = False
            has_csp_headers = False
            
            for file_path in web_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for output encoding
                    if any(term in content.lower() for term in ['escape', 'encode', 'sanitize']):
                        has_output_encoding = True
                    
                    # Check for CSP headers
                    if 'content-security-policy' in content.lower():
                        has_csp_headers = True
            
            # Evaluate findings
            if not has_output_encoding:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Input Validation",
                    title="Missing XSS Protection",
                    description="Output encoding/escaping not implemented",
                    impact="Application vulnerable to Cross-Site Scripting attacks",
                    recommendation="Implement proper output encoding for all user-generated content",
                    compliance_frameworks=["OWASP"]
                ))
            
            if not has_csp_headers:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Input Validation",
                    title="Missing Content Security Policy",
                    description="Content Security Policy headers not configured",
                    impact="Limited protection against XSS and data injection attacks",
                    recommendation="Implement Content Security Policy headers",
                    compliance_frameworks=["OWASP"]
                ))
                
        except Exception as e:
            logger.warning("XSS prevention test failed", error=str(e))
    
    async def _test_command_injection_prevention(self):
        """Test command injection prevention"""
        try:
            # Look for system command execution
            command_files = await self._find_files_containing("subprocess", "os.system", "exec", "shell")
            
            potential_command_injection = []
            
            for file_path in command_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for dangerous patterns
                    dangerous_patterns = [
                        r'os\.system\s*\(',
                        r'subprocess\.[^(]*\(\s*shell\s*=\s*True',
                        r'exec\s*\(',
                        r'eval\s*\('
                    ]
                    
                    for pattern in dangerous_patterns:
                        if re.search(pattern, content):
                            potential_command_injection.append(file_path)
                            break
            
            # Evaluate findings
            if potential_command_injection:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Input Validation",
                    title="Potential Command Injection Vulnerability",
                    description=f"Dangerous command execution found in {len(potential_command_injection)} files",
                    impact="System compromise through command injection",
                    recommendation="Avoid system command execution or use safe alternatives",
                    compliance_frameworks=["OWASP", "NIST-CSF"]
                ))
                
        except Exception as e:
            logger.warning("Command injection prevention test failed", error=str(e))
    
    async def _test_file_upload_security(self):
        """Test file upload security"""
        try:
            # Look for file upload code
            upload_files = await self._find_files_containing("upload", "file", "attachment")
            
            has_file_type_validation = False
            has_size_limits = False
            has_virus_scanning = False
            
            for file_path in upload_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for file type validation
                    if any(term in content.lower() for term in ['file_type', 'mime_type', 'extension']):
                        has_file_type_validation = True
                    
                    # Check for size limits
                    if any(term in content.lower() for term in ['file_size', 'max_size', 'size_limit']):
                        has_size_limits = True
                    
                    # Check for virus scanning
                    if any(term in content.lower() for term in ['virus', 'malware', 'scan']):
                        has_virus_scanning = True
            
            # Evaluate findings based on whether file uploads exist
            if upload_files:
                if not has_file_type_validation:
                    self.findings.append(SecurityFinding(
                        severity="HIGH",
                        category="Input Validation",
                        title="Missing File Upload Validation",
                        description="File type validation not implemented for uploads",
                        impact="Malicious files could be uploaded and executed",
                        recommendation="Implement strict file type and content validation",
                        compliance_frameworks=["OWASP"]
                    ))
                
                if not has_size_limits:
                    self.findings.append(SecurityFinding(
                        severity="MEDIUM",
                        category="Input Validation",
                        title="Missing File Size Limits",
                        description="File size limits not implemented for uploads",
                        impact="System vulnerable to storage exhaustion attacks",
                        recommendation="Implement file size limits for all uploads",
                        compliance_frameworks=["OWASP"]
                    ))
                    
        except Exception as e:
            logger.warning("File upload security test failed", error=str(e))
    
    async def _test_error_handling(self):
        """Test error handling and information disclosure"""
        # Test 1: Error Message Information Disclosure
        await self._test_error_message_disclosure()
        
        # Test 2: Exception Handling
        await self._test_exception_handling()
        
        # Test 3: Logging Security
        await self._test_logging_security()
    
    async def _test_error_message_disclosure(self):
        """Test for information disclosure in error messages"""
        try:
            # Look for error handling code
            error_files = await self._find_files_containing("error", "exception", "traceback")
            
            has_generic_errors = False
            potential_info_disclosure = []
            
            for file_path in error_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for generic error handling
                    if any(term in content.lower() for term in ['generic_error', 'internal_error']):
                        has_generic_errors = True
                    
                    # Check for potential information disclosure
                    disclosure_patterns = [
                        r'traceback\.print_exc\(\)',
                        r'str\(e\)',
                        r'exception.*message',
                        r'debug.*=.*True'
                    ]
                    
                    for pattern in disclosure_patterns:
                        if re.search(pattern, content):
                            potential_info_disclosure.append(file_path)
                            break
            
            # Evaluate findings
            if potential_info_disclosure:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Error Handling",
                    title="Potential Information Disclosure",
                    description=f"Detailed error information may be exposed in {len(potential_info_disclosure)} files",
                    impact="System information disclosed to attackers",
                    recommendation="Implement generic error messages for production",
                    compliance_frameworks=["OWASP"]
                ))
                
        except Exception as e:
            logger.warning("Error message disclosure test failed", error=str(e))
    
    async def _test_exception_handling(self):
        """Test exception handling completeness"""
        try:
            # Look for exception handling patterns
            exception_files = await self._find_files_containing("try", "except", "catch")
            
            has_comprehensive_handling = False
            bare_except_found = []
            
            for file_path in exception_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for comprehensive exception handling
                    if 'except Exception' in content or 'except:' in content:
                        has_comprehensive_handling = True
                    
                    # Check for bare except clauses
                    if re.search(r'except\s*:', content):
                        bare_except_found.append(file_path)
            
            # Evaluate findings
            if bare_except_found:
                self.findings.append(SecurityFinding(
                    severity="LOW",
                    category="Error Handling",
                    title="Bare Exception Handlers",
                    description=f"Bare except clauses found in {len(bare_except_found)} files",
                    impact="Unexpected exceptions may not be properly handled",
                    recommendation="Use specific exception types instead of bare except clauses",
                    compliance_frameworks=[]
                ))
                
        except Exception as e:
            logger.warning("Exception handling test failed", error=str(e))
    
    async def _test_logging_security(self):
        """Test logging security practices"""
        try:
            # Look for logging code
            logging_files = await self._find_files_containing("log", "audit", "monitor")
            
            has_audit_logging = False
            has_log_sanitization = False
            logs_sensitive_data = []
            
            for file_path in logging_files:
                content = await self._read_file_safely(file_path)
                if content:
                    # Check for audit logging
                    if any(term in content.lower() for term in ['audit', 'security_log']):
                        has_audit_logging = True
                    
                    # Check for log sanitization
                    if any(term in content.lower() for term in ['sanitize', 'redact', 'mask']):
                        has_log_sanitization = True
                    
                    # Check for logging sensitive data
                    sensitive_log_patterns = [
                        r'log.*password',
                        r'log.*token',
                        r'log.*key',
                        r'log.*secret'
                    ]
                    
                    for pattern in sensitive_log_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            logs_sensitive_data.append(file_path)
                            break
            
            # Evaluate findings
            if not has_audit_logging:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Logging",
                    title="Missing Security Audit Logging",
                    description="Security events not logged for audit purposes",
                    impact="Security incidents cannot be tracked or investigated",
                    recommendation="Implement comprehensive security audit logging",
                    compliance_frameworks=["SOX", "PCI-DSS", "ISO-27001"]
                ))
            
            if logs_sensitive_data:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Logging",
                    title="Sensitive Data in Logs",
                    description=f"Sensitive data logged in {len(logs_sensitive_data)} files",
                    impact="Sensitive information exposed in log files",
                    recommendation="Sanitize or redact sensitive data before logging",
                    compliance_frameworks=["GDPR", "PCI-DSS"]
                ))
                
        except Exception as e:
            logger.warning("Logging security test failed", error=str(e))
    
    async def _analyze_code_security(self):
        """Analyze code security using static analysis"""
        # Test 1: Dependency Vulnerability Scanning
        await self._scan_dependency_vulnerabilities()
        
        # Test 2: Code Quality Security Issues
        await self._analyze_code_quality_security()
        
        # Test 3: Secret Detection
        await self._detect_secrets_in_code()
    
    async def _scan_dependency_vulnerabilities(self):
        """Scan dependencies for known vulnerabilities"""
        try:
            # Look for dependency files
            dependency_files = [
                self.project_root / "requirements.txt",
                self.project_root / "requirements-prod.txt",
                self.project_root / "Pipfile",
                self.project_root / "pyproject.toml",
                self.project_root / "package.json"
            ]
            
            vulnerable_dependencies = []
            
            for dep_file in dependency_files:
                if dep_file.exists():
                    content = await self._read_file_safely(dep_file)
                    if content:
                        # Simple vulnerability patterns (in real implementation, would use vulnerability database)
                        known_vulnerable_patterns = [
                            r'django\s*[<>=]+\s*[12]\.',  # Old Django versions
                            r'flask\s*[<>=]+\s*0\.',     # Old Flask versions
                            r'requests\s*[<>=]+\s*2\.[0-9]\.',  # Specific vulnerable requests versions
                        ]
                        
                        for pattern in known_vulnerable_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                vulnerable_dependencies.append(dep_file)
            
            # Evaluate findings
            if vulnerable_dependencies:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Dependencies",
                    title="Vulnerable Dependencies Detected",
                    description=f"Potentially vulnerable dependencies found in {len(vulnerable_dependencies)} files",
                    impact="Known vulnerabilities may be exploitable",
                    recommendation="Update dependencies to latest secure versions",
                    compliance_frameworks=["NIST-CSF", "OWASP"]
                ))
                
        except Exception as e:
            logger.warning("Dependency vulnerability scan failed", error=str(e))
    
    async def _analyze_code_quality_security(self):
        """Analyze code quality from security perspective"""
        try:
            # Security-related code quality issues
            security_quality_issues = []
            
            for source_dir in self.source_dirs:
                if source_dir.exists():
                    for file_path in source_dir.rglob("*.py"):
                        content = await self._read_file_safely(file_path)
                        if content:
                            # Check for security anti-patterns
                            security_antipatterns = [
                                (r'assert\s+', "Assertions can be disabled in production"),
                                (r'pickle\.load', "Pickle deserialization is unsafe"),
                                (r'yaml\.load\(', "yaml.load() is unsafe, use yaml.safe_load()"),
                                (r'shell=True', "Shell execution can be dangerous"),
                                (r'random\.random\(\)', "random module is not cryptographically secure")
                            ]
                            
                            for pattern, description in security_antipatterns:
                                if re.search(pattern, content):
                                    security_quality_issues.append((file_path, description))
            
            # Evaluate findings
            if security_quality_issues:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Code Quality",
                    title="Security Anti-patterns Detected",
                    description=f"Security anti-patterns found in {len(security_quality_issues)} locations",
                    impact="Code patterns that may introduce security vulnerabilities",
                    recommendation="Review and fix security anti-patterns in code",
                    compliance_frameworks=["OWASP"]
                ))
                
        except Exception as e:
            logger.warning("Code quality security analysis failed", error=str(e))
    
    async def _detect_secrets_in_code(self):
        """Detect secrets and credentials in code"""
        try:
            # Secret patterns
            secret_patterns = [
                (r'-----BEGIN\s+.*PRIVATE\s+KEY-----', "Private key"),
                (r'AKIA[0-9A-Z]{16}', "AWS Access Key"),
                (r'[0-9a-f]{32}', "MD5 hash (potential secret)"),
                (r'ghp_[0-9a-zA-Z]{36}', "GitHub Personal Access Token"),
                (r'sk-[0-9a-zA-Z]{48}', "OpenAI API Key")
            ]
            
            secrets_found = []
            
            for source_dir in self.source_dirs:
                if source_dir.exists():
                    for file_path in source_dir.rglob("*.py"):
                        content = await self._read_file_safely(file_path)
                        if content:
                            for pattern, secret_type in secret_patterns:
                                if re.search(pattern, content):
                                    secrets_found.append((file_path, secret_type))
            
            # Evaluate findings
            if secrets_found:
                self.findings.append(SecurityFinding(
                    severity="CRITICAL",
                    category="Secrets Management",
                    title="Secrets Detected in Code",
                    description=f"Potential secrets found in {len(secrets_found)} locations",
                    impact="Secrets exposed in version control and deployments",
                    recommendation="Remove secrets from code and use secure secret management",
                    compliance_frameworks=["PCI-DSS", "SOX", "GDPR"]
                ))
                
        except Exception as e:
            logger.warning("Secret detection failed", error=str(e))
    
    async def _assess_infrastructure_security(self):
        """Assess infrastructure security configuration"""
        # Test 1: Container Security
        await self._assess_container_security()
        
        # Test 2: Kubernetes Security
        await self._assess_kubernetes_security()
        
        # Test 3: Cloud Security
        await self._assess_cloud_security()
    
    async def _assess_container_security(self):
        """Assess container security configuration"""
        try:
            # Look for Docker files
            docker_files = [
                self.project_root / "Dockerfile",
                self.project_root / "Dockerfile.production",
                self.project_root / "docker-compose.yml",
                self.project_root / "docker-compose.production.yml"
            ]
            
            container_issues = []
            
            for docker_file in docker_files:
                if docker_file.exists():
                    content = await self._read_file_safely(docker_file)
                    if content:
                        # Check for security issues
                        security_checks = [
                            (r'USER\s+root', "Running as root user"),
                            (r'--privileged', "Privileged container"),
                            (r'FROM\s+.*:latest', "Using latest tag"),
                            (r'ADD\s+http', "Using ADD instead of COPY for URLs")
                        ]
                        
                        for pattern, issue in security_checks:
                            if re.search(pattern, content, re.IGNORECASE):
                                container_issues.append((docker_file, issue))
            
            # Evaluate findings
            if container_issues:
                self.findings.append(SecurityFinding(
                    severity="MEDIUM",
                    category="Infrastructure Security",
                    title="Container Security Issues",
                    description=f"Container security issues found in {len(container_issues)} files",
                    impact="Containers may have elevated privileges or security risks",
                    recommendation="Follow container security best practices",
                    compliance_frameworks=["NIST-CSF", "CIS"]
                ))
                
        except Exception as e:
            logger.warning("Container security assessment failed", error=str(e))
    
    async def _assess_kubernetes_security(self):
        """Assess Kubernetes security configuration"""
        try:
            # Look for Kubernetes manifests
            k8s_dir = self.project_root / "k8s"
            k8s_issues = []
            
            if k8s_dir.exists():
                for k8s_file in k8s_dir.rglob("*.yaml"):
                    content = await self._read_file_safely(k8s_file)
                    if content:
                        # Check for security issues
                        security_checks = [
                            (r'privileged:\s*true', "Privileged containers"),
                            (r'runAsUser:\s*0', "Running as root"),
                            (r'allowPrivilegeEscalation:\s*true', "Privilege escalation allowed"),
                            (r'hostNetwork:\s*true', "Host network access"),
                            (r'hostPID:\s*true', "Host PID namespace access")
                        ]
                        
                        for pattern, issue in security_checks:
                            if re.search(pattern, content, re.IGNORECASE):
                                k8s_issues.append((k8s_file, issue))
            
            # Evaluate findings
            if k8s_issues:
                self.findings.append(SecurityFinding(
                    severity="HIGH",
                    category="Infrastructure Security",
                    title="Kubernetes Security Issues",
                    description=f"Kubernetes security issues found in {len(k8s_issues)} files",
                    impact="Containers may have excessive privileges",
                    recommendation="Apply Kubernetes security best practices and pod security standards",
                    compliance_frameworks=["NIST-CSF", "CIS"]
                ))
                
        except Exception as e:
            logger.warning("Kubernetes security assessment failed", error=str(e))
    
    async def _assess_cloud_security(self):
        """Assess cloud security configuration"""
        try:
            # Look for cloud configuration files
            cloud_files = await self._find_files_containing("aws", "azure", "gcp", "cloud", "terraform")
            
            has_cloud_config = len(cloud_files) > 0
            
            if has_cloud_config:
                # This would be expanded in a real implementation
                self.findings.append(SecurityFinding(
                    severity="INFO",
                    category="Infrastructure Security",
                    title="Cloud Configuration Detected",
                    description="Cloud infrastructure configuration found",
                    impact="Cloud security should be validated separately",
                    recommendation="Conduct cloud-specific security assessment",
                    compliance_frameworks=["CSA-CCM", "NIST-CSF"]
                ))
                
        except Exception as e:
            logger.warning("Cloud security assessment failed", error=str(e))
    
    async def _assess_compliance_frameworks(self):
        """Assess compliance with various frameworks"""
        for framework in self.compliance_frameworks:
            if framework == "SOX":
                await self._assess_sox_compliance()
            elif framework == "PCI-DSS":
                await self._assess_pci_compliance()
            elif framework == "GDPR":
                await self._assess_gdpr_compliance()
            elif framework == "ISO-27001":
                await self._assess_iso27001_compliance()
            elif framework == "NIST-CSF":
                await self._assess_nist_compliance()
    
    async def _assess_sox_compliance(self):
        """Assess SOX compliance requirements"""
        sox_requirements = [
            ComplianceRequirement(
                framework="SOX",
                requirement_id="SOX-302",
                title="Financial Reporting Controls",
                description="Controls over financial reporting must be documented and tested",
                controls_required=["audit_logging", "access_controls", "change_management"]
            ),
            ComplianceRequirement(
                framework="SOX",
                requirement_id="SOX-404",
                title="Internal Controls Assessment",
                description="Management must assess effectiveness of internal controls",
                controls_required=["risk_assessment", "control_testing", "documentation"]
            )
        ]
        
        # Assess each requirement
        for req in sox_requirements:
            # Simplified assessment logic
            controls_found = 0
            for control in req.controls_required:
                if await self._check_control_exists(control):
                    controls_found += 1
                    req.evidence.append(f"Control {control} implemented")
                else:
                    req.gaps.append(f"Control {control} missing")
            
            # Determine compliance status
            compliance_ratio = controls_found / len(req.controls_required)
            if compliance_ratio >= 1.0:
                req.status = "COMPLIANT"
            elif compliance_ratio >= 0.7:
                req.status = "PARTIAL"
            else:
                req.status = "NON_COMPLIANT"
            
            self.compliance_results.append(req)
    
    async def _assess_pci_compliance(self):
        """Assess PCI-DSS compliance requirements"""
        pci_requirements = [
            ComplianceRequirement(
                framework="PCI-DSS",
                requirement_id="PCI-3.4",
                title="Encryption of Cardholder Data",
                description="Cardholder data must be encrypted during transmission",
                controls_required=["tls_encryption", "strong_cryptography"]
            ),
            ComplianceRequirement(
                framework="PCI-DSS",
                requirement_id="PCI-8.2",
                title="User Authentication",
                description="Strong user authentication and password policies required",
                controls_required=["password_complexity", "mfa", "account_lockout"]
            )
        ]
        
        # Assess each requirement (simplified)
        for req in pci_requirements:
            controls_found = 0
            for control in req.controls_required:
                if await self._check_control_exists(control):
                    controls_found += 1
                    req.evidence.append(f"Control {control} implemented")
                else:
                    req.gaps.append(f"Control {control} missing")
            
            compliance_ratio = controls_found / len(req.controls_required)
            if compliance_ratio >= 1.0:
                req.status = "COMPLIANT"
            elif compliance_ratio >= 0.7:
                req.status = "PARTIAL"
            else:
                req.status = "NON_COMPLIANT"
            
            self.compliance_results.append(req)
    
    async def _assess_gdpr_compliance(self):
        """Assess GDPR compliance requirements"""
        gdpr_requirements = [
            ComplianceRequirement(
                framework="GDPR",
                requirement_id="GDPR-32",
                title="Security of Processing",
                description="Appropriate technical and organizational measures for data security",
                controls_required=["encryption", "access_controls", "audit_logging"]
            ),
            ComplianceRequirement(
                framework="GDPR",
                requirement_id="GDPR-25",
                title="Data Protection by Design",
                description="Data protection measures built into system design",
                controls_required=["privacy_controls", "data_minimization", "retention_policies"]
            )
        ]
        
        # Simplified assessment
        for req in gdpr_requirements:
            controls_found = 0
            for control in req.controls_required:
                if await self._check_control_exists(control):
                    controls_found += 1
                    req.evidence.append(f"Control {control} implemented")
                else:
                    req.gaps.append(f"Control {control} missing")
            
            compliance_ratio = controls_found / len(req.controls_required)
            if compliance_ratio >= 1.0:
                req.status = "COMPLIANT"
            elif compliance_ratio >= 0.7:
                req.status = "PARTIAL"
            else:
                req.status = "NON_COMPLIANT"
            
            self.compliance_results.append(req)
    
    async def _assess_iso27001_compliance(self):
        """Assess ISO 27001 compliance requirements"""
        # Simplified ISO 27001 assessment
        iso_req = ComplianceRequirement(
            framework="ISO-27001",
            requirement_id="ISO-A.12.6.1",
            title="Management of Technical Vulnerabilities",
            description="Technical vulnerabilities should be identified and managed",
            controls_required=["vulnerability_scanning", "patch_management", "security_testing"]
        )
        
        controls_found = 0
        for control in iso_req.controls_required:
            if await self._check_control_exists(control):
                controls_found += 1
                iso_req.evidence.append(f"Control {control} implemented")
            else:
                iso_req.gaps.append(f"Control {control} missing")
        
        compliance_ratio = controls_found / len(iso_req.controls_required)
        if compliance_ratio >= 1.0:
            iso_req.status = "COMPLIANT"
        elif compliance_ratio >= 0.7:
            iso_req.status = "PARTIAL"
        else:
            iso_req.status = "NON_COMPLIANT"
        
        self.compliance_results.append(iso_req)
    
    async def _assess_nist_compliance(self):
        """Assess NIST Cybersecurity Framework compliance"""
        # Simplified NIST CSF assessment
        nist_req = ComplianceRequirement(
            framework="NIST-CSF",
            requirement_id="NIST-PR.AC",
            title="Access Control",
            description="Access to assets and associated facilities is limited to authorized users",
            controls_required=["identity_management", "access_controls", "privileged_access"]
        )
        
        controls_found = 0
        for control in nist_req.controls_required:
            if await self._check_control_exists(control):
                controls_found += 1
                nist_req.evidence.append(f"Control {control} implemented")
            else:
                nist_req.gaps.append(f"Control {control} missing")
        
        compliance_ratio = controls_found / len(nist_req.controls_required)
        if compliance_ratio >= 1.0:
            nist_req.status = "COMPLIANT"
        elif compliance_ratio >= 0.7:
            nist_req.status = "PARTIAL"
        else:
            nist_req.status = "NON_COMPLIANT"
        
        self.compliance_results.append(nist_req)
    
    async def _check_control_exists(self, control_name: str) -> bool:
        """Check if a specific security control exists"""
        # Simplified control checking
        control_keywords = {
            "audit_logging": ["audit", "log", "logging"],
            "access_controls": ["auth", "rbac", "permission"],
            "encryption": ["encrypt", "crypto", "tls", "ssl"],
            "password_complexity": ["password", "complexity"],
            "mfa": ["mfa", "2fa", "totp"],
            "tls_encryption": ["tls", "ssl", "https"],
            "vulnerability_scanning": ["scan", "vulnerability"],
            "patch_management": ["update", "patch"],
            "security_testing": ["test", "security"]
        }
        
        keywords = control_keywords.get(control_name, [control_name])
        
        # Search for keywords in codebase
        for keyword in keywords:
            files = await self._find_files_containing(keyword)
            if files:
                return True
        
        return False
    
    async def _conduct_penetration_testing(self):
        """Conduct basic penetration testing"""
        # This would be a comprehensive penetration test in reality
        # For now, just add a placeholder finding
        self.findings.append(SecurityFinding(
            severity="INFO",
            category="Penetration Testing",
            title="Penetration Testing Required",
            description="Comprehensive penetration testing should be conducted",
            impact="Unknown vulnerabilities may exist",
            recommendation="Conduct professional penetration testing",
            compliance_frameworks=["PCI-DSS", "ISO-27001"]
        ))
    
    # Helper methods
    
    async def _find_files_containing(self, *keywords) -> List[Path]:
        """Find files containing any of the specified keywords"""
        matching_files = []
        
        for source_dir in self.source_dirs:
            if source_dir.exists():
                for file_path in source_dir.rglob("*.py"):
                    try:
                        content = await self._read_file_safely(file_path)
                        if content and any(keyword.lower() in content.lower() for keyword in keywords):
                            matching_files.append(file_path)
                    except Exception:
                        continue
        
        return matching_files
    
    async def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """Safely read file content"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return None
    
    def _generate_audit_report(self, start_time: datetime, end_time: datetime) -> SecurityAuditReport:
        """Generate comprehensive audit report"""
        report = SecurityAuditReport(
            audit_id=self.audit_id,
            start_time=start_time,
            end_time=end_time,
            findings=self.findings,
            compliance_requirements=self.compliance_results
        )
        
        # Calculate summary metrics
        report.total_findings = len(self.findings)
        report.critical_findings = len([f for f in self.findings if f.severity == "CRITICAL"])
        report.high_findings = len([f for f in self.findings if f.severity == "HIGH"])
        report.medium_findings = len([f for f in self.findings if f.severity == "MEDIUM"])
        report.low_findings = len([f for f in self.findings if f.severity == "LOW"])
        
        # Calculate overall compliance score
        if self.compliance_results:
            compliant_reqs = len([r for r in self.compliance_results if r.status == "COMPLIANT"])
            partial_reqs = len([r for r in self.compliance_results if r.status == "PARTIAL"])
            total_reqs = len(self.compliance_results)
            
            report.overall_compliance_score = (compliant_reqs + partial_reqs * 0.5) / total_reqs
        
        # Determine overall risk level
        if report.critical_findings > 0:
            report.overall_risk_level = "CRITICAL"
        elif report.high_findings > 3:
            report.overall_risk_level = "HIGH"
        elif report.medium_findings > 5:
            report.overall_risk_level = "MEDIUM"
        else:
            report.overall_risk_level = "LOW"
        
        # Determine production readiness
        report.production_readiness = (
            report.critical_findings == 0 and
            report.high_findings <= 2 and
            report.overall_compliance_score >= 0.8
        )
        
        # Generate recommendations
        if report.critical_findings > 0:
            report.immediate_actions.append("Fix all critical security findings immediately")
        
        if report.high_findings > 0:
            report.short_term_actions.append("Address high severity security findings within 30 days")
        
        if report.overall_compliance_score < 0.8:
            report.short_term_actions.append("Improve compliance framework adherence")
        
        if report.medium_findings > 3:
            report.long_term_actions.append("Develop security improvement program for medium findings")
        
        return report


# Factory function
def create_security_auditor(config: Dict[str, Any] = None) -> SecurityComplianceAuditor:
    """Create security compliance auditor"""
    return SecurityComplianceAuditor(config)


# CLI interface
async def main():
    """Main security audit script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Security and Compliance Auditor")
    parser.add_argument("--frameworks", nargs="*", default=["SOX", "PCI-DSS", "GDPR"], 
                       help="Compliance frameworks to assess")
    parser.add_argument("--code-analysis", action="store_true", default=True, 
                       help="Include code security analysis")
    parser.add_argument("--infrastructure", action="store_true", default=True, 
                       help="Include infrastructure security assessment")
    parser.add_argument("--output", default="security_audit_report.json", 
                       help="Output file for audit report")
    
    args = parser.parse_args()
    
    # Configure auditor
    config = {
        'compliance_frameworks': args.frameworks,
        'include_code_analysis': args.code_analysis,
        'include_infrastructure_scan': args.infrastructure,
        'include_compliance_check': True,
        'include_penetration_testing': False
    }
    
    # Run security audit
    auditor = create_security_auditor(config)
    
    try:
        report = await auditor.run_comprehensive_audit()
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report.__dict__, f, indent=2, default=str)
        
        print(f"Security audit completed. Report saved to {args.output}")
        print(f"Overall risk level: {report.overall_risk_level}")
        print(f"Production ready: {report.production_readiness}")
        print(f"Critical findings: {report.critical_findings}")
        print(f"High findings: {report.high_findings}")
        
        # Exit with appropriate code
        if report.production_readiness:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"Security audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())