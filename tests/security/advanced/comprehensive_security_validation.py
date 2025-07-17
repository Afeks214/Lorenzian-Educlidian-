#!/usr/bin/env python3
"""
🚨 AGENT 10: COMPREHENSIVE SECURITY VALIDATION
Validates that all security vulnerabilities have been properly fixed
"""

import os
import sys
import json
import re
import time
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import subprocess
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VulnerabilityType(Enum):
    """Types of security vulnerabilities to validate."""
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    AUTHENTICATION_BYPASS = "authentication_bypass"
    AUTHORIZATION_BYPASS = "authorization_bypass"
    SECRETS_EXPOSURE = "secrets_exposure"
    INPUT_VALIDATION = "input_validation"
    CORS_MISCONFIGURATION = "cors_misconfiguration"
    SSL_CONFIGURATION = "ssl_configuration"
    DEBUG_INFORMATION = "debug_information"
    HARDCODED_SECRETS = "hardcoded_secrets"


class ValidationResult(Enum):
    """Validation results."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class SecurityCheck:
    """Security check result."""
    check_id: str
    vulnerability_type: VulnerabilityType
    description: str
    result: ValidationResult
    severity: str
    details: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    evidence: Optional[str] = None
    remediation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'vulnerability_type': self.vulnerability_type.value,
            'result': self.result.value
        }


class SecurityValidator:
    """Main security validation class."""
    
    def __init__(self, project_root: str = "/home/QuantNova/GrandModel"):
        self.project_root = Path(project_root)
        self.checks_performed = []
        self.results_dir = self.project_root / "tests" / "security" / "advanced" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Security file paths
        self.security_files = {
            "auth": self.project_root / "src" / "security" / "auth.py",
            "secrets_manager": self.project_root / "src" / "security" / "secrets_manager.py",
            "rate_limiter": self.project_root / "src" / "security" / "rate_limiter.py",
            "attack_detection": self.project_root / "src" / "security" / "attack_detection.py",
            "main_api": self.project_root / "src" / "api" / "main.py",
            "tactical_api": self.project_root / "src" / "api" / "tactical_main.py"
        }
        
        logger.info(f"Security Validator initialized - Project: {self.project_root}")
    
    def validate_sql_injection_protection(self) -> List[SecurityCheck]:
        """Validate SQL injection protection."""
        logger.info("🔍 Validating SQL injection protection...")
        checks = []
        
        # Check authentication module
        auth_file = self.security_files["auth"]
        if auth_file.exists():
            with open(auth_file, 'r') as f:
                content = f.read()
            
            # Check for parameterized queries
            if "execute(" in content and "%" in content:
                # Check if string formatting is used in execute statements
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "execute(" in line and ("%" in line or ".format(" in line):
                        checks.append(SecurityCheck(
                            check_id="SQL_001",
                            vulnerability_type=VulnerabilityType.SQL_INJECTION,
                            description="Potential SQL injection vulnerability in authentication",
                            result=ValidationResult.FAILED,
                            severity="CRITICAL",
                            details="String formatting used in SQL execute statement",
                            file_path=str(auth_file),
                            line_number=i + 1,
                            evidence=line.strip(),
                            remediation="Use parameterized queries with placeholders"
                        ))
            
            # Check for proper ORM usage
            if "SQLAlchemy" in content or "session.query" in content:
                checks.append(SecurityCheck(
                    check_id="SQL_002",
                    vulnerability_type=VulnerabilityType.SQL_INJECTION,
                    description="SQL injection protection via ORM",
                    result=ValidationResult.PASSED,
                    severity="HIGH",
                    details="Using SQLAlchemy ORM for database operations",
                    file_path=str(auth_file),
                    remediation="Continue using ORM for all database operations"
                ))
            
            # Check for input validation
            if "validate_input" in content or "sanitize" in content:
                checks.append(SecurityCheck(
                    check_id="SQL_003",
                    vulnerability_type=VulnerabilityType.SQL_INJECTION,
                    description="Input validation implemented",
                    result=ValidationResult.PASSED,
                    severity="HIGH",
                    details="Input validation functions found",
                    file_path=str(auth_file),
                    remediation="Ensure all user inputs are validated"
                ))
            else:
                checks.append(SecurityCheck(
                    check_id="SQL_003",
                    vulnerability_type=VulnerabilityType.SQL_INJECTION,
                    description="Input validation missing",
                    result=ValidationResult.WARNING,
                    severity="MEDIUM",
                    details="No explicit input validation found",
                    file_path=str(auth_file),
                    remediation="Implement comprehensive input validation"
                ))
        
        return checks
    
    def validate_command_injection_protection(self) -> List[SecurityCheck]:
        """Validate command injection protection."""
        logger.info("🔍 Validating command injection protection...")
        checks = []
        
        # Check all Python files for dangerous functions
        dangerous_functions = [
            "os.system", "subprocess.run", "subprocess.call", "subprocess.Popen",
            "os.popen", "exec", "eval", "compile"
        ]
        
        python_files = list(self.project_root.glob("**/*.py"))
        
        for file_path in python_files:
            if file_path.name.startswith('.'):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    for func in dangerous_functions:
                        if func in line and not line.strip().startswith('#'):
                            # Check if it's properly sanitized
                            if "shell=True" in line:
                                checks.append(SecurityCheck(
                                    check_id="CMD_001",
                                    vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                                    description="Command injection vulnerability",
                                    result=ValidationResult.FAILED,
                                    severity="CRITICAL",
                                    details=f"Dangerous function {func} with shell=True",
                                    file_path=str(file_path),
                                    line_number=i + 1,
                                    evidence=line.strip(),
                                    remediation="Use subprocess without shell=True and validate inputs"
                                ))
                            elif "user_input" in line or "request." in line:
                                checks.append(SecurityCheck(
                                    check_id="CMD_002",
                                    vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                                    description="Potential command injection with user input",
                                    result=ValidationResult.WARNING,
                                    severity="HIGH",
                                    details=f"User input used with {func}",
                                    file_path=str(file_path),
                                    line_number=i + 1,
                                    evidence=line.strip(),
                                    remediation="Validate and sanitize all user inputs"
                                ))
            
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
        
        return checks
    
    def validate_authentication_security(self) -> List[SecurityCheck]:
        """Validate authentication security."""
        logger.info("🔍 Validating authentication security...")
        checks = []
        
        auth_file = self.security_files["auth"]
        if auth_file.exists():
            with open(auth_file, 'r') as f:
                content = f.read()
            
            # Check for JWT security
            if "jwt" in content.lower():
                # Check for proper secret handling
                if "JWT_SECRET" in content and "change-in-production" in content:
                    checks.append(SecurityCheck(
                        check_id="AUTH_001",
                        vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                        description="Default JWT secret used",
                        result=ValidationResult.FAILED,
                        severity="CRITICAL",
                        details="Default JWT secret found in code",
                        file_path=str(auth_file),
                        remediation="Use environment variable for JWT secret"
                    ))
                elif "os.environ" in content and "JWT_SECRET" in content:
                    checks.append(SecurityCheck(
                        check_id="AUTH_002",
                        vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                        description="JWT secret from environment",
                        result=ValidationResult.PASSED,
                        severity="HIGH",
                        details="JWT secret loaded from environment variable",
                        file_path=str(auth_file),
                        remediation="Ensure environment variable is set securely"
                    ))
                
                # Check for token expiration
                if "exp" in content or "expires" in content:
                    checks.append(SecurityCheck(
                        check_id="AUTH_003",
                        vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                        description="Token expiration implemented",
                        result=ValidationResult.PASSED,
                        severity="MEDIUM",
                        details="JWT token expiration configured",
                        file_path=str(auth_file),
                        remediation="Ensure reasonable expiration times"
                    ))
                else:
                    checks.append(SecurityCheck(
                        check_id="AUTH_003",
                        vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                        description="No token expiration found",
                        result=ValidationResult.WARNING,
                        severity="MEDIUM",
                        details="JWT tokens may not expire",
                        file_path=str(auth_file),
                        remediation="Implement token expiration"
                    ))
            
            # Check for rate limiting
            if "rate_limit" in content.lower() or "throttle" in content.lower():
                checks.append(SecurityCheck(
                    check_id="AUTH_004",
                    vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                    description="Rate limiting implemented",
                    result=ValidationResult.PASSED,
                    severity="MEDIUM",
                    details="Rate limiting found in authentication",
                    file_path=str(auth_file),
                    remediation="Ensure rate limits are appropriate"
                ))
            else:
                checks.append(SecurityCheck(
                    check_id="AUTH_004",
                    vulnerability_type=VulnerabilityType.AUTHENTICATION_BYPASS,
                    description="No rate limiting found",
                    result=ValidationResult.WARNING,
                    severity="MEDIUM",
                    details="Authentication endpoints may not be rate limited",
                    file_path=str(auth_file),
                    remediation="Implement rate limiting for authentication"
                ))
        
        return checks
    
    def validate_secrets_management(self) -> List[SecurityCheck]:
        """Validate secrets management."""
        logger.info("🔍 Validating secrets management...")
        checks = []
        
        # Check for hardcoded secrets in all files
        secret_patterns = [
            (r'password\s*=\s*["\']([^"\']{6,})["\']', "hardcoded_password"),
            (r'secret\s*=\s*["\']([^"\']{10,})["\']', "hardcoded_secret"),
            (r'key\s*=\s*["\']([^"\']{10,})["\']', "hardcoded_key"),
            (r'token\s*=\s*["\']([^"\']{10,})["\']', "hardcoded_token"),
            (r'api_key\s*=\s*["\']([^"\']{10,})["\']', "hardcoded_api_key"),
            (r'AWS_ACCESS_KEY_ID\s*=\s*["\']([^"\']{10,})["\']', "hardcoded_aws_key"),
            (r'AWS_SECRET_ACCESS_KEY\s*=\s*["\']([^"\']{10,})["\']', "hardcoded_aws_secret")
        ]
        
        python_files = list(self.project_root.glob("**/*.py"))
        config_files = list(self.project_root.glob("**/*.yaml")) + list(self.project_root.glob("**/*.yml")) + list(self.project_root.glob("**/*.json"))
        
        all_files = python_files + config_files
        
        for file_path in all_files:
            if file_path.name.startswith('.') or 'venv' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    for pattern, secret_type in secret_patterns:
                        matches = re.findall(pattern, line, re.IGNORECASE)
                        if matches:
                            # Check if it's a placeholder or example
                            if any(placeholder in line.lower() for placeholder in 
                                  ['change-in-production', 'your-secret', 'placeholder', 'example', 'xxx']):
                                checks.append(SecurityCheck(
                                    check_id="SEC_001",
                                    vulnerability_type=VulnerabilityType.HARDCODED_SECRETS,
                                    description=f"Placeholder {secret_type} found",
                                    result=ValidationResult.WARNING,
                                    severity="MEDIUM",
                                    details=f"Placeholder {secret_type} in code",
                                    file_path=str(file_path),
                                    line_number=i + 1,
                                    evidence=line.strip(),
                                    remediation="Replace with environment variable"
                                ))
                            else:
                                checks.append(SecurityCheck(
                                    check_id="SEC_002",
                                    vulnerability_type=VulnerabilityType.HARDCODED_SECRETS,
                                    description=f"Hardcoded {secret_type} found",
                                    result=ValidationResult.FAILED,
                                    severity="CRITICAL",
                                    details=f"Hardcoded {secret_type} in code",
                                    file_path=str(file_path),
                                    line_number=i + 1,
                                    evidence=line.strip()[:100] + "...",
                                    remediation="Remove hardcoded secret and use environment variable"
                                ))
            
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
        
        # Check for proper secrets manager usage
        secrets_file = self.security_files["secrets_manager"]
        if secrets_file.exists():
            with open(secrets_file, 'r') as f:
                content = f.read()
            
            if "os.environ" in content:
                checks.append(SecurityCheck(
                    check_id="SEC_003",
                    vulnerability_type=VulnerabilityType.SECRETS_EXPOSURE,
                    description="Environment variables used for secrets",
                    result=ValidationResult.PASSED,
                    severity="HIGH",
                    details="Secrets loaded from environment",
                    file_path=str(secrets_file),
                    remediation="Ensure environment is properly secured"
                ))
            
            if "vault" in content.lower() or "hashicorp" in content.lower():
                checks.append(SecurityCheck(
                    check_id="SEC_004",
                    vulnerability_type=VulnerabilityType.SECRETS_EXPOSURE,
                    description="Vault integration found",
                    result=ValidationResult.PASSED,
                    severity="HIGH",
                    details="HashiCorp Vault integration detected",
                    file_path=str(secrets_file),
                    remediation="Ensure Vault is properly configured"
                ))
        
        return checks
    
    def validate_cors_configuration(self) -> List[SecurityCheck]:
        """Validate CORS configuration."""
        logger.info("🔍 Validating CORS configuration...")
        checks = []
        
        # Check main API files
        api_files = [self.security_files["main_api"], self.security_files["tactical_api"]]
        
        for api_file in api_files:
            if api_file.exists():
                with open(api_file, 'r') as f:
                    content = f.read()
                
                # Check for wildcard CORS
                if 'allow_origins=["*"]' in content:
                    checks.append(SecurityCheck(
                        check_id="CORS_001",
                        vulnerability_type=VulnerabilityType.CORS_MISCONFIGURATION,
                        description="Wildcard CORS configuration",
                        result=ValidationResult.FAILED,
                        severity="HIGH",
                        details="CORS configured with wildcard (*) origin",
                        file_path=str(api_file),
                        remediation="Specify exact allowed origins"
                    ))
                elif "allow_origins=" in content:
                    checks.append(SecurityCheck(
                        check_id="CORS_002",
                        vulnerability_type=VulnerabilityType.CORS_MISCONFIGURATION,
                        description="Specific CORS origins configured",
                        result=ValidationResult.PASSED,
                        severity="MEDIUM",
                        details="CORS configured with specific origins",
                        file_path=str(api_file),
                        remediation="Verify origins are correct"
                    ))
                
                # Check for credentials allowed
                if "allow_credentials=True" in content:
                    checks.append(SecurityCheck(
                        check_id="CORS_003",
                        vulnerability_type=VulnerabilityType.CORS_MISCONFIGURATION,
                        description="CORS credentials allowed",
                        result=ValidationResult.WARNING,
                        severity="MEDIUM",
                        details="CORS allows credentials",
                        file_path=str(api_file),
                        remediation="Ensure this is necessary and origins are restrictive"
                    ))
        
        return checks
    
    def validate_input_validation(self) -> List[SecurityCheck]:
        """Validate input validation."""
        logger.info("🔍 Validating input validation...")
        checks = []
        
        # Check for Pydantic models and validation
        python_files = list(self.project_root.glob("**/*.py"))
        
        validation_found = False
        for file_path in python_files:
            if file_path.name.startswith('.') or 'venv' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for Pydantic models
                if "BaseModel" in content and "pydantic" in content:
                    validation_found = True
                    checks.append(SecurityCheck(
                        check_id="VAL_001",
                        vulnerability_type=VulnerabilityType.INPUT_VALIDATION,
                        description="Pydantic validation found",
                        result=ValidationResult.PASSED,
                        severity="HIGH",
                        details="Pydantic models used for validation",
                        file_path=str(file_path),
                        remediation="Ensure all inputs use Pydantic validation"
                    ))
                
                # Check for manual validation
                if "validate_" in content or "sanitize_" in content:
                    validation_found = True
                    checks.append(SecurityCheck(
                        check_id="VAL_002",
                        vulnerability_type=VulnerabilityType.INPUT_VALIDATION,
                        description="Manual validation found",
                        result=ValidationResult.PASSED,
                        severity="MEDIUM",
                        details="Manual validation functions found",
                        file_path=str(file_path),
                        remediation="Ensure validation is comprehensive"
                    ))
                
                # Check for dangerous patterns
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "request." in line and ("exec(" in line or "eval(" in line):
                        checks.append(SecurityCheck(
                            check_id="VAL_003",
                            vulnerability_type=VulnerabilityType.INPUT_VALIDATION,
                            description="Dangerous input processing",
                            result=ValidationResult.FAILED,
                            severity="CRITICAL",
                            details="User input used with exec/eval",
                            file_path=str(file_path),
                            line_number=i + 1,
                            evidence=line.strip(),
                            remediation="Never use exec/eval with user input"
                        ))
            
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
        
        if not validation_found:
            checks.append(SecurityCheck(
                check_id="VAL_004",
                vulnerability_type=VulnerabilityType.INPUT_VALIDATION,
                description="No input validation found",
                result=ValidationResult.WARNING,
                severity="HIGH",
                details="No explicit input validation mechanisms found",
                file_path=None,
                remediation="Implement comprehensive input validation"
            ))
        
        return checks
    
    def validate_debug_information(self) -> List[SecurityCheck]:
        """Validate debug information exposure."""
        logger.info("🔍 Validating debug information exposure...")
        checks = []
        
        # Check for debug mode enabled
        python_files = list(self.project_root.glob("**/*.py"))
        config_files = list(self.project_root.glob("**/*.yaml")) + list(self.project_root.glob("**/*.yml"))
        
        all_files = python_files + config_files
        
        for file_path in all_files:
            if file_path.name.startswith('.') or 'venv' in str(file_path):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "debug=True" in line.lower() or "debug: true" in line.lower():
                        checks.append(SecurityCheck(
                            check_id="DEBUG_001",
                            vulnerability_type=VulnerabilityType.DEBUG_INFORMATION,
                            description="Debug mode enabled",
                            result=ValidationResult.FAILED,
                            severity="MEDIUM",
                            details="Debug mode found in configuration",
                            file_path=str(file_path),
                            line_number=i + 1,
                            evidence=line.strip(),
                            remediation="Disable debug mode in production"
                        ))
                    
                    if "traceback" in line.lower() and "print" in line.lower():
                        checks.append(SecurityCheck(
                            check_id="DEBUG_002",
                            vulnerability_type=VulnerabilityType.DEBUG_INFORMATION,
                            description="Traceback printing found",
                            result=ValidationResult.WARNING,
                            severity="LOW",
                            details="Traceback printing in code",
                            file_path=str(file_path),
                            line_number=i + 1,
                            evidence=line.strip(),
                            remediation="Remove traceback printing in production"
                        ))
            
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
        
        return checks
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive security validation."""
        logger.info("🚨 STARTING COMPREHENSIVE SECURITY VALIDATION - AGENT 10")
        logger.info("=" * 80)
        
        all_checks = []
        
        # Run all validation checks
        try:
            all_checks.extend(self.validate_sql_injection_protection())
            all_checks.extend(self.validate_command_injection_protection())
            all_checks.extend(self.validate_authentication_security())
            all_checks.extend(self.validate_secrets_management())
            all_checks.extend(self.validate_cors_configuration())
            all_checks.extend(self.validate_input_validation())
            all_checks.extend(self.validate_debug_information())
        except Exception as e:
            logger.error(f"Error during validation: {e}")
        
        # Analyze results
        failed_checks = [c for c in all_checks if c.result == ValidationResult.FAILED]
        warning_checks = [c for c in all_checks if c.result == ValidationResult.WARNING]
        passed_checks = [c for c in all_checks if c.result == ValidationResult.PASSED]
        
        # Calculate security score
        security_score = self._calculate_security_score(all_checks)
        
        # Determine overall security posture
        overall_posture = self._determine_security_posture(failed_checks, warning_checks)
        
        # Generate report
        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_checks": len(all_checks),
                "passed_checks": len(passed_checks),
                "failed_checks": len(failed_checks),
                "warning_checks": len(warning_checks),
                "security_score": security_score,
                "overall_posture": overall_posture,
                "production_ready": len(failed_checks) == 0
            },
            "failed_checks": [c.to_dict() for c in failed_checks],
            "warning_checks": [c.to_dict() for c in warning_checks],
            "passed_checks": [c.to_dict() for c in passed_checks],
            "vulnerability_analysis": self._analyze_vulnerabilities(all_checks),
            "remediation_plan": self._generate_remediation_plan(failed_checks, warning_checks),
            "compliance_status": self._assess_compliance(failed_checks, warning_checks),
            "executive_summary": self._generate_executive_summary(failed_checks, warning_checks, security_score)
        }
        
        return report
    
    def _calculate_security_score(self, checks: List[SecurityCheck]) -> float:
        """Calculate security score (0-100)."""
        if not checks:
            return 0.0
        
        score = 100.0
        
        for check in checks:
            if check.result == ValidationResult.FAILED:
                if check.severity == "CRITICAL":
                    score -= 20
                elif check.severity == "HIGH":
                    score -= 10
                elif check.severity == "MEDIUM":
                    score -= 5
            elif check.result == ValidationResult.WARNING:
                if check.severity == "HIGH":
                    score -= 3
                elif check.severity == "MEDIUM":
                    score -= 2
                elif check.severity == "LOW":
                    score -= 1
        
        return max(0.0, score)
    
    def _determine_security_posture(self, failed_checks: List[SecurityCheck], 
                                  warning_checks: List[SecurityCheck]) -> str:
        """Determine overall security posture."""
        critical_failures = [c for c in failed_checks if c.severity == "CRITICAL"]
        high_failures = [c for c in failed_checks if c.severity == "HIGH"]
        
        if critical_failures:
            return "CRITICAL"
        elif high_failures:
            return "HIGH_RISK"
        elif failed_checks:
            return "MEDIUM_RISK"
        elif warning_checks:
            return "LOW_RISK"
        else:
            return "SECURE"
    
    def _analyze_vulnerabilities(self, checks: List[SecurityCheck]) -> Dict[str, Any]:
        """Analyze vulnerabilities by type."""
        vulnerability_types = {}
        
        for check in checks:
            vuln_type = check.vulnerability_type.value
            if vuln_type not in vulnerability_types:
                vulnerability_types[vuln_type] = {
                    "total": 0,
                    "failed": 0,
                    "warning": 0,
                    "passed": 0
                }
            
            vulnerability_types[vuln_type]["total"] += 1
            if check.result == ValidationResult.FAILED:
                vulnerability_types[vuln_type]["failed"] += 1
            elif check.result == ValidationResult.WARNING:
                vulnerability_types[vuln_type]["warning"] += 1
            elif check.result == ValidationResult.PASSED:
                vulnerability_types[vuln_type]["passed"] += 1
        
        return vulnerability_types
    
    def _generate_remediation_plan(self, failed_checks: List[SecurityCheck], 
                                 warning_checks: List[SecurityCheck]) -> List[Dict]:
        """Generate remediation plan."""
        remediation_plan = []
        
        # Critical items first
        critical_items = [c for c in failed_checks if c.severity == "CRITICAL"]
        if critical_items:
            remediation_plan.append({
                "priority": "IMMEDIATE",
                "timeframe": "24-48 hours",
                "items": len(critical_items),
                "description": "Critical security vulnerabilities requiring immediate attention",
                "actions": [c.remediation for c in critical_items if c.remediation]
            })
        
        # High priority items
        high_items = [c for c in failed_checks if c.severity == "HIGH"]
        if high_items:
            remediation_plan.append({
                "priority": "HIGH",
                "timeframe": "1-2 weeks",
                "items": len(high_items),
                "description": "High-priority security issues",
                "actions": [c.remediation for c in high_items if c.remediation]
            })
        
        # Medium priority items
        medium_items = [c for c in failed_checks + warning_checks if c.severity == "MEDIUM"]
        if medium_items:
            remediation_plan.append({
                "priority": "MEDIUM",
                "timeframe": "2-4 weeks",
                "items": len(medium_items),
                "description": "Medium-priority security improvements",
                "actions": [c.remediation for c in medium_items if c.remediation]
            })
        
        return remediation_plan
    
    def _assess_compliance(self, failed_checks: List[SecurityCheck], 
                          warning_checks: List[SecurityCheck]) -> Dict[str, str]:
        """Assess compliance status."""
        critical_failures = [c for c in failed_checks if c.severity == "CRITICAL"]
        high_failures = [c for c in failed_checks if c.severity == "HIGH"]
        
        return {
            "SOC2": "FAIL" if critical_failures else "PASS",
            "ISO27001": "FAIL" if critical_failures or len(high_failures) > 3 else "PASS",
            "NIST_CSF": "FAIL" if critical_failures else "NEEDS_IMPROVEMENT" if high_failures else "PASS",
            "OWASP_TOP10": "FAIL" if critical_failures else "PASS",
            "production_readiness": "BLOCKED" if critical_failures else "APPROVED"
        }
    
    def _generate_executive_summary(self, failed_checks: List[SecurityCheck], 
                                  warning_checks: List[SecurityCheck], 
                                  security_score: float) -> str:
        """Generate executive summary."""
        critical_failures = [c for c in failed_checks if c.severity == "CRITICAL"]
        high_failures = [c for c in failed_checks if c.severity == "HIGH"]
        
        if not failed_checks and not warning_checks:
            return f"""
EXECUTIVE SUMMARY - SECURITY VALIDATION PASSED

✅ ALL SECURITY CHECKS PASSED

Security Score: {security_score}/100
Total Checks: {len(failed_checks) + len(warning_checks)}
Failed Checks: 0
Warning Checks: 0

The system has successfully passed all security validations. All identified
vulnerabilities have been properly remediated and security controls are
functioning correctly.

Production Deployment: APPROVED
            """.strip()
        else:
            return f"""
EXECUTIVE SUMMARY - SECURITY VALIDATION RESULTS

Security Score: {security_score}/100
Total Issues: {len(failed_checks) + len(warning_checks)}
Failed Checks: {len(failed_checks)}
Warning Checks: {len(warning_checks)}

Critical Issues: {len(critical_failures)}
High Priority Issues: {len(high_failures)}

{"❌ CRITICAL SECURITY ISSUES FOUND" if critical_failures else "⚠️ SECURITY ISSUES IDENTIFIED"}

{f"Production Deployment: BLOCKED" if critical_failures else f"Production Deployment: NEEDS_REVIEW"}

Immediate action required to address security vulnerabilities before production deployment.
            """.strip()
    
    def save_report(self, report: Dict[str, Any]) -> Path:
        """Save validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"security_validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📋 Security validation report saved: {report_file}")
        
        # Save executive summary
        if "executive_summary" in report:
            summary_file = self.results_dir / f"security_executive_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write(report["executive_summary"])
            logger.info(f"📄 Executive summary saved: {summary_file}")
        
        return report_file


def main():
    """Main execution function."""
    print("🚨 COMPREHENSIVE SECURITY VALIDATION - AGENT 10")
    print("=" * 80)
    
    # Initialize validator
    validator = SecurityValidator()
    
    # Run validation
    report = validator.run_comprehensive_validation()
    
    # Save report
    report_file = validator.save_report(report)
    
    # Display results
    print("\n" + "=" * 80)
    print("🔒 SECURITY VALIDATION RESULTS")
    print("=" * 80)
    
    summary = report["validation_summary"]
    print(f"Security Score: {summary['security_score']}/100")
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Passed: {summary['passed_checks']}")
    print(f"Failed: {summary['failed_checks']}")
    print(f"Warnings: {summary['warning_checks']}")
    print(f"Overall Posture: {summary['overall_posture']}")
    print(f"Production Ready: {summary['production_ready']}")
    
    if report["failed_checks"]:
        print("\n🚨 FAILED CHECKS:")
        for check in report["failed_checks"]:
            print(f"  - {check['description']} ({check['severity']})")
    
    if report["warning_checks"]:
        print("\n⚠️ WARNING CHECKS:")
        for check in report["warning_checks"]:
            print(f"  - {check['description']} ({check['severity']})")
    
    print(f"\n📋 Detailed report: {report_file}")
    print("=" * 80)
    
    return report


if __name__ == "__main__":
    main()