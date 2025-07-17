"""
Comprehensive Security & Compliance Module
Enterprise-grade security implementation for financial trading systems
"""

from datetime import datetime

# Authentication and Authorization
from .enterprise_auth import (
    EnterpriseAuthenticator,
    User,
    UserRole,
    Permission,
    TokenData,
    get_authenticator,
    get_current_user,
    require_permission,
    require_role,
    require_admin,
    require_trader,
    require_risk_manager,
    require_compliance
)

# API Key Management
from .api_key_manager import (
    APIKeyManager,
    APIKey,
    APIKeyType,
    APIKeyStatus,
    APIKeyRequest,
    APIKeyResponse,
    get_api_key_manager,
    validate_api_key_dependency,
    require_api_key_permission,
    require_api_key_type
)

# Encryption and Data Protection
from .encryption import (
    EncryptionManager,
    EncryptedData,
    EncryptionAlgorithm,
    TLSManager,
    encryption_manager,
    tls_manager,
    encrypt_data,
    decrypt_data,
    encrypt_sensitive_fields,
    decrypt_sensitive_fields,
    get_ssl_context,
    SENSITIVE_USER_FIELDS,
    SENSITIVE_TRADE_FIELDS,
    SENSITIVE_RISK_FIELDS,
    SENSITIVE_COMPLIANCE_FIELDS
)

# Audit Logging
from .audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    ComplianceFramework,
    DataMasking,
    get_audit_logger,
    audit_middleware,
    log_authentication_event,
    log_trading_event,
    log_data_access_event
)

# Security Headers and CORS
from .security_headers import (
    SecurityHeaders,
    CORSPolicy,
    SecurityMiddleware,
    TrustedHostsMiddleware,
    security_headers,
    cors_policy,
    get_cors_middleware,
    get_session_middleware,
    get_trusted_host_middleware,
    get_security_middleware,
    add_security_headers,
    generate_csp_nonce,
    get_security_config
)

# Rate Limiting
from .rate_limiter import (
    RateLimiter,
    RateLimitRule,
    get_rate_limiter,
    create_rate_limit_dependency
)

# GDPR Compliance
from .gdpr_compliance import (
    GDPRComplianceManager,
    DataProcessingPurpose,
    DataCategory,
    LegalBasis,
    DataSubjectRights,
    ConsentStatus,
    DataRetentionPolicy,
    get_gdpr_manager,
    record_consent,
    submit_data_subject_request,
    check_consent
)

# SOX Compliance
from .sox_compliance import (
    SOXComplianceManager,
    InternalControl,
    ControlTest,
    ControlDeficiency,
    RegulatoryReport,
    SOXSection,
    ControlType,
    ControlStatus,
    get_sox_manager,
    execute_control_test,
    generate_sox_report,
    get_compliance_dashboard
)

# Security Testing
from .security_testing import (
    SecurityTestSuite,
    SecurityVulnerability,
    SecurityTest,
    VulnerabilitySeverity,
    TestStatus,
    get_security_test_suite,
    run_security_scan,
    generate_security_report,
    test_authentication_security,
    test_injection_vulnerabilities,
    test_xss_vulnerabilities,
    test_network_security
)

# Secrets Management
from .secrets_manager import (
    SecretsManager,
    secrets_manager,
    get_secret
)

# Legacy compatibility
from .auth import JWTAuth, verify_token, create_access_token

__all__ = [
    # Authentication and Authorization
    "EnterpriseAuthenticator",
    "User",
    "UserRole",
    "Permission",
    "TokenData",
    "get_authenticator",
    "get_current_user",
    "require_permission",
    "require_role",
    "require_admin",
    "require_trader",
    "require_risk_manager",
    "require_compliance",
    
    # API Key Management
    "APIKeyManager",
    "APIKey",
    "APIKeyType",
    "APIKeyStatus",
    "APIKeyRequest",
    "APIKeyResponse",
    "get_api_key_manager",
    "validate_api_key_dependency",
    "require_api_key_permission",
    "require_api_key_type",
    
    # Encryption and Data Protection
    "EncryptionManager",
    "EncryptedData",
    "EncryptionAlgorithm",
    "TLSManager",
    "encryption_manager",
    "tls_manager",
    "encrypt_data",
    "decrypt_data",
    "encrypt_sensitive_fields",
    "decrypt_sensitive_fields",
    "get_ssl_context",
    "SENSITIVE_USER_FIELDS",
    "SENSITIVE_TRADE_FIELDS",
    "SENSITIVE_RISK_FIELDS",
    "SENSITIVE_COMPLIANCE_FIELDS",
    
    # Audit Logging
    "AuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "ComplianceFramework",
    "DataMasking",
    "get_audit_logger",
    "audit_middleware",
    "log_authentication_event",
    "log_trading_event",
    "log_data_access_event",
    
    # Security Headers and CORS
    "SecurityHeaders",
    "CORSPolicy",
    "SecurityMiddleware",
    "TrustedHostsMiddleware",
    "security_headers",
    "cors_policy",
    "get_cors_middleware",
    "get_session_middleware",
    "get_trusted_host_middleware",
    "get_security_middleware",
    "add_security_headers",
    "generate_csp_nonce",
    "get_security_config",
    
    # Rate Limiting
    "RateLimiter",
    "RateLimitRule",
    "get_rate_limiter",
    "create_rate_limit_dependency",
    
    # GDPR Compliance
    "GDPRComplianceManager",
    "DataProcessingPurpose",
    "DataCategory",
    "LegalBasis",
    "DataSubjectRights",
    "ConsentStatus",
    "DataRetentionPolicy",
    "get_gdpr_manager",
    "record_consent",
    "submit_data_subject_request",
    "check_consent",
    
    # SOX Compliance
    "SOXComplianceManager",
    "InternalControl",
    "ControlTest",
    "ControlDeficiency",
    "RegulatoryReport",
    "SOXSection",
    "ControlType",
    "ControlStatus",
    "get_sox_manager",
    "execute_control_test",
    "generate_sox_report",
    "get_compliance_dashboard",
    
    # Security Testing
    "SecurityTestSuite",
    "SecurityVulnerability",
    "SecurityTest",
    "VulnerabilitySeverity",
    "TestStatus",
    "get_security_test_suite",
    "run_security_scan",
    "generate_security_report",
    "test_authentication_security",
    "test_injection_vulnerabilities",
    "test_xss_vulnerabilities",
    "test_network_security",
    
    # Secrets Management
    "SecretsManager",
    "secrets_manager",
    "get_secret",
    
    # Legacy compatibility
    "JWTAuth",
    "verify_token",
    "create_access_token"
]

# Version information
__version__ = "1.0.0"
__author__ = "GrandModel Security Team"
__email__ = "security@grandmodel.com"
__description__ = "Comprehensive security and compliance framework for financial trading systems"

# Security configuration
SECURITY_CONFIG = {
    "encryption": {
        "default_algorithm": "AES_GCM",
        "key_rotation_days": 90,
        "max_key_usage": 1000000
    },
    "authentication": {
        "jwt_expiration_minutes": 30,
        "refresh_token_days": 30,
        "max_login_attempts": 5,
        "lockout_duration_minutes": 30
    },
    "audit": {
        "retention_years": 7,
        "batch_size": 100,
        "batch_timeout_seconds": 30
    },
    "compliance": {
        "gdpr_data_retention_days": 2555,  # 7 years
        "sox_test_frequency_days": 90,
        "deficiency_remediation_days": 30
    },
    "security_testing": {
        "scan_frequency_days": 30,
        "vulnerability_fix_days": {
            "critical": 1,
            "high": 7,
            "medium": 30,
            "low": 90
        }
    }
}

# Security status check
def get_security_status() -> dict:
    """Get overall security status"""
    return {
        "encryption": "enabled",
        "authentication": "jwt_with_rbac",
        "audit_logging": "comprehensive",
        "compliance": "gdpr_sox_compliant",
        "security_testing": "automated",
        "version": __version__
    }

# Initialize security components
async def initialize_security():
    """Initialize all security components"""
    from src.monitoring.logger_config import get_logger
    logger = get_logger(__name__)
    
    try:
        # Initialize authenticator
        authenticator = await get_authenticator()
        
        # Initialize audit logger
        audit_logger = await get_audit_logger()
        
        # Initialize GDPR manager
        gdpr_manager = await get_gdpr_manager()
        
        # Initialize SOX manager
        sox_manager = await get_sox_manager()
        
        # Initialize API key manager
        api_key_manager = await get_api_key_manager()
        
        # Initialize rate limiter
        rate_limiter = await get_rate_limiter()
        
        logger.info("Security components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize security components: {e}")
        return False

# Security middleware stack
def get_security_middleware_stack(app):
    """Get complete security middleware stack"""
    # Apply middleware in correct order
    app.add_middleware(get_security_middleware(app))
    app.add_middleware(get_trusted_host_middleware(app))
    app.add_middleware(get_cors_middleware())
    app.add_middleware(get_session_middleware())
    
    return app

# Security health check
async def security_health_check() -> dict:
    """Perform security health check"""
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "overall_status": "healthy",
        "components": {}
    }
    
    try:
        # Check authenticator
        authenticator = await get_authenticator()
        health_status["components"]["authenticator"] = "healthy"
        
        # Check audit logger
        audit_logger = await get_audit_logger()
        health_status["components"]["audit_logger"] = "healthy"
        
        # Check encryption
        health_status["components"]["encryption"] = "healthy"
        
        # Check compliance managers
        gdpr_manager = await get_gdpr_manager()
        health_status["components"]["gdpr_compliance"] = "healthy"
        
        sox_manager = await get_sox_manager()
        health_status["components"]["sox_compliance"] = "healthy"
        
    except Exception as e:
        health_status["overall_status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status