# Security Components

## Overview

The security module provides comprehensive protection for the GrandModel trading system, implementing defense-in-depth strategies including authentication, authorization, rate limiting, and secrets management. Security is critical in financial trading systems where unauthorized access could result in significant financial losses.

## Core Components

### Authentication (`auth.py`)

Multi-layered authentication system supporting various authentication methods.

**Supported Authentication Methods:**
- JWT (JSON Web Tokens) for API access
- API Key authentication for service-to-service communication
- Multi-factor authentication (MFA) for administrative access
- Certificate-based authentication for high-security operations

**Usage:**
```python
from src.security.auth import AuthManager, JWTAuthenticator

# Initialize authentication manager
auth_config = {
    'jwt': {
        'secret_key': '${JWT_SECRET_KEY}',
        'algorithm': 'HS256',
        'expiration_minutes': 60
    },
    'api_keys': {
        'admin_key': '${ADMIN_API_KEY}',
        'readonly_key': '${READONLY_API_KEY}'
    },
    'mfa': {
        'enabled': True,
        'totp_issuer': 'GrandModel'
    }
}

auth_manager = AuthManager(auth_config)

# JWT Authentication
jwt_auth = JWTAuthenticator(auth_config['jwt'])

# Create JWT token
user_claims = {
    'user_id': 'trader_001',
    'role': 'portfolio_manager',
    'permissions': ['read_positions', 'execute_trades']
}

token = jwt_auth.create_token(user_claims)
print(f"Access Token: {token}")

# Verify JWT token
try:
    claims = jwt_auth.verify_token(token)
    print(f"User: {claims['user_id']}, Role: {claims['role']}")
except InvalidTokenError:
    print("Invalid token")
```

**Role-Based Access Control (RBAC):**
```python
class RoleBasedAccessControl:
    """RBAC implementation for GrandModel"""
    
    def __init__(self, config):
        self.roles = self._load_roles(config['roles_config'])
        self.permissions = self._load_permissions(config['permissions_config'])
    
    def check_permission(self, user_claims: Dict, required_permission: str) -> bool:
        """Check if user has required permission"""
        
        user_role = user_claims.get('role')
        if not user_role:
            return False
        
        # Check explicit permissions
        user_permissions = user_claims.get('permissions', [])
        if required_permission in user_permissions:
            return True
        
        # Check role-based permissions
        role_permissions = self.roles.get(user_role, {}).get('permissions', [])
        return required_permission in role_permissions
    
    def get_user_permissions(self, user_claims: Dict) -> List[str]:
        """Get all permissions for a user"""
        permissions = set()
        
        # Add explicit permissions
        permissions.update(user_claims.get('permissions', []))
        
        # Add role-based permissions
        user_role = user_claims.get('role')
        if user_role and user_role in self.roles:
            permissions.update(self.roles[user_role].get('permissions', []))
        
        return list(permissions)

# Permission definitions
PERMISSIONS = {
    'read_positions': 'View portfolio positions',
    'execute_trades': 'Execute trading orders',
    'modify_settings': 'Modify system settings',
    'view_logs': 'Access system logs',
    'manage_users': 'Manage user accounts',
    'emergency_stop': 'Emergency system shutdown'
}

# Role definitions
ROLES = {
    'trader': {
        'permissions': ['read_positions', 'execute_trades'],
        'description': 'Standard trading role'
    },
    'portfolio_manager': {
        'permissions': ['read_positions', 'execute_trades', 'modify_settings'],
        'description': 'Portfolio management role'
    },
    'admin': {
        'permissions': ['*'],  # All permissions
        'description': 'System administrator'
    },
    'readonly': {
        'permissions': ['read_positions', 'view_logs'],
        'description': 'Read-only access'
    }
}
```

### Rate Limiter (`rate_limiter.py`)

Advanced rate limiting system to prevent abuse and ensure system stability.

**Features:**
- Token bucket algorithm for smooth rate limiting
- Sliding window rate limiting for burst protection
- User-specific and endpoint-specific limits
- Redis-backed distributed rate limiting
- Adaptive rate limiting based on system load

**Usage:**
```python
from src.security.rate_limiter import RateLimiter, TokenBucketLimiter

# Initialize rate limiter
rate_limiter_config = {
    'redis_url': 'redis://localhost:6379',
    'default_limits': {
        'requests_per_minute': 100,
        'requests_per_hour': 1000
    },
    'endpoint_limits': {
        '/api/v1/orders': {
            'requests_per_minute': 10,
            'requests_per_second': 2
        },
        '/api/v1/positions': {
            'requests_per_minute': 60
        }
    },
    'user_limits': {
        'premium_user': {
            'requests_per_minute': 200
        },
        'basic_user': {
            'requests_per_minute': 50
        }
    }
}

rate_limiter = RateLimiter(rate_limiter_config)

# Check rate limit
user_id = 'trader_001'
endpoint = '/api/v1/orders'

if await rate_limiter.check_limit(user_id, endpoint):
    # Request allowed
    await process_request()
else:
    # Rate limit exceeded
    raise RateLimitExceededError("Rate limit exceeded")

# Get rate limit status
limit_status = await rate_limiter.get_limit_status(user_id, endpoint)
print(f"Remaining requests: {limit_status.remaining}")
print(f"Reset time: {limit_status.reset_time}")
```

**Token Bucket Implementation:**
```python
class TokenBucketLimiter:
    """Token bucket rate limiter for smooth rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float, redis_client=None):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.redis = redis_client
        
    async def consume_tokens(self, key: str, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket"""
        
        if self.redis:
            # Distributed implementation using Redis
            return await self._consume_tokens_redis(key, tokens)
        else:
            # Local implementation
            return await self._consume_tokens_local(key, tokens)
    
    async def _consume_tokens_redis(self, key: str, tokens: int) -> bool:
        """Redis-based token bucket implementation"""
        
        current_time = time.time()
        bucket_key = f"rate_limit:bucket:{key}"
        
        # Lua script for atomic token bucket operation
        lua_script = """
        local bucket_key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time
        
        -- Calculate tokens to add based on time elapsed
        local time_elapsed = current_time - last_refill
        local tokens_to_add = time_elapsed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        if tokens >= tokens_requested then
            tokens = tokens - tokens_requested
            redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', bucket_key, 3600)  -- 1 hour expiry
            return 1  -- Success
        else
            redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', bucket_key, 3600)
            return 0  -- Failure
        end
        """
        
        result = await self.redis.eval(
            lua_script, 1, bucket_key,
            self.capacity, self.refill_rate, tokens, current_time
        )
        
        return bool(result)
```

### Secrets Manager (`secrets_manager.py`)

Secure secrets management with encryption and rotation capabilities.

**Features:**
- AES-256 encryption for secrets at rest
- Environment variable integration
- Secret rotation with zero-downtime
- HSM (Hardware Security Module) support
- Audit logging for secret access

**Usage:**
```python
from src.security.secrets_manager import SecretsManager

# Initialize secrets manager
secrets_config = {
    'encryption_key': '${MASTER_ENCRYPTION_KEY}',
    'storage_backend': 'file',  # or 'vault', 'aws_secrets_manager'
    'storage_path': '/secure/secrets',
    'rotation_enabled': True,
    'rotation_interval_days': 90
}

secrets_manager = SecretsManager(secrets_config)

# Store secret
await secrets_manager.store_secret(
    'database_password',
    'super_secure_password_123',
    metadata={'description': 'Main database password', 'created_by': 'admin'}
)

# Retrieve secret
database_password = await secrets_manager.get_secret('database_password')
print(f"Database password: {database_password}")

# List secrets (metadata only)
secrets_list = await secrets_manager.list_secrets()
for secret_info in secrets_list:
    print(f"Secret: {secret_info['name']}, Created: {secret_info['created_at']}")

# Rotate secret
new_password = await secrets_manager.rotate_secret('database_password')
print(f"New password: {new_password}")
```

**Secret Encryption:**
```python
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecretEncryption:
    """Handles encryption/decryption of secrets"""
    
    def __init__(self, master_key: str, salt: bytes = None):
        if salt is None:
            salt = os.urandom(16)
        
        self.salt = salt
        
        # Derive encryption key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.fernet = Fernet(key)
    
    def encrypt_secret(self, plaintext: str) -> str:
        """Encrypt a secret value"""
        encrypted_data = self.fernet.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_secret(self, encrypted_data: str) -> str:
        """Decrypt a secret value"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = self.fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode()
    
    def rotate_encryption_key(self, new_master_key: str) -> 'SecretEncryption':
        """Create new encryptor with rotated key"""
        return SecretEncryption(new_master_key, self.salt)
```

## Security Middleware

### API Security Middleware

```python
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class SecurityMiddleware:
    """FastAPI security middleware"""
    
    def __init__(self, auth_manager: AuthManager, rate_limiter: RateLimiter):
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        self.security = HTTPBearer()
    
    async def authenticate_request(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        """Authenticate API request"""
        
        try:
            # Verify JWT token
            claims = self.auth_manager.verify_token(credentials.credentials)
            
            # Check if token is not blacklisted
            if await self.auth_manager.is_token_blacklisted(credentials.credentials):
                raise HTTPException(status_code=401, detail="Token blacklisted")
            
            return claims
            
        except InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def check_rate_limit(self, request, user_claims: Dict[str, Any]):
        """Check rate limits for request"""
        
        user_id = user_claims.get('user_id', 'anonymous')
        endpoint = request.url.path
        
        if not await self.rate_limiter.check_limit(user_id, endpoint):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission"""
        
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Get user claims from context
                user_claims = kwargs.get('user_claims')
                
                if not user_claims:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                if not self.auth_manager.check_permission(user_claims, permission):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator

# Usage in FastAPI
app = FastAPI()
security_middleware = SecurityMiddleware(auth_manager, rate_limiter)

@app.get("/api/v1/positions")
@security_middleware.require_permission("read_positions")
async def get_positions(user_claims: Dict = Depends(security_middleware.authenticate_request)):
    # Rate limiting is automatically applied
    await security_middleware.check_rate_limit(request, user_claims)
    
    # Your endpoint logic here
    return {"positions": []}
```

### Input Validation and Sanitization

```python
from pydantic import BaseModel, validator
from typing import Optional
import re

class TradeOrderRequest(BaseModel):
    """Validated trade order request model"""
    
    symbol: str
    quantity: float
    order_type: str
    side: str
    price: Optional[float] = None
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate trading symbol format"""
        if not re.match(r'^[A-Z]{1,10}$', v):
            raise ValueError('Invalid symbol format')
        return v
    
    @validator('quantity')
    def validate_quantity(cls, v):
        """Validate order quantity"""
        if v <= 0:
            raise ValueError('Quantity must be positive')
        if v > 1_000_000:  # Maximum position size
            raise ValueError('Quantity exceeds maximum allowed')
        return v
    
    @validator('order_type')
    def validate_order_type(cls, v):
        """Validate order type"""
        allowed_types = ['market', 'limit', 'stop', 'stop_limit']
        if v.lower() not in allowed_types:
            raise ValueError(f'Order type must be one of: {allowed_types}')
        return v.lower()
    
    @validator('side')
    def validate_side(cls, v):
        """Validate order side"""
        if v.lower() not in ['buy', 'sell']:
            raise ValueError('Side must be buy or sell')
        return v.lower()
    
    @validator('price')
    def validate_price(cls, v, values):
        """Validate price for limit orders"""
        order_type = values.get('order_type')
        
        if order_type in ['limit', 'stop_limit'] and v is None:
            raise ValueError('Price required for limit orders')
        
        if v is not None and v <= 0:
            raise ValueError('Price must be positive')
        
        return v

class InputSanitizer:
    """Input sanitization utilities"""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 255) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            raise ValueError("Input must be string")
        
        # Remove null bytes
        sanitized = input_str.replace('\x00', '')
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        # Truncate to max length
        sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def sanitize_numeric(input_val, min_val: float = None, max_val: float = None) -> float:
        """Sanitize numeric input"""
        try:
            value = float(input_val)
        except (ValueError, TypeError):
            raise ValueError("Invalid numeric value")
        
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Invalid numeric value")
        
        if min_val is not None and value < min_val:
            raise ValueError(f"Value below minimum: {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"Value above maximum: {max_val}")
        
        return value
```

## Security Monitoring and Logging

### Security Event Logging

```python
import logging
from datetime import datetime
from enum import Enum

class SecurityEventType(Enum):
    """Security event types for logging"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    TOKEN_EXPIRED = "token_expired"
    SECRET_ACCESSED = "secret_accessed"
    CONFIGURATION_CHANGED = "configuration_changed"

class SecurityLogger:
    """Specialized security event logger"""
    
    def __init__(self, logger_name: str = "grandmodel.security"):
        self.logger = logging.getLogger(logger_name)
        
        # Configure security-specific formatter
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        
        # Add file handler for security logs
        file_handler = logging.FileHandler('logs/security.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Add syslog handler for centralized logging
        if self._syslog_available():
            syslog_handler = logging.handlers.SysLogHandler(address=('localhost', 514))
            syslog_handler.setFormatter(formatter)
            self.logger.addHandler(syslog_handler)
    
    def log_security_event(
        self, 
        event_type: SecurityEventType, 
        user_id: str = None,
        ip_address: str = None,
        details: Dict[str, Any] = None
    ):
        """Log security event with structured data"""
        
        event_data = {
            'event_type': event_type.value,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details or {}
        }
        
        # Determine log level based on event type
        if event_type in [SecurityEventType.LOGIN_FAILURE, SecurityEventType.PERMISSION_DENIED]:
            log_level = logging.WARNING
        elif event_type in [SecurityEventType.SUSPICIOUS_ACTIVITY]:
            log_level = logging.ERROR
        else:
            log_level = logging.INFO
        
        self.logger.log(log_level, f"Security Event: {json.dumps(event_data)}")
    
    def log_authentication_success(self, user_id: str, ip_address: str, method: str):
        """Log successful authentication"""
        self.log_security_event(
            SecurityEventType.LOGIN_SUCCESS,
            user_id=user_id,
            ip_address=ip_address,
            details={'auth_method': method}
        )
    
    def log_authentication_failure(self, attempted_user: str, ip_address: str, reason: str):
        """Log failed authentication attempt"""
        self.log_security_event(
            SecurityEventType.LOGIN_FAILURE,
            user_id=attempted_user,
            ip_address=ip_address,
            details={'failure_reason': reason}
        )
    
    def log_permission_denied(self, user_id: str, required_permission: str, resource: str):
        """Log permission denied event"""
        self.log_security_event(
            SecurityEventType.PERMISSION_DENIED,
            user_id=user_id,
            details={
                'required_permission': required_permission,
                'resource': resource
            }
        )
```

### Intrusion Detection

```python
class IntrusionDetectionSystem:
    """Basic intrusion detection for trading system"""
    
    def __init__(self, config):
        self.config = config
        self.failed_attempts = defaultdict(list)
        self.suspicious_patterns = {}
        
        # Load detection rules
        self.detection_rules = self._load_detection_rules(config)
    
    def analyze_request(self, request_data: Dict[str, Any]) -> List[str]:
        """Analyze request for suspicious patterns"""
        
        alerts = []
        user_id = request_data.get('user_id')
        ip_address = request_data.get('ip_address')
        endpoint = request_data.get('endpoint')
        
        # Check for brute force attempts
        if self._detect_brute_force(user_id, ip_address):
            alerts.append("Brute force attack detected")
        
        # Check for unusual trading patterns
        if endpoint.startswith('/api/v1/orders'):
            if self._detect_unusual_trading_pattern(request_data):
                alerts.append("Unusual trading pattern detected")
        
        # Check for privilege escalation attempts
        if self._detect_privilege_escalation(request_data):
            alerts.append("Privilege escalation attempt detected")
        
        # Check for data exfiltration patterns
        if self._detect_data_exfiltration(request_data):
            alerts.append("Potential data exfiltration detected")
        
        return alerts
    
    def _detect_brute_force(self, user_id: str, ip_address: str) -> bool:
        """Detect brute force authentication attempts"""
        
        current_time = time.time()
        window_minutes = self.config.get('brute_force_window_minutes', 15)
        max_attempts = self.config.get('brute_force_max_attempts', 5)
        
        # Check failed attempts for this user/IP combination
        key = f"{user_id}:{ip_address}"
        recent_attempts = [
            attempt_time for attempt_time in self.failed_attempts[key]
            if current_time - attempt_time < (window_minutes * 60)
        ]
        
        return len(recent_attempts) >= max_attempts
    
    def _detect_unusual_trading_pattern(self, request_data: Dict[str, Any]) -> bool:
        """Detect unusual trading patterns"""
        
        user_id = request_data.get('user_id')
        order_data = request_data.get('order_data', {})
        
        # Check for unusually large orders
        quantity = order_data.get('quantity', 0)
        user_avg_quantity = self._get_user_average_quantity(user_id)
        
        if quantity > user_avg_quantity * 10:  # 10x larger than average
            return True
        
        # Check for rapid-fire orders
        recent_orders = self._get_recent_orders(user_id, minutes=5)
        if len(recent_orders) > 50:  # More than 50 orders in 5 minutes
            return True
        
        return False
    
    def record_failed_attempt(self, user_id: str, ip_address: str):
        """Record failed authentication attempt"""
        key = f"{user_id}:{ip_address}"
        self.failed_attempts[key].append(time.time())
        
        # Clean up old attempts
        current_time = time.time()
        window_seconds = self.config.get('brute_force_window_minutes', 15) * 60
        
        self.failed_attempts[key] = [
            attempt_time for attempt_time in self.failed_attempts[key]
            if current_time - attempt_time < window_seconds
        ]
```

## Configuration

### Production Security Configuration

```yaml
security:
  # Authentication settings
  authentication:
    jwt:
      secret_key: "${JWT_SECRET_KEY}"
      algorithm: "HS256"
      expiration_minutes: 60
      refresh_token_expiration_days: 30
      issuer: "grandmodel-api"
      audience: "grandmodel-clients"
    
    api_keys:
      admin_key: "${ADMIN_API_KEY}"
      service_key: "${SERVICE_API_KEY}"
      readonly_key: "${READONLY_API_KEY}"
    
    mfa:
      enabled: true
      totp_issuer: "GrandModel"
      backup_codes_count: 10
      require_for_admin: true
  
  # Authorization settings
  authorization:
    rbac_enabled: true
    default_role: "readonly"
    
    roles:
      admin:
        permissions: ["*"]
        description: "Full system access"
      
      portfolio_manager:
        permissions: [
          "read_positions", "execute_trades", "modify_settings",
          "view_reports", "manage_strategies"
        ]
        description: "Portfolio management access"
      
      trader:
        permissions: ["read_positions", "execute_trades", "view_reports"]
        description: "Standard trading access"
      
      readonly:
        permissions: ["read_positions", "view_reports"]
        description: "Read-only access"
  
  # Rate limiting
  rate_limiting:
    enabled: true
    redis_url: "${REDIS_URL}"
    
    default_limits:
      requests_per_minute: 100
      requests_per_hour: 1000
      requests_per_day: 10000
    
    endpoint_limits:
      "/api/v1/orders":
        requests_per_minute: 10
        requests_per_second: 2
      
      "/api/v1/positions":
        requests_per_minute: 60
        requests_per_second: 5
      
      "/api/v1/reports":
        requests_per_minute: 20
        requests_per_hour: 200
    
    user_tier_limits:
      premium:
        requests_per_minute: 200
        requests_per_hour: 2000
      
      standard:
        requests_per_minute: 100
        requests_per_hour: 1000
      
      basic:
        requests_per_minute: 50
        requests_per_hour: 500
  
  # Secrets management
  secrets:
    encryption_key: "${MASTER_ENCRYPTION_KEY}"
    storage_backend: "vault"  # or "file", "aws_secrets_manager"
    
    vault_config:
      url: "${VAULT_URL}"
      token: "${VAULT_TOKEN}"
      mount_path: "grandmodel"
    
    rotation:
      enabled: true
      interval_days: 90
      notify_before_days: 7
  
  # Security monitoring
  monitoring:
    security_logging: true
    log_file: "logs/security.log"
    syslog_enabled: true
    syslog_address: ["localhost", 514]
    
    intrusion_detection:
      enabled: true
      brute_force_protection: true
      brute_force_window_minutes: 15
      brute_force_max_attempts: 5
      
      anomaly_detection: true
      baseline_learning_days: 30
      
    alerts:
      email_notifications: true
      slack_notifications: true
      pagerduty_integration: true
  
  # Encryption settings
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation_enabled: true
    key_rotation_interval_days: 365
    
    tls:
      min_version: "1.2"
      cipher_suites: [
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES128-GCM-SHA256"
      ]
      
  # Session management
  sessions:
    session_timeout_minutes: 30
    max_concurrent_sessions: 3
    secure_cookies: true
    httponly_cookies: true
    samesite: "strict"
```

### Development Security Configuration

```yaml
security:
  authentication:
    jwt:
      secret_key: "dev-secret-key-not-for-production"
      expiration_minutes: 480  # 8 hours for development
    
    mfa:
      enabled: false  # Disabled for development convenience
  
  rate_limiting:
    enabled: false  # Disabled for development testing
  
  monitoring:
    security_logging: true
    log_level: "DEBUG"
    
    intrusion_detection:
      enabled: false  # Disabled for development
```

## Testing

### Security Testing

```python
# tests/unit/test_security/test_auth.py
import pytest
from src.security.auth import JWTAuthenticator, InvalidTokenError

class TestJWTAuthentication:
    def setUp(self):
        self.config = {
            'secret_key': 'test-secret-key',
            'algorithm': 'HS256',
            'expiration_minutes': 60
        }
        self.jwt_auth = JWTAuthenticator(self.config)
    
    def test_token_creation_and_verification(self):
        """Test JWT token creation and verification"""
        
        user_claims = {
            'user_id': 'test_user',
            'role': 'trader',
            'permissions': ['read_positions']
        }
        
        # Create token
        token = self.jwt_auth.create_token(user_claims)
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token
        verified_claims = self.jwt_auth.verify_token(token)
        assert verified_claims['user_id'] == 'test_user'
        assert verified_claims['role'] == 'trader'
    
    def test_invalid_token_rejection(self):
        """Test rejection of invalid tokens"""
        
        invalid_token = "invalid.token.string"
        
        with pytest.raises(InvalidTokenError):
            self.jwt_auth.verify_token(invalid_token)
    
    def test_expired_token_rejection(self):
        """Test rejection of expired tokens"""
        
        # Create token with short expiration
        short_config = self.config.copy()
        short_config['expiration_minutes'] = -1  # Already expired
        
        short_jwt_auth = JWTAuthenticator(short_config)
        expired_token = short_jwt_auth.create_token({'user_id': 'test'})
        
        with pytest.raises(InvalidTokenError):
            self.jwt_auth.verify_token(expired_token)

# tests/unit/test_security/test_rate_limiter.py
@pytest.mark.asyncio
class TestRateLimiter:
    async def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality"""
        
        config = {
            'default_limits': {'requests_per_minute': 5}
        }
        
        rate_limiter = RateLimiter(config)
        user_id = 'test_user'
        endpoint = '/api/test'
        
        # Should allow first 5 requests
        for i in range(5):
            allowed = await rate_limiter.check_limit(user_id, endpoint)
            assert allowed
        
        # 6th request should be denied
        allowed = await rate_limiter.check_limit(user_id, endpoint)
        assert not allowed
```

### Penetration Testing

```python
# tests/security/test_penetration.py
import requests
import pytest

class TestSecurityPenetration:
    """Security penetration tests"""
    
    def test_sql_injection_protection(self):
        """Test protection against SQL injection"""
        
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/*",
            "1; DELETE FROM orders; --"
        ]
        
        for malicious_input in malicious_inputs:
            response = requests.get(
                f"{API_BASE_URL}/api/v1/positions",
                params={'symbol': malicious_input},
                headers={'Authorization': f'Bearer {valid_token}'}
            )
            
            # Should return 400 Bad Request for invalid input
            assert response.status_code == 400
    
    def test_brute_force_protection(self):
        """Test brute force protection"""
        
        # Attempt multiple failed logins
        for i in range(10):
            response = requests.post(
                f"{API_BASE_URL}/auth/login",
                json={
                    'username': 'admin',
                    'password': f'wrong_password_{i}'
                }
            )
            
            if i < 5:
                assert response.status_code == 401  # Unauthorized
            else:
                # Should start rate limiting after 5 attempts
                assert response.status_code == 429  # Too Many Requests
    
    def test_unauthorized_access_prevention(self):
        """Test prevention of unauthorized access"""
        
        protected_endpoints = [
            '/api/v1/admin/users',
            '/api/v1/admin/settings',
            '/api/v1/orders',
            '/api/v1/positions'
        ]
        
        for endpoint in protected_endpoints:
            # Request without token
            response = requests.get(f"{API_BASE_URL}{endpoint}")
            assert response.status_code == 401
            
            # Request with invalid token
            response = requests.get(
                f"{API_BASE_URL}{endpoint}",
                headers={'Authorization': 'Bearer invalid_token'}
            )
            assert response.status_code == 401
```

## Troubleshooting

### Common Security Issues

**Authentication Failures:**
- Check JWT secret key configuration
- Verify token expiration settings
- Review user permissions and roles
- Check for clock skew between services

**Rate Limiting Issues:**
- Verify Redis connectivity for distributed rate limiting
- Check rate limit configurations
- Review user-specific and endpoint-specific limits
- Monitor rate limiter performance

**Secret Management Problems:**
- Verify encryption key availability
- Check secret storage backend connectivity
- Review secret rotation schedules
- Validate secret access permissions

### Security Audit Commands

```bash
# Check security configuration
python -c "
from src.security.auth import AuthManager
auth = AuthManager(config)
print(auth.get_security_status())
"

# Test rate limiter
python -c "
from src.security.rate_limiter import RateLimiter
limiter = RateLimiter(config)
print(limiter.get_rate_limit_status('test_user', '/api/test'))
"

# Validate secrets
python -c "
from src.security.secrets_manager import SecretsManager
secrets = SecretsManager(config)
print(secrets.validate_all_secrets())
"

# Security health check
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/security/health
```

## Related Documentation

- [API Documentation](../../docs/api/)
- [Deployment Security Guide](../../docs/guides/deployment_guide.md)
- [Monitoring and Alerting](../monitoring/README.md)
- [System Architecture](../../docs/architecture/system_overview.md)