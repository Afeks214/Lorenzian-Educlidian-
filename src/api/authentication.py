"""
Enterprise-grade JWT authentication system with Role-Based Access Control (RBAC).
Implements secure authentication for the Human-in-the-Loop Dashboard.
"""

import os
import jwt
import bcrypt
import secrets
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis

from src.monitoring.logger_config import get_logger

logger = get_logger(__name__)

# JWT Configuration with Vault Integration
from src.security.vault_client import get_vault_client

JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "8"))

# JWT secret will be loaded from Vault
JWT_SECRET_KEY = None

async def _get_jwt_secret() -> str:
    """Get JWT secret from Vault with fallback."""
    global JWT_SECRET_KEY
    
    if JWT_SECRET_KEY is None:
        try:
            vault_client = await get_vault_client()
            JWT_SECRET_KEY = await vault_client.get_jwt_secret()
        except Exception as e:
            logger.error(f"Failed to get JWT secret from Vault: {e}")
            # Fallback to environment or generate secure secret
            JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    
    return JWT_SECRET_KEY

# Redis client for session management
redis_client: Optional[redis.Redis] = None

class UserRole(str, Enum):
    """User roles in the system."""
    RISK_OPERATOR = "risk_operator"
    RISK_MANAGER = "risk_manager"
    SYSTEM_ADMIN = "system_admin"
    COMPLIANCE_OFFICER = "compliance_officer"
    VIEWER = "viewer"

class RolePermission(str, Enum):
    """Granular permissions for role-based access control."""
    # Dashboard permissions
    DASHBOARD_READ = "dashboard_read"
    DASHBOARD_ADMIN = "dashboard_admin"
    
    # Trade permissions
    TRADE_REVIEW = "trade_review"
    TRADE_APPROVE = "trade_approve"
    TRADE_REJECT = "trade_reject"
    HIGH_RISK_APPROVE = "high_risk_approve"
    
    # System permissions
    SYSTEM_INTEGRATION = "system_integration"
    USER_MANAGEMENT = "user_management"
    AUDIT_READ = "audit_read"
    CONFIG_MANAGE = "config_manage"
    
    # Compliance permissions
    COMPLIANCE_READ = "compliance_read"
    COMPLIANCE_REPORT = "compliance_report"

# Role to permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, List[RolePermission]] = {
    UserRole.VIEWER: [
        RolePermission.DASHBOARD_READ,
    ],
    UserRole.RISK_OPERATOR: [
        RolePermission.DASHBOARD_READ,
        RolePermission.TRADE_REVIEW,
        RolePermission.TRADE_APPROVE,
        RolePermission.TRADE_REJECT,
    ],
    UserRole.RISK_MANAGER: [
        RolePermission.DASHBOARD_READ,
        RolePermission.TRADE_REVIEW,
        RolePermission.TRADE_APPROVE,
        RolePermission.TRADE_REJECT,
        RolePermission.HIGH_RISK_APPROVE,
        RolePermission.AUDIT_READ,
        RolePermission.COMPLIANCE_READ,
    ],
    UserRole.SYSTEM_ADMIN: [
        RolePermission.DASHBOARD_READ,
        RolePermission.DASHBOARD_ADMIN,
        RolePermission.SYSTEM_INTEGRATION,
        RolePermission.USER_MANAGEMENT,
        RolePermission.CONFIG_MANAGE,
        RolePermission.AUDIT_READ,
    ],
    UserRole.COMPLIANCE_OFFICER: [
        RolePermission.DASHBOARD_READ,
        RolePermission.AUDIT_READ,
        RolePermission.COMPLIANCE_READ,
        RolePermission.COMPLIANCE_REPORT,
    ],
}

class UserInfo(BaseModel):
    """User information model."""
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    role: UserRole = Field(..., description="User role")
    permissions: List[RolePermission] = Field(..., description="User permissions")
    session_id: str = Field(..., description="Session identifier")
    login_time: datetime = Field(..., description="Login timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    mfa_enabled: bool = Field(default=False, description="Multi-factor authentication enabled")

class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    mfa_token: Optional[str] = Field(None, description="Multi-factor authentication token")

class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user_info: UserInfo = Field(..., description="User information")

class TokenPayload(BaseModel):
    """JWT token payload."""
    user_id: str
    username: str
    role: str
    session_id: str
    exp: int
    iat: int
    jti: str  # JWT ID for token revocation

# Security instance
security = HTTPBearer()

# Secure user database with Vault-managed passwords
USERS_DB: Dict[str, Dict[str, Any]] = {}

async def _init_users_db():
    """Initialize user database with Vault-managed credentials."""
    global USERS_DB
    
    try:
        vault_client = await get_vault_client()
        
        # Get user credentials from Vault
        admin_creds = await vault_client.get_secret("users/admin")
        risk_creds = await vault_client.get_secret("users/risk_manager")
        operator_creds = await vault_client.get_secret("users/operator")
        compliance_creds = await vault_client.get_secret("users/compliance")
        
        # Build user database with Vault credentials
        USERS_DB = {
            "admin": {
                "user_id": "user_001",
                "username": "admin",
                "email": "admin@grandmodel.app",
                "password_hash": bcrypt.hashpw(
                    (admin_creds.get('password', 'fallback_admin123!') if admin_creds else 'fallback_admin123!').encode(), 
                    bcrypt.gensalt()
                ).decode(),
                "role": UserRole.SYSTEM_ADMIN,
                "mfa_enabled": False,
                "active": True
            },
            "risk_manager": {
                "user_id": "user_002", 
                "username": "risk_manager",
                "email": "risk@grandmodel.app",
                "password_hash": bcrypt.hashpw(
                    (risk_creds.get('password', 'fallback_risk123!') if risk_creds else 'fallback_risk123!').encode(), 
                    bcrypt.gensalt()
                ).decode(),
                "role": UserRole.RISK_MANAGER,
                "mfa_enabled": True,
                "active": True
            },
            "operator": {
                "user_id": "user_003",
                "username": "operator", 
                "email": "operator@grandmodel.app",
                "password_hash": bcrypt.hashpw(
                    (operator_creds.get('password', 'fallback_operator123!') if operator_creds else 'fallback_operator123!').encode(), 
                    bcrypt.gensalt()
                ).decode(),
                "role": UserRole.RISK_OPERATOR,
                "mfa_enabled": False,
                "active": True
            },
            "compliance": {
                "user_id": "user_004",
                "username": "compliance",
                "email": "compliance@grandmodel.app", 
                "password_hash": bcrypt.hashpw(
                    (compliance_creds.get('password', 'fallback_compliance123!') if compliance_creds else 'fallback_compliance123!').encode(), 
                    bcrypt.gensalt()
                ).decode(),
                "role": UserRole.COMPLIANCE_OFFICER,
                "mfa_enabled": False,
                "active": True
            }
        }
        
        logger.info("User database initialized with Vault-managed credentials")
        
    except Exception as e:
        logger.error(f"Failed to initialize users from Vault: {e}")
        # Fallback to hardcoded credentials for development
        USERS_DB = {
            "admin": {
                "user_id": "user_001",
                "username": "admin",
                "email": "admin@grandmodel.app",
                "password_hash": bcrypt.hashpw(b"dev_admin123!", bcrypt.gensalt()).decode(),
                "role": UserRole.SYSTEM_ADMIN,
                "mfa_enabled": False,
                "active": True
            },
            "risk_manager": {
                "user_id": "user_002", 
                "username": "risk_manager",
                "email": "risk@grandmodel.app",
                "password_hash": bcrypt.hashpw(b"dev_risk123!", bcrypt.gensalt()).decode(),
                "role": UserRole.RISK_MANAGER,
                "mfa_enabled": True,
                "active": True
            },
            "operator": {
                "user_id": "user_003",
                "username": "operator", 
                "email": "operator@grandmodel.app",
                "password_hash": bcrypt.hashpw(b"dev_operator123!", bcrypt.gensalt()).decode(),
                "role": UserRole.RISK_OPERATOR,
                "mfa_enabled": False,
                "active": True
            },
            "compliance": {
                "user_id": "user_004",
                "username": "compliance",
                "email": "compliance@grandmodel.app", 
                "password_hash": bcrypt.hashpw(b"dev_compliance123!", bcrypt.gensalt()).decode(),
                "role": UserRole.COMPLIANCE_OFFICER,
                "mfa_enabled": False,
                "active": True
            }
        }
        
        logger.warning("Using development fallback credentials - configure Vault for production")

async def init_redis():
    """Initialize Redis connection for session management."""
    global redis_client
    if not redis_client:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)
    
    # Initialize user database if not already done
    if not USERS_DB:
        await _init_users_db()

async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

async def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials."""
    user = USERS_DB.get(username)
    if not user or not user["active"]:
        return None
    
    if not await verify_password(password, user["password_hash"]):
        return None
    
    return user

async def create_access_token(user_data: Dict[str, Any], session_id: str) -> str:
    """Create JWT access token with Vault-managed secret."""
    now = datetime.utcnow()
    expire = now + timedelta(hours=JWT_EXPIRATION_HOURS)
    
    # Get JWT secret from Vault
    jwt_secret = await _get_jwt_secret()
    
    payload = TokenPayload(
        user_id=user_data["user_id"],
        username=user_data["username"],
        role=user_data["role"].value,
        session_id=session_id,
        exp=int(expire.timestamp()),
        iat=int(now.timestamp()),
        jti=secrets.token_urlsafe(16)
    )
    
    return jwt.encode(payload.dict(), jwt_secret, algorithm=JWT_ALGORITHM)

async def create_refresh_token(user_id: str, session_id: str) -> str:
    """Create refresh token with Vault-managed secret."""
    now = datetime.utcnow()
    expire = now + timedelta(days=30)
    
    # Get JWT secret from Vault
    jwt_secret = await _get_jwt_secret()
    
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "exp": int(expire.timestamp()),
        "type": "refresh"
    }
    
    return jwt.encode(payload, jwt_secret, algorithm=JWT_ALGORITHM)

async def verify_token_payload(token: str) -> TokenPayload:
    """Verify and decode JWT token with Vault-managed secret."""
    try:
        # Get JWT secret from Vault
        jwt_secret = await _get_jwt_secret()
        payload = jwt.decode(token, jwt_secret, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInfo:
    """Dependency to verify JWT token and return user info."""
    await init_redis()
    
    token = credentials.credentials
    payload = await verify_token_payload(token)
    
    # Check if session is still valid
    if redis_client:
        try:
            session_data = await redis_client.get(f"session:{payload.session_id}")
            if not session_data:
                raise HTTPException(status_code=401, detail="Session expired")
        except Exception as e:
            logger.error(f"Redis session check failed: {e}")
            # Continue without Redis check in case of Redis issues
    
    # Get user data
    user_data = None
    for user in USERS_DB.values():
        if user["user_id"] == payload.user_id:
            user_data = user
            break
    
    if not user_data or not user_data["active"]:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    # Get permissions for user role
    permissions = ROLE_PERMISSIONS.get(user_data["role"], [])
    
    return UserInfo(
        user_id=payload.user_id,
        username=payload.username,
        email=user_data["email"],
        role=user_data["role"],
        permissions=permissions,
        session_id=payload.session_id,
        login_time=datetime.fromtimestamp(payload.iat),
        last_activity=datetime.utcnow(),
        mfa_enabled=user_data["mfa_enabled"]
    )

async def login(login_request: LoginRequest) -> LoginResponse:
    """Process user login and return tokens."""
    await init_redis()
    
    # Authenticate user
    user_data = await authenticate_user(login_request.username, login_request.password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check MFA if enabled
    if user_data["mfa_enabled"] and not login_request.mfa_token:
        raise HTTPException(status_code=401, detail="MFA token required")
    
    if user_data["mfa_enabled"] and login_request.mfa_token:
        # In production, verify MFA token here
        if login_request.mfa_token != "123456":  # Mock MFA verification
            raise HTTPException(status_code=401, detail="Invalid MFA token")
    
    # Create session
    session_id = secrets.token_urlsafe(32)
    
    # Store session in Redis
    if redis_client:
        try:
            session_data = {
                "user_id": user_data["user_id"],
                "username": user_data["username"],
                "role": user_data["role"].value,
                "login_time": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat()
            }
            # Get JWT secret for session encoding
            jwt_secret = await _get_jwt_secret()
            await redis_client.setex(
                f"session:{session_id}",
                JWT_EXPIRATION_HOURS * 3600,
                jwt.encode(session_data, jwt_secret)
            )
        except Exception as e:
            logger.error(f"Failed to store session in Redis: {e}")
    
    # Create tokens with Vault integration
    access_token = await create_access_token(user_data, session_id)
    refresh_token = await create_refresh_token(user_data["user_id"], session_id)
    
    # Get permissions
    permissions = ROLE_PERMISSIONS.get(user_data["role"], [])
    
    user_info = UserInfo(
        user_id=user_data["user_id"],
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        permissions=permissions,
        session_id=session_id,
        login_time=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        mfa_enabled=user_data["mfa_enabled"]
    )
    
    logger.info(f"User logged in: {user_data['username']} (role: {user_data['role'].value})")
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user_info=user_info
    )

async def logout(session_id: str) -> bool:
    """Logout user and invalidate session."""
    await init_redis()
    
    if redis_client:
        try:
            await redis_client.delete(f"session:{session_id}")
            logger.info(f"Session {session_id} invalidated")
            return True
        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")
    
    return False

def require_permission(required_permission: RolePermission):
    """Decorator to require specific permission."""
    def permission_checker(user: UserInfo = Depends(verify_token)) -> UserInfo:
        if required_permission not in user.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{required_permission.value}' required"
            )
        return user
    return permission_checker

def require_role(required_role: UserRole):
    """Decorator to require specific role."""
    def role_checker(user: UserInfo = Depends(verify_token)) -> UserInfo:
        if user.role != required_role:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{required_role.value}' required"
            )
        return user
    return role_checker

async def audit_log(
    user_id: str,
    action: str,
    resource: str,
    details: Dict[str, Any] = None,
    request: Request = None
) -> bool:
    """Log user action for audit trail."""
    await init_redis()
    
    audit_entry = {
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat(),
        "ip_address": request.client.host if request else "unknown",
        "user_agent": request.headers.get("user-agent") if request else "unknown"
    }
    
    if redis_client:
        try:
            # Store audit log with 1 year retention
            jwt_secret = await _get_jwt_secret()
            await redis_client.setex(
                f"audit:{datetime.utcnow().strftime('%Y%m%d')}:{secrets.token_urlsafe(8)}",
                365 * 24 * 3600,
                jwt.encode(audit_entry, jwt_secret)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store audit log: {e}")
    
    # Always log to application logger as backup
    logger.info(f"AUDIT: {audit_entry}")
    return False

# Security monitoring functions
async def check_failed_login_attempts(username: str) -> int:
    """Check number of failed login attempts for user."""
    await init_redis()
    
    if redis_client:
        try:
            count = await redis_client.get(f"failed_login:{username}")
            return int(count) if count else 0
        except Exception:
            return 0
    return 0

async def record_failed_login(username: str) -> int:
    """Record failed login attempt."""
    await init_redis()
    
    if redis_client:
        try:
            # Increment counter with 15 minute expiration
            count = await redis_client.incr(f"failed_login:{username}")
            await redis_client.expire(f"failed_login:{username}", 900)
            return count
        except Exception:
            return 0
    return 0

async def clear_failed_logins(username: str) -> bool:
    """Clear failed login attempts after successful login."""
    await init_redis()
    
    if redis_client:
        try:
            await redis_client.delete(f"failed_login:{username}")
            return True
        except Exception:
            return False
    return False