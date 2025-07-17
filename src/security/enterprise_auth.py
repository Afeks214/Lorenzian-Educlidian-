"""
Enterprise Authentication & Authorization System
Comprehensive security implementation for financial trading systems
"""

import os
import jwt
import bcrypt
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from src.monitoring.logger_config import get_logger
from src.security.secrets_manager import get_secret

logger = get_logger(__name__)

# Security configuration
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 30
PASSWORD_HASH_ROUNDS = 12
MFA_WINDOW_SECONDS = 30

class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    VIEWER = "viewer"
    AUDITOR = "auditor"
    SYSTEM_SERVICE = "system_service"

class Permission(str, Enum):
    """System permissions"""
    # Trading permissions
    TRADE_EXECUTE = "trade:execute"
    TRADE_VIEW = "trade:view"
    TRADE_HISTORY = "trade:history"
    
    # Risk management permissions
    RISK_VIEW = "risk:view"
    RISK_CONFIGURE = "risk:configure"
    RISK_OVERRIDE = "risk:override"
    
    # System administration
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_CONFIG = "admin:config"
    
    # Compliance
    COMPLIANCE_AUDIT = "compliance:audit"
    COMPLIANCE_REPORTS = "compliance:reports"
    
    # Data access
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_EXPORT = "data:export"

# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.TRADE_EXECUTE, Permission.TRADE_VIEW, Permission.TRADE_HISTORY,
        Permission.RISK_VIEW, Permission.RISK_CONFIGURE, Permission.RISK_OVERRIDE,
        Permission.ADMIN_USERS, Permission.ADMIN_SYSTEM, Permission.ADMIN_CONFIG,
        Permission.COMPLIANCE_AUDIT, Permission.COMPLIANCE_REPORTS,
        Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_EXPORT
    ],
    UserRole.TRADER: [
        Permission.TRADE_EXECUTE, Permission.TRADE_VIEW, Permission.TRADE_HISTORY,
        Permission.RISK_VIEW, Permission.DATA_READ
    ],
    UserRole.RISK_MANAGER: [
        Permission.TRADE_VIEW, Permission.TRADE_HISTORY,
        Permission.RISK_VIEW, Permission.RISK_CONFIGURE, Permission.RISK_OVERRIDE,
        Permission.DATA_READ, Permission.DATA_WRITE
    ],
    UserRole.COMPLIANCE_OFFICER: [
        Permission.TRADE_VIEW, Permission.TRADE_HISTORY,
        Permission.RISK_VIEW, Permission.COMPLIANCE_AUDIT, Permission.COMPLIANCE_REPORTS,
        Permission.DATA_READ, Permission.DATA_EXPORT
    ],
    UserRole.VIEWER: [
        Permission.TRADE_VIEW, Permission.RISK_VIEW, Permission.DATA_READ
    ],
    UserRole.AUDITOR: [
        Permission.TRADE_VIEW, Permission.TRADE_HISTORY,
        Permission.RISK_VIEW, Permission.COMPLIANCE_AUDIT, Permission.COMPLIANCE_REPORTS,
        Permission.DATA_READ
    ],
    UserRole.SYSTEM_SERVICE: [
        Permission.TRADE_EXECUTE, Permission.TRADE_VIEW, Permission.TRADE_HISTORY,
        Permission.RISK_VIEW, Permission.RISK_CONFIGURE,
        Permission.DATA_READ, Permission.DATA_WRITE
    ]
}

@dataclass
class User:
    """User model"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[Permission]
    mfa_enabled: bool = False
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Ensure permissions match role
        if not self.permissions:
            self.permissions = ROLE_PERMISSIONS.get(self.role, [])

class TokenData(BaseModel):
    """Token payload model"""
    user_id: str
    username: str
    role: UserRole
    permissions: List[Permission]
    session_id: str
    exp: int
    iat: int
    type: str  # 'access' or 'refresh'

class LoginRequest(BaseModel):
    """Login request model"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    mfa_token: Optional[str] = Field(None, min_length=6, max_length=6)
    remember_me: bool = False

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class EnterpriseAuthenticator:
    """Enterprise-grade authentication and authorization system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.security = HTTPBearer()
        self.jwt_secret = self._get_jwt_secret()
        self.cipher_suite = self._init_encryption()
        
        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 30
        self.session_timeout_hours = 8
        
        # In-memory user store (replace with database in production)
        self.users: Dict[str, User] = {}
        self._init_default_users()
        
        logger.info("Enterprise authenticator initialized")
    
    def _get_jwt_secret(self) -> str:
        """Get JWT secret from secure storage"""
        secret = get_secret("jwt_secret")
        if not secret:
            # Generate secure secret if not found
            secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            logger.warning("JWT secret generated - store securely for production")
        return secret
    
    def _init_encryption(self) -> Fernet:
        """Initialize encryption for sensitive data"""
        key = get_secret("encryption_key")
        if not key:
            key = Fernet.generate_key()
            logger.warning("Encryption key generated - store securely for production")
        
        if isinstance(key, str):
            key = key.encode()
        
        return Fernet(key)
    
    def _init_default_users(self):
        """Initialize default users for development"""
        # Admin user
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@grandmodel.com",
            role=UserRole.ADMIN,
            permissions=ROLE_PERMISSIONS[UserRole.ADMIN],
            mfa_enabled=True
        )
        
        # Trader user
        trader_user = User(
            user_id="trader_001",
            username="trader",
            email="trader@grandmodel.com",
            role=UserRole.TRADER,
            permissions=ROLE_PERMISSIONS[UserRole.TRADER]
        )
        
        # Risk manager user
        risk_user = User(
            user_id="risk_001",
            username="riskmanager",
            email="risk@grandmodel.com",
            role=UserRole.RISK_MANAGER,
            permissions=ROLE_PERMISSIONS[UserRole.RISK_MANAGER]
        )
        
        # Store users with hashed passwords
        self.users["admin"] = admin_user
        self.users["trader"] = trader_user
        self.users["riskmanager"] = risk_user
        
        # Default password (hash of 'password123')
        default_password_hash = self._hash_password("password123")
        
        # Store password hashes (in production, use proper user management)
        if self.redis:
            asyncio.create_task(self._store_password_hashes(default_password_hash))
    
    async def _store_password_hashes(self, password_hash: str):
        """Store password hashes in Redis"""
        for username in self.users.keys():
            await self.redis.set(f"password:{username}", password_hash)
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=PASSWORD_HASH_ROUNDS)
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode(), hashed.encode())
    
    async def authenticate_user(self, username: str, password: str, mfa_token: Optional[str] = None) -> Optional[User]:
        """Authenticate user with credentials"""
        try:
            # Check if user exists
            if username not in self.users:
                logger.warning("Login attempt for non-existent user", username=username)
                return None
            
            user = self.users[username]
            
            # Check if account is locked
            if user.account_locked_until and datetime.utcnow() < user.account_locked_until:
                logger.warning("Login attempt for locked account", username=username)
                return None
            
            # Get password hash
            password_hash = None
            if self.redis:
                password_hash = await self.redis.get(f"password:{username}")
            
            if not password_hash:
                logger.error("Password hash not found", username=username)
                return None
            
            # Verify password
            if not self._verify_password(password, password_hash):
                await self._handle_failed_login(user)
                return None
            
            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_token:
                    logger.warning("MFA token required but not provided", username=username)
                    return None
                
                if not await self._verify_mfa_token(user.user_id, mfa_token):
                    logger.warning("Invalid MFA token", username=username)
                    return None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.account_locked_until = None
            user.last_login = datetime.utcnow()
            
            logger.info("User authenticated successfully", username=username, role=user.role)
            return user
            
        except Exception as e:
            logger.error("Authentication error", username=username, error=str(e))
            return None
    
    async def _handle_failed_login(self, user: User):
        """Handle failed login attempt"""
        user.failed_login_attempts += 1
        
        if user.failed_login_attempts >= self.max_login_attempts:
            user.account_locked_until = datetime.utcnow() + timedelta(minutes=self.lockout_duration_minutes)
            logger.warning(
                "Account locked due to failed login attempts",
                username=user.username,
                attempts=user.failed_login_attempts
            )
    
    async def _verify_mfa_token(self, user_id: str, token: str) -> bool:
        """Verify MFA token (TOTP)"""
        # This is a simplified MFA verification
        # In production, use proper TOTP implementation
        if not self.redis:
            return True  # Skip MFA if Redis not available
        
        # Get user's MFA secret
        mfa_secret = await self.redis.get(f"mfa_secret:{user_id}")
        if not mfa_secret:
            return False
        
        # Verify TOTP token (simplified)
        import pyotp
        totp = pyotp.TOTP(mfa_secret)
        return totp.verify(token, valid_window=1)
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token"""
        session_id = str(secrets.token_urlsafe(16))
        now = datetime.utcnow()
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "session_id": session_id,
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
            "iss": "grandmodel",
            "aud": "grandmodel-api"
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=JWT_ALGORITHM)
        
        # Store session in Redis for revocation
        if self.redis:
            asyncio.create_task(self._store_session(session_id, user.user_id, token))
        
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token"""
        session_id = str(secrets.token_urlsafe(16))
        now = datetime.utcnow()
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "session_id": session_id,
            "type": "refresh",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)).timestamp()),
            "iss": "grandmodel",
            "aud": "grandmodel-api"
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=JWT_ALGORITHM)
        
        # Store refresh token in Redis
        if self.redis:
            asyncio.create_task(self._store_refresh_token(session_id, user.user_id, token))
        
        return token
    
    async def _store_session(self, session_id: str, user_id: str, token: str):
        """Store session in Redis"""
        await self.redis.setex(
            f"session:{session_id}",
            self.session_timeout_hours * 3600,
            json.dumps({
                "user_id": user_id,
                "token": token,
                "created_at": datetime.utcnow().isoformat()
            })
        )
    
    async def _store_refresh_token(self, session_id: str, user_id: str, token: str):
        """Store refresh token in Redis"""
        await self.redis.setex(
            f"refresh:{session_id}",
            JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
            json.dumps({
                "user_id": user_id,
                "token": token,
                "created_at": datetime.utcnow().isoformat()
            })
        )
    
    async def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[JWT_ALGORITHM],
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                    "require_exp": True,
                    "require_iat": True
                }
            )
            
            # Validate required fields
            required_fields = ["user_id", "username", "session_id", "type"]
            for field in required_fields:
                if field not in payload:
                    logger.warning("Missing required field in token", field=field)
                    return None
            
            # Check if session is still valid
            if self.redis:
                session_key = f"session:{payload['session_id']}"
                session_data = await self.redis.get(session_key)
                if not session_data:
                    logger.warning("Session not found or expired", session_id=payload['session_id'])
                    return None
            
            # Create TokenData
            token_data = TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                role=UserRole(payload.get("role", UserRole.VIEWER)),
                permissions=[Permission(p) for p in payload.get("permissions", [])],
                session_id=payload["session_id"],
                exp=payload["exp"],
                iat=payload["iat"],
                type=payload["type"]
            )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
        except Exception as e:
            logger.error("Token verification error", error=str(e))
            return None
    
    async def revoke_token(self, session_id: str) -> bool:
        """Revoke token by session ID"""
        if not self.redis:
            return False
        
        try:
            # Remove session
            await self.redis.delete(f"session:{session_id}")
            await self.redis.delete(f"refresh:{session_id}")
            
            logger.info("Token revoked", session_id=session_id)
            return True
        except Exception as e:
            logger.error("Error revoking token", session_id=session_id, error=str(e))
            return False
    
    def has_permission(self, user_permissions: List[Permission], required_permission: Permission) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
    
    def has_any_permission(self, user_permissions: List[Permission], required_permissions: List[Permission]) -> bool:
        """Check if user has any of the required permissions"""
        return any(perm in user_permissions for perm in required_permissions)
    
    def has_all_permissions(self, user_permissions: List[Permission], required_permissions: List[Permission]) -> bool:
        """Check if user has all required permissions"""
        return all(perm in user_permissions for perm in required_permissions)

# Global authenticator instance
auth_instance: Optional[EnterpriseAuthenticator] = None

async def get_authenticator() -> EnterpriseAuthenticator:
    """Get or create authenticator instance"""
    global auth_instance
    
    if auth_instance is None:
        # Initialize Redis client
        redis_client = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                redis_client = await redis.from_url(redis_url)
            except Exception as e:
                logger.error("Failed to connect to Redis", error=str(e))
        
        auth_instance = EnterpriseAuthenticator(redis_client)
    
    return auth_instance

# FastAPI Dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> TokenData:
    """FastAPI dependency to get current user from token"""
    authenticator = await get_authenticator()
    token_data = await authenticator.verify_token(credentials.credentials)
    
    if not token_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return token_data

def require_permission(permission: Permission):
    """Dependency to require specific permission"""
    async def permission_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        authenticator = await get_authenticator()
        
        if not authenticator.has_permission(current_user.permissions, permission):
            logger.warning(
                "Permission denied",
                user_id=current_user.user_id,
                required_permission=permission,
                user_permissions=current_user.permissions
            )
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied. Required: {permission.value}"
            )
        
        return current_user
    
    return permission_checker

def require_role(role: UserRole):
    """Dependency to require specific role"""
    async def role_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if current_user.role != role:
            logger.warning(
                "Role access denied",
                user_id=current_user.user_id,
                required_role=role,
                user_role=current_user.role
            )
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required role: {role.value}"
            )
        
        return current_user
    
    return role_checker

# Common permission dependencies
require_admin = require_role(UserRole.ADMIN)
require_trader = require_permission(Permission.TRADE_EXECUTE)
require_risk_manager = require_role(UserRole.RISK_MANAGER)
require_compliance = require_role(UserRole.COMPLIANCE_OFFICER)

# Permission-based dependencies
require_trade_execute = require_permission(Permission.TRADE_EXECUTE)
require_risk_configure = require_permission(Permission.RISK_CONFIGURE)
require_admin_access = require_permission(Permission.ADMIN_SYSTEM)
require_data_export = require_permission(Permission.DATA_EXPORT)
