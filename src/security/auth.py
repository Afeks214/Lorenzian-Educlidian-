"""
JWT Authentication and RBAC implementation.
Provides secure token validation and role-based access control.
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from src.monitoring.logger_config import get_logger

logger = get_logger(__name__)

# Security configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# JWT Configuration with Vault Integration
import secrets
from src.security.vault_client import get_vault_client

def _generate_secure_secret() -> str:
    """Generate cryptographically secure JWT secret."""
    return secrets.token_urlsafe(64)  # 512-bit secret

# Initialize JWT secret from Vault
JWT_SECRET_KEY = None

async def _init_jwt_secret():
    """Initialize JWT secret from Vault with fallback."""
    global JWT_SECRET_KEY
    
    if JWT_SECRET_KEY is None:
        try:
            # Try to get from Vault first
            vault_client = await get_vault_client()
            JWT_SECRET_KEY = await vault_client.get_jwt_secret()
            
            # Validate secret strength
            if JWT_SECRET_KEY == "your-secret-key-change-in-production":
                logger.warning("SECURITY WARNING: Default JWT secret detected - generating secure secret")
                JWT_SECRET_KEY = _generate_secure_secret()
                
        except Exception as e:
            logger.error(f"Failed to retrieve JWT secret from Vault: {e}")
            # Fallback to environment or generate secure secret
            JWT_SECRET_KEY = os.getenv("JWT_SECRET") or _generate_secure_secret()
    
    return JWT_SECRET_KEY

# Get JWT secret synchronously for immediate use
try:
    import asyncio
    JWT_SECRET_KEY = asyncio.run(_init_jwt_secret())
except:
    # Fallback for environments where async initialization fails
    JWT_SECRET_KEY = os.getenv("JWT_SECRET") or _generate_secure_secret()
    
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Validate JWT secret strength
if len(JWT_SECRET_KEY) < 32:
    logger.error("SECURITY ERROR: JWT secret too short - must be at least 32 characters")
    raise ValueError("JWT secret must be at least 32 characters for security")


class Permission:
    """Permission constants for RBAC."""
    READ = "read"
    WRITE = "write"
    TRADE = "trade"
    ADMIN = "admin"
    MODEL_RELOAD = "model_reload"
    CONFIG_UPDATE = "config_update"


class Role:
    """Role definitions with associated permissions."""
    VIEWER = {
        "name": "viewer",
        "permissions": [Permission.READ]
    }
    
    TRADER = {
        "name": "trader",
        "permissions": [Permission.READ, Permission.TRADE]
    }
    
    OPERATOR = {
        "name": "operator",
        "permissions": [Permission.READ, Permission.WRITE, Permission.TRADE]
    }
    
    ADMIN = {
        "name": "admin",
        "permissions": [
            Permission.READ,
            Permission.WRITE,
            Permission.TRADE,
            Permission.ADMIN,
            Permission.MODEL_RELOAD,
            Permission.CONFIG_UPDATE
        ]
    }


class JWTAuth:
    """JWT Authentication handler with RBAC support."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize JWT auth with secret key."""
        self.secret_key = secret_key or JWT_SECRET_KEY
        self.algorithm = JWT_ALGORITHM
        self.vault_client = None
        
    async def _ensure_vault_client(self):
        """Ensure vault client is initialized."""
        if self.vault_client is None:
            try:
                self.vault_client = await get_vault_client()
            except Exception as e:
                logger.warning(f"Failed to initialize Vault client: {e}")
                self.vault_client = None
    
    async def get_current_jwt_secret(self) -> str:
        """Get current JWT secret, refreshing from Vault if needed."""
        await self._ensure_vault_client()
        
        if self.vault_client:
            try:
                secret = await self.vault_client.get_jwt_secret()
                if secret:
                    return secret
            except Exception as e:
                logger.warning(f"Failed to get JWT secret from Vault: {e}")
        
        return self.secret_key
        
    def create_token(self, 
                    user_id: str,
                    role: str,
                    permissions: List[str],
                    expiration_hours: int = JWT_EXPIRATION_HOURS) -> str:
        """
        Create a JWT token with user info and permissions.
        
        Args:
            user_id: Unique user identifier
            role: User role name
            permissions: List of permissions
            expiration_hours: Token validity in hours
            
        Returns:
            Encoded JWT token
        """
        payload = {
            "user_id": user_id,
            "role": role,
            "permissions": permissions,
            "exp": datetime.utcnow() + timedelta(hours=expiration_hours),
            "iat": datetime.utcnow(),
            "iss": "grandmodel"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.info(
            "Token created",
            user_id=user_id,
            role=role,
            expiration_hours=expiration_hours
        )
        
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode a JWT token with enhanced security validation.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            # Enhanced token validation options
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_signature": True,
                    "require_exp": True,
                    "require_iat": True
                }
            )
            
            # Validate required fields
            required_fields = ["user_id", "permissions", "exp", "iat", "iss"]
            for field in required_fields:
                if field not in payload:
                    raise jwt.InvalidTokenError(f"Missing required field: {field}")
            
            # Validate issuer
            if payload.get("iss") != "grandmodel":
                raise jwt.InvalidTokenError("Invalid token issuer")
            
            # Additional security: Check token age (prevent very old tokens)
            import datetime
            issued_at = datetime.datetime.fromtimestamp(payload["iat"])
            max_age = datetime.timedelta(hours=JWT_EXPIRATION_HOURS + 1)  # Allow 1 hour grace
            if datetime.datetime.utcnow() - issued_at > max_age:
                raise jwt.InvalidTokenError("Token too old")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired", token_preview=token[:20])
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e), token_preview=token[:20])
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
        except Exception as e:
            logger.error("Token verification failed", error=str(e))
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials"
            )
    
    def check_permission(self, user_permissions: List[str], required_permission: str) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_permissions: List of user's permissions
            required_permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        return required_permission in user_permissions
    
    def check_permissions(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """
        Check if user has all required permissions.
        
        Args:
            user_permissions: List of user's permissions
            required_permissions: List of required permissions
            
        Returns:
            True if user has all permissions, False otherwise
        """
        return all(perm in user_permissions for perm in required_permissions)


# Global JWT handler instance
jwt_auth = JWTAuth()


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """
    FastAPI dependency for JWT token verification.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        Decoded token payload with user info
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    return jwt_auth.verify_token(token)


def create_access_token(user_id: str, role: str, permissions: List[str]) -> str:
    """
    Create an access token for a user.
    
    Args:
        user_id: Unique user identifier
        role: User role
        permissions: User permissions
        
    Returns:
        JWT access token
    """
    return jwt_auth.create_token(user_id, role, permissions)


def require_permission(permission: str):
    """
    Dependency to require specific permission.
    
    Args:
        permission: Required permission
        
    Returns:
        FastAPI dependency function
    """
    async def permission_checker(user: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
        if not jwt_auth.check_permission(user.get("permissions", []), permission):
            logger.warning(
                "Permission denied",
                user_id=user.get("user_id"),
                required_permission=permission,
                user_permissions=user.get("permissions", [])
            )
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied. Required: {permission}"
            )
        return user
    
    return permission_checker


def require_permissions(permissions: List[str]):
    """
    Dependency to require multiple permissions.
    
    Args:
        permissions: List of required permissions
        
    Returns:
        FastAPI dependency function
    """
    async def permissions_checker(user: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
        if not jwt_auth.check_permissions(user.get("permissions", []), permissions):
            logger.warning(
                "Permissions denied",
                user_id=user.get("user_id"),
                required_permissions=permissions,
                user_permissions=user.get("permissions", [])
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permissions}"
            )
        return user
    
    return permissions_checker


# Convenience dependencies for common permission checks
require_read = require_permission(Permission.READ)
require_write = require_permission(Permission.WRITE)
require_trade = require_permission(Permission.TRADE)
require_admin = require_permission(Permission.ADMIN)


def hash_password(password: str) -> str:
    """
    Hash a password for storage.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)