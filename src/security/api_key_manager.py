"""
API Key Management System
Secure API key generation, validation, and lifecycle management
"""

import os
import secrets
import hashlib
import base64
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis.asyncio as redis
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from src.monitoring.logger_config import get_logger
from src.security.secrets_manager import get_secret
from src.security.enterprise_auth import Permission, UserRole

logger = get_logger(__name__)

class APIKeyType(str, Enum):
    """API Key types"""
    TRADING = "trading"
    ANALYTICS = "analytics"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    ADMIN = "admin"
    SYSTEM = "system"
    WEBHOOK = "webhook"
    THIRD_PARTY = "third_party"

class APIKeyStatus(str, Enum):
    """API Key status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    EXPIRED = "expired"

@dataclass
class APIKeyScope:
    """API Key scope and permissions"""
    key_type: APIKeyType
    permissions: List[Permission]
    rate_limit: int = 1000  # requests per hour
    ip_whitelist: Optional[List[str]] = None
    allowed_endpoints: Optional[List[str]] = None
    data_access_level: str = "read"  # read, write, admin

@dataclass
class APIKey:
    """API Key model"""
    key_id: str
    key_name: str
    key_hash: str  # Hashed version of the key
    key_type: APIKeyType
    scope: APIKeyScope
    owner_id: str
    owner_email: str
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_count: int = 0
    rate_limit_reset: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if self.status != APIKeyStatus.ACTIVE:
            return False
        
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def is_rate_limited(self) -> bool:
        """Check if API key is rate limited"""
        if datetime.utcnow() > self.rate_limit_reset:
            # Reset rate limit counter
            self.rate_limit_count = 0
            self.rate_limit_reset = datetime.utcnow() + timedelta(hours=1)
        
        return self.rate_limit_count >= self.scope.rate_limit
    
    def increment_usage(self):
        """Increment usage counters"""
        self.usage_count += 1
        self.rate_limit_count += 1
        self.last_used_at = datetime.utcnow()

class APIKeyRequest(BaseModel):
    """API Key creation request"""
    key_name: str = Field(..., min_length=3, max_length=100)
    key_type: APIKeyType
    permissions: List[Permission]
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    rate_limit: int = Field(1000, ge=1, le=10000)
    ip_whitelist: Optional[List[str]] = None
    allowed_endpoints: Optional[List[str]] = None
    data_access_level: str = Field("read", regex="^(read|write|admin)$")
    metadata: Optional[Dict[str, Any]] = None

class APIKeyResponse(BaseModel):
    """API Key response model"""
    key_id: str
    key_name: str
    key_type: APIKeyType
    api_key: str  # Only returned once during creation
    permissions: List[Permission]
    rate_limit: int
    expires_at: Optional[datetime]
    created_at: datetime

class APIKeyManager:
    """API Key management system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.cipher_suite = self._init_encryption()
        self.key_prefix = "api_key:"
        self.usage_prefix = "api_usage:"
        
        # In-memory storage for development (replace with database)
        self.api_keys: Dict[str, APIKey] = {}
        self.key_hash_to_id: Dict[str, str] = {}
        
        logger.info("API Key Manager initialized")
    
    def _init_encryption(self) -> Fernet:
        """Initialize encryption for API key storage"""
        key = get_secret("api_key_encryption_key")
        if not key:
            key = Fernet.generate_key()
            logger.warning("API key encryption key generated - store securely")
        
        if isinstance(key, str):
            key = key.encode()
        
        return Fernet(key)
    
    def _generate_api_key(self) -> Tuple[str, str]:
        """Generate new API key and its hash"""
        # Generate secure random key
        key_bytes = secrets.token_bytes(32)
        api_key = f"gm_{base64.urlsafe_b64encode(key_bytes).decode().rstrip('=')}"
        
        # Create hash for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        return api_key, key_hash
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for lookup"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def create_api_key(self, 
                           request: APIKeyRequest,
                           owner_id: str,
                           owner_email: str) -> APIKeyResponse:
        """Create new API key"""
        try:
            # Generate API key
            api_key, key_hash = self._generate_api_key()
            key_id = str(secrets.token_urlsafe(16))
            
            # Set expiration
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
            
            # Create scope
            scope = APIKeyScope(
                key_type=request.key_type,
                permissions=request.permissions,
                rate_limit=request.rate_limit,
                ip_whitelist=request.ip_whitelist,
                allowed_endpoints=request.allowed_endpoints,
                data_access_level=request.data_access_level
            )
            
            # Create API key object
            api_key_obj = APIKey(
                key_id=key_id,
                key_name=request.key_name,
                key_hash=key_hash,
                key_type=request.key_type,
                scope=scope,
                owner_id=owner_id,
                owner_email=owner_email,
                expires_at=expires_at,
                metadata=request.metadata or {}
            )
            
            # Store in memory and Redis
            self.api_keys[key_id] = api_key_obj
            self.key_hash_to_id[key_hash] = key_id
            
            if self.redis:
                await self._store_api_key(api_key_obj)
            
            logger.info(
                "API key created",
                key_id=key_id,
                key_name=request.key_name,
                key_type=request.key_type,
                owner_id=owner_id
            )
            
            return APIKeyResponse(
                key_id=key_id,
                key_name=request.key_name,
                key_type=request.key_type,
                api_key=api_key,  # Only returned once
                permissions=request.permissions,
                rate_limit=request.rate_limit,
                expires_at=expires_at,
                created_at=api_key_obj.created_at
            )
            
        except Exception as e:
            logger.error("Error creating API key", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to create API key")
    
    async def _store_api_key(self, api_key: APIKey):
        """Store API key in Redis"""
        key_data = {
            "key_id": api_key.key_id,
            "key_name": api_key.key_name,
            "key_hash": api_key.key_hash,
            "key_type": api_key.key_type.value,
            "scope": {
                "key_type": api_key.scope.key_type.value,
                "permissions": [p.value for p in api_key.scope.permissions],
                "rate_limit": api_key.scope.rate_limit,
                "ip_whitelist": api_key.scope.ip_whitelist,
                "allowed_endpoints": api_key.scope.allowed_endpoints,
                "data_access_level": api_key.scope.data_access_level
            },
            "owner_id": api_key.owner_id,
            "owner_email": api_key.owner_email,
            "status": api_key.status.value,
            "created_at": api_key.created_at.isoformat(),
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            "usage_count": api_key.usage_count,
            "rate_limit_count": api_key.rate_limit_count,
            "rate_limit_reset": api_key.rate_limit_reset.isoformat(),
            "metadata": api_key.metadata
        }
        
        # Encrypt and store
        encrypted_data = self.cipher_suite.encrypt(json.dumps(key_data).encode())
        
        await self.redis.set(f"{self.key_prefix}{api_key.key_id}", encrypted_data)
        await self.redis.set(f"{self.key_prefix}hash:{api_key.key_hash}", api_key.key_id)
        
        # Set expiration if specified
        if api_key.expires_at:
            expire_seconds = int((api_key.expires_at - datetime.utcnow()).total_seconds())
            await self.redis.expire(f"{self.key_prefix}{api_key.key_id}", expire_seconds)
    
    async def validate_api_key(self, 
                             api_key: str,
                             request: Request) -> Optional[APIKey]:
        """Validate API key and return key object if valid"""
        try:
            key_hash = self._hash_api_key(api_key)
            
            # Find key ID
            key_id = self.key_hash_to_id.get(key_hash)
            if not key_id and self.redis:
                key_id = await self.redis.get(f"{self.key_prefix}hash:{key_hash}")
            
            if not key_id:
                logger.warning("API key not found", key_hash=key_hash[:8])
                return None
            
            # Get API key object
            api_key_obj = self.api_keys.get(key_id)
            if not api_key_obj and self.redis:
                api_key_obj = await self._load_api_key(key_id)
            
            if not api_key_obj:
                logger.warning("API key object not found", key_id=key_id)
                return None
            
            # Validate key
            if not api_key_obj.is_valid():
                logger.warning("Invalid API key", key_id=key_id, status=api_key_obj.status)
                return None
            
            # Check rate limit
            if api_key_obj.is_rate_limited():
                logger.warning("API key rate limited", key_id=key_id)
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"X-RateLimit-Reset": str(int(api_key_obj.rate_limit_reset.timestamp()))}
                )
            
            # Check IP whitelist
            if api_key_obj.scope.ip_whitelist:
                client_ip = request.client.host if request.client else "unknown"
                if client_ip not in api_key_obj.scope.ip_whitelist:
                    logger.warning("IP not whitelisted", key_id=key_id, client_ip=client_ip)
                    raise HTTPException(status_code=403, detail="IP not authorized")
            
            # Check allowed endpoints
            if api_key_obj.scope.allowed_endpoints:
                path = request.url.path
                if not any(path.startswith(endpoint) for endpoint in api_key_obj.scope.allowed_endpoints):
                    logger.warning("Endpoint not allowed", key_id=key_id, path=path)
                    raise HTTPException(status_code=403, detail="Endpoint not authorized")
            
            # Update usage
            api_key_obj.increment_usage()
            
            # Store updated usage
            if self.redis:
                await self._update_api_key_usage(api_key_obj)
            
            logger.info(
                "API key validated",
                key_id=key_id,
                key_type=api_key_obj.key_type,
                usage_count=api_key_obj.usage_count
            )
            
            return api_key_obj
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error validating API key", error=str(e))
            return None
    
    async def _load_api_key(self, key_id: str) -> Optional[APIKey]:
        """Load API key from Redis"""
        try:
            encrypted_data = await self.redis.get(f"{self.key_prefix}{key_id}")
            if not encrypted_data:
                return None
            
            # Decrypt and parse
            decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
            key_data = json.loads(decrypted_data.decode())
            
            # Reconstruct APIKey object
            scope = APIKeyScope(
                key_type=APIKeyType(key_data["scope"]["key_type"]),
                permissions=[Permission(p) for p in key_data["scope"]["permissions"]],
                rate_limit=key_data["scope"]["rate_limit"],
                ip_whitelist=key_data["scope"]["ip_whitelist"],
                allowed_endpoints=key_data["scope"]["allowed_endpoints"],
                data_access_level=key_data["scope"]["data_access_level"]
            )
            
            api_key = APIKey(
                key_id=key_data["key_id"],
                key_name=key_data["key_name"],
                key_hash=key_data["key_hash"],
                key_type=APIKeyType(key_data["key_type"]),
                scope=scope,
                owner_id=key_data["owner_id"],
                owner_email=key_data["owner_email"],
                status=APIKeyStatus(key_data["status"]),
                created_at=datetime.fromisoformat(key_data["created_at"]),
                expires_at=datetime.fromisoformat(key_data["expires_at"]) if key_data["expires_at"] else None,
                last_used_at=datetime.fromisoformat(key_data["last_used_at"]) if key_data["last_used_at"] else None,
                usage_count=key_data["usage_count"],
                rate_limit_count=key_data["rate_limit_count"],
                rate_limit_reset=datetime.fromisoformat(key_data["rate_limit_reset"]),
                metadata=key_data["metadata"]
            )
            
            # Cache in memory
            self.api_keys[key_id] = api_key
            self.key_hash_to_id[api_key.key_hash] = key_id
            
            return api_key
            
        except Exception as e:
            logger.error("Error loading API key", key_id=key_id, error=str(e))
            return None
    
    async def _update_api_key_usage(self, api_key: APIKey):
        """Update API key usage in Redis"""
        try:
            # Update usage statistics
            usage_key = f"{self.usage_prefix}{api_key.key_id}"
            usage_data = {
                "last_used_at": api_key.last_used_at.isoformat(),
                "usage_count": api_key.usage_count,
                "rate_limit_count": api_key.rate_limit_count,
                "rate_limit_reset": api_key.rate_limit_reset.isoformat()
            }
            
            await self.redis.set(usage_key, json.dumps(usage_data))
            
        except Exception as e:
            logger.error("Error updating API key usage", key_id=api_key.key_id, error=str(e))
    
    async def revoke_api_key(self, key_id: str, revoked_by: str) -> bool:
        """Revoke API key"""
        try:
            api_key = self.api_keys.get(key_id)
            if not api_key and self.redis:
                api_key = await self._load_api_key(key_id)
            
            if not api_key:
                logger.warning("API key not found for revocation", key_id=key_id)
                return False
            
            # Update status
            api_key.status = APIKeyStatus.REVOKED
            api_key.metadata["revoked_by"] = revoked_by
            api_key.metadata["revoked_at"] = datetime.utcnow().isoformat()
            
            # Update storage
            if self.redis:
                await self._store_api_key(api_key)
            
            logger.info(
                "API key revoked",
                key_id=key_id,
                revoked_by=revoked_by
            )
            
            return True
            
        except Exception as e:
            logger.error("Error revoking API key", key_id=key_id, error=str(e))
            return False
    
    async def list_api_keys(self, owner_id: str) -> List[Dict[str, Any]]:
        """List API keys for owner"""
        try:
            api_keys = []
            
            for api_key in self.api_keys.values():
                if api_key.owner_id == owner_id:
                    api_keys.append({
                        "key_id": api_key.key_id,
                        "key_name": api_key.key_name,
                        "key_type": api_key.key_type.value,
                        "status": api_key.status.value,
                        "permissions": [p.value for p in api_key.scope.permissions],
                        "rate_limit": api_key.scope.rate_limit,
                        "usage_count": api_key.usage_count,
                        "created_at": api_key.created_at.isoformat(),
                        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                        "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None
                    })
            
            return api_keys
            
        except Exception as e:
            logger.error("Error listing API keys", owner_id=owner_id, error=str(e))
            return []
    
    async def get_api_key_stats(self, key_id: str) -> Dict[str, Any]:
        """Get API key usage statistics"""
        try:
            api_key = self.api_keys.get(key_id)
            if not api_key and self.redis:
                api_key = await self._load_api_key(key_id)
            
            if not api_key:
                return {}
            
            return {
                "key_id": api_key.key_id,
                "key_name": api_key.key_name,
                "usage_count": api_key.usage_count,
                "rate_limit_count": api_key.rate_limit_count,
                "rate_limit_reset": api_key.rate_limit_reset.isoformat(),
                "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                "status": api_key.status.value,
                "created_at": api_key.created_at.isoformat(),
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None
            }
            
        except Exception as e:
            logger.error("Error getting API key stats", key_id=key_id, error=str(e))
            return {}

# Global API key manager instance
api_key_manager: Optional[APIKeyManager] = None

async def get_api_key_manager() -> APIKeyManager:
    """Get or create API key manager instance"""
    global api_key_manager
    
    if api_key_manager is None:
        # Initialize Redis client
        redis_client = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                redis_client = await redis.from_url(redis_url)
            except Exception as e:
                logger.error("Failed to connect to Redis", error=str(e))
        
        api_key_manager = APIKeyManager(redis_client)
    
    return api_key_manager

# FastAPI Security for API Key authentication
api_key_header = HTTPBearer(scheme_name="API Key")

async def validate_api_key_dependency(request: Request, 
                                     credentials: HTTPAuthorizationCredentials = Security(api_key_header)) -> APIKey:
    """FastAPI dependency for API key validation"""
    manager = await get_api_key_manager()
    api_key_obj = await manager.validate_api_key(credentials.credentials, request)
    
    if not api_key_obj:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return api_key_obj

def require_api_key_permission(permission: Permission):
    """Dependency to require specific permission for API key"""
    async def permission_checker(api_key: APIKey = Depends(validate_api_key_dependency)) -> APIKey:
        if permission not in api_key.scope.permissions:
            logger.warning(
                "API key permission denied",
                key_id=api_key.key_id,
                required_permission=permission,
                key_permissions=api_key.scope.permissions
            )
            raise HTTPException(
                status_code=403,
                detail=f"API key missing required permission: {permission.value}"
            )
        
        return api_key
    
    return permission_checker

def require_api_key_type(key_type: APIKeyType):
    """Dependency to require specific API key type"""
    async def type_checker(api_key: APIKey = Depends(validate_api_key_dependency)) -> APIKey:
        if api_key.key_type != key_type:
            logger.warning(
                "API key type mismatch",
                key_id=api_key.key_id,
                required_type=key_type,
                key_type=api_key.key_type
            )
            raise HTTPException(
                status_code=403,
                detail=f"API key type mismatch. Required: {key_type.value}"
            )
        
        return api_key
    
    return type_checker

# Common API key dependencies
require_trading_api_key = require_api_key_type(APIKeyType.TRADING)
require_analytics_api_key = require_api_key_type(APIKeyType.ANALYTICS)
require_admin_api_key = require_api_key_type(APIKeyType.ADMIN)

# Permission-based API key dependencies
require_api_trade_execute = require_api_key_permission(Permission.TRADE_EXECUTE)
require_api_risk_configure = require_api_key_permission(Permission.RISK_CONFIGURE)
require_api_data_export = require_api_key_permission(Permission.DATA_EXPORT)
