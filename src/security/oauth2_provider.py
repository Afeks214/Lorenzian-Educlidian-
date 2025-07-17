"""
Enterprise OAuth2/OpenID Connect Authentication Provider
Implements OAuth2 Authorization Server and OpenID Connect Identity Provider
for world-class security and enterprise SSO integration.
"""

import os
import json
import base64
import secrets
import hashlib
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlparse, parse_qs

import jwt
import bcrypt
import qrcode
import pyotp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
from authlib.integrations.fastapi_oauth2 import AuthorizationServer
from authlib.oauth2 import OAuth2Error
from authlib.oauth2.rfc6749 import grants
from authlib.oauth2.rfc7636 import CodeChallenge
from authlib.oidc.core import UserInfo
from authlib.oidc.core.grants import OpenIDCode

from src.monitoring.logger_config import get_logger
from src.api.authentication import UserRole, RolePermission, ROLE_PERMISSIONS

logger = get_logger(__name__)

# OAuth2 Configuration
OAUTH2_JWT_ALGORITHM = "RS256"
OAUTH2_ACCESS_TOKEN_EXPIRE_SECONDS = int(os.getenv("OAUTH2_ACCESS_TOKEN_EXPIRE_SECONDS", "3600"))
OAUTH2_REFRESH_TOKEN_EXPIRE_SECONDS = int(os.getenv("OAUTH2_REFRESH_TOKEN_EXPIRE_SECONDS", "2592000"))  # 30 days
OAUTH2_AUTHORIZATION_CODE_EXPIRE_SECONDS = 600  # 10 minutes
OAUTH2_PKCE_REQUIRED = True  # Require PKCE for security

# OpenID Connect Configuration
OIDC_ISSUER = os.getenv("OIDC_ISSUER", "https://auth.grandmodel.com")
OIDC_SUPPORTED_SCOPES = [
    "openid", "profile", "email", "roles", "permissions", 
    "trading", "risk_management", "compliance", "audit"
]
OIDC_SUPPORTED_RESPONSE_TYPES = ["code", "token", "id_token", "code token", "code id_token", "token id_token", "code token id_token"]
OIDC_SUPPORTED_GRANT_TYPES = ["authorization_code", "refresh_token", "client_credentials", "password"]

class GrantType(str, Enum):
    """OAuth2 Grant Types"""
    AUTHORIZATION_CODE = "authorization_code"
    REFRESH_TOKEN = "refresh_token"
    CLIENT_CREDENTIALS = "client_credentials"
    PASSWORD = "password"
    IMPLICIT = "implicit"

class ResponseType(str, Enum):
    """OAuth2 Response Types"""
    CODE = "code"
    TOKEN = "token"
    ID_TOKEN = "id_token"

class ClientType(str, Enum):
    """OAuth2 Client Types"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"

class TokenType(str, Enum):
    """Token Types"""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    AUTHORIZATION_CODE = "authorization_code"
    ID_TOKEN = "id_token"

@dataclass
class OAuth2Client:
    """OAuth2 Client Configuration"""
    client_id: str
    client_secret: Optional[str] = None
    client_name: str = ""
    client_type: ClientType = ClientType.CONFIDENTIAL
    redirect_uris: List[str] = field(default_factory=list)
    allowed_scopes: List[str] = field(default_factory=list)
    allowed_grant_types: List[GrantType] = field(default_factory=list)
    allowed_response_types: List[ResponseType] = field(default_factory=list)
    require_pkce: bool = True
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OAuth2Token:
    """OAuth2 Token"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = OAUTH2_ACCESS_TOKEN_EXPIRE_SECONDS
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    id_token: Optional[str] = None
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.expires_at is None:
            self.expires_at = self.issued_at + timedelta(seconds=self.expires_in)

@dataclass
class AuthorizationCode:
    """OAuth2 Authorization Code"""
    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scope: str
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None
    nonce: Optional[str] = None
    state: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(seconds=OAUTH2_AUTHORIZATION_CODE_EXPIRE_SECONDS))
    is_used: bool = False

class MFAProvider:
    """Multi-Factor Authentication Provider"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.issuer_name = "GrandModel Trading"
    
    async def generate_totp_secret(self, user_id: str) -> str:
        """Generate TOTP secret for user"""
        secret = pyotp.random_base32()
        
        # Store secret in Redis with encryption
        await self.redis.setex(
            f"mfa_secret:{user_id}",
            86400 * 30,  # 30 days
            self._encrypt_secret(secret)
        )
        
        return secret
    
    async def generate_qr_code(self, user_id: str, username: str, secret: str) -> str:
        """Generate QR code for TOTP setup"""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username,
            issuer_name=self.issuer_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        # Return base64 encoded QR code image
        import io
        from PIL import Image
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    async def verify_totp(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        try:
            encrypted_secret = await self.redis.get(f"mfa_secret:{user_id}")
            if not encrypted_secret:
                return False
            
            secret = self._decrypt_secret(encrypted_secret)
            totp = pyotp.TOTP(secret)
            
            # Verify with window of 1 (30 seconds before/after)
            return totp.verify(token, valid_window=1)
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False
    
    async def generate_backup_codes(self, user_id: str) -> List[str]:
        """Generate backup codes for MFA"""
        codes = [secrets.token_hex(4) for _ in range(10)]
        
        # Hash and store backup codes
        hashed_codes = [hashlib.sha256(code.encode()).hexdigest() for code in codes]
        await self.redis.setex(
            f"mfa_backup:{user_id}",
            86400 * 365,  # 1 year
            json.dumps(hashed_codes)
        )
        
        return codes
    
    async def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify backup code and invalidate it"""
        try:
            backup_codes_json = await self.redis.get(f"mfa_backup:{user_id}")
            if not backup_codes_json:
                return False
            
            backup_codes = json.loads(backup_codes_json)
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            
            if code_hash in backup_codes:
                # Remove used code
                backup_codes.remove(code_hash)
                await self.redis.setex(
                    f"mfa_backup:{user_id}",
                    86400 * 365,
                    json.dumps(backup_codes)
                )
                return True
            
            return False
        except Exception as e:
            logger.error(f"Backup code verification failed: {e}")
            return False
    
    def _encrypt_secret(self, secret: str) -> str:
        """Encrypt MFA secret"""
        # Use application-specific key for encryption
        key = os.getenv("MFA_ENCRYPTION_KEY", Fernet.generate_key()).encode()
        if len(key) != 32:
            key = hashlib.sha256(key).digest()
        
        fernet = Fernet(base64.urlsafe_b64encode(key))
        return fernet.encrypt(secret.encode()).decode()
    
    def _decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt MFA secret"""
        key = os.getenv("MFA_ENCRYPTION_KEY", Fernet.generate_key()).encode()
        if len(key) != 32:
            key = hashlib.sha256(key).digest()
        
        fernet = Fernet(base64.urlsafe_b64encode(key))
        return fernet.decrypt(encrypted_secret.encode()).decode()

class JWKSProvider:
    """JSON Web Key Set Provider"""
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.kid = str(uuid.uuid4())
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair for JWT signing"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
    
    def get_private_key_pem(self) -> str:
        """Get private key in PEM format"""
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
    
    def get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set"""
        public_numbers = self.public_key.public_numbers()
        
        def int_to_base64url(value: int) -> str:
            """Convert integer to base64url string"""
            byte_length = (value.bit_length() + 7) // 8
            return base64.urlsafe_b64encode(
                value.to_bytes(byte_length, 'big')
            ).decode().rstrip('=')
        
        return {
            "keys": [
                {
                    "kty": "RSA",
                    "use": "sig",
                    "kid": self.kid,
                    "alg": "RS256",
                    "n": int_to_base64url(public_numbers.n),
                    "e": int_to_base64url(public_numbers.e)
                }
            ]
        }

class OAuth2Provider:
    """OAuth2 Authorization Server and OpenID Connect Provider"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.mfa_provider = MFAProvider(redis_client)
        self.jwks_provider = JWKSProvider()
        self.clients: Dict[str, OAuth2Client] = {}
        self._load_default_clients()
    
    def _load_default_clients(self):
        """Load default OAuth2 clients"""
        # Web Application Client
        web_client = OAuth2Client(
            client_id="grandmodel-web",
            client_secret=hashlib.sha256(os.getenv("WEB_CLIENT_SECRET", secrets.token_urlsafe(32)).encode()).hexdigest(),
            client_name="GrandModel Web Application",
            client_type=ClientType.CONFIDENTIAL,
            redirect_uris=["https://app.grandmodel.com/auth/callback"],
            allowed_scopes=["openid", "profile", "email", "roles", "trading", "risk_management"],
            allowed_grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN],
            allowed_response_types=[ResponseType.CODE],
            require_pkce=True
        )
        
        # Mobile Application Client
        mobile_client = OAuth2Client(
            client_id="grandmodel-mobile",
            client_name="GrandModel Mobile App",
            client_type=ClientType.PUBLIC,
            redirect_uris=["com.grandmodel.app://auth/callback"],
            allowed_scopes=["openid", "profile", "email", "roles", "trading"],
            allowed_grant_types=[GrantType.AUTHORIZATION_CODE, GrantType.REFRESH_TOKEN],
            allowed_response_types=[ResponseType.CODE],
            require_pkce=True
        )
        
        # Service-to-Service Client
        service_client = OAuth2Client(
            client_id="grandmodel-service",
            client_secret=hashlib.sha256(os.getenv("SERVICE_CLIENT_SECRET", secrets.token_urlsafe(32)).encode()).hexdigest(),
            client_name="GrandModel Service Account",
            client_type=ClientType.CONFIDENTIAL,
            allowed_scopes=["trading", "risk_management", "compliance", "audit"],
            allowed_grant_types=[GrantType.CLIENT_CREDENTIALS],
            allowed_response_types=[],
            require_pkce=False
        )
        
        self.clients = {
            web_client.client_id: web_client,
            mobile_client.client_id: mobile_client,
            service_client.client_id: service_client
        }
    
    async def get_client(self, client_id: str) -> Optional[OAuth2Client]:
        """Get OAuth2 client by ID"""
        return self.clients.get(client_id)
    
    async def authenticate_client(self, client_id: str, client_secret: str) -> Optional[OAuth2Client]:
        """Authenticate OAuth2 client"""
        client = await self.get_client(client_id)
        if not client or not client.is_active:
            return None
        
        if client.client_type == ClientType.PUBLIC:
            return client
        
        if client.client_secret and client.client_secret == client_secret:
            client.last_used_at = datetime.utcnow()
            return client
        
        return None
    
    async def create_authorization_code(
        self,
        client_id: str,
        user_id: str,
        redirect_uri: str,
        scope: str,
        code_challenge: Optional[str] = None,
        code_challenge_method: Optional[str] = None,
        nonce: Optional[str] = None,
        state: Optional[str] = None
    ) -> str:
        """Create authorization code"""
        code = secrets.token_urlsafe(32)
        
        auth_code = AuthorizationCode(
            code=code,
            client_id=client_id,
            user_id=user_id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            nonce=nonce,
            state=state
        )
        
        # Store in Redis
        await self.redis.setex(
            f"auth_code:{code}",
            OAUTH2_AUTHORIZATION_CODE_EXPIRE_SECONDS,
            json.dumps(auth_code.__dict__, default=str)
        )
        
        return code
    
    async def get_authorization_code(self, code: str) -> Optional[AuthorizationCode]:
        """Get authorization code"""
        try:
            code_data = await self.redis.get(f"auth_code:{code}")
            if not code_data:
                return None
            
            data = json.loads(code_data)
            # Convert datetime strings back to datetime objects
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
            
            return AuthorizationCode(**data)
        except Exception as e:
            logger.error(f"Error getting authorization code: {e}")
            return None
    
    async def exchange_code_for_token(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None
    ) -> Optional[OAuth2Token]:
        """Exchange authorization code for access token"""
        auth_code = await self.get_authorization_code(code)
        if not auth_code or auth_code.is_used:
            return None
        
        if auth_code.client_id != client_id:
            return None
        
        if auth_code.redirect_uri != redirect_uri:
            return None
        
        if auth_code.expires_at < datetime.utcnow():
            return None
        
        # Verify PKCE challenge
        if auth_code.code_challenge and code_verifier:
            if not self._verify_pkce_challenge(auth_code.code_challenge, code_verifier, auth_code.code_challenge_method):
                return None
        
        # Mark code as used
        auth_code.is_used = True
        await self.redis.setex(
            f"auth_code:{code}",
            OAUTH2_AUTHORIZATION_CODE_EXPIRE_SECONDS,
            json.dumps(auth_code.__dict__, default=str)
        )
        
        # Create tokens
        return await self._create_tokens(auth_code.user_id, client_id, auth_code.scope, auth_code.nonce)
    
    async def _create_tokens(self, user_id: str, client_id: str, scope: str, nonce: Optional[str] = None) -> OAuth2Token:
        """Create access and refresh tokens"""
        now = datetime.utcnow()
        
        # Create access token
        access_token_payload = {
            "iss": OIDC_ISSUER,
            "sub": user_id,
            "aud": client_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=OAUTH2_ACCESS_TOKEN_EXPIRE_SECONDS)).timestamp()),
            "scope": scope,
            "client_id": client_id,
            "jti": str(uuid.uuid4())
        }
        
        access_token = jwt.encode(
            access_token_payload,
            self.jwks_provider.get_private_key_pem(),
            algorithm=OAUTH2_JWT_ALGORITHM,
            headers={"kid": self.jwks_provider.kid}
        )
        
        # Create refresh token
        refresh_token_payload = {
            "iss": OIDC_ISSUER,
            "sub": user_id,
            "aud": client_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=OAUTH2_REFRESH_TOKEN_EXPIRE_SECONDS)).timestamp()),
            "scope": scope,
            "client_id": client_id,
            "jti": str(uuid.uuid4()),
            "type": "refresh"
        }
        
        refresh_token = jwt.encode(
            refresh_token_payload,
            self.jwks_provider.get_private_key_pem(),
            algorithm=OAUTH2_JWT_ALGORITHM,
            headers={"kid": self.jwks_provider.kid}
        )
        
        # Create ID token if OpenID scope is requested
        id_token = None
        if "openid" in scope:
            id_token = await self._create_id_token(user_id, client_id, nonce)
        
        # Store tokens in Redis for revocation
        await self.redis.setex(
            f"access_token:{access_token_payload['jti']}",
            OAUTH2_ACCESS_TOKEN_EXPIRE_SECONDS,
            json.dumps(access_token_payload)
        )
        
        await self.redis.setex(
            f"refresh_token:{refresh_token_payload['jti']}",
            OAUTH2_REFRESH_TOKEN_EXPIRE_SECONDS,
            json.dumps(refresh_token_payload)
        )
        
        return OAuth2Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=OAUTH2_ACCESS_TOKEN_EXPIRE_SECONDS,
            scope=scope,
            id_token=id_token
        )
    
    async def _create_id_token(self, user_id: str, client_id: str, nonce: Optional[str] = None) -> str:
        """Create OpenID Connect ID token"""
        # Get user information (would typically come from database)
        user_info = await self._get_user_info(user_id)
        
        now = datetime.utcnow()
        id_token_payload = {
            "iss": OIDC_ISSUER,
            "sub": user_id,
            "aud": client_id,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=OAUTH2_ACCESS_TOKEN_EXPIRE_SECONDS)).timestamp()),
            "auth_time": int(now.timestamp()),
            "nonce": nonce,
            **user_info
        }
        
        return jwt.encode(
            id_token_payload,
            self.jwks_provider.get_private_key_pem(),
            algorithm=OAUTH2_JWT_ALGORITHM,
            headers={"kid": self.jwks_provider.kid}
        )
    
    async def _get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information for ID token"""
        # This would typically query a database
        # For now, return mock data
        return {
            "name": "John Doe",
            "email": "john.doe@grandmodel.com",
            "email_verified": True,
            "preferred_username": "john.doe",
            "roles": ["risk_manager"],
            "permissions": ["trading", "risk_management"]
        }
    
    def _verify_pkce_challenge(self, code_challenge: str, code_verifier: str, method: str) -> bool:
        """Verify PKCE code challenge"""
        if method == "S256":
            expected_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode()).digest()
            ).decode().rstrip('=')
            return expected_challenge == code_challenge
        elif method == "plain":
            return code_verifier == code_challenge
        return False
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke access or refresh token"""
        try:
            # Decode token to get JTI
            payload = jwt.decode(
                token,
                self.jwks_provider.get_public_key_pem(),
                algorithms=[OAUTH2_JWT_ALGORITHM]
            )
            
            jti = payload.get("jti")
            if not jti:
                return False
            
            # Add to revocation list
            await self.redis.setex(
                f"revoked_token:{jti}",
                OAUTH2_REFRESH_TOKEN_EXPIRE_SECONDS,  # Use longest expiration
                "revoked"
            )
            
            return True
        except Exception as e:
            logger.error(f"Error revoking token: {e}")
            return False
    
    async def is_token_revoked(self, token: str) -> bool:
        """Check if token is revoked"""
        try:
            payload = jwt.decode(
                token,
                self.jwks_provider.get_public_key_pem(),
                algorithms=[OAUTH2_JWT_ALGORITHM]
            )
            
            jti = payload.get("jti")
            if not jti:
                return False
            
            revoked = await self.redis.get(f"revoked_token:{jti}")
            return revoked is not None
        except Exception:
            return True  # If we can't decode, treat as revoked
    
    async def get_openid_configuration(self) -> Dict[str, Any]:
        """Get OpenID Connect configuration"""
        return {
            "issuer": OIDC_ISSUER,
            "authorization_endpoint": f"{OIDC_ISSUER}/oauth2/authorize",
            "token_endpoint": f"{OIDC_ISSUER}/oauth2/token",
            "userinfo_endpoint": f"{OIDC_ISSUER}/oauth2/userinfo",
            "jwks_uri": f"{OIDC_ISSUER}/.well-known/jwks.json",
            "end_session_endpoint": f"{OIDC_ISSUER}/oauth2/logout",
            "revocation_endpoint": f"{OIDC_ISSUER}/oauth2/revoke",
            "scopes_supported": OIDC_SUPPORTED_SCOPES,
            "response_types_supported": OIDC_SUPPORTED_RESPONSE_TYPES,
            "grant_types_supported": OIDC_SUPPORTED_GRANT_TYPES,
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
            "code_challenge_methods_supported": ["S256"],
            "claims_supported": [
                "sub", "iss", "aud", "exp", "iat", "auth_time", "nonce",
                "name", "email", "email_verified", "preferred_username",
                "roles", "permissions"
            ]
        }

# Global OAuth2 provider instance
oauth2_provider: Optional[OAuth2Provider] = None

async def init_oauth2_provider(redis_client: redis.Redis) -> OAuth2Provider:
    """Initialize OAuth2 provider"""
    global oauth2_provider
    if not oauth2_provider:
        oauth2_provider = OAuth2Provider(redis_client)
    return oauth2_provider

async def get_oauth2_provider() -> OAuth2Provider:
    """Get OAuth2 provider instance"""
    if not oauth2_provider:
        raise HTTPException(status_code=500, detail="OAuth2 provider not initialized")
    return oauth2_provider