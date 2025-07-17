"""
OAuth2/OpenID Connect API Endpoints
Implements enterprise-grade authentication endpoints for OAuth2 and OpenID Connect.
"""

import json
import secrets
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode, urlparse, parse_qs

from fastapi import APIRouter, HTTPException, Depends, Request, Response, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis

from src.security.oauth2_provider import (
    OAuth2Provider, OAuth2Token, MFAProvider, TokenType, GrantType, ResponseType,
    init_oauth2_provider, get_oauth2_provider, OIDC_SUPPORTED_SCOPES
)
from src.api.authentication import (
    UserInfo, UserRole, RolePermission, USERS_DB, authenticate_user, 
    verify_password, init_redis, audit_log
)
from src.monitoring.logger_config import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/oauth2", tags=["OAuth2/OpenID Connect"])

# Security
security = HTTPBearer()

class AuthorizationRequest(BaseModel):
    """OAuth2 Authorization Request"""
    response_type: str = Field(..., description="OAuth2 response type")
    client_id: str = Field(..., description="OAuth2 client ID")
    redirect_uri: str = Field(..., description="Redirect URI")
    scope: str = Field(..., description="Requested scopes")
    state: Optional[str] = Field(None, description="State parameter")
    nonce: Optional[str] = Field(None, description="Nonce for OpenID Connect")
    code_challenge: Optional[str] = Field(None, description="PKCE code challenge")
    code_challenge_method: Optional[str] = Field(None, description="PKCE code challenge method")
    
    @validator('response_type')
    def validate_response_type(cls, v):
        if v not in ['code', 'token', 'id_token']:
            raise ValueError('Invalid response type')
        return v
    
    @validator('code_challenge_method')
    def validate_code_challenge_method(cls, v):
        if v and v not in ['S256', 'plain']:
            raise ValueError('Invalid code challenge method')
        return v

class TokenRequest(BaseModel):
    """OAuth2 Token Request"""
    grant_type: str = Field(..., description="OAuth2 grant type")
    code: Optional[str] = Field(None, description="Authorization code")
    redirect_uri: Optional[str] = Field(None, description="Redirect URI")
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: Optional[str] = Field(None, description="OAuth2 client secret")
    code_verifier: Optional[str] = Field(None, description="PKCE code verifier")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    username: Optional[str] = Field(None, description="Username for password grant")
    password: Optional[str] = Field(None, description="Password for password grant")
    scope: Optional[str] = Field(None, description="Requested scopes")

class TokenResponse(BaseModel):
    """OAuth2 Token Response"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    id_token: Optional[str] = None

class TokenIntrospectionRequest(BaseModel):
    """OAuth2 Token Introspection Request"""
    token: str = Field(..., description="Token to introspect")
    token_type_hint: Optional[str] = Field(None, description="Token type hint")

class TokenIntrospectionResponse(BaseModel):
    """OAuth2 Token Introspection Response"""
    active: bool
    scope: Optional[str] = None
    client_id: Optional[str] = None
    username: Optional[str] = None
    exp: Optional[int] = None
    iat: Optional[int] = None
    sub: Optional[str] = None
    aud: Optional[str] = None
    iss: Optional[str] = None
    jti: Optional[str] = None

class RevocationRequest(BaseModel):
    """OAuth2 Token Revocation Request"""
    token: str = Field(..., description="Token to revoke")
    token_type_hint: Optional[str] = Field(None, description="Token type hint")

class MFASetupRequest(BaseModel):
    """MFA Setup Request"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class MFASetupResponse(BaseModel):
    """MFA Setup Response"""
    secret: str = Field(..., description="TOTP secret")
    qr_code: str = Field(..., description="QR code image (base64)")
    backup_codes: List[str] = Field(..., description="Backup codes")

class MFAVerifyRequest(BaseModel):
    """MFA Verification Request"""
    user_id: str = Field(..., description="User ID")
    token: str = Field(..., description="TOTP token or backup code")

class LoginForm(BaseModel):
    """Login form for authorization endpoint"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    mfa_token: Optional[str] = Field(None, description="MFA token")

# OpenID Connect Discovery
@router.get("/.well-known/openid_configuration")
async def openid_configuration():
    """OpenID Connect Discovery endpoint"""
    provider = await get_oauth2_provider()
    return await provider.get_openid_configuration()

@router.get("/.well-known/jwks.json")
async def jwks():
    """JSON Web Key Set endpoint"""
    provider = await get_oauth2_provider()
    return provider.jwks_provider.get_jwks()

# OAuth2 Authorization Endpoint
@router.get("/authorize")
async def authorize(
    request: Request,
    response_type: str = Query(..., description="OAuth2 response type"),
    client_id: str = Query(..., description="OAuth2 client ID"),
    redirect_uri: str = Query(..., description="Redirect URI"),
    scope: str = Query(..., description="Requested scopes"),
    state: Optional[str] = Query(None, description="State parameter"),
    nonce: Optional[str] = Query(None, description="Nonce for OpenID Connect"),
    code_challenge: Optional[str] = Query(None, description="PKCE code challenge"),
    code_challenge_method: Optional[str] = Query(None, description="PKCE code challenge method")
):
    """OAuth2 Authorization endpoint"""
    provider = await get_oauth2_provider()
    
    # Validate client
    client = await provider.get_client(client_id)
    if not client or not client.is_active:
        raise HTTPException(status_code=400, detail="Invalid client")
    
    # Validate redirect URI
    if redirect_uri not in client.redirect_uris:
        raise HTTPException(status_code=400, detail="Invalid redirect URI")
    
    # Validate response type
    if ResponseType(response_type) not in client.allowed_response_types:
        raise HTTPException(status_code=400, detail="Unsupported response type")
    
    # Validate scopes
    requested_scopes = set(scope.split())
    if not requested_scopes.issubset(set(client.allowed_scopes)):
        raise HTTPException(status_code=400, detail="Invalid scope")
    
    # Check if user is already authenticated
    user_id = request.session.get("user_id")
    if not user_id:
        # Show login form
        return await _show_login_form(request, {
            "response_type": response_type,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method
        })
    
    # User is authenticated, create authorization code
    code = await provider.create_authorization_code(
        client_id=client_id,
        user_id=user_id,
        redirect_uri=redirect_uri,
        scope=scope,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        nonce=nonce,
        state=state
    )
    
    # Redirect back to client with authorization code
    params = {"code": code}
    if state:
        params["state"] = state
    
    redirect_url = f"{redirect_uri}?{urlencode(params)}"
    return RedirectResponse(url=redirect_url, status_code=302)

@router.post("/authorize")
async def authorize_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    mfa_token: Optional[str] = Form(None),
    response_type: str = Form(...),
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    scope: str = Form(...),
    state: Optional[str] = Form(None),
    nonce: Optional[str] = Form(None),
    code_challenge: Optional[str] = Form(None),
    code_challenge_method: Optional[str] = Form(None)
):
    """OAuth2 Authorization endpoint (POST) - Handle login form submission"""
    provider = await get_oauth2_provider()
    
    # Authenticate user
    user_data = await authenticate_user(username, password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check MFA if required
    if user_data.get("mfa_enabled", False):
        if not mfa_token:
            raise HTTPException(status_code=401, detail="MFA token required")
        
        mfa_valid = await provider.mfa_provider.verify_totp(user_data["user_id"], mfa_token)
        if not mfa_valid:
            # Try backup code
            mfa_valid = await provider.mfa_provider.verify_backup_code(user_data["user_id"], mfa_token)
        
        if not mfa_valid:
            raise HTTPException(status_code=401, detail="Invalid MFA token")
    
    # Store user session
    request.session["user_id"] = user_data["user_id"]
    request.session["username"] = user_data["username"]
    
    # Audit log
    await audit_log(user_data["user_id"], "oauth2_login", "authorization", {
        "client_id": client_id,
        "scope": scope
    }, request)
    
    # Create authorization code
    code = await provider.create_authorization_code(
        client_id=client_id,
        user_id=user_data["user_id"],
        redirect_uri=redirect_uri,
        scope=scope,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        nonce=nonce,
        state=state
    )
    
    # Redirect back to client
    params = {"code": code}
    if state:
        params["state"] = state
    
    redirect_url = f"{redirect_uri}?{urlencode(params)}"
    return RedirectResponse(url=redirect_url, status_code=302)

# OAuth2 Token Endpoint
@router.post("/token", response_model=TokenResponse)
async def token(
    request: Request,
    grant_type: str = Form(...),
    code: Optional[str] = Form(None),
    redirect_uri: Optional[str] = Form(None),
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None),
    code_verifier: Optional[str] = Form(None),
    refresh_token: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    scope: Optional[str] = Form(None)
):
    """OAuth2 Token endpoint"""
    provider = await get_oauth2_provider()
    
    # Authenticate client
    client = await provider.authenticate_client(client_id, client_secret)
    if not client:
        raise HTTPException(status_code=401, detail="Invalid client")
    
    # Handle different grant types
    if grant_type == GrantType.AUTHORIZATION_CODE:
        return await _handle_authorization_code_grant(provider, client, code, redirect_uri, code_verifier)
    elif grant_type == GrantType.REFRESH_TOKEN:
        return await _handle_refresh_token_grant(provider, client, refresh_token)
    elif grant_type == GrantType.CLIENT_CREDENTIALS:
        return await _handle_client_credentials_grant(provider, client, scope)
    elif grant_type == GrantType.PASSWORD:
        return await _handle_password_grant(provider, client, username, password, scope, request)
    else:
        raise HTTPException(status_code=400, detail="Unsupported grant type")

# OAuth2 Token Introspection Endpoint
@router.post("/introspect", response_model=TokenIntrospectionResponse)
async def introspect(
    request: TokenIntrospectionRequest,
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None)
):
    """OAuth2 Token Introspection endpoint"""
    provider = await get_oauth2_provider()
    
    # Authenticate client
    client = await provider.authenticate_client(client_id, client_secret)
    if not client:
        raise HTTPException(status_code=401, detail="Invalid client")
    
    # Check if token is revoked
    if await provider.is_token_revoked(request.token):
        return TokenIntrospectionResponse(active=False)
    
    try:
        import jwt
        payload = jwt.decode(
            request.token,
            provider.jwks_provider.get_public_key_pem(),
            algorithms=["RS256"]
        )
        
        return TokenIntrospectionResponse(
            active=True,
            scope=payload.get("scope"),
            client_id=payload.get("client_id"),
            username=payload.get("preferred_username"),
            exp=payload.get("exp"),
            iat=payload.get("iat"),
            sub=payload.get("sub"),
            aud=payload.get("aud"),
            iss=payload.get("iss"),
            jti=payload.get("jti")
        )
    except Exception:
        return TokenIntrospectionResponse(active=False)

# OAuth2 Token Revocation Endpoint
@router.post("/revoke")
async def revoke(
    request: RevocationRequest,
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None)
):
    """OAuth2 Token Revocation endpoint"""
    provider = await get_oauth2_provider()
    
    # Authenticate client
    client = await provider.authenticate_client(client_id, client_secret)
    if not client:
        raise HTTPException(status_code=401, detail="Invalid client")
    
    # Revoke token
    await provider.revoke_token(request.token)
    
    return {"revoked": True}

# OpenID Connect UserInfo Endpoint
@router.get("/userinfo")
async def userinfo(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """OpenID Connect UserInfo endpoint"""
    provider = await get_oauth2_provider()
    
    # Verify token
    try:
        import jwt
        payload = jwt.decode(
            credentials.credentials,
            provider.jwks_provider.get_public_key_pem(),
            algorithms=["RS256"]
        )
        
        # Check if token is revoked
        if await provider.is_token_revoked(credentials.credentials):
            raise HTTPException(status_code=401, detail="Token revoked")
        
        # Get user info
        user_info = await provider._get_user_info(payload["sub"])
        return user_info
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

# MFA Setup Endpoint
@router.post("/mfa/setup", response_model=MFASetupResponse)
async def mfa_setup(request: MFASetupRequest):
    """Setup Multi-Factor Authentication"""
    provider = await get_oauth2_provider()
    
    # Authenticate user
    user_data = await authenticate_user(request.username, request.password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate TOTP secret
    secret = await provider.mfa_provider.generate_totp_secret(user_data["user_id"])
    
    # Generate QR code
    qr_code = await provider.mfa_provider.generate_qr_code(
        user_data["user_id"],
        user_data["username"],
        secret
    )
    
    # Generate backup codes
    backup_codes = await provider.mfa_provider.generate_backup_codes(user_data["user_id"])
    
    return MFASetupResponse(
        secret=secret,
        qr_code=qr_code,
        backup_codes=backup_codes
    )

@router.post("/mfa/verify")
async def mfa_verify(request: MFAVerifyRequest):
    """Verify MFA token"""
    provider = await get_oauth2_provider()
    
    # Verify TOTP
    totp_valid = await provider.mfa_provider.verify_totp(request.user_id, request.token)
    if totp_valid:
        return {"verified": True, "type": "totp"}
    
    # Try backup code
    backup_valid = await provider.mfa_provider.verify_backup_code(request.user_id, request.token)
    if backup_valid:
        return {"verified": True, "type": "backup_code"}
    
    return {"verified": False}

# Logout Endpoint
@router.post("/logout")
async def logout(request: Request):
    """OAuth2 Logout endpoint"""
    # Clear session
    request.session.clear()
    
    return {"logged_out": True}

# Helper functions
async def _show_login_form(request: Request, params: Dict[str, Any]) -> HTMLResponse:
    """Show login form for OAuth2 authorization"""
    form_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GrandModel Login</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f0f0f0; }}
            .login-container {{ max-width: 400px; margin: 100px auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .form-group {{ margin-bottom: 15px; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
            input {{ width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }}
            button {{ width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
            button:hover {{ background: #0056b3; }}
            .logo {{ text-align: center; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="logo">
                <h2>GrandModel</h2>
                <p>Enterprise Trading Platform</p>
            </div>
            <form method="post" action="/oauth2/authorize">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <div class="form-group">
                    <label for="mfa_token">MFA Token (if enabled):</label>
                    <input type="text" id="mfa_token" name="mfa_token" placeholder="6-digit code or backup code">
                </div>
                <input type="hidden" name="response_type" value="{params.get('response_type', '')}">
                <input type="hidden" name="client_id" value="{params.get('client_id', '')}">
                <input type="hidden" name="redirect_uri" value="{params.get('redirect_uri', '')}">
                <input type="hidden" name="scope" value="{params.get('scope', '')}">
                <input type="hidden" name="state" value="{params.get('state', '')}">
                <input type="hidden" name="nonce" value="{params.get('nonce', '')}">
                <input type="hidden" name="code_challenge" value="{params.get('code_challenge', '')}">
                <input type="hidden" name="code_challenge_method" value="{params.get('code_challenge_method', '')}">
                <button type="submit">Login</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=form_html)

async def _handle_authorization_code_grant(
    provider: OAuth2Provider,
    client,
    code: str,
    redirect_uri: str,
    code_verifier: Optional[str]
) -> TokenResponse:
    """Handle authorization code grant"""
    if not code or not redirect_uri:
        raise HTTPException(status_code=400, detail="Missing code or redirect_uri")
    
    # Exchange code for token
    token = await provider.exchange_code_for_token(code, client.client_id, redirect_uri, code_verifier)
    if not token:
        raise HTTPException(status_code=400, detail="Invalid authorization code")
    
    return TokenResponse(
        access_token=token.access_token,
        refresh_token=token.refresh_token,
        expires_in=token.expires_in,
        scope=token.scope,
        id_token=token.id_token
    )

async def _handle_refresh_token_grant(
    provider: OAuth2Provider,
    client,
    refresh_token: str
) -> TokenResponse:
    """Handle refresh token grant"""
    if not refresh_token:
        raise HTTPException(status_code=400, detail="Missing refresh_token")
    
    # Verify refresh token
    try:
        import jwt
        payload = jwt.decode(
            refresh_token,
            provider.jwks_provider.get_public_key_pem(),
            algorithms=["RS256"]
        )
        
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=400, detail="Invalid refresh token")
        
        if payload.get("client_id") != client.client_id:
            raise HTTPException(status_code=400, detail="Invalid client")
        
        # Check if token is revoked
        if await provider.is_token_revoked(refresh_token):
            raise HTTPException(status_code=400, detail="Token revoked")
        
        # Create new tokens
        new_token = await provider._create_tokens(
            payload["sub"],
            client.client_id,
            payload.get("scope", "")
        )
        
        return TokenResponse(
            access_token=new_token.access_token,
            refresh_token=new_token.refresh_token,
            expires_in=new_token.expires_in,
            scope=new_token.scope,
            id_token=new_token.id_token
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid refresh token")

async def _handle_client_credentials_grant(
    provider: OAuth2Provider,
    client,
    scope: Optional[str]
) -> TokenResponse:
    """Handle client credentials grant"""
    if GrantType.CLIENT_CREDENTIALS not in client.allowed_grant_types:
        raise HTTPException(status_code=400, detail="Grant type not allowed")
    
    # Use client ID as subject for service-to-service authentication
    token = await provider._create_tokens(client.client_id, client.client_id, scope or "")
    
    return TokenResponse(
        access_token=token.access_token,
        expires_in=token.expires_in,
        scope=token.scope,
        token_type="Bearer"
    )

async def _handle_password_grant(
    provider: OAuth2Provider,
    client,
    username: str,
    password: str,
    scope: Optional[str],
    request: Request
) -> TokenResponse:
    """Handle password grant (for trusted clients only)"""
    if GrantType.PASSWORD not in client.allowed_grant_types:
        raise HTTPException(status_code=400, detail="Grant type not allowed")
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Missing username or password")
    
    # Authenticate user
    user_data = await authenticate_user(username, password)
    if not user_data:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Audit log
    await audit_log(user_data["user_id"], "oauth2_password_grant", "token", {
        "client_id": client.client_id,
        "scope": scope
    }, request)
    
    # Create tokens
    token = await provider._create_tokens(user_data["user_id"], client.client_id, scope or "")
    
    return TokenResponse(
        access_token=token.access_token,
        refresh_token=token.refresh_token,
        expires_in=token.expires_in,
        scope=token.scope,
        id_token=token.id_token
    )

# Initialize OAuth2 provider on startup
@router.on_event("startup")
async def startup_event():
    """Initialize OAuth2 provider on startup"""
    await init_redis()
    redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    await init_oauth2_provider(redis_client)
    logger.info("OAuth2 provider initialized")