"""
Security Headers & CORS Policy Implementation
Comprehensive security headers for web application protection
"""

import os
import secrets
from typing import Dict, List, Optional, Any
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response as StarletteResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.monitoring.logger_config import get_logger
from src.security.secrets_manager import get_secret

logger = get_logger(__name__)

class SecurityHeaders:
    """Security headers configuration and implementation"""
    
    def __init__(self):
        self.nonce_cache = {}
        self.session_secret = self._get_session_secret()
        
    def _get_session_secret(self) -> str:
        """Get session secret key"""
        secret = get_secret("session_secret")
        if not secret:
            secret = secrets.token_urlsafe(32)
            logger.warning("Session secret generated - store securely in production")
        return secret
    
    def generate_nonce(self) -> str:
        """Generate cryptographically secure nonce"""
        return secrets.token_urlsafe(16)
    
    def get_security_headers(self, 
                           request: Request,
                           response: Response,
                           csp_nonce: Optional[str] = None) -> Dict[str, str]:
        """Generate comprehensive security headers"""
        
        # Generate nonce for CSP if not provided
        if csp_nonce is None:
            csp_nonce = self.generate_nonce()
        
        # Determine if HTTPS is being used
        is_https = request.url.scheme == "https"
        
        headers = {
            # Content Security Policy
            "Content-Security-Policy": self._get_csp_header(csp_nonce, is_https),
            
            # X-Frame-Options
            "X-Frame-Options": "DENY",
            
            # X-Content-Type-Options
            "X-Content-Type-Options": "nosniff",
            
            # X-XSS-Protection
            "X-XSS-Protection": "1; mode=block",
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions Policy
            "Permissions-Policy": self._get_permissions_policy(),
            
            # Cross-Origin Embedder Policy
            "Cross-Origin-Embedder-Policy": "require-corp",
            
            # Cross-Origin Opener Policy
            "Cross-Origin-Opener-Policy": "same-origin",
            
            # Cross-Origin Resource Policy
            "Cross-Origin-Resource-Policy": "same-origin",
            
            # Cache Control for sensitive content
            "Cache-Control": "no-cache, no-store, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0",
            
            # Server identity hiding
            "Server": "GrandModel/1.0",
            
            # Additional security headers
            "X-Permitted-Cross-Domain-Policies": "none",
            "X-Download-Options": "noopen",
        }
        
        # Add HTTPS-only headers
        if is_https:
            headers.update({
                # HTTP Strict Transport Security (HSTS)
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
                
                # Expect-CT
                "Expect-CT": "max-age=86400, enforce",
            })
        
        # Add session security headers
        if "session" in request.cookies:
            headers["Set-Cookie"] = self._get_secure_cookie_header(is_https)
        
        return headers
    
    def _get_csp_header(self, nonce: str, is_https: bool) -> str:
        """Generate Content Security Policy header"""
        # Base CSP directives
        csp_directives = {
            "default-src": "'self'",
            "script-src": f"'self' 'nonce-{nonce}' 'strict-dynamic'",
            "style-src": f"'self' 'nonce-{nonce}' 'unsafe-inline'",
            "img-src": "'self' data: https:",
            "font-src": "'self' https:",
            "connect-src": "'self' https:",
            "media-src": "'self'",
            "object-src": "'none'",
            "child-src": "'none'",
            "worker-src": "'self'",
            "manifest-src": "'self'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "frame-ancestors": "'none'",
            "block-all-mixed-content": "",
        }
        
        # Add upgrade-insecure-requests for HTTPS
        if is_https:
            csp_directives["upgrade-insecure-requests"] = ""
        
        # Environment-specific adjustments
        if os.getenv("ENVIRONMENT") == "development":
            # Allow webpack dev server in development
            csp_directives["script-src"] += " 'unsafe-eval'"
            csp_directives["connect-src"] += " ws: wss:"
        
        # Build CSP string
        csp_parts = []
        for directive, value in csp_directives.items():
            if value:
                csp_parts.append(f"{directive} {value}")
            else:
                csp_parts.append(directive)
        
        return "; ".join(csp_parts)
    
    def _get_permissions_policy(self) -> str:
        """Generate Permissions Policy header"""
        policies = {
            "accelerometer": "()",
            "ambient-light-sensor": "()",
            "autoplay": "()",
            "battery": "()",
            "camera": "()",
            "cross-origin-isolated": "()",
            "display-capture": "()",
            "document-domain": "()",
            "encrypted-media": "()",
            "execution-while-not-rendered": "()",
            "execution-while-out-of-viewport": "()",
            "fullscreen": "(self)",
            "geolocation": "()",
            "gyroscope": "()",
            "keyboard-map": "()",
            "magnetometer": "()",
            "microphone": "()",
            "midi": "()",
            "navigation-override": "()",
            "payment": "()",
            "picture-in-picture": "()",
            "publickey-credentials-get": "()",
            "screen-wake-lock": "()",
            "sync-xhr": "()",
            "usb": "()",
            "web-share": "()",
            "xr-spatial-tracking": "()"
        }
        
        return ", ".join([f"{key}={value}" for key, value in policies.items()])
    
    def _get_secure_cookie_header(self, is_https: bool) -> str:
        """Generate secure cookie header"""
        cookie_attrs = [
            "HttpOnly",
            "SameSite=Strict",
            "Path=/",
            "Max-Age=3600"  # 1 hour
        ]
        
        if is_https:
            cookie_attrs.append("Secure")
        
        return "; ".join(cookie_attrs)

class CORSPolicy:
    """CORS policy configuration"""
    
    def __init__(self):
        self.production_origins = self._get_production_origins()
        self.development_origins = self._get_development_origins()
        
    def _get_production_origins(self) -> List[str]:
        """Get production allowed origins"""
        origins = [
            "https://app.grandmodel.com",
            "https://dashboard.grandmodel.com",
            "https://api.grandmodel.com",
            "https://grandmodel.com",
            "https://www.grandmodel.com"
        ]
        
        # Add custom origins from environment
        custom_origins = os.getenv("CORS_ORIGINS", "")
        if custom_origins:
            origins.extend(custom_origins.split(","))
        
        return origins
    
    def _get_development_origins(self) -> List[str]:
        """Get development allowed origins"""
        return [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:8000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "http://0.0.0.0:3000",
            "http://0.0.0.0:8000"
        ]
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        environment = os.getenv("ENVIRONMENT", "production")
        
        if environment == "development":
            return {
                "allow_origins": self.development_origins,
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
                "allow_headers": [
                    "Accept",
                    "Accept-Language",
                    "Content-Language",
                    "Content-Type",
                    "Authorization",
                    "X-Requested-With",
                    "X-API-Key",
                    "X-Correlation-ID",
                    "X-Request-ID"
                ],
                "expose_headers": [
                    "X-Correlation-ID",
                    "X-Request-ID",
                    "X-RateLimit-Limit",
                    "X-RateLimit-Remaining",
                    "X-RateLimit-Reset"
                ],
                "max_age": 86400  # 24 hours
            }
        else:
            return {
                "allow_origins": self.production_origins,
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "allow_headers": [
                    "Accept",
                    "Content-Type",
                    "Authorization",
                    "X-API-Key",
                    "X-Correlation-ID"
                ],
                "expose_headers": [
                    "X-Correlation-ID",
                    "X-RateLimit-Limit",
                    "X-RateLimit-Remaining",
                    "X-RateLimit-Reset"
                ],
                "max_age": 3600  # 1 hour
            }

class SecurityMiddleware:
    """Security middleware implementation"""
    
    def __init__(self, app: ASGIApp):
        self.app = app
        self.security_headers = SecurityHeaders()
        self.cors_policy = CORSPolicy()
        
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        
        # Create request wrapper
        request = Request(scope, receive)
        
        # Process security headers
        async def send_wrapper(message: Any) -> None:
            if message["type"] == "http.response.start":
                # Generate nonce for this request
                nonce = self.security_headers.generate_nonce()
                
                # Create response wrapper
                response = Response()
                
                # Get security headers
                security_headers = self.security_headers.get_security_headers(
                    request, response, nonce
                )
                
                # Add security headers to response
                message["headers"] = message.get("headers", [])
                for name, value in security_headers.items():
                    message["headers"].append((name.encode(), value.encode()))
                
                # Store nonce in request state for use in templates
                if hasattr(request, "state"):
                    request.state.csp_nonce = nonce
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)

class TrustedHostsMiddleware:
    """Trusted hosts middleware for host header validation"""
    
    def __init__(self, app: ASGIApp, allowed_hosts: Optional[List[str]] = None):
        self.app = app
        self.allowed_hosts = allowed_hosts or self._get_default_allowed_hosts()
        
    def _get_default_allowed_hosts(self) -> List[str]:
        """Get default allowed hosts"""
        environment = os.getenv("ENVIRONMENT", "production")
        
        if environment == "development":
            return [
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
                "*.localhost",
                "*.local"
            ]
        else:
            return [
                "grandmodel.com",
                "*.grandmodel.com",
                "api.grandmodel.com",
                "app.grandmodel.com",
                "dashboard.grandmodel.com"
            ]
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Check host header
        headers = dict(scope.get("headers", []))
        host = headers.get(b"host", b"").decode()
        
        if not self._is_allowed_host(host):
            # Return 400 Bad Request for invalid host
            response = StarletteResponse(
                "Invalid host header",
                status_code=400,
                headers={"Content-Type": "text/plain"}
            )
            await response(scope, receive, send)
            return
        
        await self.app(scope, receive, send)
    
    def _is_allowed_host(self, host: str) -> bool:
        """Check if host is allowed"""
        if not host:
            return False
        
        # Remove port from host
        host = host.split(":")[0]
        
        for allowed_host in self.allowed_hosts:
            if allowed_host.startswith("*."):
                # Wildcard matching
                domain = allowed_host[2:]
                if host == domain or host.endswith(f".{domain}"):
                    return True
            elif host == allowed_host:
                return True
        
        return False

class RateLimitingMiddleware:
    """Rate limiting middleware"""
    
    def __init__(self, app: ASGIApp, rate_limiter):
        self.app = app
        self.rate_limiter = rate_limiter
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Apply rate limiting
        await self.rate_limiter.rate_limit_middleware(
            Request(scope, receive),
            lambda request: self.app(scope, receive, send)
        )

# Global instances
security_headers = SecurityHeaders()
cors_policy = CORSPolicy()

# Configuration functions
def get_cors_middleware() -> CORSMiddleware:
    """Get configured CORS middleware"""
    config = cors_policy.get_cors_config()
    return CORSMiddleware(**config)

def get_session_middleware() -> SessionMiddleware:
    """Get configured session middleware"""
    return SessionMiddleware(
        secret_key=security_headers.session_secret,
        max_age=3600,  # 1 hour
        https_only=os.getenv("ENVIRONMENT") == "production",
        same_site="strict"
    )

def get_trusted_host_middleware(app: ASGIApp) -> TrustedHostsMiddleware:
    """Get configured trusted host middleware"""
    return TrustedHostsMiddleware(app)

def get_security_middleware(app: ASGIApp) -> SecurityMiddleware:
    """Get configured security middleware"""
    return SecurityMiddleware(app)

# Utility functions
def add_security_headers(response: Response, request: Request) -> Response:
    """Add security headers to response"""
    headers = security_headers.get_security_headers(request, response)
    
    for name, value in headers.items():
        response.headers[name] = value
    
    return response

def generate_csp_nonce() -> str:
    """Generate CSP nonce"""
    return security_headers.generate_nonce()

def get_security_config() -> Dict[str, Any]:
    """Get security configuration summary"""
    return {
        "cors": cors_policy.get_cors_config(),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "https_only": os.getenv("ENVIRONMENT") == "production",
        "session_timeout": 3600,
        "csrf_protection": True,
        "xss_protection": True,
        "clickjacking_protection": True,
        "content_type_nosniff": True,
        "hsts_enabled": os.getenv("ENVIRONMENT") == "production",
        "referrer_policy": "strict-origin-when-cross-origin"
    }
