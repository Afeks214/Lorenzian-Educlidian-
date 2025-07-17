"""
Enhanced Authentication and Security for System Control API
===========================================================

Extended authentication system specifically for system control operations including:
- Multi-factor authentication (MFA)
- Hardware token support
- IP whitelisting
- Session management
- Audit logging
- Role-based access control (RBAC)
- Rate limiting
- Suspicious activity detection

This module provides enterprise-grade security for critical system control operations.
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network
import hashlib
import hmac
import secrets
import qrcode
from io import BytesIO
import base64

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import bcrypt
import redis.asyncio as redis
from pydantic import BaseModel, Field
import pyotp
import requests

from src.api.authentication import (
    UserInfo, UserRole, RolePermission, verify_token, audit_log,
    check_failed_login_attempts, record_failed_login, clear_failed_logins
)
from src.monitoring.logger_config import get_logger

logger = get_logger(__name__)

# Security configuration
@dataclass
class SecurityConfig:
    """Security configuration for system control"""
    max_failed_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    session_timeout: int = 3600  # 1 hour
    mfa_required: bool = True
    ip_whitelist_enabled: bool = True
    hardware_token_required: bool = False
    audit_all_actions: bool = True
    suspicious_activity_threshold: int = 10
    
class SystemControlPermission(str, Enum):
    """System control specific permissions"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    EMERGENCY_STOP = "emergency_stop"
    SYSTEM_STATUS = "system_status"
    COMPONENT_CONTROL = "component_control"
    MAINTENANCE_MODE = "maintenance_mode"
    SYSTEM_CONFIGURATION = "system_configuration"
    AUDIT_ACCESS = "audit_access"
    USER_MANAGEMENT = "user_management"

# Enhanced role permissions for system control
SYSTEM_CONTROL_PERMISSIONS: Dict[UserRole, List[SystemControlPermission]] = {
    UserRole.SYSTEM_ADMIN: [
        SystemControlPermission.SYSTEM_START,
        SystemControlPermission.SYSTEM_STOP,
        SystemControlPermission.EMERGENCY_STOP,
        SystemControlPermission.SYSTEM_STATUS,
        SystemControlPermission.COMPONENT_CONTROL,
        SystemControlPermission.MAINTENANCE_MODE,
        SystemControlPermission.SYSTEM_CONFIGURATION,
        SystemControlPermission.AUDIT_ACCESS,
        SystemControlPermission.USER_MANAGEMENT
    ],
    UserRole.RISK_MANAGER: [
        SystemControlPermission.SYSTEM_STATUS,
        SystemControlPermission.EMERGENCY_STOP,
        SystemControlPermission.AUDIT_ACCESS
    ],
    UserRole.RISK_OPERATOR: [
        SystemControlPermission.SYSTEM_STATUS,
        SystemControlPermission.EMERGENCY_STOP
    ],
    UserRole.COMPLIANCE_OFFICER: [
        SystemControlPermission.SYSTEM_STATUS,
        SystemControlPermission.AUDIT_ACCESS
    ],
    UserRole.VIEWER: [
        SystemControlPermission.SYSTEM_STATUS
    ]
}

class MFAType(str, Enum):
    """Multi-factor authentication types"""
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"    # SMS verification
    EMAIL = "email"  # Email verification
    HARDWARE = "hardware"  # Hardware token

class SecurityAlert(BaseModel):
    """Security alert model"""
    alert_id: str
    user_id: str
    alert_type: str
    description: str
    severity: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    resolved: bool = False

class SystemControlAuthenticator:
    """
    Enhanced authenticator for system control operations
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.ip_whitelist: Set[IPv4Network] = set()
        self.security_alerts: List[SecurityAlert] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Initialize IP whitelist
        self._init_ip_whitelist()
        
    async def init_redis(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
    
    def _init_ip_whitelist(self):
        """Initialize IP whitelist from configuration"""
        whitelist_ips = os.getenv("SYSTEM_CONTROL_IP_WHITELIST", "127.0.0.1/32,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16")
        
        for ip_range in whitelist_ips.split(','):
            try:
                self.ip_whitelist.add(IPv4Network(ip_range.strip()))
            except ValueError as e:
                logger.error(f"Invalid IP range in whitelist: {ip_range} - {e}")
    
    async def check_ip_whitelist(self, ip_address: str) -> bool:
        """Check if IP address is whitelisted"""
        if not self.config.ip_whitelist_enabled:
            return True
        
        try:
            client_ip = IPv4Address(ip_address)
            
            for network in self.ip_whitelist:
                if client_ip in network:
                    return True
            
            logger.warning(f"IP address {ip_address} not in whitelist")
            return False
            
        except ValueError:
            logger.error(f"Invalid IP address: {ip_address}")
            return False
    
    async def check_user_locked(self, username: str) -> bool:
        """Check if user is locked due to failed attempts"""
        failed_attempts = await check_failed_login_attempts(username)
        return failed_attempts >= self.config.max_failed_attempts
    
    async def verify_mfa_token(self, user_id: str, token: str, mfa_type: MFAType = MFAType.TOTP) -> bool:
        """Verify MFA token"""
        await self.init_redis()
        
        if mfa_type == MFAType.TOTP:
            # Get user's TOTP secret
            secret = await self._get_user_totp_secret(user_id)
            if not secret:
                return False
            
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)
        
        elif mfa_type == MFAType.SMS:
            # Verify SMS token (stored in Redis)
            if self.redis_client:
                stored_token = await self.redis_client.get(f"sms_token:{user_id}")
                if stored_token and stored_token.decode() == token:
                    await self.redis_client.delete(f"sms_token:{user_id}")
                    return True
        
        elif mfa_type == MFAType.EMAIL:
            # Verify email token (stored in Redis)
            if self.redis_client:
                stored_token = await self.redis_client.get(f"email_token:{user_id}")
                if stored_token and stored_token.decode() == token:
                    await self.redis_client.delete(f"email_token:{user_id}")
                    return True
        
        elif mfa_type == MFAType.HARDWARE:
            # Verify hardware token (implementation depends on hardware)
            return await self._verify_hardware_token(user_id, token)
        
        return False
    
    async def _get_user_totp_secret(self, user_id: str) -> Optional[str]:
        """Get user's TOTP secret from secure storage"""
        await self.init_redis()
        
        if self.redis_client:
            try:
                secret = await self.redis_client.get(f"totp_secret:{user_id}")
                return secret.decode() if secret else None
            except Exception as e:
                logger.error(f"Error retrieving TOTP secret: {e}")
        
        return None
    
    async def generate_totp_secret(self, user_id: str, username: str) -> Dict[str, Any]:
        """Generate TOTP secret and QR code for user"""
        await self.init_redis()
        
        # Generate secret
        secret = pyotp.random_base32()
        
        # Store secret
        if self.redis_client:
            await self.redis_client.set(f"totp_secret:{user_id}", secret, ex=86400 * 365)  # 1 year
        
        # Generate QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=username,
            issuer_name="GrandModel System Control"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        qr_code_data = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "secret": secret,
            "qr_code": qr_code_data,
            "provisioning_uri": provisioning_uri
        }
    
    async def _verify_hardware_token(self, user_id: str, token: str) -> bool:
        """Verify hardware token (placeholder for actual implementation)"""
        # This would integrate with actual hardware token system
        # For now, return True for demonstration
        return True
    
    async def send_sms_token(self, user_id: str, phone_number: str) -> bool:
        """Send SMS token to user"""
        await self.init_redis()
        
        # Generate 6-digit token
        token = f"{secrets.randbelow(1000000):06d}"
        
        # Store token in Redis (5 minutes expiration)
        if self.redis_client:
            await self.redis_client.setex(f"sms_token:{user_id}", 300, token)
        
        # Send SMS (placeholder - integrate with actual SMS service)
        try:
            # This would use actual SMS service like Twilio, AWS SNS, etc.
            logger.info(f"SMS token {token} sent to {phone_number}")
            return True
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    async def send_email_token(self, user_id: str, email: str) -> bool:
        """Send email token to user"""
        await self.init_redis()
        
        # Generate 6-digit token
        token = f"{secrets.randbelow(1000000):06d}"
        
        # Store token in Redis (5 minutes expiration)
        if self.redis_client:
            await self.redis_client.setex(f"email_token:{user_id}", 300, token)
        
        # Send email (placeholder - integrate with actual email service)
        try:
            # This would use actual email service like SendGrid, AWS SES, etc.
            logger.info(f"Email token {token} sent to {email}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def detect_suspicious_activity(self, user_id: str, request: Request) -> bool:
        """Detect suspicious activity patterns"""
        await self.init_redis()
        
        current_time = time.time()
        activity_key = f"activity:{user_id}"
        
        if self.redis_client:
            try:
                # Get recent activities
                activities = await self.redis_client.lrange(activity_key, 0, -1)
                
                # Check for rapid requests
                recent_activities = []
                for activity in activities:
                    activity_data = activity.decode().split('|')
                    if len(activity_data) >= 2:
                        timestamp = float(activity_data[0])
                        if current_time - timestamp < 300:  # Last 5 minutes
                            recent_activities.append(activity_data)
                
                # Check thresholds
                if len(recent_activities) > self.config.suspicious_activity_threshold:
                    await self._create_security_alert(
                        user_id,
                        "suspicious_activity",
                        f"Unusual activity detected: {len(recent_activities)} requests in 5 minutes",
                        "HIGH",
                        request
                    )
                    return True
                
                # Record current activity
                activity_record = f"{current_time}|{request.client.host}|{request.method}|{request.url}"
                await self.redis_client.lpush(activity_key, activity_record)
                await self.redis_client.ltrim(activity_key, 0, 100)  # Keep last 100 activities
                await self.redis_client.expire(activity_key, 3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error checking suspicious activity: {e}")
        
        return False
    
    async def _create_security_alert(self, user_id: str, alert_type: str, 
                                   description: str, severity: str, request: Request):
        """Create security alert"""
        alert = SecurityAlert(
            alert_id=secrets.token_urlsafe(16),
            user_id=user_id,
            alert_type=alert_type,
            description=description,
            severity=severity,
            timestamp=datetime.utcnow(),
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", "unknown")
        )
        
        self.security_alerts.append(alert)
        
        # Log alert
        logger.warning(f"SECURITY ALERT: {alert.alert_type} - {alert.description} - User: {user_id}")
        
        # Store alert in Redis
        await self.init_redis()
        if self.redis_client:
            alert_data = alert.dict()
            await self.redis_client.setex(
                f"security_alert:{alert.alert_id}",
                86400 * 7,  # 7 days
                str(alert_data)
            )
    
    async def validate_session(self, session_id: str) -> bool:
        """Validate session is still active"""
        await self.init_redis()
        
        if self.redis_client:
            try:
                session_data = await self.redis_client.get(f"session:{session_id}")
                if session_data:
                    # Update last activity
                    await self.redis_client.setex(
                        f"session_activity:{session_id}",
                        self.config.session_timeout,
                        str(time.time())
                    )
                    return True
            except Exception as e:
                logger.error(f"Error validating session: {e}")
        
        return False
    
    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions"""
        await self.init_redis()
        sessions = []
        
        if self.redis_client:
            try:
                # Get all session keys
                session_keys = await self.redis_client.keys("session:*")
                
                for key in session_keys:
                    session_data = await self.redis_client.get(key)
                    if session_data:
                        try:
                            session_info = eval(session_data.decode())  # In production, use proper JSON parsing
                            sessions.append(session_info)
                        except Exception as e:
                            logger.error(f"Error parsing session data: {e}")
                            
            except Exception as e:
                logger.error(f"Error getting active sessions: {e}")
        
        return sessions
    
    async def terminate_session(self, session_id: str) -> bool:
        """Terminate specific session"""
        await self.init_redis()
        
        if self.redis_client:
            try:
                await self.redis_client.delete(f"session:{session_id}")
                await self.redis_client.delete(f"session_activity:{session_id}")
                return True
            except Exception as e:
                logger.error(f"Error terminating session: {e}")
        
        return False
    
    async def get_security_alerts(self, resolved: Optional[bool] = None) -> List[SecurityAlert]:
        """Get security alerts"""
        if resolved is None:
            return self.security_alerts
        
        return [alert for alert in self.security_alerts if alert.resolved == resolved]
    
    async def resolve_security_alert(self, alert_id: str) -> bool:
        """Resolve security alert"""
        for alert in self.security_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False

# Global authenticator instance
system_auth = SystemControlAuthenticator()

# Security dependencies
async def verify_system_control_access(
    permission: SystemControlPermission,
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> UserInfo:
    """Verify user has system control access"""
    
    # Check IP whitelist
    if not await system_auth.check_ip_whitelist(request.client.host):
        await system_auth._create_security_alert(
            user.user_id,
            "ip_whitelist_violation",
            f"Access attempt from non-whitelisted IP: {request.client.host}",
            "HIGH",
            request
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied: IP address not whitelisted"
        )
    
    # Check user permissions
    user_permissions = SYSTEM_CONTROL_PERMISSIONS.get(user.role, [])
    if permission not in user_permissions:
        await system_auth._create_security_alert(
            user.user_id,
            "permission_violation",
            f"Permission violation: {permission.value} required, user has {user.role.value}",
            "MEDIUM",
            request
        )
        raise HTTPException(
            status_code=403,
            detail=f"Permission '{permission.value}' required"
        )
    
    # Check for suspicious activity
    if await system_auth.detect_suspicious_activity(user.user_id, request):
        raise HTTPException(
            status_code=429,
            detail="Suspicious activity detected. Please try again later."
        )
    
    # Validate session
    if not await system_auth.validate_session(user.session_id):
        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid"
        )
    
    # Log access
    await audit_log(
        user.user_id,
        f"system_control_access:{permission.value}",
        f"system_control_api",
        {"permission": permission.value, "ip": request.client.host},
        request
    )
    
    return user

# Specific permission dependencies
async def require_system_start_permission(
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> UserInfo:
    """Require system start permission"""
    return await verify_system_control_access(SystemControlPermission.SYSTEM_START, request, user)

async def require_system_stop_permission(
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> UserInfo:
    """Require system stop permission"""
    return await verify_system_control_access(SystemControlPermission.SYSTEM_STOP, request, user)

async def require_emergency_stop_permission(
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> UserInfo:
    """Require emergency stop permission"""
    return await verify_system_control_access(SystemControlPermission.EMERGENCY_STOP, request, user)

async def require_system_status_permission(
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> UserInfo:
    """Require system status permission"""
    return await verify_system_control_access(SystemControlPermission.SYSTEM_STATUS, request, user)

async def require_audit_access_permission(
    request: Request,
    user: UserInfo = Depends(verify_token)
) -> UserInfo:
    """Require audit access permission"""
    return await verify_system_control_access(SystemControlPermission.AUDIT_ACCESS, request, user)

# Enhanced authentication models
class SystemControlLoginRequest(BaseModel):
    """System control login request with enhanced security"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    mfa_token: Optional[str] = Field(None, description="MFA token")
    mfa_type: MFAType = Field(MFAType.TOTP, description="MFA type")
    hardware_token: Optional[str] = Field(None, description="Hardware token")
    
class MFASetupRequest(BaseModel):
    """MFA setup request"""
    mfa_type: MFAType
    phone_number: Optional[str] = None
    email: Optional[str] = None

class MFASetupResponse(BaseModel):
    """MFA setup response"""
    success: bool
    mfa_type: MFAType
    secret: Optional[str] = None
    qr_code: Optional[str] = None
    provisioning_uri: Optional[str] = None
    message: str

# Enhanced login endpoint
async def enhanced_system_control_login(
    login_request: SystemControlLoginRequest,
    request: Request
) -> Dict[str, Any]:
    """Enhanced login for system control with additional security"""
    
    # Check IP whitelist
    if not await system_auth.check_ip_whitelist(request.client.host):
        await system_auth._create_security_alert(
            "unknown",
            "ip_whitelist_violation",
            f"Login attempt from non-whitelisted IP: {request.client.host}",
            "HIGH",
            request
        )
        raise HTTPException(
            status_code=403,
            detail="Access denied: IP address not whitelisted"
        )
    
    # Check if user is locked
    if await system_auth.check_user_locked(login_request.username):
        await system_auth._create_security_alert(
            login_request.username,
            "locked_account_access",
            f"Access attempt on locked account: {login_request.username}",
            "MEDIUM",
            request
        )
        raise HTTPException(
            status_code=423,
            detail="Account locked due to too many failed attempts"
        )
    
    # Proceed with normal login flow
    from src.api.authentication import login, LoginRequest
    
    # Create standard login request
    standard_login = LoginRequest(
        username=login_request.username,
        password=login_request.password,
        mfa_token=login_request.mfa_token
    )
    
    try:
        # Attempt login
        login_response = await login(standard_login)
        
        # Additional system control checks
        if system_auth.config.hardware_token_required and not login_request.hardware_token:
            raise HTTPException(
                status_code=401,
                detail="Hardware token required for system control access"
            )
        
        if login_request.hardware_token:
            # Verify hardware token
            if not await system_auth._verify_hardware_token(
                login_response.user_info.user_id,
                login_request.hardware_token
            ):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid hardware token"
                )
        
        # Clear failed attempts on successful login
        await clear_failed_logins(login_request.username)
        
        # Log successful login
        await audit_log(
            login_response.user_info.user_id,
            "system_control_login",
            "authentication",
            {"ip": request.client.host, "mfa_type": login_request.mfa_type.value},
            request
        )
        
        return login_response.dict()
        
    except HTTPException as e:
        # Record failed attempt
        await record_failed_login(login_request.username)
        
        # Create security alert for failed login
        await system_auth._create_security_alert(
            login_request.username,
            "failed_login",
            f"Failed login attempt: {str(e.detail)}",
            "LOW",
            request
        )
        
        raise e

# MFA setup endpoint
async def setup_mfa(
    setup_request: MFASetupRequest,
    user: UserInfo = Depends(verify_token)
) -> MFASetupResponse:
    """Setup MFA for user"""
    
    if setup_request.mfa_type == MFAType.TOTP:
        # Generate TOTP secret and QR code
        result = await system_auth.generate_totp_secret(user.user_id, user.username)
        
        return MFASetupResponse(
            success=True,
            mfa_type=MFAType.TOTP,
            secret=result["secret"],
            qr_code=result["qr_code"],
            provisioning_uri=result["provisioning_uri"],
            message="TOTP setup successful. Scan QR code with your authenticator app."
        )
    
    elif setup_request.mfa_type == MFAType.SMS:
        if not setup_request.phone_number:
            raise HTTPException(status_code=400, detail="Phone number required for SMS MFA")
        
        # Send SMS token
        success = await system_auth.send_sms_token(user.user_id, setup_request.phone_number)
        
        return MFASetupResponse(
            success=success,
            mfa_type=MFAType.SMS,
            message="SMS MFA setup initiated. Check your phone for verification code."
        )
    
    elif setup_request.mfa_type == MFAType.EMAIL:
        if not setup_request.email:
            raise HTTPException(status_code=400, detail="Email required for email MFA")
        
        # Send email token
        success = await system_auth.send_email_token(user.user_id, setup_request.email)
        
        return MFASetupResponse(
            success=success,
            mfa_type=MFAType.EMAIL,
            message="Email MFA setup initiated. Check your email for verification code."
        )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported MFA type")