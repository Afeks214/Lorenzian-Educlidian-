"""
Security Module for Expert Trading Feedback System

This module provides authentication, authorization, and security features
for the expert trading feedback interface.
"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import bcrypt
import structlog
from cryptography.fernet import Fernet
import redis
import json

logger = structlog.get_logger()


class ExpertRole(Enum):
    """Expert role levels"""
    TRADER = "trader"
    SENIOR_TRADER = "senior_trader"
    PORTFOLIO_MANAGER = "portfolio_manager"
    RISK_MANAGER = "risk_manager"
    ADMINISTRATOR = "administrator"


class SecurityLevel(Enum):
    """Security access levels"""
    BASIC = "basic"
    ELEVATED = "elevated"
    RESTRICTED = "restricted"
    ADMIN = "admin"


@dataclass
class ExpertProfile:
    """Expert profile with security attributes"""
    expert_id: str
    name: str
    email: str
    role: ExpertRole
    security_level: SecurityLevel
    created_at: datetime
    last_login: Optional[datetime]
    login_attempts: int
    is_active: bool
    permissions: List[str]
    department: str
    supervisor_id: Optional[str]
    trading_desk: str


@dataclass
class SecurityAuditLog:
    """Security audit log entry"""
    timestamp: datetime
    expert_id: str
    action: str
    resource: str
    ip_address: str
    user_agent: str
    success: bool
    details: Optional[Dict]


class SecurityManager:
    """Comprehensive security manager for expert feedback system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.jwt_secret = self._generate_or_load_jwt_secret()
        self.encryption_key = self._generate_or_load_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Security policies
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.jwt_expiry = timedelta(hours=8)
        self.session_timeout = timedelta(hours=12)
        
        # Rate limiting
        self.rate_limits = {
            "login": (5, timedelta(minutes=15)),  # 5 attempts per 15 minutes
            "feedback": (100, timedelta(hours=1)),  # 100 feedback submissions per hour
            "api_calls": (1000, timedelta(hours=1))  # 1000 API calls per hour
        }
        
        # Expert profiles storage (in production, use proper database)
        self.expert_profiles: Dict[str, ExpertProfile] = {}
        self.audit_logs: List[SecurityAuditLog] = []
        
        logger.info("Security Manager initialized")

    def _generate_or_load_jwt_secret(self) -> str:
        """Generate or load JWT secret key"""
        secret = self.redis_client.get("jwt_secret")
        if secret:
            return secret.decode('utf-8')
        
        # Generate new secret
        new_secret = secrets.token_urlsafe(64)
        self.redis_client.set("jwt_secret", new_secret, ex=86400 * 30)  # 30 days
        return new_secret

    def _generate_or_load_encryption_key(self) -> bytes:
        """Generate or load encryption key"""
        key = self.redis_client.get("encryption_key")
        if key:
            return key
        
        # Generate new key
        new_key = Fernet.generate_key()
        self.redis_client.set("encryption_key", new_key, ex=86400 * 30)  # 30 days
        return new_key

    def create_expert_profile(
        self,
        expert_id: str,
        name: str,
        email: str,
        password: str,
        role: ExpertRole = ExpertRole.TRADER,
        security_level: SecurityLevel = SecurityLevel.BASIC,
        department: str = "Trading",
        trading_desk: str = "Main"
    ) -> bool:
        """Create a new expert profile"""
        try:
            # Validate inputs
            if expert_id in self.expert_profiles:
                logger.warning("Expert profile already exists", expert_id=expert_id)
                return False
            
            if not self._validate_password_strength(password):
                logger.warning("Password does not meet security requirements")
                return False
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Create profile
            profile = ExpertProfile(
                expert_id=expert_id,
                name=name,
                email=email,
                role=role,
                security_level=security_level,
                created_at=datetime.now(),
                last_login=None,
                login_attempts=0,
                is_active=True,
                permissions=self._get_default_permissions(role, security_level),
                department=department,
                supervisor_id=None,
                trading_desk=trading_desk
            )
            
            # Store profile and password
            self.expert_profiles[expert_id] = profile
            self.redis_client.set(f"expert_password:{expert_id}", password_hash, ex=86400 * 90)  # 90 days
            
            self._log_security_event(
                expert_id=expert_id,
                action="profile_created",
                resource="user_profile",
                ip_address="system",
                user_agent="system",
                success=True
            )
            
            logger.info("Expert profile created", expert_id=expert_id, role=role.value)
            return True
            
        except Exception as e:
            logger.error("Failed to create expert profile", error=str(e))
            return False

    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements"""
        if len(password) < 12:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return all([has_upper, has_lower, has_digit, has_special])

    def _get_default_permissions(self, role: ExpertRole, security_level: SecurityLevel) -> List[str]:
        """Get default permissions based on role and security level"""
        base_permissions = ["view_decisions", "submit_feedback", "view_own_analytics"]
        
        role_permissions = {
            ExpertRole.TRADER: [],
            ExpertRole.SENIOR_TRADER: ["mentor_junior_traders"],
            ExpertRole.PORTFOLIO_MANAGER: ["view_portfolio_analytics", "override_decisions"],
            ExpertRole.RISK_MANAGER: ["view_all_analytics", "emergency_stop"],
            ExpertRole.ADMINISTRATOR: ["manage_users", "view_audit_logs", "system_admin"]
        }
        
        security_permissions = {
            SecurityLevel.BASIC: [],
            SecurityLevel.ELEVATED: ["view_sensitive_data"],
            SecurityLevel.RESTRICTED: ["access_restricted_decisions"],
            SecurityLevel.ADMIN: ["admin_panel", "security_management"]
        }
        
        permissions = base_permissions.copy()
        permissions.extend(role_permissions.get(role, []))
        permissions.extend(security_permissions.get(security_level, []))
        
        return list(set(permissions))  # Remove duplicates

    def authenticate_expert(
        self,
        expert_id: str,
        password: str,
        ip_address: str,
        user_agent: str
    ) -> Tuple[bool, Optional[str], Optional[ExpertProfile]]:
        """Authenticate expert and return success status, JWT token, and profile"""
        
        # Check rate limiting
        if not self._check_rate_limit(expert_id, "login", ip_address):
            self._log_security_event(
                expert_id=expert_id,
                action="login_rate_limited",
                resource="authentication",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return False, None, None
        
        # Check if expert exists
        if expert_id not in self.expert_profiles:
            self._log_security_event(
                expert_id=expert_id,
                action="login_attempt_unknown_user",
                resource="authentication",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return False, None, None
        
        profile = self.expert_profiles[expert_id]
        
        # Check if account is active
        if not profile.is_active:
            self._log_security_event(
                expert_id=expert_id,
                action="login_attempt_inactive_account",
                resource="authentication",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return False, None, None
        
        # Check if account is locked
        if self._is_account_locked(expert_id):
            self._log_security_event(
                expert_id=expert_id,
                action="login_attempt_locked_account",
                resource="authentication",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return False, None, None
        
        # Verify password
        stored_password_hash = self.redis_client.get(f"expert_password:{expert_id}")
        if not stored_password_hash:
            return False, None, None
        
        if not bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
            # Increment failed attempts
            profile.login_attempts += 1
            self.redis_client.incr(f"failed_attempts:{expert_id}", 1)
            self.redis_client.expire(f"failed_attempts:{expert_id}", int(self.lockout_duration.total_seconds()))
            
            self._log_security_event(
                expert_id=expert_id,
                action="login_failed_invalid_password",
                resource="authentication",
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return False, None, None
        
        # Successful authentication
        profile.last_login = datetime.now()
        profile.login_attempts = 0
        self.redis_client.delete(f"failed_attempts:{expert_id}")
        
        # Generate JWT token
        token = self._generate_jwt_token(profile, ip_address)
        
        self._log_security_event(
            expert_id=expert_id,
            action="login_successful",
            resource="authentication",
            ip_address=ip_address,
            user_agent=user_agent,
            success=True
        )
        
        logger.info("Expert authenticated successfully", expert_id=expert_id)
        return True, token, profile

    def _check_rate_limit(self, identifier: str, action: str, ip_address: str) -> bool:
        """Check if action is within rate limits"""
        if action not in self.rate_limits:
            return True
        
        limit_count, limit_window = self.rate_limits[action]
        
        # Check both user-based and IP-based limits
        user_key = f"rate_limit:{action}:{identifier}"
        ip_key = f"rate_limit:{action}:ip:{ip_address}"
        
        user_count = self.redis_client.get(user_key)
        ip_count = self.redis_client.get(ip_key)
        
        if user_count and int(user_count) >= limit_count:
            return False
        
        if ip_count and int(ip_count) >= limit_count * 2:  # IP limit is 2x user limit
            return False
        
        # Increment counters
        pipe = self.redis_client.pipeline()
        pipe.incr(user_key, 1)
        pipe.expire(user_key, int(limit_window.total_seconds()))
        pipe.incr(ip_key, 1)
        pipe.expire(ip_key, int(limit_window.total_seconds()))
        pipe.execute()
        
        return True

    def _is_account_locked(self, expert_id: str) -> bool:
        """Check if account is locked due to failed attempts"""
        failed_attempts = self.redis_client.get(f"failed_attempts:{expert_id}")
        if failed_attempts and int(failed_attempts) >= self.max_login_attempts:
            return True
        return False

    def _generate_jwt_token(self, profile: ExpertProfile, ip_address: str) -> str:
        """Generate JWT token for authenticated expert"""
        payload = {
            "expert_id": profile.expert_id,
            "name": profile.name,
            "role": profile.role.value,
            "security_level": profile.security_level.value,
            "permissions": profile.permissions,
            "department": profile.department,
            "trading_desk": profile.trading_desk,
            "ip_address": hashlib.sha256(ip_address.encode()).hexdigest()[:16],  # Hashed IP
            "exp": datetime.utcnow() + self.jwt_expiry,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        # Store token for revocation checking
        self.redis_client.set(
            f"active_token:{payload['jti']}", 
            json.dumps({"expert_id": profile.expert_id, "ip": ip_address}),
            ex=int(self.jwt_expiry.total_seconds())
        )
        
        return token

    def verify_jwt_token(self, token: str, ip_address: str) -> Tuple[bool, Optional[Dict]]:
        """Verify JWT token and return payload if valid"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token is revoked
            token_id = payload.get("jti")
            if token_id and not self.redis_client.get(f"active_token:{token_id}"):
                return False, None
            
            # Check IP address consistency (optional security feature)
            token_ip_hash = payload.get("ip_address")
            current_ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
            if token_ip_hash != current_ip_hash:
                logger.warning("IP address mismatch in token verification", expert_id=payload.get("expert_id"))
                # Optionally enforce IP checking
                # return False, None
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            return False, None
        except jwt.InvalidTokenError:
            return False, None

    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"], options={"verify_exp": False})
            token_id = payload.get("jti")
            
            if token_id:
                self.redis_client.delete(f"active_token:{token_id}")
                self._log_security_event(
                    expert_id=payload.get("expert_id", "unknown"),
                    action="token_revoked",
                    resource="authentication",
                    ip_address="system",
                    user_agent="system",
                    success=True
                )
                return True
                
        except Exception as e:
            logger.error("Failed to revoke token", error=str(e))
        
        return False

    def check_permission(self, expert_id: str, permission: str) -> bool:
        """Check if expert has specific permission"""
        if expert_id not in self.expert_profiles:
            return False
        
        profile = self.expert_profiles[expert_id]
        return permission in profile.permissions

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

    def _log_security_event(
        self,
        expert_id: str,
        action: str,
        resource: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        details: Optional[Dict] = None
    ):
        """Log security event for audit purposes"""
        log_entry = SecurityAuditLog(
            timestamp=datetime.now(),
            expert_id=expert_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
        
        self.audit_logs.append(log_entry)
        
        # Store in Redis for persistence
        log_data = {
            "timestamp": log_entry.timestamp.isoformat(),
            "expert_id": expert_id,
            "action": action,
            "resource": resource,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "success": success,
            "details": details
        }
        
        self.redis_client.lpush("security_audit_log", json.dumps(log_data))
        self.redis_client.ltrim("security_audit_log", 0, 10000)  # Keep last 10k entries

    def get_security_metrics(self) -> Dict:
        """Get security metrics for monitoring"""
        total_experts = len(self.expert_profiles)
        active_experts = sum(1 for profile in self.expert_profiles.values() if profile.is_active)
        
        # Get recent login attempts
        recent_logs = self.audit_logs[-100:]  # Last 100 events
        successful_logins = sum(1 for log in recent_logs if log.action == "login_successful")
        failed_logins = sum(1 for log in recent_logs if "login_failed" in log.action)
        
        return {
            "total_experts": total_experts,
            "active_experts": active_experts,
            "recent_successful_logins": successful_logins,
            "recent_failed_logins": failed_logins,
            "security_events_last_hour": len([
                log for log in recent_logs 
                if log.timestamp > datetime.now() - timedelta(hours=1)
            ])
        }

    def get_jwt_secret(self) -> str:
        """Get JWT secret for token operations"""
        return self.jwt_secret

    def initialize_default_experts(self):
        """Initialize default expert accounts for testing"""
        default_experts = [
            {
                "expert_id": "trader001",
                "name": "John Smith",
                "email": "john.smith@trading.com",
                "password": "SecurePass123!",
                "role": ExpertRole.TRADER,
                "security_level": SecurityLevel.BASIC
            },
            {
                "expert_id": "senior001",
                "name": "Sarah Johnson",
                "email": "sarah.johnson@trading.com",
                "password": "SecurePass456!",
                "role": ExpertRole.SENIOR_TRADER,
                "security_level": SecurityLevel.ELEVATED
            },
            {
                "expert_id": "pm001",
                "name": "Michael Chen",
                "email": "michael.chen@trading.com",
                "password": "SecurePass789!",
                "role": ExpertRole.PORTFOLIO_MANAGER,
                "security_level": SecurityLevel.RESTRICTED
            }
        ]
        
        for expert_data in default_experts:
            self.create_expert_profile(**expert_data)
        
        logger.info("Default expert accounts initialized")