"""
Role-Based Access Control (RBAC) System
Fine-grained permissions for financial trading operations
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid
import hashlib
from functools import wraps
from contextlib import asynccontextmanager

from src.monitoring.logger_config import get_logger
from src.security.financial_audit_logger import (
    log_audit_event, AuditEventType, AuditSeverity, AuditContext
)
from src.security.vault_integration import get_vault_secret

logger = get_logger(__name__)

class PermissionScope(Enum):
    """Permission scopes for financial operations"""
    # System-wide permissions
    SYSTEM_ADMIN = "system.admin"
    SYSTEM_READ = "system.read"
    SYSTEM_WRITE = "system.write"
    
    # Trading permissions
    TRADING_VIEW = "trading.view"
    TRADING_EXECUTE = "trading.execute"
    TRADING_CANCEL = "trading.cancel"
    TRADING_MODIFY = "trading.modify"
    TRADING_APPROVE = "trading.approve"
    
    # Portfolio permissions
    PORTFOLIO_VIEW = "portfolio.view"
    PORTFOLIO_CREATE = "portfolio.create"
    PORTFOLIO_MODIFY = "portfolio.modify"
    PORTFOLIO_DELETE = "portfolio.delete"
    PORTFOLIO_REBALANCE = "portfolio.rebalance"
    
    # Risk management permissions
    RISK_VIEW = "risk.view"
    RISK_CONFIGURE = "risk.configure"
    RISK_OVERRIDE = "risk.override"
    RISK_MONITOR = "risk.monitor"
    
    # Market data permissions
    MARKET_DATA_VIEW = "market_data.view"
    MARKET_DATA_HISTORICAL = "market_data.historical"
    MARKET_DATA_REAL_TIME = "market_data.real_time"
    
    # Compliance permissions
    COMPLIANCE_VIEW = "compliance.view"
    COMPLIANCE_CONFIGURE = "compliance.configure"
    COMPLIANCE_REPORT = "compliance.report"
    
    # Audit permissions
    AUDIT_VIEW = "audit.view"
    AUDIT_EXPORT = "audit.export"
    
    # Model permissions
    MODEL_VIEW = "model.view"
    MODEL_TRAIN = "model.train"
    MODEL_DEPLOY = "model.deploy"
    MODEL_DELETE = "model.delete"
    
    # User management permissions
    USER_VIEW = "user.view"
    USER_CREATE = "user.create"
    USER_MODIFY = "user.modify"
    USER_DELETE = "user.delete"
    
    # Configuration permissions
    CONFIG_VIEW = "config.view"
    CONFIG_MODIFY = "config.modify"

class UserRole(Enum):
    """Predefined user roles"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    COMPLIANCE_OFFICER = "compliance_officer"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_SERVICE = "api_service"
    SYSTEM_SERVICE = "system_service"

class ResourceType(Enum):
    """Resource types for access control"""
    ACCOUNT = "account"
    PORTFOLIO = "portfolio"
    INSTRUMENT = "instrument"
    ORDER = "order"
    TRADE = "trade"
    POSITION = "position"
    RISK_PROFILE = "risk_profile"
    COMPLIANCE_RULE = "compliance_rule"
    AUDIT_LOG = "audit_log"
    MODEL = "model"
    CONFIGURATION = "configuration"
    USER = "user"
    MARKET_DATA = "market_data"

@dataclass
class Permission:
    """Individual permission"""
    scope: PermissionScope
    resource_type: Optional[ResourceType] = None
    resource_id: Optional[str] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    granted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    granted_by: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if permission is expired"""
        return self.expires_at is not None and datetime.now(timezone.utc) > self.expires_at
    
    def matches_resource(self, resource_type: ResourceType, resource_id: str = None) -> bool:
        """Check if permission matches resource"""
        if self.resource_type is None:
            return True  # Global permission
        
        if self.resource_type != resource_type:
            return False
        
        if self.resource_id is None:
            return True  # Permission for all resources of this type
        
        return self.resource_id == resource_id
    
    def evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """Evaluate permission conditions"""
        if not self.conditions:
            return True
        
        for condition_key, condition_value in self.conditions.items():
            if condition_key == "time_range":
                current_time = datetime.now(timezone.utc).time()
                start_time = datetime.strptime(condition_value["start"], "%H:%M").time()
                end_time = datetime.strptime(condition_value["end"], "%H:%M").time()
                
                if not (start_time <= current_time <= end_time):
                    return False
            
            elif condition_key == "ip_whitelist":
                client_ip = context.get("ip_address")
                if client_ip not in condition_value:
                    return False
            
            elif condition_key == "max_amount":
                request_amount = context.get("amount", 0)
                if request_amount > condition_value:
                    return False
            
            elif condition_key == "instruments":
                instrument_id = context.get("instrument_id")
                if instrument_id not in condition_value:
                    return False
            
            elif condition_key in context:
                if context[condition_key] != condition_value:
                    return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scope": self.scope.value,
            "resource_type": self.resource_type.value if self.resource_type else None,
            "resource_id": self.resource_id,
            "conditions": self.conditions,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "granted_by": self.granted_by
        }

@dataclass
class Role:
    """Role definition"""
    name: str
    display_name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None
    is_system_role: bool = False
    
    def add_permission(self, permission: Permission):
        """Add permission to role"""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role"""
        self.permissions.discard(permission)
    
    def has_permission(self, 
                      scope: PermissionScope, 
                      resource_type: ResourceType = None,
                      resource_id: str = None,
                      context: Dict[str, Any] = None) -> bool:
        """Check if role has specific permission"""
        context = context or {}
        
        for permission in self.permissions:
            if permission.is_expired():
                continue
            
            if permission.scope == scope:
                if permission.matches_resource(resource_type, resource_id):
                    if permission.evaluate_conditions(context):
                        return True
        
        return False
    
    def get_effective_permissions(self, rbac_system: 'RBACSystem') -> Set[Permission]:
        """Get all effective permissions including inherited ones"""
        effective_permissions = self.permissions.copy()
        
        # Add permissions from parent roles
        for parent_role_name in self.parent_roles:
            parent_role = rbac_system.get_role(parent_role_name)
            if parent_role:
                effective_permissions.update(parent_role.get_effective_permissions(rbac_system))
        
        # Remove expired permissions
        effective_permissions = {p for p in effective_permissions if not p.is_expired()}
        
        return effective_permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "permissions": [p.to_dict() for p in self.permissions],
            "parent_roles": list(self.parent_roles),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "is_system_role": self.is_system_role
        }

@dataclass
class User:
    """User with roles and permissions"""
    user_id: str
    username: str
    email: str
    full_name: str
    roles: Set[str] = field(default_factory=set)
    direct_permissions: Set[Permission] = field(default_factory=set)
    is_active: bool = True
    is_locked: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    password_expires_at: Optional[datetime] = None
    session_timeout: int = 3600  # seconds
    
    def add_role(self, role_name: str):
        """Add role to user"""
        self.roles.add(role_name)
    
    def remove_role(self, role_name: str):
        """Remove role from user"""
        self.roles.discard(role_name)
    
    def add_permission(self, permission: Permission):
        """Add direct permission to user"""
        self.direct_permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove direct permission from user"""
        self.direct_permissions.discard(permission)
    
    def is_locked_out(self) -> bool:
        """Check if user is locked out"""
        return self.is_locked or self.failed_login_attempts >= 5
    
    def is_password_expired(self) -> bool:
        """Check if password is expired"""
        return (self.password_expires_at is not None and 
                datetime.now(timezone.utc) > self.password_expires_at)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "roles": list(self.roles),
            "direct_permissions": [p.to_dict() for p in self.direct_permissions],
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "failed_login_attempts": self.failed_login_attempts,
            "password_expires_at": self.password_expires_at.isoformat() if self.password_expires_at else None,
            "session_timeout": self.session_timeout
        }

@dataclass
class Session:
    """User session"""
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return (self.expires_at is not None and 
                datetime.now(timezone.utc) > self.expires_at)
    
    def extend_session(self, timeout: int = 3600):
        """Extend session expiration"""
        self.expires_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)

class RBACSystem:
    """Role-Based Access Control System"""
    
    def __init__(self, storage_path: str = "rbac_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory storage
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.sessions: Dict[str, Session] = {}
        
        # Locks for thread safety
        self.users_lock = asyncio.Lock()
        self.roles_lock = asyncio.Lock()
        self.sessions_lock = asyncio.Lock()
        
        # Initialize default roles
        self._initialize_default_roles()
        
        # Load data from storage
        self._load_data()
        
        logger.info("RBACSystem initialized")
    
    def _initialize_default_roles(self):
        """Initialize default system roles"""
        default_roles = {
            UserRole.SUPER_ADMIN: {
                "display_name": "Super Administrator",
                "description": "Full system access",
                "permissions": [
                    Permission(PermissionScope.SYSTEM_ADMIN),
                    Permission(PermissionScope.SYSTEM_READ),
                    Permission(PermissionScope.SYSTEM_WRITE),
                    Permission(PermissionScope.USER_CREATE),
                    Permission(PermissionScope.USER_MODIFY),
                    Permission(PermissionScope.USER_DELETE),
                    Permission(PermissionScope.CONFIG_MODIFY)
                ]
            },
            UserRole.ADMIN: {
                "display_name": "Administrator",
                "description": "System administration access",
                "permissions": [
                    Permission(PermissionScope.SYSTEM_READ),
                    Permission(PermissionScope.USER_VIEW),
                    Permission(PermissionScope.USER_CREATE),
                    Permission(PermissionScope.USER_MODIFY),
                    Permission(PermissionScope.CONFIG_VIEW),
                    Permission(PermissionScope.AUDIT_VIEW),
                    Permission(PermissionScope.AUDIT_EXPORT)
                ]
            },
            UserRole.COMPLIANCE_OFFICER: {
                "display_name": "Compliance Officer",
                "description": "Compliance monitoring and reporting",
                "permissions": [
                    Permission(PermissionScope.COMPLIANCE_VIEW),
                    Permission(PermissionScope.COMPLIANCE_CONFIGURE),
                    Permission(PermissionScope.COMPLIANCE_REPORT),
                    Permission(PermissionScope.AUDIT_VIEW),
                    Permission(PermissionScope.AUDIT_EXPORT),
                    Permission(PermissionScope.TRADING_VIEW),
                    Permission(PermissionScope.PORTFOLIO_VIEW),
                    Permission(PermissionScope.RISK_VIEW)
                ]
            },
            UserRole.RISK_MANAGER: {
                "display_name": "Risk Manager",
                "description": "Risk management and oversight",
                "permissions": [
                    Permission(PermissionScope.RISK_VIEW),
                    Permission(PermissionScope.RISK_CONFIGURE),
                    Permission(PermissionScope.RISK_OVERRIDE),
                    Permission(PermissionScope.RISK_MONITOR),
                    Permission(PermissionScope.TRADING_VIEW),
                    Permission(PermissionScope.PORTFOLIO_VIEW),
                    Permission(PermissionScope.MARKET_DATA_VIEW)
                ]
            },
            UserRole.PORTFOLIO_MANAGER: {
                "display_name": "Portfolio Manager",
                "description": "Portfolio management and trading",
                "permissions": [
                    Permission(PermissionScope.PORTFOLIO_VIEW),
                    Permission(PermissionScope.PORTFOLIO_CREATE),
                    Permission(PermissionScope.PORTFOLIO_MODIFY),
                    Permission(PermissionScope.PORTFOLIO_REBALANCE),
                    Permission(PermissionScope.TRADING_VIEW),
                    Permission(PermissionScope.TRADING_EXECUTE),
                    Permission(PermissionScope.TRADING_MODIFY),
                    Permission(PermissionScope.TRADING_CANCEL),
                    Permission(PermissionScope.MARKET_DATA_VIEW),
                    Permission(PermissionScope.MARKET_DATA_HISTORICAL),
                    Permission(PermissionScope.RISK_VIEW)
                ]
            },
            UserRole.TRADER: {
                "display_name": "Trader",
                "description": "Trading operations",
                "permissions": [
                    Permission(PermissionScope.TRADING_VIEW),
                    Permission(PermissionScope.TRADING_EXECUTE),
                    Permission(PermissionScope.TRADING_MODIFY),
                    Permission(PermissionScope.TRADING_CANCEL),
                    Permission(PermissionScope.PORTFOLIO_VIEW),
                    Permission(PermissionScope.MARKET_DATA_VIEW),
                    Permission(PermissionScope.MARKET_DATA_REAL_TIME),
                    Permission(PermissionScope.RISK_VIEW)
                ]
            },
            UserRole.ANALYST: {
                "display_name": "Analyst",
                "description": "Analysis and research",
                "permissions": [
                    Permission(PermissionScope.MARKET_DATA_VIEW),
                    Permission(PermissionScope.MARKET_DATA_HISTORICAL),
                    Permission(PermissionScope.MARKET_DATA_REAL_TIME),
                    Permission(PermissionScope.PORTFOLIO_VIEW),
                    Permission(PermissionScope.TRADING_VIEW),
                    Permission(PermissionScope.MODEL_VIEW),
                    Permission(PermissionScope.MODEL_TRAIN),
                    Permission(PermissionScope.RISK_VIEW)
                ]
            },
            UserRole.VIEWER: {
                "display_name": "Viewer",
                "description": "Read-only access",
                "permissions": [
                    Permission(PermissionScope.TRADING_VIEW),
                    Permission(PermissionScope.PORTFOLIO_VIEW),
                    Permission(PermissionScope.MARKET_DATA_VIEW),
                    Permission(PermissionScope.RISK_VIEW)
                ]
            },
            UserRole.API_SERVICE: {
                "display_name": "API Service",
                "description": "API service access",
                "permissions": [
                    Permission(PermissionScope.SYSTEM_READ),
                    Permission(PermissionScope.MARKET_DATA_VIEW),
                    Permission(PermissionScope.MARKET_DATA_REAL_TIME),
                    Permission(PermissionScope.TRADING_VIEW),
                    Permission(PermissionScope.PORTFOLIO_VIEW)
                ]
            },
            UserRole.SYSTEM_SERVICE: {
                "display_name": "System Service",
                "description": "System service access",
                "permissions": [
                    Permission(PermissionScope.SYSTEM_READ),
                    Permission(PermissionScope.SYSTEM_WRITE),
                    Permission(PermissionScope.CONFIG_VIEW),
                    Permission(PermissionScope.MODEL_VIEW),
                    Permission(PermissionScope.MODEL_TRAIN),
                    Permission(PermissionScope.MODEL_DEPLOY)
                ]
            }
        }
        
        for role_enum, role_config in default_roles.items():
            role = Role(
                name=role_enum.value,
                display_name=role_config["display_name"],
                description=role_config["description"],
                permissions=set(role_config["permissions"]),
                is_system_role=True
            )
            self.roles[role_enum.value] = role
    
    def _load_data(self):
        """Load data from storage"""
        # Load users
        users_file = self.storage_path / "users.json"
        if users_file.exists():
            try:
                with open(users_file, 'r') as f:
                    users_data = json.load(f)
                    for user_data in users_data:
                        user = User(
                            user_id=user_data["user_id"],
                            username=user_data["username"],
                            email=user_data["email"],
                            full_name=user_data["full_name"],
                            roles=set(user_data["roles"]),
                            is_active=user_data["is_active"],
                            is_locked=user_data["is_locked"],
                            created_at=datetime.fromisoformat(user_data["created_at"]),
                            last_login=datetime.fromisoformat(user_data["last_login"]) if user_data["last_login"] else None,
                            failed_login_attempts=user_data["failed_login_attempts"],
                            password_expires_at=datetime.fromisoformat(user_data["password_expires_at"]) if user_data["password_expires_at"] else None,
                            session_timeout=user_data["session_timeout"]
                        )
                        
                        # Load direct permissions
                        for perm_data in user_data["direct_permissions"]:
                            permission = Permission(
                                scope=PermissionScope(perm_data["scope"]),
                                resource_type=ResourceType(perm_data["resource_type"]) if perm_data["resource_type"] else None,
                                resource_id=perm_data["resource_id"],
                                conditions=perm_data["conditions"],
                                granted_at=datetime.fromisoformat(perm_data["granted_at"]),
                                expires_at=datetime.fromisoformat(perm_data["expires_at"]) if perm_data["expires_at"] else None,
                                granted_by=perm_data["granted_by"]
                            )
                            user.direct_permissions.add(permission)
                        
                        self.users[user.user_id] = user
                        
            except Exception as e:
                logger.error(f"Failed to load users: {e}")
        
        # Load custom roles
        roles_file = self.storage_path / "roles.json"
        if roles_file.exists():
            try:
                with open(roles_file, 'r') as f:
                    roles_data = json.load(f)
                    for role_data in roles_data:
                        if not role_data["is_system_role"]:  # Only load custom roles
                            role = Role(
                                name=role_data["name"],
                                display_name=role_data["display_name"],
                                description=role_data["description"],
                                parent_roles=set(role_data["parent_roles"]),
                                created_at=datetime.fromisoformat(role_data["created_at"]),
                                created_by=role_data["created_by"],
                                is_system_role=False
                            )
                            
                            # Load permissions
                            for perm_data in role_data["permissions"]:
                                permission = Permission(
                                    scope=PermissionScope(perm_data["scope"]),
                                    resource_type=ResourceType(perm_data["resource_type"]) if perm_data["resource_type"] else None,
                                    resource_id=perm_data["resource_id"],
                                    conditions=perm_data["conditions"],
                                    granted_at=datetime.fromisoformat(perm_data["granted_at"]),
                                    expires_at=datetime.fromisoformat(perm_data["expires_at"]) if perm_data["expires_at"] else None,
                                    granted_by=perm_data["granted_by"]
                                )
                                role.permissions.add(permission)
                            
                            self.roles[role.name] = role
                            
            except Exception as e:
                logger.error(f"Failed to load roles: {e}")
    
    def _save_data(self):
        """Save data to storage"""
        # Save users
        users_file = self.storage_path / "users.json"
        try:
            with open(users_file, 'w') as f:
                users_data = [user.to_dict() for user in self.users.values()]
                json.dump(users_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
        
        # Save custom roles
        roles_file = self.storage_path / "roles.json"
        try:
            with open(roles_file, 'w') as f:
                roles_data = [role.to_dict() for role in self.roles.values()]
                json.dump(roles_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save roles: {e}")
    
    async def create_user(self, 
                         username: str,
                         email: str,
                         full_name: str,
                         roles: List[str] = None,
                         session_timeout: int = 3600,
                         password_expires_days: int = 90) -> User:
        """Create new user"""
        async with self.users_lock:
            user_id = str(uuid.uuid4())
            
            # Check if username already exists
            if any(u.username == username for u in self.users.values()):
                raise ValueError(f"Username '{username}' already exists")
            
            # Check if email already exists
            if any(u.email == email for u in self.users.values()):
                raise ValueError(f"Email '{email}' already exists")
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                full_name=full_name,
                roles=set(roles or []),
                session_timeout=session_timeout,
                password_expires_at=datetime.now(timezone.utc) + timedelta(days=password_expires_days)
            )
            
            self.users[user_id] = user
            self._save_data()
            
            # Audit log
            await log_audit_event(
                AuditEventType.USER_CREATED,
                f"User created: {username}",
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "user_id": user_id,
                    "username": username,
                    "email": email,
                    "roles": list(roles or [])
                }
            )
            
            return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    async def update_user(self, user_id: str, **kwargs) -> bool:
        """Update user"""
        async with self.users_lock:
            user = self.users.get(user_id)
            if not user:
                return False
            
            # Update allowed fields
            updatable_fields = {
                'email', 'full_name', 'is_active', 'is_locked', 
                'session_timeout', 'password_expires_at'
            }
            
            updated_fields = []
            for field, value in kwargs.items():
                if field in updatable_fields and hasattr(user, field):
                    setattr(user, field, value)
                    updated_fields.append(field)
            
            if updated_fields:
                self._save_data()
                
                # Audit log
                await log_audit_event(
                    AuditEventType.USER_MODIFIED,
                    f"User updated: {user.username}",
                    severity=AuditSeverity.MEDIUM,
                    additional_data={
                        "user_id": user_id,
                        "updated_fields": updated_fields
                    }
                )
                
                return True
            
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user"""
        async with self.users_lock:
            user = self.users.get(user_id)
            if not user:
                return False
            
            username = user.username
            del self.users[user_id]
            self._save_data()
            
            # Audit log
            await log_audit_event(
                AuditEventType.USER_DELETED,
                f"User deleted: {username}",
                severity=AuditSeverity.HIGH,
                additional_data={"user_id": user_id, "username": username}
            )
            
            return True
    
    async def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assign role to user"""
        async with self.users_lock:
            user = self.users.get(user_id)
            if not user:
                return False
            
            if role_name not in self.roles:
                return False
            
            user.add_role(role_name)
            self._save_data()
            
            # Audit log
            await log_audit_event(
                AuditEventType.ROLE_ASSIGNED,
                f"Role '{role_name}' assigned to user {user.username}",
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "user_id": user_id,
                    "username": user.username,
                    "role_name": role_name
                }
            )
            
            return True
    
    async def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Revoke role from user"""
        async with self.users_lock:
            user = self.users.get(user_id)
            if not user:
                return False
            
            user.remove_role(role_name)
            self._save_data()
            
            # Audit log
            await log_audit_event(
                AuditEventType.ROLE_REVOKED,
                f"Role '{role_name}' revoked from user {user.username}",
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "user_id": user_id,
                    "username": user.username,
                    "role_name": role_name
                }
            )
            
            return True
    
    async def create_role(self, 
                         name: str,
                         display_name: str,
                         description: str,
                         permissions: List[Permission] = None,
                         parent_roles: List[str] = None,
                         created_by: str = None) -> Role:
        """Create custom role"""
        async with self.roles_lock:
            if name in self.roles:
                raise ValueError(f"Role '{name}' already exists")
            
            role = Role(
                name=name,
                display_name=display_name,
                description=description,
                permissions=set(permissions or []),
                parent_roles=set(parent_roles or []),
                created_by=created_by,
                is_system_role=False
            )
            
            self.roles[name] = role
            self._save_data()
            
            # Audit log
            await log_audit_event(
                AuditEventType.ROLE_CREATED,
                f"Role created: {name}",
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "role_name": name,
                    "display_name": display_name,
                    "permissions_count": len(permissions or []),
                    "parent_roles": parent_roles or []
                }
            )
            
            return role
    
    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name"""
        return self.roles.get(name)
    
    async def has_permission(self, 
                           user_id: str,
                           permission_scope: PermissionScope,
                           resource_type: ResourceType = None,
                           resource_id: str = None,
                           context: Dict[str, Any] = None) -> bool:
        """Check if user has specific permission"""
        user = self.users.get(user_id)
        if not user or not user.is_active or user.is_locked_out():
            return False
        
        context = context or {}
        
        # Check direct permissions
        for permission in user.direct_permissions:
            if permission.is_expired():
                continue
            
            if permission.scope == permission_scope:
                if permission.matches_resource(resource_type, resource_id):
                    if permission.evaluate_conditions(context):
                        return True
        
        # Check role permissions
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                effective_permissions = role.get_effective_permissions(self)
                for permission in effective_permissions:
                    if permission.scope == permission_scope:
                        if permission.matches_resource(resource_type, resource_id):
                            if permission.evaluate_conditions(context):
                                return True
        
        return False
    
    async def create_session(self, user_id: str, ip_address: str = None, user_agent: str = None) -> Optional[Session]:
        """Create user session"""
        user = self.users.get(user_id)
        if not user or not user.is_active or user.is_locked_out():
            return None
        
        async with self.sessions_lock:
            session_id = str(uuid.uuid4())
            
            session = Session(
                session_id=session_id,
                user_id=user_id,
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=user.session_timeout),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.sessions[session_id] = session
            
            # Update user last login
            user.last_login = datetime.now(timezone.utc)
            user.failed_login_attempts = 0
            self._save_data()
            
            # Audit log
            await log_audit_event(
                AuditEventType.LOGIN_SUCCESS,
                f"User session created: {user.username}",
                severity=AuditSeverity.INFO,
                context=AuditContext(
                    user_id=user_id,
                    session_id=session_id,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
            )
            
            return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if session and session.is_expired():
            await self.invalidate_session(session_id)
            return None
        return session
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session"""
        async with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            session.is_active = False
            
            # Audit log
            await log_audit_event(
                AuditEventType.LOGOUT,
                f"User session invalidated",
                severity=AuditSeverity.INFO,
                context=AuditContext(
                    user_id=session.user_id,
                    session_id=session_id,
                    ip_address=session.ip_address
                )
            )
            
            return True
    
    async def extend_session(self, session_id: str) -> bool:
        """Extend session expiration"""
        async with self.sessions_lock:
            session = self.sessions.get(session_id)
            if not session or not session.is_active:
                return False
            
            user = self.users.get(session.user_id)
            if not user:
                return False
            
            session.extend_session(user.session_timeout)
            return True
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str = None, user_agent: str = None) -> Optional[Session]:
        """Authenticate user and create session"""
        user = await self.get_user_by_username(username)
        if not user:
            # Audit failed login
            await log_audit_event(
                AuditEventType.LOGIN_FAILURE,
                f"Authentication failed - user not found: {username}",
                severity=AuditSeverity.MEDIUM,
                context=AuditContext(
                    ip_address=ip_address,
                    user_agent=user_agent
                )
            )
            return None
        
        if not user.is_active or user.is_locked_out():
            # Audit failed login
            await log_audit_event(
                AuditEventType.LOGIN_FAILURE,
                f"Authentication failed - user inactive or locked: {username}",
                severity=AuditSeverity.MEDIUM,
                context=AuditContext(
                    user_id=user.user_id,
                    ip_address=ip_address,
                    user_agent=user_agent
                )
            )
            return None
        
        # In a real implementation, you would verify the password hash
        # For now, we'll assume password verification is handled elsewhere
        
        # Create session
        session = await self.create_session(user.user_id, ip_address, user_agent)
        return session
    
    def get_all_permissions(self) -> List[PermissionScope]:
        """Get all available permissions"""
        return list(PermissionScope)
    
    def get_all_roles(self) -> List[Role]:
        """Get all available roles"""
        return list(self.roles.values())
    
    def get_all_users(self) -> List[User]:
        """Get all users"""
        return list(self.users.values())
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        async with self.sessions_lock:
            expired_sessions = []
            for session_id, session in self.sessions.items():
                if session.is_expired():
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
                
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

# Permission decorators
def require_permission(permission_scope: PermissionScope, 
                     resource_type: ResourceType = None,
                     resource_id_param: str = None):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_id and context from function arguments
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None)
            context = kwargs.get('context', {})
            
            # Get resource_id from parameters if specified
            resource_id = None
            if resource_id_param:
                resource_id = kwargs.get(resource_id_param)
            
            # Check permission
            if not await rbac_system.has_permission(
                user_id=user_id,
                permission_scope=permission_scope,
                resource_type=resource_type,
                resource_id=resource_id,
                context=context
            ):
                # Audit permission denied
                await log_audit_event(
                    AuditEventType.PERMISSION_DENIED,
                    f"Permission denied: {permission_scope.value}",
                    severity=AuditSeverity.MEDIUM,
                    context=AuditContext(
                        user_id=user_id,
                        function=func.__name__,
                        component=func.__module__
                    )
                )
                raise PermissionError(f"Permission denied: {permission_scope.value}")
            
            # Audit permission granted
            await log_audit_event(
                AuditEventType.PERMISSION_GRANTED,
                f"Permission granted: {permission_scope.value}",
                severity=AuditSeverity.INFO,
                context=AuditContext(
                    user_id=user_id,
                    function=func.__name__,
                    component=func.__module__
                )
            )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def require_role(role: UserRole):
    """Decorator to require specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get('user_id') or getattr(args[0], 'user_id', None)
            
            user = await rbac_system.get_user(user_id)
            if not user or role.value not in user.roles:
                raise PermissionError(f"Role required: {role.value}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

# Global instance
rbac_system = RBACSystem()

# Utility functions
async def check_permission(user_id: str, permission_scope: PermissionScope, **kwargs) -> bool:
    """Check if user has permission"""
    return await rbac_system.has_permission(user_id, permission_scope, **kwargs)

async def get_user_permissions(user_id: str) -> Set[Permission]:
    """Get all effective permissions for user"""
    user = await rbac_system.get_user(user_id)
    if not user:
        return set()
    
    permissions = user.direct_permissions.copy()
    
    for role_name in user.roles:
        role = rbac_system.get_role(role_name)
        if role:
            permissions.update(role.get_effective_permissions(rbac_system))
    
    return permissions

async def create_admin_user(username: str, email: str, full_name: str) -> User:
    """Create admin user"""
    return await rbac_system.create_user(
        username=username,
        email=email,
        full_name=full_name,
        roles=[UserRole.ADMIN.value]
    )

# Context manager for permission checks
@asynccontextmanager
async def permission_context(user_id: str, permission_scope: PermissionScope, **kwargs):
    """Context manager for permission checks"""
    if not await rbac_system.has_permission(user_id, permission_scope, **kwargs):
        raise PermissionError(f"Permission denied: {permission_scope.value}")
    
    try:
        yield
    finally:
        pass

# Session management
async def authenticate_and_create_session(username: str, password: str, **kwargs) -> Optional[Session]:
    """Authenticate user and create session"""
    return await rbac_system.authenticate_user(username, password, **kwargs)

async def get_current_user_from_session(session_id: str) -> Optional[User]:
    """Get current user from session"""
    session = await rbac_system.get_session(session_id)
    if not session:
        return None
    
    return await rbac_system.get_user(session.user_id)

async def invalidate_user_session(session_id: str) -> bool:
    """Invalidate user session"""
    return await rbac_system.invalidate_session(session_id)
