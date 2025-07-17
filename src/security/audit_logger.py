"""
Comprehensive Audit Logging System
SOX, GDPR, and regulatory compliance audit trails
"""

import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
import redis.asyncio as redis
from fastapi import Request, Response
from pydantic import BaseModel
import asyncio
from pathlib import Path

from src.monitoring.logger_config import get_logger
from src.security.encryption import encrypt_data, decrypt_data, EncryptedData
from src.security.enterprise_auth import TokenData

logger = get_logger(__name__)

class AuditEventType(str, Enum):
    """Types of audit events"""
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLE = "mfa_enable"
    MFA_DISABLE = "mfa_disable"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"
    ROLE_CHANGE = "role_change"
    
    # Trading events
    TRADE_EXECUTED = "trade_executed"
    TRADE_CANCELLED = "trade_cancelled"
    TRADE_MODIFIED = "trade_modified"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Risk management events
    RISK_LIMIT_BREACH = "risk_limit_breach"
    RISK_LIMIT_CHANGE = "risk_limit_change"
    EMERGENCY_STOP = "emergency_stop"
    VAR_CALCULATION = "var_calculation"
    
    # Data events
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    DATA_DELETION = "data_deletion"
    DATA_MODIFICATION = "data_modification"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    MODEL_RELOAD = "model_reload"
    
    # Compliance events
    COMPLIANCE_REPORT = "compliance_report"
    REGULATORY_SUBMISSION = "regulatory_submission"
    AUDIT_REQUEST = "audit_request"
    
    # API events
    API_CALL = "api_call"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"

class AuditSeverity(str, Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComplianceFramework(str, Enum):
    """Compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    MiFID = "mifid"
    BASEL = "basel"
    DODD_FRANK = "dodd_frank"
    FINRA = "finra"
    INTERNAL = "internal"

@dataclass
class AuditEvent:
    """Audit event model"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    data_before: Optional[Dict[str, Any]] = None
    data_after: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        data["compliance_frameworks"] = [cf.value for cf in self.compliance_frameworks]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["event_type"] = AuditEventType(data["event_type"])
        data["severity"] = AuditSeverity(data["severity"])
        data["compliance_frameworks"] = [ComplianceFramework(cf) for cf in data["compliance_frameworks"]]
        return cls(**data)

class DataMasking:
    """Data masking for sensitive information in audit logs"""
    
    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email address"""
        if not email or '@' not in email:
            return email
        
        username, domain = email.split('@', 1)
        if len(username) <= 2:
            masked_username = '*' * len(username)
        else:
            masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
        
        return f"{masked_username}@{domain}"
    
    @staticmethod
    def mask_ip(ip: str) -> str:
        """Mask IP address"""
        if not ip:
            return ip
        
        # IPv4
        if '.' in ip:
            parts = ip.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.xxx.xxx"
        
        # IPv6
        if ':' in ip:
            parts = ip.split(':')
            if len(parts) >= 4:
                return f"{parts[0]}:{parts[1]}:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx"
        
        return "xxx.xxx.xxx.xxx"
    
    @staticmethod
    def mask_credit_card(card_number: str) -> str:
        """Mask credit card number"""
        if not card_number:
            return card_number
        
        # Remove spaces and hyphens
        clean_number = card_number.replace(' ', '').replace('-', '')
        
        if len(clean_number) >= 4:
            return '*' * (len(clean_number) - 4) + clean_number[-4:]
        
        return '*' * len(clean_number)
    
    @staticmethod
    def mask_ssn(ssn: str) -> str:
        """Mask Social Security Number"""
        if not ssn:
            return ssn
        
        # Remove hyphens
        clean_ssn = ssn.replace('-', '')
        
        if len(clean_ssn) == 9:
            return f"xxx-xx-{clean_ssn[-4:]}"
        
        return 'xxx-xx-xxxx'
    
    @staticmethod
    def mask_account_number(account: str) -> str:
        """Mask account number"""
        if not account or len(account) < 4:
            return '*' * len(account)
        
        return '*' * (len(account) - 4) + account[-4:]
    
    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """Mask sensitive fields in data"""
        masked_data = data.copy()
        
        for field in sensitive_fields:
            if field in masked_data:
                value = masked_data[field]
                if isinstance(value, str):
                    if 'email' in field.lower():
                        masked_data[field] = DataMasking.mask_email(value)
                    elif 'ip' in field.lower() or 'address' in field.lower():
                        masked_data[field] = DataMasking.mask_ip(value)
                    elif 'card' in field.lower() or 'credit' in field.lower():
                        masked_data[field] = DataMasking.mask_credit_card(value)
                    elif 'ssn' in field.lower() or 'social' in field.lower():
                        masked_data[field] = DataMasking.mask_ssn(value)
                    elif 'account' in field.lower():
                        masked_data[field] = DataMasking.mask_account_number(value)
                    else:
                        # Generic masking
                        if len(value) > 4:
                            masked_data[field] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                        else:
                            masked_data[field] = '*' * len(value)
        
        return masked_data

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.log_dir = Path("/var/log/grandmodel/audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Sensitive fields to mask
        self.sensitive_fields = [
            'password', 'password_hash', 'api_key', 'secret', 'token',
            'email', 'ssn', 'credit_card', 'account_number', 'ip_address',
            'personal_info', 'pii', 'financial_data'
        ]
        
        # Queue for batch processing
        self.event_queue = asyncio.Queue()
        self.batch_size = 100
        self.batch_timeout = 30  # seconds
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Audit logger initialized")
    
    def _start_background_tasks(self):
        """Start background audit processing tasks"""
        asyncio.create_task(self._process_audit_queue())
    
    async def _process_audit_queue(self):
        """Process audit events from queue"""
        batch = []
        last_flush = datetime.utcnow()
        
        while True:
            try:
                # Get event with timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    batch.append(event)
                except asyncio.TimeoutError:
                    pass
                
                # Flush batch if size limit reached or timeout
                now = datetime.utcnow()
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and (now - last_flush).seconds >= self.batch_timeout)
                )
                
                if should_flush:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = now
                    
            except Exception as e:
                logger.error("Error processing audit queue", error=str(e))
                await asyncio.sleep(1)
    
    async def _flush_batch(self, batch: List[AuditEvent]):
        """Flush batch of audit events"""
        if not batch:
            return
        
        try:
            # Store in Redis
            if self.redis:
                await self._store_batch_redis(batch)
            
            # Store in files
            await self._store_batch_files(batch)
            
            logger.info(f"Flushed {len(batch)} audit events")
            
        except Exception as e:
            logger.error("Error flushing audit batch", error=str(e))
    
    async def _store_batch_redis(self, batch: List[AuditEvent]):
        """Store batch in Redis"""
        pipe = self.redis.pipeline()
        
        for event in batch:
            # Encrypt sensitive data
            encrypted_data = encrypt_data(event.to_dict())
            
            # Store with TTL (keep for 7 years for compliance)
            ttl = 7 * 365 * 24 * 3600  # 7 years in seconds
            pipe.setex(f"audit:{event.event_id}", ttl, encrypted_data.to_dict())
            
            # Add to time-based index
            date_key = event.timestamp.strftime("%Y-%m-%d")
            pipe.zadd(f"audit_index:{date_key}", {event.event_id: event.timestamp.timestamp()})
            
            # Add to type-based index
            pipe.sadd(f"audit_type:{event.event_type.value}", event.event_id)
            
            # Add to user-based index
            if event.user_id:
                pipe.sadd(f"audit_user:{event.user_id}", event.event_id)
            
            # Add to compliance framework index
            for framework in event.compliance_frameworks:
                pipe.sadd(f"audit_compliance:{framework.value}", event.event_id)
        
        await pipe.execute()
    
    async def _store_batch_files(self, batch: List[AuditEvent]):
        """Store batch in log files"""
        # Group by date
        date_batches = {}
        for event in batch:
            date_key = event.timestamp.strftime("%Y-%m-%d")
            if date_key not in date_batches:
                date_batches[date_key] = []
            date_batches[date_key].append(event)
        
        # Write to files
        for date_key, events in date_batches.items():
            log_file = self.log_dir / f"audit_{date_key}.log"
            
            with open(log_file, 'a') as f:
                for event in events:
                    # Mask sensitive data for file storage
                    masked_data = DataMasking.mask_sensitive_data(
                        event.to_dict(), 
                        self.sensitive_fields
                    )
                    f.write(json.dumps(masked_data) + '\n')
    
    async def log_event(self, 
                       event_type: AuditEventType,
                       severity: AuditSeverity = AuditSeverity.MEDIUM,
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       resource: Optional[str] = None,
                       action: Optional[str] = None,
                       outcome: str = "success",
                       details: Optional[Dict[str, Any]] = None,
                       compliance_frameworks: Optional[List[ComplianceFramework]] = None,
                       data_before: Optional[Dict[str, Any]] = None,
                       data_after: Optional[Dict[str, Any]] = None,
                       risk_score: Optional[float] = None,
                       tags: Optional[List[str]] = None,
                       correlation_id: Optional[str] = None,
                       parent_event_id: Optional[str] = None):
        """Log audit event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            compliance_frameworks=compliance_frameworks or [],
            data_before=data_before,
            data_after=data_after,
            risk_score=risk_score,
            tags=tags or [],
            correlation_id=correlation_id,
            parent_event_id=parent_event_id
        )
        
        # Add to queue for processing
        await self.event_queue.put(event)
        
        logger.info(
            "Audit event logged",
            event_id=event.event_id,
            event_type=event_type.value,
            severity=severity.value,
            user_id=user_id
        )
    
    async def log_authentication_event(self, 
                                     event_type: AuditEventType,
                                     user_id: Optional[str] = None,
                                     ip_address: Optional[str] = None,
                                     user_agent: Optional[str] = None,
                                     outcome: str = "success",
                                     details: Optional[Dict[str, Any]] = None):
        """Log authentication event"""
        await self.log_event(
            event_type=event_type,
            severity=AuditSeverity.HIGH if outcome == "failure" else AuditSeverity.MEDIUM,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="authentication",
            action=event_type.value,
            outcome=outcome,
            details=details,
            compliance_frameworks=[ComplianceFramework.SOX, ComplianceFramework.INTERNAL],
            tags=["authentication", "security"]
        )
    
    async def log_trading_event(self, 
                              event_type: AuditEventType,
                              user_id: str,
                              trade_data: Dict[str, Any],
                              outcome: str = "success",
                              session_id: Optional[str] = None):
        """Log trading event"""
        await self.log_event(
            event_type=event_type,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            session_id=session_id,
            resource="trading",
            action=event_type.value,
            outcome=outcome,
            details=trade_data,
            compliance_frameworks=[
                ComplianceFramework.SOX, 
                ComplianceFramework.MiFID, 
                ComplianceFramework.FINRA
            ],
            tags=["trading", "financial"]
        )
    
    async def log_data_access_event(self, 
                                  user_id: str,
                                  resource: str,
                                  action: str,
                                  outcome: str = "success",
                                  data_classification: str = "internal",
                                  session_id: Optional[str] = None):
        """Log data access event"""
        severity = AuditSeverity.HIGH if data_classification == "confidential" else AuditSeverity.MEDIUM
        
        await self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            resource=resource,
            action=action,
            outcome=outcome,
            details={"data_classification": data_classification},
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOX],
            tags=["data_access", "privacy"]
        )
    
    async def log_risk_event(self, 
                           event_type: AuditEventType,
                           user_id: str,
                           risk_data: Dict[str, Any],
                           risk_score: float,
                           outcome: str = "success"):
        """Log risk management event"""
        await self.log_event(
            event_type=event_type,
            severity=AuditSeverity.CRITICAL if risk_score > 0.8 else AuditSeverity.HIGH,
            user_id=user_id,
            resource="risk_management",
            action=event_type.value,
            outcome=outcome,
            details=risk_data,
            risk_score=risk_score,
            compliance_frameworks=[
                ComplianceFramework.SOX, 
                ComplianceFramework.BASEL, 
                ComplianceFramework.DODD_FRANK
            ],
            tags=["risk_management", "compliance"]
        )
    
    async def search_audit_events(self, 
                                start_date: datetime,
                                end_date: datetime,
                                event_type: Optional[AuditEventType] = None,
                                user_id: Optional[str] = None,
                                compliance_framework: Optional[ComplianceFramework] = None,
                                limit: int = 100) -> List[AuditEvent]:
        """Search audit events"""
        events = []
        
        if not self.redis:
            return events
        
        try:
            # Get event IDs from time-based index
            current_date = start_date
            event_ids = set()
            
            while current_date <= end_date:
                date_key = current_date.strftime("%Y-%m-%d")
                
                # Get events for this date
                daily_events = await self.redis.zrangebyscore(
                    f"audit_index:{date_key}",
                    start_date.timestamp(),
                    end_date.timestamp()
                )
                
                event_ids.update(daily_events)
                current_date += timedelta(days=1)
            
            # Filter by event type
            if event_type:
                type_events = await self.redis.smembers(f"audit_type:{event_type.value}")
                event_ids = event_ids.intersection(type_events)
            
            # Filter by user
            if user_id:
                user_events = await self.redis.smembers(f"audit_user:{user_id}")
                event_ids = event_ids.intersection(user_events)
            
            # Filter by compliance framework
            if compliance_framework:
                compliance_events = await self.redis.smembers(f"audit_compliance:{compliance_framework.value}")
                event_ids = event_ids.intersection(compliance_events)
            
            # Get event data
            for event_id in list(event_ids)[:limit]:
                encrypted_data = await self.redis.get(f"audit:{event_id}")
                if encrypted_data:
                    try:
                        # Decrypt and parse event
                        decrypted_data = decrypt_data(EncryptedData.from_dict(json.loads(encrypted_data)))
                        event_data = json.loads(decrypted_data.decode())
                        events.append(AuditEvent.from_dict(event_data))
                    except Exception as e:
                        logger.error(f"Error decrypting audit event {event_id}", error=str(e))
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            logger.error("Error searching audit events", error=str(e))
        
        return events
    
    async def generate_compliance_report(self, 
                                       compliance_framework: ComplianceFramework,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report"""
        events = await self.search_audit_events(
            start_date=start_date,
            end_date=end_date,
            compliance_framework=compliance_framework
        )
        
        # Analyze events
        event_types = {}
        severities = {}
        outcomes = {}
        users = {}
        
        for event in events:
            # Count by type
            event_types[event.event_type.value] = event_types.get(event.event_type.value, 0) + 1
            
            # Count by severity
            severities[event.severity.value] = severities.get(event.severity.value, 0) + 1
            
            # Count by outcome
            outcomes[event.outcome] = outcomes.get(event.outcome, 0) + 1
            
            # Count by user
            if event.user_id:
                users[event.user_id] = users.get(event.user_id, 0) + 1
        
        report = {
            "framework": compliance_framework.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(events),
            "event_types": event_types,
            "severities": severities,
            "outcomes": outcomes,
            "top_users": dict(sorted(users.items(), key=lambda x: x[1], reverse=True)[:10]),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Log report generation
        await self.log_event(
            event_type=AuditEventType.COMPLIANCE_REPORT,
            severity=AuditSeverity.MEDIUM,
            resource="compliance",
            action="generate_report",
            details={
                "framework": compliance_framework.value,
                "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
                "event_count": len(events)
            },
            compliance_frameworks=[compliance_framework]
        )
        
        return report

# Global audit logger instance
audit_logger: Optional[AuditLogger] = None

async def get_audit_logger() -> AuditLogger:
    """Get or create audit logger instance"""
    global audit_logger
    
    if audit_logger is None:
        # Initialize Redis client
        redis_client = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                redis_client = await redis.from_url(redis_url)
            except Exception as e:
                logger.error("Failed to connect to Redis for audit logging", error=str(e))
        
        audit_logger = AuditLogger(redis_client)
    
    return audit_logger

# FastAPI middleware for automatic audit logging
async def audit_middleware(request: Request, call_next):
    """Middleware to automatically log API requests"""
    start_time = datetime.utcnow()
    correlation_id = str(uuid.uuid4())
    
    # Extract user info from request
    user_id = None
    session_id = None
    
    if hasattr(request.state, 'user'):
        user_data = request.state.user
        if isinstance(user_data, TokenData):
            user_id = user_data.user_id
            session_id = user_data.session_id
    
    # Process request
    response = await call_next(request)
    
    # Log the API call
    logger_instance = await get_audit_logger()
    
    processing_time = (datetime.utcnow() - start_time).total_seconds()
    
    await logger_instance.log_event(
        event_type=AuditEventType.API_CALL,
        severity=AuditSeverity.LOW,
        user_id=user_id,
        session_id=session_id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        resource=request.url.path,
        action=request.method,
        outcome="success" if response.status_code < 400 else "failure",
        details={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers)
        },
        correlation_id=correlation_id,
        tags=["api", "http"]
    )
    
    # Add correlation ID to response
    response.headers["X-Correlation-ID"] = correlation_id
    
    return response

# Convenience functions
async def log_authentication_event(event_type: AuditEventType, 
                                  user_id: Optional[str] = None,
                                  ip_address: Optional[str] = None,
                                  outcome: str = "success",
                                  details: Optional[Dict[str, Any]] = None):
    """Log authentication event"""
    logger_instance = await get_audit_logger()
    await logger_instance.log_authentication_event(
        event_type=event_type,
        user_id=user_id,
        ip_address=ip_address,
        outcome=outcome,
        details=details
    )

async def log_trading_event(event_type: AuditEventType,
                          user_id: str,
                          trade_data: Dict[str, Any],
                          outcome: str = "success"):
    """Log trading event"""
    logger_instance = await get_audit_logger()
    await logger_instance.log_trading_event(
        event_type=event_type,
        user_id=user_id,
        trade_data=trade_data,
        outcome=outcome
    )

async def log_data_access_event(user_id: str,
                              resource: str,
                              action: str,
                              outcome: str = "success"):
    """Log data access event"""
    logger_instance = await get_audit_logger()
    await logger_instance.log_data_access_event(
        user_id=user_id,
        resource=resource,
        action=action,
        outcome=outcome
    )
