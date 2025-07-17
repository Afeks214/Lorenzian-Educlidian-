"""
Comprehensive Financial Audit Logger
Specialized audit logging for financial trading operations with regulatory compliance
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import uuid
import hashlib
import hmac
from decimal import Decimal
from contextlib import asynccontextmanager

from src.monitoring.logger_config import get_logger
from src.security.encryption import encrypt_data, decrypt_data
from src.security.vault_integration import get_vault_secret

logger = get_logger(__name__)

class AuditEventType(Enum):
    """Financial audit event types"""
    # Authentication & Authorization
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    
    # Trading Operations
    TRADE_ORDER_CREATED = "trade_order_created"
    TRADE_ORDER_MODIFIED = "trade_order_modified"
    TRADE_ORDER_CANCELLED = "trade_order_cancelled"
    TRADE_EXECUTED = "trade_executed"
    TRADE_SETTLEMENT = "trade_settlement"
    
    # Risk Management
    RISK_LIMIT_SET = "risk_limit_set"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    RISK_OVERRIDE = "risk_override"
    POSITION_LIMIT_BREACHED = "position_limit_breached"
    
    # Portfolio Management
    PORTFOLIO_CREATED = "portfolio_created"
    PORTFOLIO_MODIFIED = "portfolio_modified"
    PORTFOLIO_DELETED = "portfolio_deleted"
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    
    # Data Access
    MARKET_DATA_ACCESSED = "market_data_accessed"
    PRICE_DATA_ACCESSED = "price_data_accessed"
    HISTORICAL_DATA_ACCESSED = "historical_data_accessed"
    
    # System Events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    
    # Compliance
    REGULATORY_REPORT_GENERATED = "regulatory_report_generated"
    COMPLIANCE_CHECK_PERFORMED = "compliance_check_performed"
    COMPLIANCE_VIOLATION = "compliance_violation"
    
    # Model Operations
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_COMPLETED = "model_training_completed"
    MODEL_INFERENCE_PERFORMED = "model_inference_performed"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_RETIRED = "model_retired"
    
    # Security Events
    SECURITY_SCAN_PERFORMED = "security_scan_performed"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    SECURITY_INCIDENT = "security_incident"
    ACCESS_CONTROL_CHANGED = "access_control_changed"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ComplianceFramework(Enum):
    """Regulatory compliance frameworks"""
    SOX = "sox"  # Sarbanes-Oxley
    SEC = "sec"  # Securities and Exchange Commission
    CFTC = "cftc"  # Commodity Futures Trading Commission
    FINRA = "finra"  # Financial Industry Regulatory Authority
    MiFID2 = "mifid2"  # Markets in Financial Instruments Directive
    GDPR = "gdpr"  # General Data Protection Regulation
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    BASEL_III = "basel_iii"  # Basel III banking regulations
    DODD_FRANK = "dodd_frank"  # Dodd-Frank Wall Street Reform

@dataclass
class AuditContext:
    """Context information for audit events"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    component: Optional[str] = None
    function: Optional[str] = None
    
@dataclass
class FinancialData:
    """Financial data for audit events"""
    account_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    instrument_id: Optional[str] = None
    quantity: Optional[Decimal] = None
    price: Optional[Decimal] = None
    value: Optional[Decimal] = None
    currency: Optional[str] = None
    market: Optional[str] = None
    order_type: Optional[str] = None
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if isinstance(value, Decimal):
                    result[key] = str(value)
                else:
                    result[key] = value
        return result

@dataclass
class AuditEvent:
    """Comprehensive audit event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType = AuditEventType.SYSTEM_STARTUP
    severity: AuditSeverity = AuditSeverity.INFO
    message: str = ""
    context: AuditContext = field(default_factory=AuditContext)
    financial_data: Optional[FinancialData] = None
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    encrypted_data: Optional[str] = None
    data_hash: Optional[str] = None
    correlation_id: Optional[str] = None
    success: bool = True
    error_details: Optional[str] = None
    risk_score: Optional[float] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.correlation_id is None:
            self.correlation_id = self.event_id
        
        # Generate data hash for integrity verification
        if self.data_hash is None:
            self.data_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of event data"""
        data_to_hash = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "message": self.message,
            "context": asdict(self.context),
            "financial_data": self.financial_data.to_dict() if self.financial_data else None,
            "success": self.success
        }
        
        data_str = json.dumps(data_to_hash, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": asdict(self.context),
            "financial_data": self.financial_data.to_dict() if self.financial_data else None,
            "compliance_frameworks": [cf.value for cf in self.compliance_frameworks],
            "encrypted_data": self.encrypted_data,
            "data_hash": self.data_hash,
            "correlation_id": self.correlation_id,
            "success": self.success,
            "error_details": self.error_details,
            "risk_score": self.risk_score,
            "additional_data": self.additional_data
        }
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using hash"""
        calculated_hash = self._calculate_hash()
        return calculated_hash == self.data_hash

class FinancialAuditLogger:
    """Comprehensive financial audit logger with regulatory compliance"""
    
    def __init__(self, 
                 log_directory: str = "logs/audit",
                 encrypt_sensitive_data: bool = True,
                 batch_size: int = 100,
                 flush_interval: int = 30):
        
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.encrypt_sensitive_data = encrypt_sensitive_data
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Event buffer for batching
        self.event_buffer: List[AuditEvent] = []
        self.buffer_lock = asyncio.Lock()
        
        # Background tasks
        self.flush_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Current log file
        self.current_log_file: Optional[Path] = None
        
        # Encryption key for sensitive data
        self.encryption_key_id = "audit_encryption_key"
        
        # Compliance mappings
        self.compliance_event_mapping = {
            ComplianceFramework.SOX: [
                AuditEventType.TRADE_EXECUTED,
                AuditEventType.TRADE_SETTLEMENT,
                AuditEventType.PORTFOLIO_REBALANCED,
                AuditEventType.RISK_OVERRIDE,
                AuditEventType.REGULATORY_REPORT_GENERATED
            ],
            ComplianceFramework.SEC: [
                AuditEventType.TRADE_ORDER_CREATED,
                AuditEventType.TRADE_ORDER_MODIFIED,
                AuditEventType.TRADE_ORDER_CANCELLED,
                AuditEventType.TRADE_EXECUTED,
                AuditEventType.MARKET_DATA_ACCESSED
            ],
            ComplianceFramework.CFTC: [
                AuditEventType.TRADE_EXECUTED,
                AuditEventType.POSITION_LIMIT_BREACHED,
                AuditEventType.RISK_LIMIT_BREACHED
            ],
            ComplianceFramework.FINRA: [
                AuditEventType.TRADE_ORDER_CREATED,
                AuditEventType.TRADE_EXECUTED,
                AuditEventType.COMPLIANCE_CHECK_PERFORMED
            ],
            ComplianceFramework.MiFID2: [
                AuditEventType.TRADE_ORDER_CREATED,
                AuditEventType.TRADE_EXECUTED,
                AuditEventType.MARKET_DATA_ACCESSED
            ],
            ComplianceFramework.GDPR: [
                AuditEventType.LOGIN_SUCCESS,
                AuditEventType.LOGIN_FAILURE,
                AuditEventType.MARKET_DATA_ACCESSED,
                AuditEventType.HISTORICAL_DATA_ACCESSED
            ]
        }
        
        logger.info("FinancialAuditLogger initialized", 
                   log_directory=str(self.log_directory),
                   encrypt_sensitive_data=self.encrypt_sensitive_data)
    
    async def start(self):
        """Start the audit logger"""
        if self.running:
            return
        
        self.running = True
        self.flush_task = asyncio.create_task(self._flush_loop())
        
        # Log startup event
        await self.log_event(
            AuditEventType.SYSTEM_STARTUP,
            "Financial audit logger started",
            AuditSeverity.INFO
        )
        
        logger.info("FinancialAuditLogger started")
    
    async def stop(self):
        """Stop the audit logger"""
        if not self.running:
            return
        
        # Log shutdown event
        await self.log_event(
            AuditEventType.SYSTEM_SHUTDOWN,
            "Financial audit logger stopping",
            AuditSeverity.INFO
        )
        
        # Flush remaining events
        await self._flush_events()
        
        # Cancel flush task
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        self.running = False
        logger.info("FinancialAuditLogger stopped")
    
    async def log_event(self,
                       event_type: AuditEventType,
                       message: str,
                       severity: AuditSeverity = AuditSeverity.INFO,
                       context: Optional[AuditContext] = None,
                       financial_data: Optional[FinancialData] = None,
                       correlation_id: Optional[str] = None,
                       success: bool = True,
                       error_details: Optional[str] = None,
                       risk_score: Optional[float] = None,
                       additional_data: Dict[str, Any] = None) -> str:
        """Log a financial audit event"""
        
        # Determine applicable compliance frameworks
        compliance_frameworks = []
        for framework, event_types in self.compliance_event_mapping.items():
            if event_type in event_types:
                compliance_frameworks.append(framework)
        
        # Create audit event
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            context=context or AuditContext(),
            financial_data=financial_data,
            compliance_frameworks=compliance_frameworks,
            correlation_id=correlation_id,
            success=success,
            error_details=error_details,
            risk_score=risk_score,
            additional_data=additional_data or {}
        )
        
        # Encrypt sensitive data if enabled
        if self.encrypt_sensitive_data and financial_data:
            try:
                sensitive_data = json.dumps(financial_data.to_dict())
                encrypted_data = encrypt_data(sensitive_data, self.encryption_key_id)
                event.encrypted_data = encrypted_data.to_dict()
            except Exception as e:
                logger.error(f"Failed to encrypt audit data: {e}")
        
        # Add to buffer
        async with self.buffer_lock:
            self.event_buffer.append(event)
            
            # Flush if buffer is full
            if len(self.event_buffer) >= self.batch_size:
                await self._flush_events()
        
        # Log to standard logger for immediate visibility
        log_method = getattr(logger, severity.value.lower(), logger.info)
        log_method(f"AUDIT: {message}", 
                  event_type=event_type.value,
                  event_id=event.event_id,
                  correlation_id=correlation_id,
                  success=success)
        
        return event.event_id
    
    async def _flush_events(self):
        """Flush events from buffer to storage"""
        if not self.event_buffer:
            return
        
        events_to_flush = self.event_buffer.copy()
        self.event_buffer.clear()
        
        # Get current log file
        log_file = self._get_current_log_file()
        
        try:
            # Write events to file
            with open(log_file, 'a') as f:
                for event in events_to_flush:
                    f.write(json.dumps(event.to_dict()) + '\n')
            
            logger.debug(f"Flushed {len(events_to_flush)} audit events to {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")
            # Return events to buffer for retry
            async with self.buffer_lock:
                self.event_buffer.extend(events_to_flush)
    
    async def _flush_loop(self):
        """Background task to flush events periodically"""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                async with self.buffer_lock:
                    if self.event_buffer:
                        await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush loop error: {e}")
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path"""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_directory / f"financial_audit_{today}.jsonl"
        
        if self.current_log_file != log_file:
            self.current_log_file = log_file
            # Create new log file with header
            if not log_file.exists():
                self._create_log_file_header(log_file)
        
        return log_file
    
    def _create_log_file_header(self, log_file: Path):
        """Create log file with header information"""
        header = {
            "log_type": "financial_audit",
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "encryption_enabled": self.encrypt_sensitive_data,
            "compliance_frameworks": [cf.value for cf in ComplianceFramework]
        }
        
        try:
            with open(log_file, 'w') as f:
                f.write(json.dumps(header) + '\n')
        except Exception as e:
            logger.error(f"Failed to create log file header: {e}")
    
    @asynccontextmanager
    async def correlation_context(self, correlation_id: str):
        """Context manager for correlated audit events"""
        old_correlation_id = getattr(self, '_current_correlation_id', None)
        self._current_correlation_id = correlation_id
        try:
            yield correlation_id
        finally:
            self._current_correlation_id = old_correlation_id
    
    async def log_trading_event(self,
                              event_type: AuditEventType,
                              message: str,
                              account_id: str,
                              instrument_id: str,
                              quantity: Decimal,
                              price: Decimal,
                              order_id: Optional[str] = None,
                              trade_id: Optional[str] = None,
                              context: Optional[AuditContext] = None,
                              correlation_id: Optional[str] = None) -> str:
        """Log trading-specific audit event"""
        
        financial_data = FinancialData(
            account_id=account_id,
            instrument_id=instrument_id,
            quantity=quantity,
            price=price,
            value=quantity * price,
            order_id=order_id,
            trade_id=trade_id
        )
        
        return await self.log_event(
            event_type=event_type,
            message=message,
            severity=AuditSeverity.MEDIUM,
            context=context,
            financial_data=financial_data,
            correlation_id=correlation_id or getattr(self, '_current_correlation_id', None)
        )
    
    async def log_risk_event(self,
                           event_type: AuditEventType,
                           message: str,
                           risk_score: float,
                           account_id: Optional[str] = None,
                           portfolio_id: Optional[str] = None,
                           context: Optional[AuditContext] = None,
                           correlation_id: Optional[str] = None) -> str:
        """Log risk management audit event"""
        
        financial_data = None
        if account_id or portfolio_id:
            financial_data = FinancialData(
                account_id=account_id,
                portfolio_id=portfolio_id
            )
        
        severity = AuditSeverity.HIGH if risk_score > 0.8 else AuditSeverity.MEDIUM
        
        return await self.log_event(
            event_type=event_type,
            message=message,
            severity=severity,
            context=context,
            financial_data=financial_data,
            risk_score=risk_score,
            correlation_id=correlation_id or getattr(self, '_current_correlation_id', None)
        )
    
    async def log_compliance_event(self,
                                 event_type: AuditEventType,
                                 message: str,
                                 compliance_framework: ComplianceFramework,
                                 context: Optional[AuditContext] = None,
                                 additional_data: Dict[str, Any] = None,
                                 correlation_id: Optional[str] = None) -> str:
        """Log compliance-specific audit event"""
        
        event = AuditEvent(
            event_type=event_type,
            severity=AuditSeverity.HIGH,
            message=message,
            context=context or AuditContext(),
            compliance_frameworks=[compliance_framework],
            correlation_id=correlation_id or getattr(self, '_current_correlation_id', None),
            additional_data=additional_data or {}
        )
        
        async with self.buffer_lock:
            self.event_buffer.append(event)
            
            if len(self.event_buffer) >= self.batch_size:
                await self._flush_events()
        
        return event.event_id
    
    async def log_authentication_event(self,
                                     event_type: AuditEventType,
                                     user_id: str,
                                     success: bool,
                                     ip_address: str,
                                     user_agent: str,
                                     session_id: Optional[str] = None,
                                     error_details: Optional[str] = None) -> str:
        """Log authentication audit event"""
        
        context = AuditContext(
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM
        message = f"User {user_id} {event_type.value.replace('_', ' ')}"
        
        return await self.log_event(
            event_type=event_type,
            message=message,
            severity=severity,
            context=context,
            success=success,
            error_details=error_details
        )
    
    async def log_model_event(self,
                            event_type: AuditEventType,
                            model_name: str,
                            model_version: str,
                            context: Optional[AuditContext] = None,
                            additional_data: Dict[str, Any] = None,
                            correlation_id: Optional[str] = None) -> str:
        """Log model operation audit event"""
        
        message = f"Model {model_name} v{model_version} {event_type.value.replace('_', ' ')}"
        
        event_data = {
            "model_name": model_name,
            "model_version": model_version,
            **(additional_data or {})
        }
        
        return await self.log_event(
            event_type=event_type,
            message=message,
            severity=AuditSeverity.INFO,
            context=context,
            correlation_id=correlation_id or getattr(self, '_current_correlation_id', None),
            additional_data=event_data
        )
    
    async def search_events(self,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          event_types: Optional[List[AuditEventType]] = None,
                          user_id: Optional[str] = None,
                          correlation_id: Optional[str] = None,
                          limit: int = 1000) -> List[AuditEvent]:
        """Search audit events with filters"""
        
        events = []
        
        # Determine date range for log files
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Iterate through log files in date range
        current_date = start_date.date()
        while current_date <= end_date.date():
            log_file = self.log_directory / f"financial_audit_{current_date.strftime('%Y-%m-%d')}.jsonl"
            
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            
                            try:
                                event_data = json.loads(line)
                                
                                # Skip header line
                                if event_data.get("log_type") == "financial_audit":
                                    continue
                                
                                # Parse event
                                event_timestamp = datetime.fromisoformat(event_data["timestamp"])
                                
                                # Apply filters
                                if event_timestamp < start_date or event_timestamp > end_date:
                                    continue
                                
                                if event_types and AuditEventType(event_data["event_type"]) not in event_types:
                                    continue
                                
                                if user_id and event_data.get("context", {}).get("user_id") != user_id:
                                    continue
                                
                                if correlation_id and event_data.get("correlation_id") != correlation_id:
                                    continue
                                
                                # Create event object
                                event = AuditEvent(
                                    event_id=event_data["event_id"],
                                    timestamp=event_timestamp,
                                    event_type=AuditEventType(event_data["event_type"]),
                                    severity=AuditSeverity(event_data["severity"]),
                                    message=event_data["message"],
                                    context=AuditContext(**event_data.get("context", {})),
                                    financial_data=FinancialData(**event_data["financial_data"]) if event_data.get("financial_data") else None,
                                    compliance_frameworks=[ComplianceFramework(cf) for cf in event_data.get("compliance_frameworks", [])],
                                    correlation_id=event_data.get("correlation_id"),
                                    success=event_data.get("success", True),
                                    error_details=event_data.get("error_details"),
                                    risk_score=event_data.get("risk_score"),
                                    additional_data=event_data.get("additional_data", {})
                                )
                                
                                events.append(event)
                                
                                if len(events) >= limit:
                                    break
                                    
                            except (json.JSONDecodeError, KeyError, ValueError) as e:
                                logger.warning(f"Failed to parse audit event: {e}")
                                continue
                                
                except Exception as e:
                    logger.error(f"Failed to read audit log file {log_file}: {e}")
            
            current_date += timedelta(days=1)
            
            if len(events) >= limit:
                break
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return events[:limit]
    
    async def generate_compliance_report(self,
                                       compliance_framework: ComplianceFramework,
                                       start_date: datetime,
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        
        # Get relevant event types for this framework
        relevant_event_types = self.compliance_event_mapping.get(compliance_framework, [])
        
        # Search for events
        events = await self.search_events(
            start_date=start_date,
            end_date=end_date,
            event_types=relevant_event_types,
            limit=10000
        )
        
        # Analyze events
        total_events = len(events)
        event_type_counts = {}
        severity_counts = {}
        error_count = 0
        
        for event in events:
            event_type_counts[event.event_type.value] = event_type_counts.get(event.event_type.value, 0) + 1
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
            
            if not event.success:
                error_count += 1
        
        # Generate report
        report = {
            "compliance_framework": compliance_framework.value,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": total_events,
                "error_count": error_count,
                "error_rate": error_count / total_events if total_events > 0 else 0
            },
            "event_type_breakdown": event_type_counts,
            "severity_breakdown": severity_counts,
            "compliance_status": "COMPLIANT" if error_count == 0 else "NON_COMPLIANT",
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return report
    
    async def verify_audit_integrity(self, 
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Verify audit log integrity"""
        
        events = await self.search_events(start_date=start_date, end_date=end_date, limit=10000)
        
        total_events = len(events)
        integrity_failures = 0
        
        for event in events:
            if not event.verify_integrity():
                integrity_failures += 1
        
        integrity_status = "VERIFIED" if integrity_failures == 0 else "COMPROMISED"
        
        return {
            "total_events_checked": total_events,
            "integrity_failures": integrity_failures,
            "integrity_rate": (total_events - integrity_failures) / total_events if total_events > 0 else 0,
            "status": integrity_status,
            "verified_at": datetime.now(timezone.utc).isoformat()
        }

# Global instance
financial_audit_logger = FinancialAuditLogger()

# Utility functions
async def log_audit_event(event_type: AuditEventType, message: str, **kwargs) -> str:
    """Log audit event using global logger"""
    return await financial_audit_logger.log_event(event_type, message, **kwargs)

async def log_trading_event(event_type: AuditEventType, message: str, **kwargs) -> str:
    """Log trading event using global logger"""
    return await financial_audit_logger.log_trading_event(event_type, message, **kwargs)

async def log_risk_event(event_type: AuditEventType, message: str, **kwargs) -> str:
    """Log risk event using global logger"""
    return await financial_audit_logger.log_risk_event(event_type, message, **kwargs)

async def log_compliance_event(event_type: AuditEventType, message: str, **kwargs) -> str:
    """Log compliance event using global logger"""
    return await financial_audit_logger.log_compliance_event(event_type, message, **kwargs)

async def log_authentication_event(event_type: AuditEventType, user_id: str, success: bool, **kwargs) -> str:
    """Log authentication event using global logger"""
    return await financial_audit_logger.log_authentication_event(event_type, user_id, success, **kwargs)

async def log_model_event(event_type: AuditEventType, model_name: str, model_version: str, **kwargs) -> str:
    """Log model event using global logger"""
    return await financial_audit_logger.log_model_event(event_type, model_name, model_version, **kwargs)

# Audit decorators
def audit_trading_operation(event_type: AuditEventType):
    """Decorator to audit trading operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            correlation_id = str(uuid.uuid4())
            
            try:
                # Log start of operation
                await log_audit_event(
                    event_type,
                    f"Started {func.__name__}",
                    correlation_id=correlation_id
                )
                
                # Execute operation
                result = await func(*args, **kwargs)
                
                # Log success
                await log_audit_event(
                    event_type,
                    f"Completed {func.__name__} successfully",
                    correlation_id=correlation_id,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Log failure
                await log_audit_event(
                    event_type,
                    f"Failed {func.__name__}",
                    correlation_id=correlation_id,
                    success=False,
                    error_details=str(e)
                )
                raise
        
        return wrapper
    return decorator

# Context managers
@asynccontextmanager
async def audit_context(correlation_id: str):
    """Context manager for correlated audit events"""
    async with financial_audit_logger.correlation_context(correlation_id):
        yield correlation_id

# Initialize on import
async def initialize_financial_audit_logger():
    """Initialize the global financial audit logger"""
    await financial_audit_logger.start()

# Cleanup on shutdown
async def shutdown_financial_audit_logger():
    """Shutdown the global financial audit logger"""
    await financial_audit_logger.stop()
