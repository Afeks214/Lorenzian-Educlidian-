"""
Enhanced Audit Logger for Enterprise-Grade Compliance
Agent 8 Mission: Comprehensive Audit Trail and Logging System

This module provides enterprise-grade audit logging capabilities with:
- Comprehensive event tracking
- Blockchain-based integrity verification
- Real-time compliance monitoring
- Regulatory reporting compliance
- Automated threat detection
"""

import json
import hashlib
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import structlog
from pathlib import Path
import sqlite3
import uuid
import threading
from contextlib import contextmanager
import os
import gzip
import pickle
from concurrent.futures import ThreadPoolExecutor
import queue

# Import blockchain audit if available
try:
    from ..xai.audit.blockchain_audit import BlockchainAudit
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False

logger = structlog.get_logger()


class AuditLevel(Enum):
    """Audit logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    FINANCIAL = "financial"


class AuditCategory(Enum):
    """Audit event categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    TRADING = "trading"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    DATA_ACCESS = "data_access"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks"""
    FINRA = "finra"
    SEC = "sec"
    GDPR = "gdpr"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    BASEL_III = "basel_iii"
    MIFID2 = "mifid2"
    DODD_FRANK = "dodd_frank"
    COMPANY_POLICY = "company_policy"


@dataclass
class AuditMetadata:
    """Metadata for audit events"""
    user_id: str
    session_id: str
    ip_address: str
    user_agent: str
    system_component: str
    correlation_id: str
    request_id: str
    trace_id: str
    geo_location: Optional[str] = None
    device_info: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceContext:
    """Compliance-specific context"""
    framework: ComplianceFramework
    requirement_id: str
    control_id: str
    risk_level: str
    impact_assessment: str
    data_classification: str
    retention_period: int  # days
    regulatory_tags: List[str] = field(default_factory=list)


@dataclass
class EnhancedAuditEvent:
    """Enhanced audit event with enterprise features"""
    event_id: str
    event_type: str
    category: AuditCategory
    level: AuditLevel
    timestamp: datetime
    metadata: AuditMetadata
    message: str
    details: Dict[str, Any]
    
    # Compliance fields
    compliance_context: Optional[ComplianceContext] = None
    regulatory_impact: bool = False
    pii_involved: bool = False
    financial_impact: bool = False
    
    # Security fields
    security_classification: str = "internal"
    threat_indicators: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    
    # Integrity fields
    checksum: str = field(default="")
    digital_signature: str = field(default="")
    blockchain_hash: str = field(default="")
    
    # Chain fields
    previous_event_id: Optional[str] = None
    chain_sequence: int = 0
    
    def __post_init__(self):
        """Calculate integrity checksums after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of event data"""
        event_data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "category": self.category.value,
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": asdict(self.metadata),
            "message": self.message,
            "details": self.details,
            "compliance_context": asdict(self.compliance_context) if self.compliance_context else None,
            "regulatory_impact": self.regulatory_impact,
            "pii_involved": self.pii_involved,
            "financial_impact": self.financial_impact,
            "security_classification": self.security_classification,
            "threat_indicators": self.threat_indicators,
            "risk_score": self.risk_score,
            "previous_event_id": self.previous_event_id,
            "chain_sequence": self.chain_sequence
        }
        
        serialized = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum"""
        calculated_checksum = self._calculate_checksum()
        return calculated_checksum == self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "category": self.category.value,
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": asdict(self.metadata),
            "message": self.message,
            "details": self.details,
            "compliance_context": asdict(self.compliance_context) if self.compliance_context else None,
            "regulatory_impact": self.regulatory_impact,
            "pii_involved": self.pii_involved,
            "financial_impact": self.financial_impact,
            "security_classification": self.security_classification,
            "threat_indicators": self.threat_indicators,
            "risk_score": self.risk_score,
            "checksum": self.checksum,
            "digital_signature": self.digital_signature,
            "blockchain_hash": self.blockchain_hash,
            "previous_event_id": self.previous_event_id,
            "chain_sequence": self.chain_sequence
        }


class AuditStorage:
    """Enhanced audit storage with enterprise features"""
    
    def __init__(self, db_path: str = "audit_enterprise.db", enable_encryption: bool = True):
        self.db_path = Path(db_path)
        self.enable_encryption = enable_encryption
        self._lock = threading.Lock()
        self._connection_pool = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize enterprise audit database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Enhanced audit events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    level TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    
                    -- Metadata fields
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    system_component TEXT NOT NULL,
                    correlation_id TEXT,
                    request_id TEXT,
                    trace_id TEXT,
                    geo_location TEXT,
                    device_info TEXT,
                    additional_context TEXT,
                    
                    -- Core fields
                    message TEXT NOT NULL,
                    details TEXT NOT NULL,
                    
                    -- Compliance fields
                    compliance_framework TEXT,
                    requirement_id TEXT,
                    control_id TEXT,
                    risk_level TEXT,
                    impact_assessment TEXT,
                    data_classification TEXT,
                    retention_period INTEGER,
                    regulatory_tags TEXT,
                    regulatory_impact BOOLEAN DEFAULT FALSE,
                    pii_involved BOOLEAN DEFAULT FALSE,
                    financial_impact BOOLEAN DEFAULT FALSE,
                    
                    -- Security fields
                    security_classification TEXT DEFAULT 'internal',
                    threat_indicators TEXT,
                    risk_score REAL DEFAULT 0.0,
                    
                    -- Integrity fields
                    checksum TEXT NOT NULL,
                    digital_signature TEXT,
                    blockchain_hash TEXT,
                    
                    -- Chain fields
                    previous_event_id TEXT,
                    chain_sequence INTEGER NOT NULL,
                    
                    -- System fields
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    archived BOOLEAN DEFAULT FALSE,
                    encrypted BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Audit compliance mapping table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_compliance_mapping (
                    mapping_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    requirement_id TEXT NOT NULL,
                    control_id TEXT NOT NULL,
                    compliance_status TEXT NOT NULL,
                    evidence_provided TEXT,
                    gap_identified TEXT,
                    remediation_required BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES audit_events (event_id)
                )
            """)
            
            # Audit blockchain registry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_blockchain_registry (
                    block_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    block_hash TEXT NOT NULL,
                    previous_block_hash TEXT,
                    merkle_root TEXT,
                    timestamp DATETIME NOT NULL,
                    nonce INTEGER,
                    difficulty INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (event_id) REFERENCES audit_events (event_id)
                )
            """)
            
            # Audit retention policies
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_retention_policies (
                    policy_id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    category TEXT NOT NULL,
                    retention_days INTEGER NOT NULL,
                    archive_after_days INTEGER,
                    delete_after_days INTEGER,
                    encryption_required BOOLEAN DEFAULT FALSE,
                    backup_required BOOLEAN DEFAULT TRUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create comprehensive indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_user_id ON audit_events(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_category ON audit_events(category)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_level ON audit_events(level)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_system_component ON audit_events(system_component)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_correlation_id ON audit_events(correlation_id)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_compliance_framework ON audit_events(compliance_framework)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_regulatory_impact ON audit_events(regulatory_impact)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_financial_impact ON audit_events(financial_impact)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_pii_involved ON audit_events(pii_involved)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_security_classification ON audit_events(security_classification)",
                "CREATE INDEX IF NOT EXISTS idx_audit_events_risk_score ON audit_events(risk_score)",
                "CREATE INDEX IF NOT EXISTS idx_audit_compliance_mapping_framework ON audit_compliance_mapping(framework)",
                "CREATE INDEX IF NOT EXISTS idx_audit_compliance_mapping_requirement ON audit_compliance_mapping(requirement_id)"
            ]
            
            for index in indexes:
                cursor.execute(index)
            
            # Initialize default retention policies
            default_policies = [
                ("finra_trading", "FINRA", "TRADING", 2555, 365, 2920, True, True),  # 7 years
                ("sec_compliance", "SEC", "COMPLIANCE", 1825, 365, 2190, True, True),  # 5 years
                ("gdpr_data", "GDPR", "DATA_ACCESS", 1095, 365, 1460, True, True),  # 3 years
                ("sox_financial", "SOX", "FINANCIAL", 2555, 365, 2920, True, True),  # 7 years
                ("pci_security", "PCI_DSS", "SECURITY", 1095, 365, 1460, True, True),  # 3 years
                ("default_operational", "COMPANY_POLICY", "OPERATIONAL", 365, 90, 730, False, True)  # 1 year
            ]
            
            for policy_data in default_policies:
                cursor.execute("""
                    INSERT OR IGNORE INTO audit_retention_policies 
                    (policy_id, framework, category, retention_days, archive_after_days, 
                     delete_after_days, encryption_required, backup_required)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, policy_data)
            
            conn.commit()
    
    def store_event(self, event: EnhancedAuditEvent) -> bool:
        """Store enhanced audit event in database"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Prepare compliance context data
                    compliance_data = None
                    if event.compliance_context:
                        compliance_data = {
                            "framework": event.compliance_context.framework.value,
                            "requirement_id": event.compliance_context.requirement_id,
                            "control_id": event.compliance_context.control_id,
                            "risk_level": event.compliance_context.risk_level,
                            "impact_assessment": event.compliance_context.impact_assessment,
                            "data_classification": event.compliance_context.data_classification,
                            "retention_period": event.compliance_context.retention_period,
                            "regulatory_tags": json.dumps(event.compliance_context.regulatory_tags)
                        }
                    
                    # Insert audit event
                    cursor.execute("""
                        INSERT INTO audit_events (
                            event_id, event_type, category, level, timestamp,
                            user_id, session_id, ip_address, user_agent, system_component,
                            correlation_id, request_id, trace_id, geo_location, device_info,
                            additional_context, message, details,
                            compliance_framework, requirement_id, control_id, risk_level,
                            impact_assessment, data_classification, retention_period,
                            regulatory_tags, regulatory_impact, pii_involved, financial_impact,
                            security_classification, threat_indicators, risk_score,
                            checksum, digital_signature, blockchain_hash,
                            previous_event_id, chain_sequence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type,
                        event.category.value,
                        event.level.value,
                        event.timestamp,
                        event.metadata.user_id,
                        event.metadata.session_id,
                        event.metadata.ip_address,
                        event.metadata.user_agent,
                        event.metadata.system_component,
                        event.metadata.correlation_id,
                        event.metadata.request_id,
                        event.metadata.trace_id,
                        event.metadata.geo_location,
                        event.metadata.device_info,
                        json.dumps(event.metadata.additional_context),
                        event.message,
                        json.dumps(event.details),
                        compliance_data["framework"] if compliance_data else None,
                        compliance_data["requirement_id"] if compliance_data else None,
                        compliance_data["control_id"] if compliance_data else None,
                        compliance_data["risk_level"] if compliance_data else None,
                        compliance_data["impact_assessment"] if compliance_data else None,
                        compliance_data["data_classification"] if compliance_data else None,
                        compliance_data["retention_period"] if compliance_data else None,
                        compliance_data["regulatory_tags"] if compliance_data else None,
                        event.regulatory_impact,
                        event.pii_involved,
                        event.financial_impact,
                        event.security_classification,
                        json.dumps(event.threat_indicators),
                        event.risk_score,
                        event.checksum,
                        event.digital_signature,
                        event.blockchain_hash,
                        event.previous_event_id,
                        event.chain_sequence
                    ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error("Failed to store audit event", event_id=event.event_id, error=str(e))
            return False
    
    def get_events_by_compliance_framework(
        self,
        framework: ComplianceFramework,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[EnhancedAuditEvent]:
        """Retrieve audit events by compliance framework"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM audit_events WHERE compliance_framework = ?"
                params = [framework.value]
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return self._rows_to_events(rows)
                
        except Exception as e:
            logger.error("Failed to retrieve compliance events", framework=framework.value, error=str(e))
            return []
    
    def get_regulatory_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[EnhancedAuditEvent]:
        """Retrieve all regulatory-impact events"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM audit_events WHERE regulatory_impact = TRUE"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return self._rows_to_events(rows)
                
        except Exception as e:
            logger.error("Failed to retrieve regulatory events", error=str(e))
            return []
    
    def _rows_to_events(self, rows: List[tuple]) -> List[EnhancedAuditEvent]:
        """Convert database rows to EnhancedAuditEvent objects"""
        events = []
        
        for row in rows:
            try:
                # Create metadata
                metadata = AuditMetadata(
                    user_id=row[5],
                    session_id=row[6],
                    ip_address=row[7],
                    user_agent=row[8] or "",
                    system_component=row[9],
                    correlation_id=row[10] or "",
                    request_id=row[11] or "",
                    trace_id=row[12] or "",
                    geo_location=row[13],
                    device_info=row[14],
                    additional_context=json.loads(row[15]) if row[15] else {}
                )
                
                # Create compliance context if present
                compliance_context = None
                if row[18]:  # compliance_framework
                    compliance_context = ComplianceContext(
                        framework=ComplianceFramework(row[18]),
                        requirement_id=row[19] or "",
                        control_id=row[20] or "",
                        risk_level=row[21] or "",
                        impact_assessment=row[22] or "",
                        data_classification=row[23] or "",
                        retention_period=row[24] or 365,
                        regulatory_tags=json.loads(row[25]) if row[25] else []
                    )
                
                # Create event
                event = EnhancedAuditEvent(
                    event_id=row[0],
                    event_type=row[1],
                    category=AuditCategory(row[2]),
                    level=AuditLevel(row[3]),
                    timestamp=datetime.fromisoformat(row[4]),
                    metadata=metadata,
                    message=row[16],
                    details=json.loads(row[17]),
                    compliance_context=compliance_context,
                    regulatory_impact=bool(row[26]),
                    pii_involved=bool(row[27]),
                    financial_impact=bool(row[28]),
                    security_classification=row[29] or "internal",
                    threat_indicators=json.loads(row[30]) if row[30] else [],
                    risk_score=float(row[31]) if row[31] else 0.0,
                    checksum=row[32],
                    digital_signature=row[33] or "",
                    blockchain_hash=row[34] or "",
                    previous_event_id=row[35],
                    chain_sequence=int(row[36])
                )
                
                events.append(event)
                
            except Exception as e:
                logger.error("Failed to parse audit event from row", error=str(e))
                continue
        
        return events


class EnhancedAuditLogger:
    """Enhanced audit logger with enterprise-grade features"""
    
    def __init__(self, storage: Optional[AuditStorage] = None, enable_blockchain: bool = False):
        self.storage = storage or AuditStorage()
        self.enable_blockchain = enable_blockchain and BLOCKCHAIN_AVAILABLE
        
        # Initialize blockchain if enabled
        if self.enable_blockchain:
            self.blockchain = BlockchainAudit()
        
        # Async processing
        self.event_queue = queue.Queue()
        self.processing_pool = ThreadPoolExecutor(max_workers=4)
        self.processing_active = True
        
        # Start background processing
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        
        # Event counters
        self.event_count = 0
        self.compliance_event_count = 0
        self.security_event_count = 0
        self.financial_event_count = 0
        
        # Threat detection
        self.threat_score_threshold = 0.7
        self.threat_indicators = []
        
        logger.info("Enhanced Audit Logger initialized")
    
    def log_event(
        self,
        event_type: str,
        category: AuditCategory,
        level: AuditLevel,
        metadata: AuditMetadata,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        compliance_context: Optional[ComplianceContext] = None,
        regulatory_impact: bool = False,
        pii_involved: bool = False,
        financial_impact: bool = False,
        security_classification: str = "internal",
        threat_indicators: Optional[List[str]] = None,
        risk_score: float = 0.0
    ) -> str:
        """Log enhanced audit event with enterprise features"""
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Create enhanced audit event
            event = EnhancedAuditEvent(
                event_id=event_id,
                event_type=event_type,
                category=category,
                level=level,
                timestamp=datetime.now(),
                metadata=metadata,
                message=message,
                details=details or {},
                compliance_context=compliance_context,
                regulatory_impact=regulatory_impact,
                pii_involved=pii_involved,
                financial_impact=financial_impact,
                security_classification=security_classification,
                threat_indicators=threat_indicators or [],
                risk_score=risk_score
            )
            
            # Add to processing queue
            self.event_queue.put(event)
            
            # Update counters
            self.event_count += 1
            
            if regulatory_impact:
                self.compliance_event_count += 1
            
            if category == AuditCategory.SECURITY:
                self.security_event_count += 1
            
            if financial_impact:
                self.financial_event_count += 1
            
            # Check for threat indicators
            if risk_score > self.threat_score_threshold:
                self._handle_threat_detection(event)
            
            logger.debug("Audit event queued", event_id=event_id, event_type=event_type)
            
            return event_id
            
        except Exception as e:
            logger.error("Failed to log audit event", error=str(e))
            return ""
    
    def _process_events(self):
        """Background thread for processing audit events"""
        while self.processing_active:
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1.0)
                
                # Process event
                self._process_single_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Error processing audit event", error=str(e))
    
    def _process_single_event(self, event: EnhancedAuditEvent):
        """Process a single audit event"""
        try:
            # Store in database
            success = self.storage.store_event(event)
            
            if not success:
                logger.error("Failed to store audit event", event_id=event.event_id)
                return
            
            # Add to blockchain if enabled
            if self.enable_blockchain and event.regulatory_impact:
                self._add_to_blockchain(event)
            
            # Handle compliance events
            if event.compliance_context:
                self._process_compliance_event(event)
            
            # Handle security events
            if event.category == AuditCategory.SECURITY:
                self._process_security_event(event)
            
            # Handle financial events
            if event.financial_impact:
                self._process_financial_event(event)
            
            logger.debug("Audit event processed", event_id=event.event_id)
            
        except Exception as e:
            logger.error("Failed to process audit event", event_id=event.event_id, error=str(e))
    
    def _add_to_blockchain(self, event: EnhancedAuditEvent):
        """Add event to blockchain for immutable audit trail"""
        try:
            if self.enable_blockchain:
                block_hash = self.blockchain.add_event(event.to_dict())
                event.blockchain_hash = block_hash
                logger.debug("Event added to blockchain", event_id=event.event_id, block_hash=block_hash)
        except Exception as e:
            logger.error("Failed to add event to blockchain", event_id=event.event_id, error=str(e))
    
    def _process_compliance_event(self, event: EnhancedAuditEvent):
        """Process compliance-related audit event"""
        try:
            # Check compliance mapping
            if event.compliance_context:
                framework = event.compliance_context.framework
                requirement_id = event.compliance_context.requirement_id
                
                logger.info(
                    "Compliance event processed",
                    event_id=event.event_id,
                    framework=framework.value,
                    requirement_id=requirement_id
                )
                
                # Trigger compliance validation if needed
                if event.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]:
                    self._trigger_compliance_alert(event)
        
        except Exception as e:
            logger.error("Failed to process compliance event", event_id=event.event_id, error=str(e))
    
    def _process_security_event(self, event: EnhancedAuditEvent):
        """Process security-related audit event"""
        try:
            # Check threat indicators
            if event.threat_indicators:
                self.threat_indicators.extend(event.threat_indicators)
                
                logger.warning(
                    "Security threat indicators detected",
                    event_id=event.event_id,
                    indicators=event.threat_indicators,
                    risk_score=event.risk_score
                )
            
            # Check for high-risk events
            if event.risk_score > self.threat_score_threshold:
                self._trigger_security_alert(event)
        
        except Exception as e:
            logger.error("Failed to process security event", event_id=event.event_id, error=str(e))
    
    def _process_financial_event(self, event: EnhancedAuditEvent):
        """Process financial-impact audit event"""
        try:
            logger.info(
                "Financial event processed",
                event_id=event.event_id,
                event_type=event.event_type,
                risk_score=event.risk_score
            )
            
            # Check for high-impact financial events
            if event.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]:
                self._trigger_financial_alert(event)
        
        except Exception as e:
            logger.error("Failed to process financial event", event_id=event.event_id, error=str(e))
    
    def _handle_threat_detection(self, event: EnhancedAuditEvent):
        """Handle threat detection from audit events"""
        try:
            logger.warning(
                "Threat detected in audit event",
                event_id=event.event_id,
                risk_score=event.risk_score,
                threat_indicators=event.threat_indicators
            )
            
            # Additional threat analysis can be implemented here
            
        except Exception as e:
            logger.error("Failed to handle threat detection", event_id=event.event_id, error=str(e))
    
    def _trigger_compliance_alert(self, event: EnhancedAuditEvent):
        """Trigger compliance alert for critical events"""
        try:
            logger.critical(
                "Compliance alert triggered",
                event_id=event.event_id,
                framework=event.compliance_context.framework.value if event.compliance_context else "unknown",
                message=event.message
            )
        except Exception as e:
            logger.error("Failed to trigger compliance alert", event_id=event.event_id, error=str(e))
    
    def _trigger_security_alert(self, event: EnhancedAuditEvent):
        """Trigger security alert for high-risk events"""
        try:
            logger.critical(
                "Security alert triggered",
                event_id=event.event_id,
                risk_score=event.risk_score,
                threat_indicators=event.threat_indicators,
                message=event.message
            )
        except Exception as e:
            logger.error("Failed to trigger security alert", event_id=event.event_id, error=str(e))
    
    def _trigger_financial_alert(self, event: EnhancedAuditEvent):
        """Trigger financial alert for high-impact events"""
        try:
            logger.critical(
                "Financial alert triggered",
                event_id=event.event_id,
                message=event.message
            )
        except Exception as e:
            logger.error("Failed to trigger financial alert", event_id=event.event_id, error=str(e))
    
    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        try:
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(days=30)
            
            # Get compliance events
            events = self.storage.get_events_by_compliance_framework(
                framework=framework,
                start_time=start_time,
                end_time=end_time
            )
            
            # Calculate statistics
            total_events = len(events)
            critical_events = len([e for e in events if e.level == AuditLevel.CRITICAL])
            error_events = len([e for e in events if e.level == AuditLevel.ERROR])
            warning_events = len([e for e in events if e.level == AuditLevel.WARNING])
            
            # Calculate compliance score
            if total_events > 0:
                compliance_score = max(0, 100 - (critical_events * 10 + error_events * 5 + warning_events * 2))
            else:
                compliance_score = 100
            
            return {
                "framework": framework.value,
                "period_start": start_time.isoformat(),
                "period_end": end_time.isoformat(),
                "total_events": total_events,
                "critical_events": critical_events,
                "error_events": error_events,
                "warning_events": warning_events,
                "compliance_score": compliance_score,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to generate compliance report", framework=framework.value, error=str(e))
            return {}
    
    def generate_security_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate security report"""
        try:
            if end_time is None:
                end_time = datetime.now()
            if start_time is None:
                start_time = end_time - timedelta(days=7)
            
            # Get security events
            events = self.storage.get_events_by_compliance_framework(
                framework=ComplianceFramework.COMPANY_POLICY,  # Using as default
                start_time=start_time,
                end_time=end_time
            )
            
            security_events = [e for e in events if e.category == AuditCategory.SECURITY]
            
            # Calculate security metrics
            total_security_events = len(security_events)
            high_risk_events = len([e for e in security_events if e.risk_score > 0.7])
            threat_indicators_count = len(set(self.threat_indicators))
            
            return {
                "period_start": start_time.isoformat(),
                "period_end": end_time.isoformat(),
                "total_security_events": total_security_events,
                "high_risk_events": high_risk_events,
                "threat_indicators_detected": threat_indicators_count,
                "security_score": max(0, 100 - high_risk_events * 5),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to generate security report", error=str(e))
            return {}
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        return {
            "total_events": self.event_count,
            "compliance_events": self.compliance_event_count,
            "security_events": self.security_event_count,
            "financial_events": self.financial_event_count,
            "threat_indicators": len(self.threat_indicators),
            "queue_size": self.event_queue.qsize(),
            "blockchain_enabled": self.enable_blockchain
        }
    
    def shutdown(self):
        """Shutdown audit logger"""
        try:
            self.processing_active = False
            
            # Wait for queue to empty
            self.event_queue.join()
            
            # Shutdown thread pool
            self.processing_pool.shutdown(wait=True)
            
            logger.info("Enhanced Audit Logger shutdown complete")
            
        except Exception as e:
            logger.error("Error during audit logger shutdown", error=str(e))


# Convenience functions for common audit scenarios
def create_trading_metadata(user_id: str, session_id: str, ip_address: str) -> AuditMetadata:
    """Create metadata for trading operations"""
    return AuditMetadata(
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent="trading_system",
        system_component="trading_engine",
        correlation_id=str(uuid.uuid4()),
        request_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4())
    )


def create_compliance_context(
    framework: ComplianceFramework,
    requirement_id: str,
    control_id: str,
    risk_level: str = "medium"
) -> ComplianceContext:
    """Create compliance context for regulatory events"""
    return ComplianceContext(
        framework=framework,
        requirement_id=requirement_id,
        control_id=control_id,
        risk_level=risk_level,
        impact_assessment="Standard regulatory requirement",
        data_classification="internal",
        retention_period=2555  # 7 years default
    )


# Global audit logger instance
_audit_logger: Optional[EnhancedAuditLogger] = None


def get_audit_logger() -> EnhancedAuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = EnhancedAuditLogger()
    return _audit_logger


def initialize_audit_logger(
    storage: Optional[AuditStorage] = None,
    enable_blockchain: bool = False
) -> EnhancedAuditLogger:
    """Initialize global audit logger"""
    global _audit_logger
    _audit_logger = EnhancedAuditLogger(storage=storage, enable_blockchain=enable_blockchain)
    return _audit_logger