"""
GDPR Compliance Implementation
Data protection, privacy rights, and regulatory compliance
"""

import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import redis.asyncio as redis
from pydantic import BaseModel, Field, EmailStr
from fastapi import HTTPException, Request

from src.monitoring.logger_config import get_logger
from src.security.encryption import encrypt_data, decrypt_data, encrypt_sensitive_fields, decrypt_sensitive_fields
from src.security.audit_logger import AuditEventType, AuditSeverity, ComplianceFramework

logger = get_logger(__name__)

class DataProcessingPurpose(str, Enum):
    """GDPR data processing purposes"""
    TRADING_EXECUTION = "trading_execution"
    RISK_MANAGEMENT = "risk_management"
    COMPLIANCE_MONITORING = "compliance_monitoring"
    CUSTOMER_SUPPORT = "customer_support"
    ANALYTICS = "analytics"
    SECURITY = "security"
    LEGAL_OBLIGATION = "legal_obligation"
    LEGITIMATE_INTEREST = "legitimate_interest"
    CONSENT = "consent"
    CONTRACT_PERFORMANCE = "contract_performance"

class DataCategory(str, Enum):
    """Categories of personal data"""
    IDENTITY = "identity"  # Name, username, employee ID
    CONTACT = "contact"  # Email, phone, address
    FINANCIAL = "financial"  # Account numbers, transaction data
    TECHNICAL = "technical"  # IP address, device info, logs
    BEHAVIORAL = "behavioral"  # Usage patterns, preferences
    BIOMETRIC = "biometric"  # Fingerprints, facial recognition
    LOCATION = "location"  # Geographic data
    SPECIAL = "special"  # Health, political, religious data

class LegalBasis(str, Enum):
    """GDPR legal basis for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

class DataSubjectRights(str, Enum):
    """GDPR data subject rights"""
    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to erasure ("right to be forgotten")
    RESTRICT_PROCESSING = "restrict_processing"  # Right to restrict processing
    DATA_PORTABILITY = "data_portability"  # Right to data portability
    OBJECT = "object"  # Right to object
    AUTOMATED_DECISION = "automated_decision"  # Rights related to automated decision making

class ConsentStatus(str, Enum):
    """Consent status"""
    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"

@dataclass
class DataProcessingRecord:
    """Record of data processing activities"""
    record_id: str
    controller_name: str
    controller_contact: str
    processing_purpose: DataProcessingPurpose
    legal_basis: LegalBasis
    data_categories: List[DataCategory]
    data_subjects: List[str]  # Categories of data subjects
    recipients: List[str]  # Categories of recipients
    third_country_transfers: List[str]  # Third countries or international organizations
    retention_period: str  # Time limits for erasure
    technical_measures: List[str]  # Technical security measures
    organizational_measures: List[str]  # Organizational security measures
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "controller_name": self.controller_name,
            "controller_contact": self.controller_contact,
            "processing_purpose": self.processing_purpose.value,
            "legal_basis": self.legal_basis.value,
            "data_categories": [dc.value for dc in self.data_categories],
            "data_subjects": self.data_subjects,
            "recipients": self.recipients,
            "third_country_transfers": self.third_country_transfers,
            "retention_period": self.retention_period,
            "technical_measures": self.technical_measures,
            "organizational_measures": self.organizational_measures,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class ConsentRecord:
    """Record of consent given by data subject"""
    consent_id: str
    data_subject_id: str
    processing_purposes: List[DataProcessingPurpose]
    status: ConsentStatus
    given_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    consent_method: str = "explicit"  # explicit, implied, inferred
    consent_evidence: Optional[str] = None  # Record of how consent was obtained
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if consent is valid"""
        if self.status != ConsentStatus.GIVEN:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "consent_id": self.consent_id,
            "data_subject_id": self.data_subject_id,
            "processing_purposes": [pp.value for pp in self.processing_purposes],
            "status": self.status.value,
            "given_at": self.given_at.isoformat() if self.given_at else None,
            "withdrawn_at": self.withdrawn_at.isoformat() if self.withdrawn_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "consent_method": self.consent_method,
            "consent_evidence": self.consent_evidence,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }

@dataclass
class DataSubjectRequest:
    """Data subject request (DSR)"""
    request_id: str
    data_subject_id: str
    request_type: DataSubjectRights
    description: str
    status: str = "pending"  # pending, in_progress, completed, rejected
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    requested_by: Optional[str] = None  # User who made the request
    processed_by: Optional[str] = None  # User who processed the request
    verification_method: Optional[str] = None
    verification_completed: bool = False
    response_data: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "data_subject_id": self.data_subject_id,
            "request_type": self.request_type.value,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "requested_by": self.requested_by,
            "processed_by": self.processed_by,
            "verification_method": self.verification_method,
            "verification_completed": self.verification_completed,
            "response_data": self.response_data,
            "rejection_reason": self.rejection_reason
        }

class DataRetentionPolicy:
    """Data retention policy implementation"""
    
    def __init__(self):
        self.retention_periods = {
            DataCategory.IDENTITY: timedelta(days=2555),  # 7 years
            DataCategory.CONTACT: timedelta(days=2555),  # 7 years
            DataCategory.FINANCIAL: timedelta(days=2555),  # 7 years
            DataCategory.TECHNICAL: timedelta(days=1095),  # 3 years
            DataCategory.BEHAVIORAL: timedelta(days=1095),  # 3 years
            DataCategory.BIOMETRIC: timedelta(days=365),  # 1 year
            DataCategory.LOCATION: timedelta(days=365),  # 1 year
            DataCategory.SPECIAL: timedelta(days=365),  # 1 year
        }
        
        # Purpose-based retention
        self.purpose_retention = {
            DataProcessingPurpose.TRADING_EXECUTION: timedelta(days=2555),  # 7 years
            DataProcessingPurpose.RISK_MANAGEMENT: timedelta(days=2555),  # 7 years
            DataProcessingPurpose.COMPLIANCE_MONITORING: timedelta(days=2555),  # 7 years
            DataProcessingPurpose.CUSTOMER_SUPPORT: timedelta(days=1095),  # 3 years
            DataProcessingPurpose.ANALYTICS: timedelta(days=1095),  # 3 years
            DataProcessingPurpose.SECURITY: timedelta(days=365),  # 1 year
            DataProcessingPurpose.LEGAL_OBLIGATION: timedelta(days=2555),  # 7 years
        }
    
    def get_retention_period(self, 
                           data_category: DataCategory,
                           processing_purpose: DataProcessingPurpose) -> timedelta:
        """Get retention period for data category and purpose"""
        # Use the longer of the two periods
        category_period = self.retention_periods.get(data_category, timedelta(days=365))
        purpose_period = self.purpose_retention.get(processing_purpose, timedelta(days=365))
        
        return max(category_period, purpose_period)
    
    def should_delete_data(self, 
                          created_at: datetime,
                          data_category: DataCategory,
                          processing_purpose: DataProcessingPurpose) -> bool:
        """Check if data should be deleted based on retention policy"""
        retention_period = self.get_retention_period(data_category, processing_purpose)
        expiry_date = created_at + retention_period
        
        return datetime.utcnow() > expiry_date

class GDPRComplianceManager:
    """GDPR compliance manager"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.retention_policy = DataRetentionPolicy()
        
        # In-memory storage for development
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.data_subject_requests: Dict[str, DataSubjectRequest] = {}
        
        # Initialize default processing records
        self._initialize_processing_records()
        
        logger.info("GDPR Compliance Manager initialized")
    
    def _initialize_processing_records(self):
        """Initialize default processing records"""
        # Trading execution processing
        trading_record = DataProcessingRecord(
            record_id="trading_execution",
            controller_name="GrandModel Trading System",
            controller_contact="privacy@grandmodel.com",
            processing_purpose=DataProcessingPurpose.TRADING_EXECUTION,
            legal_basis=LegalBasis.CONTRACT,
            data_categories=[DataCategory.IDENTITY, DataCategory.FINANCIAL],
            data_subjects=["traders", "account_holders"],
            recipients=["internal_trading_systems", "external_brokers"],
            third_country_transfers=[],
            retention_period="7 years",
            technical_measures=["encryption", "access_controls", "audit_logging"],
            organizational_measures=["staff_training", "privacy_policies", "data_protection_officer"]
        )
        
        # Risk management processing
        risk_record = DataProcessingRecord(
            record_id="risk_management",
            controller_name="GrandModel Risk Management",
            controller_contact="privacy@grandmodel.com",
            processing_purpose=DataProcessingPurpose.RISK_MANAGEMENT,
            legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
            data_categories=[DataCategory.FINANCIAL, DataCategory.BEHAVIORAL],
            data_subjects=["traders", "account_holders"],
            recipients=["internal_risk_systems", "compliance_officers"],
            third_country_transfers=[],
            retention_period="7 years",
            technical_measures=["encryption", "access_controls", "audit_logging"],
            organizational_measures=["risk_policies", "regular_reviews", "incident_response"]
        )
        
        # Analytics processing
        analytics_record = DataProcessingRecord(
            record_id="analytics",
            controller_name="GrandModel Analytics",
            controller_contact="privacy@grandmodel.com",
            processing_purpose=DataProcessingPurpose.ANALYTICS,
            legal_basis=LegalBasis.LEGITIMATE_INTERESTS,
            data_categories=[DataCategory.BEHAVIORAL, DataCategory.TECHNICAL],
            data_subjects=["users", "visitors"],
            recipients=["internal_analytics_team"],
            third_country_transfers=[],
            retention_period="3 years",
            technical_measures=["pseudonymization", "encryption", "access_controls"],
            organizational_measures=["data_minimization", "purpose_limitation", "regular_reviews"]
        )
        
        self.processing_records = {
            "trading_execution": trading_record,
            "risk_management": risk_record,
            "analytics": analytics_record
        }
    
    async def record_consent(self, 
                           data_subject_id: str,
                           processing_purposes: List[DataProcessingPurpose],
                           consent_method: str = "explicit",
                           consent_evidence: Optional[str] = None,
                           ip_address: Optional[str] = None,
                           user_agent: Optional[str] = None,
                           expires_in_days: Optional[int] = None) -> str:
        """Record consent from data subject"""
        
        consent_id = str(uuid.uuid4())
        expires_at = None
        
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        consent_record = ConsentRecord(
            consent_id=consent_id,
            data_subject_id=data_subject_id,
            processing_purposes=processing_purposes,
            status=ConsentStatus.GIVEN,
            given_at=datetime.utcnow(),
            expires_at=expires_at,
            consent_method=consent_method,
            consent_evidence=consent_evidence,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store consent record
        self.consent_records[consent_id] = consent_record
        
        if self.redis:
            await self._store_consent_record(consent_record)
        
        logger.info(
            "Consent recorded",
            consent_id=consent_id,
            data_subject_id=data_subject_id,
            purposes=[p.value for p in processing_purposes]
        )
        
        return consent_id
    
    async def withdraw_consent(self, consent_id: str, data_subject_id: str) -> bool:
        """Withdraw consent"""
        consent_record = self.consent_records.get(consent_id)
        
        if not consent_record or consent_record.data_subject_id != data_subject_id:
            return False
        
        consent_record.status = ConsentStatus.WITHDRAWN
        consent_record.withdrawn_at = datetime.utcnow()
        
        if self.redis:
            await self._store_consent_record(consent_record)
        
        logger.info(
            "Consent withdrawn",
            consent_id=consent_id,
            data_subject_id=data_subject_id
        )
        
        return True
    
    async def _store_consent_record(self, consent_record: ConsentRecord):
        """Store consent record in Redis"""
        encrypted_data = encrypt_data(consent_record.to_dict())
        
        await self.redis.setex(
            f"gdpr_consent:{consent_record.consent_id}",
            2555 * 24 * 3600,  # 7 years
            encrypted_data.to_dict()
        )
        
        # Index by data subject
        await self.redis.sadd(
            f"gdpr_consent_subject:{consent_record.data_subject_id}",
            consent_record.consent_id
        )
    
    async def submit_data_subject_request(self, 
                                        data_subject_id: str,
                                        request_type: DataSubjectRights,
                                        description: str,
                                        requested_by: Optional[str] = None) -> str:
        """Submit data subject request"""
        
        request_id = str(uuid.uuid4())
        
        dsr = DataSubjectRequest(
            request_id=request_id,
            data_subject_id=data_subject_id,
            request_type=request_type,
            description=description,
            requested_by=requested_by
        )
        
        # Store request
        self.data_subject_requests[request_id] = dsr
        
        if self.redis:
            await self._store_data_subject_request(dsr)
        
        logger.info(
            "Data subject request submitted",
            request_id=request_id,
            data_subject_id=data_subject_id,
            request_type=request_type.value
        )
        
        return request_id
    
    async def _store_data_subject_request(self, dsr: DataSubjectRequest):
        """Store data subject request in Redis"""
        encrypted_data = encrypt_data(dsr.to_dict())
        
        await self.redis.setex(
            f"gdpr_dsr:{dsr.request_id}",
            2555 * 24 * 3600,  # 7 years
            encrypted_data.to_dict()
        )
        
        # Index by data subject
        await self.redis.sadd(
            f"gdpr_dsr_subject:{dsr.data_subject_id}",
            dsr.request_id
        )
        
        # Index by status
        await self.redis.sadd(
            f"gdpr_dsr_status:{dsr.status}",
            dsr.request_id
        )
    
    async def process_access_request(self, request_id: str) -> Dict[str, Any]:
        """Process data subject access request"""
        dsr = self.data_subject_requests.get(request_id)
        
        if not dsr or dsr.request_type != DataSubjectRights.ACCESS:
            raise ValueError("Invalid access request")
        
        # Collect all data for the data subject
        data_export = {
            "data_subject_id": dsr.data_subject_id,
            "export_date": datetime.utcnow().isoformat(),
            "data_categories": {}
        }
        
        # Add consent records
        consent_records = []
        for consent_id, consent in self.consent_records.items():
            if consent.data_subject_id == dsr.data_subject_id:
                consent_records.append(consent.to_dict())
        
        data_export["consent_records"] = consent_records
        
        # Add processing records
        data_export["processing_records"] = [record.to_dict() for record in self.processing_records.values()]
        
        # Update request status
        dsr.status = "completed"
        dsr.completed_at = datetime.utcnow()
        dsr.response_data = data_export
        
        if self.redis:
            await self._store_data_subject_request(dsr)
        
        logger.info(
            "Access request processed",
            request_id=request_id,
            data_subject_id=dsr.data_subject_id
        )
        
        return data_export
    
    async def process_erasure_request(self, request_id: str) -> bool:
        """Process data subject erasure request (right to be forgotten)"""
        dsr = self.data_subject_requests.get(request_id)
        
        if not dsr or dsr.request_type != DataSubjectRights.ERASURE:
            raise ValueError("Invalid erasure request")
        
        # Check if erasure is possible (legal obligations, etc.)
        if not self._can_erase_data(dsr.data_subject_id):
            dsr.status = "rejected"
            dsr.rejection_reason = "Data required for legal obligations"
            dsr.completed_at = datetime.utcnow()
            
            if self.redis:
                await self._store_data_subject_request(dsr)
            
            return False
        
        # Perform erasure
        await self._erase_data_subject_data(dsr.data_subject_id)
        
        # Update request status
        dsr.status = "completed"
        dsr.completed_at = datetime.utcnow()
        
        if self.redis:
            await self._store_data_subject_request(dsr)
        
        logger.info(
            "Erasure request processed",
            request_id=request_id,
            data_subject_id=dsr.data_subject_id
        )
        
        return True
    
    def _can_erase_data(self, data_subject_id: str) -> bool:
        """Check if data can be erased"""
        # Check for legal obligations
        # In financial services, some data must be retained for regulatory compliance
        
        # For demonstration, we'll allow erasure except for active trading accounts
        # In production, implement proper checks based on business rules
        
        return True  # Simplified for demo
    
    async def _erase_data_subject_data(self, data_subject_id: str):
        """Erase all data for a data subject"""
        # Remove consent records
        to_remove = []
        for consent_id, consent in self.consent_records.items():
            if consent.data_subject_id == data_subject_id:
                to_remove.append(consent_id)
        
        for consent_id in to_remove:
            del self.consent_records[consent_id]
            
            if self.redis:
                await self.redis.delete(f"gdpr_consent:{consent_id}")
        
        # Remove from Redis indexes
        if self.redis:
            await self.redis.delete(f"gdpr_consent_subject:{data_subject_id}")
            await self.redis.delete(f"gdpr_dsr_subject:{data_subject_id}")
        
        # In production, this would also remove data from databases, files, etc.
    
    async def check_data_retention_compliance(self) -> Dict[str, Any]:
        """Check compliance with data retention policies"""
        compliance_report = {
            "check_date": datetime.utcnow().isoformat(),
            "expired_consents": [],
            "data_to_delete": [],
            "compliance_status": "compliant"
        }
        
        # Check expired consents
        for consent_id, consent in self.consent_records.items():
            if consent.expires_at and datetime.utcnow() > consent.expires_at:
                compliance_report["expired_consents"].append({
                    "consent_id": consent_id,
                    "data_subject_id": consent.data_subject_id,
                    "expired_at": consent.expires_at.isoformat()
                })
        
        # Check data that should be deleted based on retention policy
        # This would typically query databases for data creation dates
        
        if compliance_report["expired_consents"] or compliance_report["data_to_delete"]:
            compliance_report["compliance_status"] = "action_required"
        
        return compliance_report
    
    async def generate_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy report"""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "processing_activities": len(self.processing_records),
            "active_consents": len([c for c in self.consent_records.values() if c.status == ConsentStatus.GIVEN]),
            "withdrawn_consents": len([c for c in self.consent_records.values() if c.status == ConsentStatus.WITHDRAWN]),
            "pending_requests": len([r for r in self.data_subject_requests.values() if r.status == "pending"]),
            "completed_requests": len([r for r in self.data_subject_requests.values() if r.status == "completed"]),
            "processing_records": [record.to_dict() for record in self.processing_records.values()],
            "consent_summary": {},
            "request_summary": {}
        }
        
        # Consent summary by purpose
        for consent in self.consent_records.values():
            for purpose in consent.processing_purposes:
                if purpose.value not in report["consent_summary"]:
                    report["consent_summary"][purpose.value] = {"given": 0, "withdrawn": 0}
                
                if consent.status == ConsentStatus.GIVEN:
                    report["consent_summary"][purpose.value]["given"] += 1
                elif consent.status == ConsentStatus.WITHDRAWN:
                    report["consent_summary"][purpose.value]["withdrawn"] += 1
        
        # Request summary by type
        for request in self.data_subject_requests.values():
            request_type = request.request_type.value
            if request_type not in report["request_summary"]:
                report["request_summary"][request_type] = {"pending": 0, "completed": 0, "rejected": 0}
            
            report["request_summary"][request_type][request.status] += 1
        
        return report
    
    def get_privacy_policy(self) -> str:
        """Get privacy policy text"""
        return """
        GRANDMODEL PRIVACY POLICY
        
        This Privacy Policy explains how GrandModel ("we", "us", "our") collects, uses, and protects your personal information in accordance with the General Data Protection Regulation (GDPR) and other applicable privacy laws.
        
        DATA CONTROLLER
        GrandModel Trading System
        Email: privacy@grandmodel.com
        
        DATA WE COLLECT
        - Identity data: Name, username, employee ID
        - Contact data: Email address, phone number
        - Financial data: Account numbers, trading history
        - Technical data: IP address, browser information
        - Usage data: How you use our services
        
        LEGAL BASIS FOR PROCESSING
        - Contract performance: To provide trading services
        - Legal obligations: To comply with financial regulations
        - Legitimate interests: For risk management and analytics
        - Consent: For marketing and optional features
        
        YOUR RIGHTS
        Under GDPR, you have the right to:
        - Access your personal data
        - Rectify inaccurate data
        - Erase your data ("right to be forgotten")
        - Restrict processing
        - Data portability
        - Object to processing
        - Lodge a complaint with supervisory authority
        
        DATA RETENTION
        We retain your data for:
        - Trading data: 7 years (regulatory requirement)
        - Marketing data: Until consent withdrawn
        - Technical logs: 3 years
        
        CONTACT US
        For privacy inquiries: privacy@grandmodel.com
        Data Protection Officer: dpo@grandmodel.com
        
        Last updated: {}
        """.format(datetime.utcnow().strftime("%Y-%m-%d"))

# Global GDPR compliance manager
gdpr_manager: Optional[GDPRComplianceManager] = None

async def get_gdpr_manager() -> GDPRComplianceManager:
    """Get or create GDPR compliance manager"""
    global gdpr_manager
    
    if gdpr_manager is None:
        # Initialize Redis client
        redis_client = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                redis_client = await redis.from_url(redis_url)
            except Exception as e:
                logger.error("Failed to connect to Redis for GDPR compliance", error=str(e))
        
        gdpr_manager = GDPRComplianceManager(redis_client)
    
    return gdpr_manager

# API models
class ConsentRequest(BaseModel):
    """Consent request model"""
    data_subject_id: str = Field(..., min_length=1, max_length=100)
    processing_purposes: List[DataProcessingPurpose]
    consent_method: str = Field("explicit", regex="^(explicit|implied|inferred)$")
    consent_evidence: Optional[str] = Field(None, max_length=1000)
    expires_in_days: Optional[int] = Field(None, ge=1, le=3650)

class DataSubjectRequestModel(BaseModel):
    """Data subject request model"""
    data_subject_id: str = Field(..., min_length=1, max_length=100)
    request_type: DataSubjectRights
    description: str = Field(..., min_length=1, max_length=1000)

class PrivacySettings(BaseModel):
    """Privacy settings model"""
    analytics_consent: bool = False
    marketing_consent: bool = False
    third_party_sharing: bool = False
    data_retention_override: Optional[int] = Field(None, ge=1, le=2555)

# Convenience functions
async def record_consent(data_subject_id: str,
                        processing_purposes: List[DataProcessingPurpose],
                        request: Optional[Request] = None) -> str:
    """Record consent from data subject"""
    manager = await get_gdpr_manager()
    
    ip_address = None
    user_agent = None
    
    if request:
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
    
    return await manager.record_consent(
        data_subject_id=data_subject_id,
        processing_purposes=processing_purposes,
        ip_address=ip_address,
        user_agent=user_agent
    )

async def submit_data_subject_request(data_subject_id: str,
                                    request_type: DataSubjectRights,
                                    description: str) -> str:
    """Submit data subject request"""
    manager = await get_gdpr_manager()
    return await manager.submit_data_subject_request(
        data_subject_id=data_subject_id,
        request_type=request_type,
        description=description
    )

async def check_consent(data_subject_id: str,
                       processing_purpose: DataProcessingPurpose) -> bool:
    """Check if data subject has given consent for processing purpose"""
    manager = await get_gdpr_manager()
    
    # Check all consent records for this data subject
    for consent in manager.consent_records.values():
        if (consent.data_subject_id == data_subject_id and
            processing_purpose in consent.processing_purposes and
            consent.is_valid()):
            return True
    
    return False
