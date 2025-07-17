"""
SOX Compliance & Regulatory Reporting
Sarbanes-Oxley Act compliance for financial systems
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
from pydantic import BaseModel, Field
from fastapi import HTTPException

from src.monitoring.logger_config import get_logger
from src.security.encryption import encrypt_data, decrypt_data
from src.security.audit_logger import AuditEventType, AuditSeverity, ComplianceFramework

logger = get_logger(__name__)

class SOXSection(str, Enum):
    """SOX Act sections"""
    SECTION_302 = "302"  # Corporate Responsibility for Financial Reports
    SECTION_404 = "404"  # Management Assessment of Internal Controls
    SECTION_409 = "409"  # Real Time Issuer Disclosures
    SECTION_802 = "802"  # Criminal Penalties for Altering Documents
    SECTION_906 = "906"  # Corporate Responsibility for Financial Reports

class ControlType(str, Enum):
    """Types of internal controls"""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    COMPENSATING = "compensating"

class ControlFrequency(str, Enum):
    """Control testing frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    CONTINUOUS = "continuous"

class ControlStatus(str, Enum):
    """Control status"""
    EFFECTIVE = "effective"
    INEFFECTIVE = "ineffective"
    DEFICIENT = "deficient"
    UNDER_REVIEW = "under_review"
    NOT_TESTED = "not_tested"

class DeficiencyType(str, Enum):
    """Types of control deficiencies"""
    SIGNIFICANT_DEFICIENCY = "significant_deficiency"
    MATERIAL_WEAKNESS = "material_weakness"
    MINOR_DEFICIENCY = "minor_deficiency"

class RegulatoryFramework(str, Enum):
    """Regulatory frameworks"""
    SOX = "sox"
    COSO = "coso"
    COBIT = "cobit"
    PCAOB = "pcaob"
    SEC = "sec"
    FINRA = "finra"
    BASEL = "basel"
    MiFID = "mifid"
    DODD_FRANK = "dodd_frank"

@dataclass
class InternalControl:
    """Internal control definition"""
    control_id: str
    control_name: str
    control_description: str
    control_type: ControlType
    sox_section: SOXSection
    business_process: str
    control_objective: str
    control_activities: List[str]
    control_owner: str
    control_operator: str
    testing_frequency: ControlFrequency
    automated: bool = False
    key_control: bool = False
    entity_level: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_id": self.control_id,
            "control_name": self.control_name,
            "control_description": self.control_description,
            "control_type": self.control_type.value,
            "sox_section": self.sox_section.value,
            "business_process": self.business_process,
            "control_objective": self.control_objective,
            "control_activities": self.control_activities,
            "control_owner": self.control_owner,
            "control_operator": self.control_operator,
            "testing_frequency": self.testing_frequency.value,
            "automated": self.automated,
            "key_control": self.key_control,
            "entity_level": self.entity_level,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

@dataclass
class ControlTest:
    """Control test execution record"""
    test_id: str
    control_id: str
    test_date: datetime
    test_period_start: datetime
    test_period_end: datetime
    tester: str
    test_procedures: List[str]
    sample_size: int
    sample_selection_method: str
    test_results: Dict[str, Any]
    status: ControlStatus
    deficiencies: List[str] = field(default_factory=list)
    management_response: Optional[str] = None
    remediation_plan: Optional[str] = None
    remediation_date: Optional[datetime] = None
    evidence_location: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "control_id": self.control_id,
            "test_date": self.test_date.isoformat(),
            "test_period_start": self.test_period_start.isoformat(),
            "test_period_end": self.test_period_end.isoformat(),
            "tester": self.tester,
            "test_procedures": self.test_procedures,
            "sample_size": self.sample_size,
            "sample_selection_method": self.sample_selection_method,
            "test_results": self.test_results,
            "status": self.status.value,
            "deficiencies": self.deficiencies,
            "management_response": self.management_response,
            "remediation_plan": self.remediation_plan,
            "remediation_date": self.remediation_date.isoformat() if self.remediation_date else None,
            "evidence_location": self.evidence_location,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class ControlDeficiency:
    """Control deficiency record"""
    deficiency_id: str
    control_id: str
    test_id: str
    deficiency_type: DeficiencyType
    description: str
    root_cause: str
    potential_impact: str
    likelihood: str  # High, Medium, Low
    severity: str  # High, Medium, Low
    identified_date: datetime
    identified_by: str
    management_response: str
    remediation_plan: str
    remediation_owner: str
    target_remediation_date: datetime
    actual_remediation_date: Optional[datetime] = None
    status: str = "open"  # open, in_progress, remediated, closed
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "deficiency_id": self.deficiency_id,
            "control_id": self.control_id,
            "test_id": self.test_id,
            "deficiency_type": self.deficiency_type.value,
            "description": self.description,
            "root_cause": self.root_cause,
            "potential_impact": self.potential_impact,
            "likelihood": self.likelihood,
            "severity": self.severity,
            "identified_date": self.identified_date.isoformat(),
            "identified_by": self.identified_by,
            "management_response": self.management_response,
            "remediation_plan": self.remediation_plan,
            "remediation_owner": self.remediation_owner,
            "target_remediation_date": self.target_remediation_date.isoformat(),
            "actual_remediation_date": self.actual_remediation_date.isoformat() if self.actual_remediation_date else None,
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class RegulatoryReport:
    """Regulatory report"""
    report_id: str
    report_type: str
    regulatory_framework: RegulatoryFramework
    reporting_period_start: datetime
    reporting_period_end: datetime
    report_data: Dict[str, Any]
    prepared_by: str
    reviewed_by: Optional[str] = None
    approved_by: Optional[str] = None
    submitted_date: Optional[datetime] = None
    submission_reference: Optional[str] = None
    status: str = "draft"  # draft, under_review, approved, submitted
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "regulatory_framework": self.regulatory_framework.value,
            "reporting_period_start": self.reporting_period_start.isoformat(),
            "reporting_period_end": self.reporting_period_end.isoformat(),
            "report_data": self.report_data,
            "prepared_by": self.prepared_by,
            "reviewed_by": self.reviewed_by,
            "approved_by": self.approved_by,
            "submitted_date": self.submitted_date.isoformat() if self.submitted_date else None,
            "submission_reference": self.submission_reference,
            "status": self.status,
            "created_at": self.created_at.isoformat()
        }

class SOXComplianceManager:
    """SOX compliance management system"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self.reports_dir = Path("/var/log/grandmodel/compliance/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for development
        self.controls: Dict[str, InternalControl] = {}
        self.control_tests: Dict[str, ControlTest] = {}
        self.deficiencies: Dict[str, ControlDeficiency] = {}
        self.regulatory_reports: Dict[str, RegulatoryReport] = {}
        
        # Initialize default controls
        self._initialize_default_controls()
        
        logger.info("SOX Compliance Manager initialized")
    
    def _initialize_default_controls(self):
        """Initialize default SOX controls"""
        # Access controls
        access_control = InternalControl(
            control_id="ITG-001",
            control_name="User Access Management",
            control_description="Controls to ensure appropriate user access to trading systems",
            control_type=ControlType.PREVENTIVE,
            sox_section=SOXSection.SECTION_404,
            business_process="IT General Controls",
            control_objective="Ensure only authorized users have access to trading systems",
            control_activities=[
                "User access provisioning",
                "Periodic access reviews",
                "Access termination procedures",
                "Segregation of duties"
            ],
            control_owner="IT Security Manager",
            control_operator="IT Operations Team",
            testing_frequency=ControlFrequency.QUARTERLY,
            automated=True,
            key_control=True
        )
        
        # Change management
        change_control = InternalControl(
            control_id="ITG-002",
            control_name="Change Management",
            control_description="Controls to ensure proper authorization and testing of system changes",
            control_type=ControlType.PREVENTIVE,
            sox_section=SOXSection.SECTION_404,
            business_process="IT General Controls",
            control_objective="Ensure all system changes are properly authorized and tested",
            control_activities=[
                "Change request approval",
                "Development testing",
                "User acceptance testing",
                "Production deployment controls"
            ],
            control_owner="IT Change Manager",
            control_operator="Development Team",
            testing_frequency=ControlFrequency.QUARTERLY,
            automated=False,
            key_control=True
        )
        
        # Data backup and recovery
        backup_control = InternalControl(
            control_id="ITG-003",
            control_name="Data Backup and Recovery",
            control_description="Controls to ensure data integrity and availability",
            control_type=ControlType.DETECTIVE,
            sox_section=SOXSection.SECTION_404,
            business_process="IT General Controls",
            control_objective="Ensure data can be recovered in case of system failure",
            control_activities=[
                "Daily backup procedures",
                "Backup verification",
                "Recovery testing",
                "Offsite storage"
            ],
            control_owner="IT Operations Manager",
            control_operator="IT Operations Team",
            testing_frequency=ControlFrequency.MONTHLY,
            automated=True,
            key_control=True
        )
        
        # Financial reporting controls
        reporting_control = InternalControl(
            control_id="FIN-001",
            control_name="Financial Reporting Accuracy",
            control_description="Controls to ensure accuracy of financial reporting",
            control_type=ControlType.PREVENTIVE,
            sox_section=SOXSection.SECTION_302,
            business_process="Financial Reporting",
            control_objective="Ensure financial reports are accurate and complete",
            control_activities=[
                "Monthly reconciliations",
                "Management review",
                "Independent verification",
                "Variance analysis"
            ],
            control_owner="Finance Manager",
            control_operator="Finance Team",
            testing_frequency=ControlFrequency.MONTHLY,
            automated=False,
            key_control=True
        )
        
        # Trading authorization
        trading_control = InternalControl(
            control_id="TRD-001",
            control_name="Trading Authorization",
            control_description="Controls to ensure trading activities are properly authorized",
            control_type=ControlType.PREVENTIVE,
            sox_section=SOXSection.SECTION_404,
            business_process="Trading",
            control_objective="Ensure all trades are properly authorized and within limits",
            control_activities=[
                "Pre-trade authorization",
                "Position limits monitoring",
                "Trade approval workflow",
                "Exception reporting"
            ],
            control_owner="Trading Manager",
            control_operator="Trading Team",
            testing_frequency=ControlFrequency.DAILY,
            automated=True,
            key_control=True
        )
        
        self.controls = {
            "ITG-001": access_control,
            "ITG-002": change_control,
            "ITG-003": backup_control,
            "FIN-001": reporting_control,
            "TRD-001": trading_control
        }
    
    async def add_control(self, control: InternalControl) -> bool:
        """Add new internal control"""
        try:
            self.controls[control.control_id] = control
            
            if self.redis:
                await self._store_control(control)
            
            logger.info(
                "Control added",
                control_id=control.control_id,
                control_name=control.control_name
            )
            
            return True
            
        except Exception as e:
            logger.error("Error adding control", control_id=control.control_id, error=str(e))
            return False
    
    async def _store_control(self, control: InternalControl):
        """Store control in Redis"""
        encrypted_data = encrypt_data(control.to_dict())
        
        await self.redis.setex(
            f"sox_control:{control.control_id}",
            10 * 365 * 24 * 3600,  # 10 years
            encrypted_data.to_dict()
        )
    
    async def execute_control_test(self, 
                                 control_id: str,
                                 test_period_start: datetime,
                                 test_period_end: datetime,
                                 tester: str,
                                 test_procedures: List[str],
                                 sample_size: int,
                                 sample_selection_method: str,
                                 test_results: Dict[str, Any]) -> str:
        """Execute control test"""
        
        if control_id not in self.controls:
            raise ValueError(f"Control not found: {control_id}")
        
        test_id = str(uuid.uuid4())
        
        # Determine test status based on results
        status = ControlStatus.EFFECTIVE
        deficiencies = []
        
        # Analyze test results
        if "exceptions" in test_results and test_results["exceptions"]:
            if test_results["exceptions"] > sample_size * 0.05:  # > 5% exception rate
                status = ControlStatus.INEFFECTIVE
                deficiencies.append("High exception rate detected")
            elif test_results["exceptions"] > 0:
                status = ControlStatus.DEFICIENT
                deficiencies.append("Exceptions identified")
        
        control_test = ControlTest(
            test_id=test_id,
            control_id=control_id,
            test_date=datetime.utcnow(),
            test_period_start=test_period_start,
            test_period_end=test_period_end,
            tester=tester,
            test_procedures=test_procedures,
            sample_size=sample_size,
            sample_selection_method=sample_selection_method,
            test_results=test_results,
            status=status,
            deficiencies=deficiencies
        )
        
        self.control_tests[test_id] = control_test
        
        if self.redis:
            await self._store_control_test(control_test)
        
        # Create deficiency records if needed
        if deficiencies:
            await self._create_deficiency_records(control_test)
        
        logger.info(
            "Control test executed",
            test_id=test_id,
            control_id=control_id,
            status=status.value,
            deficiencies_count=len(deficiencies)
        )
        
        return test_id
    
    async def _store_control_test(self, control_test: ControlTest):
        """Store control test in Redis"""
        encrypted_data = encrypt_data(control_test.to_dict())
        
        await self.redis.setex(
            f"sox_test:{control_test.test_id}",
            10 * 365 * 24 * 3600,  # 10 years
            encrypted_data.to_dict()
        )
        
        # Index by control
        await self.redis.sadd(
            f"sox_control_tests:{control_test.control_id}",
            control_test.test_id
        )
    
    async def _create_deficiency_records(self, control_test: ControlTest):
        """Create deficiency records for failed test"""
        for deficiency_desc in control_test.deficiencies:
            deficiency_id = str(uuid.uuid4())
            
            # Determine deficiency type based on control and impact
            control = self.controls[control_test.control_id]
            deficiency_type = DeficiencyType.MINOR_DEFICIENCY
            
            if control.key_control:
                deficiency_type = DeficiencyType.SIGNIFICANT_DEFICIENCY
            
            if "High exception rate" in deficiency_desc:
                deficiency_type = DeficiencyType.MATERIAL_WEAKNESS
            
            deficiency = ControlDeficiency(
                deficiency_id=deficiency_id,
                control_id=control_test.control_id,
                test_id=control_test.test_id,
                deficiency_type=deficiency_type,
                description=deficiency_desc,
                root_cause="To be determined",
                potential_impact="To be assessed",
                likelihood="Medium",
                severity="Medium",
                identified_date=datetime.utcnow(),
                identified_by=control_test.tester,
                management_response="Under review",
                remediation_plan="To be developed",
                remediation_owner=control.control_owner,
                target_remediation_date=datetime.utcnow() + timedelta(days=30)
            )
            
            self.deficiencies[deficiency_id] = deficiency
            
            if self.redis:
                await self._store_deficiency(deficiency)
    
    async def _store_deficiency(self, deficiency: ControlDeficiency):
        """Store deficiency in Redis"""
        encrypted_data = encrypt_data(deficiency.to_dict())
        
        await self.redis.setex(
            f"sox_deficiency:{deficiency.deficiency_id}",
            10 * 365 * 24 * 3600,  # 10 years
            encrypted_data.to_dict()
        )
        
        # Index by control
        await self.redis.sadd(
            f"sox_control_deficiencies:{deficiency.control_id}",
            deficiency.deficiency_id
        )
        
        # Index by type
        await self.redis.sadd(
            f"sox_deficiency_type:{deficiency.deficiency_type.value}",
            deficiency.deficiency_id
        )
    
    async def generate_sox_404_report(self, 
                                    reporting_period_start: datetime,
                                    reporting_period_end: datetime,
                                    prepared_by: str) -> str:
        """Generate SOX 404 management assessment report"""
        
        report_id = str(uuid.uuid4())
        
        # Analyze controls effectiveness
        controls_summary = {}
        deficiencies_summary = {}
        
        for control_id, control in self.controls.items():
            # Get recent tests for this control
            recent_tests = [
                test for test in self.control_tests.values()
                if test.control_id == control_id and
                test.test_date >= reporting_period_start and
                test.test_date <= reporting_period_end
            ]
            
            if recent_tests:
                # Get latest test status
                latest_test = max(recent_tests, key=lambda t: t.test_date)
                controls_summary[control_id] = {
                    "control_name": control.control_name,
                    "status": latest_test.status.value,
                    "last_test_date": latest_test.test_date.isoformat(),
                    "tests_performed": len(recent_tests)
                }
            else:
                controls_summary[control_id] = {
                    "control_name": control.control_name,
                    "status": "not_tested",
                    "last_test_date": None,
                    "tests_performed": 0
                }
        
        # Analyze deficiencies
        for deficiency_id, deficiency in self.deficiencies.items():
            if (deficiency.identified_date >= reporting_period_start and
                deficiency.identified_date <= reporting_period_end):
                
                deficiency_type = deficiency.deficiency_type.value
                if deficiency_type not in deficiencies_summary:
                    deficiencies_summary[deficiency_type] = {
                        "count": 0,
                        "open": 0,
                        "remediated": 0
                    }
                
                deficiencies_summary[deficiency_type]["count"] += 1
                
                if deficiency.status == "open":
                    deficiencies_summary[deficiency_type]["open"] += 1
                elif deficiency.status == "remediated":
                    deficiencies_summary[deficiency_type]["remediated"] += 1
        
        # Determine overall assessment
        material_weaknesses = deficiencies_summary.get("material_weakness", {}).get("open", 0)
        overall_assessment = "Effective" if material_weaknesses == 0 else "Ineffective"
        
        report_data = {
            "assessment_period": {
                "start": reporting_period_start.isoformat(),
                "end": reporting_period_end.isoformat()
            },
            "overall_assessment": overall_assessment,
            "controls_tested": len([c for c in controls_summary.values() if c["status"] != "not_tested"]),
            "total_controls": len(controls_summary),
            "effective_controls": len([c for c in controls_summary.values() if c["status"] == "effective"]),
            "controls_summary": controls_summary,
            "deficiencies_summary": deficiencies_summary,
            "material_weaknesses": material_weaknesses,
            "significant_deficiencies": deficiencies_summary.get("significant_deficiency", {}).get("open", 0),
            "management_conclusion": {
                "assessment": overall_assessment,
                "basis": "Based on testing of key controls over financial reporting",
                "limitations": "Assessment based on controls in place as of the reporting date"
            }
        }
        
        report = RegulatoryReport(
            report_id=report_id,
            report_type="SOX 404 Management Assessment",
            regulatory_framework=RegulatoryFramework.SOX,
            reporting_period_start=reporting_period_start,
            reporting_period_end=reporting_period_end,
            report_data=report_data,
            prepared_by=prepared_by
        )
        
        self.regulatory_reports[report_id] = report
        
        if self.redis:
            await self._store_regulatory_report(report)
        
        # Generate report file
        await self._generate_report_file(report)
        
        logger.info(
            "SOX 404 report generated",
            report_id=report_id,
            assessment=overall_assessment,
            controls_tested=len([c for c in controls_summary.values() if c["status"] != "not_tested"])
        )
        
        return report_id
    
    async def _store_regulatory_report(self, report: RegulatoryReport):
        """Store regulatory report in Redis"""
        encrypted_data = encrypt_data(report.to_dict())
        
        await self.redis.setex(
            f"sox_report:{report.report_id}",
            10 * 365 * 24 * 3600,  # 10 years
            encrypted_data.to_dict()
        )
        
        # Index by framework
        await self.redis.sadd(
            f"sox_reports:{report.regulatory_framework.value}",
            report.report_id
        )
    
    async def _generate_report_file(self, report: RegulatoryReport):
        """Generate report file"""
        report_file = self.reports_dir / f"{report.report_id}_{report.report_type.replace(' ', '_')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        logger.info("Report file generated", report_file=str(report_file))
    
    async def get_control_testing_status(self) -> Dict[str, Any]:
        """Get control testing status summary"""
        status_summary = {
            "total_controls": len(self.controls),
            "tested_controls": 0,
            "effective_controls": 0,
            "ineffective_controls": 0,
            "deficient_controls": 0,
            "untested_controls": 0,
            "overdue_tests": [],
            "recent_tests": []
        }
        
        current_date = datetime.utcnow()
        
        for control_id, control in self.controls.items():
            # Get latest test for this control
            control_tests = [test for test in self.control_tests.values() if test.control_id == control_id]
            
            if control_tests:
                latest_test = max(control_tests, key=lambda t: t.test_date)
                status_summary["tested_controls"] += 1
                
                if latest_test.status == ControlStatus.EFFECTIVE:
                    status_summary["effective_controls"] += 1
                elif latest_test.status == ControlStatus.INEFFECTIVE:
                    status_summary["ineffective_controls"] += 1
                elif latest_test.status == ControlStatus.DEFICIENT:
                    status_summary["deficient_controls"] += 1
                
                # Check if test is overdue
                if control.testing_frequency == ControlFrequency.MONTHLY:
                    next_test_due = latest_test.test_date + timedelta(days=30)
                elif control.testing_frequency == ControlFrequency.QUARTERLY:
                    next_test_due = latest_test.test_date + timedelta(days=90)
                elif control.testing_frequency == ControlFrequency.ANNUALLY:
                    next_test_due = latest_test.test_date + timedelta(days=365)
                else:
                    next_test_due = latest_test.test_date + timedelta(days=30)
                
                if current_date > next_test_due:
                    status_summary["overdue_tests"].append({
                        "control_id": control_id,
                        "control_name": control.control_name,
                        "last_test_date": latest_test.test_date.isoformat(),
                        "due_date": next_test_due.isoformat(),
                        "days_overdue": (current_date - next_test_due).days
                    })
                
                # Add to recent tests if within last 30 days
                if (current_date - latest_test.test_date).days <= 30:
                    status_summary["recent_tests"].append({
                        "control_id": control_id,
                        "control_name": control.control_name,
                        "test_date": latest_test.test_date.isoformat(),
                        "status": latest_test.status.value,
                        "tester": latest_test.tester
                    })
            else:
                status_summary["untested_controls"] += 1
        
        return status_summary
    
    async def get_deficiencies_summary(self) -> Dict[str, Any]:
        """Get deficiencies summary"""
        summary = {
            "total_deficiencies": len(self.deficiencies),
            "by_type": {},
            "by_status": {},
            "overdue_remediations": [],
            "recent_deficiencies": []
        }
        
        current_date = datetime.utcnow()
        
        for deficiency_id, deficiency in self.deficiencies.items():
            # Count by type
            deficiency_type = deficiency.deficiency_type.value
            if deficiency_type not in summary["by_type"]:
                summary["by_type"][deficiency_type] = 0
            summary["by_type"][deficiency_type] += 1
            
            # Count by status
            if deficiency.status not in summary["by_status"]:
                summary["by_status"][deficiency.status] = 0
            summary["by_status"][deficiency.status] += 1
            
            # Check for overdue remediations
            if (deficiency.status in ["open", "in_progress"] and
                current_date > deficiency.target_remediation_date):
                summary["overdue_remediations"].append({
                    "deficiency_id": deficiency_id,
                    "control_id": deficiency.control_id,
                    "type": deficiency.deficiency_type.value,
                    "description": deficiency.description,
                    "target_date": deficiency.target_remediation_date.isoformat(),
                    "days_overdue": (current_date - deficiency.target_remediation_date).days,
                    "owner": deficiency.remediation_owner
                })
            
            # Add to recent deficiencies if identified within last 30 days
            if (current_date - deficiency.identified_date).days <= 30:
                summary["recent_deficiencies"].append({
                    "deficiency_id": deficiency_id,
                    "control_id": deficiency.control_id,
                    "type": deficiency.deficiency_type.value,
                    "description": deficiency.description,
                    "identified_date": deficiency.identified_date.isoformat(),
                    "status": deficiency.status
                })
        
        return summary
    
    async def remediate_deficiency(self, 
                                 deficiency_id: str,
                                 remediation_details: str,
                                 remediated_by: str) -> bool:
        """Mark deficiency as remediated"""
        if deficiency_id not in self.deficiencies:
            return False
        
        deficiency = self.deficiencies[deficiency_id]
        deficiency.status = "remediated"
        deficiency.actual_remediation_date = datetime.utcnow()
        deficiency.management_response = remediation_details
        
        if self.redis:
            await self._store_deficiency(deficiency)
        
        logger.info(
            "Deficiency remediated",
            deficiency_id=deficiency_id,
            control_id=deficiency.control_id,
            remediated_by=remediated_by
        )
        
        return True
    
    async def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        control_status = await self.get_control_testing_status()
        deficiencies = await self.get_deficiencies_summary()
        
        # Calculate compliance score
        total_controls = control_status["total_controls"]
        effective_controls = control_status["effective_controls"]
        compliance_score = (effective_controls / total_controls * 100) if total_controls > 0 else 0
        
        # Risk level based on material weaknesses
        material_weaknesses = deficiencies["by_type"].get("material_weakness", 0)
        if material_weaknesses > 0:
            risk_level = "HIGH"
        elif deficiencies["by_type"].get("significant_deficiency", 0) > 0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "compliance_score": round(compliance_score, 2),
            "risk_level": risk_level,
            "control_status": control_status,
            "deficiencies": deficiencies,
            "key_metrics": {
                "total_controls": total_controls,
                "effective_controls": effective_controls,
                "material_weaknesses": material_weaknesses,
                "overdue_tests": len(control_status["overdue_tests"]),
                "overdue_remediations": len(deficiencies["overdue_remediations"])
            },
            "last_updated": datetime.utcnow().isoformat()
        }

# Global SOX compliance manager
sox_manager: Optional[SOXComplianceManager] = None

async def get_sox_manager() -> SOXComplianceManager:
    """Get or create SOX compliance manager"""
    global sox_manager
    
    if sox_manager is None:
        # Initialize Redis client
        redis_client = None
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                redis_client = await redis.from_url(redis_url)
            except Exception as e:
                logger.error("Failed to connect to Redis for SOX compliance", error=str(e))
        
        sox_manager = SOXComplianceManager(redis_client)
    
    return sox_manager

# API models
class ControlTestRequest(BaseModel):
    """Control test request model"""
    control_id: str = Field(..., min_length=1, max_length=50)
    test_period_start: datetime
    test_period_end: datetime
    tester: str = Field(..., min_length=1, max_length=100)
    test_procedures: List[str] = Field(..., min_items=1)
    sample_size: int = Field(..., ge=1, le=10000)
    sample_selection_method: str = Field(..., min_length=1, max_length=200)
    test_results: Dict[str, Any]

class DeficiencyRemediationRequest(BaseModel):
    """Deficiency remediation request model"""
    deficiency_id: str = Field(..., min_length=1, max_length=50)
    remediation_details: str = Field(..., min_length=1, max_length=2000)
    remediated_by: str = Field(..., min_length=1, max_length=100)

class SOXReportRequest(BaseModel):
    """SOX report generation request model"""
    reporting_period_start: datetime
    reporting_period_end: datetime
    prepared_by: str = Field(..., min_length=1, max_length=100)
    report_type: str = Field("SOX 404 Management Assessment", min_length=1, max_length=100)

# Convenience functions
async def execute_control_test(request: ControlTestRequest) -> str:
    """Execute control test"""
    manager = await get_sox_manager()
    return await manager.execute_control_test(
        control_id=request.control_id,
        test_period_start=request.test_period_start,
        test_period_end=request.test_period_end,
        tester=request.tester,
        test_procedures=request.test_procedures,
        sample_size=request.sample_size,
        sample_selection_method=request.sample_selection_method,
        test_results=request.test_results
    )

async def generate_sox_report(request: SOXReportRequest) -> str:
    """Generate SOX report"""
    manager = await get_sox_manager()
    return await manager.generate_sox_404_report(
        reporting_period_start=request.reporting_period_start,
        reporting_period_end=request.reporting_period_end,
        prepared_by=request.prepared_by
    )

async def get_compliance_dashboard() -> Dict[str, Any]:
    """Get compliance dashboard"""
    manager = await get_sox_manager()
    return await manager.get_compliance_dashboard()
