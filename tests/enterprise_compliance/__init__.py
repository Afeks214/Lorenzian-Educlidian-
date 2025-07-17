"""
Enterprise-Grade Compliance Testing Framework
Agent 3: Regulatory Compliance Testing

Comprehensive testing for regulatory compliance across multiple jurisdictions including:
- SEC Compliance Testing (Rule 605, Regulation SHO, market maker obligations)
- MiFID II Compliance Testing (transaction reporting, best execution, systematic internalizer)
- FINRA Compliance Testing (OATS, CAT, trade reporting, ADF)
- CFTC Compliance Testing (derivatives reporting, swap dealer, position limits)
- Cross-Jurisdictional Compliance Testing (regulatory conflicts, equivalence assessment)

This framework provides comprehensive regulatory compliance validation across all major
financial regulatory regimes with automated testing and reporting capabilities.
"""

from .enterprise_audit_system import EnterpriseAuditSystem
from .forensic_analyzer import ForensicAnalyzer
from .compliance_reporter import ComplianceReporter
from .immutable_logger import ImmutableLogger
from .regulatory_validator import RegulatoryValidator

# New regulatory compliance test modules
from .test_sec_compliance import (
    SECComplianceValidator,
    SECComplianceReporter,
    OrderExecutionData,
    ShortSaleData,
    MarketMakerData
)

from .test_mifid2_compliance import (
    MiFIDIIComplianceValidator,
    MiFIDIIComplianceReporter,
    TransactionReport,
    BestExecutionData,
    SystematicInternalizerData,
    TradeTransparencyData
)

from .test_finra_compliance import (
    FINRAComplianceValidator,
    FINRAComplianceReporter,
    OATSReportData,
    CATReportData,
    TradeReportData,
    ADFSubmissionData
)

from .test_cftc_compliance import (
    CFTCComplianceValidator,
    CFTCComplianceReporter,
    DerivativesReportData,
    SwapDealerData,
    PositionLimitData,
    RTRReportData,
    SDRReportData
)

from .test_cross_jurisdictional import (
    CrossJurisdictionalValidator,
    CrossJurisdictionalReporter,
    CrossBorderTransaction,
    RegulatoryConflict,
    EquivalenceAssessment,
    ComplianceMapping
)

__all__ = [
    # Original enterprise compliance components
    'EnterpriseAuditSystem',
    'ForensicAnalyzer', 
    'ComplianceReporter',
    'ImmutableLogger',
    'RegulatoryValidator',
    
    # SEC Compliance Testing
    'SECComplianceValidator',
    'SECComplianceReporter',
    'OrderExecutionData',
    'ShortSaleData',
    'MarketMakerData',
    
    # MiFID II Compliance Testing
    'MiFIDIIComplianceValidator',
    'MiFIDIIComplianceReporter',
    'TransactionReport',
    'BestExecutionData',
    'SystematicInternalizerData',
    'TradeTransparencyData',
    
    # FINRA Compliance Testing
    'FINRAComplianceValidator',
    'FINRAComplianceReporter',
    'OATSReportData',
    'CATReportData',
    'TradeReportData',
    'ADFSubmissionData',
    
    # CFTC Compliance Testing
    'CFTCComplianceValidator',
    'CFTCComplianceReporter',
    'DerivativesReportData',
    'SwapDealerData',
    'PositionLimitData',
    'RTRReportData',
    'SDRReportData',
    
    # Cross-Jurisdictional Compliance Testing
    'CrossJurisdictionalValidator',
    'CrossJurisdictionalReporter',
    'CrossBorderTransaction',
    'RegulatoryConflict',
    'EquivalenceAssessment',
    'ComplianceMapping'
]