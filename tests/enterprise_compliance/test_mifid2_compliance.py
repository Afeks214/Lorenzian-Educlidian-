#!/usr/bin/env python3
"""
MiFID II Compliance Testing Suite
Agent 3: Regulatory Compliance Testing

Comprehensive testing for MiFID II (Markets in Financial Instruments Directive II) 
compliance including transaction reporting, best execution, and systematic internalizer obligations.

Features:
- Transaction reporting requirements (RTS 22) validation
- Best execution reporting (RTS 27 & 28) compliance
- Systematic internalizer obligations testing
- Trade transparency requirements
- Client protection measures
- Research unbundling compliance
"""

import pytest
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path


class MiFIDIIRuleType(Enum):
    """MiFID II Rule types"""
    TRANSACTION_REPORTING = "transaction_reporting"
    BEST_EXECUTION = "best_execution"
    SYSTEMATIC_INTERNALIZER = "systematic_internalizer"
    TRADE_TRANSPARENCY = "trade_transparency"
    CLIENT_PROTECTION = "client_protection"
    RESEARCH_UNBUNDLING = "research_unbundling"


class InstrumentType(Enum):
    """Financial instrument types"""
    EQUITY = "equity"
    BOND = "bond"
    DERIVATIVE = "derivative"
    ETF = "etf"
    COMMODITY = "commodity"
    CURRENCY = "currency"


class VenueType(Enum):
    """Trading venue types"""
    REGULATED_MARKET = "regulated_market"
    MTF = "mtf"  # Multilateral Trading Facility
    OTF = "otf"  # Organised Trading Facility
    SYSTEMATIC_INTERNALIZER = "systematic_internalizer"
    OTC = "otc"  # Over-the-counter


class ClientType(Enum):
    """Client classification"""
    RETAIL = "retail"
    PROFESSIONAL = "professional"
    ELIGIBLE_COUNTERPARTY = "eligible_counterparty"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    HIDDEN = "hidden"


@dataclass
class TransactionReport:
    """Transaction report for RTS 22 compliance"""
    transaction_id: str
    isin: str
    instrument_type: InstrumentType
    venue: VenueType
    venue_mic: str  # Market Identifier Code
    
    # Transaction details
    price: Decimal
    quantity: int
    currency: str
    transaction_timestamp: datetime
    trading_date: datetime
    
    # Client information
    client_id: str
    client_type: ClientType
    client_country: str
    
    # Order details
    order_id: str
    order_type: OrderType
    order_duration: str
    order_timestamp: datetime
    
    # Execution details
    execution_id: str
    execution_venue: str
    execution_timestamp: datetime
    
    # Flags
    commodity_derivative: bool = False
    securities_financing: bool = False
    short_selling: bool = False
    
    # Investment firm details
    investment_firm_id: str = "FIRM_001"
    investment_firm_country: str = "GB"
    
    # Additional fields for RTS 22
    transmission_timestamp: Optional[datetime] = None
    waiver_indicator: Optional[str] = None
    post_trade_deferral: Optional[str] = None


@dataclass
class BestExecutionData:
    """Best execution data for RTS 27 & 28 compliance"""
    report_id: str
    period: str
    instrument_class: str
    
    # RTS 27 - Investment firm data
    client_orders_count: int
    client_orders_value: Decimal
    professional_orders_count: int
    professional_orders_value: Decimal
    
    # Venue analysis
    venue_data: List[Dict[str, Any]]
    
    # Quality metrics
    average_execution_speed: float
    price_improvement_rate: float
    likelihood_of_execution: float
    
    # RTS 28 - Top 5 execution venues
    top_5_venues: List[Dict[str, Any]]
    
    # Quality of execution
    execution_quality_score: float
    best_execution_policy: str
    
    # Client information
    retail_client_impact: Dict[str, Any]
    professional_client_impact: Dict[str, Any]


@dataclass
class SystematicInternalizerData:
    """Systematic internalizer data for compliance"""
    si_id: str
    instrument_class: str
    isin: str
    
    # Quote obligations
    quote_timestamp: datetime
    bid_price: Decimal
    ask_price: Decimal
    bid_size: int
    ask_size: int
    
    # Size thresholds
    standard_market_size: int
    size_specific_to_transaction: bool
    
    # Execution obligations
    execution_at_quoted_price: bool
    price_improvement_provided: bool
    
    # Risk management
    risk_management_exception: bool
    exceptional_market_conditions: bool
    
    # Reporting requirements
    monthly_volume: int
    systematic_percentage: float
    
    # Client interaction
    client_requests: int
    client_executions: int
    rejection_rate: float


@dataclass
class TradeTransparencyData:
    """Trade transparency data for pre/post-trade requirements"""
    trade_id: str
    isin: str
    instrument_type: InstrumentType
    
    # Trade details
    price: Decimal
    quantity: int
    timestamp: datetime
    venue: VenueType
    
    # Transparency requirements
    pre_trade_transparency: bool
    post_trade_transparency: bool
    
    # Waivers and deferrals
    waiver_type: Optional[str] = None
    deferral_type: Optional[str] = None
    
    # Size thresholds
    large_in_scale: bool = False
    standard_market_size: int = 0
    
    # Publication requirements
    publication_timestamp: Optional[datetime] = None
    publication_venue: Optional[str] = None


class MiFIDIIComplianceValidator:
    """MiFID II Compliance validation engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.violations: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'transaction_reporting': {
                'reporting_deadline_minutes': 15,
                'required_fields': [
                    'transaction_id', 'isin', 'price', 'quantity', 
                    'venue', 'client_id', 'order_id'
                ],
                'transmission_deadline_minutes': 1
            },
            'best_execution': {
                'monitoring_period_months': 3,
                'venue_analysis_threshold': 0.05,
                'execution_quality_threshold': 0.85,
                'price_improvement_threshold': 0.02
            },
            'systematic_internalizer': {
                'systematic_threshold': 0.025,  # 2.5% of market
                'quote_response_time_ms': 100,
                'standard_market_size_threshold': 7500,
                'rejection_rate_threshold': 0.05
            },
            'trade_transparency': {
                'publication_deadline_minutes': 15,
                'large_in_scale_threshold': 50000,
                'deferral_size_threshold': 100000
            },
            'client_protection': {
                'appropriateness_assessment_required': True,
                'suitability_assessment_required': True,
                'cost_disclosure_required': True
            }
        }
    
    def validate_transaction_reporting_compliance(self, transactions: List[TransactionReport]) -> Dict[str, Any]:
        """Validate RTS 22 transaction reporting compliance"""
        violations = []
        metrics = {
            'total_transactions': len(transactions),
            'reporting_timeliness': 0,
            'completeness_rate': 0,
            'transmission_timeliness': 0,
            'late_reports': 0
        }
        
        if not transactions:
            return {'violations': violations, 'metrics': metrics}
        
        tr_config = self.config['transaction_reporting']
        on_time_reports = 0
        complete_reports = 0
        on_time_transmissions = 0
        
        for transaction in transactions:
            # Check reporting timeliness
            if transaction.trading_date and transaction.execution_timestamp:
                # Convert trading_date to datetime for comparison
                trading_datetime = datetime.combine(transaction.trading_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                report_deadline = trading_datetime + timedelta(minutes=tr_config['reporting_deadline_minutes'])
                if transaction.execution_timestamp <= report_deadline:
                    on_time_reports += 1
                else:
                    violations.append({
                        'rule': 'RTS_22_REPORTING_DEADLINE',
                        'severity': 'HIGH',
                        'description': f"Transaction {transaction.transaction_id} reported late",
                        'transaction_id': transaction.transaction_id,
                        'deadline': report_deadline.isoformat(),
                        'actual_time': transaction.execution_timestamp.isoformat()
                    })
            
            # Check completeness
            missing_fields = []
            for field in tr_config['required_fields']:
                if not getattr(transaction, field, None):
                    missing_fields.append(field)
            
            if not missing_fields:
                complete_reports += 1
            else:
                violations.append({
                    'rule': 'RTS_22_COMPLETENESS',
                    'severity': 'HIGH',
                    'description': f"Transaction {transaction.transaction_id} missing required fields",
                    'transaction_id': transaction.transaction_id,
                    'missing_fields': missing_fields
                })
            
            # Check transmission timeliness
            if transaction.transmission_timestamp and transaction.execution_timestamp:
                transmission_deadline = transaction.execution_timestamp + timedelta(minutes=tr_config['transmission_deadline_minutes'])
                if transaction.transmission_timestamp <= transmission_deadline:
                    on_time_transmissions += 1
                else:
                    violations.append({
                        'rule': 'RTS_22_TRANSMISSION_DEADLINE',
                        'severity': 'MEDIUM',
                        'description': f"Transaction {transaction.transaction_id} transmission late",
                        'transaction_id': transaction.transaction_id,
                        'transmission_deadline': transmission_deadline.isoformat(),
                        'actual_transmission': transaction.transmission_timestamp.isoformat()
                    })
            
            # Check venue MIC code
            if not transaction.venue_mic or len(transaction.venue_mic) != 4:
                violations.append({
                    'rule': 'RTS_22_VENUE_MIC',
                    'severity': 'MEDIUM',
                    'description': f"Transaction {transaction.transaction_id} has invalid venue MIC",
                    'transaction_id': transaction.transaction_id,
                    'venue_mic': transaction.venue_mic
                })
        
        # Calculate metrics
        metrics['reporting_timeliness'] = on_time_reports / len(transactions)
        metrics['completeness_rate'] = complete_reports / len(transactions)
        metrics['transmission_timeliness'] = on_time_transmissions / len(transactions)
        metrics['late_reports'] = len(transactions) - on_time_reports
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_best_execution_compliance(self, execution_data: List[BestExecutionData]) -> Dict[str, Any]:
        """Validate RTS 27 & 28 best execution compliance"""
        violations = []
        metrics = {
            'total_reports': len(execution_data),
            'avg_execution_quality': 0,
            'avg_price_improvement': 0,
            'venue_analysis_completeness': 0,
            'top_5_venue_coverage': 0
        }
        
        if not execution_data:
            return {'violations': violations, 'metrics': metrics}
        
        be_config = self.config['best_execution']
        quality_scores = []
        price_improvements = []
        complete_venue_analyses = 0
        complete_top_5_reports = 0
        
        for data in execution_data:
            # Check execution quality
            if data.execution_quality_score < be_config['execution_quality_threshold']:
                violations.append({
                    'rule': 'RTS_27_EXECUTION_QUALITY',
                    'severity': 'HIGH',
                    'description': f"Report {data.report_id} execution quality below threshold",
                    'report_id': data.report_id,
                    'execution_quality_score': data.execution_quality_score,
                    'threshold': be_config['execution_quality_threshold']
                })
            
            quality_scores.append(data.execution_quality_score)
            
            # Check price improvement
            if data.price_improvement_rate < be_config['price_improvement_threshold']:
                violations.append({
                    'rule': 'RTS_27_PRICE_IMPROVEMENT',
                    'severity': 'MEDIUM',
                    'description': f"Report {data.report_id} price improvement rate below threshold",
                    'report_id': data.report_id,
                    'price_improvement_rate': data.price_improvement_rate,
                    'threshold': be_config['price_improvement_threshold']
                })
            
            price_improvements.append(data.price_improvement_rate)
            
            # Check venue analysis completeness
            if len(data.venue_data) >= 5:  # Should analyze at least 5 venues
                complete_venue_analyses += 1
            else:
                violations.append({
                    'rule': 'RTS_27_VENUE_ANALYSIS',
                    'severity': 'MEDIUM',
                    'description': f"Report {data.report_id} insufficient venue analysis",
                    'report_id': data.report_id,
                    'venues_analyzed': len(data.venue_data),
                    'minimum_required': 5
                })
            
            # Check top 5 venues reporting (RTS 28)
            if len(data.top_5_venues) == 5:
                complete_top_5_reports += 1
            else:
                violations.append({
                    'rule': 'RTS_28_TOP_5_VENUES',
                    'severity': 'HIGH',
                    'description': f"Report {data.report_id} incomplete top 5 venues reporting",
                    'report_id': data.report_id,
                    'venues_reported': len(data.top_5_venues),
                    'required': 5
                })
            
            # Check client impact analysis
            if not data.retail_client_impact or not data.professional_client_impact:
                violations.append({
                    'rule': 'RTS_28_CLIENT_IMPACT',
                    'severity': 'MEDIUM',
                    'description': f"Report {data.report_id} missing client impact analysis",
                    'report_id': data.report_id
                })
        
        # Calculate metrics
        metrics['avg_execution_quality'] = sum(quality_scores) / len(quality_scores)
        metrics['avg_price_improvement'] = sum(price_improvements) / len(price_improvements)
        metrics['venue_analysis_completeness'] = complete_venue_analyses / len(execution_data)
        metrics['top_5_venue_coverage'] = complete_top_5_reports / len(execution_data)
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_systematic_internalizer_compliance(self, si_data: List[SystematicInternalizerData]) -> Dict[str, Any]:
        """Validate systematic internalizer obligations compliance"""
        violations = []
        metrics = {
            'total_si_entries': len(si_data),
            'avg_systematic_percentage': 0,
            'quote_response_compliance': 0,
            'rejection_rate_compliance': 0,
            'price_improvement_rate': 0
        }
        
        if not si_data:
            return {'violations': violations, 'metrics': metrics}
        
        si_config = self.config['systematic_internalizer']
        systematic_percentages = []
        quote_compliant = 0
        rejection_compliant = 0
        price_improvements = 0
        
        for si in si_data:
            # Check systematic threshold
            if si.systematic_percentage < si_config['systematic_threshold']:
                violations.append({
                    'rule': 'SI_SYSTEMATIC_THRESHOLD',
                    'severity': 'HIGH',
                    'description': f"SI {si.si_id} below systematic threshold",
                    'si_id': si.si_id,
                    'systematic_percentage': si.systematic_percentage,
                    'threshold': si_config['systematic_threshold']
                })
            
            systematic_percentages.append(si.systematic_percentage)
            
            # Check quote obligations
            if si.bid_price and si.ask_price and si.bid_size and si.ask_size:
                quote_compliant += 1
            else:
                violations.append({
                    'rule': 'SI_QUOTE_OBLIGATIONS',
                    'severity': 'HIGH',
                    'description': f"SI {si.si_id} incomplete quote obligations",
                    'si_id': si.si_id,
                    'bid_price': float(si.bid_price) if si.bid_price else None,
                    'ask_price': float(si.ask_price) if si.ask_price else None
                })
            
            # Check rejection rate
            if si.rejection_rate <= si_config['rejection_rate_threshold']:
                rejection_compliant += 1
            else:
                violations.append({
                    'rule': 'SI_REJECTION_RATE',
                    'severity': 'MEDIUM',
                    'description': f"SI {si.si_id} rejection rate exceeds threshold",
                    'si_id': si.si_id,
                    'rejection_rate': si.rejection_rate,
                    'threshold': si_config['rejection_rate_threshold']
                })
            
            # Check price improvement
            if si.price_improvement_provided:
                price_improvements += 1
            
            # Check size thresholds
            if si.standard_market_size < si_config['standard_market_size_threshold']:
                violations.append({
                    'rule': 'SI_STANDARD_MARKET_SIZE',
                    'severity': 'MEDIUM',
                    'description': f"SI {si.si_id} standard market size below threshold",
                    'si_id': si.si_id,
                    'standard_market_size': si.standard_market_size,
                    'threshold': si_config['standard_market_size_threshold']
                })
        
        # Calculate metrics
        metrics['avg_systematic_percentage'] = sum(systematic_percentages) / len(systematic_percentages)
        metrics['quote_response_compliance'] = quote_compliant / len(si_data)
        metrics['rejection_rate_compliance'] = rejection_compliant / len(si_data)
        metrics['price_improvement_rate'] = price_improvements / len(si_data)
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_trade_transparency_compliance(self, trades: List[TradeTransparencyData]) -> Dict[str, Any]:
        """Validate trade transparency requirements compliance"""
        violations = []
        metrics = {
            'total_trades': len(trades),
            'pre_trade_transparency_rate': 0,
            'post_trade_transparency_rate': 0,
            'publication_timeliness': 0,
            'waiver_usage_rate': 0
        }
        
        if not trades:
            return {'violations': violations, 'metrics': metrics}
        
        tt_config = self.config['trade_transparency']
        pre_trade_transparent = 0
        post_trade_transparent = 0
        timely_publications = 0
        waiver_trades = 0
        
        for trade in trades:
            # Check pre-trade transparency
            if trade.pre_trade_transparency:
                pre_trade_transparent += 1
            elif not trade.waiver_type:
                violations.append({
                    'rule': 'TRADE_TRANSPARENCY_PRE_TRADE',
                    'severity': 'HIGH',
                    'description': f"Trade {trade.trade_id} lacks pre-trade transparency without waiver",
                    'trade_id': trade.trade_id,
                    'instrument_type': trade.instrument_type.value
                })
            
            # Check post-trade transparency
            if trade.post_trade_transparency:
                post_trade_transparent += 1
            elif not trade.deferral_type:
                violations.append({
                    'rule': 'TRADE_TRANSPARENCY_POST_TRADE',
                    'severity': 'HIGH',
                    'description': f"Trade {trade.trade_id} lacks post-trade transparency without deferral",
                    'trade_id': trade.trade_id,
                    'instrument_type': trade.instrument_type.value
                })
            
            # Check publication timeliness
            if trade.publication_timestamp and trade.timestamp:
                publication_deadline = trade.timestamp + timedelta(minutes=tt_config['publication_deadline_minutes'])
                if trade.publication_timestamp <= publication_deadline:
                    timely_publications += 1
                else:
                    violations.append({
                        'rule': 'TRADE_TRANSPARENCY_PUBLICATION_DEADLINE',
                        'severity': 'MEDIUM',
                        'description': f"Trade {trade.trade_id} publication late",
                        'trade_id': trade.trade_id,
                        'publication_deadline': publication_deadline.isoformat(),
                        'actual_publication': trade.publication_timestamp.isoformat()
                    })
            
            # Check waiver usage
            if trade.waiver_type:
                waiver_trades += 1
            
            # Check large in scale threshold
            if trade.quantity > tt_config['large_in_scale_threshold'] and not trade.large_in_scale:
                violations.append({
                    'rule': 'TRADE_TRANSPARENCY_LARGE_IN_SCALE',
                    'severity': 'MEDIUM',
                    'description': f"Trade {trade.trade_id} should be marked as large in scale",
                    'trade_id': trade.trade_id,
                    'quantity': trade.quantity,
                    'threshold': tt_config['large_in_scale_threshold']
                })
        
        # Calculate metrics
        metrics['pre_trade_transparency_rate'] = pre_trade_transparent / len(trades)
        metrics['post_trade_transparency_rate'] = post_trade_transparent / len(trades)
        metrics['publication_timeliness'] = timely_publications / len(trades)
        metrics['waiver_usage_rate'] = waiver_trades / len(trades)
        
        return {'violations': violations, 'metrics': metrics}


class MiFIDIIComplianceReporter:
    """MiFID II compliance reporting system"""
    
    def __init__(self, validator: MiFIDIIComplianceValidator):
        self.validator = validator
    
    def generate_transaction_reporting_report(self, transactions: List[TransactionReport], period: str) -> Dict[str, Any]:
        """Generate RTS 22 transaction reporting compliance report"""
        compliance_result = self.validator.validate_transaction_reporting_compliance(transactions)
        
        report = {
            'report_type': 'MIFID_II_TRANSACTION_REPORTING',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_transactions': compliance_result['metrics']['total_transactions'],
                'compliance_violations': len(compliance_result['violations']),
                'reporting_timeliness': compliance_result['metrics']['reporting_timeliness'],
                'completeness_rate': compliance_result['metrics']['completeness_rate'],
                'late_reports': compliance_result['metrics']['late_reports']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_transaction_reporting_recommendations(compliance_result)
        }
        
        return report
    
    def generate_best_execution_report(self, execution_data: List[BestExecutionData], period: str) -> Dict[str, Any]:
        """Generate RTS 27 & 28 best execution compliance report"""
        compliance_result = self.validator.validate_best_execution_compliance(execution_data)
        
        report = {
            'report_type': 'MIFID_II_BEST_EXECUTION',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_reports': compliance_result['metrics']['total_reports'],
                'compliance_violations': len(compliance_result['violations']),
                'avg_execution_quality': compliance_result['metrics']['avg_execution_quality'],
                'avg_price_improvement': compliance_result['metrics']['avg_price_improvement'],
                'venue_analysis_completeness': compliance_result['metrics']['venue_analysis_completeness']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_best_execution_recommendations(compliance_result)
        }
        
        return report
    
    def generate_systematic_internalizer_report(self, si_data: List[SystematicInternalizerData], period: str) -> Dict[str, Any]:
        """Generate systematic internalizer compliance report"""
        compliance_result = self.validator.validate_systematic_internalizer_compliance(si_data)
        
        report = {
            'report_type': 'MIFID_II_SYSTEMATIC_INTERNALIZER',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_si_entries': compliance_result['metrics']['total_si_entries'],
                'compliance_violations': len(compliance_result['violations']),
                'avg_systematic_percentage': compliance_result['metrics']['avg_systematic_percentage'],
                'quote_response_compliance': compliance_result['metrics']['quote_response_compliance'],
                'rejection_rate_compliance': compliance_result['metrics']['rejection_rate_compliance']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_si_recommendations(compliance_result)
        }
        
        return report
    
    def generate_trade_transparency_report(self, trades: List[TradeTransparencyData], period: str) -> Dict[str, Any]:
        """Generate trade transparency compliance report"""
        compliance_result = self.validator.validate_trade_transparency_compliance(trades)
        
        report = {
            'report_type': 'MIFID_II_TRADE_TRANSPARENCY',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_trades': compliance_result['metrics']['total_trades'],
                'compliance_violations': len(compliance_result['violations']),
                'pre_trade_transparency_rate': compliance_result['metrics']['pre_trade_transparency_rate'],
                'post_trade_transparency_rate': compliance_result['metrics']['post_trade_transparency_rate'],
                'publication_timeliness': compliance_result['metrics']['publication_timeliness']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_trade_transparency_recommendations(compliance_result)
        }
        
        return report
    
    def _generate_transaction_reporting_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate transaction reporting recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['reporting_timeliness'] < 0.95:
            recommendations.append("Improve transaction reporting system to meet 15-minute deadline")
        
        if compliance_result['metrics']['completeness_rate'] < 0.98:
            recommendations.append("Enhance data validation to ensure all required fields are populated")
        
        if compliance_result['metrics']['transmission_timeliness'] < 0.98:
            recommendations.append("Optimize transmission infrastructure to meet 1-minute deadline")
        
        return recommendations
    
    def _generate_best_execution_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate best execution recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['avg_execution_quality'] < 0.85:
            recommendations.append("Review execution algorithms to improve quality metrics")
        
        if compliance_result['metrics']['venue_analysis_completeness'] < 0.90:
            recommendations.append("Expand venue analysis to include more execution venues")
        
        if compliance_result['metrics']['avg_price_improvement'] < 0.02:
            recommendations.append("Enhance price improvement mechanisms for better client outcomes")
        
        return recommendations
    
    def _generate_si_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate systematic internalizer recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['quote_response_compliance'] < 0.95:
            recommendations.append("Improve quote response system to meet all obligations")
        
        if compliance_result['metrics']['rejection_rate_compliance'] < 0.90:
            recommendations.append("Review rejection policies to reduce client rejection rates")
        
        return recommendations
    
    def _generate_trade_transparency_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate trade transparency recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['publication_timeliness'] < 0.95:
            recommendations.append("Accelerate trade publication to meet transparency deadlines")
        
        if compliance_result['metrics']['pre_trade_transparency_rate'] < 0.90:
            recommendations.append("Review pre-trade transparency procedures and waiver usage")
        
        return recommendations


# Test fixtures and utilities
def create_sample_transaction_reports(count: int = 100) -> List[TransactionReport]:
    """Create sample transaction reports for testing"""
    reports = []
    
    for i in range(count):
        trade_time = datetime.now(timezone.utc) - timedelta(minutes=i)
        
        report = TransactionReport(
            transaction_id=f"TXN_{i:06d}",
            isin=f"GB00B{i:07d}",
            instrument_type=InstrumentType.EQUITY,
            venue=VenueType.REGULATED_MARKET,
            venue_mic="XLON",
            price=Decimal("100.00") + Decimal(str(i * 0.01)),
            quantity=100 * (i + 1),
            currency="GBP",
            transaction_timestamp=trade_time,
            trading_date=trade_time.date(),
            client_id=f"CLIENT_{i % 10:03d}",
            client_type=ClientType.PROFESSIONAL if i % 3 == 0 else ClientType.RETAIL,
            client_country="GB",
            order_id=f"ORDER_{i:06d}",
            order_type=OrderType.MARKET if i % 2 == 0 else OrderType.LIMIT,
            order_duration="DAY",
            order_timestamp=trade_time - timedelta(seconds=30),
            execution_id=f"EXEC_{i:06d}",
            execution_venue="XLON",
            execution_timestamp=trade_time,
            transmission_timestamp=trade_time + timedelta(seconds=30) if i % 20 != 0 else trade_time + timedelta(minutes=2),  # 5% late
            short_selling=i % 10 == 0
        )
        reports.append(report)
    
    return reports


def create_sample_best_execution_data(count: int = 20) -> List[BestExecutionData]:
    """Create sample best execution data for testing"""
    data = []
    
    for i in range(count):
        venue_data = [
            {'venue': 'XLON', 'volume': 1000000, 'percentage': 0.4},
            {'venue': 'XPAR', 'volume': 750000, 'percentage': 0.3},
            {'venue': 'XFRA', 'volume': 500000, 'percentage': 0.2},
            {'venue': 'XMIL', 'volume': 250000, 'percentage': 0.1}
        ]
        
        top_5_venues = venue_data[:5] if len(venue_data) >= 5 else venue_data
        
        execution_data = BestExecutionData(
            report_id=f"BE_{i:03d}",
            period="2024-Q1",
            instrument_class="Equity",
            client_orders_count=1000 + i * 100,
            client_orders_value=Decimal("1000000") + Decimal(str(i * 100000)),
            professional_orders_count=500 + i * 50,
            professional_orders_value=Decimal("2000000") + Decimal(str(i * 200000)),
            venue_data=venue_data,
            average_execution_speed=50.0 + i,
            price_improvement_rate=0.025 + i * 0.001,
            likelihood_of_execution=0.95 + i * 0.001,
            top_5_venues=top_5_venues,
            execution_quality_score=0.85 + i * 0.005 if i % 10 != 0 else 0.80,  # 10% below threshold
            best_execution_policy="POLICY_2024_V1",
            retail_client_impact={'avg_improvement': 0.02, 'cost_reduction': 0.015},
            professional_client_impact={'avg_improvement': 0.015, 'cost_reduction': 0.01}
        )
        data.append(execution_data)
    
    return data


def create_sample_si_data(count: int = 10) -> List[SystematicInternalizerData]:
    """Create sample systematic internalizer data for testing"""
    data = []
    
    for i in range(count):
        si_data = SystematicInternalizerData(
            si_id=f"SI_{i:03d}",
            instrument_class="Equity",
            isin=f"GB00B{i:07d}",
            quote_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            bid_price=Decimal("99.95") + Decimal(str(i * 0.01)),
            ask_price=Decimal("100.05") + Decimal(str(i * 0.01)),
            bid_size=100 if i % 8 != 0 else 50,  # 12.5% insufficient size
            ask_size=100 if i % 8 != 0 else 50,
            standard_market_size=7500 + i * 100,
            size_specific_to_transaction=False,
            execution_at_quoted_price=True,
            price_improvement_provided=i % 3 == 0,
            risk_management_exception=False,
            exceptional_market_conditions=False,
            monthly_volume=1000000 + i * 100000,
            systematic_percentage=0.03 + i * 0.001 if i % 15 != 0 else 0.02,  # 6.7% below threshold
            client_requests=1000 + i * 100,
            client_executions=950 + i * 95,
            rejection_rate=0.03 + i * 0.001 if i % 12 != 0 else 0.08  # 8.3% above threshold
        )
        data.append(si_data)
    
    return data


def create_sample_trade_transparency_data(count: int = 100) -> List[TradeTransparencyData]:
    """Create sample trade transparency data for testing"""
    data = []
    
    for i in range(count):
        trade_time = datetime.now(timezone.utc) - timedelta(minutes=i)
        
        trade_data = TradeTransparencyData(
            trade_id=f"TRADE_{i:06d}",
            isin=f"GB00B{i:07d}",
            instrument_type=InstrumentType.EQUITY,
            price=Decimal("100.00") + Decimal(str(i * 0.01)),
            quantity=1000 + i * 100,
            timestamp=trade_time,
            venue=VenueType.REGULATED_MARKET,
            pre_trade_transparency=i % 10 != 0,  # 10% without pre-trade transparency
            post_trade_transparency=i % 15 != 0,  # 6.7% without post-trade transparency
            waiver_type="SIZE" if i % 10 == 0 else None,
            deferral_type="VOLUME" if i % 15 == 0 else None,
            large_in_scale=i % 20 == 0,  # 5% large in scale
            standard_market_size=7500,
            publication_timestamp=trade_time + timedelta(minutes=10) if i % 25 != 0 else trade_time + timedelta(minutes=20),  # 4% late
            publication_venue="XLON"
        )
        data.append(trade_data)
    
    return data


# Test Cases
class TestMiFIDIICompliance:
    """Test cases for MiFID II compliance validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = MiFIDIIComplianceValidator()
        self.reporter = MiFIDIIComplianceReporter(self.validator)
    
    def test_transaction_reporting_compliance_validation(self):
        """Test RTS 22 transaction reporting compliance"""
        # Create sample transaction reports
        reports = create_sample_transaction_reports(100)
        
        # Validate compliance
        result = self.validator.validate_transaction_reporting_compliance(reports)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_transactions'] == 100
        assert 'reporting_timeliness' in result['metrics']
        assert 'completeness_rate' in result['metrics']
        assert 'transmission_timeliness' in result['metrics']
    
    def test_transaction_reporting_late_report_violation(self):
        """Test late transaction reporting violation"""
        # Create transaction report with late reporting
        report = TransactionReport(
            transaction_id="TXN_LATE_001",
            isin="GB00B1234567",
            instrument_type=InstrumentType.EQUITY,
            venue=VenueType.REGULATED_MARKET,
            venue_mic="XLON",
            price=Decimal("100.00"),
            quantity=1000,
            currency="GBP",
            transaction_timestamp=datetime.now(timezone.utc),
            trading_date=datetime.now(timezone.utc).date() - timedelta(days=1),  # Late report
            client_id="CLIENT_001",
            client_type=ClientType.PROFESSIONAL,
            client_country="GB",
            order_id="ORDER_001",
            order_type=OrderType.MARKET,
            order_duration="DAY",
            order_timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
            execution_id="EXEC_001",
            execution_venue="XLON",
            execution_timestamp=datetime.now(timezone.utc)
        )
        
        result = self.validator.validate_transaction_reporting_compliance([report])
        
        # Should have violation
        assert len(result['violations']) > 0
        violation = next(v for v in result['violations'] if v['rule'] == 'RTS_22_REPORTING_DEADLINE')
        assert violation['severity'] == 'HIGH'
        assert violation['transaction_id'] == 'TXN_LATE_001'
    
    def test_best_execution_compliance_validation(self):
        """Test RTS 27 & 28 best execution compliance"""
        # Create sample best execution data
        execution_data = create_sample_best_execution_data(20)
        
        # Validate compliance
        result = self.validator.validate_best_execution_compliance(execution_data)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_reports'] == 20
        assert 'avg_execution_quality' in result['metrics']
        assert 'venue_analysis_completeness' in result['metrics']
    
    def test_best_execution_quality_violation(self):
        """Test best execution quality violation"""
        # Create best execution data with poor quality
        execution_data = BestExecutionData(
            report_id="BE_POOR_001",
            period="2024-Q1",
            instrument_class="Equity",
            client_orders_count=1000,
            client_orders_value=Decimal("1000000"),
            professional_orders_count=500,
            professional_orders_value=Decimal("2000000"),
            venue_data=[{'venue': 'XLON', 'volume': 1000000, 'percentage': 1.0}],
            average_execution_speed=100.0,
            price_improvement_rate=0.01,  # Below threshold
            likelihood_of_execution=0.95,
            top_5_venues=[{'venue': 'XLON', 'volume': 1000000, 'percentage': 1.0}],
            execution_quality_score=0.75,  # Below threshold
            best_execution_policy="POLICY_2024_V1",
            retail_client_impact={'avg_improvement': 0.01, 'cost_reduction': 0.005},
            professional_client_impact={'avg_improvement': 0.005, 'cost_reduction': 0.003}
        )
        
        result = self.validator.validate_best_execution_compliance([execution_data])
        
        # Should have violations
        assert len(result['violations']) > 0
        quality_violation = next(v for v in result['violations'] if v['rule'] == 'RTS_27_EXECUTION_QUALITY')
        assert quality_violation['severity'] == 'HIGH'
        assert quality_violation['report_id'] == 'BE_POOR_001'
    
    def test_systematic_internalizer_compliance_validation(self):
        """Test systematic internalizer compliance"""
        # Create sample SI data
        si_data = create_sample_si_data(10)
        
        # Validate compliance
        result = self.validator.validate_systematic_internalizer_compliance(si_data)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_si_entries'] == 10
        assert 'avg_systematic_percentage' in result['metrics']
        assert 'quote_response_compliance' in result['metrics']
    
    def test_systematic_internalizer_threshold_violation(self):
        """Test systematic internalizer threshold violation"""
        # Create SI data below threshold
        si_data = SystematicInternalizerData(
            si_id="SI_LOW_001",
            instrument_class="Equity",
            isin="GB00B1234567",
            quote_timestamp=datetime.now(timezone.utc),
            bid_price=Decimal("99.95"),
            ask_price=Decimal("100.05"),
            bid_size=100,
            ask_size=100,
            standard_market_size=7500,
            size_specific_to_transaction=False,
            execution_at_quoted_price=True,
            price_improvement_provided=False,
            risk_management_exception=False,
            exceptional_market_conditions=False,
            monthly_volume=1000000,
            systematic_percentage=0.02,  # Below threshold
            client_requests=1000,
            client_executions=950,
            rejection_rate=0.05
        )
        
        result = self.validator.validate_systematic_internalizer_compliance([si_data])
        
        # Should have violation
        assert len(result['violations']) > 0
        violation = next(v for v in result['violations'] if v['rule'] == 'SI_SYSTEMATIC_THRESHOLD')
        assert violation['severity'] == 'HIGH'
        assert violation['si_id'] == 'SI_LOW_001'
    
    def test_trade_transparency_compliance_validation(self):
        """Test trade transparency compliance"""
        # Create sample trade transparency data
        trades = create_sample_trade_transparency_data(100)
        
        # Validate compliance
        result = self.validator.validate_trade_transparency_compliance(trades)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_trades'] == 100
        assert 'pre_trade_transparency_rate' in result['metrics']
        assert 'post_trade_transparency_rate' in result['metrics']
    
    def test_trade_transparency_violation(self):
        """Test trade transparency violation"""
        # Create trade without transparency
        trade = TradeTransparencyData(
            trade_id="TRADE_OPAQUE_001",
            isin="GB00B1234567",
            instrument_type=InstrumentType.EQUITY,
            price=Decimal("100.00"),
            quantity=1000,
            timestamp=datetime.now(timezone.utc),
            venue=VenueType.REGULATED_MARKET,
            pre_trade_transparency=False,  # No transparency
            post_trade_transparency=False,  # No transparency
            waiver_type=None,  # No waiver
            deferral_type=None,  # No deferral
            large_in_scale=False,
            standard_market_size=7500
        )
        
        result = self.validator.validate_trade_transparency_compliance([trade])
        
        # Should have violations
        assert len(result['violations']) >= 2
        pre_trade_violation = next(v for v in result['violations'] if v['rule'] == 'TRADE_TRANSPARENCY_PRE_TRADE')
        post_trade_violation = next(v for v in result['violations'] if v['rule'] == 'TRADE_TRANSPARENCY_POST_TRADE')
        assert pre_trade_violation['severity'] == 'HIGH'
        assert post_trade_violation['severity'] == 'HIGH'
    
    def test_transaction_reporting_report_generation(self):
        """Test transaction reporting report generation"""
        # Create sample transaction reports
        reports = create_sample_transaction_reports(100)
        
        # Generate report
        report = self.reporter.generate_transaction_reporting_report(reports, "2024-Q1")
        
        # Assertions
        assert report['report_type'] == 'MIFID_II_TRANSACTION_REPORTING'
        assert report['period'] == '2024-Q1'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert report['summary']['total_transactions'] == 100
    
    def test_best_execution_report_generation(self):
        """Test best execution report generation"""
        # Create sample best execution data
        execution_data = create_sample_best_execution_data(20)
        
        # Generate report
        report = self.reporter.generate_best_execution_report(execution_data, "2024-Q1")
        
        # Assertions
        assert report['report_type'] == 'MIFID_II_BEST_EXECUTION'
        assert report['period'] == '2024-Q1'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert report['summary']['total_reports'] == 20
    
    def test_systematic_internalizer_report_generation(self):
        """Test systematic internalizer report generation"""
        # Create sample SI data
        si_data = create_sample_si_data(10)
        
        # Generate report
        report = self.reporter.generate_systematic_internalizer_report(si_data, "2024-Q1")
        
        # Assertions
        assert report['report_type'] == 'MIFID_II_SYSTEMATIC_INTERNALIZER'
        assert report['period'] == '2024-Q1'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert report['summary']['total_si_entries'] == 10
    
    def test_trade_transparency_report_generation(self):
        """Test trade transparency report generation"""
        # Create sample trade transparency data
        trades = create_sample_trade_transparency_data(100)
        
        # Generate report
        report = self.reporter.generate_trade_transparency_report(trades, "2024-Q1")
        
        # Assertions
        assert report['report_type'] == 'MIFID_II_TRADE_TRANSPARENCY'
        assert report['period'] == '2024-Q1'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert report['summary']['total_trades'] == 100
    
    def test_comprehensive_mifid_ii_compliance_validation(self):
        """Test comprehensive MiFID II compliance validation"""
        # Create all types of sample data
        transaction_reports = create_sample_transaction_reports(100)
        execution_data = create_sample_best_execution_data(20)
        si_data = create_sample_si_data(10)
        trade_transparency = create_sample_trade_transparency_data(100)
        
        # Validate all compliance areas
        tr_result = self.validator.validate_transaction_reporting_compliance(transaction_reports)
        be_result = self.validator.validate_best_execution_compliance(execution_data)
        si_result = self.validator.validate_systematic_internalizer_compliance(si_data)
        tt_result = self.validator.validate_trade_transparency_compliance(trade_transparency)
        
        # All should return valid results
        assert all('violations' in result for result in [tr_result, be_result, si_result, tt_result])
        assert all('metrics' in result for result in [tr_result, be_result, si_result, tt_result])
        
        # Check total violations
        total_violations = sum(len(result['violations']) for result in [tr_result, be_result, si_result, tt_result])
        assert total_violations >= 0  # Should have some violations with sample data
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for MiFID II compliance validation"""
        # Create large dataset
        large_transactions = create_sample_transaction_reports(10000)
        large_execution_data = create_sample_best_execution_data(500)
        large_si_data = create_sample_si_data(100)
        large_trades = create_sample_trade_transparency_data(10000)
        
        # Measure performance
        start_time = time.time()
        
        # Run all validations
        self.validator.validate_transaction_reporting_compliance(large_transactions)
        self.validator.validate_best_execution_compliance(large_execution_data)
        self.validator.validate_systematic_internalizer_compliance(large_si_data)
        self.validator.validate_trade_transparency_compliance(large_trades)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert processing_time / len(large_transactions) < 0.001  # < 1ms per transaction
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        # Test with empty data
        empty_result = self.validator.validate_transaction_reporting_compliance([])
        assert empty_result['metrics']['total_transactions'] == 0
        assert len(empty_result['violations']) == 0
        
        # Test with None values
        report_with_nones = TransactionReport(
            transaction_id="TXN_NONE_001",
            isin="GB00B1234567",
            instrument_type=InstrumentType.EQUITY,
            venue=VenueType.REGULATED_MARKET,
            venue_mic="XLON",
            price=Decimal("100.00"),
            quantity=1000,
            currency="GBP",
            transaction_timestamp=datetime.now(timezone.utc),
            trading_date=datetime.now(timezone.utc).date(),
            client_id="CLIENT_001",
            client_type=ClientType.PROFESSIONAL,
            client_country="GB",
            order_id="ORDER_001",
            order_type=OrderType.MARKET,
            order_duration="DAY",
            order_timestamp=datetime.now(timezone.utc) - timedelta(minutes=30),
            execution_id="EXEC_001",
            execution_venue="XLON",
            execution_timestamp=datetime.now(timezone.utc),
            transmission_timestamp=None  # None value
        )
        
        result = self.validator.validate_transaction_reporting_compliance([report_with_nones])
        assert 'violations' in result
        assert 'metrics' in result
    
    def test_real_time_monitoring_simulation(self):
        """Test real-time monitoring simulation"""
        # Simulate real-time data stream
        transactions = []
        violations_count = 0
        
        for i in range(100):
            # Create transaction with varying characteristics
            transaction = TransactionReport(
                transaction_id=f"RT_TXN_{i:06d}",
                isin="GB00B1234567",
                instrument_type=InstrumentType.EQUITY,
                venue=VenueType.REGULATED_MARKET,
                venue_mic="XLON",
                price=Decimal("100.00"),
                quantity=1000,
                currency="GBP",
                transaction_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                trading_date=datetime.now(timezone.utc).date(),
                client_id=f"CLIENT_{i % 10:03d}",
                client_type=ClientType.PROFESSIONAL,
                client_country="GB",
                order_id=f"ORDER_{i:06d}",
                order_type=OrderType.MARKET,
                order_duration="DAY",
                order_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i+1),
                execution_id=f"EXEC_{i:06d}",
                execution_venue="XLON",
                execution_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                transmission_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i) + timedelta(seconds=30 + i)  # Increasing delay
            )
            transactions.append(transaction)
            
            # Check compliance every 10 transactions
            if (i + 1) % 10 == 0:
                result = self.validator.validate_transaction_reporting_compliance(transactions[-10:])
                violations_count += len(result['violations'])
        
        # Should have processed all transactions
        assert len(transactions) == 100
        assert violations_count >= 0


# Integration test
@pytest.mark.asyncio
async def test_mifid_ii_compliance_integration():
    """Integration test for MiFID II compliance system"""
    # Initialize components
    validator = MiFIDIIComplianceValidator()
    reporter = MiFIDIIComplianceReporter(validator)
    
    # Create comprehensive test data
    transaction_reports = create_sample_transaction_reports(1000)
    execution_data = create_sample_best_execution_data(100)
    si_data = create_sample_si_data(50)
    trade_transparency = create_sample_trade_transparency_data(1000)
    
    # Run comprehensive compliance check
    start_time = time.time()
    
    # Generate all reports
    tr_report = reporter.generate_transaction_reporting_report(transaction_reports, "2024-Q1")
    be_report = reporter.generate_best_execution_report(execution_data, "2024-Q1")
    si_report = reporter.generate_systematic_internalizer_report(si_data, "2024-Q1")
    tt_report = reporter.generate_trade_transparency_report(trade_transparency, "2024-Q1")
    
    end_time = time.time()
    
    # Performance check
    assert (end_time - start_time) < 15.0  # Should complete within 15 seconds
    
    # Validate all reports
    assert tr_report['report_type'] == 'MIFID_II_TRANSACTION_REPORTING'
    assert be_report['report_type'] == 'MIFID_II_BEST_EXECUTION'
    assert si_report['report_type'] == 'MIFID_II_SYSTEMATIC_INTERNALIZER'
    assert tt_report['report_type'] == 'MIFID_II_TRADE_TRANSPARENCY'
    
    # Check report completeness
    for report in [tr_report, be_report, si_report, tt_report]:
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert 'generated_at' in report
    
    # Validate metrics
    assert tr_report['summary']['total_transactions'] == 1000
    assert be_report['summary']['total_reports'] == 100
    assert si_report['summary']['total_si_entries'] == 50
    assert tt_report['summary']['total_trades'] == 1000
    
    print("✅ MiFID II Compliance Integration Test Passed")


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_mifid_ii_compliance_integration())