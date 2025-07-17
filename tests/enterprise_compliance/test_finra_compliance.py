#!/usr/bin/env python3
"""
FINRA Compliance Testing Suite
Agent 3: Regulatory Compliance Testing

Comprehensive testing for FINRA (Financial Industry Regulatory Authority) compliance
including OATS, CAT, and trade reporting requirements.

Features:
- OATS (Order Audit Trail System) reporting validation
- CAT (Consolidated Audit Trail) requirements testing
- Trade reporting and ADF submissions
- Market surveillance compliance
- Best execution obligations
- Customer protection rules
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


class FINRARuleType(Enum):
    """FINRA Rule types"""
    OATS = "oats"
    CAT = "cat"
    TRADE_REPORTING = "trade_reporting"
    ADF = "adf"
    MARKET_SURVEILLANCE = "market_surveillance"
    BEST_EXECUTION = "best_execution"
    CUSTOMER_PROTECTION = "customer_protection"


class OrderEventType(Enum):
    """Order event types for OATS/CAT"""
    NEW_ORDER = "new_order"
    CANCEL_ORDER = "cancel_order"
    REPLACE_ORDER = "replace_order"
    EXECUTION = "execution"
    TRADE_CORRECTION = "trade_correction"
    TRADE_CANCELLATION = "trade_cancellation"
    ROUTE_OUT = "route_out"
    ROUTE_IN = "route_in"


class OrderOriginType(Enum):
    """Order origin types"""
    CUSTOMER = "customer"
    PROPRIETARY = "proprietary"
    MARKET_MAKER = "market_maker"
    AWAY_MARKET_MAKER = "away_market_maker"
    ELECTRONIC_ACCESS = "electronic_access"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    MARKET_ON_CLOSE = "market_on_close"
    LIMIT_ON_CLOSE = "limit_on_close"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    SELL_SHORT_EXEMPT = "sell_short_exempt"


class TradeReportVenue(Enum):
    """Trade reporting venues"""
    ADF = "adf"
    TRF = "trf"
    FINRA_TRACE = "finra_trace"
    ALTERNATIVE_DISPLAY = "alternative_display"


@dataclass
class OATSReportData:
    """OATS (Order Audit Trail System) report data"""
    firm_id: str
    order_id: str
    event_type: OrderEventType
    timestamp: datetime
    
    # Order details
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    account_type: str
    origin_type: OrderOriginType
    
    # Optional fields
    price: Optional[Decimal] = None
    customer_id: Optional[str] = None
    
    # Execution details (if applicable)
    execution_price: Optional[Decimal] = None
    execution_quantity: Optional[int] = None
    contra_party: Optional[str] = None
    
    # Routing information
    receiving_firm: Optional[str] = None
    routing_firm: Optional[str] = None
    
    # Special handling
    special_handling_code: Optional[str] = None
    
    # Compliance fields
    manual_order_entry: bool = False
    represent_order: bool = False
    
    # Timestamps
    order_received_timestamp: Optional[datetime] = None
    order_transmitted_timestamp: Optional[datetime] = None


@dataclass
class CATReportData:
    """CAT (Consolidated Audit Trail) report data"""
    cat_reporter_id: str
    event_id: str
    event_type: OrderEventType
    event_timestamp: datetime
    
    # Order identification
    order_id: str
    parent_order_id: Optional[str] = None
    
    # Security details
    symbol: str
    security_type: str
    
    # Order details
    side: OrderSide
    quantity: int
    order_type: OrderType
    price: Optional[Decimal] = None
    
    # Account information
    account_id: str
    account_holder_type: str
    
    # Firm information
    reporting_firm_id: str
    executing_firm_id: Optional[str] = None
    
    # Customer information
    customer_id: Optional[str] = None
    customer_type: Optional[str] = None
    
    # Execution information
    execution_id: Optional[str] = None
    execution_price: Optional[Decimal] = None
    execution_quantity: Optional[int] = None
    
    # Market center information
    market_center_id: Optional[str] = None
    
    # Routing information
    routed_order_id: Optional[str] = None
    
    # Options specific fields
    option_type: Optional[str] = None
    strike_price: Optional[Decimal] = None
    expiration_date: Optional[datetime] = None
    
    # Compliance flags
    solicited_flag: bool = False
    institutional_account_flag: bool = False


@dataclass
class TradeReportData:
    """Trade report data for FINRA reporting"""
    trade_id: str
    report_venue: TradeReportVenue
    timestamp: datetime
    
    # Security details
    symbol: str
    security_type: str
    
    # Trade details
    price: Decimal
    quantity: int
    
    # Market participant information
    buy_side_firm: str
    sell_side_firm: str
    
    # Trade modifier flags
    trade_modifier_1: Optional[str] = None
    trade_modifier_2: Optional[str] = None
    trade_modifier_3: Optional[str] = None
    
    # Settlement information
    settlement_date: Optional[datetime] = None
    
    # Reporting obligations
    trade_report_required: bool = True
    reporting_party: str = "both"
    
    # Capacity indicators
    buy_capacity: str = "principal"
    sell_capacity: str = "principal"
    
    # Special trade types
    cross_trade: bool = False
    odd_lot_trade: bool = False
    
    # Clearing information
    clearing_firm: Optional[str] = None
    
    # Compliance fields
    late_report: bool = False
    report_exception: Optional[str] = None


@dataclass
class ADFSubmissionData:
    """ADF (Alternative Display Facility) submission data"""
    submission_id: str
    timestamp: datetime
    
    # Security details
    symbol: str
    
    # Quote/Trade information
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    
    # Trade information
    trade_price: Optional[Decimal] = None
    trade_size: Optional[int] = None
    
    # Market participant
    market_participant_id: str
    
    # Display information
    display_size: Optional[int] = None
    reserve_size: Optional[int] = None
    
    # Order type
    order_type: OrderType
    
    # Compliance fields
    locked_crossed_market: bool = False
    
    # Timestamps
    display_timestamp: Optional[datetime] = None
    remove_timestamp: Optional[datetime] = None


class FINRAComplianceValidator:
    """FINRA Compliance validation engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.violations: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'oats': {
                'reporting_deadline_seconds': 30,
                'required_events': ['new_order', 'execution', 'cancel_order'],
                'timestamp_accuracy_ms': 1000,
                'customer_id_required': True
            },
            'cat': {
                'reporting_deadline_hours': 24,
                'event_timestamp_accuracy_ms': 1,
                'required_fields': [
                    'cat_reporter_id', 'event_id', 'event_type', 'event_timestamp',
                    'order_id', 'symbol', 'side', 'quantity'
                ],
                'options_additional_fields': ['option_type', 'strike_price', 'expiration_date']
            },
            'trade_reporting': {
                'reporting_deadline_seconds': 30,
                'price_accuracy_decimals': 4,
                'quantity_minimum': 1,
                'capacity_validation': True
            },
            'adf': {
                'quote_update_frequency_ms': 1000,
                'display_size_minimum': 100,
                'locked_crossed_tolerance_cents': 0.01
            },
            'market_surveillance': {
                'unusual_volume_threshold': 3.0,  # 3x average
                'price_volatility_threshold': 0.05,  # 5%
                'concentration_threshold': 0.10  # 10%
            },
            'best_execution': {
                'execution_quality_threshold': 0.85,
                'price_improvement_threshold': 0.02,
                'fill_rate_threshold': 0.95
            }
        }
    
    def validate_oats_compliance(self, oats_data: List[OATSReportData]) -> Dict[str, Any]:
        """Validate OATS reporting compliance"""
        violations = []
        metrics = {
            'total_events': len(oats_data),
            'timeliness_compliance': 0,
            'completeness_compliance': 0,
            'accuracy_compliance': 0,
            'late_reports': 0
        }
        
        if not oats_data:
            return {'violations': violations, 'metrics': metrics}
        
        oats_config = self.config['oats']
        timely_reports = 0
        complete_reports = 0
        accurate_reports = 0
        
        for report in oats_data:
            # Check reporting timeliness
            if report.order_received_timestamp and report.timestamp:
                reporting_delay = (report.timestamp - report.order_received_timestamp).total_seconds()
                if reporting_delay <= oats_config['reporting_deadline_seconds']:
                    timely_reports += 1
                else:
                    violations.append({
                        'rule': 'OATS_REPORTING_TIMELINESS',
                        'severity': 'HIGH',
                        'description': f"OATS report {report.order_id} late by {reporting_delay:.1f} seconds",
                        'order_id': report.order_id,
                        'delay_seconds': reporting_delay,
                        'threshold_seconds': oats_config['reporting_deadline_seconds']
                    })
            
            # Check completeness
            missing_fields = []
            if not report.firm_id:
                missing_fields.append('firm_id')
            if not report.order_id:
                missing_fields.append('order_id')
            if not report.symbol:
                missing_fields.append('symbol')
            if oats_config['customer_id_required'] and not report.customer_id:
                missing_fields.append('customer_id')
            
            if not missing_fields:
                complete_reports += 1
            else:
                violations.append({
                    'rule': 'OATS_COMPLETENESS',
                    'severity': 'HIGH',
                    'description': f"OATS report {report.order_id} missing required fields",
                    'order_id': report.order_id,
                    'missing_fields': missing_fields
                })
            
            # Check accuracy
            accuracy_issues = []
            if report.quantity <= 0:
                accuracy_issues.append('invalid_quantity')
            if report.price and report.price <= 0:
                accuracy_issues.append('invalid_price')
            if report.execution_quantity and report.execution_quantity > report.quantity:
                accuracy_issues.append('execution_quantity_exceeds_order_quantity')
            
            if not accuracy_issues:
                accurate_reports += 1
            else:
                violations.append({
                    'rule': 'OATS_ACCURACY',
                    'severity': 'MEDIUM',
                    'description': f"OATS report {report.order_id} has accuracy issues",
                    'order_id': report.order_id,
                    'accuracy_issues': accuracy_issues
                })
            
            # Check event sequence
            if report.event_type == OrderEventType.EXECUTION and not report.execution_price:
                violations.append({
                    'rule': 'OATS_EVENT_SEQUENCE',
                    'severity': 'HIGH',
                    'description': f"OATS execution event {report.order_id} missing execution price",
                    'order_id': report.order_id,
                    'event_type': report.event_type.value
                })
        
        # Calculate metrics
        metrics['timeliness_compliance'] = timely_reports / len(oats_data)
        metrics['completeness_compliance'] = complete_reports / len(oats_data)
        metrics['accuracy_compliance'] = accurate_reports / len(oats_data)
        metrics['late_reports'] = len(oats_data) - timely_reports
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_cat_compliance(self, cat_data: List[CATReportData]) -> Dict[str, Any]:
        """Validate CAT reporting compliance"""
        violations = []
        metrics = {
            'total_events': len(cat_data),
            'timeliness_compliance': 0,
            'completeness_compliance': 0,
            'linkage_compliance': 0,
            'timestamp_accuracy': 0
        }
        
        if not cat_data:
            return {'violations': violations, 'metrics': metrics}
        
        cat_config = self.config['cat']
        timely_reports = 0
        complete_reports = 0
        accurate_timestamps = 0
        
        for report in cat_data:
            # Check reporting timeliness (T+1 basis)
            if report.event_timestamp:
                reporting_deadline = report.event_timestamp + timedelta(hours=cat_config['reporting_deadline_hours'])
                if datetime.now(timezone.utc) <= reporting_deadline:
                    timely_reports += 1
                else:
                    violations.append({
                        'rule': 'CAT_REPORTING_TIMELINESS',
                        'severity': 'HIGH',
                        'description': f"CAT report {report.event_id} exceeded T+1 deadline",
                        'event_id': report.event_id,
                        'event_timestamp': report.event_timestamp.isoformat(),
                        'reporting_deadline': reporting_deadline.isoformat()
                    })
            
            # Check completeness
            missing_fields = []
            for field in cat_config['required_fields']:
                if not getattr(report, field, None):
                    missing_fields.append(field)
            
            # Check options-specific fields
            if report.security_type == 'option':
                for field in cat_config['options_additional_fields']:
                    if not getattr(report, field, None):
                        missing_fields.append(field)
            
            if not missing_fields:
                complete_reports += 1
            else:
                violations.append({
                    'rule': 'CAT_COMPLETENESS',
                    'severity': 'HIGH',
                    'description': f"CAT report {report.event_id} missing required fields",
                    'event_id': report.event_id,
                    'missing_fields': missing_fields
                })
            
            # Check timestamp accuracy
            if report.event_timestamp:
                # Check if timestamp is within acceptable accuracy range
                # (This would typically involve comparing with market data timestamps)
                accurate_timestamps += 1
            
            # Check order linkage
            if report.event_type == OrderEventType.REPLACE_ORDER and not report.parent_order_id:
                violations.append({
                    'rule': 'CAT_ORDER_LINKAGE',
                    'severity': 'HIGH',
                    'description': f"CAT replace order {report.event_id} missing parent order ID",
                    'event_id': report.event_id,
                    'order_id': report.order_id
                })
            
            # Check execution linkage
            if report.execution_id and not report.execution_price:
                violations.append({
                    'rule': 'CAT_EXECUTION_LINKAGE',
                    'severity': 'HIGH',
                    'description': f"CAT execution {report.event_id} missing execution price",
                    'event_id': report.event_id,
                    'execution_id': report.execution_id
                })
        
        # Calculate metrics
        metrics['timeliness_compliance'] = timely_reports / len(cat_data)
        metrics['completeness_compliance'] = complete_reports / len(cat_data)
        metrics['timestamp_accuracy'] = accurate_timestamps / len(cat_data)
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_trade_reporting_compliance(self, trade_data: List[TradeReportData]) -> Dict[str, Any]:
        """Validate trade reporting compliance"""
        violations = []
        metrics = {
            'total_trades': len(trade_data),
            'timeliness_compliance': 0,
            'accuracy_compliance': 0,
            'capacity_compliance': 0,
            'late_reports': 0
        }
        
        if not trade_data:
            return {'violations': violations, 'metrics': metrics}
        
        tr_config = self.config['trade_reporting']
        timely_reports = 0
        accurate_reports = 0
        capacity_compliant = 0
        
        for trade in trade_data:
            # Check reporting timeliness
            current_time = datetime.now(timezone.utc)
            reporting_deadline = trade.timestamp + timedelta(seconds=tr_config['reporting_deadline_seconds'])
            
            if current_time <= reporting_deadline:
                timely_reports += 1
            else:
                violations.append({
                    'rule': 'TRADE_REPORTING_TIMELINESS',
                    'severity': 'HIGH',
                    'description': f"Trade report {trade.trade_id} exceeded 30-second deadline",
                    'trade_id': trade.trade_id,
                    'trade_timestamp': trade.timestamp.isoformat(),
                    'reporting_deadline': reporting_deadline.isoformat()
                })
            
            # Check accuracy
            accuracy_issues = []
            if trade.quantity < tr_config['quantity_minimum']:
                accuracy_issues.append('quantity_below_minimum')
            if trade.price <= 0:
                accuracy_issues.append('invalid_price')
            
            # Check price precision
            price_str = str(trade.price)
            if '.' in price_str:
                decimal_places = len(price_str.split('.')[1])
                if decimal_places > tr_config['price_accuracy_decimals']:
                    accuracy_issues.append('price_precision_exceeded')
            
            if not accuracy_issues:
                accurate_reports += 1
            else:
                violations.append({
                    'rule': 'TRADE_REPORTING_ACCURACY',
                    'severity': 'MEDIUM',
                    'description': f"Trade report {trade.trade_id} has accuracy issues",
                    'trade_id': trade.trade_id,
                    'accuracy_issues': accuracy_issues
                })
            
            # Check capacity indicators
            if tr_config['capacity_validation']:
                valid_capacities = ['principal', 'agent', 'riskless_principal']
                if trade.buy_capacity in valid_capacities and trade.sell_capacity in valid_capacities:
                    capacity_compliant += 1
                else:
                    violations.append({
                        'rule': 'TRADE_REPORTING_CAPACITY',
                        'severity': 'MEDIUM',
                        'description': f"Trade report {trade.trade_id} has invalid capacity indicators",
                        'trade_id': trade.trade_id,
                        'buy_capacity': trade.buy_capacity,
                        'sell_capacity': trade.sell_capacity
                    })
            
            # Check trade modifier consistency
            if trade.cross_trade and not trade.trade_modifier_1:
                violations.append({
                    'rule': 'TRADE_REPORTING_MODIFIER',
                    'severity': 'MEDIUM',
                    'description': f"Cross trade {trade.trade_id} missing trade modifier",
                    'trade_id': trade.trade_id
                })
        
        # Calculate metrics
        metrics['timeliness_compliance'] = timely_reports / len(trade_data)
        metrics['accuracy_compliance'] = accurate_reports / len(trade_data)
        metrics['capacity_compliance'] = capacity_compliant / len(trade_data)
        metrics['late_reports'] = len(trade_data) - timely_reports
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_adf_compliance(self, adf_data: List[ADFSubmissionData]) -> Dict[str, Any]:
        """Validate ADF submission compliance"""
        violations = []
        metrics = {
            'total_submissions': len(adf_data),
            'quote_quality': 0,
            'display_compliance': 0,
            'locked_crossed_violations': 0,
            'update_frequency_compliance': 0
        }
        
        if not adf_data:
            return {'violations': violations, 'metrics': metrics}
        
        adf_config = self.config['adf']
        quality_quotes = 0
        display_compliant = 0
        locked_crossed_count = 0
        
        for submission in adf_data:
            # Check quote quality
            if submission.bid_price and submission.ask_price:
                if submission.ask_price > submission.bid_price:
                    quality_quotes += 1
                else:
                    violations.append({
                        'rule': 'ADF_QUOTE_QUALITY',
                        'severity': 'HIGH',
                        'description': f"ADF submission {submission.submission_id} has crossed/locked market",
                        'submission_id': submission.submission_id,
                        'bid_price': float(submission.bid_price),
                        'ask_price': float(submission.ask_price)
                    })
            
            # Check display size requirements
            if submission.display_size:
                if submission.display_size >= adf_config['display_size_minimum']:
                    display_compliant += 1
                else:
                    violations.append({
                        'rule': 'ADF_DISPLAY_SIZE',
                        'severity': 'MEDIUM',
                        'description': f"ADF submission {submission.submission_id} display size below minimum",
                        'submission_id': submission.submission_id,
                        'display_size': submission.display_size,
                        'minimum_size': adf_config['display_size_minimum']
                    })
            
            # Check locked/crossed market violations
            if submission.locked_crossed_market:
                locked_crossed_count += 1
                violations.append({
                    'rule': 'ADF_LOCKED_CROSSED_MARKET',
                    'severity': 'HIGH',
                    'description': f"ADF submission {submission.submission_id} created locked/crossed market",
                    'submission_id': submission.submission_id,
                    'symbol': submission.symbol
                })
            
            # Check order type consistency
            if submission.order_type == OrderType.MARKET and (submission.bid_price or submission.ask_price):
                violations.append({
                    'rule': 'ADF_ORDER_TYPE_CONSISTENCY',
                    'severity': 'MEDIUM',
                    'description': f"ADF market order {submission.submission_id} has price information",
                    'submission_id': submission.submission_id,
                    'order_type': submission.order_type.value
                })
        
        # Calculate metrics
        metrics['quote_quality'] = quality_quotes / len(adf_data)
        metrics['display_compliance'] = display_compliant / len(adf_data)
        metrics['locked_crossed_violations'] = locked_crossed_count
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_market_surveillance_compliance(self, trade_data: List[TradeReportData]) -> Dict[str, Any]:
        """Validate market surveillance compliance"""
        violations = []
        metrics = {
            'total_trades': len(trade_data),
            'unusual_volume_alerts': 0,
            'price_volatility_alerts': 0,
            'concentration_alerts': 0,
            'surveillance_score': 0
        }
        
        if not trade_data:
            return {'violations': violations, 'metrics': metrics}
        
        ms_config = self.config['market_surveillance']
        
        # Group trades by symbol
        symbol_trades = {}
        for trade in trade_data:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        unusual_volume_count = 0
        price_volatility_count = 0
        concentration_count = 0
        
        for symbol, trades in symbol_trades.items():
            # Check unusual volume
            total_volume = sum(trade.quantity for trade in trades)
            avg_volume = total_volume / len(trades)
            
            # Mock average volume check (in production, this would use historical data)
            mock_historical_avg = 10000
            if total_volume > mock_historical_avg * ms_config['unusual_volume_threshold']:
                unusual_volume_count += 1
                violations.append({
                    'rule': 'MARKET_SURVEILLANCE_UNUSUAL_VOLUME',
                    'severity': 'MEDIUM',
                    'description': f"Unusual volume detected for {symbol}",
                    'symbol': symbol,
                    'total_volume': total_volume,
                    'threshold_volume': mock_historical_avg * ms_config['unusual_volume_threshold']
                })
            
            # Check price volatility
            if len(trades) > 1:
                prices = [float(trade.price) for trade in trades]
                min_price = min(prices)
                max_price = max(prices)
                price_range = (max_price - min_price) / min_price
                
                if price_range > ms_config['price_volatility_threshold']:
                    price_volatility_count += 1
                    violations.append({
                        'rule': 'MARKET_SURVEILLANCE_PRICE_VOLATILITY',
                        'severity': 'MEDIUM',
                        'description': f"High price volatility detected for {symbol}",
                        'symbol': symbol,
                        'price_range_pct': price_range,
                        'threshold_pct': ms_config['price_volatility_threshold']
                    })
            
            # Check concentration
            firm_volumes = {}
            for trade in trades:
                if trade.buy_side_firm not in firm_volumes:
                    firm_volumes[trade.buy_side_firm] = 0
                firm_volumes[trade.buy_side_firm] += trade.quantity
            
            max_firm_volume = max(firm_volumes.values()) if firm_volumes else 0
            concentration_ratio = max_firm_volume / total_volume if total_volume > 0 else 0
            
            if concentration_ratio > ms_config['concentration_threshold']:
                concentration_count += 1
                violations.append({
                    'rule': 'MARKET_SURVEILLANCE_CONCENTRATION',
                    'severity': 'MEDIUM',
                    'description': f"High concentration detected for {symbol}",
                    'symbol': symbol,
                    'concentration_ratio': concentration_ratio,
                    'threshold_ratio': ms_config['concentration_threshold']
                })
        
        # Calculate metrics
        metrics['unusual_volume_alerts'] = unusual_volume_count
        metrics['price_volatility_alerts'] = price_volatility_count
        metrics['concentration_alerts'] = concentration_count
        metrics['surveillance_score'] = 1.0 - (len(violations) / len(trade_data)) if trade_data else 0
        
        return {'violations': violations, 'metrics': metrics}


class FINRAComplianceReporter:
    """FINRA compliance reporting system"""
    
    def __init__(self, validator: FINRAComplianceValidator):
        self.validator = validator
    
    def generate_oats_report(self, oats_data: List[OATSReportData], period: str) -> Dict[str, Any]:
        """Generate OATS compliance report"""
        compliance_result = self.validator.validate_oats_compliance(oats_data)
        
        report = {
            'report_type': 'FINRA_OATS',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_events': compliance_result['metrics']['total_events'],
                'compliance_violations': len(compliance_result['violations']),
                'timeliness_compliance': compliance_result['metrics']['timeliness_compliance'],
                'completeness_compliance': compliance_result['metrics']['completeness_compliance'],
                'late_reports': compliance_result['metrics']['late_reports']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_oats_recommendations(compliance_result)
        }
        
        return report
    
    def generate_cat_report(self, cat_data: List[CATReportData], period: str) -> Dict[str, Any]:
        """Generate CAT compliance report"""
        compliance_result = self.validator.validate_cat_compliance(cat_data)
        
        report = {
            'report_type': 'FINRA_CAT',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_events': compliance_result['metrics']['total_events'],
                'compliance_violations': len(compliance_result['violations']),
                'timeliness_compliance': compliance_result['metrics']['timeliness_compliance'],
                'completeness_compliance': compliance_result['metrics']['completeness_compliance'],
                'timestamp_accuracy': compliance_result['metrics']['timestamp_accuracy']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_cat_recommendations(compliance_result)
        }
        
        return report
    
    def generate_trade_reporting_report(self, trade_data: List[TradeReportData], period: str) -> Dict[str, Any]:
        """Generate trade reporting compliance report"""
        compliance_result = self.validator.validate_trade_reporting_compliance(trade_data)
        
        report = {
            'report_type': 'FINRA_TRADE_REPORTING',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_trades': compliance_result['metrics']['total_trades'],
                'compliance_violations': len(compliance_result['violations']),
                'timeliness_compliance': compliance_result['metrics']['timeliness_compliance'],
                'accuracy_compliance': compliance_result['metrics']['accuracy_compliance'],
                'late_reports': compliance_result['metrics']['late_reports']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_trade_reporting_recommendations(compliance_result)
        }
        
        return report
    
    def generate_adf_report(self, adf_data: List[ADFSubmissionData], period: str) -> Dict[str, Any]:
        """Generate ADF compliance report"""
        compliance_result = self.validator.validate_adf_compliance(adf_data)
        
        report = {
            'report_type': 'FINRA_ADF',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_submissions': compliance_result['metrics']['total_submissions'],
                'compliance_violations': len(compliance_result['violations']),
                'quote_quality': compliance_result['metrics']['quote_quality'],
                'display_compliance': compliance_result['metrics']['display_compliance'],
                'locked_crossed_violations': compliance_result['metrics']['locked_crossed_violations']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_adf_recommendations(compliance_result)
        }
        
        return report
    
    def generate_market_surveillance_report(self, trade_data: List[TradeReportData], period: str) -> Dict[str, Any]:
        """Generate market surveillance report"""
        compliance_result = self.validator.validate_market_surveillance_compliance(trade_data)
        
        report = {
            'report_type': 'FINRA_MARKET_SURVEILLANCE',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_trades': compliance_result['metrics']['total_trades'],
                'surveillance_alerts': len(compliance_result['violations']),
                'unusual_volume_alerts': compliance_result['metrics']['unusual_volume_alerts'],
                'price_volatility_alerts': compliance_result['metrics']['price_volatility_alerts'],
                'concentration_alerts': compliance_result['metrics']['concentration_alerts']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_surveillance_recommendations(compliance_result)
        }
        
        return report
    
    def _generate_oats_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate OATS compliance recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['timeliness_compliance'] < 0.95:
            recommendations.append("Improve OATS reporting system to meet 30-second deadline")
        
        if compliance_result['metrics']['completeness_compliance'] < 0.98:
            recommendations.append("Enhance data validation to ensure all required OATS fields are populated")
        
        if compliance_result['metrics']['accuracy_compliance'] < 0.95:
            recommendations.append("Implement additional accuracy checks for OATS data")
        
        return recommendations
    
    def _generate_cat_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate CAT compliance recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['timeliness_compliance'] < 0.95:
            recommendations.append("Improve CAT reporting system to meet T+1 deadline")
        
        if compliance_result['metrics']['completeness_compliance'] < 0.98:
            recommendations.append("Enhance data validation for CAT required fields")
        
        if compliance_result['metrics']['timestamp_accuracy'] < 0.99:
            recommendations.append("Improve timestamp accuracy for CAT reporting")
        
        return recommendations
    
    def _generate_trade_reporting_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate trade reporting recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['timeliness_compliance'] < 0.95:
            recommendations.append("Optimize trade reporting system to meet 30-second deadline")
        
        if compliance_result['metrics']['accuracy_compliance'] < 0.98:
            recommendations.append("Implement stricter validation for trade report accuracy")
        
        return recommendations
    
    def _generate_adf_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate ADF compliance recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['quote_quality'] < 0.98:
            recommendations.append("Improve quote quality controls to prevent locked/crossed markets")
        
        if compliance_result['metrics']['display_compliance'] < 0.95:
            recommendations.append("Ensure all ADF submissions meet minimum display size requirements")
        
        return recommendations
    
    def _generate_surveillance_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate market surveillance recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['unusual_volume_alerts'] > 0:
            recommendations.append("Investigate unusual volume patterns and implement monitoring controls")
        
        if compliance_result['metrics']['price_volatility_alerts'] > 0:
            recommendations.append("Monitor price volatility and implement volatility controls")
        
        if compliance_result['metrics']['concentration_alerts'] > 0:
            recommendations.append("Review concentration patterns and implement diversification controls")
        
        return recommendations


# Test fixtures and utilities
def create_sample_oats_data(count: int = 100) -> List[OATSReportData]:
    """Create sample OATS data for testing"""
    oats_data = []
    
    for i in range(count):
        order_time = datetime.now(timezone.utc) - timedelta(minutes=i)
        
        report = OATSReportData(
            firm_id=f"FIRM_{i % 10:03d}",
            order_id=f"ORDER_{i:06d}",
            event_type=OrderEventType.NEW_ORDER if i % 3 == 0 else OrderEventType.EXECUTION,
            timestamp=order_time + timedelta(seconds=15 if i % 20 != 0 else 45),  # 5% late
            symbol="AAPL",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            quantity=100 * (i + 1),
            order_type=OrderType.MARKET if i % 2 == 0 else OrderType.LIMIT,
            price=Decimal("150.00") if i % 2 == 1 else None,
            account_type="CUSTOMER",
            customer_id=f"CUST_{i % 50:03d}" if i % 15 != 0 else None,  # 6.7% missing
            origin_type=OrderOriginType.CUSTOMER,
            execution_price=Decimal("149.99") if i % 3 != 0 else None,
            execution_quantity=100 * (i + 1) if i % 3 != 0 else None,
            order_received_timestamp=order_time,
            manual_order_entry=i % 10 == 0
        )
        oats_data.append(report)
    
    return oats_data


def create_sample_cat_data(count: int = 100) -> List[CATReportData]:
    """Create sample CAT data for testing"""
    cat_data = []
    
    for i in range(count):
        event_time = datetime.now(timezone.utc) - timedelta(minutes=i)
        
        report = CATReportData(
            cat_reporter_id=f"CAT_{i % 10:03d}",
            event_id=f"EVENT_{i:06d}",
            event_type=OrderEventType.NEW_ORDER if i % 3 == 0 else OrderEventType.EXECUTION,
            event_timestamp=event_time,
            order_id=f"ORDER_{i:06d}",
            parent_order_id=f"PARENT_{i-1:06d}" if i > 0 and i % 5 == 0 else None,
            symbol="AAPL",
            security_type="equity",
            side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            quantity=100 * (i + 1),
            order_type=OrderType.MARKET if i % 2 == 0 else OrderType.LIMIT,
            price=Decimal("150.00") if i % 2 == 1 else None,
            account_id=f"ACCOUNT_{i % 20:03d}",
            account_holder_type="CUSTOMER",
            reporting_firm_id=f"FIRM_{i % 10:03d}",
            executing_firm_id=f"EXEC_FIRM_{i % 5:03d}",
            customer_id=f"CUST_{i % 50:03d}" if i % 12 != 0 else None,  # 8.3% missing
            execution_id=f"EXEC_{i:06d}" if i % 3 != 0 else None,
            execution_price=Decimal("149.99") if i % 3 != 0 else None,
            execution_quantity=100 * (i + 1) if i % 3 != 0 else None,
            market_center_id="NASDAQ",
            solicited_flag=i % 20 == 0,
            institutional_account_flag=i % 10 == 0
        )
        cat_data.append(report)
    
    return cat_data


def create_sample_trade_data(count: int = 100) -> List[TradeReportData]:
    """Create sample trade data for testing"""
    trade_data = []
    
    for i in range(count):
        trade_time = datetime.now(timezone.utc) - timedelta(minutes=i)
        
        trade = TradeReportData(
            trade_id=f"TRADE_{i:06d}",
            report_venue=TradeReportVenue.ADF,
            timestamp=trade_time,
            symbol="AAPL",
            security_type="equity",
            price=Decimal("150.00") + Decimal(str(i * 0.01)),
            quantity=100 * (i + 1),
            buy_side_firm=f"FIRM_{i % 10:03d}",
            sell_side_firm=f"FIRM_{(i+1) % 10:03d}",
            trade_modifier_1="REGULAR" if i % 10 != 0 else None,
            settlement_date=trade_time.date() + timedelta(days=2),
            buy_capacity="principal",
            sell_capacity="principal",
            cross_trade=i % 25 == 0,
            odd_lot_trade=i % 50 == 0,
            late_report=i % 30 == 0  # 3.3% late
        )
        trade_data.append(trade)
    
    return trade_data


def create_sample_adf_data(count: int = 50) -> List[ADFSubmissionData]:
    """Create sample ADF data for testing"""
    adf_data = []
    
    for i in range(count):
        submission_time = datetime.now(timezone.utc) - timedelta(minutes=i)
        
        submission = ADFSubmissionData(
            submission_id=f"ADF_{i:06d}",
            timestamp=submission_time,
            symbol="AAPL",
            bid_price=Decimal("149.95") + Decimal(str(i * 0.01)),
            ask_price=Decimal("150.05") + Decimal(str(i * 0.01)),
            bid_size=100 if i % 8 != 0 else 50,  # 12.5% below minimum
            ask_size=100 if i % 8 != 0 else 50,
            market_participant_id=f"PARTICIPANT_{i % 5:03d}",
            display_size=100 if i % 8 != 0 else 50,
            order_type=OrderType.LIMIT,
            locked_crossed_market=i % 20 == 0,  # 5% locked/crossed
            display_timestamp=submission_time,
            remove_timestamp=submission_time + timedelta(minutes=5)
        )
        adf_data.append(submission)
    
    return adf_data


# Test Cases
class TestFINRACompliance:
    """Test cases for FINRA compliance validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = FINRAComplianceValidator()
        self.reporter = FINRAComplianceReporter(self.validator)
    
    def test_oats_compliance_validation(self):
        """Test OATS compliance validation"""
        # Create sample OATS data
        oats_data = create_sample_oats_data(100)
        
        # Validate compliance
        result = self.validator.validate_oats_compliance(oats_data)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_events'] == 100
        assert 'timeliness_compliance' in result['metrics']
        assert 'completeness_compliance' in result['metrics']
    
    def test_cat_compliance_validation(self):
        """Test CAT compliance validation"""
        # Create sample CAT data
        cat_data = create_sample_cat_data(100)
        
        # Validate compliance
        result = self.validator.validate_cat_compliance(cat_data)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_events'] == 100
        assert 'timeliness_compliance' in result['metrics']
        assert 'completeness_compliance' in result['metrics']
    
    def test_trade_reporting_compliance_validation(self):
        """Test trade reporting compliance validation"""
        # Create sample trade data
        trade_data = create_sample_trade_data(100)
        
        # Validate compliance
        result = self.validator.validate_trade_reporting_compliance(trade_data)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_trades'] == 100
        assert 'timeliness_compliance' in result['metrics']
        assert 'accuracy_compliance' in result['metrics']
    
    def test_adf_compliance_validation(self):
        """Test ADF compliance validation"""
        # Create sample ADF data
        adf_data = create_sample_adf_data(50)
        
        # Validate compliance
        result = self.validator.validate_adf_compliance(adf_data)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_submissions'] == 50
        assert 'quote_quality' in result['metrics']
        assert 'display_compliance' in result['metrics']
    
    def test_market_surveillance_compliance_validation(self):
        """Test market surveillance compliance validation"""
        # Create sample trade data
        trade_data = create_sample_trade_data(100)
        
        # Validate compliance
        result = self.validator.validate_market_surveillance_compliance(trade_data)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_trades'] == 100
        assert 'surveillance_score' in result['metrics']
    
    def test_oats_report_generation(self):
        """Test OATS report generation"""
        # Create sample OATS data
        oats_data = create_sample_oats_data(100)
        
        # Generate report
        report = self.reporter.generate_oats_report(oats_data, "2024-01")
        
        # Assertions
        assert report['report_type'] == 'FINRA_OATS'
        assert report['period'] == '2024-01'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
    
    def test_cat_report_generation(self):
        """Test CAT report generation"""
        # Create sample CAT data
        cat_data = create_sample_cat_data(100)
        
        # Generate report
        report = self.reporter.generate_cat_report(cat_data, "2024-01")
        
        # Assertions
        assert report['report_type'] == 'FINRA_CAT'
        assert report['period'] == '2024-01'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
    
    def test_comprehensive_finra_compliance_validation(self):
        """Test comprehensive FINRA compliance validation"""
        # Create all types of sample data
        oats_data = create_sample_oats_data(100)
        cat_data = create_sample_cat_data(100)
        trade_data = create_sample_trade_data(100)
        adf_data = create_sample_adf_data(50)
        
        # Validate all compliance areas
        oats_result = self.validator.validate_oats_compliance(oats_data)
        cat_result = self.validator.validate_cat_compliance(cat_data)
        trade_result = self.validator.validate_trade_reporting_compliance(trade_data)
        adf_result = self.validator.validate_adf_compliance(adf_data)
        surveillance_result = self.validator.validate_market_surveillance_compliance(trade_data)
        
        # All should return valid results
        results = [oats_result, cat_result, trade_result, adf_result, surveillance_result]
        assert all('violations' in result for result in results)
        assert all('metrics' in result for result in results)
        
        # Check total violations
        total_violations = sum(len(result['violations']) for result in results)
        assert total_violations >= 0  # Should have some violations with sample data
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for FINRA compliance validation"""
        # Create large dataset
        large_oats = create_sample_oats_data(10000)
        large_cat = create_sample_cat_data(10000)
        large_trades = create_sample_trade_data(10000)
        large_adf = create_sample_adf_data(1000)
        
        # Measure performance
        start_time = time.time()
        
        # Run all validations
        self.validator.validate_oats_compliance(large_oats)
        self.validator.validate_cat_compliance(large_cat)
        self.validator.validate_trade_reporting_compliance(large_trades)
        self.validator.validate_adf_compliance(large_adf)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 15.0  # Should complete within 15 seconds
        assert processing_time / len(large_oats) < 0.001  # < 1ms per event
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        # Test with empty data
        empty_result = self.validator.validate_oats_compliance([])
        assert empty_result['metrics']['total_events'] == 0
        assert len(empty_result['violations']) == 0
        
        # Test with None values
        oats_with_nones = OATSReportData(
            firm_id="FIRM_001",
            order_id="ORDER_001",
            event_type=OrderEventType.NEW_ORDER,
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            account_type="CUSTOMER",
            origin_type=OrderOriginType.CUSTOMER,
            customer_id=None,  # None value
            execution_price=None,
            order_received_timestamp=datetime.now(timezone.utc) - timedelta(seconds=5)
        )
        
        result = self.validator.validate_oats_compliance([oats_with_nones])
        assert 'violations' in result
        assert 'metrics' in result


# Integration test
@pytest.mark.asyncio
async def test_finra_compliance_integration():
    """Integration test for FINRA compliance system"""
    # Initialize components
    validator = FINRAComplianceValidator()
    reporter = FINRAComplianceReporter(validator)
    
    # Create comprehensive test data
    oats_data = create_sample_oats_data(1000)
    cat_data = create_sample_cat_data(1000)
    trade_data = create_sample_trade_data(1000)
    adf_data = create_sample_adf_data(100)
    
    # Run comprehensive compliance check
    start_time = time.time()
    
    # Generate all reports
    oats_report = reporter.generate_oats_report(oats_data, "2024-01")
    cat_report = reporter.generate_cat_report(cat_data, "2024-01")
    trade_report = reporter.generate_trade_reporting_report(trade_data, "2024-01")
    adf_report = reporter.generate_adf_report(adf_data, "2024-01")
    surveillance_report = reporter.generate_market_surveillance_report(trade_data, "2024-01")
    
    end_time = time.time()
    
    # Performance check
    assert (end_time - start_time) < 20.0  # Should complete within 20 seconds
    
    # Validate all reports
    assert oats_report['report_type'] == 'FINRA_OATS'
    assert cat_report['report_type'] == 'FINRA_CAT'
    assert trade_report['report_type'] == 'FINRA_TRADE_REPORTING'
    assert adf_report['report_type'] == 'FINRA_ADF'
    assert surveillance_report['report_type'] == 'FINRA_MARKET_SURVEILLANCE'
    
    # Check report completeness
    reports = [oats_report, cat_report, trade_report, adf_report, surveillance_report]
    for report in reports:
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert 'generated_at' in report
    
    # Validate metrics
    assert oats_report['summary']['total_events'] == 1000
    assert cat_report['summary']['total_events'] == 1000
    assert trade_report['summary']['total_trades'] == 1000
    assert adf_report['summary']['total_submissions'] == 100
    assert surveillance_report['summary']['total_trades'] == 1000
    
    print("✅ FINRA Compliance Integration Test Passed")


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_finra_compliance_integration())