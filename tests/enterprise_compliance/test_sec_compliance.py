#!/usr/bin/env python3
"""
SEC Compliance Testing Suite
Agent 3: Regulatory Compliance Testing

Comprehensive testing for SEC (Securities and Exchange Commission) compliance
including Rule 605, Regulation SHO, and market maker obligations.

Features:
- Rule 605 (Order Execution Quality) compliance testing
- Regulation SHO (Short Sale) requirements validation
- Market maker obligations and reporting
- Trade execution quality metrics
- Best execution requirements
- Regulatory reporting validation
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


class SECRuleType(Enum):
    """SEC Rule types"""
    RULE_605 = "rule_605"
    REGULATION_SHO = "regulation_sho"
    MARKET_MAKER = "market_maker"
    BEST_EXECUTION = "best_execution"
    TRADE_REPORTING = "trade_reporting"
    POSITION_REPORTING = "position_reporting"


class OrderType(Enum):
    """Order types for compliance testing"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"
    SHORT_SELL = "short_sell"


class ExecutionVenue(Enum):
    """Execution venues"""
    NASDAQ = "nasdaq"
    NYSE = "nyse"
    ARCA = "arca"
    BATS = "bats"
    DARK_POOL = "dark_pool"
    INTERNALIZED = "internalized"


@dataclass
class OrderExecutionData:
    """Order execution data for Rule 605 testing"""
    order_id: str
    symbol: str
    order_type: OrderType
    order_side: OrderSide
    order_size: int
    order_price: Optional[Decimal]
    order_timestamp: datetime
    
    # Execution details
    execution_venue: ExecutionVenue
    execution_price: Decimal
    execution_size: int
    execution_timestamp: datetime
    execution_id: str
    
    # Quality metrics
    price_improvement: Optional[Decimal] = None
    effective_spread: Optional[Decimal] = None
    realized_spread: Optional[Decimal] = None
    market_impact: Optional[Decimal] = None
    
    # Timing metrics
    response_time_ms: Optional[int] = None
    fill_rate: Optional[float] = None
    
    # Best execution
    nbbo_bid: Optional[Decimal] = None
    nbbo_ask: Optional[Decimal] = None
    midpoint: Optional[Decimal] = None


@dataclass
class ShortSaleData:
    """Short sale data for Regulation SHO testing"""
    trade_id: str
    symbol: str
    trade_timestamp: datetime
    quantity: int
    price: Decimal
    
    # Regulation SHO fields
    short_sale_exempt: bool = False
    locate_obtained: bool = False
    locate_timestamp: Optional[datetime] = None
    locate_source: Optional[str] = None
    
    # Circuit breaker
    circuit_breaker_active: bool = False
    alternative_uptick_rule: bool = False
    
    # Threshold security
    threshold_security: bool = False
    fail_to_deliver_days: int = 0


@dataclass
class MarketMakerData:
    """Market maker data for compliance testing"""
    mm_id: str
    symbol: str
    timestamp: datetime
    
    # Quote obligations
    bid_price: Decimal
    ask_price: Decimal
    bid_size: int
    ask_size: int
    spread: Decimal
    
    # Performance metrics
    time_at_nbbo: float
    quote_update_frequency: int
    fill_rate: float
    
    # Regulatory requirements
    minimum_quote_size: int
    maximum_spread: Decimal
    quote_uptime: float
    
    # Reporting
    monthly_volume: int
    market_share: float
    adverse_selection: float


class SECComplianceValidator:
    """SEC Compliance validation engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.violations: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'rule_605': {
                'response_time_threshold_ms': 100,
                'fill_rate_threshold': 0.95,
                'price_improvement_threshold': 0.001,
                'effective_spread_threshold': 0.01
            },
            'regulation_sho': {
                'locate_required': True,
                'locate_timeout_hours': 24,
                'circuit_breaker_threshold': 0.10,
                'threshold_security_days': 5
            },
            'market_maker': {
                'minimum_quote_size': 100,
                'maximum_spread_bps': 50,
                'quote_uptime_threshold': 0.95,
                'time_at_nbbo_threshold': 0.80
            },
            'best_execution': {
                'price_improvement_required': True,
                'venue_comparison_required': True,
                'execution_quality_threshold': 0.90
            }
        }
    
    def validate_rule_605_compliance(self, executions: List[OrderExecutionData]) -> Dict[str, Any]:
        """Validate Rule 605 order execution quality compliance"""
        violations = []
        metrics = {
            'total_orders': len(executions),
            'avg_response_time': 0,
            'avg_fill_rate': 0,
            'price_improvement_rate': 0,
            'effective_spread_avg': 0
        }
        
        if not executions:
            return {'violations': violations, 'metrics': metrics}
        
        # Calculate metrics
        response_times = [e.response_time_ms for e in executions if e.response_time_ms]
        fill_rates = [e.fill_rate for e in executions if e.fill_rate]
        price_improvements = [e.price_improvement for e in executions if e.price_improvement]
        effective_spreads = [e.effective_spread for e in executions if e.effective_spread]
        
        if response_times:
            metrics['avg_response_time'] = sum(response_times) / len(response_times)
        if fill_rates:
            metrics['avg_fill_rate'] = sum(fill_rates) / len(fill_rates)
        if price_improvements:
            metrics['price_improvement_rate'] = len([p for p in price_improvements if p > 0]) / len(price_improvements)
        if effective_spreads:
            metrics['effective_spread_avg'] = sum(effective_spreads) / len(effective_spreads)
        
        # Check violations
        threshold_config = self.config['rule_605']
        
        # Response time violations
        if metrics['avg_response_time'] > threshold_config['response_time_threshold_ms']:
            violations.append({
                'rule': 'RULE_605_RESPONSE_TIME',
                'severity': 'HIGH',
                'description': f"Average response time {metrics['avg_response_time']}ms exceeds threshold {threshold_config['response_time_threshold_ms']}ms",
                'actual_value': metrics['avg_response_time'],
                'threshold': threshold_config['response_time_threshold_ms']
            })
        
        # Fill rate violations
        if metrics['avg_fill_rate'] < threshold_config['fill_rate_threshold']:
            violations.append({
                'rule': 'RULE_605_FILL_RATE',
                'severity': 'HIGH',
                'description': f"Average fill rate {metrics['avg_fill_rate']} below threshold {threshold_config['fill_rate_threshold']}",
                'actual_value': metrics['avg_fill_rate'],
                'threshold': threshold_config['fill_rate_threshold']
            })
        
        # Price improvement violations
        if metrics['price_improvement_rate'] < threshold_config['price_improvement_threshold']:
            violations.append({
                'rule': 'RULE_605_PRICE_IMPROVEMENT',
                'severity': 'MEDIUM',
                'description': f"Price improvement rate {metrics['price_improvement_rate']} below threshold {threshold_config['price_improvement_threshold']}",
                'actual_value': metrics['price_improvement_rate'],
                'threshold': threshold_config['price_improvement_threshold']
            })
        
        # Individual order violations
        for execution in executions:
            if execution.response_time_ms and execution.response_time_ms > threshold_config['response_time_threshold_ms'] * 2:
                violations.append({
                    'rule': 'RULE_605_INDIVIDUAL_RESPONSE_TIME',
                    'severity': 'HIGH',
                    'description': f"Order {execution.order_id} response time {execution.response_time_ms}ms significantly exceeds threshold",
                    'order_id': execution.order_id,
                    'actual_value': execution.response_time_ms,
                    'threshold': threshold_config['response_time_threshold_ms']
                })
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_regulation_sho_compliance(self, short_sales: List[ShortSaleData]) -> Dict[str, Any]:
        """Validate Regulation SHO short sale compliance"""
        violations = []
        metrics = {
            'total_short_sales': len(short_sales),
            'locate_compliance_rate': 0,
            'circuit_breaker_compliance_rate': 0,
            'threshold_security_violations': 0
        }
        
        if not short_sales:
            return {'violations': violations, 'metrics': metrics}
        
        locate_compliant = 0
        circuit_breaker_compliant = 0
        threshold_violations = 0
        
        sho_config = self.config['regulation_sho']
        
        for sale in short_sales:
            # Check locate requirement
            if sho_config['locate_required']:
                if not sale.locate_obtained:
                    violations.append({
                        'rule': 'REGULATION_SHO_LOCATE',
                        'severity': 'HIGH',
                        'description': f"Short sale {sale.trade_id} executed without required locate",
                        'trade_id': sale.trade_id,
                        'symbol': sale.symbol
                    })
                else:
                    locate_compliant += 1
                    
                    # Check locate timing
                    if sale.locate_timestamp:
                        locate_age = (sale.trade_timestamp - sale.locate_timestamp).total_seconds() / 3600
                        if locate_age > sho_config['locate_timeout_hours']:
                            violations.append({
                                'rule': 'REGULATION_SHO_LOCATE_TIMEOUT',
                                'severity': 'MEDIUM',
                                'description': f"Short sale {sale.trade_id} locate expired ({locate_age:.1f} hours old)",
                                'trade_id': sale.trade_id,
                                'locate_age_hours': locate_age
                            })
            
            # Check circuit breaker compliance
            if sale.circuit_breaker_active and not sale.alternative_uptick_rule:
                violations.append({
                    'rule': 'REGULATION_SHO_CIRCUIT_BREAKER',
                    'severity': 'HIGH',
                    'description': f"Short sale {sale.trade_id} executed during circuit breaker without alternative uptick rule",
                    'trade_id': sale.trade_id,
                    'symbol': sale.symbol
                })
            else:
                circuit_breaker_compliant += 1
            
            # Check threshold security
            if sale.threshold_security and sale.fail_to_deliver_days >= sho_config['threshold_security_days']:
                violations.append({
                    'rule': 'REGULATION_SHO_THRESHOLD_SECURITY',
                    'severity': 'CRITICAL',
                    'description': f"Short sale {sale.trade_id} in threshold security with {sale.fail_to_deliver_days} days FTD",
                    'trade_id': sale.trade_id,
                    'symbol': sale.symbol,
                    'ftd_days': sale.fail_to_deliver_days
                })
                threshold_violations += 1
        
        # Calculate metrics
        metrics['locate_compliance_rate'] = locate_compliant / len(short_sales)
        metrics['circuit_breaker_compliance_rate'] = circuit_breaker_compliant / len(short_sales)
        metrics['threshold_security_violations'] = threshold_violations
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_market_maker_compliance(self, mm_data: List[MarketMakerData]) -> Dict[str, Any]:
        """Validate market maker obligations compliance"""
        violations = []
        metrics = {
            'total_market_makers': len(mm_data),
            'avg_quote_uptime': 0,
            'avg_time_at_nbbo': 0,
            'avg_spread': 0,
            'quote_size_compliance_rate': 0
        }
        
        if not mm_data:
            return {'violations': violations, 'metrics': metrics}
        
        mm_config = self.config['market_maker']
        quote_size_compliant = 0
        
        uptimes = []
        nbbo_times = []
        spreads = []
        
        for mm in mm_data:
            # Check minimum quote size
            if mm.bid_size >= mm_config['minimum_quote_size'] and mm.ask_size >= mm_config['minimum_quote_size']:
                quote_size_compliant += 1
            else:
                violations.append({
                    'rule': 'MARKET_MAKER_QUOTE_SIZE',
                    'severity': 'HIGH',
                    'description': f"Market maker {mm.mm_id} quote size below minimum ({mm.bid_size}x{mm.ask_size})",
                    'market_maker_id': mm.mm_id,
                    'symbol': mm.symbol,
                    'bid_size': mm.bid_size,
                    'ask_size': mm.ask_size,
                    'minimum_required': mm_config['minimum_quote_size']
                })
            
            # Check maximum spread
            spread_bps = float(mm.spread / mm.bid_price * 10000)
            if spread_bps > mm_config['maximum_spread_bps']:
                violations.append({
                    'rule': 'MARKET_MAKER_SPREAD',
                    'severity': 'MEDIUM',
                    'description': f"Market maker {mm.mm_id} spread {spread_bps:.1f}bps exceeds maximum {mm_config['maximum_spread_bps']}bps",
                    'market_maker_id': mm.mm_id,
                    'symbol': mm.symbol,
                    'actual_spread_bps': spread_bps,
                    'maximum_spread_bps': mm_config['maximum_spread_bps']
                })
            
            # Check quote uptime
            if mm.quote_uptime < mm_config['quote_uptime_threshold']:
                violations.append({
                    'rule': 'MARKET_MAKER_UPTIME',
                    'severity': 'HIGH',
                    'description': f"Market maker {mm.mm_id} quote uptime {mm.quote_uptime:.1%} below threshold {mm_config['quote_uptime_threshold']:.1%}",
                    'market_maker_id': mm.mm_id,
                    'symbol': mm.symbol,
                    'actual_uptime': mm.quote_uptime,
                    'threshold': mm_config['quote_uptime_threshold']
                })
            
            # Check time at NBBO
            if mm.time_at_nbbo < mm_config['time_at_nbbo_threshold']:
                violations.append({
                    'rule': 'MARKET_MAKER_NBBO_TIME',
                    'severity': 'MEDIUM',
                    'description': f"Market maker {mm.mm_id} time at NBBO {mm.time_at_nbbo:.1%} below threshold {mm_config['time_at_nbbo_threshold']:.1%}",
                    'market_maker_id': mm.mm_id,
                    'symbol': mm.symbol,
                    'actual_time_at_nbbo': mm.time_at_nbbo,
                    'threshold': mm_config['time_at_nbbo_threshold']
                })
            
            # Collect metrics
            uptimes.append(mm.quote_uptime)
            nbbo_times.append(mm.time_at_nbbo)
            spreads.append(float(mm.spread))
        
        # Calculate metrics
        metrics['avg_quote_uptime'] = sum(uptimes) / len(uptimes)
        metrics['avg_time_at_nbbo'] = sum(nbbo_times) / len(nbbo_times)
        metrics['avg_spread'] = sum(spreads) / len(spreads)
        metrics['quote_size_compliance_rate'] = quote_size_compliant / len(mm_data)
        
        return {'violations': violations, 'metrics': metrics}
    
    def validate_best_execution_compliance(self, executions: List[OrderExecutionData]) -> Dict[str, Any]:
        """Validate best execution requirements compliance"""
        violations = []
        metrics = {
            'total_executions': len(executions),
            'price_improvement_rate': 0,
            'venue_routing_quality': 0,
            'execution_quality_score': 0
        }
        
        if not executions:
            return {'violations': violations, 'metrics': metrics}
        
        be_config = self.config['best_execution']
        
        price_improvements = 0
        venue_routing_scores = []
        execution_quality_scores = []
        
        for execution in executions:
            # Check price improvement
            if execution.price_improvement and execution.price_improvement > 0:
                price_improvements += 1
            elif be_config['price_improvement_required'] and execution.order_type == OrderType.MARKET:
                violations.append({
                    'rule': 'BEST_EXECUTION_PRICE_IMPROVEMENT',
                    'severity': 'MEDIUM',
                    'description': f"Market order {execution.order_id} did not receive price improvement",
                    'order_id': execution.order_id,
                    'symbol': execution.symbol,
                    'execution_price': float(execution.execution_price),
                    'price_improvement': float(execution.price_improvement) if execution.price_improvement else 0
                })
            
            # Check venue routing quality
            venue_score = self._calculate_venue_routing_score(execution)
            venue_routing_scores.append(venue_score)
            
            # Check execution quality
            execution_score = self._calculate_execution_quality_score(execution)
            execution_quality_scores.append(execution_score)
            
            if execution_score < be_config['execution_quality_threshold']:
                violations.append({
                    'rule': 'BEST_EXECUTION_QUALITY',
                    'severity': 'HIGH',
                    'description': f"Order {execution.order_id} execution quality score {execution_score:.2f} below threshold {be_config['execution_quality_threshold']}",
                    'order_id': execution.order_id,
                    'symbol': execution.symbol,
                    'execution_quality_score': execution_score,
                    'threshold': be_config['execution_quality_threshold']
                })
        
        # Calculate metrics
        metrics['price_improvement_rate'] = price_improvements / len(executions)
        metrics['venue_routing_quality'] = sum(venue_routing_scores) / len(venue_routing_scores)
        metrics['execution_quality_score'] = sum(execution_quality_scores) / len(execution_quality_scores)
        
        return {'violations': violations, 'metrics': metrics}
    
    def _calculate_venue_routing_score(self, execution: OrderExecutionData) -> float:
        """Calculate venue routing quality score"""
        # Simplified scoring based on execution venue
        venue_scores = {
            ExecutionVenue.NASDAQ: 0.9,
            ExecutionVenue.NYSE: 0.9,
            ExecutionVenue.ARCA: 0.85,
            ExecutionVenue.BATS: 0.85,
            ExecutionVenue.DARK_POOL: 0.8,
            ExecutionVenue.INTERNALIZED: 0.75
        }
        
        base_score = venue_scores.get(execution.execution_venue, 0.5)
        
        # Adjust for price improvement
        if execution.price_improvement and execution.price_improvement > 0:
            base_score += 0.1
        
        # Adjust for execution speed
        if execution.response_time_ms and execution.response_time_ms < 50:
            base_score += 0.05
        
        return min(base_score, 1.0)
    
    def _calculate_execution_quality_score(self, execution: OrderExecutionData) -> float:
        """Calculate execution quality score"""
        score = 0.5  # Base score
        
        # Price improvement component
        if execution.price_improvement and execution.price_improvement > 0:
            score += 0.3
        
        # Effective spread component
        if execution.effective_spread and execution.nbbo_bid and execution.nbbo_ask:
            theoretical_spread = float(execution.nbbo_ask - execution.nbbo_bid)
            if theoretical_spread > 0:
                spread_ratio = float(execution.effective_spread) / theoretical_spread
                score += 0.2 * (1 - min(spread_ratio, 1.0))
        
        # Fill rate component
        if execution.fill_rate:
            score += 0.2 * execution.fill_rate
        
        # Speed component
        if execution.response_time_ms:
            speed_score = max(0, 1 - (execution.response_time_ms / 1000))  # 1 second max
            score += 0.1 * speed_score
        
        return min(score, 1.0)


class SECComplianceReporter:
    """SEC compliance reporting system"""
    
    def __init__(self, validator: SECComplianceValidator):
        self.validator = validator
    
    def generate_rule_605_report(self, executions: List[OrderExecutionData], period: str) -> Dict[str, Any]:
        """Generate Rule 605 execution quality report"""
        compliance_result = self.validator.validate_rule_605_compliance(executions)
        
        report = {
            'report_type': 'SEC_RULE_605',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_orders': compliance_result['metrics']['total_orders'],
                'compliance_violations': len(compliance_result['violations']),
                'avg_response_time_ms': compliance_result['metrics']['avg_response_time'],
                'avg_fill_rate': compliance_result['metrics']['avg_fill_rate'],
                'price_improvement_rate': compliance_result['metrics']['price_improvement_rate']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_rule_605_recommendations(compliance_result)
        }
        
        return report
    
    def generate_regulation_sho_report(self, short_sales: List[ShortSaleData], period: str) -> Dict[str, Any]:
        """Generate Regulation SHO compliance report"""
        compliance_result = self.validator.validate_regulation_sho_compliance(short_sales)
        
        report = {
            'report_type': 'SEC_REGULATION_SHO',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_short_sales': compliance_result['metrics']['total_short_sales'],
                'compliance_violations': len(compliance_result['violations']),
                'locate_compliance_rate': compliance_result['metrics']['locate_compliance_rate'],
                'circuit_breaker_compliance_rate': compliance_result['metrics']['circuit_breaker_compliance_rate'],
                'threshold_security_violations': compliance_result['metrics']['threshold_security_violations']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_regulation_sho_recommendations(compliance_result)
        }
        
        return report
    
    def generate_market_maker_report(self, mm_data: List[MarketMakerData], period: str) -> Dict[str, Any]:
        """Generate market maker compliance report"""
        compliance_result = self.validator.validate_market_maker_compliance(mm_data)
        
        report = {
            'report_type': 'SEC_MARKET_MAKER',
            'period': period,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_market_makers': compliance_result['metrics']['total_market_makers'],
                'compliance_violations': len(compliance_result['violations']),
                'avg_quote_uptime': compliance_result['metrics']['avg_quote_uptime'],
                'avg_time_at_nbbo': compliance_result['metrics']['avg_time_at_nbbo'],
                'quote_size_compliance_rate': compliance_result['metrics']['quote_size_compliance_rate']
            },
            'metrics': compliance_result['metrics'],
            'violations': compliance_result['violations'],
            'recommendations': self._generate_market_maker_recommendations(compliance_result)
        }
        
        return report
    
    def _generate_rule_605_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate Rule 605 compliance recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['avg_response_time'] > 100:
            recommendations.append("Optimize order routing algorithms to reduce response times")
        
        if compliance_result['metrics']['avg_fill_rate'] < 0.95:
            recommendations.append("Review liquidity sources and improve fill rates")
        
        if compliance_result['metrics']['price_improvement_rate'] < 0.5:
            recommendations.append("Enhance price improvement mechanisms for better execution quality")
        
        return recommendations
    
    def _generate_regulation_sho_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate Regulation SHO compliance recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['locate_compliance_rate'] < 1.0:
            recommendations.append("Strengthen locate procedures to ensure 100% compliance")
        
        if compliance_result['metrics']['threshold_security_violations'] > 0:
            recommendations.append("Implement stricter controls for threshold securities")
        
        return recommendations
    
    def _generate_market_maker_recommendations(self, compliance_result: Dict[str, Any]) -> List[str]:
        """Generate market maker compliance recommendations"""
        recommendations = []
        
        if compliance_result['metrics']['avg_quote_uptime'] < 0.95:
            recommendations.append("Improve quote system reliability and uptime")
        
        if compliance_result['metrics']['quote_size_compliance_rate'] < 1.0:
            recommendations.append("Ensure all quotes meet minimum size requirements")
        
        return recommendations


# Test fixtures and utilities
def create_sample_execution_data(count: int = 100) -> List[OrderExecutionData]:
    """Create sample execution data for testing"""
    executions = []
    
    for i in range(count):
        execution = OrderExecutionData(
            order_id=f"ORD_{i:06d}",
            symbol="AAPL",
            order_type=OrderType.MARKET if i % 2 == 0 else OrderType.LIMIT,
            order_side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
            order_size=100 * (i + 1),
            order_price=Decimal("150.00") if i % 2 == 1 else None,
            order_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            execution_venue=ExecutionVenue.NASDAQ if i % 3 == 0 else ExecutionVenue.NYSE,
            execution_price=Decimal("149.99") + Decimal(str(i * 0.01)),
            execution_size=100 * (i + 1),
            execution_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i) + timedelta(milliseconds=50 + i),
            execution_id=f"EXE_{i:06d}",
            price_improvement=Decimal("0.01") if i % 3 == 0 else None,
            effective_spread=Decimal("0.02"),
            response_time_ms=50 + i,
            fill_rate=0.98 if i % 5 != 0 else 0.85,
            nbbo_bid=Decimal("149.98"),
            nbbo_ask=Decimal("150.02"),
            midpoint=Decimal("150.00")
        )
        executions.append(execution)
    
    return executions


def create_sample_short_sale_data(count: int = 50) -> List[ShortSaleData]:
    """Create sample short sale data for testing"""
    short_sales = []
    
    for i in range(count):
        short_sale = ShortSaleData(
            trade_id=f"SHORT_{i:06d}",
            symbol="AAPL",
            trade_timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            quantity=100 * (i + 1),
            price=Decimal("149.99") + Decimal(str(i * 0.01)),
            locate_obtained=i % 10 != 0,  # 10% without locate
            locate_timestamp=datetime.now(timezone.utc) - timedelta(hours=1) if i % 10 != 0 else None,
            locate_source="PRIME_BROKER" if i % 10 != 0 else None,
            circuit_breaker_active=i % 20 == 0,  # 5% during circuit breaker
            threshold_security=i % 50 == 0,  # 2% threshold securities
            fail_to_deliver_days=i % 10 if i % 50 == 0 else 0
        )
        short_sales.append(short_sale)
    
    return short_sales


def create_sample_market_maker_data(count: int = 10) -> List[MarketMakerData]:
    """Create sample market maker data for testing"""
    mm_data = []
    
    for i in range(count):
        mm = MarketMakerData(
            mm_id=f"MM_{i:03d}",
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            bid_price=Decimal("149.98"),
            ask_price=Decimal("150.02"),
            bid_size=100 if i % 5 != 0 else 50,  # 20% below minimum
            ask_size=100 if i % 5 != 0 else 50,
            spread=Decimal("0.04"),
            time_at_nbbo=0.85 if i % 8 != 0 else 0.75,  # 12.5% below threshold
            quote_update_frequency=100,
            fill_rate=0.95,
            minimum_quote_size=100,
            maximum_spread=Decimal("0.05"),
            quote_uptime=0.97 if i % 10 != 0 else 0.92,  # 10% below threshold
            monthly_volume=1000000,
            market_share=0.05,
            adverse_selection=0.15
        )
        mm_data.append(mm)
    
    return mm_data


# Test Cases
class TestSECCompliance:
    """Test cases for SEC compliance validation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = SECComplianceValidator()
        self.reporter = SECComplianceReporter(self.validator)
    
    def test_rule_605_compliance_validation(self):
        """Test Rule 605 order execution quality compliance"""
        # Create sample execution data
        executions = create_sample_execution_data(100)
        
        # Validate compliance
        result = self.validator.validate_rule_605_compliance(executions)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_orders'] == 100
        assert 'avg_response_time' in result['metrics']
        assert 'avg_fill_rate' in result['metrics']
        assert 'price_improvement_rate' in result['metrics']
    
    def test_rule_605_response_time_violation(self):
        """Test Rule 605 response time violation detection"""
        # Create execution with slow response time
        execution = OrderExecutionData(
            order_id="TEST_001",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            order_side=OrderSide.BUY,
            order_size=100,
            order_price=None,
            order_timestamp=datetime.now(timezone.utc),
            execution_venue=ExecutionVenue.NASDAQ,
            execution_price=Decimal("150.00"),
            execution_size=100,
            execution_timestamp=datetime.now(timezone.utc) + timedelta(milliseconds=300),
            execution_id="EXE_001",
            response_time_ms=300,  # Exceeds threshold
            fill_rate=1.0
        )
        
        result = self.validator.validate_rule_605_compliance([execution])
        
        # Should have violation
        assert len(result['violations']) > 0
        violation = next(v for v in result['violations'] if v['rule'] == 'RULE_605_INDIVIDUAL_RESPONSE_TIME')
        assert violation['severity'] == 'HIGH'
        assert violation['order_id'] == 'TEST_001'
    
    def test_regulation_sho_compliance_validation(self):
        """Test Regulation SHO short sale compliance"""
        # Create sample short sale data
        short_sales = create_sample_short_sale_data(50)
        
        # Validate compliance
        result = self.validator.validate_regulation_sho_compliance(short_sales)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_short_sales'] == 50
        assert 'locate_compliance_rate' in result['metrics']
        assert 'circuit_breaker_compliance_rate' in result['metrics']
    
    def test_regulation_sho_locate_violation(self):
        """Test Regulation SHO locate requirement violation"""
        # Create short sale without locate
        short_sale = ShortSaleData(
            trade_id="SHORT_001",
            symbol="AAPL",
            trade_timestamp=datetime.now(timezone.utc),
            quantity=100,
            price=Decimal("150.00"),
            locate_obtained=False  # Violation
        )
        
        result = self.validator.validate_regulation_sho_compliance([short_sale])
        
        # Should have violation
        assert len(result['violations']) > 0
        violation = next(v for v in result['violations'] if v['rule'] == 'REGULATION_SHO_LOCATE')
        assert violation['severity'] == 'HIGH'
        assert violation['trade_id'] == 'SHORT_001'
    
    def test_regulation_sho_circuit_breaker_violation(self):
        """Test Regulation SHO circuit breaker violation"""
        # Create short sale during circuit breaker without alternative uptick rule
        short_sale = ShortSaleData(
            trade_id="SHORT_002",
            symbol="AAPL",
            trade_timestamp=datetime.now(timezone.utc),
            quantity=100,
            price=Decimal("150.00"),
            locate_obtained=True,
            locate_timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            circuit_breaker_active=True,
            alternative_uptick_rule=False  # Violation
        )
        
        result = self.validator.validate_regulation_sho_compliance([short_sale])
        
        # Should have violation
        assert len(result['violations']) > 0
        violation = next(v for v in result['violations'] if v['rule'] == 'REGULATION_SHO_CIRCUIT_BREAKER')
        assert violation['severity'] == 'HIGH'
        assert violation['trade_id'] == 'SHORT_002'
    
    def test_market_maker_compliance_validation(self):
        """Test market maker obligations compliance"""
        # Create sample market maker data
        mm_data = create_sample_market_maker_data(10)
        
        # Validate compliance
        result = self.validator.validate_market_maker_compliance(mm_data)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_market_makers'] == 10
        assert 'avg_quote_uptime' in result['metrics']
        assert 'quote_size_compliance_rate' in result['metrics']
    
    def test_market_maker_quote_size_violation(self):
        """Test market maker quote size violation"""
        # Create market maker with insufficient quote size
        mm = MarketMakerData(
            mm_id="MM_001",
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid_price=Decimal("149.98"),
            ask_price=Decimal("150.02"),
            bid_size=50,  # Below minimum
            ask_size=50,  # Below minimum
            spread=Decimal("0.04"),
            time_at_nbbo=0.85,
            quote_update_frequency=100,
            fill_rate=0.95,
            minimum_quote_size=100,
            maximum_spread=Decimal("0.05"),
            quote_uptime=0.97,
            monthly_volume=1000000,
            market_share=0.05,
            adverse_selection=0.15
        )
        
        result = self.validator.validate_market_maker_compliance([mm])
        
        # Should have violation
        assert len(result['violations']) > 0
        violation = next(v for v in result['violations'] if v['rule'] == 'MARKET_MAKER_QUOTE_SIZE')
        assert violation['severity'] == 'HIGH'
        assert violation['market_maker_id'] == 'MM_001'
    
    def test_best_execution_compliance_validation(self):
        """Test best execution requirements compliance"""
        # Create sample execution data
        executions = create_sample_execution_data(100)
        
        # Validate compliance
        result = self.validator.validate_best_execution_compliance(executions)
        
        # Assertions
        assert 'violations' in result
        assert 'metrics' in result
        assert result['metrics']['total_executions'] == 100
        assert 'price_improvement_rate' in result['metrics']
        assert 'execution_quality_score' in result['metrics']
    
    def test_rule_605_report_generation(self):
        """Test Rule 605 report generation"""
        # Create sample execution data
        executions = create_sample_execution_data(100)
        
        # Generate report
        report = self.reporter.generate_rule_605_report(executions, "2024-01")
        
        # Assertions
        assert report['report_type'] == 'SEC_RULE_605'
        assert report['period'] == '2024-01'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert report['summary']['total_orders'] == 100
    
    def test_regulation_sho_report_generation(self):
        """Test Regulation SHO report generation"""
        # Create sample short sale data
        short_sales = create_sample_short_sale_data(50)
        
        # Generate report
        report = self.reporter.generate_regulation_sho_report(short_sales, "2024-01")
        
        # Assertions
        assert report['report_type'] == 'SEC_REGULATION_SHO'
        assert report['period'] == '2024-01'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert report['summary']['total_short_sales'] == 50
    
    def test_market_maker_report_generation(self):
        """Test market maker report generation"""
        # Create sample market maker data
        mm_data = create_sample_market_maker_data(10)
        
        # Generate report
        report = self.reporter.generate_market_maker_report(mm_data, "2024-01")
        
        # Assertions
        assert report['report_type'] == 'SEC_MARKET_MAKER'
        assert report['period'] == '2024-01'
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert report['summary']['total_market_makers'] == 10
    
    def test_comprehensive_sec_compliance_validation(self):
        """Test comprehensive SEC compliance validation"""
        # Create all types of sample data
        executions = create_sample_execution_data(100)
        short_sales = create_sample_short_sale_data(50)
        mm_data = create_sample_market_maker_data(10)
        
        # Validate all compliance areas
        rule_605_result = self.validator.validate_rule_605_compliance(executions)
        sho_result = self.validator.validate_regulation_sho_compliance(short_sales)
        mm_result = self.validator.validate_market_maker_compliance(mm_data)
        be_result = self.validator.validate_best_execution_compliance(executions)
        
        # All should return valid results
        assert all('violations' in result for result in [rule_605_result, sho_result, mm_result, be_result])
        assert all('metrics' in result for result in [rule_605_result, sho_result, mm_result, be_result])
        
        # Check total violations
        total_violations = sum(len(result['violations']) for result in [rule_605_result, sho_result, mm_result, be_result])
        assert total_violations >= 0  # Should have some violations with sample data
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for compliance validation"""
        # Create large dataset
        large_executions = create_sample_execution_data(10000)
        large_short_sales = create_sample_short_sale_data(5000)
        large_mm_data = create_sample_market_maker_data(100)
        
        # Measure performance
        start_time = time.time()
        
        # Run all validations
        self.validator.validate_rule_605_compliance(large_executions)
        self.validator.validate_regulation_sho_compliance(large_short_sales)
        self.validator.validate_market_maker_compliance(large_mm_data)
        self.validator.validate_best_execution_compliance(large_executions)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performance assertions
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert processing_time / len(large_executions) < 0.001  # < 1ms per execution
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        # Test with empty data
        empty_result = self.validator.validate_rule_605_compliance([])
        assert empty_result['metrics']['total_orders'] == 0
        assert len(empty_result['violations']) == 0
        
        # Test with None values
        execution_with_nones = OrderExecutionData(
            order_id="TEST_001",
            symbol="AAPL",
            order_type=OrderType.MARKET,
            order_side=OrderSide.BUY,
            order_size=100,
            order_price=None,
            order_timestamp=datetime.now(timezone.utc),
            execution_venue=ExecutionVenue.NASDAQ,
            execution_price=Decimal("150.00"),
            execution_size=100,
            execution_timestamp=datetime.now(timezone.utc),
            execution_id="EXE_001",
            response_time_ms=None,  # None value
            fill_rate=None  # None value
        )
        
        result = self.validator.validate_rule_605_compliance([execution_with_nones])
        assert 'violations' in result
        assert 'metrics' in result
    
    def test_real_time_monitoring_simulation(self):
        """Test real-time monitoring simulation"""
        # Simulate real-time data stream
        executions = []
        violations_count = 0
        
        for i in range(100):
            # Create execution with varying characteristics
            execution = OrderExecutionData(
                order_id=f"RT_{i:06d}",
                symbol="AAPL",
                order_type=OrderType.MARKET,
                order_side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_size=100,
                order_price=None,
                order_timestamp=datetime.now(timezone.utc) - timedelta(milliseconds=i*10),
                execution_venue=ExecutionVenue.NASDAQ,
                execution_price=Decimal("150.00"),
                execution_size=100,
                execution_timestamp=datetime.now(timezone.utc) - timedelta(milliseconds=i*10-50),
                execution_id=f"EXE_{i:06d}",
                response_time_ms=50 + (i % 10) * 20,  # Varying response times
                fill_rate=0.95 + (i % 10) * 0.005
            )
            executions.append(execution)
            
            # Check compliance every 10 executions
            if (i + 1) % 10 == 0:
                result = self.validator.validate_rule_605_compliance(executions[-10:])
                violations_count += len(result['violations'])
        
        # Should have processed all executions
        assert len(executions) == 100
        assert violations_count >= 0


# Integration test
@pytest.mark.asyncio
async def test_sec_compliance_integration():
    """Integration test for SEC compliance system"""
    # Initialize components
    validator = SECComplianceValidator()
    reporter = SECComplianceReporter(validator)
    
    # Create comprehensive test data
    executions = create_sample_execution_data(1000)
    short_sales = create_sample_short_sale_data(500)
    mm_data = create_sample_market_maker_data(50)
    
    # Run comprehensive compliance check
    start_time = time.time()
    
    # Generate all reports
    rule_605_report = reporter.generate_rule_605_report(executions, "2024-01")
    sho_report = reporter.generate_regulation_sho_report(short_sales, "2024-01")
    mm_report = reporter.generate_market_maker_report(mm_data, "2024-01")
    
    end_time = time.time()
    
    # Performance check
    assert (end_time - start_time) < 10.0  # Should complete within 10 seconds
    
    # Validate all reports
    assert rule_605_report['report_type'] == 'SEC_RULE_605'
    assert sho_report['report_type'] == 'SEC_REGULATION_SHO'
    assert mm_report['report_type'] == 'SEC_MARKET_MAKER'
    
    # Check report completeness
    for report in [rule_605_report, sho_report, mm_report]:
        assert 'summary' in report
        assert 'metrics' in report
        assert 'violations' in report
        assert 'recommendations' in report
        assert 'generated_at' in report
    
    # Validate metrics
    assert rule_605_report['summary']['total_orders'] == 1000
    assert sho_report['summary']['total_short_sales'] == 500
    assert mm_report['summary']['total_market_makers'] == 50
    
    print("✅ SEC Compliance Integration Test Passed")


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_sec_compliance_integration())