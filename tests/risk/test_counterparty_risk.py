"""
Counterparty Risk Testing Suite

This comprehensive test suite validates counterparty risk management including
credit exposure calculation, monitoring, netting and collateral management,
and counterparty default scenarios.

Key Test Areas:
1. Credit exposure calculation and monitoring
2. Netting and collateral management
3. Counterparty default scenarios and recovery
4. Credit rating and limit management
5. Concentration risk assessment
6. Regulatory capital calculations
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import uuid

from src.core.events import EventBus, Event, EventType
from src.risk.agents.portfolio_optimizer_agent import PortfolioOptimizerAgent
from src.risk.core.var_calculator import VaRCalculator, PositionData
from src.risk.core.correlation_tracker import CorrelationTracker


class CreditRating(Enum):
    """Credit rating categories"""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    CC = "CC"
    C = "C"
    D = "D"  # Default


class CounterpartyType(Enum):
    """Types of counterparties"""
    BANK = "BANK"
    BROKER_DEALER = "BROKER_DEALER"
    HEDGE_FUND = "HEDGE_FUND"
    CORPORATE = "CORPORATE"
    SOVEREIGN = "SOVEREIGN"
    INSURANCE = "INSURANCE"
    PENSION_FUND = "PENSION_FUND"
    MUTUAL_FUND = "MUTUAL_FUND"


class ExposureType(Enum):
    """Types of credit exposure"""
    CURRENT_EXPOSURE = "CURRENT_EXPOSURE"
    POTENTIAL_FUTURE_EXPOSURE = "POTENTIAL_FUTURE_EXPOSURE"
    EXPECTED_POSITIVE_EXPOSURE = "EXPECTED_POSITIVE_EXPOSURE"
    EFFECTIVE_EXPECTED_POSITIVE_EXPOSURE = "EFFECTIVE_EXPECTED_POSITIVE_EXPOSURE"


class CollateralType(Enum):
    """Types of collateral"""
    CASH = "CASH"
    GOVERNMENT_BONDS = "GOVERNMENT_BONDS"
    CORPORATE_BONDS = "CORPORATE_BONDS"
    EQUITY = "EQUITY"
    GOLD = "GOLD"
    OTHER = "OTHER"


@dataclass
class Counterparty:
    """Counterparty information"""
    id: str
    name: str
    counterparty_type: CounterpartyType
    credit_rating: CreditRating
    country: str
    sector: str
    established_date: datetime
    is_active: bool = True
    internal_rating: Optional[str] = None
    probability_of_default: float = 0.0
    recovery_rate: float = 0.4  # 40% default recovery rate
    
    
@dataclass
class CreditLimit:
    """Credit limit definition"""
    counterparty_id: str
    limit_type: str  # "TRADING", "SETTLEMENT", "TOTAL"
    limit_amount: float
    currency: str
    utilization: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class CreditExposure:
    """Credit exposure calculation"""
    counterparty_id: str
    exposure_type: ExposureType
    amount: float
    currency: str
    calculation_date: datetime
    maturity_date: Optional[datetime] = None
    netting_agreement: bool = False
    collateral_amount: float = 0.0
    net_exposure: float = 0.0


@dataclass
class CollateralItem:
    """Collateral item"""
    id: str
    counterparty_id: str
    collateral_type: CollateralType
    amount: float
    currency: str
    haircut_percentage: float
    market_value: float
    eligible_value: float
    last_marked: datetime
    maturity_date: Optional[datetime] = None


@dataclass
class Trade:
    """Trade information"""
    id: str
    counterparty_id: str
    instrument_type: str
    notional_amount: float
    currency: str
    trade_date: datetime
    maturity_date: datetime
    mark_to_market: float
    is_cleared: bool = False
    netting_set_id: Optional[str] = None


@dataclass
class DefaultScenario:
    """Counterparty default scenario"""
    counterparty_id: str
    default_date: datetime
    recovery_rate: float
    exposure_at_default: float
    loss_given_default: float
    collateral_recovery: float
    net_loss: float


class MockCreditRatingAgency:
    """Mock credit rating agency"""
    
    def __init__(self):
        self.ratings = {}
        self.rating_changes = []
        
    def get_rating(self, counterparty_id: str) -> CreditRating:
        """Get credit rating for counterparty"""
        return self.ratings.get(counterparty_id, CreditRating.BBB)
    
    def update_rating(self, counterparty_id: str, new_rating: CreditRating, reason: str):
        """Update credit rating"""
        old_rating = self.ratings.get(counterparty_id, CreditRating.BBB)
        self.ratings[counterparty_id] = new_rating
        
        self.rating_changes.append({
            "counterparty_id": counterparty_id,
            "old_rating": old_rating,
            "new_rating": new_rating,
            "reason": reason,
            "timestamp": datetime.now()
        })
    
    def get_probability_of_default(self, rating: CreditRating) -> float:
        """Get probability of default for rating"""
        pd_mapping = {
            CreditRating.AAA: 0.0001,
            CreditRating.AA: 0.0005,
            CreditRating.A: 0.0010,
            CreditRating.BBB: 0.0025,
            CreditRating.BB: 0.0100,
            CreditRating.B: 0.0300,
            CreditRating.CCC: 0.1000,
            CreditRating.CC: 0.2000,
            CreditRating.C: 0.5000,
            CreditRating.D: 1.0000
        }
        return pd_mapping.get(rating, 0.01)


class CounterpartyRiskManager:
    """Counterparty risk management system"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.counterparties: Dict[str, Counterparty] = {}
        self.credit_limits: Dict[str, List[CreditLimit]] = defaultdict(list)
        self.exposures: Dict[str, List[CreditExposure]] = defaultdict(list)
        self.collateral: Dict[str, List[CollateralItem]] = defaultdict(list)
        self.trades: Dict[str, List[Trade]] = defaultdict(list)
        self.netting_agreements: Dict[str, List[str]] = defaultdict(list)  # netting_set_id -> trade_ids
        self.rating_agency = MockCreditRatingAgency()
        
        # Risk parameters
        self.confidence_level = 0.95
        self.time_horizon_days = 10
        self.simulation_paths = 10000
        
    def add_counterparty(self, counterparty: Counterparty):
        """Add counterparty to system"""
        self.counterparties[counterparty.id] = counterparty
        
        # Initialize credit limits
        self.credit_limits[counterparty.id] = [
            CreditLimit(
                counterparty_id=counterparty.id,
                limit_type="TRADING",
                limit_amount=self._calculate_initial_limit(counterparty),
                currency="USD"
            )
        ]
        
        # Update PD from rating
        counterparty.probability_of_default = self.rating_agency.get_probability_of_default(
            counterparty.credit_rating
        )
    
    def _calculate_initial_limit(self, counterparty: Counterparty) -> float:
        """Calculate initial credit limit based on rating"""
        base_limits = {
            CreditRating.AAA: 50000000,  # $50M
            CreditRating.AA: 30000000,   # $30M
            CreditRating.A: 20000000,    # $20M
            CreditRating.BBB: 10000000,  # $10M
            CreditRating.BB: 5000000,    # $5M
            CreditRating.B: 2000000,     # $2M
            CreditRating.CCC: 1000000,   # $1M
            CreditRating.CC: 500000,     # $500K
            CreditRating.C: 100000,      # $100K
            CreditRating.D: 0            # $0
        }
        
        base_limit = base_limits.get(counterparty.credit_rating, 1000000)
        
        # Adjust for counterparty type
        type_multipliers = {
            CounterpartyType.BANK: 1.5,
            CounterpartyType.BROKER_DEALER: 1.2,
            CounterpartyType.SOVEREIGN: 2.0,
            CounterpartyType.INSURANCE: 1.1,
            CounterpartyType.HEDGE_FUND: 0.8,
            CounterpartyType.CORPORATE: 1.0,
            CounterpartyType.PENSION_FUND: 1.3,
            CounterpartyType.MUTUAL_FUND: 1.0
        }
        
        multiplier = type_multipliers.get(counterparty.counterparty_type, 1.0)
        return base_limit * multiplier
    
    def add_trade(self, trade: Trade):
        """Add trade to system"""
        self.trades[trade.counterparty_id].append(trade)
        
        # Update exposure
        self._update_exposure(trade.counterparty_id)
    
    def _update_exposure(self, counterparty_id: str):
        """Update exposure calculations for counterparty"""
        trades = self.trades.get(counterparty_id, [])
        
        if not trades:
            return
        
        # Calculate current exposure (mark-to-market)
        current_exposure = sum(max(0, trade.mark_to_market) for trade in trades)
        
        # Calculate potential future exposure (simplified Monte Carlo)
        pfe = self._calculate_potential_future_exposure(trades)
        
        # Create exposure records
        self.exposures[counterparty_id] = [
            CreditExposure(
                counterparty_id=counterparty_id,
                exposure_type=ExposureType.CURRENT_EXPOSURE,
                amount=current_exposure,
                currency="USD",
                calculation_date=datetime.now(),
                netting_agreement=self._has_netting_agreement(counterparty_id),
                collateral_amount=self._get_collateral_value(counterparty_id),
                net_exposure=max(0, current_exposure - self._get_collateral_value(counterparty_id))
            ),
            CreditExposure(
                counterparty_id=counterparty_id,
                exposure_type=ExposureType.POTENTIAL_FUTURE_EXPOSURE,
                amount=pfe,
                currency="USD",
                calculation_date=datetime.now(),
                netting_agreement=self._has_netting_agreement(counterparty_id),
                collateral_amount=self._get_collateral_value(counterparty_id),
                net_exposure=max(0, pfe - self._get_collateral_value(counterparty_id))
            )
        ]
    
    def _calculate_potential_future_exposure(self, trades: List[Trade]) -> float:
        """Calculate potential future exposure using Monte Carlo simulation"""
        if not trades:
            return 0.0
        
        # Simplified PFE calculation
        total_notional = sum(trade.notional_amount for trade in trades)
        
        # Simulate price movements
        np.random.seed(42)  # For reproducible tests
        
        # Assume 20% annual volatility, scaled to time horizon
        volatility = 0.20 * np.sqrt(self.time_horizon_days / 365)
        
        # Generate random price movements
        price_changes = np.random.normal(0, volatility, self.simulation_paths)
        
        # Calculate exposures for each path
        exposures = []
        for change in price_changes:
            path_exposure = 0
            for trade in trades:
                # Simplified: exposure = notional * price_change (for derivatives)
                trade_exposure = trade.notional_amount * max(0, change)
                path_exposure += trade_exposure
            exposures.append(path_exposure)
        
        # Return 95th percentile as PFE
        return np.percentile(exposures, 95)
    
    def _has_netting_agreement(self, counterparty_id: str) -> bool:
        """Check if netting agreement exists"""
        trades = self.trades.get(counterparty_id, [])
        return any(trade.netting_set_id for trade in trades)
    
    def _get_collateral_value(self, counterparty_id: str) -> float:
        """Get total eligible collateral value"""
        collateral_items = self.collateral.get(counterparty_id, [])
        return sum(item.eligible_value for item in collateral_items)
    
    def add_collateral(self, collateral_item: CollateralItem):
        """Add collateral item"""
        # Calculate eligible value with haircut
        collateral_item.eligible_value = collateral_item.market_value * (1 - collateral_item.haircut_percentage)
        
        self.collateral[collateral_item.counterparty_id].append(collateral_item)
        
        # Update exposure after collateral addition
        self._update_exposure(collateral_item.counterparty_id)
    
    def calculate_concentration_risk(self) -> Dict[str, float]:
        """Calculate concentration risk metrics"""
        total_exposure = 0
        counterparty_exposures = {}
        
        for counterparty_id, exposures in self.exposures.items():
            for exposure in exposures:
                if exposure.exposure_type == ExposureType.CURRENT_EXPOSURE:
                    counterparty_exposures[counterparty_id] = exposure.net_exposure
                    total_exposure += exposure.net_exposure
        
        # Calculate concentration ratios
        concentration_ratios = {}
        for counterparty_id, exposure in counterparty_exposures.items():
            concentration_ratios[counterparty_id] = exposure / total_exposure if total_exposure > 0 else 0
        
        return concentration_ratios
    
    def simulate_default_scenario(self, counterparty_id: str) -> DefaultScenario:
        """Simulate counterparty default scenario"""
        counterparty = self.counterparties.get(counterparty_id)
        if not counterparty:
            raise ValueError(f"Counterparty {counterparty_id} not found")
        
        # Get current exposure
        current_exposures = [e for e in self.exposures.get(counterparty_id, []) 
                           if e.exposure_type == ExposureType.CURRENT_EXPOSURE]
        
        exposure_at_default = current_exposures[0].net_exposure if current_exposures else 0.0
        
        # Calculate loss given default
        recovery_rate = counterparty.recovery_rate
        loss_given_default = exposure_at_default * (1 - recovery_rate)
        
        # Calculate collateral recovery
        collateral_recovery = self._get_collateral_value(counterparty_id) * 0.8  # 80% collateral recovery
        
        # Net loss
        net_loss = max(0, loss_given_default - collateral_recovery)
        
        return DefaultScenario(
            counterparty_id=counterparty_id,
            default_date=datetime.now(),
            recovery_rate=recovery_rate,
            exposure_at_default=exposure_at_default,
            loss_given_default=loss_given_default,
            collateral_recovery=collateral_recovery,
            net_loss=net_loss
        )
    
    def get_credit_utilization(self, counterparty_id: str) -> Dict[str, float]:
        """Get credit limit utilization"""
        limits = self.credit_limits.get(counterparty_id, [])
        exposures = self.exposures.get(counterparty_id, [])
        
        utilization = {}
        
        for limit in limits:
            current_exposure = 0
            for exposure in exposures:
                if exposure.exposure_type == ExposureType.CURRENT_EXPOSURE:
                    current_exposure = exposure.net_exposure
                    break
            
            utilization[limit.limit_type] = current_exposure / limit.limit_amount if limit.limit_amount > 0 else 0
        
        return utilization


class TestCounterpartyRisk:
    """Comprehensive counterparty risk testing suite"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def risk_manager(self, event_bus):
        """Create counterparty risk manager"""
        return CounterpartyRiskManager(event_bus)
    
    @pytest.fixture
    def sample_counterparties(self):
        """Create sample counterparties for testing"""
        return [
            Counterparty(
                id="CP001",
                name="Goldman Sachs",
                counterparty_type=CounterpartyType.BANK,
                credit_rating=CreditRating.A,
                country="USA",
                sector="FINANCIAL_SERVICES",
                established_date=datetime(1869, 1, 1)
            ),
            Counterparty(
                id="CP002",
                name="Bridgewater Associates",
                counterparty_type=CounterpartyType.HEDGE_FUND,
                credit_rating=CreditRating.BBB,
                country="USA",
                sector="ASSET_MANAGEMENT",
                established_date=datetime(1975, 1, 1)
            ),
            Counterparty(
                id="CP003",
                name="Apple Inc",
                counterparty_type=CounterpartyType.CORPORATE,
                credit_rating=CreditRating.AA,
                country="USA",
                sector="TECHNOLOGY",
                established_date=datetime(1976, 4, 1)
            ),
            Counterparty(
                id="CP004",
                name="Deutsche Bank",
                counterparty_type=CounterpartyType.BANK,
                credit_rating=CreditRating.BBB,
                country="GERMANY",
                sector="FINANCIAL_SERVICES",
                established_date=datetime(1870, 1, 1)
            )
        ]
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing"""
        return [
            Trade(
                id="T001",
                counterparty_id="CP001",
                instrument_type="EQUITY_SWAP",
                notional_amount=10000000,
                currency="USD",
                trade_date=datetime.now() - timedelta(days=30),
                maturity_date=datetime.now() + timedelta(days=90),
                mark_to_market=150000,
                netting_set_id="NET001"
            ),
            Trade(
                id="T002",
                counterparty_id="CP001",
                instrument_type="INTEREST_RATE_SWAP",
                notional_amount=20000000,
                currency="USD",
                trade_date=datetime.now() - timedelta(days=60),
                maturity_date=datetime.now() + timedelta(days=180),
                mark_to_market=-80000,
                netting_set_id="NET001"
            ),
            Trade(
                id="T003",
                counterparty_id="CP002",
                instrument_type="EQUITY_OPTION",
                notional_amount=5000000,
                currency="USD",
                trade_date=datetime.now() - timedelta(days=15),
                maturity_date=datetime.now() + timedelta(days=45),
                mark_to_market=75000
            ),
            Trade(
                id="T004",
                counterparty_id="CP003",
                instrument_type="CREDIT_DEFAULT_SWAP",
                notional_amount=8000000,
                currency="USD",
                trade_date=datetime.now() - timedelta(days=90),
                maturity_date=datetime.now() + timedelta(days=270),
                mark_to_market=120000
            )
        ]
    
    @pytest.fixture
    def sample_collateral(self):
        """Create sample collateral items"""
        return [
            CollateralItem(
                id="COL001",
                counterparty_id="CP001",
                collateral_type=CollateralType.CASH,
                amount=500000,
                currency="USD",
                haircut_percentage=0.0,
                market_value=500000,
                eligible_value=500000,
                last_marked=datetime.now()
            ),
            CollateralItem(
                id="COL002",
                counterparty_id="CP001",
                collateral_type=CollateralType.GOVERNMENT_BONDS,
                amount=1000000,
                currency="USD",
                haircut_percentage=0.02,
                market_value=1000000,
                eligible_value=980000,
                last_marked=datetime.now()
            ),
            CollateralItem(
                id="COL003",
                counterparty_id="CP002",
                collateral_type=CollateralType.EQUITY,
                amount=300000,
                currency="USD",
                haircut_percentage=0.15,
                market_value=300000,
                eligible_value=255000,
                last_marked=datetime.now()
            )
        ]
    
    def test_counterparty_setup_and_limits(self, risk_manager, sample_counterparties):
        """Test counterparty setup and credit limit calculation"""
        
        # Add counterparties
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        # Verify counterparties were added
        assert len(risk_manager.counterparties) == len(sample_counterparties)
        
        # Test credit limit calculation
        for counterparty in sample_counterparties:
            limits = risk_manager.credit_limits[counterparty.id]
            assert len(limits) > 0, f"No limits created for {counterparty.name}"
            
            trading_limit = next((l for l in limits if l.limit_type == "TRADING"), None)
            assert trading_limit is not None, f"No trading limit for {counterparty.name}"
            
            # Verify limit is reasonable based on rating
            expected_ranges = {
                CreditRating.AA: (20000000, 60000000),
                CreditRating.A: (15000000, 40000000),
                CreditRating.BBB: (8000000, 20000000)
            }
            
            if counterparty.credit_rating in expected_ranges:
                min_limit, max_limit = expected_ranges[counterparty.credit_rating]
                assert min_limit <= trading_limit.limit_amount <= max_limit, \
                    f"Limit {trading_limit.limit_amount} outside expected range for {counterparty.credit_rating}"
        
        print("✓ Counterparty setup and limits test successful")
        print(f"  Counterparties added: {len(sample_counterparties)}")
        
        for cp in sample_counterparties:
            limit = risk_manager.credit_limits[cp.id][0]
            print(f"  {cp.name} ({cp.credit_rating.value}): ${limit.limit_amount:,.0f}")
    
    def test_exposure_calculation(self, risk_manager, sample_counterparties, sample_trades):
        """Test credit exposure calculation"""
        
        # Setup counterparties
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        # Add trades
        for trade in sample_trades:
            risk_manager.add_trade(trade)
        
        # Verify exposures were calculated
        for counterparty_id in ["CP001", "CP002", "CP003"]:
            exposures = risk_manager.exposures.get(counterparty_id, [])
            assert len(exposures) > 0, f"No exposures calculated for {counterparty_id}"
            
            # Check exposure types
            exposure_types = [e.exposure_type for e in exposures]
            assert ExposureType.CURRENT_EXPOSURE in exposure_types, "Current exposure missing"
            assert ExposureType.POTENTIAL_FUTURE_EXPOSURE in exposure_types, "PFE missing"
            
            # Verify exposure amounts are reasonable
            current_exposure = next(e for e in exposures if e.exposure_type == ExposureType.CURRENT_EXPOSURE)
            pfe = next(e for e in exposures if e.exposure_type == ExposureType.POTENTIAL_FUTURE_EXPOSURE)
            
            assert current_exposure.amount >= 0, "Current exposure should be non-negative"
            assert pfe.amount >= 0, "PFE should be non-negative"
            
            # PFE should generally be higher than current exposure
            assert pfe.amount >= current_exposure.amount, \
                f"PFE ({pfe.amount}) should be >= current exposure ({current_exposure.amount})"
        
        print("✓ Exposure calculation test successful")
        
        for counterparty_id in ["CP001", "CP002", "CP003"]:
            exposures = risk_manager.exposures.get(counterparty_id, [])
            current_exp = next(e for e in exposures if e.exposure_type == ExposureType.CURRENT_EXPOSURE)
            pfe = next(e for e in exposures if e.exposure_type == ExposureType.POTENTIAL_FUTURE_EXPOSURE)
            
            counterparty_name = risk_manager.counterparties[counterparty_id].name
            print(f"  {counterparty_name}:")
            print(f"    Current Exposure: ${current_exp.amount:,.0f}")
            print(f"    PFE: ${pfe.amount:,.0f}")
    
    def test_netting_agreements(self, risk_manager, sample_counterparties, sample_trades):
        """Test netting agreement effects on exposure"""
        
        # Setup counterparties
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        # Add trades with netting
        for trade in sample_trades:
            risk_manager.add_trade(trade)
        
        # Test netting for CP001 (has netting agreement)
        cp001_exposures = risk_manager.exposures.get("CP001", [])
        cp001_current = next(e for e in cp001_exposures if e.exposure_type == ExposureType.CURRENT_EXPOSURE)
        
        # Verify netting agreement is recognized
        assert cp001_current.netting_agreement, "Netting agreement not recognized"
        
        # Calculate expected netted exposure
        cp001_trades = risk_manager.trades["CP001"]
        gross_positive = sum(max(0, trade.mark_to_market) for trade in cp001_trades)
        gross_negative = sum(min(0, trade.mark_to_market) for trade in cp001_trades)
        expected_netted = gross_positive + gross_negative  # Net position
        
        # Current exposure should be max(0, netted_position)
        expected_current_exposure = max(0, expected_netted)
        
        assert abs(cp001_current.amount - expected_current_exposure) < 1000, \
            f"Netting calculation incorrect: {cp001_current.amount} vs {expected_current_exposure}"
        
        print("✓ Netting agreements test successful")
        print(f"  CP001 gross positive: ${gross_positive:,.0f}")
        print(f"  CP001 gross negative: ${gross_negative:,.0f}")
        print(f"  CP001 netted exposure: ${cp001_current.amount:,.0f}")
    
    def test_collateral_management(self, risk_manager, sample_counterparties, sample_trades, sample_collateral):
        """Test collateral management and exposure reduction"""
        
        # Setup counterparties and trades
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        for trade in sample_trades:
            risk_manager.add_trade(trade)
        
        # Get exposure before collateral
        cp001_exposures_before = risk_manager.exposures.get("CP001", [])
        current_exposure_before = next(e for e in cp001_exposures_before 
                                     if e.exposure_type == ExposureType.CURRENT_EXPOSURE)
        
        # Add collateral
        for collateral in sample_collateral:
            risk_manager.add_collateral(collateral)
        
        # Get exposure after collateral
        cp001_exposures_after = risk_manager.exposures.get("CP001", [])
        current_exposure_after = next(e for e in cp001_exposures_after 
                                    if e.exposure_type == ExposureType.CURRENT_EXPOSURE)
        
        # Verify collateral was added
        cp001_collateral = risk_manager.collateral["CP001"]
        assert len(cp001_collateral) == 2, "Not all collateral items added for CP001"
        
        total_collateral_value = sum(item.eligible_value for item in cp001_collateral)
        expected_collateral = 500000 + 980000  # Cash + Government bonds (after haircut)
        
        assert abs(total_collateral_value - expected_collateral) < 1000, \
            f"Collateral calculation incorrect: {total_collateral_value} vs {expected_collateral}"
        
        # Verify net exposure reduction
        assert current_exposure_after.net_exposure < current_exposure_after.amount, \
            "Net exposure not reduced by collateral"
        
        expected_net_exposure = max(0, current_exposure_after.amount - total_collateral_value)
        assert abs(current_exposure_after.net_exposure - expected_net_exposure) < 1000, \
            f"Net exposure calculation incorrect: {current_exposure_after.net_exposure} vs {expected_net_exposure}"
        
        print("✓ Collateral management test successful")
        print(f"  CP001 gross exposure: ${current_exposure_after.amount:,.0f}")
        print(f"  CP001 collateral value: ${total_collateral_value:,.0f}")
        print(f"  CP001 net exposure: ${current_exposure_after.net_exposure:,.0f}")
    
    def test_concentration_risk_assessment(self, risk_manager, sample_counterparties, sample_trades):
        """Test concentration risk assessment"""
        
        # Setup counterparties and trades
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        for trade in sample_trades:
            risk_manager.add_trade(trade)
        
        # Calculate concentration ratios
        concentration_ratios = risk_manager.calculate_concentration_risk()
        
        # Verify concentration calculation
        assert len(concentration_ratios) > 0, "No concentration ratios calculated"
        
        # Check that ratios sum to approximately 1.0
        total_ratio = sum(concentration_ratios.values())
        assert abs(total_ratio - 1.0) < 0.01, f"Concentration ratios don't sum to 1.0: {total_ratio}"
        
        # Verify all ratios are non-negative
        for counterparty_id, ratio in concentration_ratios.items():
            assert ratio >= 0, f"Negative concentration ratio for {counterparty_id}: {ratio}"
            assert ratio <= 1.0, f"Concentration ratio > 1.0 for {counterparty_id}: {ratio}"
        
        # Check for concentration alerts (>20% threshold)
        high_concentration = {cp_id: ratio for cp_id, ratio in concentration_ratios.items() 
                            if ratio > 0.20}
        
        print("✓ Concentration risk assessment test successful")
        print(f"  Counterparties analyzed: {len(concentration_ratios)}")
        
        for counterparty_id, ratio in concentration_ratios.items():
            counterparty_name = risk_manager.counterparties[counterparty_id].name
            print(f"  {counterparty_name}: {ratio:.1%}")
        
        if high_concentration:
            print(f"  High concentration counterparties: {len(high_concentration)}")
    
    def test_credit_limit_utilization(self, risk_manager, sample_counterparties, sample_trades):
        """Test credit limit utilization calculation"""
        
        # Setup counterparties and trades
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        for trade in sample_trades:
            risk_manager.add_trade(trade)
        
        # Check utilization for each counterparty
        for counterparty_id in ["CP001", "CP002", "CP003"]:
            utilization = risk_manager.get_credit_utilization(counterparty_id)
            
            assert "TRADING" in utilization, f"No trading utilization for {counterparty_id}"
            
            trading_utilization = utilization["TRADING"]
            assert 0 <= trading_utilization <= 1.0, \
                f"Invalid utilization for {counterparty_id}: {trading_utilization}"
            
            # Get actual values for validation
            limits = risk_manager.credit_limits[counterparty_id]
            trading_limit = next(l for l in limits if l.limit_type == "TRADING")
            
            exposures = risk_manager.exposures.get(counterparty_id, [])
            current_exposure = next(e for e in exposures if e.exposure_type == ExposureType.CURRENT_EXPOSURE)
            
            expected_utilization = current_exposure.net_exposure / trading_limit.limit_amount
            assert abs(trading_utilization - expected_utilization) < 0.01, \
                f"Utilization calculation incorrect for {counterparty_id}"
        
        print("✓ Credit limit utilization test successful")
        
        for counterparty_id in ["CP001", "CP002", "CP003"]:
            utilization = risk_manager.get_credit_utilization(counterparty_id)
            counterparty_name = risk_manager.counterparties[counterparty_id].name
            trading_util = utilization["TRADING"]
            
            print(f"  {counterparty_name}: {trading_util:.1%} utilization")
    
    def test_counterparty_default_scenario(self, risk_manager, sample_counterparties, sample_trades, sample_collateral):
        """Test counterparty default scenario simulation"""
        
        # Setup full scenario
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        for trade in sample_trades:
            risk_manager.add_trade(trade)
        
        for collateral in sample_collateral:
            risk_manager.add_collateral(collateral)
        
        # Test default scenario for CP001
        default_scenario = risk_manager.simulate_default_scenario("CP001")
        
        # Verify default scenario calculation
        assert default_scenario.counterparty_id == "CP001"
        assert default_scenario.exposure_at_default >= 0
        assert default_scenario.recovery_rate > 0
        assert default_scenario.loss_given_default >= 0
        assert default_scenario.collateral_recovery >= 0
        assert default_scenario.net_loss >= 0
        
        # Verify calculation logic
        counterparty = risk_manager.counterparties["CP001"]
        expected_lgd = default_scenario.exposure_at_default * (1 - counterparty.recovery_rate)
        
        assert abs(default_scenario.loss_given_default - expected_lgd) < 1000, \
            "Loss given default calculation incorrect"
        
        # Net loss should be LGD minus collateral recovery
        expected_net_loss = max(0, default_scenario.loss_given_default - default_scenario.collateral_recovery)
        assert abs(default_scenario.net_loss - expected_net_loss) < 1000, \
            "Net loss calculation incorrect"
        
        print("✓ Counterparty default scenario test successful")
        print(f"  Counterparty: {counterparty.name}")
        print(f"  Exposure at default: ${default_scenario.exposure_at_default:,.0f}")
        print(f"  Recovery rate: {default_scenario.recovery_rate:.1%}")
        print(f"  Loss given default: ${default_scenario.loss_given_default:,.0f}")
        print(f"  Collateral recovery: ${default_scenario.collateral_recovery:,.0f}")
        print(f"  Net loss: ${default_scenario.net_loss:,.0f}")
    
    def test_credit_rating_impact(self, risk_manager, sample_counterparties):
        """Test credit rating impact on limits and pricing"""
        
        # Add counterparties
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        # Test rating changes
        original_limits = {}
        for counterparty_id in ["CP001", "CP002", "CP003"]:
            limits = risk_manager.credit_limits[counterparty_id]
            original_limits[counterparty_id] = limits[0].limit_amount
        
        # Simulate rating upgrade for CP002 (BBB -> A)
        risk_manager.counterparties["CP002"].credit_rating = CreditRating.A
        risk_manager.counterparties["CP002"].probability_of_default = \
            risk_manager.rating_agency.get_probability_of_default(CreditRating.A)
        
        # Recalculate limit
        new_limit = risk_manager._calculate_initial_limit(risk_manager.counterparties["CP002"])
        
        # Verify limit increased with rating improvement
        assert new_limit > original_limits["CP002"], \
            "Credit limit should increase with rating upgrade"
        
        # Test rating downgrade for CP003 (AA -> BBB)
        risk_manager.counterparties["CP003"].credit_rating = CreditRating.BBB
        risk_manager.counterparties["CP003"].probability_of_default = \
            risk_manager.rating_agency.get_probability_of_default(CreditRating.BBB)
        
        # Recalculate limit
        downgrade_limit = risk_manager._calculate_initial_limit(risk_manager.counterparties["CP003"])
        
        # Verify limit decreased with rating downgrade
        assert downgrade_limit < original_limits["CP003"], \
            "Credit limit should decrease with rating downgrade"
        
        print("✓ Credit rating impact test successful")
        print(f"  CP002 (BBB->A): ${original_limits['CP002']:,.0f} -> ${new_limit:,.0f}")
        print(f"  CP003 (AA->BBB): ${original_limits['CP003']:,.0f} -> ${downgrade_limit:,.0f}")
    
    def test_regulatory_capital_calculation(self, risk_manager, sample_counterparties, sample_trades):
        """Test regulatory capital calculation for counterparty risk"""
        
        # Setup counterparties and trades
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        for trade in sample_trades:
            risk_manager.add_trade(trade)
        
        # Calculate regulatory capital requirements
        total_capital_required = 0
        
        for counterparty_id, counterparty in risk_manager.counterparties.items():
            exposures = risk_manager.exposures.get(counterparty_id, [])
            
            if not exposures:
                continue
            
            # Get current exposure
            current_exposure = next(
                (e for e in exposures if e.exposure_type == ExposureType.CURRENT_EXPOSURE), 
                None
            )
            
            if current_exposure:
                # Simplified regulatory capital calculation
                # Risk weight based on credit rating
                risk_weights = {
                    CreditRating.AAA: 0.20,
                    CreditRating.AA: 0.20,
                    CreditRating.A: 0.50,
                    CreditRating.BBB: 0.50,
                    CreditRating.BB: 1.00,
                    CreditRating.B: 1.00,
                    CreditRating.CCC: 1.50,
                    CreditRating.CC: 1.50,
                    CreditRating.C: 1.50,
                    CreditRating.D: 1.50
                }
                
                risk_weight = risk_weights.get(counterparty.credit_rating, 1.00)
                risk_weighted_exposure = current_exposure.net_exposure * risk_weight
                
                # Capital requirement (8% of risk-weighted exposure)
                capital_required = risk_weighted_exposure * 0.08
                total_capital_required += capital_required
        
        # Verify capital calculation
        assert total_capital_required > 0, "No regulatory capital calculated"
        
        # Capital should be reasonable percentage of total exposure
        total_exposure = sum(
            e.net_exposure for exposures in risk_manager.exposures.values()
            for e in exposures if e.exposure_type == ExposureType.CURRENT_EXPOSURE
        )
        
        capital_ratio = total_capital_required / total_exposure if total_exposure > 0 else 0
        assert 0.01 <= capital_ratio <= 0.20, f"Capital ratio unreasonable: {capital_ratio:.2%}"
        
        print("✓ Regulatory capital calculation test successful")
        print(f"  Total exposure: ${total_exposure:,.0f}")
        print(f"  Capital required: ${total_capital_required:,.0f}")
        print(f"  Capital ratio: {capital_ratio:.2%}")
    
    def test_wrong_way_risk_detection(self, risk_manager, sample_counterparties, sample_trades):
        """Test wrong-way risk detection and assessment"""
        
        # Setup counterparties and trades
        for counterparty in sample_counterparties:
            risk_manager.add_counterparty(counterparty)
        
        for trade in sample_trades:
            risk_manager.add_trade(trade)
        
        # Identify potential wrong-way risk scenarios
        wrong_way_risks = []
        
        for counterparty_id, trades in risk_manager.trades.items():
            counterparty = risk_manager.counterparties[counterparty_id]
            
            for trade in trades:
                # Check for wrong-way risk patterns
                if (counterparty.counterparty_type == CounterpartyType.BANK and 
                    trade.instrument_type in ["CREDIT_DEFAULT_SWAP", "INTEREST_RATE_SWAP"]):
                    
                    wrong_way_risks.append({
                        "counterparty_id": counterparty_id,
                        "trade_id": trade.id,
                        "risk_type": "GENERAL_WRONG_WAY",
                        "description": f"Bank counterparty with {trade.instrument_type}"
                    })
                
                # Sector-specific wrong-way risk
                if (counterparty.sector == "FINANCIAL_SERVICES" and 
                    trade.instrument_type == "CREDIT_DEFAULT_SWAP"):
                    
                    wrong_way_risks.append({
                        "counterparty_id": counterparty_id,
                        "trade_id": trade.id,
                        "risk_type": "SPECIFIC_WRONG_WAY",
                        "description": f"Financial sector CDS exposure"
                    })
        
        # Verify wrong-way risk detection
        assert len(wrong_way_risks) > 0, "No wrong-way risks detected"
        
        # Check risk types
        risk_types = [risk["risk_type"] for risk in wrong_way_risks]
        assert "GENERAL_WRONG_WAY" in risk_types, "General wrong-way risk not detected"
        
        print("✓ Wrong-way risk detection test successful")
        print(f"  Wrong-way risks identified: {len(wrong_way_risks)}")
        
        for risk in wrong_way_risks:
            counterparty_name = risk_manager.counterparties[risk["counterparty_id"]].name
            print(f"  {counterparty_name}: {risk['risk_type']} - {risk['description']}")


if __name__ == "__main__":
    """Run counterparty risk tests directly"""
    
    print("🏛️  Starting Counterparty Risk Tests...")
    print("=" * 50)
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])