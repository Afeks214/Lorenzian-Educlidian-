"""
Real-time Risk Monitoring Testing Suite

This comprehensive test suite validates real-time position monitoring,
risk limit checking, alert systems, and automated risk reduction protocols.

Key Test Areas:
1. Real-time position monitoring and tracking
2. Risk limit checking and breach detection
3. Alert systems and notification protocols
4. Automated risk reduction and position management
5. Performance under high-frequency updates
6. Integration with trading systems and order management
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from collections import deque
import json

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationTracker, CorrelationRegime
from src.risk.core.var_calculator import VaRCalculator, VaRResult, PositionData
from src.risk.agents.real_time_risk_assessor import RealTimeRiskAssessor
from src.risk.agents.position_sizing_agent import PositionSizingAgent
from src.risk.agents.emergency_action_system import EmergencyActionSystem


class RiskLimitType(Enum):
    """Types of risk limits"""
    POSITION_LIMIT = "POSITION_LIMIT"
    VAR_LIMIT = "VAR_LIMIT"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    SECTOR_LIMIT = "SECTOR_LIMIT"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class RiskLimit:
    """Risk limit definition"""
    limit_type: RiskLimitType
    symbol: Optional[str] = None
    sector: Optional[str] = None
    limit_value: float = 0.0
    current_value: float = 0.0
    utilization: float = 0.0
    breach_threshold: float = 0.95  # 95% utilization triggers warning
    hard_limit_threshold: float = 1.0  # 100% triggers hard limit
    enabled: bool = True


@dataclass
class RiskAlert:
    """Risk alert definition"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    limit_type: RiskLimitType
    symbol: Optional[str]
    message: str
    current_value: float
    limit_value: float
    utilization: float
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    timestamp: datetime
    portfolio_value: float
    total_var: float
    leverage: float
    concentration: Dict[str, float]
    drawdown: float
    sector_exposures: Dict[str, float]
    position_count: int
    alerts_count: int


class MockTradingSystem:
    """Mock trading system for testing"""
    
    def __init__(self):
        self.positions = {}
        self.orders = []
        self.trades = []
        self.enabled = True
        
    async def reduce_position(self, symbol: str, reduction_pct: float) -> bool:
        """Simulate position reduction"""
        if not self.enabled:
            return False
            
        if symbol in self.positions:
            original_qty = self.positions[symbol]['quantity']
            new_qty = original_qty * (1 - reduction_pct)
            self.positions[symbol]['quantity'] = new_qty
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'action': 'REDUCE',
                'original_qty': original_qty,
                'new_qty': new_qty,
                'reduction_pct': reduction_pct,
                'timestamp': datetime.now()
            })
            
            return True
        return False
    
    async def halt_trading(self) -> bool:
        """Simulate trading halt"""
        self.enabled = False
        return True


class TestRealTimeRiskMonitoring:
    """Comprehensive real-time risk monitoring test suite"""
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def correlation_tracker(self, event_bus):
        """Create correlation tracker for testing"""
        tracker = CorrelationTracker(
            event_bus=event_bus,
            ewma_lambda=0.94,
            shock_threshold=0.3,
            shock_window_minutes=5
        )
        
        # Initialize with test assets
        test_assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ']
        tracker.initialize_assets(test_assets)
        
        return tracker
    
    @pytest.fixture
    def var_calculator(self, correlation_tracker, event_bus):
        """Create VaR calculator for testing"""
        return VaRCalculator(
            correlation_tracker=correlation_tracker,
            event_bus=event_bus,
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 5]
        )
    
    @pytest.fixture
    def risk_assessor(self, var_calculator, event_bus):
        """Create real-time risk assessor"""
        assessor = RealTimeRiskAssessor(
            var_calculator=var_calculator,
            event_bus=event_bus,
            update_frequency_seconds=1.0
        )
        
        # Setup risk limits
        assessor.setup_risk_limits({
            'position_limits': {
                'AAPL': 1000000,  # $1M position limit
                'GOOGL': 800000,
                'MSFT': 1200000,
                'TSLA': 500000,
                'NVDA': 600000
            },
            'portfolio_var_limit': 50000,     # $50K daily VaR
            'leverage_limit': 3.0,            # 3x leverage
            'concentration_limit': 0.20,      # 20% max position
            'drawdown_limit': 0.05,           # 5% max drawdown
            'sector_limits': {
                'TECH': 0.60,                 # 60% max tech exposure
                'CONSUMER': 0.30
            }
        })
        
        return assessor
    
    @pytest.fixture
    def position_sizing_agent(self, event_bus):
        """Create position sizing agent"""
        return PositionSizingAgent(
            event_bus=event_bus,
            max_position_size=0.20,  # 20% max position
            var_target=0.02          # 2% VaR target
        )
    
    @pytest.fixture
    def emergency_system(self, event_bus):
        """Create emergency action system"""
        return EmergencyActionSystem(
            event_bus=event_bus,
            auto_halt_threshold=0.10,    # 10% portfolio loss
            position_reduction_threshold=0.05  # 5% portfolio loss
        )
    
    @pytest.fixture
    def mock_trading_system(self):
        """Create mock trading system"""
        return MockTradingSystem()
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio for testing"""
        return {
            'AAPL': PositionData('AAPL', 1000, 180000, 180.0, 0.25),
            'GOOGL': PositionData('GOOGL', 500, 150000, 300.0, 0.30),
            'MSFT': PositionData('MSFT', 800, 240000, 300.0, 0.22),
            'TSLA': PositionData('TSLA', 200, 100000, 500.0, 0.45),
            'NVDA': PositionData('NVDA', 300, 180000, 600.0, 0.40)
        }
    
    def _setup_portfolio(self, risk_assessor, positions):
        """Helper to setup portfolio in risk assessor"""
        risk_assessor.positions = positions
        risk_assessor.portfolio_value = sum(pos.market_value for pos in positions.values())
        
        # Send position update event
        event = Event(
            event_type=EventType.POSITION_UPDATE,
            timestamp=datetime.now(),
            payload=Mock(positions=list(positions.values())),
            source='TestSetup'
        )
        risk_assessor.event_bus.publish(event)
    
    def _simulate_price_updates(self, event_bus, assets, volatility_multiplier=1.0):
        """Simulate real-time price updates"""
        base_prices = {asset: 100.0 for asset in assets}
        
        for asset in assets:
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.01 * volatility_multiplier)
            new_price = base_prices[asset] * (1 + price_change)
            
            # Create price update event
            bar_data = Mock()
            bar_data.symbol = asset
            bar_data.timestamp = datetime.now()
            bar_data.close = new_price
            bar_data.volume = np.random.randint(100000, 1000000)
            
            event = Event(
                event_type=EventType.NEW_5MIN_BAR,
                timestamp=datetime.now(),
                payload=bar_data,
                source='MarketData'
            )
            
            event_bus.publish(event)
            base_prices[asset] = new_price
    
    @pytest.mark.asyncio
    async def test_position_limit_monitoring(self, risk_assessor, sample_portfolio):
        """Test real-time position limit monitoring"""
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Initialize risk monitoring
        await risk_assessor.initialize_monitoring()
        
        # Test position within limits
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        assert risk_metrics is not None
        
        # Simulate position increase that breaches limit
        risk_assessor.positions['AAPL'].market_value = 1100000  # Exceed $1M limit
        risk_assessor.positions['AAPL'].quantity = 6111  # Adjust quantity
        
        # Trigger risk assessment
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Check for limit breach detection
        alerts = await risk_assessor.get_active_alerts()
        position_alerts = [a for a in alerts if a.limit_type == RiskLimitType.POSITION_LIMIT]
        
        assert len(position_alerts) > 0, "Position limit breach not detected"
        
        aapl_alert = next((a for a in position_alerts if a.symbol == 'AAPL'), None)
        assert aapl_alert is not None, "AAPL position limit alert not found"
        assert aapl_alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR]
        assert aapl_alert.current_value > aapl_alert.limit_value
        
        print(f"✓ Position limit monitoring successful")
        print(f"✓ AAPL position: ${aapl_alert.current_value:,.0f} (limit: ${aapl_alert.limit_value:,.0f})")
        print(f"✓ Alert severity: {aapl_alert.severity.value}")
    
    @pytest.mark.asyncio
    async def test_var_limit_monitoring(self, risk_assessor, sample_portfolio, correlation_tracker):
        """Test VaR limit monitoring and breach detection"""
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Generate some correlation data
        assets = list(sample_portfolio.keys())
        for _ in range(50):
            self._simulate_price_updates(risk_assessor.event_bus, assets)
        
        # Initialize monitoring
        await risk_assessor.initialize_monitoring()
        
        # Calculate baseline VaR
        baseline_metrics = await risk_assessor.calculate_risk_metrics()
        baseline_var = baseline_metrics.total_var
        
        # Simulate market stress to increase VaR
        correlation_tracker.simulate_correlation_shock(0.90)
        
        # Increase position volatilities
        for position in risk_assessor.positions.values():
            position.volatility *= 2.0  # Double volatility
        
        # Trigger VaR recalculation
        stressed_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Check for VaR limit breach
        alerts = await risk_assessor.get_active_alerts()
        var_alerts = [a for a in alerts if a.limit_type == RiskLimitType.VAR_LIMIT]
        
        if stressed_metrics.total_var > risk_assessor.portfolio_var_limit:
            assert len(var_alerts) > 0, "VaR limit breach not detected"
            
            var_alert = var_alerts[0]
            assert var_alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]
            assert var_alert.current_value > var_alert.limit_value
        
        print(f"✓ VaR limit monitoring successful")
        print(f"✓ Baseline VaR: ${baseline_var:,.0f}")
        print(f"✓ Stressed VaR: ${stressed_metrics.total_var:,.0f}")
        print(f"✓ VaR limit: ${risk_assessor.portfolio_var_limit:,.0f}")
    
    @pytest.mark.asyncio
    async def test_leverage_monitoring(self, risk_assessor, sample_portfolio):
        """Test leverage limit monitoring"""
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Initialize monitoring
        await risk_assessor.initialize_monitoring()
        
        # Simulate leverage increase
        risk_assessor.current_leverage = 3.5  # Exceed 3.0x limit
        
        # Trigger risk assessment
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Check for leverage limit breach
        alerts = await risk_assessor.get_active_alerts()
        leverage_alerts = [a for a in alerts if a.limit_type == RiskLimitType.LEVERAGE_LIMIT]
        
        assert len(leverage_alerts) > 0, "Leverage limit breach not detected"
        
        leverage_alert = leverage_alerts[0]
        assert leverage_alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR]
        assert leverage_alert.current_value > leverage_alert.limit_value
        
        print(f"✓ Leverage monitoring successful")
        print(f"✓ Current leverage: {leverage_alert.current_value:.1f}x")
        print(f"✓ Leverage limit: {leverage_alert.limit_value:.1f}x")
    
    @pytest.mark.asyncio
    async def test_concentration_limit_monitoring(self, risk_assessor, sample_portfolio):
        """Test concentration limit monitoring"""
        
        # Setup portfolio with one large position
        concentrated_portfolio = sample_portfolio.copy()
        concentrated_portfolio['AAPL'].market_value = 500000  # 50% of portfolio
        concentrated_portfolio['AAPL'].quantity = 2778
        
        self._setup_portfolio(risk_assessor, concentrated_portfolio)
        
        # Initialize monitoring
        await risk_assessor.initialize_monitoring()
        
        # Trigger risk assessment
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Check for concentration limit breach
        alerts = await risk_assessor.get_active_alerts()
        concentration_alerts = [a for a in alerts if a.limit_type == RiskLimitType.CONCENTRATION_LIMIT]
        
        # Calculate actual concentration
        total_value = sum(pos.market_value for pos in concentrated_portfolio.values())
        aapl_concentration = concentrated_portfolio['AAPL'].market_value / total_value
        
        if aapl_concentration > risk_assessor.concentration_limit:
            assert len(concentration_alerts) > 0, "Concentration limit breach not detected"
            
            conc_alert = concentration_alerts[0]
            assert conc_alert.symbol == 'AAPL'
            assert conc_alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR]
        
        print(f"✓ Concentration monitoring successful")
        print(f"✓ AAPL concentration: {aapl_concentration:.1%}")
        print(f"✓ Concentration limit: {risk_assessor.concentration_limit:.1%}")
    
    @pytest.mark.asyncio
    async def test_alert_system_integration(self, risk_assessor, sample_portfolio, event_bus):
        """Test alert system integration and notification"""
        
        # Setup alert listener
        alerts_received = []
        
        def alert_listener(event):
            if event.event_type == EventType.RISK_ALERT:
                alerts_received.append(event.payload)
        
        event_bus.subscribe(EventType.RISK_ALERT, alert_listener)
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Initialize monitoring
        await risk_assessor.initialize_monitoring()
        
        # Trigger multiple limit breaches
        risk_assessor.positions['AAPL'].market_value = 1100000  # Position limit breach
        risk_assessor.current_leverage = 3.5  # Leverage limit breach
        
        # Trigger risk assessment
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Wait for alerts to be processed
        await asyncio.sleep(0.1)
        
        # Verify alerts were published
        assert len(alerts_received) > 0, "No alerts received through event system"
        
        # Check alert content
        alert_types = [alert.limit_type for alert in alerts_received]
        assert RiskLimitType.POSITION_LIMIT in alert_types, "Position limit alert not published"
        assert RiskLimitType.LEVERAGE_LIMIT in alert_types, "Leverage limit alert not published"
        
        # Verify alert acknowledgment
        alerts = await risk_assessor.get_active_alerts()
        alert_id = alerts[0].alert_id
        
        success = await risk_assessor.acknowledge_alert(alert_id, "test_operator")
        assert success, "Alert acknowledgment failed"
        
        # Verify alert is marked as acknowledged
        updated_alerts = await risk_assessor.get_active_alerts()
        acknowledged_alert = next((a for a in updated_alerts if a.alert_id == alert_id), None)
        assert acknowledged_alert is not None
        assert acknowledged_alert.acknowledged, "Alert not marked as acknowledged"
        
        print(f"✓ Alert system integration successful")
        print(f"✓ Alerts received: {len(alerts_received)}")
        print(f"✓ Alert types: {[t.value for t in alert_types]}")
    
    @pytest.mark.asyncio
    async def test_automated_risk_reduction(self, risk_assessor, sample_portfolio, mock_trading_system):
        """Test automated risk reduction protocols"""
        
        # Setup trading system integration
        risk_assessor.trading_system = mock_trading_system
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Initialize monitoring with automated actions enabled
        await risk_assessor.initialize_monitoring(enable_auto_actions=True)
        
        # Simulate severe risk breach that triggers automated reduction
        risk_assessor.positions['AAPL'].market_value = 1500000  # 50% over limit
        risk_assessor.current_leverage = 4.0  # 33% over limit
        
        # Trigger risk assessment
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Wait for automated actions to process
        await asyncio.sleep(0.2)
        
        # Check if automated actions were taken
        trades = mock_trading_system.trades
        assert len(trades) > 0, "No automated trades executed"
        
        # Verify position reduction occurred
        reduction_trades = [t for t in trades if t['action'] == 'REDUCE']
        assert len(reduction_trades) > 0, "No position reduction trades"
        
        # Verify reduction was significant
        for trade in reduction_trades:
            assert trade['reduction_pct'] > 0.1, f"Insufficient reduction: {trade['reduction_pct']:.1%}"
        
        print(f"✓ Automated risk reduction successful")
        print(f"✓ Trades executed: {len(trades)}")
        print(f"✓ Reductions: {[f\"{t['symbol']}: {t['reduction_pct']:.1%}\" for t in reduction_trades]}")
    
    @pytest.mark.asyncio
    async def test_high_frequency_monitoring(self, risk_assessor, sample_portfolio, event_bus):
        """Test performance under high-frequency position updates"""
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Initialize monitoring
        await risk_assessor.initialize_monitoring()
        
        # Simulate high-frequency updates
        update_count = 100
        update_times = []
        
        for i in range(update_count):
            start_time = time.time()
            
            # Simulate position changes
            for symbol, position in risk_assessor.positions.items():
                # Small random position changes
                change_pct = np.random.uniform(-0.02, 0.02)  # ±2% change
                position.market_value *= (1 + change_pct)
                position.quantity *= (1 + change_pct)
            
            # Trigger risk assessment
            risk_metrics = await risk_assessor.calculate_risk_metrics()
            
            update_time = (time.time() - start_time) * 1000  # Convert to ms
            update_times.append(update_time)
        
        # Verify performance targets
        avg_update_time = np.mean(update_times)
        max_update_time = np.max(update_times)
        p95_update_time = np.percentile(update_times, 95)
        
        assert avg_update_time < 5.0, f"Average update time too slow: {avg_update_time:.2f}ms"
        assert p95_update_time < 10.0, f"95th percentile update time too slow: {p95_update_time:.2f}ms"
        assert max_update_time < 20.0, f"Maximum update time too slow: {max_update_time:.2f}ms"
        
        print(f"✓ High-frequency monitoring successful")
        print(f"✓ Updates processed: {update_count}")
        print(f"✓ Average update time: {avg_update_time:.2f}ms")
        print(f"✓ 95th percentile: {p95_update_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_drawdown_monitoring(self, risk_assessor, sample_portfolio):
        """Test drawdown monitoring and alerting"""
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Initialize monitoring
        await risk_assessor.initialize_monitoring()
        
        # Record initial portfolio value
        initial_value = risk_assessor.portfolio_value
        
        # Simulate portfolio decline
        decline_pct = 0.08  # 8% decline (exceeds 5% limit)
        
        for position in risk_assessor.positions.values():
            position.market_value *= (1 - decline_pct)
            position.price *= (1 - decline_pct)
        
        # Update portfolio value
        risk_assessor.portfolio_value = sum(pos.market_value for pos in risk_assessor.positions.values())
        
        # Trigger risk assessment
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Check for drawdown alert
        alerts = await risk_assessor.get_active_alerts()
        drawdown_alerts = [a for a in alerts if a.limit_type == RiskLimitType.DRAWDOWN_LIMIT]
        
        current_drawdown = (initial_value - risk_assessor.portfolio_value) / initial_value
        
        if current_drawdown > risk_assessor.drawdown_limit:
            assert len(drawdown_alerts) > 0, "Drawdown limit breach not detected"
            
            dd_alert = drawdown_alerts[0]
            assert dd_alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]
            assert dd_alert.current_value > dd_alert.limit_value
        
        print(f"✓ Drawdown monitoring successful")
        print(f"✓ Current drawdown: {current_drawdown:.1%}")
        print(f"✓ Drawdown limit: {risk_assessor.drawdown_limit:.1%}")
    
    @pytest.mark.asyncio
    async def test_sector_exposure_monitoring(self, risk_assessor, sample_portfolio):
        """Test sector exposure monitoring"""
        
        # Setup portfolio with sector classifications
        sector_portfolio = sample_portfolio.copy()
        
        # Assign sectors
        sector_map = {
            'AAPL': 'TECH',
            'GOOGL': 'TECH',
            'MSFT': 'TECH',
            'TSLA': 'CONSUMER',
            'NVDA': 'TECH'
        }
        
        risk_assessor.sector_map = sector_map
        
        # Create tech-heavy portfolio (>60% tech)
        tech_positions = ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
        for symbol in tech_positions:
            sector_portfolio[symbol].market_value *= 1.5  # Increase tech positions
        
        self._setup_portfolio(risk_assessor, sector_portfolio)
        
        # Initialize monitoring
        await risk_assessor.initialize_monitoring()
        
        # Trigger risk assessment
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Calculate tech exposure
        total_value = sum(pos.market_value for pos in sector_portfolio.values())
        tech_value = sum(pos.market_value for symbol, pos in sector_portfolio.items() 
                        if sector_map.get(symbol) == 'TECH')
        tech_exposure = tech_value / total_value
        
        # Check for sector limit breach
        alerts = await risk_assessor.get_active_alerts()
        sector_alerts = [a for a in alerts if a.limit_type == RiskLimitType.SECTOR_LIMIT]
        
        if tech_exposure > risk_assessor.sector_limits.get('TECH', 1.0):
            assert len(sector_alerts) > 0, "Sector limit breach not detected"
            
            sector_alert = next((a for a in sector_alerts if 'TECH' in a.message), None)
            assert sector_alert is not None, "Tech sector alert not found"
            assert sector_alert.severity in [AlertSeverity.WARNING, AlertSeverity.ERROR]
        
        print(f"✓ Sector exposure monitoring successful")
        print(f"✓ Tech exposure: {tech_exposure:.1%}")
        print(f"✓ Tech limit: {risk_assessor.sector_limits.get('TECH', 1.0):.1%}")
    
    @pytest.mark.asyncio
    async def test_emergency_halt_system(self, risk_assessor, sample_portfolio, mock_trading_system):
        """Test emergency trading halt system"""
        
        # Setup trading system integration
        risk_assessor.trading_system = mock_trading_system
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Initialize monitoring with emergency actions enabled
        await risk_assessor.initialize_monitoring(enable_emergency_actions=True)
        
        # Simulate catastrophic loss scenario
        loss_pct = 0.15  # 15% loss triggers emergency halt
        
        for position in risk_assessor.positions.values():
            position.market_value *= (1 - loss_pct)
            position.price *= (1 - loss_pct)
        
        # Update portfolio value
        risk_assessor.portfolio_value = sum(pos.market_value for pos in risk_assessor.positions.values())
        
        # Trigger risk assessment
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Wait for emergency actions to process
        await asyncio.sleep(0.2)
        
        # Verify trading halt was triggered
        assert not mock_trading_system.enabled, "Trading halt not triggered"
        
        # Check for emergency alerts
        alerts = await risk_assessor.get_active_alerts()
        emergency_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        
        assert len(emergency_alerts) > 0, "No emergency alerts generated"
        
        print(f"✓ Emergency halt system successful")
        print(f"✓ Portfolio loss: {loss_pct:.1%}")
        print(f"✓ Trading halted: {not mock_trading_system.enabled}")
        print(f"✓ Emergency alerts: {len(emergency_alerts)}")
    
    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, risk_assessor, sample_portfolio):
        """Test comprehensive risk metrics calculation"""
        
        # Setup portfolio
        self._setup_portfolio(risk_assessor, sample_portfolio)
        
        # Initialize monitoring
        await risk_assessor.initialize_monitoring()
        
        # Calculate risk metrics
        risk_metrics = await risk_assessor.calculate_risk_metrics()
        
        # Verify all metrics are calculated
        assert risk_metrics is not None, "Risk metrics calculation failed"
        assert risk_metrics.portfolio_value > 0, "Portfolio value not calculated"
        assert risk_metrics.total_var >= 0, "VaR not calculated"
        assert risk_metrics.leverage >= 0, "Leverage not calculated"
        assert len(risk_metrics.concentration) > 0, "Concentration not calculated"
        assert risk_metrics.position_count > 0, "Position count not calculated"
        
        # Verify concentration adds up to 1.0
        total_concentration = sum(risk_metrics.concentration.values())
        assert abs(total_concentration - 1.0) < 0.01, f"Concentration doesn't sum to 1.0: {total_concentration}"
        
        # Verify position count matches portfolio
        assert risk_metrics.position_count == len(sample_portfolio), "Position count mismatch"
        
        print(f"✓ Risk metrics calculation successful")
        print(f"✓ Portfolio value: ${risk_metrics.portfolio_value:,.0f}")
        print(f"✓ Total VaR: ${risk_metrics.total_var:,.0f}")
        print(f"✓ Leverage: {risk_metrics.leverage:.2f}x")
        print(f"✓ Position count: {risk_metrics.position_count}")


if __name__ == "__main__":
    """Run real-time risk monitoring tests directly"""
    
    print("⚡ Starting Real-time Risk Monitoring Tests...")
    print("=" * 50)
    
    # Run with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])