"""
Real-Time Risk Monitoring Service for Live Execution

This service provides comprehensive risk monitoring, breach detection, and
automatic enforcement of risk controls for live trading execution.

Key Features:
- Real-time risk limit monitoring
- Automatic position closure on breaches
- Stop-loss/take-profit enforcement
- Emergency protocol activation
- Comprehensive audit trail
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import json

from src.risk.agents.stop_target_agent import StopTargetAgent
from src.risk.agents.emergency_action_system import EmergencyActionExecutor, ActionPriority
from src.risk.core.var_calculator import VaRCalculator
from src.risk.agents.real_time_risk_assessor import RealTimeRiskAssessor
from src.core.events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class RiskBreachSeverity(Enum):
    """Risk breach severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskBreach:
    """Risk breach record"""
    timestamp: datetime
    breach_type: str
    severity: RiskBreachSeverity
    description: str
    symbol: Optional[str] = None
    position_id: Optional[str] = None
    risk_value: Optional[float] = None
    risk_limit: Optional[float] = None
    action_taken: Optional[str] = None
    resolution_time: Optional[datetime] = None


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_loss_pct: float = 0.05  # 5%
    max_daily_loss_pct: float = 0.10  # 10%
    max_drawdown_pct: float = 0.15  # 15%
    max_var_pct: float = 0.03  # 3%
    max_correlation_risk: float = 0.80  # 80%
    max_leverage: float = 2.0  # 2x
    min_liquidity_ratio: float = 0.20  # 20%
    max_concentration_pct: float = 0.25  # 25%


class RiskMonitorService:
    """
    Real-Time Risk Monitoring Service
    
    Provides comprehensive risk monitoring with automatic enforcement
    of risk controls and emergency protocols for live trading.
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        
        # Risk limits
        self.risk_limits = RiskLimits(**config.get("risk_limits", {}))
        
        # Risk management components
        self.stop_target_agent = None
        self.emergency_action_executor = None
        self.var_calculator = None
        self.real_time_risk_assessor = None
        
        # Current state
        self.running = False
        self.risk_breaches: List[RiskBreach] = []
        self.current_positions: Dict[str, Any] = {}
        self.portfolio_value: float = 0.0
        self.daily_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        
        # Monitoring intervals
        self.position_check_interval = config.get("position_check_interval", 1.0)  # seconds
        self.risk_check_interval = config.get("risk_check_interval", 0.5)  # seconds
        self.var_check_interval = config.get("var_check_interval", 5.0)  # seconds
        
        # Performance tracking
        self.checks_performed = 0
        self.breaches_detected = 0
        self.actions_taken = 0
        self.false_alarms = 0
        
        # Emergency protocols
        self.emergency_active = False
        self.emergency_reasons: List[str] = []
        
        logger.info("Risk Monitor Service initialized")
    
    async def initialize(self):
        """Initialize risk monitoring service"""
        try:
            # Initialize risk management components
            self.stop_target_agent = StopTargetAgent(self.config, self.event_bus)
            self.emergency_action_executor = EmergencyActionExecutor(self.event_bus, self.config)
            self.real_time_risk_assessor = RealTimeRiskAssessor(self.config, self.event_bus)
            
            # Initialize correlation tracker and VaR calculator
            from src.risk.core.correlation_tracker import CorrelationTracker
            correlation_tracker = CorrelationTracker(self.config.get("correlation_config", {}), self.event_bus)
            self.var_calculator = VaRCalculator(correlation_tracker, self.event_bus)
            
            # Initialize agents
            await self.stop_target_agent.initialize()
            await self.real_time_risk_assessor.initialize()
            
            # Set up event subscriptions
            self._setup_event_subscriptions()
            
            logger.info("✅ Risk Monitor Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Risk Monitor Service: {e}")
            raise
    
    def _setup_event_subscriptions(self):
        """Set up event subscriptions for monitoring"""
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_breach_event)
        self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_stop)
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_var_update)
    
    async def start(self):
        """Start risk monitoring service"""
        try:
            logger.info("🚀 Starting Risk Monitor Service...")
            
            self.running = True
            
            # Start monitoring tasks
            asyncio.create_task(self._monitor_positions())
            asyncio.create_task(self._monitor_risk_limits())
            asyncio.create_task(self._monitor_var_limits())
            asyncio.create_task(self._monitor_emergency_conditions())
            asyncio.create_task(self._cleanup_old_breaches())
            
            logger.info("✅ Risk Monitor Service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Risk Monitor Service: {e}")
            raise
    
    async def stop(self):
        """Stop risk monitoring service"""
        logger.info("🛑 Stopping Risk Monitor Service...")
        
        self.running = False
        
        # Trigger emergency stop if needed
        if self.current_positions:
            await self._trigger_emergency_stop("Service shutdown")
        
        logger.info("✅ Risk Monitor Service stopped")
    
    async def _monitor_positions(self):
        """Monitor individual position risk"""
        while self.running:
            try:
                await self._check_position_risk()
                await asyncio.sleep(self.position_check_interval)
            except Exception as e:
                logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(self.position_check_interval * 2)
    
    async def _monitor_risk_limits(self):
        """Monitor overall risk limits"""
        while self.running:
            try:
                await self._check_risk_limits()
                await asyncio.sleep(self.risk_check_interval)
            except Exception as e:
                logger.error(f"Error monitoring risk limits: {e}")
                await asyncio.sleep(self.risk_check_interval * 2)
    
    async def _monitor_var_limits(self):
        """Monitor VaR limits"""
        while self.running:
            try:
                await self._check_var_limits()
                await asyncio.sleep(self.var_check_interval)
            except Exception as e:
                logger.error(f"Error monitoring VaR limits: {e}")
                await asyncio.sleep(self.var_check_interval * 2)
    
    async def _monitor_emergency_conditions(self):
        """Monitor for emergency conditions"""
        while self.running:
            try:
                await self._check_emergency_conditions()
                await asyncio.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Error monitoring emergency conditions: {e}")
                await asyncio.sleep(2.0)
    
    async def _check_position_risk(self):
        """Check individual position risk"""
        self.checks_performed += 1
        
        for symbol, position in self.current_positions.items():
            try:
                # Skip if no position
                if position.get("quantity", 0) == 0:
                    continue
                
                # Calculate position P&L percentage
                position_value = abs(position.get("quantity", 0) * position.get("current_price", 0))
                unrealized_pnl = position.get("unrealized_pnl", 0)
                
                if position_value > 0:
                    pnl_pct = unrealized_pnl / position_value
                    
                    # Check position loss limit
                    if pnl_pct < -self.risk_limits.max_position_loss_pct:
                        await self._record_breach(
                            breach_type="position_loss_limit",
                            severity=RiskBreachSeverity.HIGH,
                            description=f"Position {symbol} loss {pnl_pct:.2%} exceeds limit {self.risk_limits.max_position_loss_pct:.2%}",
                            symbol=symbol,
                            risk_value=pnl_pct,
                            risk_limit=-self.risk_limits.max_position_loss_pct
                        )
                        
                        # Take action: close position
                        await self._close_position(symbol, f"Position loss limit exceeded: {pnl_pct:.2%}")
                
                # Check if stop-loss exists
                if not position.get("has_stop_loss", False):
                    await self._record_breach(
                        breach_type="missing_stop_loss",
                        severity=RiskBreachSeverity.MEDIUM,
                        description=f"Position {symbol} missing stop-loss order",
                        symbol=symbol
                    )
                
            except Exception as e:
                logger.error(f"Error checking position risk for {symbol}: {e}")
    
    async def _check_risk_limits(self):
        """Check overall portfolio risk limits"""
        try:
            # Check daily P&L limit
            if self.portfolio_value > 0:
                daily_pnl_pct = self.daily_pnl / self.portfolio_value
                
                if daily_pnl_pct < -self.risk_limits.max_daily_loss_pct:
                    await self._record_breach(
                        breach_type="daily_loss_limit",
                        severity=RiskBreachSeverity.CRITICAL,
                        description=f"Daily loss {daily_pnl_pct:.2%} exceeds limit {self.risk_limits.max_daily_loss_pct:.2%}",
                        risk_value=daily_pnl_pct,
                        risk_limit=-self.risk_limits.max_daily_loss_pct
                    )
                    
                    # Take action: stop trading
                    await self._trigger_emergency_stop(f"Daily loss limit exceeded: {daily_pnl_pct:.2%}")
                
                # Check drawdown limit
                drawdown_pct = self.max_drawdown / self.portfolio_value
                if drawdown_pct > self.risk_limits.max_drawdown_pct:
                    await self._record_breach(
                        breach_type="drawdown_limit",
                        severity=RiskBreachSeverity.CRITICAL,
                        description=f"Drawdown {drawdown_pct:.2%} exceeds limit {self.risk_limits.max_drawdown_pct:.2%}",
                        risk_value=drawdown_pct,
                        risk_limit=self.risk_limits.max_drawdown_pct
                    )
                    
                    # Take action: reduce positions
                    await self._reduce_positions(0.5, f"Drawdown limit exceeded: {drawdown_pct:.2%}")
            
            # Check concentration risk
            await self._check_concentration_risk()
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
    
    async def _check_var_limits(self):
        """Check VaR limits"""
        try:
            if self.var_calculator:
                var_result = await self.var_calculator.calculate_var()
                
                if var_result and self.portfolio_value > 0:
                    var_pct = var_result.portfolio_var / self.portfolio_value
                    
                    if var_pct > self.risk_limits.max_var_pct:
                        await self._record_breach(
                            breach_type="var_limit",
                            severity=RiskBreachSeverity.HIGH,
                            description=f"VaR {var_pct:.2%} exceeds limit {self.risk_limits.max_var_pct:.2%}",
                            risk_value=var_pct,
                            risk_limit=self.risk_limits.max_var_pct
                        )
                        
                        # Take action: reduce positions
                        await self._reduce_positions(0.3, f"VaR limit exceeded: {var_pct:.2%}")
                        
        except Exception as e:
            logger.error(f"Error checking VaR limits: {e}")
    
    async def _check_concentration_risk(self):
        """Check position concentration risk"""
        try:
            if not self.current_positions or self.portfolio_value <= 0:
                return
            
            # Calculate position concentrations
            for symbol, position in self.current_positions.items():
                position_value = abs(position.get("quantity", 0) * position.get("current_price", 0))
                concentration_pct = position_value / self.portfolio_value
                
                if concentration_pct > self.risk_limits.max_concentration_pct:
                    await self._record_breach(
                        breach_type="concentration_risk",
                        severity=RiskBreachSeverity.MEDIUM,
                        description=f"Position {symbol} concentration {concentration_pct:.2%} exceeds limit {self.risk_limits.max_concentration_pct:.2%}",
                        symbol=symbol,
                        risk_value=concentration_pct,
                        risk_limit=self.risk_limits.max_concentration_pct
                    )
                    
                    # Take action: reduce position
                    reduction_pct = 1 - (self.risk_limits.max_concentration_pct / concentration_pct)
                    await self._reduce_position(symbol, reduction_pct, f"Concentration risk: {concentration_pct:.2%}")
                    
        except Exception as e:
            logger.error(f"Error checking concentration risk: {e}")
    
    async def _check_emergency_conditions(self):
        """Check for emergency conditions"""
        try:
            # Check for multiple recent breaches
            recent_breaches = [
                breach for breach in self.risk_breaches
                if (datetime.now() - breach.timestamp).total_seconds() < 60  # Last minute
            ]
            
            critical_breaches = [
                breach for breach in recent_breaches
                if breach.severity in [RiskBreachSeverity.CRITICAL, RiskBreachSeverity.EMERGENCY]
            ]
            
            if len(critical_breaches) >= 3:
                await self._trigger_emergency_stop("Multiple critical breaches detected")
            
            # Check for system failures
            stop_loss_failures = [
                breach for breach in recent_breaches
                if "stop_loss" in breach.breach_type and breach.severity == RiskBreachSeverity.CRITICAL
            ]
            
            if len(stop_loss_failures) >= 2:
                await self._trigger_emergency_stop("Stop-loss system failures detected")
                
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
    
    async def _record_breach(self, breach_type: str, severity: RiskBreachSeverity, 
                           description: str, symbol: str = None, position_id: str = None,
                           risk_value: float = None, risk_limit: float = None):
        """Record a risk breach"""
        breach = RiskBreach(
            timestamp=datetime.now(),
            breach_type=breach_type,
            severity=severity,
            description=description,
            symbol=symbol,
            position_id=position_id,
            risk_value=risk_value,
            risk_limit=risk_limit
        )
        
        self.risk_breaches.append(breach)
        self.breaches_detected += 1
        
        # Log breach
        logger.warning(f"⚠️ Risk breach detected: {breach_type} - {description}")
        
        # Publish breach event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.RISK_BREACH,
                {
                    "breach_type": breach_type,
                    "severity": severity.value,
                    "description": description,
                    "symbol": symbol,
                    "risk_value": risk_value,
                    "risk_limit": risk_limit
                },
                "RiskMonitorService"
            )
        )
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a specific position"""
        try:
            logger.info(f"🔒 Closing position {symbol}: {reason}")
            
            # Use emergency action executor
            result = await self.emergency_action_executor.execute_reduce_position(
                reduction_percentage=1.0,  # Close 100%
                priority=ActionPriority.HIGH,
                positions_to_reduce=[symbol]
            )
            
            if result.success_rate > 0.8:
                logger.info(f"✅ Position {symbol} closed successfully")
                self.actions_taken += 1
                
                # Update breach record
                for breach in reversed(self.risk_breaches):
                    if breach.symbol == symbol and not breach.action_taken:
                        breach.action_taken = f"position_closed: {reason}"
                        breach.resolution_time = datetime.now()
                        break
            else:
                logger.error(f"❌ Failed to close position {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error closing position {symbol}: {e}")
    
    async def _reduce_position(self, symbol: str, reduction_pct: float, reason: str):
        """Reduce a specific position"""
        try:
            logger.info(f"📉 Reducing position {symbol} by {reduction_pct:.1%}: {reason}")
            
            result = await self.emergency_action_executor.execute_reduce_position(
                reduction_percentage=reduction_pct,
                priority=ActionPriority.HIGH,
                positions_to_reduce=[symbol]
            )
            
            if result.success_rate > 0.8:
                logger.info(f"✅ Position {symbol} reduced successfully")
                self.actions_taken += 1
            else:
                logger.error(f"❌ Failed to reduce position {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error reducing position {symbol}: {e}")
    
    async def _reduce_positions(self, reduction_pct: float, reason: str):
        """Reduce all positions"""
        try:
            logger.info(f"📉 Reducing all positions by {reduction_pct:.1%}: {reason}")
            
            result = await self.emergency_action_executor.execute_reduce_position(
                reduction_percentage=reduction_pct,
                priority=ActionPriority.HIGH
            )
            
            if result.success_rate > 0.8:
                logger.info("✅ All positions reduced successfully")
                self.actions_taken += 1
            else:
                logger.error("❌ Failed to reduce all positions")
                
        except Exception as e:
            logger.error(f"❌ Error reducing all positions: {e}")
    
    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        if self.emergency_active:
            return  # Already active
        
        try:
            logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
            
            self.emergency_active = True
            self.emergency_reasons.append(reason)
            
            # Execute emergency stop
            result = await self.emergency_action_executor.execute_close_all(
                priority=ActionPriority.EMERGENCY
            )
            
            if result.success_rate > 0.8:
                logger.info("✅ Emergency stop completed successfully")
            else:
                logger.error("❌ Emergency stop failed")
            
            self.actions_taken += 1
            
            # Record emergency breach
            await self._record_breach(
                breach_type="emergency_stop",
                severity=RiskBreachSeverity.EMERGENCY,
                description=f"Emergency stop triggered: {reason}",
                risk_value=1.0,
                risk_limit=0.0
            )
            
        except Exception as e:
            logger.error(f"❌ Emergency stop failed: {e}")
    
    async def _cleanup_old_breaches(self):
        """Clean up old breach records"""
        while self.running:
            try:
                # Remove breaches older than 24 hours
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.risk_breaches = [
                    breach for breach in self.risk_breaches
                    if breach.timestamp > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old breaches: {e}")
                await asyncio.sleep(3600)
    
    async def _handle_position_update(self, event: Event):
        """Handle position update events"""
        try:
            position_data = event.payload
            
            if hasattr(position_data, 'positions'):
                # Full portfolio update
                self.current_positions = {}
                total_value = 0.0
                
                for pos in position_data.positions:
                    self.current_positions[pos.symbol] = {
                        "quantity": pos.quantity,
                        "current_price": pos.current_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "has_stop_loss": hasattr(pos, 'has_stop_loss') and pos.has_stop_loss
                    }
                    total_value += abs(pos.quantity * pos.current_price)
                
                self.portfolio_value = total_value
                
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    async def _handle_risk_breach_event(self, event: Event):
        """Handle risk breach events from other components"""
        try:
            breach_data = event.payload
            
            # Record external breach
            await self._record_breach(
                breach_type=breach_data.get("type", "external_breach"),
                severity=RiskBreachSeverity.HIGH,
                description=breach_data.get("description", "External risk breach"),
                symbol=breach_data.get("symbol"),
                risk_value=breach_data.get("risk_value"),
                risk_limit=breach_data.get("risk_limit")
            )
            
        except Exception as e:
            logger.error(f"Error handling risk breach event: {e}")
    
    async def _handle_emergency_stop(self, event: Event):
        """Handle emergency stop events"""
        try:
            reason = event.payload.get("reason", "External emergency stop")
            await self._trigger_emergency_stop(reason)
            
        except Exception as e:
            logger.error(f"Error handling emergency stop event: {e}")
    
    async def _handle_var_update(self, event: Event):
        """Handle VaR update events"""
        try:
            # VaR monitoring is handled in _check_var_limits
            pass
            
        except Exception as e:
            logger.error(f"Error handling VaR update event: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk monitoring status"""
        return {
            "running": self.running,
            "emergency_active": self.emergency_active,
            "portfolio_value": self.portfolio_value,
            "daily_pnl": self.daily_pnl,
            "max_drawdown": self.max_drawdown,
            "checks_performed": self.checks_performed,
            "breaches_detected": self.breaches_detected,
            "actions_taken": self.actions_taken,
            "false_alarms": self.false_alarms,
            "active_positions": len(self.current_positions),
            "recent_breaches": len([
                breach for breach in self.risk_breaches
                if (datetime.now() - breach.timestamp).total_seconds() < 3600  # Last hour
            ]),
            "risk_limits": {
                "max_position_loss_pct": self.risk_limits.max_position_loss_pct,
                "max_daily_loss_pct": self.risk_limits.max_daily_loss_pct,
                "max_var_pct": self.risk_limits.max_var_pct,
                "max_concentration_pct": self.risk_limits.max_concentration_pct
            }
        }
    
    def get_breach_summary(self) -> Dict[str, Any]:
        """Get summary of risk breaches"""
        breach_types = {}
        severity_counts = {}
        
        for breach in self.risk_breaches:
            breach_types[breach.breach_type] = breach_types.get(breach.breach_type, 0) + 1
            severity_counts[breach.severity.value] = severity_counts.get(breach.severity.value, 0) + 1
        
        return {
            "total_breaches": len(self.risk_breaches),
            "breach_types": breach_types,
            "severity_counts": severity_counts,
            "recent_breaches": [
                {
                    "timestamp": breach.timestamp.isoformat(),
                    "type": breach.breach_type,
                    "severity": breach.severity.value,
                    "description": breach.description,
                    "symbol": breach.symbol,
                    "action_taken": breach.action_taken
                }
                for breach in self.risk_breaches[-10:]  # Last 10 breaches
            ]
        }