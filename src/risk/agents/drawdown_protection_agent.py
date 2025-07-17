"""
Maximum Drawdown Protection and Circuit Breaker System

Implements comprehensive drawdown protection including:
- 20% maximum drawdown circuit breaker
- Multi-level protection system (daily, weekly, monthly)
- Automatic position size reduction at drawdown levels
- Emergency stop trading functionality
- Recovery procedures and validation
- Performance impact monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import structlog
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import yaml
import threading
import time

logger = structlog.get_logger()


class DrawdownLevel(Enum):
    """Drawdown severity levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ProtectionAction(Enum):
    """Protection actions to take"""
    NONE = "none"
    REDUCE_POSITIONS_50 = "reduce_positions_50"
    REDUCE_POSITIONS_75 = "reduce_positions_75"
    STOP_TRADING = "stop_trading"
    EMERGENCY_LIQUIDATION = "emergency_liquidation"


@dataclass
class DrawdownEvent:
    """Drawdown event record"""
    timestamp: datetime
    current_equity: float
    peak_equity: float
    drawdown_percent: float
    drawdown_level: DrawdownLevel
    action_taken: ProtectionAction
    positions_affected: List[str] = field(default_factory=list)
    recovery_time: Optional[datetime] = None
    notes: str = ""


@dataclass
class ProtectionMetrics:
    """Protection system metrics"""
    total_drawdown_events: int = 0
    max_drawdown_reached: float = 0.0
    protection_activations: int = 0
    trading_stops: int = 0
    emergency_liquidations: int = 0
    avg_recovery_time: float = 0.0
    positions_reduced: int = 0
    capital_preserved: float = 0.0


class DrawdownProtectionAgent:
    """
    Maximum Drawdown Protection Agent
    
    Features:
    - Real-time drawdown monitoring
    - Multi-level circuit breaker system
    - Automatic position size reduction
    - Emergency trading halt
    - Recovery procedures
    - Performance impact tracking
    """
    
    def __init__(self, config_path: str = "/home/QuantNova/GrandModel/config/risk_management_config.yaml"):
        """Initialize Drawdown Protection Agent"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.protection_config = self.config['risk_controls']['drawdown_protection']
        self.circuit_breakers = self.protection_config['circuit_breakers']
        
        # Protection parameters
        self.max_drawdown_percent = self.protection_config['max_drawdown_percent']
        self.daily_loss_limit = self.protection_config['daily_loss_limit']
        self.weekly_loss_limit = self.protection_config['weekly_loss_limit']
        self.monthly_loss_limit = self.protection_config['monthly_loss_limit']
        
        # Circuit breaker levels
        self.level_1_threshold = self.circuit_breakers['level_1']
        self.level_2_threshold = self.circuit_breakers['level_2']
        self.level_3_threshold = self.circuit_breakers['level_3']
        
        # Portfolio tracking
        self.equity_history = []
        self.daily_pnl_history = []
        self.weekly_pnl_history = []
        self.monthly_pnl_history = []
        
        # Protection state
        self.current_peak_equity = 0.0
        self.current_drawdown = 0.0
        self.protection_active = False
        self.trading_halted = False
        self.last_protection_action = None
        self.protection_start_time = None
        
        # Event tracking
        self.drawdown_events: List[DrawdownEvent] = []
        self.protection_metrics = ProtectionMetrics()
        
        # Threading for real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # Callbacks for external systems
        self.position_reduction_callback = None
        self.trading_halt_callback = None
        self.emergency_liquidation_callback = None
        
        logger.info("Drawdown Protection Agent initialized",
                   max_drawdown_percent=self.max_drawdown_percent,
                   daily_loss_limit=self.daily_loss_limit,
                   circuit_breaker_levels=[self.level_1_threshold, self.level_2_threshold, self.level_3_threshold])
    
    def start_monitoring(self, initial_equity: float):
        """Start real-time drawdown monitoring"""
        try:
            with self.lock:
                self.current_peak_equity = initial_equity
                self.equity_history.append({
                    'timestamp': datetime.now(),
                    'equity': initial_equity,
                    'drawdown': 0.0
                })
                
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
                self.monitoring_thread.daemon = True
                self.monitoring_thread.start()
                
                logger.info("Drawdown monitoring started",
                           initial_equity=initial_equity,
                           peak_equity=self.current_peak_equity)
                
        except Exception as e:
            logger.error("Error starting drawdown monitoring", error=str(e))
    
    def stop_monitoring(self):
        """Stop real-time drawdown monitoring"""
        with self.lock:
            self.monitoring_active = False
            
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("Drawdown monitoring stopped")
    
    def update_equity(self, current_equity: float, positions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Update current equity and check for drawdown protection triggers
        
        Args:
            current_equity: Current portfolio equity
            positions: Current positions dictionary (optional)
            
        Returns:
            Dictionary with drawdown status and actions taken
        """
        try:
            with self.lock:
                now = datetime.now()
                
                # Update peak equity
                if current_equity > self.current_peak_equity:
                    self.current_peak_equity = current_equity
                
                # Calculate current drawdown
                if self.current_peak_equity > 0:
                    self.current_drawdown = (self.current_peak_equity - current_equity) / self.current_peak_equity
                else:
                    self.current_drawdown = 0.0
                
                # Add to equity history
                self.equity_history.append({
                    'timestamp': now,
                    'equity': current_equity,
                    'drawdown': self.current_drawdown
                })
                
                # Keep only recent history
                if len(self.equity_history) > 10000:
                    self.equity_history = self.equity_history[-10000:]
                
                # Check protection triggers
                protection_response = self._check_protection_triggers(current_equity, positions)
                
                # Update daily/weekly/monthly PnL tracking
                self._update_pnl_tracking(current_equity, now)
                
                return protection_response
                
        except Exception as e:
            logger.error("Error updating equity", error=str(e))
            return {'status': 'error', 'error': str(e)}
    
    def _check_protection_triggers(self, current_equity: float, 
                                 positions: Dict[str, Any] = None) -> Dict[str, Any]:
        """Check if protection triggers should be activated"""
        try:
            response = {
                'drawdown_percent': self.current_drawdown,
                'peak_equity': self.current_peak_equity,
                'current_equity': current_equity,
                'protection_level': self._get_protection_level(),
                'action_taken': ProtectionAction.NONE,
                'positions_affected': [],
                'trading_halted': self.trading_halted,
                'warnings': []
            }
            
            # Check daily loss limit
            daily_loss = self._calculate_daily_loss(current_equity)
            if daily_loss > self.daily_loss_limit:
                response['warnings'].append(f"Daily loss limit exceeded: {daily_loss:.3f}")
                if not self.protection_active:
                    self._trigger_protection(current_equity, DrawdownLevel.WARNING, 
                                           ProtectionAction.REDUCE_POSITIONS_50, positions)
                    response['action_taken'] = ProtectionAction.REDUCE_POSITIONS_50
            
            # Check weekly loss limit
            weekly_loss = self._calculate_weekly_loss(current_equity)
            if weekly_loss > self.weekly_loss_limit:
                response['warnings'].append(f"Weekly loss limit exceeded: {weekly_loss:.3f}")
                if not self.protection_active:
                    self._trigger_protection(current_equity, DrawdownLevel.CRITICAL, 
                                           ProtectionAction.REDUCE_POSITIONS_75, positions)
                    response['action_taken'] = ProtectionAction.REDUCE_POSITIONS_75
            
            # Check monthly loss limit
            monthly_loss = self._calculate_monthly_loss(current_equity)
            if monthly_loss > self.monthly_loss_limit:
                response['warnings'].append(f"Monthly loss limit exceeded: {monthly_loss:.3f}")
                if not self.trading_halted:
                    self._trigger_protection(current_equity, DrawdownLevel.EMERGENCY, 
                                           ProtectionAction.STOP_TRADING, positions)
                    response['action_taken'] = ProtectionAction.STOP_TRADING
            
            # Check circuit breaker levels
            if self.current_drawdown >= self.level_3_threshold:
                # Level 3: Emergency stop
                if not self.trading_halted:
                    self._trigger_protection(current_equity, DrawdownLevel.EMERGENCY, 
                                           ProtectionAction.EMERGENCY_LIQUIDATION, positions)
                    response['action_taken'] = ProtectionAction.EMERGENCY_LIQUIDATION
            
            elif self.current_drawdown >= self.level_2_threshold:
                # Level 2: Reduce positions by 75%
                if not self.protection_active or self.last_protection_action != ProtectionAction.REDUCE_POSITIONS_75:
                    self._trigger_protection(current_equity, DrawdownLevel.CRITICAL, 
                                           ProtectionAction.REDUCE_POSITIONS_75, positions)
                    response['action_taken'] = ProtectionAction.REDUCE_POSITIONS_75
            
            elif self.current_drawdown >= self.level_1_threshold:
                # Level 1: Reduce positions by 50%
                if not self.protection_active or self.last_protection_action != ProtectionAction.REDUCE_POSITIONS_50:
                    self._trigger_protection(current_equity, DrawdownLevel.WARNING, 
                                           ProtectionAction.REDUCE_POSITIONS_50, positions)
                    response['action_taken'] = ProtectionAction.REDUCE_POSITIONS_50
            
            # Check for recovery
            if self.protection_active and self.current_drawdown < self.level_1_threshold * 0.8:
                self._check_recovery_conditions(current_equity)
            
            return response
            
        except Exception as e:
            logger.error("Error checking protection triggers", error=str(e))
            return {'status': 'error', 'error': str(e)}
    
    def _trigger_protection(self, current_equity: float, level: DrawdownLevel, 
                          action: ProtectionAction, positions: Dict[str, Any] = None):
        """Trigger protection action"""
        try:
            now = datetime.now()
            
            # Create drawdown event
            event = DrawdownEvent(
                timestamp=now,
                current_equity=current_equity,
                peak_equity=self.current_peak_equity,
                drawdown_percent=self.current_drawdown,
                drawdown_level=level,
                action_taken=action,
                positions_affected=list(positions.keys()) if positions else []
            )
            
            # Execute protection action
            if action == ProtectionAction.REDUCE_POSITIONS_50:
                self._reduce_positions(0.5, positions)
                event.notes = "Reduced all positions by 50%"
                
            elif action == ProtectionAction.REDUCE_POSITIONS_75:
                self._reduce_positions(0.75, positions)
                event.notes = "Reduced all positions by 75%"
                
            elif action == ProtectionAction.STOP_TRADING:
                self._halt_trading()
                event.notes = "Trading halted due to excessive losses"
                
            elif action == ProtectionAction.EMERGENCY_LIQUIDATION:
                self._emergency_liquidation(positions)
                event.notes = "Emergency liquidation triggered"
            
            # Update state
            self.protection_active = True
            self.last_protection_action = action
            self.protection_start_time = now
            
            # Update metrics
            self.protection_metrics.total_drawdown_events += 1
            self.protection_metrics.protection_activations += 1
            if self.current_drawdown > self.protection_metrics.max_drawdown_reached:
                self.protection_metrics.max_drawdown_reached = self.current_drawdown
            
            # Store event
            self.drawdown_events.append(event)
            
            logger.critical("Drawdown protection triggered",
                           drawdown_percent=self.current_drawdown,
                           level=level.value,
                           action=action.value,
                           current_equity=current_equity,
                           peak_equity=self.current_peak_equity)
            
        except Exception as e:
            logger.error("Error triggering protection", error=str(e))
    
    def _reduce_positions(self, reduction_factor: float, positions: Dict[str, Any] = None):
        """Reduce position sizes by specified factor"""
        try:
            if self.position_reduction_callback and positions:
                reduced_positions = {}
                for symbol, position in positions.items():
                    new_size = position.get('size', 0) * (1 - reduction_factor)
                    reduced_positions[symbol] = {
                        'old_size': position.get('size', 0),
                        'new_size': new_size,
                        'reduction': reduction_factor
                    }
                
                # Call external position reduction function
                self.position_reduction_callback(reduced_positions)
                
                # Update metrics
                self.protection_metrics.positions_reduced += len(reduced_positions)
                
                logger.warning("Positions reduced",
                              reduction_factor=reduction_factor,
                              positions_affected=len(reduced_positions))
            
        except Exception as e:
            logger.error("Error reducing positions", error=str(e))
    
    def _halt_trading(self):
        """Halt all trading activities"""
        try:
            self.trading_halted = True
            
            if self.trading_halt_callback:
                self.trading_halt_callback()
            
            # Update metrics
            self.protection_metrics.trading_stops += 1
            
            logger.critical("Trading halted due to drawdown protection")
            
        except Exception as e:
            logger.error("Error halting trading", error=str(e))
    
    def _emergency_liquidation(self, positions: Dict[str, Any] = None):
        """Emergency liquidation of all positions"""
        try:
            if self.emergency_liquidation_callback and positions:
                self.emergency_liquidation_callback(positions)
            
            # Update metrics
            self.protection_metrics.emergency_liquidations += 1
            
            logger.critical("Emergency liquidation triggered",
                           positions_affected=len(positions) if positions else 0)
            
        except Exception as e:
            logger.error("Error executing emergency liquidation", error=str(e))
    
    def _check_recovery_conditions(self, current_equity: float):
        """Check if recovery conditions are met to reduce protection"""
        try:
            if not self.protection_active:
                return
            
            # Check if drawdown has improved significantly
            recovery_threshold = self.level_1_threshold * 0.8
            
            if self.current_drawdown < recovery_threshold:
                # Check if enough time has passed
                if self.protection_start_time:
                    time_in_protection = datetime.now() - self.protection_start_time
                    if time_in_protection.total_seconds() > 300:  # 5 minutes minimum
                        self._initiate_recovery(current_equity)
            
        except Exception as e:
            logger.error("Error checking recovery conditions", error=str(e))
    
    def _initiate_recovery(self, current_equity: float):
        """Initiate recovery from protection mode"""
        try:
            # Calculate recovery time
            recovery_time = datetime.now() - self.protection_start_time
            
            # Update last event with recovery time
            if self.drawdown_events:
                self.drawdown_events[-1].recovery_time = datetime.now()
            
            # Update metrics
            self.protection_metrics.avg_recovery_time = (
                (self.protection_metrics.avg_recovery_time * 
                 (self.protection_metrics.protection_activations - 1) +
                 recovery_time.total_seconds()) / self.protection_metrics.protection_activations
            )
            
            # Reset protection state
            self.protection_active = False
            self.last_protection_action = None
            self.protection_start_time = None
            
            logger.info("Recovery initiated",
                       recovery_time_seconds=recovery_time.total_seconds(),
                       current_drawdown=self.current_drawdown)
            
        except Exception as e:
            logger.error("Error initiating recovery", error=str(e))
    
    def _calculate_daily_loss(self, current_equity: float) -> float:
        """Calculate daily loss percentage"""
        try:
            today = datetime.now().date()
            
            # Find equity at start of day
            day_start_equity = None
            for entry in reversed(self.equity_history):
                if entry['timestamp'].date() < today:
                    day_start_equity = entry['equity']
                    break
            
            if day_start_equity is None:
                return 0.0
            
            if day_start_equity > 0:
                return (day_start_equity - current_equity) / day_start_equity
            
            return 0.0
            
        except Exception as e:
            logger.error("Error calculating daily loss", error=str(e))
            return 0.0
    
    def _calculate_weekly_loss(self, current_equity: float) -> float:
        """Calculate weekly loss percentage"""
        try:
            week_ago = datetime.now() - timedelta(days=7)
            
            # Find equity one week ago
            week_start_equity = None
            for entry in self.equity_history:
                if entry['timestamp'] >= week_ago:
                    week_start_equity = entry['equity']
                    break
            
            if week_start_equity is None:
                return 0.0
            
            if week_start_equity > 0:
                return (week_start_equity - current_equity) / week_start_equity
            
            return 0.0
            
        except Exception as e:
            logger.error("Error calculating weekly loss", error=str(e))
            return 0.0
    
    def _calculate_monthly_loss(self, current_equity: float) -> float:
        """Calculate monthly loss percentage"""
        try:
            month_ago = datetime.now() - timedelta(days=30)
            
            # Find equity one month ago
            month_start_equity = None
            for entry in self.equity_history:
                if entry['timestamp'] >= month_ago:
                    month_start_equity = entry['equity']
                    break
            
            if month_start_equity is None:
                return 0.0
            
            if month_start_equity > 0:
                return (month_start_equity - current_equity) / month_start_equity
            
            return 0.0
            
        except Exception as e:
            logger.error("Error calculating monthly loss", error=str(e))
            return 0.0
    
    def _update_pnl_tracking(self, current_equity: float, timestamp: datetime):
        """Update daily/weekly/monthly PnL tracking"""
        try:
            # This would be implemented to track PnL over different timeframes
            # For now, just store the basic information
            pass
            
        except Exception as e:
            logger.error("Error updating PnL tracking", error=str(e))
    
    def _get_protection_level(self) -> DrawdownLevel:
        """Get current protection level based on drawdown"""
        if self.current_drawdown >= self.level_3_threshold:
            return DrawdownLevel.EMERGENCY
        elif self.current_drawdown >= self.level_2_threshold:
            return DrawdownLevel.CRITICAL
        elif self.current_drawdown >= self.level_1_threshold:
            return DrawdownLevel.WARNING
        else:
            return DrawdownLevel.NORMAL
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                time.sleep(1)  # Check every second
                
                # Perform periodic checks
                if self.equity_history:
                    current_equity = self.equity_history[-1]['equity']
                    self._check_protection_triggers(current_equity)
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                time.sleep(5)  # Wait longer on error
    
    def register_callbacks(self, position_reduction_callback=None, 
                          trading_halt_callback=None, emergency_liquidation_callback=None):
        """Register callbacks for protection actions"""
        self.position_reduction_callback = position_reduction_callback
        self.trading_halt_callback = trading_halt_callback
        self.emergency_liquidation_callback = emergency_liquidation_callback
        
        logger.info("Protection callbacks registered")
    
    def manual_override(self, action: str, reason: str = "") -> Dict[str, Any]:
        """Manual override of protection system"""
        try:
            if action == "resume_trading":
                self.trading_halted = False
                self.protection_active = False
                logger.warning("Trading resumed manually", reason=reason)
                return {'status': 'success', 'action': 'trading_resumed'}
            
            elif action == "reset_protection":
                self.protection_active = False
                self.last_protection_action = None
                self.protection_start_time = None
                logger.warning("Protection reset manually", reason=reason)
                return {'status': 'success', 'action': 'protection_reset'}
            
            elif action == "update_peak":
                if self.equity_history:
                    self.current_peak_equity = self.equity_history[-1]['equity']
                    self.current_drawdown = 0.0
                    logger.warning("Peak equity updated manually", reason=reason)
                    return {'status': 'success', 'action': 'peak_updated'}
            
            else:
                return {'status': 'error', 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error("Error in manual override", error=str(e))
            return {'status': 'error', 'error': str(e)}
    
    def get_protection_status(self) -> Dict[str, Any]:
        """Get current protection status"""
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown_threshold': self.max_drawdown_percent,
            'protection_active': self.protection_active,
            'trading_halted': self.trading_halted,
            'protection_level': self._get_protection_level().value,
            'last_action': self.last_protection_action.value if self.last_protection_action else None,
            'peak_equity': self.current_peak_equity,
            'current_equity': self.equity_history[-1]['equity'] if self.equity_history else 0.0,
            'circuit_breaker_levels': {
                'level_1': self.level_1_threshold,
                'level_2': self.level_2_threshold,
                'level_3': self.level_3_threshold
            },
            'time_in_protection': (datetime.now() - self.protection_start_time).total_seconds() 
                                if self.protection_start_time else 0
        }
    
    def get_protection_metrics(self) -> Dict[str, Any]:
        """Get protection system metrics"""
        return {
            'total_drawdown_events': self.protection_metrics.total_drawdown_events,
            'max_drawdown_reached': self.protection_metrics.max_drawdown_reached,
            'protection_activations': self.protection_metrics.protection_activations,
            'trading_stops': self.protection_metrics.trading_stops,
            'emergency_liquidations': self.protection_metrics.emergency_liquidations,
            'avg_recovery_time_seconds': self.protection_metrics.avg_recovery_time,
            'positions_reduced': self.protection_metrics.positions_reduced,
            'capital_preserved': self.protection_metrics.capital_preserved,
            'recent_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'drawdown_percent': event.drawdown_percent,
                    'level': event.drawdown_level.value,
                    'action': event.action_taken.value,
                    'recovery_time': event.recovery_time.isoformat() if event.recovery_time else None
                }
                for event in self.drawdown_events[-10:]  # Last 10 events
            ]
        }
    
    def get_equity_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get equity history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            {
                'timestamp': entry['timestamp'].isoformat(),
                'equity': entry['equity'],
                'drawdown': entry['drawdown']
            }
            for entry in self.equity_history
            if entry['timestamp'] >= cutoff_time
        ]