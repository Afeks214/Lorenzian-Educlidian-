"""
Stop/Target Agent Performance Monitoring System

Real-time performance monitoring and analytics for the Stop/Target Agent (π₂),
providing comprehensive tracking of stop-loss and take-profit management
effectiveness, response times, and risk-adjusted performance metrics.

Features:
- Real-time performance tracking with <10ms monitoring overhead
- Stop/target effectiveness analysis
- Trailing stop performance metrics
- Risk-adjusted return calculations
- Volatility regime adaptation tracking
- Alert system for performance degradation
- Historical performance analysis

Author: Agent 3 - Stop/Target Agent Developer
Version: 1.0
"""

import logging


import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import structlog
import threading
import time

from ..agents.stop_target_agent import StopTargetAgent, StopTargetLevels, PositionContext
from ..agents.base_risk_agent import RiskState, RiskMetrics
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


class PerformanceAlert(Enum):
    """Performance alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class StopTargetMetrics:
    """Comprehensive stop/target performance metrics"""
    # Basic performance
    total_positions: int = 0
    stops_triggered: int = 0
    targets_hit: int = 0
    trailing_stops_activated: int = 0
    
    # Success rates
    stop_success_rate: float = 0.0
    target_success_rate: float = 0.0
    trailing_stop_success_rate: float = 0.0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    response_time_violations: int = 0
    
    # Financial metrics
    total_pnl: float = 0.0
    avg_trade_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Risk metrics
    avg_stop_distance_atr: float = 0.0
    avg_target_distance_atr: float = 0.0
    avg_hold_time_minutes: float = 0.0
    volatility_adaptation_accuracy: float = 0.0
    
    # System health
    calculation_errors: int = 0
    emergency_stops: int = 0
    last_update: Optional[datetime] = None


@dataclass
class TradeRecord:
    """Individual trade record for analysis"""
    trade_id: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    stop_loss_price: float
    take_profit_price: float
    exit_reason: str  # 'stop', 'target', 'manual', 'time'
    pnl: Optional[float]
    hold_time_minutes: Optional[int]
    atr_at_entry: float
    volatility_regime: str
    trailing_activated: bool
    stop_adjustments: int = 0
    
    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl is not None and self.pnl > 0
    
    def get_r_multiple(self) -> Optional[float]:
        """Calculate R-multiple (PnL / initial risk)"""
        if self.pnl is None:
            return None
        
        if self.position_size > 0:  # Long
            initial_risk = abs(self.entry_price - self.stop_loss_price)
        else:  # Short
            initial_risk = abs(self.stop_loss_price - self.entry_price)
        
        if initial_risk <= 0:
            return None
        
        return self.pnl / (abs(self.position_size) * initial_risk)


class StopTargetMonitor:
    """
    Performance monitoring system for Stop/Target Agent
    
    Provides comprehensive real-time monitoring and analytics for stop-loss
    and take-profit management effectiveness.
    """
    
    def __init__(self, agent: StopTargetAgent, event_bus: EventBus, config: Dict[str, Any]):
        """
        Initialize monitor
        
        Args:
            agent: Stop/Target agent to monitor
            event_bus: Event bus for communication
            config: Monitor configuration
        """
        self.agent = agent
        self.event_bus = event_bus
        self.config = config
        
        # Monitoring configuration
        self.update_interval_seconds = config.get('update_interval_seconds', 5.0)
        self.max_trade_history = config.get('max_trade_history', 10000)
        self.performance_window_minutes = config.get('performance_window_minutes', 60)
        self.alert_thresholds = config.get('alert_thresholds', {})
        
        # Data storage
        self.trade_records: Dict[str, TradeRecord] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.response_times: deque = deque(maxlen=1000)
        self.pnl_history: deque = deque(maxlen=1000)
        
        # Real-time metrics
        self.current_metrics = StopTargetMetrics()
        self.alert_history: List[Tuple[datetime, PerformanceAlert, str]] = []
        
        # Performance tracking
        self.monitoring_start_time = datetime.now()
        self.last_calculation_time: Optional[datetime] = None
        self.calculation_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Event subscriptions
        self._setup_event_subscriptions()
        
        self.logger = logger.bind(component="StopTargetMonitor", agent=agent.name)
        self.logger.info("Stop/Target monitor initialized")
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for monitoring"""
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        self.event_bus.subscribe(EventType.RISK_UPDATE, self._handle_risk_update)
        self.event_bus.subscribe(EventType.POSITION_CLOSE, self._handle_position_close)
        self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_stop)
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        with self.lock:
            if self.monitoring_active:
                return
            
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("Stop/Target monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        with self.lock:
            self.monitoring_active = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            self.logger.info("Stop/Target monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                self._update_metrics()
                self._check_performance_alerts()
                
                # Calculate monitoring overhead
                monitoring_overhead = (time.time() - start_time) * 1000
                if monitoring_overhead > 5.0:  # 5ms threshold
                    self.logger.warning("High monitoring overhead",
                                      overhead_ms=monitoring_overhead)
                
                time.sleep(self.update_interval_seconds)
                
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                time.sleep(1.0)
    
    def record_trade_entry(self, trade_id: str, entry_time: datetime,
                          entry_price: float, position_size: float,
                          levels: StopTargetLevels, risk_state: RiskState):
        """
        Record new trade entry
        
        Args:
            trade_id: Unique trade identifier
            entry_time: Trade entry timestamp
            entry_price: Entry price
            position_size: Position size
            levels: Stop/target levels
            risk_state: Current risk state
        """
        with self.lock:
            # Determine volatility regime
            if risk_state.volatility_regime <= 0.25:
                vol_regime = "low"
            elif risk_state.volatility_regime <= 0.75:
                vol_regime = "medium"
            else:
                vol_regime = "high"
            
            trade_record = TradeRecord(
                trade_id=trade_id,
                entry_time=entry_time,
                exit_time=None,
                entry_price=entry_price,
                exit_price=None,
                position_size=position_size,
                stop_loss_price=levels.stop_loss_price,
                take_profit_price=levels.take_profit_price,
                exit_reason="",
                pnl=None,
                hold_time_minutes=None,
                atr_at_entry=0.0,  # Will be calculated from levels
                volatility_regime=vol_regime,
                trailing_activated=levels.trailing_stop_active
            )
            
            # Calculate ATR from levels
            if position_size > 0:  # Long
                trade_record.atr_at_entry = abs(entry_price - levels.stop_loss_price) / levels.stop_multiplier
            else:  # Short
                trade_record.atr_at_entry = abs(levels.stop_loss_price - entry_price) / levels.stop_multiplier
            
            self.trade_records[trade_id] = trade_record
            self.current_metrics.total_positions += 1
            
            self.logger.info("Trade entry recorded",
                           trade_id=trade_id,
                           entry_price=entry_price,
                           position_size=position_size)
    
    def record_trade_exit(self, trade_id: str, exit_time: datetime,
                         exit_price: float, exit_reason: str):
        """
        Record trade exit
        
        Args:
            trade_id: Trade identifier
            exit_time: Exit timestamp
            exit_price: Exit price
            exit_reason: Reason for exit ('stop', 'target', 'manual', 'time')
        """
        with self.lock:
            if trade_id not in self.trade_records:
                self.logger.warning("Trade exit recorded for unknown trade", trade_id=trade_id)
                return
            
            trade = self.trade_records[trade_id]
            trade.exit_time = exit_time
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            
            # Calculate PnL
            if trade.position_size > 0:  # Long
                trade.pnl = (exit_price - trade.entry_price) * trade.position_size
            else:  # Short
                trade.pnl = (trade.entry_price - exit_price) * abs(trade.position_size)
            
            # Calculate hold time
            trade.hold_time_minutes = int((exit_time - trade.entry_time).total_seconds() / 60)
            
            # Update metrics based on exit reason
            if exit_reason == 'stop':
                self.current_metrics.stops_triggered += 1
            elif exit_reason == 'target':
                self.current_metrics.targets_hit += 1
            
            # Add to PnL history
            self.pnl_history.append(trade.pnl)
            
            self.logger.info("Trade exit recorded",
                           trade_id=trade_id,
                           exit_price=exit_price,
                           exit_reason=exit_reason,
                           pnl=trade.pnl)
    
    def record_stop_adjustment(self, trade_id: str, new_levels: StopTargetLevels):
        """
        Record stop/target level adjustment
        
        Args:
            trade_id: Trade identifier
            new_levels: New stop/target levels
        """
        with self.lock:
            if trade_id in self.trade_records:
                trade = self.trade_records[trade_id]
                trade.stop_adjustments += 1
                trade.stop_loss_price = new_levels.stop_loss_price
                trade.take_profit_price = new_levels.take_profit_price
                
                if new_levels.trailing_stop_active and not trade.trailing_activated:
                    trade.trailing_activated = True
                    self.current_metrics.trailing_stops_activated += 1
    
    def record_response_time(self, response_time_ms: float):
        """
        Record agent response time
        
        Args:
            response_time_ms: Response time in milliseconds
        """
        with self.lock:
            self.response_times.append(response_time_ms)
            self.calculation_count += 1
            self.last_calculation_time = datetime.now()
            
            # Check for response time violations
            if response_time_ms > self.agent.max_response_time_ms:
                self.current_metrics.response_time_violations += 1
                
                if response_time_ms > 20.0:  # Critical threshold
                    self._generate_alert(
                        PerformanceAlert.CRITICAL,
                        f"Severe response time violation: {response_time_ms:.2f}ms"
                    )
    
    def _update_metrics(self):
        """Update comprehensive metrics"""
        with self.lock:
            # Calculate response time metrics
            if self.response_times:
                self.current_metrics.avg_response_time_ms = np.mean(list(self.response_times))
                self.current_metrics.max_response_time_ms = max(self.response_times)
            
            # Calculate trade performance metrics
            completed_trades = [t for t in self.trade_records.values() if t.exit_time is not None]
            
            if completed_trades:
                # Success rates
                total_completed = len(completed_trades)
                stops_hit = sum(1 for t in completed_trades if t.exit_reason == 'stop')
                targets_hit = sum(1 for t in completed_trades if t.exit_reason == 'target')
                trailing_used = sum(1 for t in completed_trades if t.trailing_activated)
                
                self.current_metrics.stop_success_rate = stops_hit / total_completed
                self.current_metrics.target_success_rate = targets_hit / total_completed
                if trailing_used > 0:
                    self.current_metrics.trailing_stop_success_rate = trailing_used / total_completed
                
                # PnL metrics
                pnls = [t.pnl for t in completed_trades if t.pnl is not None]
                if pnls:
                    self.current_metrics.total_pnl = sum(pnls)
                    self.current_metrics.avg_trade_pnl = np.mean(pnls)
                    
                    winners = [p for p in pnls if p > 0]
                    self.current_metrics.win_rate = len(winners) / len(pnls)
                    
                    # Profit factor
                    gross_profit = sum(winners) if winners else 0
                    gross_loss = abs(sum([p for p in pnls if p < 0]))
                    if gross_loss > 0:
                        self.current_metrics.profit_factor = gross_profit / gross_loss
                    
                    # Sharpe ratio (simplified)
                    if len(pnls) > 1:
                        returns_std = np.std(pnls)
                        if returns_std > 0:
                            self.current_metrics.sharpe_ratio = self.current_metrics.avg_trade_pnl / returns_std
                    
                    # Max drawdown
                    cumulative_pnl = np.cumsum(pnls)
                    running_max = np.maximum.accumulate(cumulative_pnl)
                    drawdowns = running_max - cumulative_pnl
                    self.current_metrics.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
                
                # Distance and timing metrics
                hold_times = [t.hold_time_minutes for t in completed_trades if t.hold_time_minutes is not None]
                if hold_times:
                    self.current_metrics.avg_hold_time_minutes = np.mean(hold_times)
                
                stop_distances = []
                target_distances = []
                for trade in completed_trades:
                    if trade.atr_at_entry > 0:
                        if trade.position_size > 0:  # Long
                            stop_dist = abs(trade.entry_price - trade.stop_loss_price) / trade.atr_at_entry
                            target_dist = abs(trade.take_profit_price - trade.entry_price) / trade.atr_at_entry
                        else:  # Short
                            stop_dist = abs(trade.stop_loss_price - trade.entry_price) / trade.atr_at_entry
                            target_dist = abs(trade.entry_price - trade.take_profit_price) / trade.atr_at_entry
                        
                        stop_distances.append(stop_dist)
                        target_distances.append(target_dist)
                
                if stop_distances:
                    self.current_metrics.avg_stop_distance_atr = np.mean(stop_distances)
                if target_distances:
                    self.current_metrics.avg_target_distance_atr = np.mean(target_distances)
            
            self.current_metrics.last_update = datetime.now()
            
            # Store metrics history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'metrics': self.current_metrics
            })
    
    def _check_performance_alerts(self):
        """Check for performance alerts"""
        thresholds = self.alert_thresholds
        
        # Response time alerts
        if (self.current_metrics.avg_response_time_ms > 
            thresholds.get('response_time_warning', 8.0)):
            self._generate_alert(
                PerformanceAlert.WARNING,
                f"High average response time: {self.current_metrics.avg_response_time_ms:.2f}ms"
            )
        
        # Win rate alerts
        if (self.current_metrics.win_rate < thresholds.get('win_rate_warning', 0.4) and
            len([t for t in self.trade_records.values() if t.exit_time]) >= 10):
            self._generate_alert(
                PerformanceAlert.WARNING,
                f"Low win rate: {self.current_metrics.win_rate:.2%}"
            )
        
        # Drawdown alerts
        if self.current_metrics.max_drawdown > thresholds.get('max_drawdown_critical', 0.15):
            self._generate_alert(
                PerformanceAlert.CRITICAL,
                f"High drawdown: {self.current_metrics.max_drawdown:.2%}"
            )
        
        # Stop success rate alerts
        if (self.current_metrics.stop_success_rate > thresholds.get('stop_rate_warning', 0.7) and
            self.current_metrics.stops_triggered >= 5):
            self._generate_alert(
                PerformanceAlert.WARNING,
                f"High stop rate: {self.current_metrics.stop_success_rate:.2%}"
            )
    
    def _generate_alert(self, level: PerformanceAlert, message: str):
        """
        Generate performance alert
        
        Args:
            level: Alert level
            message: Alert message
        """
        alert_time = datetime.now()
        self.alert_history.append((alert_time, level, message))
        
        # Keep only recent alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # Publish alert event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.PERFORMANCE_ALERT,
                {
                    'component': 'StopTargetAgent',
                    'level': level.value,
                    'message': message,
                    'timestamp': alert_time,
                    'metrics': self.current_metrics
                },
                'StopTargetMonitor'
            )
        )
        
        self.logger.log(
            level.value.upper(),
            f"Performance alert: {message}",
            alert_level=level.value
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Performance report dictionary
        """
        with self.lock:
            completed_trades = [t for t in self.trade_records.values() if t.exit_time is not None]
            
            report = {
                'monitoring_duration_hours': (datetime.now() - self.monitoring_start_time).total_seconds() / 3600,
                'current_metrics': self.current_metrics,
                'trade_analysis': {
                    'total_trades': len(completed_trades),
                    'active_trades': len(self.trade_records) - len(completed_trades),
                    'by_exit_reason': {},
                    'by_volatility_regime': {},
                    'r_multiples': []
                },
                'performance_trends': {
                    'recent_performance': list(self.performance_history)[-10:],
                    'pnl_trend': list(self.pnl_history)[-20:],
                    'response_time_trend': list(self.response_times)[-20:]
                },
                'alerts': {
                    'recent_alerts': self.alert_history[-10:],
                    'alert_counts': {}
                },
                'system_health': {
                    'calculation_count': self.calculation_count,
                    'last_calculation': self.last_calculation_time,
                    'response_time_violations': self.current_metrics.response_time_violations,
                    'calculation_errors': self.current_metrics.calculation_errors
                }
            }
            
            # Analyze completed trades
            if completed_trades:
                # Group by exit reason
                exit_reasons = defaultdict(int)
                for trade in completed_trades:
                    exit_reasons[trade.exit_reason] += 1
                report['trade_analysis']['by_exit_reason'] = dict(exit_reasons)
                
                # Group by volatility regime
                vol_regimes = defaultdict(int)
                for trade in completed_trades:
                    vol_regimes[trade.volatility_regime] += 1
                report['trade_analysis']['by_volatility_regime'] = dict(vol_regimes)
                
                # R-multiples
                r_multiples = [t.get_r_multiple() for t in completed_trades]
                r_multiples = [r for r in r_multiples if r is not None]
                report['trade_analysis']['r_multiples'] = r_multiples
            
            # Alert counts
            alert_counts = defaultdict(int)
            for _, level, _ in self.alert_history:
                alert_counts[level.value] += 1
            report['alerts']['alert_counts'] = dict(alert_counts)
            
            return report
    
    def _handle_position_update(self, event: Event):
        """Handle position update events"""
        try:
            data = event.data
            trade_id = data.get('trade_id')
            
            if 'response_time_ms' in data:
                self.record_response_time(data['response_time_ms'])
            
            if 'new_levels' in data and trade_id:
                self.record_stop_adjustment(trade_id, data['new_levels'])
                
        except Exception as e:
            self.logger.error("Error handling position update", error=str(e))
    
    def _handle_risk_update(self, event: Event):
        """Handle risk update events"""
        try:
            data = event.data
            if 'response_time_ms' in data:
                self.record_response_time(data['response_time_ms'])
                
        except Exception as e:
            self.logger.error("Error handling risk update", error=str(e))
    
    def _handle_position_close(self, event: Event):
        """Handle position close events"""
        try:
            data = event.data
            trade_id = data.get('trade_id')
            
            if trade_id:
                self.record_trade_exit(
                    trade_id,
                    data.get('exit_time', datetime.now()),
                    data.get('exit_price'),
                    data.get('exit_reason', 'manual')
                )
                
        except Exception as e:
            self.logger.error("Error handling position close", error=str(e))
    
    def _handle_emergency_stop(self, event: Event):
        """Handle emergency stop events"""
        try:
            self.current_metrics.emergency_stops += 1
            self._generate_alert(
                PerformanceAlert.EMERGENCY,
                f"Emergency stop executed: {event.data.get('reason', 'Unknown')}"
            )
            
        except Exception as e:
            self.logger.error("Error handling emergency stop", error=str(e))
    
    def reset_metrics(self):
        """Reset all metrics and data"""
        with self.lock:
            self.trade_records.clear()
            self.performance_history.clear()
            self.response_times.clear()
            self.pnl_history.clear()
            self.alert_history.clear()
            
            self.current_metrics = StopTargetMetrics()
            self.monitoring_start_time = datetime.now()
            self.calculation_count = 0
            
            self.logger.info("Stop/Target monitor metrics reset")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_monitoring()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')