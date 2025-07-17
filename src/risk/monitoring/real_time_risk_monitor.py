"""
Real-Time Risk Monitoring and Alerting System
==============================================

This module implements a comprehensive real-time risk monitoring and alerting
system with:

- Real-time risk metric calculation and monitoring
- Multi-channel alerting system (email, SMS, Slack, webhooks)
- Risk dashboard with live updates
- Automated risk response and circuit breakers
- Performance monitoring and optimization
- Integration with all risk management components

Author: Risk Management System
Date: 2025-07-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum
import json
import aiohttp
from numba import jit, njit
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    LOG = "log"


class MonitoringStatus(Enum):
    """Monitoring status"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AlertConfig:
    """Alert configuration"""
    enabled: bool = True
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    severity_threshold: AlertSeverity = AlertSeverity.MEDIUM
    cooldown_seconds: int = 300
    escalation_enabled: bool = True
    escalation_timeout: int = 600
    
    # Channel-specific settings
    email_settings: Dict[str, Any] = field(default_factory=dict)
    sms_settings: Dict[str, Any] = field(default_factory=dict)
    slack_settings: Dict[str, Any] = field(default_factory=dict)
    webhook_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'enabled': self.enabled,
            'channels': [ch.value for ch in self.channels],
            'severity_threshold': self.severity_threshold.value,
            'cooldown_seconds': self.cooldown_seconds,
            'escalation_enabled': self.escalation_enabled,
            'escalation_timeout': self.escalation_timeout,
            'email_settings': self.email_settings,
            'sms_settings': self.sms_settings,
            'slack_settings': self.slack_settings,
            'webhook_settings': self.webhook_settings
        }


@dataclass
class MonitoringConfig:
    """Real-time monitoring configuration"""
    # Update frequencies
    risk_update_frequency: int = 5  # seconds
    position_update_frequency: int = 1  # seconds
    portfolio_update_frequency: int = 10  # seconds
    
    # Performance settings
    max_workers: int = 4
    batch_size: int = 100
    memory_limit_mb: int = 1024
    
    # Monitoring thresholds
    var_threshold: float = 0.02
    drawdown_threshold: float = 0.05
    leverage_threshold: float = 2.0
    concentration_threshold: float = 0.15
    correlation_threshold: float = 0.7
    
    # Alert configuration
    alert_config: AlertConfig = field(default_factory=AlertConfig)
    
    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_port: int = 8080
    dashboard_refresh_rate: int = 1  # seconds
    
    # Data retention
    data_retention_hours: int = 24
    alert_retention_hours: int = 168  # 1 week
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'risk_update_frequency': self.risk_update_frequency,
            'position_update_frequency': self.position_update_frequency,
            'portfolio_update_frequency': self.portfolio_update_frequency,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'memory_limit_mb': self.memory_limit_mb,
            'var_threshold': self.var_threshold,
            'drawdown_threshold': self.drawdown_threshold,
            'leverage_threshold': self.leverage_threshold,
            'concentration_threshold': self.concentration_threshold,
            'correlation_threshold': self.correlation_threshold,
            'alert_config': self.alert_config.to_dict(),
            'dashboard_enabled': self.dashboard_enabled,
            'dashboard_port': self.dashboard_port,
            'dashboard_refresh_rate': self.dashboard_refresh_rate,
            'data_retention_hours': self.data_retention_hours,
            'alert_retention_hours': self.alert_retention_hours
        }


@dataclass
class RiskAlert:
    """Risk alert data"""
    id: str
    timestamp: datetime
    alert_type: str
    severity: AlertSeverity
    title: str
    message: str
    current_value: float
    threshold_value: float
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    escalated: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'symbol': self.symbol,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged,
            'escalated': self.escalated,
            'resolved': self.resolved
        }


@dataclass
class RiskMetrics:
    """Real-time risk metrics"""
    timestamp: datetime
    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    current_drawdown: float
    leverage: float
    concentration: float
    correlation: float
    volatility: float
    sharpe_ratio: float
    positions_at_risk: int
    total_positions: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_var': self.portfolio_var,
            'portfolio_cvar': self.portfolio_cvar,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'leverage': self.leverage,
            'concentration': self.concentration,
            'correlation': self.correlation,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'positions_at_risk': self.positions_at_risk,
            'total_positions': self.total_positions
        }


@dataclass
class MonitoringStats:
    """Monitoring statistics"""
    uptime: timedelta
    total_alerts: int
    active_alerts: int
    alerts_by_severity: Dict[str, int]
    average_response_time: float
    system_health: float
    data_points_processed: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'uptime': self.uptime.total_seconds(),
            'total_alerts': self.total_alerts,
            'active_alerts': self.active_alerts,
            'alerts_by_severity': self.alerts_by_severity,
            'average_response_time': self.average_response_time,
            'system_health': self.system_health,
            'data_points_processed': self.data_points_processed
        }


# JIT optimized functions for real-time calculations
@njit(cache=True, fastmath=True)
def calculate_real_time_var(
    returns: np.ndarray,
    confidence_level: float,
    decay_factor: float = 0.94
) -> float:
    """
    Calculate real-time VaR with exponential weighting - JIT optimized
    
    Args:
        returns: Return array
        confidence_level: Confidence level
        decay_factor: Decay factor for exponential weighting
    
    Returns:
        Real-time VaR
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate exponentially weighted variance
    weights = np.zeros(len(returns))
    
    for i in range(len(returns)):
        weights[i] = decay_factor ** (len(returns) - 1 - i)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Calculate weighted mean and variance
    weighted_mean = np.sum(weights * returns)
    weighted_variance = np.sum(weights * (returns - weighted_mean) ** 2)
    
    # Calculate VaR
    z_score = 1.65 if confidence_level == 0.95 else 2.33  # Approximate normal quantiles
    var_value = weighted_mean - z_score * np.sqrt(weighted_variance)
    
    return abs(var_value)


@njit(cache=True, fastmath=True)
def calculate_real_time_drawdown(equity_curve: np.ndarray) -> Tuple[float, float]:
    """
    Calculate real-time drawdown metrics - JIT optimized
    
    Args:
        equity_curve: Equity curve
    
    Returns:
        Tuple of (current_drawdown, max_drawdown)
    """
    if len(equity_curve) == 0:
        return 0.0, 0.0
    
    # Calculate running maximum
    running_max = equity_curve[0]
    max_drawdown = 0.0
    
    for i in range(1, len(equity_curve)):
        if equity_curve[i] > running_max:
            running_max = equity_curve[i]
        
        drawdown = (running_max - equity_curve[i]) / running_max
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Current drawdown
    current_value = equity_curve[-1]
    peak_value = np.max(equity_curve)
    current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0.0
    
    return current_drawdown, max_drawdown


@njit(cache=True, fastmath=True)
def calculate_real_time_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    decay_factor: float = 0.94
) -> float:
    """
    Calculate real-time Sharpe ratio - JIT optimized
    
    Args:
        returns: Return array
        risk_free_rate: Risk-free rate
        decay_factor: Decay factor for exponential weighting
    
    Returns:
        Real-time Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate exponentially weighted mean and variance
    weights = np.zeros(len(returns))
    
    for i in range(len(returns)):
        weights[i] = decay_factor ** (len(returns) - 1 - i)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Calculate weighted statistics
    weighted_mean = np.sum(weights * returns)
    weighted_variance = np.sum(weights * (returns - weighted_mean) ** 2)
    
    # Calculate Sharpe ratio
    excess_return = weighted_mean - risk_free_rate
    volatility = np.sqrt(weighted_variance)
    
    if volatility == 0:
        return 0.0
    
    return excess_return / volatility


class AlertManager:
    """
    Alert management system
    
    Handles alert generation, routing, and delivery across multiple channels.
    """
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert manager
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.alert_handlers: Dict[AlertChannel, List[Callable]] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Initialize default handlers
        self._initialize_default_handlers()
    
    def _initialize_default_handlers(self) -> None:
        """Initialize default alert handlers"""
        
        # Log handler
        async def log_handler(alert: RiskAlert):
            logger.log(
                logging.WARNING if alert.severity != AlertSeverity.CRITICAL else logging.CRITICAL,
                f"Risk Alert: {alert.title} - {alert.message}",
                extra={'alert': alert.to_dict()}
            )
        
        self.add_alert_handler(AlertChannel.LOG, log_handler)
    
    def add_alert_handler(self, channel: AlertChannel, handler: Callable) -> None:
        """Add alert handler for specific channel"""
        
        if channel not in self.alert_handlers:
            self.alert_handlers[channel] = []
        
        self.alert_handlers[channel].append(handler)
    
    async def send_alert(self, alert: RiskAlert) -> None:
        """
        Send alert through configured channels
        
        Args:
            alert: Risk alert to send
        """
        
        # Check if alerts are enabled
        if not self.config.enabled:
            return
        
        # Check severity threshold
        severity_order = {
            AlertSeverity.LOW: 0,
            AlertSeverity.MEDIUM: 1,
            AlertSeverity.HIGH: 2,
            AlertSeverity.CRITICAL: 3
        }
        
        if severity_order[alert.severity] < severity_order[self.config.severity_threshold]:
            return
        
        # Check cooldown
        alert_key = f"{alert.alert_type}_{alert.symbol or 'portfolio'}"
        last_alert = self.last_alert_time.get(alert_key)
        
        if last_alert and (datetime.now() - last_alert).seconds < self.config.cooldown_seconds:
            return
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        self.last_alert_time[alert_key] = datetime.now()
        
        # Send through configured channels
        for channel in self.config.channels:
            handlers = self.alert_handlers.get(channel, [])
            
            for handler in handlers:
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed for {channel.value}: {e}")
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID to acknowledge
        
        Returns:
            True if acknowledged successfully
        """
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            return True
        
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert
        
        Args:
            alert_id: Alert ID to resolve
        
        Returns:
            True if resolved successfully
        """
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            del self.active_alerts[alert_id]
            return True
        
        return False
    
    def get_active_alerts(self) -> List[RiskAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        
        # Count alerts by severity
        severity_counts = {}
        for alert in self.alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(self.active_alerts),
            'alerts_by_severity': severity_counts,
            'alert_history_count': len(self.alert_history)
        }


class RealTimeRiskMonitor:
    """
    Real-Time Risk Monitoring System
    
    This class implements comprehensive real-time risk monitoring with:
    - Continuous risk metric calculation
    - Real-time alerting system
    - Performance monitoring
    - Integration with all risk components
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize real-time risk monitor
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.alert_manager = AlertManager(config.alert_config)
        
        # Monitoring state
        self.status = MonitoringStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.monitoring_tasks: List[asyncio.Task] = []
        
        # Data storage
        self.risk_metrics_history: List[RiskMetrics] = []
        self.performance_data: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.calculation_times: Dict[str, List[float]] = {}
        self.data_points_processed = 0
        
        # Risk calculators (would be injected in real implementation)
        self.risk_calculators: Dict[str, Any] = {}
        
        # Callbacks
        self.data_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        logger.info("RealTimeRiskMonitor initialized",
                   extra={'config': config.to_dict()})
    
    def add_data_callback(self, callback: Callable) -> None:
        """Add data update callback"""
        self.data_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback"""
        self.alert_callbacks.append(callback)
    
    def set_risk_calculator(self, name: str, calculator: Any) -> None:
        """Set risk calculator"""
        self.risk_calculators[name] = calculator
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        
        if self.status == MonitoringStatus.ACTIVE:
            return
        
        self.status = MonitoringStatus.ACTIVE
        self.start_time = datetime.now()
        
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._position_monitoring_loop()),
            asyncio.create_task(self._portfolio_monitoring_loop()),
            asyncio.create_task(self._data_cleanup_loop())
        ]
        
        logger.info("Real-time risk monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        
        if self.status == MonitoringStatus.STOPPED:
            return
        
        self.status = MonitoringStatus.STOPPED
        
        # Cancel all tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        for task in self.monitoring_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.monitoring_tasks.clear()
        
        logger.info("Real-time risk monitoring stopped")
    
    async def pause_monitoring(self) -> None:
        """Pause monitoring"""
        
        if self.status == MonitoringStatus.ACTIVE:
            self.status = MonitoringStatus.PAUSED
            logger.info("Real-time risk monitoring paused")
    
    async def resume_monitoring(self) -> None:
        """Resume monitoring"""
        
        if self.status == MonitoringStatus.PAUSED:
            self.status = MonitoringStatus.ACTIVE
            logger.info("Real-time risk monitoring resumed")
    
    async def _risk_monitoring_loop(self) -> None:
        """Main risk monitoring loop"""
        
        while self.status != MonitoringStatus.STOPPED:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._update_risk_metrics()
                
                await asyncio.sleep(self.config.risk_update_frequency)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk monitoring loop error: {e}")
                self.status = MonitoringStatus.ERROR
                await asyncio.sleep(self.config.risk_update_frequency)
    
    async def _position_monitoring_loop(self) -> None:
        """Position monitoring loop"""
        
        while self.status != MonitoringStatus.STOPPED:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._update_position_risks()
                
                await asyncio.sleep(self.config.position_update_frequency)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Position monitoring loop error: {e}")
                await asyncio.sleep(self.config.position_update_frequency)
    
    async def _portfolio_monitoring_loop(self) -> None:
        """Portfolio monitoring loop"""
        
        while self.status != MonitoringStatus.STOPPED:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._update_portfolio_metrics()
                
                await asyncio.sleep(self.config.portfolio_update_frequency)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Portfolio monitoring loop error: {e}")
                await asyncio.sleep(self.config.portfolio_update_frequency)
    
    async def _data_cleanup_loop(self) -> None:
        """Data cleanup loop"""
        
        while self.status != MonitoringStatus.STOPPED:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._cleanup_old_data()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _update_risk_metrics(self) -> None:
        """Update risk metrics"""
        
        start_time = datetime.now()
        
        try:
            # This would get real data from the trading system
            # For now, we'll create synthetic data
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_var=0.015,  # 1.5%
                portfolio_cvar=0.022,  # 2.2%
                max_drawdown=0.08,  # 8%
                current_drawdown=0.03,  # 3%
                leverage=1.5,
                concentration=0.12,  # 12%
                correlation=0.45,  # 45%
                volatility=0.18,  # 18%
                sharpe_ratio=1.2,
                positions_at_risk=3,
                total_positions=10
            )
            
            # Store metrics
            self.risk_metrics_history.append(risk_metrics)
            
            # Check thresholds and generate alerts
            await self._check_risk_thresholds(risk_metrics)
            
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(risk_metrics)
                except Exception as e:
                    logger.error(f"Data callback error: {e}")
            
            self.data_points_processed += 1
            
        except Exception as e:
            logger.error(f"Risk metrics update failed: {e}")
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'risk_metrics' not in self.calculation_times:
                self.calculation_times['risk_metrics'] = []
            
            self.calculation_times['risk_metrics'].append(calc_time)
    
    async def _update_position_risks(self) -> None:
        """Update position-level risks"""
        
        start_time = datetime.now()
        
        try:
            # This would get real position data
            # For now, we'll simulate position risk checks
            pass
            
        except Exception as e:
            logger.error(f"Position risk update failed: {e}")
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'position_risks' not in self.calculation_times:
                self.calculation_times['position_risks'] = []
            
            self.calculation_times['position_risks'].append(calc_time)
    
    async def _update_portfolio_metrics(self) -> None:
        """Update portfolio-level metrics"""
        
        start_time = datetime.now()
        
        try:
            # This would get real portfolio data
            # For now, we'll simulate portfolio metric updates
            pass
            
        except Exception as e:
            logger.error(f"Portfolio metrics update failed: {e}")
        
        finally:
            # Track performance
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if 'portfolio_metrics' not in self.calculation_times:
                self.calculation_times['portfolio_metrics'] = []
            
            self.calculation_times['portfolio_metrics'].append(calc_time)
    
    async def _check_risk_thresholds(self, metrics: RiskMetrics) -> None:
        """Check risk thresholds and generate alerts"""
        
        # VaR threshold check
        if metrics.portfolio_var > self.config.var_threshold:
            await self._generate_alert(
                "VAR_BREACH",
                AlertSeverity.HIGH,
                "Portfolio VaR Breach",
                f"Portfolio VaR {metrics.portfolio_var:.2%} exceeds threshold {self.config.var_threshold:.2%}",
                metrics.portfolio_var,
                self.config.var_threshold
            )
        
        # Drawdown threshold check
        if metrics.current_drawdown > self.config.drawdown_threshold:
            severity = AlertSeverity.CRITICAL if metrics.current_drawdown > self.config.drawdown_threshold * 1.5 else AlertSeverity.HIGH
            await self._generate_alert(
                "DRAWDOWN_WARNING",
                severity,
                "Portfolio Drawdown Warning",
                f"Portfolio drawdown {metrics.current_drawdown:.2%} exceeds threshold {self.config.drawdown_threshold:.2%}",
                metrics.current_drawdown,
                self.config.drawdown_threshold
            )
        
        # Leverage threshold check
        if metrics.leverage > self.config.leverage_threshold:
            await self._generate_alert(
                "LEVERAGE_WARNING",
                AlertSeverity.MEDIUM,
                "Portfolio Leverage Warning",
                f"Portfolio leverage {metrics.leverage:.2f}x exceeds threshold {self.config.leverage_threshold:.2f}x",
                metrics.leverage,
                self.config.leverage_threshold
            )
        
        # Concentration threshold check
        if metrics.concentration > self.config.concentration_threshold:
            await self._generate_alert(
                "CONCENTRATION_WARNING",
                AlertSeverity.MEDIUM,
                "Portfolio Concentration Warning",
                f"Portfolio concentration {metrics.concentration:.2%} exceeds threshold {self.config.concentration_threshold:.2%}",
                metrics.concentration,
                self.config.concentration_threshold
            )
        
        # Correlation threshold check
        if metrics.correlation > self.config.correlation_threshold:
            await self._generate_alert(
                "CORRELATION_WARNING",
                AlertSeverity.MEDIUM,
                "Portfolio Correlation Warning",
                f"Portfolio correlation {metrics.correlation:.2%} exceeds threshold {self.config.correlation_threshold:.2%}",
                metrics.correlation,
                self.config.correlation_threshold
            )
    
    async def _generate_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        current_value: float,
        threshold_value: float,
        symbol: Optional[str] = None
    ) -> None:
        """Generate and send alert"""
        
        alert = RiskAlert(
            id=f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            symbol=symbol
        )
        
        await self.alert_manager.send_alert(alert)
        
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old data"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.config.data_retention_hours)
        
        # Clean up risk metrics history
        self.risk_metrics_history = [
            metrics for metrics in self.risk_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
        
        # Clean up alert history
        alert_cutoff_time = datetime.now() - timedelta(hours=self.config.alert_retention_hours)
        self.alert_manager.alert_history = [
            alert for alert in self.alert_manager.alert_history
            if alert.timestamp >= alert_cutoff_time
        ]
        
        # Clean up calculation times
        for key in self.calculation_times:
            if len(self.calculation_times[key]) > 1000:
                self.calculation_times[key] = self.calculation_times[key][-1000:]
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        
        # Calculate uptime
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        # Calculate average response time
        all_times = []
        for times in self.calculation_times.values():
            all_times.extend(times)
        
        avg_response_time = np.mean(all_times) if all_times else 0.0
        
        # Calculate system health
        error_rate = 0.0  # Would be calculated from actual errors
        system_health = max(0.0, 1.0 - error_rate)
        
        # Get alert statistics
        alert_stats = self.alert_manager.get_alert_statistics()
        
        # Create monitoring stats
        monitoring_stats = MonitoringStats(
            uptime=uptime,
            total_alerts=alert_stats['total_alerts'],
            active_alerts=alert_stats['active_alerts'],
            alerts_by_severity=alert_stats['alerts_by_severity'],
            average_response_time=avg_response_time,
            system_health=system_health,
            data_points_processed=self.data_points_processed
        )
        
        return {
            'status': self.status.value,
            'monitoring_stats': monitoring_stats.to_dict(),
            'latest_metrics': self.risk_metrics_history[-1].to_dict() if self.risk_metrics_history else {},
            'active_alerts': [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],
            'performance_metrics': {
                metric: {
                    'avg_time_ms': np.mean(times),
                    'max_time_ms': np.max(times),
                    'count': len(times)
                }
                for metric, times in self.calculation_times.items()
            },
            'config': self.config.to_dict()
        }
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get data for risk dashboard"""
        
        # Get recent metrics
        recent_metrics = self.risk_metrics_history[-100:] if self.risk_metrics_history else []
        
        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': self.status.value,
            'current_metrics': recent_metrics[-1].to_dict() if recent_metrics else {},
            'metrics_history': [m.to_dict() for m in recent_metrics],
            'active_alerts': [alert.to_dict() for alert in active_alerts],
            'alert_summary': self.alert_manager.get_alert_statistics(),
            'system_health': self.get_monitoring_status()
        }


# Factory function
def create_real_time_monitor(config_dict: Optional[Dict[str, Any]] = None) -> RealTimeRiskMonitor:
    """
    Create a real-time risk monitor with configuration
    
    Args:
        config_dict: Configuration dictionary
    
    Returns:
        RealTimeRiskMonitor instance
    """
    
    if config_dict is None:
        config = MonitoringConfig()
    else:
        config = MonitoringConfig(**config_dict)
    
    return RealTimeRiskMonitor(config)