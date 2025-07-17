"""
Real-Time Risk Monitoring Dashboard

This module provides a comprehensive real-time dashboard for monitoring
risk controls, trade execution, and system health in live trading.

Key Features:
- Real-time risk metrics display
- Stop-loss/take-profit monitoring
- Risk breach alerts
- System health monitoring
- Emergency protocol status
- Performance analytics
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from src.components.live_execution_handler import LiveExecutionHandler
from src.components.risk_monitor_service import RiskMonitorService
from src.components.risk_error_handler import RiskErrorHandler
from src.core.events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class DashboardAlert(Enum):
    """Dashboard alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class RiskMetric:
    """Risk metric for dashboard display"""
    name: str
    value: float
    limit: float
    percentage: float
    status: str  # "safe", "warning", "critical"
    timestamp: datetime
    trend: str  # "up", "down", "stable"


@dataclass
class DashboardData:
    """Complete dashboard data structure"""
    timestamp: datetime
    system_status: Dict[str, Any]
    risk_metrics: List[RiskMetric]
    positions: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    performance: Dict[str, Any]
    risk_controls: Dict[str, Any]
    recent_events: List[Dict[str, Any]]


class RiskDashboard:
    """
    Real-Time Risk Monitoring Dashboard
    
    Provides comprehensive real-time monitoring of risk controls,
    trade execution, and system health for live trading.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Dashboard configuration
        self.refresh_interval = config.get("dashboard_refresh_interval", 1.0)  # seconds
        self.alert_retention_hours = config.get("alert_retention_hours", 24)
        self.max_recent_events = config.get("max_recent_events", 100)
        
        # Component references
        self.execution_handler: Optional[LiveExecutionHandler] = None
        self.risk_monitor: Optional[RiskMonitorService] = None
        self.error_handler: Optional[RiskErrorHandler] = None
        self.event_bus: Optional[EventBus] = None
        
        # Dashboard state
        self.running = False
        self.current_data: Optional[DashboardData] = None
        self.alerts: List[Dict[str, Any]] = []
        self.recent_events: List[Dict[str, Any]] = []
        self.metric_history: Dict[str, List[RiskMetric]] = {}
        
        # Performance tracking
        self.dashboard_updates = 0
        self.last_update_time = datetime.now()
        self.update_times: List[float] = []
        
        logger.info("Risk Dashboard initialized")
    
    def initialize(self, execution_handler: LiveExecutionHandler, 
                  risk_monitor: RiskMonitorService, 
                  error_handler: RiskErrorHandler,
                  event_bus: EventBus):
        """Initialize dashboard with component references"""
        self.execution_handler = execution_handler
        self.risk_monitor = risk_monitor
        self.error_handler = error_handler
        self.event_bus = event_bus
        
        # Set up event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("Risk Dashboard initialized with components")
    
    def _setup_event_subscriptions(self):
        """Set up event subscriptions for real-time updates"""
        if self.event_bus:
            self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_breach)
            self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_stop)
            self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
            self.event_bus.subscribe(EventType.ERROR, self._handle_error_event)
            self.event_bus.subscribe(EventType.TRADE_EXECUTION, self._handle_trade_execution)
    
    async def start(self):
        """Start the dashboard"""
        try:
            logger.info("🚀 Starting Risk Dashboard...")
            
            self.running = True
            
            # Start dashboard update loop
            asyncio.create_task(self._update_dashboard_loop())
            asyncio.create_task(self._cleanup_old_data())
            
            logger.info("✅ Risk Dashboard started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start Risk Dashboard: {e}")
            raise
    
    async def stop(self):
        """Stop the dashboard"""
        logger.info("🛑 Stopping Risk Dashboard...")
        self.running = False
        logger.info("✅ Risk Dashboard stopped")
    
    async def _update_dashboard_loop(self):
        """Main dashboard update loop"""
        while self.running:
            try:
                start_time = datetime.now()
                
                # Update dashboard data
                await self._update_dashboard_data()
                
                # Track performance
                update_time = (datetime.now() - start_time).total_seconds()
                self.update_times.append(update_time)
                if len(self.update_times) > 100:
                    self.update_times = self.update_times[-100:]
                
                self.dashboard_updates += 1
                self.last_update_time = datetime.now()
                
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:\n                logger.error(f"Error updating dashboard: {e}")
                await asyncio.sleep(self.refresh_interval * 2)
    
    async def _update_dashboard_data(self):
        """Update dashboard data from all components"""
        try:\n            # Get system status
            system_status = await self._get_system_status()
            
            # Get risk metrics
            risk_metrics = await self._get_risk_metrics()
            
            # Get positions
            positions = await self._get_positions()
            
            # Get alerts
            alerts = self._get_current_alerts()
            
            # Get performance data
            performance = await self._get_performance_data()
            
            # Get risk controls status
            risk_controls = await self._get_risk_controls_status()
            
            # Create dashboard data
            self.current_data = DashboardData(
                timestamp=datetime.now(),
                system_status=system_status,
                risk_metrics=risk_metrics,
                positions=positions,
                alerts=alerts,
                performance=performance,
                risk_controls=risk_controls,
                recent_events=self.recent_events[-20:]  # Last 20 events
            )
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        try:
            execution_status = self.execution_handler.get_status() if self.execution_handler else {}
            risk_monitor_status = self.risk_monitor.get_status() if self.risk_monitor else {}
            error_stats = self.error_handler.get_error_statistics() if self.error_handler else {}
            
            return {
                "execution_handler": {
                    "running": execution_status.get("running", False),
                    "broker_connected": execution_status.get("broker_connected", False),
                    "total_orders": execution_status.get("total_orders", 0),
                    "total_executions": execution_status.get("total_executions", 0),
                    "open_positions": execution_status.get("open_positions", 0)
                },
                "risk_monitor": {
                    "running": risk_monitor_status.get("running", False),
                    "emergency_active": risk_monitor_status.get("emergency_active", False),
                    "checks_performed": risk_monitor_status.get("checks_performed", 0),
                    "breaches_detected": risk_monitor_status.get("breaches_detected", 0),
                    "actions_taken": risk_monitor_status.get("actions_taken", 0)
                },
                "error_handler": {
                    "total_errors": error_stats.get("total_errors", 0),
                    "recent_errors": error_stats.get("recent_errors", 0),
                    "error_rate": error_stats.get("error_rate", 0),
                    "system_health_degraded": error_stats.get("system_health_degraded", False),
                    "trading_halted": error_stats.get("trading_halted", False)
                },
                "overall_health": self._calculate_overall_health(execution_status, risk_monitor_status, error_stats)
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def _get_risk_metrics(self) -> List[RiskMetric]:
        """Get current risk metrics"""
        try:
            metrics = []
            
            if self.execution_handler:
                status = self.execution_handler.get_status()
                risk_status = status.get("risk_status", {})
                
                # Daily P&L metric
                daily_pnl = risk_status.get("daily_pnl", 0)
                daily_pnl_limit = -5000  # $5k loss limit
                metrics.append(RiskMetric(
                    name="Daily P&L",
                    value=daily_pnl,
                    limit=daily_pnl_limit,
                    percentage=abs(daily_pnl / daily_pnl_limit) * 100 if daily_pnl_limit != 0 else 0,
                    status=self._get_metric_status(daily_pnl, daily_pnl_limit, False),
                    timestamp=datetime.now(),
                    trend=self._calculate_trend("daily_pnl", daily_pnl)
                ))
                
                # Max Drawdown metric
                max_drawdown = risk_status.get("max_drawdown", 0)
                drawdown_limit = 10000  # $10k drawdown limit
                metrics.append(RiskMetric(
                    name="Max Drawdown",
                    value=max_drawdown,
                    limit=drawdown_limit,
                    percentage=(max_drawdown / drawdown_limit) * 100 if drawdown_limit != 0 else 0,
                    status=self._get_metric_status(max_drawdown, drawdown_limit, True),
                    timestamp=datetime.now(),
                    trend=self._calculate_trend("max_drawdown", max_drawdown)
                ))
                
                # Risk Breaches metric
                risk_breaches = risk_status.get("risk_breaches", 0)
                breach_limit = 5
                metrics.append(RiskMetric(
                    name="Risk Breaches",
                    value=risk_breaches,
                    limit=breach_limit,
                    percentage=(risk_breaches / breach_limit) * 100 if breach_limit != 0 else 0,
                    status=self._get_metric_status(risk_breaches, breach_limit, True),
                    timestamp=datetime.now(),
                    trend=self._calculate_trend("risk_breaches", risk_breaches)
                ))
            
            if self.risk_monitor:
                status = self.risk_monitor.get_status()
                
                # Emergency Stops metric
                emergency_stops = status.get("emergency_stops", 0)
                emergency_limit = 1
                metrics.append(RiskMetric(
                    name="Emergency Stops",
                    value=emergency_stops,
                    limit=emergency_limit,
                    percentage=(emergency_stops / emergency_limit) * 100 if emergency_limit != 0 else 0,
                    status=self._get_metric_status(emergency_stops, emergency_limit, True),
                    timestamp=datetime.now(),
                    trend=self._calculate_trend("emergency_stops", emergency_stops)
                ))
            
            # Store metrics history
            for metric in metrics:
                if metric.name not in self.metric_history:
                    self.metric_history[metric.name] = []
                self.metric_history[metric.name].append(metric)
                
                # Keep only recent history
                if len(self.metric_history[metric.name]) > 100:
                    self.metric_history[metric.name] = self.metric_history[metric.name][-100:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return []
    
    async def _get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions with risk details"""
        try:
            if not self.execution_handler:
                return []
            
            positions = self.execution_handler.get_positions()
            position_data = []
            
            for position in positions:
                if position.quantity == 0:
                    continue
                
                # Calculate position metrics
                position_value = abs(position.quantity * position.current_price)
                pnl_pct = (position.unrealized_pnl / position_value) * 100 if position_value > 0 else 0
                
                # Check if stop-loss exists
                has_stop_loss = position.symbol in self.execution_handler.stop_loss_orders
                has_take_profit = position.symbol in self.execution_handler.take_profit_orders
                
                position_data.append({
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "side": "LONG" if position.quantity > 0 else "SHORT",
                    "current_price": position.current_price,
                    "avg_price": position.avg_price,
                    "unrealized_pnl": position.unrealized_pnl,
                    "pnl_percentage": pnl_pct,
                    "market_value": position_value,
                    "has_stop_loss": has_stop_loss,
                    "has_take_profit": has_take_profit,
                    "stop_loss_price": self.execution_handler.stop_loss_orders.get(position.symbol, {}).get("stop_price"),
                    "take_profit_price": self.execution_handler.take_profit_orders.get(position.symbol, {}).get("price"),
                    "risk_status": self._get_position_risk_status(pnl_pct, has_stop_loss),
                    "entry_time": position.entry_time.isoformat() if hasattr(position, 'entry_time') else None,
                    "last_update": position.last_update.isoformat() if hasattr(position, 'last_update') else None
                })
            
            return position_data
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def _get_current_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts"""
        try:
            # Filter recent alerts
            cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
            recent_alerts = [
                alert for alert in self.alerts
                if alert.get("timestamp", datetime.min) > cutoff_time
            ]
            
            # Sort by timestamp (newest first)
            recent_alerts.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
            
            return recent_alerts[:50]  # Return latest 50 alerts
            
        except Exception as e:
            logger.error(f"Error getting current alerts: {e}")
            return []
    
    async def _get_performance_data(self) -> Dict[str, Any]:
        """Get performance data"""
        try:
            performance = {
                "dashboard": {
                    "updates": self.dashboard_updates,
                    "last_update": self.last_update_time.isoformat(),
                    "avg_update_time_ms": (sum(self.update_times) / len(self.update_times)) * 1000 if self.update_times else 0,
                    "max_update_time_ms": max(self.update_times) * 1000 if self.update_times else 0
                }
            }
            
            if self.execution_handler:
                status = self.execution_handler.get_status()
                performance["execution"] = {
                    "avg_execution_time_ms": status.get("avg_execution_time_ms", 0),
                    "total_orders": status.get("total_orders", 0),
                    "total_executions": status.get("total_executions", 0),
                    "execution_rate": status.get("total_executions", 0) / max(1, status.get("total_orders", 1)) * 100
                }
            
            if self.risk_monitor:
                status = self.risk_monitor.get_status()
                performance["risk_monitoring"] = {
                    "checks_performed": status.get("checks_performed", 0),
                    "breaches_detected": status.get("breaches_detected", 0),
                    "actions_taken": status.get("actions_taken", 0),
                    "breach_rate": status.get("breaches_detected", 0) / max(1, status.get("checks_performed", 1)) * 100
                }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            return {}
    
    async def _get_risk_controls_status(self) -> Dict[str, Any]:
        """Get risk controls status"""
        try:
            controls = {
                "stop_loss_enforcement": False,
                "take_profit_enforcement": False,
                "position_monitoring": False,
                "var_monitoring": False,
                "emergency_protocols": False,
                "error_handling": False
            }
            
            if self.execution_handler:
                status = self.execution_handler.get_status()
                stop_loss_coverage = status.get("stop_loss_coverage", {})
                
                controls.update({
                    "stop_loss_enforcement": stop_loss_coverage.get("positions_with_stops", 0) > 0,
                    "take_profit_enforcement": status.get("risk_status", {}).get("active_target_orders", 0) > 0,
                    "position_monitoring": status.get("running", False),
                    "emergency_protocols": status.get("risk_status", {}).get("emergency_stops", 0) > 0,
                    "positions_with_stops": stop_loss_coverage.get("positions_with_stops", 0),
                    "positions_without_stops": stop_loss_coverage.get("positions_without_stops", 0),
                    "stop_loss_coverage_pct": self._calculate_stop_loss_coverage(stop_loss_coverage)
                })
            
            if self.risk_monitor:
                status = self.risk_monitor.get_status()
                controls.update({
                    "var_monitoring": status.get("running", False),
                    "real_time_monitoring": status.get("running", False)
                })
            
            if self.error_handler:
                stats = self.error_handler.get_error_statistics()
                controls.update({
                    "error_handling": not stats.get("system_health_degraded", False),
                    "system_health_ok": not stats.get("system_health_degraded", False),
                    "trading_active": not stats.get("trading_halted", False)
                })
            
            return controls
            
        except Exception as e:
            logger.error(f"Error getting risk controls status: {e}")
            return {}
    
    def _calculate_overall_health(self, execution_status: Dict, risk_monitor_status: Dict, error_stats: Dict) -> str:
        """Calculate overall system health"""
        try:
            # Health factors
            factors = []
            
            # Execution health
            if execution_status.get("running", False) and execution_status.get("broker_connected", False):
                factors.append("execution_ok")
            else:
                factors.append("execution_degraded")
            
            # Risk monitoring health
            if risk_monitor_status.get("running", False) and not risk_monitor_status.get("emergency_active", False):
                factors.append("risk_ok")
            else:
                factors.append("risk_degraded")
            
            # Error handling health
            if not error_stats.get("system_health_degraded", False) and not error_stats.get("trading_halted", False):
                factors.append("error_ok")
            else:
                factors.append("error_degraded")
            
            # Determine overall health
            if all("_ok" in factor for factor in factors):
                return "healthy"
            elif any("_degraded" in factor for factor in factors):
                return "degraded"
            else:
                return "critical"
            
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return "unknown"
    
    def _get_metric_status(self, value: float, limit: float, higher_is_worse: bool) -> str:
        """Get status for a metric based on its value and limit"""
        try:
            if higher_is_worse:
                percentage = (value / limit) * 100 if limit != 0 else 0
                if percentage >= 100:
                    return "critical"
                elif percentage >= 80:
                    return "warning"
                else:
                    return "safe"
            else:
                # Lower is worse (e.g., negative P&L)
                percentage = (abs(value) / abs(limit)) * 100 if limit != 0 else 0
                if percentage >= 100:
                    return "critical"
                elif percentage >= 80:
                    return "warning"
                else:
                    return "safe"
                    
        except Exception as e:
            logger.error(f"Error getting metric status: {e}")
            return "unknown"
    
    def _calculate_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend for a metric"""
        try:
            if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 2:
                return "stable"
            
            previous_value = self.metric_history[metric_name][-1].value
            
            if current_value > previous_value:
                return "up"
            elif current_value < previous_value:
                return "down"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return "stable"
    
    def _get_position_risk_status(self, pnl_pct: float, has_stop_loss: bool) -> str:
        """Get risk status for a position"""
        try:
            if not has_stop_loss:
                return "no_stop_loss"
            elif pnl_pct < -5:  # 5% loss
                return "high_risk"
            elif pnl_pct < -2:  # 2% loss
                return "medium_risk"
            else:
                return "low_risk"
                
        except Exception as e:
            logger.error(f"Error getting position risk status: {e}")
            return "unknown"
    
    def _calculate_stop_loss_coverage(self, stop_loss_coverage: Dict) -> float:
        """Calculate stop-loss coverage percentage"""
        try:
            with_stops = stop_loss_coverage.get("positions_with_stops", 0)
            without_stops = stop_loss_coverage.get("positions_without_stops", 0)
            total = with_stops + without_stops
            
            if total == 0:
                return 100.0
            
            return (with_stops / total) * 100
            
        except Exception as e:
            logger.error(f"Error calculating stop-loss coverage: {e}")
            return 0.0
    
    async def _cleanup_old_data(self):
        """Clean up old data periodically"""
        while self.running:
            try:
                # Clean up old alerts
                cutoff_time = datetime.now() - timedelta(hours=self.alert_retention_hours)
                self.alerts = [
                    alert for alert in self.alerts
                    if alert.get("timestamp", datetime.min) > cutoff_time
                ]
                
                # Clean up old events
                if len(self.recent_events) > self.max_recent_events:
                    self.recent_events = self.recent_events[-self.max_recent_events:]
                
                # Clean up old metric history
                for metric_name in self.metric_history:
                    if len(self.metric_history[metric_name]) > 100:
                        self.metric_history[metric_name] = self.metric_history[metric_name][-100:]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up old data: {e}")
                await asyncio.sleep(3600)
    
    def _add_alert(self, alert_type: str, level: DashboardAlert, message: str, details: Dict = None):
        """Add an alert to the dashboard"""
        try:
            alert = {
                "id": f"alert_{int(datetime.now().timestamp())}",
                "type": alert_type,
                "level": level.value,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now(),
                "resolved": False
            }
            
            self.alerts.append(alert)
            
            # Log alert
            if level == DashboardAlert.CRITICAL:
                logger.critical(f"🚨 CRITICAL ALERT: {message}")
            elif level == DashboardAlert.ERROR:
                logger.error(f"❌ ERROR ALERT: {message}")
            elif level == DashboardAlert.WARNING:
                logger.warning(f"⚠️ WARNING ALERT: {message}")
            else:
                logger.info(f"ℹ️ INFO ALERT: {message}")
                
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
    
    def _add_event(self, event_type: str, message: str, details: Dict = None):
        """Add an event to the dashboard"""
        try:
            event = {
                "id": f"event_{int(datetime.now().timestamp())}",
                "type": event_type,
                "message": message,
                "details": details or {},
                "timestamp": datetime.now()
            }
            
            self.recent_events.append(event)
            
        except Exception as e:
            logger.error(f"Error adding event: {e}")
    
    # Event handlers
    async def _handle_risk_breach(self, event: Event):
        """Handle risk breach events"""
        try:
            breach_data = event.payload
            
            self._add_alert(
                alert_type="risk_breach",
                level=DashboardAlert.ERROR,
                message=f"Risk breach: {breach_data.get('type', 'unknown')}",
                details=breach_data
            )
            
            self._add_event(
                event_type="risk_breach",
                message=f"Risk breach detected: {breach_data.get('description', 'unknown')}",
                details=breach_data
            )
            
        except Exception as e:
            logger.error(f"Error handling risk breach event: {e}")
    
    async def _handle_emergency_stop(self, event: Event):
        """Handle emergency stop events"""
        try:
            emergency_data = event.payload
            
            self._add_alert(
                alert_type="emergency_stop",
                level=DashboardAlert.CRITICAL,
                message=f"Emergency stop: {emergency_data.get('reason', 'unknown')}",
                details=emergency_data
            )
            
            self._add_event(
                event_type="emergency_stop",
                message=f"Emergency stop triggered: {emergency_data.get('reason', 'unknown')}",
                details=emergency_data
            )
            
        except Exception as e:
            logger.error(f"Error handling emergency stop event: {e}")
    
    async def _handle_position_update(self, event: Event):
        """Handle position update events"""
        try:
            position_data = event.payload
            
            self._add_event(
                event_type="position_update",
                message="Position update received",
                details={"positions_count": len(position_data.get("positions", []))}
            )
            
        except Exception as e:
            logger.error(f"Error handling position update event: {e}")
    
    async def _handle_error_event(self, event: Event):
        """Handle error events"""
        try:
            error_data = event.payload
            
            # Determine alert level based on error severity
            severity = error_data.get("severity", "error")
            if severity == "critical":
                level = DashboardAlert.CRITICAL
            elif severity == "warning":
                level = DashboardAlert.WARNING
            else:
                level = DashboardAlert.ERROR
            
            self._add_alert(
                alert_type="system_error",
                level=level,
                message=f"System error: {error_data.get('message', 'unknown')}",
                details=error_data
            )
            
            self._add_event(
                event_type="system_error",
                message=f"Error detected: {error_data.get('message', 'unknown')}",
                details=error_data
            )
            
        except Exception as e:
            logger.error(f"Error handling error event: {e}")
    
    async def _handle_trade_execution(self, event: Event):
        """Handle trade execution events"""
        try:
            trade_data = event.payload
            
            self._add_event(
                event_type="trade_execution",
                message=f"Trade executed: {trade_data.get('status', 'unknown')}",
                details=trade_data
            )
            
        except Exception as e:
            logger.error(f"Error handling trade execution event: {e}")
    
    # Public API methods
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        try:
            if self.current_data:
                return asdict(self.current_data)
            else:
                return {"error": "No data available"}
                
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status"""
        try:
            return {
                "running": self.running,
                "updates": self.dashboard_updates,
                "last_update": self.last_update_time.isoformat(),
                "avg_update_time_ms": (sum(self.update_times) / len(self.update_times)) * 1000 if self.update_times else 0,
                "alerts_count": len(self.alerts),
                "events_count": len(self.recent_events),
                "metrics_tracked": len(self.metric_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard status: {e}")
            return {"error": str(e)}
    
    def export_dashboard_data(self, format: str = "json") -> str:
        """Export dashboard data"""
        try:
            if format == "json":
                return json.dumps(self.get_dashboard_data(), indent=2, default=str)
            else:
                return "Unsupported format"
                
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            return f"Export error: {str(e)}"