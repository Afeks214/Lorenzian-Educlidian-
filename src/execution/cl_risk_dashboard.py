"""
CL Risk Monitoring Dashboard
===========================

Real-time risk monitoring dashboard for CL crude oil trading.
Provides comprehensive visualization and alerting for portfolio
risk metrics, position monitoring, and execution performance.

Key Features:
- Real-time portfolio risk metrics
- Position-level risk monitoring
- Execution performance tracking
- Risk alerts and notifications
- Interactive dashboard interface
- Historical risk analysis

Author: Agent 4 - Risk Management Mission
Date: 2025-07-17
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from collections import defaultdict, deque

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DashboardSection(Enum):
    """Dashboard sections"""
    PORTFOLIO_OVERVIEW = "portfolio_overview"
    POSITION_MONITOR = "position_monitor"
    RISK_METRICS = "risk_metrics"
    EXECUTION_PERFORMANCE = "execution_performance"
    ALERTS = "alerts"
    MARKET_CONDITIONS = "market_conditions"
    PERFORMANCE_ANALYTICS = "performance_analytics"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Metric types for display"""
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    RATIO = "ratio"
    COUNT = "count"
    TIME = "time"

@dataclass
class DashboardMetric:
    """Dashboard metric data structure"""
    name: str
    value: float
    display_value: str
    metric_type: MetricType
    change_24h: float = 0.0
    change_percent: float = 0.0
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    status: str = "normal"  # normal, warning, critical
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class DashboardAlert:
    """Dashboard alert data structure"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    section: DashboardSection
    timestamp: datetime
    acknowledged: bool = False
    auto_resolve: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    widget_type: str  # chart, table, metric, gauge
    section: DashboardSection
    data_source: str
    refresh_interval: int = 30  # seconds
    configuration: Dict[str, Any] = field(default_factory=dict)

class CLRiskDashboard:
    """
    Comprehensive real-time risk monitoring dashboard for CL trading
    
    Provides real-time visualization of portfolio risk metrics, position
    monitoring, execution performance, and market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CL Risk Dashboard
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        
        # Dashboard configuration
        self.refresh_interval = config.get('refresh_interval', 5)  # seconds
        self.history_retention = config.get('history_retention', 7)  # days
        self.alert_retention = config.get('alert_retention', 30)  # days
        
        # Risk thresholds
        self.risk_thresholds = config.get('risk_thresholds', {
            'portfolio_var_warning': 0.02,
            'portfolio_var_critical': 0.05,
            'position_loss_warning': 0.05,
            'position_loss_critical': 0.10,
            'leverage_warning': 2.0,
            'leverage_critical': 3.0,
            'drawdown_warning': 0.10,
            'drawdown_critical': 0.20
        })
        
        # Data storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts_history: List[DashboardAlert] = []
        self.performance_history: deque = deque(maxlen=1000)
        
        # Dashboard state
        self.current_metrics: Dict[str, DashboardMetric] = {}
        self.active_alerts: List[DashboardAlert] = []
        self.widget_data: Dict[str, Any] = {}
        
        # Connected components (to be injected)
        self.risk_manager = None
        self.portfolio_manager = None
        self.execution_engine = None
        self.market_analyzer = None
        
        # Dashboard widgets
        self.widgets = self._initialize_widgets()
        
        # Real-time update flag
        self.is_running = False
        self.update_task = None
        
        logger.info("✅ CL Risk Dashboard initialized")
        logger.info(f"   📊 Refresh Interval: {self.refresh_interval}s")
        logger.info(f"   📊 Widgets: {len(self.widgets)}")
        logger.info(f"   📊 Risk Thresholds: {len(self.risk_thresholds)}")
    
    def _initialize_widgets(self) -> List[DashboardWidget]:
        """Initialize dashboard widgets"""
        widgets = [
            # Portfolio Overview
            DashboardWidget(
                widget_id="portfolio_value",
                title="Portfolio Value",
                widget_type="metric",
                section=DashboardSection.PORTFOLIO_OVERVIEW,
                data_source="portfolio_manager",
                configuration={"format": "currency", "show_change": True}
            ),
            DashboardWidget(
                widget_id="daily_pnl",
                title="Daily P&L",
                widget_type="metric",
                section=DashboardSection.PORTFOLIO_OVERVIEW,
                data_source="portfolio_manager",
                configuration={"format": "currency", "show_change": True}
            ),
            DashboardWidget(
                widget_id="portfolio_chart",
                title="Portfolio Performance",
                widget_type="chart",
                section=DashboardSection.PORTFOLIO_OVERVIEW,
                data_source="portfolio_manager",
                configuration={"chart_type": "line", "timeframe": "24h"}
            ),
            
            # Position Monitor
            DashboardWidget(
                widget_id="positions_table",
                title="Active Positions",
                widget_type="table",
                section=DashboardSection.POSITION_MONITOR,
                data_source="portfolio_manager",
                configuration={"columns": ["symbol", "side", "quantity", "pnl", "risk"]}
            ),
            DashboardWidget(
                widget_id="position_heatmap",
                title="Position Risk Heatmap",
                widget_type="chart",
                section=DashboardSection.POSITION_MONITOR,
                data_source="portfolio_manager",
                configuration={"chart_type": "heatmap"}
            ),
            
            # Risk Metrics
            DashboardWidget(
                widget_id="var_gauge",
                title="Portfolio VaR",
                widget_type="gauge",
                section=DashboardSection.RISK_METRICS,
                data_source="risk_manager",
                configuration={"min": 0, "max": 0.1, "thresholds": [0.02, 0.05]}
            ),
            DashboardWidget(
                widget_id="leverage_gauge",
                title="Leverage",
                widget_type="gauge",
                section=DashboardSection.RISK_METRICS,
                data_source="portfolio_manager",
                configuration={"min": 0, "max": 5, "thresholds": [2.0, 3.0]}
            ),
            DashboardWidget(
                widget_id="risk_breakdown",
                title="Risk Breakdown",
                widget_type="chart",
                section=DashboardSection.RISK_METRICS,
                data_source="risk_manager",
                configuration={"chart_type": "donut"}
            ),
            
            # Execution Performance
            DashboardWidget(
                widget_id="execution_metrics",
                title="Execution Metrics",
                widget_type="table",
                section=DashboardSection.EXECUTION_PERFORMANCE,
                data_source="execution_engine",
                configuration={"metrics": ["fill_rate", "avg_slippage", "execution_cost"]}
            ),
            DashboardWidget(
                widget_id="slippage_chart",
                title="Slippage Trend",
                widget_type="chart",
                section=DashboardSection.EXECUTION_PERFORMANCE,
                data_source="execution_engine",
                configuration={"chart_type": "line", "timeframe": "24h"}
            ),
            
            # Market Conditions
            DashboardWidget(
                widget_id="market_conditions",
                title="Market Conditions",
                widget_type="table",
                section=DashboardSection.MARKET_CONDITIONS,
                data_source="market_analyzer",
                configuration={"conditions": ["liquidity", "volatility", "geopolitical_risk"]}
            ),
            DashboardWidget(
                widget_id="volatility_chart",
                title="Volatility Trend",
                widget_type="chart",
                section=DashboardSection.MARKET_CONDITIONS,
                data_source="market_analyzer",
                configuration={"chart_type": "line", "timeframe": "24h"}
            )
        ]
        
        return widgets
    
    def connect_components(self,
                          risk_manager=None,
                          portfolio_manager=None,
                          execution_engine=None,
                          market_analyzer=None):
        """Connect dashboard to system components"""
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.execution_engine = execution_engine
        self.market_analyzer = market_analyzer
        
        logger.info("Dashboard components connected")
    
    async def start_dashboard(self):
        """Start real-time dashboard updates"""
        if self.is_running:
            logger.warning("Dashboard already running")
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        
        logger.info("🚀 Dashboard started")
    
    async def stop_dashboard(self):
        """Stop dashboard updates"""
        self.is_running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Dashboard stopped")
    
    async def _update_loop(self):
        """Main dashboard update loop"""
        while self.is_running:
            try:
                await self._update_dashboard()
                await asyncio.sleep(self.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(self.refresh_interval)
    
    async def _update_dashboard(self):
        """Update all dashboard data"""
        try:
            # Update metrics
            await self._update_metrics()
            
            # Update widget data
            await self._update_widget_data()
            
            # Check for alerts
            await self._check_alerts()
            
            # Clean up old data
            self._cleanup_old_data()
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
    
    async def _update_metrics(self):
        """Update all dashboard metrics"""
        try:
            current_time = datetime.now()
            
            # Portfolio metrics
            if self.portfolio_manager:
                portfolio_summary = self.portfolio_manager.get_portfolio_summary()
                
                # Portfolio value
                portfolio_value = portfolio_summary.get('portfolio_value', 0)
                self.current_metrics['portfolio_value'] = DashboardMetric(
                    name="Portfolio Value",
                    value=portfolio_value,
                    display_value=f"${portfolio_value:,.2f}",
                    metric_type=MetricType.CURRENCY,
                    change_24h=self._calculate_24h_change('portfolio_value', portfolio_value),
                    last_updated=current_time
                )
                
                # Daily P&L
                daily_pnl = portfolio_summary.get('total_unrealized_pnl', 0) + portfolio_summary.get('total_realized_pnl', 0)
                self.current_metrics['daily_pnl'] = DashboardMetric(
                    name="Daily P&L",
                    value=daily_pnl,
                    display_value=f"${daily_pnl:,.2f}",
                    metric_type=MetricType.CURRENCY,
                    change_24h=self._calculate_24h_change('daily_pnl', daily_pnl),
                    last_updated=current_time
                )
                
                # Number of positions
                num_positions = portfolio_summary.get('num_positions', 0)
                self.current_metrics['num_positions'] = DashboardMetric(
                    name="Active Positions",
                    value=num_positions,
                    display_value=str(num_positions),
                    metric_type=MetricType.COUNT,
                    change_24h=self._calculate_24h_change('num_positions', num_positions),
                    last_updated=current_time
                )
                
                # Leverage
                leverage = portfolio_summary.get('exposure_summary', {}).get('crude_oil', {}).get('gross', 0) / portfolio_value if portfolio_value > 0 else 0
                self.current_metrics['leverage'] = DashboardMetric(
                    name="Leverage",
                    value=leverage,
                    display_value=f"{leverage:.2f}x",
                    metric_type=MetricType.RATIO,
                    change_24h=self._calculate_24h_change('leverage', leverage),
                    threshold_warning=self.risk_thresholds.get('leverage_warning', 2.0),
                    threshold_critical=self.risk_thresholds.get('leverage_critical', 3.0),
                    status=self._get_metric_status(leverage, 'leverage'),
                    last_updated=current_time
                )
            
            # Risk metrics
            if self.risk_manager:
                risk_metrics = self.risk_manager.get_risk_metrics(
                    {'total_value': portfolio_value, 'positions': {}}
                )
                
                # Portfolio VaR
                var_95 = risk_metrics.get('var_95', 0)
                self.current_metrics['var_95'] = DashboardMetric(
                    name="Portfolio VaR (95%)",
                    value=var_95,
                    display_value=f"{var_95:.2%}",
                    metric_type=MetricType.PERCENTAGE,
                    change_24h=self._calculate_24h_change('var_95', var_95),
                    threshold_warning=self.risk_thresholds.get('portfolio_var_warning', 0.02),
                    threshold_critical=self.risk_thresholds.get('portfolio_var_critical', 0.05),
                    status=self._get_metric_status(var_95, 'portfolio_var'),
                    last_updated=current_time
                )
                
                # Concentration ratio
                concentration = risk_metrics.get('concentration_ratio', 0)
                self.current_metrics['concentration'] = DashboardMetric(
                    name="Concentration Ratio",
                    value=concentration,
                    display_value=f"{concentration:.2%}",
                    metric_type=MetricType.PERCENTAGE,
                    change_24h=self._calculate_24h_change('concentration', concentration),
                    last_updated=current_time
                )
            
            # Execution metrics
            if self.execution_engine:
                execution_summary = self.execution_engine.get_execution_summary()
                
                # Fill rate
                fill_rate = execution_summary.get('fill_rate', 0)
                self.current_metrics['fill_rate'] = DashboardMetric(
                    name="Fill Rate",
                    value=fill_rate,
                    display_value=f"{fill_rate:.1%}",
                    metric_type=MetricType.PERCENTAGE,
                    change_24h=self._calculate_24h_change('fill_rate', fill_rate),
                    last_updated=current_time
                )
                
                # Average slippage
                avg_slippage = execution_summary.get('average_slippage', 0)
                self.current_metrics['avg_slippage'] = DashboardMetric(
                    name="Average Slippage",
                    value=avg_slippage,
                    display_value=f"{avg_slippage:.4f}",
                    metric_type=MetricType.RATIO,
                    change_24h=self._calculate_24h_change('avg_slippage', avg_slippage),
                    last_updated=current_time
                )
            
            # Market conditions
            if self.market_analyzer:
                market_summary = self.market_analyzer.get_market_summary()
                
                # Geopolitical risk
                geopolitical_risk = market_summary.get('geopolitical_risk', 0)
                self.current_metrics['geopolitical_risk'] = DashboardMetric(
                    name="Geopolitical Risk",
                    value=geopolitical_risk,
                    display_value=f"{geopolitical_risk:.2%}",
                    metric_type=MetricType.PERCENTAGE,
                    change_24h=self._calculate_24h_change('geopolitical_risk', geopolitical_risk),
                    last_updated=current_time
                )
                
                # Session liquidity
                session_liquidity = market_summary.get('session_liquidity', 0)
                self.current_metrics['session_liquidity'] = DashboardMetric(
                    name="Session Liquidity",
                    value=session_liquidity,
                    display_value=f"{session_liquidity:.2f}",
                    metric_type=MetricType.RATIO,
                    change_24h=self._calculate_24h_change('session_liquidity', session_liquidity),
                    last_updated=current_time
                )
            
            # Store metrics in history
            for metric_name, metric in self.current_metrics.items():
                self.metrics_history[metric_name].append({
                    'timestamp': current_time,
                    'value': metric.value,
                    'status': metric.status
                })
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _calculate_24h_change(self, metric_name: str, current_value: float) -> float:
        """Calculate 24-hour change for a metric"""
        try:
            history = self.metrics_history[metric_name]
            if len(history) < 2:
                return 0.0
            
            # Find value from 24 hours ago
            cutoff_time = datetime.now() - timedelta(hours=24)
            historical_values = [
                entry for entry in history 
                if entry['timestamp'] >= cutoff_time
            ]
            
            if not historical_values:
                return 0.0
            
            old_value = historical_values[0]['value']
            return current_value - old_value
            
        except Exception as e:
            logger.error(f"Error calculating 24h change for {metric_name}: {e}")
            return 0.0
    
    def _get_metric_status(self, value: float, metric_key: str) -> str:
        """Get status for a metric based on thresholds"""
        warning_threshold = self.risk_thresholds.get(f'{metric_key}_warning')
        critical_threshold = self.risk_thresholds.get(f'{metric_key}_critical')
        
        if critical_threshold and value >= critical_threshold:
            return "critical"
        elif warning_threshold and value >= warning_threshold:
            return "warning"
        else:
            return "normal"
    
    async def _update_widget_data(self):
        """Update data for all widgets"""
        try:
            for widget in self.widgets:
                widget_data = await self._get_widget_data(widget)
                self.widget_data[widget.widget_id] = widget_data
                
        except Exception as e:
            logger.error(f"Error updating widget data: {e}")
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget"""
        try:
            if widget.data_source == "portfolio_manager" and self.portfolio_manager:
                return await self._get_portfolio_widget_data(widget)
            elif widget.data_source == "risk_manager" and self.risk_manager:
                return await self._get_risk_widget_data(widget)
            elif widget.data_source == "execution_engine" and self.execution_engine:
                return await self._get_execution_widget_data(widget)
            elif widget.data_source == "market_analyzer" and self.market_analyzer:
                return await self._get_market_widget_data(widget)
            else:
                return {"error": f"Data source {widget.data_source} not available"}
                
        except Exception as e:
            logger.error(f"Error getting widget data for {widget.widget_id}: {e}")
            return {"error": str(e)}
    
    async def _get_portfolio_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get portfolio widget data"""
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            
            if widget.widget_id == "portfolio_chart":
                # Portfolio performance chart
                history = list(self.metrics_history['portfolio_value'])
                return {
                    "type": "line_chart",
                    "data": [
                        {"x": entry['timestamp'].isoformat(), "y": entry['value']}
                        for entry in history[-100:]  # Last 100 points
                    ],
                    "title": "Portfolio Value (24h)",
                    "y_axis": "Value ($)"
                }
            
            elif widget.widget_id == "positions_table":
                # Active positions table
                positions = portfolio_summary.get('position_details', [])
                return {
                    "type": "table",
                    "columns": ["Symbol", "Side", "Quantity", "P&L", "Risk%"],
                    "data": [
                        [
                            pos['symbol'],
                            pos['side'],
                            f"{pos['quantity']:.0f}",
                            f"${pos['unrealized_pnl']:.2f}",
                            f"{pos['pnl_percent']:.1%}"
                        ]
                        for pos in positions
                    ]
                }
            
            elif widget.widget_id == "position_heatmap":
                # Position risk heatmap
                positions = portfolio_summary.get('position_details', [])
                return {
                    "type": "heatmap",
                    "data": [
                        {
                            "symbol": pos['symbol'],
                            "risk": abs(pos['pnl_percent']),
                            "pnl": pos['unrealized_pnl'],
                            "size": pos['market_value']
                        }
                        for pos in positions
                    ]
                }
            
            return {"error": f"Unknown portfolio widget: {widget.widget_id}"}
            
        except Exception as e:
            logger.error(f"Error getting portfolio widget data: {e}")
            return {"error": str(e)}
    
    async def _get_risk_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get risk widget data"""
        try:
            if widget.widget_id == "var_gauge":
                # VaR gauge
                var_95 = self.current_metrics.get('var_95', DashboardMetric("", 0, "", MetricType.PERCENTAGE)).value
                return {
                    "type": "gauge",
                    "value": var_95,
                    "min": 0,
                    "max": 0.1,
                    "thresholds": [0.02, 0.05],
                    "title": "Portfolio VaR (95%)",
                    "format": "percentage"
                }
            
            elif widget.widget_id == "leverage_gauge":
                # Leverage gauge
                leverage = self.current_metrics.get('leverage', DashboardMetric("", 0, "", MetricType.RATIO)).value
                return {
                    "type": "gauge",
                    "value": leverage,
                    "min": 0,
                    "max": 5,
                    "thresholds": [2.0, 3.0],
                    "title": "Leverage",
                    "format": "ratio"
                }
            
            elif widget.widget_id == "risk_breakdown":
                # Risk breakdown chart
                # Simplified risk breakdown
                return {
                    "type": "donut_chart",
                    "data": [
                        {"label": "Market Risk", "value": 60},
                        {"label": "Concentration Risk", "value": 25},
                        {"label": "Liquidity Risk", "value": 10},
                        {"label": "Operational Risk", "value": 5}
                    ],
                    "title": "Risk Breakdown"
                }
            
            return {"error": f"Unknown risk widget: {widget.widget_id}"}
            
        except Exception as e:
            logger.error(f"Error getting risk widget data: {e}")
            return {"error": str(e)}
    
    async def _get_execution_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get execution widget data"""
        try:
            execution_summary = self.execution_engine.get_execution_summary()
            
            if widget.widget_id == "execution_metrics":
                # Execution metrics table
                return {
                    "type": "table",
                    "columns": ["Metric", "Value"],
                    "data": [
                        ["Fill Rate", f"{execution_summary.get('fill_rate', 0):.1%}"],
                        ["Avg Slippage", f"{execution_summary.get('average_slippage', 0):.4f}"],
                        ["Total Orders", f"{execution_summary.get('total_orders', 0)}"],
                        ["Total Volume", f"{execution_summary.get('total_volume', 0):.0f}"],
                        ["Execution Cost", f"${execution_summary.get('total_cost', 0):.2f}"]
                    ]
                }
            
            elif widget.widget_id == "slippage_chart":
                # Slippage trend chart
                history = list(self.metrics_history['avg_slippage'])
                return {
                    "type": "line_chart",
                    "data": [
                        {"x": entry['timestamp'].isoformat(), "y": entry['value']}
                        for entry in history[-100:]  # Last 100 points
                    ],
                    "title": "Slippage Trend (24h)",
                    "y_axis": "Slippage"
                }
            
            return {"error": f"Unknown execution widget: {widget.widget_id}"}
            
        except Exception as e:
            logger.error(f"Error getting execution widget data: {e}")
            return {"error": str(e)}
    
    async def _get_market_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get market widget data"""
        try:
            market_summary = self.market_analyzer.get_market_summary()
            
            if widget.widget_id == "market_conditions":
                # Market conditions table
                return {
                    "type": "table",
                    "columns": ["Condition", "Value", "Status"],
                    "data": [
                        ["Liquidity", f"{market_summary.get('session_liquidity', 0):.2f}", market_summary.get('market_conditions', {}).get('liquidity_level', 'unknown')],
                        ["Geopolitical Risk", f"{market_summary.get('geopolitical_risk', 0):.2%}", market_summary.get('market_conditions', {}).get('risk_level', 'unknown')],
                        ["Inventory Impact", f"{market_summary.get('inventory_impact', 0):.2%}", market_summary.get('market_conditions', {}).get('inventory_influence', 'unknown')]
                    ]
                }
            
            elif widget.widget_id == "volatility_chart":
                # Volatility trend chart
                history = list(self.metrics_history['geopolitical_risk'])
                return {
                    "type": "line_chart",
                    "data": [
                        {"x": entry['timestamp'].isoformat(), "y": entry['value']}
                        for entry in history[-100:]  # Last 100 points
                    ],
                    "title": "Geopolitical Risk Trend (24h)",
                    "y_axis": "Risk Level"
                }
            
            return {"error": f"Unknown market widget: {widget.widget_id}"}
            
        except Exception as e:
            logger.error(f"Error getting market widget data: {e}")
            return {"error": str(e)}
    
    async def _check_alerts(self):
        """Check for new alerts"""
        try:
            current_time = datetime.now()
            
            # Check metric thresholds
            for metric_name, metric in self.current_metrics.items():
                if metric.status == "critical":
                    alert = DashboardAlert(
                        alert_id=f"metric_{metric_name}_{int(current_time.timestamp())}",
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical Risk Alert: {metric.name}",
                        message=f"{metric.name} has reached critical level: {metric.display_value}",
                        section=DashboardSection.RISK_METRICS,
                        timestamp=current_time,
                        auto_resolve=True,
                        metadata={"metric": metric_name, "value": metric.value}
                    )
                    await self._add_alert(alert)
                    
                elif metric.status == "warning":
                    alert = DashboardAlert(
                        alert_id=f"metric_{metric_name}_{int(current_time.timestamp())}",
                        severity=AlertSeverity.WARNING,
                        title=f"Risk Warning: {metric.name}",
                        message=f"{metric.name} has reached warning level: {metric.display_value}",
                        section=DashboardSection.RISK_METRICS,
                        timestamp=current_time,
                        auto_resolve=True,
                        metadata={"metric": metric_name, "value": metric.value}
                    )
                    await self._add_alert(alert)
            
            # Check for position alerts
            if self.portfolio_manager:
                await self._check_position_alerts()
            
            # Check for execution alerts
            if self.execution_engine:
                await self._check_execution_alerts()
            
            # Auto-resolve old alerts
            await self._auto_resolve_alerts()
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _check_position_alerts(self):
        """Check for position-level alerts"""
        try:
            portfolio_summary = self.portfolio_manager.get_portfolio_summary()
            positions = portfolio_summary.get('position_details', [])
            
            for position in positions:
                pnl_percent = position.get('pnl_percent', 0)
                
                # Check for large losses
                if pnl_percent < -self.risk_thresholds.get('position_loss_critical', 0.10):
                    alert = DashboardAlert(
                        alert_id=f"position_loss_{position['symbol']}_{int(datetime.now().timestamp())}",
                        severity=AlertSeverity.CRITICAL,
                        title=f"Large Position Loss: {position['symbol']}",
                        message=f"Position {position['symbol']} has large loss: {pnl_percent:.1%}",
                        section=DashboardSection.POSITION_MONITOR,
                        timestamp=datetime.now(),
                        auto_resolve=True,
                        metadata={"symbol": position['symbol'], "pnl_percent": pnl_percent}
                    )
                    await self._add_alert(alert)
                    
                elif pnl_percent < -self.risk_thresholds.get('position_loss_warning', 0.05):
                    alert = DashboardAlert(
                        alert_id=f"position_loss_{position['symbol']}_{int(datetime.now().timestamp())}",
                        severity=AlertSeverity.WARNING,
                        title=f"Position Loss Warning: {position['symbol']}",
                        message=f"Position {position['symbol']} has loss: {pnl_percent:.1%}",
                        section=DashboardSection.POSITION_MONITOR,
                        timestamp=datetime.now(),
                        auto_resolve=True,
                        metadata={"symbol": position['symbol'], "pnl_percent": pnl_percent}
                    )
                    await self._add_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking position alerts: {e}")
    
    async def _check_execution_alerts(self):
        """Check for execution-related alerts"""
        try:
            execution_summary = self.execution_engine.get_execution_summary()
            
            # Check fill rate
            fill_rate = execution_summary.get('fill_rate', 0)
            if fill_rate < 0.8:  # Less than 80% fill rate
                alert = DashboardAlert(
                    alert_id=f"low_fill_rate_{int(datetime.now().timestamp())}",
                    severity=AlertSeverity.WARNING,
                    title="Low Fill Rate",
                    message=f"Fill rate is low: {fill_rate:.1%}",
                    section=DashboardSection.EXECUTION_PERFORMANCE,
                    timestamp=datetime.now(),
                    auto_resolve=True,
                    metadata={"fill_rate": fill_rate}
                )
                await self._add_alert(alert)
            
            # Check average slippage
            avg_slippage = execution_summary.get('average_slippage', 0)
            if avg_slippage > 0.001:  # More than 10 bps average slippage
                alert = DashboardAlert(
                    alert_id=f"high_slippage_{int(datetime.now().timestamp())}",
                    severity=AlertSeverity.WARNING,
                    title="High Average Slippage",
                    message=f"Average slippage is high: {avg_slippage:.4f}",
                    section=DashboardSection.EXECUTION_PERFORMANCE,
                    timestamp=datetime.now(),
                    auto_resolve=True,
                    metadata={"avg_slippage": avg_slippage}
                )
                await self._add_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking execution alerts: {e}")
    
    async def _add_alert(self, alert: DashboardAlert):
        """Add new alert"""
        try:
            # Check if similar alert already exists
            existing_alerts = [
                a for a in self.active_alerts 
                if a.alert_id == alert.alert_id or 
                (a.title == alert.title and (datetime.now() - a.timestamp).seconds < 300)
            ]
            
            if not existing_alerts:
                self.active_alerts.append(alert)
                self.alerts_history.append(alert)
                
                logger.info(f"🚨 New alert: {alert.title}")
                
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
    
    async def _auto_resolve_alerts(self):
        """Auto-resolve alerts that are no longer relevant"""
        try:
            current_time = datetime.now()
            
            # Remove auto-resolve alerts older than 5 minutes
            self.active_alerts = [
                alert for alert in self.active_alerts
                if not alert.auto_resolve or (current_time - alert.timestamp).seconds < 300
            ]
            
        except Exception as e:
            logger.error(f"Error auto-resolving alerts: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.history_retention)
            
            # Clean up alerts history
            self.alerts_history = [
                alert for alert in self.alerts_history
                if alert.timestamp >= cutoff_time
            ]
            
            # Clean up performance history
            self.performance_history = deque(
                [entry for entry in self.performance_history 
                 if entry.get('timestamp', datetime.now()) >= cutoff_time],
                maxlen=1000
            )
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get complete dashboard state"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    name: {
                        'name': metric.name,
                        'value': metric.value,
                        'display_value': metric.display_value,
                        'metric_type': metric.metric_type.value,
                        'change_24h': metric.change_24h,
                        'status': metric.status,
                        'last_updated': metric.last_updated.isoformat()
                    }
                    for name, metric in self.current_metrics.items()
                },
                'widgets': {
                    widget_id: data
                    for widget_id, data in self.widget_data.items()
                },
                'alerts': {
                    'active': [
                        {
                            'alert_id': alert.alert_id,
                            'severity': alert.severity.value,
                            'title': alert.title,
                            'message': alert.message,
                            'section': alert.section.value,
                            'timestamp': alert.timestamp.isoformat(),
                            'acknowledged': alert.acknowledged
                        }
                        for alert in self.active_alerts
                    ],
                    'count': len(self.active_alerts),
                    'critical_count': len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]),
                    'warning_count': len([a for a in self.active_alerts if a.severity == AlertSeverity.WARNING])
                },
                'system_status': {
                    'dashboard_running': self.is_running,
                    'connected_components': {
                        'risk_manager': self.risk_manager is not None,
                        'portfolio_manager': self.portfolio_manager is not None,
                        'execution_engine': self.execution_engine is not None,
                        'market_analyzer': self.market_analyzer is not None
                    },
                    'refresh_interval': self.refresh_interval,
                    'last_update': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard state: {e}")
            return {'error': str(e)}
    
    def acknowledge_alert(self, alert_id: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        try:
            for alert in self.active_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return {
                        'success': True,
                        'alert_id': alert_id,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'success': False,
                'error': f'Alert {alert_id} not found'
            }
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_historical_data(self, metric_name: str, timeframe: str = '24h') -> Dict[str, Any]:
        """Get historical data for a metric"""
        try:
            if metric_name not in self.metrics_history:
                return {'error': f'Metric {metric_name} not found'}
            
            # Parse timeframe
            if timeframe == '24h':
                cutoff_time = datetime.now() - timedelta(hours=24)
            elif timeframe == '7d':
                cutoff_time = datetime.now() - timedelta(days=7)
            elif timeframe == '30d':
                cutoff_time = datetime.now() - timedelta(days=30)
            else:
                cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Filter data
            history = [
                entry for entry in self.metrics_history[metric_name]
                if entry['timestamp'] >= cutoff_time
            ]
            
            return {
                'metric_name': metric_name,
                'timeframe': timeframe,
                'data_points': len(history),
                'data': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'value': entry['value'],
                        'status': entry.get('status', 'normal')
                    }
                    for entry in history
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting historical data for {metric_name}: {e}")
            return {'error': str(e)}
    
    def export_dashboard_data(self, format_type: str = 'json') -> Dict[str, Any]:
        """Export dashboard data"""
        try:
            dashboard_data = self.get_dashboard_state()
            
            if format_type == 'json':
                return {
                    'success': True,
                    'format': 'json',
                    'data': dashboard_data,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': f'Format {format_type} not supported'
                }
                
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            return {
                'success': False,
                'error': str(e)
            }