"""
Enhanced Correlation Tracker with EWMA and Regime Change Detection

This module implements an adaptive correlation tracking system designed to:
1. Use Exponentially Weighted Moving Average (EWMA) for faster regime adaptation
2. Detect correlation shocks in real-time
3. Trigger automated risk reduction protocols
4. Maintain performance targets <5ms for VaR calculations

Mathematical Foundation:
- EWMA: λ * previous_corr + (1-λ) * current_corr
- Regime detection: monitor average portfolio correlation spikes
- Shock threshold: >0.5 correlation increase within 10-minute window
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import asyncio
import structlog
from enum import Enum
import threading

from src.core.events import Event, EventType, EventBus
from src.safety.trading_system_controller import get_controller

logger = structlog.get_logger()


class CorrelationRegime(Enum):
    """Market correlation regimes"""
    NORMAL = "NORMAL"           # Typical correlation levels
    ELEVATED = "ELEVATED"       # Higher than normal correlations
    CRISIS = "CRISIS"          # Extreme correlation (>0.8)
    SHOCK = "SHOCK"            # Sudden correlation spike


@dataclass
class CorrelationShock:
    """Represents a detected correlation shock event"""
    timestamp: datetime
    previous_avg_corr: float
    current_avg_corr: float
    correlation_change: float
    affected_assets: List[str]
    severity: str  # "MODERATE", "HIGH", "CRITICAL"
    
    
@dataclass
class RiskReductionAction:
    """Automated risk reduction action taken"""
    timestamp: datetime
    action_type: str  # "LEVERAGE_REDUCTION", "POSITION_CLOSE", "TRADING_HALT"
    leverage_before: float
    leverage_after: float
    trigger_reason: str
    manual_reset_required: bool = True


class CorrelationTracker:
    """
    Enhanced correlation tracker with EWMA and regime change detection.
    
    Key Features:
    - EWMA-based correlation calculation for faster adaptation
    - Real-time correlation shock detection
    - Automated risk reduction protocols
    - Performance optimized for <5ms VaR calculations
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        ewma_lambda: float = 0.94,  # Standard RiskMetrics decay factor
        shock_threshold: float = 0.5,  # Correlation increase threshold
        shock_window_minutes: int = 10,  # Detection window
        max_correlation_history: int = 1000,  # Memory efficiency
        performance_target_ms: float = 5.0  # Performance target
    ):
        self.event_bus = event_bus
        self.ewma_lambda = ewma_lambda
        self.shock_threshold = shock_threshold
        self.shock_window = timedelta(minutes=shock_window_minutes)
        self.max_history = max_correlation_history
        self.performance_target = performance_target_ms
        
        # Correlation data storage
        self.correlation_matrix: Optional[np.ndarray] = None
        self.asset_returns: Dict[str, deque] = {}
        self.correlation_history: deque = deque(maxlen=max_correlation_history)
        self.shock_alerts: List[CorrelationShock] = []
        self.risk_actions: List[RiskReductionAction] = []
        
        # Performance tracking
        self.calculation_times: deque = deque(maxlen=100)
        
        # Thread safety for correlation matrix operations
        self._correlation_lock = threading.RLock()
        
        # Risk management state
        self.current_regime = CorrelationRegime.NORMAL
        self.current_leverage: float = 1.0
        self.max_allowed_leverage: float = 4.0
        self.manual_reset_required: bool = False
        
        # Asset universe
        self.assets: List[str] = []
        self.asset_index: Dict[str, int] = {}
        
        # Callbacks for risk actions
        self.leverage_reduction_callback: Optional[Callable] = None
        
        # Subscribe to events
        self._setup_event_subscriptions()
        
        # Register with trading system controller
        system_controller = get_controller()
        if system_controller:
            system_controller.register_component("correlation_tracker", {
                "ewma_lambda": ewma_lambda,
                "shock_threshold": shock_threshold,
                "shock_window_minutes": shock_window_minutes,
                "performance_target_ms": performance_target_ms
            })
        
        logger.info("CorrelationTracker initialized", 
                   ewma_lambda=ewma_lambda,
                   shock_threshold=shock_threshold,
                   shock_window_minutes=shock_window_minutes)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for real-time updates"""
        self.event_bus.subscribe(EventType.NEW_5MIN_BAR, self._handle_price_update)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
    
    def register_leverage_callback(self, callback: Callable[[float], None]):
        """Register callback for automated leverage reduction"""
        self.leverage_reduction_callback = callback
        logger.info("Leverage reduction callback registered")
    
    def initialize_assets(self, asset_list: List[str]):
        """Initialize the asset universe for correlation tracking"""
        self.assets = asset_list
        self.asset_index = {asset: i for i, asset in enumerate(asset_list)}
        
        # Initialize correlation matrix
        n_assets = len(asset_list)
        self.correlation_matrix = np.eye(n_assets)  # Start with identity matrix
        
        # Initialize return storage
        for asset in asset_list:
            self.asset_returns[asset] = deque(maxlen=252)  # 1 year of daily returns
            
        logger.info("Asset universe initialized", 
                   n_assets=n_assets, 
                   assets=asset_list[:5])  # Log first 5 for brevity
    
    def _handle_price_update(self, event: Event):
        """Handle new price data for correlation updates"""
        # Check if system is ON before processing new data
        system_controller = get_controller()
        if system_controller and not system_controller.is_system_on():
            logger.debug("System is OFF - skipping correlation tracking update")
            return
            
        bar_data = event.payload
        if not hasattr(bar_data, 'symbol') or bar_data.symbol not in self.asset_index:
            return
            
        # Calculate return
        if len(self.asset_returns[bar_data.symbol]) > 0:
            previous_price = self.asset_returns[bar_data.symbol][-1][1]  # [timestamp, price]
            return_pct = (bar_data.close - previous_price) / previous_price
            
            # Store return with timestamp
            self.asset_returns[bar_data.symbol].append((bar_data.timestamp, return_pct))
            
            # Update correlation matrix if we have sufficient data
            if self._sufficient_data_available():
                start_time = datetime.now()
                self._update_correlation_matrix()
                calc_time = (datetime.now() - start_time).total_seconds() * 1000
                self.calculation_times.append(calc_time)
                
                # Cache the updated correlation matrix
                if system_controller and self.correlation_matrix is not None:
                    system_controller.cache_value("correlation_matrix", self.correlation_matrix.copy(), ttl_seconds=300)
                    system_controller.cache_value("correlation_regime", self.current_regime, ttl_seconds=300)
                    logger.debug("Cached correlation matrix and regime for OFF-system access")
                
                # Check for correlation shocks
                self._check_correlation_shock()
                
        # Always store the price for next calculation
        if bar_data.symbol in self.asset_returns:
            # Store current price for next return calculation
            self.asset_returns[bar_data.symbol].append((bar_data.timestamp, bar_data.close))
    
    def _handle_position_update(self, event: Event):
        """Handle position updates to track current leverage"""
        position_data = event.payload
        if hasattr(position_data, 'total_leverage'):
            self.current_leverage = position_data.total_leverage
    
    def _sufficient_data_available(self) -> bool:
        """Check if we have sufficient data for correlation calculation"""
        min_observations = 30  # Minimum observations for stable correlation
        return all(len(returns) >= min_observations for returns in self.asset_returns.values())
    
    def _update_correlation_matrix(self):
        """Update correlation matrix using EWMA methodology"""
        with self._correlation_lock:
            if not self._sufficient_data_available():
                return
            
            # Get latest returns for all assets
            latest_returns = []
            for asset in self.assets:
                if len(self.asset_returns[asset]) > 0:
                    # Get most recent return
                    latest_returns.append(self.asset_returns[asset][-1][1])  # [timestamp, return]
                else:
                    latest_returns.append(0.0)
            
            latest_returns = np.array(latest_returns)
            
            # Calculate sample correlation matrix from recent returns (for comparison)
            returns_matrix = []
            min_length = min(len(returns) for returns in self.asset_returns.values())
            recent_window = min(60, min_length)  # Use last 60 observations or available data
            
            for asset in self.assets:
                asset_returns_list = [ret[1] for ret in list(self.asset_returns[asset])[-recent_window:]]
                returns_matrix.append(asset_returns_list)
            
            returns_matrix = np.array(returns_matrix)
            
            if returns_matrix.shape[1] < 2:  # Need at least 2 observations
                return
                
            # Calculate current sample correlation
            current_corr = np.corrcoef(returns_matrix)
            
            # Apply EWMA update: λ * previous + (1-λ) * current
            if self.correlation_matrix is not None:
                self.correlation_matrix = (
                    self.ewma_lambda * self.correlation_matrix + 
                    (1 - self.ewma_lambda) * current_corr
                )
            else:
                self.correlation_matrix = current_corr
            
            # Ensure correlation matrix properties
            np.fill_diagonal(self.correlation_matrix, 1.0)  # Diagonal = 1
            
            # Store correlation history for regime detection
            avg_correlation = self._calculate_average_correlation()
            self.correlation_history.append((datetime.now(), avg_correlation))
            
            # Publish VaR update event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.VAR_UPDATE,
                    {
                        'correlation_matrix': self.correlation_matrix.copy(),
                        'average_correlation': avg_correlation,
                        'regime': self.current_regime.value
                    },
                    'CorrelationTracker'
                )
            )
    
    def _calculate_average_correlation(self) -> float:
        """Calculate average off-diagonal correlation"""
        if self.correlation_matrix is None:
            return 0.0
            
        # Get upper triangular matrix (excluding diagonal)
        upper_tri = np.triu(self.correlation_matrix, k=1)
        n_assets = self.correlation_matrix.shape[0]
        
        if n_assets < 2:
            return 0.0
            
        # Average of upper triangular elements
        return np.sum(upper_tri) / (n_assets * (n_assets - 1) / 2)
    
    def _check_correlation_shock(self):
        """Check for correlation shock and trigger alerts if necessary"""
        if len(self.correlation_history) < 2:
            return
            
        current_time = datetime.now()
        current_avg_corr = self.correlation_history[-1][1]
        
        # Look for correlation spikes within the shock window
        window_start = current_time - self.shock_window
        historical_correlations = [
            corr for timestamp, corr in self.correlation_history
            if timestamp >= window_start
        ]
        
        if len(historical_correlations) < 2:
            return
            
        min_historical_corr = min(historical_correlations[:-1])  # Exclude current
        correlation_change = current_avg_corr - min_historical_corr
        
        # Detect correlation shock
        if correlation_change > self.shock_threshold:
            severity = self._classify_shock_severity(correlation_change, current_avg_corr)
            
            shock = CorrelationShock(
                timestamp=current_time,
                previous_avg_corr=min_historical_corr,
                current_avg_corr=current_avg_corr,
                correlation_change=correlation_change,
                affected_assets=self.assets.copy(),
                severity=severity
            )
            
            self.shock_alerts.append(shock)
            self._trigger_correlation_shock_response(shock)
            
            logger.critical("Correlation shock detected",
                          correlation_change=correlation_change,
                          current_avg_corr=current_avg_corr,
                          severity=severity)
    
    def _classify_shock_severity(self, change: float, current_corr: float) -> str:
        """Classify the severity of correlation shock"""
        if current_corr > 0.9 or change > 0.7:
            return "CRITICAL"
        elif current_corr > 0.8 or change > 0.6:
            return "HIGH"
        else:
            return "MODERATE"
    
    def _trigger_correlation_shock_response(self, shock: CorrelationShock):
        """Trigger automated risk reduction in response to correlation shock"""
        if shock.severity in ["HIGH", "CRITICAL"]:
            # Automatic leverage reduction
            new_leverage = self.current_leverage * 0.5  # 50% reduction
            
            action = RiskReductionAction(
                timestamp=datetime.now(),
                action_type="LEVERAGE_REDUCTION",
                leverage_before=self.current_leverage,
                leverage_after=new_leverage,
                trigger_reason=f"Correlation shock: {shock.severity}",
                manual_reset_required=True
            )
            
            self.risk_actions.append(action)
            self.manual_reset_required = True
            
            # Execute leverage reduction
            if self.leverage_reduction_callback:
                self.leverage_reduction_callback(new_leverage)
            
            # Update regime
            if shock.severity == "CRITICAL":
                self.current_regime = CorrelationRegime.CRISIS
            else:
                self.current_regime = CorrelationRegime.ELEVATED
            
            # Publish risk breach event
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.RISK_BREACH,
                    {
                        'type': 'CORRELATION_SHOCK',
                        'severity': shock.severity,
                        'action': action,
                        'manual_reset_required': True
                    },
                    'CorrelationTracker'
                )
            )
            
            logger.critical("Automated risk reduction triggered",
                          action_type=action.action_type,
                          leverage_reduction=f"{action.leverage_before:.2f} -> {action.leverage_after:.2f}",
                          manual_reset_required=True)
    
    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get current correlation matrix"""
        with self._correlation_lock:
            # Check if system is ON before returning live correlation matrix
            system_controller = get_controller()
            if system_controller and not system_controller.is_system_on():
                logger.debug("System is OFF - returning cached correlation matrix")
                
                # Try to return cached correlation matrix
                cached_matrix = system_controller.get_cached_value("correlation_matrix")
                if cached_matrix is not None:
                    return cached_matrix.copy()
                
                # Fall back to current matrix if available
                if self.correlation_matrix is not None:
                    logger.debug("Using current correlation matrix as fallback while system is OFF")
                    return self.correlation_matrix.copy()
                
                logger.warning("No correlation matrix available while system is OFF")
                return None
            
            return self.correlation_matrix.copy() if self.correlation_matrix is not None else None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.calculation_times:
            return {"avg_calc_time_ms": 0, "max_calc_time_ms": 0, "target_met": True}
            
        avg_time = np.mean(self.calculation_times)
        max_time = np.max(self.calculation_times)
        target_met = avg_time <= self.performance_target
        
        return {
            "avg_calc_time_ms": avg_time,
            "max_calc_time_ms": max_time,
            "target_met": target_met,
            "calculation_count": len(self.calculation_times)
        }
    
    def get_regime_status(self) -> Dict:
        """Get current correlation regime status"""
        current_avg_corr = self._calculate_average_correlation()
        
        return {
            "current_regime": self.current_regime.value,
            "average_correlation": current_avg_corr,
            "current_leverage": self.current_leverage,
            "max_allowed_leverage": self.max_allowed_leverage,
            "manual_reset_required": self.manual_reset_required,
            "recent_shocks": len([s for s in self.shock_alerts 
                                if s.timestamp > datetime.now() - timedelta(hours=24)]),
            "total_risk_actions": len(self.risk_actions)
        }
    
    def manual_reset_risk_controls(self, operator_id: str, reason: str):
        """Manual reset of risk controls after investigation"""
        if not self.manual_reset_required:
            logger.warning("Manual reset attempted but not required")
            return False
            
        self.manual_reset_required = False
        self.current_regime = CorrelationRegime.NORMAL
        self.max_allowed_leverage = 4.0  # Reset to normal levels
        
        logger.info("Risk controls manually reset",
                   operator_id=operator_id,
                   reason=reason,
                   timestamp=datetime.now())
        
        # Publish reset event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.RISK_UPDATE,
                {
                    'type': 'MANUAL_RESET',
                    'operator_id': operator_id,
                    'reason': reason,
                    'new_regime': self.current_regime.value
                },
                'CorrelationTracker'
            )
        )
        
        return True
    
    def simulate_correlation_shock(self, shock_magnitude: float = 0.8):
        """Simulate correlation shock for testing purposes"""
        if self.correlation_matrix is None:
            logger.warning("Cannot simulate shock: no correlation matrix initialized")
            return
            
        # Save original matrix
        original_matrix = self.correlation_matrix.copy()
        
        # Create shocked matrix (increase all correlations)
        n_assets = self.correlation_matrix.shape[0]
        shock_matrix = np.full((n_assets, n_assets), shock_magnitude)
        np.fill_diagonal(shock_matrix, 1.0)
        
        # Update correlation matrix
        self.correlation_matrix = shock_matrix
        
        # Force shock detection
        current_avg_corr = self._calculate_average_correlation()
        self.correlation_history.append((datetime.now(), current_avg_corr))
        
        # Check for shock
        self._check_correlation_shock()
        
        logger.warning("Correlation shock simulated",
                      shock_magnitude=shock_magnitude,
                      new_avg_correlation=current_avg_corr)
        
        return original_matrix  # Return for restoration if needed