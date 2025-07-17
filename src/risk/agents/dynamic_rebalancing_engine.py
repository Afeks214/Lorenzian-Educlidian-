"""
High-Performance Dynamic Rebalancing Engine

This module implements a high-performance dynamic rebalancing engine for real-time portfolio
allocation adjustments with <10ms response time target. Features advanced optimization 
techniques, caching, and performance monitoring for ultra-low latency rebalancing decisions.

Key Features:
- <10ms response time target for rebalancing decisions
- Real-time portfolio drift monitoring and triggers
- Intelligent rebalancing frequency optimization
- Transaction cost-aware rebalancing
- Multi-threading and async processing for performance
- Advanced caching and pre-computation strategies
- Market microstructure-aware timing optimization
- Emergency rebalancing protocols
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
import time
import structlog
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from src.core.events import EventBus, Event, EventType
from src.risk.core.correlation_tracker import CorrelationRegime

logger = structlog.get_logger()


class RebalanceTrigger(Enum):
    """Rebalancing trigger types"""
    DRIFT_THRESHOLD = "drift_threshold"
    TIME_BASED = "time_based"
    VOLATILITY_REGIME = "volatility_regime"
    CORRELATION_SHOCK = "correlation_shock"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    PERFORMANCE_DIVERGENCE = "performance_divergence"
    EMERGENCY = "emergency"
    MANUAL = "manual"


class RebalanceUrgency(Enum):
    """Rebalancing urgency levels"""
    LOW = "low"          # Can wait for optimal timing
    MEDIUM = "medium"    # Should execute within minutes
    HIGH = "high"        # Should execute within seconds
    CRITICAL = "critical"  # Immediate execution required


@dataclass
class RebalanceSignal:
    """Rebalancing signal and metadata"""
    timestamp: datetime
    trigger: RebalanceTrigger
    urgency: RebalanceUrgency
    current_weights: np.ndarray
    target_weights: np.ndarray
    weight_drift: np.ndarray
    max_drift: float
    expected_cost: float
    risk_impact: float
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'trigger': self.trigger.value,
            'urgency': self.urgency.value,
            'current_weights': self.current_weights.tolist(),
            'target_weights': self.target_weights.tolist(),
            'max_drift': self.max_drift,
            'expected_cost': self.expected_cost,
            'risk_impact': self.risk_impact,
            'reason': self.reason
        }


@dataclass
class RebalanceExecution:
    """Rebalancing execution result"""
    timestamp: datetime
    signal: RebalanceSignal
    executed_weights: np.ndarray
    execution_time_ms: float
    transaction_costs: float
    slippage: float
    market_impact: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'executed_weights': self.executed_weights.tolist(),
            'execution_time_ms': self.execution_time_ms,
            'transaction_costs': self.transaction_costs,
            'slippage': self.slippage,
            'market_impact': self.market_impact,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class RebalanceConfig:
    """Rebalancing configuration parameters"""
    # Drift thresholds
    drift_threshold: float = 0.05  # 5% weight drift threshold
    emergency_drift_threshold: float = 0.15  # 15% emergency threshold
    
    # Time-based rebalancing
    rebalance_frequency_hours: int = 24  # Daily rebalancing
    min_rebalance_interval_minutes: int = 15  # Minimum time between rebalances
    
    # Transaction cost parameters
    transaction_cost_bps: float = 5.0  # 5 basis points per transaction
    market_impact_factor: float = 0.1  # Market impact coefficient
    
    # Performance thresholds
    max_response_time_ms: float = 10.0  # Target response time
    performance_monitor_window: int = 100  # Performance monitoring window
    
    # Risk thresholds
    volatility_threshold: float = 0.3  # 30% volatility threshold
    correlation_threshold: float = 0.8  # 80% correlation threshold
    
    # Optimization parameters
    use_async_processing: bool = True
    max_concurrent_calculations: int = 4
    enable_precomputation: bool = True
    cache_ttl_seconds: int = 60


class DynamicRebalancingEngine:
    """
    High-Performance Dynamic Rebalancing Engine with <10ms response time target.
    
    Implements ultra-low latency portfolio rebalancing with intelligent triggering,
    cost optimization, and real-time performance monitoring.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        n_strategies: int = 5,
        config: Optional[RebalanceConfig] = None
    ):
        """
        Initialize Dynamic Rebalancing Engine
        
        Args:
            event_bus: Event bus for real-time communication
            n_strategies: Number of strategies in portfolio
            config: Rebalancing configuration
        """
        self.event_bus = event_bus
        self.n_strategies = n_strategies
        self.config = config or RebalanceConfig()
        
        # Current state
        self.current_weights = np.array([1.0 / n_strategies] * n_strategies)
        self.target_weights = self.current_weights.copy()
        self.last_rebalance_time: Optional[datetime] = None
        self.last_signal_time: Optional[datetime] = None
        
        # Performance tracking
        self.response_times: List[float] = []
        self.execution_times: List[float] = []
        self.rebalance_count = 0
        self.performance_violations = 0
        
        # Rebalancing history
        self.signal_history: List[RebalanceSignal] = []
        self.execution_history: List[RebalanceExecution] = []
        
        # Market state monitoring
        self.market_volatility = 0.15  # Default 15% volatility
        self.correlation_regime = CorrelationRegime.NORMAL
        self.market_stress_level = 0.0  # 0-1 scale
        
        # Caching and precomputation
        self.optimization_cache: Dict[str, Tuple[datetime, np.ndarray]] = {}
        self.cost_cache: Dict[str, Tuple[datetime, float]] = {}
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_calculations)
        self.processing_lock = threading.Lock()
        
        # Market timing optimization
        self.optimal_execution_windows = self._initialize_execution_windows()
        
        # Emergency protocols
        self.emergency_mode = False
        self.emergency_protocols = {
            'max_position_size': 0.3,
            'force_diversification': True,
            'disable_cost_optimization': True
        }
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logger.info("Dynamic Rebalancing Engine initialized",
                   n_strategies=n_strategies,
                   target_response_time_ms=self.config.max_response_time_ms,
                   drift_threshold=self.config.drift_threshold)
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for real-time monitoring"""
        self.event_bus.subscribe(EventType.RISK_UPDATE, self._handle_risk_update)
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        self.event_bus.subscribe(EventType.VAR_UPDATE, self._handle_var_update)
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_breach)
    
    async def process_rebalance_signal(
        self,
        new_target_weights: np.ndarray,
        trigger: RebalanceTrigger = RebalanceTrigger.MANUAL,
        urgency: RebalanceUrgency = RebalanceUrgency.MEDIUM,
        reason: str = "Manual rebalancing"
    ) -> Optional[RebalanceExecution]:
        """
        Process rebalancing signal with <10ms response time target
        
        Args:
            new_target_weights: New target portfolio weights
            trigger: Rebalancing trigger type
            urgency: Rebalancing urgency level
            reason: Reason for rebalancing
            
        Returns:
            Rebalancing execution result
        """
        start_time = time.perf_counter()
        
        try:
            # Validate inputs quickly
            if not self._fast_validate_weights(new_target_weights):
                logger.warning("Invalid target weights provided")
                return None
            
            # Calculate weight drift
            weight_drift = new_target_weights - self.current_weights
            max_drift = np.max(np.abs(weight_drift))
            
            # Quick decision: is rebalancing needed?
            if not self._should_rebalance(max_drift, trigger, urgency):
                return None
            
            # Create rebalance signal
            signal = RebalanceSignal(
                timestamp=datetime.now(),
                trigger=trigger,
                urgency=urgency,
                current_weights=self.current_weights.copy(),
                target_weights=new_target_weights.copy(),
                weight_drift=weight_drift,
                max_drift=max_drift,
                expected_cost=0.0,  # Will be calculated
                risk_impact=0.0,   # Will be calculated
                reason=reason
            )
            
            # Store signal
            self.signal_history.append(signal)
            self.last_signal_time = datetime.now()
            
            # Fast-track critical rebalancing
            if urgency == RebalanceUrgency.CRITICAL:
                execution = await self._execute_critical_rebalance(signal)
            else:
                # Normal processing with optimization
                execution = await self._execute_optimized_rebalance(signal)
            
            # Track response time
            response_time = (time.perf_counter() - start_time) * 1000
            self.response_times.append(response_time)
            
            # Check performance target
            if response_time > self.config.max_response_time_ms:
                self.performance_violations += 1
                logger.warning("Response time exceeded target",
                              response_time_ms=response_time,
                              target_ms=self.config.max_response_time_ms)
            
            # Update current weights if successful
            if execution and execution.success:
                self.current_weights = execution.executed_weights.copy()
                self.target_weights = new_target_weights.copy()
                self.last_rebalance_time = datetime.now()
                self.rebalance_count += 1
                
                # Publish rebalancing event
                self._publish_rebalancing_event(execution)
            
            return execution
            
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            logger.error("Rebalancing signal processing failed",
                        error=str(e),
                        response_time_ms=response_time)
            return None
    
    async def _execute_critical_rebalance(self, signal: RebalanceSignal) -> RebalanceExecution:
        """Execute critical rebalancing with minimal latency"""
        start_time = time.perf_counter()
        
        try:
            # Emergency mode: bypass most optimizations
            executed_weights = signal.target_weights.copy()
            
            # Basic safety checks only
            executed_weights = self._apply_emergency_constraints(executed_weights)
            
            # Estimate costs quickly (simplified)
            transaction_costs = self._fast_estimate_costs(
                signal.current_weights, executed_weights
            )
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            execution = RebalanceExecution(
                timestamp=datetime.now(),
                signal=signal,
                executed_weights=executed_weights,
                execution_time_ms=execution_time,
                transaction_costs=transaction_costs,
                slippage=0.01,  # Estimated
                market_impact=0.005,  # Estimated
                success=True
            )
            
            self.execution_history.append(execution)
            self.execution_times.append(execution_time)
            
            logger.info("Critical rebalancing executed",
                       execution_time_ms=execution_time,
                       max_drift=signal.max_drift)
            
            return execution
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error("Critical rebalancing failed", error=str(e))
            
            return RebalanceExecution(
                timestamp=datetime.now(),
                signal=signal,
                executed_weights=signal.current_weights,
                execution_time_ms=execution_time,
                transaction_costs=0,
                slippage=0,
                market_impact=0,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_optimized_rebalance(self, signal: RebalanceSignal) -> RebalanceExecution:
        """Execute optimized rebalancing with cost and timing optimization"""
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(signal.target_weights)
            if cache_key in self.optimization_cache:
                cache_time, cached_weights = self.optimization_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl_seconds:
                    executed_weights = cached_weights
                    logger.debug("Using cached optimization result")
                else:
                    # Cache expired, optimize
                    executed_weights = await self._optimize_execution_weights(signal)
                    self.optimization_cache[cache_key] = (datetime.now(), executed_weights)
            else:
                # Not in cache, optimize
                executed_weights = await self._optimize_execution_weights(signal)
                self.optimization_cache[cache_key] = (datetime.now(), executed_weights)
            
            # Calculate transaction costs
            transaction_costs = await self._calculate_transaction_costs(
                signal.current_weights, executed_weights
            )
            
            # Estimate market impact
            market_impact = self._estimate_market_impact(
                signal.current_weights, executed_weights
            )
            
            # Calculate slippage
            slippage = self._estimate_slippage(executed_weights)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            execution = RebalanceExecution(
                timestamp=datetime.now(),
                signal=signal,
                executed_weights=executed_weights,
                execution_time_ms=execution_time,
                transaction_costs=transaction_costs,
                slippage=slippage,
                market_impact=market_impact,
                success=True
            )
            
            self.execution_history.append(execution)
            self.execution_times.append(execution_time)
            
            logger.info("Optimized rebalancing executed",
                       execution_time_ms=execution_time,
                       transaction_costs=transaction_costs,
                       market_impact=market_impact)
            
            return execution
            
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error("Optimized rebalancing failed", error=str(e))
            
            return RebalanceExecution(
                timestamp=datetime.now(),
                signal=signal,
                executed_weights=signal.current_weights,
                execution_time_ms=execution_time,
                transaction_costs=0,
                slippage=0,
                market_impact=0,
                success=False,
                error_message=str(e)
            )
    
    async def _optimize_execution_weights(self, signal: RebalanceSignal) -> np.ndarray:
        """Optimize execution weights considering costs and constraints"""
        
        target_weights = signal.target_weights
        current_weights = signal.current_weights
        
        # If urgency is high, minimize optimization
        if signal.urgency == RebalanceUrgency.HIGH:
            return self._apply_basic_constraints(target_weights)
        
        # Full optimization for lower urgency
        try:
            # Transaction cost optimization
            optimized_weights = await self._cost_optimized_weights(
                current_weights, target_weights
            )
            
            # Apply constraints
            optimized_weights = self._apply_constraints(optimized_weights)
            
            # Market timing adjustment
            if self._is_optimal_execution_time():
                return optimized_weights
            else:
                # Partial rebalancing in suboptimal times
                rebalance_fraction = self._get_optimal_rebalance_fraction()
                return current_weights + rebalance_fraction * (optimized_weights - current_weights)
        
        except Exception as e:
            logger.warning("Weight optimization failed, using basic constraints", error=str(e))
            return self._apply_basic_constraints(target_weights)
    
    async def _cost_optimized_weights(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> np.ndarray:
        """Optimize weights considering transaction costs"""
        
        # Simple cost optimization: minimize turnover
        if self.config.use_async_processing:
            # Async optimization
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._minimize_turnover,
                current_weights,
                target_weights
            )
        else:
            return self._minimize_turnover(current_weights, target_weights)
    
    def _minimize_turnover(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> np.ndarray:
        """Minimize portfolio turnover while approaching target"""
        
        # Calculate required turnover
        turnover = np.sum(np.abs(target_weights - current_weights))
        
        # If turnover is small, execute fully
        if turnover < 0.1:  # 10% total turnover threshold
            return target_weights
        
        # Otherwise, scale down the move to reduce costs
        cost_factor = max(0.5, 1.0 - turnover)  # Reduce aggressiveness with high turnover
        adjusted_weights = current_weights + cost_factor * (target_weights - current_weights)
        
        return adjusted_weights
    
    async def _calculate_transaction_costs(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> float:
        """Calculate expected transaction costs"""
        
        # Check cache
        cache_key = f"{hash(current_weights.tobytes())}_{hash(target_weights.tobytes())}"
        if cache_key in self.cost_cache:
            cache_time, cached_cost = self.cost_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.config.cache_ttl_seconds:
                return cached_cost
        
        # Calculate turnover
        turnover = np.sum(np.abs(target_weights - current_weights))
        
        # Base transaction cost
        base_cost = turnover * self.config.transaction_cost_bps / 10000.0
        
        # Market impact adjustment
        market_impact_cost = turnover**1.5 * self.config.market_impact_factor / 100.0
        
        total_cost = base_cost + market_impact_cost
        
        # Cache result
        self.cost_cache[cache_key] = (datetime.now(), total_cost)
        
        return total_cost
    
    def _estimate_market_impact(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray
    ) -> float:
        """Estimate market impact of rebalancing"""
        
        turnover = np.sum(np.abs(target_weights - current_weights))
        
        # Square-root impact model
        market_impact = np.sqrt(turnover) * self.config.market_impact_factor / 100.0
        
        # Adjust for market conditions
        stress_multiplier = 1.0 + self.market_stress_level
        
        return market_impact * stress_multiplier
    
    def _estimate_slippage(self, weights: np.ndarray) -> float:
        """Estimate execution slippage"""
        # Simplified slippage model
        portfolio_concentration = np.sum(weights**2)  # Herfindahl index
        base_slippage = 0.002  # 20 basis points base
        
        # Higher slippage for concentrated portfolios
        concentration_penalty = portfolio_concentration * 0.005
        
        return base_slippage + concentration_penalty
    
    def _should_rebalance(
        self,
        max_drift: float,
        trigger: RebalanceTrigger,
        urgency: RebalanceUrgency
    ) -> bool:
        """Fast decision on whether rebalancing is needed"""
        
        # Always rebalance critical and emergency triggers
        if urgency == RebalanceUrgency.CRITICAL or trigger == RebalanceTrigger.EMERGENCY:
            return True
        
        # Check minimum time interval
        if (self.last_rebalance_time and 
            (datetime.now() - self.last_rebalance_time).total_seconds() < 
            self.config.min_rebalance_interval_minutes * 60):
            return False
        
        # Check drift threshold
        if trigger == RebalanceTrigger.DRIFT_THRESHOLD:
            return max_drift > self.config.drift_threshold
        
        # Always allow manual rebalancing
        if trigger == RebalanceTrigger.MANUAL:
            return True
        
        # Default behavior
        return max_drift > self.config.drift_threshold
    
    def _fast_validate_weights(self, weights: np.ndarray) -> bool:
        """Fast validation of portfolio weights"""
        if weights is None or len(weights) != self.n_strategies:
            return False
        
        if np.any(weights < 0) or np.any(weights > 1):
            return False
        
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return False
        
        return True
    
    def _apply_emergency_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply emergency constraints to weights"""
        constrained_weights = weights.copy()
        
        # Maximum position size
        max_position = self.emergency_protocols['max_position_size']
        constrained_weights = np.minimum(constrained_weights, max_position)
        
        # Force diversification if needed
        if self.emergency_protocols['force_diversification']:
            min_weight = 0.05  # 5% minimum per strategy
            constrained_weights = np.maximum(constrained_weights, min_weight)
        
        # Renormalize
        constrained_weights /= np.sum(constrained_weights)
        
        return constrained_weights
    
    def _apply_basic_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply basic constraints to weights"""
        constrained_weights = np.clip(weights, 0.01, 0.8)  # 1% to 80% bounds
        constrained_weights /= np.sum(constrained_weights)
        return constrained_weights
    
    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Apply full constraints to weights"""
        # Weight bounds
        constrained_weights = np.clip(weights, 0.01, 0.5)
        
        # Concentration limits
        if np.max(constrained_weights) > 0.4:
            # Scale down large positions
            scale_factor = 0.4 / np.max(constrained_weights)
            constrained_weights *= scale_factor
        
        # Renormalize
        constrained_weights /= np.sum(constrained_weights)
        
        return constrained_weights
    
    def _is_optimal_execution_time(self) -> bool:
        """Check if current time is optimal for execution"""
        current_hour = datetime.now().hour
        
        # Avoid market open/close (simplified)
        if current_hour in [9, 15, 16]:  # Market open/close hours
            return False
        
        return True
    
    def _get_optimal_rebalance_fraction(self) -> float:
        """Get optimal fraction of rebalancing to execute now"""
        if self.market_stress_level > 0.5:
            return 0.3  # Partial rebalancing in stressed markets
        else:
            return 0.7  # More aggressive in normal markets
    
    def _initialize_execution_windows(self) -> Dict[str, List[int]]:
        """Initialize optimal execution time windows"""
        return {
            'normal': [10, 11, 12, 13, 14],  # Mid-day hours
            'stressed': [11, 12, 13],        # Narrow window in stress
            'emergency': list(range(24))     # Any time for emergency
        }
    
    def _generate_cache_key(self, weights: np.ndarray) -> str:
        """Generate cache key for optimization results"""
        return hash(weights.tobytes())
    
    def _handle_risk_update(self, event: Event):
        """Handle risk update events for rebalancing triggers"""
        risk_data = event.payload
        
        if isinstance(risk_data, dict):
            # Check for correlation shocks
            if risk_data.get('type') == 'CORRELATION_SHOCK':
                severity = risk_data.get('severity', 'MODERATE')
                urgency = RebalanceUrgency.HIGH if severity in ['HIGH', 'CRITICAL'] else RebalanceUrgency.MEDIUM
                
                # Trigger rebalancing (would need target weights from optimizer)
                logger.info("Correlation shock detected, may trigger rebalancing",
                           severity=severity, urgency=urgency.value)
    
    def _handle_position_update(self, event: Event):
        """Handle position updates"""
        # Update current weights from position data
        pass
    
    def _handle_var_update(self, event: Event):
        """Handle VaR updates"""
        # Monitor for VaR limit breaches
        pass
    
    def _handle_risk_breach(self, event: Event):
        """Handle risk breach events"""
        risk_data = event.payload
        
        if isinstance(risk_data, dict):
            breach_type = risk_data.get('type')
            if breach_type in ['VAR_LIMIT_BREACH', 'CORRELATION_SHOCK']:
                logger.warning("Risk breach detected, emergency rebalancing may be required",
                              breach_type=breach_type)
    
    def _publish_rebalancing_event(self, execution: RebalanceExecution):
        """Publish rebalancing execution event"""
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.RISK_UPDATE,
                {
                    'type': 'PORTFOLIO_REBALANCE_EXECUTED',
                    'execution': execution.to_dict(),
                    'new_weights': execution.executed_weights.tolist(),
                    'execution_time_ms': execution.execution_time_ms,
                    'transaction_costs': execution.transaction_costs,
                    'rebalance_count': self.rebalance_count
                },
                'DynamicRebalancingEngine'
            )
        )
    
    def get_performance_metrics(self) -> Dict:
        """Get rebalancing engine performance metrics"""
        
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
        
        success_rate = 0
        if self.execution_history:
            successful_executions = sum(1 for ex in self.execution_history if ex.success)
            success_rate = successful_executions / len(self.execution_history)
        
        return {
            "rebalance_count": self.rebalance_count,
            "avg_response_time_ms": avg_response_time,
            "max_response_time_ms": np.max(self.response_times) if self.response_times else 0,
            "target_response_time_ms": self.config.max_response_time_ms,
            "response_time_violations": self.performance_violations,
            "avg_execution_time_ms": avg_execution_time,
            "success_rate": success_rate,
            "signal_count": len(self.signal_history),
            "cache_hit_rate": len(self.optimization_cache) / max(1, len(self.signal_history)),
            "emergency_mode": self.emergency_mode,
            "current_weights": self.current_weights.tolist(),
            "target_weights": self.target_weights.tolist()
        }
    
    def enable_emergency_mode(self, reason: str):
        """Enable emergency rebalancing mode"""
        self.emergency_mode = True
        logger.critical("Emergency rebalancing mode enabled", reason=reason)
        
        # Publish emergency mode event
        self.event_bus.publish(
            self.event_bus.create_event(
                EventType.EMERGENCY_STOP,
                {
                    'type': 'EMERGENCY_REBALANCING_MODE',
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                },
                'DynamicRebalancingEngine'
            )
        )
    
    def disable_emergency_mode(self, reason: str):
        """Disable emergency rebalancing mode"""
        self.emergency_mode = False
        logger.info("Emergency rebalancing mode disabled", reason=reason)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("Dynamic Rebalancing Engine cleanup completed")