"""
Superposition Order Generator (Agent 7 Implementation)
==================================================

This module implements the quantum-inspired superposition order generator that
converts multi-layered MARL superposition states into executable market orders.

Key Features:
- Quantum-inspired superposition collapse to definite orders
- Multi-layer coherence analysis (strategic, tactical, risk)
- Entanglement-aware order sizing and timing
- Coherence-based confidence scoring
- Market microstructure integration
- Real-time order adaptation

Superposition Processing:
1. Collect superposition states from all MARL layers
2. Analyze coherence and entanglement patterns
3. Apply quantum-inspired collapse mechanisms
4. Generate executable order parameters
5. Integrate market microstructure constraints
6. Optimize execution timing and sizing

Author: Claude Code (Agent 7 Mission)
Version: 1.0
Date: 2025-07-17
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from scipy.optimize import minimize
from scipy.stats import entropy

logger = structlog.get_logger()


class OrderType(Enum):
    """Order types for execution"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"
    IMPLEMENTATION_SHORTFALL = "IS"


class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"


class TimeInForce(Enum):
    """Time in force options"""
    DAY = "DAY"
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTC = "GTC"  # Good Till Canceled
    GTD = "GTD"  # Good Till Date


@dataclass
class SuperpositionState:
    """Superposition state from MARL layer"""
    probabilities: np.ndarray  # [buy, hold, sell] or [bearish, neutral, bullish]
    confidence: float
    coherence: float
    layer_id: str
    timestamp: datetime
    
    def entropy(self) -> float:
        """Calculate quantum entropy of superposition"""
        return entropy(self.probabilities + 1e-10)
    
    def dominant_state(self) -> int:
        """Get dominant state index"""
        return np.argmax(self.probabilities)
    
    def is_coherent(self, threshold: float = 0.7) -> bool:
        """Check if superposition is coherent"""
        return self.coherence >= threshold


@dataclass
class EntanglementMetrics:
    """Entanglement metrics between layers"""
    strategic_tactical: float = 0.0
    tactical_risk: float = 0.0
    strategic_risk: float = 0.0
    
    def overall_entanglement(self) -> float:
        """Calculate overall entanglement score"""
        return (self.strategic_tactical + self.tactical_risk + self.strategic_risk) / 3.0


@dataclass
class OrderParameters:
    """Generated order parameters"""
    # Core order details
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.IOC
    
    # Execution parameters
    venue: str = "SMART"
    max_participation_rate: float = 0.1
    target_completion_time: float = 300.0  # seconds
    
    # Risk parameters
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    max_slippage_bps: float = 10.0
    
    # Quantum parameters
    superposition_confidence: float = 0.0
    coherence_score: float = 0.0
    entanglement_score: float = 0.0
    collapse_probability: float = 0.0
    
    # Metadata
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation_timestamp: datetime = field(default_factory=datetime.now)
    generator_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value,
            'venue': self.venue,
            'max_participation_rate': self.max_participation_rate,
            'target_completion_time': self.target_completion_time,
            'stop_loss_level': self.stop_loss_level,
            'take_profit_level': self.take_profit_level,
            'max_slippage_bps': self.max_slippage_bps,
            'superposition_confidence': self.superposition_confidence,
            'coherence_score': self.coherence_score,
            'entanglement_score': self.entanglement_score,
            'collapse_probability': self.collapse_probability,
            'generation_timestamp': self.generation_timestamp.isoformat(),
            'generator_version': self.generator_version
        }


@dataclass
class MarketContext:
    """Market context for order generation"""
    current_price: float
    bid_price: float
    ask_price: float
    spread_bps: float
    volatility: float
    volume: float
    market_impact: float
    liquidity_depth: float
    
    # Venue information
    venue_spreads: Dict[str, float] = field(default_factory=dict)
    venue_liquidity: Dict[str, float] = field(default_factory=dict)
    venue_latency: Dict[str, float] = field(default_factory=dict)
    
    def effective_spread(self, venue: str = "SMART") -> float:
        """Get effective spread for venue"""
        return self.venue_spreads.get(venue, self.spread_bps)
    
    def available_liquidity(self, venue: str = "SMART") -> float:
        """Get available liquidity for venue"""
        return self.venue_liquidity.get(venue, self.liquidity_depth)


class QuantumCollapseEngine:
    """Quantum-inspired collapse engine for superposition states"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collapse_threshold = config.get('collapse_threshold', 0.6)
        self.coherence_weight = config.get('coherence_weight', 0.3)
        self.entanglement_weight = config.get('entanglement_weight', 0.2)
        
    def collapse_superposition(self, 
                             superposition_states: List[SuperpositionState],
                             entanglement_metrics: EntanglementMetrics) -> Tuple[int, float]:
        """
        Collapse superposition states to definite action
        
        Args:
            superposition_states: List of superposition states from all layers
            entanglement_metrics: Entanglement metrics between layers
            
        Returns:
            Tuple of (collapsed_action, collapse_confidence)
        """
        if not superposition_states:
            return 1, 0.0  # Default to hold with no confidence
        
        # Calculate weighted probabilities
        weighted_probs = self._calculate_weighted_probabilities(
            superposition_states, entanglement_metrics
        )
        
        # Apply quantum collapse mechanism
        collapsed_action = self._apply_collapse_mechanism(weighted_probs)
        
        # Calculate collapse confidence
        collapse_confidence = self._calculate_collapse_confidence(
            superposition_states, entanglement_metrics, collapsed_action
        )
        
        return collapsed_action, collapse_confidence
    
    def _calculate_weighted_probabilities(self, 
                                        superposition_states: List[SuperpositionState],
                                        entanglement_metrics: EntanglementMetrics) -> np.ndarray:
        """Calculate weighted probabilities from all layers"""
        if not superposition_states:
            return np.array([0.33, 0.34, 0.33])  # Uniform distribution
        
        # Start with uniform weights
        weights = np.ones(len(superposition_states)) / len(superposition_states)
        
        # Adjust weights based on coherence
        for i, state in enumerate(superposition_states):
            if state.is_coherent():
                weights[i] *= (1 + self.coherence_weight)
            else:
                weights[i] *= (1 - self.coherence_weight)
        
        # Adjust weights based on entanglement
        overall_entanglement = entanglement_metrics.overall_entanglement()
        if overall_entanglement > 0.5:  # High entanglement
            # Increase weight for layers with high entanglement
            for i, state in enumerate(superposition_states):
                if state.layer_id in ['strategic', 'tactical']:
                    weights[i] *= (1 + self.entanglement_weight * overall_entanglement)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted average probabilities
        weighted_probs = np.zeros(3)
        for i, state in enumerate(superposition_states):
            weighted_probs += weights[i] * state.probabilities
        
        return weighted_probs
    
    def _apply_collapse_mechanism(self, weighted_probs: np.ndarray) -> int:
        """Apply quantum collapse mechanism"""
        # Check if any probability is above collapse threshold
        max_prob = np.max(weighted_probs)
        max_idx = np.argmax(weighted_probs)
        
        if max_prob >= self.collapse_threshold:
            return max_idx
        
        # If no clear winner, use probabilistic collapse
        return np.random.choice(len(weighted_probs), p=weighted_probs)
    
    def _calculate_collapse_confidence(self, 
                                     superposition_states: List[SuperpositionState],
                                     entanglement_metrics: EntanglementMetrics,
                                     collapsed_action: int) -> float:
        """Calculate confidence in collapse decision"""
        if not superposition_states:
            return 0.0
        
        # Base confidence from probability mass
        weighted_probs = self._calculate_weighted_probabilities(
            superposition_states, entanglement_metrics
        )
        base_confidence = weighted_probs[collapsed_action]
        
        # Adjust for coherence
        avg_coherence = np.mean([state.coherence for state in superposition_states])
        coherence_bonus = avg_coherence * 0.2
        
        # Adjust for entanglement
        entanglement_score = entanglement_metrics.overall_entanglement()
        entanglement_bonus = entanglement_score * 0.1
        
        # Adjust for consensus
        consensus_score = self._calculate_consensus_score(superposition_states, collapsed_action)
        consensus_bonus = consensus_score * 0.2
        
        total_confidence = base_confidence + coherence_bonus + entanglement_bonus + consensus_bonus
        
        return min(1.0, total_confidence)
    
    def _calculate_consensus_score(self, 
                                 superposition_states: List[SuperpositionState],
                                 collapsed_action: int) -> float:
        """Calculate consensus score across layers"""
        if not superposition_states:
            return 0.0
        
        agreements = 0
        for state in superposition_states:
            if state.dominant_state() == collapsed_action:
                agreements += 1
        
        return agreements / len(superposition_states)


class OrderSizingEngine:
    """Engine for calculating optimal order sizes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.2)
        self.kelly_lookback = config.get('kelly_lookback', 252)
        self.risk_free_rate = config.get('risk_free_rate', 0.02)
        
    def calculate_optimal_size(self, 
                             collapsed_action: int,
                             collapse_confidence: float,
                             market_context: MarketContext,
                             risk_allocation: float) -> float:
        """
        Calculate optimal order size
        
        Args:
            collapsed_action: Collapsed action (0=sell, 1=hold, 2=buy)
            collapse_confidence: Confidence in collapse
            market_context: Market context
            risk_allocation: Risk allocation from upstream
            
        Returns:
            Optimal order size
        """
        if collapsed_action == 1:  # Hold
            return 0.0
        
        # Base size from risk allocation
        base_size = abs(risk_allocation)
        
        # Apply Kelly criterion adjustment
        kelly_size = self._calculate_kelly_size(
            collapse_confidence, market_context.volatility
        )
        
        # Apply market impact adjustment
        impact_adjustment = self._calculate_impact_adjustment(
            base_size, market_context
        )
        
        # Apply liquidity constraints
        liquidity_adjustment = self._calculate_liquidity_adjustment(
            base_size, market_context
        )
        
        # Combine adjustments
        optimal_size = base_size * kelly_size * impact_adjustment * liquidity_adjustment
        
        # Apply position limits
        optimal_size = min(optimal_size, self.max_position_size)
        
        return optimal_size
    
    def _calculate_kelly_size(self, confidence: float, volatility: float) -> float:
        """Calculate Kelly criterion size adjustment"""
        # Simplified Kelly formula
        # f* = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        
        win_prob = confidence
        loss_prob = 1 - confidence
        
        # Estimate odds from volatility (simplified)
        odds = 1 / max(volatility, 0.01)
        
        kelly_fraction = (odds * win_prob - loss_prob) / odds
        kelly_fraction = max(0, min(kelly_fraction, 1.0))  # Clamp to [0, 1]
        
        # Apply conservative scaling
        return kelly_fraction * 0.5  # 50% of full Kelly
    
    def _calculate_impact_adjustment(self, size: float, market_context: MarketContext) -> float:
        """Calculate market impact adjustment"""
        if size <= 0:
            return 1.0
        
        # Estimate market impact
        participation_rate = size / market_context.volume
        impact_factor = market_context.market_impact
        
        # Square root impact model
        estimated_impact = impact_factor * np.sqrt(participation_rate)
        
        # Reduce size if impact is too high
        if estimated_impact > 0.01:  # 100 bps threshold
            return 0.5  # Reduce by 50%
        elif estimated_impact > 0.005:  # 50 bps threshold
            return 0.8  # Reduce by 20%
        else:
            return 1.0  # No reduction
    
    def _calculate_liquidity_adjustment(self, size: float, market_context: MarketContext) -> float:
        """Calculate liquidity adjustment"""
        if size <= 0:
            return 1.0
        
        # Check if size exceeds available liquidity
        available_liquidity = market_context.liquidity_depth
        
        if size > available_liquidity:
            # Reduce size to fit available liquidity
            return available_liquidity / size
        else:
            return 1.0


class OrderTimingEngine:
    """Engine for optimal order timing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_urgency = config.get('base_urgency', 0.5)
        self.volatility_factor = config.get('volatility_factor', 2.0)
        self.spread_factor = config.get('spread_factor', 1.5)
        
    def calculate_optimal_timing(self, 
                               collapsed_action: int,
                               collapse_confidence: float,
                               market_context: MarketContext,
                               entanglement_metrics: EntanglementMetrics) -> Tuple[float, OrderType]:
        """
        Calculate optimal order timing and type
        
        Args:
            collapsed_action: Collapsed action
            collapse_confidence: Confidence in collapse
            market_context: Market context
            entanglement_metrics: Entanglement metrics
            
        Returns:
            Tuple of (urgency_score, optimal_order_type)
        """
        if collapsed_action == 1:  # Hold
            return 0.0, OrderType.MARKET
        
        # Base urgency from confidence
        urgency = self.base_urgency + (collapse_confidence - 0.5) * 0.5
        
        # Adjust for market conditions
        volatility_adjustment = self._calculate_volatility_adjustment(
            market_context.volatility
        )
        
        spread_adjustment = self._calculate_spread_adjustment(
            market_context.spread_bps
        )
        
        entanglement_adjustment = self._calculate_entanglement_adjustment(
            entanglement_metrics
        )
        
        # Combine adjustments
        final_urgency = urgency * volatility_adjustment * spread_adjustment * entanglement_adjustment
        final_urgency = max(0.0, min(1.0, final_urgency))  # Clamp to [0, 1]
        
        # Determine order type based on urgency
        optimal_order_type = self._determine_order_type(
            final_urgency, market_context
        )
        
        return final_urgency, optimal_order_type
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate volatility adjustment for timing"""
        if volatility > 0.3:  # High volatility
            return 1.5  # Increase urgency
        elif volatility > 0.2:  # Medium volatility
            return 1.2
        elif volatility < 0.1:  # Low volatility
            return 0.8  # Decrease urgency
        else:
            return 1.0
    
    def _calculate_spread_adjustment(self, spread_bps: float) -> float:
        """Calculate spread adjustment for timing"""
        if spread_bps > 20:  # Wide spread
            return 0.7  # Decrease urgency, wait for better prices
        elif spread_bps > 10:  # Medium spread
            return 0.9
        else:
            return 1.0
    
    def _calculate_entanglement_adjustment(self, entanglement_metrics: EntanglementMetrics) -> float:
        """Calculate entanglement adjustment for timing"""
        overall_entanglement = entanglement_metrics.overall_entanglement()
        
        if overall_entanglement > 0.7:  # High entanglement
            return 1.3  # Increase urgency due to coordinated signal
        elif overall_entanglement > 0.5:  # Medium entanglement
            return 1.1
        else:
            return 1.0
    
    def _determine_order_type(self, urgency: float, market_context: MarketContext) -> OrderType:
        """Determine optimal order type based on urgency and market context"""
        if urgency > 0.8:
            return OrderType.MARKET  # High urgency, use market order
        elif urgency > 0.6:
            return OrderType.LIMIT  # Medium urgency, use limit order
        elif urgency > 0.4:
            return OrderType.TWAP  # Low urgency, use TWAP
        else:
            return OrderType.VWAP  # Very low urgency, use VWAP


class SuperpositionOrderGenerator:
    """
    Main superposition order generator
    
    Converts multi-layer MARL superposition states into executable market orders
    using quantum-inspired collapse mechanisms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize engines
        self.collapse_engine = QuantumCollapseEngine(self.config)
        self.sizing_engine = OrderSizingEngine(self.config)
        self.timing_engine = OrderTimingEngine(self.config)
        
        # Performance tracking
        self.generation_history = []
        self.performance_metrics = {
            'orders_generated': 0,
            'successful_collapses': 0,
            'average_confidence': 0.0,
            'average_coherence': 0.0,
            'average_entanglement': 0.0
        }
        
        logger.info("SuperpositionOrderGenerator initialized",
                   config_keys=list(self.config.keys()))
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'collapse_threshold': 0.6,
            'coherence_weight': 0.3,
            'entanglement_weight': 0.2,
            'max_position_size': 0.2,
            'kelly_lookback': 252,
            'risk_free_rate': 0.02,
            'base_urgency': 0.5,
            'volatility_factor': 2.0,
            'spread_factor': 1.5,
            'default_symbol': 'SPY',
            'default_venue': 'SMART',
            'default_tif': 'IOC',
            'max_slippage_bps': 10.0
        }
    
    def generate_order(self, 
                      strategic_superposition: SuperpositionState,
                      tactical_superposition: SuperpositionState,
                      risk_superposition: SuperpositionState,
                      entanglement_metrics: EntanglementMetrics,
                      market_context: MarketContext,
                      risk_allocation: float = 0.1,
                      symbol: str = "SPY") -> Optional[OrderParameters]:
        """
        Generate order from superposition states
        
        Args:
            strategic_superposition: Strategic layer superposition
            tactical_superposition: Tactical layer superposition
            risk_superposition: Risk layer superposition
            entanglement_metrics: Entanglement metrics
            market_context: Market context
            risk_allocation: Risk allocation
            symbol: Trading symbol
            
        Returns:
            OrderParameters or None if no order should be generated
        """
        generation_start = time.time()
        
        try:
            # Collect all superposition states
            superposition_states = [
                strategic_superposition,
                tactical_superposition,
                risk_superposition
            ]
            
            # Collapse superposition to definite action
            collapsed_action, collapse_confidence = self.collapse_engine.collapse_superposition(
                superposition_states, entanglement_metrics
            )
            
            # Check if we should generate an order
            if collapsed_action == 1:  # Hold action
                logger.debug("Superposition collapsed to HOLD, no order generated")
                return None
            
            if collapse_confidence < 0.3:  # Low confidence
                logger.debug("Collapse confidence too low, no order generated",
                           confidence=collapse_confidence)
                return None
            
            # Calculate optimal order size
            optimal_size = self.sizing_engine.calculate_optimal_size(
                collapsed_action, collapse_confidence, market_context, risk_allocation
            )
            
            if optimal_size < 0.001:  # Minimum size threshold
                logger.debug("Optimal size too small, no order generated",
                           size=optimal_size)
                return None
            
            # Calculate optimal timing
            urgency, optimal_order_type = self.timing_engine.calculate_optimal_timing(
                collapsed_action, collapse_confidence, market_context, entanglement_metrics
            )
            
            # Determine order side
            order_side = OrderSide.BUY if collapsed_action == 2 else OrderSide.SELL
            
            # Calculate price based on order type
            order_price = self._calculate_order_price(
                optimal_order_type, order_side, market_context, urgency
            )
            
            # Generate order parameters
            order_params = OrderParameters(
                symbol=symbol,
                side=order_side,
                quantity=optimal_size,
                order_type=optimal_order_type,
                price=order_price,
                time_in_force=TimeInForce.IOC,
                venue=self.config['default_venue'],
                max_participation_rate=min(0.2, optimal_size / market_context.volume),
                target_completion_time=self._calculate_completion_time(urgency),
                max_slippage_bps=self.config['max_slippage_bps'],
                superposition_confidence=collapse_confidence,
                coherence_score=self._calculate_overall_coherence(superposition_states),
                entanglement_score=entanglement_metrics.overall_entanglement(),
                collapse_probability=collapse_confidence
            )
            
            # Update performance metrics
            self._update_performance_metrics(
                superposition_states, entanglement_metrics, collapse_confidence
            )
            
            # Store generation history
            generation_time = (time.time() - generation_start) * 1000  # Convert to ms
            self.generation_history.append({
                'timestamp': datetime.now(),
                'order_params': order_params,
                'generation_time_ms': generation_time,
                'collapsed_action': collapsed_action,
                'collapse_confidence': collapse_confidence
            })
            
            # Keep only recent history
            if len(self.generation_history) > 1000:
                self.generation_history.pop(0)
            
            logger.info("Order generated from superposition",
                       order_id=order_params.order_id,
                       side=order_params.side.value,
                       quantity=order_params.quantity,
                       order_type=order_params.order_type.value,
                       confidence=collapse_confidence,
                       generation_time_ms=generation_time)
            
            return order_params
            
        except Exception as e:
            logger.error("Error generating order from superposition",
                        error=str(e))
            return None
    
    def _calculate_order_price(self, 
                             order_type: OrderType,
                             order_side: OrderSide,
                             market_context: MarketContext,
                             urgency: float) -> Optional[float]:
        """Calculate order price based on type and market conditions"""
        if order_type == OrderType.MARKET:
            return None  # Market orders don't have price
        
        # Base price
        if order_side == OrderSide.BUY:
            base_price = market_context.bid_price
        else:
            base_price = market_context.ask_price
        
        # Adjust for urgency
        if order_type == OrderType.LIMIT:
            if order_side == OrderSide.BUY:
                # For buy orders, increase price for higher urgency
                price_adjustment = urgency * (market_context.spread_bps / 10000) * market_context.current_price
                return base_price + price_adjustment
            else:
                # For sell orders, decrease price for higher urgency
                price_adjustment = urgency * (market_context.spread_bps / 10000) * market_context.current_price
                return base_price - price_adjustment
        
        return base_price
    
    def _calculate_completion_time(self, urgency: float) -> float:
        """Calculate target completion time based on urgency"""
        # High urgency = short completion time
        base_time = 300.0  # 5 minutes
        urgency_factor = 2.0 - urgency  # Range from 1.0 to 2.0
        
        return base_time * urgency_factor
    
    def _calculate_overall_coherence(self, superposition_states: List[SuperpositionState]) -> float:
        """Calculate overall coherence score"""
        if not superposition_states:
            return 0.0
        
        return np.mean([state.coherence for state in superposition_states])
    
    def _update_performance_metrics(self, 
                                  superposition_states: List[SuperpositionState],
                                  entanglement_metrics: EntanglementMetrics,
                                  collapse_confidence: float):
        """Update performance metrics"""
        self.performance_metrics['orders_generated'] += 1
        
        if collapse_confidence > 0.5:
            self.performance_metrics['successful_collapses'] += 1
        
        # Update averages
        n = self.performance_metrics['orders_generated']
        old_avg_conf = self.performance_metrics['average_confidence']
        self.performance_metrics['average_confidence'] = (
            (n - 1) * old_avg_conf + collapse_confidence
        ) / n
        
        old_avg_coherence = self.performance_metrics['average_coherence']
        avg_coherence = self._calculate_overall_coherence(superposition_states)
        self.performance_metrics['average_coherence'] = (
            (n - 1) * old_avg_coherence + avg_coherence
        ) / n
        
        old_avg_entanglement = self.performance_metrics['average_entanglement']
        entanglement_score = entanglement_metrics.overall_entanglement()
        self.performance_metrics['average_entanglement'] = (
            (n - 1) * old_avg_entanglement + entanglement_score
        ) / n
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        success_rate = 0.0
        if self.performance_metrics['orders_generated'] > 0:
            success_rate = (
                self.performance_metrics['successful_collapses'] / 
                self.performance_metrics['orders_generated']
            )
        
        return {
            **self.performance_metrics,
            'success_rate': success_rate,
            'recent_generations': len(self.generation_history)
        }
    
    def get_generation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent generation history"""
        return self.generation_history[-limit:]


# Example usage and testing
if __name__ == "__main__":
    # Example superposition states
    strategic_superposition = SuperpositionState(
        probabilities=np.array([0.2, 0.3, 0.5]),  # Buy bias
        confidence=0.8,
        coherence=0.7,
        layer_id="strategic",
        timestamp=datetime.now()
    )
    
    tactical_superposition = SuperpositionState(
        probabilities=np.array([0.1, 0.2, 0.7]),  # Strong buy bias
        confidence=0.9,
        coherence=0.8,
        layer_id="tactical",
        timestamp=datetime.now()
    )
    
    risk_superposition = SuperpositionState(
        probabilities=np.array([0.3, 0.4, 0.3]),  # Neutral
        confidence=0.6,
        coherence=0.5,
        layer_id="risk",
        timestamp=datetime.now()
    )
    
    # Example entanglement metrics
    entanglement_metrics = EntanglementMetrics(
        strategic_tactical=0.6,
        tactical_risk=0.4,
        strategic_risk=0.3
    )
    
    # Example market context
    market_context = MarketContext(
        current_price=100.0,
        bid_price=99.95,
        ask_price=100.05,
        spread_bps=10.0,
        volatility=0.15,
        volume=1000.0,
        market_impact=0.001,
        liquidity_depth=0.8
    )
    
    # Create generator
    generator = SuperpositionOrderGenerator()
    
    # Generate order
    order = generator.generate_order(
        strategic_superposition,
        tactical_superposition,
        risk_superposition,
        entanglement_metrics,
        market_context,
        risk_allocation=0.1,
        symbol="SPY"
    )
    
    if order:
        print("Generated Order:")
        print(f"  Order ID: {order.order_id}")
        print(f"  Symbol: {order.symbol}")
        print(f"  Side: {order.side.value}")
        print(f"  Quantity: {order.quantity:.6f}")
        print(f"  Order Type: {order.order_type.value}")
        print(f"  Price: {order.price}")
        print(f"  Confidence: {order.superposition_confidence:.3f}")
        print(f"  Coherence: {order.coherence_score:.3f}")
        print(f"  Entanglement: {order.entanglement_score:.3f}")
        
        # Convert to dictionary
        order_dict = order.to_dict()
        print(f"\\nOrder Dictionary: {order_dict}")
    else:
        print("No order generated")
    
    # Performance metrics
    metrics = generator.get_performance_metrics()
    print(f"\\nPerformance Metrics: {metrics}")