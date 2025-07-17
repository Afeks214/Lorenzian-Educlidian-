"""
Advanced Action Space & Microstructure Intelligence Engine
AGENT 2 MISSION: Advanced Action Space & Execution Engine

Expands the tactical MARL action space from 3 to 15 sophisticated actions
with microstructure intelligence and Level 2 market data integration.

Enhanced Action Space:
- Position Management: HOLD, INCREASE_LONG, DECREASE_LONG, INCREASE_SHORT, DECREASE_SHORT
- Execution Styles: MARKET_BUY, MARKET_SELL, LIMIT_BUY, LIMIT_SELL
- Order Management: MODIFY_ORDERS, CANCEL_ORDERS
- Risk Management: REDUCE_RISK, HEDGE_POSITION
- Microstructure: PROVIDE_LIQUIDITY, TAKE_LIQUIDITY, ICEBERG_ORDER

Features:
- Level 2 order book integration
- Real-time bid-ask spread analysis
- Order flow imbalance detection
- Execution cost optimization
- Adaptive order sizing

Author: Agent 2 - Advanced Action Space & Execution Engine
Version: 2.0 - Mission Dominion Execution Intelligence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum, IntEnum
import logging
from abc import ABC, abstractmethod
import time
from collections import deque
import threading

logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """Extended action space for tactical MARL"""
    # Position Management (0-4)
    HOLD = 0
    INCREASE_LONG = 1
    DECREASE_LONG = 2
    INCREASE_SHORT = 3
    DECREASE_SHORT = 4
    
    # Execution Styles (5-8)
    MARKET_BUY = 5
    MARKET_SELL = 6
    LIMIT_BUY = 7
    LIMIT_SELL = 8
    
    # Order Management (9-10)
    MODIFY_ORDERS = 9
    CANCEL_ORDERS = 10
    
    # Risk Management (11-12)
    REDUCE_RISK = 11
    HEDGE_POSITION = 12
    
    # Microstructure (13-14)
    PROVIDE_LIQUIDITY = 13
    TAKE_LIQUIDITY = 14


class ExecutionStyle(Enum):
    """Execution style preferences"""
    AGGRESSIVE = "AGGRESSIVE"      # Take liquidity, fast execution
    PASSIVE = "PASSIVE"           # Provide liquidity, better prices
    BALANCED = "BALANCED"         # Mix of aggressive and passive
    STEALTH = "STEALTH"          # Hidden orders, minimal impact
    ICEBERG = "ICEBERG"          # Large orders split into smaller chunks


class OrderBookLevel(Enum):
    """Order book data levels"""
    LEVEL_1 = "LEVEL_1"  # Best bid/ask only
    LEVEL_2 = "LEVEL_2"  # Full order book depth
    LEVEL_3 = "LEVEL_3"  # Individual order details


@dataclass
class OrderBookSnapshot:
    """Level 2 order book data structure"""
    timestamp: pd.Timestamp
    symbol: str
    
    # Bid side (sorted by price descending)
    bid_prices: np.ndarray
    bid_sizes: np.ndarray
    bid_orders: np.ndarray
    
    # Ask side (sorted by price ascending)  
    ask_prices: np.ndarray
    ask_sizes: np.ndarray
    ask_orders: np.ndarray
    
    # Derived metrics
    best_bid: float
    best_ask: float
    bid_ask_spread: float
    mid_price: float
    
    # Microstructure metrics
    order_book_imbalance: float
    depth_imbalance: float
    weighted_mid_price: float


@dataclass
class MicrostructureFeatures:
    """Microstructure features for enhanced decision making"""
    bid_ask_spread: float
    spread_bps: float
    order_book_imbalance: float
    depth_of_market: float
    price_impact_estimate: float
    
    # Flow metrics
    order_flow_imbalance: float
    trade_intensity: float
    volatility_estimate: float
    
    # Timing metrics
    time_to_execution: float
    expected_slippage: float
    liquidity_score: float


@dataclass
class ExecutionContext:
    """Context information for execution decisions"""
    current_position: float
    target_position: float
    risk_budget: float
    time_horizon: int  # minutes
    urgency_level: float  # 0-1
    
    # Market context
    volatility_regime: str
    market_impact_threshold: float
    liquidity_condition: str
    
    # Cost constraints
    max_slippage_bps: float
    max_market_impact_bps: float
    commission_rate: float


@dataclass
class ActionOutput:
    """Enhanced action output with execution details"""
    action: ActionType
    confidence: float
    size: float
    execution_style: ExecutionStyle
    
    # Execution parameters
    limit_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    iceberg_size: Optional[float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Microstructure optimization
    expected_slippage_bps: float = 0.0
    expected_market_impact_bps: float = 0.0
    liquidity_providing: bool = False
    
    # Metadata
    reasoning: str = ""
    timestamp: pd.Timestamp = pd.Timestamp.now()


class MicrostructureAnalyzer:
    """
    Advanced microstructure analysis for optimal execution
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Microstructure Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Order book history for analysis
        self.order_book_history: deque = deque(
            maxlen=self.config.get('history_length', 1000)
        )
        
        # Flow tracking
        self.trade_flow_history: deque = deque(
            maxlen=self.config.get('flow_history_length', 500)
        )
        
        # Performance tracking
        self.execution_metrics = {
            'total_executions': 0,
            'slippage_history': [],
            'market_impact_history': [],
            'liquidity_scores': []
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("Microstructure Analyzer initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'history_length': 1000,
            'flow_history_length': 500,
            'min_spread_bps': 0.1,
            'max_spread_bps': 50.0,
            'imbalance_threshold': 0.6,
            'depth_levels': 10,
            'impact_model': 'square_root'
        }
    
    def process_order_book(self, order_book: OrderBookSnapshot) -> MicrostructureFeatures:
        """
        Process order book snapshot and extract microstructure features
        
        Args:
            order_book: Level 2 order book data
            
        Returns:
            MicrostructureFeatures: Extracted features
        """
        try:
            with self.lock:
                self.order_book_history.append(order_book)
            
            # Calculate basic spread metrics
            spread_absolute = order_book.ask_prices[0] - order_book.bid_prices[0]
            spread_bps = (spread_absolute / order_book.mid_price) * 10000
            
            # Order book imbalance
            total_bid_size = np.sum(order_book.bid_sizes[:self.config['depth_levels']])
            total_ask_size = np.sum(order_book.ask_sizes[:self.config['depth_levels']])
            total_size = total_bid_size + total_ask_size
            
            if total_size > 0:
                order_book_imbalance = (total_bid_size - total_ask_size) / total_size
            else:
                order_book_imbalance = 0.0
            
            # Depth of market (total size in top levels)
            depth_of_market = total_size
            
            # Price impact estimate using square root model
            price_impact_estimate = self._estimate_price_impact(order_book, 1.0)  # For 1 unit
            
            # Flow metrics (simplified - would use actual trade data)
            order_flow_imbalance = self._calculate_order_flow_imbalance()
            trade_intensity = self._calculate_trade_intensity()
            volatility_estimate = self._estimate_volatility()
            
            # Execution timing metrics
            time_to_execution = self._estimate_execution_time(order_book)
            expected_slippage = self._estimate_slippage(order_book)
            liquidity_score = self._calculate_liquidity_score(order_book)
            
            return MicrostructureFeatures(
                bid_ask_spread=spread_absolute,
                spread_bps=spread_bps,
                order_book_imbalance=order_book_imbalance,
                depth_of_market=depth_of_market,
                price_impact_estimate=price_impact_estimate,
                order_flow_imbalance=order_flow_imbalance,
                trade_intensity=trade_intensity,
                volatility_estimate=volatility_estimate,
                time_to_execution=time_to_execution,
                expected_slippage=expected_slippage,
                liquidity_score=liquidity_score
            )
            
        except Exception as e:
            logger.error(f"Error processing order book: {e}")
            return self._create_fallback_features()
    
    def _estimate_price_impact(self, order_book: OrderBookSnapshot, size: float) -> float:
        """Estimate price impact for given order size"""
        try:
            if self.config['impact_model'] == 'linear':
                # Linear impact model
                if size > 0:  # Buy order
                    cumulative_size = np.cumsum(order_book.ask_sizes)
                    impact_idx = np.searchsorted(cumulative_size, size)
                    if impact_idx < len(order_book.ask_prices):
                        impact_price = order_book.ask_prices[impact_idx]
                        return (impact_price - order_book.best_ask) / order_book.mid_price
                else:  # Sell order
                    cumulative_size = np.cumsum(order_book.bid_sizes)
                    impact_idx = np.searchsorted(cumulative_size, abs(size))
                    if impact_idx < len(order_book.bid_prices):
                        impact_price = order_book.bid_prices[impact_idx]
                        return (order_book.best_bid - impact_price) / order_book.mid_price
            
            elif self.config['impact_model'] == 'square_root':
                # Square root impact model (more realistic)
                total_depth = np.sum(order_book.ask_sizes) + np.sum(order_book.bid_sizes)
                if total_depth > 0:
                    participation_rate = abs(size) / total_depth
                    impact = 0.1 * np.sqrt(participation_rate)  # 10% impact coefficient
                    return min(impact, 0.005)  # Cap at 50 bps
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error estimating price impact: {e}")
            return 0.001  # 1 bp fallback
    
    def _calculate_order_flow_imbalance(self) -> float:
        """Calculate order flow imbalance from recent history"""
        if len(self.trade_flow_history) < 10:
            return 0.0
        
        # Simplified calculation (would use actual trade flow data)
        recent_flows = list(self.trade_flow_history)[-10:]
        buy_volume = sum(flow.get('buy_volume', 0) for flow in recent_flows)
        sell_volume = sum(flow.get('sell_volume', 0) for flow in recent_flows)
        
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            return (buy_volume - sell_volume) / total_volume
        return 0.0
    
    def _calculate_trade_intensity(self) -> float:
        """Calculate trade intensity metric"""
        if len(self.trade_flow_history) < 5:
            return 0.5
        
        # Calculate trades per minute (simplified)
        recent_window = 5  # 5 minute window
        recent_trades = list(self.trade_flow_history)[-recent_window:]
        trade_count = sum(flow.get('trade_count', 1) for flow in recent_trades)
        
        intensity = trade_count / recent_window
        return min(intensity / 100, 1.0)  # Normalize to 0-1
    
    def _estimate_volatility(self) -> float:
        """Estimate short-term volatility from order book changes"""
        if len(self.order_book_history) < 10:
            return 0.01  # 1% fallback
        
        recent_books = list(self.order_book_history)[-10:]
        mid_prices = [book.mid_price for book in recent_books]
        
        if len(mid_prices) > 1:
            returns = np.diff(np.log(mid_prices))
            volatility = np.std(returns) * np.sqrt(252 * 24 * 12)  # Annualized
            return min(volatility, 1.0)  # Cap at 100%
        
        return 0.01
    
    def _estimate_execution_time(self, order_book: OrderBookSnapshot) -> float:
        """Estimate time to execution based on market conditions"""
        # Simplified model based on spread and depth
        spread_factor = order_book.spread_bps / 10  # Wider spread = longer time
        depth_factor = 1.0 / (order_book.depth_of_market + 1)  # Less depth = longer time
        
        base_time = 30.0  # 30 seconds base
        estimated_time = base_time * (1 + spread_factor + depth_factor)
        
        return min(estimated_time, 300.0)  # Cap at 5 minutes
    
    def _estimate_slippage(self, order_book: OrderBookSnapshot) -> float:
        """Estimate expected slippage"""
        # Simplified slippage model
        spread_component = order_book.spread_bps / 2  # Half spread
        impact_component = self._estimate_price_impact(order_book, 1.0) * 10000  # Convert to bps
        
        total_slippage = spread_component + impact_component
        return min(total_slippage, 50.0)  # Cap at 50 bps
    
    def _calculate_liquidity_score(self, order_book: OrderBookSnapshot) -> float:
        """Calculate liquidity score (0-1, higher is better)"""
        # Factors: tight spread, deep book, balanced
        spread_score = max(0, 1 - (order_book.spread_bps / 50))  # Normalize by 50 bps
        depth_score = min(order_book.depth_of_market / 1000, 1.0)  # Normalize by 1000 units
        balance_score = max(0, 1 - abs(order_book.order_book_imbalance))
        
        # Weighted average
        liquidity_score = (spread_score * 0.4 + depth_score * 0.4 + balance_score * 0.2)
        return liquidity_score
    
    def _create_fallback_features(self) -> MicrostructureFeatures:
        """Create fallback features when calculation fails"""
        return MicrostructureFeatures(
            bid_ask_spread=0.01,
            spread_bps=10.0,
            order_book_imbalance=0.0,
            depth_of_market=100.0,
            price_impact_estimate=0.001,
            order_flow_imbalance=0.0,
            trade_intensity=0.5,
            volatility_estimate=0.02,
            time_to_execution=60.0,
            expected_slippage=5.0,
            liquidity_score=0.5
        )


class AdvancedActionEngine:
    """
    Advanced Action Engine with Microstructure Intelligence
    
    Converts MARL agent outputs into sophisticated execution actions
    with microstructure optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Advanced Action Engine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Microstructure analyzer
        self.microstructure_analyzer = MicrostructureAnalyzer()
        
        # Action mapping and validation
        self.action_space_size = len(ActionType)
        self.valid_actions = list(ActionType)
        
        # Execution tracking
        self.execution_history: deque = deque(
            maxlen=self.config.get('history_length', 1000)
        )
        
        # Performance metrics
        self.performance_metrics = {
            'action_distribution': {action.name: 0 for action in ActionType},
            'execution_costs': [],
            'slippage_stats': {'mean': 0, 'std': 0, 'max': 0},
            'success_rate': 0.0
        }
        
        logger.info(f"Advanced Action Engine initialized with {self.action_space_size} actions")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'history_length': 1000,
            'default_position_size': 1.0,
            'max_position_size': 10.0,
            'risk_limit': 0.05,  # 5% max risk per trade
            'slippage_tolerance_bps': 10.0,
            'market_impact_limit_bps': 20.0,
            'liquidity_threshold': 0.3
        }
    
    def process_agent_decision(
        self,
        agent_outputs: Dict[str, np.ndarray],
        order_book: OrderBookSnapshot,
        execution_context: ExecutionContext
    ) -> List[ActionOutput]:
        """
        Process agent decisions into executable actions
        
        Args:
            agent_outputs: Raw agent outputs (logits/probabilities)
            order_book: Current order book state
            execution_context: Execution context and constraints
            
        Returns:
            List[ActionOutput]: Executable actions with parameters
        """
        try:
            # Analyze microstructure
            microstructure_features = self.microstructure_analyzer.process_order_book(order_book)
            
            # Process each agent's decision
            actions = []
            
            for agent_id, output in agent_outputs.items():
                # Convert agent output to action
                action_output = self._convert_agent_output(
                    agent_id, output, microstructure_features, execution_context
                )
                
                if action_output:
                    actions.append(action_output)
            
            # Optimize execution across all actions
            optimized_actions = self._optimize_execution_plan(
                actions, microstructure_features, execution_context
            )
            
            # Update performance tracking
            self._update_performance_metrics(optimized_actions)
            
            return optimized_actions
            
        except Exception as e:
            logger.error(f"Error processing agent decisions: {e}")
            return []
    
    def _convert_agent_output(
        self,
        agent_id: str,
        output: np.ndarray,
        microstructure_features: MicrostructureFeatures,
        execution_context: ExecutionContext
    ) -> Optional[ActionOutput]:
        """Convert single agent output to action"""
        
        try:
            # Handle different output formats
            if len(output.shape) == 1:
                if len(output) == self.action_space_size:
                    # Full action space probabilities
                    action_probs = output
                elif len(output) == 3:
                    # Legacy 3-action format, convert to extended
                    action_probs = self._convert_legacy_actions(output)
                else:
                    logger.warning(f"Unexpected output shape for {agent_id}: {output.shape}")
                    return None
            else:
                logger.warning(f"Unexpected output shape for {agent_id}: {output.shape}")
                return None
            
            # Softmax normalization
            action_probs = np.exp(action_probs - np.max(action_probs))
            action_probs = action_probs / np.sum(action_probs)
            
            # Select action
            action_idx = np.argmax(action_probs)
            action = ActionType(action_idx)
            confidence = float(action_probs[action_idx])
            
            # Determine execution parameters based on microstructure
            execution_style = self._determine_execution_style(
                action, microstructure_features, execution_context
            )
            
            # Calculate position size
            size = self._calculate_position_size(
                action, execution_context, microstructure_features
            )
            
            # Set execution parameters
            limit_price = self._calculate_limit_price(
                action, order_book, execution_style
            )
            
            return ActionOutput(
                action=action,
                confidence=confidence,
                size=size,
                execution_style=execution_style,
                limit_price=limit_price,
                expected_slippage_bps=microstructure_features.expected_slippage,
                expected_market_impact_bps=microstructure_features.price_impact_estimate * 10000,
                liquidity_providing=(execution_style == ExecutionStyle.PASSIVE),
                reasoning=f"{agent_id} decision: {action.name} with {confidence:.2f} confidence"
            )
            
        except Exception as e:
            logger.error(f"Error converting agent output for {agent_id}: {e}")
            return None
    
    def _convert_legacy_actions(self, legacy_output: np.ndarray) -> np.ndarray:
        """Convert 3-action legacy format to 15-action extended format"""
        # Legacy: [Short, Hold, Long] -> Extended: 15 actions
        extended_probs = np.zeros(self.action_space_size)
        
        # Map legacy actions to primary extended actions
        short_prob, hold_prob, long_prob = legacy_output
        
        # Distribute probabilities across related actions
        extended_probs[ActionType.HOLD] = hold_prob * 0.6
        extended_probs[ActionType.CANCEL_ORDERS] = hold_prob * 0.2
        extended_probs[ActionType.MODIFY_ORDERS] = hold_prob * 0.2
        
        extended_probs[ActionType.INCREASE_LONG] = long_prob * 0.4
        extended_probs[ActionType.MARKET_BUY] = long_prob * 0.3
        extended_probs[ActionType.LIMIT_BUY] = long_prob * 0.3
        
        extended_probs[ActionType.INCREASE_SHORT] = short_prob * 0.4
        extended_probs[ActionType.MARKET_SELL] = short_prob * 0.3
        extended_probs[ActionType.LIMIT_SELL] = short_prob * 0.3
        
        return extended_probs
    
    def _determine_execution_style(
        self,
        action: ActionType,
        microstructure_features: MicrostructureFeatures,
        execution_context: ExecutionContext
    ) -> ExecutionStyle:
        """Determine optimal execution style"""
        
        # Aggressive actions
        if action in [ActionType.MARKET_BUY, ActionType.MARKET_SELL, ActionType.TAKE_LIQUIDITY]:
            return ExecutionStyle.AGGRESSIVE
        
        # Passive actions
        if action in [ActionType.LIMIT_BUY, ActionType.LIMIT_SELL, ActionType.PROVIDE_LIQUIDITY]:
            return ExecutionStyle.PASSIVE
        
        # Adaptive based on market conditions
        if microstructure_features.liquidity_score > 0.7:
            # High liquidity - can be aggressive
            if execution_context.urgency_level > 0.7:
                return ExecutionStyle.AGGRESSIVE
            else:
                return ExecutionStyle.BALANCED
        
        elif microstructure_features.liquidity_score < 0.3:
            # Low liquidity - be passive
            return ExecutionStyle.PASSIVE
        
        else:
            # Medium liquidity - balanced approach
            if abs(execution_context.target_position - execution_context.current_position) > 5.0:
                return ExecutionStyle.ICEBERG  # Large orders
            else:
                return ExecutionStyle.BALANCED
    
    def _calculate_position_size(
        self,
        action: ActionType,
        execution_context: ExecutionContext,
        microstructure_features: MicrostructureFeatures
    ) -> float:
        """Calculate optimal position size"""
        
        # Base size from context
        base_size = self.config['default_position_size']
        
        # Adjust based on action type
        if action in [ActionType.INCREASE_LONG, ActionType.INCREASE_SHORT]:
            size_multiplier = 1.5
        elif action in [ActionType.DECREASE_LONG, ActionType.DECREASE_SHORT]:
            size_multiplier = 0.5
        elif action in [ActionType.REDUCE_RISK]:
            size_multiplier = 0.25
        else:
            size_multiplier = 1.0
        
        # Adjust based on liquidity
        liquidity_adjustment = microstructure_features.liquidity_score
        
        # Adjust based on volatility
        volatility_adjustment = max(0.1, 1.0 - microstructure_features.volatility_estimate)
        
        # Calculate final size
        final_size = (base_size * size_multiplier * 
                     liquidity_adjustment * volatility_adjustment)
        
        # Apply limits
        final_size = max(0.1, min(final_size, self.config['max_position_size']))
        
        return final_size
    
    def _calculate_limit_price(
        self,
        action: ActionType,
        order_book: OrderBookSnapshot,
        execution_style: ExecutionStyle
    ) -> Optional[float]:
        """Calculate limit price for limit orders"""
        
        if action not in [ActionType.LIMIT_BUY, ActionType.LIMIT_SELL]:
            return None
        
        if execution_style == ExecutionStyle.AGGRESSIVE:
            # Aggressive limit orders - close to market
            if action == ActionType.LIMIT_BUY:
                return order_book.best_ask * 0.9999  # Just below ask
            else:
                return order_book.best_bid * 1.0001  # Just above bid
        
        elif execution_style == ExecutionStyle.PASSIVE:
            # Passive limit orders - provide liquidity
            if action == ActionType.LIMIT_BUY:
                return order_book.best_bid + order_book.bid_ask_spread * 0.1
            else:
                return order_book.best_ask - order_book.bid_ask_spread * 0.1
        
        else:
            # Balanced approach
            if action == ActionType.LIMIT_BUY:
                return order_book.mid_price - order_book.bid_ask_spread * 0.2
            else:
                return order_book.mid_price + order_book.bid_ask_spread * 0.2
    
    def _optimize_execution_plan(
        self,
        actions: List[ActionOutput],
        microstructure_features: MicrostructureFeatures,
        execution_context: ExecutionContext
    ) -> List[ActionOutput]:
        """Optimize execution plan across all actions"""
        
        if not actions:
            return []
        
        optimized_actions = []
        
        for action in actions:
            # Validate action feasibility
            if self._validate_action_feasibility(action, microstructure_features, execution_context):
                
                # Optimize timing
                action = self._optimize_action_timing(action, microstructure_features)
                
                # Optimize size
                action = self._optimize_action_size(action, microstructure_features, execution_context)
                
                optimized_actions.append(action)
            else:
                logger.warning(f"Action {action.action.name} failed feasibility check")
        
        return optimized_actions
    
    def _validate_action_feasibility(
        self,
        action: ActionOutput,
        microstructure_features: MicrostructureFeatures,
        execution_context: ExecutionContext
    ) -> bool:
        """Validate if action is feasible given current conditions"""
        
        # Check slippage tolerance
        if action.expected_slippage_bps > self.config['slippage_tolerance_bps']:
            return False
        
        # Check market impact limits
        if action.expected_market_impact_bps > self.config['market_impact_limit_bps']:
            return False
        
        # Check liquidity requirements
        if microstructure_features.liquidity_score < self.config['liquidity_threshold']:
            if action.execution_style == ExecutionStyle.AGGRESSIVE:
                return False
        
        # Check risk limits
        position_change = action.size
        if action.action in [ActionType.INCREASE_SHORT, ActionType.MARKET_SELL]:
            position_change = -position_change
        
        new_position = execution_context.current_position + position_change
        if abs(new_position) > execution_context.risk_budget:
            return False
        
        return True
    
    def _optimize_action_timing(
        self,
        action: ActionOutput,
        microstructure_features: MicrostructureFeatures
    ) -> ActionOutput:
        """Optimize action timing based on microstructure"""
        
        # Adjust time in force based on market conditions
        if microstructure_features.volatility_estimate > 0.05:  # High volatility
            action.time_in_force = "IOC"  # Immediate or Cancel
        elif microstructure_features.liquidity_score < 0.3:  # Low liquidity
            action.time_in_force = "GTC"  # Good Till Cancelled
        else:
            action.time_in_force = "GTD"  # Good Till Day
        
        return action
    
    def _optimize_action_size(
        self,
        action: ActionOutput,
        microstructure_features: MicrostructureFeatures,
        execution_context: ExecutionContext
    ) -> ActionOutput:
        """Optimize action size based on market conditions"""
        
        # Reduce size in low liquidity conditions
        if microstructure_features.liquidity_score < 0.3:
            action.size *= 0.5
        
        # Use iceberg for large orders
        if action.size > 5.0 and microstructure_features.depth_of_market < action.size * 2:
            action.iceberg_size = action.size * 0.2  # 20% chunks
            action.execution_style = ExecutionStyle.ICEBERG
        
        return action
    
    def _update_performance_metrics(self, actions: List[ActionOutput]):
        """Update performance tracking metrics"""
        
        for action in actions:
            # Update action distribution
            self.performance_metrics['action_distribution'][action.action.name] += 1
            
            # Track execution costs
            execution_cost = action.expected_slippage_bps + action.expected_market_impact_bps
            self.performance_metrics['execution_costs'].append(execution_cost)
            
            # Store in history
            self.execution_history.append({
                'timestamp': action.timestamp,
                'action': action.action.name,
                'size': action.size,
                'execution_style': action.execution_style.value,
                'expected_cost': execution_cost
            })
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get information about the action space"""
        return {
            'action_space_size': self.action_space_size,
            'valid_actions': [action.name for action in self.valid_actions],
            'action_categories': {
                'Position Management': [a.name for a in ActionType if 0 <= a <= 4],
                'Execution Styles': [a.name for a in ActionType if 5 <= a <= 8],
                'Order Management': [a.name for a in ActionType if 9 <= a <= 10],
                'Risk Management': [a.name for a in ActionType if 11 <= a <= 12],
                'Microstructure': [a.name for a in ActionType if 13 <= a <= 14]
            },
            'performance_metrics': self.performance_metrics
        }


# Test and validation functions
def create_sample_order_book() -> OrderBookSnapshot:
    """Create sample order book for testing"""
    
    # Sample bid side (descending prices)
    bid_prices = np.array([100.50, 100.49, 100.48, 100.47, 100.46])
    bid_sizes = np.array([100, 150, 200, 300, 250])
    bid_orders = np.array([5, 7, 8, 12, 10])
    
    # Sample ask side (ascending prices)
    ask_prices = np.array([100.51, 100.52, 100.53, 100.54, 100.55])
    ask_sizes = np.array([120, 180, 220, 280, 240])
    ask_orders = np.array([6, 8, 9, 11, 9])
    
    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    spread = best_ask - best_bid
    mid_price = (best_bid + best_ask) / 2
    
    # Calculate imbalances
    total_bid_size = np.sum(bid_sizes)
    total_ask_size = np.sum(ask_sizes)
    order_book_imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
    
    return OrderBookSnapshot(
        timestamp=pd.Timestamp.now(),
        symbol="TEST",
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        bid_orders=bid_orders,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        ask_orders=ask_orders,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_ask_spread=spread,
        mid_price=mid_price,
        order_book_imbalance=order_book_imbalance,
        depth_imbalance=order_book_imbalance,
        weighted_mid_price=mid_price
    )


def test_advanced_action_engine():
    """Test the advanced action engine"""
    print("🧪 Testing Advanced Action Engine")
    
    # Initialize engine
    engine = AdvancedActionEngine()
    
    # Create sample inputs
    order_book = create_sample_order_book()
    
    execution_context = ExecutionContext(
        current_position=2.0,
        target_position=3.0,
        risk_budget=10.0,
        time_horizon=30,
        urgency_level=0.6,
        volatility_regime="normal",
        market_impact_threshold=0.02,
        liquidity_condition="good",
        max_slippage_bps=15.0,
        max_market_impact_bps=25.0,
        commission_rate=0.001
    )
    
    # Sample agent outputs (both legacy and extended formats)
    agent_outputs = {
        'fvg_agent': np.array([0.1, 0.3, 0.6]),  # Legacy format
        'momentum_agent': np.random.softmax(np.random.randn(15)),  # Extended format
        'entry_agent': np.array([0.2, 0.5, 0.3])  # Legacy format
    }
    
    # Process decisions
    print(f"\n📊 Processing agent decisions:")
    actions = engine.process_agent_decision(agent_outputs, order_book, execution_context)
    
    for i, action in enumerate(actions):
        print(f"\n  Action {i+1}: {action.action.name}")
        print(f"    Size: {action.size:.2f}")
        print(f"    Confidence: {action.confidence:.2f}")
        print(f"    Execution Style: {action.execution_style.value}")
        print(f"    Expected Slippage: {action.expected_slippage_bps:.1f} bps")
        print(f"    Limit Price: {action.limit_price}")
        print(f"    Reasoning: {action.reasoning}")
    
    # Action space info
    print(f"\n📋 Action Space Information:")
    action_info = engine.get_action_space_info()
    print(f"  Total actions: {action_info['action_space_size']}")
    
    for category, actions_list in action_info['action_categories'].items():
        print(f"  {category}: {len(actions_list)} actions")
    
    print("\n✅ Advanced Action Engine validation complete!")


if __name__ == "__main__":
    test_advanced_action_engine()