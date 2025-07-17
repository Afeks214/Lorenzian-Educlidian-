"""
Choice Generation System for Expert Decision Points

This module analyzes market conditions and MARL agent outputs to generate
meaningful strategy choices for expert evaluation in complex scenarios.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import uuid
import structlog

from .feedback_api import (
    DecisionPoint, TradingStrategy, MarketContext, 
    DecisionComplexity, StrategyType
)
from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    CRISIS = "crisis"


class DecisionTrigger(Enum):
    """Triggers that require expert input"""
    HIGH_UNCERTAINTY = "high_uncertainty"
    CONFLICTING_SIGNALS = "conflicting_signals"
    REGIME_CHANGE = "regime_change"
    BLACK_SWAN_EVENT = "black_swan_event"
    CORRELATION_SHOCK = "correlation_shock"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    NEWS_EVENT = "news_event"


@dataclass
class AgentOutput:
    """Output from a MARL agent"""
    agent_id: str
    action: str
    confidence: float
    reasoning: str
    risk_score: float
    expected_return: float


@dataclass
class MarketSignal:
    """Market signal from various sources"""
    signal_type: str
    strength: float
    confidence: float
    timeframe: str
    source: str


class ChoiceGenerator:
    """Generates meaningful strategy choices for expert evaluation"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        
        # Complexity thresholds
        self.uncertainty_threshold = 0.3
        self.conflict_threshold = 0.4
        self.volatility_threshold = 0.02
        self.volume_threshold = 1.5  # vs average
        
        # Strategy templates
        self.strategy_templates = self._initialize_strategy_templates()
        
        logger.info("Choice Generator initialized")

    def _initialize_strategy_templates(self) -> Dict[StrategyType, Dict[str, Any]]:
        """Initialize strategy templates with base parameters"""
        return {
            StrategyType.AGGRESSIVE: {
                "position_size_multiplier": 1.5,
                "stop_loss_multiplier": 0.8,
                "take_profit_multiplier": 2.0,
                "time_horizon": 15,  # minutes
                "risk_tolerance": 0.05
            },
            StrategyType.CONSERVATIVE: {
                "position_size_multiplier": 0.5,
                "stop_loss_multiplier": 1.5,
                "take_profit_multiplier": 1.2,
                "time_horizon": 60,
                "risk_tolerance": 0.02
            },
            StrategyType.MOMENTUM: {
                "position_size_multiplier": 1.2,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.8,
                "time_horizon": 30,
                "risk_tolerance": 0.03
            },
            StrategyType.MEAN_REVERSION: {
                "position_size_multiplier": 0.8,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 1.5,
                "time_horizon": 45,
                "risk_tolerance": 0.025
            },
            StrategyType.BREAKOUT: {
                "position_size_multiplier": 1.3,
                "stop_loss_multiplier": 0.9,
                "take_profit_multiplier": 2.5,
                "time_horizon": 20,
                "risk_tolerance": 0.04
            },
            StrategyType.SCALPING: {
                "position_size_multiplier": 2.0,
                "stop_loss_multiplier": 0.5,
                "take_profit_multiplier": 0.8,
                "time_horizon": 5,
                "risk_tolerance": 0.01
            }
        }

    def analyze_decision_complexity(
        self, 
        agent_outputs: List[AgentOutput],
        market_signals: List[MarketSignal],
        market_context: MarketContext
    ) -> Tuple[DecisionComplexity, List[DecisionTrigger]]:
        """Analyze the complexity of a trading decision"""
        
        triggers = []
        complexity_score = 0.0
        
        # Check agent uncertainty and conflicts
        if agent_outputs:
            confidences = [output.confidence for output in agent_outputs]
            avg_confidence = np.mean(confidences)
            confidence_variance = np.var(confidences)
            
            if avg_confidence < self.uncertainty_threshold:
                triggers.append(DecisionTrigger.HIGH_UNCERTAINTY)
                complexity_score += 0.3
            
            if confidence_variance > self.conflict_threshold:
                triggers.append(DecisionTrigger.CONFLICTING_SIGNALS)
                complexity_score += 0.4
        
        # Check market conditions
        if market_context.volatility > self.volatility_threshold:
            triggers.append(DecisionTrigger.REGIME_CHANGE)
            complexity_score += 0.2
        
        if market_context.correlation_shock:
            triggers.append(DecisionTrigger.CORRELATION_SHOCK)
            complexity_score += 0.5
        
        # Check volume anomalies
        if market_context.volume > self.volume_threshold:
            complexity_score += 0.1
        
        # Determine complexity level
        if complexity_score >= 0.8:
            complexity = DecisionComplexity.CRITICAL
        elif complexity_score >= 0.5:
            complexity = DecisionComplexity.HIGH
        elif complexity_score >= 0.3:
            complexity = DecisionComplexity.MEDIUM
        else:
            complexity = DecisionComplexity.LOW
        
        return complexity, triggers

    def generate_strategy_alternatives(
        self,
        market_context: MarketContext,
        agent_outputs: List[AgentOutput],
        current_position: Optional[Dict[str, Any]] = None
    ) -> List[TradingStrategy]:
        """Generate alternative trading strategies for expert evaluation"""
        
        strategies = []
        
        # Determine market regime
        regime = self._classify_market_regime(market_context)
        
        # Generate top 3-4 most relevant strategies based on context
        strategy_types = self._select_relevant_strategies(regime, market_context, agent_outputs)
        
        for i, strategy_type in enumerate(strategy_types):
            strategy = self._create_strategy(
                strategy_type=strategy_type,
                market_context=market_context,
                agent_outputs=agent_outputs,
                current_position=current_position,
                variant=i
            )
            strategies.append(strategy)
        
        # Sort by confidence score
        strategies.sort(key=lambda s: s.confidence_score, reverse=True)
        
        return strategies[:3]  # Return top 3 strategies

    def _classify_market_regime(self, context: MarketContext) -> MarketRegime:
        """Classify current market regime"""
        
        if context.correlation_shock:
            return MarketRegime.CRISIS
        
        if context.volatility > 0.03:
            return MarketRegime.VOLATILE
        elif context.volatility < 0.005:
            return MarketRegime.QUIET
        
        if context.trend_strength > 0.7:
            if context.price > context.resistance_level:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        return MarketRegime.RANGING

    def _select_relevant_strategies(
        self,
        regime: MarketRegime,
        context: MarketContext,
        agent_outputs: List[AgentOutput]
    ) -> List[StrategyType]:
        """Select most relevant strategy types for current conditions"""
        
        strategy_scores = {}
        
        # Score strategies based on market regime
        regime_preferences = {
            MarketRegime.TRENDING_UP: [StrategyType.MOMENTUM, StrategyType.BREAKOUT, StrategyType.AGGRESSIVE],
            MarketRegime.TRENDING_DOWN: [StrategyType.CONSERVATIVE, StrategyType.MEAN_REVERSION, StrategyType.MOMENTUM],
            MarketRegime.RANGING: [StrategyType.MEAN_REVERSION, StrategyType.SCALPING, StrategyType.CONSERVATIVE],
            MarketRegime.VOLATILE: [StrategyType.CONSERVATIVE, StrategyType.SCALPING, StrategyType.AGGRESSIVE],
            MarketRegime.QUIET: [StrategyType.SCALPING, StrategyType.MOMENTUM, StrategyType.MEAN_REVERSION],
            MarketRegime.CRISIS: [StrategyType.CONSERVATIVE, StrategyType.MEAN_REVERSION, StrategyType.AGGRESSIVE]
        }
        
        preferred_strategies = regime_preferences.get(regime, list(StrategyType))
        
        # Initialize scores
        for strategy_type in StrategyType:
            strategy_scores[strategy_type] = 0.0
        
        # Boost preferred strategies
        for i, strategy_type in enumerate(preferred_strategies):
            strategy_scores[strategy_type] += (len(preferred_strategies) - i) * 0.2
        
        # Consider agent recommendations
        for output in agent_outputs:
            if "aggressive" in output.action.lower():
                strategy_scores[StrategyType.AGGRESSIVE] += output.confidence * 0.3
            elif "conservative" in output.action.lower():
                strategy_scores[StrategyType.CONSERVATIVE] += output.confidence * 0.3
            elif "momentum" in output.action.lower():
                strategy_scores[StrategyType.MOMENTUM] += output.confidence * 0.3
        
        # Sort by score and return top strategies
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [strategy_type for strategy_type, _ in sorted_strategies[:4]]

    def _create_strategy(
        self,
        strategy_type: StrategyType,
        market_context: MarketContext,
        agent_outputs: List[AgentOutput],
        current_position: Optional[Dict[str, Any]],
        variant: int = 0
    ) -> TradingStrategy:
        """Create a specific trading strategy"""
        
        template = self.strategy_templates[strategy_type]
        
        # Base position size (would be calculated from Kelly Criterion in production)
        base_position_size = 1000.0  # Example base size
        
        # Adjust for current market conditions
        volatility_adjustment = 1.0 - (market_context.volatility * 10)  # Reduce size in high vol
        position_size = base_position_size * template["position_size_multiplier"] * volatility_adjustment
        
        # Add variant adjustments
        if variant == 1:
            position_size *= 0.8  # More conservative variant
            template = template.copy()
            template["stop_loss_multiplier"] *= 1.2
        elif variant == 2:
            position_size *= 1.2  # More aggressive variant
            template = template.copy()
            template["take_profit_multiplier"] *= 1.1
        
        # Calculate entry, stop loss, and take profit
        entry_price = market_context.price
        
        if strategy_type in [StrategyType.MOMENTUM, StrategyType.BREAKOUT, StrategyType.AGGRESSIVE]:
            # Long bias strategies
            stop_loss = entry_price * (1 - template["risk_tolerance"] * template["stop_loss_multiplier"])
            take_profit = entry_price * (1 + template["risk_tolerance"] * template["take_profit_multiplier"])
        else:
            # Mean reversion or conservative strategies might be shorts
            if market_context.price > (market_context.support_level + market_context.resistance_level) / 2:
                # Price above midpoint, consider short
                stop_loss = entry_price * (1 + template["risk_tolerance"] * template["stop_loss_multiplier"])
                take_profit = entry_price * (1 - template["risk_tolerance"] * template["take_profit_multiplier"])
                position_size *= -1  # Short position
            else:
                # Long position
                stop_loss = entry_price * (1 - template["risk_tolerance"] * template["stop_loss_multiplier"])
                take_profit = entry_price * (1 + template["risk_tolerance"] * template["take_profit_multiplier"])
        
        # Calculate risk-reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Calculate expected PnL and max drawdown
        expected_pnl = position_size * (take_profit - entry_price) * 0.6  # 60% success rate assumption
        max_drawdown = abs(position_size * (stop_loss - entry_price))
        
        # Generate confidence score based on multiple factors
        confidence_score = self._calculate_strategy_confidence(
            strategy_type, market_context, agent_outputs, risk_reward_ratio
        )
        
        # Generate reasoning
        reasoning = self._generate_strategy_reasoning(strategy_type, market_context, variant)
        
        return TradingStrategy(
            strategy_id=str(uuid.uuid4()),
            strategy_type=strategy_type,
            entry_price=round(entry_price, 4),
            position_size=round(position_size, 2),
            stop_loss=round(stop_loss, 4),
            take_profit=round(take_profit, 4),
            time_horizon=template["time_horizon"],
            risk_reward_ratio=round(risk_reward_ratio, 2),
            confidence_score=round(confidence_score, 3),
            reasoning=reasoning,
            expected_pnl=round(expected_pnl, 2),
            max_drawdown=round(max_drawdown, 2)
        )

    def _calculate_strategy_confidence(
        self,
        strategy_type: StrategyType,
        market_context: MarketContext,
        agent_outputs: List[AgentOutput],
        risk_reward_ratio: float
    ) -> float:
        """Calculate confidence score for a strategy"""
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on market conditions alignment
        if strategy_type == StrategyType.MOMENTUM and market_context.trend_strength > 0.7:
            confidence += 0.2
        elif strategy_type == StrategyType.MEAN_REVERSION and market_context.trend_strength < 0.3:
            confidence += 0.2
        elif strategy_type == StrategyType.CONSERVATIVE and market_context.volatility > 0.02:
            confidence += 0.15
        
        # Adjust based on agent consensus
        if agent_outputs:
            relevant_outputs = [
                output for output in agent_outputs 
                if strategy_type.value.lower() in output.action.lower()
            ]
            if relevant_outputs:
                avg_agent_confidence = np.mean([output.confidence for output in relevant_outputs])
                confidence += (avg_agent_confidence - 0.5) * 0.3
        
        # Adjust based on risk-reward ratio
        if risk_reward_ratio > 2.0:
            confidence += 0.1
        elif risk_reward_ratio < 1.0:
            confidence -= 0.1
        
        # Ensure confidence is within bounds
        return max(0.1, min(0.95, confidence))

    def _generate_strategy_reasoning(
        self,
        strategy_type: StrategyType,
        market_context: MarketContext,
        variant: int
    ) -> str:
        """Generate human-readable reasoning for the strategy"""
        
        base_reasoning = {
            StrategyType.AGGRESSIVE: f"High-conviction play targeting {market_context.symbol} with strong momentum signals. Current price {market_context.price} shows potential for quick moves.",
            StrategyType.CONSERVATIVE: f"Risk-managed approach for {market_context.symbol} given current volatility of {market_context.volatility:.3f}. Focus on capital preservation.",
            StrategyType.MOMENTUM: f"Momentum-based strategy riding the current trend strength of {market_context.trend_strength:.2f} in {market_context.symbol}.",
            StrategyType.MEAN_REVERSION: f"Counter-trend opportunity in {market_context.symbol} as price deviates from fair value between support {market_context.support_level} and resistance {market_context.resistance_level}.",
            StrategyType.BREAKOUT: f"Breakout strategy anticipating move beyond resistance {market_context.resistance_level} in {market_context.symbol} with volume confirmation.",
            StrategyType.SCALPING: f"Short-term scalping approach exploiting micro-movements in {market_context.symbol} during current market conditions."
        }
        
        reasoning = base_reasoning.get(strategy_type, "Strategy targeting current market opportunity.")
        
        # Add variant-specific notes
        if variant == 1:
            reasoning += " (Conservative variant with enhanced risk management)"
        elif variant == 2:
            reasoning += " (Aggressive variant with increased profit targets)"
        
        # Add market regime context
        if market_context.correlation_shock:
            reasoning += " Note: Correlation shock detected - heightened caution advised."
        
        return reasoning

    def should_request_expert_input(
        self,
        agent_outputs: List[AgentOutput],
        market_signals: List[MarketSignal],
        market_context: MarketContext
    ) -> bool:
        """Determine if expert input should be requested"""
        
        complexity, triggers = self.analyze_decision_complexity(
            agent_outputs, market_signals, market_context
        )
        
        # Request expert input for medium complexity and above
        if complexity in [DecisionComplexity.MEDIUM, DecisionComplexity.HIGH, DecisionComplexity.CRITICAL]:
            return True
        
        # Also request for specific triggers
        critical_triggers = [
            DecisionTrigger.BLACK_SWAN_EVENT,
            DecisionTrigger.CORRELATION_SHOCK,
            DecisionTrigger.LIQUIDITY_CRISIS
        ]
        
        return any(trigger in critical_triggers for trigger in triggers)

    def create_decision_point(
        self,
        market_context: MarketContext,
        agent_outputs: List[AgentOutput],
        market_signals: List[MarketSignal],
        current_position: Optional[Dict[str, Any]] = None
    ) -> Optional[DecisionPoint]:
        """Create a decision point for expert evaluation"""
        
        if not self.should_request_expert_input(agent_outputs, market_signals, market_context):
            return None
        
        complexity, triggers = self.analyze_decision_complexity(
            agent_outputs, market_signals, market_context
        )
        
        strategies = self.generate_strategy_alternatives(
            market_context, agent_outputs, current_position
        )
        
        # Set deadline based on complexity
        deadline_minutes = {
            DecisionComplexity.LOW: 30,
            DecisionComplexity.MEDIUM: 15,
            DecisionComplexity.HIGH: 10,
            DecisionComplexity.CRITICAL: 5
        }
        
        deadline = datetime.now() + timedelta(minutes=deadline_minutes[complexity])
        
        # Get model recommendation (highest confidence strategy)
        model_recommendation = strategies[0].strategy_id if strategies else "no_action"
        
        decision_point = DecisionPoint(
            decision_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            context=market_context,
            complexity=complexity,
            strategies=strategies,
            current_position=current_position,
            expert_deadline=deadline,
            model_recommendation=model_recommendation,
            confidence_threshold=0.7
        )
        
        logger.info(
            "Decision point created",
            decision_id=decision_point.decision_id,
            complexity=complexity.value,
            strategies_count=len(strategies),
            triggers=[t.value for t in triggers]
        )
        
        return decision_point