"""
Position Sizing Agent (π₁) - Intelligent Position Sizing with Kelly Criterion and MARL Integration

This module implements the Position Sizing Agent that determines optimal position sizes using
Kelly Criterion enhanced with multi-factor risk analysis and MARL capabilities.

Technical Specifications:
- Agent: Position Sizing Agent (π₁)
- Action Space: Discrete(5) → {1, 2, 3, 4, 5} contracts
- Observation Space: Box(-∞, +∞, (10,)) → Risk state vector
- Decision Factors: Kelly Criterion, account size, volatility, correlation risk
- Performance Target: >95% optimal sizing accuracy
- Response Time: <10ms for sizing decisions

Author: Agent 2 - Position Sizing Specialist
Date: 2025-07-13
Mission: Create intelligent position sizing system with bulletproof Kelly integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import structlog

from src.risk.agents.base_risk_agent import BaseRiskAgent, RiskState, RiskMetrics
from src.risk.core.kelly_calculator import KellyCalculator, KellyOutput
from src.core.events import EventBus, Event, EventType

logger = structlog.get_logger()


@dataclass
class PositionSizingDecision:
    """Position sizing decision output with detailed reasoning"""
    contracts: int                      # Final position size (1-5 contracts)
    kelly_fraction: float              # Raw Kelly fraction
    position_size_fraction: float      # Adjusted position size fraction
    confidence: float                  # Decision confidence (0-1)
    reasoning: Dict[str, Any]          # Detailed reasoning breakdown
    risk_adjustments: List[str]        # List of applied risk adjustments
    computation_time_ms: float         # Processing time
    timestamp: datetime


@dataclass
class SizingFactors:
    """Multi-factor inputs for position sizing"""
    kelly_suggestion: float            # Kelly Criterion suggestion
    volatility_adjustment: float       # Volatility-based adjustment
    correlation_adjustment: float      # Correlation risk adjustment
    account_equity_factor: float       # Account equity consideration
    drawdown_penalty: float           # Current drawdown penalty
    market_stress_adjustment: float   # Market stress adjustment
    liquidity_factor: float          # Liquidity consideration
    time_of_day_factor: float        # Time-based risk factor


class PositionSizingNetwork(nn.Module):
    """
    Neural network for position sizing with multi-factor integration.
    
    Architecture:
    - Input: 10-dimensional risk state vector + Kelly suggestion
    - Hidden layers: Multi-scale feature processing
    - Output: Probability distribution over {1,2,3,4,5} contracts
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Network configuration
        self.input_dim = 11  # 10D risk vector + Kelly suggestion
        self.hidden_dims = config.get('hidden_dims', [128, 64, 32])
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.num_contracts = 5  # Discrete action space: {1,2,3,4,5}
        
        # Multi-scale feature processing layers
        self.feature_extractor = self._build_feature_extractor()
        
        # Position sizing decision layers
        self.sizing_head = self._build_sizing_head()
        
        # Confidence estimation branch
        self.confidence_head = self._build_confidence_head()
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_feature_extractor(self) -> nn.Module:
        """Build multi-scale feature extraction network"""
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_sizing_head(self) -> nn.Module:
        """Build position sizing decision head"""
        return nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, self.num_contracts),
            nn.Softmax(dim=-1)
        )
    
    def _build_confidence_head(self) -> nn.Module:
        """Build confidence estimation head"""
        return nn.Sequential(
            nn.Linear(self.hidden_dims[-1], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, risk_state: torch.Tensor, kelly_suggestion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            risk_state: 10D risk state vector
            kelly_suggestion: Kelly Criterion suggestion
            
        Returns:
            Tuple of (action_probabilities, confidence)
        """
        # Combine inputs
        combined_input = torch.cat([risk_state, kelly_suggestion.unsqueeze(-1)], dim=-1)
        
        # Extract features
        features = self.feature_extractor(combined_input)
        
        # Get action probabilities and confidence
        action_probs = self.sizing_head(features)
        confidence = self.confidence_head(features)
        
        return action_probs, confidence


class PositionSizingAgentV2(BaseRiskAgent):
    """
    Position Sizing Agent (π₁) with Kelly Criterion integration and MARL capabilities.
    
    This agent intelligently determines optimal position sizes using:
    - Kelly Criterion for mathematical optimality
    - Multi-factor risk adjustments
    - Real-time market condition adaptation
    - MARL-based learning and optimization
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """
        Initialize Position Sizing Agent
        
        Args:
            config: Agent configuration
            event_bus: Event bus for real-time communication
        """
        # Initialize base risk agent
        super().__init__(config, event_bus)
        self.name = "PositionSizingAgentV2"
        
        # Position sizing specific configuration
        self.sizing_config = config.get('position_sizing', {})
        
        # Kelly Calculator integration
        self.kelly_calculator = KellyCalculator(enable_rolling_validation=True)
        
        # Position sizing constraints
        self.min_contracts = 1
        self.max_contracts = 5
        self.max_position_fraction = self.sizing_config.get('max_position_fraction', 0.25)
        self.min_account_equity = self.sizing_config.get('min_account_equity', 10000)
        
        # Risk adjustment parameters
        self.volatility_threshold = self.sizing_config.get('volatility_threshold', 0.3)
        self.correlation_threshold = self.sizing_config.get('correlation_threshold', 0.7)
        self.drawdown_threshold = self.sizing_config.get('drawdown_threshold', 0.1)
        self.stress_threshold = self.sizing_config.get('stress_threshold', 0.8)
        
        # Neural network for position sizing
        self.device = torch.device(config.get('device', 'cpu'))
        self.network = PositionSizingNetwork(self.sizing_config).to(self.device)
        
        # Performance tracking
        self.sizing_decisions = deque(maxlen=1000)
        self.kelly_accuracy_history = deque(maxlen=100)
        self.sizing_performance_metrics = deque(maxlen=500)
        
        # Risk state validation
        self.last_valid_state: Optional[RiskState] = None
        self.state_validation_failures = 0
        
        logger.info("Position Sizing Agent V2 initialized",
                   name=self.name,
                   min_contracts=self.min_contracts,
                   max_contracts=self.max_contracts,
                   max_position_fraction=self.max_position_fraction)
    
    def calculate_risk_action(self, risk_state: RiskState) -> Tuple[int, float]:
        """
        Calculate optimal position size based on risk state
        
        Args:
            risk_state: Current risk state
            
        Returns:
            Tuple of (contracts, confidence)
        """
        start_time = datetime.now()
        
        try:
            # 1. Validate risk state
            if not self._validate_risk_state(risk_state):
                logger.warning("Invalid risk state, using safe fallback")
                return self._get_safe_position_size(), 0.1
            
            # 2. Calculate Kelly Criterion suggestion
            kelly_output = self._calculate_kelly_suggestion(risk_state)
            
            # 3. Extract multi-factor sizing inputs
            sizing_factors = self._extract_sizing_factors(risk_state, kelly_output)
            
            # 4. Neural network decision
            contracts, confidence = self._make_neural_sizing_decision(risk_state, sizing_factors)
            
            # 5. Apply final safety constraints
            final_contracts = self._apply_safety_constraints(contracts, risk_state, sizing_factors)
            
            # 6. Calculate detailed reasoning
            reasoning = self._generate_decision_reasoning(risk_state, sizing_factors, final_contracts)
            
            # 7. Create decision record
            decision = PositionSizingDecision(
                contracts=final_contracts,
                kelly_fraction=kelly_output.kelly_fraction,
                position_size_fraction=kelly_output.kelly_fraction * final_contracts / self.max_contracts,
                confidence=confidence,
                reasoning=reasoning,
                risk_adjustments=reasoning.get('adjustments_applied', []),
                computation_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                timestamp=datetime.now()
            )
            
            # 8. Track performance and publish event
            self._track_decision_performance(decision)
            self._publish_sizing_decision(decision)
            
            return final_contracts, confidence
            
        except Exception as e:
            logger.error("Error in position sizing calculation", error=str(e))
            return self._get_safe_position_size(), 0.1
    
    def validate_risk_constraints(self, risk_state: RiskState) -> bool:
        """
        Validate risk state against position sizing constraints
        
        Args:
            risk_state: Current risk state
            
        Returns:
            True if constraints are satisfied
        """
        constraints_passed = True
        
        # Account equity constraint
        if risk_state.account_equity_normalized < 0.5:  # Below 50% of initial capital
            logger.warning("Low account equity constraint violation",
                          equity_ratio=risk_state.account_equity_normalized)
            constraints_passed = False
        
        # Drawdown constraint
        if risk_state.current_drawdown_pct > self.drawdown_threshold:
            logger.warning("Drawdown constraint violation",
                          current_drawdown=risk_state.current_drawdown_pct,
                          threshold=self.drawdown_threshold)
            constraints_passed = False
        
        # Margin usage constraint
        if risk_state.margin_usage_pct > 0.8:  # Above 80% margin usage
            logger.warning("High margin usage constraint violation",
                          margin_usage=risk_state.margin_usage_pct)
            constraints_passed = False
        
        # Market stress constraint
        if risk_state.market_stress_level > self.stress_threshold:
            logger.warning("Market stress constraint violation",
                          stress_level=risk_state.market_stress_level,
                          threshold=self.stress_threshold)
            constraints_passed = False
        
        return constraints_passed
    
    def _validate_risk_state(self, risk_state: RiskState) -> bool:
        """Validate risk state for position sizing"""
        try:
            # Check for valid numeric values
            state_vector = risk_state.to_vector()
            if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
                self.state_validation_failures += 1
                return False
            
            # Check reasonable ranges
            if (risk_state.account_equity_normalized < 0 or
                risk_state.volatility_regime < 0 or risk_state.volatility_regime > 1 or
                risk_state.margin_usage_pct < 0 or risk_state.margin_usage_pct > 1):
                self.state_validation_failures += 1
                return False
            
            self.last_valid_state = risk_state
            return True
            
        except Exception as e:
            logger.error("Risk state validation error", error=str(e))
            self.state_validation_failures += 1
            return False
    
    def _calculate_kelly_suggestion(self, risk_state: RiskState) -> KellyOutput:
        """Calculate Kelly Criterion position size suggestion"""
        try:
            # Estimate win probability from market conditions
            # This is a simplified heuristic - in production, use more sophisticated estimation
            base_win_prob = 0.51  # Slight edge assumption
            
            # Adjust based on market stress and volatility
            stress_adjustment = (1 - risk_state.market_stress_level) * 0.1
            volatility_adjustment = (1 - risk_state.volatility_regime) * 0.05
            
            win_probability = base_win_prob + stress_adjustment + volatility_adjustment
            win_probability = max(0.505, min(0.65, win_probability))  # Reasonable bounds
            
            # Estimate payout ratio based on volatility and liquidity
            base_payout = 1.5  # 1.5:1 risk-reward assumption
            liquidity_bonus = risk_state.liquidity_conditions * 0.5
            
            payout_ratio = base_payout + liquidity_bonus
            payout_ratio = max(1.1, min(3.0, payout_ratio))  # Reasonable bounds
            
            # Calculate Kelly position
            return self.kelly_calculator.calculate_position_size(
                win_probability=win_probability,
                payout_ratio=payout_ratio,
                capital=1.0  # Normalized capital
            )
            
        except Exception as e:
            logger.error("Kelly calculation error", error=str(e))
            # Return conservative fallback
            return KellyOutput(
                kelly_fraction=0.1,
                position_size=0.1,
                inputs=None,
                calculation_time_ms=0.0,
                security_warnings=[],
                capped_by_validation=True
            )
    
    def _extract_sizing_factors(self, risk_state: RiskState, kelly_output: KellyOutput) -> SizingFactors:
        """Extract multi-factor inputs for position sizing"""
        
        # Volatility adjustment (reduce size in high volatility)
        volatility_adjustment = 1.0 - min(risk_state.volatility_regime, 0.5)
        
        # Correlation adjustment (reduce size when high correlation)
        correlation_adjustment = 1.0 - max(0, risk_state.correlation_risk - 0.5)
        
        # Account equity factor (scale with available capital)
        account_equity_factor = min(1.0, risk_state.account_equity_normalized)
        
        # Drawdown penalty (reduce size when in drawdown)
        drawdown_penalty = 1.0 - min(risk_state.current_drawdown_pct * 2, 0.5)
        
        # Market stress adjustment (reduce size in stressed markets)
        market_stress_adjustment = 1.0 - min(risk_state.market_stress_level, 0.6)
        
        # Liquidity factor (adjust for liquidity conditions)
        liquidity_factor = 0.8 + (risk_state.liquidity_conditions * 0.4)
        
        # Time of day factor (reduce size during risky periods)
        time_of_day_factor = 1.0 - (risk_state.time_of_day_risk * 0.3)
        
        return SizingFactors(
            kelly_suggestion=kelly_output.kelly_fraction,
            volatility_adjustment=volatility_adjustment,
            correlation_adjustment=correlation_adjustment,
            account_equity_factor=account_equity_factor,
            drawdown_penalty=drawdown_penalty,
            market_stress_adjustment=market_stress_adjustment,
            liquidity_factor=liquidity_factor,
            time_of_day_factor=time_of_day_factor
        )
    
    def _make_neural_sizing_decision(self, risk_state: RiskState, sizing_factors: SizingFactors) -> Tuple[int, float]:
        """Make position sizing decision using neural network"""
        try:
            # Prepare inputs
            risk_vector = torch.FloatTensor(risk_state.to_vector()).unsqueeze(0).to(self.device)
            kelly_tensor = torch.FloatTensor([sizing_factors.kelly_suggestion]).to(self.device)
            
            # Neural network forward pass
            with torch.no_grad():
                action_probs, confidence = self.network(risk_vector, kelly_tensor)
                action_probs = action_probs.cpu().numpy().flatten()
                confidence = confidence.cpu().item()
            
            # Sample action from probability distribution
            contracts = np.random.choice(range(1, 6), p=action_probs) 
            
            return contracts, confidence
            
        except Exception as e:
            logger.error("Neural sizing decision error", error=str(e))
            # Fallback to Kelly-based heuristic
            kelly_contracts = max(1, min(5, int(sizing_factors.kelly_suggestion * 5)))
            return kelly_contracts, 0.5
    
    def _apply_safety_constraints(self, contracts: int, risk_state: RiskState, sizing_factors: SizingFactors) -> int:
        """Apply final safety constraints to position size"""
        
        # Ensure within bounds
        contracts = max(self.min_contracts, min(self.max_contracts, contracts))
        
        # Apply multi-factor reduction
        reduction_factor = (
            sizing_factors.volatility_adjustment *
            sizing_factors.correlation_adjustment *
            sizing_factors.account_equity_factor *
            sizing_factors.drawdown_penalty *
            sizing_factors.market_stress_adjustment *
            sizing_factors.time_of_day_factor
        )
        
        # Reduce position if factors suggest high risk
        if reduction_factor < 0.8:
            contracts = max(1, contracts - 1)
        if reduction_factor < 0.6:
            contracts = max(1, contracts - 1)
        if reduction_factor < 0.4:
            contracts = 1  # Minimum position only
        
        # Emergency constraints
        if risk_state.current_drawdown_pct > 0.15:  # 15% drawdown
            contracts = 1
        if risk_state.margin_usage_pct > 0.9:  # 90% margin usage
            contracts = 1
        
        return contracts
    
    def _generate_decision_reasoning(self, risk_state: RiskState, sizing_factors: SizingFactors, final_contracts: int) -> Dict[str, Any]:
        """Generate detailed reasoning for the sizing decision"""
        
        adjustments_applied = []
        
        # Check what adjustments were applied
        if sizing_factors.volatility_adjustment < 0.9:
            adjustments_applied.append("volatility_reduction")
        if sizing_factors.correlation_adjustment < 0.9:
            adjustments_applied.append("correlation_reduction")
        if sizing_factors.drawdown_penalty < 0.9:
            adjustments_applied.append("drawdown_penalty")
        if sizing_factors.market_stress_adjustment < 0.9:
            adjustments_applied.append("stress_reduction")
        
        return {
            'kelly_fraction': sizing_factors.kelly_suggestion,
            'final_contracts': final_contracts,
            'kelly_suggested_contracts': int(sizing_factors.kelly_suggestion * 5),
            'reduction_factors': {
                'volatility': sizing_factors.volatility_adjustment,
                'correlation': sizing_factors.correlation_adjustment,
                'equity': sizing_factors.account_equity_factor,
                'drawdown': sizing_factors.drawdown_penalty,
                'stress': sizing_factors.market_stress_adjustment,
                'liquidity': sizing_factors.liquidity_factor,
                'time': sizing_factors.time_of_day_factor
            },
            'adjustments_applied': adjustments_applied,
            'risk_state_summary': {
                'volatility_regime': risk_state.volatility_regime,
                'correlation_risk': risk_state.correlation_risk,
                'drawdown_pct': risk_state.current_drawdown_pct,
                'stress_level': risk_state.market_stress_level,
                'margin_usage': risk_state.margin_usage_pct
            }
        }
    
    def _get_safe_position_size(self) -> int:
        """Get safe default position size"""
        return 1  # Always return minimum safe position
    
    def _track_decision_performance(self, decision: PositionSizingDecision):
        """Track decision performance metrics"""
        self.sizing_decisions.append(decision)
        
        # Track Kelly accuracy (how close our decision was to Kelly suggestion)
        kelly_suggested_contracts = max(1, min(5, int(decision.kelly_fraction * 5)))
        kelly_accuracy = 1.0 - abs(decision.contracts - kelly_suggested_contracts) / 4.0
        self.kelly_accuracy_history.append(kelly_accuracy)
        
        # Track overall performance
        self.sizing_performance_metrics.append({
            'contracts': decision.contracts,
            'kelly_fraction': decision.kelly_fraction,
            'confidence': decision.confidence,
            'computation_time_ms': decision.computation_time_ms,
            'adjustments_count': len(decision.risk_adjustments)
        })
    
    def _publish_sizing_decision(self, decision: PositionSizingDecision):
        """Publish sizing decision event"""
        if self.event_bus:
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.POSITION_SIZE_UPDATE,
                    {
                        'agent': self.name,
                        'contracts': decision.contracts,
                        'kelly_fraction': decision.kelly_fraction,
                        'confidence': decision.confidence,
                        'reasoning': decision.reasoning,
                        'computation_time_ms': decision.computation_time_ms
                    },
                    self.name
                )
            )
    
    def get_sizing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive position sizing performance metrics"""
        if not self.sizing_decisions:
            return {'status': 'no_decisions_made'}
        
        # Calculate Kelly accuracy
        avg_kelly_accuracy = np.mean(self.kelly_accuracy_history) if self.kelly_accuracy_history else 0.0
        
        # Calculate average response time
        avg_response_time = np.mean([d.computation_time_ms for d in self.sizing_decisions])
        
        # Calculate decision distribution
        contract_distribution = {}
        for i in range(1, 6):
            count = sum(1 for d in self.sizing_decisions if d.contracts == i)
            contract_distribution[f'contracts_{i}'] = count / len(self.sizing_decisions)
        
        # Calculate adjustment frequency
        adjustment_frequency = {}
        all_adjustments = []
        for decision in self.sizing_decisions:
            all_adjustments.extend(decision.risk_adjustments)
        
        for adj in set(all_adjustments):
            adjustment_frequency[adj] = all_adjustments.count(adj) / len(self.sizing_decisions)
        
        return {
            'total_decisions': len(self.sizing_decisions),
            'kelly_accuracy_avg': avg_kelly_accuracy,
            'kelly_accuracy_target': 0.95,  # >95% target
            'avg_response_time_ms': avg_response_time,
            'response_time_target': 10.0,  # <10ms target
            'contract_distribution': contract_distribution,
            'adjustment_frequency': adjustment_frequency,
            'state_validation_failures': self.state_validation_failures,
            'performance_targets_met': {
                'kelly_accuracy': avg_kelly_accuracy >= 0.95,
                'response_time': avg_response_time <= 10.0
            }
        }
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get risk metrics specific to position sizing"""
        base_metrics = super().get_risk_metrics()
        
        # Enhanced metrics for position sizing
        sizing_metrics = self.get_sizing_metrics()
        
        return RiskMetrics(
            total_risk_decisions=base_metrics.total_risk_decisions,
            risk_events_detected=base_metrics.risk_events_detected,
            false_positive_rate=base_metrics.false_positive_rate,
            avg_response_time_ms=sizing_metrics.get('avg_response_time_ms', 0.0),
            risk_adjusted_return=0.0,  # To be calculated from trading results
            max_drawdown=0.0,  # To be calculated from trading results
            sharpe_ratio=0.0,  # To be calculated from trading results
            var_accuracy=sizing_metrics.get('kelly_accuracy_avg', 0.0),
            correlation_prediction_accuracy=sizing_metrics.get('kelly_accuracy_avg', 0.0)
        )
    
    def update_from_trading_results(self, contracts_used: int, pnl: float, kelly_fraction: float):
        """Update agent learning from actual trading results"""
        try:
            # Calculate Kelly accuracy for this trade
            kelly_suggested_contracts = max(1, min(5, int(kelly_fraction * 5)))
            kelly_accuracy = 1.0 - abs(contracts_used - kelly_suggested_contracts) / 4.0
            
            # Store result for learning (this would be used for training in production)
            trading_result = {
                'contracts_used': contracts_used,
                'pnl': pnl,
                'kelly_fraction': kelly_fraction,
                'kelly_accuracy': kelly_accuracy,
                'timestamp': datetime.now()
            }
            
            logger.info("Trading result recorded for agent learning",
                       contracts=contracts_used,
                       pnl=pnl,
                       kelly_accuracy=kelly_accuracy)
            
        except Exception as e:
            logger.error("Error updating from trading results", error=str(e))
    
    def __str__(self) -> str:
        """String representation"""
        return f"PositionSizingAgentV2(decisions={len(self.sizing_decisions)})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        metrics = self.get_sizing_metrics()
        return (f"PositionSizingAgentV2("
                f"decisions={len(self.sizing_decisions)}, "
                f"kelly_accuracy={metrics.get('kelly_accuracy_avg', 0.0):.3f}, "
                f"avg_response_time={metrics.get('avg_response_time_ms', 0.0):.2f}ms)")


def create_position_sizing_agent_v2(config: Dict[str, Any], event_bus: Optional[EventBus] = None) -> PositionSizingAgentV2:
    """
    Factory function to create Position Sizing Agent V2 with validated configuration
    
    Args:
        config: Agent configuration
        event_bus: Optional event bus for real-time communication
        
    Returns:
        Configured PositionSizingAgentV2 instance
    """
    # Set default configuration if not provided
    default_config = {
        'position_sizing': {
            'hidden_dims': [128, 64, 32],
            'dropout_rate': 0.1,
            'max_position_fraction': 0.25,
            'min_account_equity': 10000,
            'volatility_threshold': 0.3,
            'correlation_threshold': 0.7,
            'drawdown_threshold': 0.1,
            'stress_threshold': 0.8
        },
        'max_response_time_ms': 10.0,
        'risk_tolerance': 0.02,
        'enable_emergency_stop': True,
        'device': 'cpu'
    }
    
    # Merge configurations
    merged_config = {**default_config, **config}
    
    return PositionSizingAgentV2(merged_config, event_bus)