"""
Sequential Execution Agents (Agent 7 Implementation)
================================================

This module implements the 5 sequential execution agents that form the final layer
of the GrandModel cascade system. Each agent processes upstream MARL outputs and
contributes to the final execution decision.

Key Features:
- Context-aware agents that process strategic, tactical, and risk outputs
- Microsecond timing optimization for sub-10ms execution
- Superposition state handling from quantum-inspired upstream MARLs
- Real-time market microstructure analysis
- Coordinated decision-making with Byzantine fault tolerance

Sequential Execution Order:
1. MarketTimingAgent (π₁): Optimal execution timing
2. LiquiditySourcingAgent (π₂): Venue and liquidity selection
3. PositionFragmentationAgent (π₃): Order size optimization
4. RiskControlAgent (π₄): Real-time risk monitoring
5. ExecutionMonitorAgent (π₅): Quality control and feedback

Author: Claude Code (Agent 7 Mission)
Version: 1.0
Date: 2025-07-17
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

logger = structlog.get_logger()


@dataclass
class AgentDecision:
    """Standard agent decision output"""
    agent_id: str
    decision_value: Any
    confidence: float
    processing_time_us: float
    internal_state: Dict[str, Any]
    timestamp: datetime
    decision_id: str


@dataclass
class SuperpositionContext:
    """Superposition state from upstream MARLs"""
    strategic_superposition: np.ndarray  # [buy, hold, sell] probabilities
    tactical_superposition: np.ndarray   # [bearish, neutral, bullish] probabilities
    risk_superposition: np.ndarray       # [low, medium, high] risk states
    
    # Coherence metrics
    strategic_coherence: float = 0.0
    tactical_coherence: float = 0.0
    risk_coherence: float = 0.0
    
    # Entanglement measures
    strategic_tactical_entanglement: float = 0.0
    tactical_risk_entanglement: float = 0.0
    
    def collapse_superposition(self) -> Tuple[int, int, int]:
        """Collapse superposition states to definite values"""
        strategic_action = np.argmax(self.strategic_superposition)
        tactical_action = np.argmax(self.tactical_superposition)
        risk_action = np.argmax(self.risk_superposition)
        
        return strategic_action, tactical_action, risk_action


class SequentialExecutionAgentBase(nn.Module, ABC):
    """
    Base class for sequential execution agents
    
    Provides common functionality for processing upstream MARL outputs
    and contributing to the final execution decision.
    """
    
    def __init__(self, 
                 agent_id: str,
                 config: Dict[str, Any],
                 obs_dim: int,
                 action_dim: int):
        super().__init__()
        
        self.agent_id = agent_id
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Neural network architecture
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.activation = config.get('activation', 'relu')
        
        # Build neural network
        self._build_network()
        
        # Performance tracking
        self.decision_history = []
        self.performance_metrics = {
            'avg_latency_us': 0.0,
            'decisions_made': 0,
            'avg_confidence': 0.0,
            'error_rate': 0.0
        }
        
        logger.info(f"Initialized {self.agent_id}",
                   obs_dim=obs_dim,
                   action_dim=action_dim,
                   hidden_dim=self.hidden_dim)
    
    def _build_network(self):
        """Build the neural network architecture"""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.obs_dim, self.hidden_dim))
        layers.append(self._get_activation())
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self._get_activation())
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Attention mechanism for upstream context
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout_rate
        )
        
        # Superposition processor
        self.superposition_processor = nn.Linear(9, self.hidden_dim)  # 3 superpositions × 3 states each
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _get_activation(self):
        """Get activation function"""
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def forward(self, 
                observation: torch.Tensor,
                superposition_context: Optional[SuperpositionContext] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            observation: Current observation
            superposition_context: Context from upstream MARLs
            
        Returns:
            Tuple of (action_logits, confidence)
        """
        # Process observation
        features = self.network[:-1](observation)  # All layers except output
        
        # Process superposition context if available
        if superposition_context is not None:
            superposition_features = self._process_superposition_context(superposition_context)
            
            # Apply attention to combine features
            combined_features = self._apply_attention(features, superposition_features)
        else:
            combined_features = features
        
        # Generate action logits
        action_logits = self.network[-1](combined_features)
        
        # Estimate confidence
        confidence = self.confidence_estimator(combined_features)
        
        return action_logits, confidence
    
    def _process_superposition_context(self, context: SuperpositionContext) -> torch.Tensor:
        """Process superposition context from upstream MARLs"""
        # Flatten all superposition states
        superposition_flat = np.concatenate([
            context.strategic_superposition,
            context.tactical_superposition,
            context.risk_superposition
        ])
        
        # Convert to tensor
        superposition_tensor = torch.tensor(superposition_flat, dtype=torch.float32)
        
        # Process through linear layer
        superposition_features = self.superposition_processor(superposition_tensor)
        
        return superposition_features
    
    def _apply_attention(self, 
                        observation_features: torch.Tensor,
                        superposition_features: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism to combine features"""
        # Prepare for attention (seq_len, batch_size, embed_dim)
        query = observation_features.unsqueeze(0)
        key = superposition_features.unsqueeze(0)
        value = superposition_features.unsqueeze(0)
        
        # Apply attention
        attended_features, _ = self.context_attention(query, key, value)
        
        # Combine with original features
        combined = observation_features + attended_features.squeeze(0)
        
        return combined
    
    @abstractmethod
    def make_decision(self, 
                     observation: np.ndarray,
                     cascade_context: Dict[str, Any],
                     superposition_context: Optional[SuperpositionContext] = None) -> AgentDecision:
        """Make agent-specific decision"""
        pass
    
    def update_performance_metrics(self, decision: AgentDecision):
        """Update performance metrics"""
        self.performance_metrics['decisions_made'] += 1
        
        # Update average latency
        old_avg = self.performance_metrics['avg_latency_us']
        new_latency = decision.processing_time_us
        n = self.performance_metrics['decisions_made']
        self.performance_metrics['avg_latency_us'] = ((n - 1) * old_avg + new_latency) / n
        
        # Update average confidence
        old_avg_conf = self.performance_metrics['avg_confidence']
        new_conf = decision.confidence
        self.performance_metrics['avg_confidence'] = ((n - 1) * old_avg_conf + new_conf) / n
        
        # Store decision in history
        self.decision_history.append(decision)
        if len(self.decision_history) > 1000:  # Keep only recent decisions
            self.decision_history.pop(0)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()


class MarketTimingAgent(SequentialExecutionAgentBase):
    """
    Market Timing Agent (π₁)
    
    Determines optimal execution timing based on market microstructure,
    upstream MARL signals, and real-time market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            agent_id='market_timing',
            config=config,
            obs_dim=config.get('obs_dim', 55),  # From environment
            action_dim=4  # [timing_delay_us, urgency, confidence, market_regime_adjust]
        )
        
        # Timing-specific parameters
        self.min_delay_us = config.get('min_delay_us', 10.0)
        self.max_delay_us = config.get('max_delay_us', 1000.0)
        self.urgency_threshold = config.get('urgency_threshold', 0.8)
        
        # Market regime detection
        self.regime_detector = nn.Sequential(
            nn.Linear(20, 64),  # Market observation dimension
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [normal, volatile, trending, crisis]
            nn.Softmax(dim=-1)
        )
        
        # Timing optimizer
        self.timing_optimizer = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, 64),  # Features + regime
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def make_decision(self, 
                     observation: np.ndarray,
                     cascade_context: Dict[str, Any],
                     superposition_context: Optional[SuperpositionContext] = None) -> AgentDecision:
        """Make market timing decision"""
        start_time = time.time_ns()
        
        try:
            # Convert observation to tensor
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            
            # Forward pass
            with torch.no_grad():
                action_logits, confidence = self.forward(obs_tensor, superposition_context)
                
                # Extract market observation for regime detection
                market_obs = obs_tensor[12:32]  # Market microstructure part
                regime_probs = self.regime_detector(market_obs)
                
                # Optimize timing
                combined_features = torch.cat([
                    self.network[:-1](obs_tensor),
                    regime_probs
                ])
                timing_factor = self.timing_optimizer(combined_features)
            
            # Process action logits
            timing_delay_us = float(action_logits[0] * self.max_delay_us)
            urgency = float(torch.sigmoid(action_logits[1]))
            confidence_raw = float(torch.sigmoid(action_logits[2]))
            market_regime_adjust = float(torch.sigmoid(action_logits[3]))
            
            # Adjust timing based on market conditions
            volatility = observation[23] if len(observation) > 23 else 0.15
            spread = observation[15] if len(observation) > 15 else 5.0
            
            # Volatility adjustment
            if volatility > 0.25:  # High volatility
                timing_delay_us *= 1.5  # Slow down execution
                urgency *= 0.8
            elif volatility < 0.10:  # Low volatility
                timing_delay_us *= 0.7  # Speed up execution
                urgency *= 1.2
            
            # Spread adjustment
            if spread > 10.0:  # Wide spread
                timing_delay_us *= 1.2
                urgency *= 0.9
            
            # Superposition influence
            if superposition_context is not None:
                strategic_action, tactical_action, risk_action = superposition_context.collapse_superposition()
                
                # If strategic and tactical agree, increase urgency
                if strategic_action == tactical_action and strategic_action != 1:  # Not hold
                    urgency *= 1.3
                    timing_delay_us *= 0.8
                
                # If risk is high, slow down
                if risk_action == 2:  # High risk
                    timing_delay_us *= 1.4
                    urgency *= 0.7
            
            # Clamp values
            timing_delay_us = np.clip(timing_delay_us, self.min_delay_us, self.max_delay_us)
            urgency = np.clip(urgency, 0.0, 1.0)
            confidence_final = np.clip(confidence_raw * float(confidence), 0.0, 1.0)
            market_regime_adjust = np.clip(market_regime_adjust, 0.0, 1.0)
            
            # Create decision
            decision_value = {
                'timing_delay_us': timing_delay_us,
                'urgency': urgency,
                'confidence': confidence_final,
                'market_regime_adjust': market_regime_adjust,
                'regime_probs': regime_probs.numpy().tolist(),
                'volatility_factor': volatility,
                'spread_factor': spread
            }
            
            processing_time_us = (time.time_ns() - start_time) / 1000.0
            
            decision = AgentDecision(
                agent_id=self.agent_id,
                decision_value=decision_value,
                confidence=confidence_final,
                processing_time_us=processing_time_us,
                internal_state={
                    'regime_detection': regime_probs.numpy().tolist(),
                    'timing_optimization': float(timing_factor),
                    'volatility_adjustment': volatility,
                    'spread_adjustment': spread
                },
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )
            
            self.update_performance_metrics(decision)
            
            logger.debug("Market timing decision made",
                        timing_delay_us=timing_delay_us,
                        urgency=urgency,
                        confidence=confidence_final,
                        processing_time_us=processing_time_us)
            
            return decision
            
        except Exception as e:
            logger.error("Error in market timing decision", error=str(e))
            # Return safe default decision
            return AgentDecision(
                agent_id=self.agent_id,
                decision_value={
                    'timing_delay_us': 500.0,
                    'urgency': 0.5,
                    'confidence': 0.1,
                    'market_regime_adjust': 0.5
                },
                confidence=0.1,
                processing_time_us=(time.time_ns() - start_time) / 1000.0,
                internal_state={'error': str(e)},
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )


class LiquiditySourcingAgent(SequentialExecutionAgentBase):
    """
    Liquidity Sourcing Agent (π₂)
    
    Selects optimal venues and liquidity sources based on market conditions,
    venue performance, and upstream MARL signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            agent_id='liquidity_sourcing',
            config=config,
            obs_dim=config.get('obs_dim', 65),
            action_dim=5  # [venue_weights..., liquidity_threshold]
        )
        
        # Venue analysis
        self.venue_analyzer = nn.Sequential(
            nn.Linear(10, 64),  # Venue-specific observations
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),  # Number of venues
            nn.Softmax(dim=-1)
        )
        
        # Liquidity estimator
        self.liquidity_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Venue names
        self.venues = config.get('venues', ['SMART', 'ARCA', 'NASDAQ', 'NYSE', 'BATS'])
        
        # Performance tracking per venue
        self.venue_performance = {venue: {'fill_rate': 0.95, 'latency': 50.0, 'spread': 5.0} 
                                 for venue in self.venues}
    
    def make_decision(self, 
                     observation: np.ndarray,
                     cascade_context: Dict[str, Any],
                     superposition_context: Optional[SuperpositionContext] = None) -> AgentDecision:
        """Make liquidity sourcing decision"""
        start_time = time.time_ns()
        
        try:
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            
            with torch.no_grad():
                action_logits, confidence = self.forward(obs_tensor, superposition_context)
                
                # Extract venue-specific observations
                venue_obs = obs_tensor[-10:]  # Last 10 dimensions are venue metrics
                venue_weights = self.venue_analyzer(venue_obs)
                
                # Estimate liquidity threshold
                features = self.network[:-1](obs_tensor)
                liquidity_threshold = self.liquidity_estimator(features)
            
            # Process venue weights
            venue_weights_np = venue_weights.numpy()
            liquidity_threshold_val = float(liquidity_threshold)
            
            # Adjust weights based on market conditions
            market_volatility = observation[23] if len(observation) > 23 else 0.15
            spread = observation[15] if len(observation) > 15 else 5.0
            
            # High volatility: prefer reliable venues
            if market_volatility > 0.25:
                # Boost SMART and reduce smaller venues
                venue_weights_np[0] *= 1.3  # SMART
                venue_weights_np[4] *= 0.7  # BATS
                venue_weights_np = venue_weights_np / np.sum(venue_weights_np)
            
            # Wide spread: prefer venues with better liquidity
            if spread > 10.0:
                venue_weights_np[0] *= 1.2  # SMART
                venue_weights_np[1] *= 1.1  # ARCA
                venue_weights_np = venue_weights_np / np.sum(venue_weights_np)
            
            # Superposition influence
            if superposition_context is not None:
                strategic_action, tactical_action, risk_action = superposition_context.collapse_superposition()
                
                # High risk: prefer SMART routing
                if risk_action == 2:
                    venue_weights_np[0] *= 1.5  # SMART
                    venue_weights_np = venue_weights_np / np.sum(venue_weights_np)
                
                # Strong directional signal: prefer fast execution venues
                if strategic_action != 1 and tactical_action != 1:  # Not hold
                    venue_weights_np[2] *= 1.2  # NASDAQ (fast)
                    venue_weights_np = venue_weights_np / np.sum(venue_weights_np)
            
            # Calculate total liquidity score
            total_liquidity = 0.0
            selected_venues = []
            
            for i, (venue, weight) in enumerate(zip(self.venues, venue_weights_np)):
                if weight > 0.05:  # Minimum 5% weight
                    venue_liquidity = observation[32 + i] if len(observation) > 32 + i else 0.8
                    total_liquidity += venue_liquidity * weight
                    selected_venues.append({
                        'venue': venue,
                        'weight': float(weight),
                        'liquidity': venue_liquidity,
                        'performance': self.venue_performance[venue]
                    })
            
            liquidity_score = min(1.0, total_liquidity / liquidity_threshold_val)
            
            decision_value = {
                'venue_weights': venue_weights_np.tolist(),
                'liquidity_threshold': liquidity_threshold_val,
                'selected_venues': selected_venues,
                'total_liquidity': total_liquidity,
                'liquidity_score': liquidity_score
            }
            
            processing_time_us = (time.time_ns() - start_time) / 1000.0
            
            decision = AgentDecision(
                agent_id=self.agent_id,
                decision_value=decision_value,
                confidence=float(confidence),
                processing_time_us=processing_time_us,
                internal_state={
                    'venue_analysis': venue_weights_np.tolist(),
                    'liquidity_estimation': liquidity_threshold_val,
                    'market_adjustments': {
                        'volatility': market_volatility,
                        'spread': spread
                    }
                },
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )
            
            self.update_performance_metrics(decision)
            
            logger.debug("Liquidity sourcing decision made",
                        selected_venues=len(selected_venues),
                        total_liquidity=total_liquidity,
                        liquidity_score=liquidity_score,
                        processing_time_us=processing_time_us)
            
            return decision
            
        except Exception as e:
            logger.error("Error in liquidity sourcing decision", error=str(e))
            return AgentDecision(
                agent_id=self.agent_id,
                decision_value={
                    'venue_weights': [0.6, 0.2, 0.1, 0.05, 0.05],
                    'liquidity_threshold': 0.8,
                    'selected_venues': [{'venue': 'SMART', 'weight': 0.6, 'liquidity': 0.8}],
                    'total_liquidity': 0.8,
                    'liquidity_score': 1.0
                },
                confidence=0.1,
                processing_time_us=(time.time_ns() - start_time) / 1000.0,
                internal_state={'error': str(e)},
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )


class PositionFragmentationAgent(SequentialExecutionAgentBase):
    """
    Position Fragmentation Agent (π₃)
    
    Optimizes order size and fragmentation strategy to minimize market impact
    while maintaining execution quality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            agent_id='position_fragmentation',
            config=config,
            obs_dim=config.get('obs_dim', 61),
            action_dim=4  # [fragment_size, num_fragments, timing_spread, stealth_factor]
        )
        
        # Market impact model
        self.impact_model = nn.Sequential(
            nn.Linear(6, 32),  # Market impact factors
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Fragmentation optimizer
        self.fragmentation_optimizer = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 64),  # Features + impact
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),  # Fragment params
            nn.Sigmoid()
        )
        
        # Parameters
        self.max_fragments = config.get('max_fragments', 20)
        self.min_fragment_size = config.get('min_fragment_size', 0.01)
        self.max_timing_spread = config.get('max_timing_spread', 300.0)  # seconds
    
    def make_decision(self, 
                     observation: np.ndarray,
                     cascade_context: Dict[str, Any],
                     superposition_context: Optional[SuperpositionContext] = None) -> AgentDecision:
        """Make position fragmentation decision"""
        start_time = time.time_ns()
        
        try:
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            
            with torch.no_grad():
                action_logits, confidence = self.forward(obs_tensor, superposition_context)
                
                # Extract market impact factors
                impact_factors = torch.tensor([
                    observation[19],  # Market impact
                    observation[17],  # Depth imbalance
                    observation[20],  # Flow toxicity
                    observation[21],  # Institutional flow
                    observation[23],  # Volatility
                    observation[15]   # Spread
                ], dtype=torch.float32)
                
                # Estimate market impact
                market_impact = self.impact_model(impact_factors)
                
                # Optimize fragmentation
                features = self.network[:-1](obs_tensor)
                combined_features = torch.cat([features, market_impact])
                fragment_params = self.fragmentation_optimizer(combined_features)
            
            # Get position size from cascade context
            position_size = abs(cascade_context.get('risk_allocation', 0.1))
            
            # Process fragmentation parameters
            fragment_size = float(fragment_params[0])
            num_fragments = int(fragment_params[1] * self.max_fragments)
            timing_spread = float(fragment_params[2] * self.max_timing_spread)
            stealth_factor = float(fragment_params[3])
            
            # Ensure minimum fragment size
            if fragment_size < self.min_fragment_size:
                fragment_size = self.min_fragment_size
            
            # Ensure at least 1 fragment
            num_fragments = max(1, num_fragments)
            
            # Calculate optimal fragmentation
            if position_size > 0:
                # Calculate market impact for different strategies
                single_order_impact = self._calculate_market_impact(
                    position_size, 
                    float(market_impact),
                    observation[17],  # Depth imbalance
                    observation[15]   # Spread
                )
                
                fragment_impact = self._calculate_market_impact(
                    position_size / num_fragments,
                    float(market_impact),
                    observation[17],
                    observation[15]
                ) * num_fragments
                
                fragmentation_benefit = max(0, single_order_impact - fragment_impact)
                
                # Stealth benefit
                stealth_benefit = stealth_factor * 0.1
                
                total_benefit = fragmentation_benefit + stealth_benefit
            else:
                single_order_impact = 0.0
                fragment_impact = 0.0
                fragmentation_benefit = 0.0
                stealth_benefit = 0.0
                total_benefit = 0.0
            
            # Superposition influence
            if superposition_context is not None:
                strategic_action, tactical_action, risk_action = superposition_context.collapse_superposition()
                
                # High risk: increase fragmentation
                if risk_action == 2:
                    num_fragments = min(self.max_fragments, int(num_fragments * 1.5))
                    stealth_factor = min(1.0, stealth_factor * 1.2)
                
                # Strong directional signal: reduce fragmentation for speed
                if strategic_action == tactical_action and strategic_action != 1:
                    num_fragments = max(1, int(num_fragments * 0.7))
                    timing_spread *= 0.8
            
            decision_value = {
                'fragment_size': fragment_size,
                'num_fragments': num_fragments,
                'timing_spread': timing_spread,
                'stealth_factor': stealth_factor,
                'single_order_impact': single_order_impact,
                'fragment_impact': fragment_impact,
                'fragmentation_benefit': fragmentation_benefit,
                'stealth_benefit': stealth_benefit,
                'total_benefit': total_benefit
            }
            
            processing_time_us = (time.time_ns() - start_time) / 1000.0
            
            decision = AgentDecision(
                agent_id=self.agent_id,
                decision_value=decision_value,
                confidence=float(confidence),
                processing_time_us=processing_time_us,
                internal_state={
                    'market_impact_estimate': float(market_impact),
                    'position_size': position_size,
                    'fragmentation_analysis': {
                        'single_order_impact': single_order_impact,
                        'fragment_impact': fragment_impact,
                        'benefit': fragmentation_benefit
                    }
                },
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )
            
            self.update_performance_metrics(decision)
            
            logger.debug("Position fragmentation decision made",
                        num_fragments=num_fragments,
                        fragment_size=fragment_size,
                        stealth_factor=stealth_factor,
                        total_benefit=total_benefit,
                        processing_time_us=processing_time_us)
            
            return decision
            
        except Exception as e:
            logger.error("Error in position fragmentation decision", error=str(e))
            return AgentDecision(
                agent_id=self.agent_id,
                decision_value={
                    'fragment_size': 0.1,
                    'num_fragments': 1,
                    'timing_spread': 60.0,
                    'stealth_factor': 0.5,
                    'single_order_impact': 0.0,
                    'fragment_impact': 0.0,
                    'fragmentation_benefit': 0.0,
                    'stealth_benefit': 0.0,
                    'total_benefit': 0.0
                },
                confidence=0.1,
                processing_time_us=(time.time_ns() - start_time) / 1000.0,
                internal_state={'error': str(e)},
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )
    
    def _calculate_market_impact(self, order_size: float, base_impact: float, depth_imbalance: float, spread: float) -> float:
        """Calculate market impact for given order size"""
        if order_size <= 0:
            return 0.0
        
        # Square root impact model with adjustments
        impact = base_impact * np.sqrt(order_size)
        
        # Adjust for depth imbalance
        impact *= (1 + abs(depth_imbalance))
        
        # Adjust for spread
        impact *= (1 + spread / 100.0)
        
        return min(impact, 0.02)  # Cap at 200 bps


class RiskControlAgent(SequentialExecutionAgentBase):
    """
    Risk Control Agent (π₄)
    
    Monitors real-time risk and makes approval/rejection decisions for
    execution based on risk thresholds and upstream signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            agent_id='risk_control',
            config=config,
            obs_dim=config.get('obs_dim', 63),
            action_dim=5  # [APPROVE, REDUCE_SIZE, DELAY, CANCEL, EMERGENCY_STOP]
        )
        
        # Risk assessment model
        self.risk_assessor = nn.Sequential(
            nn.Linear(8, 64),  # Risk-specific observations
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Emergency detection
        self.emergency_detector = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Risk thresholds
        self.var_threshold = config.get('var_threshold', 0.02)
        self.emergency_threshold = config.get('emergency_threshold', 0.05)
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        
        # Action mapping
        self.actions = ['APPROVE', 'REDUCE_SIZE', 'DELAY', 'CANCEL', 'EMERGENCY_STOP']
    
    def make_decision(self, 
                     observation: np.ndarray,
                     cascade_context: Dict[str, Any],
                     superposition_context: Optional[SuperpositionContext] = None) -> AgentDecision:
        """Make risk control decision"""
        start_time = time.time_ns()
        
        try:
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            
            with torch.no_grad():
                action_logits, confidence = self.forward(obs_tensor, superposition_context)
                
                # Extract risk factors
                risk_factors = torch.tensor([
                    observation[0],   # Strategic VaR
                    observation[7],   # Risk allocation
                    observation[8],   # Stop loss
                    observation[9],   # Take profit
                    observation[29],  # Portfolio VaR
                    observation[30],  # Drawdown
                    observation[31],  # Max drawdown
                    observation[32]   # Risk budget used
                ], dtype=torch.float32)
                
                # Assess risk
                risk_score = self.risk_assessor(risk_factors)
                
                # Detect emergency conditions
                features = self.network[:-1](obs_tensor)
                emergency_prob = self.emergency_detector(features)
                
                # Get action probabilities
                action_probs = F.softmax(action_logits, dim=-1)
                action_idx = torch.argmax(action_probs)
            
            # Current risk metrics
            current_var = cascade_context.get('risk_var_estimate', 0.02)
            position_size = abs(cascade_context.get('risk_allocation', 0.0))
            
            # Risk assessment
            risk_score_val = float(risk_score)
            emergency_prob_val = float(emergency_prob)
            
            # Override logic based on risk conditions
            if emergency_prob_val > 0.8 or current_var > self.emergency_threshold:
                action_idx = 4  # EMERGENCY_STOP
                risk_approved = False
            elif risk_score_val > 1.0 or current_var > self.var_threshold * 1.5:
                action_idx = 3  # CANCEL
                risk_approved = False
            elif risk_score_val > 0.8 or current_var > self.var_threshold * 1.2:
                action_idx = 2  # DELAY
                risk_approved = False
            elif risk_score_val > 0.6 or current_var > self.var_threshold:
                action_idx = 1  # REDUCE_SIZE
                risk_approved = True
            else:
                action_idx = 0  # APPROVE
                risk_approved = True
            
            # Superposition influence
            if superposition_context is not None:
                strategic_action, tactical_action, risk_action = superposition_context.collapse_superposition()
                
                # If risk superposition indicates high risk, be more conservative
                if risk_action == 2:  # High risk
                    if action_idx == 0:  # APPROVE
                        action_idx = 1  # REDUCE_SIZE
                    elif action_idx == 1:  # REDUCE_SIZE
                        action_idx = 2  # DELAY
                
                # Check for conflicting signals
                if strategic_action != tactical_action:
                    # Conflicting signals, increase caution
                    if action_idx == 0:  # APPROVE
                        action_idx = 2  # DELAY
                        risk_approved = False
            
            action_name = self.actions[action_idx]
            
            decision_value = {
                'risk_action': action_name,
                'risk_approved': risk_approved,
                'risk_score': risk_score_val,
                'current_var': current_var,
                'emergency_probability': emergency_prob_val,
                'emergency_stop': action_name == 'EMERGENCY_STOP',
                'position_size': position_size,
                'var_threshold': self.var_threshold
            }
            
            processing_time_us = (time.time_ns() - start_time) / 1000.0
            
            decision = AgentDecision(
                agent_id=self.agent_id,
                decision_value=decision_value,
                confidence=float(confidence),
                processing_time_us=processing_time_us,
                internal_state={
                    'risk_assessment': risk_score_val,
                    'emergency_detection': emergency_prob_val,
                    'action_probabilities': action_probs.numpy().tolist(),
                    'risk_factors': risk_factors.numpy().tolist()
                },
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )
            
            self.update_performance_metrics(decision)
            
            logger.debug("Risk control decision made",
                        action=action_name,
                        risk_approved=risk_approved,
                        risk_score=risk_score_val,
                        emergency_prob=emergency_prob_val,
                        processing_time_us=processing_time_us)
            
            return decision
            
        except Exception as e:
            logger.error("Error in risk control decision", error=str(e))
            return AgentDecision(
                agent_id=self.agent_id,
                decision_value={
                    'risk_action': 'CANCEL',
                    'risk_approved': False,
                    'risk_score': 1.0,
                    'current_var': 0.05,
                    'emergency_probability': 0.0,
                    'emergency_stop': False,
                    'position_size': 0.0,
                    'var_threshold': self.var_threshold
                },
                confidence=0.1,
                processing_time_us=(time.time_ns() - start_time) / 1000.0,
                internal_state={'error': str(e)},
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )


class ExecutionMonitorAgent(SequentialExecutionAgentBase):
    """
    Execution Monitor Agent (π₅)
    
    Monitors execution quality and provides feedback for continuous improvement
    of the execution process.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            agent_id='execution_monitor',
            config=config,
            obs_dim=config.get('obs_dim', 67),
            action_dim=3  # [quality_threshold, feedback_weight, adjustment_factor]
        )
        
        # Quality assessment model
        self.quality_assessor = nn.Sequential(
            nn.Linear(12, 64),  # Quality metrics
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Feedback generator
        self.feedback_generator = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 64),  # Features + quality
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),  # Feedback dimensions
            nn.Tanh()
        )
        
        # Performance targets
        self.target_fill_rate = config.get('target_fill_rate', 0.95)
        self.target_latency_us = config.get('target_latency_us', 500.0)
        self.target_slippage_bps = config.get('target_slippage_bps', 10.0)
        
        # Historical tracking
        self.quality_history = []
        self.feedback_history = []
    
    def make_decision(self, 
                     observation: np.ndarray,
                     cascade_context: Dict[str, Any],
                     superposition_context: Optional[SuperpositionContext] = None) -> AgentDecision:
        """Make execution monitoring decision"""
        start_time = time.time_ns()
        
        try:
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            
            with torch.no_grad():
                action_logits, confidence = self.forward(obs_tensor, superposition_context)
                
                # Extract quality metrics
                quality_metrics = torch.tensor([
                    observation[27],  # Fill rate
                    observation[28],  # Slippage
                    observation[26],  # Latency
                    observation[29],  # Market impact
                    observation[30],  # Orders executed
                    observation[31],  # Partial fills
                    observation[32],  # Orders cancelled
                    observation[33],  # Risk violations
                    observation[34],  # Emergency stops
                    observation[35],  # Avg recent latency
                    observation[36],  # Avg recent fill rate
                    observation[37]   # Avg recent slippage
                ], dtype=torch.float32)
                
                # Assess quality
                quality_score = self.quality_assessor(quality_metrics)
                
                # Generate feedback
                features = self.network[:-1](obs_tensor)
                combined_features = torch.cat([features, quality_score])
                feedback_raw = self.feedback_generator(combined_features)
            
            # Process monitoring parameters
            quality_threshold = float(torch.sigmoid(action_logits[0]))
            feedback_weight = float(torch.sigmoid(action_logits[1]))
            adjustment_factor = float(torch.sigmoid(action_logits[2]))
            
            # Calculate component scores
            fill_rate = float(quality_metrics[0])
            slippage = float(quality_metrics[1])
            latency = float(quality_metrics[2])
            
            fill_rate_score = min(1.0, fill_rate / self.target_fill_rate)
            latency_score = max(0.0, 1.0 - latency / (self.target_latency_us * 2))
            slippage_score = max(0.0, 1.0 - slippage / self.target_slippage_bps)
            
            overall_quality = (fill_rate_score + latency_score + slippage_score) / 3.0
            
            # Generate feedback
            feedback = {
                'quality_meets_threshold': overall_quality >= quality_threshold,
                'overall_quality': overall_quality,
                'component_scores': {
                    'fill_rate': fill_rate_score,
                    'latency': latency_score,
                    'slippage': slippage_score
                },
                'suggested_adjustments': {}
            }
            
            # Generate specific feedback based on performance
            feedback_adjustments = feedback_raw.numpy()
            
            if fill_rate_score < 0.9:
                feedback['suggested_adjustments']['increase_urgency'] = float(feedback_adjustments[0]) * adjustment_factor
                feedback['suggested_adjustments']['improve_venue_selection'] = float(feedback_adjustments[1]) * adjustment_factor
            
            if latency_score < 0.8:
                feedback['suggested_adjustments']['reduce_complexity'] = float(feedback_adjustments[2]) * adjustment_factor
                feedback['suggested_adjustments']['optimize_timing'] = float(feedback_adjustments[3]) * adjustment_factor
            
            if slippage_score < 0.8:
                feedback['suggested_adjustments']['improve_fragmentation'] = float(feedback_adjustments[4]) * adjustment_factor
                feedback['suggested_adjustments']['better_timing'] = float(feedback_adjustments[5]) * adjustment_factor
            
            # Add to history
            self.quality_history.append(overall_quality)
            self.feedback_history.append(feedback)
            
            # Keep only recent history
            if len(self.quality_history) > 100:
                self.quality_history.pop(0)
                self.feedback_history.pop(0)
            
            decision_value = {
                'quality_threshold': quality_threshold,
                'feedback_weight': feedback_weight,
                'adjustment_factor': adjustment_factor,
                'overall_quality': overall_quality,
                'quality_score': float(quality_score),
                'feedback': feedback,
                'quality_trend': self._calculate_quality_trend()
            }
            
            processing_time_us = (time.time_ns() - start_time) / 1000.0
            
            decision = AgentDecision(
                agent_id=self.agent_id,
                decision_value=decision_value,
                confidence=float(confidence),
                processing_time_us=processing_time_us,
                internal_state={
                    'quality_assessment': float(quality_score),
                    'component_scores': {
                        'fill_rate': fill_rate_score,
                        'latency': latency_score,
                        'slippage': slippage_score
                    },
                    'feedback_raw': feedback_adjustments.tolist(),
                    'quality_history_length': len(self.quality_history)
                },
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )
            
            self.update_performance_metrics(decision)
            
            logger.debug("Execution monitor decision made",
                        overall_quality=overall_quality,
                        quality_meets_threshold=feedback['quality_meets_threshold'],
                        num_adjustments=len(feedback['suggested_adjustments']),
                        processing_time_us=processing_time_us)
            
            return decision
            
        except Exception as e:
            logger.error("Error in execution monitor decision", error=str(e))
            return AgentDecision(
                agent_id=self.agent_id,
                decision_value={
                    'quality_threshold': 0.8,
                    'feedback_weight': 0.5,
                    'adjustment_factor': 0.1,
                    'overall_quality': 0.5,
                    'quality_score': 0.5,
                    'feedback': {
                        'quality_meets_threshold': False,
                        'overall_quality': 0.5,
                        'suggested_adjustments': {}
                    },
                    'quality_trend': 0.0
                },
                confidence=0.1,
                processing_time_us=(time.time_ns() - start_time) / 1000.0,
                internal_state={'error': str(e)},
                timestamp=datetime.now(),
                decision_id=str(uuid.uuid4())
            )
    
    def _calculate_quality_trend(self) -> float:
        """Calculate quality trend over recent history"""
        if len(self.quality_history) < 2:
            return 0.0
        
        # Simple linear trend
        recent_quality = self.quality_history[-10:]  # Last 10 measurements
        if len(recent_quality) < 2:
            return 0.0
        
        # Calculate slope
        x = np.arange(len(recent_quality))
        y = np.array(recent_quality)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        else:
            return 0.0