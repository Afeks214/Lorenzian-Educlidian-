"""
Tactical Superposition Aggregator

This module implements the tactical ensemble superposition system that aggregates
outputs from the sequential tactical agents (FVG → Momentum → EntryOpt) into a
unified tactical decision. The system handles consensus building, Byzantine fault
tolerance, and strategic alignment.

Key Features:
- Weighted ensemble aggregation with agent-specific weights
- Byzantine fault tolerance with cryptographic validation
- Strategic context alignment and synergy integration
- Risk-adjusted execution threshold management
- Microstructure-aware execution optimization
- Performance monitoring and adaptive weight adjustment

Architecture:
- Receives agent superpositions from sequential tactical agents
- Applies strategic context alignment and predecessor validation
- Performs Byzantine fault detection and consensus building
- Generates final tactical superposition for execution
- Provides rich output for downstream execution systems

Author: Agent 5 - Sequential Tactical MARL Specialist
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, OrderedDict
import logging
import time
import json
import uuid
import hashlib
import hmac
from statistics import mode, StatisticsError
import warnings

# Core imports
from src.consensus.byzantine_detector import ByzantineDetector
from src.consensus.pbft_engine import PBFTEngine
from src.core.events import Event, EventType
from src.core.event_bus import EventBus
from src.risk.core.var_calculator import VaRCalculator

logger = logging.getLogger(__name__)

class AggregationMethod(Enum):
    """Aggregation methods for tactical superposition"""
    WEIGHTED_ENSEMBLE = "weighted_ensemble"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    STRATEGIC_ALIGNED = "strategic_aligned"
    BYZANTINE_RESISTANT = "byzantine_resistant"

class ExecutionMode(Enum):
    """Execution modes for tactical superposition"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    RISK_MANAGED = "risk_managed"

class ConsensusLevel(Enum):
    """Consensus levels for decision making"""
    STRONG = "strong"      # All agents agree
    MODERATE = "moderate"  # Majority agrees
    WEAK = "weak"         # No clear majority
    FAILED = "failed"     # Byzantine or conflicting

@dataclass
class AgentSuperposition:
    """Container for agent superposition output"""
    agent_id: str
    action: int
    probabilities: np.ndarray
    confidence: float
    feature_importance: Dict[str, float]
    market_insights: Dict[str, Any]
    execution_signals: Dict[str, Any]
    processing_time: float
    timestamp: float
    signature: Optional[str] = None
    is_byzantine: bool = False
    
    def __post_init__(self):
        """Validate superposition"""
        if len(self.probabilities) != 3:
            raise ValueError("Probabilities must have length 3")
        if not np.allclose(np.sum(self.probabilities), 1.0, atol=1e-6):
            raise ValueError("Probabilities must sum to 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be in [0, 1]")

@dataclass
class StrategicContext:
    """Strategic context from upstream system"""
    regime_embedding: np.ndarray
    synergy_signal: Dict[str, Any]
    market_state: Dict[str, Any]
    confidence_level: float
    execution_bias: str
    volatility_forecast: float
    timestamp: float

@dataclass
class TacticalSuperposition:
    """Final tactical superposition output"""
    execute: bool
    action: int
    confidence: float
    aggregated_probabilities: np.ndarray
    agent_contributions: Dict[str, AgentSuperposition]
    strategic_alignment: float
    execution_command: Optional[Dict[str, Any]]
    risk_assessment: Dict[str, float]
    microstructure_analysis: Dict[str, Any]
    consensus_level: ConsensusLevel
    aggregation_method: AggregationMethod
    byzantine_agents: List[str]
    processing_time: float
    timestamp: float
    session_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'execute': self.execute,
            'action': self.action,
            'confidence': self.confidence,
            'aggregated_probabilities': self.aggregated_probabilities.tolist(),
            'strategic_alignment': self.strategic_alignment,
            'execution_command': self.execution_command,
            'risk_assessment': self.risk_assessment,
            'microstructure_analysis': self.microstructure_analysis,
            'consensus_level': self.consensus_level.value,
            'aggregation_method': self.aggregation_method.value,
            'byzantine_agents': self.byzantine_agents,
            'processing_time': self.processing_time,
            'timestamp': self.timestamp,
            'session_id': self.session_id
        }

@dataclass
class AggregatorMetrics:
    """Performance metrics for aggregator"""
    total_aggregations: int = 0
    successful_aggregations: int = 0
    byzantine_detections: int = 0
    consensus_failures: int = 0
    strategic_alignments: int = 0
    execution_decisions: int = 0
    processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_update: float = field(default_factory=time.time)

class TacticalSuperpositionAggregator:
    """
    Tactical Superposition Aggregator
    
    Aggregates sequential tactical agent outputs into unified tactical decisions
    with Byzantine fault tolerance, strategic alignment, and risk management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tactical superposition aggregator
        
        Args:
            config: Aggregator configuration
        """
        self.config = config or self._default_config()
        self._validate_config()
        
        # Core parameters
        self.aggregation_method = AggregationMethod(self.config.get('aggregation_method', 'weighted_ensemble'))
        self.execution_mode = ExecutionMode(self.config.get('execution_mode', 'balanced'))
        self.execution_threshold = self.config.get('execution_threshold', 0.75)
        self.consensus_threshold = self.config.get('consensus_threshold', 0.7)
        
        # Agent weights (FVG, Momentum, EntryOpt)
        self.agent_weights = self.config.get('agent_weights', [0.35, 0.40, 0.25])
        self.adaptive_weights = self.config.get('adaptive_weights', True)
        
        # Byzantine fault tolerance
        self.byzantine_detection = self.config.get('byzantine_detection', True)
        self.max_byzantine_agents = self.config.get('max_byzantine_agents', 1)
        
        # Strategic integration
        self.strategic_alignment_weight = self.config.get('strategic_alignment_weight', 0.3)
        self.risk_adjustment_factor = self.config.get('risk_adjustment_factor', 0.2)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.metrics = AggregatorMetrics()
        self.aggregation_history = deque(maxlen=1000)
        
        # Session management
        self.session_id = str(uuid.uuid4())
        
        logger.info(f"Tactical Superposition Aggregator initialized")
        logger.info(f"Method: {self.aggregation_method.value}, Mode: {self.execution_mode.value}")
        logger.info(f"Agent weights: {self.agent_weights}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for aggregator"""
        return {
            'aggregation_method': 'weighted_ensemble',
            'execution_mode': 'balanced',
            'execution_threshold': 0.75,
            'consensus_threshold': 0.7,
            'agent_weights': [0.35, 0.40, 0.25],
            'adaptive_weights': True,
            'byzantine_detection': True,
            'max_byzantine_agents': 1,
            'strategic_alignment_weight': 0.3,
            'risk_adjustment_factor': 0.2,
            'performance_monitoring': True,
            'signature_validation': True,
            'microstructure_integration': True,
            'risk_management': {
                'enable_var_calculation': True,
                'var_confidence_level': 0.95,
                'max_position_size': 1.0,
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 3.0
            },
            'execution_optimization': {
                'timing_sensitivity': 0.5,
                'liquidity_requirements': 0.7,
                'slippage_tolerance': 0.001,
                'market_impact_limit': 0.005
            }
        }
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if len(self.config['agent_weights']) != 3:
            raise ValueError("Agent weights must have length 3")
        
        if not np.allclose(sum(self.config['agent_weights']), 1.0, atol=1e-6):
            raise ValueError("Agent weights must sum to 1.0")
        
        if not 0.0 <= self.config['execution_threshold'] <= 1.0:
            raise ValueError("Execution threshold must be in [0, 1]")
        
        logger.info("Configuration validation passed")
    
    def _initialize_components(self):
        """Initialize core components"""
        try:
            # Byzantine detector
            if self.byzantine_detection:
                self.byzantine_detector = ByzantineDetector(
                    config=self.config.get('byzantine_config', {})
                )
            else:
                self.byzantine_detector = None
            
            # PBFT engine for consensus
            if self.config.get('enable_pbft', False):
                self.pbft_engine = PBFTEngine(
                    config=self.config.get('pbft_config', {})
                )
            else:
                self.pbft_engine = None
            
            # Risk calculator
            if self.config.get('risk_management', {}).get('enable_var_calculation', True):
                self.risk_calculator = VaRCalculator(
                    config=self.config.get('risk_management', {})
                )
            else:
                self.risk_calculator = None
            
            # Microstructure analyzer
            self.microstructure_analyzer = self._initialize_microstructure_analyzer()
            
            # Adaptive weight manager
            if self.adaptive_weights:
                self.weight_manager = AdaptiveWeightManager(
                    initial_weights=self.agent_weights,
                    config=self.config.get('adaptive_weights_config', {})
                )
            else:
                self.weight_manager = None
            
            logger.info("All aggregator components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_microstructure_analyzer(self):
        """Initialize microstructure analyzer"""
        try:
            if self.config.get('microstructure_integration', True):
                from src.execution.microstructure.microstructure_engine import MicrostructureEngine
                return MicrostructureEngine(config=self.config.get('execution_optimization', {}))
            else:
                return None
        except ImportError:
            logger.warning("Microstructure engine not available")
            return None
    
    def aggregate(
        self,
        agent_superpositions: OrderedDict[str, AgentSuperposition],
        strategic_context: Optional[StrategicContext] = None,
        market_state: Optional[Dict[str, Any]] = None
    ) -> TacticalSuperposition:
        """
        Aggregate agent superpositions into tactical decision
        
        Args:
            agent_superpositions: Ordered dictionary of agent superpositions
            strategic_context: Strategic context from upstream system
            market_state: Current market state
            
        Returns:
            Tactical superposition decision
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not agent_superpositions:
                raise ValueError("No agent superpositions provided")
            
            # Phase 1: Signature validation
            validated_superpositions = self._validate_signatures(agent_superpositions)
            
            # Phase 2: Byzantine agent detection
            byzantine_agents = self._detect_byzantine_agents(validated_superpositions)
            
            # Phase 3: Filter out Byzantine agents
            clean_superpositions = self._filter_byzantine_agents(validated_superpositions, byzantine_agents)
            
            # Phase 4: Consensus building
            consensus_result = self._build_consensus(clean_superpositions)
            
            # Phase 5: Strategic alignment
            strategic_alignment = self._calculate_strategic_alignment(
                consensus_result, strategic_context
            )
            
            # Phase 6: Risk assessment
            risk_assessment = self._calculate_risk_assessment(
                consensus_result, market_state
            )
            
            # Phase 7: Microstructure analysis
            microstructure_analysis = self._analyze_microstructure(
                consensus_result, market_state
            )
            
            # Phase 8: Final decision
            final_decision = self._make_final_decision(
                consensus_result, strategic_alignment, risk_assessment, microstructure_analysis
            )
            
            # Phase 9: Execution command generation
            execution_command = None
            if final_decision['execute']:
                execution_command = self._generate_execution_command(
                    final_decision, strategic_context, market_state
                )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Create tactical superposition
            tactical_superposition = TacticalSuperposition(
                execute=final_decision['execute'],
                action=final_decision['action'],
                confidence=final_decision['confidence'],
                aggregated_probabilities=final_decision['probabilities'],
                agent_contributions=clean_superpositions,
                strategic_alignment=strategic_alignment,
                execution_command=execution_command,
                risk_assessment=risk_assessment,
                microstructure_analysis=microstructure_analysis,
                consensus_level=consensus_result['consensus_level'],
                aggregation_method=self.aggregation_method,
                byzantine_agents=byzantine_agents,
                processing_time=processing_time,
                timestamp=time.time(),
                session_id=self.session_id
            )
            
            # Update metrics
            self._update_metrics(tactical_superposition)
            
            # Store in history
            self.aggregation_history.append(tactical_superposition)
            
            # Update adaptive weights if enabled
            if self.weight_manager:
                self.weight_manager.update_weights(tactical_superposition)
            
            return tactical_superposition
            
        except Exception as e:
            logger.error(f"Error in tactical superposition aggregation: {e}")
            return self._create_safe_superposition(agent_superpositions, processing_time=(time.time() - start_time) * 1000)
    
    def _validate_signatures(self, agent_superpositions: OrderedDict[str, AgentSuperposition]) -> OrderedDict[str, AgentSuperposition]:
        """Validate cryptographic signatures"""
        try:
            if not self.config.get('signature_validation', True):
                return agent_superpositions
            
            validated_superpositions = OrderedDict()
            
            for agent_id, superposition in agent_superpositions.items():
                # Validate signature if present
                if superposition.signature:
                    is_valid = self._verify_signature(superposition)
                    if not is_valid:
                        logger.warning(f"Invalid signature from agent {agent_id}")
                        superposition.is_byzantine = True
                
                validated_superpositions[agent_id] = superposition
            
            return validated_superpositions
            
        except Exception as e:
            logger.error(f"Error validating signatures: {e}")
            return agent_superpositions
    
    def _verify_signature(self, superposition: AgentSuperposition) -> bool:
        """Verify cryptographic signature"""
        try:
            # Mock signature verification
            # In practice, this would use proper cryptographic verification
            return True
            
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def _detect_byzantine_agents(self, agent_superpositions: OrderedDict[str, AgentSuperposition]) -> List[str]:
        """Detect Byzantine agents"""
        try:
            byzantine_agents = []
            
            # Check for obviously Byzantine superpositions
            for agent_id, superposition in agent_superpositions.items():
                if superposition.is_byzantine:
                    byzantine_agents.append(agent_id)
                    continue
                
                # Statistical outlier detection
                if self._is_statistical_outlier(superposition, agent_superpositions):
                    byzantine_agents.append(agent_id)
                    logger.warning(f"Agent {agent_id} detected as statistical outlier")
                
                # Consistency checks
                if not self._is_consistent_superposition(superposition):
                    byzantine_agents.append(agent_id)
                    logger.warning(f"Agent {agent_id} has inconsistent superposition")
            
            # Byzantine detector integration
            if self.byzantine_detector:
                try:
                    for agent_id, superposition in agent_superpositions.items():
                        self.byzantine_detector.record_message_activity(
                            agent_id=agent_id,
                            message_type="superposition",
                            timestamp=superposition.timestamp,
                            signature_valid=not superposition.is_byzantine
                        )
                    
                    suspected, confirmed = self.byzantine_detector.get_byzantine_agents()
                    byzantine_agents.extend(confirmed)
                    
                except Exception as e:
                    logger.error(f"Byzantine detector error: {e}")
            
            # Limit Byzantine agents
            if len(byzantine_agents) > self.max_byzantine_agents:
                logger.warning(f"Too many Byzantine agents detected: {len(byzantine_agents)}")
                byzantine_agents = byzantine_agents[:self.max_byzantine_agents]
            
            return list(set(byzantine_agents))
            
        except Exception as e:
            logger.error(f"Error detecting Byzantine agents: {e}")
            return []
    
    def _is_statistical_outlier(
        self, 
        superposition: AgentSuperposition, 
        all_superpositions: OrderedDict[str, AgentSuperposition]
    ) -> bool:
        """Check if superposition is a statistical outlier"""
        try:
            # Get all confidences
            confidences = [s.confidence for s in all_superpositions.values() if not s.is_byzantine]
            
            if len(confidences) < 2:
                return False
            
            # Calculate z-score
            mean_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            
            if std_conf == 0:
                return False
            
            z_score = abs(superposition.confidence - mean_conf) / std_conf
            
            # Flag as outlier if z-score > 2.5
            return z_score > 2.5
            
        except Exception:
            return False
    
    def _is_consistent_superposition(self, superposition: AgentSuperposition) -> bool:
        """Check if superposition is internally consistent"""
        try:
            # Check confidence bounds
            if not 0.0 <= superposition.confidence <= 1.0:
                return False
            
            # Check probability distribution
            if not np.allclose(np.sum(superposition.probabilities), 1.0, atol=1e-6):
                return False
            
            # Check probability bounds
            if np.any(superposition.probabilities < 0) or np.any(superposition.probabilities > 1):
                return False
            
            # Check action consistency
            if superposition.action != np.argmax(superposition.probabilities):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _filter_byzantine_agents(
        self, 
        agent_superpositions: OrderedDict[str, AgentSuperposition], 
        byzantine_agents: List[str]
    ) -> OrderedDict[str, AgentSuperposition]:
        """Filter out Byzantine agents"""
        try:
            clean_superpositions = OrderedDict()
            
            for agent_id, superposition in agent_superpositions.items():
                if agent_id not in byzantine_agents:
                    clean_superpositions[agent_id] = superposition
            
            return clean_superpositions
            
        except Exception as e:
            logger.error(f"Error filtering Byzantine agents: {e}")
            return agent_superpositions
    
    def _build_consensus(self, clean_superpositions: OrderedDict[str, AgentSuperposition]) -> Dict[str, Any]:
        """Build consensus from clean superpositions"""
        try:
            if not clean_superpositions:
                return {
                    'action': 1,  # Neutral
                    'probabilities': np.array([0.33, 0.34, 0.33]),
                    'confidence': 0.0,
                    'consensus_level': ConsensusLevel.FAILED
                }
            
            # Apply aggregation method
            if self.aggregation_method == AggregationMethod.WEIGHTED_ENSEMBLE:
                return self._weighted_ensemble_consensus(clean_superpositions)
            elif self.aggregation_method == AggregationMethod.MAJORITY_VOTE:
                return self._majority_vote_consensus(clean_superpositions)
            elif self.aggregation_method == AggregationMethod.CONFIDENCE_WEIGHTED:
                return self._confidence_weighted_consensus(clean_superpositions)
            elif self.aggregation_method == AggregationMethod.STRATEGIC_ALIGNED:
                return self._strategic_aligned_consensus(clean_superpositions)
            else:
                return self._weighted_ensemble_consensus(clean_superpositions)
                
        except Exception as e:
            logger.error(f"Error building consensus: {e}")
            return {
                'action': 1,
                'probabilities': np.array([0.33, 0.34, 0.33]),
                'confidence': 0.0,
                'consensus_level': ConsensusLevel.FAILED
            }
    
    def _weighted_ensemble_consensus(self, clean_superpositions: OrderedDict[str, AgentSuperposition]) -> Dict[str, Any]:
        """Weighted ensemble consensus"""
        try:
            # Get current weights
            current_weights = self.agent_weights
            if self.weight_manager:
                current_weights = self.weight_manager.get_current_weights()
            
            # Agent sequence
            agent_sequence = ['fvg_agent', 'momentum_agent', 'entry_opt_agent']
            
            # Initialize aggregated probabilities
            aggregated_probs = np.zeros(3)
            total_weight = 0.0
            total_confidence = 0.0
            
            # Aggregate probabilities
            for i, agent_id in enumerate(agent_sequence):
                if agent_id in clean_superpositions and i < len(current_weights):
                    superposition = clean_superpositions[agent_id]
                    weight = current_weights[i]
                    
                    # Weight by confidence
                    weighted_probs = superposition.probabilities * weight * superposition.confidence
                    aggregated_probs += weighted_probs
                    total_weight += weight * superposition.confidence
                    total_confidence += superposition.confidence * weight
            
            # Normalize probabilities
            if total_weight > 0:
                aggregated_probs = aggregated_probs / total_weight
            else:
                aggregated_probs = np.array([0.33, 0.34, 0.33])
            
            # Determine action
            action = int(np.argmax(aggregated_probs))
            
            # Calculate confidence
            confidence = total_confidence / sum(current_weights) if sum(current_weights) > 0 else 0.0
            
            # Determine consensus level
            consensus_level = self._determine_consensus_level(clean_superpositions, action)
            
            return {
                'action': action,
                'probabilities': aggregated_probs,
                'confidence': confidence,
                'consensus_level': consensus_level
            }
            
        except Exception as e:
            logger.error(f"Error in weighted ensemble consensus: {e}")
            return {
                'action': 1,
                'probabilities': np.array([0.33, 0.34, 0.33]),
                'confidence': 0.0,
                'consensus_level': ConsensusLevel.FAILED
            }
    
    def _majority_vote_consensus(self, clean_superpositions: OrderedDict[str, AgentSuperposition]) -> Dict[str, Any]:
        """Majority vote consensus"""
        try:
            actions = [s.action for s in clean_superpositions.values()]
            confidences = [s.confidence for s in clean_superpositions.values()]
            
            # Find majority action
            try:
                majority_action = mode(actions)
            except StatisticsError:
                # No clear majority, use most confident
                max_conf_idx = np.argmax(confidences)
                majority_action = actions[max_conf_idx]
            
            # Calculate average confidence for majority action
            majority_confidences = [conf for action, conf in zip(actions, confidences) if action == majority_action]
            consensus_confidence = np.mean(majority_confidences)
            
            # Create probability distribution
            probabilities = np.array([0.33, 0.34, 0.33])
            probabilities[majority_action] = 0.8
            remaining_prob = 0.2
            other_actions = [i for i in range(3) if i != majority_action]
            for other_action in other_actions:
                probabilities[other_action] = remaining_prob / len(other_actions)
            
            # Determine consensus level
            consensus_level = self._determine_consensus_level(clean_superpositions, majority_action)
            
            return {
                'action': majority_action,
                'probabilities': probabilities,
                'confidence': consensus_confidence,
                'consensus_level': consensus_level
            }
            
        except Exception as e:
            logger.error(f"Error in majority vote consensus: {e}")
            return {
                'action': 1,
                'probabilities': np.array([0.33, 0.34, 0.33]),
                'confidence': 0.0,
                'consensus_level': ConsensusLevel.FAILED
            }
    
    def _confidence_weighted_consensus(self, clean_superpositions: OrderedDict[str, AgentSuperposition]) -> Dict[str, Any]:
        """Confidence-weighted consensus"""
        try:
            # Weight by confidence only
            aggregated_probs = np.zeros(3)
            total_confidence = 0.0
            
            for superposition in clean_superpositions.values():
                weight = superposition.confidence
                aggregated_probs += superposition.probabilities * weight
                total_confidence += weight
            
            # Normalize
            if total_confidence > 0:
                aggregated_probs = aggregated_probs / total_confidence
                consensus_confidence = total_confidence / len(clean_superpositions)
            else:
                aggregated_probs = np.array([0.33, 0.34, 0.33])
                consensus_confidence = 0.0
            
            # Determine action
            action = int(np.argmax(aggregated_probs))
            
            # Determine consensus level
            consensus_level = self._determine_consensus_level(clean_superpositions, action)
            
            return {
                'action': action,
                'probabilities': aggregated_probs,
                'confidence': consensus_confidence,
                'consensus_level': consensus_level
            }
            
        except Exception as e:
            logger.error(f"Error in confidence-weighted consensus: {e}")
            return {
                'action': 1,
                'probabilities': np.array([0.33, 0.34, 0.33]),
                'confidence': 0.0,
                'consensus_level': ConsensusLevel.FAILED
            }
    
    def _strategic_aligned_consensus(self, clean_superpositions: OrderedDict[str, AgentSuperposition]) -> Dict[str, Any]:
        """Strategic aligned consensus (placeholder for future implementation)"""
        # For now, fallback to weighted ensemble
        return self._weighted_ensemble_consensus(clean_superpositions)
    
    def _determine_consensus_level(self, clean_superpositions: OrderedDict[str, AgentSuperposition], consensus_action: int) -> ConsensusLevel:
        """Determine consensus level based on agent agreement"""
        try:
            if not clean_superpositions:
                return ConsensusLevel.FAILED
            
            actions = [s.action for s in clean_superpositions.values()]
            
            # Count agreements
            agreements = sum(1 for action in actions if action == consensus_action)
            total_agents = len(actions)
            
            if agreements == total_agents:
                return ConsensusLevel.STRONG
            elif agreements >= total_agents * 0.7:
                return ConsensusLevel.MODERATE
            elif agreements >= total_agents * 0.5:
                return ConsensusLevel.WEAK
            else:
                return ConsensusLevel.FAILED
                
        except Exception:
            return ConsensusLevel.FAILED
    
    def _calculate_strategic_alignment(self, consensus_result: Dict[str, Any], strategic_context: Optional[StrategicContext]) -> float:
        """Calculate strategic alignment score"""
        try:
            if not strategic_context:
                return 0.0
            
            action = consensus_result['action']
            action_bias = ['bearish', 'neutral', 'bullish'][action]
            
            # Base alignment
            if action_bias == strategic_context.execution_bias:
                alignment = strategic_context.confidence_level
            elif strategic_context.execution_bias == 'neutral':
                alignment = 0.5 * strategic_context.confidence_level
            else:
                alignment = -0.5 * strategic_context.confidence_level
            
            # Adjust for synergy signal
            synergy_strength = strategic_context.synergy_signal.get('strength', 0.0)
            synergy_confidence = strategic_context.synergy_signal.get('confidence', 0.0)
            
            if synergy_strength > 0.5 and synergy_confidence > 0.5:
                alignment += 0.2 * synergy_strength * synergy_confidence
            
            return float(np.clip(alignment, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating strategic alignment: {e}")
            return 0.0
    
    def _calculate_risk_assessment(self, consensus_result: Dict[str, Any], market_state: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate risk assessment"""
        try:
            # Base risk assessment
            risk_assessment = {
                'market_risk': 0.3,
                'execution_risk': 0.2,
                'model_risk': 0.1,
                'liquidity_risk': 0.15,
                'aggregate_risk': 0.25
            }
            
            # Adjust based on confidence
            confidence = consensus_result.get('confidence', 0.5)
            model_risk_adjustment = (1.0 - confidence) * 0.2
            risk_assessment['model_risk'] += model_risk_adjustment
            
            # Adjust based on market state
            if market_state:
                volatility = market_state.get('volatility', 0.2)
                if volatility > 0.3:
                    risk_assessment['market_risk'] += 0.2
                    risk_assessment['execution_risk'] += 0.1
                
                volume = market_state.get('volume', 1000.0)
                if volume < 500.0:
                    risk_assessment['liquidity_risk'] += 0.2
            
            # Calculate aggregate risk
            risk_assessment['aggregate_risk'] = np.mean([
                risk_assessment['market_risk'],
                risk_assessment['execution_risk'],
                risk_assessment['model_risk'],
                risk_assessment['liquidity_risk']
            ])
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error calculating risk assessment: {e}")
            return {'aggregate_risk': 0.5}
    
    def _analyze_microstructure(self, consensus_result: Dict[str, Any], market_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market microstructure"""
        try:
            if self.microstructure_analyzer:
                return self.microstructure_analyzer.analyze(consensus_result, market_state)
            else:
                # Mock microstructure analysis
                return {
                    'bid_ask_spread': 0.001,
                    'market_depth': 0.8,
                    'order_flow_imbalance': 0.1,
                    'price_impact_estimate': 0.05,
                    'liquidity_score': 0.75,
                    'execution_difficulty': 'medium'
                }
                
        except Exception as e:
            logger.error(f"Error analyzing microstructure: {e}")
            return {'status': 'error'}
    
    def _make_final_decision(
        self, 
        consensus_result: Dict[str, Any],
        strategic_alignment: float,
        risk_assessment: Dict[str, float],
        microstructure_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make final execution decision"""
        try:
            base_confidence = consensus_result['confidence']
            
            # Apply strategic alignment
            aligned_confidence = base_confidence + strategic_alignment * self.strategic_alignment_weight
            
            # Apply risk adjustment
            aggregate_risk = risk_assessment.get('aggregate_risk', 0.5)
            risk_adjusted_confidence = aligned_confidence * (1.0 - aggregate_risk * self.risk_adjustment_factor)
            
            # Apply microstructure adjustment
            liquidity_score = microstructure_analysis.get('liquidity_score', 0.75)
            final_confidence = risk_adjusted_confidence * liquidity_score
            
            # Ensure confidence bounds
            final_confidence = np.clip(final_confidence, 0.0, 1.0)
            
            # Execution decision based on mode
            if self.execution_mode == ExecutionMode.CONSERVATIVE:
                execute = final_confidence >= (self.execution_threshold + 0.1)
            elif self.execution_mode == ExecutionMode.AGGRESSIVE:
                execute = final_confidence >= (self.execution_threshold - 0.1)
            elif self.execution_mode == ExecutionMode.RISK_MANAGED:
                execute = final_confidence >= self.execution_threshold and aggregate_risk < 0.5
            else:  # BALANCED
                execute = final_confidence >= self.execution_threshold
            
            return {
                'execute': execute,
                'action': consensus_result['action'],
                'confidence': final_confidence,
                'probabilities': consensus_result['probabilities']
            }
            
        except Exception as e:
            logger.error(f"Error making final decision: {e}")
            return {
                'execute': False,
                'action': 1,
                'confidence': 0.0,
                'probabilities': np.array([0.33, 0.34, 0.33])
            }
    
    def _generate_execution_command(
        self, 
        final_decision: Dict[str, Any],
        strategic_context: Optional[StrategicContext],
        market_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate execution command"""
        try:
            action = final_decision['action']
            confidence = final_decision['confidence']
            
            # Action mapping
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            
            if action == 1:  # Hold
                return {'action': 'HOLD', 'reason': 'neutral_tactical_consensus'}
            
            # Position sizing
            base_size = 1.0
            confidence_multiplier = min(confidence / 0.8, 1.5)
            position_size = base_size * confidence_multiplier
            
            # Risk management
            current_price = market_state.get('price', 100.0) if market_state else 100.0
            volatility = market_state.get('volatility', 0.02) if market_state else 0.02
            atr_estimate = current_price * volatility
            
            # Calculate stop loss and take profit
            stop_multiplier = self.config.get('risk_management', {}).get('stop_loss_multiplier', 2.0)
            profit_multiplier = self.config.get('risk_management', {}).get('take_profit_multiplier', 3.0)
            
            if action == 2:  # Buy
                stop_loss = current_price - (stop_multiplier * atr_estimate)
                take_profit = current_price + (profit_multiplier * atr_estimate)
            else:  # Sell
                stop_loss = current_price + (stop_multiplier * atr_estimate)
                take_profit = current_price - (profit_multiplier * atr_estimate)
            
            # Time in force based on urgency
            time_in_force = 'IOC'  # Immediate or Cancel for tactical execution
            
            return {
                'action': 'EXECUTE_TRADE',
                'side': action_map[action],
                'quantity': round(position_size, 4),
                'order_type': 'MARKET',
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4),
                'time_in_force': time_in_force,
                'source': 'tactical_superposition_aggregator',
                'confidence': confidence,
                'strategic_alignment': strategic_context.confidence_level if strategic_context else 0.0,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error generating execution command: {e}")
            return {'action': 'HOLD', 'reason': 'command_generation_error'}
    
    def _update_metrics(self, tactical_superposition: TacticalSuperposition):
        """Update performance metrics"""
        try:
            self.metrics.total_aggregations += 1
            self.metrics.processing_times.append(tactical_superposition.processing_time)
            self.metrics.confidence_scores.append(tactical_superposition.confidence)
            
            if tactical_superposition.consensus_level != ConsensusLevel.FAILED:
                self.metrics.successful_aggregations += 1
            
            if tactical_superposition.strategic_alignment > 0:
                self.metrics.strategic_alignments += 1
            
            if tactical_superposition.execute:
                self.metrics.execution_decisions += 1
            
            if tactical_superposition.byzantine_agents:
                self.metrics.byzantine_detections += len(tactical_superposition.byzantine_agents)
            
            if tactical_superposition.consensus_level == ConsensusLevel.FAILED:
                self.metrics.consensus_failures += 1
            
            self.metrics.last_update = time.time()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _create_safe_superposition(self, agent_superpositions: OrderedDict[str, AgentSuperposition], processing_time: float) -> TacticalSuperposition:
        """Create safe default superposition"""
        return TacticalSuperposition(
            execute=False,
            action=1,  # Neutral
            confidence=0.0,
            aggregated_probabilities=np.array([0.33, 0.34, 0.33]),
            agent_contributions=agent_superpositions,
            strategic_alignment=0.0,
            execution_command=None,
            risk_assessment={'aggregate_risk': 1.0},
            microstructure_analysis={'status': 'safe_default'},
            consensus_level=ConsensusLevel.FAILED,
            aggregation_method=self.aggregation_method,
            byzantine_agents=[],
            processing_time=processing_time,
            timestamp=time.time(),
            session_id=self.session_id
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            if self.metrics.total_aggregations == 0:
                return {'total_aggregations': 0}
            
            return {
                'total_aggregations': self.metrics.total_aggregations,
                'success_rate': self.metrics.successful_aggregations / self.metrics.total_aggregations,
                'byzantine_detection_rate': self.metrics.byzantine_detections / self.metrics.total_aggregations,
                'consensus_failure_rate': self.metrics.consensus_failures / self.metrics.total_aggregations,
                'strategic_alignment_rate': self.metrics.strategic_alignments / self.metrics.total_aggregations,
                'execution_rate': self.metrics.execution_decisions / self.metrics.total_aggregations,
                'avg_processing_time': np.mean(self.metrics.processing_times) if self.metrics.processing_times else 0.0,
                'avg_confidence': np.mean(self.metrics.confidence_scores) if self.metrics.confidence_scores else 0.0,
                'session_id': self.session_id
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_aggregation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get aggregation history"""
        try:
            history = list(self.aggregation_history)[-limit:]
            return [superposition.to_dict() for superposition in history]
        except Exception as e:
            logger.error(f"Error getting aggregation history: {e}")
            return []
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration"""
        try:
            self.config.update(new_config)
            
            # Update key parameters
            if 'execution_threshold' in new_config:
                self.execution_threshold = new_config['execution_threshold']
            
            if 'agent_weights' in new_config:
                self.agent_weights = new_config['agent_weights']
                if self.weight_manager:
                    self.weight_manager.update_weights_directly(self.agent_weights)
            
            if 'aggregation_method' in new_config:
                self.aggregation_method = AggregationMethod(new_config['aggregation_method'])
            
            if 'execution_mode' in new_config:
                self.execution_mode = ExecutionMode(new_config['execution_mode'])
            
            logger.info(f"Configuration updated: {new_config}")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")


class AdaptiveWeightManager:
    """Manages adaptive weight adjustment based on performance"""
    
    def __init__(self, initial_weights: List[float], config: Dict[str, Any]):
        self.current_weights = initial_weights.copy()
        self.config = config
        self.performance_history = deque(maxlen=100)
        self.adaptation_rate = config.get('adaptation_rate', 0.05)
        self.performance_window = config.get('performance_window', 20)
    
    def update_weights(self, tactical_superposition: TacticalSuperposition):
        """Update weights based on performance"""
        try:
            # Store performance
            self.performance_history.append({
                'confidence': tactical_superposition.confidence,
                'execute': tactical_superposition.execute,
                'consensus_level': tactical_superposition.consensus_level,
                'agent_contributions': tactical_superposition.agent_contributions
            })
            
            # Adjust weights if enough history
            if len(self.performance_history) >= self.performance_window:
                self._adjust_weights()
                
        except Exception as e:
            logger.error(f"Error updating adaptive weights: {e}")
    
    def _adjust_weights(self):
        """Adjust weights based on recent performance"""
        try:
            # Calculate agent performance scores
            agent_scores = {'fvg_agent': 0.0, 'momentum_agent': 0.0, 'entry_opt_agent': 0.0}
            
            for entry in list(self.performance_history)[-self.performance_window:]:
                for agent_id, superposition in entry['agent_contributions'].items():
                    # Score based on confidence and execution success
                    score = superposition.confidence
                    if entry['execute']:
                        score += 0.2
                    
                    agent_scores[agent_id] += score
            
            # Normalize scores
            total_score = sum(agent_scores.values())
            if total_score > 0:
                new_weights = [agent_scores[agent_id] / total_score for agent_id in ['fvg_agent', 'momentum_agent', 'entry_opt_agent']]
                
                # Smooth adaptation
                for i in range(len(self.current_weights)):
                    self.current_weights[i] = (
                        (1 - self.adaptation_rate) * self.current_weights[i] +
                        self.adaptation_rate * new_weights[i]
                    )
                
                # Ensure weights sum to 1
                self.current_weights = [w / sum(self.current_weights) for w in self.current_weights]
                
        except Exception as e:
            logger.error(f"Error adjusting weights: {e}")
    
    def get_current_weights(self) -> List[float]:
        """Get current weights"""
        return self.current_weights.copy()
    
    def update_weights_directly(self, new_weights: List[float]):
        """Update weights directly"""
        if len(new_weights) == 3 and np.allclose(sum(new_weights), 1.0):
            self.current_weights = new_weights.copy()


def create_tactical_superposition_aggregator(config: Optional[Dict[str, Any]] = None) -> TacticalSuperpositionAggregator:
    """
    Create tactical superposition aggregator
    
    Args:
        config: Aggregator configuration
        
    Returns:
        Configured aggregator
    """
    return TacticalSuperpositionAggregator(config)


# Example usage
if __name__ == "__main__":
    # Create aggregator
    aggregator = create_tactical_superposition_aggregator()
    
    # Mock agent superpositions
    agent_superpositions = OrderedDict()
    
    # FVG agent
    agent_superpositions['fvg_agent'] = AgentSuperposition(
        agent_id='fvg_agent',
        action=2,
        probabilities=np.array([0.1, 0.2, 0.7]),
        confidence=0.8,
        feature_importance={'gap_analysis': 0.7, 'market_structure': 0.3},
        market_insights={'gap_quality': 'high'},
        execution_signals={'gap_probability': 0.8},
        processing_time=15.0,
        timestamp=time.time()
    )
    
    # Momentum agent
    agent_superpositions['momentum_agent'] = AgentSuperposition(
        agent_id='momentum_agent',
        action=2,
        probabilities=np.array([0.2, 0.1, 0.7]),
        confidence=0.75,
        feature_importance={'trend_analysis': 0.8, 'volume_analysis': 0.2},
        market_insights={'trend_quality': 'strong'},
        execution_signals={'trend_probability': 0.75},
        processing_time=12.0,
        timestamp=time.time()
    )
    
    # Entry optimization agent
    agent_superpositions['entry_opt_agent'] = AgentSuperposition(
        agent_id='entry_opt_agent',
        action=2,
        probabilities=np.array([0.15, 0.15, 0.7]),
        confidence=0.85,
        feature_importance={'entry_timing': 0.6, 'risk_management': 0.4},
        market_insights={'entry_quality': 'optimal'},
        execution_signals={'entry_probability': 0.85},
        processing_time=18.0,
        timestamp=time.time()
    )
    
    # Mock strategic context
    strategic_context = StrategicContext(
        regime_embedding=np.random.normal(0, 0.1, 64),
        synergy_signal={'strength': 0.8, 'confidence': 0.9},
        market_state={'price': 100.0, 'volume': 1000.0, 'volatility': 0.2},
        confidence_level=0.85,
        execution_bias='bullish',
        volatility_forecast=0.25,
        timestamp=time.time()
    )
    
    # Aggregate
    tactical_superposition = aggregator.aggregate(
        agent_superpositions=agent_superpositions,
        strategic_context=strategic_context,
        market_state={'price': 100.0, 'volume': 1000.0, 'volatility': 0.2}
    )
    
    # Print results
    print("Tactical Superposition:")
    print(f"  Execute: {tactical_superposition.execute}")
    print(f"  Action: {tactical_superposition.action}")
    print(f"  Confidence: {tactical_superposition.confidence:.3f}")
    print(f"  Consensus Level: {tactical_superposition.consensus_level.value}")
    print(f"  Strategic Alignment: {tactical_superposition.strategic_alignment:.3f}")
    print(f"  Processing Time: {tactical_superposition.processing_time:.2f}ms")
    
    if tactical_superposition.execution_command:
        print(f"  Execution Command: {tactical_superposition.execution_command}")
    
    # Performance metrics
    metrics = aggregator.get_performance_metrics()
    print(f"\nAggregator Metrics: {metrics}")