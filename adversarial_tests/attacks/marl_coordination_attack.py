#!/usr/bin/env python3
"""
🚨 AGENT GAMMA MISSION - MULTI-AGENT COORDINATION ATTACK MODULE
Advanced MARL Attack Development: Multi-Agent Coordination Disruption

This module implements sophisticated attacks targeting the coordination mechanisms
between Strategic and Tactical MARL agents, focusing on disrupting:
- Agent consensus mechanisms
- Strategic-tactical coordination bridges
- Gating network vulnerabilities
- Inter-agent communication patterns

Key Attack Vectors:
1. Consensus Disruption: Attack agent consensus mechanisms
2. Communication Jamming: Disrupt inter-agent communication
3. Gating Network Exploitation: Exploit gating network vulnerabilities
4. Coordination Timing Attacks: Exploit coordination timing windows
5. Strategic-Tactical Desynchronization: Disrupt multi-timeframe coordination

MISSION OBJECTIVE: Achieve >80% attack success rate against current MARL coordination defenses
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from enum import Enum
import asyncio
import time

# Attack Result Tracking
@dataclass
class CoordinationAttackResult:
    """Results from a coordination attack attempt."""
    attack_type: str
    success: bool
    confidence: float
    disrupted_agents: List[str]
    gating_weights_before: np.ndarray
    gating_weights_after: np.ndarray
    coordination_disruption_score: float
    execution_time_ms: float
    attack_payload: Dict[str, Any]
    timestamp: datetime

class AttackType(Enum):
    """Types of coordination attacks."""
    CONSENSUS_DISRUPTION = "consensus_disruption"
    COMMUNICATION_JAMMING = "communication_jamming"
    GATING_EXPLOITATION = "gating_exploitation"
    TIMING_ATTACK = "timing_attack"
    DESYNCHRONIZATION = "desynchronization"

class MARLCoordinationAttacker:
    """
    Advanced Multi-Agent Coordination Attack System.
    
    This system implements sophisticated attacks targeting the coordination
    mechanisms between Strategic and Tactical MARL agents.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the MARL Coordination Attacker.
        
        Args:
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.device = device
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Attack history and analytics
        self.attack_history = []
        self.success_rates = {attack_type: 0.0 for attack_type in AttackType}
        self.disruption_metrics = {
            'total_attempts': 0,
            'successful_attacks': 0,
            'avg_disruption_score': 0.0,
            'max_disruption_score': 0.0,
            'agents_compromised': set()
        }
        
        # Attack parameters
        self.attack_strength = 0.5  # Base attack strength
        self.coordination_threshold = 0.3  # Threshold for coordination disruption
        self.timing_window_ms = 100  # Timing attack window
        
        # Gating network attack parameters
        self.gating_noise_scale = 0.1
        self.consensus_disruption_factor = 0.8
        
        self.logger.info(f"MARLCoordinationAttacker initialized: device={device}")
    
    def generate_consensus_disruption_attack(
        self, 
        agent_predictions: List[Dict[str, Any]],
        shared_context: Dict[str, Any],
        target_agents: List[str] = None
    ) -> Tuple[Dict[str, Any], CoordinationAttackResult]:
        """
        🎯 ATTACK 1: CONSENSUS DISRUPTION ATTACK
        
        Targets agent consensus mechanisms by creating conflicting predictions
        that disrupt the ensemble decision-making process.
        
        Args:
            agent_predictions: List of agent predictions to attack
            shared_context: Shared market context
            target_agents: Specific agents to target (or None for all)
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if target_agents is None:
            target_agents = ['MLMI', 'NWRQK', 'Regime']
        
        # Store original predictions for comparison
        original_predictions = [pred.copy() for pred in agent_predictions]
        
        # Generate attack payload
        attack_payload = {
            'attack_type': AttackType.CONSENSUS_DISRUPTION.value,
            'target_agents': target_agents,
            'disruption_patterns': {},
            'confidence_manipulation': {},
            'probability_injection': {}
        }
        
        disrupted_agents = []
        
        # For each target agent, create conflicting predictions
        for i, agent_pred in enumerate(agent_predictions):
            agent_name = agent_pred.get('agent_name', f'Agent_{i}')
            
            if agent_name not in target_agents:
                continue
            
            # Original probabilities
            original_probs = np.array(agent_pred.get('action_probabilities', [0.33, 0.34, 0.33]))
            original_confidence = agent_pred.get('confidence', 0.5)
            
            # Create consensus disruption patterns
            if agent_name == 'MLMI':
                # Make MLMI extremely bullish when others are bearish
                disrupted_probs = np.array([0.95, 0.03, 0.02])  # Extreme buy signal
                disrupted_confidence = 0.98
                
            elif agent_name == 'NWRQK':
                # Make NWRQK extremely bearish when others are bullish
                disrupted_probs = np.array([0.02, 0.03, 0.95])  # Extreme sell signal
                disrupted_confidence = 0.97
                
            elif agent_name == 'Regime':
                # Make Regime suggest extreme uncertainty
                disrupted_probs = np.array([0.05, 0.90, 0.05])  # Extreme hold signal
                disrupted_confidence = 0.15  # Very low confidence
            
            else:
                # Generic consensus disruption
                disrupted_probs = 1.0 - original_probs  # Invert probabilities
                disrupted_probs = disrupted_probs / np.sum(disrupted_probs)  # Normalize
                disrupted_confidence = 1.0 - original_confidence
            
            # Apply disruption
            agent_pred['action_probabilities'] = disrupted_probs.tolist()
            agent_pred['confidence'] = disrupted_confidence
            agent_pred['consensus_disrupted'] = True
            
            # Store attack details
            attack_payload['disruption_patterns'][agent_name] = {
                'original_probs': original_probs.tolist(),
                'disrupted_probs': disrupted_probs.tolist(),
                'original_confidence': original_confidence,
                'disrupted_confidence': disrupted_confidence
            }
            
            disrupted_agents.append(agent_name)
        
        # Calculate disruption score
        disruption_score = self._calculate_consensus_disruption_score(
            original_predictions, agent_predictions
        )
        
        # Determine attack success
        success = disruption_score > self.coordination_threshold
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = CoordinationAttackResult(
            attack_type=AttackType.CONSENSUS_DISRUPTION.value,
            success=success,
            confidence=disruption_score,
            disrupted_agents=disrupted_agents,
            gating_weights_before=np.array([0.33, 0.33, 0.34]),  # Placeholder
            gating_weights_after=np.array([0.5, 0.3, 0.2]),   # Placeholder
            coordination_disruption_score=disruption_score,
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Consensus disruption attack executed: "
            f"success={success}, disruption_score={disruption_score:.3f}, "
            f"agents_disrupted={len(disrupted_agents)}"
        )
        
        return attack_payload, attack_result
    
    def generate_communication_jamming_attack(
        self,
        shared_context: Dict[str, Any],
        communication_channels: List[str] = None
    ) -> Tuple[Dict[str, Any], CoordinationAttackResult]:
        """
        🎯 ATTACK 2: COMMUNICATION JAMMING ATTACK
        
        Disrupts inter-agent communication by injecting noise into shared context
        and communication channels.
        
        Args:
            shared_context: Shared market context to attack
            communication_channels: Specific channels to jam
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if communication_channels is None:
            communication_channels = ['volatility_30', 'volume_ratio', 'momentum_20', 'momentum_50']
        
        # Store original context
        original_context = shared_context.copy()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': AttackType.COMMUNICATION_JAMMING.value,
            'target_channels': communication_channels,
            'noise_injections': {},
            'context_corruption': {},
            'jamming_patterns': {}
        }
        
        disrupted_channels = []
        
        # Inject noise into communication channels
        for channel in communication_channels:
            if channel in shared_context:
                original_value = shared_context[channel]
                
                # Different jamming patterns for different channels
                if channel == 'volatility_30':
                    # Inject extreme volatility spikes
                    noise = np.random.uniform(0.5, 2.0)  # Multiply by 1.5x to 3x
                    jammed_value = original_value * noise
                    
                elif channel == 'volume_ratio':
                    # Create fake volume spikes
                    noise = np.random.uniform(5.0, 20.0)  # Extreme volume multiplication
                    jammed_value = original_value * noise
                    
                elif channel in ['momentum_20', 'momentum_50']:
                    # Inject momentum reversals
                    noise = np.random.uniform(-2.0, 2.0)  # Strong momentum reversal
                    jammed_value = original_value + noise
                    
                else:
                    # Generic noise injection
                    noise = np.random.normal(0, abs(original_value) * 0.5)
                    jammed_value = original_value + noise
                
                # Apply jamming
                shared_context[channel] = jammed_value
                
                # Store attack details
                attack_payload['noise_injections'][channel] = {
                    'original_value': original_value,
                    'noise_applied': noise,
                    'jammed_value': jammed_value,
                    'corruption_ratio': abs(jammed_value - original_value) / (abs(original_value) + 1e-8)
                }
                
                disrupted_channels.append(channel)
        
        # Add false market regime signals
        shared_context['market_regime'] = 'attack_injected'
        shared_context['regime_confidence'] = 0.95  # High confidence in false regime
        
        # Calculate disruption score
        disruption_score = self._calculate_communication_disruption_score(
            original_context, shared_context
        )
        
        # Determine attack success
        success = disruption_score > self.coordination_threshold
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = CoordinationAttackResult(
            attack_type=AttackType.COMMUNICATION_JAMMING.value,
            success=success,
            confidence=disruption_score,
            disrupted_agents=disrupted_channels,  # Channels instead of agents
            gating_weights_before=np.array([0.33, 0.33, 0.34]),  # Placeholder
            gating_weights_after=np.array([0.2, 0.4, 0.4]),   # Placeholder
            coordination_disruption_score=disruption_score,
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Communication jamming attack executed: "
            f"success={success}, disruption_score={disruption_score:.3f}, "
            f"channels_jammed={len(disrupted_channels)}"
        )
        
        return attack_payload, attack_result
    
    def generate_gating_exploitation_attack(
        self,
        shared_context: Dict[str, Any],
        target_agent_weights: np.ndarray = None
    ) -> Tuple[Dict[str, Any], CoordinationAttackResult]:
        """
        🎯 ATTACK 3: GATING NETWORK EXPLOITATION ATTACK
        
        Exploits gating network vulnerabilities to manipulate agent weights
        and force suboptimal coordination decisions.
        
        Args:
            shared_context: Shared market context
            target_agent_weights: Desired agent weights after attack
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if target_agent_weights is None:
            # Force extreme weight concentration on weakest agent
            target_agent_weights = np.array([0.05, 0.05, 0.90])  # Force Regime dominance
        
        # Store original context
        original_context = shared_context.copy()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': AttackType.GATING_EXPLOITATION.value,
            'target_weights': target_agent_weights.tolist(),
            'context_manipulation': {},
            'bias_injections': {},
            'weight_forcing_patterns': {}
        }
        
        # Manipulate context features to force desired gating weights
        # Based on gating network bias patterns from gating_network.py
        
        # Force high volatility to bias toward Regime agent
        if 'volatility_30' in shared_context:
            original_vol = shared_context['volatility_30']
            # Extreme volatility to trigger regime bias
            shared_context['volatility_30'] = 0.15  # Very high volatility
            attack_payload['context_manipulation']['volatility_30'] = {
                'original': original_vol,
                'manipulated': 0.15,
                'bias_target': 'Regime'
            }
        
        # Suppress momentum signals to reduce MLMI influence
        for momentum_key in ['momentum_20', 'momentum_50']:
            if momentum_key in shared_context:
                original_momentum = shared_context[momentum_key]
                shared_context[momentum_key] = 0.001  # Minimal momentum
                attack_payload['context_manipulation'][momentum_key] = {
                    'original': original_momentum,
                    'manipulated': 0.001,
                    'bias_target': 'suppress_MLMI'
                }
        
        # Reduce volume signals to suppress NWRQK
        if 'volume_ratio' in shared_context:
            original_volume = shared_context['volume_ratio']
            shared_context['volume_ratio'] = 0.5  # Low volume
            attack_payload['context_manipulation']['volume_ratio'] = {
                'original': original_volume,
                'manipulated': 0.5,
                'bias_target': 'suppress_NWRQK'
            }
        
        # Amplify MMD score to strengthen regime detection
        if 'mmd_score' in shared_context:
            original_mmd = shared_context['mmd_score']
            shared_context['mmd_score'] = 0.8  # High MMD score
            attack_payload['context_manipulation']['mmd_score'] = {
                'original': original_mmd,
                'manipulated': 0.8,
                'bias_target': 'strengthen_Regime'
            }
        
        # Calculate disruption score
        disruption_score = self._calculate_gating_disruption_score(
            original_context, shared_context, target_agent_weights
        )
        
        # Determine attack success
        success = disruption_score > self.coordination_threshold
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = CoordinationAttackResult(
            attack_type=AttackType.GATING_EXPLOITATION.value,
            success=success,
            confidence=disruption_score,
            disrupted_agents=['Gating_Network'],
            gating_weights_before=np.array([0.33, 0.33, 0.34]),  # Placeholder
            gating_weights_after=target_agent_weights,
            coordination_disruption_score=disruption_score,
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Gating exploitation attack executed: "
            f"success={success}, disruption_score={disruption_score:.3f}, "
            f"target_weights={target_agent_weights}"
        )
        
        return attack_payload, attack_result
    
    def generate_timing_attack(
        self,
        coordination_delay_ms: float = 50.0,
        desync_pattern: str = 'staggered'
    ) -> Tuple[Dict[str, Any], CoordinationAttackResult]:
        """
        🎯 ATTACK 4: COORDINATION TIMING ATTACK
        
        Exploits coordination timing windows to desynchronize agent interactions
        and disrupt ensemble decision-making.
        
        Args:
            coordination_delay_ms: Delay to inject into coordination
            desync_pattern: Type of desynchronization pattern
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': AttackType.TIMING_ATTACK.value,
            'coordination_delay_ms': coordination_delay_ms,
            'desync_pattern': desync_pattern,
            'timing_manipulations': {},
            'delay_injections': {}
        }
        
        # Different timing attack patterns
        if desync_pattern == 'staggered':
            # Stagger agent execution times
            agent_delays = {
                'MLMI': 0.0,  # No delay
                'NWRQK': coordination_delay_ms,  # Medium delay
                'Regime': coordination_delay_ms * 2  # Maximum delay
            }
            
        elif desync_pattern == 'random':
            # Random delays for each agent
            agent_delays = {
                'MLMI': np.random.uniform(0, coordination_delay_ms),
                'NWRQK': np.random.uniform(0, coordination_delay_ms),
                'Regime': np.random.uniform(0, coordination_delay_ms)
            }
            
        elif desync_pattern == 'burst':
            # All agents delayed simultaneously
            agent_delays = {
                'MLMI': coordination_delay_ms,
                'NWRQK': coordination_delay_ms,
                'Regime': coordination_delay_ms
            }
            
        else:
            # Default staggered pattern
            agent_delays = {
                'MLMI': 0.0,
                'NWRQK': coordination_delay_ms * 0.5,
                'Regime': coordination_delay_ms
            }
        
        # Record timing manipulations
        attack_payload['timing_manipulations'] = agent_delays
        
        # Simulate the timing attack (in practice, this would delay actual execution)
        disrupted_agents = []
        for agent_name, delay in agent_delays.items():
            if delay > 0:
                # In practice, this would inject actual delays
                attack_payload['delay_injections'][agent_name] = {
                    'delay_ms': delay,
                    'desync_severity': delay / coordination_delay_ms
                }
                disrupted_agents.append(agent_name)
        
        # Calculate disruption score based on timing desynchronization
        max_delay = max(agent_delays.values())
        min_delay = min(agent_delays.values())
        timing_spread = max_delay - min_delay
        
        disruption_score = min(timing_spread / self.timing_window_ms, 1.0)
        
        # Determine attack success
        success = disruption_score > self.coordination_threshold
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = CoordinationAttackResult(
            attack_type=AttackType.TIMING_ATTACK.value,
            success=success,
            confidence=disruption_score,
            disrupted_agents=disrupted_agents,
            gating_weights_before=np.array([0.33, 0.33, 0.34]),  # Placeholder
            gating_weights_after=np.array([0.33, 0.33, 0.34]),  # No change for timing
            coordination_disruption_score=disruption_score,
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Timing attack executed: "
            f"success={success}, disruption_score={disruption_score:.3f}, "
            f"timing_spread={timing_spread:.1f}ms"
        )
        
        return attack_payload, attack_result
    
    def generate_desynchronization_attack(
        self,
        strategic_context: Dict[str, Any],
        tactical_context: Dict[str, Any],
        desync_factor: float = 0.8
    ) -> Tuple[Dict[str, Any], CoordinationAttackResult]:
        """
        🎯 ATTACK 5: STRATEGIC-TACTICAL DESYNCHRONIZATION ATTACK
        
        Disrupts coordination between strategic (30min) and tactical (5min) 
        MARL systems by creating inconsistent market views.
        
        Args:
            strategic_context: Strategic agent context
            tactical_context: Tactical agent context
            desync_factor: Strength of desynchronization
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        # Store original contexts
        original_strategic = strategic_context.copy()
        original_tactical = tactical_context.copy()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': AttackType.DESYNCHRONIZATION.value,
            'desync_factor': desync_factor,
            'strategic_manipulations': {},
            'tactical_manipulations': {},
            'context_divergence': {}
        }
        
        # Create opposing market views between strategic and tactical
        # Strategic: Make it appear bullish
        if 'momentum_20' in strategic_context:
            original_momentum = strategic_context['momentum_20']
            strategic_context['momentum_20'] = abs(original_momentum) * 2.0  # Strong positive momentum
            attack_payload['strategic_manipulations']['momentum_20'] = {
                'original': original_momentum,
                'manipulated': strategic_context['momentum_20'],
                'bias': 'bullish'
            }
        
        if 'price_trend' in strategic_context:
            original_trend = strategic_context['price_trend']
            strategic_context['price_trend'] = abs(original_trend) * 1.5  # Positive trend
            attack_payload['strategic_manipulations']['price_trend'] = {
                'original': original_trend,
                'manipulated': strategic_context['price_trend'],
                'bias': 'bullish'
            }
        
        # Tactical: Make it appear bearish
        if 'momentum_20' in tactical_context:
            original_momentum = tactical_context['momentum_20']
            tactical_context['momentum_20'] = -abs(original_momentum) * 2.0  # Strong negative momentum
            attack_payload['tactical_manipulations']['momentum_20'] = {
                'original': original_momentum,
                'manipulated': tactical_context['momentum_20'],
                'bias': 'bearish'
            }
        
        if 'price_trend' in tactical_context:
            original_trend = tactical_context['price_trend']
            tactical_context['price_trend'] = -abs(original_trend) * 1.5  # Negative trend
            attack_payload['tactical_manipulations']['price_trend'] = {
                'original': original_trend,
                'manipulated': tactical_context['price_trend'],
                'bias': 'bearish'
            }
        
        # Create volatility desynchronization
        if 'volatility_30' in strategic_context and 'volatility_30' in tactical_context:
            # Strategic: Low volatility (trending market)
            strategic_context['volatility_30'] = 0.005  # Very low volatility
            # Tactical: High volatility (choppy market)
            tactical_context['volatility_30'] = 0.08   # Very high volatility
            
            attack_payload['context_divergence']['volatility_desync'] = {
                'strategic_volatility': 0.005,
                'tactical_volatility': 0.08,
                'divergence_factor': 16.0  # 16x difference
            }
        
        # Calculate disruption score
        disruption_score = self._calculate_desync_disruption_score(
            original_strategic, original_tactical,
            strategic_context, tactical_context
        )
        
        # Determine attack success
        success = disruption_score > self.coordination_threshold
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = CoordinationAttackResult(
            attack_type=AttackType.DESYNCHRONIZATION.value,
            success=success,
            confidence=disruption_score,
            disrupted_agents=['Strategic_System', 'Tactical_System'],
            gating_weights_before=np.array([0.33, 0.33, 0.34]),  # Placeholder
            gating_weights_after=np.array([0.33, 0.33, 0.34]),  # No change for desync
            coordination_disruption_score=disruption_score,
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Desynchronization attack executed: "
            f"success={success}, disruption_score={disruption_score:.3f}, "
            f"desync_factor={desync_factor}"
        )
        
        return attack_payload, attack_result
    
    def _calculate_consensus_disruption_score(
        self,
        original_predictions: List[Dict[str, Any]],
        disrupted_predictions: List[Dict[str, Any]]
    ) -> float:
        """Calculate consensus disruption score."""
        if not original_predictions or not disrupted_predictions:
            return 0.0
        
        total_disruption = 0.0
        
        for orig, disr in zip(original_predictions, disrupted_predictions):
            orig_probs = np.array(orig.get('action_probabilities', [0.33, 0.33, 0.34]))
            disr_probs = np.array(disr.get('action_probabilities', [0.33, 0.33, 0.34]))
            
            # Calculate KL divergence as disruption measure
            kl_div = np.sum(orig_probs * np.log((orig_probs + 1e-8) / (disr_probs + 1e-8)))
            total_disruption += kl_div
        
        return min(total_disruption / len(original_predictions), 1.0)
    
    def _calculate_communication_disruption_score(
        self,
        original_context: Dict[str, Any],
        jammed_context: Dict[str, Any]
    ) -> float:
        """Calculate communication disruption score."""
        disruption_score = 0.0
        count = 0
        
        for key in original_context:
            if key in jammed_context and isinstance(original_context[key], (int, float)):
                orig_val = original_context[key]
                jammed_val = jammed_context[key]
                
                # Calculate relative change
                rel_change = abs(jammed_val - orig_val) / (abs(orig_val) + 1e-8)
                disruption_score += min(rel_change, 1.0)
                count += 1
        
        return disruption_score / max(count, 1)
    
    def _calculate_gating_disruption_score(
        self,
        original_context: Dict[str, Any],
        manipulated_context: Dict[str, Any],
        target_weights: np.ndarray
    ) -> float:
        """Calculate gating network disruption score."""
        # Context manipulation score
        context_score = self._calculate_communication_disruption_score(
            original_context, manipulated_context
        )
        
        # Weight manipulation potential (heuristic)
        weight_entropy = -np.sum(target_weights * np.log(target_weights + 1e-8))
        normalized_entropy = weight_entropy / np.log(len(target_weights))
        weight_score = 1.0 - normalized_entropy  # Lower entropy = higher disruption
        
        return (context_score + weight_score) / 2.0
    
    def _calculate_desync_disruption_score(
        self,
        orig_strategic: Dict[str, Any],
        orig_tactical: Dict[str, Any],
        manip_strategic: Dict[str, Any],
        manip_tactical: Dict[str, Any]
    ) -> float:
        """Calculate desynchronization disruption score."""
        strategic_score = self._calculate_communication_disruption_score(
            orig_strategic, manip_strategic
        )
        tactical_score = self._calculate_communication_disruption_score(
            orig_tactical, manip_tactical
        )
        
        # Divergence between strategic and tactical
        divergence_score = 0.0
        count = 0
        
        for key in manip_strategic:
            if (key in manip_tactical and 
                isinstance(manip_strategic[key], (int, float)) and
                isinstance(manip_tactical[key], (int, float))):
                
                strategic_val = manip_strategic[key]
                tactical_val = manip_tactical[key]
                
                # Calculate divergence
                if abs(strategic_val) + abs(tactical_val) > 1e-8:
                    divergence = abs(strategic_val - tactical_val) / (abs(strategic_val) + abs(tactical_val))
                    divergence_score += min(divergence, 1.0)
                    count += 1
        
        divergence_score = divergence_score / max(count, 1)
        
        return (strategic_score + tactical_score + divergence_score) / 3.0
    
    def _record_attack_result(self, result: CoordinationAttackResult):
        """Record attack result for analytics."""
        self.attack_history.append(result)
        
        # Update metrics
        self.disruption_metrics['total_attempts'] += 1
        if result.success:
            self.disruption_metrics['successful_attacks'] += 1
        
        # Update success rates
        attack_type = AttackType(result.attack_type)
        type_attempts = len([r for r in self.attack_history if r.attack_type == result.attack_type])
        type_successes = len([r for r in self.attack_history if r.attack_type == result.attack_type and r.success])
        self.success_rates[attack_type] = type_successes / type_attempts
        
        # Update disruption metrics
        self.disruption_metrics['avg_disruption_score'] = np.mean([r.coordination_disruption_score for r in self.attack_history])
        self.disruption_metrics['max_disruption_score'] = max(self.disruption_metrics['max_disruption_score'], result.coordination_disruption_score)
        self.disruption_metrics['agents_compromised'].update(result.disrupted_agents)
        
        # Keep history manageable
        if len(self.attack_history) > 1000:
            self.attack_history = self.attack_history[-500:]
    
    def get_attack_analytics(self) -> Dict[str, Any]:
        """Get comprehensive attack analytics."""
        if not self.attack_history:
            return {'status': 'no_attacks_recorded'}
        
        recent_attacks = self.attack_history[-100:]  # Last 100 attacks
        
        return {
            'total_attacks': len(self.attack_history),
            'recent_attacks': len(recent_attacks),
            'overall_success_rate': self.disruption_metrics['successful_attacks'] / self.disruption_metrics['total_attempts'],
            'success_rates_by_type': {attack_type.value: rate for attack_type, rate in self.success_rates.items()},
            'disruption_metrics': self.disruption_metrics.copy(),
            'recent_performance': {
                'avg_disruption_score': np.mean([r.coordination_disruption_score for r in recent_attacks]),
                'max_disruption_score': max([r.coordination_disruption_score for r in recent_attacks]),
                'avg_execution_time_ms': np.mean([r.execution_time_ms for r in recent_attacks]),
                'success_rate': len([r for r in recent_attacks if r.success]) / len(recent_attacks)
            },
            'attack_type_distribution': {
                attack_type.value: len([r for r in recent_attacks if r.attack_type == attack_type.value])
                for attack_type in AttackType
            }
        }
    
    def reset_attack_history(self):
        """Reset attack history and metrics."""
        self.attack_history.clear()
        self.success_rates = {attack_type: 0.0 for attack_type in AttackType}
        self.disruption_metrics = {
            'total_attempts': 0,
            'successful_attacks': 0,
            'avg_disruption_score': 0.0,
            'max_disruption_score': 0.0,
            'agents_compromised': set()
        }
        self.logger.info("Attack history and metrics reset")

# Example usage and testing functions
def run_coordination_attack_demo():
    """Demonstrate coordination attack capabilities."""
    print("🚨" * 50)
    print("AGENT GAMMA MISSION - MULTI-AGENT COORDINATION ATTACK DEMO")
    print("🚨" * 50)
    
    attacker = MARLCoordinationAttacker()
    
    # Mock agent predictions
    agent_predictions = [
        {
            'agent_name': 'MLMI',
            'action_probabilities': [0.6, 0.2, 0.2],
            'confidence': 0.7
        },
        {
            'agent_name': 'NWRQK',
            'action_probabilities': [0.3, 0.4, 0.3],
            'confidence': 0.6
        },
        {
            'agent_name': 'Regime',
            'action_probabilities': [0.4, 0.3, 0.3],
            'confidence': 0.8
        }
    ]
    
    # Mock shared context
    shared_context = {
        'volatility_30': 0.02,
        'volume_ratio': 1.5,
        'momentum_20': 0.01,
        'momentum_50': 0.005,
        'mmd_score': 0.1,
        'price_trend': 0.008,
        'market_regime': 'trending'
    }
    
    print("\n🎯 ATTACK 1: CONSENSUS DISRUPTION")
    payload, result = attacker.generate_consensus_disruption_attack(
        agent_predictions.copy(), shared_context.copy()
    )
    print(f"Success: {result.success}, Disruption Score: {result.coordination_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 2: COMMUNICATION JAMMING")
    payload, result = attacker.generate_communication_jamming_attack(
        shared_context.copy()
    )
    print(f"Success: {result.success}, Disruption Score: {result.coordination_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 3: GATING EXPLOITATION")
    payload, result = attacker.generate_gating_exploitation_attack(
        shared_context.copy()
    )
    print(f"Success: {result.success}, Disruption Score: {result.coordination_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 4: TIMING ATTACK")
    payload, result = attacker.generate_timing_attack()
    print(f"Success: {result.success}, Disruption Score: {result.coordination_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 5: DESYNCHRONIZATION")
    payload, result = attacker.generate_desynchronization_attack(
        shared_context.copy(), shared_context.copy()
    )
    print(f"Success: {result.success}, Disruption Score: {result.coordination_disruption_score:.3f}")
    
    print("\n📊 ATTACK ANALYTICS")
    analytics = attacker.get_attack_analytics()
    print(f"Overall Success Rate: {analytics['overall_success_rate']:.2%}")
    print(f"Average Disruption Score: {analytics['disruption_metrics']['avg_disruption_score']:.3f}")
    print(f"Agents Compromised: {list(analytics['disruption_metrics']['agents_compromised'])}")
    
    return attacker

if __name__ == "__main__":
    run_coordination_attack_demo()