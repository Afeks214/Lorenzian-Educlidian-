#!/usr/bin/env python3
"""
🚨 AGENT GAMMA MISSION - POLICY GRADIENT ATTACK MODULE
Advanced MARL Attack Development: Policy Gradient Exploitation

This module implements sophisticated attacks targeting the policy gradient
mechanisms in Strategic and Tactical MARL systems:
- FGSM/PGD adaptations for RL policy gradient systems
- Action space manipulation attacks
- Reward function vulnerability exploitation
- Policy optimization disruption
- Gradient-based adversarial perturbations

Key Attack Vectors:
1. Fast Gradient Sign Method (FGSM) for Policy Networks
2. Projected Gradient Descent (PGD) for Action Spaces
3. Reward Function Poisoning
4. Policy Gradient Reversal
5. Action Space Boundary Attacks

MISSION OBJECTIVE: Achieve >80% attack success rate against policy gradient defenses
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
import copy

# Attack Result Tracking
@dataclass
class PolicyGradientAttackResult:
    """Results from a policy gradient attack attempt."""
    attack_type: str
    success: bool
    confidence: float
    policy_disruption_score: float
    gradient_manipulation_strength: float
    action_space_corruption: float
    reward_function_impact: float
    original_policy_output: Dict[str, Any]
    attacked_policy_output: Dict[str, Any]
    execution_time_ms: float
    attack_payload: Dict[str, Any]
    timestamp: datetime

class PolicyAttackType(Enum):
    """Types of policy gradient attacks."""
    FGSM_POLICY = "fgsm_policy"
    PGD_ACTION_SPACE = "pgd_action_space"
    REWARD_POISONING = "reward_poisoning"
    GRADIENT_REVERSAL = "gradient_reversal"
    BOUNDARY_ATTACK = "boundary_attack"

class PolicyGradientAttacker:
    """
    Advanced Policy Gradient Attack System.
    
    This system implements sophisticated attacks targeting policy gradient
    mechanisms in MARL systems using adapted FGSM/PGD techniques.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the Policy Gradient Attacker.
        
        Args:
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.device = device
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Attack history and analytics
        self.attack_history = []
        self.success_rates = {attack_type: 0.0 for attack_type in PolicyAttackType}
        self.policy_metrics = {
            'total_attempts': 0,
            'successful_attacks': 0,
            'avg_disruption_score': 0.0,
            'max_disruption_score': 0.0,
            'policies_compromised': set()
        }
        
        # Policy gradient attack parameters
        self.epsilon = 0.1  # Attack strength
        self.alpha = 0.01   # Step size for iterative attacks
        self.max_iterations = 10  # Maximum iterations for PGD
        self.action_space_bounds = [-1.0, 1.0]  # Action space bounds
        
        # Reward function parameters
        self.reward_scale = 1.0
        self.reward_bias = 0.0
        
        self.logger.info(f"PolicyGradientAttacker initialized: device={device}")
    
    def generate_fgsm_policy_attack(
        self,
        policy_network: nn.Module,
        input_state: torch.Tensor,
        target_action: torch.Tensor = None,
        epsilon: float = None
    ) -> Tuple[Dict[str, Any], PolicyGradientAttackResult]:
        """
        🎯 ATTACK 1: FAST GRADIENT SIGN METHOD (FGSM) FOR POLICY NETWORKS
        
        Applies FGSM to policy networks to generate adversarial inputs that
        cause the policy to produce suboptimal actions.
        
        Args:
            policy_network: Target policy network
            input_state: Input state tensor
            target_action: Target action to achieve (optional)
            epsilon: Attack strength (optional)
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if epsilon is None:
            epsilon = self.epsilon
        
        # Store original inputs
        original_state = input_state.clone()
        
        # Enable gradient computation
        input_state.requires_grad_(True)
        
        # Get original policy output
        policy_network.eval()
        with torch.no_grad():
            original_output = policy_network(original_state)
            original_action_probs = F.softmax(original_output, dim=-1)
        
        # Generate attack payload
        attack_payload = {
            'attack_type': PolicyAttackType.FGSM_POLICY.value,
            'epsilon': epsilon,
            'original_state': original_state.detach().clone(),
            'target_action': target_action,
            'gradient_info': {},
            'perturbation_details': {}
        }
        
        # FGSM attack implementation
        if target_action is not None:
            # Targeted attack: push policy towards target action
            target_action_tensor = target_action
        else:
            # Untargeted attack: push policy away from original action
            target_action_tensor = torch.argmax(original_action_probs, dim=-1)
        
        # Forward pass
        policy_output = policy_network(input_state)
        
        # Calculate loss based on attack type
        if target_action is not None:
            # Targeted: minimize distance to target action
            loss = F.cross_entropy(policy_output, target_action_tensor)
        else:
            # Untargeted: maximize entropy (make policy uncertain)
            action_probs = F.softmax(policy_output, dim=-1)
            loss = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
        
        # Backward pass to get gradients
        loss.backward()
        
        # Get gradients
        data_grad = input_state.grad.data
        
        # FGSM perturbation
        if target_action is not None:
            # Targeted attack: step towards target
            perturbation = -epsilon * data_grad.sign()
        else:
            # Untargeted attack: step away from original
            perturbation = epsilon * data_grad.sign()
        
        # Apply perturbation
        adversarial_state = original_state + perturbation
        
        # Clip to maintain valid state space
        adversarial_state = torch.clamp(adversarial_state, -3.0, 3.0)
        
        # Get attacked policy output
        with torch.no_grad():
            attacked_output = policy_network(adversarial_state)
            attacked_action_probs = F.softmax(attacked_output, dim=-1)
        
        # Record attack details
        attack_payload['gradient_info'] = {
            'gradient_norm': torch.norm(data_grad).item(),
            'gradient_direction': data_grad.sign().tolist(),
            'loss_value': loss.item()
        }
        
        attack_payload['perturbation_details'] = {
            'perturbation': perturbation.tolist(),
            'perturbation_norm': torch.norm(perturbation).item(),
            'adversarial_state': adversarial_state.tolist()
        }
        
        # Calculate disruption metrics
        policy_disruption = self._calculate_policy_disruption(
            original_action_probs, attacked_action_probs
        )
        
        gradient_strength = torch.norm(data_grad).item()
        
        # Determine attack success
        if target_action is not None:
            # Targeted success: achieved target action
            success = torch.argmax(attacked_action_probs, dim=-1) == target_action_tensor
        else:
            # Untargeted success: significant policy change
            success = policy_disruption > 0.3
        
        success = success.item() if torch.is_tensor(success) else success
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = PolicyGradientAttackResult(
            attack_type=PolicyAttackType.FGSM_POLICY.value,
            success=success,
            confidence=policy_disruption,
            policy_disruption_score=policy_disruption,
            gradient_manipulation_strength=gradient_strength,
            action_space_corruption=0.0,  # Not applicable for FGSM
            reward_function_impact=0.0,   # Not applicable for FGSM
            original_policy_output={
                'action_probs': original_action_probs.tolist(),
                'raw_output': original_output.tolist()
            },
            attacked_policy_output={
                'action_probs': attacked_action_probs.tolist(),
                'raw_output': attacked_output.tolist()
            },
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"FGSM policy attack executed: "
            f"success={success}, disruption_score={policy_disruption:.3f}, "
            f"epsilon={epsilon}"
        )
        
        # Return adversarial state in attack payload
        attack_payload['adversarial_state'] = adversarial_state
        
        return attack_payload, attack_result
    
    def generate_pgd_action_space_attack(
        self,
        policy_network: nn.Module,
        input_state: torch.Tensor,
        target_action: torch.Tensor = None,
        epsilon: float = None,
        alpha: float = None,
        max_iterations: int = None
    ) -> Tuple[Dict[str, Any], PolicyGradientAttackResult]:
        """
        🎯 ATTACK 2: PROJECTED GRADIENT DESCENT (PGD) FOR ACTION SPACES
        
        Applies iterative PGD attack to manipulate action space exploration
        and force suboptimal action selection.
        
        Args:
            policy_network: Target policy network
            input_state: Input state tensor
            target_action: Target action to achieve (optional)
            epsilon: Attack strength (optional)
            alpha: Step size (optional)
            max_iterations: Maximum iterations (optional)
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if epsilon is None:
            epsilon = self.epsilon
        if alpha is None:
            alpha = self.alpha
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        # Store original inputs
        original_state = input_state.clone()
        
        # Get original policy output
        policy_network.eval()
        with torch.no_grad():
            original_output = policy_network(original_state)
            original_action_probs = F.softmax(original_output, dim=-1)
        
        # Generate attack payload
        attack_payload = {
            'attack_type': PolicyAttackType.PGD_ACTION_SPACE.value,
            'epsilon': epsilon,
            'alpha': alpha,
            'max_iterations': max_iterations,
            'original_state': original_state.detach().clone(),
            'target_action': target_action,
            'pgd_trajectory': [],
            'iteration_losses': []
        }
        
        # Initialize adversarial state
        adversarial_state = original_state.clone()
        
        # PGD iterative attack
        for iteration in range(max_iterations):
            adversarial_state.requires_grad_(True)
            
            # Forward pass
            policy_output = policy_network(adversarial_state)
            
            # Calculate loss
            if target_action is not None:
                # Targeted: minimize distance to target action
                loss = F.cross_entropy(policy_output, target_action)
            else:
                # Untargeted: maximize action space corruption
                action_probs = F.softmax(policy_output, dim=-1)
                # Minimize confidence in top action
                max_prob = torch.max(action_probs, dim=-1)[0]
                loss = max_prob
            
            # Backward pass
            loss.backward()
            
            # Get gradients
            data_grad = adversarial_state.grad.data
            
            # PGD step
            if target_action is not None:
                # Targeted: step towards target
                perturbation = -alpha * data_grad.sign()
            else:
                # Untargeted: step to maximize uncertainty
                perturbation = alpha * data_grad.sign()
            
            # Apply perturbation
            adversarial_state = adversarial_state.detach() + perturbation
            
            # Project back to epsilon ball
            perturbation_total = adversarial_state - original_state
            perturbation_total = torch.clamp(perturbation_total, -epsilon, epsilon)
            adversarial_state = original_state + perturbation_total
            
            # Clip to maintain valid state space
            adversarial_state = torch.clamp(adversarial_state, -3.0, 3.0)
            
            # Record iteration
            attack_payload['pgd_trajectory'].append({
                'iteration': iteration,
                'loss': loss.item(),
                'gradient_norm': torch.norm(data_grad).item(),
                'perturbation_norm': torch.norm(perturbation).item()
            })
            
            attack_payload['iteration_losses'].append(loss.item())
        
        # Get final attacked policy output
        with torch.no_grad():
            attacked_output = policy_network(adversarial_state)
            attacked_action_probs = F.softmax(attacked_output, dim=-1)
        
        # Calculate disruption metrics
        policy_disruption = self._calculate_policy_disruption(
            original_action_probs, attacked_action_probs
        )
        
        action_space_corruption = self._calculate_action_space_corruption(
            original_action_probs, attacked_action_probs
        )
        
        # Determine attack success
        if target_action is not None:
            # Targeted success: achieved target action
            success = torch.argmax(attacked_action_probs, dim=-1) == target_action
        else:
            # Untargeted success: significant action space corruption
            success = action_space_corruption > 0.4
        
        success = success.item() if torch.is_tensor(success) else success
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = PolicyGradientAttackResult(
            attack_type=PolicyAttackType.PGD_ACTION_SPACE.value,
            success=success,
            confidence=policy_disruption,
            policy_disruption_score=policy_disruption,
            gradient_manipulation_strength=np.mean([t['gradient_norm'] for t in attack_payload['pgd_trajectory']]),
            action_space_corruption=action_space_corruption,
            reward_function_impact=0.0,   # Not applicable for PGD
            original_policy_output={
                'action_probs': original_action_probs.tolist(),
                'raw_output': original_output.tolist()
            },
            attacked_policy_output={
                'action_probs': attacked_action_probs.tolist(),
                'raw_output': attacked_output.tolist()
            },
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"PGD action space attack executed: "
            f"success={success}, disruption_score={policy_disruption:.3f}, "
            f"action_corruption={action_space_corruption:.3f}, iterations={max_iterations}"
        )
        
        # Return adversarial state in attack payload
        attack_payload['adversarial_state'] = adversarial_state
        
        return attack_payload, attack_result
    
    def generate_reward_poisoning_attack(
        self,
        reward_function: Callable,
        state: torch.Tensor,
        action: torch.Tensor,
        poisoning_type: str = 'reward_reversal',
        poisoning_strength: float = 1.0
    ) -> Tuple[Dict[str, Any], PolicyGradientAttackResult]:
        """
        🎯 ATTACK 3: REWARD FUNCTION POISONING
        
        Poisons the reward function to mislead policy learning and cause
        suboptimal policy updates.
        
        Args:
            reward_function: Target reward function
            state: Current state
            action: Current action
            poisoning_type: Type of reward poisoning
            poisoning_strength: Strength of poisoning
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        # Get original reward
        original_reward = reward_function(state, action)
        
        # Generate attack payload
        attack_payload = {
            'attack_type': PolicyAttackType.REWARD_POISONING.value,
            'poisoning_type': poisoning_type,
            'poisoning_strength': poisoning_strength,
            'original_reward': original_reward,
            'poisoning_methods': {}
        }
        
        # Apply reward poisoning
        if poisoning_type == 'reward_reversal':
            # Reverse the reward signal
            poisoned_reward = -original_reward * poisoning_strength
            
            attack_payload['poisoning_methods']['reversal'] = {
                'factor': -poisoning_strength,
                'description': 'Reverse reward signal'
            }
        
        elif poisoning_type == 'reward_scaling':
            # Scale down positive rewards, amplify negative rewards
            if original_reward > 0:
                poisoned_reward = original_reward * (1 - poisoning_strength)
            else:
                poisoned_reward = original_reward * (1 + poisoning_strength)
            
            attack_payload['poisoning_methods']['scaling'] = {
                'positive_factor': (1 - poisoning_strength),
                'negative_factor': (1 + poisoning_strength),
                'description': 'Asymmetric reward scaling'
            }
        
        elif poisoning_type == 'reward_noise':
            # Add adversarial noise to rewards
            noise = torch.randn_like(original_reward) * poisoning_strength
            poisoned_reward = original_reward + noise
            
            attack_payload['poisoning_methods']['noise'] = {
                'noise_std': poisoning_strength,
                'noise_value': noise.item(),
                'description': 'Adversarial reward noise'
            }
        
        elif poisoning_type == 'reward_delay':
            # Delay reward signal (simulated)
            poisoned_reward = original_reward * 0.1  # Greatly reduced immediate reward
            
            attack_payload['poisoning_methods']['delay'] = {
                'delay_factor': 0.1,
                'description': 'Delayed reward signal'
            }
        
        else:
            # Default to reward reversal
            poisoned_reward = -original_reward * poisoning_strength
            attack_payload['poisoning_methods']['default'] = {
                'factor': -poisoning_strength,
                'description': 'Default reward reversal'
            }
        
        # Calculate reward function impact
        reward_impact = abs(original_reward - poisoned_reward) / (abs(original_reward) + 1e-8)
        reward_impact = reward_impact.item() if torch.is_tensor(reward_impact) else reward_impact
        
        # Determine attack success
        success = reward_impact > 0.5  # Significant reward change
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = PolicyGradientAttackResult(
            attack_type=PolicyAttackType.REWARD_POISONING.value,
            success=success,
            confidence=reward_impact,
            policy_disruption_score=0.0,  # Not directly applicable
            gradient_manipulation_strength=0.0,  # Not applicable
            action_space_corruption=0.0,  # Not applicable
            reward_function_impact=reward_impact,
            original_policy_output={
                'reward': original_reward.item() if torch.is_tensor(original_reward) else original_reward
            },
            attacked_policy_output={
                'reward': poisoned_reward.item() if torch.is_tensor(poisoned_reward) else poisoned_reward
            },
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Reward poisoning attack executed: "
            f"success={success}, reward_impact={reward_impact:.3f}, "
            f"poisoning_type={poisoning_type}"
        )
        
        # Return poisoned reward in attack payload
        attack_payload['poisoned_reward'] = poisoned_reward
        
        return attack_payload, attack_result
    
    def generate_gradient_reversal_attack(
        self,
        policy_network: nn.Module,
        input_state: torch.Tensor,
        loss_function: Callable,
        reversal_strength: float = 1.0
    ) -> Tuple[Dict[str, Any], PolicyGradientAttackResult]:
        """
        🎯 ATTACK 4: POLICY GRADIENT REVERSAL
        
        Reverses policy gradients to cause the policy to learn in the opposite
        direction, leading to suboptimal behavior.
        
        Args:
            policy_network: Target policy network
            input_state: Input state tensor
            loss_function: Loss function to reverse
            reversal_strength: Strength of gradient reversal
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        # Store original policy parameters
        original_params = {}
        for name, param in policy_network.named_parameters():
            original_params[name] = param.clone()
        
        # Generate attack payload
        attack_payload = {
            'attack_type': PolicyAttackType.GRADIENT_REVERSAL.value,
            'reversal_strength': reversal_strength,
            'gradient_manipulations': {},
            'parameter_changes': {}
        }
        
        # Get original policy output
        policy_network.eval()
        with torch.no_grad():
            original_output = policy_network(input_state)
            original_action_probs = F.softmax(original_output, dim=-1)
        
        # Enable training mode for gradient computation
        policy_network.train()
        
        # Forward pass
        policy_output = policy_network(input_state)
        
        # Calculate loss
        loss = loss_function(policy_output, input_state)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Reverse gradients
        for name, param in policy_network.named_parameters():
            if param.grad is not None:
                # Store original gradient
                original_grad = param.grad.clone()
                
                # Reverse gradient
                param.grad = -param.grad * reversal_strength
                
                # Record manipulation
                attack_payload['gradient_manipulations'][name] = {
                    'original_grad_norm': torch.norm(original_grad).item(),
                    'reversed_grad_norm': torch.norm(param.grad).item(),
                    'reversal_strength': reversal_strength
                }
        
        # Simulate gradient step (without actually updating the network)
        with torch.no_grad():
            for name, param in policy_network.named_parameters():
                if param.grad is not None:
                    # Simulate parameter update
                    param_change = -0.01 * param.grad  # Assume learning rate of 0.01
                    attack_payload['parameter_changes'][name] = {
                        'param_change_norm': torch.norm(param_change).item(),
                        'relative_change': torch.norm(param_change).item() / (torch.norm(param).item() + 1e-8)
                    }
        
        # Get attacked policy output (with reversed gradients applied)
        with torch.no_grad():
            attacked_output = policy_network(input_state)
            attacked_action_probs = F.softmax(attacked_output, dim=-1)
        
        # Restore original parameters (for testing purposes)
        for name, param in policy_network.named_parameters():
            param.data = original_params[name]
        
        # Calculate disruption metrics
        policy_disruption = self._calculate_policy_disruption(
            original_action_probs, attacked_action_probs
        )
        
        gradient_strength = np.mean([
            info['reversed_grad_norm'] for info in attack_payload['gradient_manipulations'].values()
        ])
        
        # Determine attack success
        success = policy_disruption > 0.2  # Significant policy change
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = PolicyGradientAttackResult(
            attack_type=PolicyAttackType.GRADIENT_REVERSAL.value,
            success=success,
            confidence=policy_disruption,
            policy_disruption_score=policy_disruption,
            gradient_manipulation_strength=gradient_strength,
            action_space_corruption=0.0,  # Not directly applicable
            reward_function_impact=0.0,   # Not applicable
            original_policy_output={
                'action_probs': original_action_probs.tolist(),
                'raw_output': original_output.tolist()
            },
            attacked_policy_output={
                'action_probs': attacked_action_probs.tolist(),
                'raw_output': attacked_output.tolist()
            },
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Gradient reversal attack executed: "
            f"success={success}, disruption_score={policy_disruption:.3f}, "
            f"gradient_strength={gradient_strength:.3f}"
        )
        
        return attack_payload, attack_result
    
    def generate_boundary_attack(
        self,
        policy_network: nn.Module,
        input_state: torch.Tensor,
        action_bounds: Tuple[float, float] = None,
        boundary_type: str = 'hard_boundary'
    ) -> Tuple[Dict[str, Any], PolicyGradientAttackResult]:
        """
        🎯 ATTACK 5: ACTION SPACE BOUNDARY ATTACK
        
        Attacks the boundaries of the action space to force the policy to
        select actions at the extreme boundaries, causing suboptimal behavior.
        
        Args:
            policy_network: Target policy network
            input_state: Input state tensor
            action_bounds: Action space bounds
            boundary_type: Type of boundary attack
            
        Returns:
            Tuple of (attack_payload, attack_result)
        """
        start_time = time.time()
        
        if action_bounds is None:
            action_bounds = tuple(self.action_space_bounds)
        
        # Generate attack payload
        attack_payload = {
            'attack_type': PolicyAttackType.BOUNDARY_ATTACK.value,
            'action_bounds': action_bounds,
            'boundary_type': boundary_type,
            'boundary_manipulations': {},
            'action_forcing': {}
        }
        
        # Get original policy output
        policy_network.eval()
        with torch.no_grad():
            original_output = policy_network(input_state)
            original_action_probs = F.softmax(original_output, dim=-1)
        
        # Apply boundary attack
        if boundary_type == 'hard_boundary':
            # Force actions to extreme boundaries
            attacked_action_probs = original_action_probs.clone()
            
            # Find dominant action
            dominant_action = torch.argmax(attacked_action_probs, dim=-1)
            
            # Force to boundary actions (buy=0, sell=2)
            if dominant_action == 1:  # If hold, force to buy or sell
                attacked_action_probs[0, 0] = 0.9  # Force buy
                attacked_action_probs[0, 1] = 0.05  # Reduce hold
                attacked_action_probs[0, 2] = 0.05  # Reduce sell
            else:
                # Amplify existing boundary action
                attacked_action_probs[0, dominant_action] = 0.95
                attacked_action_probs[0, 1] = 0.025  # Reduce hold
                attacked_action_probs[0, 2-dominant_action] = 0.025  # Reduce opposite
            
            attack_payload['boundary_manipulations']['hard_boundary'] = {
                'forced_action': dominant_action.item(),
                'boundary_probability': 0.95,
                'description': 'Force actions to extreme boundaries'
            }
        
        elif boundary_type == 'soft_boundary':
            # Gradually push actions toward boundaries
            attacked_action_probs = original_action_probs.clone()
            
            # Reduce middle action (hold) probability
            attacked_action_probs[0, 1] *= 0.3  # Reduce hold
            
            # Amplify boundary actions
            attacked_action_probs[0, 0] *= 1.5  # Amplify buy
            attacked_action_probs[0, 2] *= 1.5  # Amplify sell
            
            # Renormalize
            attacked_action_probs = attacked_action_probs / torch.sum(attacked_action_probs, dim=-1, keepdim=True)
            
            attack_payload['boundary_manipulations']['soft_boundary'] = {
                'hold_reduction': 0.3,
                'boundary_amplification': 1.5,
                'description': 'Gradually push toward boundaries'
            }
        
        elif boundary_type == 'oscillating_boundary':
            # Create oscillating boundary preferences
            attacked_action_probs = original_action_probs.clone()
            
            # Create oscillation pattern
            oscillation = torch.sin(torch.tensor(time.time()))
            
            if oscillation > 0:
                # Favor buy boundary
                attacked_action_probs[0, 0] = 0.8
                attacked_action_probs[0, 1] = 0.1
                attacked_action_probs[0, 2] = 0.1
            else:
                # Favor sell boundary
                attacked_action_probs[0, 0] = 0.1
                attacked_action_probs[0, 1] = 0.1
                attacked_action_probs[0, 2] = 0.8
            
            attack_payload['boundary_manipulations']['oscillating_boundary'] = {
                'oscillation_value': oscillation.item(),
                'favored_boundary': 'buy' if oscillation > 0 else 'sell',
                'description': 'Oscillating boundary preferences'
            }
        
        # Calculate disruption metrics
        policy_disruption = self._calculate_policy_disruption(
            original_action_probs, attacked_action_probs
        )
        
        action_space_corruption = self._calculate_action_space_corruption(
            original_action_probs, attacked_action_probs
        )
        
        # Determine attack success
        success = action_space_corruption > 0.3  # Significant boundary manipulation
        
        # Create attack result
        execution_time_ms = (time.time() - start_time) * 1000
        attack_result = PolicyGradientAttackResult(
            attack_type=PolicyAttackType.BOUNDARY_ATTACK.value,
            success=success,
            confidence=policy_disruption,
            policy_disruption_score=policy_disruption,
            gradient_manipulation_strength=0.0,  # Not applicable
            action_space_corruption=action_space_corruption,
            reward_function_impact=0.0,   # Not applicable
            original_policy_output={
                'action_probs': original_action_probs.tolist(),
                'raw_output': original_output.tolist()
            },
            attacked_policy_output={
                'action_probs': attacked_action_probs.tolist(),
                'raw_output': attacked_action_probs.tolist()  # Use probs as output
            },
            execution_time_ms=execution_time_ms,
            attack_payload=attack_payload,
            timestamp=datetime.now()
        )
        
        self._record_attack_result(attack_result)
        
        self.logger.info(
            f"Boundary attack executed: "
            f"success={success}, disruption_score={policy_disruption:.3f}, "
            f"action_corruption={action_space_corruption:.3f}, boundary_type={boundary_type}"
        )
        
        # Return attacked action probabilities in attack payload
        attack_payload['attacked_action_probs'] = attacked_action_probs
        
        return attack_payload, attack_result
    
    def _calculate_policy_disruption(
        self,
        original_probs: torch.Tensor,
        attacked_probs: torch.Tensor
    ) -> float:
        """Calculate policy disruption score using KL divergence."""
        # Calculate KL divergence
        kl_div = F.kl_div(
            torch.log(attacked_probs + 1e-8),
            original_probs,
            reduction='batchmean'
        )
        
        # Normalize to [0, 1] range
        return min(kl_div.item(), 1.0)
    
    def _calculate_action_space_corruption(
        self,
        original_probs: torch.Tensor,
        attacked_probs: torch.Tensor
    ) -> float:
        """Calculate action space corruption score."""
        # Calculate L2 distance between probability distributions
        l2_distance = torch.norm(original_probs - attacked_probs, p=2)
        
        # Normalize by maximum possible distance
        max_distance = torch.norm(torch.ones_like(original_probs), p=2)
        
        corruption_score = l2_distance / max_distance
        
        return min(corruption_score.item(), 1.0)
    
    def _record_attack_result(self, result: PolicyGradientAttackResult):
        """Record attack result for analytics."""
        self.attack_history.append(result)
        
        # Update metrics
        self.policy_metrics['total_attempts'] += 1
        if result.success:
            self.policy_metrics['successful_attacks'] += 1
        
        # Update success rates
        attack_type = PolicyAttackType(result.attack_type)
        type_attempts = len([r for r in self.attack_history if r.attack_type == result.attack_type])
        type_successes = len([r for r in self.attack_history if r.attack_type == result.attack_type and r.success])
        self.success_rates[attack_type] = type_successes / type_attempts
        
        # Update disruption metrics
        self.policy_metrics['avg_disruption_score'] = np.mean([r.policy_disruption_score for r in self.attack_history])
        self.policy_metrics['max_disruption_score'] = max(self.policy_metrics['max_disruption_score'], result.policy_disruption_score)
        self.policy_metrics['policies_compromised'].add(result.attack_type)
        
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
            'overall_success_rate': self.policy_metrics['successful_attacks'] / self.policy_metrics['total_attempts'],
            'success_rates_by_type': {attack_type.value: rate for attack_type, rate in self.success_rates.items()},
            'policy_metrics': self.policy_metrics.copy(),
            'recent_performance': {
                'avg_disruption_score': np.mean([r.policy_disruption_score for r in recent_attacks]),
                'max_disruption_score': max([r.policy_disruption_score for r in recent_attacks]),
                'avg_gradient_strength': np.mean([r.gradient_manipulation_strength for r in recent_attacks if r.gradient_manipulation_strength > 0]),
                'avg_action_corruption': np.mean([r.action_space_corruption for r in recent_attacks if r.action_space_corruption > 0]),
                'success_rate': len([r for r in recent_attacks if r.success]) / len(recent_attacks)
            },
            'attack_type_distribution': {
                attack_type.value: len([r for r in recent_attacks if r.attack_type == attack_type.value])
                for attack_type in PolicyAttackType
            }
        }

# Mock policy network for testing
class MockPolicyNetwork(nn.Module):
    """Mock policy network for testing purposes."""
    
    def __init__(self, input_dim: int = 13, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Example usage and testing functions
def run_policy_gradient_attack_demo():
    """Demonstrate policy gradient attack capabilities."""
    print("🚨" * 50)
    print("AGENT GAMMA MISSION - POLICY GRADIENT ATTACK DEMO")
    print("🚨" * 50)
    
    attacker = PolicyGradientAttacker()
    
    # Create mock policy network
    policy_network = MockPolicyNetwork()
    
    # Mock input state
    input_state = torch.randn(1, 13)  # Batch size 1, 13 features
    
    # Mock target action
    target_action = torch.tensor([0])  # Buy action
    
    # Mock reward function
    def mock_reward_function(state, action):
        return torch.randn(1)  # Random reward
    
    # Mock loss function
    def mock_loss_function(output, state):
        return torch.mean(output ** 2)  # Simple loss
    
    print("\n🎯 ATTACK 1: FGSM POLICY ATTACK")
    payload, result = attacker.generate_fgsm_policy_attack(
        policy_network, input_state.clone(), target_action
    )
    print(f"Success: {result.success}, Disruption Score: {result.policy_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 2: PGD ACTION SPACE ATTACK")
    payload, result = attacker.generate_pgd_action_space_attack(
        policy_network, input_state.clone(), target_action
    )
    print(f"Success: {result.success}, Disruption Score: {result.policy_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 3: REWARD POISONING ATTACK")
    payload, result = attacker.generate_reward_poisoning_attack(
        mock_reward_function, input_state.clone(), target_action, 'reward_reversal'
    )
    print(f"Success: {result.success}, Reward Impact: {result.reward_function_impact:.3f}")
    
    print("\n🎯 ATTACK 4: GRADIENT REVERSAL ATTACK")
    payload, result = attacker.generate_gradient_reversal_attack(
        policy_network, input_state.clone(), mock_loss_function
    )
    print(f"Success: {result.success}, Disruption Score: {result.policy_disruption_score:.3f}")
    
    print("\n🎯 ATTACK 5: BOUNDARY ATTACK")
    payload, result = attacker.generate_boundary_attack(
        policy_network, input_state.clone(), boundary_type='hard_boundary'
    )
    print(f"Success: {result.success}, Action Corruption: {result.action_space_corruption:.3f}")
    
    print("\n📊 POLICY GRADIENT ATTACK ANALYTICS")
    analytics = attacker.get_attack_analytics()
    print(f"Overall Success Rate: {analytics['overall_success_rate']:.2%}")
    print(f"Average Disruption Score: {analytics['policy_metrics']['avg_disruption_score']:.3f}")
    print(f"Policies Compromised: {list(analytics['policy_metrics']['policies_compromised'])}")
    
    return attacker

if __name__ == "__main__":
    run_policy_gradient_attack_demo()