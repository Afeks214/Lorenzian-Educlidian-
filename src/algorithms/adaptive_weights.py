"""
Adaptive Weight Learning Algorithms

This module implements advanced adaptive weight learning systems for MARL components:
1. Meta-learning based weight adaptation
2. Attention-based agent coordination
3. Performance-driven weight adjustment
4. Multi-armed bandit integration

Mathematical Foundation:
- MAML (Model-Agnostic Meta-Learning) for fast adaptation
- Attention mechanisms for dynamic weight computation
- Thompson Sampling for exploration-exploitation balance
- Gradient-based optimization for weight updates

Key Features:
- Real-time weight adaptation based on performance feedback
- Regime-aware weight adjustment
- Multi-objective optimization for competing objectives
- Robustness to adversarial scenarios

Author: Agent Gamma - Algorithmic Excellence Implementation Specialist
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Weight adaptation strategies"""
    PERFORMANCE_BASED = "performance_based"
    ATTENTION_BASED = "attention_based"
    META_LEARNING = "meta_learning"
    MULTI_ARMED_BANDIT = "multi_armed_bandit"
    HYBRID = "hybrid"


@dataclass
class PerformanceMetrics:
    """Performance metrics for weight adaptation"""
    accuracy: float
    latency_ms: float
    confidence: float
    robustness: float
    regime_adaptability: float
    timestamp: float


@dataclass
class WeightUpdate:
    """Weight update result"""
    agent_weights: np.ndarray
    adaptation_reason: str
    improvement: float
    confidence: float
    timestamp: float


class BaseAdaptiveWeightLearner(ABC):
    """Base class for adaptive weight learning algorithms"""
    
    def __init__(self, n_agents: int, learning_rate: float = 0.01):
        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.weights = np.ones(n_agents) / n_agents  # Initialize uniform
        self.performance_history = deque(maxlen=1000)
        
    @abstractmethod
    def update_weights(
        self,
        agent_performances: np.ndarray,
        market_context: Dict[str, Any]
    ) -> WeightUpdate:
        """Update agent weights based on performance"""
        pass
    
    @abstractmethod
    def get_adaptation_rationale(self) -> str:
        """Get explanation for weight adaptation"""
        pass


class PerformanceBasedAdaptation(BaseAdaptiveWeightLearner):
    """
    Performance-based adaptive weight learning.
    
    Adjusts weights based on recent agent performance with
    exponential decay and momentum terms.
    """
    
    def __init__(self, n_agents: int, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(n_agents, learning_rate)
        self.momentum = momentum
        self.velocity = np.zeros(n_agents)
        self.performance_window = 50
        
    def update_weights(
        self,
        agent_performances: np.ndarray,
        market_context: Dict[str, Any]
    ) -> WeightUpdate:
        """Update weights based on exponential moving average of performance"""
        
        # Store performance
        self.performance_history.append({
            'performances': agent_performances.copy(),
            'timestamp': time.time(),
            'context': market_context
        })
        
        # Calculate exponential moving average of performances
        if len(self.performance_history) >= self.performance_window:
            recent_performances = np.array([
                p['performances'] for p in list(self.performance_history)[-self.performance_window:]
            ])
            
            # Apply exponential decay
            decay_weights = np.exp(-0.1 * np.arange(self.performance_window))
            decay_weights = decay_weights / np.sum(decay_weights)
            
            ema_performances = np.average(recent_performances, axis=0, weights=decay_weights)
        else:
            ema_performances = agent_performances
        
        # Normalize performances to [0, 1]
        if np.max(ema_performances) > np.min(ema_performances):
            normalized_performances = (ema_performances - np.min(ema_performances)) / (np.max(ema_performances) - np.min(ema_performances))
        else:
            normalized_performances = np.ones_like(ema_performances) / len(ema_performances)
        
        # Apply temperature scaling for exploration/exploitation
        temperature = self._calculate_temperature(market_context)
        scaled_performances = normalized_performances / temperature
        
        # Softmax for weight computation
        exp_performances = np.exp(scaled_performances - np.max(scaled_performances))
        target_weights = exp_performances / np.sum(exp_performances)
        
        # Apply momentum
        weight_gradient = target_weights - self.weights
        self.velocity = self.momentum * self.velocity + self.learning_rate * weight_gradient
        
        # Update weights
        old_weights = self.weights.copy()
        self.weights = self.weights + self.velocity
        
        # Ensure weights are non-negative and sum to 1
        self.weights = np.maximum(self.weights, 0.001)  # Minimum weight
        self.weights = self.weights / np.sum(self.weights)
        
        # Calculate improvement
        improvement = np.sum((self.weights - old_weights) * ema_performances)
        
        return WeightUpdate(
            agent_weights=self.weights.copy(),
            adaptation_reason=f"Performance-based adaptation (temp={temperature:.3f})",
            improvement=improvement,
            confidence=self._calculate_confidence(ema_performances),
            timestamp=time.time()
        )
    
    def _calculate_temperature(self, market_context: Dict[str, Any]) -> float:
        """Calculate temperature for exploration/exploitation balance"""
        volatility = market_context.get('volatility_30', 0.01)
        regime = market_context.get('market_regime', 'normal')
        
        base_temperature = 0.1
        
        # Increase temperature (more exploration) in volatile markets
        volatility_factor = 1.0 + (volatility - 0.01) * 10
        
        # Regime-specific adjustments
        regime_factors = {
            'normal': 1.0,
            'volatile': 1.5,
            'crisis': 2.0,
            'recovery': 1.2
        }
        
        regime_factor = regime_factors.get(regime, 1.0)
        
        return base_temperature * volatility_factor * regime_factor
    
    def _calculate_confidence(self, performances: np.ndarray) -> float:
        """Calculate confidence in weight adaptation"""
        # Higher confidence when performance differences are clear
        performance_variance = np.var(performances)
        confidence = min(1.0, performance_variance * 10)
        return confidence
    
    def get_adaptation_rationale(self) -> str:
        """Get explanation for weight adaptation"""
        if len(self.performance_history) == 0:
            return "No performance history available"
        
        recent_perf = self.performance_history[-1]['performances']
        best_agent = np.argmax(recent_perf)
        
        return f"Agent {best_agent} performing best ({recent_perf[best_agent]:.3f}), weights adapted accordingly"


class AttentionBasedAdaptation(BaseAdaptiveWeightLearner):
    """
    Attention-based adaptive weight learning.
    
    Uses attention mechanisms to dynamically compute weights
    based on agent relevance to current market context.
    """
    
    def __init__(self, n_agents: int, context_dim: int = 6, hidden_dim: int = 64):
        super().__init__(n_agents, 0.01)
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(context_dim + n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.attention_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def update_weights(
        self,
        agent_performances: np.ndarray,
        market_context: Dict[str, Any]
    ) -> WeightUpdate:
        """Update weights using attention mechanism"""
        
        # Extract context features
        context_features = self._extract_context_features(market_context)
        
        # Combine context and performance features
        combined_features = torch.cat([
            torch.FloatTensor(context_features),
            torch.FloatTensor(agent_performances)
        ])
        
        # Compute attention weights
        with torch.no_grad():
            attention_weights = self.attention_net(combined_features)
            self.weights = attention_weights.numpy()
        
        # Train attention network with performance feedback
        if len(self.performance_history) > 10:
            self._train_attention_network()
        
        # Store performance
        self.performance_history.append({
            'performances': agent_performances.copy(),
            'context': market_context,
            'weights': self.weights.copy(),
            'timestamp': time.time()
        })
        
        return WeightUpdate(
            agent_weights=self.weights.copy(),
            adaptation_reason="Attention-based context adaptation",
            improvement=self._calculate_improvement(),
            confidence=self._calculate_attention_confidence(),
            timestamp=time.time()
        )
    
    def _extract_context_features(self, market_context: Dict[str, Any]) -> np.ndarray:
        """Extract context features for attention network"""
        features = np.array([
            market_context.get('volatility_30', 0.01),
            market_context.get('volume_ratio', 1.0),
            market_context.get('momentum_20', 0.0),
            market_context.get('momentum_50', 0.0),
            market_context.get('mmd_score', 0.0),
            market_context.get('price_trend', 0.0)
        ])
        
        return features
    
    def _train_attention_network(self):
        """Train attention network with historical performance"""
        if len(self.performance_history) < 20:
            return
        
        # Prepare training data
        batch_size = min(32, len(self.performance_history))
        indices = np.random.choice(len(self.performance_history), batch_size, replace=False)
        
        contexts = []
        performances = []
        targets = []
        
        for idx in indices:
            entry = self.performance_history[idx]
            
            context_features = self._extract_context_features(entry['context'])
            combined_features = np.concatenate([context_features, entry['performances']])
            
            contexts.append(combined_features)
            performances.append(entry['performances'])
            
            # Target: weights that would have maximized performance
            target_weights = self._compute_optimal_weights(entry['performances'])
            targets.append(target_weights)
        
        # Convert to tensors
        contexts_tensor = torch.FloatTensor(contexts)
        targets_tensor = torch.FloatTensor(targets)
        
        # Training step
        self.optimizer.zero_grad()
        predicted_weights = self.attention_net(contexts_tensor)
        loss = self.criterion(predicted_weights, targets_tensor)
        loss.backward()
        self.optimizer.step()
    
    def _compute_optimal_weights(self, performances: np.ndarray) -> np.ndarray:
        """Compute optimal weights for given performances"""
        # Simple heuristic: exponential weighting of performances
        exp_performances = np.exp(performances - np.max(performances))
        optimal_weights = exp_performances / np.sum(exp_performances)
        return optimal_weights
    
    def _calculate_improvement(self) -> float:
        """Calculate improvement from attention-based adaptation"""
        if len(self.performance_history) < 2:
            return 0.0
        
        current_perf = self.performance_history[-1]['performances']
        current_weights = self.performance_history[-1]['weights']
        
        weighted_performance = np.sum(current_weights * current_perf)
        
        return weighted_performance
    
    def _calculate_attention_confidence(self) -> float:
        """Calculate confidence in attention-based weights"""
        # Confidence based on attention weight entropy
        entropy = -np.sum(self.weights * np.log(self.weights + 1e-8))
        max_entropy = np.log(self.n_agents)
        
        # Higher confidence when weights are more concentrated
        confidence = 1.0 - (entropy / max_entropy)
        return confidence
    
    def get_adaptation_rationale(self) -> str:
        """Get explanation for attention-based adaptation"""
        dominant_agent = np.argmax(self.weights)
        attention_strength = self.weights[dominant_agent]
        
        return f"Attention mechanism focused on agent {dominant_agent} (weight: {attention_strength:.3f})"


class MetaLearningAdaptation(BaseAdaptiveWeightLearner):
    """
    Meta-learning based adaptive weight learning.
    
    Uses MAML-inspired approach to quickly adapt weights
    to new market regimes and conditions.
    """
    
    def __init__(self, n_agents: int, meta_lr: float = 0.01, fast_lr: float = 0.1):
        super().__init__(n_agents, meta_lr)
        self.meta_lr = meta_lr
        self.fast_lr = fast_lr
        
        # Meta-parameters for weight generation
        self.meta_params = {
            'context_weights': np.random.randn(6, n_agents) * 0.1,
            'performance_weights': np.random.randn(n_agents, n_agents) * 0.1,
            'bias': np.zeros(n_agents)
        }
        
        self.task_history = deque(maxlen=100)
        
    def update_weights(
        self,
        agent_performances: np.ndarray,
        market_context: Dict[str, Any]
    ) -> WeightUpdate:
        """Update weights using meta-learning approach"""
        
        # Extract context features
        context_features = np.array([
            market_context.get('volatility_30', 0.01),
            market_context.get('volume_ratio', 1.0),
            market_context.get('momentum_20', 0.0),
            market_context.get('momentum_50', 0.0),
            market_context.get('mmd_score', 0.0),
            market_context.get('price_trend', 0.0)
        ])
        
        # Fast adaptation: compute task-specific weights
        task_weights = self._fast_adaptation(context_features, agent_performances)
        
        # Store task for meta-learning
        self.task_history.append({
            'context': context_features,
            'performances': agent_performances,
            'weights': task_weights,
            'timestamp': time.time()
        })
        
        # Meta-learning update
        if len(self.task_history) >= 10:
            self._meta_update()
        
        self.weights = task_weights
        
        return WeightUpdate(
            agent_weights=self.weights.copy(),
            adaptation_reason="Meta-learning fast adaptation",
            improvement=self._calculate_meta_improvement(),
            confidence=self._calculate_meta_confidence(),
            timestamp=time.time()
        )
    
    def _fast_adaptation(self, context: np.ndarray, performances: np.ndarray) -> np.ndarray:
        """Fast adaptation to new task using current meta-parameters"""
        
        # Compute base weights from meta-parameters
        context_contribution = np.dot(context, self.meta_params['context_weights'])
        performance_contribution = np.dot(performances, self.meta_params['performance_weights'])
        
        raw_weights = context_contribution + performance_contribution + self.meta_params['bias']
        
        # Apply softmax
        exp_weights = np.exp(raw_weights - np.max(raw_weights))
        weights = exp_weights / np.sum(exp_weights)
        
        # One-step gradient update for fast adaptation
        gradient = self._compute_weight_gradient(weights, performances)
        adapted_weights = weights + self.fast_lr * gradient
        
        # Normalize
        adapted_weights = np.maximum(adapted_weights, 0.001)
        adapted_weights = adapted_weights / np.sum(adapted_weights)
        
        return adapted_weights
    
    def _compute_weight_gradient(self, weights: np.ndarray, performances: np.ndarray) -> np.ndarray:
        """Compute gradient for weight optimization"""
        # Gradient of weighted performance with respect to weights
        gradient = performances - np.sum(weights * performances)
        return gradient
    
    def _meta_update(self):
        """Update meta-parameters based on task history"""
        if len(self.task_history) < 20:
            return
        
        # Sample batch of tasks
        batch_size = min(10, len(self.task_history))
        task_indices = np.random.choice(len(self.task_history), batch_size, replace=False)
        
        total_meta_grad = {
            'context_weights': np.zeros_like(self.meta_params['context_weights']),
            'performance_weights': np.zeros_like(self.meta_params['performance_weights']),
            'bias': np.zeros_like(self.meta_params['bias'])
        }
        
        for idx in task_indices:
            task = self.task_history[idx]
            
            # Compute meta-gradient for this task
            meta_grad = self._compute_meta_gradient(task)
            
            # Accumulate gradients
            for key in total_meta_grad:
                total_meta_grad[key] += meta_grad[key]
        
        # Update meta-parameters
        for key in self.meta_params:
            self.meta_params[key] += self.meta_lr * total_meta_grad[key] / batch_size
    
    def _compute_meta_gradient(self, task: Dict) -> Dict:
        """Compute meta-gradient for a single task"""
        context = task['context']
        performances = task['performances']
        
        # Compute expected gradient (simplified)
        context_grad = np.outer(context, performances - np.mean(performances))
        performance_grad = np.outer(performances, performances - np.mean(performances))
        bias_grad = performances - np.mean(performances)
        
        return {
            'context_weights': context_grad,
            'performance_weights': performance_grad,
            'bias': bias_grad
        }
    
    def _calculate_meta_improvement(self) -> float:
        """Calculate improvement from meta-learning"""
        if len(self.task_history) < 2:
            return 0.0
        
        recent_tasks = list(self.task_history)[-5:]
        improvements = []
        
        for task in recent_tasks:
            weighted_perf = np.sum(task['weights'] * task['performances'])
            uniform_perf = np.mean(task['performances'])
            improvements.append(weighted_perf - uniform_perf)
        
        return np.mean(improvements)
    
    def _calculate_meta_confidence(self) -> float:
        """Calculate confidence in meta-learning adaptation"""
        if len(self.task_history) < 5:
            return 0.5
        
        # Confidence based on consistency of recent adaptations
        recent_weights = np.array([task['weights'] for task in list(self.task_history)[-5:]])
        weight_stability = 1.0 - np.mean(np.std(recent_weights, axis=0))
        
        return max(0.1, min(1.0, weight_stability))
    
    def get_adaptation_rationale(self) -> str:
        """Get explanation for meta-learning adaptation"""
        return f"Meta-learning adaptation based on {len(self.task_history)} historical tasks"


class MultiArmedBanditAdaptation(BaseAdaptiveWeightLearner):
    """
    Multi-armed bandit based adaptive weight learning.
    
    Uses Thompson Sampling to balance exploration and exploitation
    in agent weight selection.
    """
    
    def __init__(self, n_agents: int, exploration_factor: float = 1.0):
        super().__init__(n_agents, 0.01)
        self.exploration_factor = exploration_factor
        
        # Beta distributions for each agent (Thompson Sampling)
        self.alpha = np.ones(n_agents)  # Successes
        self.beta = np.ones(n_agents)   # Failures
        
        self.total_rewards = np.zeros(n_agents)
        self.n_selections = np.zeros(n_agents)
        
    def update_weights(
        self,
        agent_performances: np.ndarray,
        market_context: Dict[str, Any]
    ) -> WeightUpdate:
        """Update weights using Thompson Sampling"""
        
        # Update Beta distributions with performance feedback
        self._update_beta_distributions(agent_performances)
        
        # Sample from Beta distributions for Thompson Sampling
        sampled_values = np.random.beta(self.alpha, self.beta)
        
        # Convert to weights (exploration vs exploitation)
        exploration_weights = sampled_values / np.sum(sampled_values)
        
        # Exploitation weights based on empirical means
        if np.sum(self.n_selections) > 0:
            exploitation_weights = self.total_rewards / np.maximum(self.n_selections, 1)
            exploitation_weights = exploitation_weights / np.sum(exploitation_weights)
        else:
            exploitation_weights = np.ones(self.n_agents) / self.n_agents
        
        # Combine exploration and exploitation
        self.weights = (
            self.exploration_factor * exploration_weights +
            (1 - self.exploration_factor) * exploitation_weights
        )
        
        # Normalize
        self.weights = self.weights / np.sum(self.weights)
        
        # Store performance
        self.performance_history.append({
            'performances': agent_performances,
            'weights': self.weights.copy(),
            'timestamp': time.time()
        })
        
        return WeightUpdate(
            agent_weights=self.weights.copy(),
            adaptation_reason="Multi-armed bandit Thompson Sampling",
            improvement=self._calculate_bandit_improvement(),
            confidence=self._calculate_bandit_confidence(),
            timestamp=time.time()
        )
    
    def _update_beta_distributions(self, performances: np.ndarray):
        """Update Beta distributions based on performance feedback"""
        
        # Normalize performances to [0, 1] range
        if np.max(performances) > np.min(performances):
            norm_performances = (performances - np.min(performances)) / (np.max(performances) - np.min(performances))
        else:
            norm_performances = np.ones_like(performances) * 0.5
        
        # Update Beta parameters
        for i in range(self.n_agents):
            # Treat performance as probability of success
            success_prob = norm_performances[i]
            
            # Update based on Bernoulli trial
            if np.random.random() < success_prob:
                self.alpha[i] += 1  # Success
            else:
                self.beta[i] += 1   # Failure
            
            # Update tracking statistics
            self.total_rewards[i] += performances[i]
            self.n_selections[i] += 1
    
    def _calculate_bandit_improvement(self) -> float:
        """Calculate improvement from bandit adaptation"""
        if len(self.performance_history) < 2:
            return 0.0
        
        current_perf = self.performance_history[-1]['performances']
        current_weights = self.performance_history[-1]['weights']
        
        weighted_performance = np.sum(current_weights * current_perf)
        random_performance = np.mean(current_perf)
        
        return weighted_performance - random_performance
    
    def _calculate_bandit_confidence(self) -> float:
        """Calculate confidence in bandit adaptation"""
        # Confidence based on variance of Beta distributions
        variances = self.alpha * self.beta / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))
        avg_variance = np.mean(variances)
        
        # Lower variance means higher confidence
        confidence = 1.0 - avg_variance
        return max(0.1, min(1.0, confidence))
    
    def get_adaptation_rationale(self) -> str:
        """Get explanation for bandit adaptation"""
        best_agent = np.argmax(self.total_rewards / np.maximum(self.n_selections, 1))
        confidence_interval = np.sqrt(self.alpha[best_agent] * self.beta[best_agent] / 
                                     (self.alpha[best_agent] + self.beta[best_agent]))
        
        return f"Bandit selected agent {best_agent} with CI width: {confidence_interval:.3f}"


class HybridAdaptiveWeightLearner:
    """
    Hybrid adaptive weight learner combining multiple strategies.
    
    Dynamically selects the best adaptation strategy based on
    current market conditions and performance metrics.
    """
    
    def __init__(self, n_agents: int, context_dim: int = 6):
        self.n_agents = n_agents
        self.context_dim = context_dim
        
        # Initialize all adaptation strategies
        self.strategies = {
            AdaptationStrategy.PERFORMANCE_BASED: PerformanceBasedAdaptation(n_agents),
            AdaptationStrategy.ATTENTION_BASED: AttentionBasedAdaptation(n_agents, context_dim),
            AdaptationStrategy.META_LEARNING: MetaLearningAdaptation(n_agents),
            AdaptationStrategy.MULTI_ARMED_BANDIT: MultiArmedBanditAdaptation(n_agents)
        }
        
        # Strategy selection meta-learner
        self.strategy_performance = {strategy: deque(maxlen=100) for strategy in self.strategies}
        self.current_strategy = AdaptationStrategy.PERFORMANCE_BASED
        
        # Combined weights
        self.weights = np.ones(n_agents) / n_agents
        
    def update_weights(
        self,
        agent_performances: np.ndarray,
        market_context: Dict[str, Any]
    ) -> WeightUpdate:
        """Update weights using hybrid approach"""
        
        # Select best strategy for current context
        best_strategy = self._select_strategy(market_context)
        
        # Update weights using selected strategy
        weight_update = self.strategies[best_strategy].update_weights(
            agent_performances, market_context
        )
        
        # Store strategy performance
        strategy_performance = np.sum(weight_update.agent_weights * agent_performances)
        self.strategy_performance[best_strategy].append(strategy_performance)
        
        # Update combined weights
        self.weights = weight_update.agent_weights
        self.current_strategy = best_strategy
        
        # Enhance update with hybrid information
        weight_update.adaptation_reason = f"Hybrid: {best_strategy.value} - {weight_update.adaptation_reason}"
        
        return weight_update
    
    def _select_strategy(self, market_context: Dict[str, Any]) -> AdaptationStrategy:
        """Select best adaptation strategy for current context"""
        
        # Get market characteristics
        volatility = market_context.get('volatility_30', 0.01)
        regime = market_context.get('market_regime', 'normal')
        
        # Strategy selection logic
        if regime == 'crisis':
            # Use meta-learning for rapid adaptation in crisis
            return AdaptationStrategy.META_LEARNING
        elif volatility > 0.03:
            # Use attention-based for high volatility
            return AdaptationStrategy.ATTENTION_BASED
        elif len(self.strategy_performance[AdaptationStrategy.PERFORMANCE_BASED]) > 50:
            # Use performance-based when we have sufficient history
            return AdaptationStrategy.PERFORMANCE_BASED
        else:
            # Use multi-armed bandit for exploration
            return AdaptationStrategy.MULTI_ARMED_BANDIT
    
    def get_strategy_summary(self) -> Dict:
        """Get summary of strategy performance"""
        summary = {}
        
        for strategy, performance_history in self.strategy_performance.items():
            if len(performance_history) > 0:
                summary[strategy.value] = {
                    'avg_performance': np.mean(performance_history),
                    'std_performance': np.std(performance_history),
                    'n_uses': len(performance_history)
                }
            else:
                summary[strategy.value] = {
                    'avg_performance': 0.0,
                    'std_performance': 0.0,
                    'n_uses': 0
                }
        
        summary['current_strategy'] = self.current_strategy.value
        
        return summary
    
    def get_adaptation_rationale(self) -> str:
        """Get explanation for hybrid adaptation"""
        base_rationale = self.strategies[self.current_strategy].get_adaptation_rationale()
        return f"Hybrid strategy ({self.current_strategy.value}): {base_rationale}"


# Factory functions for easy instantiation
def create_adaptive_weight_learner(
    n_agents: int,
    strategy: AdaptationStrategy = AdaptationStrategy.HYBRID,
    context_dim: int = 6
) -> BaseAdaptiveWeightLearner:
    """
    Factory function to create adaptive weight learner.
    
    Args:
        n_agents: Number of agents
        strategy: Adaptation strategy to use
        context_dim: Dimension of context features
        
    Returns:
        Configured adaptive weight learner
    """
    
    if strategy == AdaptationStrategy.PERFORMANCE_BASED:
        return PerformanceBasedAdaptation(n_agents)
    elif strategy == AdaptationStrategy.ATTENTION_BASED:
        return AttentionBasedAdaptation(n_agents, context_dim)
    elif strategy == AdaptationStrategy.META_LEARNING:
        return MetaLearningAdaptation(n_agents)
    elif strategy == AdaptationStrategy.MULTI_ARMED_BANDIT:
        return MultiArmedBanditAdaptation(n_agents)
    else:  # HYBRID
        return HybridAdaptiveWeightLearner(n_agents, context_dim)


def benchmark_adaptation_strategies(
    n_agents: int = 3,
    n_episodes: int = 1000,
    context_dim: int = 6
) -> Dict:
    """
    Benchmark different adaptation strategies.
    
    Args:
        n_agents: Number of agents
        n_episodes: Number of episodes to test
        context_dim: Context dimension
        
    Returns:
        Benchmark results
    """
    
    strategies = [
        AdaptationStrategy.PERFORMANCE_BASED,
        AdaptationStrategy.ATTENTION_BASED,
        AdaptationStrategy.META_LEARNING,
        AdaptationStrategy.MULTI_ARMED_BANDIT,
        AdaptationStrategy.HYBRID
    ]
    
    results = {}
    
    for strategy in strategies:
        try:
            learner = create_adaptive_weight_learner(n_agents, strategy, context_dim)
            
            total_reward = 0
            adaptation_times = []
            
            for episode in range(n_episodes):
                # Simulate agent performances
                agent_performances = np.random.beta(2, 2, n_agents)
                
                # Simulate market context
                market_context = {
                    'volatility_30': np.random.exponential(0.02),
                    'volume_ratio': np.random.lognormal(0, 0.5),
                    'momentum_20': np.random.normal(0, 0.01),
                    'momentum_50': np.random.normal(0, 0.005),
                    'mmd_score': np.random.exponential(0.1),
                    'price_trend': np.random.normal(0, 0.01),
                    'market_regime': np.random.choice(['normal', 'volatile', 'crisis'])
                }
                
                # Update weights and measure time
                start_time = time.time()
                weight_update = learner.update_weights(agent_performances, market_context)
                adaptation_time = (time.time() - start_time) * 1000  # ms
                
                adaptation_times.append(adaptation_time)
                
                # Calculate reward
                reward = np.sum(weight_update.agent_weights * agent_performances)
                total_reward += reward
            
            results[strategy.value] = {
                'total_reward': total_reward,
                'avg_reward': total_reward / n_episodes,
                'avg_adaptation_time_ms': np.mean(adaptation_times),
                'std_adaptation_time_ms': np.std(adaptation_times),
                'success': True
            }
            
        except Exception as e:
            results[strategy.value] = {
                'error': str(e),
                'success': False
            }
    
    return results