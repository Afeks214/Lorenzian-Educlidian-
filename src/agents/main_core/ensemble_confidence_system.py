"""
Ensemble Confidence System - Replacement for MC Dropout in Strategic Layer.

This module provides an efficient ensemble-based confidence mechanism that
replaces MC Dropout for strategic decision-making while maintaining
decision quality and improving performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfidenceMetrics:
    """Comprehensive confidence metrics from ensemble evaluation."""
    agreement_score: float
    consensus_strength: float
    divergence_metric: float
    prediction_entropy: float
    confidence_score: float
    decision_boundary_distance: float
    ensemble_variance: float
    weighted_consensus: torch.Tensor
    member_predictions: List[torch.Tensor]
    

@dataclass
class EnsembleResult:
    """Complete ensemble evaluation result."""
    should_proceed: bool
    predicted_action: int
    action_probabilities: torch.Tensor
    confidence_metrics: EnsembleConfidenceMetrics
    ensemble_statistics: Dict[str, Any]
    performance_metrics: Dict[str, float]


class EnsembleConfidenceBase(ABC):
    """Base class for ensemble confidence mechanisms."""
    
    @abstractmethod
    def evaluate(self, models: List[nn.Module], input_state: torch.Tensor,
                 market_context: Optional[Dict[str, Any]] = None) -> EnsembleResult:
        """Evaluate ensemble confidence."""
        pass
    
    @abstractmethod
    def update_weights(self, performance_history: List[Dict[str, Any]]):
        """Update ensemble member weights based on performance."""
        pass


class WeightedEnsembleConfidence(EnsembleConfidenceBase):
    """
    Weighted ensemble confidence mechanism.
    
    Uses multiple model snapshots with adaptive weights based on
    recent performance to create consensus decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.65)
        self.n_ensemble_members = config.get('n_ensemble_members', 5)
        self.weight_decay = config.get('weight_decay', 0.95)
        self.min_weight = config.get('min_weight', 0.1)
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Initialize equal weights
        self.member_weights = torch.ones(self.n_ensemble_members) / self.n_ensemble_members
        self.member_weights = self.member_weights.to(self.device)
        
        # Performance tracking
        self.performance_history = []
        self.evaluation_count = 0
        
    def evaluate(self, models: List[nn.Module], input_state: torch.Tensor,
                 market_context: Optional[Dict[str, Any]] = None) -> EnsembleResult:
        """
        Evaluate ensemble confidence.
        
        Args:
            models: List of model snapshots or ensemble members
            input_state: Input state tensor
            market_context: Market context information
            
        Returns:
            EnsembleResult with confidence metrics and decision
        """
        start_time = time.time()
        
        # Ensure we have enough models
        if len(models) < self.n_ensemble_members:
            logger.warning(f"Only {len(models)} models available, need {self.n_ensemble_members}")
            # Pad with repeated models if necessary
            models = models + [models[-1]] * (self.n_ensemble_members - len(models))
        
        # Move input to device
        input_state = input_state.to(self.device)
        
        # Collect predictions from all ensemble members
        member_predictions = []
        
        with torch.no_grad():
            for i, model in enumerate(models[:self.n_ensemble_members]):
                model.eval()
                
                # Get prediction from this ensemble member
                output = model(input_state)
                
                # Extract action probabilities
                if isinstance(output, dict):
                    probs = output.get('action_probs', output.get('probs'))
                else:
                    probs = F.softmax(output, dim=-1)
                
                member_predictions.append(probs)
        
        # Stack predictions: [n_members, batch_size, n_actions]
        ensemble_predictions = torch.stack(member_predictions)
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(
            ensemble_predictions, 
            self.member_weights
        )
        
        # Apply adaptive thresholding
        threshold = self._calculate_adaptive_threshold(
            confidence_metrics,
            market_context
        )
        
        # Make final decision
        predicted_action = confidence_metrics.weighted_consensus.argmax(dim=-1)
        confidence = confidence_metrics.weighted_consensus.max(dim=-1)[0]
        should_proceed = (predicted_action == 0) & (confidence >= threshold)
        
        # Calculate performance metrics
        inference_time = (time.time() - start_time) * 1000  # ms
        performance_metrics = {
            'inference_time_ms': inference_time,
            'ensemble_size': len(models),
            'consensus_strength': confidence_metrics.consensus_strength,
            'threshold_used': threshold
        }
        
        # Update evaluation count
        self.evaluation_count += 1
        
        return EnsembleResult(
            should_proceed=should_proceed.item(),
            predicted_action=predicted_action.item(),
            action_probabilities=confidence_metrics.weighted_consensus,
            confidence_metrics=confidence_metrics,
            ensemble_statistics=self._calculate_ensemble_statistics(ensemble_predictions),
            performance_metrics=performance_metrics
        )
    
    def _calculate_confidence_metrics(self, predictions: torch.Tensor, 
                                    weights: torch.Tensor) -> EnsembleConfidenceMetrics:
        """Calculate comprehensive confidence metrics from ensemble predictions."""
        
        # Weighted consensus
        weighted_consensus = torch.sum(
            predictions * weights.view(-1, 1, 1), 
            dim=0
        )
        
        # Agreement score (how much members agree)
        pairwise_agreement = torch.zeros(len(predictions))
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # KL divergence between predictions
                kl_div = F.kl_div(
                    F.log_softmax(predictions[i], dim=-1),
                    F.softmax(predictions[j], dim=-1),
                    reduction='batchmean'
                )
                pairwise_agreement[i] += 1.0 / (1.0 + kl_div)
                pairwise_agreement[j] += 1.0 / (1.0 + kl_div)
        
        agreement_score = pairwise_agreement.mean().item()
        
        # Consensus strength (entropy of weighted consensus)
        consensus_entropy = -torch.sum(
            weighted_consensus * torch.log(weighted_consensus + 1e-8),
            dim=-1
        ).mean().item()
        consensus_strength = 1.0 / (1.0 + consensus_entropy)
        
        # Divergence metric (variance across ensemble)
        ensemble_variance = predictions.var(dim=0).mean().item()
        divergence_metric = ensemble_variance
        
        # Prediction entropy
        prediction_entropy = -torch.sum(
            weighted_consensus * torch.log(weighted_consensus + 1e-8),
            dim=-1
        ).mean().item()
        
        # Overall confidence score
        confidence_score = (agreement_score + consensus_strength) / 2.0
        
        # Decision boundary distance
        max_prob = weighted_consensus.max(dim=-1)[0]
        second_max_prob = torch.topk(weighted_consensus, 2, dim=-1)[0][:, 1]
        decision_boundary_distance = (max_prob - second_max_prob).mean().item()
        
        return EnsembleConfidenceMetrics(
            agreement_score=agreement_score,
            consensus_strength=consensus_strength,
            divergence_metric=divergence_metric,
            prediction_entropy=prediction_entropy,
            confidence_score=confidence_score,
            decision_boundary_distance=decision_boundary_distance,
            ensemble_variance=ensemble_variance,
            weighted_consensus=weighted_consensus,
            member_predictions=[pred.clone() for pred in predictions]
        )
    
    def _calculate_adaptive_threshold(self, metrics: EnsembleConfidenceMetrics,
                                    market_context: Optional[Dict[str, Any]]) -> float:
        """Calculate adaptive confidence threshold based on context."""
        
        base_threshold = self.confidence_threshold
        
        # Adjust based on ensemble agreement
        if metrics.agreement_score < 0.7:
            base_threshold += 0.05  # Require higher confidence when ensemble disagrees
        
        # Adjust based on market context
        if market_context:
            volatility = market_context.get('volatility', 1.0)
            if volatility > 2.0:
                base_threshold += 0.1  # Higher threshold in volatile markets
            
            regime = market_context.get('regime', 'normal')
            if regime == 'crisis':
                base_threshold += 0.15  # Much higher threshold in crisis
        
        # Adjust based on ensemble variance
        if metrics.ensemble_variance > 0.1:
            base_threshold += 0.05  # Higher threshold when ensemble is uncertain
        
        return np.clip(base_threshold, 0.5, 0.95)
    
    def _calculate_ensemble_statistics(self, predictions: torch.Tensor) -> Dict[str, Any]:
        """Calculate statistical properties of ensemble predictions."""
        
        return {
            'mean_prediction': predictions.mean(dim=0),
            'std_prediction': predictions.std(dim=0),
            'median_prediction': predictions.median(dim=0)[0],
            'min_prediction': predictions.min(dim=0)[0],
            'max_prediction': predictions.max(dim=0)[0],
            'prediction_range': predictions.max(dim=0)[0] - predictions.min(dim=0)[0],
            'coefficient_of_variation': predictions.std(dim=0) / (predictions.mean(dim=0) + 1e-8)
        }
    
    def update_weights(self, performance_history: List[Dict[str, Any]]):
        """Update ensemble member weights based on recent performance."""
        
        if len(performance_history) < 10:
            return  # Need enough history to update weights
        
        # Calculate performance scores for each member
        member_scores = torch.zeros(self.n_ensemble_members)
        
        for perf_record in performance_history[-50:]:  # Use last 50 records
            member_predictions = perf_record.get('member_predictions', [])
            actual_outcome = perf_record.get('actual_outcome', 0)
            
            for i, prediction in enumerate(member_predictions):
                if i < self.n_ensemble_members:
                    # Calculate accuracy score
                    predicted_action = prediction.argmax().item()
                    accuracy = 1.0 if predicted_action == actual_outcome else 0.0
                    
                    # Calculate confidence calibration
                    confidence = prediction.max().item()
                    calibration = 1.0 - abs(confidence - accuracy)
                    
                    # Combined score
                    member_scores[i] += (accuracy + calibration) / 2.0
        
        # Normalize scores
        member_scores = member_scores / len(performance_history[-50:])
        
        # Update weights with exponential decay
        self.member_weights = self.member_weights * self.weight_decay + \
                             member_scores * (1 - self.weight_decay)
        
        # Ensure minimum weight
        self.member_weights = torch.clamp(self.member_weights, self.min_weight, 1.0)
        
        # Normalize weights
        self.member_weights = self.member_weights / self.member_weights.sum()
        
        logger.info(f"Updated ensemble weights: {self.member_weights.tolist()}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the ensemble system."""
        
        return {
            'ensemble_size': self.n_ensemble_members,
            'member_weights': self.member_weights.tolist(),
            'evaluation_count': self.evaluation_count,
            'weight_entropy': -torch.sum(
                self.member_weights * torch.log(self.member_weights + 1e-8)
            ).item(),
            'effective_ensemble_size': 1.0 / torch.sum(self.member_weights ** 2).item(),
            'performance_history_length': len(self.performance_history)
        }


class SnapshotEnsembleConfidence(EnsembleConfidenceBase):
    """
    Snapshot ensemble confidence using model checkpoints.
    
    Maintains multiple snapshots of the same model taken at different
    training points to create diversity without MC Dropout overhead.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.snapshot_paths = config.get('snapshot_paths', [])
        self.confidence_threshold = config.get('confidence_threshold', 0.65)
        self.device = torch.device(config.get('device', 'cpu'))
        
        # Load snapshots
        self.snapshots = []
        self._load_snapshots()
        
    def _load_snapshots(self):
        """Load model snapshots from disk."""
        
        for path in self.snapshot_paths:
            try:
                # In practice, this would load actual model checkpoints
                # For now, we'll use placeholder logic
                logger.info(f"Loading snapshot from {path}")
                # snapshot = torch.load(path, map_location=self.device)
                # self.snapshots.append(snapshot)
            except Exception as e:
                logger.warning(f"Failed to load snapshot from {path}: {e}")
        
        if not self.snapshots:
            logger.warning("No snapshots loaded, using single model")
    
    def evaluate(self, models: List[nn.Module], input_state: torch.Tensor,
                 market_context: Optional[Dict[str, Any]] = None) -> EnsembleResult:
        """Evaluate using snapshot ensemble."""
        
        # Use the main model if no snapshots available
        if not self.snapshots:
            return self._single_model_evaluation(models[0], input_state, market_context)
        
        # Use snapshots for ensemble evaluation
        return self._snapshot_ensemble_evaluation(input_state, market_context)
    
    def _single_model_evaluation(self, model: nn.Module, input_state: torch.Tensor,
                                market_context: Optional[Dict[str, Any]]) -> EnsembleResult:
        """Fallback to single model evaluation."""
        
        with torch.no_grad():
            model.eval()
            output = model(input_state)
            
            if isinstance(output, dict):
                probs = output.get('action_probs', output.get('probs'))
            else:
                probs = F.softmax(output, dim=-1)
        
        # Create mock confidence metrics
        confidence_metrics = EnsembleConfidenceMetrics(
            agreement_score=1.0,
            consensus_strength=probs.max().item(),
            divergence_metric=0.0,
            prediction_entropy=-torch.sum(probs * torch.log(probs + 1e-8)).item(),
            confidence_score=probs.max().item(),
            decision_boundary_distance=0.0,
            ensemble_variance=0.0,
            weighted_consensus=probs,
            member_predictions=[probs]
        )
        
        predicted_action = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1)[0]
        should_proceed = (predicted_action == 0) & (confidence >= self.confidence_threshold)
        
        return EnsembleResult(
            should_proceed=should_proceed.item(),
            predicted_action=predicted_action.item(),
            action_probabilities=probs,
            confidence_metrics=confidence_metrics,
            ensemble_statistics={'single_model': True},
            performance_metrics={'inference_time_ms': 0.0}
        )
    
    def _snapshot_ensemble_evaluation(self, input_state: torch.Tensor,
                                    market_context: Optional[Dict[str, Any]]) -> EnsembleResult:
        """Evaluate using snapshot ensemble."""
        
        # This would implement actual snapshot ensemble evaluation
        # For now, return placeholder
        return self._single_model_evaluation(None, input_state, market_context)
    
    def update_weights(self, performance_history: List[Dict[str, Any]]):
        """Update snapshot weights based on performance."""
        pass  # Snapshots have equal weights by default


class DiversityEnsembleConfidence(EnsembleConfidenceBase):
    """
    Diversity-based ensemble confidence.
    
    Uses different model architectures or initialization seeds
    to create diverse ensemble members.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.diversity_method = config.get('diversity_method', 'initialization')
        self.confidence_threshold = config.get('confidence_threshold', 0.65)
        self.device = torch.device(config.get('device', 'cpu'))
        
    def evaluate(self, models: List[nn.Module], input_state: torch.Tensor,
                 market_context: Optional[Dict[str, Any]] = None) -> EnsembleResult:
        """Evaluate using diversity ensemble."""
        
        # Use weighted ensemble as base implementation
        weighted_ensemble = WeightedEnsembleConfidence(self.config)
        return weighted_ensemble.evaluate(models, input_state, market_context)
    
    def update_weights(self, performance_history: List[Dict[str, Any]]):
        """Update diversity weights based on performance."""
        pass


class EnsembleConfidenceFactory:
    """Factory for creating ensemble confidence mechanisms."""
    
    @staticmethod
    def create_ensemble_confidence(config: Dict[str, Any]) -> EnsembleConfidenceBase:
        """Create ensemble confidence mechanism based on config."""
        
        method = config.get('method', 'weighted')
        
        if method == 'weighted':
            return WeightedEnsembleConfidence(config)
        elif method == 'snapshot':
            return SnapshotEnsembleConfidence(config)
        elif method == 'diversity':
            return DiversityEnsembleConfidence(config)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")


class EnsembleConfidenceManager:
    """
    Manager for ensemble confidence system.
    
    Provides high-level interface for strategic agents to use
    ensemble confidence instead of MC Dropout.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ensemble_confidence = EnsembleConfidenceFactory.create_ensemble_confidence(config)
        self.performance_tracker = []
        
    def evaluate_confidence(self, models: List[nn.Module], input_state: torch.Tensor,
                          market_context: Optional[Dict[str, Any]] = None) -> EnsembleResult:
        """Main interface for confidence evaluation."""
        
        result = self.ensemble_confidence.evaluate(models, input_state, market_context)
        
        # Log evaluation
        logger.debug(f"Ensemble confidence: {result.confidence_metrics.confidence_score:.3f}")
        
        return result
    
    def update_performance(self, evaluation_result: EnsembleResult, 
                         actual_outcome: Any):
        """Update performance tracking for ensemble learning."""
        
        performance_record = {
            'timestamp': time.time(),
            'predicted_action': evaluation_result.predicted_action,
            'confidence': evaluation_result.confidence_metrics.confidence_score,
            'actual_outcome': actual_outcome,
            'member_predictions': evaluation_result.confidence_metrics.member_predictions
        }
        
        self.performance_tracker.append(performance_record)
        
        # Keep limited history
        if len(self.performance_tracker) > 1000:
            self.performance_tracker = self.performance_tracker[-500:]
        
        # Update ensemble weights periodically
        if len(self.performance_tracker) % 50 == 0:
            self.ensemble_confidence.update_weights(self.performance_tracker)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        
        base_diagnostics = self.ensemble_confidence.get_diagnostics()
        
        if self.performance_tracker:
            recent_performance = self.performance_tracker[-100:]
            accuracy = sum(1 for p in recent_performance 
                         if p['predicted_action'] == p['actual_outcome']) / len(recent_performance)
            
            base_diagnostics.update({
                'recent_accuracy': accuracy,
                'total_evaluations': len(self.performance_tracker),
                'avg_confidence': np.mean([p['confidence'] for p in recent_performance])
            })
        
        return base_diagnostics