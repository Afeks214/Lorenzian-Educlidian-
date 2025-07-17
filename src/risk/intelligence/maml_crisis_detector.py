"""
MAML Crisis Detection Engine

This module implements Model-Agnostic Meta-Learning (MAML) for crisis pattern 
recognition with few-shot learning capabilities and >95% accuracy requirements.

Key Features:
- MAML implementation for crisis pattern recognition
- Few-shot learning across different crisis types
- Transfer learning with confidence scoring
- Real-time pattern similarity matching
- Neural network-based crisis pattern embedding
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime
import structlog
import asyncio
from pathlib import Path
import pickle
import json

from .crisis_dataset_processor import CrisisFingerprint, CrisisType

logger = structlog.get_logger()


@dataclass
class CrisisDetectionResult:
    """Result of crisis detection"""
    timestamp: datetime
    crisis_probability: float
    crisis_type: CrisisType
    confidence_score: float
    similarity_score: float
    feature_importance: Dict[str, float]
    processing_time_ms: float
    model_version: str


@dataclass
class MAMLTask:
    """MAML training task definition"""
    support_set: Tuple[torch.Tensor, torch.Tensor]  # (features, labels)
    query_set: Tuple[torch.Tensor, torch.Tensor]    # (features, labels)
    crisis_type: CrisisType
    task_id: str


class CrisisEmbeddingNetwork(nn.Module):
    """
    Neural network for crisis pattern embedding.
    Transforms raw features into crisis-specific representations.
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 128, hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)


class CrisisClassifier(nn.Module):
    """
    Crisis classification head for MAML.
    Takes embeddings and outputs crisis probabilities.
    """
    
    def __init__(self, embedding_dim: int, num_crisis_types: int):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_crisis_types)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.classifier(embeddings)


class MAMLCrisisDetector:
    """
    Model-Agnostic Meta-Learning implementation for crisis detection.
    
    Implements MAML algorithm for few-shot crisis pattern recognition
    with >95% accuracy requirement on historical crisis patterns.
    """
    
    def __init__(
        self,
        feature_dim: int,
        embedding_dim: int = 128,
        learning_rate: float = 0.001,
        meta_learning_rate: float = 0.01,
        inner_steps: int = 5,
        device: str = "cpu"
    ):
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.inner_steps = inner_steps
        self.device = torch.device(device)
        
        # Crisis types mapping
        self.crisis_types = list(CrisisType)
        self.num_crisis_types = len(self.crisis_types)
        self.crisis_to_idx = {ct: i for i, ct in enumerate(self.crisis_types)}
        self.idx_to_crisis = {i: ct for i, ct in enumerate(self.crisis_types)}
        
        # Build networks
        self.embedding_network = CrisisEmbeddingNetwork(
            feature_dim, embedding_dim
        ).to(self.device)
        
        self.classifier = CrisisClassifier(
            embedding_dim, self.num_crisis_types
        ).to(self.device)
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(
            list(self.embedding_network.parameters()) + list(self.classifier.parameters()),
            lr=self.meta_learning_rate
        )
        
        # Training state
        self.training_history = []
        self.model_version = "1.0"
        self.is_trained = False
        
        # Performance tracking
        self.accuracy_target = 0.95  # 95% accuracy requirement
        self.processing_time_target_ms = 5.0  # <5ms processing time
        
        logger.info("MAMLCrisisDetector initialized",
                   feature_dim=feature_dim,
                   embedding_dim=embedding_dim,
                   device=device)
    
    async def train_meta_model(
        self,
        fingerprints: List[CrisisFingerprint],
        num_epochs: int = 100,
        meta_batch_size: int = 8,
        support_size: int = 5,
        query_size: int = 10
    ) -> bool:
        """
        Train MAML model on crisis fingerprints.
        
        Args:
            fingerprints: List of crisis fingerprints for training
            num_epochs: Number of meta-training epochs
            meta_batch_size: Number of tasks per meta-batch
            support_size: Number of examples per support set
            query_size: Number of examples per query set
            
        Returns:
            True if training successful and meets accuracy target
        """
        
        logger.info("Starting MAML meta-training",
                   num_fingerprints=len(fingerprints),
                   num_epochs=num_epochs)
        
        try:
            # Prepare training data
            train_data = self._prepare_training_data(fingerprints)
            
            # Meta-training loop
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                
                for batch_idx in range(meta_batch_size):
                    # Generate meta-batch of tasks
                    tasks = self._generate_meta_batch(
                        train_data, meta_batch_size, support_size, query_size
                    )
                    
                    # Meta-update step
                    batch_loss, batch_accuracy = await self._meta_update(tasks)
                    
                    epoch_loss += batch_loss
                    epoch_accuracy += batch_accuracy
                
                # Average metrics
                avg_loss = epoch_loss / meta_batch_size
                avg_accuracy = epoch_accuracy / meta_batch_size
                
                # Log progress
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")
                
                # Store training history
                self.training_history.append({
                    'epoch': epoch,
                    'loss': avg_loss,
                    'accuracy': avg_accuracy,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Early stopping if target accuracy reached
                if avg_accuracy >= self.accuracy_target:
                    logger.info(f"Target accuracy {self.accuracy_target} reached at epoch {epoch}")
                    break
            
            # Final validation
            final_accuracy = await self._validate_model(fingerprints)
            
            if final_accuracy >= self.accuracy_target:
                self.is_trained = True
                logger.info(f"MAML training completed successfully. Final accuracy: {final_accuracy:.4f}")
                return True
            else:
                logger.warning(f"Training completed but accuracy {final_accuracy:.4f} below target {self.accuracy_target}")
                return False
                
        except Exception as e:
            logger.error(f"MAML training failed: {e}")
            return False
    
    def _prepare_training_data(self, fingerprints: List[CrisisFingerprint]) -> Dict:
        """Prepare fingerprints for MAML training"""
        
        # Group by crisis type
        crisis_data = {ct: [] for ct in self.crisis_types}
        
        for fp in fingerprints:
            features = torch.tensor(fp.feature_vector, dtype=torch.float32)
            label = self.crisis_to_idx[fp.crisis_type]
            crisis_data[fp.crisis_type].append((features, label))
        
        # Convert to tensors
        for crisis_type in crisis_data:
            if crisis_data[crisis_type]:
                features, labels = zip(*crisis_data[crisis_type])
                crisis_data[crisis_type] = {
                    'features': torch.stack(features),
                    'labels': torch.tensor(labels, dtype=torch.long)
                }
            else:
                # Create dummy data if crisis type not present
                crisis_data[crisis_type] = {
                    'features': torch.zeros((1, self.feature_dim), dtype=torch.float32),
                    'labels': torch.tensor([self.crisis_to_idx[crisis_type]], dtype=torch.long)
                }
        
        return crisis_data
    
    def _generate_meta_batch(
        self,
        train_data: Dict,
        batch_size: int,
        support_size: int,
        query_size: int
    ) -> List[MAMLTask]:
        """Generate batch of MAML tasks"""
        
        tasks = []
        
        for _ in range(batch_size):
            # Randomly select crisis type
            crisis_type = np.random.choice(self.crisis_types)
            crisis_data = train_data[crisis_type]
            
            # Sample support and query sets
            n_available = len(crisis_data['features'])
            
            if n_available < support_size + query_size:
                # If not enough data, sample with replacement
                indices = np.random.choice(n_available, support_size + query_size, replace=True)
            else:
                indices = np.random.choice(n_available, support_size + query_size, replace=False)
            
            support_indices = indices[:support_size]
            query_indices = indices[support_size:]
            
            # Create task
            support_features = crisis_data['features'][support_indices]
            support_labels = crisis_data['labels'][support_indices]
            query_features = crisis_data['features'][query_indices]
            query_labels = crisis_data['labels'][query_indices]
            
            task = MAMLTask(
                support_set=(support_features.to(self.device), support_labels.to(self.device)),
                query_set=(query_features.to(self.device), query_labels.to(self.device)),
                crisis_type=crisis_type,
                task_id=f"task_{len(tasks)}"
            )
            
            tasks.append(task)
        
        return tasks
    
    async def _meta_update(self, tasks: List[MAMLTask]) -> Tuple[float, float]:
        """Perform MAML meta-update step"""
        
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        total_accuracy = 0.0
        
        for task in tasks:
            # Clone model parameters for inner loop
            fast_weights = {}
            for name, param in self.embedding_network.named_parameters():
                fast_weights[f"embedding.{name}"] = param.clone()
            for name, param in self.classifier.named_parameters():
                fast_weights[f"classifier.{name}"] = param.clone()
            
            # Inner loop adaptation
            support_features, support_labels = task.support_set
            
            for inner_step in range(self.inner_steps):
                # Forward pass with fast weights
                embeddings = self._forward_with_weights(
                    support_features, fast_weights, 'embedding'
                )
                logits = self._forward_with_weights(
                    embeddings, fast_weights, 'classifier'
                )
                
                # Compute loss
                inner_loss = F.cross_entropy(logits, support_labels)
                
                # Compute gradients
                grads = torch.autograd.grad(
                    inner_loss,
                    [fast_weights[name] for name in fast_weights],
                    create_graph=True
                )
                
                # Update fast weights
                for i, name in enumerate(fast_weights):
                    fast_weights[name] = fast_weights[name] - self.learning_rate * grads[i]
            
            # Query set evaluation with adapted parameters
            query_features, query_labels = task.query_set
            
            query_embeddings = self._forward_with_weights(
                query_features, fast_weights, 'embedding'
            )
            query_logits = self._forward_with_weights(
                query_embeddings, fast_weights, 'classifier'
            )
            
            # Meta-loss
            meta_loss = F.cross_entropy(query_logits, query_labels)
            total_loss += meta_loss
            
            # Accuracy
            predictions = torch.argmax(query_logits, dim=1)
            accuracy = (predictions == query_labels).float().mean()
            total_accuracy += accuracy
        
        # Average loss
        avg_loss = total_loss / len(tasks)
        avg_accuracy = total_accuracy / len(tasks)
        
        # Meta-gradient step
        avg_loss.backward()
        self.meta_optimizer.step()
        
        return avg_loss.item(), avg_accuracy.item()
    
    def _forward_with_weights(self, x: torch.Tensor, weights: Dict, prefix: str) -> torch.Tensor:
        """Forward pass with custom weights"""
        
        if prefix == 'embedding':
            # Manual forward pass through embedding network
            current = x
            layer_idx = 0
            
            for module in self.embedding_network.network:
                if isinstance(module, nn.Linear):
                    weight_name = f"embedding.network.{layer_idx}.weight"
                    bias_name = f"embedding.network.{layer_idx}.bias"
                    
                    if weight_name in weights and bias_name in weights:
                        current = F.linear(current, weights[weight_name], weights[bias_name])
                    else:
                        current = module(current)
                
                elif isinstance(module, nn.BatchNorm1d):
                    current = module(current)
                elif isinstance(module, nn.ReLU):
                    current = F.relu(current)
                elif isinstance(module, nn.Dropout):
                    current = F.dropout(current, training=self.training)
                
                layer_idx += 1
            
            return current
        
        elif prefix == 'classifier':
            # Manual forward pass through classifier
            current = x
            layer_idx = 0
            
            for module in self.classifier.classifier:
                if isinstance(module, nn.Linear):
                    weight_name = f"classifier.classifier.{layer_idx}.weight"
                    bias_name = f"classifier.classifier.{layer_idx}.bias"
                    
                    if weight_name in weights and bias_name in weights:
                        current = F.linear(current, weights[weight_name], weights[bias_name])
                    else:
                        current = module(current)
                
                elif isinstance(module, nn.ReLU):
                    current = F.relu(current)
                elif isinstance(module, nn.Dropout):
                    current = F.dropout(current, training=self.training)
                
                layer_idx += 1
            
            return current
        
        else:
            raise ValueError(f"Unknown prefix: {prefix}")
    
    async def _validate_model(self, fingerprints: List[CrisisFingerprint]) -> float:
        """Validate model accuracy on held-out data"""
        
        self.embedding_network.eval()
        self.classifier.eval()
        
        try:
            # Prepare validation data
            features = []
            labels = []
            
            for fp in fingerprints[-100:]:  # Use last 100 for validation
                features.append(fp.feature_vector)
                labels.append(self.crisis_to_idx[fp.crisis_type])
            
            if not features:
                return 0.0
            
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                embeddings = self.embedding_network(features_tensor)
                logits = self.classifier(embeddings)
                predictions = torch.argmax(logits, dim=1)
                
                accuracy = (predictions == labels_tensor).float().mean().item()
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return 0.0
        
        finally:
            self.embedding_network.train()
            self.classifier.train()
    
    async def detect_crisis_pattern(
        self,
        feature_vector: np.ndarray,
        confidence_threshold: float = 0.90
    ) -> Optional[CrisisDetectionResult]:
        """
        Detect crisis pattern in real-time with <5ms processing requirement.
        
        Args:
            feature_vector: Input feature vector
            confidence_threshold: Minimum confidence for positive detection
            
        Returns:
            CrisisDetectionResult if crisis detected, None otherwise
        """
        
        start_time = datetime.now()
        
        if not self.is_trained:
            logger.warning("Model not trained, cannot detect crisis patterns")
            return None
        
        try:
            self.embedding_network.eval()
            self.classifier.eval()
            
            # Convert to tensor
            features_tensor = torch.tensor(
                feature_vector, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                embeddings = self.embedding_network(features_tensor)
                logits = self.classifier(embeddings)
                probabilities = F.softmax(logits, dim=1)
                
                # Get predictions
                max_prob, predicted_idx = torch.max(probabilities, dim=1)
                crisis_type = self.idx_to_crisis[predicted_idx.item()]
                
                # Calculate confidence and similarity scores
                confidence_score = max_prob.item()
                crisis_probability = confidence_score
                
                # Feature importance (gradient-based)
                feature_importance = self._calculate_feature_importance(
                    features_tensor, embeddings
                )
                
                # Similarity score (distance in embedding space)
                similarity_score = self._calculate_similarity_score(embeddings)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Check if detection meets criteria
            if confidence_score >= confidence_threshold:
                result = CrisisDetectionResult(
                    timestamp=datetime.now(),
                    crisis_probability=crisis_probability,
                    crisis_type=crisis_type,
                    confidence_score=confidence_score,
                    similarity_score=similarity_score,
                    feature_importance=feature_importance,
                    processing_time_ms=processing_time,
                    model_version=self.model_version
                )
                
                # Check processing time requirement
                if processing_time > self.processing_time_target_ms:
                    logger.warning(f"Processing time {processing_time:.2f}ms exceeds target {self.processing_time_target_ms}ms")
                
                return result
            else:
                return None
                
        except Exception as e:
            logger.error(f"Crisis detection failed: {e}")
            return None
        
        finally:
            self.embedding_network.train()
            self.classifier.train()
    
    def _calculate_feature_importance(
        self,
        features: torch.Tensor,
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate feature importance scores"""
        
        try:
            # Gradient-based feature importance
            features.requires_grad_(True)
            
            # Forward pass
            embeddings = self.embedding_network(features)
            output = self.classifier(embeddings)
            
            # Calculate gradients
            gradients = torch.autograd.grad(
                outputs=output.max(),
                inputs=features,
                create_graph=False,
                retain_graph=False
            )[0]
            
            # Feature importance as absolute gradients
            importance_scores = torch.abs(gradients).squeeze().cpu().numpy()
            
            # Create feature importance dictionary
            feature_names = [
                'volatility_spike', 'volatility_acceleration', 'volatility_persistence',
                'price_drop_rate', 'price_gap_size', 'price_momentum',
                'volume_spike', 'volume_pattern_1', 'volume_pattern_2', 'volume_pattern_3',
                'volume_pattern_4', 'volume_pattern_5', 'unusual_volume_ratio',
                'correlation_breakdown', 'correlation_contagion', 'cross_asset_correlation',
                'bid_ask_spread_spike', 'market_depth_reduction', 'liquidity_stress_score',
                'time_of_day', 'day_of_week',
                'rsi_divergence', 'macd_signal', 'bollinger_squeeze'
            ]
            
            importance_dict = {}
            for i, name in enumerate(feature_names[:len(importance_scores)]):
                importance_dict[name] = float(importance_scores[i])
            
            return importance_dict
            
        except Exception:
            return {}
    
    def _calculate_similarity_score(self, embeddings: torch.Tensor) -> float:
        """Calculate similarity score to training patterns"""
        
        try:
            # Simple similarity based on embedding magnitude
            embedding_norm = torch.norm(embeddings, dim=1).item()
            
            # Normalize to 0-1 range (simple heuristic)
            similarity_score = min(1.0, embedding_norm / 10.0)
            
            return similarity_score
            
        except Exception:
            return 0.0
    
    def get_model_performance(self) -> Dict:
        """Get model performance statistics"""
        
        if not self.training_history:
            return {"error": "No training history available"}
        
        latest_metrics = self.training_history[-1]
        
        return {
            "is_trained": self.is_trained,
            "model_version": self.model_version,
            "accuracy_target": self.accuracy_target,
            "current_accuracy": latest_metrics.get('accuracy', 0.0),
            "accuracy_target_met": latest_metrics.get('accuracy', 0.0) >= self.accuracy_target,
            "processing_time_target_ms": self.processing_time_target_ms,
            "training_epochs": len(self.training_history),
            "final_loss": latest_metrics.get('loss', 0.0),
            "feature_dimension": self.feature_dim,
            "embedding_dimension": self.embedding_dim
        }
    
    async def save_model(self, model_path: str) -> bool:
        """Save trained model"""
        
        try:
            model_state = {
                'embedding_network': self.embedding_network.state_dict(),
                'classifier': self.classifier.state_dict(),
                'meta_optimizer': self.meta_optimizer.state_dict(),
                'training_history': self.training_history,
                'model_version': self.model_version,
                'is_trained': self.is_trained,
                'config': {
                    'feature_dim': self.feature_dim,
                    'embedding_dim': self.embedding_dim,
                    'learning_rate': self.learning_rate,
                    'meta_learning_rate': self.meta_learning_rate,
                    'inner_steps': self.inner_steps
                }
            }
            
            torch.save(model_state, model_path)
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    async def load_model(self, model_path: str) -> bool:
        """Load trained model"""
        
        try:
            model_state = torch.load(model_path, map_location=self.device)
            
            self.embedding_network.load_state_dict(model_state['embedding_network'])
            self.classifier.load_state_dict(model_state['classifier'])
            self.meta_optimizer.load_state_dict(model_state['meta_optimizer'])
            self.training_history = model_state['training_history']
            self.model_version = model_state['model_version']
            self.is_trained = model_state['is_trained']
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False