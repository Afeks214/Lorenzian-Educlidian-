"""
Adversarial Detection Engine

Detects model poisoning, gradient manipulation, and other adversarial attacks
in the trading system with real-time monitoring and automated response.
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Create placeholder stats module
    class stats:
        @staticmethod
        def norm(*args, **kwargs):
            return type('obj', (object,), {'pdf': lambda x: 1.0})()
        @staticmethod
        def chi2(*args, **kwargs):
            return type('obj', (object,), {'pdf': lambda x: 1.0})()
        @staticmethod
        def ks_2samp(*args, **kwargs):
            return (0.0, 1.0)
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create placeholder classes
    class IsolationForest:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            pass
        def predict(self, *args, **kwargs):
            return []
    
    class StandardScaler:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            pass
        def transform(self, *args, **kwargs):
            return []
import threading
import hashlib
import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.event_bus import EventBus
from src.core.events import Event


class AttackType(Enum):
    MODEL_POISONING = "model_poisoning"
    GRADIENT_MANIPULATION = "gradient_manipulation"
    DATA_POISONING = "data_poisoning"
    BACKDOOR_ATTACK = "backdoor_attack"
    ADVERSARIAL_EXAMPLES = "adversarial_examples"
    BYZANTINE_ATTACK = "byzantine_attack"
    REWARD_HACKING = "reward_hacking"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AttackSignature:
    """Signature of a detected attack"""
    attack_type: AttackType
    threat_level: ThreatLevel
    confidence: float
    timestamp: datetime
    agent_id: str
    description: str
    evidence: Dict[str, Any]
    mitigation_actions: List[str] = field(default_factory=list)
    false_positive_probability: float = 0.0


@dataclass
class ModelFingerprint:
    """Fingerprint of a model for integrity checking"""
    model_id: str
    timestamp: datetime
    weight_hash: str
    weight_statistics: Dict[str, float]
    gradient_statistics: Dict[str, float]
    performance_metrics: Dict[str, float]
    layer_signatures: Dict[str, str]


class GradientMonitor:
    """Monitor gradient patterns for manipulation detection"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.gradient_history = deque(maxlen=history_size)
        self.gradient_norms = deque(maxlen=history_size)
        self.gradient_similarities = deque(maxlen=history_size)
        self.anomaly_threshold = 2.0
        self.lock = threading.Lock()
    
    def add_gradient(self, gradients: Dict[str, torch.Tensor]):
        """Add gradient information to monitoring"""
        with self.lock:
            # Calculate gradient norms
            norms = {}
            for name, grad in gradients.items():
                if grad is not None:
                    norms[name] = torch.norm(grad).item()
            
            # Calculate overall gradient norm
            total_norm = sum(norms.values())
            
            # Calculate similarity with previous gradients
            similarity = 0.0
            if len(self.gradient_history) > 0:
                last_norms = self.gradient_history[-1]
                similarity = self._calculate_cosine_similarity(norms, last_norms)
            
            # Store in history
            self.gradient_history.append(norms)
            self.gradient_norms.append(total_norm)
            self.gradient_similarities.append(similarity)
    
    def _calculate_cosine_similarity(self, grad1: Dict, grad2: Dict) -> float:
        """Calculate cosine similarity between gradient dictionaries"""
        common_keys = set(grad1.keys()) & set(grad2.keys())
        if not common_keys:
            return 0.0
        
        vec1 = np.array([grad1[k] for k in common_keys])
        vec2 = np.array([grad2[k] for k in common_keys])
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def detect_anomalies(self) -> List[Dict]:
        """Detect gradient anomalies"""
        with self.lock:
            if len(self.gradient_norms) < 10:
                return []
            
            anomalies = []
            
            # Check for gradient explosion
            recent_norms = list(self.gradient_norms)[-10:]
            norm_mean = np.mean(recent_norms)
            norm_std = np.std(recent_norms)
            
            if norm_std > 0 and recent_norms[-1] > norm_mean + self.anomaly_threshold * norm_std:
                anomalies.append({
                    'type': 'gradient_explosion',
                    'severity': 'high',
                    'value': recent_norms[-1],
                    'threshold': norm_mean + self.anomaly_threshold * norm_std
                })
            
            # Check for gradient vanishing
            if recent_norms[-1] < norm_mean - self.anomaly_threshold * norm_std:
                anomalies.append({
                    'type': 'gradient_vanishing',
                    'severity': 'medium',
                    'value': recent_norms[-1],
                    'threshold': norm_mean - self.anomaly_threshold * norm_std
                })
            
            # Check for suspicious similarity patterns
            recent_similarities = list(self.gradient_similarities)[-10:]
            if len(recent_similarities) > 5:
                avg_similarity = np.mean(recent_similarities)
                if avg_similarity > 0.95:  # Too similar
                    anomalies.append({
                        'type': 'gradient_repetition',
                        'severity': 'high',
                        'value': avg_similarity,
                        'threshold': 0.95
                    })
            
            return anomalies


class ModelIntegrityChecker:
    """Check model integrity for poisoning detection"""
    
    def __init__(self):
        self.model_fingerprints = {}
        self.baseline_performances = {}
        self.performance_threshold = 0.1  # 10% performance drop threshold
        self.lock = threading.Lock()
    
    def create_fingerprint(self, model: nn.Module, model_id: str, 
                          performance_metrics: Dict[str, float]) -> ModelFingerprint:
        """Create a fingerprint for a model"""
        with self.lock:
            # Calculate weight hash
            weight_hash = self._calculate_model_hash(model)
            
            # Calculate weight statistics
            weight_stats = self._calculate_weight_statistics(model)
            
            # Calculate gradient statistics (if available)
            gradient_stats = self._calculate_gradient_statistics(model)
            
            # Calculate layer signatures
            layer_signatures = self._calculate_layer_signatures(model)
            
            fingerprint = ModelFingerprint(
                model_id=model_id,
                timestamp=datetime.now(),
                weight_hash=weight_hash,
                weight_statistics=weight_stats,
                gradient_statistics=gradient_stats,
                performance_metrics=performance_metrics,
                layer_signatures=layer_signatures
            )
            
            self.model_fingerprints[model_id] = fingerprint
            return fingerprint
    
    def _calculate_model_hash(self, model: nn.Module) -> str:
        """Calculate hash of model weights"""
        model_bytes = b""
        for param in model.parameters():
            model_bytes += param.data.cpu().numpy().tobytes()
        return hashlib.sha256(model_bytes).hexdigest()
    
    def _calculate_weight_statistics(self, model: nn.Module) -> Dict[str, float]:
        """Calculate statistical properties of model weights"""
        all_weights = []
        layer_stats = {}
        
        for name, param in model.named_parameters():
            weights = param.data.cpu().numpy().flatten()
            all_weights.extend(weights)
            
            layer_stats[name] = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'min': float(np.min(weights)),
                'max': float(np.max(weights)),
                'l1_norm': float(np.sum(np.abs(weights))),
                'l2_norm': float(np.sqrt(np.sum(weights ** 2)))
            }
        
        all_weights = np.array(all_weights)
        return {
            'global_mean': float(np.mean(all_weights)),
            'global_std': float(np.std(all_weights)),
            'global_min': float(np.min(all_weights)),
            'global_max': float(np.max(all_weights)),
            'global_l1_norm': float(np.sum(np.abs(all_weights))),
            'global_l2_norm': float(np.sqrt(np.sum(all_weights ** 2))),
            'layer_stats': layer_stats
        }
    
    def _calculate_gradient_statistics(self, model: nn.Module) -> Dict[str, float]:
        """Calculate gradient statistics if available"""
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data.cpu().numpy().flatten()
                gradient_stats[name] = {
                    'mean': float(np.mean(grad)),
                    'std': float(np.std(grad)),
                    'norm': float(np.linalg.norm(grad))
                }
        
        return gradient_stats
    
    def _calculate_layer_signatures(self, model: nn.Module) -> Dict[str, str]:
        """Calculate unique signatures for each layer"""
        layer_signatures = {}
        
        for name, param in model.named_parameters():
            weights = param.data.cpu().numpy()
            # Create signature from weight distribution
            signature = f"{weights.shape}_{np.mean(weights):.6f}_{np.std(weights):.6f}"
            layer_signatures[name] = hashlib.md5(signature.encode()).hexdigest()
        
        return layer_signatures
    
    def check_integrity(self, model: nn.Module, model_id: str,
                       performance_metrics: Dict[str, float]) -> List[AttackSignature]:
        """Check model integrity against baseline"""
        with self.lock:
            if model_id not in self.model_fingerprints:
                # Create baseline fingerprint
                self.create_fingerprint(model, model_id, performance_metrics)
                return []
            
            baseline = self.model_fingerprints[model_id]
            current = self.create_fingerprint(model, f"{model_id}_current", performance_metrics)
            
            attacks = []
            
            # Check weight hash
            if baseline.weight_hash != current.weight_hash:
                attacks.append(AttackSignature(
                    attack_type=AttackType.MODEL_POISONING,
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.8,
                    timestamp=datetime.now(),
                    agent_id=model_id,
                    description="Model weights have been modified",
                    evidence={
                        'baseline_hash': baseline.weight_hash,
                        'current_hash': current.weight_hash,
                        'weight_diff': self._calculate_weight_difference(baseline, current)
                    },
                    mitigation_actions=['restore_model', 'investigate_source']
                ))
            
            # Check performance degradation
            performance_drop = self._calculate_performance_drop(baseline, current)
            if performance_drop > self.performance_threshold:
                attacks.append(AttackSignature(
                    attack_type=AttackType.PERFORMANCE_DEGRADATION,
                    threat_level=ThreatLevel.MEDIUM,
                    confidence=0.7,
                    timestamp=datetime.now(),
                    agent_id=model_id,
                    description=f"Performance degradation detected: {performance_drop:.2%}",
                    evidence={
                        'baseline_performance': baseline.performance_metrics,
                        'current_performance': current.performance_metrics,
                        'performance_drop': performance_drop
                    },
                    mitigation_actions=['investigate_training', 'check_data_quality']
                ))
            
            # Check for backdoor signatures
            backdoor_score = self._detect_backdoor_patterns(baseline, current)
            if backdoor_score > 0.7:
                attacks.append(AttackSignature(
                    attack_type=AttackType.BACKDOOR_ATTACK,
                    threat_level=ThreatLevel.CRITICAL,
                    confidence=backdoor_score,
                    timestamp=datetime.now(),
                    agent_id=model_id,
                    description="Potential backdoor pattern detected",
                    evidence={
                        'backdoor_score': backdoor_score,
                        'suspicious_layers': self._identify_suspicious_layers(baseline, current)
                    },
                    mitigation_actions=['isolate_model', 'forensic_analysis', 'retrain_model']
                ))
            
            return attacks
    
    def _calculate_weight_difference(self, baseline: ModelFingerprint, 
                                   current: ModelFingerprint) -> Dict[str, float]:
        """Calculate difference in weight statistics"""
        diff = {}
        
        for key in ['global_mean', 'global_std', 'global_l1_norm', 'global_l2_norm']:
            if key in baseline.weight_statistics and key in current.weight_statistics:
                diff[key] = abs(baseline.weight_statistics[key] - current.weight_statistics[key])
        
        return diff
    
    def _calculate_performance_drop(self, baseline: ModelFingerprint,
                                  current: ModelFingerprint) -> float:
        """Calculate performance drop percentage"""
        if not baseline.performance_metrics or not current.performance_metrics:
            return 0.0
        
        # Look for common performance metrics
        common_metrics = set(baseline.performance_metrics.keys()) & set(current.performance_metrics.keys())
        
        if not common_metrics:
            return 0.0
        
        drops = []
        for metric in common_metrics:
            baseline_val = baseline.performance_metrics[metric]
            current_val = current.performance_metrics[metric]
            
            if baseline_val > 0:
                drop = (baseline_val - current_val) / baseline_val
                drops.append(drop)
        
        return max(drops) if drops else 0.0
    
    def _detect_backdoor_patterns(self, baseline: ModelFingerprint,
                                 current: ModelFingerprint) -> float:
        """Detect potential backdoor patterns"""
        # This is a simplified backdoor detection
        # In practice, this would be more sophisticated
        
        suspicious_score = 0.0
        
        # Check for unusual weight patterns
        baseline_stats = baseline.weight_statistics
        current_stats = current.weight_statistics
        
        # Check for layers with dramatic changes
        if 'layer_stats' in baseline_stats and 'layer_stats' in current_stats:
            baseline_layers = baseline_stats['layer_stats']
            current_layers = current_stats['layer_stats']
            
            for layer_name in baseline_layers:
                if layer_name in current_layers:
                    baseline_layer = baseline_layers[layer_name]
                    current_layer = current_layers[layer_name]
                    
                    # Check for unusual std changes (backdoors often have specific patterns)
                    std_change = abs(baseline_layer['std'] - current_layer['std']) / baseline_layer['std']
                    if std_change > 2.0:  # More than 200% change
                        suspicious_score += 0.2
                    
                    # Check for unusual norm changes
                    norm_change = abs(baseline_layer['l2_norm'] - current_layer['l2_norm']) / baseline_layer['l2_norm']
                    if norm_change > 3.0:  # More than 300% change
                        suspicious_score += 0.3
        
        return min(suspicious_score, 1.0)
    
    def _identify_suspicious_layers(self, baseline: ModelFingerprint,
                                   current: ModelFingerprint) -> List[str]:
        """Identify layers with suspicious changes"""
        suspicious_layers = []
        
        if 'layer_stats' in baseline.weight_statistics and 'layer_stats' in current.weight_statistics:
            baseline_layers = baseline.weight_statistics['layer_stats']
            current_layers = current.weight_statistics['layer_stats']
            
            for layer_name in baseline_layers:
                if layer_name in current_layers:
                    baseline_layer = baseline_layers[layer_name]
                    current_layer = current_layers[layer_name]
                    
                    # Check for significant changes
                    changes = []
                    for stat in ['mean', 'std', 'l1_norm', 'l2_norm']:
                        if baseline_layer[stat] != 0:
                            change = abs(baseline_layer[stat] - current_layer[stat]) / abs(baseline_layer[stat])
                            changes.append(change)
                    
                    if changes and max(changes) > 1.0:  # More than 100% change
                        suspicious_layers.append(layer_name)
        
        return suspicious_layers


class ByzantineDetector:
    """Detect Byzantine attacks in multi-agent systems"""
    
    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self.agent_decisions = defaultdict(lambda: deque(maxlen=history_size))
        self.agent_performances = defaultdict(lambda: deque(maxlen=history_size))
        self.consensus_history = deque(maxlen=history_size)
        self.lock = threading.Lock()
    
    def add_agent_decision(self, agent_id: str, decision: Dict[str, Any], 
                          performance: float):
        """Add agent decision for Byzantine detection"""
        with self.lock:
            self.agent_decisions[agent_id].append({
                'decision': decision,
                'timestamp': datetime.now(),
                'performance': performance
            })
            self.agent_performances[agent_id].append(performance)
    
    def detect_byzantine_behavior(self) -> List[AttackSignature]:
        """Detect Byzantine behavior patterns"""
        with self.lock:
            if len(self.agent_decisions) < 2:
                return []
            
            attacks = []
            
            # Check for agents with consistently deviant decisions
            for agent_id in self.agent_decisions:
                if len(self.agent_decisions[agent_id]) < 10:
                    continue
                
                # Calculate deviation from consensus
                deviation_score = self._calculate_deviation_score(agent_id)
                
                if deviation_score > 0.8:
                    attacks.append(AttackSignature(
                        attack_type=AttackType.BYZANTINE_ATTACK,
                        threat_level=ThreatLevel.HIGH,
                        confidence=deviation_score,
                        timestamp=datetime.now(),
                        agent_id=agent_id,
                        description=f"Agent shows Byzantine behavior pattern",
                        evidence={
                            'deviation_score': deviation_score,
                            'recent_decisions': list(self.agent_decisions[agent_id])[-5:]
                        },
                        mitigation_actions=['isolate_agent', 'investigate_behavior']
                    ))
            
            # Check for coordinated attacks
            coordination_score = self._detect_coordination_patterns()
            if coordination_score > 0.7:
                attacks.append(AttackSignature(
                    attack_type=AttackType.BYZANTINE_ATTACK,
                    threat_level=ThreatLevel.CRITICAL,
                    confidence=coordination_score,
                    timestamp=datetime.now(),
                    agent_id='multiple_agents',
                    description="Coordinated Byzantine attack detected",
                    evidence={
                        'coordination_score': coordination_score,
                        'involved_agents': list(self.agent_decisions.keys())
                    },
                    mitigation_actions=['emergency_shutdown', 'forensic_analysis']
                ))
            
            return attacks
    
    def _calculate_deviation_score(self, agent_id: str) -> float:
        """Calculate how much an agent deviates from consensus"""
        agent_decisions = self.agent_decisions[agent_id]
        
        if len(agent_decisions) < 5:
            return 0.0
        
        # Calculate average deviation from other agents
        total_deviation = 0.0
        comparisons = 0
        
        for other_agent_id in self.agent_decisions:
            if other_agent_id == agent_id:
                continue
            
            other_decisions = self.agent_decisions[other_agent_id]
            
            # Compare recent decisions
            min_len = min(len(agent_decisions), len(other_decisions))
            for i in range(min_len):
                deviation = self._calculate_decision_distance(
                    agent_decisions[i]['decision'],
                    other_decisions[i]['decision']
                )
                total_deviation += deviation
                comparisons += 1
        
        return total_deviation / comparisons if comparisons > 0 else 0.0
    
    def _calculate_decision_distance(self, decision1: Dict, decision2: Dict) -> float:
        """Calculate distance between two decisions"""
        # Simple distance calculation - in practice this would be more sophisticated
        common_keys = set(decision1.keys()) & set(decision2.keys())
        
        if not common_keys:
            return 1.0
        
        distances = []
        for key in common_keys:
            val1 = decision1[key]
            val2 = decision2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical distance
                max_val = max(abs(val1), abs(val2), 1.0)
                distances.append(abs(val1 - val2) / max_val)
            elif val1 != val2:
                # Categorical difference
                distances.append(1.0)
            else:
                distances.append(0.0)
        
        return sum(distances) / len(distances)
    
    def _detect_coordination_patterns(self) -> float:
        """Detect coordinated attack patterns"""
        if len(self.agent_decisions) < 3:
            return 0.0
        
        # Check for unusual synchronization
        sync_score = 0.0
        
        # Check if agents are making identical decisions
        agent_ids = list(self.agent_decisions.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent1_decisions = self.agent_decisions[agent_ids[i]]
                agent2_decisions = self.agent_decisions[agent_ids[j]]
                
                # Check recent decisions for unusual similarity
                min_len = min(len(agent1_decisions), len(agent2_decisions))
                if min_len >= 5:
                    similar_count = 0
                    for k in range(min_len):
                        distance = self._calculate_decision_distance(
                            agent1_decisions[k]['decision'],
                            agent2_decisions[k]['decision']
                        )
                        if distance < 0.1:  # Very similar
                            similar_count += 1
                    
                    similarity_ratio = similar_count / min_len
                    if similarity_ratio > 0.8:  # More than 80% similar
                        sync_score += 0.3
        
        return min(sync_score, 1.0)


class AdversarialDetector:
    """
    Main adversarial detection engine that coordinates all detection components
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or EventBus()
        self.gradient_monitor = GradientMonitor()
        self.integrity_checker = ModelIntegrityChecker()
        self.byzantine_detector = ByzantineDetector()
        
        # Detection thresholds
        self.thresholds = {
            'gradient_anomaly': 0.7,
            'model_integrity': 0.6,
            'byzantine_behavior': 0.8,
            'performance_drop': 0.15
        }
        
        # Detection history
        self.detection_history = deque(maxlen=1000)
        self.active_attacks = {}
        
        # Anomaly detection models
        self.anomaly_models = {
            'gradient': IsolationForest(contamination=0.1, random_state=42),
            'performance': IsolationForest(contamination=0.1, random_state=42),
            'decision': IsolationForest(contamination=0.1, random_state=42)
        }
        
        self.scalers = {
            'gradient': StandardScaler(),
            'performance': StandardScaler(),
            'decision': StandardScaler()
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('adversarial_detector.log'),
                logging.StreamHandler()
            ]
        )
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Adversarial detection monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Adversarial detection monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Perform periodic checks
                await self._perform_periodic_checks()
                
                # Update anomaly models
                await self._update_anomaly_models()
                
                # Clean up old detections
                self._cleanup_old_detections()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _perform_periodic_checks(self):
        """Perform periodic adversarial checks"""
        # Check gradient anomalies
        gradient_anomalies = self.gradient_monitor.detect_anomalies()
        for anomaly in gradient_anomalies:
            await self._handle_gradient_anomaly(anomaly)
        
        # Check for Byzantine behavior
        byzantine_attacks = self.byzantine_detector.detect_byzantine_behavior()
        for attack in byzantine_attacks:
            await self._handle_attack_detection(attack)
    
    async def _handle_gradient_anomaly(self, anomaly: Dict):
        """Handle detected gradient anomaly"""
        attack_signature = AttackSignature(
            attack_type=AttackType.GRADIENT_MANIPULATION,
            threat_level=ThreatLevel.HIGH if anomaly['severity'] == 'high' else ThreatLevel.MEDIUM,
            confidence=0.8,
            timestamp=datetime.now(),
            agent_id='gradient_monitor',
            description=f"Gradient anomaly detected: {anomaly['type']}",
            evidence=anomaly,
            mitigation_actions=['investigate_training', 'check_optimizer', 'monitor_gradients']
        )
        
        await self._handle_attack_detection(attack_signature)
    
    async def _handle_attack_detection(self, attack: AttackSignature):
        """Handle detected attack"""
        # Store in history
        self.detection_history.append(attack)
        
        # Add to active attacks
        self.active_attacks[f"{attack.agent_id}_{attack.attack_type.value}"] = attack
        
        # Emit event
        await self.event_bus.emit(Event(
            type="adversarial_attack_detected",
            data={
                "attack_type": attack.attack_type.value,
                "threat_level": attack.threat_level.value,
                "confidence": attack.confidence,
                "agent_id": attack.agent_id,
                "description": attack.description,
                "evidence": attack.evidence,
                "mitigation_actions": attack.mitigation_actions,
                "timestamp": attack.timestamp.isoformat()
            }
        ))
        
        # Log detection
        self.logger.warning(
            f"ATTACK DETECTED: {attack.attack_type.value} on {attack.agent_id} "
            f"(confidence: {attack.confidence:.2f}, threat: {attack.threat_level.value})"
        )
        
        # Trigger mitigation if threat level is high
        if attack.threat_level.value >= ThreatLevel.HIGH.value:
            await self._trigger_mitigation(attack)
    
    async def _trigger_mitigation(self, attack: AttackSignature):
        """Trigger mitigation actions"""
        self.logger.critical(f"Triggering mitigation for {attack.attack_type.value}")
        
        # Execute mitigation actions
        for action in attack.mitigation_actions:
            await self._execute_mitigation_action(action, attack)
    
    async def _execute_mitigation_action(self, action: str, attack: AttackSignature):
        """Execute a specific mitigation action"""
        self.logger.info(f"Executing mitigation action: {action}")
        
        # Emit mitigation event
        await self.event_bus.emit(Event(
            type="mitigation_action_executed",
            data={
                "action": action,
                "attack_type": attack.attack_type.value,
                "agent_id": attack.agent_id,
                "timestamp": datetime.now().isoformat()
            }
        ))
    
    async def _update_anomaly_models(self):
        """Update anomaly detection models with new data"""
        # This is a simplified update - in practice you'd retrain models
        pass
    
    def _cleanup_old_detections(self):
        """Clean up old attack detections"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove old active attacks
        to_remove = []
        for key, attack in self.active_attacks.items():
            if attack.timestamp < cutoff_time:
                to_remove.append(key)
        
        for key in to_remove:
            del self.active_attacks[key]
    
    async def analyze_model(self, model: nn.Module, model_id: str,
                          performance_metrics: Dict[str, float]) -> List[AttackSignature]:
        """Analyze a model for adversarial attacks"""
        attacks = []
        
        # Check model integrity
        integrity_attacks = self.integrity_checker.check_integrity(
            model, model_id, performance_metrics
        )
        attacks.extend(integrity_attacks)
        
        # Handle detected attacks
        for attack in integrity_attacks:
            await self._handle_attack_detection(attack)
        
        return attacks
    
    async def analyze_gradients(self, gradients: Dict[str, torch.Tensor],
                               agent_id: str) -> List[AttackSignature]:
        """Analyze gradients for manipulation"""
        # Add to monitoring
        self.gradient_monitor.add_gradient(gradients)
        
        # Detect anomalies
        anomalies = self.gradient_monitor.detect_anomalies()
        
        attacks = []
        for anomaly in anomalies:
            attack = AttackSignature(
                attack_type=AttackType.GRADIENT_MANIPULATION,
                threat_level=ThreatLevel.HIGH if anomaly['severity'] == 'high' else ThreatLevel.MEDIUM,
                confidence=0.8,
                timestamp=datetime.now(),
                agent_id=agent_id,
                description=f"Gradient anomaly: {anomaly['type']}",
                evidence=anomaly,
                mitigation_actions=['investigate_training', 'check_optimizer']
            )
            attacks.append(attack)
            await self._handle_attack_detection(attack)
        
        return attacks
    
    async def analyze_agent_decisions(self, agent_id: str, decision: Dict[str, Any],
                                    performance: float) -> List[AttackSignature]:
        """Analyze agent decisions for Byzantine behavior"""
        # Add to Byzantine detector
        self.byzantine_detector.add_agent_decision(agent_id, decision, performance)
        
        # Detect Byzantine behavior
        byzantine_attacks = self.byzantine_detector.detect_byzantine_behavior()
        
        # Handle detected attacks
        for attack in byzantine_attacks:
            await self._handle_attack_detection(attack)
        
        return byzantine_attacks
    
    def get_detection_summary(self) -> Dict:
        """Get summary of recent detections"""
        now = datetime.now()
        recent_attacks = [
            attack for attack in self.detection_history
            if (now - attack.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Count by type
        attack_counts = defaultdict(int)
        threat_levels = defaultdict(int)
        
        for attack in recent_attacks:
            attack_counts[attack.attack_type.value] += 1
            threat_levels[attack.threat_level.value] += 1
        
        return {
            'total_recent_attacks': len(recent_attacks),
            'active_attacks': len(self.active_attacks),
            'attack_types': dict(attack_counts),
            'threat_levels': dict(threat_levels),
            'detection_history_size': len(self.detection_history),
            'monitoring_active': self.monitoring_active
        }
    
    def get_active_attacks(self) -> List[AttackSignature]:
        """Get list of currently active attacks"""
        return list(self.active_attacks.values())
    
    async def reset_detection_state(self):
        """Reset detection state (for testing)"""
        self.detection_history.clear()
        self.active_attacks.clear()
        self.gradient_monitor.gradient_history.clear()
        self.gradient_monitor.gradient_norms.clear()
        self.gradient_monitor.gradient_similarities.clear()
        
        self.logger.info("Detection state reset")


# Example usage and testing
async def demo_adversarial_detector():
    """Demonstration of adversarial detector"""
    detector = AdversarialDetector()
    
    # Start monitoring
    await detector.start_monitoring()
    
    # Create fake model for testing
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Analyze model
    attacks = await detector.analyze_model(model, "demo_model", {"accuracy": 0.95})
    print(f"Model analysis found {len(attacks)} attacks")
    
    # Simulate gradient analysis
    fake_gradients = {
        "layer1.weight": torch.randn(20, 10),
        "layer1.bias": torch.randn(20),
        "layer2.weight": torch.randn(5, 20),
        "layer2.bias": torch.randn(5)
    }
    
    gradient_attacks = await detector.analyze_gradients(fake_gradients, "demo_agent")
    print(f"Gradient analysis found {len(gradient_attacks)} attacks")
    
    # Simulate agent decision analysis
    decision_attacks = await detector.analyze_agent_decisions(
        "demo_agent", 
        {"action": "buy", "amount": 100},
        0.8
    )
    print(f"Decision analysis found {len(decision_attacks)} attacks")
    
    # Wait a bit for monitoring
    await asyncio.sleep(10)
    
    # Get summary
    summary = detector.get_detection_summary()
    print("\n=== ADVERSARIAL DETECTOR SUMMARY ===")
    print(json.dumps(summary, indent=2))
    
    # Stop monitoring
    await detector.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(demo_adversarial_detector())