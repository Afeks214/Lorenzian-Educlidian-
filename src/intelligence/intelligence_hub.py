"""
Intelligence Hub - Central coordination for all AI intelligence upgrades.

Integrates dynamic feature selection, intelligent gating network, and regime-aware
rewards into a cohesive system optimized for <5ms inference performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
import psutil

from .regime_detector import RegimeDetector, RegimeAnalysis
from .gating_network import GatingNetwork
from .regime_aware_reward import RegimeAwareRewardFunction
from .attention_optimizer import AttentionOptimizer

@dataclass
class IntelligenceMetrics:
    """Performance metrics for intelligence components."""
    attention_time_ms: float
    gating_time_ms: float
    regime_detection_time_ms: float
    total_intelligence_overhead_ms: float
    memory_usage_mb: float
    decisions_per_second: float

class IntelligenceHub:
    """
    Central coordination hub for all intelligence upgrades.
    
    Manages the integration of:
    - Dynamic feature selection (attention mechanisms)
    - Intelligent gating network (expert coordination)
    - Regime-aware reward function (contextual learning)
    
    Optimized for <5ms total inference while providing adaptive intelligence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance targets
        self.max_intelligence_overhead_ms = config.get('max_intelligence_overhead_ms', 1.0)
        self.performance_monitoring = config.get('performance_monitoring', True)
        
        # Initialize intelligence components with performance optimization
        self._initialize_optimized_components()
        
        # Performance tracking
        self.performance_metrics = IntelligenceMetrics(
            attention_time_ms=0.0,
            gating_time_ms=0.0,
            regime_detection_time_ms=0.0,
            total_intelligence_overhead_ms=0.0,
            memory_usage_mb=0.0,
            decisions_per_second=0.0
        )
        
        # JIT compilation for performance
        self._compile_for_performance()
        
        # Caching for repeated computations
        self._setup_intelligent_caching()
        
        # Integration state tracking
        self.integration_stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_activations': 0
        }
        
        self.logger.info("Intelligence Hub initialized successfully")
        
    def _initialize_optimized_components(self):
        """Initialize all intelligence components with performance optimizations."""
        
        # Regime detector with optimized thresholds
        regime_config = self.config.get('regime_detection', {})
        regime_config.update({
            'fast_mode': True,              # Enable fast detection mode
            'cache_analysis': True,         # Cache regime analysis
            'min_confidence_threshold': 0.7 # Skip low-confidence detailed analysis
        })
        self.regime_detector = RegimeDetector(regime_config)
        
        # Gating network with reduced complexity for speed
        gating_config = self.config.get('gating_network', {})
        self.gating_network = GatingNetwork(
            shared_context_dim=6,           # Optimized context dimension
            n_agents=3,
            hidden_dim=32                   # Reduced from 64 for speed
        )
        
        # Regime-aware reward function
        reward_config = self.config.get('regime_aware_reward', {})
        self.regime_reward_function = RegimeAwareRewardFunction(reward_config)
        
        # Attention mechanism optimizer
        attention_config = self.config.get('attention', {})
        self.attention_optimizer = AttentionOptimizer(attention_config)
        
        self.logger.info("Intelligence components initialized with performance optimizations")
    
    def _compile_for_performance(self):
        """Apply JIT compilation and other performance optimizations."""
        try:
            # JIT compile the gating network for faster inference
            example_context = torch.randn(1, 6)
            self.gating_network_jit = torch.jit.trace(self.gating_network, example_context)
            
            # Optimize attention computations
            self._optimize_attention_kernels()
            
            self.logger.info("JIT compilation completed successfully")
            
        except Exception as e:
            self.logger.warning(f"JIT compilation failed, using eager mode: {e}")
            self.gating_network_jit = self.gating_network
    
    def _optimize_attention_kernels(self):
        """Pre-optimize attention computation kernels."""
        try:
            # Pre-warm attention optimizer with common patterns
            common_patterns = [
                torch.randn(1, 7),  # Single sample
                torch.randn(4, 7),  # Small batch
            ]
            
            for pattern in common_patterns:
                _ = self.attention_optimizer.optimize_attention_computation(pattern)
            
            self.logger.info("Attention kernels optimized")
            
        except Exception as e:
            self.logger.warning(f"Attention kernel optimization failed: {e}")
    
    def _setup_intelligent_caching(self):
        """Set up intelligent caching for repeated computations."""
        self.regime_cache = {}
        self.context_cache = {}
        self.gating_cache = {}
        
        # Cache size limits
        self.max_cache_size = 1000
        self.cache_ttl_seconds = 30  # 30 second TTL for cached results
        
        self.logger.info("Intelligent caching system initialized")
    
    def process_intelligence_pipeline(
        self, 
        market_context: Dict[str, Any],
        agent_predictions: List[Dict[str, Any]],
        agent_attention_weights: Optional[List[torch.Tensor]] = None
    ) -> Tuple[Dict[str, Any], IntelligenceMetrics]:
        """
        Process the complete intelligence pipeline with performance monitoring.
        
        Args:
            market_context: Current market context
            agent_predictions: Predictions from all three agents
            agent_attention_weights: Optional attention weights from each agent
            
        Returns:
            Tuple of (intelligence_result, performance_metrics)
        """
        start_time = time.perf_counter()
        self.integration_stats['total_operations'] += 1
        
        try:
            # Step 1: Regime Detection (Target: <0.2ms)
            regime_start = time.perf_counter()
            regime_analysis = self._fast_regime_detection(market_context)
            regime_time = (time.perf_counter() - regime_start) * 1000
            
            # Step 2: Dynamic Gating (Target: <0.3ms)
            gating_start = time.perf_counter()
            gating_weights = self._fast_gating_computation(market_context, regime_analysis)
            gating_time = (time.perf_counter() - gating_start) * 1000
            
            # Step 3: Attention Analysis (Target: <0.2ms)
            attention_start = time.perf_counter()
            attention_analysis = self._analyze_attention_patterns(agent_attention_weights, regime_analysis)
            attention_time = (time.perf_counter() - attention_start) * 1000
            
            # Step 4: Integrated Decision (Target: <0.3ms)
            integration_start = time.perf_counter()
            integrated_result = self._integrate_intelligence_components(
                regime_analysis, gating_weights, attention_analysis, agent_predictions
            )
            integration_time = (time.perf_counter() - integration_start) * 1000
            
            # Calculate total intelligence overhead
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Update performance metrics
            self.performance_metrics = IntelligenceMetrics(
                attention_time_ms=attention_time,
                gating_time_ms=gating_time,
                regime_detection_time_ms=regime_time,
                total_intelligence_overhead_ms=total_time,
                memory_usage_mb=self._get_memory_usage(),
                decisions_per_second=1000.0 / total_time if total_time > 0 else 0.0
            )
            
            # Performance validation
            if total_time > self.max_intelligence_overhead_ms:
                self.logger.warning(
                    f"Intelligence overhead {total_time:.3f}ms exceeds target "
                    f"{self.max_intelligence_overhead_ms}ms"
                )
            
            # Add performance data to result
            integrated_result['intelligence_metrics'] = self.performance_metrics
            integrated_result['regime_analysis'] = regime_analysis
            
            return integrated_result, self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error in intelligence pipeline: {e}")
            self.integration_stats['fallback_activations'] += 1
            
            # Return fallback result
            fallback_result = self._create_fallback_result(agent_predictions)
            return fallback_result, self.performance_metrics
    
    def _fast_regime_detection(self, market_context: Dict[str, Any]) -> RegimeAnalysis:
        """Optimized regime detection with caching."""
        
        # Create cache key from context
        context_key = self._create_context_cache_key(market_context)
        
        # Check cache first
        if context_key in self.regime_cache:
            cached_result, timestamp = self.regime_cache[context_key]
            if time.time() - timestamp < self.cache_ttl_seconds:
                self.integration_stats['cache_hits'] += 1
                return cached_result
        
        # Perform regime detection
        self.integration_stats['cache_misses'] += 1
        regime_analysis = self.regime_detector.detect_regime(market_context)
        
        # Cache result
        self.regime_cache[context_key] = (regime_analysis, time.time())
        
        # Maintain cache size
        if len(self.regime_cache) > self.max_cache_size:
            self._clean_cache(self.regime_cache)
        
        return regime_analysis
    
    def _fast_gating_computation(
        self, 
        market_context: Dict[str, Any], 
        regime_analysis: RegimeAnalysis
    ) -> torch.Tensor:
        """Optimized gating weight computation."""
        
        # Extract shared context tensor efficiently
        context_tensor = self._extract_shared_context_optimized(market_context)
        
        # Use JIT-compiled gating network
        with torch.no_grad():
            gating_result = self.gating_network_jit(context_tensor)
            
        return gating_result['gating_weights'] if isinstance(gating_result, dict) else gating_result
    
    def _analyze_attention_patterns(
        self, 
        attention_weights: Optional[List[torch.Tensor]],
        regime_analysis: RegimeAnalysis
    ) -> Dict[str, Any]:
        """Fast analysis of attention patterns across agents."""
        
        if attention_weights is None or not attention_weights:
            # Return default attention analysis
            return {
                'attention_entropy': [1.0, 1.0, 1.0],
                'attention_focus': [0.33, 0.33, 0.33],
                'regime_alignment': 0.5,
                'agent_specialization': {},
                'attention_available': False
            }
        
        attention_analysis = {
            'attention_entropy': [],
            'attention_focus': [],
            'regime_alignment': 0.0,
            'agent_specialization': {},
            'attention_available': True
        }
        
        agent_names = ['MLMI', 'NWRQK', 'Regime']
        
        for i, weights in enumerate(attention_weights):
            if weights is not None and weights.numel() > 0:
                # Ensure weights are properly normalized
                if weights.sum() > 0:
                    normalized_weights = weights / weights.sum()
                else:
                    normalized_weights = torch.ones_like(weights) / weights.numel()
                
                # Calculate attention entropy (focus measure)
                entropy = -torch.sum(normalized_weights * torch.log(normalized_weights + 1e-8)).item()
                attention_analysis['attention_entropy'].append(entropy)
                
                # Calculate attention focus (max weight)
                focus = torch.max(normalized_weights).item()
                attention_analysis['attention_focus'].append(focus)
                
                # Determine agent specialization
                max_idx = torch.argmax(normalized_weights).item()
                agent_name = agent_names[i] if i < len(agent_names) else f'agent_{i}'
                attention_analysis['agent_specialization'][agent_name] = {
                    'primary_feature': max_idx,
                    'focus_strength': focus,
                    'weights': normalized_weights.tolist() if len(normalized_weights) <= 10 else None
                }
            else:
                # Default values for missing attention
                attention_analysis['attention_entropy'].append(1.0)
                attention_analysis['attention_focus'].append(0.33)
        
        # Calculate regime alignment score
        attention_analysis['regime_alignment'] = self._calculate_regime_alignment(
            attention_weights, regime_analysis
        )
        
        return attention_analysis
    
    def _integrate_intelligence_components(
        self,
        regime_analysis: RegimeAnalysis,
        gating_weights: torch.Tensor,
        attention_analysis: Dict[str, Any],
        agent_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Integrate all intelligence components into final decision."""
        
        # Convert gating weights to numpy for processing
        if isinstance(gating_weights, torch.Tensor):
            gating_weights_np = gating_weights.detach().cpu().numpy().flatten()
        else:
            gating_weights_np = np.array(gating_weights).flatten()
        
        # Ensure we have the right number of weights
        n_agents = len(agent_predictions)
        if len(gating_weights_np) < n_agents:
            # Pad with uniform weights
            padding = np.ones(n_agents - len(gating_weights_np)) / n_agents
            gating_weights_np = np.concatenate([gating_weights_np, padding])
        elif len(gating_weights_np) > n_agents:
            # Truncate
            gating_weights_np = gating_weights_np[:n_agents]
        
        # Weighted combination of agent predictions using dynamic gating
        final_probabilities = np.zeros(3)  # [buy, hold, sell]
        total_weight = 0.0
        
        for i, prediction in enumerate(agent_predictions):
            if i < len(gating_weights_np):
                # Extract prediction probabilities
                if 'action_probabilities' in prediction:
                    probs = np.array(prediction['action_probabilities'])
                elif 'probabilities' in prediction:
                    probs = np.array(prediction['probabilities'])
                else:
                    # Default uniform probabilities
                    probs = np.array([0.33, 0.34, 0.33])
                
                # Ensure probabilities are the right shape
                if len(probs) != 3:
                    probs = np.array([0.33, 0.34, 0.33])
                
                # Apply gating weight and agent confidence
                agent_confidence = prediction.get('confidence', 1.0)
                weight = gating_weights_np[i] * agent_confidence
                
                final_probabilities += weight * probs
                total_weight += weight
        
        # Normalize probabilities
        if total_weight > 0:
            final_probabilities /= total_weight
        else:
            final_probabilities = np.array([0.33, 0.34, 0.33])
        
        # Ensure probabilities are valid
        final_probabilities = np.clip(final_probabilities, 0.01, 0.99)
        final_probabilities /= final_probabilities.sum()
        
        # Calculate overall confidence
        regime_confidence = regime_analysis.confidence
        gating_confidence = self._calculate_gating_confidence(gating_weights_np)
        attention_confidence = np.mean(attention_analysis.get('attention_focus', [0.5]))
        
        overall_confidence = (regime_confidence + gating_confidence + attention_confidence) / 3.0
        overall_confidence = np.clip(overall_confidence, 0.1, 0.95)
        
        return {
            'final_probabilities': final_probabilities.tolist(),
            'overall_confidence': float(overall_confidence),
            'regime': regime_analysis.regime.value,
            'regime_confidence': float(regime_confidence),
            'gating_weights': gating_weights_np.tolist(),
            'attention_analysis': attention_analysis,
            'intelligence_active': True,
            'integration_stats': self.integration_stats.copy()
        }
    
    def _calculate_regime_alignment(
        self, 
        attention_weights: List[torch.Tensor], 
        regime_analysis: RegimeAnalysis
    ) -> float:
        """Calculate how well attention patterns align with current regime."""
        
        try:
            if not attention_weights or regime_analysis is None:
                return 0.5
            
            # Simple alignment score based on attention focus patterns
            total_alignment = 0.0
            valid_weights = 0
            
            for weights in attention_weights:
                if weights is not None and weights.numel() > 0:
                    # Check if attention is focused (good for trending regimes)
                    # or distributed (good for sideways regimes)
                    focus = torch.max(weights).item()
                    
                    if regime_analysis.regime.value in ['bull_trend', 'bear_trend']:
                        # Trending regimes benefit from focused attention
                        alignment = focus
                    elif regime_analysis.regime.value == 'sideways':
                        # Sideways regime benefits from distributed attention
                        alignment = 1.0 - focus
                    else:  # crisis, recovery
                        # Crisis/recovery benefit from moderate focus
                        alignment = 1.0 - abs(focus - 0.5)
                    
                    total_alignment += alignment
                    valid_weights += 1
            
            if valid_weights > 0:
                return total_alignment / valid_weights
            else:
                return 0.5
                
        except Exception as e:
            self.logger.error(f"Error calculating regime alignment: {e}")
            return 0.5
    
    def _calculate_gating_confidence(self, gating_weights: np.ndarray) -> float:
        """Calculate confidence score from gating weights distribution."""
        
        try:
            if len(gating_weights) == 0:
                return 0.5
            
            # Higher confidence when weights are more decisive (less uniform)
            uniformity = np.std(gating_weights) / np.mean(gating_weights) if np.mean(gating_weights) > 0 else 0
            confidence = min(1.0, uniformity)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.error(f"Error calculating gating confidence: {e}")
            return 0.5
    
    def _extract_shared_context_optimized(self, market_context: Dict[str, Any]) -> torch.Tensor:
        """Extract shared context tensor efficiently."""
        
        # Extract key features for gating network (6 dimensions)
        context_values = [
            market_context.get('volatility_30', 0.5),
            market_context.get('mmd_score', 0.3),
            market_context.get('momentum_20', 0.0),
            market_context.get('momentum_50', 0.0),
            market_context.get('volume_ratio', 1.0),
            market_context.get('price_trend', 0.0)
        ]
        
        # Convert to tensor
        context_tensor = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)
        
        return context_tensor
    
    def _create_context_cache_key(self, market_context: Dict[str, Any]) -> str:
        """Create cache key from market context."""
        key_values = [
            round(market_context.get('volatility_30', 0), 3),
            round(market_context.get('mmd_score', 0), 3),
            round(market_context.get('momentum_20', 0), 4),
            round(market_context.get('momentum_50', 0), 4),
            round(market_context.get('volume_ratio', 1), 2),
            round(market_context.get('price_trend', 0), 4)
        ]
        return f"context_{hash(tuple(key_values))}"
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _clean_cache(self, cache_dict: Dict):
        """Clean oldest entries from cache."""
        if len(cache_dict) > self.max_cache_size:
            # Remove oldest 10% of entries
            items_to_remove = len(cache_dict) // 10
            oldest_keys = sorted(cache_dict.keys(), 
                               key=lambda k: cache_dict[k][1] if isinstance(cache_dict[k], tuple) else 0)[:items_to_remove]
            
            for key in oldest_keys:
                del cache_dict[key]
    
    def _create_fallback_result(self, agent_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create fallback result when intelligence pipeline fails."""
        
        # Simple average of agent predictions
        if agent_predictions:
            avg_probs = np.mean([
                p.get('action_probabilities', [0.33, 0.34, 0.33]) 
                for p in agent_predictions
            ], axis=0)
            avg_confidence = np.mean([
                p.get('confidence', 0.5) 
                for p in agent_predictions
            ])
        else:
            avg_probs = np.array([0.33, 0.34, 0.33])
            avg_confidence = 0.5
        
        return {
            'final_probabilities': avg_probs.tolist(),
            'overall_confidence': float(avg_confidence),
            'regime': 'sideways',
            'regime_confidence': 0.5,
            'gating_weights': [1.0/len(agent_predictions) if agent_predictions else 1.0] * (len(agent_predictions) or 1),
            'attention_analysis': {'attention_available': False},
            'intelligence_active': False,
            'fallback_mode': True
        }
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        
        return {
            'performance_metrics': {
                'mean_attention_time_ms': self.performance_metrics.attention_time_ms,
                'mean_gating_time_ms': self.performance_metrics.gating_time_ms,
                'mean_regime_detection_ms': self.performance_metrics.regime_detection_time_ms,
                'mean_total_overhead_ms': self.performance_metrics.total_intelligence_overhead_ms,
                'current_memory_mb': self.performance_metrics.memory_usage_mb,
                'decisions_per_second': self.performance_metrics.decisions_per_second
            },
            'integration_stats': self.integration_stats,
            'cache_stats': {
                'regime_cache_size': len(self.regime_cache),
                'context_cache_size': len(self.context_cache),
                'gating_cache_size': len(self.gating_cache),
                'cache_hit_rate': (
                    self.integration_stats['cache_hits'] / 
                    max(1, self.integration_stats['cache_hits'] + self.integration_stats['cache_misses'])
                )
            },
            'component_status': {
                'regime_detector': 'active',
                'gating_network': 'active',
                'attention_optimizer': 'active',
                'reward_function': 'active'
            }
        }
    
    def reset_intelligence_state(self):
        """Reset intelligence hub state for clean restart."""
        
        # Clear caches
        self.regime_cache.clear()
        self.context_cache.clear()
        self.gating_cache.clear()
        
        # Reset component states
        self.regime_detector.reset_cache()
        self.attention_optimizer.reset_tracking()
        self.regime_reward_function.reset_tracking()
        
        # Reset integration stats
        self.integration_stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'fallback_activations': 0
        }
        
        self.logger.info("Intelligence Hub state reset completed")
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update intelligence hub configuration dynamically."""
        
        try:
            # Update performance targets
            if 'max_intelligence_overhead_ms' in new_config:
                self.max_intelligence_overhead_ms = new_config['max_intelligence_overhead_ms']
            
            # Update component configurations
            if 'regime_detection' in new_config:
                # Regime detector config updates
                pass  # Implementation would update regime detector settings
            
            if 'gating_network' in new_config:
                # Gating network config updates
                pass  # Implementation would update gating network settings
            
            self.config.update(new_config)
            self.logger.info("Intelligence Hub configuration updated")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")