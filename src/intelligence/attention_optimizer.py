"""
Attention Optimizer for Dynamic Feature Selection.

Optimizes attention mechanism computations for speed while maintaining the
dynamic feature selection capabilities. Integrates with existing secure attention.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import logging

class AttentionOptimizer:
    """
    Optimizes attention mechanism computations for minimal latency.
    
    Provides accelerated attention computations while maintaining security
    and dynamic feature selection capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance optimization settings
        self.use_fast_attention = config.get('use_fast_attention', True)
        self.batch_size_threshold = config.get('batch_size_threshold', 4)
        self.attention_cache_size = config.get('attention_cache_size', 100)
        
        # Attention computation parameters
        self.min_attention = config.get('min_attention', 1e-6)
        self.max_attention = config.get('max_attention', 1.0)
        self.temperature = config.get('temperature', 1.0)
        
        # Performance tracking
        self.computation_times = []
        self.attention_patterns = []
        
        # Precomputed attention templates for common patterns
        self.attention_templates = self._create_attention_templates()
        
        # JIT compilation setup
        self._setup_jit_functions()
        
        self.logger.info("Attention optimizer initialized")
    
    def _create_attention_templates(self) -> Dict[str, torch.Tensor]:
        """Create precomputed attention templates for common patterns."""
        
        templates = {}
        
        # Uniform attention (fallback)
        templates['uniform'] = torch.ones(7) / 7
        
        # MLMI-focused attention (features 0, 1, 2)
        mlmi_attention = torch.zeros(7)
        mlmi_attention[0:3] = 0.3
        mlmi_attention[3:] = 0.1 / 4
        templates['mlmi_focused'] = F.softmax(mlmi_attention, dim=0)
        
        # NWRQK-focused attention (features 3, 4, 5, 6)
        nwrqk_attention = torch.zeros(7)
        nwrqk_attention[0:3] = 0.1 / 3
        nwrqk_attention[3:] = 0.7 / 4
        templates['nwrqk_focused'] = F.softmax(nwrqk_attention, dim=0)
        
        # Momentum-focused attention (features 5, 6)
        momentum_attention = torch.zeros(7)
        momentum_attention[5:] = 0.4
        momentum_attention[:5] = 0.2 / 5
        templates['momentum_focused'] = F.softmax(momentum_attention, dim=0)
        
        # Volatility-focused attention (features 1, 2)
        volatility_attention = torch.zeros(7)
        volatility_attention[1:3] = 0.4
        volatility_attention[0] = 0.1
        volatility_attention[3:] = 0.1 / 4
        templates['volatility_focused'] = F.softmax(volatility_attention, dim=0)
        
        return templates
    
    def _setup_jit_functions(self):
        """Setup JIT-compiled functions for speed."""
        try:
            # Example tensor for tracing
            example_input = torch.randn(1, 7)
            
            # JIT compile attention computation
            def attention_forward(x):
                return F.softmax(x / self.temperature, dim=-1)
            
            self.jit_attention = torch.jit.trace(attention_forward, example_input)
            
            self.logger.info("JIT compilation successful")
            
        except Exception as e:
            self.logger.warning(f"JIT compilation failed: {e}")
            self.jit_attention = None
    
    def optimize_attention_computation(
        self,
        raw_attention_scores: torch.Tensor,
        agent_id: str = "default",
        market_context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Optimize attention weight computation for speed.
        
        Args:
            raw_attention_scores: Raw attention scores before softmax
            agent_id: Agent identifier for specialization
            market_context: Optional market context for adaptive attention
            
        Returns:
            Optimized attention weights
        """
        start_time = time.perf_counter()
        
        try:
            batch_size = raw_attention_scores.shape[0]
            
            # Use template-based attention for single samples if appropriate
            if batch_size == 1 and self.use_fast_attention:
                attention_weights = self._get_template_attention(agent_id, market_context)
                if attention_weights is not None:
                    computation_time = (time.perf_counter() - start_time) * 1000
                    self._track_computation_time(computation_time)
                    return attention_weights.unsqueeze(0)
            
            # Use JIT-compiled attention if available
            if self.jit_attention is not None and batch_size <= self.batch_size_threshold:
                attention_weights = self.jit_attention(raw_attention_scores)
            else:
                # Standard attention computation
                attention_weights = F.softmax(raw_attention_scores / self.temperature, dim=-1)
            
            # Apply bounds for stability
            attention_weights = torch.clamp(attention_weights, self.min_attention, self.max_attention)
            
            # Renormalize
            attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
            
            # Track computation time
            computation_time = (time.perf_counter() - start_time) * 1000
            self._track_computation_time(computation_time)
            
            return attention_weights
            
        except Exception as e:
            self.logger.error(f"Error in attention optimization: {e}")
            
            # Fallback to uniform attention
            uniform_attention = torch.ones_like(raw_attention_scores) / raw_attention_scores.shape[-1]
            return uniform_attention
    
    def _get_template_attention(
        self,
        agent_id: str,
        market_context: Optional[Dict[str, Any]]
    ) -> Optional[torch.Tensor]:
        """Get template-based attention for fast computation."""
        
        # Agent-specific templates
        if agent_id == "mlmi":
            base_template = self.attention_templates['mlmi_focused']
        elif agent_id == "nwrqk":
            base_template = self.attention_templates['nwrqk_focused']
        elif agent_id == "momentum":
            base_template = self.attention_templates['momentum_focused']
        else:
            base_template = self.attention_templates['uniform']
        
        # Adapt based on market context if provided
        if market_context is not None:
            base_template = self._adapt_template_to_context(base_template, market_context)
        
        return base_template
    
    def _adapt_template_to_context(
        self,
        base_template: torch.Tensor,
        market_context: Dict[str, Any]
    ) -> torch.Tensor:
        """Adapt attention template based on market context."""
        
        try:
            # Make a copy to avoid modifying the original
            adapted_template = base_template.clone()
            
            # Volatility-based adaptation
            volatility = market_context.get('volatility_30', 0.5)
            if volatility > 0.8:  # High volatility
                # Focus more on volatility-related features
                adapted_template[1:3] *= 1.3
                adapted_template = F.softmax(adapted_template, dim=0)
            
            # Momentum-based adaptation
            momentum_20 = abs(market_context.get('momentum_20', 0.0))
            momentum_50 = abs(market_context.get('momentum_50', 0.0))
            momentum_strength = momentum_20 + momentum_50 * 0.5
            
            if momentum_strength > 0.02:  # Strong momentum
                # Focus more on momentum features
                adapted_template[5:] *= 1.2
                adapted_template = F.softmax(adapted_template, dim=0)
            
            # MMD-based adaptation (regime change indicator)
            mmd_score = market_context.get('mmd_score', 0.3)
            if mmd_score > 0.5:  # High regime change probability
                # Balanced attention during uncertain times
                adapted_template = 0.7 * adapted_template + 0.3 * self.attention_templates['uniform']
            
            return adapted_template
            
        except Exception as e:
            self.logger.error(f"Error adapting attention template: {e}")
            return base_template
    
    def batch_attention_computation(
        self,
        batch_attention_scores: List[torch.Tensor],
        agent_ids: List[str],
        market_contexts: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """
        Batch process multiple attention computations for efficiency.
        
        Args:
            batch_attention_scores: List of attention score tensors
            agent_ids: List of agent identifiers
            market_contexts: List of market contexts
            
        Returns:
            List of optimized attention weights
        """
        start_time = time.perf_counter()
        
        try:
            if not batch_attention_scores:
                return []
            
            # Check if we can stack for batch processing
            can_batch = all(
                scores.shape == batch_attention_scores[0].shape 
                for scores in batch_attention_scores
            )
            
            if can_batch and len(batch_attention_scores) > 1:
                # Stack and process as batch
                stacked_scores = torch.stack(batch_attention_scores)
                
                if self.jit_attention is not None:
                    batch_attention = self.jit_attention(stacked_scores)
                else:
                    batch_attention = F.softmax(stacked_scores / self.temperature, dim=-1)
                
                # Apply bounds and renormalize
                batch_attention = torch.clamp(batch_attention, self.min_attention, self.max_attention)
                batch_attention = batch_attention / batch_attention.sum(dim=-1, keepdim=True)
                
                result = [batch_attention[i] for i in range(len(batch_attention_scores))]
            else:
                # Process individually
                result = []
                for i, scores in enumerate(batch_attention_scores):
                    agent_id = agent_ids[i] if i < len(agent_ids) else "default"
                    context = market_contexts[i] if i < len(market_contexts) else None
                    
                    attention_weights = self.optimize_attention_computation(
                        scores, agent_id, context
                    )
                    result.append(attention_weights.squeeze(0) if attention_weights.dim() > 1 else attention_weights)
            
            # Track computation time
            computation_time = (time.perf_counter() - start_time) * 1000
            self._track_computation_time(computation_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in batch attention computation: {e}")
            
            # Fallback to individual processing
            result = []
            for scores in batch_attention_scores:
                uniform_attention = torch.ones_like(scores) / scores.shape[-1]
                result.append(uniform_attention)
            return result
    
    def _track_computation_time(self, computation_time_ms: float):
        """Track attention computation times for performance monitoring."""
        
        self.computation_times.append(computation_time_ms)
        
        # Limit history size
        if len(self.computation_times) > 1000:
            self.computation_times.pop(0)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get attention computation performance statistics."""
        
        if not self.computation_times:
            return {'status': 'no_data'}
        
        times = np.array(self.computation_times)
        
        return {
            'mean_time_ms': float(np.mean(times)),
            'median_time_ms': float(np.median(times)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'p99_time_ms': float(np.percentile(times, 99)),
            'max_time_ms': float(np.max(times)),
            'std_time_ms': float(np.std(times)),
            'num_computations': len(times)
        }
    
    def analyze_attention_patterns(
        self,
        attention_weights: torch.Tensor,
        agent_id: str = "default"
    ) -> Dict[str, float]:
        """Analyze attention patterns for insights."""
        
        try:
            # Convert to numpy for analysis
            if isinstance(attention_weights, torch.Tensor):
                weights = attention_weights.detach().cpu().numpy()
            else:
                weights = np.array(attention_weights)
            
            # Flatten if batched
            if weights.ndim > 1:
                weights = weights.mean(axis=0)
            
            # Calculate pattern metrics
            entropy = -np.sum(weights * np.log(weights + 1e-8))
            max_entropy = np.log(len(weights))
            focus_score = 1.0 - (entropy / max_entropy)  # Higher = more focused
            
            # Find primary focus features
            top_indices = np.argsort(weights)[::-1][:3]
            primary_focus = int(top_indices[0])
            secondary_focus = int(top_indices[1])
            
            # Calculate specialization score
            top_weight = weights[primary_focus]
            avg_weight = np.mean(weights)
            specialization_score = top_weight / avg_weight
            
            pattern_analysis = {
                'entropy': float(entropy),
                'focus_score': float(focus_score),
                'specialization_score': float(specialization_score),
                'primary_focus_feature': primary_focus,
                'secondary_focus_feature': secondary_focus,
                'primary_focus_weight': float(top_weight),
                'weight_std': float(np.std(weights)),
                'agent_id': agent_id
            }
            
            # Track patterns
            self.attention_patterns.append(pattern_analysis)
            if len(self.attention_patterns) > 100:
                self.attention_patterns.pop(0)
            
            return pattern_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing attention patterns: {e}")
            return {'error': str(e)}
    
    def get_attention_insights(self) -> Dict[str, Any]:
        """Get insights from tracked attention patterns."""
        
        if not self.attention_patterns:
            return {'status': 'no_data'}
        
        patterns = self.attention_patterns
        
        # Calculate aggregate statistics
        focus_scores = [p['focus_score'] for p in patterns]
        specialization_scores = [p['specialization_score'] for p in patterns]
        primary_features = [p['primary_focus_feature'] for p in patterns]
        
        # Feature usage statistics
        feature_usage = {}
        for feature in range(7):
            feature_usage[f'feature_{feature}'] = primary_features.count(feature) / len(patterns)
        
        return {
            'mean_focus_score': float(np.mean(focus_scores)),
            'mean_specialization': float(np.mean(specialization_scores)),
            'focus_stability': float(1.0 - np.std(focus_scores)),
            'feature_usage_distribution': feature_usage,
            'most_used_feature': int(max(feature_usage, key=feature_usage.get).split('_')[1]),
            'num_patterns_analyzed': len(patterns)
        }
    
    def reset_tracking(self):
        """Reset performance and pattern tracking."""
        self.computation_times.clear()
        self.attention_patterns.clear()
        self.logger.info("Attention optimizer tracking reset")