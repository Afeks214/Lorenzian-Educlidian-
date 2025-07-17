"""
Strategic MARL Component - Core orchestration component for the Strategic MARL system.

This module implements the central StrategicMARLComponent that handles SYNERGY_DETECTED
events and coordinates the three specialized agents (MLMI, NWRQK, Regime Detection).

Key Features:
- Async event handling for SYNERGY_DETECTED events
- Parallel execution of three specialized agents
- Decision aggregation with superposition calculation
- Performance monitoring and error handling
- Mathematical validation and circuit breaker patterns
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import numpy as np
import torch
from functools import lru_cache
import warnings
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextvars import ContextVar
from threading import Lock
from weakref import WeakKeyDictionary

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Import numba with fallback
try:
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# GPU acceleration setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32

from src.core.component_base import ComponentBase
from src.core.events import EventType, Event
from src.agents.strategic_agent_base import (
    MLMIStrategicAgent, 
    NWRQKStrategicAgent, 
    RegimeDetectionAgent,
    AgentPrediction
)
from src.agents.gating_network import GatingNetwork, GatingNetworkTrainer
from src.algorithms.adaptive_weights import create_adaptive_weight_learner, AdaptationStrategy


@dataclass
class StrategicDecision:
    """Represents a strategic decision made by the ensemble of agents."""
    action: str
    confidence: float
    uncertainty: float
    should_proceed: bool
    reasoning: str
    timestamp: datetime
    agent_contributions: Dict[str, float]
    performance_metrics: Dict[str, float]


class StrategicMARLComponent(ComponentBase):
    """
    Strategic MARL Component - Core orchestration for the 30-minute Strategic MARL system.
    
    This component:
    1. Listens for SYNERGY_DETECTED events from the synergy detection system
    2. Extracts and validates the 48x13 matrix data
    3. Coordinates three specialized agents (MLMI, NWRQK, Regime Detection)
    4. Aggregates their decisions using superposition calculation
    5. Publishes final strategic decisions with confidence scores
    """
    
    def __init__(self, kernel):
        """
        Initialize the Strategic MARL Component.
        
        Args:
            kernel: Reference to the AlgoSpace kernel for system integration
        """
        super().__init__("StrategicMARLComponent", kernel)
        
        # Set up structured logging
        self.logger = logging.getLogger('strategic_marl')
        
        # Load strategic configuration
        self.strategic_config = self._load_strategic_config()
        
        # JIT compilation and caching setup
        self._context_cache = {}
        self._tensor_cache = {}
        self._agent_result_cache = {}
        self._result_cache_lock = Lock()
        self._cache_ttl_seconds = 30  # Cache results for 30 seconds
        self._cache_max_size = 1000
        
        # Thread pool for CPU-bound operations
        self._cpu_executor = ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix='strategic_marl_cpu'
        )
        
        # Process pool for heavy computations (if needed)
        self._process_executor = ProcessPoolExecutor(
            max_workers=2,
            context=None  # Use default context
        )
        
        # Parallel execution metrics
        self._parallel_metrics = {
            'agent_execution_times': deque(maxlen=100),
            'cache_hit_rate': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_speedup': 0.0,
            'thread_pool_utilization': 0.0
        }
        
        # Pre-allocated tensor buffers for performance (GPU-ready)
        self._tensor_buffer_size = 1000
        self._preallocated_tensors = {
            'context_tensor': torch.zeros(6, dtype=TORCH_DTYPE, device=DEVICE),
            'agent_probs': torch.zeros(3, 3, dtype=TORCH_DTYPE, device=DEVICE),
            'ensemble_probs': torch.zeros(3, dtype=TORCH_DTYPE, device=DEVICE),
            'weights': torch.zeros(3, dtype=TORCH_DTYPE, device=DEVICE),
            'matrix_buffer': torch.zeros(48, 13, dtype=TORCH_DTYPE, device=DEVICE)
        }
        
        # LRU cache for context extraction
        self._context_cache_size = 100
        self._context_cache = {}
        
        # Initialize agent placeholders (will be created in initialize())
        self.mlmi_agent: Optional[MLMIStrategicAgent] = None
        self.nwrqk_agent: Optional[NWRQKStrategicAgent] = None
        self.regime_agent: Optional[RegimeDetectionAgent] = None
        
        # Performance metrics tracking
        self.performance_metrics = {
            'total_inferences': 0,
            'total_inference_time_ms': 0.0,
            'avg_inference_time_ms': 0.0,
            'max_inference_time_ms': 0.0,
            'timeout_count': 0,
            'error_count': 0,
            'success_count': 0
        }
        
        # Circuit breaker state
        self.circuit_breaker = {
            'consecutive_failures': 0,
            'max_failures': self.strategic_config.get('safety', {}).get('max_consecutive_failures', 5),
            'is_open': False,
            'last_failure_time': None
        }
        
        # Gating Network configuration (replaces static ensemble weights)
        ensemble_config = self.strategic_config.get('ensemble', {})
        gating_config = ensemble_config.get('gating_network', {})
        
        self.confidence_threshold = ensemble_config.get('confidence_threshold', 0.65)
        
        # Adaptive weight learning configuration
        self.enable_adaptive_weights = ensemble_config.get('enable_adaptive_weights', True)
        self.adaptive_weight_update_frequency = ensemble_config.get('adaptive_update_frequency', 5)
        self.adaptive_weight_counter = 0
        
        # Initialize intelligent gating network with config
        self.gating_network = GatingNetwork(
            shared_context_dim=self._get_shared_context_dim(),
            n_agents=3,
            hidden_dim=gating_config.get('hidden_dim', 64)
        )
        
        # Add gating network optimizer and trainer
        self.gating_trainer = GatingNetworkTrainer(
            self.gating_network,
            learning_rate=gating_config.get('learning_rate', 1e-4)
        )
        
        # Initialize adaptive weight learner for enhanced coordination
        self.adaptive_weight_learner = create_adaptive_weight_learner(
            n_agents=3,
            strategy=AdaptationStrategy.HYBRID,
            context_dim=self._get_shared_context_dim()
        )
        
        # Performance tracking for adaptive learning
        self.agent_performance_history = deque(maxlen=1000)
        self.weight_adaptation_history = deque(maxlen=500)
        
        # Performance requirements
        perf_config = self.strategic_config.get('performance', {})
        self.max_inference_latency_ms = perf_config.get('max_inference_latency_ms', 10.0)  # Target <10ms
        self.agent_timeout_ms = perf_config.get('agent_timeout_ms', 8.0)  # Individual agent timeout
        
        # Performance monitoring setup
        self.performance_alerts = {
            'latency_threshold_ms': 10.0,
            'consecutive_slow_inferences': 0,
            'max_consecutive_slow': 3,
            'alert_sent': False
        }
        
        # Warmup JIT compilation
        self._warmup_jit_compilation()
        
        self.logger.info(
            f"StrategicMARLComponent initialized with intelligent gating: "
            f"gating_network_dim={self._get_shared_context_dim()}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"max_latency_ms={self.max_inference_latency_ms}"
        )
    
    def _load_strategic_config(self) -> Dict[str, Any]:
        """Load strategic configuration from the config manager."""
        try:
            # Get strategic config section from the kernel's config manager
            strategic_config = self.config.get('strategic', {})
            
            # If no strategic config in main config, try to load from strategic_config.yaml
            if not strategic_config:
                import yaml
                config_path = "/home/QuantNova/GrandModel/configs/strategic_config.yaml"
                try:
                    with open(config_path, 'r') as f:
                        strategic_config = yaml.safe_load(f)
                    self.logger.info(f"Loaded strategic config from {config_path}")
                except FileNotFoundError:
                    self.logger.warning(f"Strategic config file not found at {config_path}, using defaults")
                    strategic_config = self._get_default_config()
                except Exception as e:
                    self.logger.error(f"Error loading strategic config: {e}, using defaults")
                    strategic_config = self._get_default_config()
            
            return strategic_config
        except Exception as e:
            self.logger.error(f"Error in _load_strategic_config: {e}, using defaults")
            return self._get_default_config()

    def _get_shared_context_dim(self) -> int:
        """Get dimension of shared context vector for gating network."""
        return 6  # [volatility_30, volume_ratio, momentum_20, momentum_50, mmd_score, price_trend]

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'environment': {
                'matrix_shape': [48, 13],
                'feature_indices': {
                    'mlmi_expert': [0, 1, 9, 10],
                    'nwrqk_expert': [2, 3, 4, 5],
                    'regime_expert': [6, 7, 8, 11, 12]
                }
            },
            'ensemble': {
                'confidence_threshold': 0.7,
                'max_inference_latency_ms': 5,
                'adaptive_weights': True
            },
            'performance': {
                'max_latency_ms': 5,
                'enable_metrics': True
            }
        }

# PyTorch JIT-compiled performance-critical functions
@torch.jit.script
def _jit_extract_context_fast(matrix_data: torch.Tensor) -> torch.Tensor:
    """
    PyTorch JIT-compiled fast context extraction with vectorized operations.
    
    Args:
        matrix_data: Input matrix (48, 13) as torch.Tensor
        
    Returns:
        Context vector [volatility_30, volume_ratio, momentum_20, momentum_50, mmd_score, price_trend]
    """
    rows, cols = matrix_data.shape
    
    # Initialize context vector
    context = torch.zeros(6, dtype=matrix_data.dtype, device=matrix_data.device)
    
    # Vectorized calculations with bounds checking
    if cols > 11:
        context[0] = torch.std(matrix_data[:, 11], unbiased=False)  # volatility_30 (match NumPy ddof=0)
    if cols > 12:
        context[1] = torch.mean(matrix_data[:, 12])  # volume_ratio
    if cols > 9:
        context[2] = torch.mean(matrix_data[:, 9])   # momentum_20
    if cols > 10:
        context[3] = torch.mean(matrix_data[:, 10])  # momentum_50
    if cols > 0:
        context[5] = torch.mean(matrix_data[:, 0])   # price_trend
    
    # Enhanced MMD score calculation (vectorized)
    if cols > 10 and rows > 24:
        recent_window = matrix_data[-24:, 10]
        historical_window = matrix_data[:-24, 10]
        if len(recent_window) > 0 and len(historical_window) > 0:
            context[4] = torch.abs(torch.mean(recent_window) - torch.mean(historical_window))
    
    return context

@torch.jit.script
def _jit_aggregate_decisions_fast(
    agent_probs: torch.Tensor,
    weights: torch.Tensor,
    confidences: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch JIT-compiled fast decision aggregation with vectorized operations.
    
    Args:
        agent_probs: Agent probabilities [n_agents, n_actions]
        weights: Agent weights [n_agents]
        confidences: Agent confidences [n_agents]
        
    Returns:
        Ensemble probabilities [n_actions]
    """
    n_agents, n_actions = agent_probs.shape
    
    # Vectorized weighted aggregation using tensor operations
    effective_weights = weights * confidences  # [n_agents]
    weighted_probs = agent_probs * effective_weights.unsqueeze(1)  # [n_agents, n_actions]
    ensemble_probs = torch.sum(weighted_probs, dim=0)  # [n_actions]
    
    # Normalize probabilities
    total_weight = torch.sum(effective_weights)
    if total_weight > 0:
        ensemble_probs = ensemble_probs / total_weight
    else:
        # Fallback uniform distribution
        ensemble_probs = torch.ones(n_actions, dtype=agent_probs.dtype, device=agent_probs.device) / n_actions
    
    return ensemble_probs

@torch.jit.script
def _jit_calculate_ensemble_confidence(
    ensemble_probs: torch.Tensor,
    gating_confidence: float
) -> Tuple[float, float]:
    """
    PyTorch JIT-compiled confidence and uncertainty calculation.
    
    Args:
        ensemble_probs: Ensemble probabilities [n_actions]
        gating_confidence: Gating network confidence
        
    Returns:
        Tuple of (confidence, uncertainty)
    """
    # Confidence based on max probability and gating confidence
    confidence = float(torch.max(ensemble_probs)) * gating_confidence
    
    # Uncertainty based on entropy (with numerical stability)
    epsilon = 1e-8
    safe_probs = torch.clamp(ensemble_probs, min=epsilon)
    uncertainty = float(-torch.sum(safe_probs * torch.log(safe_probs)))
    
    return confidence, uncertainty


# Caching utility functions
def _create_cache_key(matrix_data: np.ndarray, shared_context: Dict[str, Any]) -> str:
    """Create a cache key from matrix data and shared context."""
    # Create hash from matrix data
    matrix_hash = hashlib.md5(matrix_data.tobytes()).hexdigest()[:16]
    
    # Create hash from relevant context fields
    context_key = {
        'market_volatility': shared_context.get('market_volatility', 0.0),
        'volume_profile': shared_context.get('volume_profile', 0.0),
        'momentum_signal': shared_context.get('momentum_signal', 0.0),
        'trend_strength': shared_context.get('trend_strength', 0.0),
        'mmd_score': shared_context.get('mmd_score', 0.0),
        'market_regime': shared_context.get('market_regime', 'ranging')
    }
    
    context_hash = hashlib.md5(
        json.dumps(context_key, sort_keys=True).encode()
    ).hexdigest()[:16]
    
    return f"{matrix_hash}_{context_hash}"


    def _cleanup_cache(self, cache_dict: Dict, max_size: int, ttl_seconds: int):
        """Clean up cache entries based on size and TTL."""
        current_time = time.time()
        
        # Remove expired entries
        expired_keys = []
        for key, (timestamp, _) in cache_dict.items():
            if current_time - timestamp > ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            cache_dict.pop(key, None)
        
        # Remove oldest entries if over max size
        if len(cache_dict) > max_size:
            # Sort by timestamp and remove oldest
            sorted_items = sorted(cache_dict.items(), key=lambda x: x[1][0])
            excess_count = len(cache_dict) - max_size
            for key, _ in sorted_items[:excess_count]:
                cache_dict.pop(key, None)

    async def initialize(self):
        """Initialize the Strategic MARL Component."""
        try:
            # Initialize agents
            await self._initialize_agents()
            
            # Validate configuration
            self._validate_configuration()
            
            # Subscribe to SYNERGY_DETECTED events
            if hasattr(self, 'event_bus') and self.event_bus:
                await self.event_bus.subscribe(EventType.SYNERGY_DETECTED, self.process_synergy_event)
            
            self._initialized = True
            self.logger.info("Strategic MARL Component initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Strategic MARL Component: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the component gracefully."""
        self.logger.info("Shutting down Strategic MARL Component...")
        
        # Shutdown executors
        await self._shutdown_executors()
        
        # Clear caches
        with self._result_cache_lock:
            self._agent_result_cache.clear()
            self._context_cache.clear()
            self._tensor_cache.clear()
        
        # Call parent shutdown
        await super().shutdown()
        
        self.logger.info("Strategic MARL Component shutdown complete")
    
    async def _initialize_agents(self):
        """Initialize the three specialized agents."""
        try:
            # Initialize agents with their respective configurations
            agent_configs = self.strategic_config.get('agents', {})
            
            # Initialize MLMI agent
            mlmi_config = agent_configs.get('mlmi_expert', {})
            self.mlmi_agent = MLMIStrategicAgent(
                feature_indices=self.strategic_config['environment']['feature_indices']['mlmi_expert'],
                hidden_dims=mlmi_config.get('hidden_dims', [64, 32]),
                dropout_rate=mlmi_config.get('dropout_rate', 0.1)
            )
            await self.mlmi_agent.initialize()
            
            # Initialize NWRQK agent
            nwrqk_config = agent_configs.get('nwrqk_expert', {})
            self.nwrqk_agent = NWRQKStrategicAgent(
                feature_indices=self.strategic_config['environment']['feature_indices']['nwrqk_expert'],
                hidden_dims=nwrqk_config.get('hidden_dims', [64, 32]),
                dropout_rate=nwrqk_config.get('dropout_rate', 0.1)
            )
            await self.nwrqk_agent.initialize()
            
            # Initialize Regime Detection agent
            regime_config = agent_configs.get('regime_expert', {})
            self.regime_agent = RegimeDetectionAgent(
                feature_indices=self.strategic_config['environment']['feature_indices']['regime_expert'],
                hidden_dims=regime_config.get('hidden_dims', [64, 32]),
                dropout_rate=regime_config.get('dropout_rate', 0.15)
            )
            await self.regime_agent.initialize()
            
            self.logger.info("All strategic agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            raise
    
    def _validate_configuration(self):
        """Validate the configuration."""
        required_sections = ['environment', 'ensemble', 'performance']
        for section in required_sections:
            if section not in self.strategic_config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate ensemble weights
        weights = self.strategic_config['ensemble'].get('weights', [0.33, 0.33, 0.34])
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Ensemble weights must sum to 1.0")
        
        # Set ensemble weights
        self.ensemble_weights = np.array(weights, dtype=np.float32)
        
        self.logger.info("Configuration validation passed")
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached agent result if available and valid."""
        with self._result_cache_lock:
            if cache_key in self._agent_result_cache:
                timestamp, result = self._agent_result_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl_seconds:
                    self._parallel_metrics['cache_hits'] += 1
                    self._update_cache_hit_rate()
                    return result
                else:
                    # Remove expired entry
                    del self._agent_result_cache[cache_key]
            
            self._parallel_metrics['cache_misses'] += 1
            self._update_cache_hit_rate()
            return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """Cache agent result with timestamp."""
        with self._result_cache_lock:
            # Clean up cache if needed
            _cleanup_cache(self._agent_result_cache, self._cache_max_size, self._cache_ttl_seconds)
            
            # Add new result
            self._agent_result_cache[cache_key] = (time.time(), result)
    
    def _update_cache_hit_rate(self):
        """Update cache hit rate metric."""
        total_requests = self._parallel_metrics['cache_hits'] + self._parallel_metrics['cache_misses']
        if total_requests > 0:
            self._parallel_metrics['cache_hit_rate'] = self._parallel_metrics['cache_hits'] / total_requests
    
    async def process_synergy_event(self, event_data: Dict[str, Any]):
        """Process SYNERGY_DETECTED event."""
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if self.circuit_breaker['is_open']:
                self.logger.warning("Circuit breaker is open, skipping event")
                return
            
            # Extract and validate matrix data
            matrix_data = self._extract_and_validate_matrix(event_data)
            
            # Extract shared context
            shared_context = self._extract_shared_context(matrix_data)
            
            # Execute agents in parallel
            agent_results = await self._execute_agents_parallel(matrix_data, shared_context)
            
            # Combine agent outputs into strategic decision
            decision = self._combine_agent_outputs(agent_results)
            
            # Publish strategic decision
            await self._publish_strategic_decision(decision)
            
            # Update performance metrics
            inference_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(inference_time_ms, success=True)
            
            self.logger.info(f"Strategic decision processed in {inference_time_ms:.2f}ms")
            
        except Exception as e:
            inference_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(inference_time_ms, success=False)
            self._handle_processing_error(e)
            raise
    
    def _extract_and_validate_matrix(self, event_data: Dict[str, Any]) -> np.ndarray:
        """Extract and validate matrix data from event."""
        if 'matrix_data' not in event_data:
            raise ValueError("Event data missing required 'matrix_data' field")
        
        matrix_data = event_data['matrix_data']
        
        # Validate shape
        expected_shape = tuple(self.strategic_config['environment']['matrix_shape'])
        if matrix_data.shape != expected_shape:
            raise ValueError(f"Invalid matrix shape: {matrix_data.shape}, expected {expected_shape}")
        
        # Validate data quality
        if np.any(np.isnan(matrix_data)):
            raise ValueError("Matrix contains NaN values")
        
        if np.any(np.isinf(matrix_data)):
            raise ValueError("Matrix contains infinite values")
        
        return matrix_data
    
    @lru_cache(maxsize=100)
    def _extract_shared_context_cached(self, matrix_hash: str, matrix_data: np.ndarray) -> Dict[str, Any]:
        """Extract shared context using cached JIT-compiled function."""
        # Convert to PyTorch tensor for JIT compilation
        matrix_tensor = torch.from_numpy(matrix_data).to(device=DEVICE, dtype=TORCH_DTYPE)
        
        # Use PyTorch JIT-compiled function for fast context extraction
        context_vector = _jit_extract_context_fast(matrix_tensor)
        
        # Create structured context
        context = {
            'market_volatility': float(context_vector[0].item()),
            'volume_profile': float(context_vector[1].item()),
            'momentum_signal': float(context_vector[2].item()),
            'trend_strength': float(context_vector[3].item()),
            'mmd_score': float(context_vector[4].item()),
            'price_trend': float(context_vector[5].item()),
            'market_regime': self._detect_market_regime(matrix_data),
            'timestamp': datetime.now().isoformat()
        }
        
        return context
    
    def _extract_shared_context(self, matrix_data: np.ndarray) -> Dict[str, Any]:
        """Extract shared context using cached JIT-compiled function."""
        # Create hash for caching
        matrix_hash = str(hash(matrix_data.tobytes()))
        return self._extract_shared_context_cached(matrix_hash, matrix_data)
    
    def _detect_market_regime(self, matrix_data: np.ndarray) -> str:
        """Detect market regime from matrix data."""
        # Simple regime detection based on volatility and momentum
        volatility = np.std(matrix_data[:, -1])  # Last column volatility
        momentum = np.mean(matrix_data[:, 9]) if matrix_data.shape[1] > 9 else 0
        
        if volatility > 0.05:
            return 'volatile'
        elif abs(momentum) > 0.01:
            return 'trending'
        else:
            return 'ranging'
    
    async def _execute_agents_parallel(self, matrix_data: np.ndarray, shared_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute agents in parallel with advanced optimizations."""
        parallel_start_time = time.time()
        
        # Check cache first
        cache_key = _create_cache_key(matrix_data, shared_context)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for key: {cache_key[:16]}...")
            return cached_result
        
        # Agent timeout configuration
        agent_timeout_seconds = self.agent_timeout_ms / 1000.0
        overall_timeout_seconds = self.max_inference_latency_ms / 1000.0
        
        # Agent execution with comprehensive error handling and fallback
        async def execute_agent_optimized(agent, agent_name: str, feature_indices: List[int]):
            """Execute single agent with optimizations."""
            agent_start_time = time.time()
            
            try:
                # Pre-process data in thread pool for CPU-bound operations
                loop = asyncio.get_event_loop()
                
                # Extract features in thread pool
                features_future = loop.run_in_executor(
                    self._cpu_executor,
                    self._extract_agent_features,
                    matrix_data,
                    feature_indices
                )
                
                # Execute agent prediction with timeout
                prediction_task = agent.predict(matrix_data, shared_context)
                
                # Wait for both feature extraction and prediction
                try:
                    features, result = await asyncio.gather(
                        asyncio.wait_for(features_future, timeout=agent_timeout_seconds * 0.3),
                        asyncio.wait_for(prediction_task, timeout=agent_timeout_seconds),
                        return_exceptions=True
                    )
                    
                    # Check for exceptions
                    if isinstance(features, Exception):
                        self.logger.warning(f"{agent_name} feature extraction failed: {features}")
                        features = None
                    
                    if isinstance(result, Exception):
                        raise result
                    
                    # Add execution time to result
                    execution_time = (time.time() - agent_start_time) * 1000
                    result['computation_time_ms'] = execution_time
                    result['features_extracted'] = features is not None
                    result['agent_name'] = agent_name
                    
                    return result
                    
                except asyncio.TimeoutError:
                    self.performance_metrics['timeout_count'] += 1
                    self.logger.warning(f"{agent_name} agent timed out after {agent_timeout_seconds}s")
                    return self._get_fallback_result(agent_name)
                    
            except Exception as e:
                self.logger.error(f"{agent_name} agent failed: {e}")
                return self._get_fallback_result(agent_name)
        
        # Create agent tasks with staggered execution for load balancing
        agent_configs = [
            (self.mlmi_agent, 'MLMI', self.strategic_config['environment']['feature_indices']['mlmi_expert']),
            (self.nwrqk_agent, 'NWRQK', self.strategic_config['environment']['feature_indices']['nwrqk_expert']),
            (self.regime_agent, 'Regime', self.strategic_config['environment']['feature_indices']['regime_expert'])
        ]
        
        # Create tasks with individual monitoring
        agent_tasks = []
        for agent, name, indices in agent_configs:
            task = asyncio.create_task(
                execute_agent_optimized(agent, name, indices),
                name=f"{name}_agent_task"
            )
            agent_tasks.append(task)
        
        # Execute with comprehensive timeout and fallback handling
        try:
            # Use asyncio.gather with return_exceptions=True for better error handling
            results = await asyncio.wait_for(
                asyncio.gather(*agent_tasks, return_exceptions=True),
                timeout=overall_timeout_seconds
            )
            
            # Process results and handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    agent_name = agent_configs[i][1]
                    self.logger.error(f"{agent_name} agent failed with exception: {result}")
                    final_results.append(self._get_fallback_result(agent_name))
                else:
                    final_results.append(result)
            
            # Calculate parallel execution metrics
            parallel_time = (time.time() - parallel_start_time) * 1000
            self._parallel_metrics['agent_execution_times'].append(parallel_time)
            
            # Calculate speedup (estimated sequential time vs parallel time)
            estimated_sequential_time = sum(r.get('computation_time_ms', 0) for r in final_results)
            if estimated_sequential_time > 0:
                speedup = estimated_sequential_time / parallel_time
                self._parallel_metrics['parallel_speedup'] = speedup
            
            # Cache successful results
            if all(not r.get('fallback', False) for r in final_results):
                self._cache_result(cache_key, final_results)
            
            return final_results
            
        except asyncio.TimeoutError:
            self.logger.error(f"Overall agent execution timed out after {overall_timeout_seconds}s")
            # Cancel remaining tasks
            for task in agent_tasks:
                if not task.done():
                    task.cancel()
            
            # Return fallback results for all agents
            return [
                self._get_fallback_result(config[1]) 
                for config in agent_configs
            ]
        
        except Exception as e:
            self.logger.error(f"Critical error in parallel execution: {e}")
            # Cancel remaining tasks
            for task in agent_tasks:
                if not task.done():
                    task.cancel()
            
            # Return fallback results for all agents
            return [
                self._get_fallback_result(config[1]) 
                for config in agent_configs
            ]
    
    def _extract_agent_features(self, matrix_data: np.ndarray, feature_indices: List[int]) -> Optional[np.ndarray]:
        """Extract features for agent in thread pool (CPU-bound operation)."""
        try:
            if not feature_indices or matrix_data.shape[1] <= max(feature_indices):
                return None
            
            # Extract relevant features
            features = matrix_data[:, feature_indices]
            
            # Basic feature processing
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def _get_fallback_result(self, agent_name: str) -> Dict[str, Any]:
        """Get fallback result for failed agent that SUPPORTS strategy decisions."""
        return {
            'agent_name': agent_name,
            'action_probabilities': [0.33, 0.34, 0.33],  # Uniform distribution
            'confidence': 0.1,  # Low confidence for fallback
            'features_used': [],
            'feature_importance': {},
            'internal_state': {},
            'computation_time_ms': 0.0,
            'fallback': True,
            'strategy_support_mode': True,  # Flag to indicate strategy support
            'strategy_override_allowed': False  # Prevent overriding strategy decisions
        }
    
    def _combine_agent_outputs(self, agent_results: List[Dict[str, Any]]) -> StrategicDecision:
        """Combine agent outputs into strategic decision that SUPPORTS strategy signals."""
        # Check if any agent is in strategy support mode
        strategy_support_agents = [r for r in agent_results if r.get('strategy_support_mode', False)]
        
        # Prepare data for PyTorch JIT compilation
        n_agents = len(agent_results)
        n_actions = len(agent_results[0]['action_probabilities'])
        
        # Convert to PyTorch tensors on GPU if available
        agent_probs = torch.tensor(
            [result['action_probabilities'] for result in agent_results], 
            dtype=TORCH_DTYPE, 
            device=DEVICE
        )
        confidences = torch.tensor(
            [result['confidence'] for result in agent_results], 
            dtype=TORCH_DTYPE, 
            device=DEVICE
        )
        
        # Reduce confidence for fallback agents to ensure strategy decisions dominate
        for i, result in enumerate(agent_results):
            if result.get('fallback', False):
                confidences[i] *= 0.1  # Heavily penalize fallback agents
        
        # Get gating network weights
        context_vector = torch.tensor([
            agent_results[0].get('market_volatility', 0.0),
            agent_results[0].get('volume_profile', 0.0),
            agent_results[0].get('momentum_signal', 0.0),
            agent_results[0].get('trend_strength', 0.0),
            agent_results[0].get('mmd_score', 0.0),
            agent_results[0].get('price_trend', 0.0)
        ], dtype=TORCH_DTYPE, device=DEVICE)
        
        # Use gating network for dynamic weight computation
        gating_weights, gating_confidence = self.gating_network.compute_weights(
            context_vector.unsqueeze(0)
        )
        weights = gating_weights[0].to(device=DEVICE, dtype=TORCH_DTYPE)
        
        # Use PyTorch JIT-compiled aggregation
        ensemble_probs = _jit_aggregate_decisions_fast(agent_probs, weights, confidences)
        
        # Calculate final confidence and uncertainty
        confidence, uncertainty = _jit_calculate_ensemble_confidence(ensemble_probs, float(gating_confidence))
        
        # Determine action with strategy support logic
        action_idx = int(torch.argmax(ensemble_probs).item())
        action_map = {0: 'buy', 1: 'hold', 2: 'sell'}
        action = action_map[action_idx]
        
        # Determine if should proceed - favor strategy decisions
        should_proceed = confidence >= self.confidence_threshold
        
        # Check for strategy override protection
        strategy_override_blocked = any(r.get('strategy_override_allowed', True) == False for r in agent_results)
        if strategy_override_blocked:
            # Agents are protecting a strategy decision, reduce agent autonomy
            should_proceed = should_proceed and confidence >= (self.confidence_threshold * 1.2)
        
        # Create agent contributions (convert back to CPU for storage)
        agent_contributions = {}
        weights_cpu = weights.cpu().numpy()
        confidences_cpu = confidences.cpu().numpy()
        for i, result in enumerate(agent_results):
            agent_contributions[result['agent_name']] = float(weights_cpu[i] * confidences_cpu[i])
        
        # Create performance metrics
        performance_metrics = {
            'ensemble_probabilities': ensemble_probs.cpu().numpy().tolist(),
            'individual_confidences': confidences.cpu().numpy().tolist(),
            'gating_weights': weights.cpu().numpy().tolist(),
            'total_computation_time_ms': sum(r.get('computation_time_ms', 0) for r in agent_results)
        }
        
        # Create reasoning with strategy support context
        reasoning = f"Ensemble decision based on {n_agents} agents with {confidence:.2f} confidence. "
        reasoning += f"Gating network assigned weights: {weights.cpu().numpy().tolist()}"
        
        # Add strategy support context
        fallback_count = sum(1 for r in agent_results if r.get('fallback', False))
        if fallback_count > 0:
            reasoning += f" ({fallback_count} agents in fallback mode - strategy decisions prioritized)"
        
        if strategy_override_blocked:
            reasoning += " (Strategy override protection active)"
        
        return StrategicDecision(
            action=action,
            confidence=confidence,
            uncertainty=uncertainty,
            should_proceed=should_proceed,
            reasoning=reasoning,
            timestamp=datetime.now(),
            agent_contributions=agent_contributions,
            performance_metrics=performance_metrics
        )
    
    async def _publish_strategic_decision(self, decision: StrategicDecision):
        """Publish strategic decision to event bus."""
        event = Event(
            type=EventType.STRATEGIC_DECISION,
            data={
                'decision': decision,
                'component': self.name,
                'timestamp': decision.timestamp.isoformat()
            }
        )
        
        await self.event_bus.publish(event)
        
        self.logger.info(f"Published strategic decision: {decision.action} (confidence: {decision.confidence:.2f})")
    
    def _update_performance_metrics(self, inference_time_ms: float, success: bool):
        """Update performance metrics with parallel execution tracking."""
        self.performance_metrics['total_inferences'] += 1
        self.performance_metrics['total_inference_time_ms'] += inference_time_ms
        
        if success:
            self.performance_metrics['success_count'] += 1
        else:
            self.performance_metrics['error_count'] += 1
        
        # Update average
        self.performance_metrics['avg_inference_time_ms'] = (
            self.performance_metrics['total_inference_time_ms'] / 
            self.performance_metrics['total_inferences']
        )
        
        # Update max
        if inference_time_ms > self.performance_metrics['max_inference_time_ms']:
            self.performance_metrics['max_inference_time_ms'] = inference_time_ms
        
        # Update thread pool utilization
        if hasattr(self, '_cpu_executor'):
            active_threads = getattr(self._cpu_executor, '_threads', set())
            max_workers = self._cpu_executor._max_workers
            self._parallel_metrics['thread_pool_utilization'] = len(active_threads) / max_workers
        
        # Check performance alerts
        if inference_time_ms > self.performance_alerts['latency_threshold_ms']:
            self.performance_alerts['consecutive_slow_inferences'] += 1
            
            if (self.performance_alerts['consecutive_slow_inferences'] >= 
                self.performance_alerts['max_consecutive_slow'] and 
                not self.performance_alerts['alert_sent']):
                
                self.logger.warning(
                    f"Performance alert: {self.performance_alerts['consecutive_slow_inferences']} "
                    f"consecutive inferences exceeded {self.performance_alerts['latency_threshold_ms']}ms"
                )
                self.performance_alerts['alert_sent'] = True
        else:
            self.performance_alerts['consecutive_slow_inferences'] = 0
            self.performance_alerts['alert_sent'] = False
        
        # Log performance insights
        if self.performance_metrics['total_inferences'] % 100 == 0:
            self._log_performance_insights()
    
    def _handle_processing_error(self, error: Exception):
        """Handle processing errors and update circuit breaker."""
        self.circuit_breaker['consecutive_failures'] += 1
        self.circuit_breaker['last_failure_time'] = time.time()
        
        if (self.circuit_breaker['consecutive_failures'] >= 
            self.circuit_breaker['max_failures']):
            self.circuit_breaker['is_open'] = True
            self.logger.error(f"Circuit breaker opened after {self.circuit_breaker['consecutive_failures']} failures")
        
        self.logger.error(f"Processing error: {error}")
    
    def _log_performance_insights(self):
        """Log performance insights and optimization recommendations."""
        avg_parallel_time = (
            sum(self._parallel_metrics['agent_execution_times']) / 
            len(self._parallel_metrics['agent_execution_times'])
            if self._parallel_metrics['agent_execution_times'] else 0
        )
        
        insights = []
        
        if avg_parallel_time > self.max_inference_latency_ms:
            insights.append(f"Average parallel execution time ({avg_parallel_time:.2f}ms) exceeds target ({self.max_inference_latency_ms}ms)")
        
        if self._parallel_metrics['cache_hit_rate'] < 0.3:
            insights.append(f"Low cache hit rate ({self._parallel_metrics['cache_hit_rate']:.2f}), consider increasing cache TTL")
        
        if self._parallel_metrics['parallel_speedup'] < 1.5:
            insights.append(f"Low parallel speedup ({self._parallel_metrics['parallel_speedup']:.2f}), check for bottlenecks")
        
        if insights:
            self.logger.info(f"Performance insights: {'; '.join(insights)}")
        else:
            self.logger.info(f"Performance optimal: avg={avg_parallel_time:.2f}ms, cache_hit_rate={self._parallel_metrics['cache_hit_rate']:.2f}, speedup={self._parallel_metrics['parallel_speedup']:.2f}")
    
    async def _shutdown_executors(self):
        """Shutdown thread and process pools gracefully."""
        if hasattr(self, '_cpu_executor'):
            self._cpu_executor.shutdown(wait=True)
        if hasattr(self, '_process_executor'):
            self._process_executor.shutdown(wait=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status with parallel execution metrics."""
        return {
            'name': self.name,
            'initialized': self.initialized,
            'circuit_breaker_open': self.circuit_breaker['is_open'],
            'consecutive_failures': self.circuit_breaker['consecutive_failures'],
            'performance_metrics': self.performance_metrics.copy(),
            'parallel_metrics': {
                'cache_hit_rate': self._parallel_metrics['cache_hit_rate'],
                'cache_hits': self._parallel_metrics['cache_hits'],
                'cache_misses': self._parallel_metrics['cache_misses'],
                'parallel_speedup': self._parallel_metrics['parallel_speedup'],
                'thread_pool_utilization': self._parallel_metrics['thread_pool_utilization'],
                'avg_parallel_execution_time_ms': (
                    sum(self._parallel_metrics['agent_execution_times']) / 
                    len(self._parallel_metrics['agent_execution_times'])
                    if self._parallel_metrics['agent_execution_times'] else 0
                )
            },
            'ensemble_weights': self.ensemble_weights.tolist() if hasattr(self, 'ensemble_weights') else [],
            'confidence_threshold': self.confidence_threshold,
            'performance_targets': {
                'max_inference_latency_ms': self.max_inference_latency_ms,
                'agent_timeout_ms': self.agent_timeout_ms,
                'cache_ttl_seconds': self._cache_ttl_seconds
            },
            'agents_status': {
                'mlmi_initialized': self.mlmi_agent is not None,
                'nwrqk_initialized': self.nwrqk_agent is not None,
                'regime_initialized': self.regime_agent is not None
            },
            'execution_pools': {
                'cpu_executor_max_workers': self._cpu_executor._max_workers if hasattr(self, '_cpu_executor') else 0,
                'process_executor_max_workers': self._process_executor._max_workers if hasattr(self, '_process_executor') else 0
            },
            'device': str(DEVICE),
            'tensor_cache_size': len(self._context_cache)
        }
    
    def _warmup_jit_compilation(self):
        """Warmup JIT compilation with dummy data."""
        try:
            # Create dummy data for warmup
            dummy_matrix = torch.randn(48, 13, dtype=TORCH_DTYPE, device=DEVICE)
            dummy_agent_probs = torch.randn(3, 3, dtype=TORCH_DTYPE, device=DEVICE)
            dummy_weights = torch.ones(3, dtype=TORCH_DTYPE, device=DEVICE) / 3
            dummy_confidences = torch.ones(3, dtype=TORCH_DTYPE, device=DEVICE) * 0.8
            
            # Warmup context extraction
            _ = _jit_extract_context_fast(dummy_matrix)
            
            # Warmup decision aggregation
            ensemble_probs = _jit_aggregate_decisions_fast(dummy_agent_probs, dummy_weights, dummy_confidences)
            
            # Warmup confidence calculation
            _ = _jit_calculate_ensemble_confidence(ensemble_probs, 0.8)
            
            self.logger.info("JIT compilation warmup completed successfully")
            
        except Exception as e:
            self.logger.warning(f"JIT compilation warmup failed: {e}")
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark performance of critical functions."""
        import time
        
        # Generate test data
        test_matrix = torch.randn(48, 13, dtype=TORCH_DTYPE, device=DEVICE)
        test_agent_probs = torch.randn(3, 3, dtype=TORCH_DTYPE, device=DEVICE)
        test_weights = torch.ones(3, dtype=TORCH_DTYPE, device=DEVICE) / 3
        test_confidences = torch.ones(3, dtype=TORCH_DTYPE, device=DEVICE) * 0.8
        
        # Benchmark context extraction
        start_time = time.time()
        for _ in range(num_iterations):
            _ = _jit_extract_context_fast(test_matrix)
        context_time = (time.time() - start_time) * 1000 / num_iterations
        
        # Benchmark decision aggregation
        start_time = time.time()
        ensemble_probs_list = []
        for _ in range(num_iterations):
            ensemble_probs = _jit_aggregate_decisions_fast(test_agent_probs, test_weights, test_confidences)
            ensemble_probs_list.append(ensemble_probs)
        aggregation_time = (time.time() - start_time) * 1000 / num_iterations
        
        # Benchmark confidence calculation
        start_time = time.time()
        for ensemble_probs in ensemble_probs_list:
            _ = _jit_calculate_ensemble_confidence(ensemble_probs, 0.8)
        confidence_time = (time.time() - start_time) * 1000 / num_iterations
        
        total_time = context_time + aggregation_time + confidence_time
        
        return {
            'context_extraction_ms': context_time,
            'decision_aggregation_ms': aggregation_time,
            'confidence_calculation_ms': confidence_time,
            'total_latency_ms': total_time,
            'target_met': total_time < 50.0,
            'device': str(DEVICE)
        }
    
    def clear_caches(self):
        """Clear all caches for memory management."""
        self._context_cache.clear()
        self._extract_shared_context_cached.cache_clear()
        self.logger.info("All caches cleared")
