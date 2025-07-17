"""
Decision Context Processor

Agent Beta: Real-time streaming specialist
Mission: Advanced context extraction and vector processing for LLM explanations

This module implements the decision context processor that extracts key decision
factors from MARL outputs, formats context for vector storage and LLM processing,
and calculates performance metrics with integration to the existing event bus system.

Key Features:
- Advanced feature extraction from decision contexts
- Vector embeddings for similarity search
- Performance metrics calculation and tracking
- LLM-ready context formatting
- Integration with existing event bus system
- Caching and optimization for high-frequency processing

Author: Agent Beta - Real-time Streaming Specialist
Version: 1.0 - Decision Context Processor
"""

import asyncio
import logging
import time
import uuid
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import json
import numpy as np
from collections import deque, defaultdict
import threading
from enum import Enum

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, using NumPy fallback for embeddings")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using simple embeddings")

from ...core.events import EventType, Event, EventBus
from ...core.component_base import ComponentBase
from .decision_capture import DecisionContext

logger = logging.getLogger(__name__)


class ProcessingPriority(Enum):
    """Processing priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ContextFeatureType(Enum):
    """Types of context features"""
    MARKET_INDICATOR = "market_indicator"
    AGENT_CONTRIBUTION = "agent_contribution"
    PERFORMANCE_METRIC = "performance_metric"
    TEMPORAL_FEATURE = "temporal_feature"
    RISK_FACTOR = "risk_factor"
    SENTIMENT_INDICATOR = "sentiment_indicator"


@dataclass
class ProcessedContext:
    """Processed decision context ready for LLM and streaming"""
    
    # Core Context
    decision_id: str
    timestamp: datetime
    processing_latency_ms: float
    
    # Decision Summary
    decision_summary: Dict[str, Any]
    key_factors: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    confidence_breakdown: Dict[str, float]
    
    # Feature Vectors
    feature_vector: np.ndarray
    embedding_vector: Optional[np.ndarray]
    similarity_hash: str
    
    # LLM Context
    llm_context: Dict[str, Any]
    explanation_template: str
    context_quality_score: float
    
    # Performance Metrics
    agent_performance: Dict[str, float]
    market_regime_score: float
    decision_complexity_score: float
    
    # Streaming Metadata
    priority: ProcessingPriority
    target_audiences: List[str]
    streaming_ready: bool


@dataclass
class ProcessorMetrics:
    """Context processor performance metrics"""
    
    total_contexts_processed: int = 0
    total_processing_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    
    feature_extraction_time_ms: float = 0.0
    embedding_generation_time_ms: float = 0.0
    llm_formatting_time_ms: float = 0.0
    
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    
    processing_errors: int = 0
    vector_generation_errors: int = 0
    
    queue_size: int = 0
    priority_queue_size: int = 0


class ContextProcessor(ComponentBase):
    """
    Decision Context Processor
    
    Processes captured decision contexts to extract key factors, generate feature
    vectors, and format context for LLM processing and real-time streaming.
    """
    
    def __init__(self, kernel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Context Processor
        
        Args:
            kernel: Reference to the AlgoSpace kernel
            config: Configuration dictionary
        """
        super().__init__("ContextProcessor", kernel)
        
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger('xai.context_processor')
        
        # Processing queues
        self.processing_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['queue_size']
        )
        self.priority_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['priority_queue_size']
        )
        
        # Embedding model
        self.embedding_model = None
        self.embedding_dim = self.config['embedding_dim']
        
        # Feature extractors
        self.feature_extractors: Dict[str, Callable] = {}
        self.feature_weights: Dict[str, float] = {}
        
        # Context cache for performance
        self.context_cache: Dict[str, ProcessedContext] = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = self.config['cache_size']
        
        # Performance metrics
        self.metrics = ProcessorMetrics()
        self.metrics_lock = threading.Lock()
        
        # Processing state
        self.active = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Performance monitoring
        self.processing_window = deque(maxlen=1000)
        self.quality_window = deque(maxlen=500)
        
        # Market regime tracking
        self.regime_history = deque(maxlen=100)
        self.volatility_tracker = deque(maxlen=50)
        
        self.logger.info(
            f"ContextProcessor initialized: "
            f"embedding_model={'sentence-transformers' if SENTENCE_TRANSFORMERS_AVAILABLE else 'simple'}, "
            f"embedding_dim={self.embedding_dim}, "
            f"cache_size={self.max_cache_size}"
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Processing Configuration
            'queue_size': 5000,
            'priority_queue_size': 1000,
            'batch_size': 50,
            'processing_interval_ms': 5,
            'max_processing_latency_ms': 50,
            
            # Feature Extraction
            'feature_vector_dim': 128,
            'embedding_dim': 384,
            'embedding_model': 'all-MiniLM-L6-v2',
            'feature_selection_threshold': 0.1,
            
            # Caching
            'cache_size': 2000,
            'cache_ttl_minutes': 30,
            'similarity_threshold': 0.95,
            
            # LLM Context
            'max_context_length': 4000,
            'context_template_version': 'v1',
            'include_technical_details': True,
            'explanation_depth': 'comprehensive',
            
            # Performance Monitoring
            'quality_score_threshold': 0.7,
            'complexity_threshold': 0.8,
            'monitoring_interval_seconds': 60,
            
            # Feature Weights
            'feature_weights': {
                'confidence': 0.25,
                'agent_consensus': 0.20,
                'market_volatility': 0.15,
                'momentum_strength': 0.15,
                'risk_score': 0.15,
                'temporal_context': 0.10
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the Context Processor"""
        try:
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Register feature extractors
            self._register_feature_extractors()
            
            # Subscribe to decision context events
            self.event_bus.subscribe(
                EventType.INDICATOR_UPDATE,  # Temporary - will use XAI_DECISION_CAPTURED
                self._handle_decision_context
            )
            
            # Start processing task
            self.active = True
            self.processing_task = asyncio.create_task(self._process_contexts())
            
            self._initialized = True
            self.logger.info("ContextProcessor initialized successfully")
            
            # Publish component started event
            event = self.event_bus.create_event(
                EventType.COMPONENT_STARTED,
                {"component": self.name, "status": "ready"},
                source=self.name
            )
            self.event_bus.publish(event)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ContextProcessor: {e}")
            raise
    
    async def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(
                    self.config['embedding_model']
                )
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                self.logger.info(f"Loaded embedding model: {self.config['embedding_model']}")
            else:
                self.logger.warning("Using simple embedding fallback")
                
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}, using fallback")
            self.embedding_model = None
    
    def _register_feature_extractors(self) -> None:
        """Register feature extraction functions"""
        self.feature_extractors.update({
            'confidence_features': self._extract_confidence_features,
            'agent_consensus_features': self._extract_agent_consensus_features,
            'market_features': self._extract_market_features,
            'temporal_features': self._extract_temporal_features,
            'risk_features': self._extract_risk_features,
            'performance_features': self._extract_performance_features
        })
        
        # Set feature weights from config
        self.feature_weights = self.config['feature_weights']
    
    def _handle_decision_context(self, event: Event) -> None:
        """
        Handle decision context events
        
        Args:
            event: Event containing decision context
        """
        try:
            # Extract context data
            event_payload = event.payload
            
            # Check if this is a decision context event
            if 'context' not in event_payload:
                return  # Not a decision context event
            
            context_data = event_payload['context']
            decision_id = event_payload.get('decision_id')
            priority = event_payload.get('processing_priority', 'normal')
            
            # Reconstruct DecisionContext
            context = DecisionContext(**context_data)
            
            # Schedule async processing
            asyncio.create_task(
                self._queue_context_for_processing(context, ProcessingPriority(priority))
            )
            
        except Exception as e:
            self.logger.error(f"Error handling decision context event: {e}")
            with self.metrics_lock:
                self.metrics.processing_errors += 1
    
    async def _queue_context_for_processing(
        self, 
        context: DecisionContext, 
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> None:
        """
        Queue decision context for processing
        
        Args:
            context: Decision context to process
            priority: Processing priority
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(context)
            cached_result = self._get_cached_context(cache_key)
            
            if cached_result:
                # Use cached result
                await self._publish_processed_context(cached_result)
                with self.metrics_lock:
                    self.metrics.cache_hits += 1
                return
            
            with self.metrics_lock:
                self.metrics.cache_misses += 1
            
            # Queue for processing
            processing_item = (context, priority, time.perf_counter())
            
            if priority in [ProcessingPriority.HIGH, ProcessingPriority.CRITICAL]:
                await self.priority_queue.put(processing_item)
                with self.metrics_lock:
                    self.metrics.priority_queue_size = self.priority_queue.qsize()
            else:
                await self.processing_queue.put(processing_item)
                with self.metrics_lock:
                    self.metrics.queue_size = self.processing_queue.qsize()
                    
        except asyncio.QueueFull:
            self.logger.warning(f"Processing queue full, dropping context {context.decision_id}")
        except Exception as e:
            self.logger.error(f"Error queuing context for processing: {e}")
    
    async def _process_contexts(self) -> None:
        """Process queued decision contexts"""
        while self.active:
            try:
                # Process priority queue first
                try:
                    item = await asyncio.wait_for(self.priority_queue.get(), timeout=0.001)
                    await self._process_single_context(*item)
                    continue
                except asyncio.TimeoutError:
                    pass
                
                # Process regular queue
                try:
                    item = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=self.config['processing_interval_ms'] / 1000.0
                    )
                    await self._process_single_context(*item)
                except asyncio.TimeoutError:
                    pass
                    
            except Exception as e:
                self.logger.error(f"Context processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_single_context(
        self, 
        context: DecisionContext, 
        priority: ProcessingPriority, 
        queue_time: float
    ) -> None:
        """
        Process a single decision context
        
        Args:
            context: Decision context to process
            priority: Processing priority
            queue_time: Time when context was queued
        """
        processing_start = time.perf_counter()
        
        try:
            # Extract features
            feature_vector = await self._extract_feature_vector(context)
            
            # Generate embeddings
            embedding_vector = await self._generate_embedding(context)
            
            # Calculate quality scores
            quality_scores = await self._calculate_quality_scores(context)
            
            # Format for LLM
            llm_context = await self._format_llm_context(context, feature_vector)
            
            # Create processed context
            processing_latency_ms = (time.perf_counter() - processing_start) * 1000
            queue_latency_ms = (processing_start - queue_time) * 1000
            
            processed_context = ProcessedContext(
                decision_id=context.decision_id,
                timestamp=context.timestamp,
                processing_latency_ms=processing_latency_ms,
                
                decision_summary=self._create_decision_summary(context),
                key_factors=self._extract_key_factors(context, feature_vector),
                risk_assessment=self._assess_risk_factors(context),
                confidence_breakdown=self._break_down_confidence(context),
                
                feature_vector=feature_vector,
                embedding_vector=embedding_vector,
                similarity_hash=self._generate_similarity_hash(feature_vector),
                
                llm_context=llm_context,
                explanation_template=self._select_explanation_template(context),
                context_quality_score=quality_scores['overall'],
                
                agent_performance=self._calculate_agent_performance(context),
                market_regime_score=quality_scores['market_regime'],
                decision_complexity_score=quality_scores['complexity'],
                
                priority=priority,
                target_audiences=self._determine_target_audiences(context),
                streaming_ready=True
            )
            
            # Cache the result
            cache_key = self._generate_cache_key(context)
            self._cache_processed_context(cache_key, processed_context)
            
            # Update metrics
            self._update_processing_metrics(processing_latency_ms, queue_latency_ms, quality_scores['overall'])
            
            # Publish processed context
            await self._publish_processed_context(processed_context)
            
            self.logger.debug(
                f"Processed context {context.decision_id}: "
                f"processing_ms={processing_latency_ms:.1f}, "
                f"quality={quality_scores['overall']:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process context {context.decision_id}: {e}")
            with self.metrics_lock:
                self.metrics.processing_errors += 1
    
    async def _extract_feature_vector(self, context: DecisionContext) -> np.ndarray:
        """
        Extract comprehensive feature vector from decision context
        
        Args:
            context: Decision context
            
        Returns:
            Feature vector array
        """
        feature_start = time.perf_counter()
        
        try:
            all_features = []
            
            # Extract features using registered extractors
            for extractor_name, extractor_func in self.feature_extractors.items():
                try:
                    features = await extractor_func(context)
                    all_features.extend(features)
                except Exception as e:
                    self.logger.debug(f"Feature extractor {extractor_name} failed: {e}")
                    # Add zero features as fallback
                    all_features.extend([0.0] * 10)  # Assume 10 features per extractor
            
            # Pad or truncate to target dimension
            target_dim = self.config['feature_vector_dim']
            if len(all_features) > target_dim:
                feature_vector = np.array(all_features[:target_dim])
            else:
                feature_vector = np.zeros(target_dim)
                feature_vector[:len(all_features)] = all_features
            
            # Normalize
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm
            
            # Update metrics
            feature_time = (time.perf_counter() - feature_start) * 1000
            with self.metrics_lock:
                self.metrics.feature_extraction_time_ms += feature_time
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.zeros(self.config['feature_vector_dim'])
    
    async def _extract_confidence_features(self, context: DecisionContext) -> List[float]:
        """Extract confidence-related features"""
        return [
            context.confidence,
            context.gating_confidence,
            1.0 - context.uncertainty,
            float(context.should_proceed),
            max(context.ensemble_probabilities) if context.ensemble_probabilities else 0.5,
            np.std(context.ensemble_probabilities) if context.ensemble_probabilities else 0.0,
            len([p for p in context.ensemble_probabilities if p > 0.3]) / 3.0 if context.ensemble_probabilities else 0.0,
            context.inference_time_ms / 10.0,  # Normalized by expected max (10ms)
            float(context.confidence > 0.8),
            float(context.uncertainty < 0.2)
        ]
    
    async def _extract_agent_consensus_features(self, context: DecisionContext) -> List[float]:
        """Extract agent consensus features"""
        agent_contributions = list(context.agent_contributions.values()) if context.agent_contributions else [0.33, 0.33, 0.34]
        gating_weights = context.gating_weights if context.gating_weights else [0.33, 0.33, 0.34]
        
        return [
            np.std(agent_contributions),  # Consensus spread
            max(agent_contributions) - min(agent_contributions),  # Consensus range
            np.mean(agent_contributions),  # Average contribution
            len([c for c in agent_contributions if c > 0.4]),  # Dominant agents
            np.std(gating_weights),  # Gating diversity
            max(gating_weights),  # Dominant gating weight
            np.corrcoef(agent_contributions, gating_weights)[0, 1] if len(agent_contributions) > 1 else 0.0,
            float(max(agent_contributions) > 0.6),  # Strong consensus
            float(np.std(agent_contributions) < 0.1),  # High agreement
            np.entropy(agent_contributions) if sum(agent_contributions) > 0 else 0.0  # Consensus entropy
        ]
    
    async def _extract_market_features(self, context: DecisionContext) -> List[float]:
        """Extract market-related features"""
        return [
            context.volatility,
            context.volume_ratio,
            self._classify_regime_numeric(context.regime_classification),
            self._get_momentum_strength(context.momentum_indicators),
            self._calculate_trend_strength(context.momentum_indicators),
            float(context.volatility > 0.03),  # High volatility flag
            float(context.volume_ratio > 1.5),  # High volume flag
            self._get_market_session_numeric(context.timestamp),
            self._calculate_market_stress_score(context),
            self._get_volatility_regime_score(context.volatility)
        ]
    
    async def _extract_temporal_features(self, context: DecisionContext) -> List[float]:
        """Extract temporal features"""
        now = datetime.now(timezone.utc)
        
        return [
            context.timestamp.hour / 24.0,  # Hour of day
            context.timestamp.weekday() / 7.0,  # Day of week
            (now - context.timestamp).total_seconds() / 3600.0,  # Age in hours
            context.capture_latency_ns / 1_000_000.0,  # Capture latency in ms
            context.inference_time_ms,
            float(context.timestamp.hour in range(9, 16)),  # Market hours
            float(context.timestamp.weekday() < 5),  # Weekday
            (context.timestamp.timestamp() % 86400) / 86400.0,  # Time of day normalized
            self._get_market_regime_persistence_score(),
            float((now - context.timestamp).total_seconds() < 60)  # Recent decision
        ]
    
    async def _extract_risk_features(self, context: DecisionContext) -> List[float]:
        """Extract risk-related features"""
        return [
            1.0 - context.confidence,  # Confidence risk
            context.uncertainty,
            context.volatility,
            self._calculate_position_risk_score(context),
            self._calculate_drawdown_risk(context),
            float(context.action in ['buy', 'sell']),  # Directional risk
            self._calculate_correlation_risk(context),
            self._calculate_liquidity_risk(context),
            self._calculate_regime_transition_risk(context),
            self._calculate_execution_risk(context)
        ]
    
    async def _extract_performance_features(self, context: DecisionContext) -> List[float]:
        """Extract performance-related features"""
        return [
            context.inference_time_ms / 5.0,  # Normalized by target (5ms)
            context.capture_latency_ns / 100_000.0,  # Normalized capture latency
            float(context.inference_time_ms < 5.0),  # Fast inference
            self._calculate_prediction_quality_score(context),
            self._calculate_consistency_score(context),
            self._get_recent_accuracy_score(),
            self._calculate_feature_importance_score(context),
            float(len(context.agent_contributions) >= 3),  # Full agent participation
            self._calculate_gating_effectiveness_score(context),
            self._calculate_overall_system_health_score()
        ]
    
    async def _generate_embedding(self, context: DecisionContext) -> Optional[np.ndarray]:
        """
        Generate semantic embedding for context
        
        Args:
            context: Decision context
            
        Returns:
            Embedding vector or None if unavailable
        """
        embedding_start = time.perf_counter()
        
        try:
            if not self.embedding_model:
                return None
            
            # Create text representation of context
            context_text = self._create_context_text(context)
            
            if SENTENCE_TRANSFORMERS_AVAILABLE and self.embedding_model:
                # Generate embedding using sentence transformer
                embedding = self.embedding_model.encode(context_text)
                embedding_vector = np.array(embedding, dtype=np.float32)
            else:
                # Simple hash-based embedding fallback
                embedding_vector = self._simple_embedding(context_text)
            
            # Update metrics
            embedding_time = (time.perf_counter() - embedding_start) * 1000
            with self.metrics_lock:
                self.metrics.embedding_generation_time_ms += embedding_time
            
            return embedding_vector
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            with self.metrics_lock:
                self.metrics.vector_generation_errors += 1
            return None
    
    def _create_context_text(self, context: DecisionContext) -> str:
        """Create text representation of context for embedding"""
        text_parts = [
            f"Trading decision: {context.action}",
            f"Confidence: {context.confidence:.2f}",
            f"Market volatility: {context.volatility:.3f}",
            f"Volume ratio: {context.volume_ratio:.2f}",
            f"Market regime: {context.regime_classification}",
            f"Reasoning: {context.reasoning[:200]}"  # Truncate reasoning
        ]
        
        # Add agent contributions
        if context.agent_contributions:
            for agent, contribution in context.agent_contributions.items():
                text_parts.append(f"{agent} contribution: {contribution:.2f}")
        
        return " | ".join(text_parts)
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple hash-based embedding fallback"""
        # Create hash-based embedding
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to vector
        embedding = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
        
        # Pad or truncate to embedding dimension
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        else:
            padded = np.zeros(self.embedding_dim)
            padded[:len(embedding)] = embedding
            embedding = padded
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    async def _calculate_quality_scores(self, context: DecisionContext) -> Dict[str, float]:
        """Calculate context quality scores"""
        scores = {}
        
        # Overall quality based on confidence and consensus
        scores['overall'] = (
            context.confidence * 0.4 +
            (1.0 - context.uncertainty) * 0.3 +
            self._calculate_consensus_quality(context) * 0.3
        )
        
        # Market regime quality
        scores['market_regime'] = self._calculate_market_regime_quality(context)
        
        # Decision complexity
        scores['complexity'] = self._calculate_decision_complexity(context)
        
        # Data completeness
        scores['completeness'] = self._calculate_data_completeness(context)
        
        return scores
    
    async def _format_llm_context(self, context: DecisionContext, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Format context for LLM processing
        
        Args:
            context: Decision context
            feature_vector: Extracted feature vector
            
        Returns:
            LLM-ready context dictionary
        """
        llm_start = time.perf_counter()
        
        try:
            llm_context = {
                # Core Decision Data
                'decision': {
                    'action': context.action,
                    'confidence': context.confidence,
                    'reasoning': context.reasoning,
                    'timestamp': context.timestamp.isoformat(),
                    'should_proceed': context.should_proceed
                },
                
                # Agent Analysis
                'agents': {
                    'contributions': context.agent_contributions,
                    'gating_weights': context.gating_weights,
                    'ensemble_probabilities': context.ensemble_probabilities,
                    'consensus_score': self._calculate_consensus_quality(context)
                },
                
                # Market Context
                'market': {
                    'volatility': context.volatility,
                    'volume_ratio': context.volume_ratio,
                    'regime': context.regime_classification,
                    'momentum': context.momentum_indicators,
                    'session': self._classify_market_session(context.timestamp)
                },
                
                # Performance Metrics
                'performance': {
                    'inference_time_ms': context.inference_time_ms,
                    'capture_latency_ns': context.capture_latency_ns,
                    'gating_confidence': context.gating_confidence,
                    'uncertainty': context.uncertainty
                },
                
                # Risk Assessment
                'risk': {
                    'overall_risk_score': self._calculate_overall_risk_score(context),
                    'volatility_risk': min(context.volatility / 0.05, 1.0),
                    'consensus_risk': 1.0 - self._calculate_consensus_quality(context),
                    'execution_risk': self._calculate_execution_risk(context)
                },
                
                # Feature Analysis
                'features': {
                    'top_features': self._get_top_feature_importances(feature_vector),
                    'feature_summary': self._summarize_features(feature_vector),
                    'quality_score': self._calculate_feature_quality(feature_vector)
                }
            }
            
            # Update metrics
            llm_time = (time.perf_counter() - llm_start) * 1000
            with self.metrics_lock:
                self.metrics.llm_formatting_time_ms += llm_time
            
            return llm_context
            
        except Exception as e:
            self.logger.error(f"LLM context formatting failed: {e}")
            return {'error': 'Context formatting failed'}
    
    def _create_decision_summary(self, context: DecisionContext) -> Dict[str, Any]:
        """Create high-level decision summary"""
        return {
            'action': context.action,
            'confidence': context.confidence,
            'certainty_level': self._classify_certainty_level(context.confidence),
            'consensus_strength': self._classify_consensus_strength(context),
            'market_conditions': self._summarize_market_conditions(context),
            'key_insight': self._generate_key_insight(context)
        }
    
    def _extract_key_factors(self, context: DecisionContext, feature_vector: np.ndarray) -> List[Dict[str, Any]]:
        """Extract key decision factors"""
        factors = []
        
        # Confidence factor
        factors.append({
            'factor': 'Decision Confidence',
            'value': context.confidence,
            'importance': 0.9 if context.confidence > 0.8 else 0.6,
            'description': f"System confidence in {context.action} decision"
        })
        
        # Agent consensus factor
        consensus_score = self._calculate_consensus_quality(context)
        factors.append({
            'factor': 'Agent Consensus',
            'value': consensus_score,
            'importance': 0.8 if consensus_score > 0.7 else 0.5,
            'description': f"Agreement level among trading agents"
        })
        
        # Market volatility factor
        factors.append({
            'factor': 'Market Volatility',
            'value': context.volatility,
            'importance': 0.7 if context.volatility > 0.03 else 0.4,
            'description': f"Current market volatility level"
        })
        
        return sorted(factors, key=lambda x: x['importance'], reverse=True)
    
    # Utility methods
    def _classify_regime_numeric(self, regime: str) -> float:
        """Convert regime classification to numeric value"""
        regime_map = {
            'trending': 1.0,
            'ranging': 0.5,
            'volatile': 0.8,
            'transitional': 0.3,
            'unknown': 0.0
        }
        return regime_map.get(regime, 0.0)
    
    def _get_momentum_strength(self, momentum_indicators: Dict[str, float]) -> float:
        """Calculate momentum strength from indicators"""
        if not momentum_indicators:
            return 0.0
        
        return np.mean([abs(v) for v in momentum_indicators.values()])
    
    def _calculate_trend_strength(self, momentum_indicators: Dict[str, float]) -> float:
        """Calculate trend strength"""
        if not momentum_indicators:
            return 0.0
        
        values = list(momentum_indicators.values())
        if len(values) < 2:
            return abs(values[0]) if values else 0.0
        
        # Consistency of momentum direction
        signs = [np.sign(v) for v in values]
        consistency = abs(np.mean(signs))
        magnitude = np.mean([abs(v) for v in values])
        
        return consistency * magnitude
    
    def _get_market_session_numeric(self, timestamp: datetime) -> float:
        """Get numeric representation of market session"""
        hour = timestamp.hour
        if 9 <= hour < 16:
            return 1.0  # Regular session
        elif 4 <= hour < 9 or 16 <= hour < 20:
            return 0.5  # Extended session
        else:
            return 0.0  # Closed
    
    def _calculate_consensus_quality(self, context: DecisionContext) -> float:
        """Calculate quality of agent consensus"""
        if not context.agent_contributions:
            return 0.5
        
        contributions = list(context.agent_contributions.values())
        
        # Higher quality when there's clear agreement but not complete dominance
        std_dev = np.std(contributions)
        max_contrib = max(contributions)
        
        # Ideal is strong agreement with some diversity
        if 0.4 <= max_contrib <= 0.7 and std_dev < 0.2:
            return 0.9
        elif max_contrib > 0.8:
            return 0.6  # Too much dominance
        else:
            return 0.3  # Poor consensus
    
    def _calculate_market_regime_quality(self, context: DecisionContext) -> float:
        """Calculate market regime detection quality"""
        if context.regime_classification == 'unknown':
            return 0.2
        
        # Higher quality for stable regimes with clear characteristics
        regime_scores = {
            'trending': 0.9,
            'ranging': 0.7,
            'volatile': 0.5,
            'transitional': 0.3
        }
        
        return regime_scores.get(context.regime_classification, 0.4)
    
    def _calculate_decision_complexity(self, context: DecisionContext) -> float:
        """Calculate decision complexity score"""
        complexity_factors = [
            context.uncertainty,
            1.0 - self._calculate_consensus_quality(context),
            min(context.volatility / 0.05, 1.0),
            float(context.regime_classification == 'transitional') * 0.5
        ]
        
        return np.mean(complexity_factors)
    
    def _calculate_data_completeness(self, context: DecisionContext) -> float:
        """Calculate data completeness score"""
        completeness_score = 0.0
        total_factors = 0
        
        # Check presence of key data
        factors = [
            (context.agent_contributions, 0.3),
            (context.gating_weights, 0.2),
            (context.ensemble_probabilities, 0.2),
            (context.momentum_indicators, 0.1),
            (context.reasoning, 0.1),
            (context.volatility > 0, 0.1)
        ]
        
        for factor, weight in factors:
            total_factors += weight
            if factor:
                completeness_score += weight
        
        return completeness_score / total_factors if total_factors > 0 else 0.0
    
    # Additional utility methods for risk calculations
    def _calculate_position_risk_score(self, context: DecisionContext) -> float:
        """Calculate position risk score"""
        # Placeholder - would integrate with position management
        if context.action in ['buy', 'sell']:
            return context.volatility * (1.0 - context.confidence)
        return 0.0
    
    def _calculate_overall_risk_score(self, context: DecisionContext) -> float:
        """Calculate overall risk score"""
        risk_factors = [
            context.uncertainty * 0.3,
            min(context.volatility / 0.05, 1.0) * 0.3,
            (1.0 - self._calculate_consensus_quality(context)) * 0.2,
            self._calculate_execution_risk(context) * 0.2
        ]
        
        return np.mean(risk_factors)
    
    def _calculate_execution_risk(self, context: DecisionContext) -> float:
        """Calculate execution risk"""
        # Higher risk for high volatility and low confidence
        vol_risk = min(context.volatility / 0.05, 1.0)
        conf_risk = 1.0 - context.confidence
        timing_risk = min(context.inference_time_ms / 10.0, 1.0)
        
        return np.mean([vol_risk, conf_risk, timing_risk])
    
    # Cache management methods
    def _generate_cache_key(self, context: DecisionContext) -> str:
        """Generate cache key for context"""
        key_data = f"{context.action}_{context.confidence:.2f}_{context.volatility:.3f}_{context.regime_classification}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_context(self, cache_key: str) -> Optional[ProcessedContext]:
        """Get cached processed context"""
        with self.cache_lock:
            return self.context_cache.get(cache_key)
    
    def _cache_processed_context(self, cache_key: str, processed_context: ProcessedContext) -> None:
        """Cache processed context"""
        with self.cache_lock:
            # Simple cache eviction - remove oldest if at capacity
            if len(self.context_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.context_cache))
                del self.context_cache[oldest_key]
            
            self.context_cache[cache_key] = processed_context
    
    def _generate_similarity_hash(self, feature_vector: np.ndarray) -> str:
        """Generate similarity hash for feature vector"""
        # Quantize vector for similarity matching
        quantized = (feature_vector * 1000).astype(int)
        return hashlib.md5(quantized.tobytes()).hexdigest()
    
    # Publishing methods
    async def _publish_processed_context(self, processed_context: ProcessedContext) -> None:
        """
        Publish processed context for streaming
        
        Args:
            processed_context: Processed decision context
        """
        try:
            event_payload = {
                'decision_id': processed_context.decision_id,
                'processed_context': asdict(processed_context),
                'priority': processed_context.priority.value,
                'streaming_ready': processed_context.streaming_ready,
                'timestamp': processed_context.timestamp.isoformat()
            }
            
            event = self.event_bus.create_event(
                EventType.INDICATOR_UPDATE,  # Temporary - will use XAI_CONTEXT_PROCESSED
                event_payload,
                source=self.name
            )
            
            self.event_bus.publish(event)
            
        except Exception as e:
            self.logger.error(f"Failed to publish processed context: {e}")
    
    def _update_processing_metrics(self, processing_time_ms: float, queue_time_ms: float, quality_score: float) -> None:
        """Update processing metrics"""
        with self.metrics_lock:
            self.metrics.total_contexts_processed += 1
            self.metrics.total_processing_time_ms += processing_time_ms
            
            self.metrics.avg_processing_time_ms = (
                self.metrics.total_processing_time_ms / self.metrics.total_contexts_processed
            )
            
            if processing_time_ms > self.metrics.max_processing_time_ms:
                self.metrics.max_processing_time_ms = processing_time_ms
            
            # Update cache hit rate
            if self.metrics.cache_hits + self.metrics.cache_misses > 0:
                self.metrics.cache_hit_rate = (
                    self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
                )
        
        # Add to performance windows
        self.processing_window.append(processing_time_ms)
        self.quality_window.append(quality_score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processor metrics"""
        with self.metrics_lock:
            metrics_dict = asdict(self.metrics)
        
        # Add performance analysis
        if self.processing_window:
            recent_times = list(self.processing_window)
            metrics_dict['performance_analysis'] = {
                'recent_avg_time_ms': np.mean(recent_times),
                'recent_max_time_ms': np.max(recent_times),
                'time_std_ms': np.std(recent_times),
                'processing_rate_per_sec': len(recent_times) / (max(recent_times) / 1000.0) if recent_times else 0.0
            }
        
        if self.quality_window:
            recent_quality = list(self.quality_window)
            metrics_dict['quality_analysis'] = {
                'avg_quality_score': np.mean(recent_quality),
                'quality_std': np.std(recent_quality),
                'high_quality_rate': len([q for q in recent_quality if q > 0.8]) / len(recent_quality)
            }
        
        # Add system status
        metrics_dict['system_status'] = {
            'active': self.active,
            'cache_size': len(self.context_cache),
            'embedding_model_loaded': self.embedding_model is not None,
            'initialized': self._initialized
        }
        
        return metrics_dict
    
    async def shutdown(self) -> None:
        """Shutdown the Context Processor"""
        try:
            self.active = False
            
            # Wait for processing task to complete
            if self.processing_task:
                try:
                    await asyncio.wait_for(self.processing_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self.processing_task.cancel()
            
            # Unsubscribe from events
            self.event_bus.unsubscribe(
                EventType.INDICATOR_UPDATE,
                self._handle_decision_context
            )
            
            self.logger.info(
                f"ContextProcessor shutdown complete: "
                f"final_metrics={self.get_metrics()}"
            )
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise
    
    # Placeholder methods for missing functionality
    def _classify_market_session(self, timestamp: datetime) -> str:
        """Classify market session"""
        hour = timestamp.hour
        if 9 <= hour < 16:
            return 'regular'
        elif 4 <= hour < 9 or 16 <= hour < 20:
            return 'extended'
        else:
            return 'closed'
    
    def _classify_certainty_level(self, confidence: float) -> str:
        """Classify certainty level"""
        if confidence > 0.8:
            return 'high'
        elif confidence > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _classify_consensus_strength(self, context: DecisionContext) -> str:
        """Classify consensus strength"""
        consensus_score = self._calculate_consensus_quality(context)
        if consensus_score > 0.8:
            return 'strong'
        elif consensus_score > 0.6:
            return 'moderate'
        else:
            return 'weak'
    
    def _summarize_market_conditions(self, context: DecisionContext) -> str:
        """Summarize market conditions"""
        conditions = []
        
        if context.volatility > 0.03:
            conditions.append('high volatility')
        if context.volume_ratio > 1.5:
            conditions.append('high volume')
        
        conditions.append(f"{context.regime_classification} regime")
        
        return ', '.join(conditions) if conditions else 'normal conditions'
    
    def _generate_key_insight(self, context: DecisionContext) -> str:
        """Generate key insight for decision"""
        if context.confidence > 0.8:
            return f"Strong {context.action} signal with high confidence"
        elif context.uncertainty > 0.5:
            return f"Uncertain market conditions affecting {context.action} decision"
        else:
            return f"Moderate {context.action} signal based on current conditions"
    
    # Additional missing methods (placeholders)
    def _assess_risk_factors(self, context: DecisionContext) -> Dict[str, Any]:
        """Assess risk factors"""
        return {
            'overall_risk': self._calculate_overall_risk_score(context),
            'volatility_risk': min(context.volatility / 0.05, 1.0),
            'execution_risk': self._calculate_execution_risk(context),
            'regime_risk': 0.8 if context.regime_classification == 'transitional' else 0.3
        }
    
    def _break_down_confidence(self, context: DecisionContext) -> Dict[str, float]:
        """Break down confidence components"""
        return {
            'model_confidence': context.confidence,
            'gating_confidence': context.gating_confidence,
            'consensus_confidence': self._calculate_consensus_quality(context),
            'timing_confidence': max(0.0, 1.0 - context.inference_time_ms / 10.0)
        }
    
    def _select_explanation_template(self, context: DecisionContext) -> str:
        """Select explanation template"""
        if context.confidence > 0.8:
            return "high_confidence_template"
        elif context.uncertainty > 0.5:
            return "uncertain_conditions_template"
        else:
            return "standard_explanation_template"
    
    def _calculate_agent_performance(self, context: DecisionContext) -> Dict[str, float]:
        """Calculate agent performance scores"""
        return {
            agent: contribution * context.confidence
            for agent, contribution in context.agent_contributions.items()
        }
    
    def _determine_target_audiences(self, context: DecisionContext) -> List[str]:
        """Determine target audiences for explanation"""
        audiences = ['trader']
        
        if context.confidence < 0.6 or context.uncertainty > 0.4:
            audiences.append('risk_manager')
        
        if context.volatility > 0.03:
            audiences.append('compliance')
        
        return audiences
    
    # Additional missing calculation methods
    def _calculate_market_stress_score(self, context: DecisionContext) -> float:
        """Calculate market stress score"""
        return min(context.volatility / 0.05, 1.0)
    
    def _get_volatility_regime_score(self, volatility: float) -> float:
        """Get volatility regime score"""
        if volatility > 0.05:
            return 1.0  # High volatility
        elif volatility > 0.02:
            return 0.5  # Medium volatility
        else:
            return 0.0  # Low volatility
    
    def _get_market_regime_persistence_score(self) -> float:
        """Get market regime persistence score"""
        return 0.5  # Placeholder
    
    def _calculate_drawdown_risk(self, context: DecisionContext) -> float:
        """Calculate drawdown risk"""
        return context.volatility * (1.0 - context.confidence)
    
    def _calculate_correlation_risk(self, context: DecisionContext) -> float:
        """Calculate correlation risk"""
        return 0.3  # Placeholder
    
    def _calculate_liquidity_risk(self, context: DecisionContext) -> float:
        """Calculate liquidity risk"""
        return max(0.0, 1.0 - context.volume_ratio)
    
    def _calculate_regime_transition_risk(self, context: DecisionContext) -> float:
        """Calculate regime transition risk"""
        return 0.8 if context.regime_classification == 'transitional' else 0.2
    
    def _calculate_prediction_quality_score(self, context: DecisionContext) -> float:
        """Calculate prediction quality score"""
        return context.confidence * self._calculate_consensus_quality(context)
    
    def _calculate_consistency_score(self, context: DecisionContext) -> float:
        """Calculate consistency score"""
        return 1.0 - context.uncertainty
    
    def _get_recent_accuracy_score(self) -> float:
        """Get recent accuracy score"""
        return 0.75  # Placeholder
    
    def _calculate_feature_importance_score(self, context: DecisionContext) -> float:
        """Calculate feature importance score"""
        return context.confidence
    
    def _calculate_gating_effectiveness_score(self, context: DecisionContext) -> float:
        """Calculate gating effectiveness score"""
        return context.gating_confidence
    
    def _calculate_overall_system_health_score(self) -> float:
        """Calculate overall system health score"""
        return 0.85  # Placeholder
    
    def _get_top_feature_importances(self, feature_vector: np.ndarray) -> List[Dict[str, Any]]:
        """Get top feature importances"""
        # Get indices of top features by magnitude
        top_indices = np.argsort(np.abs(feature_vector))[-5:][::-1]
        
        feature_names = [
            'confidence', 'consensus', 'volatility', 'momentum', 'risk_score',
            'volume', 'trend_strength', 'regime_score', 'timing', 'performance'
        ]
        
        return [
            {
                'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                'importance': float(feature_vector[i]),
                'rank': rank + 1
            }
            for rank, i in enumerate(top_indices)
        ]
    
    def _summarize_features(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Summarize feature vector"""
        return {
            'mean': float(np.mean(feature_vector)),
            'std': float(np.std(feature_vector)),
            'max': float(np.max(feature_vector)),
            'min': float(np.min(feature_vector)),
            'non_zero_count': int(np.count_nonzero(feature_vector))
        }
    
    def _calculate_feature_quality(self, feature_vector: np.ndarray) -> float:
        """Calculate feature quality score"""
        # Quality based on information content
        non_zero_ratio = np.count_nonzero(feature_vector) / len(feature_vector)
        magnitude = np.linalg.norm(feature_vector)
        diversity = np.std(feature_vector)
        
        return np.mean([non_zero_ratio, min(magnitude, 1.0), min(diversity, 1.0)])


# Test function
async def test_context_processor():
    """Test the Context Processor"""
    print("🧪 Testing Context Processor")
    
    # Mock kernel and event bus
    class MockKernel:
        def __init__(self):
            self.event_bus = EventBus()
    
    kernel = MockKernel()
    
    # Initialize context processor
    processor = ContextProcessor(kernel)
    await processor.initialize()
    
    # Create mock decision context
    mock_context = DecisionContext(
        decision_id="test-123",
        timestamp=datetime.now(timezone.utc),
        symbol="NQ",
        action="buy",
        confidence=0.85,
        
        strategic_decision={'test': 'data'},
        agent_contributions={'MLMI': 0.4, 'NWRQK': 0.35, 'Regime': 0.25},
        ensemble_probabilities=[0.1, 0.2, 0.7],
        gating_weights=[0.4, 0.35, 0.25],
        reasoning="Strong momentum signals detected",
        
        market_data={'volatility': 0.025},
        volatility=0.025,
        volume_ratio=1.2,
        momentum_indicators={'short': 0.02, 'long': 0.015},
        regime_classification='trending',
        
        inference_time_ms=2.5,
        gating_confidence=0.9,
        uncertainty=0.15,
        should_proceed=True,
        
        capture_latency_ns=75_000,
        processing_priority="high",
        client_connections=3
    )
    
    # Process context
    print("\n🔄 Processing decision context...")
    await processor._queue_context_for_processing(mock_context, ProcessingPriority.HIGH)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    # Check metrics
    metrics = processor.get_metrics()
    print(f"\n📊 Processor Metrics:")
    print(f"  Contexts processed: {metrics['total_contexts_processed']}")
    print(f"  Avg processing time: {metrics['avg_processing_time_ms']:.1f}ms")
    print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    print(f"  Embedding model loaded: {metrics['system_status']['embedding_model_loaded']}")
    
    # Shutdown
    await processor.shutdown()
    print("\n✅ Context Processor test complete!")


if __name__ == "__main__":
    asyncio.run(test_context_processor())