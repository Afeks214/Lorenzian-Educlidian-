"""
Real-time Decision Capture Pipeline

Agent Beta: Real-time streaming specialist
Mission: Zero-latency decision capture for instant explanations

This module implements the real-time decision capture pipeline that hooks into
Strategic MARL decision events and captures comprehensive decision context
with zero impact on trading performance.

Key Features:
- Zero-latency event hooking via async processing
- Comprehensive decision context capture
- Redis-backed queuing for reliability
- Performance monitoring and metrics
- Bulletproof error handling

Author: Agent Beta - Real-time Streaming Specialist
Version: 1.0 - Real-time Decision Capture Pipeline
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union
import json
import numpy as np
from collections import deque
import threading

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory queue fallback")

from ...core.events import EventType, Event, EventBus
from ...core.component_base import ComponentBase

logger = logging.getLogger(__name__)


@dataclass
class DecisionContext:
    """Comprehensive decision context for XAI processing"""
    
    # Core Decision Data
    decision_id: str
    timestamp: datetime
    symbol: str
    action: str
    confidence: float
    
    # Strategic MARL Context
    strategic_decision: Dict[str, Any]
    agent_contributions: Dict[str, float]
    ensemble_probabilities: List[float]
    gating_weights: List[float]
    reasoning: str
    
    # Market Context
    market_data: Dict[str, Any]
    volatility: float
    volume_ratio: float
    momentum_indicators: Dict[str, float]
    regime_classification: str
    
    # Performance Metrics
    inference_time_ms: float
    gating_confidence: float
    uncertainty: float
    should_proceed: bool
    
    # XAI Metadata
    capture_latency_ns: int
    processing_priority: str = "normal"
    client_connections: int = 0


@dataclass
class CaptureMetrics:
    """Metrics for decision capture performance"""
    
    total_decisions_captured: int = 0
    total_capture_time_ns: int = 0
    avg_capture_latency_ns: float = 0.0
    max_capture_latency_ns: int = 0
    min_capture_latency_ns: int = float('inf')
    
    queue_size: int = 0
    max_queue_size: int = 0
    queue_overflows: int = 0
    
    redis_connections: int = 0
    redis_errors: int = 0
    fallback_activations: int = 0
    
    memory_usage_mb: float = 0.0
    processing_errors: int = 0


class DecisionCapture(ComponentBase):
    """
    Real-time Decision Capture Pipeline
    
    Provides zero-latency capture of Strategic MARL decisions with comprehensive
    context extraction for real-time explanation generation.
    """
    
    def __init__(self, kernel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Decision Capture Pipeline
        
        Args:
            kernel: Reference to the AlgoSpace kernel
            config: Configuration dictionary
        """
        super().__init__("DecisionCapture", kernel)
        
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger('xai.decision_capture')
        
        # Redis connection for reliable queuing
        self.redis_client: Optional[redis.Redis] = None
        self.redis_connected = False
        
        # In-memory fallback queue
        self.fallback_queue: deque = deque(
            maxlen=self.config['fallback_queue_size']
        )
        
        # Performance metrics
        self.metrics = CaptureMetrics()
        self.metrics_lock = threading.Lock()
        
        # Event hooks for zero-latency capture
        self.decision_hooks: List[Callable] = []
        self.context_extractors: Dict[str, Callable] = {}
        
        # Processing state
        self.active = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Performance monitoring
        self.performance_window = deque(maxlen=1000)  # Last 1000 decisions
        self.alert_thresholds = {
            'max_latency_ns': self.config['max_capture_latency_ns'],
            'queue_size_warning': self.config['queue_size_warning'],
            'error_rate_threshold': 0.01  # 1% error rate threshold
        }
        
        self.logger.info(
            f"DecisionCapture initialized: "
            f"redis_enabled={REDIS_AVAILABLE}, "
            f"max_latency_ns={self.config['max_capture_latency_ns']}, "
            f"queue_size={self.config['queue_size']}"
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Performance Requirements
            'max_capture_latency_ns': 100_000,  # 100 microseconds
            'queue_size': 10000,
            'fallback_queue_size': 1000,
            'batch_size': 100,
            'processing_interval_ms': 10,
            
            # Redis Configuration
            'redis_url': 'redis://localhost:6379/1',
            'redis_key_prefix': 'xai:decisions',
            'redis_ttl_seconds': 3600,  # 1 hour TTL
            'redis_max_connections': 20,
            
            # Queue Management
            'queue_size_warning': 8000,
            'queue_overflow_action': 'drop_oldest',
            'priority_processing': True,
            
            # Error Handling
            'max_retries': 3,
            'retry_delay_ms': 100,
            'fallback_on_redis_error': True,
            
            # Monitoring
            'metrics_collection_enabled': True,
            'performance_logging_interval': 60,  # seconds
        }
    
    async def initialize(self) -> None:
        """Initialize the Decision Capture Pipeline"""
        try:
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Subscribe to Strategic MARL decision events
            self.event_bus.subscribe(
                EventType.STRATEGIC_DECISION, 
                self._handle_strategic_decision
            )
            
            # Register context extractors
            self._register_context_extractors()
            
            # Start processing task
            self.active = True
            self.processing_task = asyncio.create_task(self._process_queue())
            
            self._initialized = True
            self.logger.info("DecisionCapture initialized successfully")
            
            # Publish component started event
            event = self.event_bus.create_event(
                EventType.COMPONENT_STARTED,
                {"component": self.name, "status": "ready"},
                source=self.name
            )
            self.event_bus.publish(event)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DecisionCapture: {e}")
            raise
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for reliable queuing"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, using fallback queue only")
            return
        
        try:
            self.redis_client = redis.from_url(
                self.config['redis_url'],
                max_connections=self.config['redis_max_connections'],
                decode_responses=False
            )
            
            # Test connection
            await self.redis_client.ping()
            self.redis_connected = True
            
            with self.metrics_lock:
                self.metrics.redis_connections = 1
            
            self.logger.info(f"Redis connected: {self.config['redis_url']}")
            
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}, using fallback")
            self.redis_connected = False
            with self.metrics_lock:
                self.metrics.redis_errors += 1
    
    def _register_context_extractors(self) -> None:
        """Register context extraction functions"""
        self.context_extractors.update({
            'market_context': self._extract_market_context,
            'performance_context': self._extract_performance_context,
            'agent_context': self._extract_agent_context,
            'timing_context': self._extract_timing_context
        })
    
    def _handle_strategic_decision(self, event: Event) -> None:
        """
        Handle Strategic MARL decision events with zero-latency capture
        
        Args:
            event: Strategic decision event from MARL system
        """
        # Capture start time immediately for latency measurement
        capture_start_ns = time.perf_counter_ns()
        
        try:
            # Schedule async processing without blocking the event handler
            asyncio.create_task(
                self._capture_decision_async(event.payload, capture_start_ns)
            )
            
        except Exception as e:
            # Log error but don't raise to avoid impacting trading system
            self.logger.error(f"Error scheduling decision capture: {e}")
            with self.metrics_lock:
                self.metrics.processing_errors += 1
    
    async def _capture_decision_async(
        self, 
        decision_payload: Dict[str, Any], 
        capture_start_ns: int
    ) -> None:
        """
        Asynchronously capture decision with comprehensive context
        
        Args:
            decision_payload: Strategic decision payload
            capture_start_ns: Capture start timestamp in nanoseconds
        """
        try:
            # Extract core decision data
            decision_context = await self._build_decision_context(
                decision_payload, capture_start_ns
            )
            
            # Queue for processing (non-blocking)
            await self._queue_decision_context(decision_context)
            
            # Update metrics
            capture_latency_ns = time.perf_counter_ns() - capture_start_ns
            self._update_capture_metrics(capture_latency_ns)
            
            # Performance alerts
            if capture_latency_ns > self.alert_thresholds['max_latency_ns']:
                self.logger.warning(
                    f"High capture latency: {capture_latency_ns / 1000:.1f}μs "
                    f"(threshold: {self.alert_thresholds['max_latency_ns'] / 1000:.1f}μs)"
                )
            
        except Exception as e:
            self.logger.error(f"Decision capture failed: {e}")
            with self.metrics_lock:
                self.metrics.processing_errors += 1
    
    async def _build_decision_context(
        self, 
        decision_payload: Dict[str, Any], 
        capture_start_ns: int
    ) -> DecisionContext:
        """
        Build comprehensive decision context from Strategic MARL decision
        
        Args:
            decision_payload: Raw decision payload from Strategic MARL
            capture_start_ns: Capture start timestamp
            
        Returns:
            DecisionContext: Complete decision context for XAI processing
        """
        # Generate unique decision ID
        decision_id = str(uuid.uuid4())
        
        # Extract core decision data
        action = decision_payload.get('action', 'unknown')
        confidence = float(decision_payload.get('confidence', 0.0))
        reasoning = decision_payload.get('reasoning', '')
        
        # Extract agent contributions
        agent_contributions = decision_payload.get('agent_contributions', {})
        
        # Extract performance metrics
        performance_metrics = decision_payload.get('performance_metrics', {})
        ensemble_probabilities = performance_metrics.get('ensemble_probabilities', [0.33, 0.33, 0.34])
        gating_weights = performance_metrics.get('dynamic_weights', [0.33, 0.33, 0.34])
        gating_confidence = performance_metrics.get('gating_confidence', 0.5)
        
        # Extract market context using registered extractors
        market_context = {}
        for extractor_name, extractor_func in self.context_extractors.items():
            try:
                context_data = await extractor_func(decision_payload)
                market_context[extractor_name] = context_data
            except Exception as e:
                self.logger.debug(f"Context extractor {extractor_name} failed: {e}")
                market_context[extractor_name] = {}
        
        # Calculate capture latency
        capture_latency_ns = time.perf_counter_ns() - capture_start_ns
        
        return DecisionContext(
            decision_id=decision_id,
            timestamp=datetime.now(timezone.utc),
            symbol="NQ",  # Default symbol - should be extracted from payload
            action=action,
            confidence=confidence,
            
            strategic_decision=decision_payload,
            agent_contributions=agent_contributions,
            ensemble_probabilities=ensemble_probabilities,
            gating_weights=gating_weights,
            reasoning=reasoning,
            
            market_data=market_context.get('market_context', {}),
            volatility=market_context.get('market_context', {}).get('volatility', 0.02),
            volume_ratio=market_context.get('market_context', {}).get('volume_ratio', 1.0),
            momentum_indicators=market_context.get('market_context', {}).get('momentum', {}),
            regime_classification=market_context.get('market_context', {}).get('regime', 'unknown'),
            
            inference_time_ms=performance_metrics.get('inference_time_ms', 0.0),
            gating_confidence=gating_confidence,
            uncertainty=decision_payload.get('uncertainty', 0.0),
            should_proceed=decision_payload.get('should_proceed', True),
            
            capture_latency_ns=capture_latency_ns,
            processing_priority="high" if confidence > 0.8 else "normal"
        )
    
    async def _extract_market_context(self, decision_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market context from decision payload"""
        return {
            'volatility': 0.02,  # Would extract from actual market data
            'volume_ratio': 1.0,
            'momentum': {'short': 0.0, 'long': 0.0},
            'regime': 'normal',
            'trend_strength': 0.5
        }
    
    async def _extract_performance_context(self, decision_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance context from decision payload"""
        performance_metrics = decision_payload.get('performance_metrics', {})
        return {
            'inference_time_ms': performance_metrics.get('inference_time_ms', 0.0),
            'max_confidence': performance_metrics.get('max_confidence', 0.0),
            'min_confidence': performance_metrics.get('min_confidence', 0.0),
            'total_weight': performance_metrics.get('total_weight', 0.0)
        }
    
    async def _extract_agent_context(self, decision_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent-specific context from decision payload"""
        agent_contributions = decision_payload.get('agent_contributions', {})
        return {
            'mlmi_contribution': agent_contributions.get('MLMI', 0.0),
            'nwrqk_contribution': agent_contributions.get('NWRQK', 0.0),
            'regime_contribution': agent_contributions.get('Regime', 0.0),
            'consensus_achieved': len(agent_contributions) >= 2
        }
    
    async def _extract_timing_context(self, decision_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract timing context from decision payload"""
        timestamp_str = decision_payload.get('timestamp')
        if timestamp_str:
            decision_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            decision_timestamp = datetime.now(timezone.utc)
        
        return {
            'decision_timestamp': decision_timestamp.isoformat(),
            'processing_delay_ms': (datetime.now(timezone.utc) - decision_timestamp).total_seconds() * 1000,
            'market_session': self._classify_market_session(decision_timestamp)
        }
    
    def _classify_market_session(self, timestamp: datetime) -> str:
        """Classify market session based on timestamp"""
        hour = timestamp.hour
        if 9 <= hour < 16:
            return 'regular'
        elif 4 <= hour < 9 or 16 <= hour < 20:
            return 'extended'
        else:
            return 'closed'
    
    async def _queue_decision_context(self, context: DecisionContext) -> None:
        """
        Queue decision context for processing
        
        Args:
            context: Decision context to queue
        """
        context_data = asdict(context)
        context_json = json.dumps(context_data, default=str)
        
        # Try Redis first, fallback to in-memory queue
        if self.redis_connected and self.redis_client:
            try:
                key = f"{self.config['redis_key_prefix']}:{context.decision_id}"
                await self.redis_client.setex(
                    key, 
                    self.config['redis_ttl_seconds'], 
                    context_json
                )
                
                # Add to processing queue
                queue_key = f"{self.config['redis_key_prefix']}:queue"
                await self.redis_client.lpush(queue_key, context.decision_id)
                
                # Trim queue to max size
                await self.redis_client.ltrim(queue_key, 0, self.config['queue_size'] - 1)
                
                return
                
            except Exception as e:
                self.logger.error(f"Redis queue failed: {e}, using fallback")
                with self.metrics_lock:
                    self.metrics.redis_errors += 1
                self.redis_connected = False
        
        # Fallback to in-memory queue
        self.fallback_queue.append(context)
        with self.metrics_lock:
            self.metrics.fallback_activations += 1
            self.metrics.queue_size = len(self.fallback_queue)
            
            if self.metrics.queue_size > self.metrics.max_queue_size:
                self.metrics.max_queue_size = self.metrics.queue_size
    
    async def _process_queue(self) -> None:
        """Process queued decision contexts"""
        while self.active:
            try:
                # Process batch of contexts
                contexts = await self._get_batch_from_queue()
                
                if contexts:
                    # Publish contexts for explanation processing
                    for context in contexts:
                        await self._publish_context_event(context)
                    
                    self.logger.debug(f"Processed batch of {len(contexts)} decision contexts")
                
                # Small delay to avoid busy waiting
                await asyncio.sleep(self.config['processing_interval_ms'] / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1)  # Longer delay on error
    
    async def _get_batch_from_queue(self) -> List[DecisionContext]:
        """Get batch of contexts from queue"""
        contexts = []
        batch_size = self.config['batch_size']
        
        # Try Redis first
        if self.redis_connected and self.redis_client:
            try:
                queue_key = f"{self.config['redis_key_prefix']}:queue"
                
                for _ in range(batch_size):
                    decision_id = await self.redis_client.rpop(queue_key)
                    if not decision_id:
                        break
                    
                    context_key = f"{self.config['redis_key_prefix']}:{decision_id.decode()}"
                    context_data = await self.redis_client.get(context_key)
                    
                    if context_data:
                        context_dict = json.loads(context_data.decode())
                        # Convert dict back to DecisionContext
                        context = DecisionContext(**context_dict)
                        contexts.append(context)
                        
                        # Clean up processed context
                        await self.redis_client.delete(context_key)
                
                return contexts
                
            except Exception as e:
                self.logger.error(f"Redis batch retrieval failed: {e}")
                self.redis_connected = False
        
        # Fallback to in-memory queue
        for _ in range(min(batch_size, len(self.fallback_queue))):
            if self.fallback_queue:
                contexts.append(self.fallback_queue.popleft())
        
        with self.metrics_lock:
            self.metrics.queue_size = len(self.fallback_queue)
        
        return contexts
    
    async def _publish_context_event(self, context: DecisionContext) -> None:
        """
        Publish decision context event for downstream processing
        
        Args:
            context: Decision context to publish
        """
        try:
            # Create event for context processors and streaming engines
            event_payload = {
                'decision_id': context.decision_id,
                'context': asdict(context),
                'processing_priority': context.processing_priority,
                'timestamp': context.timestamp.isoformat()
            }
            
            event = self.event_bus.create_event(
                EventType.INDICATOR_UPDATE,  # Temporary - will add XAI_DECISION_CAPTURED
                event_payload,
                source=self.name
            )
            
            self.event_bus.publish(event)
            
        except Exception as e:
            self.logger.error(f"Failed to publish context event: {e}")
    
    def _update_capture_metrics(self, capture_latency_ns: int) -> None:
        """Update capture performance metrics"""
        with self.metrics_lock:
            self.metrics.total_decisions_captured += 1
            self.metrics.total_capture_time_ns += capture_latency_ns
            
            # Update averages
            self.metrics.avg_capture_latency_ns = (
                self.metrics.total_capture_time_ns / self.metrics.total_decisions_captured
            )
            
            # Update min/max
            if capture_latency_ns > self.metrics.max_capture_latency_ns:
                self.metrics.max_capture_latency_ns = capture_latency_ns
            
            if capture_latency_ns < self.metrics.min_capture_latency_ns:
                self.metrics.min_capture_latency_ns = capture_latency_ns
        
        # Add to performance window for trend analysis
        self.performance_window.append(capture_latency_ns)
    
    def add_decision_hook(self, hook: Callable[[DecisionContext], None]) -> None:
        """
        Add custom decision hook for additional processing
        
        Args:
            hook: Callable that receives DecisionContext
        """
        self.decision_hooks.append(hook)
        self.logger.info(f"Added decision hook: {hook.__name__}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive capture metrics"""
        with self.metrics_lock:
            metrics_dict = asdict(self.metrics)
        
        # Add performance analysis
        if self.performance_window:
            recent_latencies = list(self.performance_window)
            metrics_dict['performance_analysis'] = {
                'recent_avg_latency_ns': np.mean(recent_latencies),
                'recent_max_latency_ns': np.max(recent_latencies),
                'recent_min_latency_ns': np.min(recent_latencies),
                'latency_std_ns': np.std(recent_latencies),
                'latency_95th_percentile_ns': np.percentile(recent_latencies, 95),
                'samples_count': len(recent_latencies)
            }
        
        metrics_dict['system_status'] = {
            'active': self.active,
            'redis_connected': self.redis_connected,
            'initialized': self._initialized,
            'processing_task_running': self.processing_task is not None and not self.processing_task.done()
        }
        
        return metrics_dict
    
    async def shutdown(self) -> None:
        """Shutdown the Decision Capture Pipeline"""
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
                EventType.STRATEGIC_DECISION, 
                self._handle_strategic_decision
            )
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            self.logger.info(
                f"DecisionCapture shutdown complete: "
                f"final_metrics={self.get_metrics()}"
            )
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise


# Test function
async def test_decision_capture():
    """Test the Decision Capture Pipeline"""
    print("🧪 Testing Decision Capture Pipeline")
    
    # Mock kernel and event bus
    class MockKernel:
        def __init__(self):
            self.event_bus = EventBus()
    
    kernel = MockKernel()
    
    # Initialize decision capture
    capture = DecisionCapture(kernel)
    await capture.initialize()
    
    # Create mock strategic decision
    mock_decision = {
        'action': 'buy',
        'confidence': 0.85,
        'uncertainty': 0.15,
        'should_proceed': True,
        'reasoning': 'Strong momentum signals detected',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'agent_contributions': {
            'MLMI': 0.4,
            'NWRQK': 0.35,
            'Regime': 0.25
        },
        'performance_metrics': {
            'ensemble_probabilities': [0.1, 0.2, 0.7],
            'dynamic_weights': [0.4, 0.35, 0.25],
            'gating_confidence': 0.9,
            'inference_time_ms': 2.5
        }
    }
    
    # Create and publish event
    event = kernel.event_bus.create_event(
        EventType.STRATEGIC_DECISION,
        mock_decision,
        source="test_strategic_marl"
    )
    
    print("\n📡 Publishing mock strategic decision...")
    kernel.event_bus.publish(event)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check metrics
    metrics = capture.get_metrics()
    print(f"\n📊 Capture Metrics:")
    print(f"  Decisions captured: {metrics['total_decisions_captured']}")
    print(f"  Avg latency: {metrics['avg_capture_latency_ns'] / 1000:.1f}μs")
    print(f"  Queue size: {metrics['queue_size']}")
    print(f"  Redis connected: {metrics['system_status']['redis_connected']}")
    
    # Shutdown
    await capture.shutdown()
    print("\n✅ Decision Capture Pipeline test complete!")


if __name__ == "__main__":
    asyncio.run(test_decision_capture())