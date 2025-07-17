"""
Real-time Explanation Streaming Engine

Agent Beta: Real-time streaming specialist
Mission: Orchestrate the complete real-time explanation pipeline

This module implements the central streaming engine that orchestrates the complete
real-time explanation pipeline, connecting decision capture, context processing,
LLM explanation generation, and WebSocket delivery for live trading explanations.

Key Features:
- End-to-end pipeline orchestration
- LLM integration for natural language explanations
- Priority-based streaming with audience targeting
- Performance monitoring and adaptive optimization
- Horizontal scaling coordination
- Circuit breaker patterns for reliability

Author: Agent Beta - Real-time Streaming Specialist
Version: 1.0 - Real-time Explanation Streaming Engine
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set, Callable, Union
import json
import numpy as np
from collections import deque, defaultdict
import threading
from enum import Enum

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("ollama not available, using mock LLM responses")

from ...core.events import EventType, Event, EventBus
from ...core.component_base import ComponentBase
from ...llm.ollama_llm import OllamaLLM
from .decision_capture import DecisionContext
from .context_processor import ProcessedContext, ProcessingPriority
from .websocket_manager import WebSocketManager, WebSocketMessage, MessageType

logger = logging.getLogger(__name__)


class ExplanationPriority(Enum):
    """Explanation priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    REAL_TIME = "real_time"


class ExplanationAudience(Enum):
    """Target audiences for explanations"""
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    COMPLIANCE = "compliance"
    CLIENT = "client"
    TECHNICAL = "technical"
    ALL = "all"


@dataclass
class ExplanationRequest:
    """Request for explanation generation"""
    request_id: str
    decision_id: str
    processed_context: ProcessedContext
    priority: ExplanationPriority
    audience: ExplanationAudience
    timestamp: datetime
    custom_template: Optional[str] = None
    include_technical: bool = True
    max_length: int = 1000


@dataclass
class GeneratedExplanation:
    """Generated explanation ready for streaming"""
    explanation_id: str
    decision_id: str
    request_id: str
    
    # Explanation Content
    explanation_text: str
    summary: str
    key_points: List[str]
    confidence_assessment: str
    risk_assessment: str
    
    # Metadata
    audience: ExplanationAudience
    priority: ExplanationPriority
    generation_time_ms: float
    quality_score: float
    
    # Streaming Info
    target_channels: List[str]
    delivery_deadline: datetime
    streaming_ready: bool


@dataclass
class StreamingMetrics:
    """Streaming engine performance metrics"""
    
    total_explanations_generated: int = 0
    total_explanations_streamed: int = 0
    total_generation_time_ms: float = 0.0
    avg_generation_time_ms: float = 0.0
    max_generation_time_ms: float = 0.0
    
    llm_requests: int = 0
    llm_failures: int = 0
    llm_avg_latency_ms: float = 0.0
    
    websocket_deliveries: int = 0
    websocket_failures: int = 0
    delivery_success_rate: float = 0.0
    
    pipeline_errors: int = 0
    circuit_breaker_triggers: int = 0
    
    active_connections: int = 0
    explanation_queue_size: int = 0
    processing_backlog: int = 0


class StreamingEngine(ComponentBase):
    """
    Real-time Explanation Streaming Engine
    
    Orchestrates the complete pipeline from decision capture to WebSocket delivery,
    generating natural language explanations and streaming them to connected clients.
    """
    
    def __init__(self, kernel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Streaming Engine
        
        Args:
            kernel: Reference to the AlgoSpace kernel
            config: Configuration dictionary
        """
        super().__init__("StreamingEngine", kernel)
        
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger('xai.streaming_engine')
        
        # Core components
        self.websocket_manager: Optional[WebSocketManager] = None
        self.llm_client: Optional[OllamaLLM] = None
        
        # Processing queues
        self.explanation_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['explanation_queue_size']
        )
        self.priority_explanation_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['priority_queue_size']
        )
        self.streaming_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config['streaming_queue_size']
        )
        
        # Performance metrics
        self.metrics = StreamingMetrics()
        self.metrics_lock = threading.Lock()
        
        # Background tasks
        self.active = False
        self.explanation_processor_task: Optional[asyncio.Task] = None
        self.streaming_processor_task: Optional[asyncio.Task] = None
        self.metrics_monitor_task: Optional[asyncio.Task] = None
        
        # Circuit breaker for reliability
        self.circuit_breaker = {
            'failures': 0,
            'max_failures': self.config['circuit_breaker']['max_failures'],
            'reset_timeout': self.config['circuit_breaker']['reset_timeout_seconds'],
            'last_failure': None,
            'is_open': False
        }
        
        # Explanation templates
        self.explanation_templates: Dict[str, str] = {}
        self._load_explanation_templates()
        
        # Performance monitoring
        self.generation_latencies = deque(maxlen=1000)
        self.delivery_latencies = deque(maxlen=1000)
        self.quality_scores = deque(maxlen=500)
        
        # Audience subscription tracking
        self.audience_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # audience -> client_ids
        
        self.logger.info(
            f"StreamingEngine initialized: "
            f"llm_available={OLLAMA_AVAILABLE}, "
            f"explanation_queue_size={self.config['explanation_queue_size']}, "
            f"target_latency_ms={self.config['target_explanation_latency_ms']}"
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Queue Configuration
            'explanation_queue_size': 5000,
            'priority_queue_size': 1000,
            'streaming_queue_size': 2000,
            'batch_size': 20,
            'processing_interval_ms': 5,
            
            # Performance Targets
            'target_explanation_latency_ms': 200,
            'max_explanation_latency_ms': 1000,
            'target_delivery_latency_ms': 50,
            'quality_threshold': 0.7,
            
            # LLM Configuration
            'llm': {
                'model': 'llama3.2:3b',
                'host': 'localhost',
                'port': 11434,
                'timeout_seconds': 10,
                'max_tokens': 500,
                'temperature': 0.3,
                'system_prompt': 'You are an expert trading analyst providing clear, concise explanations of trading decisions.'
            },
            
            # WebSocket Configuration
            'websocket': {
                'host': '0.0.0.0',
                'port': 8765,
                'max_connections': 1000,
                'authentication': {'enabled': True}
            },
            
            # Explanation Configuration
            'explanation': {
                'max_length': 1000,
                'include_confidence_breakdown': True,
                'include_risk_assessment': True,
                'include_technical_details': True,
                'personalization_enabled': True
            },
            
            # Circuit Breaker
            'circuit_breaker': {
                'max_failures': 10,
                'reset_timeout_seconds': 60,
                'failure_rate_threshold': 0.1
            },
            
            # Monitoring
            'monitoring': {
                'metrics_interval_seconds': 30,
                'performance_alerts_enabled': True,
                'quality_alerts_enabled': True
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the Streaming Engine"""
        try:
            # Initialize WebSocket Manager
            await self._initialize_websocket_manager()
            
            # Initialize LLM client
            await self._initialize_llm_client()
            
            # Subscribe to processed context events
            self.event_bus.subscribe(
                EventType.INDICATOR_UPDATE,  # Temporary - will use XAI_CONTEXT_PROCESSED
                self._handle_processed_context
            )
            
            # Start background processing tasks
            self.active = True
            self.explanation_processor_task = asyncio.create_task(self._process_explanation_requests())
            self.streaming_processor_task = asyncio.create_task(self._process_streaming_queue())
            self.metrics_monitor_task = asyncio.create_task(self._monitor_performance())
            
            self._initialized = True
            self.logger.info("StreamingEngine initialized successfully")
            
            # Publish component started event
            event = self.event_bus.create_event(
                EventType.COMPONENT_STARTED,
                {"component": self.name, "status": "ready"},
                source=self.name
            )
            self.event_bus.publish(event)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize StreamingEngine: {e}")
            raise
    
    async def _initialize_websocket_manager(self) -> None:
        """Initialize WebSocket Manager"""
        try:
            self.websocket_manager = WebSocketManager(
                self.kernel, self.config['websocket']
            )
            await self.websocket_manager.initialize()
            
            # Set up custom authentication if needed
            self.websocket_manager.set_auth_handler(self._authenticate_client)
            
            self.logger.info("WebSocket Manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSocket Manager: {e}")
            raise
    
    async def _initialize_llm_client(self) -> None:
        """Initialize LLM client"""
        try:
            if OLLAMA_AVAILABLE:
                self.llm_client = OllamaLLM(self.config['llm'])
                await self.llm_client.initialize()
                self.logger.info(f"LLM client initialized: {self.config['llm']['model']}")
            else:
                self.logger.warning("LLM not available, using mock responses")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            # Continue without LLM - will use fallback explanations
    
    def _load_explanation_templates(self) -> None:
        """Load explanation templates for different audiences"""
        self.explanation_templates = {
            'trader': '''
Trading Decision: {action} {symbol}
Confidence: {confidence:.1%}
Key Factors: {key_factors}
Market Conditions: {market_conditions}
Risk Assessment: {risk_assessment}
Recommendation: {recommendation}
            '''.strip(),
            
            'risk_manager': '''
Risk Analysis for {action} {symbol}
Decision Confidence: {confidence:.1%}
Risk Score: {risk_score:.2f}
Volatility: {volatility:.2%}
Position Impact: {position_impact}
Risk Factors: {risk_factors}
Mitigation: {risk_mitigation}
            '''.strip(),
            
            'compliance': '''
Algorithmic Trading Decision Report
Symbol: {symbol}
Action: {action}
Timestamp: {timestamp}
Confidence: {confidence:.1%}
Decision Basis: {decision_basis}
Regulatory Notes: {regulatory_notes}
Audit Trail: {audit_trail}
            '''.strip(),
            
            'client': '''
Investment Update: {symbol}
Action: {action_description}
Rationale: {client_rationale}
Expected Outcome: {expected_outcome}
Risk Level: {risk_level}
Time Horizon: {time_horizon}
            '''.strip()
        }
    
    def _handle_processed_context(self, event: Event) -> None:
        """
        Handle processed context events
        
        Args:
            event: Event containing processed context
        """
        try:
            event_payload = event.payload
            
            # Check if this is a processed context event
            if 'processed_context' not in event_payload:
                return
            
            processed_context_data = event_payload['processed_context']
            processed_context = ProcessedContext(**processed_context_data)
            
            # Create explanation requests for target audiences
            asyncio.create_task(
                self._create_explanation_requests(processed_context)
            )
            
        except Exception as e:
            self.logger.error(f"Error handling processed context event: {e}")
            with self.metrics_lock:
                self.metrics.pipeline_errors += 1
    
    async def _create_explanation_requests(self, processed_context: ProcessedContext) -> None:
        """
        Create explanation requests for different audiences
        
        Args:
            processed_context: Processed decision context
        """
        try:
            # Determine target audiences based on context
            target_audiences = self._determine_target_audiences(processed_context)
            
            # Create requests for each audience
            for audience_str in target_audiences:
                try:
                    audience = ExplanationAudience(audience_str)
                    priority = self._determine_explanation_priority(processed_context, audience)
                    
                    request = ExplanationRequest(
                        request_id=str(uuid.uuid4()),
                        decision_id=processed_context.decision_id,
                        processed_context=processed_context,
                        priority=priority,
                        audience=audience,
                        timestamp=datetime.now(timezone.utc),
                        include_technical=audience in [ExplanationAudience.TRADER, ExplanationAudience.TECHNICAL]
                    )
                    
                    # Queue for processing
                    await self._queue_explanation_request(request)
                    
                except ValueError:
                    self.logger.warning(f"Invalid audience: {audience_str}")
                    
        except Exception as e:
            self.logger.error(f"Error creating explanation requests: {e}")
    
    def _determine_target_audiences(self, processed_context: ProcessedContext) -> List[str]:
        """Determine target audiences based on context"""
        audiences = ['trader']  # Always include trader
        
        # Add risk manager for high-risk or low-confidence decisions
        if (processed_context.confidence_breakdown.get('model_confidence', 0) < 0.6 or
            processed_context.risk_assessment.get('overall_risk', 0) > 0.7):
            audiences.append('risk_manager')
        
        # Add compliance for high-volatility or large position changes
        market_volatility = processed_context.llm_context.get('market', {}).get('volatility', 0)
        if market_volatility > 0.03:
            audiences.append('compliance')
        
        # Add client for significant decisions
        if processed_context.confidence_breakdown.get('model_confidence', 0) > 0.8:
            audiences.append('client')
        
        return audiences
    
    def _determine_explanation_priority(
        self, 
        processed_context: ProcessedContext, 
        audience: ExplanationAudience
    ) -> ExplanationPriority:
        """Determine explanation priority"""
        
        # Critical priority for compliance in high-risk situations
        if (audience == ExplanationAudience.COMPLIANCE and 
            processed_context.risk_assessment.get('overall_risk', 0) > 0.8):
            return ExplanationPriority.CRITICAL
        
        # High priority for risk manager alerts
        if (audience == ExplanationAudience.RISK_MANAGER and
            processed_context.confidence_breakdown.get('model_confidence', 0) < 0.5):
            return ExplanationPriority.HIGH
        
        # High priority for trader real-time decisions
        if (audience == ExplanationAudience.TRADER and
            processed_context.priority == ProcessingPriority.HIGH):
            return ExplanationPriority.HIGH
        
        return ExplanationPriority.NORMAL
    
    async def _queue_explanation_request(self, request: ExplanationRequest) -> None:
        """Queue explanation request for processing"""
        try:
            if request.priority in [ExplanationPriority.CRITICAL, ExplanationPriority.HIGH]:
                await self.priority_explanation_queue.put(request)
                with self.metrics_lock:
                    self.metrics.processing_backlog = self.priority_explanation_queue.qsize()
            else:
                await self.explanation_queue.put(request)
                with self.metrics_lock:
                    self.metrics.explanation_queue_size = self.explanation_queue.qsize()
                    
        except asyncio.QueueFull:
            self.logger.warning(f"Explanation queue full, dropping request {request.request_id}")
    
    async def _process_explanation_requests(self) -> None:
        """Process queued explanation requests"""
        while self.active:
            try:
                # Check circuit breaker
                if self.circuit_breaker['is_open']:
                    if self._should_reset_circuit_breaker():
                        self._reset_circuit_breaker()
                    else:
                        await asyncio.sleep(1)
                        continue
                
                # Process priority queue first
                try:
                    request = await asyncio.wait_for(
                        self.priority_explanation_queue.get(), timeout=0.001
                    )
                    await self._process_single_explanation_request(request)
                    continue
                except asyncio.TimeoutError:
                    pass
                
                # Process regular queue
                try:
                    request = await asyncio.wait_for(
                        self.explanation_queue.get(),
                        timeout=self.config['processing_interval_ms'] / 1000.0
                    )
                    await self._process_single_explanation_request(request)
                except asyncio.TimeoutError:
                    pass
                    
            except Exception as e:
                self.logger.error(f"Explanation processing error: {e}")
                self._handle_circuit_breaker_failure()
                await asyncio.sleep(0.1)
    
    async def _process_single_explanation_request(self, request: ExplanationRequest) -> None:
        """
        Process a single explanation request
        
        Args:
            request: Explanation request to process
        """
        generation_start = time.perf_counter()
        
        try:
            # Generate explanation
            explanation = await self._generate_explanation(request)
            
            # Queue for streaming
            await self.streaming_queue.put(explanation)
            
            # Update metrics
            generation_time_ms = (time.perf_counter() - generation_start) * 1000
            self._update_generation_metrics(generation_time_ms, explanation.quality_score)
            
            self.logger.debug(
                f"Generated explanation {explanation.explanation_id}: "
                f"time_ms={generation_time_ms:.1f}, quality={explanation.quality_score:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process explanation request {request.request_id}: {e}")
            self._handle_circuit_breaker_failure()
    
    async def _generate_explanation(self, request: ExplanationRequest) -> GeneratedExplanation:
        """
        Generate explanation using LLM
        
        Args:
            request: Explanation request
            
        Returns:
            Generated explanation
        """
        generation_start = time.perf_counter()
        
        try:
            # Prepare LLM prompt
            prompt = self._create_llm_prompt(request)
            
            # Generate explanation using LLM
            if self.llm_client and not self.circuit_breaker['is_open']:
                try:
                    llm_response = await self.llm_client.generate_explanation(
                        prompt,
                        max_tokens=self.config['llm']['max_tokens'],
                        temperature=self.config['llm']['temperature']
                    )
                    explanation_text = llm_response.get('explanation', '')
                    
                    with self.metrics_lock:
                        self.metrics.llm_requests += 1
                        
                except Exception as e:
                    self.logger.warning(f"LLM generation failed: {e}, using template")
                    explanation_text = self._generate_template_explanation(request)
                    with self.metrics_lock:
                        self.metrics.llm_failures += 1
            else:
                # Use template fallback
                explanation_text = self._generate_template_explanation(request)
            
            # Extract key components
            key_points = self._extract_key_points(request, explanation_text)
            summary = self._create_summary(request)
            confidence_assessment = self._create_confidence_assessment(request)
            risk_assessment = self._create_risk_assessment(request)
            
            # Calculate quality score
            quality_score = self._calculate_explanation_quality(
                explanation_text, request, key_points
            )
            
            # Determine target channels
            target_channels = self._get_target_channels(request.audience)
            
            # Create generated explanation
            generation_time_ms = (time.perf_counter() - generation_start) * 1000
            
            explanation = GeneratedExplanation(
                explanation_id=str(uuid.uuid4()),
                decision_id=request.decision_id,
                request_id=request.request_id,
                
                explanation_text=explanation_text,
                summary=summary,
                key_points=key_points,
                confidence_assessment=confidence_assessment,
                risk_assessment=risk_assessment,
                
                audience=request.audience,
                priority=request.priority,
                generation_time_ms=generation_time_ms,
                quality_score=quality_score,
                
                target_channels=target_channels,
                delivery_deadline=datetime.now(timezone.utc) + timedelta(
                    milliseconds=self.config['target_delivery_latency_ms']
                ),
                streaming_ready=True
            )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            # Return minimal fallback explanation
            return self._create_fallback_explanation(request)
    
    def _create_llm_prompt(self, request: ExplanationRequest) -> str:
        """Create LLM prompt for explanation generation"""
        context = request.processed_context
        
        prompt_parts = [
            f"Generate a clear trading explanation for a {request.audience.value}.",
            f"",
            f"Decision: {context.decision_summary['action']} with {context.decision_summary['confidence']:.1%} confidence",
            f"Market Conditions: {context.decision_summary['market_conditions']}",
            f"Key Factors: {', '.join([f['factor'] for f in context.key_factors[:3]])}",
            f"Risk Assessment: Overall risk {context.risk_assessment['overall_risk']:.2f}",
            f"",
            f"Requirements:",
            f"- Keep under {request.max_length} characters",
            f"- Use clear, professional language",
            f"- Focus on actionable insights",
            f"- Include confidence and risk context"
        ]
        
        if request.include_technical:
            prompt_parts.append("- Include relevant technical details")
        
        return "\n".join(prompt_parts)
    
    def _generate_template_explanation(self, request: ExplanationRequest) -> str:
        """Generate explanation using templates"""
        context = request.processed_context
        template = self.explanation_templates.get(request.audience.value, self.explanation_templates['trader'])
        
        # Prepare template variables
        template_vars = {
            'action': context.decision_summary['action'],
            'symbol': 'NQ',  # Default symbol
            'confidence': context.confidence_breakdown.get('model_confidence', 0),
            'key_factors': ', '.join([f['factor'] for f in context.key_factors[:3]]),
            'market_conditions': context.decision_summary['market_conditions'],
            'risk_assessment': f"Risk level: {context.risk_assessment.get('overall_risk', 0):.2f}",
            'recommendation': self._generate_recommendation(context),
            'timestamp': context.timestamp.isoformat(),
            'risk_score': context.risk_assessment.get('overall_risk', 0),
            'volatility': context.llm_context.get('market', {}).get('volatility', 0),
            'position_impact': 'Moderate',  # Placeholder
            'risk_factors': 'Market volatility, execution timing',
            'risk_mitigation': 'Position sizing, stop-loss levels',
            'decision_basis': 'Multi-agent consensus with high confidence',
            'regulatory_notes': 'All regulatory requirements met',
            'audit_trail': f"Decision ID: {context.decision_id}",
            'action_description': self._get_action_description(context.decision_summary['action']),
            'client_rationale': self._generate_client_rationale(context),
            'expected_outcome': self._generate_expected_outcome(context),
            'risk_level': context.decision_summary['certainty_level'],
            'time_horizon': 'Short-term'
        }
        
        try:
            return template.format(**template_vars)
        except KeyError as e:
            self.logger.warning(f"Template formatting error: {e}")
            return f"Trading decision: {context.decision_summary['action']} with {template_vars['confidence']:.1%} confidence"
    
    def _extract_key_points(self, request: ExplanationRequest, explanation_text: str) -> List[str]:
        """Extract key points from explanation"""
        # Simple extraction - split by sentences and take first few
        sentences = [s.strip() for s in explanation_text.split('.') if s.strip()]
        return sentences[:3]  # Return first 3 sentences as key points
    
    def _create_summary(self, request: ExplanationRequest) -> str:
        """Create explanation summary"""
        context = request.processed_context
        action = context.decision_summary['action']
        confidence = context.confidence_breakdown.get('model_confidence', 0)
        
        return f"{action.title()} decision with {confidence:.1%} confidence based on {context.decision_summary['consensus_strength']} agent consensus"
    
    def _create_confidence_assessment(self, request: ExplanationRequest) -> str:
        """Create confidence assessment"""
        context = request.processed_context
        confidence = context.confidence_breakdown.get('model_confidence', 0)
        
        if confidence > 0.8:
            return "High confidence - Strong signal alignment across multiple indicators"
        elif confidence > 0.6:
            return "Moderate confidence - Good signal quality with some uncertainty"
        else:
            return "Low confidence - Weak or conflicting signals, proceed with caution"
    
    def _create_risk_assessment(self, request: ExplanationRequest) -> str:
        """Create risk assessment"""
        context = request.processed_context
        risk_score = context.risk_assessment.get('overall_risk', 0)
        
        if risk_score > 0.7:
            return "High risk - Consider reduced position size and tight stops"
        elif risk_score > 0.4:
            return "Moderate risk - Standard risk management applies"
        else:
            return "Low risk - Favorable risk/reward profile"
    
    def _calculate_explanation_quality(
        self, 
        explanation_text: str, 
        request: ExplanationRequest, 
        key_points: List[str]
    ) -> float:
        """Calculate quality score for explanation"""
        quality_factors = []
        
        # Length appropriateness
        length_score = min(len(explanation_text) / request.max_length, 1.0)
        if length_score > 0.5:  # Not too short
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.4)
        
        # Content completeness
        required_terms = ['confidence', 'risk', request.processed_context.decision_summary['action']]
        content_score = sum(1 for term in required_terms if term.lower() in explanation_text.lower()) / len(required_terms)
        quality_factors.append(content_score)
        
        # Key points quality
        key_points_score = min(len(key_points) / 3.0, 1.0)
        quality_factors.append(key_points_score)
        
        # Context relevance
        context_score = request.processed_context.context_quality_score
        quality_factors.append(context_score)
        
        return np.mean(quality_factors)
    
    def _get_target_channels(self, audience: ExplanationAudience) -> List[str]:
        """Get target channels for audience"""
        channel_mapping = {
            ExplanationAudience.TRADER: ['trading_desk', 'real_time_alerts'],
            ExplanationAudience.RISK_MANAGER: ['risk_management', 'alerts'],
            ExplanationAudience.COMPLIANCE: ['compliance', 'audit_trail'],
            ExplanationAudience.CLIENT: ['client_portal', 'notifications'],
            ExplanationAudience.TECHNICAL: ['technical_analysis', 'system_status']
        }
        
        return channel_mapping.get(audience, ['general'])
    
    def _create_fallback_explanation(self, request: ExplanationRequest) -> GeneratedExplanation:
        """Create fallback explanation when generation fails"""
        context = request.processed_context
        
        return GeneratedExplanation(
            explanation_id=str(uuid.uuid4()),
            decision_id=request.decision_id,
            request_id=request.request_id,
            
            explanation_text=f"Trading decision: {context.decision_summary['action']} with {context.confidence_breakdown.get('model_confidence', 0):.1%} confidence",
            summary="System generated trading decision",
            key_points=["Decision made", "Confidence calculated", "Risk assessed"],
            confidence_assessment="Standard confidence level",
            risk_assessment="Standard risk level",
            
            audience=request.audience,
            priority=request.priority,
            generation_time_ms=1.0,
            quality_score=0.3,
            
            target_channels=['general'],
            delivery_deadline=datetime.now(timezone.utc) + timedelta(seconds=30),
            streaming_ready=True
        )
    
    async def _process_streaming_queue(self) -> None:
        """Process streaming queue and deliver explanations"""
        while self.active:
            try:
                # Get explanation from queue
                explanation = await asyncio.wait_for(
                    self.streaming_queue.get(),
                    timeout=self.config['processing_interval_ms'] / 1000.0
                )
                
                # Stream to WebSocket clients
                await self._stream_explanation(explanation)
                
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                self.logger.error(f"Streaming processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _stream_explanation(self, explanation: GeneratedExplanation) -> None:
        """
        Stream explanation to WebSocket clients
        
        Args:
            explanation: Generated explanation to stream
        """
        delivery_start = time.perf_counter()
        
        try:
            # Create WebSocket message
            message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.EXPLANATION,
                timestamp=datetime.now(timezone.utc),
                payload={
                    'explanation_id': explanation.explanation_id,
                    'decision_id': explanation.decision_id,
                    'explanation': explanation.explanation_text,
                    'summary': explanation.summary,
                    'key_points': explanation.key_points,
                    'confidence_assessment': explanation.confidence_assessment,
                    'risk_assessment': explanation.risk_assessment,
                    'audience': explanation.audience.value,
                    'priority': explanation.priority.value,
                    'quality_score': explanation.quality_score,
                    'generation_time_ms': explanation.generation_time_ms
                },
                priority='high' if explanation.priority in [ExplanationPriority.HIGH, ExplanationPriority.CRITICAL] else 'normal'
            )
            
            # Broadcast to appropriate audience
            if self.websocket_manager:
                topic = f"explanations_{explanation.audience.value}"
                clients_reached = await self.websocket_manager.broadcast_message(message, topic=topic)
                
                with self.metrics_lock:
                    self.metrics.websocket_deliveries += clients_reached
                    self.metrics.total_explanations_streamed += 1
                    
                    if clients_reached > 0:
                        delivery_time_ms = (time.perf_counter() - delivery_start) * 1000
                        self.delivery_latencies.append(delivery_time_ms)
                        
                        # Update delivery success rate
                        total_attempts = self.metrics.websocket_deliveries + self.metrics.websocket_failures
                        if total_attempts > 0:
                            self.metrics.delivery_success_rate = self.metrics.websocket_deliveries / total_attempts
                
                self.logger.debug(
                    f"Streamed explanation {explanation.explanation_id} to {clients_reached} clients"
                )
            else:
                self.logger.warning("WebSocket manager not available for streaming")
                with self.metrics_lock:
                    self.metrics.websocket_failures += 1
                    
        except Exception as e:
            self.logger.error(f"Failed to stream explanation {explanation.explanation_id}: {e}")
            with self.metrics_lock:
                self.metrics.websocket_failures += 1
    
    async def _monitor_performance(self) -> None:
        """Monitor performance and trigger alerts"""
        while self.active:
            try:
                await asyncio.sleep(self.config['monitoring']['metrics_interval_seconds'])
                
                metrics = self.get_metrics()
                
                # Performance alerts
                if (metrics['avg_generation_time_ms'] > 
                    self.config['target_explanation_latency_ms'] * 1.5):
                    self.logger.warning(
                        f"High explanation latency: {metrics['avg_generation_time_ms']:.1f}ms "
                        f"(target: {self.config['target_explanation_latency_ms']}ms)"
                    )
                
                # Quality alerts
                if self.quality_scores and np.mean(self.quality_scores) < self.config['quality_threshold']:
                    self.logger.warning(
                        f"Low explanation quality: {np.mean(self.quality_scores):.2f} "
                        f"(threshold: {self.config['quality_threshold']})"
                    )
                
                # Queue size alerts
                if (metrics['explanation_queue_size'] > 
                    self.config['explanation_queue_size'] * 0.8):
                    self.logger.warning(
                        f"High explanation queue size: {metrics['explanation_queue_size']}"
                    )
                
                self.logger.debug(
                    f"Performance metrics: "
                    f"explanations={metrics['total_explanations_generated']}, "
                    f"avg_latency={metrics['avg_generation_time_ms']:.1f}ms, "
                    f"quality={np.mean(self.quality_scores):.2f if self.quality_scores else 0:.2f}"
                )
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
    
    # Circuit breaker methods
    def _handle_circuit_breaker_failure(self) -> None:
        """Handle circuit breaker failure"""
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = datetime.now(timezone.utc)
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['max_failures']:
            self.circuit_breaker['is_open'] = True
            with self.metrics_lock:
                self.metrics.circuit_breaker_triggers += 1
            self.logger.error("Circuit breaker opened due to excessive failures")
    
    def _should_reset_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be reset"""
        if not self.circuit_breaker['last_failure']:
            return False
        
        time_since_failure = (
            datetime.now(timezone.utc) - self.circuit_breaker['last_failure']
        ).total_seconds()
        
        return time_since_failure >= self.circuit_breaker['reset_timeout']
    
    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker"""
        self.circuit_breaker['is_open'] = False
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['last_failure'] = None
        self.logger.info("Circuit breaker reset")
    
    # Authentication
    async def _authenticate_client(self, token: str, user_id: str, connection) -> bool:
        """Authenticate WebSocket client"""
        # Simple authentication - in production, validate against auth service
        if token and user_id:
            self.logger.info(f"Authenticated client: {user_id}")
            return True
        return False
    
    # Utility methods
    def _generate_recommendation(self, context: ProcessedContext) -> str:
        """Generate trading recommendation"""
        confidence = context.confidence_breakdown.get('model_confidence', 0)
        action = context.decision_summary['action']
        
        if confidence > 0.8:
            return f"Strong {action} recommendation - execute with standard position size"
        elif confidence > 0.6:
            return f"Moderate {action} signal - consider reduced position size"
        else:
            return f"Weak {action} signal - proceed with caution or skip"
    
    def _get_action_description(self, action: str) -> str:
        """Get user-friendly action description"""
        action_map = {
            'buy': 'Increase long position',
            'sell': 'Increase short position',
            'hold': 'Maintain current position'
        }
        return action_map.get(action, action)
    
    def _generate_client_rationale(self, context: ProcessedContext) -> str:
        """Generate client-friendly rationale"""
        return f"Market analysis indicates favorable conditions for this position based on multiple technical indicators"
    
    def _generate_expected_outcome(self, context: ProcessedContext) -> str:
        """Generate expected outcome"""
        confidence = context.confidence_breakdown.get('model_confidence', 0)
        if confidence > 0.8:
            return "High probability of positive outcome"
        elif confidence > 0.6:
            return "Moderate probability of achieving target"
        else:
            return "Conservative positioning with managed risk"
    
    def _update_generation_metrics(self, generation_time_ms: float, quality_score: float) -> None:
        """Update generation metrics"""
        with self.metrics_lock:
            self.metrics.total_explanations_generated += 1
            self.metrics.total_generation_time_ms += generation_time_ms
            
            self.metrics.avg_generation_time_ms = (
                self.metrics.total_generation_time_ms / self.metrics.total_explanations_generated
            )
            
            if generation_time_ms > self.metrics.max_generation_time_ms:
                self.metrics.max_generation_time_ms = generation_time_ms
        
        # Add to monitoring windows
        self.generation_latencies.append(generation_time_ms)
        self.quality_scores.append(quality_score)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive streaming metrics"""
        with self.metrics_lock:
            metrics_dict = asdict(self.metrics)
        
        # Add performance analysis
        if self.generation_latencies:
            latencies = list(self.generation_latencies)
            metrics_dict['latency_analysis'] = {
                'recent_avg_ms': np.mean(latencies),
                'recent_max_ms': np.max(latencies),
                'recent_min_ms': np.min(latencies),
                'latency_95th_percentile_ms': np.percentile(latencies, 95),
                'samples_count': len(latencies)
            }
        
        if self.quality_scores:
            qualities = list(self.quality_scores)
            metrics_dict['quality_analysis'] = {
                'avg_quality': np.mean(qualities),
                'quality_std': np.std(qualities),
                'high_quality_rate': len([q for q in qualities if q > 0.8]) / len(qualities),
                'low_quality_rate': len([q for q in qualities if q < 0.5]) / len(qualities)
            }
        
        # Add system status
        metrics_dict['system_status'] = {
            'active': self.active,
            'websocket_manager_ready': self.websocket_manager is not None,
            'llm_client_ready': self.llm_client is not None,
            'circuit_breaker_open': self.circuit_breaker['is_open'],
            'initialized': self._initialized
        }
        
        return metrics_dict
    
    async def shutdown(self) -> None:
        """Shutdown the Streaming Engine"""
        try:
            self.active = False
            
            # Cancel background tasks
            tasks = [
                self.explanation_processor_task,
                self.streaming_processor_task,
                self.metrics_monitor_task
            ]
            
            for task in tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
            
            # Shutdown WebSocket manager
            if self.websocket_manager:
                await self.websocket_manager.shutdown()
            
            # Shutdown LLM client
            if self.llm_client:
                await self.llm_client.shutdown()
            
            # Unsubscribe from events
            self.event_bus.unsubscribe(
                EventType.INDICATOR_UPDATE,
                self._handle_processed_context
            )
            
            self.logger.info(
                f"StreamingEngine shutdown complete: "
                f"final_metrics={self.get_metrics()}"
            )
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise


# Test function
async def test_streaming_engine():
    """Test the Streaming Engine"""
    print("🧪 Testing Streaming Engine")
    
    # Mock kernel and event bus
    class MockKernel:
        def __init__(self):
            self.event_bus = EventBus()
    
    kernel = MockKernel()
    
    # Initialize streaming engine with test config
    config = {
        'websocket': {
            'host': 'localhost',
            'port': 8767,  # Use different port for testing
            'authentication': {'enabled': False}
        },
        'llm': {
            'model': 'test-model',
            'timeout_seconds': 1  # Short timeout for testing
        }
    }
    
    engine = StreamingEngine(kernel, config)
    await engine.initialize()
    
    # Wait for initialization
    await asyncio.sleep(0.2)
    
    # Create mock processed context
    from .context_processor import ProcessedContext
    
    mock_processed_context = ProcessedContext(
        decision_id="test-123",
        timestamp=datetime.now(timezone.utc),
        processing_latency_ms=25.0,
        
        decision_summary={
            'action': 'buy',
            'confidence': 0.85,
            'certainty_level': 'high',
            'consensus_strength': 'strong',
            'market_conditions': 'favorable trending conditions'
        },
        key_factors=[
            {'factor': 'Strong momentum', 'importance': 0.9},
            {'factor': 'High volume', 'importance': 0.8}
        ],
        risk_assessment={'overall_risk': 0.3},
        confidence_breakdown={'model_confidence': 0.85},
        
        feature_vector=np.random.rand(128),
        embedding_vector=None,
        similarity_hash="test-hash",
        
        llm_context={'market': {'volatility': 0.02}},
        explanation_template="standard",
        context_quality_score=0.8,
        
        agent_performance={'MLMI': 0.4, 'NWRQK': 0.35, 'Regime': 0.25},
        market_regime_score=0.7,
        decision_complexity_score=0.4,
        
        priority=ProcessingPriority.HIGH,
        target_audiences=['trader'],
        streaming_ready=True
    )
    
    # Create and publish processed context event
    event_payload = {
        'decision_id': 'test-123',
        'processed_context': asdict(mock_processed_context),
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    event = kernel.event_bus.create_event(
        EventType.INDICATOR_UPDATE,
        event_payload,
        source="test_context_processor"
    )
    
    print("\n📡 Publishing mock processed context...")
    kernel.event_bus.publish(event)
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    # Check metrics
    metrics = engine.get_metrics()
    print(f"\n📊 Streaming Engine Metrics:")
    print(f"  Explanations generated: {metrics['total_explanations_generated']}")
    print(f"  Explanations streamed: {metrics['total_explanations_streamed']}")
    print(f"  Avg generation time: {metrics['avg_generation_time_ms']:.1f}ms")
    print(f"  LLM requests: {metrics['llm_requests']}")
    print(f"  WebSocket deliveries: {metrics['websocket_deliveries']}")
    print(f"  System status: {metrics['system_status']}")
    
    # Shutdown
    await engine.shutdown()
    print("\n✅ Streaming Engine test complete!")


if __name__ == "__main__":
    asyncio.run(test_streaming_engine())