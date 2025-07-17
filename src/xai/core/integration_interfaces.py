"""
XAI Core Engine Integration Interfaces
Agent Alpha Mission: Seamless Integration Layer

Provides standardized interfaces for integrating the XAI Core Engine
with other system components for real-time trading explanations.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol, Union
from dataclasses import dataclass
from enum import Enum

from .vector_store import TradingDecisionVectorStore, TradingDecisionRecord, SearchResult
from .llm_engine import OllamaExplanationEngine, ExplanationContext, ExplanationResult, ExplanationStyle
from .embedding_pipeline import EmbeddingPipeline, EmbeddingRequest, EmbeddingType

logger = logging.getLogger(__name__)


class ExplanationPriority(Enum):
    """Priority levels for explanation requests"""
    CRITICAL = 1    # Real-time trading decisions
    HIGH = 2        # Risk alerts and compliance
    MEDIUM = 3      # Performance analysis
    LOW = 4         # Historical analysis


@dataclass
class TradingDecisionInput:
    """Standardized input for trading decisions requiring explanation"""
    decision_id: str
    symbol: str
    asset_class: str
    action: str
    confidence: float
    timestamp: datetime
    features: Dict[str, float]
    market_conditions: Dict[str, Any]
    agent_votes: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None


@dataclass
class ExplanationRequest:
    """Comprehensive explanation request"""
    request_id: str
    decision_input: TradingDecisionInput
    explanation_style: ExplanationStyle
    priority: ExplanationPriority
    target_audience: str  # trader, risk_manager, regulator, client
    context_window_hours: int = 24
    include_similar_decisions: bool = True
    max_similar_decisions: int = 5
    custom_context: Optional[Dict[str, Any]] = None


@dataclass
class ExplanationResponse:
    """Complete explanation response"""
    request_id: str
    explanation_text: str
    confidence_score: float
    generation_time_ms: float
    similar_decisions: List[SearchResult]
    context_factors: List[str]
    performance_metrics: Dict[str, Any]
    cached: bool
    error_message: Optional[str] = None


class TradingSystemInterface(Protocol):
    """Interface for trading system integration"""
    
    async def get_market_context(self, symbol: str, timestamp: datetime) -> Dict[str, Any]:
        """Get current market context for symbol"""
        ...
    
    async def get_position_info(self, symbol: str) -> Dict[str, Any]:
        """Get current position information"""
        ...
    
    async def get_risk_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get current risk metrics"""
        ...


class NotificationInterface(Protocol):
    """Interface for sending notifications"""
    
    async def send_explanation(
        self, 
        response: ExplanationResponse, 
        recipients: List[str]
    ) -> bool:
        """Send explanation to recipients"""
        ...
    
    async def send_alert(self, message: str, severity: str) -> bool:
        """Send system alert"""
        ...


class XAICoreEngineOrchestrator:
    """
    Central orchestrator for XAI Core Engine operations
    
    Coordinates vector store, LLM engine, and embedding pipeline
    to provide seamless explanation generation for trading systems.
    """
    
    def __init__(
        self,
        vector_store: TradingDecisionVectorStore,
        llm_engine: OllamaExplanationEngine,
        embedding_pipeline: EmbeddingPipeline,
        trading_system: Optional[TradingSystemInterface] = None,
        notification_system: Optional[NotificationInterface] = None
    ):
        """Initialize XAI Core Engine Orchestrator"""
        self.vector_store = vector_store
        self.llm_engine = llm_engine
        self.embedding_pipeline = embedding_pipeline
        self.trading_system = trading_system
        self.notification_system = notification_system
        
        # Request queue for prioritized processing
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Active request tracking
        self.active_requests: Dict[str, ExplanationRequest] = {}
        
        # Performance metrics
        self.orchestrator_metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'avg_processing_time_ms': 0.0,
            'queue_size': 0
        }
        
        # Background task for processing requests
        self.processing_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        logger.info("XAI Core Engine Orchestrator initialized")
    
    async def start(self):
        """Start the orchestrator background processing"""
        if not self.processing_task:
            self.processing_task = asyncio.create_task(self._process_requests())
            logger.info("XAI Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.shutdown_event.set()
        if self.processing_task:
            await self.processing_task
            self.processing_task = None
        logger.info("XAI Orchestrator stopped")
    
    async def request_explanation(
        self,
        request: ExplanationRequest
    ) -> ExplanationResponse:
        """
        Request explanation for trading decision
        
        Args:
            request: Explanation request with decision details
            
        Returns:
            ExplanationResponse: Generated explanation
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.debug(f"Processing explanation request {request.request_id}")
            
            # Store decision in vector database first
            await self._store_decision(request.decision_input)
            
            # Generate explanation context
            context = await self._build_explanation_context(request)
            
            # Find similar decisions
            similar_decisions = await self._find_similar_decisions(request, context)
            
            # Generate explanation
            explanation_result = await self.llm_engine.generate_explanation(
                context=context,
                style=request.explanation_style,
                use_cache=True
            )
            
            # Build response
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            response = ExplanationResponse(
                request_id=request.request_id,
                explanation_text=explanation_result.explanation_text,
                confidence_score=explanation_result.confidence_score,
                generation_time_ms=processing_time_ms,
                similar_decisions=similar_decisions,
                context_factors=explanation_result.context_factors,
                performance_metrics=self._get_performance_summary(),
                cached=explanation_result.cached
            )
            
            # Update metrics
            self._update_metrics(processing_time_ms, success=True)
            
            # Send notifications if configured
            if self.notification_system and request.priority == ExplanationPriority.CRITICAL:
                await self._send_notification(response, request)
            
            logger.debug(f"Explanation generated in {processing_time_ms:.1f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate explanation for {request.request_id}: {e}")
            
            processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            self._update_metrics(processing_time_ms, success=False)
            
            return ExplanationResponse(
                request_id=request.request_id,
                explanation_text=f"Unable to generate explanation: {str(e)}",
                confidence_score=0.0,
                generation_time_ms=processing_time_ms,
                similar_decisions=[],
                context_factors=[],
                performance_metrics=self._get_performance_summary(),
                cached=False,
                error_message=str(e)
            )
    
    async def _store_decision(self, decision_input: TradingDecisionInput):
        """Store trading decision in vector database"""
        
        record = TradingDecisionRecord(
            decision_id=decision_input.decision_id,
            timestamp=decision_input.timestamp,
            symbol=decision_input.symbol,
            asset_class=decision_input.asset_class,
            action=decision_input.action,
            confidence=decision_input.confidence,
            features=decision_input.features,
            market_conditions=decision_input.market_conditions,
            execution_result=decision_input.execution_result
        )
        
        success = await self.vector_store.store_trading_decision(record)
        if not success:
            logger.warning(f"Failed to store decision {decision_input.decision_id}")
    
    async def _build_explanation_context(
        self,
        request: ExplanationRequest
    ) -> ExplanationContext:
        """Build explanation context from request and external data"""
        
        decision = request.decision_input
        
        # Get additional context from trading system if available
        performance_metrics = None
        risk_metrics = None
        
        if self.trading_system:
            try:
                performance_metrics = await self.trading_system.get_market_context(
                    decision.symbol, decision.timestamp
                )
                risk_metrics = await self.trading_system.get_risk_metrics(decision.symbol)
            except Exception as e:
                logger.warning(f"Failed to get trading system context: {e}")
        
        # Build context
        context = ExplanationContext(
            symbol=decision.symbol,
            action=decision.action,
            confidence=decision.confidence,
            timestamp=decision.timestamp,
            market_features=decision.features,
            similar_decisions=[],  # Will be populated later
            performance_metrics=performance_metrics,
            risk_metrics=risk_metrics,
            user_preferences=request.custom_context
        )
        
        return context
    
    async def _find_similar_decisions(
        self,
        request: ExplanationRequest,
        context: ExplanationContext
    ) -> List[SearchResult]:
        """Find similar historical decisions"""
        
        if not request.include_similar_decisions:
            return []
        
        try:
            # Create query text for similarity search
            query_text = (
                f"{context.action} {context.symbol} with {context.confidence:.1%} confidence "
                f"based on market signals"
            )
            
            # Search for similar decisions
            similar_decisions = await self.vector_store.find_similar_decisions(
                query_text=query_text,
                symbol=context.symbol,
                time_window_hours=request.context_window_hours,
                limit=request.max_similar_decisions
            )
            
            # Update context with similar decisions
            context.similar_decisions = [
                {
                    'score': result.score,
                    'metadata': result.metadata,
                    'document': result.document
                }
                for result in similar_decisions
            ]
            
            return similar_decisions
            
        except Exception as e:
            logger.error(f"Failed to find similar decisions: {e}")
            return []
    
    async def _send_notification(
        self,
        response: ExplanationResponse,
        request: ExplanationRequest
    ):
        """Send notification for critical explanations"""
        
        if not self.notification_system:
            return
        
        try:
            # Determine recipients based on target audience
            recipients = self._get_notification_recipients(request.target_audience)
            
            await self.notification_system.send_explanation(response, recipients)
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _get_notification_recipients(self, target_audience: str) -> List[str]:
        """Get notification recipients for target audience"""
        
        # This would typically be configured externally
        recipient_map = {
            'trader': ['trading-desk@company.com'],
            'risk_manager': ['risk-team@company.com'],
            'regulator': ['compliance@company.com'],
            'client': ['client-services@company.com']
        }
        
        return recipient_map.get(target_audience, ['default@company.com'])
    
    def _update_metrics(self, processing_time_ms: float, success: bool):
        """Update orchestrator performance metrics"""
        
        self.orchestrator_metrics['total_requests'] += 1
        
        if success:
            self.orchestrator_metrics['completed_requests'] += 1
        else:
            self.orchestrator_metrics['failed_requests'] += 1
        
        # Update average processing time
        total_requests = self.orchestrator_metrics['total_requests']
        old_avg = self.orchestrator_metrics['avg_processing_time_ms']
        
        self.orchestrator_metrics['avg_processing_time_ms'] = (
            (old_avg * (total_requests - 1) + processing_time_ms) / total_requests
        )
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics from all components"""
        
        return {
            'orchestrator': self.orchestrator_metrics.copy(),
            'queue_size': self.request_queue.qsize()
        }
    
    async def _process_requests(self):
        """Background task to process queued requests"""
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for request with timeout
                try:
                    priority, request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process request
                response = await self.request_explanation(request)
                
                # Mark task as done
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in request processing: {e}")
                await asyncio.sleep(0.1)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        try:
            # Get health from all components
            vector_health = await self.vector_store.get_health_status()
            llm_health = await self.llm_engine.get_health_status()
            embedding_health = await self.embedding_pipeline.get_health_status()
            
            # Aggregate health status
            all_healthy = all(
                health.get('status') == 'healthy' 
                for health in [vector_health, llm_health, embedding_health]
            )
            
            # Check latency targets
            vector_latency_ok = vector_health.get('test_query_latency_ms', 999) < 100
            embedding_latency_ok = embedding_health.get('test_embedding_latency_ms', 999) < 50
            llm_latency_ok = llm_health.get('performance_stats', {}).get('avg_generation_time_ms', 999) < 2000
            
            overall_status = 'healthy' if all_healthy else 'degraded'
            
            return {
                'overall_status': overall_status,
                'components': {
                    'vector_store': vector_health,
                    'llm_engine': llm_health,
                    'embedding_pipeline': embedding_health
                },
                'performance_targets': {
                    'vector_query_target_met': vector_latency_ok,
                    'embedding_target_met': embedding_latency_ok,
                    'llm_target_met': llm_latency_ok,
                    'overall_target_met': vector_latency_ok and embedding_latency_ok
                },
                'orchestrator_metrics': self.orchestrator_metrics.copy(),
                'queue_status': {
                    'queue_size': self.request_queue.qsize(),
                    'processing_active': self.processing_task is not None and not self.processing_task.done()
                }
            }
            
        except Exception as e:
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'components': {},
                'performance_targets': {},
                'orchestrator_metrics': self.orchestrator_metrics.copy()
            }
    
    async def clear_all_caches(self):
        """Clear all component caches"""
        
        await asyncio.gather(
            self.vector_store.clear_cache(),
            self.llm_engine.clear_cache(),
            self.embedding_pipeline.clear_cache()
        )
        
        logger.info("All XAI component caches cleared")
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        vector_perf = await self.vector_store.get_performance_metrics()
        llm_perf = await self.llm_engine.get_performance_metrics()
        embedding_perf = await self.embedding_pipeline.get_performance_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'orchestrator': self.orchestrator_metrics.copy(),
            'vector_store': vector_perf,
            'llm_engine': llm_perf,
            'embedding_pipeline': embedding_perf,
            'summary': {
                'total_explanations_generated': self.orchestrator_metrics['completed_requests'],
                'success_rate': (
                    self.orchestrator_metrics['completed_requests'] / 
                    max(1, self.orchestrator_metrics['total_requests'])
                ),
                'avg_end_to_end_latency_ms': self.orchestrator_metrics['avg_processing_time_ms'],
                'latency_target_met': self.orchestrator_metrics['avg_processing_time_ms'] < 100
            }
        }


# Factory function for easy initialization

async def create_xai_core_engine(
    vector_store_config: Optional[Dict[str, Any]] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    embedding_config: Optional[Dict[str, Any]] = None
) -> XAICoreEngineOrchestrator:
    """
    Factory function to create and initialize XAI Core Engine
    
    Args:
        vector_store_config: Vector store configuration
        llm_config: LLM engine configuration
        embedding_config: Embedding pipeline configuration
        
    Returns:
        XAICoreEngineOrchestrator: Initialized orchestrator
    """
    
    from .vector_store import VectorStoreConfig
    from .llm_engine import OllamaConfig
    from .embedding_pipeline import EmbeddingConfig
    
    # Create components with provided or default configs
    vector_store = TradingDecisionVectorStore(
        VectorStoreConfig(**vector_store_config) if vector_store_config else VectorStoreConfig()
    )
    
    llm_engine = OllamaExplanationEngine(
        OllamaConfig(**llm_config) if llm_config else OllamaConfig()
    )
    
    embedding_pipeline = EmbeddingPipeline(
        EmbeddingConfig(**embedding_config) if embedding_config else EmbeddingConfig()
    )
    
    # Initialize LLM engine session
    await llm_engine._initialize_session()
    
    # Create orchestrator
    orchestrator = XAICoreEngineOrchestrator(
        vector_store=vector_store,
        llm_engine=llm_engine,
        embedding_pipeline=embedding_pipeline
    )
    
    # Start orchestrator
    await orchestrator.start()
    
    logger.info("XAI Core Engine created and initialized successfully")
    return orchestrator