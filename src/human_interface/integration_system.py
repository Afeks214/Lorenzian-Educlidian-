"""
Human Feedback Integration System

This module integrates the human feedback system with the existing MARL components,
creating a seamless bridge between expert decisions and AI training.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import structlog
import redis
from concurrent.futures import ThreadPoolExecutor

from .feedback_api import FeedbackAPI, DecisionPoint, ExpertChoice
from .choice_generator import ChoiceGenerator, AgentOutput, MarketSignal
from .rlhf_trainer import RLHFTrainer, PreferenceDatabase
from .security import SecurityManager
from ..core.event_bus import EventBus, Event, EventType
from ..core.config_manager import ConfigManager

logger = structlog.get_logger()


class HumanFeedbackCoordinator:
    """Coordinates human feedback collection and integration with MARL training"""
    
    def __init__(
        self,
        event_bus: EventBus,
        config_manager: ConfigManager,
        redis_client: redis.Redis
    ):
        self.event_bus = event_bus
        self.config_manager = config_manager
        self.redis_client = redis_client
        
        # Initialize components
        self.security_manager = SecurityManager(redis_client)
        self.preference_db = PreferenceDatabase()
        self.choice_generator = ChoiceGenerator(event_bus)
        self.feedback_api = FeedbackAPI(event_bus, redis_client)
        self.rlhf_trainer = RLHFTrainer(event_bus, self.preference_db)
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Active decision tracking
        self.pending_decisions: Dict[str, DecisionPoint] = {}
        self.decision_timeouts: Dict[str, datetime] = {}
        
        # Performance metrics
        self.metrics = {
            "decisions_created": 0,
            "expert_responses": 0,
            "training_updates": 0,
            "model_accuracy": 0.0,
            "response_time_avg": 0.0
        }
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Initialize default experts for demo
        self.security_manager.initialize_default_experts()
        
        logger.info("Human Feedback Coordinator initialized")

    def _setup_event_handlers(self):
        """Setup event handlers for system integration"""
        
        # Listen for new market data that might require expert input
        self.event_bus.subscribe(EventType.NEW_5MIN_BAR, self._handle_market_update)
        self.event_bus.subscribe(EventType.NEW_30MIN_BAR, self._handle_market_update)
        
        # Listen for MARL agent decisions
        self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._handle_agent_decision)
        
        # Listen for risk events that require immediate expert input
        self.event_bus.subscribe(EventType.RISK_BREACH, self._handle_risk_event)
        self.event_bus.subscribe(EventType.MARKET_STRESS, self._handle_market_stress)
        self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_event)
        
        # Listen for completed expert feedback
        self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._handle_expert_feedback)

    async def _handle_market_update(self, event: Event):
        """Handle new market data and check if expert input is needed"""
        try:
            bar_data = event.payload
            
            # Extract market context
            market_context = self._extract_market_context(bar_data)
            
            # Get current agent outputs
            agent_outputs = await self._get_current_agent_outputs(market_context)
            
            # Get market signals
            market_signals = await self._get_market_signals(market_context)
            
            # Check if expert input is needed
            decision_point = self.choice_generator.create_decision_point(
                market_context=market_context,
                agent_outputs=agent_outputs,
                market_signals=market_signals
            )
            
            if decision_point:
                await self._process_decision_point(decision_point)
                
        except Exception as e:
            logger.error("Error handling market update", error=str(e))

    async def _handle_agent_decision(self, event: Event):
        """Handle MARL agent decisions that may need expert validation"""
        try:
            if event.payload.get("type") == "expert_feedback":
                # This is feedback from an expert, handle it
                await self._process_expert_feedback(event.payload)
            else:
                # This is an agent decision, evaluate if expert input is needed
                agent_decision = event.payload
                
                # Check decision confidence and complexity
                if self._requires_expert_validation(agent_decision):
                    await self._request_expert_validation(agent_decision)
                    
        except Exception as e:
            logger.error("Error handling agent decision", error=str(e))

    async def _handle_risk_event(self, event: Event):
        """Handle risk events requiring immediate expert input"""
        try:
            risk_data = event.payload
            
            # Create high-priority decision point
            decision_point = self._create_risk_decision_point(risk_data)
            
            if decision_point:
                # Set shorter deadline for risk events
                decision_point.expert_deadline = datetime.now() + timedelta(minutes=5)
                decision_point.complexity = decision_point.complexity  # Keep existing complexity
                
                await self._process_decision_point(decision_point, priority=True)
                
        except Exception as e:
            logger.error("Error handling risk event", error=str(e))

    async def _handle_market_stress(self, event: Event):
        """Handle market stress events"""
        try:
            stress_data = event.payload
            
            # Notify all experts of market stress
            await self._broadcast_market_stress_alert(stress_data)
            
            # Create decision points for critical positions
            await self._create_stress_decision_points(stress_data)
            
        except Exception as e:
            logger.error("Error handling market stress", error=str(e))

    async def _handle_emergency_event(self, event: Event):
        """Handle emergency events requiring immediate expert intervention"""
        try:
            emergency_data = event.payload
            
            # Create critical decision point with 2-minute deadline
            decision_point = self._create_emergency_decision_point(emergency_data)
            
            if decision_point:
                await self._process_decision_point(decision_point, priority=True, urgent=True)
                
        except Exception as e:
            logger.error("Error handling emergency event", error=str(e))

    def _extract_market_context(self, bar_data) -> Any:
        """Extract market context from bar data"""
        # This would typically interface with existing market data systems
        # For now, creating a mock context
        from .feedback_api import MarketContext
        
        return MarketContext(
            symbol=bar_data.get('symbol', 'ETH-USD'),
            price=bar_data.get('close', 2000.0),
            volatility=np.random.uniform(0.01, 0.05),  # Mock volatility
            volume=bar_data.get('volume', 1000000),
            trend_strength=np.random.uniform(0.2, 0.8),  # Mock trend strength
            support_level=bar_data.get('close', 2000.0) * 0.98,
            resistance_level=bar_data.get('close', 2000.0) * 1.02,
            time_of_day="market_hours",
            market_regime="trending",
            correlation_shock=False
        )

    async def _get_current_agent_outputs(self, market_context) -> List[AgentOutput]:
        """Get current outputs from MARL agents"""
        # Mock agent outputs - in production, this would query actual agents
        return [
            AgentOutput(
                agent_id="mlmi_agent",
                action="momentum_long",
                confidence=0.75,
                reasoning="Strong momentum signals detected",
                risk_score=0.3,
                expected_return=0.05
            ),
            AgentOutput(
                agent_id="nwrqk_agent",
                action="conservative_hold",
                confidence=0.6,
                reasoning="Mixed signals, prefer caution",
                risk_score=0.2,
                expected_return=0.02
            )
        ]

    async def _get_market_signals(self, market_context) -> List[MarketSignal]:
        """Get current market signals"""
        # Mock market signals - in production, this would query signal systems
        return [
            MarketSignal(
                signal_type="momentum",
                strength=0.7,
                confidence=0.8,
                timeframe="5m",
                source="technical_analysis"
            ),
            MarketSignal(
                signal_type="volatility",
                strength=0.4,
                confidence=0.9,
                timeframe="30m",
                source="vol_surface"
            )
        ]

    async def _process_decision_point(self, decision_point: DecisionPoint, priority: bool = False, urgent: bool = False):
        """Process a decision point requiring expert input"""
        try:
            # Store decision point
            self.pending_decisions[decision_point.decision_id] = decision_point
            self.decision_timeouts[decision_point.decision_id] = decision_point.expert_deadline
            
            # Submit to feedback API
            success = await self.feedback_api.submit_decision_for_expert_input(decision_point)
            
            if success:
                self.metrics["decisions_created"] += 1
                
                # If urgent, send additional alerts
                if urgent:
                    await self._send_urgent_alerts(decision_point)
                
                logger.info(
                    "Decision point created for expert input",
                    decision_id=decision_point.decision_id,
                    complexity=decision_point.complexity.value,
                    priority=priority,
                    urgent=urgent
                )
            else:
                logger.error("Failed to submit decision for expert input")
                
        except Exception as e:
            logger.error("Error processing decision point", error=str(e))

    async def _process_expert_feedback(self, feedback_payload):
        """Process received expert feedback"""
        try:
            expert_choice = ExpertChoice(**feedback_payload["expert_choice"])
            decision_context = feedback_payload["decision_context"]
            
            # Store in preference database
            self.preference_db.store_expert_choice(expert_choice, decision_context)
            
            # Remove from pending decisions
            if expert_choice.decision_id in self.pending_decisions:
                del self.pending_decisions[expert_choice.decision_id]
                del self.decision_timeouts[expert_choice.decision_id]
            
            # Update metrics
            self.metrics["expert_responses"] += 1
            
            # Trigger RLHF training update if enough new data
            await self._check_training_trigger()
            
            logger.info(
                "Expert feedback processed",
                decision_id=expert_choice.decision_id,
                expert_id=expert_choice.expert_id,
                confidence=expert_choice.confidence
            )
            
        except Exception as e:
            logger.error("Error processing expert feedback", error=str(e))

    def _requires_expert_validation(self, agent_decision) -> bool:
        """Determine if agent decision requires expert validation"""
        # Check confidence thresholds
        confidence = agent_decision.get("confidence", 1.0)
        if confidence < 0.7:
            return True
        
        # Check for conflicting agent opinions
        if agent_decision.get("conflict_detected", False):
            return True
        
        # Check position size
        position_size = agent_decision.get("position_size", 0)
        if abs(position_size) > 5000:  # Large position threshold
            return True
        
        return False

    async def _request_expert_validation(self, agent_decision):
        """Request expert validation for an agent decision"""
        # Create decision point from agent decision
        # This would need to be implemented based on agent decision format
        pass

    def _create_risk_decision_point(self, risk_data) -> Optional[DecisionPoint]:
        """Create decision point for risk events"""
        # Implementation would depend on risk data format
        return None

    def _create_emergency_decision_point(self, emergency_data) -> Optional[DecisionPoint]:
        """Create decision point for emergency events"""
        # Implementation would depend on emergency data format
        return None

    async def _broadcast_market_stress_alert(self, stress_data):
        """Broadcast market stress alert to all experts"""
        # Implementation for alerting system
        pass

    async def _create_stress_decision_points(self, stress_data):
        """Create decision points for market stress scenarios"""
        # Implementation for stress-related decisions
        pass

    async def _send_urgent_alerts(self, decision_point: DecisionPoint):
        """Send urgent alerts for critical decisions"""
        # Implementation for urgent notification system
        pass

    async def _check_training_trigger(self):
        """Check if RLHF training should be triggered"""
        recent_feedback_count = len(self.preference_db.get_preference_records(limit=10))
        
        if recent_feedback_count >= 10:
            # Trigger training in background
            self.executor.submit(self._run_training_update)

    def _run_training_update(self):
        """Run RLHF training update"""
        try:
            results = self.rlhf_trainer.train_reward_model(epochs=5)
            self.metrics["training_updates"] += 1
            self.metrics["model_accuracy"] = results.get("val_accuracy", 0.0)
            
            logger.info("RLHF training update completed", results=results)
            
        except Exception as e:
            logger.error("Error in training update", error=str(e))

    async def cleanup_expired_decisions(self):
        """Clean up expired decisions that didn't receive expert input"""
        now = datetime.now()
        expired_decisions = []
        
        for decision_id, deadline in self.decision_timeouts.items():
            if now > deadline:
                expired_decisions.append(decision_id)
        
        for decision_id in expired_decisions:
            if decision_id in self.pending_decisions:
                decision = self.pending_decisions[decision_id]
                
                # Use model recommendation as fallback
                await self._apply_model_fallback(decision)
                
                # Clean up
                del self.pending_decisions[decision_id]
                del self.decision_timeouts[decision_id]
                
                logger.info(
                    "Decision expired, applied model fallback",
                    decision_id=decision_id
                )

    async def _apply_model_fallback(self, decision: DecisionPoint):
        """Apply model recommendation when expert input times out"""
        # Find the model-recommended strategy
        recommended_strategy = None
        for strategy in decision.strategies:
            if strategy.strategy_id == decision.model_recommendation:
                recommended_strategy = strategy
                break
        
        if recommended_strategy:
            # Execute the model's recommendation
            execution_event = self.event_bus.create_event(
                event_type=EventType.EXECUTE_TRADE,
                payload={
                    "strategy": recommended_strategy,
                    "context": decision.context,
                    "source": "model_fallback",
                    "reason": "expert_timeout"
                },
                source="human_feedback_coordinator"
            )
            
            self.event_bus.publish(execution_event)

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and metrics"""
        training_status = self.rlhf_trainer.get_training_status()
        
        return {
            "pending_decisions": len(self.pending_decisions),
            "active_experts": len(self.security_manager.expert_profiles),
            "metrics": self.metrics,
            "training_status": training_status,
            "system_health": "operational"
        }

    def get_feedback_api_app(self):
        """Get the FastAPI app for the feedback interface"""
        return self.feedback_api.get_api_app()

    async def start_monitoring_loop(self):
        """Start the monitoring loop for expired decisions"""
        while True:
            try:
                await self.cleanup_expired_decisions()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)

    def shutdown(self):
        """Shutdown the coordinator"""
        self.executor.shutdown(wait=True)
        logger.info("Human Feedback Coordinator shutdown complete")