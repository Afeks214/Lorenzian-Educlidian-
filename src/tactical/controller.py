"""
Tactical MARL Controller - Event-Driven Decision Engine

High-frequency controller that listens for SYNERGY_DETECTED events
via Redis Streams and orchestrates tactical trading decisions.
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import torch

try:
    import redis.asyncio as redis
    from redis.asyncio.client import Redis
    from redis.exceptions import ConnectionError, TimeoutError, LockError
except ImportError:
    # For testing without redis installed
    redis = None
    Redis = None
    ConnectionError = Exception
    TimeoutError = Exception
    LockError = Exception

# Import tactical components
from src.monitoring.tactical_metrics import tactical_metrics
from src.tactical.environment import TacticalEnvironment
from src.tactical.aggregator import TacticalDecisionAggregator
from src.matrix.assembler_5m import MatrixAssembler5m
from src.tactical.async_inference_pool import AsyncInferencePool, get_global_inference_pool

logger = logging.getLogger(__name__)

@dataclass
class SynergyEvent:
    """Structure for SYNERGY_DETECTED events."""
    synergy_type: str
    direction: int  # 1 for long, -1 for short
    confidence: float
    signal_sequence: List[Dict[str, Any]]
    market_context: Dict[str, Any]
    correlation_id: str
    timestamp: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SynergyEvent':
        """Create SynergyEvent from dictionary."""
        return cls(
            synergy_type=data.get('synergy_type', ''),
            direction=data.get('direction', 0),
            confidence=data.get('confidence', 0.0),
            signal_sequence=data.get('signal_sequence', []),
            market_context=data.get('market_context', {}),
            correlation_id=data.get('correlation_id', ''),
            timestamp=data.get('timestamp', time.time())
        )

@dataclass
class TacticalDecision:
    """Structure for tactical decisions."""
    action: str  # "long", "short", "hold"
    confidence: float
    agent_votes: List[Dict[str, Any]]
    consensus_breakdown: Dict[str, float]
    execution_command: Dict[str, Any]
    timing: Dict[str, float]
    correlation_id: str
    timestamp: float

class TacticalMARLController:
    """
    Event-driven tactical MARL controller.
    
    Orchestrates the tactical decision-making process by:
    1. Listening for SYNERGY_DETECTED events via Redis Streams
    2. Fetching current 5-minute matrix state
    3. Getting decisions from tactical agents
    4. Aggregating decisions with confidence weighting
    5. Executing trades when consensus is reached
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        """Initialize tactical controller."""
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        
        # Stream configuration
        self.stream_name = "synergy_events"
        self.consumer_group = "tactical_group"
        self.consumer_name = "tactical_consumer_1"
        
        # Components
        self.tactical_env = TacticalEnvironment()
        self.decision_aggregator = TacticalDecisionAggregator()
        self.matrix_assembler = MatrixAssembler5m({})  # Will be configured properly
        self.inference_pool: Optional[AsyncInferencePool] = None
        
        # State tracking
        self.current_position = None
        self.last_decision_time = None
        self.decisions_processed = 0
        self.running = False
        
        # Performance tracking
        self.processing_times = []
        self.decision_history = []
        
        logger.info("Tactical MARL Controller initialized")
    
    async def initialize(self):
        """Initialize Redis connection and consumer group."""
        try:
            # Connect to Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Create consumer group (ignore error if exists)
            try:
                await self.redis_client.xgroup_create(
                    self.stream_name,
                    self.consumer_group,
                    id='0',
                    mkstream=True
                )
                logger.info(f"Created consumer group: {self.consumer_group}")
            except Exception as e:
                if "BUSYGROUP" in str(e):
                    logger.info(f"Consumer group already exists: {self.consumer_group}")
                else:
                    logger.error(f"Failed to create consumer group: {e}")
                    raise
            
            # Initialize tactical environment
            await self.tactical_env.initialize()
            
            # Initialize async inference pool
            self.inference_pool = await get_global_inference_pool()
            
            logger.info("Tactical controller initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tactical controller: {e}")
            raise
    
    async def start_event_listener(self):
        """Start listening for SYNERGY_DETECTED events."""
        if self.running:
            logger.warning("Event listener already running")
            return
        
        self.running = True
        logger.info("🎯 Starting tactical event listener")
        
        try:
            while self.running:
                try:
                    # Read new messages from stream
                    messages = await self.redis_client.xreadgroup(
                        self.consumer_group,
                        self.consumer_name,
                        {self.stream_name: '>'},
                        count=1,
                        block=1000  # Block for 1 second
                    )
                    
                    for stream, stream_messages in messages:
                        for message_id, fields in stream_messages:
                            await self._process_synergy_event(
                                message_id.decode(),
                                fields
                            )
                    
                    # Process any pending messages
                    await self._process_pending_messages()
                    
                except asyncio.CancelledError:
                    logger.info("Event listener cancelled")
                    break
                except (ConnectionError, TimeoutError) as e:
                    logger.error(f"Redis connection error: {e}")
                    await asyncio.sleep(5)  # Retry after 5 seconds
                except Exception as e:
                    logger.error(f"Error in event listener: {e}")
                    tactical_metrics.record_error(
                        error_type="event_listener_error",
                        component="controller",
                        severity="error"
                    )
                    await asyncio.sleep(1)  # Brief pause before retry
        
        finally:
            self.running = False
            logger.info("🛑 Tactical event listener stopped")
    
    async def _process_synergy_event(self, message_id: str, fields: Dict[bytes, bytes]):
        """Process a SYNERGY_DETECTED event."""
        start_time = time.perf_counter()
        
        try:
            # Parse event data
            event_data = {}
            for key, value in fields.items():
                try:
                    event_data[key.decode()] = json.loads(value.decode())
                except json.JSONDecodeError:
                    event_data[key.decode()] = value.decode()
            
            synergy_event = SynergyEvent.from_dict(event_data)
            
            logger.info(
                f"Processing synergy event: {synergy_event.synergy_type} "
                f"(confidence: {synergy_event.confidence:.3f}, "
                f"direction: {synergy_event.direction})",
                correlation_id=synergy_event.correlation_id
            )
            
            # Record event processing
            tactical_metrics.record_stream_processing(
                stream_name=self.stream_name,
                consumer_group=self.consumer_group,
                status="processing"
            )
            
            # Process the event
            async with tactical_metrics.measure_decision_latency(
                synergy_type=synergy_event.synergy_type,
                decision_type="synergy_response"
            ):
                decision = await self.on_synergy_detected(synergy_event)
            
            # Record successful processing
            tactical_metrics.record_stream_processing(
                stream_name=self.stream_name,
                consumer_group=self.consumer_group,
                status="success"
            )
            
            # Acknowledge message
            await self.redis_client.xack(
                self.stream_name,
                self.consumer_group,
                message_id
            )
            
            # Update metrics
            processing_time = time.perf_counter() - start_time
            self.processing_times.append(processing_time)
            self.decisions_processed += 1
            self.last_decision_time = datetime.utcnow()
            
            # Record decision
            tactical_metrics.record_synergy_response(
                synergy_type=synergy_event.synergy_type,
                response_action=decision.action,
                confidence=decision.confidence
            )
            
            logger.info(
                f"✅ Synergy event processed: {decision.action} "
                f"(confidence: {decision.confidence:.3f}, "
                f"time: {processing_time*1000:.2f}ms)",
                correlation_id=synergy_event.correlation_id
            )
            
        except Exception as e:
            logger.error(f"Failed to process synergy event: {e}")
            tactical_metrics.record_error(
                error_type="synergy_event_processing_failed",
                component="controller",
                severity="error"
            )
            
            # Record failed processing
            tactical_metrics.record_stream_processing(
                stream_name=self.stream_name,
                consumer_group=self.consumer_group,
                status="failed"
            )
    
    async def _process_pending_messages(self):
        """Process any pending messages that weren't acknowledged."""
        try:
            # Get pending messages
            pending_info = await self.redis_client.xpending(
                self.stream_name,
                self.consumer_group
            )
            
            if pending_info and pending_info[0] > 0:  # Count of pending messages
                logger.info(f"Processing {pending_info[0]} pending messages")
                
                # Get detailed pending messages
                pending_messages = await self.redis_client.xpending_range(
                    self.stream_name,
                    self.consumer_group,
                    min="-",
                    max="+",
                    count=10
                )
                
                for message_info in pending_messages:
                    message_id = message_info[0]
                    consumer = message_info[1]
                    idle_time = message_info[2]
                    
                    # If message is idle for more than 10 seconds, reclaim it
                    if idle_time > 10000:  # 10 seconds in milliseconds
                        try:
                            # Claim the message
                            claimed = await self.redis_client.xclaim(
                                self.stream_name,
                                self.consumer_group,
                                self.consumer_name,
                                min_idle_time=idle_time,
                                message_ids=[message_id]
                            )
                            
                            for stream, messages in claimed:
                                for msg_id, fields in messages:
                                    await self._process_synergy_event(
                                        msg_id.decode(),
                                        fields
                                    )
                        except Exception as e:
                            logger.error(f"Failed to claim pending message: {e}")
        
        except Exception as e:
            logger.error(f"Error processing pending messages: {e}")
    
    async def on_synergy_detected(self, synergy_event: SynergyEvent) -> TacticalDecision:
        """
        Process SYNERGY_DETECTED event and generate tactical response.
        
        This is the core decision-making method that:
        1. Acquires distributed lock to prevent race conditions
        2. Checks for duplicate event processing (idempotency)
        3. Fetches current 5-minute matrix state
        4. Gets decisions from tactical agents
        5. Aggregates decisions with confidence weighting
        6. Creates execution command if consensus is reached
        """
        timing = {}
        lock = None
        lock_acquired = False
        
        # Step 0: Acquire distributed lock to prevent race conditions
        lock_start = time.perf_counter()
        
        if not synergy_event.correlation_id:
            logger.error("Event missing correlation_id - cannot acquire lock")
            return self._create_error_response(synergy_event, "missing_correlation_id")
        
        lock_key = f"tactical:event_lock:{synergy_event.correlation_id}"
        lock = self.redis_client.lock(
            lock_key,
            timeout=10,  # 10 second timeout
            sleep=0.01,  # 10ms retry interval
            blocking_timeout=0.1  # Don't block for more than 100ms
        )
        
        try:
            # Try to acquire lock with short timeout to maintain low latency
            lock_acquired = await lock.acquire(blocking=False)
            timing['lock_acquisition_ms'] = (time.perf_counter() - lock_start) * 1000
            
            if not lock_acquired:
                logger.warning(
                    f"Event {synergy_event.correlation_id} is already being processed by another instance. Discarding.",
                    correlation_id=synergy_event.correlation_id
                )
                return self._create_duplicate_event_response(synergy_event)
            
            logger.debug(
                f"Acquired lock for event {synergy_event.correlation_id} in {timing['lock_acquisition_ms']:.2f}ms",
                correlation_id=synergy_event.correlation_id
            )
            
            # Step 1: Idempotency check (now protected by lock)
            idempotency_start = time.perf_counter()
            if await self._is_duplicate_event(synergy_event):
                logger.warning(
                    f"Duplicate event detected, skipping processing: {synergy_event.correlation_id}",
                    correlation_id=synergy_event.correlation_id
                )
                return self._create_duplicate_event_response(synergy_event)
            
            # Mark event as being processed (atomic with lock)
            await self._mark_event_processing(synergy_event)
            timing['idempotency_check_ms'] = (time.perf_counter() - idempotency_start) * 1000
            
            # Step 2: Get current matrix state
            matrix_start = time.perf_counter()
            matrix_state = await self._get_current_matrix_state()
            timing['matrix_fetch_ms'] = (time.perf_counter() - matrix_start) * 1000
            
            # Step 3: Get agent decisions
            agent_start = time.perf_counter()
            agent_decisions = await self._get_agent_decisions(matrix_state, synergy_event)
            timing['agent_inference_ms'] = (time.perf_counter() - agent_start) * 1000
            
            # Step 4: Aggregate decisions
            aggregation_start = time.perf_counter()
            aggregated_decision = await self._aggregate_decisions(
                agent_decisions,
                synergy_event
            )
            timing['aggregation_ms'] = (time.perf_counter() - aggregation_start) * 1000
            
            # Step 5: Create execution command
            execution_start = time.perf_counter()
            execution_command = await self._create_execution_command(
                aggregated_decision,
                synergy_event
            )
            timing['execution_ms'] = (time.perf_counter() - execution_start) * 1000
            
            # Create tactical decision
            decision = TacticalDecision(
                action=aggregated_decision['action'],
                confidence=aggregated_decision['confidence'],
                agent_votes=agent_decisions,
                consensus_breakdown=aggregated_decision['consensus_breakdown'],
                execution_command=execution_command,
                timing=timing,
                correlation_id=synergy_event.correlation_id,
                timestamp=time.time()
            )
            
            # Store decision history
            self.decision_history.append(decision)
            if len(self.decision_history) > 100:
                self.decision_history = self.decision_history[-100:]
            
            # Emit execution event if trade should be executed
            if decision.action != "hold" and execution_command.get("action") == "execute_trade":
                await self._emit_execution_event(decision, synergy_event)
            
            # Mark event as completed (idempotency)
            await self._mark_event_completed(synergy_event, decision)
            
            return decision
            
        except Exception as e:
            logger.error(
                f"Error processing synergy event {synergy_event.correlation_id}: {e}",
                correlation_id=synergy_event.correlation_id
            )
            # Record error in metrics
            tactical_metrics.record_error(
                error_type="synergy_processing_error",
                component="controller",
                severity="error"
            )
            return self._create_error_response(synergy_event, str(e))
            
        finally:
            # Always release the lock
            if lock and lock_acquired:
                try:
                    await lock.release()
                    timing['lock_total_ms'] = (time.perf_counter() - lock_start) * 1000
                    logger.debug(
                        f"Released lock for event {synergy_event.correlation_id} after {timing.get('lock_total_ms', 0):.2f}ms",
                        correlation_id=synergy_event.correlation_id
                    )
                except Exception as lock_error:
                    logger.error(
                        f"Failed to release lock for event {synergy_event.correlation_id}: {lock_error}",
                        correlation_id=synergy_event.correlation_id
                    )
    
    async def _get_current_matrix_state(self) -> np.ndarray:
        """Fetch current 60×7 matrix from MatrixAssembler5m."""
        try:
            async with tactical_metrics.measure_pipeline_component(
                "matrix_assembler",
                "get_matrix"
            ):
                # In production, this would fetch from the actual matrix assembler
                # For now, simulate with realistic data
                matrix = np.random.randn(60, 7).astype(np.float32)
                
                # Simulate FVG features
                matrix[:, 0] = np.random.choice([0, 1], size=60, p=[0.8, 0.2])  # fvg_bullish_active
                matrix[:, 1] = np.random.choice([0, 1], size=60, p=[0.8, 0.2])  # fvg_bearish_active
                matrix[:, 2] = np.random.uniform(0.99, 1.01, size=60)  # fvg_nearest_level
                matrix[:, 3] = np.random.exponential(5, size=60)  # fvg_age
                matrix[:, 4] = np.random.choice([0, 1], size=60, p=[0.9, 0.1])  # fvg_mitigation_signal
                matrix[:, 5] = np.random.uniform(-5, 5, size=60)  # price_momentum_5
                matrix[:, 6] = np.random.uniform(0.5, 2.0, size=60)  # volume_ratio
                
                return matrix
                
        except Exception as e:
            logger.error(f"Failed to get matrix state: {e}")
            tactical_metrics.record_error(
                error_type="matrix_fetch_failed",
                component="matrix_assembler",
                severity="error"
            )
            # Return default matrix
            return np.zeros((60, 7), dtype=np.float32)
    
    async def _get_agent_decisions(
        self,
        matrix_state: np.ndarray,
        synergy_event: SynergyEvent
    ) -> List[Dict[str, Any]]:
        """Get decisions from all tactical agents using parallel inference pool."""
        try:
            # Use the async inference pool for parallel processing
            if self.inference_pool:
                async with tactical_metrics.measure_pipeline_component(
                    "agent_inference", 
                    "parallel_processing"
                ):
                    agent_decisions = await self.inference_pool.submit_inference_jobs(
                        matrix_state=matrix_state,
                        synergy_event=asdict(synergy_event),
                        correlation_id=synergy_event.correlation_id,
                        timeout_seconds=2.0  # Aggressive timeout for high frequency
                    )
                
                logger.debug(
                    f"Parallel inference completed: {len(agent_decisions)} agents in pool",
                    correlation_id=synergy_event.correlation_id
                )
                
                return agent_decisions
            
            else:
                # Fallback to sequential processing if pool not available
                logger.warning("Inference pool not available, falling back to sequential processing")
                return await self._get_agent_decisions_sequential(matrix_state, synergy_event)
                
        except Exception as e:
            logger.error(f"Parallel inference failed: {e}, falling back to sequential")
            tactical_metrics.record_error(
                error_type="parallel_inference_failed",
                component="inference_pool",
                severity="warning"
            )
            
            # Fallback to sequential processing
            return await self._get_agent_decisions_sequential(matrix_state, synergy_event)
    
    async def _get_agent_decisions_sequential(
        self,
        matrix_state: np.ndarray,
        synergy_event: SynergyEvent
    ) -> List[Dict[str, Any]]:
        """Fallback sequential agent decision processing."""
        agent_decisions = []
        
        # Agent configurations
        agents = [
            {"name": "fvg_agent", "type": "fvg"},
            {"name": "momentum_agent", "type": "momentum"},
            {"name": "entry_agent", "type": "entry"}
        ]
        
        try:
            for agent_config in agents:
                agent_name = agent_config["name"]
                agent_type = agent_config["type"]
                
                async with tactical_metrics.measure_inference_latency(
                    model_type=agent_type,
                    agent_name=agent_name,
                    correlation_id=synergy_event.correlation_id
                ):
                    # Get agent decision
                    decision = await self._get_single_agent_decision(
                        agent_config,
                        matrix_state,
                        synergy_event
                    )
                    
                    agent_decisions.append({
                        "agent_name": agent_name,
                        "agent_type": agent_type,
                        "action": decision["action"],
                        "probabilities": decision["probabilities"],
                        "confidence": decision["confidence"],
                        "reasoning": decision.get("reasoning", {}),
                        "correlation_id": synergy_event.correlation_id
                    })
        
        except Exception as e:
            logger.error(f"Failed to get agent decisions: {e}")
            tactical_metrics.record_error(
                error_type="agent_inference_failed",
                component="agents",
                severity="error"
            )
            
            # Return default decisions
            for agent_config in agents:
                agent_decisions.append({
                    "agent_name": agent_config["name"],
                    "agent_type": agent_config["type"],
                    "action": 0,  # Hold
                    "probabilities": [0.33, 0.34, 0.33],
                    "confidence": 0.5,
                    "reasoning": {"error": "inference_failed"},
                    "correlation_id": synergy_event.correlation_id
                })
        
        return agent_decisions
    
    async def _get_single_agent_decision(
        self,
        agent_config: Dict[str, str],
        matrix_state: np.ndarray,
        synergy_event: SynergyEvent
    ) -> Dict[str, Any]:
        """Get decision from a single agent."""
        
        # Simulate agent inference (in production, this would use actual models)
        agent_type = agent_config["type"]
        
        if agent_type == "fvg":
            # FVG agent focuses on gap patterns
            fvg_active = np.any(matrix_state[:, 0] > 0) or np.any(matrix_state[:, 1] > 0)
            if fvg_active:
                if synergy_event.direction > 0:
                    probabilities = [0.7, 0.2, 0.1]  # Bullish
                else:
                    probabilities = [0.1, 0.2, 0.7]  # Bearish
            else:
                probabilities = [0.3, 0.4, 0.3]  # Neutral
        
        elif agent_type == "momentum":
            # Momentum agent focuses on price momentum
            momentum = np.mean(matrix_state[:, 5])  # price_momentum_5
            if momentum > 2.0:
                probabilities = [0.6, 0.3, 0.1]  # Bullish
            elif momentum < -2.0:
                probabilities = [0.1, 0.3, 0.6]  # Bearish
            else:
                probabilities = [0.35, 0.3, 0.35]  # Neutral
        
        else:  # entry agent
            # Entry agent focuses on timing
            volume_ratio = np.mean(matrix_state[:, 6])
            if volume_ratio > 1.5:
                probabilities = [0.4, 0.5, 0.1]  # Execute
            else:
                probabilities = [0.2, 0.6, 0.2]  # Wait
        
        # Normalize probabilities
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        # Sample action
        action = np.random.choice(3, p=probabilities) - 1  # -1, 0, 1
        confidence = float(probabilities.max())
        
        return {
            "action": action,
            "probabilities": probabilities.tolist(),
            "confidence": confidence,
            "reasoning": {
                "agent_type": agent_type,
                "synergy_alignment": synergy_event.direction == np.sign(action) if action != 0 else True
            }
        }
    
    async def _aggregate_decisions(
        self,
        agent_decisions: List[Dict[str, Any]],
        synergy_event: SynergyEvent
    ) -> Dict[str, Any]:
        """Aggregate agent decisions using weighted voting."""
        
        # Use decision aggregator
        aggregated = await self.decision_aggregator.aggregate_decisions(
            agent_decisions,
            synergy_event
        )
        
        return aggregated
    
    async def _create_execution_command(
        self,
        decision: Dict[str, Any],
        synergy_event: SynergyEvent
    ) -> Dict[str, Any]:
        """Create execution command for trading system."""
        
        if decision["action"] == "hold":
            return {
                "action": "hold",
                "reason": "no_consensus_or_hold_decision",
                "correlation_id": synergy_event.correlation_id
            }
        
        # Check minimum confidence threshold for execution
        min_confidence = 0.6  # Lower threshold to allow more trades
        if decision["confidence"] < min_confidence:
            return {
                "action": "hold",
                "reason": f"confidence_too_low_{decision['confidence']:.3f}",
                "correlation_id": synergy_event.correlation_id
            }
        
        # Create trading command
        current_price = 100.0  # Would be fetched from market data
        atr_5min = 0.5  # Would be calculated from actual data
        
        if decision["action"] == "long":
            stop_loss = current_price - (2.0 * atr_5min)
            take_profit = current_price + (3.0 * atr_5min)
            side = "BUY"
        else:  # short
            stop_loss = current_price + (2.0 * atr_5min)
            take_profit = current_price - (3.0 * atr_5min)
            side = "SELL"
        
        # Position sizing based on confidence
        base_quantity = 1
        confidence_multiplier = min(decision["confidence"] / 0.8, 1.5)
        quantity = int(base_quantity * confidence_multiplier)
        
        return {
            "action": "execute_trade",
            "order_type": "MARKET",
            "side": side,
            "quantity": quantity,
            "symbol": "EURUSD",  # Would be configurable
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "time_in_force": "IOC",
            "metadata": {
                "source": "tactical_marl",
                "synergy_type": synergy_event.synergy_type,
                "confidence": decision["confidence"],
                "correlation_id": synergy_event.correlation_id,
                "timestamp": time.time()
            }
        }
    
    async def process_decision_request(
        self,
        matrix_state: List[List[float]],
        synergy_context: Optional[Dict[str, Any]] = None,
        override_params: Dict[str, Any] = None,
        correlation_id: str = ""
    ) -> Dict[str, Any]:
        """Process a manual decision request."""
        
        # Convert matrix state to numpy array
        matrix_np = np.array(matrix_state, dtype=np.float32)
        
        # Create synergy event from context
        if synergy_context:
            synergy_event = SynergyEvent.from_dict(synergy_context)
        else:
            synergy_event = SynergyEvent(
                synergy_type="manual",
                direction=0,
                confidence=0.5,
                signal_sequence=[],
                market_context={},
                correlation_id=correlation_id,
                timestamp=time.time()
            )
        
        # Process decision
        decision = await self.on_synergy_detected(synergy_event)
        
        return {
            "decision": {
                "action": decision.action,
                "confidence": decision.confidence,
                "execution_command": decision.execution_command
            },
            "agent_breakdown": {
                "agent_votes": decision.agent_votes,
                "consensus_breakdown": decision.consensus_breakdown
            },
            "timing": decision.timing
        }
    
    async def stop_event_listener(self):
        """Stop the event listener."""
        self.running = False
        logger.info("Stopping tactical event listener")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_event_listener()
        
        # Cleanup inference pool
        if self.inference_pool:
            # Note: Don't stop global pool here as other instances may be using it
            self.inference_pool = None
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Tactical controller cleanup complete")
    
    async def _is_duplicate_event(self, synergy_event: SynergyEvent) -> bool:
        """
        Check if this event has already been processed (idempotency check).
        
        Uses Redis SET to track processed events with TTL for cleanup.
        """
        if not synergy_event.correlation_id:
            # No correlation ID, assume it's a new event
            return False
        
        try:
            # Check if event is already being processed
            processing_key = f"tactical:processing:{synergy_event.correlation_id}"
            is_processing = await self.redis_client.exists(processing_key)
            
            if is_processing:
                logger.warning(f"Event already being processed: {synergy_event.correlation_id}")
                return True
            
            # Check if event has already been completed
            completed_key = f"tactical:completed:{synergy_event.correlation_id}"
            is_completed = await self.redis_client.exists(completed_key)
            
            if is_completed:
                logger.info(f"Event already completed: {synergy_event.correlation_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check event duplication: {e}")
            # On error, assume it's not a duplicate to avoid blocking processing
            return False
    
    async def _mark_event_processing(self, synergy_event: SynergyEvent):
        """Mark event as being processed."""
        if not synergy_event.correlation_id:
            return
        
        try:
            processing_key = f"tactical:processing:{synergy_event.correlation_id}"
            # Set with 5 minute TTL (should be more than enough for processing)
            await self.redis_client.setex(processing_key, 300, "1")
            
            # Store event details for debugging
            event_details = {
                "synergy_type": synergy_event.synergy_type,
                "direction": synergy_event.direction,
                "confidence": synergy_event.confidence,
                "timestamp": synergy_event.timestamp,
                "processing_start": time.time()
            }
            
            details_key = f"tactical:event_details:{synergy_event.correlation_id}"
            await self.redis_client.setex(details_key, 300, json.dumps(event_details))
            
        except Exception as e:
            logger.error(f"Failed to mark event as processing: {e}")
    
    async def _mark_event_completed(self, synergy_event: SynergyEvent, decision: TacticalDecision):
        """Mark event as completed and remove from processing."""
        if not synergy_event.correlation_id:
            return
        
        try:
            processing_key = f"tactical:processing:{synergy_event.correlation_id}"
            completed_key = f"tactical:completed:{synergy_event.correlation_id}"
            
            # Remove from processing
            await self.redis_client.delete(processing_key)
            
            # Mark as completed with 1 hour TTL
            completion_data = {
                "action": decision.action,
                "confidence": decision.confidence,
                "execution_command": decision.execution_command,
                "completion_time": time.time(),
                "correlation_id": synergy_event.correlation_id
            }
            
            await self.redis_client.setex(completed_key, 3600, json.dumps(completion_data))
            
            # Record completion metrics
            tactical_metrics.record_event_completion(
                synergy_type=synergy_event.synergy_type,
                action=decision.action,
                confidence=decision.confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to mark event as completed: {e}")
    
    def _create_duplicate_event_response(self, synergy_event: SynergyEvent) -> TacticalDecision:
        """Create a response for duplicate events."""
        return TacticalDecision(
            action="hold",
            confidence=0.0,
            agent_votes=[],
            consensus_breakdown={"duplicate_event": 1.0},
            execution_command={
                "action": "hold",
                "reason": "duplicate_event",
                "correlation_id": synergy_event.correlation_id
            },
            timing={"duplicate_check_ms": 0.0},
            correlation_id=synergy_event.correlation_id,
            timestamp=time.time()
        )
    
    async def _emit_execution_event(self, decision: TacticalDecision, synergy_event: SynergyEvent):
        """Emit execution event for trade execution pipeline."""
        try:
            # Create execution event payload
            execution_payload = {
                'action': decision.action,
                'confidence': decision.confidence,
                'execution_command': decision.execution_command,
                'correlation_id': synergy_event.correlation_id,
                'timestamp': time.time(),
                'source': 'tactical_controller'
            }
            
            # Publish to Redis for execution engine
            await self.redis_client.publish(
                'execution_events',
                json.dumps(execution_payload)
            )
            
            # Also publish to Redis streams for persistence
            await self.redis_client.xadd(
                'execution_stream',
                execution_payload
            )
            
            logger.info(
                f"📤 Execution event emitted: {decision.action}",
                correlation_id=synergy_event.correlation_id,
                confidence=decision.confidence
            )
            
        except Exception as e:
            logger.error(
                f"Failed to emit execution event: {e}",
                correlation_id=synergy_event.correlation_id
            )
    
    def _create_error_response(self, synergy_event: SynergyEvent, error_message: str) -> TacticalDecision:
        """Create a response for error conditions."""
        return TacticalDecision(
            action="hold",
            confidence=0.0,
            agent_votes=[],
            consensus_breakdown={"error": 1.0},
            execution_command={
                "action": "hold",
                "reason": f"error: {error_message}",
                "correlation_id": synergy_event.correlation_id
            },
            timing={"error_response_ms": 0.0},
            correlation_id=synergy_event.correlation_id,
            timestamp=time.time()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "decisions_processed": self.decisions_processed,
            "last_decision_time": self.last_decision_time.isoformat() if self.last_decision_time else None,
            "average_processing_time_ms": np.mean(self.processing_times) * 1000 if self.processing_times else 0,
            "running": self.running,
            "current_position": self.current_position,
            "decisions_history_length": len(self.decision_history),
            "inference_pool": {
                "available": self.inference_pool is not None,
                "stats": self.inference_pool.get_stats() if self.inference_pool else None
            }
        }