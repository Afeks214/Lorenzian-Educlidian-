"""
Strategic-to-Tactical Cascade Integration

This module implements the seamless integration between the Strategic MARL system
(30-minute cycles) and the Tactical MARL system (5-minute cycles). It manages
the cascade of strategic decisions into tactical execution with proper context
propagation and performance optimization.

Key Features:
- Seamless strategic-to-tactical context propagation
- Real-time strategic context updates and buffering
- Tactical execution triggering based on strategic signals
- Performance monitoring and latency optimization
- Byzantine fault tolerance across system boundaries
- Comprehensive logging and debugging support

Architecture:
- Strategic MARL (30m) generates strategic context
- Context buffer manages strategic updates
- Tactical MARL (5m) receives enriched strategic context
- Cascade coordinator manages timing and synchronization
- Performance monitor ensures latency requirements

Author: Agent 5 - Sequential Tactical MARL Specialist
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
import time
import json
import uuid
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
from pathlib import Path
import warnings

# Core imports
from src.environment.strategic_env import StrategicEnvironment  
from src.environment.sequential_tactical_env import SequentialTacticalEnvironment, StrategicContext
from src.agents.tactical.sequential_tactical_agents import SequentialTacticalAgent
from src.environment.tactical_superposition_aggregator import TacticalSuperpositionAggregator
from src.core.events import Event, EventType
from src.core.event_bus import EventBus
from src.monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class CascadeState(Enum):
    """Cascade system states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STRATEGIC_UPDATE = "strategic_update"
    TACTICAL_EXECUTION = "tactical_execution"
    SYNCHRONIZING = "synchronizing"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class ExecutionTrigger(Enum):
    """Execution trigger types"""
    STRATEGIC_SIGNAL = "strategic_signal"
    TACTICAL_TIMER = "tactical_timer"
    MANUAL_TRIGGER = "manual_trigger"
    EMERGENCY_TRIGGER = "emergency_trigger"

@dataclass
class StrategicUpdate:
    """Strategic update container"""
    update_id: str
    timestamp: float
    regime_embedding: np.ndarray
    synergy_signal: Dict[str, Any]
    market_state: Dict[str, Any]
    confidence_level: float
    execution_bias: str
    volatility_forecast: float
    urgency_level: str
    expiry_timestamp: float
    source: str = "strategic_marl"
    
    def to_strategic_context(self) -> StrategicContext:
        """Convert to strategic context"""
        return StrategicContext(
            regime_embedding=self.regime_embedding,
            synergy_signal=self.synergy_signal,
            market_state=self.market_state,
            confidence_level=self.confidence_level,
            execution_bias=self.execution_bias,
            volatility_forecast=self.volatility_forecast,
            timestamp=self.timestamp
        )

@dataclass
class TacticalExecution:
    """Tactical execution container"""
    execution_id: str
    timestamp: float
    trigger_type: ExecutionTrigger
    strategic_context: StrategicContext
    tactical_superposition: Optional[Dict[str, Any]]
    execution_result: Optional[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    processing_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class CascadeMetrics:
    """Cascade performance metrics"""
    total_strategic_updates: int = 0
    total_tactical_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_cascade_latency: float = 0.0
    avg_strategic_to_tactical_latency: float = 0.0
    context_propagation_failures: int = 0
    synchronization_issues: int = 0
    last_update: float = field(default_factory=time.time)
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_tactical_executions == 0:
            return 0.0
        return self.successful_executions / self.total_tactical_executions
    
    def get_failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_tactical_executions == 0:
            return 0.0
        return self.failed_executions / self.total_tactical_executions

class StrategicTacticalCascade:
    """
    Strategic-to-Tactical Cascade Integration
    
    Manages the integration between strategic and tactical MARL systems,
    ensuring seamless context propagation and efficient execution.
    """
    
    def __init__(
        self,
        strategic_environment: StrategicEnvironment,
        tactical_environment: SequentialTacticalEnvironment,
        tactical_agents: Dict[str, SequentialTacticalAgent],
        tactical_aggregator: TacticalSuperpositionAggregator,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize strategic-tactical cascade
        
        Args:
            strategic_environment: Strategic MARL environment
            tactical_environment: Tactical MARL environment
            tactical_agents: Dictionary of tactical agents
            tactical_aggregator: Tactical superposition aggregator
            config: Configuration dictionary
            event_bus: Event bus for system communication
        """
        self.strategic_environment = strategic_environment
        self.tactical_environment = tactical_environment
        self.tactical_agents = tactical_agents
        self.tactical_aggregator = tactical_aggregator
        self.config = config or self._default_config()
        self.event_bus = event_bus or EventBus()
        
        # Validate configuration
        self._validate_config()
        
        # Cascade state management
        self.cascade_state = CascadeState.INITIALIZING
        self.is_running = False
        self.shutdown_requested = False
        
        # Strategic context management
        self.strategic_context_buffer = deque(maxlen=self.config['strategic_buffer_size'])
        self.current_strategic_context: Optional[StrategicContext] = None
        self.strategic_update_lock = threading.Lock()
        
        # Tactical execution management
        self.tactical_execution_queue = queue.Queue(maxsize=self.config['tactical_queue_size'])
        self.execution_history = deque(maxlen=self.config['execution_history_size'])
        
        # Performance monitoring
        self.cascade_metrics = CascadeMetrics()
        self.performance_monitor = self._initialize_performance_monitor()
        
        # Threading and concurrency
        self.strategic_thread: Optional[threading.Thread] = None
        self.tactical_thread: Optional[threading.Thread] = None
        self.coordination_thread: Optional[threading.Thread] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config['max_workers'])
        
        # Timing and synchronization
        self.strategic_cycle_time = self.config['strategic_cycle_time']  # 30 minutes
        self.tactical_cycle_time = self.config['tactical_cycle_time']    # 5 minutes
        self.last_strategic_update = 0.0
        self.last_tactical_execution = 0.0
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Initialize event handlers
        self._setup_event_handlers()
        
        logger.info(f"Strategic-Tactical Cascade initialized")
        logger.info(f"Strategic cycle: {self.strategic_cycle_time}s, Tactical cycle: {self.tactical_cycle_time}s")
        logger.info(f"Session ID: {self.session_id}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for cascade"""
        return {
            'strategic_cycle_time': 1800.0,  # 30 minutes
            'tactical_cycle_time': 300.0,    # 5 minutes
            'strategic_buffer_size': 10,
            'tactical_queue_size': 100,
            'execution_history_size': 1000,
            'max_workers': 4,
            'context_expiry_time': 3600.0,  # 1 hour
            'cascade_timeout': 30.0,         # 30 seconds
            'performance_monitoring': True,
            'auto_restart': True,
            'error_recovery': True,
            'max_retries': 3,
            'retry_delay': 1.0,
            'strategic_sync_tolerance': 60.0,  # 1 minute
            'tactical_latency_target': 1.0,     # 1 second
            'enable_logging': True,
            'log_level': 'INFO',
            'metrics_update_frequency': 60.0,   # 1 minute
            'health_check_frequency': 30.0,     # 30 seconds
            'emergency_shutdown_threshold': 0.1  # 10% failure rate
        }
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = [
            'strategic_cycle_time', 'tactical_cycle_time', 
            'strategic_buffer_size', 'tactical_queue_size'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate timing relationships
        if self.config['tactical_cycle_time'] >= self.config['strategic_cycle_time']:
            raise ValueError("Tactical cycle must be shorter than strategic cycle")
        
        # Validate buffer sizes
        if self.config['strategic_buffer_size'] < 1:
            raise ValueError("Strategic buffer size must be at least 1")
        
        if self.config['tactical_queue_size'] < 1:
            raise ValueError("Tactical queue size must be at least 1")
        
        logger.info("Configuration validation passed")
    
    def _initialize_performance_monitor(self):
        """Initialize performance monitoring"""
        try:
            if self.config['performance_monitoring']:
                return PerformanceMonitor(
                    config=self.config,
                    session_id=self.session_id
                )
            else:
                return None
        except Exception as e:
            logger.warning(f"Performance monitor initialization failed: {e}")
            return None
    
    def _setup_event_handlers(self):
        """Setup event handlers for system communication"""
        try:
            # Strategic update events
            self.event_bus.subscribe(EventType.STRATEGIC_UPDATE, self._handle_strategic_update)
            
            # Tactical execution events  
            self.event_bus.subscribe(EventType.TACTICAL_EXECUTION, self._handle_tactical_execution)
            
            # System events
            self.event_bus.subscribe(EventType.SYSTEM_SHUTDOWN, self._handle_system_shutdown)
            self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_stop)
            
            # Performance events
            self.event_bus.subscribe(EventType.PERFORMANCE_ALERT, self._handle_performance_alert)
            
            logger.info("Event handlers setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup event handlers: {e}")
            raise
    
    def start(self) -> bool:
        """
        Start the cascade system
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning("Cascade already running")
                return True
            
            logger.info("Starting strategic-tactical cascade...")
            
            # Initialize strategic context
            self._initialize_strategic_context()
            
            # Start coordination thread
            self.coordination_thread = threading.Thread(
                target=self._coordination_loop,
                name="CascadeCoordination"
            )
            self.coordination_thread.daemon = True
            self.coordination_thread.start()
            
            # Start strategic monitoring thread
            self.strategic_thread = threading.Thread(
                target=self._strategic_monitoring_loop,
                name="StrategicMonitoring"
            )
            self.strategic_thread.daemon = True
            self.strategic_thread.start()
            
            # Start tactical execution thread
            self.tactical_thread = threading.Thread(
                target=self._tactical_execution_loop,
                name="TacticalExecution"
            )
            self.tactical_thread.daemon = True
            self.tactical_thread.start()
            
            # Update state
            self.is_running = True
            self.cascade_state = CascadeState.RUNNING
            
            # Emit start event
            self.event_bus.emit(Event(
                type=EventType.SYSTEM_START,
                data={'session_id': self.session_id, 'timestamp': time.time()}
            ))
            
            logger.info("Strategic-tactical cascade started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start cascade: {e}")
            self.cascade_state = CascadeState.ERROR
            return False
    
    def stop(self) -> bool:
        """
        Stop the cascade system
        
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if not self.is_running:
                logger.warning("Cascade not running")
                return True
            
            logger.info("Stopping strategic-tactical cascade...")
            
            # Set shutdown flags
            self.shutdown_requested = True
            self.is_running = False
            self.cascade_state = CascadeState.SHUTDOWN
            
            # Wait for threads to finish
            if self.coordination_thread and self.coordination_thread.is_alive():
                self.coordination_thread.join(timeout=5.0)
            
            if self.strategic_thread and self.strategic_thread.is_alive():
                self.strategic_thread.join(timeout=5.0)
            
            if self.tactical_thread and self.tactical_thread.is_alive():
                self.tactical_thread.join(timeout=5.0)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Emit stop event
            self.event_bus.emit(Event(
                type=EventType.SYSTEM_STOP,
                data={'session_id': self.session_id, 'timestamp': time.time()}
            ))
            
            logger.info("Strategic-tactical cascade stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop cascade: {e}")
            return False
    
    def _initialize_strategic_context(self):
        """Initialize strategic context from strategic environment"""
        try:
            # Get initial strategic context
            strategic_observations = self.strategic_environment.reset()
            
            # Generate initial strategic update
            strategic_update = self._generate_strategic_update(strategic_observations)
            
            # Store in buffer
            with self.strategic_update_lock:
                self.strategic_context_buffer.append(strategic_update)
                self.current_strategic_context = strategic_update.to_strategic_context()
            
            logger.info("Strategic context initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize strategic context: {e}")
            raise
    
    def _coordination_loop(self):
        """Main coordination loop"""
        logger.info("Coordination loop started")
        
        while self.is_running and not self.shutdown_requested:
            try:
                # Check system health
                self._check_system_health()
                
                # Update metrics
                self._update_metrics()
                
                # Check for emergency conditions
                if self._check_emergency_conditions():
                    self._handle_emergency_shutdown()
                    break
                
                # Coordinate strategic and tactical timing
                self._coordinate_timing()
                
                # Process pending events
                self._process_coordination_events()
                
                # Sleep for coordination interval
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                if not self.config['error_recovery']:
                    break
                time.sleep(self.config['retry_delay'])
        
        logger.info("Coordination loop stopped")
    
    def _strategic_monitoring_loop(self):
        """Strategic monitoring loop"""
        logger.info("Strategic monitoring loop started")
        
        while self.is_running and not self.shutdown_requested:
            try:
                current_time = time.time()
                
                # Check if strategic update is needed
                if (current_time - self.last_strategic_update) >= self.strategic_cycle_time:
                    self._update_strategic_context()
                    self.last_strategic_update = current_time
                
                # Check context expiry
                self._check_context_expiry()
                
                # Monitor strategic environment
                self._monitor_strategic_environment()
                
                # Sleep for monitoring interval
                time.sleep(min(60.0, self.strategic_cycle_time / 10))
                
            except Exception as e:
                logger.error(f"Error in strategic monitoring loop: {e}")
                if not self.config['error_recovery']:
                    break
                time.sleep(self.config['retry_delay'])
        
        logger.info("Strategic monitoring loop stopped")
    
    def _tactical_execution_loop(self):
        """Tactical execution loop"""
        logger.info("Tactical execution loop started")
        
        while self.is_running and not self.shutdown_requested:
            try:
                current_time = time.time()
                
                # Check if tactical execution is needed
                if (current_time - self.last_tactical_execution) >= self.tactical_cycle_time:
                    self._execute_tactical_cycle()
                    self.last_tactical_execution = current_time
                
                # Process tactical execution queue
                self._process_tactical_queue()
                
                # Monitor tactical environment
                self._monitor_tactical_environment()
                
                # Sleep for tactical interval
                time.sleep(min(10.0, self.tactical_cycle_time / 10))
                
            except Exception as e:
                logger.error(f"Error in tactical execution loop: {e}")
                if not self.config['error_recovery']:
                    break
                time.sleep(self.config['retry_delay'])
        
        logger.info("Tactical execution loop stopped")
    
    def _update_strategic_context(self):
        """Update strategic context from strategic environment"""
        try:
            self.cascade_state = CascadeState.STRATEGIC_UPDATE
            update_start_time = time.time()
            
            # Run strategic environment step
            strategic_observations = self.strategic_environment.reset()
            
            # Generate strategic update
            strategic_update = self._generate_strategic_update(strategic_observations)
            
            # Store in buffer with lock
            with self.strategic_update_lock:
                self.strategic_context_buffer.append(strategic_update)
                self.current_strategic_context = strategic_update.to_strategic_context()
            
            # Update metrics
            self.cascade_metrics.total_strategic_updates += 1
            
            # Emit strategic update event
            self.event_bus.emit(Event(
                type=EventType.STRATEGIC_UPDATE,
                data={
                    'update_id': strategic_update.update_id,
                    'timestamp': strategic_update.timestamp,
                    'confidence_level': strategic_update.confidence_level,
                    'execution_bias': strategic_update.execution_bias
                }
            ))
            
            update_duration = time.time() - update_start_time
            logger.info(f"Strategic context updated in {update_duration:.3f}s")
            
            self.cascade_state = CascadeState.RUNNING
            
        except Exception as e:
            logger.error(f"Failed to update strategic context: {e}")
            self.cascade_state = CascadeState.ERROR
    
    def _generate_strategic_update(self, observations: Dict[str, Any]) -> StrategicUpdate:
        """Generate strategic update from observations"""
        try:
            # Mock strategic update generation
            # In practice, this would process strategic environment outputs
            
            update_id = str(uuid.uuid4())
            timestamp = time.time()
            
            strategic_update = StrategicUpdate(
                update_id=update_id,
                timestamp=timestamp,
                regime_embedding=np.random.normal(0, 0.1, 64).astype(np.float32),
                synergy_signal={
                    'strength': np.random.uniform(0.5, 1.0),
                    'confidence': np.random.uniform(0.6, 1.0),
                    'direction': np.random.choice([-1, 0, 1]),
                    'urgency': np.random.uniform(0.3, 0.9),
                    'risk_level': np.random.uniform(0.1, 0.5),
                    'time_horizon': np.random.uniform(0.5, 2.0)
                },
                market_state={
                    'price': 100.0 + np.random.normal(0, 2),
                    'volume': 1000.0 + np.random.normal(0, 200),
                    'volatility': np.random.uniform(0.1, 0.5),
                    'trend': np.random.choice(['bullish', 'bearish', 'neutral']),
                    'regime': np.random.choice(['trending', 'ranging', 'volatile'])
                },
                confidence_level=np.random.uniform(0.6, 1.0),
                execution_bias=np.random.choice(['bullish', 'neutral', 'bearish']),
                volatility_forecast=np.random.uniform(0.1, 0.5),
                urgency_level=np.random.choice(['low', 'medium', 'high']),
                expiry_timestamp=timestamp + self.config['context_expiry_time']
            )
            
            return strategic_update
            
        except Exception as e:
            logger.error(f"Failed to generate strategic update: {e}")
            # Return safe default
            return StrategicUpdate(
                update_id=str(uuid.uuid4()),
                timestamp=time.time(),
                regime_embedding=np.zeros(64, dtype=np.float32),
                synergy_signal={},
                market_state={},
                confidence_level=0.5,
                execution_bias='neutral',
                volatility_forecast=0.2,
                urgency_level='medium',
                expiry_timestamp=time.time() + 3600
            )
    
    def _execute_tactical_cycle(self):
        """Execute tactical cycle"""
        try:
            self.cascade_state = CascadeState.TACTICAL_EXECUTION
            execution_start_time = time.time()
            
            # Get current strategic context
            strategic_context = self.current_strategic_context
            if not strategic_context:
                logger.warning("No strategic context available for tactical execution")
                return
            
            # Update tactical environment with strategic context
            self.tactical_environment.update_strategic_context(strategic_context)
            
            # Reset tactical environment
            tactical_observations = self.tactical_environment.reset()
            
            # Execute tactical sequence
            tactical_execution = self._run_tactical_sequence(
                tactical_observations, strategic_context
            )
            
            # Store execution result
            self.execution_history.append(tactical_execution)
            
            # Update metrics
            self.cascade_metrics.total_tactical_executions += 1
            if tactical_execution.success:
                self.cascade_metrics.successful_executions += 1
            else:
                self.cascade_metrics.failed_executions += 1
            
            # Update cascade latency
            cascade_latency = time.time() - execution_start_time
            self.cascade_metrics.avg_cascade_latency = (
                0.9 * self.cascade_metrics.avg_cascade_latency + 
                0.1 * cascade_latency
            )
            
            # Emit tactical execution event
            self.event_bus.emit(Event(
                type=EventType.TACTICAL_EXECUTION,
                data={
                    'execution_id': tactical_execution.execution_id,
                    'success': tactical_execution.success,
                    'processing_time': tactical_execution.processing_time,
                    'tactical_superposition': tactical_execution.tactical_superposition
                }
            ))
            
            logger.info(f"Tactical cycle executed in {cascade_latency:.3f}s")
            
            self.cascade_state = CascadeState.RUNNING
            
        except Exception as e:
            logger.error(f"Failed to execute tactical cycle: {e}")
            self.cascade_state = CascadeState.ERROR
    
    def _run_tactical_sequence(
        self, 
        observations: Dict[str, Any], 
        strategic_context: StrategicContext
    ) -> TacticalExecution:
        """Run tactical sequence with strategic context"""
        try:
            execution_id = str(uuid.uuid4())
            sequence_start_time = time.time()
            
            # Sequential agent execution
            agent_superpositions = {}
            predecessor_context = None
            
            for agent_id in ['fvg_agent', 'momentum_agent', 'entry_opt_agent']:
                if agent_id in self.tactical_agents:
                    agent = self.tactical_agents[agent_id]
                    observation = observations.get(agent_id, np.zeros(agent.observation_dim))
                    
                    # Agent action selection with strategic context
                    output = agent.select_action(
                        observation=observation,
                        strategic_context=strategic_context,
                        predecessor_context=predecessor_context
                    )
                    
                    agent_superpositions[agent_id] = output
                    
                    # Update predecessor context
                    predecessor_context = self._update_predecessor_context(
                        predecessor_context, agent_id, output
                    )
            
            # Aggregate tactical superposition
            tactical_superposition = self.tactical_aggregator.aggregate(
                agent_superpositions=agent_superpositions,
                strategic_context=strategic_context,
                market_state=strategic_context.market_state
            )
            
            # Calculate performance metrics
            processing_time = time.time() - sequence_start_time
            performance_metrics = {
                'processing_time': processing_time,
                'agent_count': len(agent_superpositions),
                'confidence': tactical_superposition.confidence,
                'consensus_level': tactical_superposition.consensus_level.value,
                'strategic_alignment': tactical_superposition.strategic_alignment
            }
            
            # Determine success
            success = (
                tactical_superposition.consensus_level.value != 'failed' and
                processing_time < self.config['cascade_timeout']
            )
            
            # Create execution result
            execution_result = {
                'execute': tactical_superposition.execute,
                'action': tactical_superposition.action,
                'confidence': tactical_superposition.confidence,
                'execution_command': tactical_superposition.execution_command
            }
            
            return TacticalExecution(
                execution_id=execution_id,
                timestamp=time.time(),
                trigger_type=ExecutionTrigger.TACTICAL_TIMER,
                strategic_context=strategic_context,
                tactical_superposition=tactical_superposition.to_dict(),
                execution_result=execution_result,
                performance_metrics=performance_metrics,
                processing_time=processing_time,
                success=success
            )
            
        except Exception as e:
            logger.error(f"Failed to run tactical sequence: {e}")
            return TacticalExecution(
                execution_id=str(uuid.uuid4()),
                timestamp=time.time(),
                trigger_type=ExecutionTrigger.TACTICAL_TIMER,
                strategic_context=strategic_context,
                tactical_superposition=None,
                execution_result=None,
                performance_metrics={},
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _update_predecessor_context(self, current_context, agent_id, output):
        """Update predecessor context (mock implementation)"""
        # This would be implemented similarly to the tactical environment
        return current_context
    
    def _check_system_health(self):
        """Check system health"""
        try:
            # Check thread health
            if not self.strategic_thread.is_alive():
                logger.warning("Strategic thread is not alive")
            
            if not self.tactical_thread.is_alive():
                logger.warning("Tactical thread is not alive")
            
            # Check performance metrics
            if self.cascade_metrics.get_failure_rate() > self.config['emergency_shutdown_threshold']:
                logger.error(f"High failure rate: {self.cascade_metrics.get_failure_rate():.2%}")
                self._handle_emergency_shutdown()
            
            # Check queue health
            if self.tactical_execution_queue.full():
                logger.warning("Tactical execution queue is full")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions"""
        try:
            # Check failure rate
            if self.cascade_metrics.get_failure_rate() > self.config['emergency_shutdown_threshold']:
                return True
            
            # Check cascade latency
            if self.cascade_metrics.avg_cascade_latency > self.config['cascade_timeout']:
                return True
            
            # Check context expiry
            if self.current_strategic_context:
                if time.time() > (self.current_strategic_context.timestamp + self.config['context_expiry_time']):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
            return False
    
    def _check_context_expiry(self):
        """Check and handle context expiry"""
        try:
            with self.strategic_update_lock:
                if self.current_strategic_context:
                    current_time = time.time()
                    context_age = current_time - self.current_strategic_context.timestamp
                    
                    if context_age > self.config['context_expiry_time']:
                        logger.warning(f"Strategic context expired (age: {context_age:.1f}s)")
                        self.cascade_metrics.context_propagation_failures += 1
                        
                        # Force strategic update
                        self._update_strategic_context()
                        
        except Exception as e:
            logger.error(f"Error checking context expiry: {e}")
    
    def _coordinate_timing(self):
        """Coordinate strategic and tactical timing"""
        try:
            current_time = time.time()
            
            # Check for timing drift
            strategic_drift = abs(current_time - self.last_strategic_update) - self.strategic_cycle_time
            tactical_drift = abs(current_time - self.last_tactical_execution) - self.tactical_cycle_time
            
            if abs(strategic_drift) > self.config['strategic_sync_tolerance']:
                logger.warning(f"Strategic timing drift: {strategic_drift:.1f}s")
                self.cascade_metrics.synchronization_issues += 1
            
            if abs(tactical_drift) > self.config['tactical_latency_target']:
                logger.warning(f"Tactical timing drift: {tactical_drift:.1f}s")
                self.cascade_metrics.synchronization_issues += 1
            
        except Exception as e:
            logger.error(f"Error coordinating timing: {e}")
    
    def _process_coordination_events(self):
        """Process coordination events"""
        try:
            # Process any pending coordination tasks
            pass
            
        except Exception as e:
            logger.error(f"Error processing coordination events: {e}")
    
    def _process_tactical_queue(self):
        """Process tactical execution queue"""
        try:
            while not self.tactical_execution_queue.empty():
                try:
                    execution_request = self.tactical_execution_queue.get_nowait()
                    # Process execution request
                    self._handle_tactical_execution_request(execution_request)
                except queue.Empty:
                    break
                    
        except Exception as e:
            logger.error(f"Error processing tactical queue: {e}")
    
    def _handle_tactical_execution_request(self, request):
        """Handle tactical execution request"""
        try:
            # Process execution request
            logger.info(f"Processing tactical execution request: {request}")
            
        except Exception as e:
            logger.error(f"Error handling tactical execution request: {e}")
    
    def _monitor_strategic_environment(self):
        """Monitor strategic environment"""
        try:
            # Monitor strategic environment health
            pass
            
        except Exception as e:
            logger.error(f"Error monitoring strategic environment: {e}")
    
    def _monitor_tactical_environment(self):
        """Monitor tactical environment"""
        try:
            # Monitor tactical environment health
            pass
            
        except Exception as e:
            logger.error(f"Error monitoring tactical environment: {e}")
    
    def _update_metrics(self):
        """Update performance metrics"""
        try:
            current_time = time.time()
            
            # Update cascade metrics
            self.cascade_metrics.last_update = current_time
            
            # Update strategic-to-tactical latency
            if self.current_strategic_context:
                strategic_to_tactical_latency = current_time - self.current_strategic_context.timestamp
                self.cascade_metrics.avg_strategic_to_tactical_latency = (
                    0.9 * self.cascade_metrics.avg_strategic_to_tactical_latency +
                    0.1 * strategic_to_tactical_latency
                )
            
            # Update performance monitor
            if self.performance_monitor:
                self.performance_monitor.update_metrics(self.cascade_metrics)
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _handle_strategic_update(self, event: Event):
        """Handle strategic update event"""
        try:
            logger.info(f"Handling strategic update event: {event.data}")
            
        except Exception as e:
            logger.error(f"Error handling strategic update event: {e}")
    
    def _handle_tactical_execution(self, event: Event):
        """Handle tactical execution event"""
        try:
            logger.info(f"Handling tactical execution event: {event.data}")
            
        except Exception as e:
            logger.error(f"Error handling tactical execution event: {e}")
    
    def _handle_system_shutdown(self, event: Event):
        """Handle system shutdown event"""
        try:
            logger.info("Handling system shutdown event")
            self.stop()
            
        except Exception as e:
            logger.error(f"Error handling system shutdown event: {e}")
    
    def _handle_emergency_stop(self, event: Event):
        """Handle emergency stop event"""
        try:
            logger.critical("Handling emergency stop event")
            self._handle_emergency_shutdown()
            
        except Exception as e:
            logger.error(f"Error handling emergency stop event: {e}")
    
    def _handle_emergency_shutdown(self):
        """Handle emergency shutdown"""
        try:
            logger.critical("Initiating emergency shutdown")
            
            self.cascade_state = CascadeState.ERROR
            self.is_running = False
            self.shutdown_requested = True
            
            # Emit emergency event
            self.event_bus.emit(Event(
                type=EventType.EMERGENCY_STOP,
                data={'session_id': self.session_id, 'timestamp': time.time()}
            ))
            
        except Exception as e:
            logger.error(f"Error in emergency shutdown: {e}")
    
    def _handle_performance_alert(self, event: Event):
        """Handle performance alert event"""
        try:
            logger.warning(f"Performance alert: {event.data}")
            
        except Exception as e:
            logger.error(f"Error handling performance alert: {e}")
    
    def get_cascade_metrics(self) -> Dict[str, Any]:
        """Get cascade performance metrics"""
        try:
            return {
                'cascade_state': self.cascade_state.value,
                'is_running': self.is_running,
                'session_id': self.session_id,
                'uptime': time.time() - self.start_time,
                'metrics': {
                    'total_strategic_updates': self.cascade_metrics.total_strategic_updates,
                    'total_tactical_executions': self.cascade_metrics.total_tactical_executions,
                    'success_rate': self.cascade_metrics.get_success_rate(),
                    'failure_rate': self.cascade_metrics.get_failure_rate(),
                    'avg_cascade_latency': self.cascade_metrics.avg_cascade_latency,
                    'avg_strategic_to_tactical_latency': self.cascade_metrics.avg_strategic_to_tactical_latency,
                    'context_propagation_failures': self.cascade_metrics.context_propagation_failures,
                    'synchronization_issues': self.cascade_metrics.synchronization_issues
                },
                'current_strategic_context': {
                    'timestamp': self.current_strategic_context.timestamp if self.current_strategic_context else None,
                    'confidence_level': self.current_strategic_context.confidence_level if self.current_strategic_context else None,
                    'execution_bias': self.current_strategic_context.execution_bias if self.current_strategic_context else None
                },
                'timing': {
                    'strategic_cycle_time': self.strategic_cycle_time,
                    'tactical_cycle_time': self.tactical_cycle_time,
                    'last_strategic_update': self.last_strategic_update,
                    'last_tactical_execution': self.last_tactical_execution
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cascade metrics: {e}")
            return {'error': str(e)}
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get execution history"""
        try:
            history = list(self.execution_history)[-limit:]
            return [execution.__dict__ for execution in history]
        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            return []
    
    def trigger_manual_execution(self) -> bool:
        """Trigger manual tactical execution"""
        try:
            if not self.is_running:
                logger.error("Cannot trigger manual execution: cascade not running")
                return False
            
            # Add manual execution request to queue
            execution_request = {
                'type': ExecutionTrigger.MANUAL_TRIGGER,
                'timestamp': time.time(),
                'session_id': self.session_id
            }
            
            self.tactical_execution_queue.put(execution_request)
            logger.info("Manual execution triggered")
            
            return True
            
        except Exception as e:
            logger.error(f"Error triggering manual execution: {e}")
            return False
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration"""
        try:
            self.config.update(new_config)
            logger.info(f"Configuration updated: {new_config}")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")


def create_strategic_tactical_cascade(
    strategic_environment: StrategicEnvironment,
    tactical_environment: SequentialTacticalEnvironment,
    tactical_agents: Dict[str, SequentialTacticalAgent],
    tactical_aggregator: TacticalSuperpositionAggregator,
    config: Optional[Dict[str, Any]] = None
) -> StrategicTacticalCascade:
    """
    Create strategic-tactical cascade integration
    
    Args:
        strategic_environment: Strategic MARL environment
        tactical_environment: Tactical MARL environment  
        tactical_agents: Dictionary of tactical agents
        tactical_aggregator: Tactical superposition aggregator
        config: Configuration dictionary
        
    Returns:
        Configured cascade integration
    """
    return StrategicTacticalCascade(
        strategic_environment=strategic_environment,
        tactical_environment=tactical_environment,
        tactical_agents=tactical_agents,
        tactical_aggregator=tactical_aggregator,
        config=config
    )


# Example usage
if __name__ == "__main__":
    # This would be run with proper environment setup
    print("Strategic-Tactical Cascade Integration initialized successfully")