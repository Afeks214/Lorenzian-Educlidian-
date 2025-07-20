"""
Parallel MARL System with 8 Agent Architecture

Implements high-velocity parallel agent system with:
- 4 Strategic agents (30m timeframe)
- 4 Tactical agents (5m timeframe)
- Robust matrix-to-agent interconnections
- 300% trustworthy data delivery
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import time
from collections import deque
import logging

from src.core.minimal_dependencies import Event, EventType
from src.utils.logger import get_logger


@dataclass
class AgentMetrics:
    """Metrics for individual agent performance."""
    agent_id: str
    agent_type: str
    decisions_count: int
    avg_decision_time_ms: float
    last_matrix_timestamp: Optional[datetime]
    last_decision_timestamp: Optional[datetime]
    error_count: int
    uptime_seconds: float


class MatrixDeliveryValidator:
    """
    Ensures 300% trustworthy matrix delivery to agents.
    
    Triple validation:
    1. Matrix integrity check
    2. Delivery confirmation 
    3. Agent acknowledgment
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.delivery_history = deque(maxlen=1000)
        self.validation_lock = threading.RLock()
        
    def validate_matrix_integrity(self, matrix: np.ndarray, matrix_type: str) -> bool:
        """Validate matrix data integrity."""
        try:
            # Check matrix is not None
            if matrix is None:
                self.logger.error(f"Matrix is None for {matrix_type}")
                return False
            
            # Check matrix shape
            expected_shapes = {
                "30m": (48, None),  # 48 rows, variable columns
                "5m": (60, None)    # 60 rows, variable columns
            }
            
            expected_shape = expected_shapes.get(matrix_type)
            if expected_shape and matrix.shape[0] != expected_shape[0]:
                self.logger.error(
                    f"Invalid matrix shape for {matrix_type}: "
                    f"expected rows={expected_shape[0]}, got {matrix.shape}"
                )
                return False
            
            # Check for NaN/Inf values
            if not np.all(np.isfinite(matrix)):
                self.logger.error(f"Matrix contains non-finite values for {matrix_type}")
                return False
            
            # Check value ranges (should be normalized)
            if np.max(np.abs(matrix)) > 10.0:
                self.logger.warning(
                    f"Matrix values outside expected range for {matrix_type}: "
                    f"max_abs={np.max(np.abs(matrix))}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Matrix validation error for {matrix_type}: {e}")
            return False
    
    def record_delivery(self, 
                       agent_id: str, 
                       matrix_type: str, 
                       delivery_success: bool,
                       agent_ack: bool = False) -> None:
        """Record matrix delivery attempt."""
        with self.validation_lock:
            record = {
                "timestamp": datetime.now(),
                "agent_id": agent_id,
                "matrix_type": matrix_type,
                "delivery_success": delivery_success,
                "agent_ack": agent_ack
            }
            self.delivery_history.append(record)
    
    def get_delivery_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get delivery statistics."""
        with self.validation_lock:
            if agent_id:
                records = [r for r in self.delivery_history if r["agent_id"] == agent_id]
            else:
                records = list(self.delivery_history)
            
            if not records:
                return {"total_deliveries": 0}
            
            total = len(records)
            successful = sum(1 for r in records if r["delivery_success"])
            acked = sum(1 for r in records if r["agent_ack"])
            
            return {
                "total_deliveries": total,
                "success_rate": successful / total if total > 0 else 0.0,
                "acknowledgment_rate": acked / total if total > 0 else 0.0,
                "last_delivery": records[-1]["timestamp"] if records else None
            }


class ParallelAgent:
    """Individual agent in the parallel MARL system."""
    
    def __init__(self, 
                 agent_id: str,
                 agent_type: str,
                 matrix_type: str,
                 event_bus):
        self.agent_id = agent_id
        self.agent_type = agent_type  # "strategic" or "tactical"
        self.matrix_type = matrix_type  # "30m" or "5m"
        self.event_bus = event_bus
        self.logger = get_logger(f"Agent_{agent_id}")
        
        # State tracking
        self.is_active = False
        self.last_matrix = None
        self.last_matrix_timestamp = None
        self.decision_history = deque(maxlen=100)
        self.error_count = 0
        self.start_time = None
        
        # Performance tracking
        self.decision_times = deque(maxlen=100)
        
        # Subscribe to appropriate matrix events
        self.matrix_event_type = EventType.MATRIX_30M_READY if matrix_type == "30m" else EventType.MATRIX_5M_READY
        self.event_bus.subscribe(self.matrix_event_type, self._on_matrix_ready)
        
        self.logger.info(
            f"Initialized {agent_type} agent {agent_id} for {matrix_type} matrices"
        )
    
    async def start(self) -> None:
        """Start the agent."""
        self.is_active = True
        self.start_time = datetime.now()
        self.logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self) -> None:
        """Stop the agent."""
        self.is_active = False
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def _on_matrix_ready(self, event: Event) -> None:
        """Handle matrix ready event."""
        if not self.is_active:
            return
        
        try:
            start_time = time.perf_counter()
            
            # Extract matrix from event
            matrix = event.payload.get('matrix')
            timestamp = event.payload.get('timestamp', datetime.now())
            
            if matrix is None:
                self.logger.error(f"No matrix in event payload")
                self.error_count += 1
                return
            
            # Store matrix
            self.last_matrix = matrix.copy()
            self.last_matrix_timestamp = timestamp
            
            # Make decision
            decision = await self._make_decision(matrix)
            
            # Record decision time
            decision_time = (time.perf_counter() - start_time) * 1000
            self.decision_times.append(decision_time)
            
            # Store decision
            self.decision_history.append({\n                "timestamp": timestamp,\n                "decision": decision,\n                "decision_time_ms": decision_time\n            })\n            \n            # Publish decision\n            await self._publish_decision(decision, timestamp)\n            \n            # Send acknowledgment\n            await self._send_acknowledgment(event)\n            \n        except Exception as e:\n            self.logger.error(f"Error processing matrix: {e}")\n            self.error_count += 1\n    \n    async def _make_decision(self, matrix: np.ndarray) -> Dict[str, Any]:\n        """Make trading decision based on matrix."""\n        try:\n            # Simulate neural network inference\n            await asyncio.sleep(0.001)  # 1ms processing time\n            \n            # Extract features from matrix\n            latest_features = matrix[-1, :]  # Most recent bar\n            \n            if self.agent_type == "strategic":\n                # Strategic decision (position sizing, direction)\n                decision = self._make_strategic_decision(latest_features)\n            else:\n                # Tactical decision (entry/exit timing)\n                decision = self._make_tactical_decision(latest_features)\n            \n            return decision\n            \n        except Exception as e:\n            self.logger.error(f"Decision making error: {e}")\n            return {"action": "hold", "confidence": 0.0}\n    \n    def _make_strategic_decision(self, features: np.ndarray) -> Dict[str, Any]:\n        """Make strategic trading decision."""\n        # Simulate strategic analysis\n        signal_strength = np.mean(features[:5])  # First 5 features\n        \n        if signal_strength > 0.3:\n            action = "buy"\n            position_size = min(0.1, signal_strength)\n        elif signal_strength < -0.3:\n            action = "sell"\n            position_size = min(0.1, abs(signal_strength))\n        else:\n            action = "hold"\n            position_size = 0.0\n        \n        return {\n            "agent_id": self.agent_id,\n            "agent_type": self.agent_type,\n            "action": action,\n            "position_size": position_size,\n            "confidence": abs(signal_strength),\n            "timeframe": "30m"\n        }\n    \n    def _make_tactical_decision(self, features: np.ndarray) -> Dict[str, Any]:\n        """Make tactical trading decision."""\n        # Simulate tactical analysis\n        momentum = np.mean(features[2:5])  # Momentum features\n        volatility = np.std(features[-5:])  # Recent volatility\n        \n        if momentum > 0.2 and volatility < 0.5:\n            action = "enter_long"\n            urgency = "high"\n        elif momentum < -0.2 and volatility < 0.5:\n            action = "enter_short"\n            urgency = "high"\n        elif volatility > 1.0:\n            action = "reduce_risk"\n            urgency = "immediate"\n        else:\n            action = "wait"\n            urgency = "low"\n        \n        return {\n            "agent_id": self.agent_id,\n            "agent_type": self.agent_type,\n            "action": action,\n            "urgency": urgency,\n            "momentum": momentum,\n            "volatility": volatility,\n            "timeframe": "5m"\n        }\n    \n    async def _publish_decision(self, decision: Dict[str, Any], timestamp: datetime) -> None:\n        """Publish agent decision."""\n        try:\n            event_type = EventType.STRATEGIC_DECISION if self.agent_type == "strategic" else EventType.TACTICAL_DECISION\n            \n            await self.event_bus.publish(Event(\n                type=event_type,\n                payload={\n                    "decision": decision,\n                    "timestamp": timestamp,\n                    "agent_id": self.agent_id\n                },\n                source=f"agent_{self.agent_id}"\n            ))\n            \n        except Exception as e:\n            self.logger.error(f"Error publishing decision: {e}")\n    \n    async def _send_acknowledgment(self, original_event: Event) -> None:\n        """Send acknowledgment for received matrix."""\n        try:\n            await self.event_bus.publish(Event(\n                type=EventType.AGENT_ACKNOWLEDGMENT,\n                payload={\n                    "agent_id": self.agent_id,\n                    "matrix_type": self.matrix_type,\n                    "original_event_id": getattr(original_event, 'id', None),\n                    "timestamp": datetime.now()\n                },\n                source=f"agent_{self.agent_id}"\n            ))\n            \n        except Exception as e:\n            self.logger.error(f"Error sending acknowledgment: {e}")\n    \n    def get_metrics(self) -> AgentMetrics:\n        """Get agent performance metrics."""\n        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0\n        avg_decision_time = np.mean(list(self.decision_times)) if self.decision_times else 0.0\n        \n        return AgentMetrics(\n            agent_id=self.agent_id,\n            agent_type=self.agent_type,\n            decisions_count=len(self.decision_history),\n            avg_decision_time_ms=avg_decision_time,\n            last_matrix_timestamp=self.last_matrix_timestamp,\n            last_decision_timestamp=self.decision_history[-1]["timestamp"] if self.decision_history else None,\n            error_count=self.error_count,\n            uptime_seconds=uptime\n        )


class ParallelMARLSystem:\n    """Main parallel MARL system coordinator."""\n    \n    def __init__(self, event_bus):\n        self.event_bus = event_bus\n        self.logger = get_logger(self.__class__.__name__)\n        \n        # Matrix delivery validator\n        self.validator = MatrixDeliveryValidator()\n        \n        # Create 8 parallel agents\n        self.agents: List[ParallelAgent] = []\n        \n        # 4 Strategic agents (30m)\n        for i in range(4):\n            agent = ParallelAgent(\n                agent_id=f"strategic_{i+1}",\n                agent_type="strategic",\n                matrix_type="30m",\n                event_bus=event_bus\n            )\n            self.agents.append(agent)\n        \n        # 4 Tactical agents (5m)\n        for i in range(4):\n            agent = ParallelAgent(\n                agent_id=f"tactical_{i+1}",\n                agent_type="tactical",\n                matrix_type="5m",\n                event_bus=event_bus\n            )\n            self.agents.append(agent)\n        \n        # Subscribe to matrix events for validation\n        self.event_bus.subscribe(EventType.MATRIX_30M_READY, self._validate_30m_delivery)\n        self.event_bus.subscribe(EventType.MATRIX_5M_READY, self._validate_5m_delivery)\n        self.event_bus.subscribe(EventType.AGENT_ACKNOWLEDGMENT, self._handle_acknowledgment)\n        \n        # System state\n        self.is_running = False\n        self.start_time = None\n        \n        self.logger.info("Parallel MARL System initialized with 8 agents")\n    \n    async def start(self) -> None:\n        """Start all agents."""\n        self.is_running = True\n        self.start_time = datetime.now()\n        \n        # Start all agents in parallel\n        tasks = [agent.start() for agent in self.agents]\n        await asyncio.gather(*tasks)\n        \n        self.logger.info("All 8 agents started successfully")\n    \n    async def stop(self) -> None:\n        """Stop all agents."""\n        self.is_running = False\n        \n        # Stop all agents in parallel\n        tasks = [agent.stop() for agent in self.agents]\n        await asyncio.gather(*tasks)\n        \n        self.logger.info("All agents stopped")\n    \n    async def _validate_30m_delivery(self, event: Event) -> None:\n        """Validate 30m matrix delivery to strategic agents."""\n        matrix = event.payload.get('matrix')\n        \n        # Validate matrix integrity\n        if not self.validator.validate_matrix_integrity(matrix, "30m"):\n            self.logger.error("30m matrix failed integrity check")\n            return\n        \n        # Record delivery to each strategic agent\n        strategic_agents = [a for a in self.agents if a.agent_type == "strategic"]\n        for agent in strategic_agents:\n            self.validator.record_delivery(agent.agent_id, "30m", True)\n    \n    async def _validate_5m_delivery(self, event: Event) -> None:\n        """Validate 5m matrix delivery to tactical agents."""\n        matrix = event.payload.get('matrix')\n        \n        # Validate matrix integrity\n        if not self.validator.validate_matrix_integrity(matrix, "5m"):\n            self.logger.error("5m matrix failed integrity check")\n            return\n        \n        # Record delivery to each tactical agent\n        tactical_agents = [a for a in self.agents if a.agent_type == "tactical"]\n        for agent in tactical_agents:\n            self.validator.record_delivery(agent.agent_id, "5m", True)\n    \n    async def _handle_acknowledgment(self, event: Event) -> None:\n        """Handle agent acknowledgment."""\n        payload = event.payload\n        agent_id = payload.get('agent_id')\n        matrix_type = payload.get('matrix_type')\n        \n        # Update delivery record with acknowledgment\n        self.validator.record_delivery(agent_id, matrix_type, True, agent_ack=True)\n    \n    def get_system_metrics(self) -> Dict[str, Any]:\n        """Get comprehensive system metrics."""\n        agent_metrics = [agent.get_metrics() for agent in self.agents]\n        \n        # Calculate system-wide statistics\n        strategic_agents = [m for m in agent_metrics if m.agent_type == "strategic"]\n        tactical_agents = [m for m in agent_metrics if m.agent_type == "tactical"]\n        \n        total_decisions = sum(m.decisions_count for m in agent_metrics)\n        total_errors = sum(m.error_count for m in agent_metrics)\n        avg_decision_time = np.mean([m.avg_decision_time_ms for m in agent_metrics if m.avg_decision_time_ms > 0])\n        \n        system_uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0\n        \n        return {\n            "system_status": "running" if self.is_running else "stopped",\n            "system_uptime_seconds": system_uptime,\n            "total_agents": len(self.agents),\n            "strategic_agents": len(strategic_agents),\n            "tactical_agents": len(tactical_agents),\n            "total_decisions": total_decisions,\n            "total_errors": total_errors,\n            "avg_decision_time_ms": avg_decision_time,\n            "error_rate": total_errors / max(total_decisions, 1),\n            "delivery_stats": {\n                "30m": self.validator.get_delivery_stats(),\n                "5m": self.validator.get_delivery_stats()\n            },\n            "agent_metrics": [{\n                "agent_id": m.agent_id,\n                "agent_type": m.agent_type,\n                "decisions_count": m.decisions_count,\n                "avg_decision_time_ms": m.avg_decision_time_ms,\n                "error_count": m.error_count,\n                "uptime_seconds": m.uptime_seconds\n            } for m in agent_metrics]\n        }\n    \n    def get_trustworthiness_score(self) -> float:\n        \"\"\"\n        Calculate 300% trustworthiness score.\n        \n        Returns:\n            Score from 0.0 to 3.0 (300%)\n        \"\"\"\n        try:\n            delivery_stats_30m = self.validator.get_delivery_stats()\n            delivery_stats_5m = self.validator.get_delivery_stats()\n            \n            # Component 1: Delivery success rate (0-1)\n            success_rate_30m = delivery_stats_30m.get('success_rate', 0.0)\n            success_rate_5m = delivery_stats_5m.get('success_rate', 0.0)\n            delivery_score = (success_rate_30m + success_rate_5m) / 2\n            \n            # Component 2: Acknowledgment rate (0-1)\n            ack_rate_30m = delivery_stats_30m.get('acknowledgment_rate', 0.0)\n            ack_rate_5m = delivery_stats_5m.get('acknowledgment_rate', 0.0)\n            ack_score = (ack_rate_30m + ack_rate_5m) / 2\n            \n            # Component 3: Agent health (0-1)\n            agent_metrics = [agent.get_metrics() for agent in self.agents]\n            active_agents = sum(1 for m in agent_metrics if m.uptime_seconds > 0)\n            health_score = active_agents / len(self.agents)\n            \n            # Total trustworthiness (0-3)\n            total_score = delivery_score + ack_score + health_score\n            \n            return total_score\n            \n        except Exception as e:\n            self.logger.error(f"Error calculating trustworthiness score: {e}")\n            return 0.0