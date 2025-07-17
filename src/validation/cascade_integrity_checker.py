"""
Cascade Integrity Checker for MARL Systems.

This module ensures that the cascade of Multi-Agent Reinforcement Learning (MARL)
systems maintains integrity throughout the superposition framework. It validates:

- Inter-agent coordination and communication integrity
- Information flow consistency across agent hierarchies
- Cascade stability under various market conditions
- Agent synchronization and temporal consistency
- Error propagation and fault tolerance

The checker ensures that the entire MARL cascade operates reliably
and maintains system integrity under all conditions.
"""

import numpy as np
import torch
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import time
from collections import defaultdict, deque
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


class CascadeLevel(Enum):
    """Levels in the MARL cascade hierarchy."""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    EXECUTION = "execution"
    RISK_MANAGEMENT = "risk_management"
    COORDINATION = "coordination"


class IntegrityStatus(Enum):
    """Integrity status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    COMPROMISED = "compromised"
    FAILED = "failed"


class CommunicationChannel(Enum):
    """Communication channels between agents."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"


@dataclass
class AgentNode:
    """Represents an agent in the MARL cascade."""
    agent_id: str
    level: CascadeLevel
    expected_latency_ms: float
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    communication_channels: Dict[str, CommunicationChannel] = field(default_factory=dict)
    health_status: IntegrityStatus = IntegrityStatus.HEALTHY
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'level': self.level.value,
            'expected_latency_ms': self.expected_latency_ms,
            'dependencies': self.dependencies,
            'dependents': self.dependents,
            'communication_channels': {k: v.value for k, v in self.communication_channels.items()},
            'health_status': self.health_status.value,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'metadata': self.metadata
        }


@dataclass
class CascadeIntegrityReport:
    """Report on cascade integrity status."""
    timestamp: datetime
    overall_status: IntegrityStatus
    agent_statuses: Dict[str, IntegrityStatus]
    communication_health: Dict[str, float]
    dependency_violations: List[str]
    performance_issues: List[str]
    recovery_suggestions: List[str]
    cascade_latency_ms: float
    error_propagation_risk: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status.value,
            'agent_statuses': {k: v.value for k, v in self.agent_statuses.items()},
            'communication_health': self.communication_health,
            'dependency_violations': self.dependency_violations,
            'performance_issues': self.performance_issues,
            'recovery_suggestions': self.recovery_suggestions,
            'cascade_latency_ms': self.cascade_latency_ms,
            'error_propagation_risk': self.error_propagation_risk
        }


class CascadeIntegrityChecker:
    """
    Comprehensive cascade integrity checker for MARL systems.
    
    This checker ensures that the entire cascade of MARL agents maintains
    integrity, proper communication, and consistent performance throughout
    the superposition framework.
    """
    
    def __init__(
        self,
        max_cascade_latency_ms: float = 10.0,
        heartbeat_interval_ms: float = 1000.0,
        dependency_timeout_ms: float = 5000.0,
        communication_timeout_ms: float = 2000.0,
        error_threshold: float = 0.1
    ):
        """
        Initialize the cascade integrity checker.
        
        Args:
            max_cascade_latency_ms: Maximum allowed cascade latency
            heartbeat_interval_ms: Interval for agent heartbeats
            dependency_timeout_ms: Timeout for dependency responses
            communication_timeout_ms: Timeout for communication channels
            error_threshold: Threshold for error propagation risk
        """
        self.max_cascade_latency_ms = max_cascade_latency_ms
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.dependency_timeout_ms = dependency_timeout_ms
        self.communication_timeout_ms = communication_timeout_ms
        self.error_threshold = error_threshold
        
        # Setup logging
        self.logger = logging.getLogger('cascade_integrity_checker')
        self.logger.setLevel(logging.INFO)
        
        # Agent registry and cascade graph
        self.agents: Dict[str, AgentNode] = {}
        self.cascade_graph: nx.DiGraph = nx.DiGraph()
        
        # Communication monitoring
        self.communication_log: deque = deque(maxlen=10000)
        self.communication_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Integrity monitoring
        self.integrity_history: deque = deque(maxlen=1000)
        self.active_violations: Set[str] = set()
        
        # Performance tracking
        self.cascade_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Thread safety
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        # Recovery mechanisms
        self.recovery_strategies: Dict[str, callable] = {}
        self.auto_recovery_enabled = True
        
        self.logger.info(f"Cascade integrity checker initialized with "
                        f"max_latency={max_cascade_latency_ms}ms")
    
    def register_agent(
        self,
        agent_id: str,
        level: CascadeLevel,
        expected_latency_ms: float,
        dependencies: Optional[List[str]] = None,
        communication_channels: Optional[Dict[str, CommunicationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an agent in the cascade.
        
        Args:
            agent_id: Unique identifier for the agent
            level: Level in the cascade hierarchy
            expected_latency_ms: Expected response latency
            dependencies: List of agent IDs this agent depends on
            communication_channels: Communication channels to other agents
            metadata: Additional metadata
        """
        with self._lock:
            # Create agent node
            agent_node = AgentNode(
                agent_id=agent_id,
                level=level,
                expected_latency_ms=expected_latency_ms,
                dependencies=dependencies or [],
                communication_channels=communication_channels or {},
                metadata=metadata or {}
            )
            
            # Update dependents for existing agents
            for dep_id in agent_node.dependencies:
                if dep_id in self.agents:
                    self.agents[dep_id].dependents.append(agent_id)
            
            # Register agent
            self.agents[agent_id] = agent_node
            
            # Update cascade graph
            self.cascade_graph.add_node(agent_id, **agent_node.to_dict())
            
            # Add dependency edges
            for dep_id in agent_node.dependencies:
                if dep_id in self.agents:
                    self.cascade_graph.add_edge(dep_id, agent_id)
            
            self.logger.info(f"Registered agent {agent_id} at level {level.value}")
    
    def update_agent_heartbeat(self, agent_id: str, timestamp: Optional[datetime] = None) -> None:
        """
        Update agent heartbeat timestamp.
        
        Args:
            agent_id: Agent identifier
            timestamp: Heartbeat timestamp (default: now)
        """
        if agent_id not in self.agents:
            self.logger.warning(f"Unknown agent {agent_id} sent heartbeat")
            return
        
        with self._lock:
            self.agents[agent_id].last_heartbeat = timestamp or datetime.now()
    
    def log_communication(
        self,
        from_agent: str,
        to_agent: str,
        channel: CommunicationChannel,
        latency_ms: float,
        success: bool,
        payload_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log communication between agents.
        
        Args:
            from_agent: Source agent ID
            to_agent: Destination agent ID
            channel: Communication channel used
            latency_ms: Communication latency
            success: Whether communication was successful
            payload_size: Size of payload in bytes
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now(),
            'from_agent': from_agent,
            'to_agent': to_agent,
            'channel': channel.value,
            'latency_ms': latency_ms,
            'success': success,
            'payload_size': payload_size,
            'metadata': metadata or {}
        }
        
        with self._lock:
            self.communication_log.append(log_entry)
            
            # Update communication statistics
            comm_key = f"{from_agent}->{to_agent}"
            stats = self.communication_stats[comm_key]
            stats['total_attempts'] += 1
            stats['total_latency_ms'] += latency_ms
            
            if success:
                stats['successful_attempts'] += 1
            else:
                stats['failed_attempts'] += 1
            
            # Update averages
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total_attempts']
            stats['success_rate'] = stats['successful_attempts'] / stats['total_attempts']
    
    def check_cascade_integrity(self) -> CascadeIntegrityReport:
        """
        Perform comprehensive cascade integrity check.
        
        Returns:
            Detailed integrity report
        """
        start_time = time.perf_counter()
        
        # Check individual agent health
        agent_statuses = self._check_agent_health()
        
        # Check communication health
        communication_health = self._check_communication_health()
        
        # Check dependency violations
        dependency_violations = self._check_dependency_violations()
        
        # Check performance issues
        performance_issues = self._check_performance_issues()
        
        # Calculate cascade latency
        cascade_latency_ms = self._calculate_cascade_latency()
        
        # Calculate error propagation risk
        error_propagation_risk = self._calculate_error_propagation_risk()
        
        # Generate recovery suggestions
        recovery_suggestions = self._generate_recovery_suggestions(
            agent_statuses, dependency_violations, performance_issues
        )
        
        # Determine overall status
        overall_status = self._determine_overall_status(
            agent_statuses, communication_health, dependency_violations,
            performance_issues, cascade_latency_ms, error_propagation_risk
        )
        
        # Create report
        report = CascadeIntegrityReport(
            timestamp=datetime.now(),
            overall_status=overall_status,
            agent_statuses=agent_statuses,
            communication_health=communication_health,
            dependency_violations=dependency_violations,
            performance_issues=performance_issues,
            recovery_suggestions=recovery_suggestions,
            cascade_latency_ms=cascade_latency_ms,
            error_propagation_risk=error_propagation_risk
        )
        
        # Store in history
        with self._lock:
            self.integrity_history.append(report)
        
        # Log completion time
        check_time_ms = (time.perf_counter() - start_time) * 1000
        self.logger.info(f"Cascade integrity check completed in {check_time_ms:.2f}ms, "
                        f"status: {overall_status.value}")
        
        return report
    
    def _check_agent_health(self) -> Dict[str, IntegrityStatus]:
        """Check health status of all agents."""
        agent_statuses = {}
        current_time = datetime.now()
        
        for agent_id, agent in self.agents.items():
            status = IntegrityStatus.HEALTHY
            
            # Check heartbeat freshness
            if agent.last_heartbeat:
                heartbeat_age = (current_time - agent.last_heartbeat).total_seconds() * 1000
                if heartbeat_age > self.heartbeat_interval_ms * 3:  # 3x interval threshold
                    status = IntegrityStatus.DEGRADED
                if heartbeat_age > self.heartbeat_interval_ms * 5:  # 5x interval threshold
                    status = IntegrityStatus.FAILED
            else:
                # No heartbeat received
                status = IntegrityStatus.COMPROMISED
            
            agent_statuses[agent_id] = status
            
            # Update agent health status
            agent.health_status = status
        
        return agent_statuses
    
    def _check_communication_health(self) -> Dict[str, float]:
        """Check health of communication channels."""
        communication_health = {}
        
        for comm_key, stats in self.communication_stats.items():
            if stats['total_attempts'] == 0:
                health_score = 0.0
            else:
                # Health score based on success rate and latency
                success_score = stats['success_rate']
                
                # Latency score (inversely proportional to latency)
                avg_latency = stats['avg_latency_ms']
                latency_score = max(0, 1 - (avg_latency / self.communication_timeout_ms))
                
                # Combined health score
                health_score = 0.7 * success_score + 0.3 * latency_score
            
            communication_health[comm_key] = health_score
        
        return communication_health
    
    def _check_dependency_violations(self) -> List[str]:
        """Check for dependency violations in the cascade."""
        violations = []
        
        for agent_id, agent in self.agents.items():
            # Check if dependencies are healthy
            for dep_id in agent.dependencies:
                if dep_id not in self.agents:
                    violations.append(f"Agent {agent_id} depends on unknown agent {dep_id}")
                    continue
                
                dep_agent = self.agents[dep_id]
                if dep_agent.health_status in [IntegrityStatus.COMPROMISED, IntegrityStatus.FAILED]:
                    violations.append(f"Agent {agent_id} depends on unhealthy agent {dep_id}")
            
            # Check for circular dependencies
            if self._has_circular_dependency(agent_id):
                violations.append(f"Circular dependency detected involving agent {agent_id}")
            
            # Check dependency timeout
            if self._has_dependency_timeout(agent_id):
                violations.append(f"Dependency timeout detected for agent {agent_id}")
        
        return violations
    
    def _check_performance_issues(self) -> List[str]:
        """Check for performance issues in the cascade."""
        issues = []
        
        # Check cascade latency
        cascade_latency = self._calculate_cascade_latency()
        if cascade_latency > self.max_cascade_latency_ms:
            issues.append(f"Cascade latency {cascade_latency:.2f}ms exceeds limit {self.max_cascade_latency_ms}ms")
        
        # Check individual agent performance
        for agent_id, agent in self.agents.items():
            recent_performance = self._get_recent_performance(agent_id)
            if recent_performance:
                avg_latency = np.mean(recent_performance)
                if avg_latency > agent.expected_latency_ms * 2:
                    issues.append(f"Agent {agent_id} performance degraded: {avg_latency:.2f}ms vs expected {agent.expected_latency_ms}ms")
        
        # Check communication bottlenecks
        for comm_key, health_score in self._check_communication_health().items():
            if health_score < 0.5:
                issues.append(f"Communication bottleneck detected: {comm_key} (health: {health_score:.2f})")
        
        return issues
    
    def _calculate_cascade_latency(self) -> float:
        """Calculate total cascade latency."""
        if not self.cascade_graph.nodes:
            return 0.0
        
        try:
            # Find longest path in the cascade (critical path)
            longest_path_length = 0
            
            # Get topological sort to find valid orderings
            if nx.is_directed_acyclic_graph(self.cascade_graph):
                topo_sort = list(nx.topological_sort(self.cascade_graph))
                
                # Calculate longest path using dynamic programming
                distances = {node: 0 for node in self.cascade_graph.nodes}
                
                for node in topo_sort:
                    agent = self.agents[node]
                    current_distance = distances[node] + agent.expected_latency_ms
                    
                    for successor in self.cascade_graph.successors(node):
                        distances[successor] = max(distances[successor], current_distance)
                
                longest_path_length = max(distances.values()) if distances else 0
            else:
                # Graph has cycles, use approximation
                longest_path_length = sum(agent.expected_latency_ms for agent in self.agents.values())
            
            return longest_path_length
            
        except Exception as e:
            self.logger.error(f"Error calculating cascade latency: {e}")
            return float('inf')
    
    def _calculate_error_propagation_risk(self) -> float:
        """Calculate risk of error propagation through the cascade."""
        if not self.agents:
            return 0.0
        
        # Calculate based on unhealthy agents and their connectivity
        unhealthy_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.health_status in [IntegrityStatus.DEGRADED, IntegrityStatus.COMPROMISED, IntegrityStatus.FAILED]
        ]
        
        if not unhealthy_agents:
            return 0.0
        
        # Risk increases with number of unhealthy agents and their connectivity
        unhealthy_ratio = len(unhealthy_agents) / len(self.agents)
        
        # Calculate connectivity impact
        connectivity_impact = 0.0
        for agent_id in unhealthy_agents:
            # Impact is proportional to number of dependents
            num_dependents = len(self.agents[agent_id].dependents)
            connectivity_impact += num_dependents / len(self.agents)
        
        # Combined risk score
        error_propagation_risk = min(1.0, unhealthy_ratio * 0.5 + connectivity_impact * 0.5)
        
        return error_propagation_risk
    
    def _generate_recovery_suggestions(
        self,
        agent_statuses: Dict[str, IntegrityStatus],
        dependency_violations: List[str],
        performance_issues: List[str]
    ) -> List[str]:
        """Generate recovery suggestions based on detected issues."""
        suggestions = []
        
        # Suggestions for unhealthy agents
        for agent_id, status in agent_statuses.items():
            if status == IntegrityStatus.FAILED:
                suggestions.append(f"Restart agent {agent_id} immediately")
            elif status == IntegrityStatus.COMPROMISED:
                suggestions.append(f"Investigate agent {agent_id} communication issues")
            elif status == IntegrityStatus.DEGRADED:
                suggestions.append(f"Monitor agent {agent_id} performance closely")
        
        # Suggestions for dependency violations
        for violation in dependency_violations:
            if "circular dependency" in violation.lower():
                suggestions.append("Resolve circular dependencies by redesigning agent interactions")
            elif "timeout" in violation.lower():
                suggestions.append("Increase timeout thresholds or optimize agent performance")
            elif "unknown agent" in violation.lower():
                suggestions.append("Ensure all dependent agents are properly registered")
        
        # Suggestions for performance issues
        for issue in performance_issues:
            if "cascade latency" in issue.lower():
                suggestions.append("Optimize critical path agents to reduce cascade latency")
            elif "performance degraded" in issue.lower():
                suggestions.append("Investigate and optimize underperforming agents")
            elif "communication bottleneck" in issue.lower():
                suggestions.append("Optimize communication channels and reduce payload sizes")
        
        # General suggestions
        if len(agent_statuses) > 0:
            unhealthy_ratio = len([s for s in agent_statuses.values() 
                                 if s != IntegrityStatus.HEALTHY]) / len(agent_statuses)
            
            if unhealthy_ratio > 0.3:
                suggestions.append("Consider system-wide restart due to high failure rate")
            elif unhealthy_ratio > 0.1:
                suggestions.append("Implement graceful degradation strategies")
        
        return suggestions
    
    def _determine_overall_status(
        self,
        agent_statuses: Dict[str, IntegrityStatus],
        communication_health: Dict[str, float],
        dependency_violations: List[str],
        performance_issues: List[str],
        cascade_latency_ms: float,
        error_propagation_risk: float
    ) -> IntegrityStatus:
        """Determine overall cascade integrity status."""
        # Check for critical failures
        failed_agents = [s for s in agent_statuses.values() if s == IntegrityStatus.FAILED]
        if len(failed_agents) > 0:
            return IntegrityStatus.FAILED
        
        # Check for compromised state
        compromised_agents = [s for s in agent_statuses.values() if s == IntegrityStatus.COMPROMISED]
        if len(compromised_agents) > 0 or len(dependency_violations) > 0:
            return IntegrityStatus.COMPROMISED
        
        # Check for degraded state
        degraded_agents = [s for s in agent_statuses.values() if s == IntegrityStatus.DEGRADED]
        poor_communication = [h for h in communication_health.values() if h < 0.5]
        
        if (len(degraded_agents) > 0 or 
            len(poor_communication) > 0 or 
            len(performance_issues) > 0 or
            cascade_latency_ms > self.max_cascade_latency_ms or
            error_propagation_risk > self.error_threshold):
            return IntegrityStatus.DEGRADED
        
        return IntegrityStatus.HEALTHY
    
    def _has_circular_dependency(self, agent_id: str) -> bool:
        """Check if agent is part of a circular dependency."""
        try:
            # Use DFS to detect cycles starting from this agent
            visited = set()
            rec_stack = set()
            
            def dfs(node):
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                
                visited.add(node)
                rec_stack.add(node)
                
                for dep in self.agents.get(node, AgentNode("", CascadeLevel.STRATEGIC, 0)).dependencies:
                    if dfs(dep):
                        return True
                
                rec_stack.remove(node)
                return False
            
            return dfs(agent_id)
            
        except Exception as e:
            self.logger.error(f"Error checking circular dependency for {agent_id}: {e}")
            return False
    
    def _has_dependency_timeout(self, agent_id: str) -> bool:
        """Check if agent has dependency timeout issues."""
        # Check recent communication logs for timeouts
        recent_time = datetime.now() - timedelta(milliseconds=self.dependency_timeout_ms)
        
        for log_entry in self.communication_log:
            if (log_entry['timestamp'] > recent_time and
                log_entry['to_agent'] == agent_id and
                log_entry['latency_ms'] > self.dependency_timeout_ms):
                return True
        
        return False
    
    def _get_recent_performance(self, agent_id: str) -> List[float]:
        """Get recent performance metrics for an agent."""
        return list(self.cascade_performance.get(agent_id, []))
    
    def record_agent_performance(self, agent_id: str, latency_ms: float) -> None:
        """Record performance metric for an agent."""
        with self._lock:
            self.cascade_performance[agent_id].append(latency_ms)
    
    def start_monitoring(self) -> None:
        """Start continuous cascade monitoring."""
        if self._monitoring_active:
            self.logger.warning("Cascade monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Started cascade integrity monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop continuous cascade monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        
        self.logger.info("Stopped cascade integrity monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Perform integrity check
                report = self.check_cascade_integrity()
                
                # Handle auto-recovery if enabled
                if self.auto_recovery_enabled and report.overall_status in [
                    IntegrityStatus.DEGRADED, IntegrityStatus.COMPROMISED
                ]:
                    self._attempt_auto_recovery(report)
                
                # Sleep for monitoring interval
                time.sleep(self.heartbeat_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Error in cascade monitoring loop: {e}")
                time.sleep(1.0)  # Prevent tight loop on error
    
    def _attempt_auto_recovery(self, report: CascadeIntegrityReport) -> None:
        """Attempt automatic recovery based on integrity report."""
        for suggestion in report.recovery_suggestions:
            # Simple recovery strategies
            if "restart agent" in suggestion.lower():
                # Extract agent ID and attempt restart
                # This would integrate with actual agent management system
                self.logger.info(f"Auto-recovery suggestion: {suggestion}")
            
            elif "increase timeout" in suggestion.lower():
                # Temporarily increase timeouts
                self.dependency_timeout_ms *= 1.5
                self.communication_timeout_ms *= 1.5
                self.logger.info(f"Auto-recovery: Increased timeout thresholds")
    
    def get_cascade_topology(self) -> Dict[str, Any]:
        """Get current cascade topology information."""
        with self._lock:
            return {
                'nodes': len(self.cascade_graph.nodes),
                'edges': len(self.cascade_graph.edges),
                'is_acyclic': nx.is_directed_acyclic_graph(self.cascade_graph),
                'levels': {level.value: len([a for a in self.agents.values() if a.level == level]) 
                          for level in CascadeLevel},
                'critical_path_length': self._calculate_cascade_latency(),
                'average_connectivity': (sum(len(agent.dependencies) + len(agent.dependents) 
                                           for agent in self.agents.values()) / 
                                       max(1, len(self.agents)) if self.agents else 0)
            }
    
    def export_cascade_data(self) -> Dict[str, Any]:
        """Export cascade data for analysis."""
        with self._lock:
            return {
                'agents': {agent_id: agent.to_dict() for agent_id, agent in self.agents.items()},
                'communication_stats': dict(self.communication_stats),
                'recent_integrity_reports': [report.to_dict() for report in list(self.integrity_history)[-10:]],
                'cascade_topology': self.get_cascade_topology(),
                'configuration': {
                    'max_cascade_latency_ms': self.max_cascade_latency_ms,
                    'heartbeat_interval_ms': self.heartbeat_interval_ms,
                    'dependency_timeout_ms': self.dependency_timeout_ms,
                    'communication_timeout_ms': self.communication_timeout_ms,
                    'error_threshold': self.error_threshold
                }
            }
    
    def register_recovery_strategy(self, issue_type: str, strategy: callable) -> None:
        """Register a recovery strategy for a specific issue type."""
        self.recovery_strategies[issue_type] = strategy
        self.logger.info(f"Registered recovery strategy for {issue_type}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()