"""
Data Lineage Tracking System
Agent Delta: Data Pipeline Transformation Specialist

Comprehensive data lineage tracking system that provides complete visibility
into data transformations, dependencies, and quality throughout the pipeline.
Implements immutable audit trails, dependency graphs, and impact analysis.

Key Features:
- Immutable lineage records with cryptographic integrity
- Real-time dependency graph construction
- Transformation impact analysis
- Data quality propagation tracking
- Compliance and audit trail support
- Performance-optimized lineage queries
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import hashlib
import json
import structlog
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from abc import ABC, abstractmethod

logger = structlog.get_logger(__name__)

# =============================================================================
# LINEAGE ENUMERATIONS
# =============================================================================

class LineageEventType(str, Enum):
    """Types of lineage events"""
    CREATED = "created"
    TRANSFORMED = "transformed"
    VALIDATED = "validated"
    AGGREGATED = "aggregated"
    ENRICHED = "enriched"
    FILTERED = "filtered"
    JOINED = "joined"
    SPLIT = "split"
    DELETED = "deleted"
    ARCHIVED = "archived"

class DataFlowDirection(str, Enum):
    """Data flow direction"""
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    BIDIRECTIONAL = "bidirectional"

class LineageStatus(str, Enum):
    """Lineage record status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    INVALID = "invalid"

class TransformationType(str, Enum):
    """Types of data transformations"""
    NORMALIZATION = "normalization"
    AGGREGATION = "aggregation"
    ENRICHMENT = "enrichment"
    VALIDATION = "validation"
    FILTERING = "filtering"
    JOINING = "joining"
    SPLITTING = "splitting"
    CALCULATION = "calculation"
    FORMATTING = "formatting"
    CLEANING = "cleaning"

# =============================================================================
# CORE LINEAGE STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class LineageNode:
    """Immutable lineage node representing a data entity"""
    
    node_id: str
    data_type: str
    schema_version: str
    timestamp: datetime
    
    # Data identification
    data_hash: str
    data_signature: str
    
    # Metadata
    source_system: str
    component: str
    quality_score: float
    
    # Context
    business_context: Dict[str, Any] = field(default_factory=dict)
    technical_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate node integrity"""
        if not (0 <= self.quality_score <= 1):
            raise ValueError("Quality score must be between 0 and 1")
        
        if not self.node_id:
            raise ValueError("Node ID cannot be empty")

@dataclass(frozen=True)
class LineageEdge:
    """Immutable lineage edge representing a transformation"""
    
    edge_id: str
    source_node_id: str
    target_node_id: str
    transformation_type: TransformationType
    timestamp: datetime
    
    # Transformation details
    transformation_code: Optional[str] = None
    transformation_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Quality impact
    quality_impact: float = 0.0  # Change in quality score
    success: bool = True
    error_message: Optional[str] = None
    
    # Performance metrics
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Validation
    validation_rules: List[str] = field(default_factory=list)
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)

@dataclass
class LineageTrace:
    """Complete lineage trace for a data entity"""
    
    trace_id: str
    root_node_id: str
    target_node_id: str
    
    # Trace data
    nodes: List[LineageNode] = field(default_factory=list)
    edges: List[LineageEdge] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Quality metrics
    overall_quality_score: float = 0.0
    quality_degradation: float = 0.0
    
    # Path analysis
    critical_path: List[str] = field(default_factory=list)
    transformation_count: int = 0
    total_processing_time_ms: float = 0.0
    
    def update_metrics(self):
        """Update calculated metrics"""
        self.transformation_count = len(self.edges)
        self.total_processing_time_ms = sum(edge.execution_time_ms for edge in self.edges)
        
        if self.nodes:
            self.overall_quality_score = sum(node.quality_score for node in self.nodes) / len(self.nodes)
            
            # Calculate quality degradation
            if len(self.nodes) > 1:
                initial_quality = self.nodes[0].quality_score
                final_quality = self.nodes[-1].quality_score
                self.quality_degradation = max(0, initial_quality - final_quality)
        
        self.last_updated = datetime.utcnow()

# =============================================================================
# LINEAGE TRACKING ENGINE
# =============================================================================

class LineageTracker:
    """Core lineage tracking engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Storage
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[str, LineageEdge] = {}
        self.traces: Dict[str, LineageTrace] = {}
        
        # Graph structures
        self.dependency_graph = nx.DiGraph()
        self.reverse_dependency_graph = nx.DiGraph()
        
        # Indices for fast lookup
        self.node_by_data_hash: Dict[str, str] = {}
        self.edges_by_source: Dict[str, List[str]] = defaultdict(list)
        self.edges_by_target: Dict[str, List[str]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Performance monitoring
        self.operation_stats = {
            'nodes_created': 0,
            'edges_created': 0,
            'traces_built': 0,
            'queries_executed': 0,
            'avg_query_time_ms': 0.0
        }
        
        # Background processing
        self.processing_queue = deque()
        self.processing_thread = None
        self.running = False
        
        logger.info("Lineage tracker initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'max_nodes': 100000,
            'max_edges': 500000,
            'max_trace_depth': 50,
            'enable_background_processing': True,
            'batch_size': 100,
            'retention_days': 30,
            'enable_integrity_checks': True,
            'enable_performance_monitoring': True
        }
    
    def start(self):
        """Start background processing"""
        if self.config['enable_background_processing'] and not self.running:
            self.running = True
            self.processing_thread = threading.Thread(
                target=self._background_processor,
                name="lineage_processor"
            )
            self.processing_thread.start()
            logger.info("Lineage background processing started")
    
    def stop(self):
        """Stop background processing"""
        if self.running:
            self.running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=5.0)
            logger.info("Lineage background processing stopped")
    
    def create_node(self, 
                   data_type: str,
                   data_hash: str,
                   source_system: str,
                   component: str,
                   quality_score: float,
                   business_context: Optional[Dict[str, Any]] = None,
                   technical_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new lineage node
        
        Args:
            data_type: Type of data
            data_hash: Hash of the data
            source_system: Source system identifier
            component: Component identifier
            quality_score: Quality score (0-1)
            business_context: Business context information
            technical_context: Technical context information
            
        Returns:
            str: Node ID
        """
        with self.lock:
            node_id = str(uuid.uuid4())
            
            # Create data signature
            signature_data = {
                'data_type': data_type,
                'data_hash': data_hash,
                'source_system': source_system,
                'component': component,
                'timestamp': datetime.utcnow().isoformat()
            }
            data_signature = hashlib.sha256(
                json.dumps(signature_data, sort_keys=True).encode()
            ).hexdigest()
            
            # Create node
            node = LineageNode(
                node_id=node_id,
                data_type=data_type,
                schema_version="1.0",
                timestamp=datetime.utcnow(),
                data_hash=data_hash,
                data_signature=data_signature,
                source_system=source_system,
                component=component,
                quality_score=quality_score,
                business_context=business_context or {},
                technical_context=technical_context or {}
            )
            
            # Store node
            self.nodes[node_id] = node
            self.node_by_data_hash[data_hash] = node_id
            
            # Update graph
            self.dependency_graph.add_node(node_id, **{
                'data_type': data_type,
                'quality_score': quality_score,
                'timestamp': node.timestamp
            })
            
            # Update statistics
            self.operation_stats['nodes_created'] += 1
            
            logger.debug(f"Created lineage node: {node_id}")
            return node_id
    
    def create_edge(self,
                   source_node_id: str,
                   target_node_id: str,
                   transformation_type: TransformationType,
                   transformation_code: Optional[str] = None,
                   transformation_parameters: Optional[Dict[str, Any]] = None,
                   quality_impact: float = 0.0,
                   success: bool = True,
                   error_message: Optional[str] = None,
                   execution_time_ms: float = 0.0,
                   memory_usage_mb: float = 0.0,
                   validation_rules: Optional[List[str]] = None,
                   validation_passed: bool = True,
                   validation_errors: Optional[List[str]] = None) -> str:
        """
        Create a new lineage edge
        
        Args:
            source_node_id: Source node ID
            target_node_id: Target node ID
            transformation_type: Type of transformation
            transformation_code: Optional transformation code
            transformation_parameters: Transformation parameters
            quality_impact: Impact on quality score
            success: Whether transformation succeeded
            error_message: Error message if failed
            execution_time_ms: Execution time in milliseconds
            memory_usage_mb: Memory usage in MB
            validation_rules: Validation rules applied
            validation_passed: Whether validation passed
            validation_errors: Validation errors
            
        Returns:
            str: Edge ID
        """
        with self.lock:
            # Validate nodes exist
            if source_node_id not in self.nodes:
                raise ValueError(f"Source node {source_node_id} not found")
            if target_node_id not in self.nodes:
                raise ValueError(f"Target node {target_node_id} not found")
            
            edge_id = str(uuid.uuid4())
            
            # Create edge
            edge = LineageEdge(
                edge_id=edge_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                transformation_type=transformation_type,
                timestamp=datetime.utcnow(),
                transformation_code=transformation_code,
                transformation_parameters=transformation_parameters or {},
                quality_impact=quality_impact,
                success=success,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                validation_rules=validation_rules or [],
                validation_passed=validation_passed,
                validation_errors=validation_errors or []
            )
            
            # Store edge
            self.edges[edge_id] = edge
            self.edges_by_source[source_node_id].append(edge_id)
            self.edges_by_target[target_node_id].append(edge_id)
            
            # Update graph
            self.dependency_graph.add_edge(source_node_id, target_node_id, **{
                'edge_id': edge_id,
                'transformation_type': transformation_type.value,
                'timestamp': edge.timestamp,
                'quality_impact': quality_impact
            })
            
            # Update reverse graph
            self.reverse_dependency_graph.add_edge(target_node_id, source_node_id, **{
                'edge_id': edge_id,
                'transformation_type': transformation_type.value,
                'timestamp': edge.timestamp,
                'quality_impact': quality_impact
            })
            
            # Update statistics
            self.operation_stats['edges_created'] += 1
            
            logger.debug(f"Created lineage edge: {edge_id}")
            return edge_id
    
    def build_trace(self, node_id: str, direction: DataFlowDirection = DataFlowDirection.UPSTREAM) -> LineageTrace:
        """
        Build complete lineage trace for a node
        
        Args:
            node_id: Node ID to trace
            direction: Direction of trace
            
        Returns:
            LineageTrace: Complete lineage trace
        """
        start_time = time.time()
        
        with self.lock:
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found")
            
            trace_id = str(uuid.uuid4())
            trace = LineageTrace(
                trace_id=trace_id,
                root_node_id=node_id,
                target_node_id=node_id
            )
            
            # Choose appropriate graph
            graph = self.dependency_graph if direction == DataFlowDirection.DOWNSTREAM else self.reverse_dependency_graph
            
            # Build trace using BFS
            visited = set()
            queue = deque([node_id])
            
            while queue and len(trace.nodes) < self.config['max_trace_depth']:
                current_node_id = queue.popleft()
                
                if current_node_id in visited:
                    continue
                
                visited.add(current_node_id)
                
                # Add node to trace
                if current_node_id in self.nodes:
                    trace.nodes.append(self.nodes[current_node_id])
                
                # Add edges and connected nodes
                for neighbor in graph.neighbors(current_node_id):
                    if neighbor not in visited:
                        queue.append(neighbor)
                        
                        # Add edge to trace
                        edge_data = graph.get_edge_data(current_node_id, neighbor)
                        if edge_data and 'edge_id' in edge_data:
                            edge_id = edge_data['edge_id']
                            if edge_id in self.edges:
                                trace.edges.append(self.edges[edge_id])
            
            # Update trace metrics
            trace.update_metrics()
            
            # Store trace
            self.traces[trace_id] = trace
            
            # Update statistics
            execution_time_ms = (time.time() - start_time) * 1000
            self.operation_stats['traces_built'] += 1
            self.operation_stats['queries_executed'] += 1
            
            # Update average query time
            current_avg = self.operation_stats['avg_query_time_ms']
            query_count = self.operation_stats['queries_executed']
            self.operation_stats['avg_query_time_ms'] = (
                (current_avg * (query_count - 1) + execution_time_ms) / query_count
            )
            
            logger.debug(f"Built lineage trace: {trace_id} with {len(trace.nodes)} nodes")
            return trace
    
    def find_dependencies(self, node_id: str, max_depth: int = 10) -> List[str]:
        """
        Find all dependencies for a node
        
        Args:
            node_id: Node ID
            max_depth: Maximum depth to search
            
        Returns:
            List[str]: List of dependency node IDs
        """
        with self.lock:
            if node_id not in self.nodes:
                return []
            
            dependencies = set()
            queue = deque([(node_id, 0)])
            
            while queue:
                current_node_id, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                # Get upstream dependencies
                for predecessor in self.reverse_dependency_graph.predecessors(current_node_id):
                    if predecessor not in dependencies:
                        dependencies.add(predecessor)
                        queue.append((predecessor, depth + 1))
            
            return list(dependencies)
    
    def find_dependents(self, node_id: str, max_depth: int = 10) -> List[str]:
        """
        Find all dependents for a node
        
        Args:
            node_id: Node ID
            max_depth: Maximum depth to search
            
        Returns:
            List[str]: List of dependent node IDs
        """
        with self.lock:
            if node_id not in self.nodes:
                return []
            
            dependents = set()
            queue = deque([(node_id, 0)])
            
            while queue:
                current_node_id, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                # Get downstream dependents
                for successor in self.dependency_graph.successors(current_node_id):
                    if successor not in dependents:
                        dependents.add(successor)
                        queue.append((successor, depth + 1))
            
            return list(dependents)
    
    def analyze_impact(self, node_id: str) -> Dict[str, Any]:
        """
        Analyze potential impact of changes to a node
        
        Args:
            node_id: Node ID to analyze
            
        Returns:
            Dict[str, Any]: Impact analysis results
        """
        with self.lock:
            if node_id not in self.nodes:
                return {'error': f'Node {node_id} not found'}
            
            node = self.nodes[node_id]
            dependencies = self.find_dependencies(node_id)
            dependents = self.find_dependents(node_id)
            
            # Calculate quality impact
            quality_impacts = []
            for dependent_id in dependents:
                dependent_node = self.nodes.get(dependent_id)
                if dependent_node:
                    quality_impacts.append(dependent_node.quality_score)
            
            # Calculate processing time impact
            processing_times = []
            for edge_id in self.edges_by_source[node_id]:
                edge = self.edges[edge_id]
                processing_times.append(edge.execution_time_ms)
            
            return {
                'node_id': node_id,
                'node_type': node.data_type,
                'quality_score': node.quality_score,
                'dependencies_count': len(dependencies),
                'dependents_count': len(dependents),
                'dependencies': dependencies,
                'dependents': dependents,
                'avg_dependent_quality': sum(quality_impacts) / len(quality_impacts) if quality_impacts else 0,
                'total_processing_time_ms': sum(processing_times),
                'risk_level': self._calculate_risk_level(len(dependencies), len(dependents), node.quality_score)
            }
    
    def _calculate_risk_level(self, dependencies_count: int, dependents_count: int, quality_score: float) -> str:
        """Calculate risk level based on dependencies and quality"""
        risk_score = (dependencies_count * 0.3 + dependents_count * 0.5 + (1 - quality_score) * 0.2)
        
        if risk_score < 2:
            return "LOW"
        elif risk_score < 5:
            return "MEDIUM"
        elif risk_score < 10:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def get_quality_degradation_path(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get quality degradation path for a node
        
        Args:
            node_id: Node ID
            
        Returns:
            List[Dict[str, Any]]: Quality degradation path
        """
        trace = self.build_trace(node_id, DataFlowDirection.UPSTREAM)
        
        # Build quality degradation path
        path = []
        for i, node in enumerate(trace.nodes):
            path_entry = {
                'node_id': node.node_id,
                'data_type': node.data_type,
                'quality_score': node.quality_score,
                'timestamp': node.timestamp
            }
            
            # Add transformation info if not first node
            if i > 0:
                edge = trace.edges[i-1]
                path_entry['transformation'] = {
                    'type': edge.transformation_type.value,
                    'quality_impact': edge.quality_impact,
                    'execution_time_ms': edge.execution_time_ms,
                    'success': edge.success
                }
            
            path.append(path_entry)
        
        return path
    
    def _background_processor(self):
        """Background processing thread"""
        while self.running:
            try:
                # Process cleanup tasks
                self._cleanup_old_records()
                
                # Process integrity checks
                if self.config['enable_integrity_checks']:
                    self._verify_integrity()
                
                # Sleep before next iteration
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in background processor: {e}")
                time.sleep(10)
    
    def _cleanup_old_records(self):
        """Clean up old lineage records"""
        cutoff_time = datetime.utcnow() - timedelta(days=self.config['retention_days'])
        
        with self.lock:
            # Find old nodes
            old_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.timestamp < cutoff_time
            ]
            
            # Remove old nodes and associated edges
            for node_id in old_nodes:
                self._remove_node(node_id)
            
            if old_nodes:
                logger.info(f"Cleaned up {len(old_nodes)} old lineage records")
    
    def _remove_node(self, node_id: str):
        """Remove a node and all associated edges"""
        if node_id not in self.nodes:
            return
        
        # Remove edges
        edges_to_remove = []
        edges_to_remove.extend(self.edges_by_source[node_id])
        edges_to_remove.extend(self.edges_by_target[node_id])
        
        for edge_id in edges_to_remove:
            if edge_id in self.edges:
                del self.edges[edge_id]
        
        # Remove from indices
        del self.edges_by_source[node_id]
        del self.edges_by_target[node_id]
        
        # Remove from hash index
        node = self.nodes[node_id]
        if node.data_hash in self.node_by_data_hash:
            del self.node_by_data_hash[node.data_hash]
        
        # Remove from graphs
        if self.dependency_graph.has_node(node_id):
            self.dependency_graph.remove_node(node_id)
        if self.reverse_dependency_graph.has_node(node_id):
            self.reverse_dependency_graph.remove_node(node_id)
        
        # Remove node
        del self.nodes[node_id]
    
    def _verify_integrity(self):
        """Verify lineage integrity"""
        issues = []
        
        with self.lock:
            # Check for orphaned edges
            for edge_id, edge in self.edges.items():
                if edge.source_node_id not in self.nodes:
                    issues.append(f"Edge {edge_id} references missing source node {edge.source_node_id}")
                if edge.target_node_id not in self.nodes:
                    issues.append(f"Edge {edge_id} references missing target node {edge.target_node_id}")
            
            # Check graph consistency
            graph_nodes = set(self.dependency_graph.nodes())
            stored_nodes = set(self.nodes.keys())
            
            if graph_nodes != stored_nodes:
                issues.append("Graph nodes inconsistent with stored nodes")
        
        if issues:
            logger.warning(f"Lineage integrity issues found: {issues}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lineage tracking statistics"""
        with self.lock:
            return {
                'nodes_count': len(self.nodes),
                'edges_count': len(self.edges),
                'traces_count': len(self.traces),
                'operation_stats': self.operation_stats.copy(),
                'graph_stats': {
                    'nodes': self.dependency_graph.number_of_nodes(),
                    'edges': self.dependency_graph.number_of_edges(),
                    'is_dag': nx.is_directed_acyclic_graph(self.dependency_graph)
                }
            }
    
    def export_lineage(self, format: str = 'json') -> str:
        """Export lineage data"""
        with self.lock:
            if format == 'json':
                export_data = {
                    'nodes': [
                        {
                            'node_id': node.node_id,
                            'data_type': node.data_type,
                            'timestamp': node.timestamp.isoformat(),
                            'quality_score': node.quality_score,
                            'source_system': node.source_system,
                            'component': node.component
                        }
                        for node in self.nodes.values()
                    ],
                    'edges': [
                        {
                            'edge_id': edge.edge_id,
                            'source_node_id': edge.source_node_id,
                            'target_node_id': edge.target_node_id,
                            'transformation_type': edge.transformation_type.value,
                            'timestamp': edge.timestamp.isoformat(),
                            'quality_impact': edge.quality_impact,
                            'execution_time_ms': edge.execution_time_ms
                        }
                        for edge in self.edges.values()
                    ]
                }
                return json.dumps(export_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")


# =============================================================================
# LINEAGE QUERY ENGINE
# =============================================================================

class LineageQueryEngine:
    """Advanced lineage query engine"""
    
    def __init__(self, tracker: LineageTracker):
        self.tracker = tracker
        
    def find_root_causes(self, node_id: str, quality_threshold: float = 0.8) -> List[str]:
        """Find root causes of quality issues"""
        trace = self.tracker.build_trace(node_id, DataFlowDirection.UPSTREAM)
        
        root_causes = []
        for node in trace.nodes:
            if node.quality_score < quality_threshold:
                # Check if this is a root cause (no upstream dependencies with similar issues)
                dependencies = self.tracker.find_dependencies(node.node_id, max_depth=1)
                
                is_root_cause = True
                for dep_id in dependencies:
                    dep_node = self.tracker.nodes.get(dep_id)
                    if dep_node and dep_node.quality_score < quality_threshold:
                        is_root_cause = False
                        break
                
                if is_root_cause:
                    root_causes.append(node.node_id)
        
        return root_causes
    
    def find_transformation_bottlenecks(self, node_id: str, time_threshold_ms: float = 1000) -> List[str]:
        """Find transformation bottlenecks in lineage"""
        trace = self.tracker.build_trace(node_id, DataFlowDirection.UPSTREAM)
        
        bottlenecks = []
        for edge in trace.edges:
            if edge.execution_time_ms > time_threshold_ms:
                bottlenecks.append(edge.edge_id)
        
        return bottlenecks
    
    def calculate_lineage_metrics(self, node_id: str) -> Dict[str, Any]:
        """Calculate comprehensive lineage metrics"""
        trace = self.tracker.build_trace(node_id, DataFlowDirection.UPSTREAM)
        
        if not trace.nodes:
            return {}
        
        # Quality metrics
        quality_scores = [node.quality_score for node in trace.nodes]
        
        # Performance metrics
        execution_times = [edge.execution_time_ms for edge in trace.edges]
        
        # Transformation metrics
        transformation_types = [edge.transformation_type.value for edge in trace.edges]
        transformation_counts = {t: transformation_types.count(t) for t in set(transformation_types)}
        
        return {
            'node_count': len(trace.nodes),
            'edge_count': len(trace.edges),
            'avg_quality_score': sum(quality_scores) / len(quality_scores),
            'min_quality_score': min(quality_scores),
            'max_quality_score': max(quality_scores),
            'total_execution_time_ms': sum(execution_times),
            'avg_execution_time_ms': sum(execution_times) / len(execution_times) if execution_times else 0,
            'transformation_counts': transformation_counts,
            'quality_degradation': trace.quality_degradation,
            'critical_path_length': len(trace.critical_path)
        }


# =============================================================================
# GLOBAL LINEAGE MANAGER
# =============================================================================

class GlobalLineageManager:
    """Global lineage manager singleton"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.tracker = LineageTracker()
            self.query_engine = LineageQueryEngine(self.tracker)
            self.initialized = True
            
            # Start background processing
            self.tracker.start()
    
    def __del__(self):
        if hasattr(self, 'tracker'):
            self.tracker.stop()


# Global instance
lineage_manager = GlobalLineageManager()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def track_transformation(source_data_hash: str,
                        target_data_hash: str,
                        transformation_type: TransformationType,
                        component: str,
                        **kwargs) -> str:
    """
    Convenient function to track a data transformation
    
    Args:
        source_data_hash: Hash of source data
        target_data_hash: Hash of target data
        transformation_type: Type of transformation
        component: Component performing transformation
        **kwargs: Additional parameters
        
    Returns:
        str: Edge ID
    """
    # Find or create nodes
    source_node_id = lineage_manager.tracker.node_by_data_hash.get(source_data_hash)
    target_node_id = lineage_manager.tracker.node_by_data_hash.get(target_data_hash)
    
    if not source_node_id or not target_node_id:
        raise ValueError("Source or target node not found")
    
    return lineage_manager.tracker.create_edge(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        transformation_type=transformation_type,
        **kwargs
    )

def get_data_lineage(data_hash: str) -> Optional[LineageTrace]:
    """
    Get complete lineage for data
    
    Args:
        data_hash: Hash of data
        
    Returns:
        Optional[LineageTrace]: Lineage trace or None if not found
    """
    node_id = lineage_manager.tracker.node_by_data_hash.get(data_hash)
    if not node_id:
        return None
    
    return lineage_manager.tracker.build_trace(node_id)

def analyze_data_quality_impact(data_hash: str) -> Dict[str, Any]:
    """
    Analyze quality impact for data
    
    Args:
        data_hash: Hash of data
        
    Returns:
        Dict[str, Any]: Impact analysis
    """
    node_id = lineage_manager.tracker.node_by_data_hash.get(data_hash)
    if not node_id:
        return {'error': 'Data not found'}
    
    return lineage_manager.tracker.analyze_impact(node_id)

# Export key components
__all__ = [
    'LineageEventType',
    'TransformationType',
    'LineageNode',
    'LineageEdge',
    'LineageTrace',
    'LineageTracker',
    'LineageQueryEngine',
    'GlobalLineageManager',
    'lineage_manager',
    'track_transformation',
    'get_data_lineage',
    'analyze_data_quality_impact'
]