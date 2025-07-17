"""
Data lineage tracking and validation system

This module implements comprehensive data lineage tracking to monitor data flow,
transformations, and dependencies throughout the data pipeline.
"""

import time
import threading
import uuid
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
from pathlib import Path
import sqlite3
from datetime import datetime
import networkx as nx
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LineageEventType(Enum):
    """Types of lineage events"""
    DATA_INGESTION = "data_ingestion"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_VALIDATION = "data_validation"
    DATA_AGGREGATION = "data_aggregation"
    DATA_STORAGE = "data_storage"
    DATA_RETRIEVAL = "data_retrieval"
    DATA_DELETION = "data_deletion"
    INDICATOR_CALCULATION = "indicator_calculation"
    CACHE_OPERATION = "cache_operation"
    QUALITY_CHECK = "quality_check"

class DataOperation(Enum):
    """Types of data operations"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    VALIDATE = "validate"
    CACHE = "cache"

@dataclass
class DataAsset:
    """Represents a data asset in the lineage graph"""
    asset_id: str
    name: str
    asset_type: str  # 'dataset', 'table', 'file', 'stream', 'indicator'
    schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if isinstance(self.tags, list):
            self.tags = set(self.tags)

@dataclass
class DataTransformation:
    """Represents a data transformation operation"""
    transformation_id: str
    name: str
    transformation_type: str  # 'filter', 'aggregate', 'join', 'calculate', 'validate'
    input_assets: List[str]
    output_assets: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    code: Optional[str] = None
    executed_by: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LineageEvent:
    """Represents a lineage event"""
    event_id: str
    event_type: LineageEventType
    operation: DataOperation
    asset_id: str
    transformation_id: Optional[str] = None
    source_assets: List[str] = field(default_factory=list)
    target_assets: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class ValidationRule:
    """Represents a data validation rule"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # 'schema', 'quality', 'business', 'referential'
    target_assets: List[str]
    conditions: Dict[str, Any]
    severity: str  # 'error', 'warning', 'info'
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of validation rule execution"""
    validation_id: str
    rule_id: str
    asset_id: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class DataLineageTracker:
    """Comprehensive data lineage tracking system"""
    
    def __init__(self, 
                 enable_persistence: bool = True,
                 persistence_path: Optional[str] = None,
                 max_events: int = 100000,
                 enable_graph_analysis: bool = True):
        
        self.enable_persistence = enable_persistence
        self.persistence_path = Path(persistence_path) if persistence_path else Path("/tmp/lineage_tracker")
        self.max_events = max_events
        self.enable_graph_analysis = enable_graph_analysis
        
        # Create storage directory
        if enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self.db_path = self.persistence_path / "lineage.db"
            self._init_database()
        
        # In-memory storage
        self.assets = {}  # asset_id -> DataAsset
        self.transformations = {}  # transformation_id -> DataTransformation
        self.events = deque(maxlen=max_events)  # LineageEvent queue
        self.validation_rules = {}  # rule_id -> ValidationRule
        self.validation_results = deque(maxlen=max_events)  # ValidationResult queue
        
        # Thread safety
        self.assets_lock = threading.RLock()
        self.transformations_lock = threading.RLock()
        self.events_lock = threading.RLock()
        self.rules_lock = threading.RLock()
        self.results_lock = threading.RLock()
        
        # Lineage graph for analysis
        if enable_graph_analysis:
            self.lineage_graph = nx.DiGraph()
            self.graph_lock = threading.RLock()
        
        # Active sessions tracking
        self.active_sessions = {}  # session_id -> session_info
        self.session_lock = threading.RLock()
        
        # Performance metrics
        self.total_events = 0
        self.total_validations = 0
        self.event_processing_times = deque(maxlen=1000)
        
        # Background executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup cleanup
        self._setup_cleanup()
    
    def _setup_cleanup(self):
        """Setup cleanup on exit"""
        import atexit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Cleanup resources"""
        if self.enable_persistence:
            self._persist_all_data()
        
        self.executor.shutdown(wait=True)
        logger.info("DataLineageTracker cleanup completed")
    
    def _init_database(self):
        """Initialize database for persistence"""
        with sqlite3.connect(self.db_path) as conn:
            # Assets table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    schema TEXT,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    tags TEXT
                )
            ''')
            
            # Transformations table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transformations (
                    transformation_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    transformation_type TEXT NOT NULL,
                    input_assets TEXT NOT NULL,
                    output_assets TEXT NOT NULL,
                    parameters TEXT,
                    code TEXT,
                    executed_by TEXT,
                    execution_time REAL,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS lineage_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    transformation_id TEXT,
                    source_assets TEXT,
                    target_assets TEXT,
                    timestamp REAL NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT,
                    success BOOLEAN,
                    error_message TEXT
                )
            ''')
            
            # Validation rules table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS validation_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    rule_type TEXT NOT NULL,
                    target_assets TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    enabled BOOLEAN,
                    created_at REAL NOT NULL,
                    metadata TEXT
                )
            ''')
            
            # Validation results table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS validation_results (
                    validation_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    message TEXT,
                    details TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (rule_id) REFERENCES validation_rules (rule_id)
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON lineage_events (timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_events_asset_id ON lineage_events (asset_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_validation_results_timestamp ON validation_results (timestamp)')
            
            conn.commit()
    
    def register_asset(self, asset: DataAsset) -> str:
        """Register a new data asset"""
        with self.assets_lock:
            self.assets[asset.asset_id] = asset
        
        # Update graph
        if self.enable_graph_analysis:
            with self.graph_lock:
                self.lineage_graph.add_node(asset.asset_id, 
                                           name=asset.name,
                                           asset_type=asset.asset_type,
                                           created_at=asset.created_at)
        
        # Log event
        self._log_event(LineageEvent(
            event_id=str(uuid.uuid4()),
            event_type=LineageEventType.DATA_INGESTION,
            operation=DataOperation.CREATE,
            asset_id=asset.asset_id,
            metadata={'asset_type': asset.asset_type, 'name': asset.name}
        ))
        
        return asset.asset_id
    
    def register_transformation(self, transformation: DataTransformation) -> str:
        """Register a data transformation"""
        with self.transformations_lock:
            self.transformations[transformation.transformation_id] = transformation
        
        # Update graph relationships
        if self.enable_graph_analysis:
            with self.graph_lock:
                # Add transformation as node
                self.lineage_graph.add_node(transformation.transformation_id,
                                           name=transformation.name,
                                           node_type='transformation',
                                           transformation_type=transformation.transformation_type)
                
                # Add edges from inputs to transformation
                for input_asset in transformation.input_assets:
                    self.lineage_graph.add_edge(input_asset, transformation.transformation_id)
                
                # Add edges from transformation to outputs
                for output_asset in transformation.output_assets:
                    self.lineage_graph.add_edge(transformation.transformation_id, output_asset)
        
        # Log event
        self._log_event(LineageEvent(
            event_id=str(uuid.uuid4()),
            event_type=LineageEventType.DATA_TRANSFORMATION,
            operation=DataOperation.TRANSFORM,
            asset_id=transformation.transformation_id,
            transformation_id=transformation.transformation_id,
            source_assets=transformation.input_assets,
            target_assets=transformation.output_assets,
            metadata={
                'transformation_type': transformation.transformation_type,
                'execution_time': transformation.execution_time
            }
        ))
        
        return transformation.transformation_id
    
    def track_data_flow(self, 
                       source_asset_id: str,
                       target_asset_id: str,
                       transformation_id: Optional[str] = None,
                       operation: DataOperation = DataOperation.TRANSFORM,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track data flow between assets"""
        
        event_id = str(uuid.uuid4())
        
        # Update graph
        if self.enable_graph_analysis:
            with self.graph_lock:
                if transformation_id:
                    # Flow through transformation
                    self.lineage_graph.add_edge(source_asset_id, transformation_id)
                    self.lineage_graph.add_edge(transformation_id, target_asset_id)
                else:
                    # Direct flow
                    self.lineage_graph.add_edge(source_asset_id, target_asset_id)
        
        # Log event
        self._log_event(LineageEvent(
            event_id=event_id,
            event_type=LineageEventType.DATA_TRANSFORMATION,
            operation=operation,
            asset_id=target_asset_id,
            transformation_id=transformation_id,
            source_assets=[source_asset_id],
            target_assets=[target_asset_id],
            metadata=metadata or {}
        ))
        
        return event_id
    
    def _log_event(self, event: LineageEvent):
        """Log a lineage event"""
        start_time = time.time_ns()
        
        with self.events_lock:
            self.events.append(event)
            self.total_events += 1
        
        # Persist to database if enabled
        if self.enable_persistence:
            self.executor.submit(self._persist_event, event)
        
        # Record processing time
        end_time = time.time_ns()
        processing_time_us = (end_time - start_time) / 1000
        self.event_processing_times.append(processing_time_us)
    
    def _persist_event(self, event: LineageEvent):
        """Persist event to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'INSERT INTO lineage_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (event.event_id, event.event_type.value, event.operation.value,
                     event.asset_id, event.transformation_id,
                     json.dumps(event.source_assets), json.dumps(event.target_assets),
                     event.timestamp, event.user_id, event.session_id,
                     json.dumps(event.metadata), event.success, event.error_message)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error persisting event: {str(e)}")
    
    def _persist_all_data(self):
        """Persist all in-memory data to database"""
        if not self.enable_persistence:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute('DELETE FROM assets')
                conn.execute('DELETE FROM transformations')
                
                # Persist assets
                with self.assets_lock:
                    for asset in self.assets.values():
                        conn.execute(
                            'INSERT OR REPLACE INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                            (asset.asset_id, asset.name, asset.asset_type,
                             json.dumps(asset.schema), json.dumps(asset.metadata),
                             asset.created_at, asset.updated_at, json.dumps(list(asset.tags)))
                        )
                
                # Persist transformations
                with self.transformations_lock:
                    for transformation in self.transformations.values():
                        conn.execute(
                            'INSERT OR REPLACE INTO transformations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                            (transformation.transformation_id, transformation.name,
                             transformation.transformation_type,
                             json.dumps(transformation.input_assets),
                             json.dumps(transformation.output_assets),
                             json.dumps(transformation.parameters),
                             transformation.code, transformation.executed_by,
                             transformation.execution_time, transformation.timestamp,
                             json.dumps(transformation.metadata))
                        )
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error persisting data: {str(e)}")
    
    def add_validation_rule(self, rule: ValidationRule) -> str:
        """Add a data validation rule"""
        with self.rules_lock:
            self.validation_rules[rule.rule_id] = rule
        
        # Persist to database
        if self.enable_persistence:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'INSERT OR REPLACE INTO validation_rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (rule.rule_id, rule.name, rule.description, rule.rule_type,
                         json.dumps(rule.target_assets), json.dumps(rule.conditions),
                         rule.severity, rule.enabled, rule.created_at,
                         json.dumps(rule.metadata))
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Error persisting validation rule: {str(e)}")
        
        return rule.rule_id
    
    def validate_asset(self, asset_id: str, rule_id: Optional[str] = None) -> List[ValidationResult]:
        """Validate an asset against rules"""
        results = []
        
        with self.rules_lock:
            rules_to_check = []
            
            if rule_id:
                if rule_id in self.validation_rules:
                    rules_to_check.append(self.validation_rules[rule_id])
            else:
                # Check all applicable rules
                for rule in self.validation_rules.values():
                    if rule.enabled and (not rule.target_assets or asset_id in rule.target_assets):
                        rules_to_check.append(rule)
        
        # Execute validation rules
        for rule in rules_to_check:
            try:
                result = self._execute_validation_rule(asset_id, rule)
                results.append(result)
                
                # Store result
                with self.results_lock:
                    self.validation_results.append(result)
                    self.total_validations += 1
                
                # Persist result
                if self.enable_persistence:
                    self.executor.submit(self._persist_validation_result, result)
            
            except Exception as e:
                logger.error(f"Error executing validation rule {rule.rule_id}: {str(e)}")
        
        return results
    
    def _execute_validation_rule(self, asset_id: str, rule: ValidationRule) -> ValidationResult:
        """Execute a validation rule against an asset"""
        # Get asset
        with self.assets_lock:
            asset = self.assets.get(asset_id)
        
        if not asset:
            return ValidationResult(
                validation_id=str(uuid.uuid4()),
                rule_id=rule.rule_id,
                asset_id=asset_id,
                passed=False,
                message=f"Asset {asset_id} not found"
            )
        
        # Simple rule execution (can be extended for complex rules)
        passed = True
        message = "Validation passed"
        details = {}
        
        try:
            # Schema validation
            if rule.rule_type == 'schema':
                if 'required_fields' in rule.conditions:
                    required_fields = rule.conditions['required_fields']
                    if asset.schema:
                        asset_fields = set(asset.schema.keys())
                        missing_fields = set(required_fields) - asset_fields
                        if missing_fields:
                            passed = False
                            message = f"Missing required fields: {list(missing_fields)}"
                            details['missing_fields'] = list(missing_fields)
            
            # Quality validation
            elif rule.rule_type == 'quality':
                if 'min_records' in rule.conditions:
                    min_records = rule.conditions['min_records']
                    record_count = asset.metadata.get('record_count', 0)
                    if record_count < min_records:
                        passed = False
                        message = f"Record count {record_count} below minimum {min_records}"
                        details['record_count'] = record_count
                        details['min_records'] = min_records
            
            # Business rule validation
            elif rule.rule_type == 'business':
                # Custom business logic can be implemented here
                pass
            
            # Referential integrity validation
            elif rule.rule_type == 'referential':
                if 'referenced_assets' in rule.conditions:
                    referenced_assets = rule.conditions['referenced_assets']
                    with self.assets_lock:
                        for ref_asset_id in referenced_assets:
                            if ref_asset_id not in self.assets:
                                passed = False
                                message = f"Referenced asset {ref_asset_id} not found"
                                details['missing_reference'] = ref_asset_id
                                break
        
        except Exception as e:
            passed = False
            message = f"Validation error: {str(e)}"
            details['error'] = str(e)
        
        return ValidationResult(
            validation_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            asset_id=asset_id,
            passed=passed,
            message=message,
            details=details
        )
    
    def _persist_validation_result(self, result: ValidationResult):
        """Persist validation result to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'INSERT INTO validation_results VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (result.validation_id, result.rule_id, result.asset_id,
                     result.passed, result.message, json.dumps(result.details),
                     result.timestamp)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error persisting validation result: {str(e)}")
    
    def get_asset_lineage(self, asset_id: str, direction: str = 'both') -> Dict[str, Any]:
        """Get lineage information for an asset"""
        if not self.enable_graph_analysis:
            return {'error': 'Graph analysis not enabled'}
        
        with self.graph_lock:
            if asset_id not in self.lineage_graph:
                return {'error': f'Asset {asset_id} not found in lineage graph'}
            
            lineage = {
                'asset_id': asset_id,
                'upstream': [],
                'downstream': [],
                'transformations': []
            }
            
            # Get upstream dependencies
            if direction in ['upstream', 'both']:
                predecessors = list(self.lineage_graph.predecessors(asset_id))
                for pred in predecessors:
                    node_data = self.lineage_graph.nodes[pred]
                    lineage['upstream'].append({
                        'id': pred,
                        'name': node_data.get('name', pred),
                        'type': node_data.get('asset_type', 'unknown')
                    })
            
            # Get downstream dependencies
            if direction in ['downstream', 'both']:
                successors = list(self.lineage_graph.successors(asset_id))
                for succ in successors:
                    node_data = self.lineage_graph.nodes[succ]
                    lineage['downstream'].append({
                        'id': succ,
                        'name': node_data.get('name', succ),
                        'type': node_data.get('asset_type', 'unknown')
                    })
            
            # Get related transformations
            for node_id in self.lineage_graph.nodes():
                node_data = self.lineage_graph.nodes[node_id]
                if node_data.get('node_type') == 'transformation':
                    # Check if this transformation is connected to our asset
                    if (self.lineage_graph.has_edge(asset_id, node_id) or 
                        self.lineage_graph.has_edge(node_id, asset_id)):
                        lineage['transformations'].append({
                            'id': node_id,
                            'name': node_data.get('name', node_id),
                            'type': node_data.get('transformation_type', 'unknown')
                        })
            
            return lineage
    
    def get_impact_analysis(self, asset_id: str) -> Dict[str, Any]:
        """Get impact analysis for an asset"""
        if not self.enable_graph_analysis:
            return {'error': 'Graph analysis not enabled'}
        
        with self.graph_lock:
            if asset_id not in self.lineage_graph:
                return {'error': f'Asset {asset_id} not found in lineage graph'}
            
            # Get all downstream nodes
            downstream_nodes = nx.descendants(self.lineage_graph, asset_id)
            
            impact = {
                'asset_id': asset_id,
                'total_impacted_assets': len(downstream_nodes),
                'impacted_assets': [],
                'impact_levels': defaultdict(list)
            }
            
            # Calculate impact levels (distance from source)
            for node in downstream_nodes:
                try:
                    distance = nx.shortest_path_length(self.lineage_graph, asset_id, node)
                    node_data = self.lineage_graph.nodes[node]
                    
                    asset_info = {
                        'id': node,
                        'name': node_data.get('name', node),
                        'type': node_data.get('asset_type', 'unknown'),
                        'distance': distance
                    }
                    
                    impact['impacted_assets'].append(asset_info)
                    impact['impact_levels'][distance].append(asset_info)
                
                except nx.NetworkXNoPath:
                    continue
            
            # Convert defaultdict to regular dict
            impact['impact_levels'] = dict(impact['impact_levels'])
            
            return impact
    
    def get_data_quality_summary(self) -> Dict[str, Any]:
        """Get data quality summary based on validation results"""
        with self.results_lock:
            recent_results = list(self.validation_results)[-1000:]  # Last 1000 results
        
        if not recent_results:
            return {'message': 'No validation results available'}
        
        # Calculate statistics
        total_validations = len(recent_results)
        passed_validations = sum(1 for r in recent_results if r.passed)
        failed_validations = total_validations - passed_validations
        
        # Group by asset
        asset_quality = defaultdict(lambda: {'passed': 0, 'failed': 0})
        for result in recent_results:
            if result.passed:
                asset_quality[result.asset_id]['passed'] += 1
            else:
                asset_quality[result.asset_id]['failed'] += 1
        
        # Group by rule
        rule_performance = defaultdict(lambda: {'passed': 0, 'failed': 0})
        for result in recent_results:
            if result.passed:
                rule_performance[result.rule_id]['passed'] += 1
            else:
                rule_performance[result.rule_id]['failed'] += 1
        
        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': failed_validations,
            'success_rate': (passed_validations / total_validations) * 100 if total_validations > 0 else 0,
            'asset_quality': dict(asset_quality),
            'rule_performance': dict(rule_performance)
        }
    
    def get_lineage_summary(self) -> Dict[str, Any]:
        """Get lineage tracking summary"""
        with self.assets_lock:
            total_assets = len(self.assets)
            asset_types = defaultdict(int)
            for asset in self.assets.values():
                asset_types[asset.asset_type] += 1
        
        with self.transformations_lock:
            total_transformations = len(self.transformations)
            transformation_types = defaultdict(int)
            for transformation in self.transformations.values():
                transformation_types[transformation.transformation_type] += 1
        
        with self.events_lock:
            recent_events = list(self.events)[-100:]  # Last 100 events
        
        # Performance metrics
        avg_processing_time = (sum(self.event_processing_times) / len(self.event_processing_times) 
                              if self.event_processing_times else 0)
        
        return {
            'total_assets': total_assets,
            'asset_types': dict(asset_types),
            'total_transformations': total_transformations,
            'transformation_types': dict(transformation_types),
            'total_events': self.total_events,
            'total_validations': self.total_validations,
            'recent_events_count': len(recent_events),
            'avg_event_processing_time_us': avg_processing_time,
            'graph_nodes': len(self.lineage_graph.nodes()) if self.enable_graph_analysis else 0,
            'graph_edges': len(self.lineage_graph.edges()) if self.enable_graph_analysis else 0
        }
    
    def export_lineage_graph(self, format: str = 'json') -> str:
        """Export lineage graph in specified format"""
        if not self.enable_graph_analysis:
            return '{"error": "Graph analysis not enabled"}'
        
        with self.graph_lock:
            if format == 'json':
                # Convert graph to JSON format
                graph_data = {
                    'nodes': [],
                    'edges': []
                }
                
                for node_id, node_data in self.lineage_graph.nodes(data=True):
                    graph_data['nodes'].append({
                        'id': node_id,
                        'data': node_data
                    })
                
                for source, target, edge_data in self.lineage_graph.edges(data=True):
                    graph_data['edges'].append({
                        'source': source,
                        'target': target,
                        'data': edge_data
                    })
                
                return json.dumps(graph_data, indent=2)
            
            elif format == 'dot':
                # Convert to DOT format for visualization
                try:
                    import pydot
                    return nx.nx_pydot.to_pydot(self.lineage_graph).to_string()
                except ImportError:
                    return '{"error": "pydot not installed for DOT export"}'
            
            else:
                return f'{{"error": "Unsupported format: {format}"}}'

# Utility functions
def create_lineage_tracker(enable_persistence: bool = True) -> DataLineageTracker:
    """Create lineage tracker with default settings"""
    return DataLineageTracker(enable_persistence=enable_persistence)

def create_data_asset(name: str, asset_type: str, **kwargs) -> DataAsset:
    """Create a data asset with unique ID"""
    return DataAsset(
        asset_id=str(uuid.uuid4()),
        name=name,
        asset_type=asset_type,
        **kwargs
    )

def create_transformation(name: str, transformation_type: str, 
                        input_assets: List[str], output_assets: List[str], 
                        **kwargs) -> DataTransformation:
    """Create a data transformation with unique ID"""
    return DataTransformation(
        transformation_id=str(uuid.uuid4()),
        name=name,
        transformation_type=transformation_type,
        input_assets=input_assets,
        output_assets=output_assets,
        **kwargs
    )

def create_validation_rule(name: str, rule_type: str, 
                         target_assets: List[str], conditions: Dict[str, Any], 
                         severity: str = 'error', **kwargs) -> ValidationRule:
    """Create a validation rule with unique ID"""
    return ValidationRule(
        rule_id=str(uuid.uuid4()),
        name=name,
        description=kwargs.get('description', ''),
        rule_type=rule_type,
        target_assets=target_assets,
        conditions=conditions,
        severity=severity,
        **kwargs
    )
