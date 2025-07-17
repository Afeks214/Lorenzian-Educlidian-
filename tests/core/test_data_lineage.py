"""
Comprehensive test suite for data lineage tracking system.
Tests all functionality including lineage tracking, dependency graphs, 
audit trails, and integration tests for lineage event processing.
"""

import pytest
import asyncio
import threading
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.data_lineage import (
    LineageEventType,
    TransformationType,
    LineageNode,
    LineageEdge,
    LineageTrace,
    LineageTracker,
    LineageQueryEngine,
    GlobalLineageManager,
    DataFlowDirection,
    LineageStatus,
    lineage_manager,
    track_transformation,
    get_data_lineage,
    analyze_data_quality_impact
)


class TestLineageDataStructures:
    """Test data structures for lineage tracking."""
    
    def test_lineage_node_creation(self):
        """Test LineageNode creation and validation."""
        node = LineageNode(
            node_id="test-node-1",
            data_type="market_data",
            schema_version="1.0",
            timestamp=datetime.utcnow(),
            data_hash="abc123",
            data_signature="signature123",
            source_system="market_feed",
            component="data_ingestion",
            quality_score=0.95
        )
        
        assert node.node_id == "test-node-1"
        assert node.data_type == "market_data"
        assert node.quality_score == 0.95
        assert node.business_context == {}
        assert node.technical_context == {}
    
    def test_lineage_node_quality_score_validation(self):
        """Test quality score validation."""
        # Test invalid quality score > 1
        with pytest.raises(ValueError, match="Quality score must be between 0 and 1"):
            LineageNode(
                node_id="test-node-1",
                data_type="market_data",
                schema_version="1.0",
                timestamp=datetime.utcnow(),
                data_hash="abc123",
                data_signature="signature123",
                source_system="market_feed",
                component="data_ingestion",
                quality_score=1.5
            )
        
        # Test invalid quality score < 0
        with pytest.raises(ValueError, match="Quality score must be between 0 and 1"):
            LineageNode(
                node_id="test-node-1",
                data_type="market_data",
                schema_version="1.0",
                timestamp=datetime.utcnow(),
                data_hash="abc123",
                data_signature="signature123",
                source_system="market_feed",
                component="data_ingestion",
                quality_score=-0.1
            )
    
    def test_lineage_node_empty_id_validation(self):
        """Test empty node ID validation."""
        with pytest.raises(ValueError, match="Node ID cannot be empty"):
            LineageNode(
                node_id="",
                data_type="market_data",
                schema_version="1.0",
                timestamp=datetime.utcnow(),
                data_hash="abc123",
                data_signature="signature123",
                source_system="market_feed",
                component="data_ingestion",
                quality_score=0.95
            )
    
    def test_lineage_edge_creation(self):
        """Test LineageEdge creation."""
        edge = LineageEdge(
            edge_id="edge-1",
            source_node_id="node-1",
            target_node_id="node-2",
            transformation_type=TransformationType.NORMALIZATION,
            timestamp=datetime.utcnow(),
            quality_impact=0.05,
            execution_time_ms=150.0
        )
        
        assert edge.edge_id == "edge-1"
        assert edge.source_node_id == "node-1"
        assert edge.target_node_id == "node-2"
        assert edge.transformation_type == TransformationType.NORMALIZATION
        assert edge.quality_impact == 0.05
        assert edge.execution_time_ms == 150.0
        assert edge.success is True
    
    def test_lineage_trace_update_metrics(self):
        """Test LineageTrace metrics calculation."""
        trace = LineageTrace(
            trace_id="trace-1",
            root_node_id="root-1",
            target_node_id="target-1"
        )
        
        # Add test nodes
        node1 = LineageNode(
            node_id="node-1",
            data_type="raw_data",
            schema_version="1.0",
            timestamp=datetime.utcnow(),
            data_hash="hash1",
            data_signature="sig1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2 = LineageNode(
            node_id="node-2",
            data_type="processed_data",
            schema_version="1.0",
            timestamp=datetime.utcnow(),
            data_hash="hash2",
            data_signature="sig2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        trace.nodes = [node1, node2]
        
        # Add test edge
        edge = LineageEdge(
            edge_id="edge-1",
            source_node_id="node-1",
            target_node_id="node-2",
            transformation_type=TransformationType.NORMALIZATION,
            timestamp=datetime.utcnow(),
            execution_time_ms=100.0
        )
        
        trace.edges = [edge]
        
        # Update metrics
        trace.update_metrics()
        
        assert trace.transformation_count == 1
        assert trace.total_processing_time_ms == 100.0
        assert trace.overall_quality_score == 0.85  # (0.9 + 0.8) / 2
        assert trace.quality_degradation == 0.1  # 0.9 - 0.8


class TestLineageTracker:
    """Test the LineageTracker class."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.tracker = LineageTracker()
    
    def teardown_method(self):
        """Teardown method for each test."""
        if hasattr(self.tracker, 'stop'):
            self.tracker.stop()
    
    def test_tracker_initialization(self):
        """Test LineageTracker initialization."""
        assert self.tracker.nodes == {}
        assert self.tracker.edges == {}
        assert self.tracker.traces == {}
        assert self.tracker.operation_stats['nodes_created'] == 0
        assert self.tracker.operation_stats['edges_created'] == 0
    
    def test_create_node(self):
        """Test node creation."""
        node_id = self.tracker.create_node(
            data_type="market_data",
            data_hash="hash123",
            source_system="bloomberg",
            component="data_feed",
            quality_score=0.95
        )
        
        assert node_id in self.tracker.nodes
        assert self.tracker.nodes[node_id].data_type == "market_data"
        assert self.tracker.nodes[node_id].quality_score == 0.95
        assert self.tracker.node_by_data_hash["hash123"] == node_id
        assert self.tracker.operation_stats['nodes_created'] == 1
    
    def test_create_edge(self):
        """Test edge creation."""
        # Create two nodes first
        node1_id = self.tracker.create_node(
            data_type="raw_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="processed_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        # Create edge
        edge_id = self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION,
            quality_impact=0.05,
            execution_time_ms=150.0
        )
        
        assert edge_id in self.tracker.edges
        assert self.tracker.edges[edge_id].source_node_id == node1_id
        assert self.tracker.edges[edge_id].target_node_id == node2_id
        assert self.tracker.edges[edge_id].transformation_type == TransformationType.NORMALIZATION
        assert self.tracker.operation_stats['edges_created'] == 1
    
    def test_create_edge_invalid_nodes(self):
        """Test edge creation with invalid nodes."""
        with pytest.raises(ValueError, match="Source node .* not found"):
            self.tracker.create_edge(
                source_node_id="invalid-node",
                target_node_id="another-invalid-node",
                transformation_type=TransformationType.NORMALIZATION
            )
    
    def test_build_trace_upstream(self):
        """Test building upstream trace."""
        # Create a chain of nodes
        node1_id = self.tracker.create_node(
            data_type="raw_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="processed_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        node3_id = self.tracker.create_node(
            data_type="final_data",
            data_hash="hash3",
            source_system="source3",
            component="comp3",
            quality_score=0.7
        )
        
        # Create edges
        edge1_id = self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION,
            execution_time_ms=100.0
        )
        
        edge2_id = self.tracker.create_edge(
            source_node_id=node2_id,
            target_node_id=node3_id,
            transformation_type=TransformationType.AGGREGATION,
            execution_time_ms=200.0
        )
        
        # Build trace
        trace = self.tracker.build_trace(node3_id, DataFlowDirection.UPSTREAM)
        
        assert len(trace.nodes) == 3
        assert len(trace.edges) == 2
        assert trace.transformation_count == 2
        assert trace.total_processing_time_ms == 300.0
        assert trace.overall_quality_score == 0.8  # (0.9 + 0.8 + 0.7) / 3
    
    def test_build_trace_nonexistent_node(self):
        """Test building trace for non-existent node."""
        with pytest.raises(ValueError, match="Node .* not found"):
            self.tracker.build_trace("nonexistent-node")
    
    def test_find_dependencies(self):
        """Test finding dependencies."""
        # Create nodes and edges
        node1_id = self.tracker.create_node(
            data_type="source_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="intermediate_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        node3_id = self.tracker.create_node(
            data_type="final_data",
            data_hash="hash3",
            source_system="source3",
            component="comp3",
            quality_score=0.7
        )
        
        # Create dependency chain: node1 -> node2 -> node3
        self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION
        )
        
        self.tracker.create_edge(
            source_node_id=node2_id,
            target_node_id=node3_id,
            transformation_type=TransformationType.AGGREGATION
        )
        
        # Find dependencies for node3
        dependencies = self.tracker.find_dependencies(node3_id)
        
        assert len(dependencies) == 2
        assert node1_id in dependencies
        assert node2_id in dependencies
    
    def test_find_dependents(self):
        """Test finding dependents."""
        # Create nodes and edges
        node1_id = self.tracker.create_node(
            data_type="source_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="intermediate_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        node3_id = self.tracker.create_node(
            data_type="final_data",
            data_hash="hash3",
            source_system="source3",
            component="comp3",
            quality_score=0.7
        )
        
        # Create dependency chain: node1 -> node2 -> node3
        self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION
        )
        
        self.tracker.create_edge(
            source_node_id=node2_id,
            target_node_id=node3_id,
            transformation_type=TransformationType.AGGREGATION
        )
        
        # Find dependents for node1
        dependents = self.tracker.find_dependents(node1_id)
        
        assert len(dependents) == 2
        assert node2_id in dependents
        assert node3_id in dependents
    
    def test_analyze_impact(self):
        """Test impact analysis."""
        # Create nodes
        node1_id = self.tracker.create_node(
            data_type="source_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="dependent_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        # Create edge
        self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION,
            execution_time_ms=100.0
        )
        
        # Analyze impact
        impact = self.tracker.analyze_impact(node1_id)
        
        assert impact['node_id'] == node1_id
        assert impact['quality_score'] == 0.9
        assert impact['dependents_count'] == 1
        assert impact['dependencies_count'] == 0
        assert impact['avg_dependent_quality'] == 0.8
        assert impact['total_processing_time_ms'] == 100.0
        assert impact['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    def test_analyze_impact_nonexistent_node(self):
        """Test impact analysis for non-existent node."""
        impact = self.tracker.analyze_impact("nonexistent-node")
        assert 'error' in impact
        assert 'not found' in impact['error']
    
    def test_get_quality_degradation_path(self):
        """Test quality degradation path calculation."""
        # Create nodes with decreasing quality
        node1_id = self.tracker.create_node(
            data_type="high_quality_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.95
        )
        
        node2_id = self.tracker.create_node(
            data_type="medium_quality_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.85
        )
        
        node3_id = self.tracker.create_node(
            data_type="low_quality_data",
            data_hash="hash3",
            source_system="source3",
            component="comp3",
            quality_score=0.75
        )
        
        # Create edges
        self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION,
            quality_impact=-0.1,
            execution_time_ms=100.0
        )
        
        self.tracker.create_edge(
            source_node_id=node2_id,
            target_node_id=node3_id,
            transformation_type=TransformationType.AGGREGATION,
            quality_impact=-0.1,
            execution_time_ms=200.0
        )
        
        # Get degradation path
        path = self.tracker.get_quality_degradation_path(node3_id)
        
        assert len(path) == 3
        assert path[0]['quality_score'] == 0.75
        assert path[1]['quality_score'] == 0.85
        assert path[2]['quality_score'] == 0.95
        
        # Check transformation info
        assert 'transformation' in path[0]
        assert path[0]['transformation']['type'] == 'aggregation'
        assert path[0]['transformation']['quality_impact'] == -0.1
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Create some test data
        node1_id = self.tracker.create_node(
            data_type="data1",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="data2",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION
        )
        
        stats = self.tracker.get_statistics()
        
        assert stats['nodes_count'] == 2
        assert stats['edges_count'] == 1
        assert stats['traces_count'] == 0
        assert stats['operation_stats']['nodes_created'] == 2
        assert stats['operation_stats']['edges_created'] == 1
        assert stats['graph_stats']['nodes'] == 2
        assert stats['graph_stats']['edges'] == 1
        assert stats['graph_stats']['is_dag'] is True
    
    def test_export_lineage_json(self):
        """Test lineage export in JSON format."""
        # Create test data
        node1_id = self.tracker.create_node(
            data_type="source_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="target_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        edge_id = self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION,
            execution_time_ms=100.0
        )
        
        # Export and parse
        export_data = self.tracker.export_lineage()
        parsed_data = json.loads(export_data)
        
        assert 'nodes' in parsed_data
        assert 'edges' in parsed_data
        assert len(parsed_data['nodes']) == 2
        assert len(parsed_data['edges']) == 1
        
        # Check node data
        node_data = parsed_data['nodes'][0]
        assert 'node_id' in node_data
        assert 'data_type' in node_data
        assert 'quality_score' in node_data
        
        # Check edge data
        edge_data = parsed_data['edges'][0]
        assert 'edge_id' in edge_data
        assert 'source_node_id' in edge_data
        assert 'target_node_id' in edge_data
        assert 'transformation_type' in edge_data
    
    def test_export_lineage_invalid_format(self):
        """Test export with invalid format."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            self.tracker.export_lineage(format='xml')
    
    def test_background_processing(self):
        """Test background processing functionality."""
        # Enable background processing
        config = self.tracker._default_config()
        config['enable_background_processing'] = True
        config['retention_days'] = 0  # Immediate cleanup for testing
        
        tracker = LineageTracker(config)
        
        # Create old node
        old_timestamp = datetime.utcnow() - timedelta(days=1)
        with patch('src.core.data_lineage.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = old_timestamp
            
            node_id = tracker.create_node(
                data_type="old_data",
                data_hash="old_hash",
                source_system="old_source",
                component="old_comp",
                quality_score=0.9
            )
        
        # Start background processing
        tracker.start()
        
        # Wait for cleanup
        time.sleep(0.1)
        
        # Stop background processing
        tracker.stop()
        
        # Check that old node was cleaned up
        assert node_id not in tracker.nodes
    
    def test_node_removal(self):
        """Test node removal functionality."""
        # Create nodes and edges
        node1_id = self.tracker.create_node(
            data_type="data1",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="data2",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        edge_id = self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION
        )
        
        # Remove node
        self.tracker._remove_node(node1_id)
        
        # Check that node and associated data were removed
        assert node1_id not in self.tracker.nodes
        assert edge_id not in self.tracker.edges
        assert "hash1" not in self.tracker.node_by_data_hash
        assert not self.tracker.dependency_graph.has_node(node1_id)
        assert not self.tracker.reverse_dependency_graph.has_node(node1_id)
    
    def test_integrity_verification(self):
        """Test integrity verification."""
        # Create nodes and edges
        node1_id = self.tracker.create_node(
            data_type="data1",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="data2",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        edge_id = self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION
        )
        
        # Manually break integrity by removing node but keeping edge
        del self.tracker.nodes[node1_id]
        
        # Verify integrity (should detect the issue)
        with patch('src.core.data_lineage.logger') as mock_logger:
            self.tracker._verify_integrity()
            mock_logger.warning.assert_called()
            
            # Check that warning contains information about missing node
            warning_args = mock_logger.warning.call_args[0]
            assert "missing source node" in warning_args[0]


class TestLineageQueryEngine:
    """Test the LineageQueryEngine class."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.tracker = LineageTracker()
        self.query_engine = LineageQueryEngine(self.tracker)
    
    def teardown_method(self):
        """Teardown method for each test."""
        if hasattr(self.tracker, 'stop'):
            self.tracker.stop()
    
    def test_find_root_causes(self):
        """Test finding root causes of quality issues."""
        # Create nodes with varying quality scores
        high_quality_node = self.tracker.create_node(
            data_type="high_quality",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.95
        )
        
        low_quality_node = self.tracker.create_node(
            data_type="low_quality",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.6  # Below threshold
        )
        
        final_node = self.tracker.create_node(
            data_type="final_data",
            data_hash="hash3",
            source_system="source3",
            component="comp3",
            quality_score=0.7
        )
        
        # Create edges
        self.tracker.create_edge(
            source_node_id=high_quality_node,
            target_node_id=final_node,
            transformation_type=TransformationType.NORMALIZATION
        )
        
        self.tracker.create_edge(
            source_node_id=low_quality_node,
            target_node_id=final_node,
            transformation_type=TransformationType.AGGREGATION
        )
        
        # Find root causes
        root_causes = self.query_engine.find_root_causes(final_node, quality_threshold=0.8)
        
        assert low_quality_node in root_causes
        assert high_quality_node not in root_causes
        assert final_node in root_causes
    
    def test_find_transformation_bottlenecks(self):
        """Test finding transformation bottlenecks."""
        # Create nodes
        node1_id = self.tracker.create_node(
            data_type="source_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="intermediate_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        node3_id = self.tracker.create_node(
            data_type="final_data",
            data_hash="hash3",
            source_system="source3",
            component="comp3",
            quality_score=0.7
        )
        
        # Create edges with different execution times
        fast_edge = self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION,
            execution_time_ms=500.0  # Fast
        )
        
        slow_edge = self.tracker.create_edge(
            source_node_id=node2_id,
            target_node_id=node3_id,
            transformation_type=TransformationType.AGGREGATION,
            execution_time_ms=2000.0  # Slow (bottleneck)
        )
        
        # Find bottlenecks
        bottlenecks = self.query_engine.find_transformation_bottlenecks(
            node3_id, 
            time_threshold_ms=1000.0
        )
        
        assert slow_edge in bottlenecks
        assert fast_edge not in bottlenecks
    
    def test_calculate_lineage_metrics(self):
        """Test comprehensive lineage metrics calculation."""
        # Create nodes with different quality scores
        node1_id = self.tracker.create_node(
            data_type="high_quality",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.tracker.create_node(
            data_type="medium_quality",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        node3_id = self.tracker.create_node(
            data_type="low_quality",
            data_hash="hash3",
            source_system="source3",
            component="comp3",
            quality_score=0.7
        )
        
        # Create edges with different execution times
        self.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION,
            execution_time_ms=100.0
        )
        
        self.tracker.create_edge(
            source_node_id=node2_id,
            target_node_id=node3_id,
            transformation_type=TransformationType.AGGREGATION,
            execution_time_ms=200.0
        )
        
        # Calculate metrics
        metrics = self.query_engine.calculate_lineage_metrics(node3_id)
        
        assert metrics['node_count'] == 3
        assert metrics['edge_count'] == 2
        assert metrics['avg_quality_score'] == 0.8  # (0.9 + 0.8 + 0.7) / 3
        assert metrics['min_quality_score'] == 0.7
        assert metrics['max_quality_score'] == 0.9
        assert metrics['total_execution_time_ms'] == 300.0
        assert metrics['avg_execution_time_ms'] == 150.0
        assert metrics['transformation_counts']['normalization'] == 1
        assert metrics['transformation_counts']['aggregation'] == 1
    
    def test_calculate_lineage_metrics_empty_trace(self):
        """Test metrics calculation for empty trace."""
        # Create isolated node
        node_id = self.tracker.create_node(
            data_type="isolated_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        # Calculate metrics
        metrics = self.query_engine.calculate_lineage_metrics(node_id)
        
        assert metrics['node_count'] == 1
        assert metrics['edge_count'] == 0
        assert metrics['avg_quality_score'] == 0.9
        assert metrics['total_execution_time_ms'] == 0
        assert metrics['avg_execution_time_ms'] == 0
        assert metrics['transformation_counts'] == {}


class TestGlobalLineageManager:
    """Test the GlobalLineageManager singleton."""
    
    def test_singleton_behavior(self):
        """Test that GlobalLineageManager is a singleton."""
        manager1 = GlobalLineageManager()
        manager2 = GlobalLineageManager()
        
        assert manager1 is manager2
        assert id(manager1) == id(manager2)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = GlobalLineageManager()
        
        assert hasattr(manager, 'tracker')
        assert hasattr(manager, 'query_engine')
        assert hasattr(manager, 'initialized')
        assert manager.initialized is True
        assert isinstance(manager.tracker, LineageTracker)
        assert isinstance(manager.query_engine, LineageQueryEngine)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def setup_method(self):
        """Setup method for each test."""
        # Reset global manager state
        GlobalLineageManager._instance = None
        self.manager = GlobalLineageManager()
    
    def teardown_method(self):
        """Teardown method for each test."""
        if hasattr(self.manager, 'tracker'):
            self.manager.tracker.stop()
    
    def test_track_transformation(self):
        """Test track_transformation utility function."""
        # Create source and target nodes
        source_node_id = self.manager.tracker.create_node(
            data_type="source_data",
            data_hash="source_hash",
            source_system="source_system",
            component="source_comp",
            quality_score=0.9
        )
        
        target_node_id = self.manager.tracker.create_node(
            data_type="target_data",
            data_hash="target_hash",
            source_system="target_system",
            component="target_comp",
            quality_score=0.8
        )
        
        # Track transformation
        edge_id = track_transformation(
            source_data_hash="source_hash",
            target_data_hash="target_hash",
            transformation_type=TransformationType.NORMALIZATION,
            component="transformation_comp",
            execution_time_ms=100.0
        )
        
        assert edge_id in self.manager.tracker.edges
        edge = self.manager.tracker.edges[edge_id]
        assert edge.source_node_id == source_node_id
        assert edge.target_node_id == target_node_id
        assert edge.transformation_type == TransformationType.NORMALIZATION
        assert edge.execution_time_ms == 100.0
    
    def test_track_transformation_missing_nodes(self):
        """Test track_transformation with missing nodes."""
        with pytest.raises(ValueError, match="Source or target node not found"):
            track_transformation(
                source_data_hash="nonexistent_hash",
                target_data_hash="another_nonexistent_hash",
                transformation_type=TransformationType.NORMALIZATION,
                component="test_comp"
            )
    
    def test_get_data_lineage(self):
        """Test get_data_lineage utility function."""
        # Create nodes
        node1_id = self.manager.tracker.create_node(
            data_type="source_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.manager.tracker.create_node(
            data_type="target_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        # Create edge
        self.manager.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION
        )
        
        # Get lineage
        lineage = get_data_lineage("hash2")
        
        assert lineage is not None
        assert isinstance(lineage, LineageTrace)
        assert len(lineage.nodes) >= 1
        assert lineage.target_node_id == node2_id
    
    def test_get_data_lineage_not_found(self):
        """Test get_data_lineage for non-existent data."""
        lineage = get_data_lineage("nonexistent_hash")
        assert lineage is None
    
    def test_analyze_data_quality_impact(self):
        """Test analyze_data_quality_impact utility function."""
        # Create node
        node_id = self.manager.tracker.create_node(
            data_type="test_data",
            data_hash="test_hash",
            source_system="test_source",
            component="test_comp",
            quality_score=0.85
        )
        
        # Analyze impact
        impact = analyze_data_quality_impact("test_hash")
        
        assert 'node_id' in impact
        assert impact['node_id'] == node_id
        assert impact['quality_score'] == 0.85
        assert 'risk_level' in impact
    
    def test_analyze_data_quality_impact_not_found(self):
        """Test analyze_data_quality_impact for non-existent data."""
        impact = analyze_data_quality_impact("nonexistent_hash")
        assert 'error' in impact
        assert impact['error'] == 'Data not found'


class TestLineageIntegration:
    """Integration tests for lineage system."""
    
    def setup_method(self):
        """Setup method for each test."""
        # Reset global manager state
        GlobalLineageManager._instance = None
        self.manager = GlobalLineageManager()
    
    def teardown_method(self):
        """Teardown method for each test."""
        if hasattr(self.manager, 'tracker'):
            self.manager.tracker.stop()
    
    def test_complex_lineage_scenario(self):
        """Test complex lineage tracking scenario."""
        # Create a complex data pipeline
        
        # Raw data sources
        raw_market_data = self.manager.tracker.create_node(
            data_type="raw_market_data",
            data_hash="raw_hash_1",
            source_system="bloomberg",
            component="market_feed",
            quality_score=0.98
        )
        
        raw_news_data = self.manager.tracker.create_node(
            data_type="raw_news_data",
            data_hash="raw_hash_2",
            source_system="reuters",
            component="news_feed",
            quality_score=0.85
        )
        
        # First level processing
        normalized_market_data = self.manager.tracker.create_node(
            data_type="normalized_market_data",
            data_hash="norm_hash_1",
            source_system="internal",
            component="normalizer",
            quality_score=0.95
        )
        
        processed_news_data = self.manager.tracker.create_node(
            data_type="processed_news_data",
            data_hash="proc_hash_1",
            source_system="internal",
            component="nlp_processor",
            quality_score=0.80
        )
        
        # Second level processing
        enriched_market_data = self.manager.tracker.create_node(
            data_type="enriched_market_data",
            data_hash="enrich_hash_1",
            source_system="internal",
            component="enricher",
            quality_score=0.90
        )
        
        # Final aggregated data
        final_data = self.manager.tracker.create_node(
            data_type="aggregated_data",
            data_hash="final_hash",
            source_system="internal",
            component="aggregator",
            quality_score=0.85
        )
        
        # Create transformation edges
        self.manager.tracker.create_edge(
            source_node_id=raw_market_data,
            target_node_id=normalized_market_data,
            transformation_type=TransformationType.NORMALIZATION,
            execution_time_ms=150.0,
            quality_impact=-0.03
        )
        
        self.manager.tracker.create_edge(
            source_node_id=raw_news_data,
            target_node_id=processed_news_data,
            transformation_type=TransformationType.CLEANING,
            execution_time_ms=300.0,
            quality_impact=-0.05
        )
        
        self.manager.tracker.create_edge(
            source_node_id=normalized_market_data,
            target_node_id=enriched_market_data,
            transformation_type=TransformationType.ENRICHMENT,
            execution_time_ms=200.0,
            quality_impact=-0.05
        )
        
        self.manager.tracker.create_edge(
            source_node_id=enriched_market_data,
            target_node_id=final_data,
            transformation_type=TransformationType.AGGREGATION,
            execution_time_ms=100.0,
            quality_impact=-0.05
        )
        
        self.manager.tracker.create_edge(
            source_node_id=processed_news_data,
            target_node_id=final_data,
            transformation_type=TransformationType.JOINING,
            execution_time_ms=250.0,
            quality_impact=0.0
        )
        
        # Test comprehensive lineage analysis
        trace = self.manager.tracker.build_trace(final_data, DataFlowDirection.UPSTREAM)
        
        # Verify trace structure
        assert len(trace.nodes) == 6  # All nodes in the lineage
        assert len(trace.edges) == 5  # All transformations
        
        # Test quality degradation analysis
        degradation_path = self.manager.tracker.get_quality_degradation_path(final_data)
        assert len(degradation_path) >= 3
        
        # Test impact analysis
        impact = self.manager.tracker.analyze_impact(raw_market_data)
        assert impact['dependents_count'] > 0
        assert impact['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        # Test query engine functionality
        root_causes = self.manager.query_engine.find_root_causes(final_data, quality_threshold=0.9)
        assert len(root_causes) > 0
        
        bottlenecks = self.manager.query_engine.find_transformation_bottlenecks(final_data, time_threshold_ms=200.0)
        assert len(bottlenecks) > 0
        
        metrics = self.manager.query_engine.calculate_lineage_metrics(final_data)
        assert metrics['node_count'] == 6
        assert metrics['edge_count'] == 5
        assert metrics['total_execution_time_ms'] == 1000.0
    
    def test_concurrent_lineage_operations(self):
        """Test concurrent lineage operations."""
        def create_lineage_chain(chain_id: int):
            """Create a lineage chain in a thread."""
            nodes = []
            for i in range(5):
                node_id = self.manager.tracker.create_node(
                    data_type=f"data_type_{chain_id}_{i}",
                    data_hash=f"hash_{chain_id}_{i}",
                    source_system=f"source_{chain_id}",
                    component=f"comp_{chain_id}",
                    quality_score=0.9 - (i * 0.1)
                )
                nodes.append(node_id)
            
            # Create edges between nodes
            for i in range(len(nodes) - 1):
                self.manager.tracker.create_edge(
                    source_node_id=nodes[i],
                    target_node_id=nodes[i + 1],
                    transformation_type=TransformationType.NORMALIZATION,
                    execution_time_ms=100.0
                )
            
            return nodes[-1]  # Return final node
        
        # Run multiple lineage chains concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_lineage_chain, i) for i in range(10)]
            final_nodes = [future.result() for future in as_completed(futures)]
        
        # Verify all chains were created
        assert len(final_nodes) == 10
        
        # Verify statistics
        stats = self.manager.tracker.get_statistics()
        assert stats['nodes_count'] == 50  # 10 chains * 5 nodes each
        assert stats['edges_count'] == 40  # 10 chains * 4 edges each
        
        # Test concurrent queries
        def query_lineage(node_id: str):
            """Query lineage in a thread."""
            trace = self.manager.tracker.build_trace(node_id, DataFlowDirection.UPSTREAM)
            return len(trace.nodes)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(query_lineage, node_id) for node_id in final_nodes]
            node_counts = [future.result() for future in as_completed(futures)]
        
        # Verify all queries returned expected results
        assert all(count == 5 for count in node_counts)
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for critical operations."""
        # Create a large lineage graph
        nodes = []
        for i in range(1000):
            node_id = self.manager.tracker.create_node(
                data_type=f"data_type_{i}",
                data_hash=f"hash_{i}",
                source_system="test_source",
                component="test_comp",
                quality_score=0.9
            )
            nodes.append(node_id)
        
        # Create edges (chain structure)
        for i in range(len(nodes) - 1):
            self.manager.tracker.create_edge(
                source_node_id=nodes[i],
                target_node_id=nodes[i + 1],
                transformation_type=TransformationType.NORMALIZATION,
                execution_time_ms=10.0
            )
        
        # Benchmark trace building
        start_time = time.time()
        trace = self.manager.tracker.build_trace(nodes[-1], DataFlowDirection.UPSTREAM)
        trace_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Verify performance requirements
        assert trace_time < 1000  # Should complete within 1 second
        assert len(trace.nodes) == 1000
        assert len(trace.edges) == 999
        
        # Benchmark dependency finding
        start_time = time.time()
        dependencies = self.manager.tracker.find_dependencies(nodes[-1])
        dep_time = (time.time() - start_time) * 1000
        
        assert dep_time < 500  # Should complete within 500ms
        assert len(dependencies) == 999
        
        # Benchmark impact analysis
        start_time = time.time()
        impact = self.manager.tracker.analyze_impact(nodes[0])
        impact_time = (time.time() - start_time) * 1000
        
        assert impact_time < 200  # Should complete within 200ms
        assert impact['dependents_count'] == 999
    
    def test_lineage_export_import_roundtrip(self):
        """Test export/import roundtrip functionality."""
        # Create test lineage
        node1_id = self.manager.tracker.create_node(
            data_type="source_data",
            data_hash="hash1",
            source_system="source1",
            component="comp1",
            quality_score=0.9
        )
        
        node2_id = self.manager.tracker.create_node(
            data_type="target_data",
            data_hash="hash2",
            source_system="source2",
            component="comp2",
            quality_score=0.8
        )
        
        edge_id = self.manager.tracker.create_edge(
            source_node_id=node1_id,
            target_node_id=node2_id,
            transformation_type=TransformationType.NORMALIZATION,
            execution_time_ms=100.0
        )
        
        # Export lineage
        export_data = self.manager.tracker.export_lineage()
        
        # Parse exported data
        parsed_data = json.loads(export_data)
        
        # Verify export structure
        assert 'nodes' in parsed_data
        assert 'edges' in parsed_data
        assert len(parsed_data['nodes']) == 2
        assert len(parsed_data['edges']) == 1
        
        # Verify node data integrity
        node_data = parsed_data['nodes'][0]
        assert all(key in node_data for key in [
            'node_id', 'data_type', 'timestamp', 'quality_score',
            'source_system', 'component'
        ])
        
        # Verify edge data integrity
        edge_data = parsed_data['edges'][0]
        assert all(key in edge_data for key in [
            'edge_id', 'source_node_id', 'target_node_id',
            'transformation_type', 'timestamp', 'execution_time_ms'
        ])
    
    def test_mathematical_validation(self):
        """Test mathematical properties of lineage system."""
        # Test DAG property maintenance
        nodes = []
        for i in range(10):
            node_id = self.manager.tracker.create_node(
                data_type=f"data_{i}",
                data_hash=f"hash_{i}",
                source_system="test",
                component="test",
                quality_score=0.9
            )
            nodes.append(node_id)
        
        # Create edges that maintain DAG property
        for i in range(len(nodes) - 1):
            self.manager.tracker.create_edge(
                source_node_id=nodes[i],
                target_node_id=nodes[i + 1],
                transformation_type=TransformationType.NORMALIZATION
            )
        
        # Verify DAG property
        stats = self.manager.tracker.get_statistics()
        assert stats['graph_stats']['is_dag'] is True
        
        # Test quality score constraints
        for node_id in nodes:
            node = self.manager.tracker.nodes[node_id]
            assert 0 <= node.quality_score <= 1
        
        # Test trace metrics consistency
        trace = self.manager.tracker.build_trace(nodes[-1], DataFlowDirection.UPSTREAM)
        trace.update_metrics()
        
        # Verify metrics are consistent
        assert trace.transformation_count == len(trace.edges)
        assert trace.overall_quality_score == sum(node.quality_score for node in trace.nodes) / len(trace.nodes)
        
        # Test lineage path consistency
        dependencies = self.manager.tracker.find_dependencies(nodes[-1])
        assert len(dependencies) == len(nodes) - 1
        
        dependents = self.manager.tracker.find_dependents(nodes[0])
        assert len(dependents) == len(nodes) - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])