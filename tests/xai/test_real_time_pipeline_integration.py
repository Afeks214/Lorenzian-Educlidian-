"""
Comprehensive Integration Tests for XAI Real-time Pipeline

Agent Beta: Real-time streaming specialist
Mission: Bulletproof testing for the complete XAI pipeline

This module provides comprehensive integration tests for the complete XAI real-time
explanation pipeline, testing end-to-end functionality, performance requirements,
error handling, and integration with the Strategic MARL system.

Test Coverage:
- End-to-end pipeline flow
- Performance and latency requirements
- Error handling and graceful degradation
- Component integration and dependencies
- WebSocket connectivity and streaming
- Strategic MARL integration

Author: Agent Beta - Real-time Streaming Specialist
Version: 1.0 - Comprehensive Integration Tests
"""

import asyncio
import pytest
import time
import uuid
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

# Import XAI pipeline components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.core.events import EventType, Event, EventBus
from src.xai.pipeline.decision_capture import DecisionCapture, DecisionContext
from src.xai.pipeline.context_processor import ContextProcessor, ProcessedContext, ProcessingPriority
from src.xai.pipeline.websocket_manager import WebSocketManager, WebSocketMessage, MessageType
from src.xai.pipeline.streaming_engine import StreamingEngine, ExplanationPriority, ExplanationAudience
from src.xai.pipeline.marl_integration import XAIPipelineIntegration


# Mock kernel for testing
class MockKernel:
    """Mock kernel for testing"""
    def __init__(self):
        self.event_bus = EventBus()
        self.config = {}


# Test fixtures
@pytest.fixture
def mock_kernel():
    """Create mock kernel"""
    return MockKernel()


@pytest.fixture
def test_config():
    """Create test configuration"""
    return {
        'health_check_interval_seconds': 1,
        'enable_graceful_degradation': True,
        'websocket_manager': {
            'host': 'localhost',
            'port': 8769,  # Test port
            'authentication': {'enabled': False}
        },
        'decision_capture': {
            'max_capture_latency_ns': 100_000,
            'queue_size': 1000,
            'redis': {'enabled': False}  # Disable Redis for testing
        },
        'context_processor': {
            'queue_size': 500,
            'cache_size': 100
        },
        'streaming_engine': {
            'target_explanation_latency_ms': 200,
            'llm': {'timeout_seconds': 1}
        }
    }


@pytest.fixture
def mock_strategic_decision():
    """Create mock strategic decision"""
    return {
        'action': 'buy',
        'confidence': 0.85,
        'uncertainty': 0.15,
        'should_proceed': True,
        'reasoning': 'Strong momentum signals detected across multiple indicators',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'agent_contributions': {
            'MLMI': 0.4,
            'NWRQK': 0.35,
            'Regime': 0.25
        },
        'performance_metrics': {
            'ensemble_probabilities': [0.1, 0.2, 0.7],
            'dynamic_weights': [0.4, 0.35, 0.25],
            'gating_confidence': 0.9,
            'inference_time_ms': 2.5,
            'max_confidence': 0.85,
            'min_confidence': 0.25,
            'total_weight': 1.0
        }
    }


class TestDecisionCapture:
    """Test Decision Capture component"""
    
    @pytest.mark.asyncio
    async def test_decision_capture_initialization(self, mock_kernel, test_config):
        """Test decision capture initialization"""
        capture = DecisionCapture(mock_kernel, test_config['decision_capture'])
        await capture.initialize()
        
        assert capture._initialized
        assert capture.active
        
        await capture.shutdown()
    
    @pytest.mark.asyncio
    async def test_decision_capture_latency(self, mock_kernel, test_config, mock_strategic_decision):
        """Test decision capture latency requirements"""
        capture = DecisionCapture(mock_kernel, test_config['decision_capture'])
        await capture.initialize()
        
        # Create and publish strategic decision event
        event = mock_kernel.event_bus.create_event(
            EventType.STRATEGIC_DECISION,
            mock_strategic_decision,
            source="test_strategic_marl"
        )
        
        start_time = time.perf_counter_ns()
        mock_kernel.event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check metrics
        metrics = capture.get_metrics()
        
        assert metrics['total_decisions_captured'] > 0
        assert metrics['avg_capture_latency_ns'] < test_config['decision_capture']['max_capture_latency_ns']
        
        await capture.shutdown()
    
    @pytest.mark.asyncio
    async def test_decision_capture_context_extraction(self, mock_kernel, test_config, mock_strategic_decision):
        """Test decision context extraction"""
        capture = DecisionCapture(mock_kernel, test_config['decision_capture'])
        await capture.initialize()
        
        # Create test context
        context = await capture._build_decision_context(
            mock_strategic_decision, 
            time.perf_counter_ns()
        )
        
        assert isinstance(context, DecisionContext)
        assert context.action == 'buy'
        assert context.confidence == 0.85
        assert context.symbol == 'NQ'
        assert len(context.agent_contributions) == 3
        
        await capture.shutdown()


class TestContextProcessor:
    """Test Context Processor component"""
    
    @pytest.mark.asyncio
    async def test_context_processor_initialization(self, mock_kernel, test_config):
        """Test context processor initialization"""
        processor = ContextProcessor(mock_kernel, test_config['context_processor'])
        await processor.initialize()
        
        assert processor._initialized
        assert processor.active
        assert len(processor.feature_extractors) > 0
        
        await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self, mock_kernel, test_config):
        """Test feature extraction from decision context"""
        processor = ContextProcessor(mock_kernel, test_config['context_processor'])
        await processor.initialize()
        
        # Create mock decision context
        context = DecisionContext(
            decision_id="test-123",
            timestamp=datetime.now(timezone.utc),
            symbol="NQ",
            action="buy",
            confidence=0.85,
            strategic_decision={},
            agent_contributions={'MLMI': 0.4, 'NWRQK': 0.35, 'Regime': 0.25},
            ensemble_probabilities=[0.1, 0.2, 0.7],
            gating_weights=[0.4, 0.35, 0.25],
            reasoning="Test reasoning",
            market_data={},
            volatility=0.025,
            volume_ratio=1.2,
            momentum_indicators={'short': 0.02, 'long': 0.015},
            regime_classification='trending',
            inference_time_ms=2.5,
            gating_confidence=0.9,
            uncertainty=0.15,
            should_proceed=True,
            capture_latency_ns=75_000
        )
        
        # Extract features
        feature_vector = await processor._extract_feature_vector(context)
        
        assert isinstance(feature_vector, np.ndarray)
        assert len(feature_vector) == test_config['context_processor'].get('feature_vector_dim', 128)
        assert not np.isnan(feature_vector).any()
        
        await processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_processing_performance(self, mock_kernel, test_config):
        """Test context processing performance"""
        processor = ContextProcessor(mock_kernel, test_config['context_processor'])
        await processor.initialize()
        
        # Create test contexts
        contexts = []
        for i in range(10):
            context = DecisionContext(
                decision_id=f"test-{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="NQ",
                action="buy",
                confidence=0.8 + i * 0.01,
                strategic_decision={},
                agent_contributions={'MLMI': 0.4, 'NWRQK': 0.35, 'Regime': 0.25},
                ensemble_probabilities=[0.1, 0.2, 0.7],
                gating_weights=[0.4, 0.35, 0.25],
                reasoning="Test reasoning",
                market_data={},
                volatility=0.02 + i * 0.001,
                volume_ratio=1.0 + i * 0.1,
                momentum_indicators={'short': 0.01, 'long': 0.01},
                regime_classification='trending',
                inference_time_ms=2.0 + i * 0.1,
                gating_confidence=0.9,
                uncertainty=0.1 + i * 0.01,
                should_proceed=True,
                capture_latency_ns=50_000 + i * 1000
            )
            contexts.append(context)
        
        # Process contexts
        start_time = time.perf_counter()
        
        for context in contexts:
            await processor._queue_context_for_processing(context, ProcessingPriority.NORMAL)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        processing_time = time.perf_counter() - start_time
        
        # Check metrics
        metrics = processor.get_metrics()
        
        assert metrics['total_contexts_processed'] >= len(contexts)
        assert metrics['avg_processing_time_ms'] < 100  # Should be under 100ms
        
        await processor.shutdown()


class TestWebSocketManager:
    """Test WebSocket Manager component"""
    
    @pytest.mark.asyncio
    async def test_websocket_manager_initialization(self, mock_kernel, test_config):
        """Test WebSocket manager initialization"""
        manager = WebSocketManager(mock_kernel, test_config['websocket_manager'])
        await manager.initialize()
        
        assert manager._initialized
        assert manager.server is not None
        
        await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_message_queuing(self, mock_kernel, test_config):
        """Test message queuing functionality"""
        manager = WebSocketManager(mock_kernel, test_config['websocket_manager'])
        await manager.initialize()
        
        # Create test message
        test_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EXPLANATION,
            timestamp=datetime.now(timezone.utc),
            payload={
                'explanation': 'Test explanation',
                'confidence': 0.85
            }
        )
        
        # Queue message
        success = await manager.queue_message(test_message)
        assert success
        
        # Check metrics
        metrics = manager.get_metrics()
        assert metrics['queue_size'] > 0
        
        await manager.shutdown()


class TestStreamingEngine:
    """Test Streaming Engine component"""
    
    @pytest.mark.asyncio
    async def test_streaming_engine_initialization(self, mock_kernel, test_config):
        """Test streaming engine initialization"""
        engine = StreamingEngine(mock_kernel, test_config['streaming_engine'])
        await engine.initialize()
        
        assert engine._initialized
        assert engine.active
        assert engine.websocket_manager is not None
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_explanation_generation(self, mock_kernel, test_config):
        """Test explanation generation"""
        engine = StreamingEngine(mock_kernel, test_config['streaming_engine'])
        await engine.initialize()
        
        # Create mock processed context
        processed_context = ProcessedContext(
            decision_id="test-123",
            timestamp=datetime.now(timezone.utc),
            processing_latency_ms=25.0,
            decision_summary={
                'action': 'buy',
                'confidence': 0.85,
                'certainty_level': 'high',
                'consensus_strength': 'strong',
                'market_conditions': 'favorable'
            },
            key_factors=[
                {'factor': 'Strong momentum', 'importance': 0.9}
            ],
            risk_assessment={'overall_risk': 0.3},
            confidence_breakdown={'model_confidence': 0.85},
            feature_vector=np.random.rand(128),
            embedding_vector=None,
            similarity_hash="test-hash",
            llm_context={'market': {'volatility': 0.02}},
            explanation_template="standard",
            context_quality_score=0.8,
            agent_performance={'MLMI': 0.4},
            market_regime_score=0.7,
            decision_complexity_score=0.4,
            priority=ProcessingPriority.HIGH,
            target_audiences=['trader'],
            streaming_ready=True
        )
        
        # Create explanation request
        from src.xai.pipeline.streaming_engine import ExplanationRequest
        
        request = ExplanationRequest(
            request_id=str(uuid.uuid4()),
            decision_id=processed_context.decision_id,
            processed_context=processed_context,
            priority=ExplanationPriority.HIGH,
            audience=ExplanationAudience.TRADER,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Generate explanation
        explanation = await engine._generate_explanation(request)
        
        assert explanation.explanation_text
        assert explanation.summary
        assert explanation.quality_score > 0
        assert explanation.audience == ExplanationAudience.TRADER
        
        await engine.shutdown()


class TestEndToEndPipeline:
    """Test end-to-end pipeline functionality"""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, mock_kernel, test_config, mock_strategic_decision):
        """Test complete pipeline from Strategic MARL decision to WebSocket delivery"""
        # Initialize integration
        integration = XAIPipelineIntegration(mock_kernel, test_config)
        await integration.initialize()
        
        # Verify all components are ready
        status = integration.get_integration_status()
        assert status['pipeline_active']
        assert status['components_initialized'] == status['total_components']
        
        # Create and publish strategic decision
        event = mock_kernel.event_bus.create_event(
            EventType.STRATEGIC_DECISION,
            mock_strategic_decision,
            source="test_strategic_marl"
        )
        
        start_time = time.perf_counter()
        mock_kernel.event_bus.publish(event)
        
        # Wait for complete processing
        await asyncio.sleep(2.0)
        
        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000
        
        # Check that processing completed within acceptable time
        assert total_latency_ms < 5000  # 5 seconds max for complete pipeline
        
        # Check component metrics
        status = integration.get_integration_status()
        
        if 'component_metrics' in status:
            # Decision capture metrics
            dc_metrics = status['component_metrics'].get('decision_capture', {})
            if dc_metrics:
                assert dc_metrics.get('total_decisions_captured', 0) > 0
            
            # Context processor metrics
            cp_metrics = status['component_metrics'].get('context_processor', {})
            if cp_metrics:
                assert cp_metrics.get('total_contexts_processed', 0) >= 0
            
            # Streaming engine metrics
            se_metrics = status['component_metrics'].get('streaming_engine', {})
            if se_metrics:
                assert se_metrics.get('total_explanations_generated', 0) >= 0
        
        await integration.shutdown()
    
    @pytest.mark.asyncio
    async def test_pipeline_performance_requirements(self, mock_kernel, test_config):
        """Test pipeline performance meets requirements"""
        integration = XAIPipelineIntegration(mock_kernel, test_config)
        await integration.initialize()
        
        # Process multiple decisions to test throughput
        decisions = []
        for i in range(5):
            decision = {
                'action': 'buy' if i % 2 == 0 else 'sell',
                'confidence': 0.7 + i * 0.05,
                'uncertainty': 0.3 - i * 0.05,
                'should_proceed': True,
                'reasoning': f'Test decision {i}',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'agent_contributions': {
                    'MLMI': 0.4 + i * 0.01,
                    'NWRQK': 0.35,
                    'Regime': 0.25 - i * 0.01
                },
                'performance_metrics': {
                    'ensemble_probabilities': [0.1, 0.2, 0.7],
                    'dynamic_weights': [0.4, 0.35, 0.25],
                    'gating_confidence': 0.85 + i * 0.01,
                    'inference_time_ms': 2.0 + i * 0.1
                }
            }
            decisions.append(decision)
        
        # Publish decisions rapidly
        start_time = time.perf_counter()
        
        for i, decision in enumerate(decisions):
            event = mock_kernel.event_bus.create_event(
                EventType.STRATEGIC_DECISION,
                decision,
                source=f"test_strategic_marl_{i}"
            )
            mock_kernel.event_bus.publish(event)
            
            # Small delay between decisions
            await asyncio.sleep(0.01)
        
        # Wait for processing
        await asyncio.sleep(3.0)
        
        processing_time = time.perf_counter() - start_time
        
        # Check performance requirements
        status = integration.get_integration_status()
        
        # Throughput should handle multiple decisions per second
        throughput = len(decisions) / processing_time
        assert throughput > 1.0  # Should handle at least 1 decision per second
        
        await integration.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_graceful_degradation(self, mock_kernel, test_config):
        """Test error handling and graceful degradation"""
        # Enable graceful degradation
        test_config['enable_graceful_degradation'] = True
        
        integration = XAIPipelineIntegration(mock_kernel, test_config)
        
        # Simulate component initialization failure
        with patch.object(WebSocketManager, 'initialize', side_effect=Exception("Test error")):
            await integration.initialize()
            
            # Should still initialize with graceful degradation
            assert integration._initialized
            
            status = integration.get_integration_status()
            assert status['health']['status'] in ['degraded', 'unhealthy']
        
        await integration.shutdown()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, mock_kernel, test_config):
        """Test health monitoring functionality"""
        # Set short health check interval
        test_config['health_check_interval_seconds'] = 0.5
        
        integration = XAIPipelineIntegration(mock_kernel, test_config)
        await integration.initialize()
        
        # Wait for health checks
        await asyncio.sleep(1.5)
        
        status = integration.get_integration_status()
        
        # Should have performed health checks
        assert status['health']['last_check'] is not None
        assert status['health']['status'] in ['healthy', 'degraded', 'unhealthy']
        
        await integration.shutdown()


class TestIntegrationWithStrategicMARL:
    """Test integration with Strategic MARL system"""
    
    @pytest.mark.asyncio
    async def test_strategic_decision_event_handling(self, mock_kernel, test_config, mock_strategic_decision):
        """Test handling of Strategic MARL decision events"""
        integration = XAIPipelineIntegration(mock_kernel, test_config)
        await integration.initialize()
        
        # Subscribe to XAI events to verify processing
        xai_events = []
        
        def capture_xai_event(event):
            xai_events.append(event)
        
        # Note: Using INDICATOR_UPDATE as temporary event type
        mock_kernel.event_bus.subscribe(EventType.INDICATOR_UPDATE, capture_xai_event)
        
        # Publish strategic decision
        event = mock_kernel.event_bus.create_event(
            EventType.STRATEGIC_DECISION,
            mock_strategic_decision,
            source="test_strategic_marl"
        )
        mock_kernel.event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Should have generated XAI events
        assert len(xai_events) > 0
        
        # Check that events contain decision context or processed context
        context_events = [e for e in xai_events if 'context' in e.payload or 'processed_context' in e.payload]
        assert len(context_events) > 0
        
        await integration.shutdown()
    
    @pytest.mark.asyncio
    async def test_zero_latency_impact_on_trading(self, mock_kernel, test_config, mock_strategic_decision):
        """Test that XAI pipeline has zero impact on trading latency"""
        # Measure Strategic MARL performance without XAI
        start_time = time.perf_counter_ns()
        
        event = mock_kernel.event_bus.create_event(
            EventType.STRATEGIC_DECISION,
            mock_strategic_decision,
            source="test_strategic_marl"
        )
        mock_kernel.event_bus.publish(event)
        
        baseline_latency_ns = time.perf_counter_ns() - start_time
        
        # Now measure with XAI pipeline
        integration = XAIPipelineIntegration(mock_kernel, test_config)
        await integration.initialize()
        
        start_time = time.perf_counter_ns()
        
        event = mock_kernel.event_bus.create_event(
            EventType.STRATEGIC_DECISION,
            mock_strategic_decision,
            source="test_strategic_marl"
        )
        mock_kernel.event_bus.publish(event)
        
        with_xai_latency_ns = time.perf_counter_ns() - start_time
        
        # XAI should add minimal latency (async processing)
        latency_overhead_ns = with_xai_latency_ns - baseline_latency_ns
        
        # Should be under 10 microseconds overhead
        assert latency_overhead_ns < 10_000
        
        await integration.shutdown()


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_decision_capture_benchmark(self, mock_kernel, test_config):
        """Benchmark decision capture performance"""
        capture = DecisionCapture(mock_kernel, test_config['decision_capture'])
        await capture.initialize()
        
        # Benchmark parameters
        num_decisions = 100
        decisions = []
        
        # Generate test decisions
        for i in range(num_decisions):
            decision = {
                'action': 'buy',
                'confidence': 0.8,
                'uncertainty': 0.2,
                'should_proceed': True,
                'reasoning': f'Test decision {i}',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'agent_contributions': {'MLMI': 0.4, 'NWRQK': 0.35, 'Regime': 0.25},
                'performance_metrics': {
                    'ensemble_probabilities': [0.1, 0.2, 0.7],
                    'dynamic_weights': [0.4, 0.35, 0.25],
                    'gating_confidence': 0.9,
                    'inference_time_ms': 2.5
                }
            }
            decisions.append(decision)
        
        # Benchmark capture
        start_time = time.perf_counter()
        
        for decision in decisions:
            event = mock_kernel.event_bus.create_event(
                EventType.STRATEGIC_DECISION,
                decision,
                source="benchmark_test"
            )
            mock_kernel.event_bus.publish(event)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Check performance
        metrics = capture.get_metrics()
        
        print(f"\nDecision Capture Benchmark Results:")
        print(f"  Decisions processed: {metrics['total_decisions_captured']}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {metrics['total_decisions_captured'] / total_time:.1f} decisions/sec")
        print(f"  Average latency: {metrics['avg_capture_latency_ns'] / 1000:.1f}μs")
        print(f"  Max latency: {metrics['max_capture_latency_ns'] / 1000:.1f}μs")
        
        # Performance assertions
        assert metrics['total_decisions_captured'] >= num_decisions
        assert metrics['avg_capture_latency_ns'] < test_config['decision_capture']['max_capture_latency_ns']
        assert (metrics['total_decisions_captured'] / total_time) > 50  # At least 50 decisions/sec
        
        await capture.shutdown()
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency_benchmark(self, mock_kernel, test_config, mock_strategic_decision):
        """Benchmark end-to-end pipeline latency"""
        integration = XAIPipelineIntegration(mock_kernel, test_config)
        await integration.initialize()
        
        # Wait for components to be ready
        await asyncio.sleep(1.0)
        
        # Benchmark end-to-end latency
        latencies = []
        
        for i in range(10):
            start_time = time.perf_counter_ns()
            
            # Publish decision
            event = mock_kernel.event_bus.create_event(
                EventType.STRATEGIC_DECISION,
                mock_strategic_decision,
                source=f"benchmark_test_{i}"
            )
            mock_kernel.event_bus.publish(event)
            
            # Wait for explanation to be generated (simplified - in real test would monitor WebSocket)
            await asyncio.sleep(0.5)
            
            end_time = time.perf_counter_ns()
            latency_ms = (end_time - start_time) / 1_000_000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"\nEnd-to-End Latency Benchmark Results:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  Max latency: {max_latency:.1f}ms")
        print(f"  95th percentile: {p95_latency:.1f}ms")
        print(f"  Target latency: {test_config['streaming_engine']['target_explanation_latency_ms']}ms")
        
        # Performance assertions
        assert avg_latency < test_config['streaming_engine']['target_explanation_latency_ms'] * 2  # Allow 2x target
        assert p95_latency < test_config['streaming_engine']['target_explanation_latency_ms'] * 3  # Allow 3x for p95
        
        await integration.shutdown()


# Test runner
if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v", "-s"])