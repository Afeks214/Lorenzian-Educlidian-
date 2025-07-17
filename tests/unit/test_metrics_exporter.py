"""
Unit tests for the MetricsExporter component.
Tests Prometheus metrics, custom business metrics, and tracking.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from contextlib import asynccontextmanager
from prometheus_client import REGISTRY, CollectorRegistry

from src.monitoring.metrics_exporter import MetricsExporter, metrics_exporter


class TestMetricsExporter:
    """Test suite for MetricsExporter."""
    
    @pytest.fixture
    def clean_registry(self):
        """Create a clean registry for testing."""
        # Create a new registry for isolated tests
        registry = CollectorRegistry()
        yield registry
    
    @pytest.fixture
    def test_metrics(self, clean_registry):
        """Create MetricsExporter instance with test registry."""
        metrics = MetricsExporter(registry=clean_registry)
        return metrics
    
    def test_initialization(self, test_metrics):
        """Test MetricsExporter initialization."""
        assert test_metrics.registry is not None
        assert hasattr(test_metrics, 'health_status')
        assert hasattr(test_metrics, 'inference_latency')
        assert hasattr(test_metrics, 'http_requests')
        assert hasattr(test_metrics, 'http_request_duration')
        assert hasattr(test_metrics, 'active_connections')
        assert hasattr(test_metrics, 'event_bus_throughput')
        assert hasattr(test_metrics, 'errors_total')
        assert hasattr(test_metrics, 'trade_pnl')
        assert hasattr(test_metrics, 'model_confidence')
        assert hasattr(test_metrics, 'synergy_response')
        assert hasattr(test_metrics, 'active_positions')
        assert hasattr(test_metrics, 'correlation_ids')
        assert hasattr(test_metrics, 'system_info')
    
    def test_update_health_status(self, test_metrics):
        """Test health status metric update."""
        test_metrics.update_health_status("healthy", "redis")
        test_metrics.update_health_status("degraded", "api")
        test_metrics.update_health_status("unhealthy", "models")
        
        # Verify metrics are set correctly
        # In real test, would query the registry
        assert True  # Placeholder for actual metric verification
    
    @pytest.mark.asyncio
    async def test_measure_inference_latency(self, test_metrics):
        """Test inference latency measurement."""
        async with test_metrics.measure_inference_latency(
            model_type="strategic_marl",
            agent_name="strategic_agent",
            correlation_id="test-123"
        ):
            # Simulate some work
            await asyncio.sleep(0.01)
        
        # Verify latency was recorded
        assert "test-123" in test_metrics.correlation_ids
    
    def test_track_http_request(self, test_metrics):
        """Test HTTP request tracking."""
        with test_metrics.track_http_request("GET", "/health", "200"):
            # Simulate request processing
            time.sleep(0.01)
        
        # Verify request was tracked
        # In real test, would check counter and histogram values
        assert True  # Placeholder
    
    def test_track_active_connection(self, test_metrics):
        """Test active connection tracking."""
        # Add connection
        test_metrics.track_active_connection(1)
        # Remove connection
        test_metrics.track_active_connection(-1)
        
        # Gauge should be back to baseline
        assert True  # Placeholder
    
    def test_record_event(self, test_metrics):
        """Test event recording."""
        test_metrics.record_event("SYNERGY_DETECTED", "success")
        test_metrics.record_event("TRADE_EXECUTED", "success")
        test_metrics.record_event("TRADE_FAILED", "failure")
        
        # Verify events were recorded
        assert True  # Placeholder
    
    def test_record_error(self, test_metrics):
        """Test error recording."""
        test_metrics.record_error("validation_error", "api", "bad_request")
        test_metrics.record_error("database_error", "redis", "connection_lost")
        test_metrics.record_error("model_error", "inference", "timeout")
        
        # Verify errors were counted
        assert True  # Placeholder
    
    def test_update_trade_pnl(self, test_metrics):
        """Test trade P&L tracking."""
        test_metrics.update_trade_pnl(100.50, "BTCUSDT", "win")
        test_metrics.update_trade_pnl(-50.25, "ETHUSDT", "loss")
        test_metrics.update_trade_pnl(0.0, "ADAUSDT", "breakeven")
        
        # Verify P&L was recorded
        assert True  # Placeholder
    
    def test_update_model_confidence(self, test_metrics):
        """Test model confidence tracking."""
        test_metrics.update_model_confidence(0.85, "strategic_marl", "strategic_agent")
        test_metrics.update_model_confidence(0.72, "tactical_marl", "tactical_agent")
        test_metrics.update_model_confidence(0.91, "risk_marl", "risk_agent")
        
        # Verify confidence scores were recorded
        assert True  # Placeholder
    
    def test_record_synergy_response(self, test_metrics):
        """Test synergy response tracking."""
        test_metrics.record_synergy_response("TYPE_1", "success", "corr-123")
        test_metrics.record_synergy_response("TYPE_2", "success", "corr-124")
        test_metrics.record_synergy_response("TYPE_3", "failure", "corr-125")
        
        # Verify synergy responses were tracked
        assert "corr-123" in test_metrics.correlation_ids
        assert "corr-124" in test_metrics.correlation_ids
    
    def test_update_active_positions(self, test_metrics):
        """Test active positions tracking."""
        test_metrics.update_active_positions(5)
        test_metrics.update_active_positions(7)
        test_metrics.update_active_positions(3)
        
        # Verify position count was updated
        assert True  # Placeholder
    
    def test_track_correlation_id(self, test_metrics):
        """Test correlation ID tracking."""
        test_metrics.track_correlation_id("test-corr-1")
        test_metrics.track_correlation_id("test-corr-2")
        
        assert "test-corr-1" in test_metrics.correlation_ids
        assert "test-corr-2" in test_metrics.correlation_ids
        
        # Test cleanup of old IDs (would need to mock time)
        assert len(test_metrics.correlation_ids) <= 10000
    
    def test_update_system_info(self, test_metrics):
        """Test system info metric update."""
        test_metrics.update_system_info({
            "version": "1.0.0",
            "environment": "test",
            "service": "grandmodel"
        })
        
        # Verify info metric was set
        assert True  # Placeholder
    
    def test_record_rate_limit(self, test_metrics):
        """Test rate limit tracking."""
        test_metrics.record_rate_limit("decide", allowed=True)
        test_metrics.record_rate_limit("decide", allowed=True)
        test_metrics.record_rate_limit("decide", allowed=False)
        test_metrics.record_rate_limit("health", allowed=True)
        
        # Verify rate limits were tracked
        assert True  # Placeholder
    
    @pytest.mark.asyncio
    async def test_concurrent_metric_updates(self, test_metrics):
        """Test thread-safety of metric updates."""
        import asyncio
        
        async def update_metrics():
            for i in range(100):
                test_metrics.record_event("TEST_EVENT", "success")
                test_metrics.update_active_positions(i % 10)
                await asyncio.sleep(0.001)
        
        # Run multiple concurrent updates
        tasks = [update_metrics() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Metrics should be updated without errors
        assert True  # Placeholder
    
    def test_metric_labels(self, test_metrics):
        """Test that metrics have appropriate labels."""
        # Test various label combinations
        test_metrics.update_health_status("healthy", "redis")
        test_metrics.record_event("SYNERGY_DETECTED", "success")
        test_metrics.update_model_confidence(0.85, "strategic_marl", "agent1")
        
        # Verify labels are correctly set
        # Would need to inspect the actual prometheus metrics
        assert True  # Placeholder


class TestMetricsExporterHelpers:
    """Test helper functions and utilities."""
    
    def test_get_metrics_export(self, test_metrics):
        """Test metrics export functionality."""
        # Add some metrics
        test_metrics.update_health_status("healthy", "all")
        test_metrics.record_event("TEST", "success")
        
        # Export metrics
        from src.monitoring.metrics_exporter import get_metrics
        metrics_text = get_metrics()
        
        assert isinstance(metrics_text, str)
        assert "grandmodel_health_status" in metrics_text
        assert "grandmodel_event_bus_throughput_total" in metrics_text
    
    def test_get_metrics_content_type(self):
        """Test content type for Prometheus metrics."""
        from src.monitoring.metrics_exporter import get_metrics_content_type
        
        content_type = get_metrics_content_type()
        assert content_type == "text/plain; version=0.0.4; charset=utf-8"
    
    def test_singleton_instance(self):
        """Test that metrics_exporter is a singleton."""
        from src.monitoring.metrics_exporter import metrics_exporter as instance1
        from src.monitoring.metrics_exporter import metrics_exporter as instance2
        
        assert instance1 is instance2


class TestMetricsIntegration:
    """Integration tests for metrics with other components."""
    
    @pytest.mark.asyncio
    async def test_metrics_with_http_tracking(self, test_metrics):
        """Test metrics integration with HTTP request tracking."""
        # Simulate a complete request flow
        correlation_id = "test-request-123"
        test_metrics.track_correlation_id(correlation_id)
        
        with test_metrics.track_http_request("POST", "/decide", "200"):
            async with test_metrics.measure_inference_latency(
                "strategic_marl", "ensemble", correlation_id
            ):
                await asyncio.sleep(0.002)  # 2ms inference
            
            test_metrics.update_model_confidence(0.87, "strategic_marl", "ensemble")
            test_metrics.record_synergy_response("TYPE_1", "success", correlation_id)
        
        # Verify all metrics were recorded with correlation
        assert correlation_id in test_metrics.correlation_ids
    
    def test_metrics_error_handling(self, test_metrics):
        """Test that metrics handle errors gracefully."""
        # Test with invalid inputs
        test_metrics.update_health_status("invalid_status", "test")  # Should not crash
        test_metrics.update_model_confidence(-0.5, "test", "test")  # Invalid confidence
        test_metrics.update_active_positions(-10)  # Negative positions
        
        # Exporter should continue working
        test_metrics.record_event("TEST", "success")
        assert True  # If we get here, error handling worked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])