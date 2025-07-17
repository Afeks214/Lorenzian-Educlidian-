"""
Comprehensive Test Suite for Universal Observation Enrichment System

This test suite validates all components of the sequential architecture
including performance benchmarks and integration testing.
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any
import numpy as np

from ..events import EventBus, EventType
from .universal_observation_enricher import UniversalObservationEnricher, ContextLayer
from .dynamic_attention_mechanism import DynamicAttentionMechanism, AttentionType
from .observation_cache import ObservationCache, CacheEvictionPolicy
from .correlation_tracker import CorrelationTracker, CorrelationType


class TestUniversalObservationEnricher:
    """Test suite for Universal Observation Enricher"""
    
    def setup_method(self):
        """Setup test environment"""
        self.event_bus = EventBus()
        self.config = {
            "max_enrichment_time_ms": 5.0,
            "enable_temporal_context": True,
            "temporal_window_size": 20,
            "enable_correlation_tracking": True,
            "cache": {
                "max_size": 1000,
                "max_memory_mb": 64,
                "eviction_policy": "adaptive"
            },
            "attention": {
                "attention_temperature": 1.0,
                "min_attention_threshold": 0.01,
                "max_context_sources": 20
            },
            "correlation": {
                "correlation_window_size": 50,
                "min_correlation_samples": 5,
                "correlation_update_interval": 1.0
            }
        }
        self.enricher = UniversalObservationEnricher(self.event_bus, self.config)
    
    def test_agent_registration(self):
        """Test agent registration functionality"""
        agent_id = "test_agent"
        marl_system = "test_marl"
        metadata = {"agent_type": "test_type"}
        
        self.enricher.register_agent(agent_id, marl_system, metadata)
        
        assert agent_id in self.enricher.agent_registry
        assert self.enricher.agent_registry[agent_id]["marl_system"] == marl_system
        assert self.enricher.agent_registry[agent_id]["metadata"] == metadata
    
    def test_marl_system_registration(self):
        """Test MARL system registration"""
        system_id = "test_system"
        metadata = {"system_type": "test"}
        
        self.enricher.register_marl_system(system_id, metadata)
        
        assert system_id in self.enricher.marl_system_registry
        assert self.enricher.marl_system_registry[system_id]["metadata"] == metadata
    
    def test_basic_observation_enrichment(self):
        """Test basic observation enrichment"""
        # Register test agents
        self.enricher.register_agent("agent1", "marl1", {"agent_type": "test"})
        self.enricher.register_agent("agent2", "marl1", {"agent_type": "test"})
        
        # Test observation
        base_obs = {
            "price": 100.0,
            "volume": 1000,
            "signal": 0.5
        }
        
        enriched_obs = self.enricher.enrich_observation("agent1", base_obs)
        
        assert enriched_obs.base_observation == base_obs
        assert isinstance(enriched_obs.intra_marl_context, dict)
        assert isinstance(enriched_obs.inter_marl_context, dict)
        assert isinstance(enriched_obs.temporal_context, dict)
        assert isinstance(enriched_obs.attention_weights, dict)
        assert enriched_obs.enrichment_time_ms >= 0
    
    def test_enrichment_performance(self):
        """Test enrichment performance meets <5ms target"""
        # Register multiple agents
        for i in range(10):
            self.enricher.register_agent(f"agent_{i}", "marl1", {"agent_type": "test"})
        
        # Test observation
        base_obs = {
            "price": 100.0,
            "volume": 1000,
            "signal": 0.5,
            "timestamp": datetime.now()
        }
        
        # Measure enrichment time
        start_time = time.time()
        enriched_obs = self.enricher.enrich_observation("agent_0", base_obs)
        end_time = time.time()
        
        enrichment_time_ms = (end_time - start_time) * 1000
        
        # Should be under 5ms
        assert enrichment_time_ms < 5.0
        assert enriched_obs.enrichment_time_ms < 5.0
    
    def test_context_layer_enrichment(self):
        """Test different context layer enrichment"""
        # Register agents in different systems
        self.enricher.register_agent("agent1", "strategic", {"agent_type": "entry"})
        self.enricher.register_agent("agent2", "strategic", {"agent_type": "momentum"})
        self.enricher.register_agent("agent3", "tactical", {"agent_type": "fvg"})
        
        self.enricher.register_marl_system("strategic", {"system_type": "strategic"})
        self.enricher.register_marl_system("tactical", {"system_type": "tactical"})
        
        # Add some observations to create context
        obs1 = {"price": 100.0, "signal": 0.6}
        obs2 = {"price": 101.0, "signal": 0.7}
        obs3 = {"price": 102.0, "signal": 0.8}
        
        # Enrich observations to build context
        self.enricher.enrich_observation("agent1", obs1)
        self.enricher.enrich_observation("agent2", obs2)
        self.enricher.enrich_observation("agent3", obs3)
        
        # Test enrichment with built context
        enriched_obs = self.enricher.enrich_observation("agent1", obs1)
        
        # Should have intra-MARL context (agent2 in same system)
        assert len(enriched_obs.intra_marl_context) > 0
        
        # Should have inter-MARL context (agent3 in different system)
        assert len(enriched_obs.inter_marl_context) > 0
    
    def test_temporal_context(self):
        """Test temporal context enrichment"""
        self.enricher.register_agent("agent1", "marl1", {"agent_type": "test"})
        
        # Add multiple observations to build temporal context
        for i in range(10):
            obs = {"price": 100.0 + i, "signal": 0.5 + i * 0.1}
            self.enricher.enrich_observation("agent1", obs)
        
        # Test enrichment with temporal context
        test_obs = {"price": 110.0, "signal": 0.9}
        enriched_obs = self.enricher.enrich_observation("agent1", test_obs)
        
        # Should have temporal context
        assert len(enriched_obs.temporal_context) > 0
        assert "recent_trend" in enriched_obs.temporal_context or "statistics" in enriched_obs.temporal_context


class TestDynamicAttentionMechanism:
    """Test suite for Dynamic Attention Mechanism"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            "attention_temperature": 1.0,
            "min_attention_threshold": 0.01,
            "max_context_sources": 20,
            "enable_adaptive_temperature": True
        }
        self.attention_mechanism = DynamicAttentionMechanism(self.config)
    
    def test_attention_weight_computation(self):
        """Test attention weight computation"""
        agent_id = "test_agent"
        base_observation = {"price": 100.0, "volume": 1000}
        context_sources = {
            "source1": {"price": 101.0, "volume": 1100},
            "source2": {"price": 99.0, "volume": 900},
            "source3": {"price": 102.0, "volume": 1200}
        }
        
        attention_weights = self.attention_mechanism.compute_attention_weights(
            agent_id, base_observation, context_sources
        )
        
        # Should return weights for all sources
        assert len(attention_weights) == len(context_sources)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(attention_weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # All weights should be positive
        for weight in attention_weights.values():
            assert weight > 0
    
    def test_attention_weight_normalization(self):
        """Test attention weight normalization"""
        agent_id = "test_agent"
        base_observation = {"price": 100.0}
        context_sources = {f"source{i}": {"price": 100.0 + i} for i in range(5)}
        
        attention_weights = self.attention_mechanism.compute_attention_weights(
            agent_id, base_observation, context_sources
        )
        
        # Weights should sum to 1.0
        total_weight = sum(attention_weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # All weights should be above minimum threshold
        for weight in attention_weights.values():
            assert weight >= self.config["min_attention_threshold"]
    
    def test_performance_requirements(self):
        """Test attention computation performance"""
        agent_id = "test_agent"
        base_observation = {"price": 100.0, "volume": 1000}
        context_sources = {f"source{i}": {"price": 100.0 + i, "volume": 1000 + i} for i in range(50)}
        
        start_time = time.time()
        attention_weights = self.attention_mechanism.compute_attention_weights(
            agent_id, base_observation, context_sources
        )
        end_time = time.time()
        
        computation_time_ms = (end_time - start_time) * 1000
        
        # Should be fast enough for real-time use
        assert computation_time_ms < 10.0  # 10ms limit
        assert len(attention_weights) > 0


class TestObservationCache:
    """Test suite for Observation Cache"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            "max_size": 1000,
            "max_memory_mb": 64,
            "eviction_policy": "lru",
            "enable_compression": False
        }
        self.cache = ObservationCache(self.config)
    
    def test_basic_cache_operations(self):
        """Test basic cache operations"""
        key = "test_key"
        value = {"data": "test_value"}
        
        # Put and get
        self.cache.put(key, value)
        retrieved_value = self.cache.get(key)
        
        assert retrieved_value == value
    
    def test_cache_performance(self):
        """Test cache performance requirements"""
        # Test put performance
        start_time = time.time()
        for i in range(100):
            self.cache.put(f"key_{i}", {"data": f"value_{i}"})
        put_time = (time.time() - start_time) * 1000
        
        # Test get performance
        start_time = time.time()
        for i in range(100):
            self.cache.get(f"key_{i}")
        get_time = (time.time() - start_time) * 1000
        
        # Should be fast enough
        assert put_time < 100.0  # 100ms for 100 puts
        assert get_time < 50.0   # 50ms for 100 gets
    
    def test_cache_eviction(self):
        """Test cache eviction policies"""
        # Fill cache beyond capacity
        for i in range(self.config["max_size"] + 100):
            self.cache.put(f"key_{i}", {"data": f"value_{i}"})
        
        # Cache should not exceed max size
        assert self.cache.l2_cache.size() <= self.config["max_size"]
    
    def test_cache_hit_rate(self):
        """Test cache hit rate"""
        # Put some values
        for i in range(10):
            self.cache.put(f"key_{i}", {"data": f"value_{i}"})
        
        # Access them multiple times
        for _ in range(5):
            for i in range(10):
                self.cache.get(f"key_{i}")
        
        metrics = self.cache.get_performance_metrics()
        
        # Should have high hit rate
        assert metrics["cache_hit_rate"] > 0.8


class TestCorrelationTracker:
    """Test suite for Correlation Tracker"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            "correlation_window_size": 50,
            "min_correlation_samples": 5,
            "correlation_update_interval": 0.1
        }
        self.correlation_tracker = CorrelationTracker(self.config)
    
    def test_correlation_tracking(self):
        """Test correlation tracking functionality"""
        # Add observations for multiple agents
        for i in range(10):
            obs1 = {"price": 100.0 + i, "volume": 1000 + i * 10}
            obs2 = {"price": 100.0 + i * 0.8, "volume": 1000 + i * 8}  # Correlated
            obs3 = {"price": 100.0 - i * 0.5, "volume": 1000 - i * 5}  # Anti-correlated
            
            self.correlation_tracker.update_correlations("agent1", obs1, {})
            self.correlation_tracker.update_correlations("agent2", obs2, {})
            self.correlation_tracker.update_correlations("agent3", obs3, {})
        
        # Get correlation matrix
        correlation_matrix = self.correlation_tracker.get_correlation_matrix()
        
        # Should have correlations computed
        assert len(correlation_matrix) > 0
        
        # Check agent correlations
        agent1_correlations = self.correlation_tracker.get_agent_correlations("agent1")
        assert len(agent1_correlations) > 0
    
    def test_superposition_relevance(self):
        """Test superposition relevance computation"""
        # Add observations to build superposition states
        for i in range(10):
            obs = {"price": 100.0 + i, "signal": 0.5 + i * 0.05}
            self.correlation_tracker.update_correlations("agent1", obs, {})
        
        # Compute superposition relevance
        base_obs = {"price": 105.0, "signal": 0.75}
        attention_weights = {"agent1": 0.8}
        
        relevance_scores = self.correlation_tracker.compute_superposition_relevance(
            "agent2", base_obs, attention_weights
        )
        
        # Should return relevance scores
        assert isinstance(relevance_scores, dict)
    
    def test_correlation_performance(self):
        """Test correlation computation performance"""
        # Add many observations
        start_time = time.time()
        for i in range(100):
            obs = {"price": 100.0 + i, "volume": 1000 + i}
            self.correlation_tracker.update_correlations("agent1", obs, {})
        
        computation_time = (time.time() - start_time) * 1000
        
        # Should be reasonably fast
        assert computation_time < 1000.0  # 1 second for 100 updates


class TestIntegration:
    """Integration test suite"""
    
    def test_full_system_integration(self):
        """Test full system integration"""
        # Setup complete system
        event_bus = EventBus()
        config = {
            "max_enrichment_time_ms": 5.0,
            "enable_temporal_context": True,
            "temporal_window_size": 20,
            "enable_correlation_tracking": True,
            "cache": {
                "max_size": 1000,
                "max_memory_mb": 64,
                "eviction_policy": "adaptive"
            },
            "attention": {
                "attention_temperature": 1.0,
                "min_attention_threshold": 0.01,
                "max_context_sources": 20
            },
            "correlation": {
                "correlation_window_size": 50,
                "min_correlation_samples": 5,
                "correlation_update_interval": 0.1
            }
        }
        
        enricher = UniversalObservationEnricher(event_bus, config)
        
        # Register agents
        enricher.register_agent("strategic_agent", "strategic_marl", {"agent_type": "entry"})
        enricher.register_agent("tactical_agent", "tactical_marl", {"agent_type": "fvg"})
        enricher.register_agent("risk_agent", "risk_marl", {"agent_type": "risk"})
        
        # Register MARL systems
        enricher.register_marl_system("strategic_marl", {"system_type": "strategic"})
        enricher.register_marl_system("tactical_marl", {"system_type": "tactical"})
        enricher.register_marl_system("risk_marl", {"system_type": "risk"})
        
        # Test enrichment workflow
        base_obs = {
            "price": 100.0,
            "volume": 1000,
            "timestamp": datetime.now(),
            "indicators": {
                "momentum": 0.6,
                "volatility": 0.25
            }
        }
        
        # Enrich observation
        enriched_obs = enricher.enrich_observation("strategic_agent", base_obs)
        
        # Validate enrichment
        assert enriched_obs.base_observation == base_obs
        assert enriched_obs.enrichment_time_ms < 5.0
        assert enriched_obs.total_context_sources >= 0
        
        # Check performance metrics
        metrics = enricher.get_performance_metrics()
        assert metrics["registered_agents"] == 3
        assert metrics["registered_marl_systems"] == 3
    
    def test_concurrent_enrichment(self):
        """Test concurrent enrichment operations"""
        event_bus = EventBus()
        config = {
            "max_enrichment_time_ms": 5.0,
            "cache": {"max_size": 1000, "max_memory_mb": 64},
            "attention": {"attention_temperature": 1.0},
            "correlation": {"correlation_window_size": 50}
        }
        
        enricher = UniversalObservationEnricher(event_bus, config)
        
        # Register agents
        for i in range(5):
            enricher.register_agent(f"agent_{i}", "marl1", {"agent_type": "test"})
        
        # Define concurrent enrichment function
        def enrich_observations(agent_id, num_observations):
            for i in range(num_observations):
                obs = {"price": 100.0 + i, "volume": 1000 + i}
                enricher.enrich_observation(agent_id, obs)
        
        # Start concurrent threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=enrich_observations,
                args=(f"agent_{i}", 20)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check final metrics
        metrics = enricher.get_performance_metrics()
        assert metrics["registered_agents"] == 5
        
        # Check that all enrichments completed successfully
        correlation_summary = enricher.correlation_tracker.get_correlation_summary()
        assert len(correlation_summary["correlation_matrices"]) > 0


def run_performance_benchmark():
    """Run performance benchmark"""
    print("Running Performance Benchmark...")
    
    # Setup system
    event_bus = EventBus()
    config = {
        "max_enrichment_time_ms": 5.0,
        "cache": {"max_size": 10000, "max_memory_mb": 128},
        "attention": {"attention_temperature": 1.0},
        "correlation": {"correlation_window_size": 100}
    }
    
    enricher = UniversalObservationEnricher(event_bus, config)
    
    # Register many agents
    for i in range(50):
        enricher.register_agent(f"agent_{i}", f"marl_{i % 5}", {"agent_type": "test"})
    
    # Benchmark enrichment
    start_time = time.time()
    for i in range(1000):
        agent_id = f"agent_{i % 50}"
        obs = {
            "price": 100.0 + i,
            "volume": 1000 + i,
            "timestamp": datetime.now(),
            "indicators": {"signal": 0.5 + (i % 10) * 0.05}
        }
        enriched_obs = enricher.enrich_observation(agent_id, obs)
        
        # Ensure performance target is met
        if enriched_obs.enrichment_time_ms > 5.0:
            print(f"WARNING: Enrichment time {enriched_obs.enrichment_time_ms:.2f}ms exceeds target")
    
    total_time = time.time() - start_time
    avg_time_per_enrichment = (total_time / 1000) * 1000
    
    print(f"Benchmark Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average enrichment time: {avg_time_per_enrichment:.2f}ms")
    print(f"  Target: <5ms per enrichment")
    print(f"  Result: {'PASS' if avg_time_per_enrichment < 5.0 else 'FAIL'}")
    
    # Get final metrics
    metrics = enricher.get_performance_metrics()
    print(f"  Cache hit rate: {metrics['cache_performance']['cache_hit_rate']:.2%}")
    print(f"  Correlation computations: {metrics['correlation_performance']['correlation_computations']}")


if __name__ == "__main__":
    # Run performance benchmark
    run_performance_benchmark()
    
    # Run tests
    pytest.main([__file__, "-v"])