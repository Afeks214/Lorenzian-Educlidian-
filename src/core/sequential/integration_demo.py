"""
Integration Demo for Universal Observation Enrichment System

This script demonstrates how to integrate the Universal Observation Enrichment System
with the existing GrandModel event system and MARL components.
"""

import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import structlog

from ..events import EventBus, EventType, Event
from ..kernel import AlgoSpaceKernel
from .universal_observation_enricher import UniversalObservationEnricher
from .dynamic_attention_mechanism import DynamicAttentionMechanism
from .observation_cache import ObservationCache
from .correlation_tracker import CorrelationTracker

logger = structlog.get_logger(__name__)


class SequentialArchitectureIntegration:
    """
    Integration class for Sequential Architecture with GrandModel
    
    This class demonstrates how to integrate the Universal Observation Enrichment System
    with the existing GrandModel system components.
    """
    
    def __init__(self, kernel: AlgoSpaceKernel, config: Dict[str, Any]):
        """
        Initialize the integration
        
        Args:
            kernel: AlgoSpaceKernel instance
            config: Configuration for sequential architecture
        """
        self.kernel = kernel
        self.config = config
        self.event_bus = kernel.get_event_bus()
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Initialize Universal Observation Enricher
        enricher_config = config.get("observation_enricher", {})
        self.observation_enricher = UniversalObservationEnricher(
            self.event_bus, enricher_config
        )
        
        # Setup integration with existing components
        self._setup_agent_registration()
        self._setup_marl_system_registration()
        self._setup_event_handlers()
        
        self.logger.info("Sequential Architecture Integration initialized")
    
    def _setup_agent_registration(self) -> None:
        """Register known agents from the kernel"""
        
        # Strategic MARL agents
        strategic_agents = [
            ("entry_agent", "strategic_marl", {"agent_type": "entry_agent", "timeframe": "30m"}),
            ("momentum_agent", "strategic_marl", {"agent_type": "momentum_agent", "timeframe": "30m"}),
            ("fvg_agent", "strategic_marl", {"agent_type": "fvg_agent", "timeframe": "30m"}),
            ("mlmi_agent", "strategic_marl", {"agent_type": "mlmi_agent", "timeframe": "30m"}),
            ("nwrqk_agent", "strategic_marl", {"agent_type": "nwrqk_agent", "timeframe": "30m"}),
        ]
        
        for agent_id, marl_system, metadata in strategic_agents:
            self.observation_enricher.register_agent(agent_id, marl_system, metadata)
        
        # Tactical MARL agents
        tactical_agents = [
            ("tactical_entry", "tactical_marl", {"agent_type": "entry_agent", "timeframe": "5m"}),
            ("tactical_momentum", "tactical_marl", {"agent_type": "momentum_agent", "timeframe": "5m"}),
            ("tactical_fvg", "tactical_marl", {"agent_type": "fvg_agent", "timeframe": "5m"}),
        ]
        
        for agent_id, marl_system, metadata in tactical_agents:
            self.observation_enricher.register_agent(agent_id, marl_system, metadata)
        
        # Risk management agents
        risk_agents = [
            ("risk_monitor", "risk_marl", {"agent_type": "risk_agent", "focus": "monitoring"}),
            ("position_sizing", "risk_marl", {"agent_type": "risk_agent", "focus": "position_sizing"}),
            ("stop_target", "risk_marl", {"agent_type": "risk_agent", "focus": "stop_target"}),
        ]
        
        for agent_id, marl_system, metadata in risk_agents:
            self.observation_enricher.register_agent(agent_id, marl_system, metadata)
        
        # Execution agents
        execution_agents = [
            ("execution_router", "execution_marl", {"agent_type": "execution_agent", "focus": "routing"}),
            ("execution_timing", "execution_marl", {"agent_type": "execution_agent", "focus": "timing"}),
        ]
        
        for agent_id, marl_system, metadata in execution_agents:
            self.observation_enricher.register_agent(agent_id, marl_system, metadata)
        
        self.logger.info("Agent registration completed")
    
    def _setup_marl_system_registration(self) -> None:
        """Register MARL systems"""
        
        marl_systems = [
            ("strategic_marl", {
                "system_type": "strategic",
                "timeframe": "30m",
                "agents": ["entry_agent", "momentum_agent", "fvg_agent", "mlmi_agent", "nwrqk_agent"],
                "priority": "high"
            }),
            ("tactical_marl", {
                "system_type": "tactical",
                "timeframe": "5m",
                "agents": ["tactical_entry", "tactical_momentum", "tactical_fvg"],
                "priority": "medium"
            }),
            ("risk_marl", {
                "system_type": "risk_management",
                "timeframe": "real_time",
                "agents": ["risk_monitor", "position_sizing", "stop_target"],
                "priority": "critical"
            }),
            ("execution_marl", {
                "system_type": "execution",
                "timeframe": "real_time",
                "agents": ["execution_router", "execution_timing"],
                "priority": "high"
            })
        ]
        
        for system_id, metadata in marl_systems:
            self.observation_enricher.register_marl_system(system_id, metadata)
        
        self.logger.info("MARL system registration completed")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for sequential architecture"""
        
        # Handler for strategic decisions
        self.event_bus.subscribe(
            EventType.STRATEGIC_DECISION,
            self._handle_strategic_decision
        )
        
        # Handler for tactical decisions
        self.event_bus.subscribe(
            EventType.TACTICAL_DECISION,
            self._handle_tactical_decision
        )
        
        # Handler for synergy detection
        self.event_bus.subscribe(
            EventType.SYNERGY_DETECTED,
            self._handle_synergy_detection
        )
        
        # Handler for risk updates
        self.event_bus.subscribe(
            EventType.RISK_UPDATE,
            self._handle_risk_update
        )
        
        # Handler for VaR updates
        self.event_bus.subscribe(
            EventType.VAR_UPDATE,
            self._handle_var_update
        )
        
        # Handler for trade execution
        self.event_bus.subscribe(
            EventType.EXECUTE_TRADE,
            self._handle_trade_execution
        )
        
        self.logger.info("Event handlers configured")
    
    def _handle_strategic_decision(self, event: Event) -> None:
        """Handle strategic decision events"""
        
        payload = event.payload
        agent_id = payload.get("agent_id", "unknown_strategic")
        
        # Enrich the observation with context
        enriched_obs = self.observation_enricher.enrich_observation(
            agent_id, payload
        )
        
        # Log enrichment results
        self.logger.info(
            "Strategic decision enriched",
            agent_id=agent_id,
            context_sources=enriched_obs.total_context_sources,
            enrichment_time_ms=enriched_obs.enrichment_time_ms
        )
        
        # Create enriched event
        enriched_event = self.event_bus.create_event(
            EventType.STRATEGIC_DECISION,
            {
                "original_payload": payload,
                "enriched_observation": enriched_obs
            },
            source=f"sequential_enricher_{agent_id}"
        )
        
        # You could publish this back to event bus for other components to use
        # self.event_bus.publish(enriched_event)
    
    def _handle_tactical_decision(self, event: Event) -> None:
        """Handle tactical decision events"""
        
        payload = event.payload
        agent_id = payload.get("agent_id", "unknown_tactical")
        
        # Enrich the observation with context
        enriched_obs = self.observation_enricher.enrich_observation(
            agent_id, payload
        )
        
        self.logger.info(
            "Tactical decision enriched",
            agent_id=agent_id,
            context_sources=enriched_obs.total_context_sources,
            enrichment_time_ms=enriched_obs.enrichment_time_ms
        )
    
    def _handle_synergy_detection(self, event: Event) -> None:
        """Handle synergy detection events"""
        
        payload = event.payload
        
        # Enrich synergy information with context from all agents
        enriched_obs = self.observation_enricher.enrich_observation(
            "synergy_detector", payload
        )
        
        self.logger.info(
            "Synergy detection enriched",
            synergy_type=payload.get("synergy_type", "unknown"),
            context_sources=enriched_obs.total_context_sources,
            enrichment_time_ms=enriched_obs.enrichment_time_ms
        )
    
    def _handle_risk_update(self, event: Event) -> None:
        """Handle risk update events"""
        
        payload = event.payload
        agent_id = payload.get("agent_id", "risk_monitor")
        
        # Enrich risk information with context
        enriched_obs = self.observation_enricher.enrich_observation(
            agent_id, payload
        )
        
        self.logger.info(
            "Risk update enriched",
            agent_id=agent_id,
            context_sources=enriched_obs.total_context_sources,
            enrichment_time_ms=enriched_obs.enrichment_time_ms
        )
    
    def _handle_var_update(self, event: Event) -> None:
        """Handle VaR update events"""
        
        payload = event.payload
        
        # Enrich VaR information with context
        enriched_obs = self.observation_enricher.enrich_observation(
            "var_calculator", payload
        )
        
        self.logger.info(
            "VaR update enriched",
            var_value=payload.get("var_value", "unknown"),
            context_sources=enriched_obs.total_context_sources,
            enrichment_time_ms=enriched_obs.enrichment_time_ms
        )
    
    def _handle_trade_execution(self, event: Event) -> None:
        """Handle trade execution events"""
        
        payload = event.payload
        agent_id = payload.get("agent_id", "execution_router")
        
        # Enrich execution information with context
        enriched_obs = self.observation_enricher.enrich_observation(
            agent_id, payload
        )
        
        self.logger.info(
            "Trade execution enriched",
            agent_id=agent_id,
            context_sources=enriched_obs.total_context_sources,
            enrichment_time_ms=enriched_obs.enrichment_time_ms
        )
    
    def get_enrichment_statistics(self) -> Dict[str, Any]:
        """Get enrichment statistics"""
        
        return {
            "observation_enricher": self.observation_enricher.get_performance_metrics(),
            "registered_agents": len(self.observation_enricher.agent_registry),
            "registered_marl_systems": len(self.observation_enricher.marl_system_registry)
        }
    
    def get_correlation_analysis(self) -> Dict[str, Any]:
        """Get correlation analysis"""
        
        return self.observation_enricher.correlation_tracker.get_correlation_summary()
    
    def demonstrate_enrichment(self) -> None:
        """Demonstrate the enrichment system"""
        
        # Simulate some agent observations
        self.logger.info("Starting enrichment demonstration")
        
        # Simulate strategic agent observation
        strategic_obs = {
            "agent_id": "entry_agent",
            "timestamp": datetime.now(),
            "market_conditions": {
                "price": 4200.50,
                "volume": 1500000,
                "volatility": 0.25
            },
            "indicators": {
                "mlmi_signal": 0.75,
                "momentum": 0.60,
                "entry_confidence": 0.85
            },
            "risk_metrics": {
                "position_size": 0.02,
                "stop_loss": 4150.00,
                "take_profit": 4300.00
            }
        }
        
        # Enrich the observation
        enriched_obs = self.observation_enricher.enrich_observation(
            "entry_agent", strategic_obs
        )
        
        self.logger.info(
            "Demonstration enrichment completed",
            enrichment_time_ms=enriched_obs.enrichment_time_ms,
            context_sources=enriched_obs.total_context_sources,
            attention_weights=list(enriched_obs.attention_weights.keys())[:5]
        )
        
        # Simulate tactical agent observation
        tactical_obs = {
            "agent_id": "tactical_entry",
            "timestamp": datetime.now(),
            "micro_structure": {
                "bid_ask_spread": 0.25,
                "order_book_depth": 50000,
                "trade_flow": "bullish"
            },
            "fvg_analysis": {
                "active_gaps": 3,
                "nearest_gap": 4195.00,
                "gap_strength": 0.70
            }
        }
        
        enriched_tactical = self.observation_enricher.enrich_observation(
            "tactical_entry", tactical_obs
        )
        
        self.logger.info(
            "Tactical demonstration enrichment completed",
            enrichment_time_ms=enriched_tactical.enrichment_time_ms,
            context_sources=enriched_tactical.total_context_sources
        )
        
        # Show performance metrics
        metrics = self.get_enrichment_statistics()
        self.logger.info(
            "Enrichment system performance",
            **metrics
        )


def create_sequential_architecture_integration(kernel: AlgoSpaceKernel) -> SequentialArchitectureIntegration:
    """
    Factory function to create Sequential Architecture Integration
    
    Args:
        kernel: AlgoSpaceKernel instance
        
    Returns:
        SequentialArchitectureIntegration instance
    """
    
    # Default configuration
    config = {
        "observation_enricher": {
            "max_enrichment_time_ms": 5.0,
            "enable_temporal_context": True,
            "temporal_window_size": 20,
            "enable_correlation_tracking": True,
            "cache": {
                "max_size": 10000,
                "max_memory_mb": 512,
                "eviction_policy": "adaptive"
            },
            "attention": {
                "attention_temperature": 1.0,
                "min_attention_threshold": 0.01,
                "max_context_sources": 50,
                "enable_adaptive_temperature": True
            },
            "correlation": {
                "correlation_window_size": 100,
                "min_correlation_samples": 10,
                "correlation_update_interval": 5.0
            }
        }
    }
    
    return SequentialArchitectureIntegration(kernel, config)


if __name__ == "__main__":
    # Demo usage
    import sys
    sys.path.append("/home/QuantNova/GrandModel")
    
    # Create a demo kernel
    kernel = AlgoSpaceKernel()
    
    # Create integration
    integration = create_sequential_architecture_integration(kernel)
    
    # Run demonstration
    integration.demonstrate_enrichment()
    
    # Show statistics
    stats = integration.get_enrichment_statistics()
    print("Enrichment Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show correlation analysis
    correlation_analysis = integration.get_correlation_analysis()
    print("\nCorrelation Analysis:")
    print(f"  Top correlated pairs: {len(correlation_analysis.get('top_correlated_pairs', []))}")
    print(f"  Correlation statistics: {correlation_analysis.get('correlation_statistics', {})}")