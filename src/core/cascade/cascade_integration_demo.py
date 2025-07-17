"""
Cascade Integration Demo - Comprehensive demonstration of the Inter-MARL Cascade Management System

This demo showcases the complete integration of all cascade components:
- SuperpositionCascadeManager
- MARLCoordinationEngine
- CascadePerformanceMonitor
- CascadeValidationFramework
- EmergencyCascadeProtocols
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import structlog

from ..events import EventBus, Event, EventType
from ..performance.performance_monitor import PerformanceMonitor
from .superposition_cascade_manager import SuperpositionCascadeManager, SuperpositionPacket, SuperpositionType
from .marl_coordination_engine import MARLCoordinationEngine
from .cascade_performance_monitor import CascadePerformanceMonitor
from .cascade_validation_framework import CascadeValidationFramework
from .emergency_cascade_protocols import EmergencyCascadeProtocols, EmergencyType, EmergencyLevel


class CascadeIntegrationDemo:
    """
    Comprehensive demonstration of the Inter-MARL Cascade Management System.
    Shows the complete integration and interaction between all cascade components.
    """

    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        
        # Initialize event bus
        self.event_bus = EventBus()
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize cascade components
        self.cascade_manager = SuperpositionCascadeManager(
            event_bus=self.event_bus,
            performance_monitor=self.performance_monitor,
            max_concurrent_flows=100,
            cascade_timeout_ms=100
        )
        
        self.coordination_engine = MARLCoordinationEngine(
            event_bus=self.event_bus,
            performance_monitor=self.performance_monitor,
            heartbeat_interval=1.0,
            sync_timeout=10.0
        )
        
        self.performance_monitor_cascade = CascadePerformanceMonitor(
            event_bus=self.event_bus,
            performance_monitor=self.performance_monitor,
            target_end_to_end_latency_ms=100.0,
            sampling_interval=0.1
        )
        
        self.validation_framework = CascadeValidationFramework(
            event_bus=self.event_bus,
            validation_interval=5.0,
            deep_validation_interval=30.0
        )
        
        self.emergency_protocols = EmergencyCascadeProtocols(
            event_bus=self.event_bus,
            cascade_manager=self.cascade_manager,
            coordination_engine=self.coordination_engine,
            validation_framework=self.validation_framework,
            emergency_response_timeout=30.0
        )
        
        # Demo metrics
        self.demo_metrics = {
            "packets_processed": 0,
            "average_latency": 0.0,
            "coordination_events": 0,
            "validation_checks": 0,
            "emergency_events": 0
        }
        
    def setup_demo_systems(self):
        """Setup demo MARL systems"""
        self.logger.info("Setting up demo MARL systems")
        
        # Register MARL systems with coordination engine
        self.coordination_engine.register_marl_system(
            system_id="strategic",
            system_name="Strategic MARL System",
            capabilities=["long_term_planning", "market_analysis", "risk_assessment"],
            configuration={
                "timeframe": "30min",
                "agents": ["entry_agent", "momentum_agent", "mlmi_agent"],
                "performance_target": "high_accuracy"
            }
        )
        
        self.coordination_engine.register_marl_system(
            system_id="tactical",
            system_name="Tactical MARL System",
            capabilities=["signal_generation", "timing_optimization", "execution_planning"],
            configuration={
                "timeframe": "5min",
                "agents": ["fvg_agent", "nwrqk_agent", "momentum_agent"],
                "performance_target": "low_latency"
            }
        )
        
        self.coordination_engine.register_marl_system(
            system_id="risk",
            system_name="Risk Management MARL System",
            capabilities=["risk_monitoring", "position_sizing", "correlation_tracking"],
            configuration={
                "var_method": "monte_carlo",
                "confidence_level": 0.95,
                "performance_target": "risk_minimization"
            }
        )
        
        self.coordination_engine.register_marl_system(
            system_id="execution",
            system_name="Execution MARL System",
            capabilities=["order_routing", "execution_optimization", "slippage_control"],
            configuration={
                "venues": ["alpaca", "interactive_brokers"],
                "algorithms": ["twap", "vwap", "implementation_shortfall"],
                "performance_target": "execution_quality"
            }
        )
        
        # Register systems with cascade manager
        self.cascade_manager.register_marl_system(
            system_id="strategic",
            system_name="Strategic MARL System",
            input_callback=self._strategic_processor,
            output_callback=self._strategic_output_handler
        )
        
        self.cascade_manager.register_marl_system(
            system_id="tactical",
            system_name="Tactical MARL System",
            input_callback=self._tactical_processor,
            output_callback=self._tactical_output_handler
        )
        
        self.cascade_manager.register_marl_system(
            system_id="risk",
            system_name="Risk Management MARL System",
            input_callback=self._risk_processor,
            output_callback=self._risk_output_handler
        )
        
        self.cascade_manager.register_marl_system(
            system_id="execution",
            system_name="Execution MARL System",
            input_callback=self._execution_processor,
            output_callback=self._execution_output_handler
        )
        
        # Setup performance monitoring
        self.performance_monitor_cascade.add_alert_handler(self._handle_performance_alert)
        
        # Setup validation framework
        self.validation_framework.add_alert_handler(self._handle_validation_alert)
        
        # Setup emergency protocols
        self.emergency_protocols.add_emergency_contact(
            name="System Administrator",
            contact_info="admin@grandmodel.com",
            contact_type="email"
        )
        
        self.emergency_protocols.add_notification_handler(self._handle_emergency_notification)
        
        self.logger.info("Demo systems setup complete")
        
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration of cascade system"""
        self.logger.info("Starting comprehensive cascade demonstration")
        
        # Setup demo systems
        self.setup_demo_systems()
        
        # Run demo scenarios
        self._run_normal_flow_demo()
        self._run_coordination_demo()
        self._run_performance_monitoring_demo()
        self._run_validation_demo()
        self._run_emergency_scenario_demo()
        
        # Generate final report
        self._generate_demo_report()
        
        self.logger.info("Comprehensive cascade demonstration complete")
        
    def _run_normal_flow_demo(self):
        """Demonstrate normal cascade flow"""
        self.logger.info("=== Normal Flow Demo ===")
        
        # Inject market data context
        market_context = {
            "symbol": "NQ",
            "timestamp": datetime.now().isoformat(),
            "price": 15750.25,
            "volume": 1000,
            "market_conditions": {
                "volatility": 0.15,
                "trend": "upward",
                "support": 15700.0,
                "resistance": 15800.0
            }
        }
        
        # Start performance tracking
        packet_id = self.cascade_manager.inject_superposition(
            packet_type=SuperpositionType.CONTEXT_UPDATE,
            data=market_context,
            source_system="market_data",
            priority=1
        )
        
        if packet_id:
            self.performance_monitor_cascade.track_packet_start(
                SuperpositionPacket(
                    packet_id=packet_id,
                    packet_type=SuperpositionType.CONTEXT_UPDATE,
                    source_system="market_data",
                    target_system="strategic",
                    timestamp=datetime.now(),
                    data=market_context,
                    context={}
                )
            )
            
        # Allow processing time
        time.sleep(2)
        
        # Check cascade status
        status = self.cascade_manager.get_cascade_status()
        self.logger.info(f"Cascade status: {status['state']}")
        self.logger.info(f"Packets processed: {status['metrics']['total_packets_processed']}")
        
        self.demo_metrics["packets_processed"] += 1
        
    def _run_coordination_demo(self):
        """Demonstrate coordination between systems"""
        self.logger.info("=== Coordination Demo ===")
        
        # Coordinate a decision across systems
        decision_result = self.coordination_engine.coordinate_decision(
            requesting_system="strategic",
            decision_type="entry_signal",
            decision_data={
                "signal": "buy",
                "strength": 0.8,
                "confidence": 0.75,
                "target_position": 100
            },
            affected_systems=["tactical", "risk", "execution"],
            priority=2
        )
        
        if decision_result:
            self.logger.info(f"Coordination result: {decision_result['success']}")
            if decision_result.get("conflicts_resolved"):
                self.logger.info(f"Conflicts resolved: {decision_result['conflicts_resolved']}")
                
        # Test synchronization
        sync_results = self.coordination_engine.synchronize_systems("fast_sync")
        self.logger.info(f"Synchronization results: {sync_results}")
        
        # Get coordination status
        coord_status = self.coordination_engine.get_coordination_status()
        self.logger.info(f"Coordination state: {coord_status['coordination_state']}")
        self.logger.info(f"Active systems: {coord_status['active_systems']}")
        
        self.demo_metrics["coordination_events"] += 1
        
    def _run_performance_monitoring_demo(self):
        """Demonstrate performance monitoring capabilities"""
        self.logger.info("=== Performance Monitoring Demo ===")
        
        # Get real-time metrics
        real_time_metrics = self.performance_monitor_cascade.get_real_time_metrics()
        self.logger.info(f"End-to-end latency: {real_time_metrics['end_to_end_latency']['average']:.2f}ms")
        self.logger.info(f"Throughput: {real_time_metrics['throughput']:.2f} packets/sec")
        self.logger.info(f"Performance health: {real_time_metrics['performance_health']:.1f}%")
        
        # Generate performance report
        performance_report = self.performance_monitor_cascade.generate_performance_report()
        self.logger.info(f"Performance report generated: {performance_report.report_id}")
        self.logger.info(f"Performance score: {performance_report.performance_score:.1f}%")
        
        # Test performance under load
        self._simulate_performance_load()
        
        self.demo_metrics["average_latency"] = real_time_metrics['end_to_end_latency']['average']
        
    def _run_validation_demo(self):
        """Demonstrate validation framework"""
        self.logger.info("=== Validation Demo ===")
        
        # Run comprehensive validation
        validation_report = self.validation_framework.run_comprehensive_validation()
        self.logger.info(f"Validation report: {validation_report.report_id}")
        self.logger.info(f"Total validations: {validation_report.summary['total_validations']}")
        self.logger.info(f"Critical issues: {validation_report.summary['critical']}")
        self.logger.info(f"Cascade integrity: {validation_report.cascade_integrity_score:.1f}%")
        
        # Test validation with problematic data
        self._test_validation_with_bad_data()
        
        # Get validation status
        validation_status = self.validation_framework.get_validation_status()
        self.logger.info(f"Validation metrics: {validation_status['metrics']}")
        
        self.demo_metrics["validation_checks"] += validation_report.summary['total_validations']
        
    def _run_emergency_scenario_demo(self):
        """Demonstrate emergency protocols"""
        self.logger.info("=== Emergency Scenario Demo ===")
        
        # Simulate system failure
        emergency_id = self.emergency_protocols.declare_emergency(
            emergency_type=EmergencyType.SYSTEM_FAILURE,
            emergency_level=EmergencyLevel.WARNING,
            source_system="tactical",
            affected_systems=["tactical"],
            description="Simulated system failure for demo",
            context={"reason": "demo_scenario"}
        )
        
        self.logger.info(f"Emergency declared: {emergency_id}")
        
        # Allow emergency protocols to respond
        time.sleep(3)
        
        # Check emergency status
        emergency_status = self.emergency_protocols.get_emergency_status()
        self.logger.info(f"Active emergencies: {len(emergency_status.active_emergencies)}")
        self.logger.info(f"Recovery plans: {len(emergency_status.recovery_plans_active)}")
        
        # Simulate performance degradation
        self.emergency_protocols.declare_emergency(
            emergency_type=EmergencyType.PERFORMANCE_DEGRADATION,
            emergency_level=EmergencyLevel.CAUTION,
            source_system="cascade_manager",
            affected_systems=["strategic", "tactical", "risk", "execution"],
            description="Simulated performance degradation",
            context={"latency_increase": 50}
        )
        
        self.demo_metrics["emergency_events"] += 2
        
    def _simulate_performance_load(self):
        """Simulate performance load for testing"""
        self.logger.info("Simulating performance load...")
        
        # Inject multiple packets rapidly
        for i in range(10):
            packet_id = self.cascade_manager.inject_superposition(
                packet_type=SuperpositionType.STRATEGIC_SIGNAL,
                data={"signal": f"test_signal_{i}", "strength": 0.5},
                source_system="load_test",
                priority=1
            )
            
            if packet_id:
                self.performance_monitor_cascade.track_packet_start(
                    SuperpositionPacket(
                        packet_id=packet_id,
                        packet_type=SuperpositionType.STRATEGIC_SIGNAL,
                        source_system="load_test",
                        target_system="tactical",
                        timestamp=datetime.now(),
                        data={"signal": f"test_signal_{i}", "strength": 0.5},
                        context={}
                    )
                )
                
            time.sleep(0.1)  # 100ms between packets
            
        self.logger.info("Performance load simulation complete")
        
    def _test_validation_with_bad_data(self):
        """Test validation with problematic data"""
        self.logger.info("Testing validation with bad data...")
        
        # Create packet with invalid data
        bad_packet = SuperpositionPacket(
            packet_id="bad_packet_test",
            packet_type=SuperpositionType.STRATEGIC_SIGNAL,
            source_system="test",
            target_system="tactical",
            timestamp=datetime.now(),
            data={},  # Empty data - should trigger validation error
            context={}
        )
        
        # Validate the bad packet
        validation_results = self.validation_framework.validate_packet(bad_packet)
        self.logger.info(f"Validation results for bad packet: {len(validation_results)} issues found")
        
        for result in validation_results:
            if result.level.value in ["ERROR", "CRITICAL"]:
                self.logger.warning(f"Validation issue: {result.message}")
                
    def _generate_demo_report(self):
        """Generate final demo report"""
        self.logger.info("=== Demo Report ===")
        
        # Get final system status
        cascade_status = self.cascade_manager.get_cascade_status()
        coord_status = self.coordination_engine.get_coordination_status()
        perf_metrics = self.performance_monitor_cascade.get_real_time_metrics()
        validation_status = self.validation_framework.get_validation_status()
        emergency_status = self.emergency_protocols.get_emergency_status()
        
        report = {
            "demo_summary": {
                "timestamp": datetime.now().isoformat(),
                "duration": "~5 minutes",
                "scenarios_tested": 5,
                "metrics": self.demo_metrics
            },
            "cascade_manager": {
                "state": cascade_status["state"],
                "systems_registered": len(cascade_status["systems"]),
                "health_score": cascade_status["metrics"]["cascade_health_score"],
                "total_packets": cascade_status["metrics"]["total_packets_processed"],
                "success_rate": cascade_status["metrics"]["success_rate"]
            },
            "coordination_engine": {
                "state": coord_status["coordination_state"],
                "active_systems": coord_status["active_systems"],
                "successful_coordinations": coord_status["metrics"]["successful_coordinations"],
                "coordination_efficiency": coord_status["metrics"]["coordination_efficiency"]
            },
            "performance_monitor": {
                "average_latency_ms": perf_metrics["end_to_end_latency"]["average"],
                "p95_latency_ms": perf_metrics["end_to_end_latency"]["p95"],
                "throughput": perf_metrics["throughput"],
                "performance_health": perf_metrics["performance_health"],
                "target_met": perf_metrics["end_to_end_latency"]["average"] <= 100
            },
            "validation_framework": {
                "total_validations": validation_status["metrics"]["total_validations"],
                "passed_validations": validation_status["metrics"]["passed_validations"],
                "failed_validations": validation_status["metrics"]["failed_validations"],
                "system_health_score": validation_status["metrics"]["system_health_score"]
            },
            "emergency_protocols": {
                "active_emergencies": len(emergency_status.active_emergencies),
                "total_emergencies": self.emergency_protocols.get_emergency_metrics()["total_emergencies"],
                "successful_recoveries": self.emergency_protocols.get_emergency_metrics()["successful_recoveries"],
                "system_availability": self.emergency_protocols.get_emergency_metrics()["system_availability"]
            }
        }
        
        self.logger.info("Demo Report:")
        for section, data in report.items():
            self.logger.info(f"  {section}: {data}")
            
        return report
        
    # System processors (mock implementations)
    def _strategic_processor(self, packet: SuperpositionPacket) -> SuperpositionPacket:
        """Mock strategic system processor"""
        self.performance_monitor_cascade.track_system_entry(packet.packet_id, "strategic")
        
        # Simulate processing delay
        time.sleep(0.02)  # 20ms processing time
        
        # Transform packet
        result_packet = SuperpositionPacket(
            packet_id=f"strategic_{packet.packet_id}",
            packet_type=SuperpositionType.STRATEGIC_SIGNAL,
            source_system="strategic",
            target_system="tactical",
            timestamp=datetime.now(),
            data={
                "signal": "buy",
                "strength": 0.8,
                "confidence": 0.75,
                "source_data": packet.data
            },
            context={"processed_by": "strategic"}
        )
        
        self.performance_monitor_cascade.track_system_exit(packet.packet_id, "strategic", True)
        return result_packet
        
    def _tactical_processor(self, packet: SuperpositionPacket) -> SuperpositionPacket:
        """Mock tactical system processor"""
        self.performance_monitor_cascade.track_system_entry(packet.packet_id, "tactical")
        
        # Simulate processing delay
        time.sleep(0.015)  # 15ms processing time
        
        # Transform packet
        result_packet = SuperpositionPacket(
            packet_id=f"tactical_{packet.packet_id}",
            packet_type=SuperpositionType.TACTICAL_SIGNAL,
            source_system="tactical",
            target_system="risk",
            timestamp=datetime.now(),
            data={
                "action": "enter_position",
                "size": 100,
                "timing": "immediate",
                "source_signal": packet.data
            },
            context={"processed_by": "tactical"}
        )
        
        self.performance_monitor_cascade.track_system_exit(packet.packet_id, "tactical", True)
        return result_packet
        
    def _risk_processor(self, packet: SuperpositionPacket) -> SuperpositionPacket:
        """Mock risk system processor"""
        self.performance_monitor_cascade.track_system_entry(packet.packet_id, "risk")
        
        # Simulate processing delay
        time.sleep(0.025)  # 25ms processing time
        
        # Transform packet
        result_packet = SuperpositionPacket(
            packet_id=f"risk_{packet.packet_id}",
            packet_type=SuperpositionType.RISK_ASSESSMENT,
            source_system="risk",
            target_system="execution",
            timestamp=datetime.now(),
            data={
                "risk_score": 0.3,
                "approved_size": 80,  # Reduced from 100
                "var_impact": 0.05,
                "source_action": packet.data
            },
            context={"processed_by": "risk"}
        )
        
        self.performance_monitor_cascade.track_system_exit(packet.packet_id, "risk", True)
        return result_packet
        
    def _execution_processor(self, packet: SuperpositionPacket) -> SuperpositionPacket:
        """Mock execution system processor"""
        self.performance_monitor_cascade.track_system_entry(packet.packet_id, "execution")
        
        # Simulate processing delay
        time.sleep(0.02)  # 20ms processing time
        
        # Transform packet
        result_packet = SuperpositionPacket(
            packet_id=f"execution_{packet.packet_id}",
            packet_type=SuperpositionType.EXECUTION_PLAN,
            source_system="execution",
            target_system="complete",
            timestamp=datetime.now(),
            data={
                "orders": [
                    {
                        "symbol": "NQ",
                        "quantity": 80,
                        "order_type": "market",
                        "venue": "alpaca"
                    }
                ],
                "execution_quality": 0.95,
                "source_assessment": packet.data
            },
            context={"processed_by": "execution"}
        )
        
        self.performance_monitor_cascade.track_system_exit(packet.packet_id, "execution", True)
        self.performance_monitor_cascade.track_packet_completion(packet.packet_id, True)
        return result_packet
        
    # Output handlers
    def _strategic_output_handler(self, packet: SuperpositionPacket) -> None:
        """Handle strategic system output"""
        self.logger.debug(f"Strategic output: {packet.packet_id}")
        
    def _tactical_output_handler(self, packet: SuperpositionPacket) -> None:
        """Handle tactical system output"""
        self.logger.debug(f"Tactical output: {packet.packet_id}")
        
    def _risk_output_handler(self, packet: SuperpositionPacket) -> None:
        """Handle risk system output"""
        self.logger.debug(f"Risk output: {packet.packet_id}")
        
    def _execution_output_handler(self, packet: SuperpositionPacket) -> None:
        """Handle execution system output"""
        self.logger.debug(f"Execution output: {packet.packet_id}")
        
    # Alert handlers
    def _handle_performance_alert(self, alert: Dict[str, Any]) -> None:
        """Handle performance alerts"""
        self.logger.warning(f"Performance alert: {alert['message']}")
        
    def _handle_validation_alert(self, alert: Any) -> None:
        """Handle validation alerts"""
        self.logger.warning(f"Validation alert: {alert.message}")
        
    def _handle_emergency_notification(self, emergency_event: Any, contact: Dict[str, str]) -> None:
        """Handle emergency notifications"""
        self.logger.critical(f"Emergency notification to {contact['name']}: {emergency_event.description}")


def main():
    """Main demonstration function"""
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run demo
    demo = CascadeIntegrationDemo()
    demo.run_comprehensive_demo()
    
    # Keep demo running for a bit to see background processes
    print("Demo running... Press Ctrl+C to stop")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("Demo stopped")
        
    # Cleanup
    demo.cascade_manager.shutdown()
    demo.coordination_engine.shutdown()
    demo.performance_monitor_cascade.shutdown()
    demo.validation_framework.shutdown()
    demo.emergency_protocols.shutdown()
    
    print("Demo complete")


if __name__ == "__main__":
    main()