#!/usr/bin/env python3
"""
Universal Superposition System Validation Script.

This script demonstrates the complete validation and monitoring framework
for the universal superposition system, showing all components working
together to ensure system integrity and performance.
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import system components
from src.validation.universal_superposition_validator import (
    UniversalSuperpositionValidator,
    SuperpositionState,
    ValidationLevel
)
from src.monitoring.superposition_performance_monitor import (
    SuperpositionPerformanceMonitor,
    MonitoringMode
)
from src.validation.cascade_integrity_checker import (
    CascadeIntegrityChecker,
    CascadeLevel,
    CommunicationChannel
)
from src.monitoring.superposition_quality_metrics import (
    SuperpositionQualityMetrics,
    SuperpositionOutput
)


class SuperpositionSystemValidator:
    """Complete validation system for universal superposition framework."""
    
    def __init__(self):
        """Initialize all validation and monitoring components."""
        logger.info("🚀 Initializing Universal Superposition System Validator")
        
        # Initialize components
        self.validator = UniversalSuperpositionValidator(
            tolerance=1e-6,
            validation_level=ValidationLevel.COMPREHENSIVE,
            performance_target_ms=5.0
        )
        
        self.performance_monitor = SuperpositionPerformanceMonitor(
            target_latency_ms=5.0,
            monitoring_mode=MonitoringMode.REALTIME,
            alert_threshold_ms=4.0,
            critical_threshold_ms=8.0
        )
        
        self.cascade_checker = CascadeIntegrityChecker(
            max_cascade_latency_ms=10.0,
            heartbeat_interval_ms=1000.0,
            dependency_timeout_ms=5000.0
        )
        
        self.quality_metrics = SuperpositionQualityMetrics(
            history_size=1000,
            consistency_threshold=0.8,
            coherence_threshold=0.7,
            calibration_threshold=0.75
        )
        
        # Setup system components
        self._setup_system()
        
        logger.info("✅ All components initialized successfully")
    
    def _setup_system(self):
        """Set up the complete system architecture."""
        logger.info("🔧 Setting up system architecture")
        
        # Register components in performance monitor
        self.performance_monitor.register_component(
            "superposition_validator",
            expected_latency_ms=5.0,
            critical_path=True,
            optimization_hints={'parallelizable': True, 'gpu_accelerated': True}
        )
        
        self.performance_monitor.register_component(
            "cascade_integrity_checker",
            expected_latency_ms=8.0,
            critical_path=True,
            optimization_hints={'cacheable': True}
        )
        
        self.performance_monitor.register_component(
            "quality_metrics",
            expected_latency_ms=3.0,
            critical_path=False,
            optimization_hints={'parallelizable': True}
        )
        
        # Register MARL agents in cascade checker
        self.cascade_checker.register_agent(
            "strategic_agent",
            CascadeLevel.STRATEGIC,
            expected_latency_ms=5.0,
            dependencies=[],
            communication_channels={'tactical': CommunicationChannel.HIERARCHICAL}
        )
        
        self.cascade_checker.register_agent(
            "tactical_agent",
            CascadeLevel.TACTICAL,
            expected_latency_ms=3.0,
            dependencies=["strategic_agent"],
            communication_channels={
                'strategic': CommunicationChannel.HIERARCHICAL,
                'execution': CommunicationChannel.DIRECT
            }
        )
        
        self.cascade_checker.register_agent(
            "execution_agent",
            CascadeLevel.EXECUTION,
            expected_latency_ms=2.0,
            dependencies=["tactical_agent"],
            communication_channels={'tactical': CommunicationChannel.DIRECT}
        )
        
        self.cascade_checker.register_agent(
            "risk_management_agent",
            CascadeLevel.RISK_MANAGEMENT,
            expected_latency_ms=4.0,
            dependencies=["strategic_agent", "tactical_agent"],
            communication_channels={
                'strategic': CommunicationChannel.BROADCAST,
                'tactical': CommunicationChannel.BROADCAST
            }
        )
        
        logger.info("🏗️ System architecture setup complete")
    
    def create_sample_superposition_state(self) -> SuperpositionState:
        """Create a realistic superposition state for testing."""
        n_agents = 4
        
        # Create normalized amplitudes
        raw_amplitudes = np.random.dirichlet([2, 3, 2, 1])  # Weighted toward tactical
        amplitudes = raw_amplitudes / np.linalg.norm(raw_amplitudes)
        
        # Create phases with some structure
        phases = np.array([0.0, np.pi/6, np.pi/3, np.pi/2])
        
        # Agent contributions (normalized)
        agent_contributions = {
            'strategic_agent': float(amplitudes[0]),
            'tactical_agent': float(amplitudes[1]),
            'execution_agent': float(amplitudes[2]),
            'risk_management_agent': float(amplitudes[3])
        }
        
        # Confidence scores (varying by agent type)
        confidence_scores = {
            'strategic_agent': 0.85,
            'tactical_agent': 0.75,
            'execution_agent': 0.65,
            'risk_management_agent': 0.90
        }
        
        # Create coherence matrix (symmetric, positive definite)
        base_coherence = np.random.random((n_agents, n_agents)) * 0.3
        coherence_matrix = (base_coherence + base_coherence.T) / 2
        np.fill_diagonal(coherence_matrix, 1.0)
        
        # Ensure positive definite
        eigenvals = np.linalg.eigvals(coherence_matrix)
        if np.min(eigenvals) < 0.1:
            coherence_matrix += np.eye(n_agents) * (0.1 - np.min(eigenvals))
        
        return SuperpositionState(
            amplitudes=amplitudes,
            phases=phases,
            agent_contributions=agent_contributions,
            confidence_scores=confidence_scores,
            coherence_matrix=coherence_matrix
        )
    
    def create_superposition_output(self, state: SuperpositionState) -> SuperpositionOutput:
        """Create superposition output from state."""
        # Calculate ensemble confidence
        ensemble_confidence = np.mean(list(state.confidence_scores.values()))
        
        # Calculate decision value (weighted by confidence)
        decision_value = sum(
            contrib * state.confidence_scores[agent_id]
            for agent_id, contrib in state.agent_contributions.items()
        ) / len(state.agent_contributions)
        
        return SuperpositionOutput(
            timestamp=datetime.now(),
            decision_probabilities=state.amplitudes,
            agent_contributions=state.agent_contributions,
            confidence_scores=state.confidence_scores,
            ensemble_confidence=ensemble_confidence,
            decision_value=decision_value
        )
    
    def simulate_agent_heartbeats(self):
        """Simulate agent heartbeats for cascade monitoring."""
        current_time = datetime.now()
        for agent_id in self.cascade_checker.agents:
            self.cascade_checker.update_agent_heartbeat(agent_id, current_time)
    
    def simulate_agent_communication(self):
        """Simulate communication between agents."""
        communications = [
            ("strategic_agent", "tactical_agent", CommunicationChannel.HIERARCHICAL, 2.5),
            ("tactical_agent", "execution_agent", CommunicationChannel.DIRECT, 1.8),
            ("strategic_agent", "risk_management_agent", CommunicationChannel.BROADCAST, 3.2),
            ("tactical_agent", "risk_management_agent", CommunicationChannel.BROADCAST, 2.1)
        ]
        
        for from_agent, to_agent, channel, latency in communications:
            self.cascade_checker.log_communication(
                from_agent=from_agent,
                to_agent=to_agent,
                channel=channel,
                latency_ms=latency,
                success=True
            )
    
    def validate_single_superposition(self, state: SuperpositionState) -> Dict[str, Any]:
        """Validate a single superposition state through the complete pipeline."""
        results = {}
        
        # 1. Validate superposition state
        with self.performance_monitor.measure_performance("superposition_validator", "validation"):
            validation_results = self.validator.validate_superposition_state(state)
            results['validation'] = validation_results
        
        # 2. Check cascade integrity
        with self.performance_monitor.measure_performance("cascade_integrity_checker", "integrity_check"):
            integrity_report = self.cascade_checker.check_cascade_integrity()
            results['cascade_integrity'] = integrity_report
        
        # 3. Assess quality
        output = self.create_superposition_output(state)
        with self.performance_monitor.measure_performance("quality_metrics", "quality_assessment"):
            self.quality_metrics.record_output(output)
            results['quality_output'] = output
        
        return results
    
    def run_comprehensive_validation(self, num_iterations: int = 10) -> Dict[str, Any]:
        """Run comprehensive validation across multiple iterations."""
        logger.info(f"🔍 Starting comprehensive validation with {num_iterations} iterations")
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        self.cascade_checker.start_monitoring()
        
        try:
            all_results = []
            start_time = time.perf_counter()
            
            for i in range(num_iterations):
                logger.info(f"🔄 Running iteration {i+1}/{num_iterations}")
                
                # Create realistic superposition state
                state = self.create_sample_superposition_state()
                
                # Simulate system activity
                self.simulate_agent_heartbeats()
                self.simulate_agent_communication()
                
                # Record cascade performance
                for agent_id, agent in self.cascade_checker.agents.items():
                    self.cascade_checker.record_agent_performance(
                        agent_id, 
                        agent.expected_latency_ms * (0.8 + np.random.random() * 0.4)
                    )
                
                # Validate through complete pipeline
                iteration_results = self.validate_single_superposition(state)
                all_results.append(iteration_results)
                
                # Brief pause between iterations
                time.sleep(0.1)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Generate comprehensive report
            report = self._generate_comprehensive_report(all_results, total_time)
            
            return report
            
        finally:
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
            self.cascade_checker.stop_monitoring()
    
    def _generate_comprehensive_report(self, all_results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        logger.info("📊 Generating comprehensive validation report")
        
        # Performance summary
        performance_summary = self.performance_monitor.get_performance_summary()
        
        # Cascade topology
        cascade_topology = self.cascade_checker.get_cascade_topology()
        
        # Quality summary
        quality_summary = self.quality_metrics.get_quality_summary()
        
        # Final quality assessment
        final_quality_report = self.quality_metrics.assess_quality()
        
        # Validation statistics
        validation_stats = self._calculate_validation_statistics(all_results)
        
        # System health assessment
        system_health = self._assess_system_health(performance_summary, cascade_topology, quality_summary)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_configuration': {
                'iterations': len(all_results),
                'total_time_seconds': total_time,
                'avg_time_per_iteration_ms': (total_time / len(all_results)) * 1000
            },
            'performance_summary': performance_summary,
            'cascade_topology': cascade_topology,
            'quality_summary': quality_summary,
            'validation_statistics': validation_stats,
            'system_health': system_health,
            'final_quality_report': final_quality_report.to_dict(),
            'recommendations': self._generate_recommendations(system_health, validation_stats)
        }
        
        return report
    
    def _calculate_validation_statistics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from validation results."""
        total_validations = 0
        passed_validations = 0
        validation_scores = []
        
        for result in all_results:
            if 'validation' in result:
                for validation_result in result['validation'].values():
                    total_validations += 1
                    if validation_result.passed:
                        passed_validations += 1
                    validation_scores.append(validation_result.score)
        
        cascade_statuses = []
        cascade_latencies = []
        
        for result in all_results:
            if 'cascade_integrity' in result:
                cascade_statuses.append(result['cascade_integrity'].overall_status.value)
                cascade_latencies.append(result['cascade_integrity'].cascade_latency_ms)
        
        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'validation_pass_rate': passed_validations / max(1, total_validations),
            'avg_validation_score': np.mean(validation_scores) if validation_scores else 0,
            'cascade_health_rate': cascade_statuses.count('healthy') / max(1, len(cascade_statuses)),
            'avg_cascade_latency_ms': np.mean(cascade_latencies) if cascade_latencies else 0,
            'max_cascade_latency_ms': np.max(cascade_latencies) if cascade_latencies else 0
        }
    
    def _assess_system_health(self, performance_summary: Dict[str, Any], cascade_topology: Dict[str, Any], quality_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health."""
        health_score = 100.0
        issues = []
        
        # Performance health
        if performance_summary.get('performance_violation_rate', 0) > 0.1:
            health_score -= 20
            issues.append("High performance violation rate")
        
        # Cascade health
        if not cascade_topology.get('is_acyclic', True):
            health_score -= 30
            issues.append("Circular dependencies in cascade")
        
        if cascade_topology.get('critical_path_length', 0) > 15:
            health_score -= 15
            issues.append("Critical path too long")
        
        # Quality health
        if quality_summary.get('status') != 'active':
            health_score -= 25
            issues.append("Quality assessment system inactive")
        
        # Determine health level
        if health_score >= 90:
            health_level = "excellent"
        elif health_score >= 75:
            health_level = "good"
        elif health_score >= 60:
            health_level = "acceptable"
        elif health_score >= 40:
            health_level = "poor"
        else:
            health_level = "critical"
        
        return {
            'health_score': health_score,
            'health_level': health_level,
            'issues': issues,
            'monitoring_active': performance_summary.get('monitoring_active', False)
        }
    
    def _generate_recommendations(self, system_health: Dict[str, Any], validation_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Performance recommendations
        if validation_stats.get('validation_pass_rate', 0) < 0.8:
            recommendations.append("Review validation thresholds - low pass rate detected")
        
        if validation_stats.get('avg_cascade_latency_ms', 0) > 8:
            recommendations.append("Optimize cascade performance - high latency detected")
        
        # Health recommendations
        if system_health['health_score'] < 80:
            recommendations.append("System health below optimal - investigate identified issues")
        
        # System-specific recommendations
        if system_health['health_level'] == 'critical':
            recommendations.append("URGENT: System requires immediate attention")
        
        if not system_health.get('monitoring_active', False):
            recommendations.append("Enable continuous monitoring for production deployment")
        
        return recommendations
    
    def export_validation_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Export validation report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"superposition_validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Validation report exported to {filename}")
        return filename
    
    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable summary of validation results."""
        print("\n" + "="*80)
        print("🎯 UNIVERSAL SUPERPOSITION SYSTEM VALIDATION SUMMARY")
        print("="*80)
        
        # Test configuration
        config = report['test_configuration']
        print(f"📋 Test Configuration:")
        print(f"   • Iterations: {config['iterations']}")
        print(f"   • Total Time: {config['total_time_seconds']:.2f}s")
        print(f"   • Avg Time/Iteration: {config['avg_time_per_iteration_ms']:.2f}ms")
        
        # System health
        health = report['system_health']
        print(f"\n🏥 System Health: {health['health_level'].upper()} ({health['health_score']:.1f}/100)")
        if health['issues']:
            print(f"   ⚠️  Issues: {', '.join(health['issues'])}")
        
        # Performance summary
        perf = report['performance_summary']
        print(f"\n⚡ Performance Summary:")
        print(f"   • Total Tests: {perf.get('total_tests', 0)}")
        print(f"   • Success Rate: {perf.get('success_rate', 0):.1%}")
        print(f"   • Performance Violations: {perf.get('performance_violations', 0)}")
        
        # Validation statistics
        val_stats = report['validation_statistics']
        print(f"\n✅ Validation Statistics:")
        print(f"   • Total Validations: {val_stats['total_validations']}")
        print(f"   • Pass Rate: {val_stats['validation_pass_rate']:.1%}")
        print(f"   • Avg Score: {val_stats['avg_validation_score']:.3f}")
        print(f"   • Cascade Health Rate: {val_stats['cascade_health_rate']:.1%}")
        print(f"   • Avg Cascade Latency: {val_stats['avg_cascade_latency_ms']:.2f}ms")
        
        # Quality summary
        quality = report['quality_summary']
        if quality.get('status') == 'active':
            print(f"\n🎨 Quality Summary:")
            print(f"   • Overall Quality: {quality['overall_quality']}")
            print(f"   • Trend: {quality['trend_direction']}")
            print(f"   • Outputs Assessed: {quality['total_outputs_assessed']}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        print("✨ VALIDATION COMPLETE")
        print("="*80)


def main():
    """Main execution function."""
    print("🚀 Universal Superposition System Validation")
    print("=" * 50)
    
    # Initialize validation system
    validator = SuperpositionSystemValidator()
    
    # Run comprehensive validation
    report = validator.run_comprehensive_validation(num_iterations=5)
    
    # Export report
    filename = validator.export_validation_report(report)
    
    # Print summary
    validator.print_summary(report)
    
    # Final status
    health_level = report['system_health']['health_level']
    if health_level in ['excellent', 'good']:
        print("🎉 System validation PASSED - Ready for production!")
    elif health_level == 'acceptable':
        print("⚠️  System validation PASSED with warnings - Review recommendations")
    else:
        print("❌ System validation FAILED - Critical issues detected")
    
    print(f"\n📄 Detailed report saved to: {filename}")


if __name__ == "__main__":
    main()