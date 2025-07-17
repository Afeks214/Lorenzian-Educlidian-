#!/usr/bin/env python3
"""
🚨 AGENT GAMMA MISSION - ADVANCED MARL ATTACK INTEGRATION FRAMEWORK
Advanced MARL Attack Development: Integration and Orchestration

This module provides a unified interface for all advanced MARL attack modules
and integrates them with the existing adversarial test framework.

Key Components:
1. Attack Module Registry
2. Unified Attack Interface
3. Attack Orchestration System
4. Integration with Existing Framework
5. Attack Analytics and Reporting

MISSION OBJECTIVE: Seamless integration of all attack modules with >80% success rate
"""

from .marl_coordination_attack import (
    MARLCoordinationAttacker,
    CoordinationAttackResult,
    AttackType as CoordAttackType
)

from .temporal_sequence_attack import (
    TemporalSequenceAttacker,
    TemporalAttackResult,
    TemporalAttackType
)

from .policy_gradient_attack import (
    PolicyGradientAttacker,
    PolicyGradientAttackResult,
    PolicyAttackType
)

from .regime_transition_attack import (
    RegimeTransitionAttacker,
    RegimeAttackResult,
    RegimeAttackType
)

from .advanced_marl_scenarios import (
    AdvancedMARLScenarioAttacker,
    ScenarioAttackResult,
    AttackScenario
)

# Export all attack modules for easy access
__all__ = [
    'MARLCoordinationAttacker',
    'TemporalSequenceAttacker', 
    'PolicyGradientAttacker',
    'RegimeTransitionAttacker',
    'AdvancedMARLScenarioAttacker',
    'CoordinationAttackResult',
    'TemporalAttackResult',
    'PolicyGradientAttackResult',
    'RegimeAttackResult',
    'ScenarioAttackResult',
    'CoordAttackType',
    'TemporalAttackType',
    'PolicyAttackType',
    'RegimeAttackType',
    'AttackScenario',
    'UnifiedAttackOrchestrator'
]

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time

@dataclass
class UnifiedAttackResult:
    """Unified result structure for all attack types."""
    attack_module: str
    attack_type: str
    success: bool
    confidence: float
    disruption_score: float
    execution_time_ms: float
    original_result: Any
    metadata: Dict[str, Any]
    timestamp: datetime

class AttackModule(Enum):
    """Available attack modules."""
    COORDINATION = "coordination"
    TEMPORAL = "temporal"
    POLICY = "policy"
    REGIME = "regime"
    SCENARIO = "scenario"

class UnifiedAttackOrchestrator:
    """
    Unified Attack Orchestrator.
    
    This class provides a single interface for executing all types of
    advanced MARL attacks and integrating with the existing framework.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the Unified Attack Orchestrator.
        
        Args:
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        self.device = device
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize all attack modules
        self.coordination_attacker = MARLCoordinationAttacker(device)
        self.temporal_attacker = TemporalSequenceAttacker(device)
        self.policy_attacker = PolicyGradientAttacker(device)
        self.regime_attacker = RegimeTransitionAttacker(device)
        self.scenario_attacker = AdvancedMARLScenarioAttacker(device)
        
        # Attack history and analytics
        self.unified_history = []
        self.module_success_rates = {module: 0.0 for module in AttackModule}
        self.orchestrator_metrics = {
            'total_attacks': 0,
            'successful_attacks': 0,
            'avg_disruption_score': 0.0,
            'avg_execution_time_ms': 0.0,
            'modules_used': set()
        }
        
        # Attack module registry
        self.attack_registry = {
            AttackModule.COORDINATION: {
                'attacker': self.coordination_attacker,
                'methods': {
                    'consensus_disruption': self.coordination_attacker.generate_consensus_disruption_attack,
                    'communication_jamming': self.coordination_attacker.generate_communication_jamming_attack,
                    'gating_exploitation': self.coordination_attacker.generate_gating_exploitation_attack,
                    'timing_attack': self.coordination_attacker.generate_timing_attack,
                    'desynchronization': self.coordination_attacker.generate_desynchronization_attack
                }
            },
            AttackModule.TEMPORAL: {
                'attacker': self.temporal_attacker,
                'methods': {
                    'correlation_poisoning': self.temporal_attacker.generate_correlation_poisoning_attack,
                    'pattern_disruption': self.temporal_attacker.generate_pattern_disruption_attack,
                    'memory_exploitation': self.temporal_attacker.generate_memory_exploitation_attack,
                    'timeframe_desync': self.temporal_attacker.generate_timeframe_desync_attack,
                    'gradient_attack': self.temporal_attacker.generate_gradient_attack
                }
            },
            AttackModule.POLICY: {
                'attacker': self.policy_attacker,
                'methods': {
                    'fgsm_policy': self.policy_attacker.generate_fgsm_policy_attack,
                    'pgd_action_space': self.policy_attacker.generate_pgd_action_space_attack,
                    'reward_poisoning': self.policy_attacker.generate_reward_poisoning_attack,
                    'gradient_reversal': self.policy_attacker.generate_gradient_reversal_attack,
                    'boundary_attack': self.policy_attacker.generate_boundary_attack
                }
            },
            AttackModule.REGIME: {
                'attacker': self.regime_attacker,
                'methods': {
                    'false_bull_signal': self.regime_attacker.generate_false_bull_signal_attack,
                    'false_bear_signal': self.regime_attacker.generate_false_bear_signal_attack,
                    'transition_confusion': self.regime_attacker.generate_transition_confusion_attack,
                    'volatility_manipulation': self.regime_attacker.generate_volatility_manipulation_attack,
                    'mmd_poisoning': self.regime_attacker.generate_mmd_poisoning_attack
                }
            },
            AttackModule.SCENARIO: {
                'attacker': self.scenario_attacker,
                'methods': {
                    'bull_trap_coordination': self.scenario_attacker.execute_bull_trap_coordination_attack,
                    'whipsaw_multi_agent': self.scenario_attacker.execute_whipsaw_multi_agent_attack,
                    'fake_breakout_manipulation': self.scenario_attacker.execute_fake_breakout_manipulation_attack,
                    'coordinated_correlation_gaming': self.scenario_attacker.execute_coordinated_correlation_gaming_attack
                }
            }
        }
        
        self.logger.info(f"UnifiedAttackOrchestrator initialized with {len(self.attack_registry)} modules")
    
    def execute_attack(
        self,
        attack_module: Union[AttackModule, str],
        attack_method: str,
        *args,
        **kwargs
    ) -> UnifiedAttackResult:
        """
        Execute a specific attack using the unified interface.
        
        Args:
            attack_module: Attack module to use
            attack_method: Specific attack method to execute
            *args: Positional arguments for the attack method
            **kwargs: Keyword arguments for the attack method
            
        Returns:
            UnifiedAttackResult with standardized result format
        """
        start_time = time.time()
        
        # Convert string to enum if needed
        if isinstance(attack_module, str):
            attack_module = AttackModule(attack_module)
        
        # Get attack method
        if attack_module not in self.attack_registry:
            raise ValueError(f"Unknown attack module: {attack_module}")
        
        module_info = self.attack_registry[attack_module]
        if attack_method not in module_info['methods']:
            raise ValueError(f"Unknown attack method: {attack_method} for module: {attack_module}")
        
        attack_func = module_info['methods'][attack_method]
        
        # Execute attack
        try:
            self.logger.info(f"Executing {attack_module.value}.{attack_method}")
            
            # Handle different return patterns
            if attack_module == AttackModule.SCENARIO:
                # Scenario attacks return single result
                original_result = attack_func(*args, **kwargs)
                attack_payload = None
            else:
                # Other attacks return (payload, result)
                attack_payload, original_result = attack_func(*args, **kwargs)
            
            # Extract success and confidence
            success = original_result.success
            
            # Get confidence/disruption score based on result type
            if hasattr(original_result, 'coordination_disruption_score'):
                disruption_score = original_result.coordination_disruption_score
            elif hasattr(original_result, 'temporal_disruption_score'):
                disruption_score = original_result.temporal_disruption_score
            elif hasattr(original_result, 'policy_disruption_score'):
                disruption_score = original_result.policy_disruption_score
            elif hasattr(original_result, 'regime_disruption_score'):
                disruption_score = original_result.regime_disruption_score
            elif hasattr(original_result, 'scenario_effectiveness'):
                disruption_score = original_result.scenario_effectiveness
            else:
                disruption_score = original_result.confidence
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Create unified result
            unified_result = UnifiedAttackResult(
                attack_module=attack_module.value,
                attack_type=attack_method,
                success=success,
                confidence=disruption_score,
                disruption_score=disruption_score,
                execution_time_ms=execution_time_ms,
                original_result=original_result,
                metadata={
                    'attack_payload': attack_payload,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                },
                timestamp=datetime.now()
            )
            
            self._record_unified_result(unified_result, attack_module)
            
            self.logger.info(
                f"Attack completed: {attack_module.value}.{attack_method} - "
                f"success={success}, disruption={disruption_score:.3f}"
            )
            
            return unified_result
            
        except Exception as e:
            self.logger.error(f"Attack failed: {attack_module.value}.{attack_method} - {str(e)}")
            raise
    
    def execute_attack_sequence(
        self,
        attack_sequence: List[Dict[str, Any]],
        continue_on_failure: bool = True
    ) -> List[UnifiedAttackResult]:
        """
        Execute a sequence of attacks.
        
        Args:
            attack_sequence: List of attack specifications
            continue_on_failure: Whether to continue if an attack fails
            
        Returns:
            List of UnifiedAttackResult objects
        """
        results = []
        
        for i, attack_spec in enumerate(attack_sequence):
            try:
                attack_module = attack_spec['module']
                attack_method = attack_spec['method']
                args = attack_spec.get('args', [])
                kwargs = attack_spec.get('kwargs', {})
                
                result = self.execute_attack(attack_module, attack_method, *args, **kwargs)
                results.append(result)
                
                if not result.success and not continue_on_failure:
                    self.logger.warning(f"Attack sequence stopped at step {i+1} due to failure")
                    break
                    
            except Exception as e:
                self.logger.error(f"Attack sequence failed at step {i+1}: {str(e)}")
                if not continue_on_failure:
                    break
        
        return results
    
    def get_available_attacks(self) -> Dict[str, List[str]]:
        """Get all available attack modules and methods."""
        return {
            module.value: list(info['methods'].keys())
            for module, info in self.attack_registry.items()
        }
    
    def get_attack_analytics(self) -> Dict[str, Any]:
        """Get comprehensive attack analytics across all modules."""
        if not self.unified_history:
            return {'status': 'no_attacks_recorded'}
        
        recent_attacks = self.unified_history[-200:]  # Last 200 attacks
        
        # Calculate per-module analytics
        module_analytics = {}
        for module in AttackModule:
            module_attacks = [a for a in recent_attacks if a.attack_module == module.value]
            if module_attacks:
                module_analytics[module.value] = {
                    'total_attacks': len(module_attacks),
                    'success_rate': len([a for a in module_attacks if a.success]) / len(module_attacks),
                    'avg_disruption_score': np.mean([a.disruption_score for a in module_attacks]),
                    'avg_execution_time_ms': np.mean([a.execution_time_ms for a in module_attacks])
                }
        
        # Calculate overall analytics
        overall_analytics = {
            'total_attacks': len(self.unified_history),
            'recent_attacks': len(recent_attacks),
            'overall_success_rate': len([a for a in recent_attacks if a.success]) / len(recent_attacks),
            'avg_disruption_score': np.mean([a.disruption_score for a in recent_attacks]),
            'avg_execution_time_ms': np.mean([a.execution_time_ms for a in recent_attacks]),
            'modules_used': len(set(a.attack_module for a in recent_attacks))
        }
        
        # Get individual module analytics
        coordination_analytics = self.coordination_attacker.get_attack_analytics()
        temporal_analytics = self.temporal_attacker.get_attack_analytics()
        policy_analytics = self.policy_attacker.get_attack_analytics()
        regime_analytics = self.regime_attacker.get_attack_analytics()
        scenario_analytics = self.scenario_attacker.get_scenario_analytics()
        
        return {
            'unified_analytics': overall_analytics,
            'module_analytics': module_analytics,
            'module_success_rates': self.module_success_rates,
            'orchestrator_metrics': self.orchestrator_metrics,
            'individual_module_analytics': {
                'coordination': coordination_analytics,
                'temporal': temporal_analytics,
                'policy': policy_analytics,
                'regime': regime_analytics,
                'scenario': scenario_analytics
            }
        }
    
    def generate_attack_report(self) -> Dict[str, Any]:
        """Generate comprehensive attack report."""
        analytics = self.get_attack_analytics()
        
        if analytics.get('status') == 'no_attacks_recorded':
            return {'report': 'No attacks have been recorded yet.'}
        
        # Calculate success rates by attack type
        attack_type_performance = {}
        for attack in self.unified_history[-200:]:
            attack_key = f"{attack.attack_module}.{attack.attack_type}"
            if attack_key not in attack_type_performance:
                attack_type_performance[attack_key] = {'attempts': 0, 'successes': 0}
            
            attack_type_performance[attack_key]['attempts'] += 1
            if attack.success:
                attack_type_performance[attack_key]['successes'] += 1
        
        # Calculate success rates
        for key, stats in attack_type_performance.items():
            stats['success_rate'] = stats['successes'] / stats['attempts']
        
        # Find most and least effective attacks
        sorted_attacks = sorted(
            attack_type_performance.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        most_effective = sorted_attacks[:3] if sorted_attacks else []
        least_effective = sorted_attacks[-3:] if sorted_attacks else []
        
        report = {
            'summary': {
                'total_attacks': analytics['unified_analytics']['total_attacks'],
                'overall_success_rate': analytics['unified_analytics']['overall_success_rate'],
                'avg_disruption_score': analytics['unified_analytics']['avg_disruption_score'],
                'modules_tested': analytics['unified_analytics']['modules_used']
            },
            'module_performance': analytics['module_analytics'],
            'attack_effectiveness': {
                'most_effective': [
                    {'attack': attack, 'success_rate': stats['success_rate'], 'attempts': stats['attempts']}
                    for attack, stats in most_effective
                ],
                'least_effective': [
                    {'attack': attack, 'success_rate': stats['success_rate'], 'attempts': stats['attempts']}
                    for attack, stats in least_effective
                ]
            },
            'recommendations': self._generate_attack_recommendations(analytics),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_attack_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on attack performance."""
        recommendations = []
        
        # Check overall success rate
        overall_success = analytics['unified_analytics']['overall_success_rate']
        if overall_success < 0.5:
            recommendations.append("Overall attack success rate is below 50%. Consider increasing attack strength or targeting different vulnerabilities.")
        elif overall_success > 0.8:
            recommendations.append("High attack success rate detected. The system may need stronger defenses.")
        
        # Check module performance
        module_analytics = analytics.get('module_analytics', {})
        for module, stats in module_analytics.items():
            if stats['success_rate'] < 0.3:
                recommendations.append(f"{module.title()} attacks have low success rate ({stats['success_rate']:.1%}). Consider refining attack techniques.")
            elif stats['success_rate'] > 0.9:
                recommendations.append(f"{module.title()} attacks are highly effective ({stats['success_rate']:.1%}). Consider developing defenses.")
        
        # Check execution time
        avg_time = analytics['unified_analytics']['avg_execution_time_ms']
        if avg_time > 1000:
            recommendations.append("Average attack execution time is high. Consider optimizing attack algorithms.")
        
        # Check disruption scores
        avg_disruption = analytics['unified_analytics']['avg_disruption_score']
        if avg_disruption < 0.3:
            recommendations.append("Low disruption scores indicate attacks may not be significantly affecting the system.")
        elif avg_disruption > 0.8:
            recommendations.append("High disruption scores suggest attacks are very effective. System defenses may be insufficient.")
        
        return recommendations
    
    def _record_unified_result(self, result: UnifiedAttackResult, attack_module: AttackModule):
        """Record unified attack result for analytics."""
        self.unified_history.append(result)
        
        # Update metrics
        self.orchestrator_metrics['total_attacks'] += 1
        if result.success:
            self.orchestrator_metrics['successful_attacks'] += 1
        
        # Update module success rates
        module_attempts = len([r for r in self.unified_history if r.attack_module == attack_module.value])
        module_successes = len([r for r in self.unified_history if r.attack_module == attack_module.value and r.success])
        self.module_success_rates[attack_module] = module_successes / module_attempts
        
        # Update orchestrator metrics
        self.orchestrator_metrics['avg_disruption_score'] = np.mean([r.disruption_score for r in self.unified_history])
        self.orchestrator_metrics['avg_execution_time_ms'] = np.mean([r.execution_time_ms for r in self.unified_history])
        self.orchestrator_metrics['modules_used'].add(attack_module.value)
        
        # Keep history manageable
        if len(self.unified_history) > 2000:
            self.unified_history = self.unified_history[-1000:]

# Integration with existing framework
def integrate_with_existing_framework():
    """
    Integration function to connect with existing adversarial test framework.
    This function should be called to set up the integration.
    """
    try:
        # Import existing framework components
        import sys
        import os
        
        # Add adversarial_tests to path
        adversarial_path = os.path.join(os.path.dirname(__file__), '..')
        if adversarial_path not in sys.path:
            sys.path.insert(0, adversarial_path)
        
        # Try to import existing components
        try:
            from market_manipulation_scenarios import MarketManipulationGenerator
            from extreme_data_attacks import ExtremeDataGenerator
            existing_framework_available = True
        except ImportError:
            existing_framework_available = False
        
        if existing_framework_available:
            print("✅ Successfully integrated with existing adversarial test framework")
            return True
        else:
            print("⚠️ Existing adversarial test framework not found, running independently")
            return False
            
    except Exception as e:
        print(f"❌ Error integrating with existing framework: {e}")
        return False

# Example usage
def run_integration_demo():
    """Demonstrate the unified attack integration."""
    print("🚨" * 50)
    print("AGENT GAMMA MISSION - UNIFIED ATTACK INTEGRATION DEMO")
    print("🚨" * 50)
    
    # Initialize orchestrator
    orchestrator = UnifiedAttackOrchestrator()
    
    # Show available attacks
    print("\n📋 Available Attacks:")
    available_attacks = orchestrator.get_available_attacks()
    for module, methods in available_attacks.items():
        print(f"  {module.upper()}:")
        for method in methods:
            print(f"    - {method}")
    
    # Execute some sample attacks
    print("\n🎯 Executing Sample Attacks:")
    
    # Sample data
    market_data = np.random.randn(48, 13)
    regime_indicators = {'market_regime': 'sideways', 'regime_confidence': 0.6}
    agent_predictions = [{'agent_name': 'MLMI', 'action_probabilities': [0.4, 0.3, 0.3], 'confidence': 0.7}]
    shared_context = {'volatility_30': 0.02, 'volume_ratio': 1.5}
    
    # Execute individual attacks
    try:
        result1 = orchestrator.execute_attack(
            'coordination', 'consensus_disruption',
            agent_predictions, shared_context
        )
        print(f"  Coordination Attack: Success={result1.success}, Disruption={result1.disruption_score:.3f}")
        
        result2 = orchestrator.execute_attack(
            'temporal', 'correlation_poisoning',
            market_data
        )
        print(f"  Temporal Attack: Success={result2.success}, Disruption={result2.disruption_score:.3f}")
        
        result3 = orchestrator.execute_attack(
            'regime', 'false_bull_signal',
            market_data, regime_indicators
        )
        print(f"  Regime Attack: Success={result3.success}, Disruption={result3.disruption_score:.3f}")
        
    except Exception as e:
        print(f"  Error executing attacks: {e}")
    
    # Generate report
    print("\n📊 Attack Report:")
    report = orchestrator.generate_attack_report()
    
    if 'summary' in report:
        print(f"  Total Attacks: {report['summary']['total_attacks']}")
        print(f"  Success Rate: {report['summary']['overall_success_rate']:.1%}")
        print(f"  Avg Disruption: {report['summary']['avg_disruption_score']:.3f}")
    
    # Test integration
    print("\n🔗 Testing Framework Integration:")
    integration_success = integrate_with_existing_framework()
    
    return orchestrator

if __name__ == "__main__":
    run_integration_demo()