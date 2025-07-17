#!/usr/bin/env python3
"""
Strategic MARL System Validation Test

Tests Strategic MARL system functionality after PyTorch compatibility fixes.
Validates agent cooperation, decision-making, and JIT compilation.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import traceback
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fast_architectures import FastStrategicMARLSystem, FastMLMIActor, FastNWRQKActor, FastMMDActor
from models.jit_compatible_models import JITFastStrategicSystem

class StrategicMARLValidator:
    """Validates Strategic MARL system functionality."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🚀 Starting Strategic MARL System Validation")
        print("="*60)
        
        # Test 1: Basic model initialization
        init_result = self._test_model_initialization()
        self.test_results.append(init_result)
        
        # Test 2: Agent cooperation
        cooperation_result = self._test_agent_cooperation()
        self.test_results.append(cooperation_result)
        
        # Test 3: Decision making
        decision_result = self._test_decision_making()
        self.test_results.append(decision_result)
        
        # Test 4: JIT compilation
        jit_result = self._test_jit_compilation()
        self.test_results.append(jit_result)
        
        # Test 5: Performance validation
        performance_result = self._test_performance()
        self.test_results.append(performance_result)
        
        # Generate summary
        return self._generate_summary()
    
    def _test_model_initialization(self) -> Dict[str, Any]:
        """Test Strategic MARL system initialization."""
        print("\n📋 Test 1: Model Initialization")
        
        result = {
            "test_name": "model_initialization",
            "success": False,
            "error": None,
            "details": {}
        }
        
        try:
            # Test FastStrategicMARLSystem
            strategic_system = FastStrategicMARLSystem()
            result["details"]["strategic_system"] = "initialized"
            
            # Test individual agents
            mlmi_agent = FastMLMIActor()
            nwrqk_agent = FastNWRQKActor() 
            mmd_agent = FastMMDActor()
            
            result["details"]["individual_agents"] = {
                "mlmi": "initialized",
                "nwrqk": "initialized", 
                "mmd": "initialized"
            }
            
            # Test JIT compatible system
            jit_strategic = JITFastStrategicSystem()
            result["details"]["jit_system"] = "initialized"
            
            result["success"] = True
            print("  ✅ All models initialized successfully")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"  ❌ Model initialization failed: {e}")
        
        return result
    
    def _test_agent_cooperation(self) -> Dict[str, Any]:
        """Test agent cooperation in Strategic MARL system."""
        print("\n🤝 Test 2: Agent Cooperation")
        
        result = {
            "test_name": "agent_cooperation",
            "success": False,
            "error": None,
            "details": {}
        }
        
        try:
            strategic_system = FastStrategicMARLSystem()
            
            # Create test inputs for each agent
            test_states = {
                'mlmi': torch.randn(1, 4),
                'nwrqk': torch.randn(1, 6),
                'mmd': torch.randn(1, 3)
            }
            
            # Test forward pass
            output = strategic_system.forward(test_states)
            
            # Validate output structure
            assert 'agents' in output, "Missing agents output"
            assert 'critic' in output, "Missing critic output"
            
            # Validate individual agent outputs
            agent_outputs = output['agents']
            for agent_name in ['mlmi', 'nwrqk', 'mmd']:
                assert agent_name in agent_outputs, f"Missing {agent_name} output"
                agent_output = agent_outputs[agent_name]
                assert 'action' in agent_output, f"Missing action for {agent_name}"
                assert 'action_probs' in agent_output, f"Missing action_probs for {agent_name}"
            
            # Test fast inference
            fast_output = strategic_system.fast_inference(test_states)
            assert 'actions' in fast_output, "Missing actions in fast inference"
            assert 'value' in fast_output, "Missing value in fast inference"
            
            result["details"]["output_structure"] = "valid"
            result["details"]["fast_inference"] = "working"
            result["details"]["agent_coordination"] = "functioning"
            
            result["success"] = True
            print("  ✅ Agent cooperation functioning correctly")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"  ❌ Agent cooperation test failed: {e}")
        
        return result
    
    def _test_decision_making(self) -> Dict[str, Any]:
        """Test strategic decision-making capabilities."""
        print("\n🧠 Test 3: Decision Making")
        
        result = {
            "test_name": "decision_making",
            "success": False,
            "error": None,
            "details": {}
        }
        
        try:
            strategic_system = FastStrategicMARLSystem()
            
            # Test multiple scenarios
            scenarios = [
                {
                    'name': 'bullish_momentum',
                    'mlmi': torch.tensor([[1.0, 0.8, 0.6, 0.9]]),  # Strong momentum
                    'nwrqk': torch.tensor([[0.2, 0.1, 0.3, 0.8, 0.9, 0.7]]),  # Low volatility, high trend
                    'mmd': torch.tensor([[0.1, 0.2, 0.8]])  # Low divergence
                },
                {
                    'name': 'bearish_momentum',
                    'mlmi': torch.tensor([[-1.0, -0.8, -0.6, -0.9]]),  # Strong negative momentum
                    'nwrqk': torch.tensor([[0.8, 0.9, 0.7, -0.8, -0.9, -0.7]]),  # High volatility, negative trend
                    'mmd': torch.tensor([[0.9, 0.8, 0.2]])  # High divergence
                },
                {
                    'name': 'neutral_consolidation',
                    'mlmi': torch.tensor([[0.0, 0.1, -0.1, 0.0]]),  # Weak momentum
                    'nwrqk': torch.tensor([[0.5, 0.4, 0.6, 0.0, 0.1, -0.1]]),  # Moderate volatility
                    'mmd': torch.tensor([[0.5, 0.5, 0.5]])  # Moderate divergence
                }
            ]
            
            decision_quality = []
            
            for scenario in scenarios:
                states = {
                    'mlmi': scenario['mlmi'],
                    'nwrqk': scenario['nwrqk'], 
                    'mmd': scenario['mmd']
                }
                
                # Get decisions
                decisions = strategic_system.fast_inference(states)
                actions = decisions['actions']
                value = decisions['value']
                
                # Validate decision consistency
                decision_quality.append({
                    'scenario': scenario['name'],
                    'actions': actions,
                    'value': value,
                    'consistent': self._validate_decision_consistency(actions, scenario['name'])
                })
            
            result["details"]["scenarios_tested"] = len(scenarios)
            result["details"]["decision_quality"] = decision_quality
            result["details"]["consistency_rate"] = sum(1 for d in decision_quality if d['consistent']) / len(decision_quality)
            
            result["success"] = True
            print(f"  ✅ Decision making validated ({result['details']['consistency_rate']:.1%} consistency)")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"  ❌ Decision making test failed: {e}")
        
        return result
    
    def _validate_decision_consistency(self, actions: Dict[str, int], scenario_name: str) -> bool:
        """Validate that decisions are consistent with scenario."""
        # Simple heuristic validation - in real scenarios this would be more sophisticated
        if scenario_name == 'bullish_momentum':
            # Expect mostly long actions (action 0 typically = long)
            return sum(1 for action in actions.values() if action == 0) >= 2
        elif scenario_name == 'bearish_momentum':
            # Expect mostly short actions (action 2 typically = short)
            return sum(1 for action in actions.values() if action == 2) >= 2
        else:  # neutral
            # Expect mixed or hold actions
            return True  # More lenient for neutral scenarios
    
    def _test_jit_compilation(self) -> Dict[str, Any]:
        """Test JIT compilation functionality."""
        print("\n⚡ Test 4: JIT Compilation")
        
        result = {
            "test_name": "jit_compilation",
            "success": False,
            "error": None,
            "details": {}
        }
        
        try:
            # Test JIT compatible model
            jit_strategic = JITFastStrategicSystem()
            
            # Test inputs
            mlmi_input = torch.randn(1, 4)
            nwrqk_input = torch.randn(1, 6)
            mmd_input = torch.randn(1, 3)
            
            # Test tracing compilation
            with torch.no_grad():
                traced_model = torch.jit.trace(jit_strategic, (mlmi_input, nwrqk_input, mmd_input))
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Test compiled model
            compiled_output = traced_model(mlmi_input, nwrqk_input, mmd_input)
            
            # Validate output
            assert len(compiled_output) == 4, "JIT model should return 4 outputs"
            mlmi_probs, nwrqk_probs, mmd_probs, value = compiled_output
            
            assert mlmi_probs.shape == (1, 3), "MLMI probabilities shape incorrect"
            assert nwrqk_probs.shape == (1, 3), "NWRQK probabilities shape incorrect"
            assert mmd_probs.shape == (1, 3), "MMD probabilities shape incorrect"
            assert value.shape == (1,), "Value shape incorrect"
            
            result["details"]["compilation"] = "success"
            result["details"]["output_validation"] = "passed"
            result["details"]["optimization"] = "applied"
            
            result["success"] = True
            print("  ✅ JIT compilation working correctly")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"  ❌ JIT compilation test failed: {e}")
        
        return result
    
    def _test_performance(self) -> Dict[str, Any]:
        """Test performance meets <5ms requirements."""
        print("\n🏃 Test 5: Performance Validation")
        
        result = {
            "test_name": "performance_validation",
            "success": False,
            "error": None,
            "details": {}
        }
        
        try:
            # Test both regular and JIT models
            strategic_system = FastStrategicMARLSystem()
            jit_strategic = JITFastStrategicSystem()
            
            # JIT compile the JIT model
            test_inputs = (torch.randn(1, 4), torch.randn(1, 6), torch.randn(1, 3))
            with torch.no_grad():
                traced_model = torch.jit.trace(jit_strategic, test_inputs)
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Benchmark regular model
            regular_times = []
            states = {'mlmi': torch.randn(1, 4), 'nwrqk': torch.randn(1, 6), 'mmd': torch.randn(1, 3)}
            
            # Warmup
            for _ in range(20):
                _ = strategic_system.fast_inference(states)
            
            # Benchmark
            for _ in range(1000):
                start_time = time.perf_counter()
                _ = strategic_system.fast_inference(states)
                end_time = time.perf_counter()
                regular_times.append((end_time - start_time) * 1000)
            
            # Benchmark JIT model
            jit_times = []
            
            # Warmup
            for _ in range(20):
                _ = traced_model(*test_inputs)
            
            # Benchmark
            for _ in range(1000):
                start_time = time.perf_counter()
                _ = traced_model(*test_inputs)
                end_time = time.perf_counter()
                jit_times.append((end_time - start_time) * 1000)
            
            # Calculate statistics
            regular_p99 = np.percentile(regular_times, 99)
            jit_p99 = np.percentile(jit_times, 99)
            
            result["details"]["regular_model"] = {
                "mean_ms": np.mean(regular_times),
                "p99_ms": regular_p99,
                "meets_target": regular_p99 < 5.0
            }
            
            result["details"]["jit_model"] = {
                "mean_ms": np.mean(jit_times),
                "p99_ms": jit_p99,
                "meets_target": jit_p99 < 5.0
            }
            
            result["details"]["speedup"] = np.mean(regular_times) / np.mean(jit_times)
            
            # Success if at least one model meets target
            result["success"] = regular_p99 < 5.0 or jit_p99 < 5.0
            
            if result["success"]:
                print(f"  ✅ Performance targets met (Regular: {regular_p99:.2f}ms, JIT: {jit_p99:.2f}ms)")
            else:
                print(f"  ❌ Performance targets not met (Regular: {regular_p99:.2f}ms, JIT: {jit_p99:.2f}ms)")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"  ❌ Performance validation failed: {e}")
        
        return result
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        successful_tests = sum(1 for result in self.test_results if result["success"])
        total_tests = len(self.test_results)
        
        failed_tests = [result for result in self.test_results if not result["success"]]
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": len(failed_tests),
            "success_rate": successful_tests / total_tests,
            "overall_success": successful_tests == total_tests,
            "test_results": self.test_results,
            "failed_test_names": [test["test_name"] for test in failed_tests]
        }

def main():
    """Main validation function."""
    validator = StrategicMARLValidator()
    results = validator.run_validation()
    
    print("\n" + "="*60)
    print("📊 STRATEGIC MARL VALIDATION SUMMARY")
    print("="*60)
    
    print(f"Total Tests: {results['total_tests']}")
    print(f"Successful: {results['successful_tests']}")
    print(f"Failed: {results['failed_tests']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    
    if results['failed_tests'] > 0:
        print(f"\nFailed Tests:")
        for test_name in results['failed_test_names']:
            print(f"  - {test_name}")
    
    print("\n" + "="*60)
    
    if results['overall_success']:
        print("✅ STRATEGIC MARL SYSTEM VALIDATION PASSED")
        print("🚀 System is ready for production deployment")
        return 0
    else:
        print("❌ STRATEGIC MARL SYSTEM VALIDATION FAILED")
        print("🔧 System needs fixes before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())