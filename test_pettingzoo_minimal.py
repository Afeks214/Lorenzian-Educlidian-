#!/usr/bin/env python3
"""
Minimal PettingZoo Environment Test
=================================

This test checks basic PettingZoo environment structure without requiring
heavy dependencies like torch, complex ML libraries, or external data.
"""

import sys
import os
import importlib
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class MinimalPettingZooTest:
    """Minimal test for PettingZoo environments"""
    
    def __init__(self):
        self.results = []
    
    def test_environment_basic_structure(self, module_path: str, class_name: str, env_name: str) -> Dict[str, Any]:
        """
        Test basic environment structure
        
        Args:
            module_path: Python module path
            class_name: Environment class name
            env_name: Human-readable environment name
            
        Returns:
            Test results dictionary
        """
        
        result = {
            'name': env_name,
            'module_path': module_path,
            'class_name': class_name,
            'import_success': False,
            'instantiation_success': False,
            'has_required_attributes': [],
            'has_required_methods': [],
            'space_check_success': False,
            'agent_count': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Import the module
            print(f"🧪 Testing {env_name}")
            print(f"   📦 Importing {module_path}...")
            
            module = importlib.import_module(module_path)
            env_class = getattr(module, class_name)
            result['import_success'] = True
            print(f"   ✅ Import successful")
            
            # Step 2: Try to instantiate with minimal config
            print(f"   🏗️  Instantiating {class_name}...")
            
            # Create minimal config
            minimal_config = self._create_minimal_config(env_name)
            
            try:
                env = env_class(minimal_config)
                result['instantiation_success'] = True
                print(f"   ✅ Instantiation successful")
                
                # Step 3: Check required attributes
                print(f"   🔍 Checking required attributes...")
                required_attrs = ['possible_agents', 'agents', 'action_spaces', 'observation_spaces']
                for attr in required_attrs:
                    if hasattr(env, attr):
                        result['has_required_attributes'].append(attr)
                        print(f"   ✅ Has {attr}")
                    else:
                        result['warnings'].append(f"Missing attribute: {attr}")
                        print(f"   ⚠️  Missing {attr}")
                
                # Step 4: Check required methods
                print(f"   🔧 Checking required methods...")
                required_methods = ['reset', 'step', 'observe', 'render', 'close']
                for method in required_methods:
                    if hasattr(env, method) and callable(getattr(env, method)):
                        result['has_required_methods'].append(method)
                        print(f"   ✅ Has {method}()")
                    else:
                        result['warnings'].append(f"Missing method: {method}")
                        print(f"   ⚠️  Missing {method}()")
                
                # Step 5: Check agent count
                if hasattr(env, 'possible_agents'):
                    result['agent_count'] = len(env.possible_agents)
                    print(f"   ✅ Agent count: {result['agent_count']}")
                
                # Step 6: Check action/observation spaces
                if hasattr(env, 'action_spaces') and hasattr(env, 'observation_spaces'):
                    try:
                        action_spaces = env.action_spaces
                        observation_spaces = env.observation_spaces
                        
                        if isinstance(action_spaces, dict) and isinstance(observation_spaces, dict):
                            result['space_check_success'] = True
                            print(f"   ✅ Action/observation spaces are dictionaries")
                        else:
                            result['warnings'].append("Action/observation spaces are not dictionaries")
                            print(f"   ⚠️  Action/observation spaces are not dictionaries")
                    except Exception as e:
                        result['warnings'].append(f"Error checking spaces: {str(e)}")
                        print(f"   ⚠️  Error checking spaces: {str(e)}")
                
                # Step 7: Try basic reset (if possible)
                print(f"   🔄 Testing reset method...")
                try:
                    env.reset()
                    print(f"   ✅ Reset successful")
                except Exception as e:
                    result['warnings'].append(f"Reset failed: {str(e)}")
                    print(f"   ⚠️  Reset failed: {str(e)}")
                
                # Step 8: Clean up
                try:
                    env.close()
                    print(f"   ✅ Close successful")
                except Exception as e:
                    result['warnings'].append(f"Close failed: {str(e)}")
                    print(f"   ⚠️  Close failed: {str(e)}")
                
            except Exception as e:
                result['errors'].append(f"Instantiation failed: {str(e)}")
                print(f"   ❌ Instantiation failed: {str(e)}")
                
        except ImportError as e:
            result['errors'].append(f"Import failed: {str(e)}")
            print(f"   ❌ Import failed: {str(e)}")
        except Exception as e:
            result['errors'].append(f"Unexpected error: {str(e)}")
            print(f"   ❌ Unexpected error: {str(e)}")
        
        return result
    
    def _create_minimal_config(self, env_name: str) -> Dict[str, Any]:
        """Create minimal configuration for environment"""
        
        # Base minimal config
        base_config = {
            'max_steps': 10,
            'initial_capital': 100000.0,
            'max_position_size': 0.1,
            'risk_tolerance': 0.05
        }
        
        # Environment-specific configs
        if 'tactical' in env_name.lower():
            return {
                'tactical_marl': {
                    'environment': {
                        'matrix_shape': [60, 7],
                        'max_episode_steps': 10
                    },
                    'agents': {
                        'fvg_agent': {'attention_weights': [0.4, 0.4, 0.1, 0.05, 0.05]},
                        'momentum_agent': {'attention_weights': [0.05, 0.05, 0.1, 0.3, 0.5]},
                        'entry_opt_agent': {'attention_weights': [0.2, 0.2, 0.2, 0.2, 0.2]}
                    }
                }
            }
        elif 'strategic' in env_name.lower():
            return {
                'matrix_shape': [48, 13],
                'max_timesteps': 10,
                'time_window': 5
            }
        elif 'risk' in env_name.lower():
            return {
                'initial_capital': 100000.0,
                'max_steps': 10,
                'risk_tolerance': 0.05,
                'asset_universe': ['SPY', 'QQQ', 'TLT'],
                'scenario': 'normal'
            }
        elif 'execution' in env_name.lower():
            # Return ExecutionEnvironmentConfig-like structure
            return {
                'max_steps': 10,
                'initial_portfolio_value': 100000.0,
                'max_position_size': 0.1,
                'transaction_cost_bps': 2.0
            }
        else:
            return base_config
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all environment tests"""
        
        print("🚀 Minimal PettingZoo Environment Test Suite")
        print("=" * 60)
        
        # Define test cases
        test_cases = [
            {
                'module_path': 'environment.tactical_env',
                'class_name': 'TacticalMarketEnv',
                'env_name': 'Tactical Environment'
            },
            {
                'module_path': 'environment.strategic_env',
                'class_name': 'StrategicMarketEnv',
                'env_name': 'Strategic Environment'
            },
            {
                'module_path': 'src.environment.tactical_env',
                'class_name': 'TacticalMarketEnv',
                'env_name': 'Tactical Environment (src)'
            },
            {
                'module_path': 'src.environment.strategic_env',
                'class_name': 'StrategicMarketEnv',
                'env_name': 'Strategic Environment (src)'
            },
            {
                'module_path': 'src.environment.risk_env',
                'class_name': 'RiskManagementEnv',
                'env_name': 'Risk Management Environment'
            },
            {
                'module_path': 'src.environment.execution_env',
                'class_name': 'ExecutionEnvironment',
                'env_name': 'Execution Environment'
            }
        ]
        
        # Run tests
        for test_case in test_cases:
            print(f"\n{'-' * 60}")
            result = self.test_environment_basic_structure(
                test_case['module_path'],
                test_case['class_name'],
                test_case['env_name']
            )
            self.results.append(result)
        
        # Print summary
        self._print_summary()
        
        return {
            'results': self.results,
            'summary': self._get_summary_stats()
        }
    
    def _print_summary(self):
        """Print test summary"""
        
        print(f"\n{'=' * 60}")
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        if not self.results:
            print("❌ No tests were run")
            return
        
        total_tests = len(self.results)
        import_success = sum(1 for r in self.results if r['import_success'])
        instantiation_success = sum(1 for r in self.results if r['instantiation_success'])
        
        print(f"🧪 Total environments tested: {total_tests}")
        print(f"✅ Import successful: {import_success}/{total_tests}")
        print(f"✅ Instantiation successful: {instantiation_success}/{total_tests}")
        
        if instantiation_success > 0:
            print(f"\n📋 Detailed Results:")
            for result in self.results:
                if result['instantiation_success']:
                    print(f"  ✅ {result['name']}")
                    print(f"     • Agents: {result['agent_count']}")
                    print(f"     • Attributes: {len(result['has_required_attributes'])}/4")
                    print(f"     • Methods: {len(result['has_required_methods'])}/5")
                    print(f"     • Spaces OK: {result['space_check_success']}")
                    
                    if result['warnings']:
                        print(f"     • Warnings: {len(result['warnings'])}")
                else:
                    print(f"  ❌ {result['name']}")
                    if result['errors']:
                        print(f"     • Errors: {result['errors'][0]}")
        
        # Overall assessment
        good_envs = sum(1 for r in self.results if r['instantiation_success'] and 
                       len(r['has_required_attributes']) >= 3 and
                       len(r['has_required_methods']) >= 4)
        
        print(f"\n🎯 Overall Assessment: {good_envs}/{total_tests} environments are well-structured")
        
        if good_envs == total_tests and total_tests > 0:
            print("🎉 ALL TESTED ENVIRONMENTS PASSED BASIC STRUCTURE CHECKS!")
        elif good_envs > 0:
            print("✅ Some environments passed - check warnings for improvements")
        else:
            print("⚠️  No environments passed all basic checks - review errors above")
    
    def _get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        
        if not self.results:
            return {}
        
        total = len(self.results)
        return {
            'total_tested': total,
            'import_success_rate': sum(1 for r in self.results if r['import_success']) / total,
            'instantiation_success_rate': sum(1 for r in self.results if r['instantiation_success']) / total,
            'average_agent_count': sum(r['agent_count'] for r in self.results) / total,
            'average_attributes': sum(len(r['has_required_attributes']) for r in self.results) / total,
            'average_methods': sum(len(r['has_required_methods']) for r in self.results) / total,
            'total_errors': sum(len(r['errors']) for r in self.results),
            'total_warnings': sum(len(r['warnings']) for r in self.results)
        }


def main():
    """Main test function"""
    
    # Create and run test suite
    test_suite = MinimalPettingZooTest()
    results = test_suite.run_all_tests()
    
    print(f"\n{'=' * 60}")
    print("🏁 Test Complete")
    print("=" * 60)
    
    stats = results['summary']
    if stats:
        print(f"📈 Success Rate: {stats['instantiation_success_rate']:.1%}")
        print(f"🔢 Average Agent Count: {stats['average_agent_count']:.1f}")
        print(f"⚠️  Total Warnings: {stats['total_warnings']}")
        print(f"❌ Total Errors: {stats['total_errors']}")
    
    print(f"\n💡 Next Steps:")
    print(f"1. Address any errors shown above")
    print(f"2. Review warnings for potential improvements")
    print(f"3. Run full PettingZoo API check when ready")
    print(f"4. Test with actual MARL training when dependencies are available")
    
    return results


if __name__ == "__main__":
    main()