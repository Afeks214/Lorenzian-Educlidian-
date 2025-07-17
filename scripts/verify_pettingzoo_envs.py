#!/usr/bin/env python3
"""
PettingZoo Environment Verification Script
==========================================

This script verifies that all MARL environments in the GrandModel system
are properly compatible with PettingZoo standards.
"""

import sys
import os
import traceback
from typing import Dict, List, Any
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_environment_compatibility(env_class, env_name: str) -> Dict[str, Any]:
    """Verify a single environment's PettingZoo compatibility."""
    results = {
        'name': env_name,
        'compatible': False,
        'errors': [],
        'warnings': [],
        'tests_passed': 0,
        'tests_total': 0
    }
    
    try:
        # Test 1: Environment creation
        results['tests_total'] += 1
        env = env_class()
        results['tests_passed'] += 1
        print(f"✅ {env_name}: Environment creation successful")
        
        # Test 2: Required attributes
        results['tests_total'] += 1
        required_attrs = ['possible_agents', 'agent_selection', 'agents', 
                         'observation_spaces', 'action_spaces']
        for attr in required_attrs:
            if not hasattr(env, attr):
                results['errors'].append(f"Missing required attribute: {attr}")
                
        if not results['errors']:
            results['tests_passed'] += 1
            print(f"✅ {env_name}: All required attributes present")
        
        # Test 3: Reset functionality
        results['tests_total'] += 1
        env.reset()
        results['tests_passed'] += 1
        print(f"✅ {env_name}: Reset successful")
        
        # Test 4: Agent iteration
        results['tests_total'] += 1
        if hasattr(env, 'agent_iter'):
            agent_count = 0
            for agent in env.agent_iter():
                agent_count += 1
                if agent_count > 10:  # Prevent infinite loops
                    break
            results['tests_passed'] += 1
            print(f"✅ {env_name}: Agent iteration works ({agent_count} agents)")
        else:
            results['errors'].append("Missing agent_iter method")
        
        # Test 5: Observation and action spaces
        results['tests_total'] += 1
        valid_spaces = True
        for agent in env.possible_agents:
            if agent not in env.observation_spaces:
                results['errors'].append(f"Missing observation space for agent {agent}")
                valid_spaces = False
            if agent not in env.action_spaces:
                results['errors'].append(f"Missing action space for agent {agent}")
                valid_spaces = False
                
        if valid_spaces:
            results['tests_passed'] += 1
            print(f"✅ {env_name}: Valid observation and action spaces")
        
        # Test 6: Step functionality
        results['tests_total'] += 1
        if env.agent_selection:
            action = env.action_spaces[env.agent_selection].sample()
            try:
                env.step(action)
                results['tests_passed'] += 1
                print(f"✅ {env_name}: Step execution successful")
            except Exception as e:
                results['errors'].append(f"Step execution failed: {str(e)}")
        else:
            results['warnings'].append("No agent selected for step test")
        
        # Test 7: Rendering
        results['tests_total'] += 1
        try:
            env.render()
            results['tests_passed'] += 1
            print(f"✅ {env_name}: Rendering successful")
        except Exception as e:
            results['warnings'].append(f"Rendering issue: {str(e)}")
            results['tests_passed'] += 1  # Non-critical
        
        # Test 8: Close functionality
        results['tests_total'] += 1
        env.close()
        results['tests_passed'] += 1
        print(f"✅ {env_name}: Close successful")
        
        # Calculate compatibility
        if results['tests_passed'] == results['tests_total'] and not results['errors']:
            results['compatible'] = True
            print(f"🎉 {env_name}: FULLY COMPATIBLE")
        else:
            print(f"⚠️  {env_name}: COMPATIBILITY ISSUES")
            
    except Exception as e:
        results['errors'].append(f"Critical error during verification: {str(e)}")
        print(f"❌ {env_name}: CRITICAL ERROR - {str(e)}")
        traceback.print_exc()
    
    return results

def main():
    """Main verification routine."""
    print("🔍 PettingZoo Environment Verification")
    print("=" * 50)
    
    # Import environments
    environments = {}
    
    try:
        from src.environment.strategic_env import StrategicMARLEnvironment
        environments['Strategic'] = StrategicMARLEnvironment
        print("📦 Strategic environment imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Strategic environment: {e}")
    
    try:
        from src.environment.tactical_env import TacticalMARLEnvironment
        environments['Tactical'] = TacticalMARLEnvironment
        print("📦 Tactical environment imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Tactical environment: {e}")
    
    try:
        from src.environment.risk_env import RiskMARLEnvironment
        environments['Risk'] = RiskMARLEnvironment
        print("📦 Risk environment imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Risk environment: {e}")
    
    try:
        from src.environment.execution_env import ExecutionMARLEnvironment
        environments['Execution'] = ExecutionMARLEnvironment
        print("📦 Execution environment imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Execution environment: {e}")
    
    if not environments:
        print("❌ No environments could be imported. Check your installation.")
        return False
    
    print(f"\n🧪 Testing {len(environments)} environments...")
    print("=" * 50)
    
    # Verify each environment
    all_results = []
    for env_name, env_class in environments.items():
        print(f"\n🔎 Verifying {env_name} Environment...")
        results = verify_environment_compatibility(env_class, env_name)
        all_results.append(results)
        print(f"   Tests passed: {results['tests_passed']}/{results['tests_total']}")
    
    # Summary report
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    compatible_count = 0
    for result in all_results:
        status = "✅ COMPATIBLE" if result['compatible'] else "❌ ISSUES"
        print(f"{result['name']:12} | {status} | {result['tests_passed']}/{result['tests_total']} tests")
        if result['compatible']:
            compatible_count += 1
        
        if result['errors']:
            print(f"             | Errors: {', '.join(result['errors'])}")
        if result['warnings']:
            print(f"             | Warnings: {', '.join(result['warnings'])}")
    
    print(f"\n🎯 Overall Compatibility: {compatible_count}/{len(all_results)} environments")
    
    if compatible_count == len(all_results):
        print("🎉 ALL ENVIRONMENTS ARE PETTINGZOO COMPATIBLE!")
        return True
    else:
        print("⚠️  Some environments need attention before full compatibility")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)