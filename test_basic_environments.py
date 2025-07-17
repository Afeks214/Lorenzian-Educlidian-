#!/usr/bin/env python3
"""
Basic PettingZoo Environment Test
===============================

This is a minimal test that checks if environments can be instantiated
without requiring the full dependency stack.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_basic_instantiation():
    """Test basic environment instantiation"""
    
    print("🧪 Basic Environment Instantiation Test")
    print("=" * 50)
    
    # Test environments with minimal dependencies
    test_cases = [
        {
            'name': 'Tactical Environment',
            'import_path': 'environment.tactical_env',
            'class_name': 'TacticalMarketEnv',
            'config': {}
        },
        {
            'name': 'Strategic Environment', 
            'import_path': 'environment.strategic_env',
            'class_name': 'StrategicMarketEnv',
            'config': {}
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        result = {'name': test_case['name'], 'success': False, 'error': None}
        
        try:
            # Try to import the module
            module = __import__(test_case['import_path'], fromlist=[test_case['class_name']])
            env_class = getattr(module, test_case['class_name'])
            
            # Try to instantiate with minimal config
            env = env_class(test_case['config'])
            
            # Check basic attributes
            if hasattr(env, 'possible_agents'):
                print(f"✅ {test_case['name']}: Basic instantiation successful")
                print(f"   • Agents: {env.possible_agents}")
                result['success'] = True
            else:
                print(f"⚠️  {test_case['name']}: Missing possible_agents attribute")
                result['error'] = "Missing possible_agents attribute"
                
        except ImportError as e:
            print(f"❌ {test_case['name']}: Import failed - {str(e)}")
            result['error'] = f"Import failed: {str(e)}"
        except Exception as e:
            print(f"❌ {test_case['name']}: Instantiation failed - {str(e)}")
            result['error'] = f"Instantiation failed: {str(e)}"
        
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n🎯 Test Results: {successful}/{len(results)} environments passed basic test")
    
    return results

if __name__ == "__main__":
    test_basic_instantiation()
