#!/usr/bin/env python3
"""
Simple PettingZoo Environment Structure Verification
==================================================

This script verifies the basic structure of PettingZoo environments without
requiring heavy dependencies like torch. It checks:
1. File existence and basic import structure
2. Required PettingZoo methods and attributes
3. Basic instantiation without full dependencies
4. Action and observation space definitions

This is intended as a lightweight verification tool for development.
"""

import os
import sys
import importlib.util
import inspect
from typing import Dict, List, Any, Optional
from pathlib import Path
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class PettingZooVerifier:
    """Simple verification class for PettingZoo environments"""
    
    def __init__(self):
        self.results = {
            'environments_found': [],
            'environments_tested': [],
            'verification_results': {},
            'errors': [],
            'warnings': []
        }
    
    def verify_environment_file(self, env_path: Path) -> Dict[str, Any]:
        """
        Verify a single environment file
        
        Args:
            env_path: Path to environment file
            
        Returns:
            Dictionary with verification results
        """
        result = {
            'file_path': str(env_path),
            'exists': False,
            'importable': False,
            'has_pettingzoo_base': False,
            'has_required_methods': [],
            'has_required_attributes': [],
            'class_found': None,
            'errors': []
        }
        
        try:
            # Check if file exists
            if not env_path.exists():
                result['errors'].append(f"File does not exist: {env_path}")
                return result
            
            result['exists'] = True
            
            # Try to import the file
            spec = importlib.util.spec_from_file_location("env_module", env_path)
            if spec is None:
                result['errors'].append("Could not create module spec")
                return result
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute the module to check basic syntax
            try:
                spec.loader.exec_module(module)
                result['importable'] = True
            except Exception as e:
                # If there are import errors, try to continue with partial verification
                result['errors'].append(f"Import error: {str(e)}")
                # Still try to analyze the source code
                
            # Analyze source code for structure
            with open(env_path, 'r') as f:
                source_code = f.read()
            
            # Check for PettingZoo imports
            if 'from pettingzoo' in source_code or 'import pettingzoo' in source_code:
                result['has_pettingzoo_base'] = True
            
            # Check for AECEnv class
            if 'AECEnv' in source_code:
                result['has_pettingzoo_base'] = True
            
            # Look for environment classes
            if result['importable']:
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and name.endswith('Env'):
                        result['class_found'] = name
                        result = self._verify_class_structure(obj, result)
                        break
            
            # Check for required methods in source code
            required_methods = ['reset', 'step', 'observe', 'render', 'close']
            for method in required_methods:
                if f'def {method}(' in source_code:
                    result['has_required_methods'].append(method)
            
            # Check for required attributes in source code
            required_attrs = ['possible_agents', 'agents', 'action_spaces', 'observation_spaces']
            for attr in required_attrs:
                if f'self.{attr}' in source_code or f'{attr} =' in source_code:
                    result['has_required_attributes'].append(attr)
            
        except Exception as e:
            result['errors'].append(f"Unexpected error: {str(e)}")
        
        return result
    
    def _verify_class_structure(self, cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify the structure of an environment class
        
        Args:
            cls: Environment class
            result: Current verification result
            
        Returns:
            Updated verification result
        """
        try:
            # Check if it's a subclass of something that looks like AECEnv
            if hasattr(cls, '__bases__'):
                for base in cls.__bases__:
                    if 'AECEnv' in str(base):
                        result['has_pettingzoo_base'] = True
                        break
            
            # Check for required methods
            required_methods = ['reset', 'step', 'observe', 'render', 'close']
            for method in required_methods:
                if hasattr(cls, method):
                    result['has_required_methods'].append(method)
            
            # Check for required attributes (if the class has them defined)
            required_attrs = ['possible_agents', 'agents', 'action_spaces', 'observation_spaces']
            for attr in required_attrs:
                if hasattr(cls, attr):
                    result['has_required_attributes'].append(attr)
            
        except Exception as e:
            result['errors'].append(f"Error verifying class structure: {str(e)}")
        
        return result
    
    def verify_environments(self) -> Dict[str, Any]:
        """
        Verify all PettingZoo environments in the project
        
        Returns:
            Comprehensive verification results
        """
        
        # Define environment files to check
        env_files = [
            PROJECT_ROOT / 'environment' / 'tactical_env.py',
            PROJECT_ROOT / 'environment' / 'strategic_env.py',
            PROJECT_ROOT / 'src' / 'environment' / 'tactical_env.py',
            PROJECT_ROOT / 'src' / 'environment' / 'strategic_env.py',
            PROJECT_ROOT / 'src' / 'environment' / 'risk_env.py',
            PROJECT_ROOT / 'src' / 'environment' / 'execution_env.py',
        ]
        
        print("🔍 PettingZoo Environment Structure Verification")
        print("=" * 60)
        
        for env_file in env_files:
            if env_file.exists():
                self.results['environments_found'].append(str(env_file))
                print(f"\n📁 Found: {env_file.name}")
                
                # Verify the environment
                verification = self.verify_environment_file(env_file)
                self.results['verification_results'][str(env_file)] = verification
                self.results['environments_tested'].append(str(env_file))
                
                # Print results
                self._print_verification_results(env_file.name, verification)
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_verification_results(self, filename: str, result: Dict[str, Any]):
        """Print verification results for a single file"""
        
        print(f"  ✅ Exists: {result['exists']}")
        print(f"  ✅ Importable: {result['importable']}")
        print(f"  ✅ Has PettingZoo base: {result['has_pettingzoo_base']}")
        
        if result['class_found']:
            print(f"  ✅ Environment class found: {result['class_found']}")
        
        if result['has_required_methods']:
            print(f"  ✅ Required methods: {', '.join(result['has_required_methods'])}")
        
        if result['has_required_attributes']:
            print(f"  ✅ Required attributes: {', '.join(result['has_required_attributes'])}")
        
        if result['errors']:
            print(f"  ❌ Errors:")
            for error in result['errors']:
                print(f"     • {error}")
    
    def _print_summary(self):
        """Print verification summary"""
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        
        total_found = len(self.results['environments_found'])
        total_tested = len(self.results['environments_tested'])
        
        print(f"📁 Environment files found: {total_found}")
        print(f"🧪 Environment files tested: {total_tested}")
        
        if total_tested > 0:
            importable_count = sum(1 for result in self.results['verification_results'].values() 
                                 if result['importable'])
            pettingzoo_count = sum(1 for result in self.results['verification_results'].values() 
                                 if result['has_pettingzoo_base'])
            
            print(f"✅ Importable environments: {importable_count}/{total_tested}")
            print(f"✅ PettingZoo-based environments: {pettingzoo_count}/{total_tested}")
            
            # Method compliance
            required_methods = ['reset', 'step', 'observe', 'render', 'close']
            for method in required_methods:
                count = sum(1 for result in self.results['verification_results'].values() 
                           if method in result['has_required_methods'])
                print(f"   • {method}: {count}/{total_tested}")
            
            # Attribute compliance
            required_attrs = ['possible_agents', 'agents', 'action_spaces', 'observation_spaces']
            for attr in required_attrs:
                count = sum(1 for result in self.results['verification_results'].values() 
                           if attr in result['has_required_attributes'])
                print(f"   • {attr}: {count}/{total_tested}")
        
        # Overall assessment
        if total_tested > 0:
            good_envs = sum(1 for result in self.results['verification_results'].values() 
                           if result['importable'] and result['has_pettingzoo_base'] and 
                           len(result['has_required_methods']) >= 4)
            
            print(f"\n🎯 Overall: {good_envs}/{total_tested} environments appear well-structured")
            
            if good_envs == total_tested:
                print("🎉 ALL ENVIRONMENTS HAVE GOOD BASIC STRUCTURE!")
            else:
                print("⚠️  Some environments may need attention")
        else:
            print("❌ No environments found to test")


def create_basic_test():
    """Create a basic test that doesn't require full dependencies"""
    
    basic_test_code = '''#!/usr/bin/env python3
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
    print(f"\\n🎯 Test Results: {successful}/{len(results)} environments passed basic test")
    
    return results

if __name__ == "__main__":
    test_basic_instantiation()
'''
    
    # Write the basic test file
    test_file = PROJECT_ROOT / 'test_basic_environments.py'
    with open(test_file, 'w') as f:
        f.write(basic_test_code)
    
    print(f"✅ Created basic test file: {test_file}")
    return test_file


def main():
    """Main verification function"""
    
    print("🚀 Starting PettingZoo Environment Verification")
    print("=" * 60)
    
    # Run structure verification
    verifier = PettingZooVerifier()
    results = verifier.verify_environments()
    
    # Create basic test
    print("\n" + "=" * 60)
    print("🧪 Creating Basic Test File")
    print("=" * 60)
    
    test_file = create_basic_test()
    
    print(f"\n📝 Next Steps:")
    print(f"1. Run the basic test: python {test_file.name}")
    print(f"2. Check any errors reported above")
    print(f"3. Ensure all required methods are implemented")
    print(f"4. Test with actual PettingZoo API checker when ready")
    
    return results


if __name__ == "__main__":
    main()