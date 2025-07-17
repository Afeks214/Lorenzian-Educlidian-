#!/usr/bin/env python3
"""
Comprehensive PettingZoo Environment Verification
===============================================

This script provides comprehensive verification of PettingZoo environments
without requiring actual PettingZoo installation by:
1. Analyzing source code structure
2. Creating mock PettingZoo classes
3. Testing environment instantiation with mocks
4. Validating required methods and attributes
"""

import sys
import os
import ast
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock PettingZoo classes for testing
class MockSpace:
    """Mock gymnasium/pettingzoo space"""
    def __init__(self, *args, **kwargs):
        pass
    
    def sample(self):
        return None
    
    def contains(self, x):
        return True

class MockBox(MockSpace):
    """Mock Box space"""
    def __init__(self, low, high, shape=None, dtype=None):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        super().__init__()

class MockDiscrete(MockSpace):
    """Mock Discrete space"""
    def __init__(self, n):
        self.n = n
        super().__init__()

class MockAECEnv:
    """Mock AEC Environment base class"""
    def __init__(self):
        self.agents = []
        self.possible_agents = []
        self.agent_selection = None
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.action_spaces = {}
        self.observation_spaces = {}
    
    def reset(self, seed=None, options=None):
        pass
    
    def step(self, action):
        pass
    
    def observe(self, agent):
        return None
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass

class MockAgentSelector:
    """Mock agent selector"""
    def __init__(self, agents):
        self.agents = agents
        self.current_idx = 0
    
    def next(self):
        if self.agents:
            agent = self.agents[self.current_idx % len(self.agents)]
            self.current_idx += 1
            return agent
        return None
    
    def reset(self):
        self.current_idx = 0
        return self.next()

# Mock imports
class MockPettingZoo:
    """Mock pettingzoo module"""
    AECEnv = MockAECEnv
    
    class utils:
        agent_selector = MockAgentSelector
        
        class wrappers:
            @staticmethod
            def OrderEnforcingWrapper(env):
                return env
            
            @staticmethod
            def AssertOutOfBoundsWrapper(env):
                return env

class MockGymnasium:
    """Mock gymnasium module"""
    class spaces:
        Box = MockBox
        Discrete = MockDiscrete
        Dict = dict

# Install mocks
sys.modules['pettingzoo'] = MockPettingZoo()
sys.modules['pettingzoo.utils'] = MockPettingZoo.utils()
sys.modules['pettingzoo.utils.wrappers'] = MockPettingZoo.utils.wrappers()
sys.modules['pettingzoo.utils.agent_selector'] = MockPettingZoo.utils()
sys.modules['gymnasium'] = MockGymnasium()
sys.modules['gymnasium.spaces'] = MockGymnasium.spaces()
sys.modules['gym'] = MockGymnasium()  # Also mock gym for compatibility

class PettingZooEnvironmentAnalyzer:
    """Comprehensive PettingZoo environment analyzer"""
    
    def __init__(self):
        self.analysis_results = {}
        self.test_results = {}
    
    def analyze_environment_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze a PettingZoo environment file
        
        Args:
            file_path: Path to environment file
            
        Returns:
            Analysis results
        """
        
        result = {
            'file_path': str(file_path),
            'exists': file_path.exists(),
            'source_analysis': {},
            'import_test': {},
            'instantiation_test': {},
            'method_analysis': {},
            'space_analysis': {},
            'agent_analysis': {},
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        if not result['exists']:
            result['errors'].append(f"File does not exist: {file_path}")
            return result
        
        # Analyze source code
        result['source_analysis'] = self._analyze_source_code(file_path)
        
        # Test imports
        result['import_test'] = self._test_imports(file_path)
        
        # Test instantiation
        if result['import_test']['success']:
            result['instantiation_test'] = self._test_instantiation(
                result['import_test']['module'],
                result['import_test']['env_class']
            )
        else:
            result['instantiation_test'] = {'success': False, 'error': 'Import failed'}
        
        # Analyze methods
        result['method_analysis'] = self._analyze_methods(result['source_analysis'])
        
        # Analyze spaces
        result['space_analysis'] = self._analyze_spaces(result['source_analysis'])
        
        # Analyze agents
        result['agent_analysis'] = self._analyze_agents(result['source_analysis'])
        
        # Generate recommendations
        result['recommendations'] = self._generate_recommendations(result)
        
        return result
    
    def _analyze_source_code(self, file_path: Path) -> Dict[str, Any]:
        """Analyze source code structure"""
        
        with open(file_path, 'r') as f:
            source = f.read()
        
        analysis = {
            'lines_of_code': len(source.splitlines()),
            'imports': self._extract_imports(source),
            'classes': self._extract_classes(source),
            'methods': self._extract_methods(source),
            'attributes': self._extract_attributes(source),
            'pettingzoo_references': self._find_pettingzoo_references(source)
        }
        
        return analysis
    
    def _extract_imports(self, source: str) -> List[str]:
        """Extract import statements"""
        imports = []
        for line in source.splitlines():
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        return imports
    
    def _extract_classes(self, source: str) -> List[Dict[str, Any]]:
        """Extract class definitions"""
        classes = []
        
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    base_classes = [base.id if isinstance(base, ast.Name) else str(base) 
                                  for base in node.bases]
                    classes.append({
                        'name': node.name,
                        'bases': base_classes,
                        'lineno': node.lineno
                    })
        except SyntaxError:
            # If AST parsing fails, use regex fallback
            class_pattern = r'class\s+(\w+)\s*\(([^)]*)\):'
            for match in re.finditer(class_pattern, source):
                classes.append({
                    'name': match.group(1),
                    'bases': [b.strip() for b in match.group(2).split(',') if b.strip()],
                    'lineno': source[:match.start()].count('\n') + 1
                })
        
        return classes
    
    def _extract_methods(self, source: str) -> List[Dict[str, Any]]:
        """Extract method definitions"""
        methods = []
        
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    methods.append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'is_method': len(node.args.args) > 0 and node.args.args[0].arg == 'self'
                    })
        except SyntaxError:
            # Regex fallback
            method_pattern = r'def\s+(\w+)\s*\([^)]*\):'
            for match in re.finditer(method_pattern, source):
                methods.append({
                    'name': match.group(1),
                    'lineno': source[:match.start()].count('\n') + 1,
                    'args': [],
                    'is_method': True
                })
        
        return methods
    
    def _extract_attributes(self, source: str) -> List[str]:
        """Extract attribute assignments"""
        attributes = []
        
        # Look for self.attribute = ... patterns
        attr_pattern = r'self\.(\w+)\s*='
        for match in re.finditer(attr_pattern, source):
            attr_name = match.group(1)
            if attr_name not in attributes:
                attributes.append(attr_name)
        
        return attributes
    
    def _find_pettingzoo_references(self, source: str) -> Dict[str, List[str]]:
        """Find PettingZoo-specific references"""
        references = {
            'aec_env': [],
            'agent_selector': [],
            'spaces': [],
            'wrappers': []
        }
        
        # Check for AECEnv
        if 'AECEnv' in source:
            references['aec_env'].append('AECEnv class reference found')
        
        # Check for agent_selector
        if 'agent_selector' in source:
            references['agent_selector'].append('agent_selector usage found')
        
        # Check for spaces
        if 'spaces.Box' in source or 'spaces.Discrete' in source:
            references['spaces'].append('Gymnasium/PettingZoo spaces found')
        
        # Check for wrappers
        if 'wrappers.' in source:
            references['wrappers'].append('PettingZoo wrappers found')
        
        return references
    
    def _test_imports(self, file_path: Path) -> Dict[str, Any]:
        """Test if file can be imported"""
        
        result = {
            'success': False,
            'module': None,
            'env_class': None,
            'env_class_name': None,
            'error': None
        }
        
        try:
            # Create module spec
            spec = importlib.util.spec_from_file_location("test_env", file_path)
            if spec is None:
                result['error'] = "Could not create module spec"
                return result
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute module
            spec.loader.exec_module(module)
            
            # Find environment class
            env_class = None
            env_class_name = None
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith('Env'):
                    env_class = obj
                    env_class_name = name
                    break
            
            if env_class is None:
                result['error'] = "No environment class found"
                return result
            
            result['success'] = True
            result['module'] = module
            result['env_class'] = env_class
            result['env_class_name'] = env_class_name
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _test_instantiation(self, module, env_class) -> Dict[str, Any]:
        """Test environment instantiation"""
        
        result = {
            'success': False,
            'env_instance': None,
            'agents': [],
            'action_spaces': {},
            'observation_spaces': {},
            'methods_available': [],
            'error': None
        }
        
        try:
            # Create minimal config
            config = self._create_minimal_config(env_class.__name__)
            
            # Try to instantiate
            env = env_class(config)
            
            # Check basic attributes
            if hasattr(env, 'possible_agents'):
                result['agents'] = env.possible_agents
            
            if hasattr(env, 'action_spaces'):
                result['action_spaces'] = dict(env.action_spaces)
            
            if hasattr(env, 'observation_spaces'):
                result['observation_spaces'] = dict(env.observation_spaces)
            
            # Check methods
            required_methods = ['reset', 'step', 'observe', 'render', 'close']
            for method in required_methods:
                if hasattr(env, method) and callable(getattr(env, method)):
                    result['methods_available'].append(method)
            
            result['success'] = True
            result['env_instance'] = env
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _create_minimal_config(self, class_name: str) -> Dict[str, Any]:
        """Create minimal configuration for environment"""
        
        if 'Tactical' in class_name:
            return {
                'tactical_marl': {
                    'environment': {
                        'matrix_shape': [60, 7],
                        'max_episode_steps': 10
                    }
                }
            }
        elif 'Strategic' in class_name:
            return {
                'matrix_shape': [48, 13],
                'max_timesteps': 10
            }
        elif 'Risk' in class_name:
            return {
                'initial_capital': 100000.0,
                'max_steps': 10,
                'risk_tolerance': 0.05,
                'asset_universe': ['SPY', 'QQQ']
            }
        elif 'Execution' in class_name:
            return {
                'max_steps': 10,
                'initial_portfolio_value': 100000.0
            }
        else:
            return {}
    
    def _analyze_methods(self, source_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze method compliance"""
        
        required_methods = ['reset', 'step', 'observe', 'render', 'close']
        methods_found = [m['name'] for m in source_analysis['methods']]
        
        return {
            'required_methods': required_methods,
            'methods_found': methods_found,
            'missing_methods': [m for m in required_methods if m not in methods_found],
            'extra_methods': [m for m in methods_found if m not in required_methods and not m.startswith('_')],
            'compliance_score': len([m for m in required_methods if m in methods_found]) / len(required_methods)
        }
    
    def _analyze_spaces(self, source_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze action/observation spaces"""
        
        attributes = source_analysis['attributes']
        
        return {
            'has_action_spaces': 'action_spaces' in attributes,
            'has_observation_spaces': 'observation_spaces' in attributes,
            'space_types_found': [imp for imp in source_analysis['imports'] if 'spaces' in imp],
            'compliance': 'action_spaces' in attributes and 'observation_spaces' in attributes
        }
    
    def _analyze_agents(self, source_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent configuration"""
        
        attributes = source_analysis['attributes']
        
        return {
            'has_possible_agents': 'possible_agents' in attributes,
            'has_agents': 'agents' in attributes,
            'has_agent_selector': 'agent_selector' in attributes,
            'agent_compliance': all(attr in attributes for attr in ['possible_agents', 'agents'])
        }
    
    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improvement"""
        
        recommendations = []
        
        # Check method compliance
        method_analysis = result['method_analysis']
        if method_analysis['missing_methods']:
            recommendations.append(f"Implement missing methods: {', '.join(method_analysis['missing_methods'])}")
        
        # Check space compliance
        space_analysis = result['space_analysis']
        if not space_analysis['has_action_spaces']:
            recommendations.append("Define action_spaces attribute")
        if not space_analysis['has_observation_spaces']:
            recommendations.append("Define observation_spaces attribute")
        
        # Check agent compliance
        agent_analysis = result['agent_analysis']
        if not agent_analysis['has_possible_agents']:
            recommendations.append("Define possible_agents attribute")
        if not agent_analysis['has_agents']:
            recommendations.append("Define agents attribute")
        
        # Check instantiation
        if result['instantiation_test'] and not result['instantiation_test']['success']:
            recommendations.append("Fix instantiation issues")
        
        return recommendations
    
    def analyze_all_environments(self) -> Dict[str, Any]:
        """Analyze all environments in the project"""
        
        print("🔍 Comprehensive PettingZoo Environment Analysis")
        print("=" * 70)
        
        # Find environment files
        env_files = [
            PROJECT_ROOT / 'environment' / 'tactical_env.py',
            PROJECT_ROOT / 'environment' / 'strategic_env.py',
            PROJECT_ROOT / 'src' / 'environment' / 'tactical_env.py',
            PROJECT_ROOT / 'src' / 'environment' / 'strategic_env.py',
            PROJECT_ROOT / 'src' / 'environment' / 'risk_env.py',
            PROJECT_ROOT / 'src' / 'environment' / 'execution_env.py',
        ]
        
        results = {}
        
        for env_file in env_files:
            if env_file.exists():
                print(f"\n📄 Analyzing {env_file.name}...")
                result = self.analyze_environment_file(env_file)
                results[str(env_file)] = result
                self._print_analysis_summary(env_file.name, result)
        
        # Print overall summary
        self._print_overall_summary(results)
        
        return results
    
    def _print_analysis_summary(self, filename: str, result: Dict[str, Any]):
        """Print analysis summary for a single file"""
        
        print(f"  📊 Source Analysis:")
        print(f"    • Lines of code: {result['source_analysis']['lines_of_code']}")
        print(f"    • Classes found: {len(result['source_analysis']['classes'])}")
        print(f"    • Methods found: {len(result['source_analysis']['methods'])}")
        
        print(f"  🔧 Method Compliance:")
        method_analysis = result['method_analysis']
        print(f"    • Score: {method_analysis['compliance_score']:.1%}")
        if method_analysis['missing_methods']:
            print(f"    • Missing: {', '.join(method_analysis['missing_methods'])}")
        
        print(f"  🎯 Agent Configuration:")
        agent_analysis = result['agent_analysis']
        print(f"    • Agent compliance: {agent_analysis['agent_compliance']}")
        
        print(f"  💻 Import Test:")
        import_test = result['import_test']
        print(f"    • Success: {import_test['success']}")
        if import_test['success']:
            print(f"    • Class: {import_test['env_class_name']}")
        
        print(f"  🏗️  Instantiation Test:")
        inst_test = result['instantiation_test']
        print(f"    • Success: {inst_test['success']}")
        if inst_test['success']:
            print(f"    • Agents: {len(inst_test.get('agents', []))}")
            print(f"    • Methods: {len(inst_test.get('methods_available', []))}/5")
        elif inst_test.get('error'):
            print(f"    • Error: {inst_test['error']}")
        
        if result['recommendations']:
            print(f"  💡 Recommendations:")
            for rec in result['recommendations']:
                print(f"    • {rec}")
    
    def _print_overall_summary(self, results: Dict[str, Any]):
        """Print overall analysis summary"""
        
        print(f"\n{'=' * 70}")
        print("🎯 OVERALL ANALYSIS SUMMARY")
        print("=" * 70)
        
        if not results:
            print("❌ No environments found")
            return
        
        total_envs = len(results)
        successful_imports = sum(1 for r in results.values() if r['import_test']['success'])
        successful_instantiations = sum(1 for r in results.values() if r['instantiation_test'].get('success', False))
        
        print(f"📊 Total environments analyzed: {total_envs}")
        print(f"✅ Successful imports: {successful_imports}/{total_envs}")
        print(f"✅ Successful instantiations: {successful_instantiations}/{total_envs}")
        
        # Method compliance
        avg_method_compliance = sum(r['method_analysis']['compliance_score'] for r in results.values()) / total_envs
        print(f"🔧 Average method compliance: {avg_method_compliance:.1%}")
        
        # Agent compliance
        agent_compliant = sum(1 for r in results.values() if r['agent_analysis']['agent_compliance'])
        print(f"🎯 Agent configuration compliance: {agent_compliant}/{total_envs}")
        
        # Overall assessment
        well_structured = sum(1 for r in results.values() if (
            r['import_test']['success'] and 
            r['instantiation_test'].get('success', False) and
            r['method_analysis']['compliance_score'] >= 0.8 and
            r['agent_analysis']['agent_compliance']
        ))
        
        print(f"\n🏆 Overall Assessment: {well_structured}/{total_envs} environments are well-structured")
        
        if well_structured == total_envs:
            print("🎉 ALL ENVIRONMENTS ARE WELL-STRUCTURED!")
        elif well_structured > 0:
            print("✅ Some environments are well-structured")
        else:
            print("⚠️  All environments need improvements")


def main():
    """Main analysis function"""
    
    analyzer = PettingZooEnvironmentAnalyzer()
    results = analyzer.analyze_all_environments()
    
    print(f"\n{'=' * 70}")
    print("🏁 Analysis Complete")
    print("=" * 70)
    
    print(f"📝 Next Steps:")
    print(f"1. Review recommendations above")
    print(f"2. Install PettingZoo for full testing: pip install pettingzoo")
    print(f"3. Run official PettingZoo API tests")
    print(f"4. Test with actual MARL training")
    
    return results


if __name__ == "__main__":
    main()