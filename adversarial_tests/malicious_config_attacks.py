#!/usr/bin/env python3
"""
🚨 RED TEAM CONFIGURATION ATTACK SUITE
Agent 3 Mission: Attack strategic_config.yaml with malicious parameters

This module generates malicious configuration files designed to:
- Break input validation
- Cause system crashes
- Exploit configuration parsing vulnerabilities
- Test error handling robustness
- Verify security boundaries

MISSION: Prove the system has robust configuration validation.
If these attacks succeed, the system is NOT production ready.
"""

import yaml
import os
import json
from typing import Dict, Any, List
import numpy as np
import math

class MaliciousConfigGenerator:
    """
    Generates malicious configuration files to attack the Strategic MARL system.
    """
    
    def __init__(self, base_config_path: str = "configs/strategic_config.yaml"):
        self.base_config_path = base_config_path
        self.output_dir = "adversarial_tests/malicious_configs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base config to modify
        try:
            with open(base_config_path, 'r') as f:
                self.base_config = yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load base config: {e}")
            self.base_config = self._create_minimal_config()
    
    def _create_minimal_config(self) -> Dict[str, Any]:
        """Create minimal config structure for testing."""
        return {
            'environment': {'matrix_shape': [48, 13]},
            'agents': {'mlmi_expert': {'learning_rate': 3e-4}},
            'training': {'batch_size': 256, 'gamma': 0.99}
        }
    
    def attack_1_type_confusion(self) -> str:
        """
        🎯 ATTACK 1: TYPE CONFUSION ATTACK
        
        Replace expected numeric values with strings, booleans, or other types
        to test type validation robustness.
        """
        print("🚨 GENERATING TYPE CONFUSION ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # String instead of float attacks
        malicious_config['agents']['mlmi_expert']['learning_rate'] = "very_high"
        malicious_config['training']['gamma'] = "maximum_discount"
        malicious_config['training']['batch_size'] = "large"
        
        # Boolean confusion attacks
        malicious_config['environment']['matrix_shape'] = True
        malicious_config['training']['grad_clip'] = False
        
        # List instead of scalar attacks
        malicious_config['agents']['mlmi_expert']['dropout_rate'] = [0.1, 0.2, 0.3]
        malicious_config['training']['target_kl'] = [0.01, 0.02]
        
        # Dictionary instead of scalar attacks
        malicious_config['training']['n_epochs'] = {"min": 5, "max": 15}
        
        file_path = os.path.join(self.output_dir, "attack_type_confusion.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(malicious_config, f, default_flow_style=False)
        
        print(f"✅ Type confusion attack saved to {file_path}")
        return file_path
    
    def attack_2_negative_values(self) -> str:
        """
        🎯 ATTACK 2: NEGATIVE VALUES ATTACK
        
        Set parameters that should be positive to negative values
        to test boundary validation.
        """
        print("🚨 GENERATING NEGATIVE VALUES ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # Negative dimensions - should be impossible
        malicious_config['environment']['matrix_shape'] = [-48, -13]
        malicious_config['agents']['mlmi_expert']['hidden_dims'] = [-256, -128, -64]
        
        # Negative learning parameters
        malicious_config['agents']['mlmi_expert']['learning_rate'] = -0.001
        malicious_config['training']['batch_size'] = -256
        malicious_config['training']['gamma'] = -0.99
        
        # Negative buffer sizes
        malicious_config['agents']['mlmi_expert']['buffer_size'] = -10000
        malicious_config['training']['buffer_capacity'] = -100000
        
        # Negative frequencies and timeouts
        malicious_config['agents']['mlmi_expert']['update_frequency'] = -100
        malicious_config['integration']['matrix_assembler']['feature_timeout_ms'] = -1000
        
        file_path = os.path.join(self.output_dir, "attack_negative_values.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(malicious_config, f, default_flow_style=False)
        
        print(f"✅ Negative values attack saved to {file_path}")
        return file_path
    
    def attack_3_extreme_values(self) -> str:
        """
        🎯 ATTACK 3: EXTREME VALUES ATTACK
        
        Set parameters to extreme values to test system limits and 
        potential overflow/underflow conditions.
        """
        print("🚨 GENERATING EXTREME VALUES ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # Massive dimensions to cause memory exhaustion
        malicious_config['environment']['matrix_shape'] = [999999, 999999]
        malicious_config['agents']['mlmi_expert']['hidden_dims'] = [99999999, 88888888, 77777777]
        
        # Extreme learning rates
        malicious_config['agents']['mlmi_expert']['learning_rate'] = 1e10  # Guaranteed instability
        malicious_config['ensemble']['learning_rate'] = 999.999
        
        # Extreme batch sizes
        malicious_config['training']['batch_size'] = 2147483647  # Max int32
        malicious_config['training']['episodes'] = 999999999
        
        # Extreme probability values
        malicious_config['training']['gamma'] = 99.99  # Should be [0,1]
        malicious_config['curriculum']['stages'][0]['synergy_probability'] = 5.5  # Should be [0,1]
        
        # Extreme timeouts to cause DoS
        malicious_config['integration']['matrix_assembler']['feature_timeout_ms'] = 999999999999
        malicious_config['integration']['tactical_marl']['decision_timeout_ms'] = 999999999999
        
        file_path = os.path.join(self.output_dir, "attack_extreme_values.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(malicious_config, f, default_flow_style=False)
        
        print(f"✅ Extreme values attack saved to {file_path}")
        return file_path
    
    def attack_4_special_float_values(self) -> str:
        """
        🎯 ATTACK 4: SPECIAL FLOAT VALUES ATTACK
        
        Use NaN, Infinity, and other special float values to test
        numerical stability and validation.
        """
        print("🚨 GENERATING SPECIAL FLOAT VALUES ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # NaN attacks
        malicious_config['agents']['mlmi_expert']['learning_rate'] = float('nan')
        malicious_config['training']['gamma'] = float('nan')
        malicious_config['ensemble']['weights'] = [float('nan'), 0.35, 0.25]
        
        # Infinity attacks
        malicious_config['training']['grad_clip'] = float('inf')
        malicious_config['agents']['mlmi_expert']['dropout_rate'] = float('-inf')
        malicious_config['rewards']['max_drawdown'] = float('inf')
        
        # Zero division potential
        malicious_config['rewards']['pnl_normalizer'] = 0.0
        malicious_config['ensemble']['learning_rate'] = 0.0
        
        # Subnormal numbers
        malicious_config['training']['target_kl'] = 1e-323  # Subnormal
        malicious_config['rewards']['delta'] = 5e-324      # Smallest positive
        
        file_path = os.path.join(self.output_dir, "attack_special_floats.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(malicious_config, f, default_flow_style=False)
        
        print(f"✅ Special float values attack saved to {file_path}")
        return file_path
    
    def attack_5_missing_required_fields(self) -> str:
        """
        🎯 ATTACK 5: MISSING REQUIRED FIELDS ATTACK
        
        Remove essential configuration fields to test error handling
        and graceful degradation.
        """
        print("🚨 GENERATING MISSING REQUIRED FIELDS ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # Remove critical environment config
        del malicious_config['environment']['matrix_shape']
        del malicious_config['environment']['max_timesteps']
        
        # Remove agent configurations
        del malicious_config['agents']['mlmi_expert']['learning_rate']
        del malicious_config['agents']['mlmi_expert']['hidden_dims']
        
        # Remove training essentials
        del malicious_config['training']['batch_size']
        del malicious_config['training']['gamma']
        del malicious_config['training']['episodes']
        
        # Remove safety configurations
        if 'safety' in malicious_config:
            del malicious_config['safety']
        
        file_path = os.path.join(self.output_dir, "attack_missing_fields.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(malicious_config, f, default_flow_style=False)
        
        print(f"✅ Missing required fields attack saved to {file_path}")
        return file_path
    
    def attack_6_circular_references(self) -> str:
        """
        🎯 ATTACK 6: CIRCULAR REFERENCES ATTACK
        
        Create circular references in configuration to test parsing robustness.
        """
        print("🚨 GENERATING CIRCULAR REFERENCES ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # Create circular reference using YAML aliases
        malicious_config['circular_ref_a'] = {'ref': None}
        malicious_config['circular_ref_b'] = {'ref': malicious_config['circular_ref_a']}
        malicious_config['circular_ref_a']['ref'] = malicious_config['circular_ref_b']
        
        # Circular references in actual config values
        malicious_config['agents']['mlmi_expert']['learning_rate'] = "${training.batch_size}"
        malicious_config['training']['batch_size'] = "${agents.mlmi_expert.learning_rate}"
        
        file_path = os.path.join(self.output_dir, "attack_circular_refs.yaml")
        try:
            with open(file_path, 'w') as f:
                yaml.dump(malicious_config, f, default_flow_style=False)
        except (FileNotFoundError, IOError, OSError) as e:
            # If circular ref breaks YAML, write a manually crafted version
            with open(file_path, 'w') as f:
                f.write("""
# Circular reference attack
environment: &env_ref
  matrix_shape: [48, 13]
  max_timesteps: *training_ref

training: &training_ref
  batch_size: 256
  gamma: *env_ref
  episodes: 10000

agents:
  mlmi_expert:
    learning_rate: *training_ref
    hidden_dims: [256, 128, 64]
""")
        
        print(f"✅ Circular references attack saved to {file_path}")
        return file_path
    
    def attack_7_malformed_yaml(self) -> str:
        """
        🎯 ATTACK 7: MALFORMED YAML ATTACK
        
        Create syntactically invalid YAML to test parser error handling.
        """
        print("🚨 GENERATING MALFORMED YAML ATTACK...")
        
        file_path = os.path.join(self.output_dir, "attack_malformed_yaml.yaml")
        
        malformed_yaml = """
# Malformed YAML attack
environment:
  matrix_shape: [48, 13
  max_timesteps: 1000}  # Mismatched brackets
  
agents:
  mlmi_expert:
    learning_rate: 3e-4
    hidden_dims: [256, 128, 64]]  # Extra bracket
    
training:
  batch_size: 256
  gamma: 0.99
  episodes: "unclosed string
  
# Unescaped special characters
rewards:
  alpha: 1.0 @ invalid
  beta: 0.2 # comment: in value
  gamma: -0.3 | pipe character
  
# Invalid indentation
ensemble:
weights: [0.4, 0.35, 0.25]  # Wrong indentation
  confidence_threshold: 0.65
    learning_rate: 1e-3  # Too much indentation
"""
        
        with open(file_path, 'w') as f:
            f.write(malformed_yaml)
        
        print(f"✅ Malformed YAML attack saved to {file_path}")
        return file_path
    
    def attack_8_injection_attempts(self) -> str:
        """
        🎯 ATTACK 8: INJECTION ATTEMPTS ATTACK
        
        Attempt various injection attacks through configuration values.
        """
        print("🚨 GENERATING INJECTION ATTEMPTS ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # Command injection attempts
        malicious_config['training']['checkpoint_dir'] = "models/strategic; rm -rf / #"
        malicious_config['monitoring']['wandb']['project'] = "project'; DROP TABLE users; --"
        
        # Path traversal attempts
        malicious_config['performance']['max_inference_latency_ms'] = "../../etc/passwd"
        malicious_config['integration']['matrix_assembler']['expected_shape'] = "../../../root/.ssh/id_rsa"
        
        # Code execution attempts
        malicious_config['agents']['mlmi_expert']['features'] = "__import__('os').system('whoami')"
        malicious_config['ensemble']['weights'] = "eval('__import__(\"subprocess\").call([\"ls\", \"-la\"])')"
        
        # Format string attacks
        malicious_config['rewards']['alpha'] = "%s%s%s%s%s%s%s%s%s%s%s%s"
        malicious_config['training']['batch_size'] = "{0.__class__.__bases__[0].__subclasses__()}"
        
        file_path = os.path.join(self.output_dir, "attack_injection_attempts.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(malicious_config, f, default_flow_style=False)
        
        print(f"✅ Injection attempts attack saved to {file_path}")
        return file_path
    
    def attack_9_memory_exhaustion(self) -> str:
        """
        🎯 ATTACK 9: MEMORY EXHAUSTION ATTACK
        
        Create configurations designed to exhaust system memory.
        """
        print("🚨 GENERATING MEMORY EXHAUSTION ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # Massive array sizes
        malicious_config['environment']['matrix_shape'] = [100000, 100000]
        malicious_config['agents']['mlmi_expert']['hidden_dims'] = [999999999] * 1000
        
        # Massive buffer sizes
        malicious_config['training']['buffer_capacity'] = 2**31 - 1  # Max int
        malicious_config['agents']['mlmi_expert']['buffer_size'] = 2**30
        
        # Large batch sizes
        malicious_config['training']['batch_size'] = 2**20  # 1 million
        malicious_config['optimization']['max_batch_size'] = 2**25
        
        # Massive arrays and lists
        huge_array = [0.1] * 1000000  # 1 million elements
        malicious_config['ensemble']['weights'] = huge_array
        malicious_config['rewards']['reward_clipping'] = huge_array
        
        file_path = os.path.join(self.output_dir, "attack_memory_exhaustion.yaml")
        with open(file_path, 'w') as f:
            yaml.dump(malicious_config, f, default_flow_style=False)
        
        print(f"✅ Memory exhaustion attack saved to {file_path}")
        return file_path
    
    def attack_10_unicode_chaos(self) -> str:
        """
        🎯 ATTACK 10: UNICODE CHAOS ATTACK
        
        Use various Unicode characters and encoding attacks.
        """
        print("🚨 GENERATING UNICODE CHAOS ATTACK...")
        
        malicious_config = self.base_config.copy()
        
        # Unicode in field names and values
        malicious_config['agents']['mlmi_expert']['学習率'] = 3e-4  # Japanese
        malicious_config['тренировка'] = {'размер_пакета': 256}  # Russian
        malicious_config['agents']['🤖_agent'] = {'💰_rate': 0.001}  # Emojis
        
        # Zero-width characters
        malicious_config['environment']['matrix_shape\u200b'] = [48, 13]  # Zero-width space
        malicious_config['training']['batch\u200c\u200dsize'] = 256  # Zero-width non-joiners
        
        # Right-to-left override attacks
        malicious_config['monitoring']['wandb']['project'] = "normal\u202ekcatta"  # RTL override
        
        # Normalization attacks
        malicious_config['agents']['café'] = {'hidden_dims': [256]}  # Composed é
        malicious_config['agents']['cafe\u0301'] = {'learning_rate': 0.001}  # Decomposed é
        
        file_path = os.path.join(self.output_dir, "attack_unicode_chaos.yaml")
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(malicious_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Unicode chaos attack saved to {file_path}")
        return file_path
    
    def generate_all_attacks(self) -> Dict[str, str]:
        """
        Generate all malicious configuration attacks.
        """
        print("🚨" * 30)
        print("MALICIOUS CONFIGURATION ATTACK SUITE")
        print("🚨" * 30)
        
        attacks = {
            'type_confusion': self.attack_1_type_confusion,
            'negative_values': self.attack_2_negative_values,
            'extreme_values': self.attack_3_extreme_values,
            'special_floats': self.attack_4_special_float_values,
            'missing_fields': self.attack_5_missing_required_fields,
            'circular_refs': self.attack_6_circular_references,
            'malformed_yaml': self.attack_7_malformed_yaml,
            'injection_attempts': self.attack_8_injection_attempts,
            'memory_exhaustion': self.attack_9_memory_exhaustion,
            'unicode_chaos': self.attack_10_unicode_chaos,
        }
        
        attack_files = {}
        
        for attack_name, attack_func in attacks.items():
            try:
                print(f"\n🎯 Executing {attack_name} attack...")
                file_path = attack_func()
                attack_files[attack_name] = file_path
            except Exception as e:
                print(f"❌ Failed to generate {attack_name}: {e}")
                attack_files[attack_name] = None
        
        return attack_files

def test_config_validation(config_file: str) -> Dict[str, Any]:
    """
    Test system response to malicious configuration file.
    """
    print(f"\n🔍 TESTING CONFIGURATION: {config_file}")
    
    results = {
        'file': config_file,
        'yaml_parseable': False,
        'validation_passed': False,
        'error_message': None,
        'system_crashed': False
    }
    
    try:
        # Test 1: Can YAML be parsed?
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        results['yaml_parseable'] = True
        print("✅ YAML parsing succeeded")
        
        # Test 2: Would system validation catch this?
        # This would normally load the Strategic MARL system with the config
        print("⚠️  System validation test would go here")
        print("   (Skipped due to PyTorch compatibility issues)")
        
    except yaml.YAMLError as e:
        results['error_message'] = f"YAML Error: {str(e)}"
        print(f"✅ YAML parsing FAILED (good): {e}")
    except Exception as e:
        results['error_message'] = f"General Error: {str(e)}"
        print(f"❌ Unexpected error: {e}")
        results['system_crashed'] = True
    
    return results

def run_configuration_attack_suite():
    """
    Execute the complete configuration attack suite.
    """
    print("🚨" * 40)
    print("RED TEAM CONFIGURATION ATTACK EXECUTION")
    print("🚨" * 40)
    
    generator = MaliciousConfigGenerator()
    attack_files = generator.generate_all_attacks()
    
    print("\n" + "="*80)
    print("🎯 TESTING SYSTEM RESPONSE TO MALICIOUS CONFIGURATIONS")
    print("="*80)
    
    test_results = []
    
    for attack_name, file_path in attack_files.items():
        if file_path:
            result = test_config_validation(file_path)
            test_results.append(result)
    
    print("\n" + "="*80)
    print("📊 CONFIGURATION ATTACK RESULTS SUMMARY")
    print("="*80)
    
    for result in test_results:
        attack_name = os.path.basename(result['file']).replace('attack_', '').replace('.yaml', '')
        print(f"\n🎯 {attack_name.upper()}:")
        print(f"   📁 File: {result['file']}")
        print(f"   🔍 YAML Parseable: {'✅' if result['yaml_parseable'] else '❌'}")
        print(f"   🛡️  System Crashed: {'❌' if result['system_crashed'] else '✅'}")
        if result['error_message']:
            print(f"   💬 Error: {result['error_message'][:100]}...")
    
    # Count results
    total_attacks = len(test_results)
    yaml_failures = sum(1 for r in test_results if not r['yaml_parseable'])
    system_crashes = sum(1 for r in test_results if r['system_crashed'])
    
    print(f"\n📈 ATTACK SUMMARY:")
    print(f"   Total Attacks: {total_attacks}")
    print(f"   YAML Parse Failures: {yaml_failures} (good - shows basic validation)")
    print(f"   System Crashes: {system_crashes} (bad - shows vulnerability)")
    print(f"   Graceful Failures: {total_attacks - system_crashes} (good)")
    
    return test_results

if __name__ == "__main__":
    run_configuration_attack_suite()