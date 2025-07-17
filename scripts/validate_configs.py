#!/usr/bin/env python3
"""
GrandModel Configuration Validation Script
AGENT 5 - Configuration Recovery Mission
Validates all configuration files and reports any issues
"""

import sys
import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.config_manager import ConfigManager, Environment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigValidator:
    """Configuration validation utility"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate_yaml_file(self, file_path: Path) -> bool:
        """Validate YAML file syntax"""
        try:
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            self.errors.append(f"YAML syntax error in {file_path}: {e}")
            return False
        except FileNotFoundError:
            self.errors.append(f"Configuration file not found: {file_path}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")
            return False
            
    def validate_system_config(self, config: Dict[str, Any]) -> bool:
        """Validate system configuration"""
        valid = True
        
        # Required top-level sections
        required_sections = ['system']
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section: {section}")
                valid = False
                
        if 'system' not in config:
            return valid
            
        system_config = config['system']
        
        # Required system subsections
        required_subsections = ['security', 'database', 'redis', 'performance']
        for subsection in required_subsections:
            if subsection not in system_config:
                self.errors.append(f"Missing system subsection: {subsection}")
                valid = False
                
        # Validate database config
        if 'database' in system_config:
            db_config = system_config['database']
            required_db_fields = ['host', 'port', 'name', 'username', 'password']
            for field in required_db_fields:
                if field not in db_config:
                    self.errors.append(f"Missing database field: {field}")
                    valid = False
                    
        # Validate Redis config
        if 'redis' in system_config:
            redis_config = system_config['redis']
            required_redis_fields = ['host', 'port', 'db']
            for field in required_redis_fields:
                if field not in redis_config:
                    self.errors.append(f"Missing Redis field: {field}")
                    valid = False
                    
        return valid
        
    def validate_trading_config(self, config: Dict[str, Any], agent_type: str) -> bool:
        """Validate trading agent configuration"""
        valid = True
        
        # Expected agent key
        agent_key = f"{agent_type}_agent"
        if agent_key not in config:
            self.errors.append(f"Missing agent configuration: {agent_key}")
            return False
            
        agent_config = config[agent_key]
        
        # Required fields
        required_fields = ['name', 'type', 'behavior', 'rewards', 'learning']
        for field in required_fields:
            if field not in agent_config:
                self.errors.append(f"Missing {agent_type} agent field: {field}")
                valid = False
                
        # Validate reward weights sum to reasonable value
        if 'rewards' in agent_config:
            rewards = agent_config['rewards']
            weight_fields = [k for k in rewards.keys() if 'weight' in k]
            if weight_fields:
                total_weight = sum(rewards.get(field, 0) for field in weight_fields)
                if total_weight > 1.1 or total_weight < 0.9:
                    self.warnings.append(f"{agent_type} agent reward weights sum to {total_weight}, expected ~1.0")
                    
        return valid
        
    def validate_model_config(self, config: Dict[str, Any], model_type: str) -> bool:
        """Validate model configuration"""
        valid = True
        
        if model_type == 'mappo':
            required_sections = ['mappo', 'training', 'multi_agent']
            for section in required_sections:
                if section not in config:
                    self.errors.append(f"Missing MAPPO section: {section}")
                    valid = False
                    
        elif model_type == 'networks':
            required_sections = ['networks', 'input_processing', 'agent_networks']
            for section in required_sections:
                if section not in config:
                    self.errors.append(f"Missing networks section: {section}")
                    valid = False
                    
        elif model_type == 'hyperparameters':
            required_sections = ['hyperparameters', 'learning_rates', 'optimization']
            for section in required_sections:
                if section not in config:
                    self.errors.append(f"Missing hyperparameters section: {section}")
                    valid = False
                    
        return valid
        
    def validate_environment_config(self, config: Dict[str, Any], env_type: str) -> bool:
        """Validate environment configuration"""
        valid = True
        
        if env_type == 'market':
            env_key = 'market_environment'
        elif env_type == 'simulation':
            env_key = 'simulation_environment'
        else:
            self.errors.append(f"Unknown environment type: {env_type}")
            return False
            
        if env_key not in config:
            self.errors.append(f"Missing environment configuration: {env_key}")
            return False
            
        env_config = config[env_key]
        
        # Common required fields
        required_fields = ['name', 'version', 'type']
        for field in required_fields:
            if field not in env_config:
                self.errors.append(f"Missing {env_type} environment field: {field}")
                valid = False
                
        return valid
        
    def validate_all_configs(self) -> Tuple[bool, int, int]:
        """Validate all configuration files"""
        logger.info("Starting configuration validation...")
        
        configs_dir = Path(__file__).parent.parent / "configs"
        
        # Validate YAML syntax for all config files
        config_files = [
            configs_dir / "system" / "production.yaml",
            configs_dir / "system" / "development.yaml", 
            configs_dir / "system" / "testing.yaml",
            configs_dir / "trading" / "strategic_config.yaml",
            configs_dir / "trading" / "tactical_config.yaml",
            configs_dir / "trading" / "risk_config.yaml",
            configs_dir / "models" / "mappo_config.yaml",
            configs_dir / "models" / "network_config.yaml",
            configs_dir / "models" / "hyperparameters.yaml",
            configs_dir / "environments" / "market_config.yaml",
            configs_dir / "environments" / "simulation_config.yaml"
        ]
        
        for config_file in config_files:
            self.validate_yaml_file(config_file)
            
        # Validate configuration content using ConfigManager
        try:
            for env in [Environment.PRODUCTION, Environment.DEVELOPMENT, Environment.TESTING]:
                logger.info(f"Validating {env.value} configuration...")
                
                config_manager = ConfigManager(env)
                
                # Validate system config
                system_config = config_manager.get_system_config()
                self.validate_system_config({'system': system_config})
                
                # Validate trading configs
                for agent_type in ['strategic', 'tactical', 'risk']:
                    trading_config = config_manager.get_trading_config(agent_type)
                    self.validate_trading_config(trading_config, agent_type)
                    
                # Validate model configs
                for model_type in ['mappo', 'networks', 'hyperparameters']:
                    model_config = config_manager.get_model_config(model_type)
                    self.validate_model_config(model_config, model_type)
                    
                # Validate environment configs
                for env_type in ['market', 'simulation']:
                    env_config = config_manager.get_environment_config(env_type)
                    self.validate_environment_config(env_config, env_type)
                    
                # Run ConfigManager validation
                if not config_manager.validate_configs():
                    self.errors.append(f"ConfigManager validation failed for {env.value}")
                    
        except Exception as e:
            self.errors.append(f"Error during configuration validation: {e}")
            
        # Report results
        self.report_results()
        
        return len(self.errors) == 0, len(self.errors), len(self.warnings)
        
    def report_results(self):
        """Report validation results"""
        if self.errors:
            logger.error(f"Found {len(self.errors)} configuration errors:")
            for error in self.errors:
                logger.error(f"  ❌ {error}")
                
        if self.warnings:
            logger.warning(f"Found {len(self.warnings)} configuration warnings:")
            for warning in self.warnings:
                logger.warning(f"  ⚠️  {warning}")
                
        if not self.errors and not self.warnings:
            logger.info("✅ All configurations are valid!")
        elif not self.errors:
            logger.info("✅ All configurations are valid (with warnings)")
        else:
            logger.error("❌ Configuration validation failed")

def main():
    """Main validation function"""
    validator = ConfigValidator()
    
    success, num_errors, num_warnings = validator.validate_all_configs()
    
    print(f"\nValidation Summary:")
    print(f"  Errors: {num_errors}")
    print(f"  Warnings: {num_warnings}")
    print(f"  Status: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())