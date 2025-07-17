"""
GrandModel Configuration Manager
AGENT 5 - Configuration Recovery Mission
Centralized configuration management system for all MARL components
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class Environment(Enum):
    """Environment types"""
    PRODUCTION = "production"
    DEVELOPMENT = "development" 
    TESTING = "testing"

@dataclass
class ConfigPaths:
    """Configuration file paths"""
    base_dir: Path
    system_dir: Path
    trading_dir: Path
    models_dir: Path
    environments_dir: Path
    monitoring_dir: Path

class ConfigManager:
    """
    Centralized configuration management system
    Handles loading, validation, and access to all configuration files
    """
    
    def __init__(self, environment: Environment = Environment.PRODUCTION):
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.base_path = Path(__file__).parent.parent.parent / "configs"
        self.paths = ConfigPaths(
            base_dir=self.base_path,
            system_dir=self.base_path / "system",
            trading_dir=self.base_path / "trading", 
            models_dir=self.base_path / "models",
            environments_dir=self.base_path / "environments",
            monitoring_dir=self.base_path / "monitoring"
        )
        
        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
        
        # Load configurations
        self._load_all_configs()
        
    def _load_all_configs(self) -> None:
        """Load all configuration files"""
        try:
            # System configurations
            self._config_cache['system'] = self._load_yaml(
                self.paths.system_dir / f"{self.environment.value}.yaml"
            )
            
            # Trading configurations
            self._config_cache['strategic'] = self._load_yaml(
                self.paths.trading_dir / "strategic_config.yaml"
            )
            self._config_cache['tactical'] = self._load_yaml(
                self.paths.trading_dir / "tactical_config.yaml"
            )
            self._config_cache['risk'] = self._load_yaml(
                self.paths.trading_dir / "risk_config.yaml"
            )
            
            # Model configurations
            self._config_cache['mappo'] = self._load_yaml(
                self.paths.models_dir / "mappo_config.yaml"
            )
            self._config_cache['networks'] = self._load_yaml(
                self.paths.models_dir / "network_config.yaml"
            )
            self._config_cache['hyperparameters'] = self._load_yaml(
                self.paths.models_dir / "hyperparameters.yaml"
            )
            
            # Environment configurations
            self._config_cache['market'] = self._load_yaml(
                self.paths.environments_dir / "market_config.yaml"
            )
            self._config_cache['simulation'] = self._load_yaml(
                self.paths.environments_dir / "simulation_config.yaml"
            )
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            self.logger.info(f"Successfully loaded configurations for {self.environment.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
            raise
            
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not file_path.exists():
            self.logger.warning(f"Configuration file not found: {file_path}")
            return {}
            
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            return {}
            
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides"""
        
        # Database configuration
        if 'system' in self._config_cache:
            db_config = self._config_cache['system'].get('database', {})
            db_config['host'] = os.getenv('DB_HOST', db_config.get('host'))
            db_config['port'] = int(os.getenv('DB_PORT', db_config.get('port', 5432)))
            db_config['name'] = os.getenv('DB_NAME', db_config.get('name'))
            db_config['username'] = os.getenv('DB_USERNAME', db_config.get('username'))
            db_config['password'] = os.getenv('DB_PASSWORD', db_config.get('password'))
            
            # Redis configuration
            redis_config = self._config_cache['system'].get('redis', {})
            redis_config['host'] = os.getenv('REDIS_HOST', redis_config.get('host'))
            redis_config['port'] = int(os.getenv('REDIS_PORT', redis_config.get('port', 6379)))
            redis_config['password'] = os.getenv('REDIS_PASSWORD', redis_config.get('password'))
            
            # Security configuration
            security_config = self._config_cache['system'].get('security', {})
            security_config['jwt_secret_key'] = os.getenv('JWT_SECRET_KEY', 
                                                        security_config.get('jwt_secret_key'))
            
    def get_config(self, config_name: str, section: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """
        Get configuration by name and optional section
        
        Args:
            config_name: Name of the configuration (e.g., 'system', 'strategic')
            section: Optional section within the configuration
            
        Returns:
            Configuration dictionary or value
        """
        if config_name not in self._config_cache:
            self.logger.warning(f"Configuration '{config_name}' not found")
            return {}
            
        config = self._config_cache[config_name]
        
        if section:
            return config.get(section, {})
        
        return config
        
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.get_config('system')
        
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get_config('system', 'database')
        
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return self.get_config('system', 'redis')
        
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.get_config('system', 'security')
        
    def get_trading_config(self, agent_type: str) -> Dict[str, Any]:
        """
        Get trading configuration for specific agent
        
        Args:
            agent_type: Type of agent ('strategic', 'tactical', 'risk')
            
        Returns:
            Agent-specific trading configuration
        """
        if agent_type not in ['strategic', 'tactical', 'risk']:
            raise ValueError(f"Invalid agent type: {agent_type}")
            
        return self.get_config(agent_type)
        
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Get model configuration
        
        Args:
            model_type: Type of model ('mappo', 'networks', 'hyperparameters')
            
        Returns:
            Model configuration
        """
        return self.get_config(model_type)
        
    def get_environment_config(self, env_type: str) -> Dict[str, Any]:
        """
        Get environment configuration
        
        Args:
            env_type: Type of environment ('market', 'simulation')
            
        Returns:
            Environment configuration
        """
        return self.get_config(env_type)
        
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration values
        
        Args:
            config_name: Name of the configuration to update
            updates: Dictionary of updates to apply
        """
        if config_name not in self._config_cache:
            self.logger.warning(f"Configuration '{config_name}' not found for update")
            return
            
        self._deep_update(self._config_cache[config_name], updates)
        self.logger.info(f"Updated configuration '{config_name}'")
        
    def _deep_update(self, original: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep update dictionary"""
        for key, value in updates.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
                
    def validate_configs(self) -> bool:
        """
        Validate all loaded configurations
        
        Returns:
            True if all configurations are valid
        """
        try:
            # Validate required configurations exist
            required_configs = ['system', 'strategic', 'tactical', 'risk', 'mappo']
            for config_name in required_configs:
                if config_name not in self._config_cache:
                    self.logger.error(f"Required configuration '{config_name}' missing")
                    return False
                    
            # Validate system configuration
            system_config = self.get_system_config()
            required_system_sections = ['database', 'redis', 'security']
            for section in required_system_sections:
                if section not in system_config:
                    self.logger.error(f"Required system section '{section}' missing")
                    return False
                    
            # Validate database configuration
            db_config = self.get_database_config()
            required_db_fields = ['host', 'port', 'name', 'username', 'password']
            for field in required_db_fields:
                if not db_config.get(field):
                    self.logger.error(f"Required database field '{field}' missing")
                    return False
                    
            self.logger.info("All configurations validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
            
    def reload_configs(self) -> None:
        """Reload all configurations from files"""
        self.logger.info("Reloading configurations...")
        self._config_cache.clear()
        self._load_all_configs()
        
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all loaded configurations"""
        return self._config_cache.copy()
        
    def export_config(self, config_name: str, file_path: Path) -> None:
        """
        Export configuration to file
        
        Args:
            config_name: Name of the configuration to export
            file_path: Path to export file
        """
        if config_name not in self._config_cache:
            raise ValueError(f"Configuration '{config_name}' not found")
            
        with open(file_path, 'w') as f:
            yaml.dump(self._config_cache[config_name], f, default_flow_style=False)
            
        self.logger.info(f"Exported configuration '{config_name}' to {file_path}")

# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager(environment: Optional[Environment] = None) -> ConfigManager:
    """
    Get global configuration manager instance
    
    Args:
        environment: Environment type (only used for first initialization)
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        env = environment or Environment(os.getenv('ENVIRONMENT', 'production'))
        _config_manager = ConfigManager(env)
        
    return _config_manager

def get_config(config_name: str, section: Optional[str] = None) -> Union[Dict[str, Any], Any]:
    """
    Convenience function to get configuration
    
    Args:
        config_name: Name of the configuration
        section: Optional section within the configuration
        
    Returns:
        Configuration dictionary or value
    """
    return get_config_manager().get_config(config_name, section)