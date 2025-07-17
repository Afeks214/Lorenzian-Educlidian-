"""
Advanced Configuration Management System
Centralized configuration management with environment-specific configurations,
versioning, rollback capabilities, and runtime updates.
"""

import os
import json
import yaml
import logging
import hashlib
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager

from .config_validator import ConfigValidator
from .secrets_manager import SecretsManager


class Environment(Enum):
    """Environment types"""
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TESTING = "testing"


@dataclass
class ConfigVersion:
    """Configuration version metadata"""
    version: str
    timestamp: datetime
    checksum: str
    author: str
    description: str
    config_data: Dict[str, Any]


@dataclass
class ConfigChangeEvent:
    """Configuration change event"""
    config_name: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    user: str
    reason: str


class ConfigManager:
    """
    Advanced configuration management system with:
    - Environment-specific configurations
    - Configuration versioning and rollback
    - Runtime configuration updates
    - Configuration validation
    - Change event tracking
    """

    def __init__(self, environment: Environment = Environment.PRODUCTION,
                 base_path: Optional[Path] = None):
        self.environment = environment
        self.base_path = base_path or Path(__file__).parent.parent.parent / "config"
        self.logger = logging.getLogger(__name__)
        
        # Configuration storage
        self._configs: Dict[str, Any] = {}
        self._config_versions: Dict[str, List[ConfigVersion]] = {}
        self._change_listeners: List[Callable[[ConfigChangeEvent], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize components
        self.validator = ConfigValidator()
        self.secrets_manager = SecretsManager()
        
        # Configuration paths
        self.config_paths = {
            'base': self.base_path,
            'environments': self.base_path / "environments",
            'schemas': self.base_path / "schemas",
            'versions': self.base_path / "versions",
            'backups': self.base_path / "backups"
        }
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Load configurations
        self._load_configurations()
        
        self.logger.info(f"ConfigManager initialized for {environment.value}")

    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for path in self.config_paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def _load_configurations(self):
        """Load all configurations for the current environment"""
        with self._lock:
            try:
                # Load base configurations
                self._load_base_configs()
                
                # Load environment-specific configurations
                self._load_environment_configs()
                
                # Apply secrets
                self._apply_secrets()
                
                # Validate configurations
                self._validate_all_configs()
                
                self.logger.info("All configurations loaded successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to load configurations: {e}")
                raise

    def _load_base_configs(self):
        """Load base configuration files"""
        base_files = [
            "settings.yaml",
            "data_pipeline.yaml",
            "model_configs.yaml",
            "risk_management_config.yaml"
        ]
        
        for file_name in base_files:
            file_path = self.config_paths['base'] / file_name
            if file_path.exists():
                config_name = file_name.replace('.yaml', '').replace('.yml', '')
                self._configs[config_name] = self._load_yaml_file(file_path)

    def _load_environment_configs(self):
        """Load environment-specific configuration overrides"""
        env_dir = self.config_paths['environments'] / self.environment.value
        if env_dir.exists():
            for config_file in env_dir.glob("*.yaml"):
                config_name = config_file.stem
                env_config = self._load_yaml_file(config_file)
                
                # Merge with base config
                if config_name in self._configs:
                    self._configs[config_name] = self._deep_merge(
                        self._configs[config_name], env_config
                    )
                else:
                    self._configs[config_name] = env_config

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file with error handling"""
        try:
            with open(file_path, 'r') as f:
                content = yaml.safe_load(f) or {}
                
            # Process environment variable substitutions
            return self._process_env_vars(content)
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            return {}

    def _process_env_vars(self, obj: Any) -> Any:
        """Process environment variable substitutions in configuration"""
        if isinstance(obj, dict):
            return {k: self._process_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._process_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Handle ${VAR:default} pattern
            if obj.startswith("${") and obj.endswith("}"):
                var_spec = obj[2:-1]
                if ":" in var_spec:
                    var_name, default_value = var_spec.split(":", 1)
                    return os.getenv(var_name, default_value)
                else:
                    return os.getenv(var_spec, obj)
        return obj

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result

    def _apply_secrets(self):
        """Apply secrets to configurations"""
        for config_name, config_data in self._configs.items():
            self._configs[config_name] = self.secrets_manager.apply_secrets(config_data)

    def _validate_all_configs(self):
        """Validate all loaded configurations"""
        for config_name, config_data in self._configs.items():
            if not self.validator.validate_config(config_name, config_data):
                self.logger.warning(f"Configuration validation failed for {config_name}")

    def get_config(self, config_name: str, section: Optional[str] = None) -> Any:
        """Get configuration value"""
        with self._lock:
            if config_name not in self._configs:
                self.logger.warning(f"Configuration '{config_name}' not found")
                return {}
                
            config = self._configs[config_name]
            
            if section:
                return config.get(section, {})
            
            return config

    def set_config(self, config_name: str, config_data: Dict[str, Any], 
                   user: str = "system", reason: str = "Runtime update"):
        """Set configuration with versioning and validation"""
        with self._lock:
            # Validate new configuration
            if not self.validator.validate_config(config_name, config_data):
                raise ValueError(f"Configuration validation failed for {config_name}")
            
            # Create version if config exists
            if config_name in self._configs:
                self._create_version(config_name, self._configs[config_name], user, reason)
            
            # Store old value for change event
            old_value = self._configs.get(config_name)
            
            # Update configuration
            self._configs[config_name] = config_data
            
            # Create change event
            event = ConfigChangeEvent(
                config_name=config_name,
                old_value=old_value,
                new_value=config_data,
                timestamp=datetime.now(),
                user=user,
                reason=reason
            )
            
            # Notify listeners
            self._notify_change_listeners(event)
            
            # Save to disk
            self._save_config_to_disk(config_name, config_data)
            
            self.logger.info(f"Configuration '{config_name}' updated by {user}: {reason}")

    def update_config(self, config_name: str, updates: Dict[str, Any], 
                      user: str = "system", reason: str = "Runtime update"):
        """Update specific configuration values"""
        with self._lock:
            if config_name not in self._configs:
                raise ValueError(f"Configuration '{config_name}' not found")
            
            # Create a deep copy and apply updates
            updated_config = self._deep_merge(self._configs[config_name], updates)
            
            # Set the updated configuration
            self.set_config(config_name, updated_config, user, reason)

    def _create_version(self, config_name: str, config_data: Dict[str, Any], 
                       user: str, reason: str):
        """Create a version of the configuration"""
        version_id = f"v{len(self._config_versions.get(config_name, []))+1}"
        checksum = self._calculate_checksum(config_data)
        
        version = ConfigVersion(
            version=version_id,
            timestamp=datetime.now(),
            checksum=checksum,
            author=user,
            description=reason,
            config_data=config_data
        )
        
        if config_name not in self._config_versions:
            self._config_versions[config_name] = []
        
        self._config_versions[config_name].append(version)
        
        # Save version to disk
        self._save_version_to_disk(config_name, version)

    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate MD5 checksum of configuration data"""
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get_versions(self, config_name: str) -> List[ConfigVersion]:
        """Get all versions of a configuration"""
        return self._config_versions.get(config_name, [])

    def rollback_config(self, config_name: str, version: str, 
                       user: str = "system", reason: str = "Rollback"):
        """Rollback configuration to a specific version"""
        with self._lock:
            versions = self._config_versions.get(config_name, [])
            target_version = None
            
            for v in versions:
                if v.version == version:
                    target_version = v
                    break
            
            if not target_version:
                raise ValueError(f"Version '{version}' not found for '{config_name}'")
            
            # Set configuration to the target version
            self.set_config(config_name, target_version.config_data, user, 
                          f"Rollback to {version}: {reason}")

    def _save_config_to_disk(self, config_name: str, config_data: Dict[str, Any]):
        """Save configuration to disk"""
        env_dir = self.config_paths['environments'] / self.environment.value
        env_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = env_dir / f"{config_name}.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def _save_version_to_disk(self, config_name: str, version: ConfigVersion):
        """Save version to disk"""
        version_dir = self.config_paths['versions'] / config_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        version_file = version_dir / f"{version.version}.json"
        
        with open(version_file, 'w') as f:
            json.dump(asdict(version), f, indent=2, default=str)

    def add_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """Add a configuration change listener"""
        self._change_listeners.append(listener)

    def remove_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """Remove a configuration change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)

    def _notify_change_listeners(self, event: ConfigChangeEvent):
        """Notify all change listeners"""
        for listener in self._change_listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"Error in change listener: {e}")

    def reload_config(self, config_name: str):
        """Reload a specific configuration from disk"""
        with self._lock:
            # Reload base config
            base_file = self.config_paths['base'] / f"{config_name}.yaml"
            if base_file.exists():
                base_config = self._load_yaml_file(base_file)
            else:
                base_config = {}
            
            # Reload environment-specific config
            env_file = (self.config_paths['environments'] / 
                       self.environment.value / f"{config_name}.yaml")
            if env_file.exists():
                env_config = self._load_yaml_file(env_file)
                config_data = self._deep_merge(base_config, env_config)
            else:
                config_data = base_config
            
            # Apply secrets and validate
            config_data = self.secrets_manager.apply_secrets(config_data)
            
            if self.validator.validate_config(config_name, config_data):
                self._configs[config_name] = config_data
                self.logger.info(f"Configuration '{config_name}' reloaded")
            else:
                self.logger.error(f"Failed to reload '{config_name}': validation failed")

    def reload_all_configs(self):
        """Reload all configurations from disk"""
        with self._lock:
            self._configs.clear()
            self._load_configurations()

    def backup_configs(self, backup_name: Optional[str] = None):
        """Create a backup of all configurations"""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = self.config_paths['backups'] / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup all configurations
        for config_name, config_data in self._configs.items():
            backup_file = backup_dir / f"{config_name}.yaml"
            with open(backup_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        
        # Backup metadata
        metadata = {
            'environment': self.environment.value,
            'timestamp': datetime.now().isoformat(),
            'configs': list(self._configs.keys())
        }
        
        metadata_file = backup_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Configuration backup created: {backup_name}")
        return backup_name

    def restore_configs(self, backup_name: str, user: str = "system"):
        """Restore configurations from backup"""
        backup_dir = self.config_paths['backups'] / backup_name
        
        if not backup_dir.exists():
            raise ValueError(f"Backup '{backup_name}' not found")
        
        # Load metadata
        metadata_file = backup_dir / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Backup metadata not found for '{backup_name}'")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Restore configurations
        for config_name in metadata['configs']:
            config_file = backup_dir / f"{config_name}.yaml"
            if config_file.exists():
                config_data = self._load_yaml_file(config_file)
                self.set_config(config_name, config_data, user, 
                              f"Restored from backup: {backup_name}")
        
        self.logger.info(f"Configurations restored from backup: {backup_name}")

    @contextmanager
    def config_transaction(self, config_name: str):
        """Context manager for configuration transactions"""
        with self._lock:
            # Save current state
            original_config = self._configs.get(config_name, {}).copy()
            
            try:
                yield
            except Exception:
                # Rollback on error
                if config_name in self._configs:
                    self._configs[config_name] = original_config
                raise

    def get_config_diff(self, config_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """Get difference between two configuration versions"""
        versions = self._config_versions.get(config_name, [])
        
        v1_data = None
        v2_data = None
        
        for v in versions:
            if v.version == version1:
                v1_data = v.config_data
            elif v.version == version2:
                v2_data = v.config_data
        
        if v1_data is None or v2_data is None:
            raise ValueError("One or both versions not found")
        
        return self._calculate_diff(v1_data, v2_data)

    def _calculate_diff(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between two dictionaries"""
        diff = {}
        
        # Check for changed/added keys
        for key in dict2:
            if key not in dict1:
                diff[key] = {'type': 'added', 'value': dict2[key]}
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    subdiff = self._calculate_diff(dict1[key], dict2[key])
                    if subdiff:
                        diff[key] = {'type': 'changed', 'diff': subdiff}
                else:
                    diff[key] = {
                        'type': 'changed',
                        'old_value': dict1[key],
                        'new_value': dict2[key]
                    }
        
        # Check for removed keys
        for key in dict1:
            if key not in dict2:
                diff[key] = {'type': 'removed', 'value': dict1[key]}
        
        return diff

    def get_config_status(self) -> Dict[str, Any]:
        """Get overall configuration status"""
        status = {
            'environment': self.environment.value,
            'configs_loaded': len(self._configs),
            'config_names': list(self._configs.keys()),
            'versions_tracked': {
                name: len(versions) for name, versions in self._config_versions.items()
            },
            'change_listeners': len(self._change_listeners),
            'last_update': datetime.now().isoformat()
        }
        
        return status