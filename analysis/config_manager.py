"""
Analysis Configuration Manager
Agent 5 - System Integration and Production Architecture

Unified configuration system with schema validation, dynamic reloading,
and environment-specific configuration management for analysis components.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from datetime import datetime
import hashlib
import jsonschema
from contextlib import contextmanager
import copy


class ConfigType(Enum):
    """Configuration types for analysis components"""
    BACKTEST = "backtest"
    METRICS = "metrics"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    BASELINE = "baseline"
    INTEGRATION = "integration"


@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    name: str
    schema: Dict[str, Any]
    required_fields: List[str] = field(default_factory=list)
    default_values: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Callable] = field(default_factory=dict)


@dataclass
class ConfigChangeEvent:
    """Configuration change event"""
    config_type: ConfigType
    changed_keys: List[str]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    timestamp: datetime
    source: str


class ConfigWatcher:
    """File system watcher for configuration changes"""
    
    def __init__(self, config_manager: 'AnalysisConfigManager'):
        self.config_manager = config_manager
        self.watching = False
        self.watch_thread = None
        self._file_mtimes = {}
        self._lock = threading.Lock()
        
    def start_watching(self) -> None:
        """Start watching configuration files"""
        if self.watching:
            return
            
        self.watching = True
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        
    def stop_watching(self) -> None:
        """Stop watching configuration files"""
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=1.0)
            
    def _watch_loop(self) -> None:
        """Main watch loop"""
        while self.watching:
            try:
                self._check_file_changes()
                time.sleep(1.0)  # Check every second
            except Exception as e:
                logging.error(f"Configuration watcher error: {e}")
                time.sleep(5.0)  # Longer sleep on error
                
    def _check_file_changes(self) -> None:
        """Check for file changes"""
        with self._lock:
            for config_path in self.config_manager.config_paths.values():
                if not config_path.exists():
                    continue
                    
                current_mtime = config_path.stat().st_mtime
                if config_path not in self._file_mtimes:
                    self._file_mtimes[config_path] = current_mtime
                    continue
                    
                if current_mtime > self._file_mtimes[config_path]:
                    self._file_mtimes[config_path] = current_mtime
                    self.config_manager._reload_config_file(config_path)


class AnalysisConfigManager:
    """
    Unified configuration management system for analysis components
    
    Features:
    - Schema validation
    - Dynamic configuration reloading
    - Environment-specific configuration
    - Parameter optimization framework integration
    - Configuration change notifications
    - File system watching
    """
    
    def __init__(self, environment: str = None, enable_watching: bool = True):
        self.environment = environment or os.getenv('ENVIRONMENT', 'production')
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._change_callbacks: List[Callable[[ConfigChangeEvent], None]] = []
        
        # Configuration storage
        self._configs: Dict[ConfigType, Dict[str, Any]] = {}
        self._schemas: Dict[ConfigType, ConfigSchema] = {}
        self._config_hashes: Dict[ConfigType, str] = {}
        
        # Configuration paths
        self.base_path = Path(__file__).parent.parent / "configs"
        self.config_paths: Dict[ConfigType, Path] = {
            ConfigType.BACKTEST: self.base_path / "analysis" / "backtest_config.yaml",
            ConfigType.METRICS: self.base_path / "analysis" / "metrics_config.yaml",
            ConfigType.VALIDATION: self.base_path / "analysis" / "validation_config.yaml",
            ConfigType.OPTIMIZATION: self.base_path / "analysis" / "optimization_config.yaml",
            ConfigType.BASELINE: self.base_path / "analysis" / "baseline_config.yaml",
            ConfigType.INTEGRATION: self.base_path / "analysis" / "integration_config.yaml"
        }
        
        # Initialize schemas
        self._initialize_schemas()
        
        # Load configurations
        self._load_all_configs()
        
        # Start file watcher
        self.watcher = ConfigWatcher(self) if enable_watching else None
        if self.watcher:
            self.watcher.start_watching()
            
    def _initialize_schemas(self) -> None:
        """Initialize configuration schemas"""
        
        # Backtest configuration schema
        backtest_schema = ConfigSchema(
            name="backtest",
            schema={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "initial_capital": {"type": "number", "minimum": 0},
                    "transaction_costs": {"type": "number", "minimum": 0},
                    "slippage": {"type": "number", "minimum": 0},
                    "position_sizing": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["fixed", "percent", "kelly"]},
                            "value": {"type": "number", "minimum": 0}
                        }
                    },
                    "risk_management": {
                        "type": "object",
                        "properties": {
                            "max_position_size": {"type": "number", "minimum": 0, "maximum": 1},
                            "max_drawdown": {"type": "number", "minimum": 0, "maximum": 1},
                            "stop_loss": {"type": "number", "minimum": 0},
                            "take_profit": {"type": "number", "minimum": 0}
                        }
                    }
                },
                "required": ["start_date", "end_date", "initial_capital"]
            },
            required_fields=["start_date", "end_date", "initial_capital"],
            default_values={
                "initial_capital": 100000,
                "transaction_costs": 0.001,
                "slippage": 0.0005,
                "position_sizing": {"method": "percent", "value": 0.02}
            }
        )
        
        # Metrics configuration schema
        metrics_schema = ConfigSchema(
            name="metrics",
            schema={
                "type": "object",
                "properties": {
                    "risk_free_rate": {"type": "number", "minimum": 0},
                    "periods_per_year": {"type": "integer", "minimum": 1},
                    "benchmark": {"type": "string"},
                    "metrics_to_calculate": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "rolling_window": {"type": "integer", "minimum": 1},
                    "confidence_levels": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            },
            default_values={
                "risk_free_rate": 0.02,
                "periods_per_year": 252,
                "benchmark": "SPY",
                "metrics_to_calculate": [
                    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                    "max_drawdown", "win_rate", "profit_factor"
                ],
                "rolling_window": 252,
                "confidence_levels": [0.95, 0.99]
            }
        )
        
        # Validation configuration schema
        validation_schema = ConfigSchema(
            name="validation",
            schema={
                "type": "object",
                "properties": {
                    "cross_validation": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["time_series", "walk_forward"]},
                            "n_splits": {"type": "integer", "minimum": 2},
                            "test_size": {"type": "number", "minimum": 0.1, "maximum": 0.9}
                        }
                    },
                    "statistical_tests": {
                        "type": "object",
                        "properties": {
                            "significance_level": {"type": "number", "minimum": 0, "maximum": 1},
                            "multiple_testing_correction": {"type": "string"},
                            "test_types": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "robustness_checks": {
                        "type": "object",
                        "properties": {
                            "stress_test_scenarios": {"type": "array"},
                            "monte_carlo_runs": {"type": "integer", "minimum": 100}
                        }
                    }
                }
            },
            default_values={
                "cross_validation": {
                    "method": "walk_forward",
                    "n_splits": 5,
                    "test_size": 0.2
                },
                "statistical_tests": {
                    "significance_level": 0.05,
                    "multiple_testing_correction": "bonferroni",
                    "test_types": ["t_test", "mann_whitney", "kolmogorov_smirnov"]
                },
                "robustness_checks": {
                    "stress_test_scenarios": ["market_crash", "high_volatility", "low_liquidity"],
                    "monte_carlo_runs": 1000
                }
            }
        )
        
        # Optimization configuration schema
        optimization_schema = ConfigSchema(
            name="optimization",
            schema={
                "type": "object",
                "properties": {
                    "optimizer": {
                        "type": "object",
                        "properties": {
                            "algorithm": {"type": "string", "enum": ["bayesian", "genetic", "grid_search"]},
                            "n_trials": {"type": "integer", "minimum": 10},
                            "timeout": {"type": "number", "minimum": 0},
                            "parallel_jobs": {"type": "integer", "minimum": 1}
                        }
                    },
                    "objective_function": {
                        "type": "object",
                        "properties": {
                            "primary_metric": {"type": "string"},
                            "secondary_metrics": {"type": "array", "items": {"type": "string"}},
                            "weights": {"type": "object"},
                            "constraints": {"type": "object"}
                        }
                    },
                    "parameter_spaces": {"type": "object"}
                }
            },
            default_values={
                "optimizer": {
                    "algorithm": "bayesian",
                    "n_trials": 100,
                    "timeout": 3600,
                    "parallel_jobs": 4
                },
                "objective_function": {
                    "primary_metric": "sharpe_ratio",
                    "secondary_metrics": ["max_drawdown", "win_rate"],
                    "weights": {"sharpe_ratio": 0.7, "max_drawdown": 0.3}
                }
            }
        )
        
        # Baseline configuration schema
        baseline_schema = ConfigSchema(
            name="baseline",
            schema={
                "type": "object",
                "properties": {
                    "agents": {
                        "type": "object",
                        "properties": {
                            "random_agent": {"type": "object"},
                            "rule_based_agent": {"type": "object"},
                            "benchmark_agent": {"type": "object"}
                        }
                    },
                    "comparison_metrics": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "performance_thresholds": {"type": "object"}
                }
            },
            default_values={
                "agents": {
                    "random_agent": {"enabled": True, "seed": 42},
                    "rule_based_agent": {"enabled": True, "rules": "conservative"},
                    "benchmark_agent": {"enabled": True, "benchmark": "SPY"}
                },
                "comparison_metrics": ["sharpe_ratio", "max_drawdown", "win_rate"],
                "performance_thresholds": {
                    "minimum_sharpe": 0.5,
                    "maximum_drawdown": 0.2
                }
            }
        )
        
        # Integration configuration schema
        integration_schema = ConfigSchema(
            name="integration",
            schema={
                "type": "object",
                "properties": {
                    "data_sources": {"type": "array", "items": {"type": "string"}},
                    "model_endpoints": {"type": "object"},
                    "performance_monitoring": {"type": "object"},
                    "alerting": {"type": "object"}
                }
            },
            default_values={
                "data_sources": ["risk_system", "execution_engine", "strategic_agents"],
                "model_endpoints": {
                    "strategic_marl": "/api/v1/strategic",
                    "tactical_marl": "/api/v1/tactical",
                    "risk_management": "/api/v1/risk"
                },
                "performance_monitoring": {
                    "enabled": True,
                    "update_frequency": 60,
                    "metrics_buffer_size": 1000
                },
                "alerting": {
                    "enabled": True,
                    "thresholds": {
                        "performance_degradation": 0.1,
                        "error_rate": 0.05
                    }
                }
            }
        )
        
        # Store schemas
        self._schemas = {
            ConfigType.BACKTEST: backtest_schema,
            ConfigType.METRICS: metrics_schema,
            ConfigType.VALIDATION: validation_schema,
            ConfigType.OPTIMIZATION: optimization_schema,
            ConfigType.BASELINE: baseline_schema,
            ConfigType.INTEGRATION: integration_schema
        }
        
    def _load_all_configs(self) -> None:
        """Load all configuration files"""
        with self._lock:
            for config_type in ConfigType:
                self._load_config(config_type)
                
    def _load_config(self, config_type: ConfigType) -> None:
        """Load a specific configuration"""
        config_path = self.config_paths[config_type]
        schema = self._schemas[config_type]
        
        # Load from file if exists
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.suffix == '.yaml':
                        config_data = yaml.safe_load(f) or {}
                    else:
                        config_data = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load {config_path}: {e}")
                config_data = {}
        else:
            config_data = {}
            
        # Apply defaults
        config_data = self._apply_defaults(config_data, schema.default_values)
        
        # Apply environment overrides
        config_data = self._apply_environment_overrides(config_data, config_type)
        
        # Validate configuration
        self._validate_config(config_data, schema)
        
        # Store configuration
        old_config = self._configs.get(config_type, {})
        self._configs[config_type] = config_data
        
        # Update hash
        config_hash = self._calculate_hash(config_data)
        old_hash = self._config_hashes.get(config_type)
        self._config_hashes[config_type] = config_hash
        
        # Notify of changes
        if old_hash and old_hash != config_hash:
            self._notify_config_change(config_type, old_config, config_data)
            
    def _apply_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration"""
        result = copy.deepcopy(defaults)
        self._deep_update(result, config)
        return result
        
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep update dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
                
    def _apply_environment_overrides(self, config: Dict[str, Any], config_type: ConfigType) -> Dict[str, Any]:
        """Apply environment-specific overrides"""
        # Environment-specific overrides
        env_prefix = f"{config_type.value.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                # Simple type conversion
                if value.lower() in ['true', 'false']:
                    config[config_key] = value.lower() == 'true'
                elif value.isdigit():
                    config[config_key] = int(value)
                elif value.replace('.', '').isdigit():
                    config[config_key] = float(value)
                else:
                    config[config_key] = value
                    
        return config
        
    def _validate_config(self, config: Dict[str, Any], schema: ConfigSchema) -> None:
        """Validate configuration against schema"""
        try:
            jsonschema.validate(config, schema.schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
            
        # Custom validation rules
        for field, validator in schema.validation_rules.items():
            if field in config:
                if not validator(config[field]):
                    raise ValueError(f"Custom validation failed for field '{field}'")
                    
    def _calculate_hash(self, config: Dict[str, Any]) -> str:
        """Calculate configuration hash"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
        
    def _notify_config_change(self, config_type: ConfigType, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Notify configuration change"""
        # Find changed keys
        changed_keys = []
        old_values = {}
        new_values = {}
        
        all_keys = set(old_config.keys()) | set(new_config.keys())
        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            if old_val != new_val:
                changed_keys.append(key)
                old_values[key] = old_val
                new_values[key] = new_val
                
        # Create event
        event = ConfigChangeEvent(
            config_type=config_type,
            changed_keys=changed_keys,
            old_values=old_values,
            new_values=new_values,
            timestamp=datetime.now(),
            source="file_reload"
        )
        
        # Notify callbacks
        for callback in self._change_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Configuration change callback failed: {e}")
                
    def _reload_config_file(self, config_path: Path) -> None:
        """Reload configuration from file"""
        for config_type, path in self.config_paths.items():
            if path == config_path:
                self.logger.info(f"Reloading configuration: {config_type.value}")
                self._load_config(config_type)
                break
                
    def get_config(self, config_type: ConfigType) -> Dict[str, Any]:
        """Get configuration by type"""
        with self._lock:
            return copy.deepcopy(self._configs.get(config_type, {}))
            
    def get_config_value(self, config_type: ConfigType, key: str, default: Any = None) -> Any:
        """Get specific configuration value"""
        config = self.get_config(config_type)
        return config.get(key, default)
        
    def update_config(self, config_type: ConfigType, updates: Dict[str, Any]) -> None:
        """Update configuration"""
        with self._lock:
            old_config = self._configs.get(config_type, {})
            new_config = copy.deepcopy(old_config)
            self._deep_update(new_config, updates)
            
            # Validate updated configuration
            schema = self._schemas[config_type]
            self._validate_config(new_config, schema)
            
            # Store updated configuration
            self._configs[config_type] = new_config
            
            # Update hash
            config_hash = self._calculate_hash(new_config)
            old_hash = self._config_hashes.get(config_type)
            self._config_hashes[config_type] = config_hash
            
            # Notify of changes
            if old_hash != config_hash:
                self._notify_config_change(config_type, old_config, new_config)
                
    def save_config(self, config_type: ConfigType) -> None:
        """Save configuration to file"""
        with self._lock:
            config = self._configs.get(config_type, {})
            config_path = self.config_paths[config_type]
            
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(config_path, 'w') as f:
                if config_path.suffix == '.yaml':
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=2)
                    
            self.logger.info(f"Saved configuration: {config_type.value}")
            
    def register_change_callback(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Register callback for configuration changes"""
        self._change_callbacks.append(callback)
        
    def unregister_change_callback(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """Unregister callback for configuration changes"""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
            
    def reload_all_configs(self) -> None:
        """Reload all configurations"""
        with self._lock:
            self._load_all_configs()
            
    def get_all_configs(self) -> Dict[ConfigType, Dict[str, Any]]:
        """Get all configurations"""
        with self._lock:
            return {k: copy.deepcopy(v) for k, v in self._configs.items()}
            
    def export_config(self, config_type: ConfigType, output_path: Path) -> None:
        """Export configuration to file"""
        config = self.get_config(config_type)
        
        with open(output_path, 'w') as f:
            if output_path.suffix == '.yaml':
                yaml.dump(config, f, default_flow_style=False)
            else:
                json.dump(config, f, indent=2)
                
    def import_config(self, config_type: ConfigType, input_path: Path) -> None:
        """Import configuration from file"""
        with open(input_path, 'r') as f:
            if input_path.suffix == '.yaml':
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
                
        # Validate and update
        schema = self._schemas[config_type]
        self._validate_config(config_data, schema)
        
        with self._lock:
            old_config = self._configs.get(config_type, {})
            self._configs[config_type] = config_data
            
            # Update hash
            config_hash = self._calculate_hash(config_data)
            old_hash = self._config_hashes.get(config_type)
            self._config_hashes[config_type] = config_hash
            
            # Notify of changes
            if old_hash != config_hash:
                self._notify_config_change(config_type, old_config, config_data)
                
    @contextmanager
    def temporary_config(self, config_type: ConfigType, temp_config: Dict[str, Any]):
        """Context manager for temporary configuration changes"""
        original_config = self.get_config(config_type)
        try:
            self.update_config(config_type, temp_config)
            yield
        finally:
            with self._lock:
                self._configs[config_type] = original_config
                
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.watcher:
            self.watcher.stop_watching()


# Global configuration manager instance
_config_manager: Optional[AnalysisConfigManager] = None


def get_analysis_config_manager(environment: str = None) -> AnalysisConfigManager:
    """Get global analysis configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = AnalysisConfigManager(environment)
    return _config_manager


def get_config(config_type: ConfigType) -> Dict[str, Any]:
    """Convenience function to get configuration"""
    return get_analysis_config_manager().get_config(config_type)


def get_config_value(config_type: ConfigType, key: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    return get_analysis_config_manager().get_config_value(config_type, key, default)