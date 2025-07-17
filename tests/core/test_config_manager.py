"""
Comprehensive test suite for configuration management system.
Tests configuration loading, validation, environment handling, error scenarios,
and security tests for config injection attacks.
"""

import pytest
import os
import yaml
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any

from src.core.config_manager import (
    ConfigManager,
    Environment,
    ConfigPaths,
    get_config_manager,
    get_config,
    _config_manager
)


class TestEnvironmentEnum:
    """Test the Environment enum."""
    
    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.PRODUCTION.value == "production"
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
    
    def test_environment_string_conversion(self):
        """Test string conversion of environment enum."""
        assert str(Environment.PRODUCTION) == "Environment.PRODUCTION"
        assert Environment.PRODUCTION.value == "production"


class TestConfigPaths:
    """Test the ConfigPaths dataclass."""
    
    def test_config_paths_creation(self):
        """Test ConfigPaths creation."""
        base_dir = Path("/test/configs")
        paths = ConfigPaths(
            base_dir=base_dir,
            system_dir=base_dir / "system",
            trading_dir=base_dir / "trading",
            models_dir=base_dir / "models",
            environments_dir=base_dir / "environments",
            monitoring_dir=base_dir / "monitoring"
        )
        
        assert paths.base_dir == base_dir
        assert paths.system_dir == base_dir / "system"
        assert paths.trading_dir == base_dir / "trading"
        assert paths.models_dir == base_dir / "models"
        assert paths.environments_dir == base_dir / "environments"
        assert paths.monitoring_dir == base_dir / "monitoring"


class TestConfigManager:
    """Test the ConfigManager class."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)
        
        # Create subdirectories
        (self.config_dir / "system").mkdir()
        (self.config_dir / "trading").mkdir()
        (self.config_dir / "models").mkdir()
        (self.config_dir / "environments").mkdir()
        (self.config_dir / "monitoring").mkdir()
        
        # Create test configuration files
        self._create_test_configs()
    
    def teardown_method(self):
        """Teardown method for each test."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_configs(self):
        """Create test configuration files."""
        # System configuration
        system_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'test_db',
                'username': 'test_user',
                'password': 'test_pass'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'password': 'redis_pass'
            },
            'security': {
                'jwt_secret_key': 'test_secret'
            }
        }
        
        with open(self.config_dir / "system" / "production.yaml", 'w') as f:
            yaml.dump(system_config, f)
        
        with open(self.config_dir / "system" / "development.yaml", 'w') as f:
            yaml.dump(system_config, f)
        
        with open(self.config_dir / "system" / "testing.yaml", 'w') as f:
            yaml.dump(system_config, f)
        
        # Trading configurations
        strategic_config = {
            'agent_type': 'strategic',
            'lookback_period': 30,
            'risk_threshold': 0.1
        }
        
        with open(self.config_dir / "trading" / "strategic_config.yaml", 'w') as f:
            yaml.dump(strategic_config, f)
        
        tactical_config = {
            'agent_type': 'tactical',
            'lookback_period': 5,
            'risk_threshold': 0.05
        }
        
        with open(self.config_dir / "trading" / "tactical_config.yaml", 'w') as f:
            yaml.dump(tactical_config, f)
        
        risk_config = {
            'max_position_size': 0.1,
            'stop_loss': 0.02,
            'take_profit': 0.06
        }
        
        with open(self.config_dir / "trading" / "risk_config.yaml", 'w') as f:
            yaml.dump(risk_config, f)
        
        # Model configurations
        mappo_config = {
            'algorithm': 'mappo',
            'learning_rate': 0.0003,
            'batch_size': 32
        }
        
        with open(self.config_dir / "models" / "mappo_config.yaml", 'w') as f:
            yaml.dump(mappo_config, f)
        
        network_config = {
            'hidden_dim': 256,
            'num_layers': 3,
            'activation': 'relu'
        }
        
        with open(self.config_dir / "models" / "network_config.yaml", 'w') as f:
            yaml.dump(network_config, f)
        
        hyperparams_config = {
            'epochs': 100,
            'patience': 10,
            'optimizer': 'adam'
        }
        
        with open(self.config_dir / "models" / "hyperparameters.yaml", 'w') as f:
            yaml.dump(hyperparams_config, f)
        
        # Environment configurations
        market_config = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'timeframe': '1min',
            'lookback': 100
        }
        
        with open(self.config_dir / "environments" / "market_config.yaml", 'w') as f:
            yaml.dump(market_config, f)
        
        simulation_config = {
            'episodes': 1000,
            'max_steps': 1000,
            'reward_scaling': 1.0
        }
        
        with open(self.config_dir / "environments" / "simulation_config.yaml", 'w') as f:
            yaml.dump(simulation_config, f)
    
    @patch('src.core.config_manager.Path')
    def test_config_manager_initialization(self, mock_path):
        """Test ConfigManager initialization."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        assert manager.environment == Environment.PRODUCTION
        assert hasattr(manager, 'paths')
        assert hasattr(manager, '_config_cache')
        assert len(manager._config_cache) > 0
    
    @patch('src.core.config_manager.Path')
    def test_load_all_configs(self, mock_path):
        """Test loading all configurations."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Check that all required configs are loaded
        expected_configs = [
            'system', 'strategic', 'tactical', 'risk', 'mappo',
            'networks', 'hyperparameters', 'market', 'simulation'
        ]
        
        for config_name in expected_configs:
            assert config_name in manager._config_cache
            assert isinstance(manager._config_cache[config_name], dict)
    
    @patch('src.core.config_manager.Path')
    def test_load_yaml_file_exists(self, mock_path):
        """Test loading YAML file that exists."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        config_path = self.config_dir / "trading" / "strategic_config.yaml"
        
        config = manager._load_yaml(config_path)
        
        assert config['agent_type'] == 'strategic'
        assert config['lookback_period'] == 30
        assert config['risk_threshold'] == 0.1
    
    @patch('src.core.config_manager.Path')
    def test_load_yaml_file_not_exists(self, mock_path):
        """Test loading YAML file that doesn't exist."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        nonexistent_path = self.config_dir / "nonexistent.yaml"
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            config = manager._load_yaml(nonexistent_path)
            
            assert config == {}
            mock_logger.return_value.warning.assert_called_once()
    
    @patch('src.core.config_manager.Path')
    def test_load_yaml_invalid_yaml(self, mock_path):
        """Test loading invalid YAML file."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        # Create invalid YAML file
        invalid_yaml_path = self.config_dir / "invalid.yaml"
        with open(invalid_yaml_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            config = manager._load_yaml(invalid_yaml_path)
            
            assert config == {}
            mock_logger.return_value.error.assert_called_once()
    
    @patch('src.core.config_manager.Path')
    def test_apply_env_overrides(self, mock_path):
        """Test applying environment variable overrides."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        # Set environment variables
        test_env_vars = {
            'DB_HOST': 'override_host',
            'DB_PORT': '5433',
            'DB_NAME': 'override_db',
            'DB_USERNAME': 'override_user',
            'DB_PASSWORD': 'override_pass',
            'REDIS_HOST': 'override_redis',
            'REDIS_PORT': '6380',
            'REDIS_PASSWORD': 'override_redis_pass',
            'JWT_SECRET_KEY': 'override_secret'
        }
        
        with patch.dict(os.environ, test_env_vars):
            manager = ConfigManager(Environment.PRODUCTION)
            
            # Check database config overrides
            db_config = manager.get_database_config()
            assert db_config['host'] == 'override_host'
            assert db_config['port'] == 5433
            assert db_config['name'] == 'override_db'
            assert db_config['username'] == 'override_user'
            assert db_config['password'] == 'override_pass'
            
            # Check Redis config overrides
            redis_config = manager.get_redis_config()
            assert redis_config['host'] == 'override_redis'
            assert redis_config['port'] == 6380
            assert redis_config['password'] == 'override_redis_pass'
            
            # Check security config overrides
            security_config = manager.get_security_config()
            assert security_config['jwt_secret_key'] == 'override_secret'
    
    @patch('src.core.config_manager.Path')
    def test_get_config(self, mock_path):
        """Test getting configuration by name."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Test getting full configuration
        strategic_config = manager.get_config('strategic')
        assert strategic_config['agent_type'] == 'strategic'
        
        # Test getting configuration section
        db_config = manager.get_config('system', 'database')
        assert db_config['host'] == 'localhost'
        assert db_config['port'] == 5432
        
        # Test getting non-existent configuration
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            empty_config = manager.get_config('nonexistent')
            assert empty_config == {}
            mock_logger.return_value.warning.assert_called_once()
    
    @patch('src.core.config_manager.Path')
    def test_get_system_config(self, mock_path):
        """Test getting system configuration."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        system_config = manager.get_system_config()
        
        assert 'database' in system_config
        assert 'redis' in system_config
        assert 'security' in system_config
    
    @patch('src.core.config_manager.Path')
    def test_get_trading_config(self, mock_path):
        """Test getting trading configuration for different agents."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Test valid agent types
        strategic_config = manager.get_trading_config('strategic')
        assert strategic_config['agent_type'] == 'strategic'
        
        tactical_config = manager.get_trading_config('tactical')
        assert tactical_config['agent_type'] == 'tactical'
        
        risk_config = manager.get_trading_config('risk')
        assert risk_config['max_position_size'] == 0.1
        
        # Test invalid agent type
        with pytest.raises(ValueError, match="Invalid agent type"):
            manager.get_trading_config('invalid_agent')
    
    @patch('src.core.config_manager.Path')
    def test_get_model_config(self, mock_path):
        """Test getting model configuration."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        mappo_config = manager.get_model_config('mappo')
        assert mappo_config['algorithm'] == 'mappo'
        
        network_config = manager.get_model_config('networks')
        assert network_config['hidden_dim'] == 256
        
        hyperparams_config = manager.get_model_config('hyperparameters')
        assert hyperparams_config['epochs'] == 100
    
    @patch('src.core.config_manager.Path')
    def test_get_environment_config(self, mock_path):
        """Test getting environment configuration."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        market_config = manager.get_environment_config('market')
        assert market_config['symbols'] == ['AAPL', 'GOOGL', 'MSFT']
        
        simulation_config = manager.get_environment_config('simulation')
        assert simulation_config['episodes'] == 1000
    
    @patch('src.core.config_manager.Path')
    def test_update_config(self, mock_path):
        """Test updating configuration values."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Update strategic config
        updates = {
            'lookback_period': 60,
            'new_field': 'new_value'
        }
        
        manager.update_config('strategic', updates)
        
        strategic_config = manager.get_config('strategic')
        assert strategic_config['lookback_period'] == 60
        assert strategic_config['new_field'] == 'new_value'
        assert strategic_config['agent_type'] == 'strategic'  # Unchanged
    
    @patch('src.core.config_manager.Path')
    def test_update_config_nonexistent(self, mock_path):
        """Test updating non-existent configuration."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            manager.update_config('nonexistent', {'key': 'value'})
            mock_logger.return_value.warning.assert_called_once()
    
    @patch('src.core.config_manager.Path')
    def test_deep_update(self, mock_path):
        """Test deep update functionality."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Test deep update of nested dictionary
        updates = {
            'database': {
                'host': 'new_host',
                'new_field': 'new_value'
            }
        }
        
        manager.update_config('system', updates)
        
        system_config = manager.get_config('system')
        assert system_config['database']['host'] == 'new_host'
        assert system_config['database']['new_field'] == 'new_value'
        assert system_config['database']['port'] == 5432  # Unchanged
    
    @patch('src.core.config_manager.Path')
    def test_validate_configs_success(self, mock_path):
        """Test successful configuration validation."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            result = manager.validate_configs()
            
            assert result is True
            mock_logger.return_value.info.assert_called_with("All configurations validated successfully")
    
    @patch('src.core.config_manager.Path')
    def test_validate_configs_missing_required_config(self, mock_path):
        """Test validation with missing required configuration."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Remove required config
        del manager._config_cache['system']
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            result = manager.validate_configs()
            
            assert result is False
            mock_logger.return_value.error.assert_called()
    
    @patch('src.core.config_manager.Path')
    def test_validate_configs_missing_system_section(self, mock_path):
        """Test validation with missing system section."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Remove required system section
        del manager._config_cache['system']['database']
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            result = manager.validate_configs()
            
            assert result is False
            mock_logger.return_value.error.assert_called()
    
    @patch('src.core.config_manager.Path')
    def test_validate_configs_missing_db_field(self, mock_path):
        """Test validation with missing database field."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Remove required database field
        del manager._config_cache['system']['database']['host']
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            result = manager.validate_configs()
            
            assert result is False
            mock_logger.return_value.error.assert_called()
    
    @patch('src.core.config_manager.Path')
    def test_validate_configs_exception(self, mock_path):
        """Test validation with exception."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        with patch.object(manager, 'get_system_config', side_effect=Exception("Test error")):
            with patch('src.core.config_manager.logging.getLogger') as mock_logger:
                result = manager.validate_configs()
                
                assert result is False
                mock_logger.return_value.error.assert_called()
    
    @patch('src.core.config_manager.Path')
    def test_reload_configs(self, mock_path):
        """Test reloading configurations."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Modify a config value
        original_value = manager.get_config('strategic')['lookback_period']
        manager.update_config('strategic', {'lookback_period': 999})
        
        # Verify the change
        assert manager.get_config('strategic')['lookback_period'] == 999
        
        # Reload configs
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            manager.reload_configs()
            
            # Verify the value was reset
            assert manager.get_config('strategic')['lookback_period'] == original_value
            mock_logger.return_value.info.assert_called_with("Reloading configurations...")
    
    @patch('src.core.config_manager.Path')
    def test_get_all_configs(self, mock_path):
        """Test getting all configurations."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        all_configs = manager.get_all_configs()
        
        # Should be a copy, not the original
        assert id(all_configs) != id(manager._config_cache)
        
        # Should contain all loaded configs
        expected_configs = [
            'system', 'strategic', 'tactical', 'risk', 'mappo',
            'networks', 'hyperparameters', 'market', 'simulation'
        ]
        
        for config_name in expected_configs:
            assert config_name in all_configs
    
    @patch('src.core.config_manager.Path')
    def test_export_config(self, mock_path):
        """Test exporting configuration to file."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        # Export strategic config
        export_path = self.config_dir / "exported_strategic.yaml"
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            manager.export_config('strategic', export_path)
            
            # Verify file was created
            assert export_path.exists()
            
            # Verify content
            with open(export_path, 'r') as f:
                exported_config = yaml.safe_load(f)
            
            assert exported_config['agent_type'] == 'strategic'
            assert exported_config['lookback_period'] == 30
            
            mock_logger.return_value.info.assert_called()
    
    @patch('src.core.config_manager.Path')
    def test_export_config_nonexistent(self, mock_path):
        """Test exporting non-existent configuration."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = ConfigManager(Environment.PRODUCTION)
        
        export_path = self.config_dir / "exported_nonexistent.yaml"
        
        with pytest.raises(ValueError, match="Configuration 'nonexistent' not found"):
            manager.export_config('nonexistent', export_path)
    
    @patch('src.core.config_manager.Path')
    def test_initialization_error_handling(self, mock_path):
        """Test error handling during initialization."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        # Remove a required config file
        os.remove(self.config_dir / "system" / "production.yaml")
        
        with patch('src.core.config_manager.logging.getLogger') as mock_logger:
            with pytest.raises(Exception):
                ConfigManager(Environment.PRODUCTION)
            
            mock_logger.return_value.error.assert_called()


class TestGlobalConfigManager:
    """Test global configuration manager functions."""
    
    def setup_method(self):
        """Setup method for each test."""
        # Clear global state
        global _config_manager
        _config_manager = None
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)
        
        # Create minimal config structure
        (self.config_dir / "system").mkdir()
        (self.config_dir / "trading").mkdir()
        (self.config_dir / "models").mkdir()
        (self.config_dir / "environments").mkdir()
        
        # Create minimal configs
        self._create_minimal_configs()
    
    def teardown_method(self):
        """Teardown method for each test."""
        global _config_manager
        _config_manager = None
        shutil.rmtree(self.temp_dir)
    
    def _create_minimal_configs(self):
        """Create minimal configuration files."""
        minimal_config = {
            'database': {'host': 'localhost', 'port': 5432, 'name': 'test', 'username': 'test', 'password': 'test'},
            'redis': {'host': 'localhost', 'port': 6379, 'password': 'test'},
            'security': {'jwt_secret_key': 'test'}
        }
        
        for env in ['production', 'development', 'testing']:
            with open(self.config_dir / "system" / f"{env}.yaml", 'w') as f:
                yaml.dump(minimal_config, f)
        
        for config_name in ['strategic_config', 'tactical_config', 'risk_config']:
            with open(self.config_dir / "trading" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)
        
        for config_name in ['mappo_config', 'network_config', 'hyperparameters']:
            with open(self.config_dir / "models" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)
        
        for config_name in ['market_config', 'simulation_config']:
            with open(self.config_dir / "environments" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)
    
    @patch('src.core.config_manager.Path')
    def test_get_config_manager_first_call(self, mock_path):
        """Test first call to get_config_manager."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager = get_config_manager(Environment.PRODUCTION)
        
        assert isinstance(manager, ConfigManager)
        assert manager.environment == Environment.PRODUCTION
    
    @patch('src.core.config_manager.Path')
    def test_get_config_manager_subsequent_calls(self, mock_path):
        """Test subsequent calls to get_config_manager return same instance."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        manager1 = get_config_manager(Environment.PRODUCTION)
        manager2 = get_config_manager(Environment.DEVELOPMENT)  # Should ignore environment
        
        assert manager1 is manager2
        assert id(manager1) == id(manager2)
    
    @patch('src.core.config_manager.Path')
    def test_get_config_manager_with_env_var(self, mock_path):
        """Test get_config_manager with environment variable."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            manager = get_config_manager()
            
            assert manager.environment == Environment.DEVELOPMENT
    
    @patch('src.core.config_manager.Path')
    def test_get_config_function(self, mock_path):
        """Test get_config convenience function."""
        mock_path.return_value.parent.parent.parent = self.config_dir
        
        config = get_config('strategic')
        assert config['test'] == 'value'
        
        config_section = get_config('system', 'database')
        assert config_section['host'] == 'localhost'


class TestConfigSecurityAndValidation:
    """Test security and validation aspects of configuration management."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)
        
        # Create config structure
        (self.config_dir / "system").mkdir()
        (self.config_dir / "trading").mkdir()
        (self.config_dir / "models").mkdir()
        (self.config_dir / "environments").mkdir()
    
    def teardown_method(self):
        """Teardown method for each test."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_injection_attack_prevention(self):
        """Test prevention of configuration injection attacks."""
        # Create malicious config with code injection attempt
        malicious_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'test_db',
                'username': 'test_user',
                'password': 'test_pass'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'password': 'redis_pass'
            },
            'security': {
                'jwt_secret_key': 'test_secret'
            },
            'malicious_code': {
                'exec_attempt': '__import__("os").system("rm -rf /")',
                'eval_attempt': 'eval("print(\'injected code\')")'
            }
        }
        
        with open(self.config_dir / "system" / "production.yaml", 'w') as f:
            yaml.dump(malicious_config, f)
        
        # Create other required configs
        self._create_minimal_configs()
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            # Configuration should load without executing malicious code
            manager = ConfigManager(Environment.PRODUCTION)
            
            # Verify malicious code is stored as strings, not executed
            system_config = manager.get_system_config()
            assert 'malicious_code' in system_config
            assert isinstance(system_config['malicious_code']['exec_attempt'], str)
            assert isinstance(system_config['malicious_code']['eval_attempt'], str)
    
    def test_yaml_bomb_protection(self):
        """Test protection against YAML bombs."""
        # Create a YAML bomb (exponential expansion)
        yaml_bomb = """
        a: &a
          - <<: *a
          - <<: *a
          - <<: *a
        """
        
        with open(self.config_dir / "system" / "production.yaml", 'w') as f:
            f.write(yaml_bomb)
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            # Should handle the malformed YAML gracefully
            with patch('src.core.config_manager.logging.getLogger') as mock_logger:
                manager = ConfigManager(Environment.PRODUCTION)
                
                # Should log error and continue with empty config
                mock_logger.return_value.error.assert_called()
                assert manager.get_system_config() == {}
    
    def test_unicode_injection_protection(self):
        """Test protection against Unicode injection attacks."""
        # Create config with Unicode injection attempts
        unicode_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'test\u0000db',  # Null byte injection
                'username': 'test\u202euser',  # Right-to-left override
                'password': 'test\ufeffpass'  # Byte order mark
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'password': 'redis_pass'
            },
            'security': {
                'jwt_secret_key': 'test_secret'
            }
        }
        
        with open(self.config_dir / "system" / "production.yaml", 'w') as f:
            yaml.dump(unicode_config, f)
        
        self._create_minimal_configs()
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            manager = ConfigManager(Environment.PRODUCTION)
            
            # Configuration should load, but Unicode characters should be preserved as-is
            db_config = manager.get_database_config()
            assert '\u0000' in db_config['name']
            assert '\u202e' in db_config['username']
            assert '\ufeff' in db_config['password']
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        # Attempt to access files outside config directory
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            manager = ConfigManager(Environment.PRODUCTION)
            
            # Attempt to load file outside config directory
            malicious_path = Path("../../../etc/passwd")
            
            with patch('src.core.config_manager.logging.getLogger') as mock_logger:
                config = manager._load_yaml(malicious_path)
                
                # Should return empty config and log warning
                assert config == {}
                mock_logger.return_value.warning.assert_called()
    
    def test_large_config_file_handling(self):
        """Test handling of extremely large configuration files."""
        # Create a large configuration
        large_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'test_db',
                'username': 'test_user',
                'password': 'test_pass'
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'password': 'redis_pass'
            },
            'security': {
                'jwt_secret_key': 'test_secret'
            },
            'large_data': {f'key_{i}': f'value_{i}' * 1000 for i in range(1000)}
        }
        
        with open(self.config_dir / "system" / "production.yaml", 'w') as f:
            yaml.dump(large_config, f)
        
        self._create_minimal_configs()
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            # Should handle large config gracefully
            manager = ConfigManager(Environment.PRODUCTION)
            
            system_config = manager.get_system_config()
            assert 'large_data' in system_config
            assert len(system_config['large_data']) == 1000
    
    def test_malformed_yaml_handling(self):
        """Test handling of malformed YAML files."""
        malformed_yaml_cases = [
            "invalid: yaml: content: [",  # Unclosed bracket
            "key: value\n  invalid_indent",  # Invalid indentation
            "key: value\n\ttab_mixed_with_spaces: value",  # Mixed tabs and spaces
            "key: value\n---\n... invalid",  # Invalid document separator
        ]
        
        for i, malformed_yaml in enumerate(malformed_yaml_cases):
            config_file = self.config_dir / f"malformed_{i}.yaml"
            with open(config_file, 'w') as f:
                f.write(malformed_yaml)
            
            with patch('src.core.config_manager.Path') as mock_path:
                mock_path.return_value.parent.parent.parent = self.config_dir
                
                manager = ConfigManager(Environment.PRODUCTION)
                
                with patch('src.core.config_manager.logging.getLogger') as mock_logger:
                    config = manager._load_yaml(config_file)
                    
                    # Should return empty config and log error
                    assert config == {}
                    mock_logger.return_value.error.assert_called()
    
    def test_environment_variable_injection(self):
        """Test security of environment variable injection."""
        # Test with malicious environment variables
        malicious_env_vars = {
            'DB_HOST': 'localhost; rm -rf /',
            'DB_PORT': '5432; echo "injected"',
            'DB_NAME': 'test_db`whoami`',
            'JWT_SECRET_KEY': 'secret$(malicious_command)'
        }
        
        self._create_minimal_configs()
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            with patch.dict(os.environ, malicious_env_vars):
                manager = ConfigManager(Environment.PRODUCTION)
                
                # Values should be stored as strings without execution
                db_config = manager.get_database_config()
                assert db_config['host'] == 'localhost; rm -rf /'
                assert db_config['port'] == '5432; echo "injected"'  # String, not executed
                assert db_config['name'] == 'test_db`whoami`'
                
                security_config = manager.get_security_config()
                assert security_config['jwt_secret_key'] == 'secret$(malicious_command)'
    
    def test_config_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        # Test with missing required fields
        invalid_configs = [
            # Missing database host
            {
                'database': {'port': 5432, 'name': 'test', 'username': 'test', 'password': 'test'},
                'redis': {'host': 'localhost', 'port': 6379, 'password': 'test'},
                'security': {'jwt_secret_key': 'test'}
            },
            # Missing redis section
            {
                'database': {'host': 'localhost', 'port': 5432, 'name': 'test', 'username': 'test', 'password': 'test'},
                'security': {'jwt_secret_key': 'test'}
            },
            # Missing security section
            {
                'database': {'host': 'localhost', 'port': 5432, 'name': 'test', 'username': 'test', 'password': 'test'},
                'redis': {'host': 'localhost', 'port': 6379, 'password': 'test'}
            }
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            config_file = self.config_dir / "system" / f"invalid_{i}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(invalid_config, f)
            
            with patch('src.core.config_manager.Path') as mock_path:
                mock_path.return_value.parent.parent.parent = self.config_dir
                
                # Mock the system config file path to use our invalid config
                original_load_yaml = ConfigManager._load_yaml
                
                def mock_load_yaml(self, path):
                    if path.name == "production.yaml":
                        return invalid_config
                    return original_load_yaml(self, path)
                
                with patch.object(ConfigManager, '_load_yaml', mock_load_yaml):
                    manager = ConfigManager(Environment.PRODUCTION)
                    
                    with patch('src.core.config_manager.logging.getLogger') as mock_logger:
                        result = manager.validate_configs()
                        
                        assert result is False
                        mock_logger.return_value.error.assert_called()
    
    def _create_minimal_configs(self):
        """Create minimal configuration files for testing."""
        minimal_config = {
            'database': {'host': 'localhost', 'port': 5432, 'name': 'test', 'username': 'test', 'password': 'test'},
            'redis': {'host': 'localhost', 'port': 6379, 'password': 'test'},
            'security': {'jwt_secret_key': 'test'}
        }
        
        for env in ['production', 'development', 'testing']:
            with open(self.config_dir / "system" / f"{env}.yaml", 'w') as f:
                yaml.dump(minimal_config, f)
        
        for config_name in ['strategic_config', 'tactical_config', 'risk_config']:
            with open(self.config_dir / "trading" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)
        
        for config_name in ['mappo_config', 'network_config', 'hyperparameters']:
            with open(self.config_dir / "models" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)
        
        for config_name in ['market_config', 'simulation_config']:
            with open(self.config_dir / "environments" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)


class TestConfigPerformance:
    """Test performance aspects of configuration management."""
    
    def setup_method(self):
        """Setup method for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir(parents=True)
        
        # Create config structure
        (self.config_dir / "system").mkdir()
        (self.config_dir / "trading").mkdir()
        (self.config_dir / "models").mkdir()
        (self.config_dir / "environments").mkdir()
        
        self._create_test_configs()
    
    def teardown_method(self):
        """Teardown method for each test."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_configs(self):
        """Create test configuration files."""
        base_config = {
            'database': {'host': 'localhost', 'port': 5432, 'name': 'test', 'username': 'test', 'password': 'test'},
            'redis': {'host': 'localhost', 'port': 6379, 'password': 'test'},
            'security': {'jwt_secret_key': 'test'}
        }
        
        for env in ['production', 'development', 'testing']:
            with open(self.config_dir / "system" / f"{env}.yaml", 'w') as f:
                yaml.dump(base_config, f)
        
        for config_name in ['strategic_config', 'tactical_config', 'risk_config']:
            with open(self.config_dir / "trading" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)
        
        for config_name in ['mappo_config', 'network_config', 'hyperparameters']:
            with open(self.config_dir / "models" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)
        
        for config_name in ['market_config', 'simulation_config']:
            with open(self.config_dir / "environments" / f"{config_name}.yaml", 'w') as f:
                yaml.dump({'test': 'value'}, f)
    
    def test_config_loading_performance(self):
        """Test configuration loading performance."""
        import time
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            start_time = time.time()
            manager = ConfigManager(Environment.PRODUCTION)
            end_time = time.time()
            
            loading_time = end_time - start_time
            
            # Configuration loading should be fast (< 1 second)
            assert loading_time < 1.0
            
            # Should have loaded all configs
            assert len(manager._config_cache) >= 9
    
    def test_config_access_performance(self):
        """Test configuration access performance."""
        import time
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            manager = ConfigManager(Environment.PRODUCTION)
            
            # Test repeated access performance
            start_time = time.time()
            for _ in range(1000):
                config = manager.get_config('strategic')
                section = manager.get_config('system', 'database')
            end_time = time.time()
            
            access_time = end_time - start_time
            
            # 1000 accesses should be very fast (< 0.1 seconds)
            assert access_time < 0.1
    
    def test_config_update_performance(self):
        """Test configuration update performance."""
        import time
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            manager = ConfigManager(Environment.PRODUCTION)
            
            # Test update performance
            start_time = time.time()
            for i in range(100):
                manager.update_config('strategic', {'test_field': f'value_{i}'})
            end_time = time.time()
            
            update_time = end_time - start_time
            
            # 100 updates should be fast (< 0.1 seconds)
            assert update_time < 0.1
    
    def test_config_validation_performance(self):
        """Test configuration validation performance."""
        import time
        
        with patch('src.core.config_manager.Path') as mock_path:
            mock_path.return_value.parent.parent.parent = self.config_dir
            
            manager = ConfigManager(Environment.PRODUCTION)
            
            # Test validation performance
            start_time = time.time()
            for _ in range(100):
                result = manager.validate_configs()
            end_time = time.time()
            
            validation_time = end_time - start_time
            
            # 100 validations should be fast (< 0.5 seconds)
            assert validation_time < 0.5
            assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])