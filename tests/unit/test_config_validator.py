"""
Unit tests for configuration validation with Pydantic.
Tests config loading, validation, and secret management.
"""

import pytest
import os
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from pydantic import ValidationError

from src.core.config import (
    Config, ProductionConfig, RedisConfig, LoggingConfig,
    TradingConfig, IndicatorConfig, APIConfig, MonitoringConfig,
    PerformanceConfig
)


class TestConfigModels:
    """Test Pydantic configuration models."""
    
    def test_redis_config_defaults(self):
        """Test RedisConfig with defaults."""
        config = RedisConfig()
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.max_connections == 100
    
    def test_redis_config_validation(self):
        """Test RedisConfig validation."""
        # Valid config
        config = RedisConfig(host="redis-server", port=6380, db=1)
        assert config.host == "redis-server"
        assert config.port == 6380
        
        # Invalid port
        with pytest.raises(ValidationError) as exc_info:
            RedisConfig(port=70000)
        assert "less than or equal to 65535" in str(exc_info.value)
        
        # Invalid db
        with pytest.raises(ValidationError) as exc_info:
            RedisConfig(db=-1)
        assert "greater than or equal to 0" in str(exc_info.value)
    
    def test_logging_config_validation(self):
        """Test LoggingConfig validation."""
        # Valid config
        config = LoggingConfig(level="DEBUG", format="json")
        assert config.level == "DEBUG"
        
        # Case insensitive level
        config = LoggingConfig(level="info")
        assert config.level == "INFO"
        
        # Invalid level
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfig(level="INVALID")
        assert "Invalid log level" in str(exc_info.value)
    
    def test_trading_config_validation(self):
        """Test TradingConfig validation."""
        # Valid config
        config = TradingConfig(
            tick_buffer_size=500,
            bar_periods=[60, 300, 900],
            risk_per_trade=0.01
        )
        assert config.tick_buffer_size == 500
        assert len(config.bar_periods) == 3
        
        # Invalid buffer size
        with pytest.raises(ValidationError) as exc_info:
            TradingConfig(tick_buffer_size=50)
        assert "greater than or equal to 100" in str(exc_info.value)
        
        # Invalid risk
        with pytest.raises(ValidationError) as exc_info:
            TradingConfig(risk_per_trade=0.2)
        assert "less than or equal to 0.1" in str(exc_info.value)
        
        # Invalid bar period
        with pytest.raises(ValidationError) as exc_info:
            TradingConfig(bar_periods=[30])  # Too short
        assert "Bar period 30s too short" in str(exc_info.value)
    
    def test_indicator_config_validation(self):
        """Test IndicatorConfig validation."""
        # Valid config
        config = IndicatorConfig(
            mlmi_period=25,
            nwrqk_period=20,
            fvg_lookback=150,
            lvn_bins=75
        )
        assert config.mlmi_period == 25
        
        # Invalid periods
        with pytest.raises(ValidationError) as exc_info:
            IndicatorConfig(mlmi_period=150)  # Too large
        assert "less than or equal to 100" in str(exc_info.value)
    
    def test_api_config_secret_loading(self):
        """Test APIConfig with secret loading."""
        with patch('src.security.secrets_manager.secrets_manager.get_secret') as mock_get:
            mock_get.return_value = "test-jwt-secret"
            
            config = APIConfig()
            assert config.jwt_secret == "test-jwt-secret"
            mock_get.assert_called_with('jwt_secret', required=True)
    
    def test_performance_config_validation(self):
        """Test PerformanceConfig validation."""
        # Valid config
        config = PerformanceConfig(
            max_inference_latency_ms=3.5,
            max_memory_usage_mb=1024,
            max_cpu_usage_percent=75.0
        )
        assert config.max_inference_latency_ms == 3.5
        
        # Invalid latency
        with pytest.raises(ValidationError) as exc_info:
            PerformanceConfig(max_inference_latency_ms=150.0)
        assert "less than or equal to 100" in str(exc_info.value)
    
    def test_production_config_complete(self):
        """Test complete ProductionConfig."""
        with patch('src.security.secrets_manager.secrets_manager.get_secret') as mock_get:
            mock_get.return_value = "test-secret"
            
            config = ProductionConfig(
                environment="production",
                debug=False,
                feature_flags={"enable_ml": True}
            )
            
            assert config.environment == "production"
            assert config.debug is False
            assert config.feature_flags["enable_ml"] is True
            assert isinstance(config.redis, RedisConfig)
            assert isinstance(config.logging, LoggingConfig)
            assert isinstance(config.trading, TradingConfig)
    
    def test_production_config_environment_validation(self):
        """Test environment validation."""
        # Valid environments
        for env in ["development", "staging", "production"]:
            config = ProductionConfig(environment=env)
            assert config.environment == env
        
        # Invalid environment
        with pytest.raises(ValidationError) as exc_info:
            ProductionConfig(environment="testing")
        assert "Invalid environment" in str(exc_info.value)


class TestConfigManager:
    """Test Config class for configuration management."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'redis': {'host': 'test-redis', 'port': 6379},
                'logging': {'level': 'DEBUG'},
                'trading': {'tick_buffer_size': 2000}
            }
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def temp_json_config(self):
        """Create temporary JSON config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                'api': {'port': 8080},
                'monitoring': {'metrics_port': 9091}
            }
            json.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_config_initialization_default(self):
        """Test Config initialization with defaults."""
        with patch('os.path.exists', return_value=False):
            config = Config()
            
            assert config.config['redis']['host'] == 'localhost'
            assert config.config['logging']['level'] == 'INFO'
            assert config.config['trading']['tick_buffer_size'] == 1000
    
    def test_config_load_yaml(self, temp_config_file):
        """Test loading YAML configuration."""
        config = Config(config_path=temp_config_file)
        
        assert config.config['redis']['host'] == 'test-redis'
        assert config.config['logging']['level'] == 'DEBUG'
        assert config.config['trading']['tick_buffer_size'] == 2000
    
    def test_config_load_json(self, temp_json_config):
        """Test loading JSON configuration."""
        config = Config(config_path=temp_json_config)
        
        assert config.config['api']['port'] == 8080
        assert config.config['monitoring']['metrics_port'] == 9091
    
    def test_config_deep_merge(self):
        """Test deep merge functionality."""
        config = Config()
        base = {'a': {'b': 1, 'c': 2}, 'd': 3}
        update = {'a': {'b': 10, 'e': 4}, 'f': 5}
        
        result = config._deep_merge(base, update)
        
        assert result['a']['b'] == 10  # Updated
        assert result['a']['c'] == 2   # Preserved
        assert result['a']['e'] == 4   # Added
        assert result['d'] == 3         # Preserved
        assert result['f'] == 5         # Added
    
    def test_config_get(self):
        """Test getting config values by dot notation."""
        config = Config()
        config.config = {
            'redis': {'host': 'localhost', 'port': 6379},
            'api': {'jwt': {'secret': 'test-secret'}}
        }
        
        assert config.get('redis.host') == 'localhost'
        assert config.get('redis.port') == 6379
        assert config.get('api.jwt.secret') == 'test-secret'
        assert config.get('nonexistent', 'default') == 'default'
    
    def test_config_set(self):
        """Test setting config values."""
        with patch.object(Config, 'validate_config'):
            config = Config()
            
            config.set('redis.host', 'new-host')
            config.set('api.new_setting', 'value')
            
            assert config.config['redis']['host'] == 'new-host'
            assert config.config['api']['new_setting'] == 'value'
    
    def test_config_validation(self):
        """Test configuration validation."""
        with patch('src.security.secrets_manager.secrets_manager.get_secret') as mock_get:
            mock_get.return_value = "test-secret"
            
            config = Config()
            validated = config.validate_config()
            
            assert isinstance(validated, ProductionConfig)
            assert validated.redis.host == 'localhost'
    
    def test_config_validation_error(self):
        """Test configuration validation with errors."""
        config = Config()
        config.config['trading']['risk_per_trade'] = 0.5  # Too high
        
        with pytest.raises(ValidationError) as exc_info:
            config.validate_config()
        
        assert "less than or equal to 0.1" in str(exc_info.value)
    
    def test_check_required_secrets(self):
        """Test required secrets validation."""
        with patch('src.security.secrets_manager.secrets_manager.validate_secrets') as mock_validate:
            mock_validate.return_value = {
                'jwt_secret': True,
                'db_password': True,
                'db_username': True
            }
            
            config = Config()
            result = config.check_required_secrets()
            
            assert result is True
            mock_validate.assert_called_once()
    
    def test_check_required_secrets_missing(self):
        """Test required secrets validation with missing secrets."""
        with patch('src.security.secrets_manager.secrets_manager.validate_secrets') as mock_validate:
            mock_validate.return_value = {
                'jwt_secret': True,
                'db_password': False,  # Missing
                'db_username': True
            }
            
            config = Config()
            result = config.check_required_secrets()
            
            assert result is False
    
    def test_save_config_yaml(self):
        """Test saving configuration to YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_config.yaml"
            
            config = Config()
            config.config['test'] = {'value': 123}
            config.save_config(str(save_path))
            
            assert save_path.exists()
            
            # Load and verify
            with open(save_path) as f:
                loaded = yaml.safe_load(f)
                assert loaded['test']['value'] == 123
    
    def test_save_config_json(self):
        """Test saving configuration to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_config.json"
            
            config = Config()
            config.config['test'] = {'value': 456}
            config.save_config(str(save_path))
            
            assert save_path.exists()
            
            # Load and verify
            with open(save_path) as f:
                loaded = json.load(f)
                assert loaded['test']['value'] == 456
    
    def test_config_file_not_found_warning(self):
        """Test warning when config file not found."""
        with patch('src.monitoring.logger_config.get_logger') as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance
            
            Config(config_path="/nonexistent/path.yaml")
            
            logger_instance.warning.assert_called()
            args = logger_instance.warning.call_args[0]
            assert "Config file not found" in args[0]


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_config_lifecycle(self):
        """Test complete configuration lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            
            # Create initial config
            initial_data = {
                'environment': 'staging',
                'redis': {'host': 'staging-redis'},
                'api': {'port': 8001}
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(initial_data, f)
            
            # Load config
            with patch('src.security.secrets_manager.secrets_manager.get_secret') as mock_get:
                mock_get.return_value = "test-secret"
                
                config = Config(str(config_path))
                
                # Validate
                validated = config.validate_config()
                assert validated.environment == 'staging'
                assert validated.redis.host == 'staging-redis'
                assert validated.api.port == 8001
                
                # Modify
                config.set('redis.port', 6380)
                
                # Save
                config.save_config()
                
                # Reload and verify
                new_config = Config(str(config_path))
                assert new_config.get('redis.port') == 6380


if __name__ == "__main__":
    pytest.main([__file__, "-v"])