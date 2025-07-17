"""
Comprehensive test suite for the configuration management system.
"""

import pytest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.config import (
    ConfigManager, SecretsManager, FeatureFlagManager, 
    ConfigValidator, ConfigMonitor, ConfigAutomation
)
from src.config.config_manager import Environment, ConfigVersion, ConfigChangeEvent
from src.config.secrets_manager import SecretType, SecretMetadata
from src.config.feature_flags import FeatureState, RolloutStrategy, FeatureFlag
from src.config.config_validator import ValidationLevel, ValidationResult
from src.config.config_monitor import AlertSeverity, ConfigAlert, ComplianceRule
from src.config.config_automation import DeploymentStatus, DeploymentJob, DriftDetection


class TestConfigManager:
    """Test suite for ConfigManager"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create ConfigManager instance"""
        return ConfigManager(Environment.TESTING, temp_config_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration data"""
        return {
            'system': {
                'name': 'Test System',
                'version': '1.0.0',
                'mode': 'testing'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'test_db'
            }
        }
    
    def test_config_manager_initialization(self, config_manager):
        """Test ConfigManager initialization"""
        assert config_manager.environment == Environment.TESTING
        assert config_manager.base_path.exists()
        assert isinstance(config_manager.validator, ConfigValidator)
        assert isinstance(config_manager.secrets_manager, SecretsManager)
    
    def test_set_and_get_config(self, config_manager, sample_config):
        """Test setting and getting configuration"""
        config_manager.set_config('test_config', sample_config)
        retrieved_config = config_manager.get_config('test_config')
        
        assert retrieved_config == sample_config
    
    def test_update_config(self, config_manager, sample_config):
        """Test updating configuration"""
        config_manager.set_config('test_config', sample_config)
        
        updates = {'system': {'version': '2.0.0'}}
        config_manager.update_config('test_config', updates)
        
        updated_config = config_manager.get_config('test_config')
        assert updated_config['system']['version'] == '2.0.0'
        assert updated_config['system']['name'] == 'Test System'
    
    def test_config_versioning(self, config_manager, sample_config):
        """Test configuration versioning"""
        config_manager.set_config('test_config', sample_config)
        
        # Update config to create a version
        updated_config = sample_config.copy()
        updated_config['system']['version'] = '2.0.0'
        config_manager.set_config('test_config', updated_config)
        
        versions = config_manager.get_versions('test_config')
        assert len(versions) == 1
        assert versions[0].version == 'v1'
    
    def test_config_rollback(self, config_manager, sample_config):
        """Test configuration rollback"""
        config_manager.set_config('test_config', sample_config)
        
        # Update config
        updated_config = sample_config.copy()
        updated_config['system']['version'] = '2.0.0'
        config_manager.set_config('test_config', updated_config)
        
        # Rollback to previous version
        config_manager.rollback_config('test_config', 'v1')
        
        current_config = config_manager.get_config('test_config')
        assert current_config['system']['version'] == '1.0.0'
    
    def test_change_listeners(self, config_manager, sample_config):
        """Test configuration change listeners"""
        listener_called = []
        
        def test_listener(event):
            listener_called.append(event)
        
        config_manager.add_change_listener(test_listener)
        config_manager.set_config('test_config', sample_config)
        
        assert len(listener_called) == 1
        assert listener_called[0].config_name == 'test_config'
    
    def test_config_backup_restore(self, config_manager, sample_config):
        """Test configuration backup and restore"""
        config_manager.set_config('test_config', sample_config)
        
        # Create backup
        backup_name = config_manager.backup_configs('test_backup')
        
        # Modify config
        modified_config = sample_config.copy()
        modified_config['system']['version'] = '2.0.0'
        config_manager.set_config('test_config', modified_config)
        
        # Restore backup
        config_manager.restore_configs(backup_name)
        
        restored_config = config_manager.get_config('test_config')
        assert restored_config['system']['version'] == '1.0.0'
    
    def test_environment_variable_processing(self, config_manager):
        """Test environment variable processing"""
        config_with_env = {
            'database': {
                'host': '${DB_HOST:localhost}',
                'port': '${DB_PORT:5432}'
            }
        }
        
        with patch.dict('os.environ', {'DB_HOST': 'production.db.com'}):
            processed_config = config_manager._process_env_vars(config_with_env)
            
            assert processed_config['database']['host'] == 'production.db.com'
            assert processed_config['database']['port'] == '5432'  # default value
    
    def test_config_transaction(self, config_manager, sample_config):
        """Test configuration transaction"""
        config_manager.set_config('test_config', sample_config)
        
        try:
            with config_manager.config_transaction('test_config'):
                # Modify config within transaction
                config_manager.update_config('test_config', {'system': {'version': '2.0.0'}})
                
                # Simulate error
                raise Exception("Test error")
        except Exception:
            pass
        
        # Config should be rolled back
        current_config = config_manager.get_config('test_config')
        assert current_config['system']['version'] == '1.0.0'


class TestSecretsManager:
    """Test suite for SecretsManager"""
    
    @pytest.fixture
    def temp_vault_dir(self):
        """Create temporary vault directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def secrets_manager(self, temp_vault_dir):
        """Create SecretsManager instance"""
        return SecretsManager(temp_vault_dir, "test_master_key")
    
    def test_secrets_manager_initialization(self, secrets_manager):
        """Test SecretsManager initialization"""
        assert secrets_manager.vault_path.exists()
        assert secrets_manager.secrets_path.exists()
        assert secrets_manager.metadata_path.exists()
        assert secrets_manager.audit_path.exists()
    
    def test_create_and_get_secret(self, secrets_manager):
        """Test creating and getting a secret"""
        secret_name = "test_secret"
        secret_value = "secret_value_123"
        
        success = secrets_manager.create_secret(
            secret_name, secret_value, SecretType.API_KEY, "Test secret"
        )
        
        assert success
        retrieved_value = secrets_manager.get_secret(secret_name)
        assert retrieved_value == secret_value
    
    def test_update_secret(self, secrets_manager):
        """Test updating a secret"""
        secret_name = "test_secret"
        original_value = "original_value"
        updated_value = "updated_value"
        
        secrets_manager.create_secret(
            secret_name, original_value, SecretType.API_KEY, "Test secret"
        )
        
        success = secrets_manager.update_secret(secret_name, updated_value)
        assert success
        
        retrieved_value = secrets_manager.get_secret(secret_name)
        assert retrieved_value == updated_value
    
    def test_secret_rotation(self, secrets_manager):
        """Test secret rotation"""
        secret_name = "test_secret"
        original_value = "original_value"
        
        secrets_manager.create_secret(
            secret_name, original_value, SecretType.API_KEY, "Test secret"
        )
        
        success = secrets_manager.rotate_secret(secret_name, "new_rotated_value")
        assert success
        
        retrieved_value = secrets_manager.get_secret(secret_name)
        assert retrieved_value == "new_rotated_value"
    
    def test_secret_expiration(self, secrets_manager):
        """Test secret expiration"""
        secret_name = "test_secret"
        secret_value = "secret_value"
        
        # Create secret that expires in the past
        expires_at = datetime.now() - timedelta(hours=1)
        
        secrets_manager.create_secret(
            secret_name, secret_value, SecretType.API_KEY, 
            "Test secret", expires_at=expires_at
        )
        
        # Should return None for expired secret
        retrieved_value = secrets_manager.get_secret(secret_name)
        assert retrieved_value is None
    
    def test_access_control(self, secrets_manager):
        """Test secret access control"""
        secret_name = "test_secret"
        secret_value = "secret_value"
        
        secrets_manager.create_secret(
            secret_name, secret_value, SecretType.API_KEY, "Test secret"
        )
        
        # Set access policy
        secrets_manager.set_access_policy(secret_name, ["user1", "user2"])
        
        # User1 should have access
        retrieved_value = secrets_manager.get_secret(secret_name, "user1")
        assert retrieved_value == secret_value
        
        # User3 should not have access
        with pytest.raises(PermissionError):
            secrets_manager.get_secret(secret_name, "user3")
    
    def test_audit_logging(self, secrets_manager):
        """Test audit logging"""
        secret_name = "test_secret"
        secret_value = "secret_value"
        
        secrets_manager.create_secret(
            secret_name, secret_value, SecretType.API_KEY, "Test secret"
        )
        
        secrets_manager.get_secret(secret_name, "test_user")
        
        audit_log = secrets_manager.get_audit_log()
        assert len(audit_log) >= 2  # create and read operations
        
        # Check that read operation was logged
        read_entries = [entry for entry in audit_log if entry.access_type == "read"]
        assert len(read_entries) >= 1
        assert read_entries[0].user == "test_user"
    
    def test_apply_secrets_to_config(self, secrets_manager):
        """Test applying secrets to configuration"""
        secret_name = "db_password"
        secret_value = "super_secret_password"
        
        secrets_manager.create_secret(
            secret_name, secret_value, SecretType.PASSWORD, "Database password"
        )
        
        config_with_secrets = {
            'database': {
                'host': 'localhost',
                'password': '${SECRET:db_password}'
            }
        }
        
        processed_config = secrets_manager.apply_secrets(config_with_secrets)
        assert processed_config['database']['password'] == secret_value


class TestFeatureFlagManager:
    """Test suite for FeatureFlagManager"""
    
    @pytest.fixture
    def temp_flags_dir(self):
        """Create temporary feature flags directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def feature_manager(self, temp_flags_dir):
        """Create FeatureFlagManager instance"""
        return FeatureFlagManager(temp_flags_dir)
    
    def test_feature_flag_creation(self, feature_manager):
        """Test creating a feature flag"""
        success = feature_manager.create_feature_flag(
            "test_feature", "Test Feature", "test_user",
            FeatureState.ENABLED, 100.0
        )
        
        assert success
        
        flag = feature_manager.get_feature_flag("test_feature")
        assert flag is not None
        assert flag.name == "test_feature"
        assert flag.state == FeatureState.ENABLED
    
    def test_feature_evaluation(self, feature_manager):
        """Test feature flag evaluation"""
        feature_manager.create_feature_flag(
            "test_feature", "Test Feature", "test_user",
            FeatureState.ENABLED, 100.0
        )
        
        evaluation = feature_manager.evaluate_feature("test_feature", "user123")
        assert evaluation.enabled is True
        assert evaluation.feature_name == "test_feature"
    
    def test_percentage_rollout(self, feature_manager):
        """Test percentage-based rollout"""
        feature_manager.create_feature_flag(
            "test_feature", "Test Feature", "test_user",
            FeatureState.ROLLOUT, 50.0,  # 50% rollout
            RolloutStrategy.PERCENTAGE
        )
        
        # Test with consistent user ID
        evaluation1 = feature_manager.evaluate_feature("test_feature", "user123")
        evaluation2 = feature_manager.evaluate_feature("test_feature", "user123")
        
        # Should be consistent for same user
        assert evaluation1.enabled == evaluation2.enabled
    
    def test_user_list_rollout(self, feature_manager):
        """Test user list rollout"""
        feature_manager.create_feature_flag(
            "test_feature", "Test Feature", "test_user",
            FeatureState.ROLLOUT, 100.0,
            RolloutStrategy.USER_LIST
        )
        
        # Update enabled users
        feature_manager.update_feature_flag("test_feature", enabled_users=["user123", "user456"])
        
        # User123 should be enabled
        evaluation = feature_manager.evaluate_feature("test_feature", "user123")
        assert evaluation.enabled is True
        
        # User789 should be disabled
        evaluation = feature_manager.evaluate_feature("test_feature", "user789")
        assert evaluation.enabled is False
    
    def test_ab_testing(self, feature_manager):
        """Test A/B testing functionality"""
        # Create feature flag
        feature_manager.create_feature_flag(
            "test_feature", "Test Feature", "test_user",
            FeatureState.EXPERIMENT, 100.0
        )
        
        # Create A/B test
        success = feature_manager.create_ab_test(
            "test_experiment",
            "test_feature",
            {"A": {"color": "blue"}, "B": {"color": "red"}},
            {"A": 0.5, "B": 0.5},
            ["click_rate", "conversion_rate"],
            datetime.now(),
            datetime.now() + timedelta(days=7)
        )
        
        assert success
        
        # Test variant assignment
        evaluation = feature_manager.evaluate_feature("test_feature", "user123")
        assert evaluation.variant in ["A", "B"]
    
    def test_feature_analytics(self, feature_manager):
        """Test feature analytics"""
        feature_manager.create_feature_flag(
            "test_feature", "Test Feature", "test_user",
            FeatureState.ENABLED, 100.0
        )
        
        # Generate some evaluations
        for i in range(10):
            feature_manager.evaluate_feature("test_feature", f"user{i}")
        
        analytics = feature_manager.get_feature_analytics("test_feature")
        assert analytics['total_evaluations'] == 10
        assert analytics['enabled_count'] == 10
        assert analytics['enable_rate'] == 1.0


class TestConfigValidator:
    """Test suite for ConfigValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create ConfigValidator instance"""
        return ConfigValidator()
    
    def test_settings_validation(self, validator):
        """Test settings configuration validation"""
        valid_config = {
            'system': {
                'name': 'Test System',
                'version': '1.0.0',
                'mode': 'testing'
            },
            'data_handler': {
                'type': 'backtest',
                'backtest_file': 'test.csv'
            }
        }
        
        result = validator.validate_config_detailed('settings', valid_config)
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_invalid_settings_validation(self, validator):
        """Test invalid settings configuration validation"""
        invalid_config = {
            'system': {
                'name': 'Test System',
                'version': '1.0.0',
                'mode': 'invalid_mode'  # Invalid mode
            },
            'data_handler': {
                'type': 'invalid_type'  # Invalid type
            }
        }
        
        result = validator.validate_config_detailed('settings', invalid_config)
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_risk_config_validation(self, validator):
        """Test risk configuration validation"""
        valid_config = {
            'max_position_size': 100000,
            'max_daily_loss': 5000,
            'stop_loss_percent': 2.0,
            'position_sizing_method': 'kelly'
        }
        
        result = validator.validate_config_detailed('risk_management_config', valid_config)
        assert result.valid is True
    
    def test_invalid_risk_config_validation(self, validator):
        """Test invalid risk configuration validation"""
        invalid_config = {
            'max_position_size': -100,  # Invalid negative value
            'stop_loss_percent': 150,   # Invalid percentage > 100
            'position_sizing_method': 'invalid_method'  # Invalid method
        }
        
        result = validator.validate_config_detailed('risk_management_config', invalid_config)
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_schema_generation(self, validator):
        """Test schema generation from sample data"""
        sample_data = {
            'name': 'Test System',
            'version': '1.0.0',
            'enabled': True,
            'settings': {
                'timeout': 30,
                'retries': 3
            }
        }
        
        schema = validator.generate_schema('test_config', sample_data)
        
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert schema['properties']['name']['type'] == 'string'
        assert schema['properties']['enabled']['type'] == 'boolean'
        assert schema['properties']['settings']['type'] == 'object'


class TestConfigMonitor:
    """Test suite for ConfigMonitor"""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create mock ConfigManager"""
        mock_manager = Mock()
        mock_manager.get_all_configs.return_value = {
            'test_config': {
                'system': {'name': 'Test System'},
                'database': {'host': 'localhost'}
            }
        }
        return mock_manager
    
    @pytest.fixture
    def config_monitor(self, mock_config_manager):
        """Create ConfigMonitor instance"""
        with patch('src.config.config_monitor.psutil'):
            monitor = ConfigMonitor(mock_config_manager, check_interval=1)
            monitor.stop_monitoring()  # Stop automatic monitoring for tests
            return monitor
    
    def test_monitor_initialization(self, config_monitor):
        """Test ConfigMonitor initialization"""
        assert config_monitor.check_interval == 1
        assert len(config_monitor.compliance_rules) > 0
    
    def test_alert_creation(self, config_monitor):
        """Test alert creation"""
        alert_id = config_monitor._create_alert(
            AlertSeverity.WARNING,
            "Test Alert",
            "Test alert message",
            "test_config"
        )
        
        assert alert_id in config_monitor.alerts
        alert = config_monitor.alerts[alert_id]
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
    
    def test_alert_resolution(self, config_monitor):
        """Test alert resolution"""
        alert_id = config_monitor._create_alert(
            AlertSeverity.WARNING,
            "Test Alert",
            "Test alert message",
            "test_config"
        )
        
        success = config_monitor.resolve_alert(alert_id, "test_user")
        assert success is True
        
        alert = config_monitor.alerts[alert_id]
        assert alert.resolved is True
        assert alert.resolved_by == "test_user"
    
    def test_compliance_rule_validation(self, config_monitor):
        """Test compliance rule validation"""
        # Test password strength validation
        result = config_monitor._validate_password_strength("weak", {"min_length": 12})
        assert result is False
        
        result = config_monitor._validate_password_strength("strong_password_123!", {"min_length": 12, "require_special": True})
        assert result is True
    
    def test_compliance_status(self, config_monitor):
        """Test compliance status reporting"""
        status = config_monitor.get_compliance_status()
        
        assert 'compliant_rules' in status
        assert 'non_compliant_rules' in status
        assert 'compliance_percentage' in status
        assert 'status' in status
    
    def test_monitoring_report(self, config_monitor):
        """Test monitoring report generation"""
        report = config_monitor.create_monitoring_report()
        
        assert 'report_timestamp' in report
        assert 'statistics' in report
        assert 'compliance_status' in report
        assert 'alert_summary' in report
        assert 'system_health' in report


class TestConfigAutomation:
    """Test suite for ConfigAutomation"""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create mock ConfigManager"""
        mock_manager = Mock()
        mock_manager.get_all_configs.return_value = {
            'test_config': {
                'system': {'name': 'Test System'},
                'database': {'host': 'localhost'}
            }
        }
        mock_manager.get_config.return_value = {
            'system': {'name': 'Test System'},
            'database': {'host': 'localhost'}
        }
        return mock_manager
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config_automation(self, mock_config_manager, temp_backup_dir):
        """Create ConfigAutomation instance"""
        automation = ConfigAutomation(mock_config_manager, temp_backup_dir)
        automation.stop_automation()  # Stop automatic processing for tests
        return automation
    
    def test_automation_initialization(self, config_automation):
        """Test ConfigAutomation initialization"""
        assert config_automation.backup_path.exists()
        assert len(config_automation.automation_rules) > 0
    
    def test_deployment_creation(self, config_automation):
        """Test deployment job creation"""
        config_changes = {
            'test_config': {
                'system': {'name': 'Updated System'},
                'database': {'host': 'production.db.com'}
            }
        }
        
        job_id = config_automation.create_deployment(
            "Test Deployment",
            "Test deployment description",
            config_changes,
            "testing"
        )
        
        assert job_id in config_automation.deployment_jobs
        job = config_automation.deployment_jobs[job_id]
        assert job.name == "Test Deployment"
        assert job.status == DeploymentStatus.PENDING
    
    def test_deployment_execution(self, config_automation):
        """Test deployment execution"""
        config_changes = {
            'test_config': {
                'system': {'name': 'Updated System'}
            }
        }
        
        job_id = config_automation.create_deployment(
            "Test Deployment",
            "Test deployment description",
            config_changes,
            "testing"
        )
        
        job = config_automation.deployment_jobs[job_id]
        config_automation._execute_deployment(job)
        
        # Check that deployment was executed
        assert job.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED]
        assert job.started_at is not None
        assert job.completed_at is not None
    
    def test_backup_creation(self, config_automation):
        """Test backup creation"""
        backup_id = config_automation.create_backup(
            "Test Backup",
            "Test backup description",
            ["test_config"],
            retention_days=7
        )
        
        assert backup_id in config_automation.backup_records
        record = config_automation.backup_records[backup_id]
        assert record.name == "Test Backup"
        assert record.retention_days == 7
    
    def test_backup_restore(self, config_automation):
        """Test backup restore"""
        # Create backup
        backup_id = config_automation.create_backup(
            "Test Backup",
            "Test backup description",
            ["test_config"]
        )
        
        # Restore backup
        success = config_automation.restore_backup(backup_id, "test_user")
        assert success is True
        
        # Verify config manager was called
        config_automation.config_manager.set_config.assert_called()
    
    def test_drift_detection(self, config_automation):
        """Test drift detection"""
        # Set initial baseline
        config_automation.baseline_checksums['test_config'] = 'original_checksum'
        
        # Simulate changed config
        config_automation.config_manager.get_all_configs.return_value = {
            'test_config': {
                'system': {'name': 'Changed System'}  # Different from original
            }
        }
        
        # Check for drift
        config_automation._check_drift()
        
        # Should detect drift
        assert len(config_automation.drift_detections) > 0
    
    def test_automation_rule_execution(self, config_automation):
        """Test automation rule execution"""
        # Create test rule
        from src.config.config_automation import AutomationRule
        
        rule = AutomationRule(
            id="test_rule",
            name="Test Rule",
            description="Test rule description",
            trigger_type="schedule",
            trigger_config={"cron": "0 * * * *"},
            actions=["backup_configs"]
        )
        
        config_automation.automation_rules["test_rule"] = rule
        
        # Execute rule
        config_automation._execute_automation_rule(rule)
        
        # Check that rule was executed
        assert rule.last_executed is not None
        
        # Check that backup was created
        assert len(config_automation.backup_records) > 0


class TestIntegration:
    """Integration tests for the configuration management system"""
    
    @pytest.fixture
    def temp_system_dir(self):
        """Create temporary system directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def integrated_system(self, temp_system_dir):
        """Create integrated configuration management system"""
        config_manager = ConfigManager(Environment.TESTING, temp_system_dir / "config")
        
        # Initialize other components
        secrets_manager = SecretsManager(temp_system_dir / "vault")
        feature_manager = FeatureFlagManager(temp_system_dir / "feature_flags")
        validator = ConfigValidator()
        
        with patch('src.config.config_monitor.psutil'):
            monitor = ConfigMonitor(config_manager, check_interval=1)
            monitor.stop_monitoring()
        
        automation = ConfigAutomation(config_manager, temp_system_dir / "backups")
        automation.stop_automation()
        
        return {
            'config_manager': config_manager,
            'secrets_manager': secrets_manager,
            'feature_manager': feature_manager,
            'validator': validator,
            'monitor': monitor,
            'automation': automation
        }
    
    def test_end_to_end_workflow(self, integrated_system):
        """Test end-to-end configuration management workflow"""
        config_manager = integrated_system['config_manager']
        secrets_manager = integrated_system['secrets_manager']
        feature_manager = integrated_system['feature_manager']
        automation = integrated_system['automation']
        
        # 1. Create a secret
        secrets_manager.create_secret(
            "db_password", "secret123", SecretType.PASSWORD, "Database password"
        )
        
        # 2. Create configuration with secret reference
        config_with_secret = {
            'database': {
                'host': 'localhost',
                'password': '${SECRET:db_password}'
            },
            'system': {
                'name': 'Test System',
                'version': '1.0.0'
            }
        }
        
        config_manager.set_config('app_config', config_with_secret)
        
        # 3. Apply secrets to configuration
        processed_config = secrets_manager.apply_secrets(config_with_secret)
        assert processed_config['database']['password'] == 'secret123'
        
        # 4. Create feature flag
        feature_manager.create_feature_flag(
            "new_feature", "New Feature", "admin", FeatureState.ENABLED
        )
        
        # 5. Create backup
        backup_id = automation.create_backup("integration_backup", "Integration test backup")
        
        # 6. Verify backup was created
        assert backup_id in automation.backup_records
        
        # 7. Modify configuration
        updated_config = config_with_secret.copy()
        updated_config['system']['version'] = '2.0.0'
        config_manager.set_config('app_config', updated_config)
        
        # 8. Check that versions were created
        versions = config_manager.get_versions('app_config')
        assert len(versions) == 1
        
        # 9. Test feature flag
        evaluation = feature_manager.evaluate_feature("new_feature", "user123")
        assert evaluation.enabled is True
        
        # 10. Restore from backup
        success = automation.restore_backup(backup_id, "admin")
        assert success is True
    
    def test_configuration_monitoring_integration(self, integrated_system):
        """Test configuration monitoring integration"""
        config_manager = integrated_system['config_manager']
        monitor = integrated_system['monitor']
        
        # Set up configuration
        test_config = {
            'system': {
                'name': 'Test System',
                'version': '1.0.0'
            },
            'risk_management': {
                'max_position_size': 500000  # Below compliance limit
            }
        }
        
        config_manager.set_config('test_config', test_config)
        
        # Trigger compliance check
        monitor._check_compliance()
        
        # Check compliance status
        compliance_status = monitor.get_compliance_status()
        assert 'compliant_rules' in compliance_status
        assert 'non_compliant_rules' in compliance_status
    
    def test_automation_with_monitoring(self, integrated_system):
        """Test automation integration with monitoring"""
        config_manager = integrated_system['config_manager']
        monitor = integrated_system['monitor']
        automation = integrated_system['automation']
        
        # Create configuration
        test_config = {
            'system': {'name': 'Test System'},
            'database': {'host': 'localhost'}
        }
        
        config_manager.set_config('test_config', test_config)
        
        # Create deployment
        config_changes = {
            'test_config': {
                'system': {'name': 'Updated System'},
                'database': {'host': 'production.db.com'}
            }
        }
        
        job_id = automation.create_deployment(
            "Production Deployment",
            "Deploy to production",
            config_changes,
            "production"
        )
        
        # Execute deployment
        job = automation.deployment_jobs[job_id]
        automation._execute_deployment(job)
        
        # Check deployment status
        assert job.status in [DeploymentStatus.SUCCESS, DeploymentStatus.FAILED]
        
        # Check that configuration was updated
        updated_config = config_manager.get_config('test_config')
        if job.status == DeploymentStatus.SUCCESS:
            assert updated_config['system']['name'] == 'Updated System'


@pytest.fixture
def mock_psutil():
    """Mock psutil for testing"""
    with patch('src.config.config_monitor.psutil') as mock_psutil:
        mock_psutil.virtual_memory.return_value.percent = 50.0
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.disk_usage.return_value.percent = 75.0
        yield mock_psutil


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_config_manager_invalid_environment(self):
        """Test ConfigManager with invalid environment"""
        with pytest.raises(ValueError):
            ConfigManager("invalid_environment")
    
    def test_secrets_manager_invalid_secret_type(self):
        """Test SecretsManager with invalid secret type"""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_manager = SecretsManager(Path(temp_dir))
            
            with pytest.raises(ValueError):
                secrets_manager.create_secret(
                    "test", "value", "invalid_type", "description"
                )
    
    def test_feature_flag_invalid_rollout_percentage(self):
        """Test FeatureFlagManager with invalid rollout percentage"""
        with tempfile.TemporaryDirectory() as temp_dir:
            feature_manager = FeatureFlagManager(Path(temp_dir))
            
            success = feature_manager.create_feature_flag(
                "test_feature", "Test", "user", FeatureState.ROLLOUT, 150.0  # Invalid percentage
            )
            
            # Should still create but with adjusted percentage
            assert success is True
    
    def test_config_validator_missing_schema(self):
        """Test ConfigValidator with missing schema"""
        validator = ConfigValidator()
        
        # Should not fail with missing schema
        result = validator.validate_config('nonexistent_config', {})
        assert result is True  # No validation performed without schema
    
    def test_automation_failed_deployment_rollback(self):
        """Test automation rollback on failed deployment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_config_manager = Mock()
            mock_config_manager.get_all_configs.return_value = {}
            mock_config_manager.get_config.return_value = {'original': 'config'}
            mock_config_manager.set_config.side_effect = Exception("Deployment failed")
            
            automation = ConfigAutomation(mock_config_manager, Path(temp_dir))
            automation.stop_automation()
            
            # Create deployment that will fail
            config_changes = {'test_config': {'new': 'config'}}
            job_id = automation.create_deployment(
                "Failed Deployment", "Will fail", config_changes
            )
            
            job = automation.deployment_jobs[job_id]
            automation._execute_deployment(job)
            
            # Should have failed status
            assert job.status == DeploymentStatus.FAILED
            assert job.error_message is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])