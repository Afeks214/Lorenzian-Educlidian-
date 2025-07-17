#!/usr/bin/env python3
"""
Configuration Management System Demo
Demonstrates the complete configuration management system capabilities.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import (
    ConfigManager, SecretsManager, FeatureFlagManager,
    ConfigValidator, ConfigMonitor, ConfigAutomation
)
from config.config_manager import Environment, ConfigChangeEvent
from config.secrets_manager import SecretType
from config.feature_flags import FeatureState, RolloutStrategy
from config.config_validator import ValidationLevel
from config.config_monitor import AlertSeverity, ComplianceRule
from config.config_automation import AutomationRule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigManagementDemo:
    """Complete configuration management system demonstration"""
    
    def __init__(self):
        self.demo_path = Path(__file__).parent / "demo_config_system"
        self.demo_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.config_manager = ConfigManager(Environment.DEVELOPMENT, self.demo_path / "config")
        self.secrets_manager = SecretsManager(self.demo_path / "vault")
        self.feature_manager = FeatureFlagManager(self.demo_path / "feature_flags")
        self.validator = ConfigValidator()
        self.monitor = ConfigMonitor(self.config_manager, check_interval=5)
        self.automation = ConfigAutomation(self.config_manager, self.demo_path / "backups")
        
        # Setup event handlers
        self.setup_event_handlers()
        
        logger.info("Configuration Management Demo initialized")
    
    def setup_event_handlers(self):
        """Setup event handlers for demonstration"""
        
        # Configuration change listener
        def config_change_handler(event: ConfigChangeEvent):
            logger.info(f"Configuration changed: {event.config_name} by {event.user}")
            logger.info(f"Reason: {event.reason}")
        
        self.config_manager.add_change_listener(config_change_handler)
        
        # Alert handler
        def alert_handler(alert):
            logger.warning(f"ALERT: {alert.title} - {alert.message}")
        
        self.monitor.add_alert_handler(alert_handler)
        
        # Deployment handler
        def deployment_handler(job):
            logger.info(f"Deployment {job.name} status: {job.status.value}")
        
        self.automation.add_deployment_handler(deployment_handler)
    
    def demo_basic_configuration(self):
        """Demonstrate basic configuration management"""
        logger.info("=== Basic Configuration Management Demo ===")
        
        # Create sample configuration
        sample_config = {
            'system': {
                'name': 'GrandModel Trading System',
                'version': '1.0.0',
                'mode': 'development',
                'environment': 'development'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'grandmodel_dev',
                'connection_pool': {
                    'min_connections': 2,
                    'max_connections': 20
                }
            },
            'risk_management': {
                'max_position_size': 100000,
                'max_daily_loss': 5000,
                'stop_loss_percent': 2.0,
                'position_sizing_method': 'fixed'
            }
        }
        
        # Set configuration
        self.config_manager.set_config('demo_config', sample_config, 'demo_user', 'Initial configuration')
        
        # Get configuration
        retrieved_config = self.config_manager.get_config('demo_config')
        logger.info(f"Retrieved config system name: {retrieved_config['system']['name']}")
        
        # Update configuration
        updates = {
            'system': {
                'version': '1.1.0'
            },
            'risk_management': {
                'max_position_size': 150000
            }
        }
        
        self.config_manager.update_config('demo_config', updates, 'demo_user', 'Version update')
        
        # Show versions
        versions = self.config_manager.get_versions('demo_config')
        logger.info(f"Configuration versions: {[v.version for v in versions]}")
        
        # Rollback demonstration
        if versions:
            logger.info("Demonstrating rollback...")
            self.config_manager.rollback_config('demo_config', versions[0].version, 'demo_user', 'Rollback demo')
            
            current_config = self.config_manager.get_config('demo_config')
            logger.info(f"After rollback, version: {current_config['system']['version']}")
    
    def demo_secrets_management(self):
        """Demonstrate secrets management"""
        logger.info("=== Secrets Management Demo ===")
        
        # Create secrets
        secrets_data = [
            ('db_password', 'super_secret_db_password_123!', SecretType.PASSWORD, 'Database password'),
            ('api_key', 'abc123def456ghi789jkl012mno345pqr', SecretType.API_KEY, 'External API key'),
            ('jwt_secret', 'jwt_secret_key_for_authentication_tokens', SecretType.TOKEN, 'JWT secret key')
        ]
        
        for name, value, secret_type, description in secrets_data:
            success = self.secrets_manager.create_secret(name, value, secret_type, description)
            logger.info(f"Created secret '{name}': {success}")
        
        # Demonstrate secret access
        db_password = self.secrets_manager.get_secret('db_password', 'demo_user')
        logger.info(f"Retrieved DB password: {'*' * len(db_password) if db_password else 'None'}")
        
        # Set access policy
        self.secrets_manager.set_access_policy('db_password', ['demo_user', 'admin'])
        
        # Demonstrate access control
        try:
            self.secrets_manager.get_secret('db_password', 'unauthorized_user')
        except PermissionError as e:
            logger.info(f"Access denied as expected: {e}")
        
        # Secret rotation
        success = self.secrets_manager.rotate_secret('api_key', 'new_rotated_api_key_xyz789', 'demo_user')
        logger.info(f"Secret rotation successful: {success}")
        
        # Show audit log
        audit_log = self.secrets_manager.get_audit_log(hours_back=1)
        logger.info(f"Audit log entries: {len(audit_log)}")
        
        # Apply secrets to configuration
        config_with_secrets = {
            'database': {
                'host': 'localhost',
                'password': '${SECRET:db_password}',
                'username': 'grandmodel_user'
            },
            'external_api': {
                'key': '${SECRET:api_key}',
                'endpoint': 'https://api.example.com'
            }
        }
        
        processed_config = self.secrets_manager.apply_secrets(config_with_secrets)
        logger.info(f"Config with secrets applied: password masked = {'*' * 10}")
    
    def demo_feature_flags(self):
        """Demonstrate feature flags and A/B testing"""
        logger.info("=== Feature Flags Demo ===")
        
        # Create feature flags
        feature_flags = [
            ('new_algorithm', 'New Trading Algorithm', FeatureState.ROLLOUT, 50.0),
            ('enhanced_ui', 'Enhanced User Interface', FeatureState.ENABLED, 100.0),
            ('beta_features', 'Beta Features', FeatureState.EXPERIMENT, 25.0),
            ('debug_mode', 'Debug Mode', FeatureState.DISABLED, 0.0)
        ]
        
        for name, description, state, percentage in feature_flags:
            success = self.feature_manager.create_feature_flag(
                name, description, 'demo_user', state, percentage, RolloutStrategy.PERCENTAGE
            )
            logger.info(f"Created feature flag '{name}': {success}")
        
        # Test feature evaluations
        test_users = ['user1', 'user2', 'user3', 'user4', 'user5']
        
        for user in test_users:
            evaluation = self.feature_manager.evaluate_feature('new_algorithm', user)
            logger.info(f"Feature 'new_algorithm' for {user}: {evaluation.enabled} ({evaluation.reason})")
        
        # Create A/B test
        ab_test_success = self.feature_manager.create_ab_test(
            'algorithm_test',
            'new_algorithm',
            {
                'A': {'algorithm': 'original', 'params': {'threshold': 0.5}},
                'B': {'algorithm': 'enhanced', 'params': {'threshold': 0.7}}
            },
            {'A': 0.5, 'B': 0.5},
            ['success_rate', 'latency'],
            datetime.now(),
            datetime.now() + timedelta(days=7)
        )
        
        logger.info(f"Created A/B test: {ab_test_success}")
        
        # Test A/B variant assignment
        for user in test_users:
            evaluation = self.feature_manager.evaluate_feature('new_algorithm', user)
            if evaluation.enabled:
                logger.info(f"User {user} assigned to variant: {evaluation.variant}")
        
        # Get feature analytics
        analytics = self.feature_manager.get_feature_analytics('new_algorithm', hours_back=1)
        logger.info(f"Feature analytics: {analytics}")
    
    def demo_configuration_validation(self):
        """Demonstrate configuration validation"""
        logger.info("=== Configuration Validation Demo ===")
        
        # Test valid configuration
        valid_config = {
            'system': {
                'name': 'Test System',
                'version': '1.0.0',
                'mode': 'development'
            },
            'data_handler': {
                'type': 'backtest',
                'backtest_file': 'test.csv'
            }
        }
        
        result = self.validator.validate_config_detailed('settings', valid_config)
        logger.info(f"Valid config validation result: {result.valid}")
        
        # Test invalid configuration
        invalid_config = {
            'system': {
                'name': 'Test System',
                'version': 'invalid-version',  # Invalid version format
                'mode': 'invalid_mode'  # Invalid mode
            },
            'data_handler': {
                'type': 'invalid_type'  # Invalid type
            }
        }
        
        result = self.validator.validate_config_detailed('settings', invalid_config)
        logger.info(f"Invalid config validation result: {result.valid}")
        
        if result.errors:
            logger.info("Validation errors:")
            for error in result.errors:
                logger.info(f"  - {error.field}: {error.message}")
        
        # Generate schema from sample data
        schema = self.validator.generate_schema('demo_config', valid_config)
        logger.info(f"Generated schema properties: {list(schema['properties'].keys())}")
    
    def demo_monitoring_alerting(self):
        """Demonstrate monitoring and alerting"""
        logger.info("=== Monitoring and Alerting Demo ===")
        
        # Create test configuration with potential issues
        test_config = {
            'system': {
                'name': 'Test System',
                'version': '1.0.0',
                'ssl_enabled': False  # Will trigger compliance alert in production
            },
            'database': {
                'password': 'weak',  # Will trigger password strength alert
                'host': 'localhost'
            },
            'risk_management': {
                'max_position_size': 2000000  # Will trigger position size alert
            }
        }
        
        self.config_manager.set_config('monitored_config', test_config, 'demo_user', 'Test config for monitoring')
        
        # Trigger compliance check
        self.monitor._check_compliance()
        
        # Get alerts
        alerts = self.monitor.get_alerts(hours_back=1)
        logger.info(f"Generated alerts: {len(alerts)}")
        
        for alert in alerts:
            logger.info(f"Alert: {alert.title} - {alert.severity.value}")
        
        # Get compliance status
        compliance = self.monitor.get_compliance_status()
        logger.info(f"Compliance status: {compliance}")
        
        # Create monitoring report
        report = self.monitor.create_monitoring_report(hours_back=1)
        logger.info(f"Monitoring report sections: {list(report.keys())}")
        
        # Add custom compliance rule
        custom_rule = ComplianceRule(
            id='custom_demo_rule',
            name='Demo Custom Rule',
            description='Custom rule for demonstration',
            category='demo',
            severity=AlertSeverity.INFO,
            config_path='system.name',
            validation_func='validate_system_name',
            parameters={'required_prefix': 'GrandModel'}
        )
        
        # Add custom validation function
        def validate_system_name(name, params):
            required_prefix = params.get('required_prefix', '')
            return name.startswith(required_prefix) if name else False
        
        # This would be added to the validator in a real implementation
        logger.info(f"Custom rule created: {custom_rule.name}")
    
    def demo_automation(self):
        """Demonstrate configuration automation"""
        logger.info("=== Configuration Automation Demo ===")
        
        # Create initial configuration
        initial_config = {
            'system': {
                'name': 'GrandModel System',
                'version': '1.0.0',
                'environment': 'development'
            },
            'features': {
                'new_feature': False,
                'enhanced_logging': True
            }
        }
        
        self.config_manager.set_config('automation_demo', initial_config, 'demo_user', 'Initial config')
        
        # Create backup
        backup_id = self.automation.create_backup(
            'demo_backup',
            'Backup for automation demo',
            ['automation_demo'],
            retention_days=7
        )
        
        logger.info(f"Created backup: {backup_id}")
        
        # Create deployment
        config_changes = {
            'automation_demo': {
                'system': {
                    'name': 'GrandModel System',
                    'version': '1.1.0',
                    'environment': 'development'
                },
                'features': {
                    'new_feature': True,
                    'enhanced_logging': True,
                    'beta_mode': True
                }
            }
        }
        
        deployment_id = self.automation.create_deployment(
            'Demo Deployment',
            'Demonstration deployment',
            config_changes,
            'development'
        )
        
        logger.info(f"Created deployment: {deployment_id}")
        
        # Wait for deployment to process
        time.sleep(2)
        
        # Check deployment status
        deployment = self.automation.get_deployment_status(deployment_id)
        logger.info(f"Deployment status: {deployment.status.value}")
        
        # Simulate configuration drift
        self.automation.baseline_checksums['automation_demo'] = 'fake_original_checksum'
        
        # Check for drift
        self.automation._check_drift()
        
        # Get drift detections
        drift_detections = self.automation.get_drift_detections()
        logger.info(f"Drift detections: {len(drift_detections)}")
        
        # Get automation status
        status = self.automation.get_automation_status()
        logger.info(f"Automation status: {status}")
        
        # Add custom automation rule
        custom_rule = AutomationRule(
            id='demo_rule',
            name='Demo Automation Rule',
            description='Demonstration automation rule',
            trigger_type='schedule',
            trigger_config={'interval': 'daily'},
            actions=['backup_configs']
        )
        
        success = self.automation.add_automation_rule(custom_rule)
        logger.info(f"Added automation rule: {success}")
    
    def demo_integration(self):
        """Demonstrate system integration"""
        logger.info("=== System Integration Demo ===")
        
        # Create configuration with secrets
        config_with_secrets = {
            'system': {
                'name': 'Integrated System',
                'version': '1.0.0'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'password': '${SECRET:db_password}'
            },
            'api': {
                'key': '${SECRET:api_key}',
                'endpoint': 'https://api.example.com'
            }
        }
        
        # Set configuration
        self.config_manager.set_config('integrated_config', config_with_secrets, 'demo_user', 'Integrated config')
        
        # Apply secrets
        processed_config = self.secrets_manager.apply_secrets(config_with_secrets)
        logger.info(f"Configuration with secrets applied successfully")
        
        # Create feature flag for the configuration
        self.feature_manager.create_feature_flag(
            'integrated_feature',
            'Integrated Feature',
            'demo_user',
            FeatureState.ENABLED,
            100.0
        )
        
        # Check feature flag in configuration context
        feature_enabled = self.feature_manager.is_feature_enabled('integrated_feature', 'demo_user')
        logger.info(f"Feature enabled in integrated context: {feature_enabled}")
        
        # Create backup of integrated system
        backup_id = self.automation.create_backup(
            'integrated_backup',
            'Backup of integrated system',
            ['integrated_config'],
            retention_days=30
        )
        
        # Get system status
        config_status = self.config_manager.get_config_status()
        secrets_status = self.secrets_manager.get_secrets_status()
        features_status = self.feature_manager.get_system_status()
        monitoring_stats = self.monitor.get_monitoring_stats()
        automation_status = self.automation.get_automation_status()
        
        logger.info("=== System Status Summary ===")
        logger.info(f"Configurations: {config_status['configs_loaded']}")
        logger.info(f"Secrets: {secrets_status['total_secrets']}")
        logger.info(f"Feature Flags: {features_status['total_flags']}")
        logger.info(f"Active Alerts: {monitoring_stats.active_alerts}")
        logger.info(f"Automation Active: {automation_status['automation_active']}")
    
    def run_demo(self):
        """Run the complete demonstration"""
        logger.info("Starting Configuration Management System Demo")
        
        try:
            # Run all demo sections
            self.demo_basic_configuration()
            self.demo_secrets_management()
            self.demo_feature_flags()
            self.demo_configuration_validation()
            self.demo_monitoring_alerting()
            self.demo_automation()
            self.demo_integration()
            
            logger.info("Configuration Management System Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Clean up demo resources"""
        logger.info("Cleaning up demo resources...")
        
        # Stop monitoring and automation
        self.monitor.stop_monitoring()
        self.automation.stop_automation()
        
        # Clean up demo directory
        import shutil
        if self.demo_path.exists():
            shutil.rmtree(self.demo_path)
        
        logger.info("Demo cleanup completed")


def main():
    """Main demo function"""
    print("GrandModel Configuration Management System Demo")
    print("=" * 50)
    
    # Create and run demo
    demo = ConfigManagementDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()