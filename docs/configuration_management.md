# Configuration Management System

## Overview

The GrandModel Configuration Management System provides enterprise-grade configuration management with centralized configuration, dynamic updates, secrets management, feature flags, monitoring, and automation capabilities.

## Architecture

### Core Components

1. **ConfigManager**: Centralized configuration management with environment-specific overrides
2. **SecretsManager**: Secure storage and management of sensitive data
3. **FeatureFlagManager**: Dynamic feature flags and A/B testing
4. **ConfigValidator**: Configuration validation and compliance checking
5. **ConfigMonitor**: Real-time monitoring and alerting
6. **ConfigAutomation**: Automated deployment and drift detection

## Features

### 1. Centralized Configuration

- Environment-specific configurations (production, staging, development, testing)
- Configuration versioning and rollback capabilities
- Runtime configuration updates without restart
- Configuration validation with JSON schemas
- Environment variable substitution

### 2. Secrets Management

- Encrypted storage of sensitive data
- Access control and audit logging
- Secret rotation capabilities
- Integration with configuration system
- Compliance with security standards

### 3. Feature Flags & A/B Testing

- Dynamic feature toggles
- Multiple rollout strategies (percentage, user list, hash-based)
- A/B testing framework
- Real-time feature analytics
- Gradual rollout capabilities

### 4. Monitoring & Alerting

- Real-time configuration monitoring
- Compliance checking with customizable rules
- Alert management and resolution
- Performance metrics collection
- Health monitoring

### 5. Automation

- Automated configuration deployment
- Drift detection and correction
- Backup and recovery automation
- CI/CD integration
- Compliance automation

## Usage

### Basic Configuration Management

```python
from src.config import ConfigManager, Environment

# Initialize configuration manager
config_manager = ConfigManager(Environment.PRODUCTION)

# Get configuration
config = config_manager.get_config('settings')

# Update configuration
config_manager.update_config('settings', {
    'system': {'version': '2.0.0'}
}, user='admin', reason='Version update')

# Rollback configuration
config_manager.rollback_config('settings', 'v1', user='admin')
```

### Secrets Management

```python
from src.config import SecretsManager, SecretType

# Initialize secrets manager
secrets_manager = SecretsManager()

# Create a secret
secrets_manager.create_secret(
    'database_password',
    'super_secret_password',
    SecretType.PASSWORD,
    'Database password for production'
)

# Get secret
password = secrets_manager.get_secret('database_password')

# Rotate secret
secrets_manager.rotate_secret('database_password', 'new_password')
```

### Feature Flags

```python
from src.config import FeatureFlagManager, FeatureState

# Initialize feature flag manager
feature_manager = FeatureFlagManager()

# Create feature flag
feature_manager.create_feature_flag(
    'new_algorithm',
    'New trading algorithm',
    'engineering_team',
    FeatureState.ROLLOUT,
    enabled_percentage=25.0
)

# Check if feature is enabled
is_enabled = feature_manager.is_feature_enabled('new_algorithm', user_id='user123')

# Evaluate feature with details
evaluation = feature_manager.evaluate_feature('new_algorithm', user_id='user123')
```

### Configuration Validation

```python
from src.config import ConfigValidator

# Initialize validator
validator = ConfigValidator()

# Validate configuration
result = validator.validate_config_detailed('settings', config_data)

if not result.valid:
    for error in result.errors:
        print(f"Error: {error.field} - {error.message}")
```

### Monitoring and Alerting

```python
from src.config import ConfigMonitor

# Initialize monitor
monitor = ConfigMonitor(config_manager)

# Get alerts
alerts = monitor.get_alerts(severity=AlertSeverity.CRITICAL, resolved=False)

# Create compliance rule
rule = ComplianceRule(
    id='custom_rule',
    name='Custom Rule',
    description='Custom compliance rule',
    category='security',
    severity=AlertSeverity.ERROR,
    config_path='system.ssl_enabled',
    validation_func='validate_ssl_enabled',
    parameters={'required_for_prod': True}
)

monitor.add_compliance_rule(rule)
```

### Automation

```python
from src.config import ConfigAutomation

# Initialize automation
automation = ConfigAutomation(config_manager)

# Create deployment
config_changes = {
    'settings': {
        'system': {'version': '2.0.0'},
        'database': {'host': 'new-db-host.com'}
    }
}

job_id = automation.create_deployment(
    'Production Deployment',
    'Deploy new configuration to production',
    config_changes,
    'production'
)

# Create backup
backup_id = automation.create_backup(
    'pre_deployment_backup',
    'Backup before deployment',
    retention_days=30
)

# Check drift
drift_detections = automation.get_drift_detections(resolved=False)
```

## Configuration Structure

### Environment-Specific Configurations

```yaml
# config/environments/production.yaml
system:
  environment: production
  ssl_enabled: true
  backup_enabled: true

database:
  host: ${DB_HOST:production.db.com}
  password: ${SECRET:db_password}

security:
  jwt_secret_key: ${SECRET:jwt_secret}
```

### Schema Validation

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "system": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "mode": {"enum": ["live", "paper", "backtest"]}
      },
      "required": ["name", "version", "mode"]
    }
  },
  "required": ["system"]
}
```

## Security Features

### Secrets Encryption

- All secrets are encrypted at rest using Fernet (AES 128)
- Master key derivation using PBKDF2
- Secure key management practices
- Audit logging for all secret operations

### Access Control

- Role-based access control for secrets
- User-specific access policies
- Operation-level permissions (read, write, rotate)
- Audit trail for compliance

### Compliance

- Built-in compliance rules for common security requirements
- Custom compliance rule framework
- Automated compliance checking
- Violation reporting and alerting

## Best Practices

### Configuration Management

1. **Use Environment Variables**: Store sensitive data in environment variables
2. **Version Control**: Track configuration changes with versioning
3. **Validation**: Always validate configurations before deployment
4. **Backup**: Regular backups of configuration state
5. **Monitoring**: Monitor configuration changes and compliance

### Secrets Management

1. **Rotation**: Regularly rotate secrets
2. **Least Privilege**: Grant minimal necessary access
3. **Audit**: Monitor secret access and operations
4. **Encryption**: Use strong encryption for secret storage
5. **Compliance**: Follow security compliance requirements

### Feature Flags

1. **Gradual Rollouts**: Use percentage-based rollouts for new features
2. **Monitoring**: Monitor feature flag performance and usage
3. **Cleanup**: Remove unused feature flags
4. **Testing**: Test feature flags in non-production environments
5. **Documentation**: Document feature flag purpose and usage

## Monitoring and Alerting

### Alert Types

- **Configuration Changes**: Alerts on configuration modifications
- **Compliance Violations**: Alerts on compliance rule violations
- **Drift Detection**: Alerts on configuration drift
- **Performance Issues**: Alerts on system performance degradation
- **Security Events**: Alerts on security-related events

### Metrics

- Configuration change frequency
- Compliance score
- Secret access patterns
- Feature flag usage
- System performance metrics

## Integration

### CI/CD Integration

```python
# Example CI/CD pipeline integration
def deploy_configuration():
    # Validate configuration
    if not validator.validate_config('settings', new_config):
        raise ValueError("Configuration validation failed")
    
    # Create backup
    backup_id = automation.create_backup('pre_deployment_backup')
    
    # Deploy configuration
    job_id = automation.create_deployment(
        'Automated Deployment',
        'CI/CD pipeline deployment',
        config_changes,
        'production'
    )
    
    # Monitor deployment
    job = automation.get_deployment_status(job_id)
    if job.status == DeploymentStatus.FAILED:
        # Restore backup
        automation.restore_backup(backup_id)
        raise Exception("Deployment failed, restored backup")
```

### Monitoring Integration

```python
# Example monitoring integration
def setup_monitoring():
    # Add alert handler
    def alert_handler(alert):
        if alert.severity == AlertSeverity.CRITICAL:
            send_pager_alert(alert)
        else:
            send_email_alert(alert)
    
    monitor.add_alert_handler(alert_handler)
    
    # Add compliance rules
    for rule in custom_compliance_rules:
        monitor.add_compliance_rule(rule)
```

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**
   - Check schema definitions
   - Verify required fields
   - Validate data types

2. **Secret Access Denied**
   - Check access policies
   - Verify user permissions
   - Review audit logs

3. **Feature Flag Not Working**
   - Check rollout configuration
   - Verify user assignment
   - Review evaluation logs

4. **Deployment Failures**
   - Check configuration validation
   - Review deployment logs
   - Verify system resources

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode
config_manager = ConfigManager(Environment.DEVELOPMENT)
```

## API Reference

### ConfigManager

- `get_config(name, section=None)`: Get configuration
- `set_config(name, data, user, reason)`: Set configuration
- `update_config(name, updates, user, reason)`: Update configuration
- `rollback_config(name, version, user, reason)`: Rollback configuration
- `get_versions(name)`: Get configuration versions
- `backup_configs(name)`: Create backup
- `restore_configs(name)`: Restore backup

### SecretsManager

- `create_secret(name, value, type, description)`: Create secret
- `get_secret(name, user)`: Get secret
- `update_secret(name, value, user)`: Update secret
- `rotate_secret(name, new_value, user)`: Rotate secret
- `delete_secret(name, user)`: Delete secret
- `list_secrets(user)`: List secrets
- `set_access_policy(name, users)`: Set access policy

### FeatureFlagManager

- `create_feature_flag(name, description, owner, state, percentage)`: Create feature flag
- `is_feature_enabled(name, user_id, context)`: Check if feature is enabled
- `evaluate_feature(name, user_id, context)`: Evaluate feature
- `update_feature_flag(name, **kwargs)`: Update feature flag
- `create_ab_test(name, feature_flag, variants, allocation, metrics)`: Create A/B test
- `get_feature_analytics(name, hours_back)`: Get feature analytics

### ConfigValidator

- `validate_config(name, data)`: Validate configuration
- `validate_config_detailed(name, data)`: Detailed validation
- `add_validation_rule(name, func)`: Add validation rule
- `generate_schema(name, sample_data)`: Generate schema
- `save_schema(name, schema)`: Save schema

### ConfigMonitor

- `get_alerts(severity, resolved, config_name, hours_back)`: Get alerts
- `resolve_alert(alert_id, resolved_by)`: Resolve alert
- `get_compliance_status(config_name)`: Get compliance status
- `add_compliance_rule(rule)`: Add compliance rule
- `create_monitoring_report(hours_back)`: Create monitoring report

### ConfigAutomation

- `create_deployment(name, description, changes, environment)`: Create deployment
- `get_deployment_status(job_id)`: Get deployment status
- `create_backup(name, description, configs, retention_days)`: Create backup
- `restore_backup(backup_id, user)`: Restore backup
- `get_drift_detections(resolved)`: Get drift detections
- `add_automation_rule(rule)`: Add automation rule