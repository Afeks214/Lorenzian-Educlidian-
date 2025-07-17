"""
Configuration Validator
Validates configuration data against schemas and business rules.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logging.warning("jsonschema not available - using basic validation")


class ValidationLevel(Enum):
    """Validation levels"""
    STRICT = "strict"
    NORMAL = "normal"
    RELAXED = "relaxed"


@dataclass
class ValidationError:
    """Validation error details"""
    field: str
    message: str
    value: Any
    severity: str = "error"


@dataclass
class ValidationResult:
    """Validation result"""
    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]


class ConfigValidator:
    """
    Configuration validator with schema validation and business rules.
    """

    def __init__(self, schemas_path: Optional[Path] = None,
                 validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.schemas_path = schemas_path or Path(__file__).parent.parent.parent / "config" / "schemas"
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Load schemas
        self.schemas = {}
        self._load_schemas()
        
        # Built-in validation rules
        self.validation_rules = {
            'settings': self._validate_settings,
            'risk_management_config': self._validate_risk_config,
            'model_configs': self._validate_model_config,
            'data_pipeline': self._validate_data_pipeline
        }

    def _load_schemas(self):
        """Load JSON schemas from the schemas directory"""
        if not self.schemas_path.exists():
            self.schemas_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created schemas directory: {self.schemas_path}")
            return
        
        for schema_file in self.schemas_path.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema_name = schema_file.stem
                    self.schemas[schema_name] = json.load(f)
                    self.logger.debug(f"Loaded schema: {schema_name}")
            except Exception as e:
                self.logger.error(f"Failed to load schema {schema_file}: {e}")

    def validate_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """
        Validate configuration data
        
        Args:
            config_name: Name of the configuration
            config_data: Configuration data to validate
            
        Returns:
            True if valid, False otherwise
        """
        result = self.validate_config_detailed(config_name, config_data)
        
        if not result.valid:
            self.logger.error(f"Configuration validation failed for {config_name}:")
            for error in result.errors:
                self.logger.error(f"  {error.field}: {error.message}")
        
        if result.warnings:
            self.logger.warning(f"Configuration warnings for {config_name}:")
            for warning in result.warnings:
                self.logger.warning(f"  {warning.field}: {warning.message}")
        
        return result.valid

    def validate_config_detailed(self, config_name: str, config_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate configuration with detailed results
        
        Args:
            config_name: Name of the configuration
            config_data: Configuration data to validate
            
        Returns:
            ValidationResult with detailed error information
        """
        errors = []
        warnings = []
        
        # Schema validation
        if JSONSCHEMA_AVAILABLE and config_name in self.schemas:
            schema_errors = self._validate_schema(config_name, config_data)
            errors.extend(schema_errors)
        
        # Business rule validation
        if config_name in self.validation_rules:
            rule_errors, rule_warnings = self._validate_business_rules(config_name, config_data)
            errors.extend(rule_errors)
            warnings.extend(rule_warnings)
        
        # General validation
        general_errors, general_warnings = self._validate_general_rules(config_data)
        errors.extend(general_errors)
        warnings.extend(general_warnings)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _validate_schema(self, config_name: str, config_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate against JSON schema"""
        errors = []
        
        try:
            schema = self.schemas[config_name]
            jsonschema.validate(config_data, schema)
        except jsonschema.ValidationError as e:
            errors.append(ValidationError(
                field=".".join(str(x) for x in e.path),
                message=e.message,
                value=e.instance
            ))
        except Exception as e:
            errors.append(ValidationError(
                field="schema",
                message=f"Schema validation error: {e}",
                value=config_data
            ))
        
        return errors

    def _validate_business_rules(self, config_name: str, config_data: Dict[str, Any]) -> tuple:
        """Validate business-specific rules"""
        validator_func = self.validation_rules[config_name]
        return validator_func(config_data)

    def _validate_general_rules(self, config_data: Dict[str, Any]) -> tuple:
        """Validate general configuration rules"""
        errors = []
        warnings = []
        
        # Check for empty values
        empty_fields = self._find_empty_fields(config_data)
        for field in empty_fields:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(ValidationError(
                    field=field,
                    message="Empty value not allowed in strict mode",
                    value=None
                ))
            else:
                warnings.append(ValidationError(
                    field=field,
                    message="Empty value detected",
                    value=None,
                    severity="warning"
                ))
        
        # Check for suspicious values
        suspicious_fields = self._find_suspicious_values(config_data)
        for field, value in suspicious_fields.items():
            warnings.append(ValidationError(
                field=field,
                message=f"Suspicious value detected: {value}",
                value=value,
                severity="warning"
            ))
        
        return errors, warnings

    def _find_empty_fields(self, data: Dict[str, Any], prefix: str = "") -> List[str]:
        """Find fields with empty values"""
        empty_fields = []
        
        for key, value in data.items():
            field_path = f"{prefix}.{key}" if prefix else key
            
            if value is None or value == "":
                empty_fields.append(field_path)
            elif isinstance(value, dict):
                empty_fields.extend(self._find_empty_fields(value, field_path))
            elif isinstance(value, list) and len(value) == 0:
                empty_fields.append(field_path)
        
        return empty_fields

    def _find_suspicious_values(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Find potentially suspicious values"""
        suspicious = {}
        
        for key, value in data.items():
            field_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, str):
                # Check for potential injection attempts
                if any(pattern in value.lower() for pattern in ['<script', 'javascript:', 'eval(']):
                    suspicious[field_path] = value
                # Check for placeholder values
                elif value in ['TODO', 'FIXME', 'CHANGE_ME', 'password', '123456']:
                    suspicious[field_path] = value
            elif isinstance(value, dict):
                suspicious.update(self._find_suspicious_values(value, field_path))
        
        return suspicious

    def _validate_settings(self, config_data: Dict[str, Any]) -> tuple:
        """Validate settings configuration"""
        errors = []
        warnings = []
        
        # Check system configuration
        if 'system' not in config_data:
            errors.append(ValidationError(
                field="system",
                message="System configuration is required",
                value=None
            ))
        else:
            system_config = config_data['system']
            
            # Check required system fields
            required_fields = ['name', 'version', 'mode']
            for field in required_fields:
                if field not in system_config:
                    errors.append(ValidationError(
                        field=f"system.{field}",
                        message=f"Required field '{field}' is missing",
                        value=None
                    ))
            
            # Validate mode
            if 'mode' in system_config:
                valid_modes = ['live', 'paper', 'backtest']
                if system_config['mode'] not in valid_modes:
                    errors.append(ValidationError(
                        field="system.mode",
                        message=f"Invalid mode. Must be one of: {valid_modes}",
                        value=system_config['mode']
                    ))
        
        # Check data handler configuration
        if 'data_handler' in config_data:
            data_config = config_data['data_handler']
            
            if 'type' not in data_config:
                errors.append(ValidationError(
                    field="data_handler.type",
                    message="Data handler type is required",
                    value=None
                ))
            else:
                valid_types = ['rithmic', 'ib', 'backtest']
                if data_config['type'] not in valid_types:
                    errors.append(ValidationError(
                        field="data_handler.type",
                        message=f"Invalid data handler type. Must be one of: {valid_types}",
                        value=data_config['type']
                    ))
            
            # For backtest mode, check file existence
            if data_config.get('type') == 'backtest' and 'backtest_file' in data_config:
                backtest_file = Path(data_config['backtest_file'])
                if not backtest_file.exists():
                    warnings.append(ValidationError(
                        field="data_handler.backtest_file",
                        message="Backtest file does not exist",
                        value=str(backtest_file),
                        severity="warning"
                    ))
        
        return errors, warnings

    def _validate_risk_config(self, config_data: Dict[str, Any]) -> tuple:
        """Validate risk management configuration"""
        errors = []
        warnings = []
        
        # Check required risk parameters
        required_params = ['max_position_size', 'max_daily_loss', 'stop_loss_percent']
        for param in required_params:
            if param not in config_data:
                errors.append(ValidationError(
                    field=param,
                    message=f"Required risk parameter '{param}' is missing",
                    value=None
                ))
        
        # Validate numeric ranges
        if 'max_position_size' in config_data:
            max_pos = config_data['max_position_size']
            if not isinstance(max_pos, (int, float)) or max_pos <= 0:
                errors.append(ValidationError(
                    field="max_position_size",
                    message="Max position size must be a positive number",
                    value=max_pos
                ))
        
        if 'stop_loss_percent' in config_data:
            stop_loss = config_data['stop_loss_percent']
            if not isinstance(stop_loss, (int, float)) or stop_loss <= 0 or stop_loss > 100:
                errors.append(ValidationError(
                    field="stop_loss_percent",
                    message="Stop loss percent must be between 0 and 100",
                    value=stop_loss
                ))
        
        # Check position sizing method
        if 'position_sizing_method' in config_data:
            valid_methods = ['fixed', 'kelly', 'risk_parity']
            if config_data['position_sizing_method'] not in valid_methods:
                errors.append(ValidationError(
                    field="position_sizing_method",
                    message=f"Invalid position sizing method. Must be one of: {valid_methods}",
                    value=config_data['position_sizing_method']
                ))
        
        return errors, warnings

    def _validate_model_config(self, config_data: Dict[str, Any]) -> tuple:
        """Validate model configuration"""
        errors = []
        warnings = []
        
        # Check model paths
        if 'models' in config_data:
            models = config_data['models']
            
            for model_name, model_path in models.items():
                if model_path:
                    path_obj = Path(model_path)
                    if not path_obj.parent.exists():
                        warnings.append(ValidationError(
                            field=f"models.{model_name}",
                            message=f"Model directory does not exist: {path_obj.parent}",
                            value=model_path,
                            severity="warning"
                        ))
        
        # Check device configuration
        if 'device' in config_data:
            device = config_data['device']
            if device not in ['cpu', 'cuda', 'auto']:
                warnings.append(ValidationError(
                    field="device",
                    message=f"Unusual device configuration: {device}",
                    value=device,
                    severity="warning"
                ))
        
        return errors, warnings

    def _validate_data_pipeline(self, config_data: Dict[str, Any]) -> tuple:
        """Validate data pipeline configuration"""
        errors = []
        warnings = []
        
        # Check buffer sizes
        if 'buffer_sizes' in config_data:
            buffers = config_data['buffer_sizes']
            
            for buffer_name, size in buffers.items():
                if not isinstance(size, int) or size <= 0:
                    errors.append(ValidationError(
                        field=f"buffer_sizes.{buffer_name}",
                        message="Buffer size must be a positive integer",
                        value=size
                    ))
                elif size > 1000000:  # 1M limit
                    warnings.append(ValidationError(
                        field=f"buffer_sizes.{buffer_name}",
                        message="Very large buffer size detected",
                        value=size,
                        severity="warning"
                    ))
        
        # Check memory limits
        if 'memory_limits' in config_data:
            memory = config_data['memory_limits']
            
            if 'max_memory_mb' in memory:
                max_mem = memory['max_memory_mb']
                if not isinstance(max_mem, (int, float)) or max_mem <= 0:
                    errors.append(ValidationError(
                        field="memory_limits.max_memory_mb",
                        message="Max memory must be a positive number",
                        value=max_mem
                    ))
                elif max_mem > 32000:  # 32GB limit
                    warnings.append(ValidationError(
                        field="memory_limits.max_memory_mb",
                        message="Very high memory limit detected",
                        value=max_mem,
                        severity="warning"
                    ))
        
        return errors, warnings

    def add_validation_rule(self, config_name: str, 
                          validator_func: callable):
        """Add a custom validation rule"""
        self.validation_rules[config_name] = validator_func

    def remove_validation_rule(self, config_name: str):
        """Remove a validation rule"""
        if config_name in self.validation_rules:
            del self.validation_rules[config_name]

    def create_schema_template(self, config_name: str) -> Dict[str, Any]:
        """Create a JSON schema template for a configuration"""
        # Basic template
        template = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": f"{config_name} Configuration Schema",
            "properties": {},
            "required": []
        }
        
        # Add specific templates based on config type
        if config_name == "settings":
            template["properties"] = {
                "system": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "version": {"type": "string"},
                        "mode": {"enum": ["live", "paper", "backtest"]}
                    },
                    "required": ["name", "version", "mode"]
                },
                "data_handler": {
                    "type": "object",
                    "properties": {
                        "type": {"enum": ["rithmic", "ib", "backtest"]},
                        "backtest_file": {"type": "string"}
                    },
                    "required": ["type"]
                }
            }
            template["required"] = ["system", "data_handler"]
        
        return template

    def generate_schema(self, config_name: str, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a JSON schema from sample configuration data"""
        def infer_type(value):
            if isinstance(value, bool):
                return "boolean"
            elif isinstance(value, int):
                return "integer"
            elif isinstance(value, float):
                return "number"
            elif isinstance(value, str):
                return "string"
            elif isinstance(value, list):
                return "array"
            elif isinstance(value, dict):
                return "object"
            else:
                return "string"
        
        def build_properties(data):
            properties = {}
            required = []
            
            for key, value in data.items():
                if value is not None:
                    prop_type = infer_type(value)
                    properties[key] = {"type": prop_type}
                    
                    if prop_type == "object" and isinstance(value, dict):
                        properties[key]["properties"] = build_properties(value)[0]
                    elif prop_type == "array" and value:
                        if isinstance(value[0], dict):
                            properties[key]["items"] = {
                                "type": "object",
                                "properties": build_properties(value[0])[0]
                            }
                        else:
                            properties[key]["items"] = {"type": infer_type(value[0])}
                    
                    required.append(key)
            
            return properties, required
        
        properties, required = build_properties(sample_data)
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "title": f"{config_name} Configuration Schema (Auto-generated)",
            "properties": properties,
            "required": required
        }
        
        return schema

    def save_schema(self, config_name: str, schema: Dict[str, Any]):
        """Save a JSON schema to disk"""
        self.schemas_path.mkdir(parents=True, exist_ok=True)
        
        schema_file = self.schemas_path / f"{config_name}.json"
        
        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        # Update in-memory schemas
        self.schemas[config_name] = schema
        
        self.logger.info(f"Schema saved: {config_name}")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation system summary"""
        return {
            'validation_level': self.validation_level.value,
            'schemas_loaded': len(self.schemas),
            'validation_rules': len(self.validation_rules),
            'schema_names': list(self.schemas.keys()),
            'rule_names': list(self.validation_rules.keys()),
            'jsonschema_available': JSONSCHEMA_AVAILABLE
        }