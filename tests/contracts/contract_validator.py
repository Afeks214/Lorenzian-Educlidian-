"""
Contract Validator
==================

Validates service interactions against contract definitions.
Provides comprehensive validation for requests, responses, and events.
"""

import json
import jsonschema
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re
import threading
from .contract_registry import ContractRegistry, ContractDefinition, ContractEndpoint, ContractEvent


class ValidationResult(Enum):
    """Validation result enumeration"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


@dataclass
class ValidationError:
    """Validation error details"""
    field: str
    message: str
    expected: Any = None
    actual: Any = None
    error_type: str = "validation"


@dataclass
class ValidationReport:
    """Validation report structure"""
    validation_id: str
    contract_id: str
    contract_version: str
    endpoint_path: str
    result: ValidationResult
    timestamp: datetime
    duration_ms: float
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestValidation:
    """Request validation configuration"""
    validate_headers: bool = True
    validate_query_params: bool = True
    validate_body: bool = True
    validate_content_type: bool = True
    strict_validation: bool = False


@dataclass
class ResponseValidation:
    """Response validation configuration"""
    validate_headers: bool = True
    validate_status_code: bool = True
    validate_body: bool = True
    validate_content_type: bool = True
    strict_validation: bool = False


class ContractValidator:
    """
    Contract validator for service interactions
    
    Features:
    - Request/response validation
    - Event validation
    - Schema validation using JSON Schema
    - Configurable validation rules
    - Detailed error reporting
    - Performance metrics
    """
    
    def __init__(self, registry: ContractRegistry):
        self.registry = registry
        self.validation_cache: Dict[str, Any] = {}
        self.validation_reports: List[ValidationReport] = []
        self.lock = threading.RLock()
        
        # Default validation configurations
        self.default_request_validation = RequestValidation()
        self.default_response_validation = ResponseValidation()
        
        # JSON Schema validator
        self.schema_validator = jsonschema.Draft7Validator
    
    def validate_request(self, contract_id: str, version: str, endpoint_path: str,
                        method: str, request_data: Dict[str, Any],
                        validation_config: Optional[RequestValidation] = None) -> ValidationReport:
        """Validate request against contract"""
        
        start_time = time.time()
        validation_id = f"req_{contract_id}_{version}_{int(time.time() * 1000)}"
        
        config = validation_config or self.default_request_validation
        
        try:
            # Get contract
            contract = self.registry.get_contract(contract_id, version)
            if not contract:
                return self._create_error_report(
                    validation_id, contract_id, version, endpoint_path,
                    f"Contract not found: {contract_id}:{version}"
                )
            
            # Find endpoint
            endpoint = self._find_endpoint(contract, endpoint_path, method)
            if not endpoint:
                return self._create_error_report(
                    validation_id, contract_id, version, endpoint_path,
                    f"Endpoint not found: {method} {endpoint_path}"
                )
            
            # Perform validation
            errors = []
            warnings = []
            
            # Validate headers
            if config.validate_headers:
                header_errors = self._validate_headers(
                    request_data.get('headers', {}),
                    endpoint.headers,
                    config.strict_validation
                )
                errors.extend(header_errors)
            
            # Validate query parameters
            if config.validate_query_params:
                query_errors = self._validate_query_params(
                    request_data.get('query_params', {}),
                    endpoint.query_params,
                    config.strict_validation
                )
                errors.extend(query_errors)
            
            # Validate request body
            if config.validate_body and 'body' in request_data:
                body_errors = self._validate_schema(
                    request_data['body'],
                    endpoint.request_schema,
                    "request_body"
                )
                errors.extend(body_errors)
            
            # Validate content type
            if config.validate_content_type:
                content_type_errors = self._validate_content_type(
                    request_data.get('headers', {}),
                    endpoint.metadata.get('content_type', 'application/json')
                )
                errors.extend(content_type_errors)
            
            # Determine result
            result = ValidationResult.PASS if not errors else ValidationResult.FAIL
            
            # Create report
            duration_ms = (time.time() - start_time) * 1000
            
            report = ValidationReport(
                validation_id=validation_id,
                contract_id=contract_id,
                contract_version=version,
                endpoint_path=endpoint_path,
                result=result,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                errors=errors,
                warnings=warnings,
                metadata={
                    'method': method,
                    'validation_type': 'request',
                    'config': config
                }
            )
            
            self._store_validation_report(report)
            return report
            
        except Exception as e:
            return self._create_error_report(
                validation_id, contract_id, version, endpoint_path,
                f"Validation error: {str(e)}"
            )
    
    def validate_response(self, contract_id: str, version: str, endpoint_path: str,
                         method: str, response_data: Dict[str, Any],
                         validation_config: Optional[ResponseValidation] = None) -> ValidationReport:
        """Validate response against contract"""
        
        start_time = time.time()
        validation_id = f"resp_{contract_id}_{version}_{int(time.time() * 1000)}"
        
        config = validation_config or self.default_response_validation
        
        try:
            # Get contract
            contract = self.registry.get_contract(contract_id, version)
            if not contract:
                return self._create_error_report(
                    validation_id, contract_id, version, endpoint_path,
                    f"Contract not found: {contract_id}:{version}"
                )
            
            # Find endpoint
            endpoint = self._find_endpoint(contract, endpoint_path, method)
            if not endpoint:
                return self._create_error_report(
                    validation_id, contract_id, version, endpoint_path,
                    f"Endpoint not found: {method} {endpoint_path}"
                )
            
            # Perform validation
            errors = []
            warnings = []
            
            # Validate status code
            if config.validate_status_code:
                status_code_errors = self._validate_status_code(
                    response_data.get('status_code', 200),
                    endpoint.status_codes
                )
                errors.extend(status_code_errors)
            
            # Validate headers
            if config.validate_headers:
                header_errors = self._validate_response_headers(
                    response_data.get('headers', {}),
                    endpoint.metadata.get('response_headers', {}),
                    config.strict_validation
                )
                errors.extend(header_errors)
            
            # Validate response body
            if config.validate_body and 'body' in response_data:
                body_errors = self._validate_schema(
                    response_data['body'],
                    endpoint.response_schema,
                    "response_body"
                )
                errors.extend(body_errors)
            
            # Validate content type
            if config.validate_content_type:
                content_type_errors = self._validate_content_type(
                    response_data.get('headers', {}),
                    endpoint.metadata.get('response_content_type', 'application/json')
                )
                errors.extend(content_type_errors)
            
            # Determine result
            result = ValidationResult.PASS if not errors else ValidationResult.FAIL
            
            # Create report
            duration_ms = (time.time() - start_time) * 1000
            
            report = ValidationReport(
                validation_id=validation_id,
                contract_id=contract_id,
                contract_version=version,
                endpoint_path=endpoint_path,
                result=result,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                errors=errors,
                warnings=warnings,
                metadata={
                    'method': method,
                    'validation_type': 'response',
                    'config': config
                }
            )
            
            self._store_validation_report(report)
            return report
            
        except Exception as e:
            return self._create_error_report(
                validation_id, contract_id, version, endpoint_path,
                f"Validation error: {str(e)}"
            )
    
    def validate_event(self, contract_id: str, version: str, event_type: str,
                      event_data: Dict[str, Any]) -> ValidationReport:
        """Validate event against contract"""
        
        start_time = time.time()
        validation_id = f"event_{contract_id}_{version}_{int(time.time() * 1000)}"
        
        try:
            # Get contract
            contract = self.registry.get_contract(contract_id, version)
            if not contract:
                return self._create_error_report(
                    validation_id, contract_id, version, event_type,
                    f"Contract not found: {contract_id}:{version}"
                )
            
            # Find event
            event = self._find_event(contract, event_type)
            if not event:
                return self._create_error_report(
                    validation_id, contract_id, version, event_type,
                    f"Event not found: {event_type}"
                )
            
            # Validate event schema
            errors = self._validate_schema(
                event_data,
                event.event_schema,
                "event_data"
            )
            
            # Determine result
            result = ValidationResult.PASS if not errors else ValidationResult.FAIL
            
            # Create report
            duration_ms = (time.time() - start_time) * 1000
            
            report = ValidationReport(
                validation_id=validation_id,
                contract_id=contract_id,
                contract_version=version,
                endpoint_path=event_type,
                result=result,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                errors=errors,
                warnings=[],
                metadata={
                    'event_type': event_type,
                    'validation_type': 'event',
                    'routing_key': event.routing_key,
                    'exchange': event.exchange
                }
            )
            
            self._store_validation_report(report)
            return report
            
        except Exception as e:
            return self._create_error_report(
                validation_id, contract_id, version, event_type,
                f"Event validation error: {str(e)}"
            )
    
    def _find_endpoint(self, contract: ContractDefinition, path: str, method: str) -> Optional[ContractEndpoint]:
        """Find endpoint in contract"""
        for endpoint in contract.endpoints:
            if endpoint.path == path and endpoint.method.upper() == method.upper():
                return endpoint
        return None
    
    def _find_event(self, contract: ContractDefinition, event_type: str) -> Optional[ContractEvent]:
        """Find event in contract"""
        for event in contract.events:
            if event.event_type == event_type:
                return event
        return None
    
    def _validate_headers(self, actual_headers: Dict[str, str], 
                         expected_headers: Dict[str, str], 
                         strict: bool) -> List[ValidationError]:
        """Validate request headers"""
        errors = []
        
        # Check required headers
        for header_name, expected_value in expected_headers.items():
            if header_name not in actual_headers:
                errors.append(ValidationError(
                    field=f"headers.{header_name}",
                    message=f"Required header missing: {header_name}",
                    expected=expected_value,
                    actual=None,
                    error_type="missing_header"
                ))
            elif strict and actual_headers[header_name] != expected_value:
                errors.append(ValidationError(
                    field=f"headers.{header_name}",
                    message=f"Header value mismatch: {header_name}",
                    expected=expected_value,
                    actual=actual_headers[header_name],
                    error_type="header_mismatch"
                ))
        
        return errors
    
    def _validate_response_headers(self, actual_headers: Dict[str, str], 
                                  expected_headers: Dict[str, str], 
                                  strict: bool) -> List[ValidationError]:
        """Validate response headers"""
        return self._validate_headers(actual_headers, expected_headers, strict)
    
    def _validate_query_params(self, actual_params: Dict[str, Any], 
                              expected_params: Dict[str, Any], 
                              strict: bool) -> List[ValidationError]:
        """Validate query parameters"""
        errors = []
        
        for param_name, param_config in expected_params.items():
            if isinstance(param_config, dict) and param_config.get('required', False):
                if param_name not in actual_params:
                    errors.append(ValidationError(
                        field=f"query_params.{param_name}",
                        message=f"Required query parameter missing: {param_name}",
                        expected=param_config,
                        actual=None,
                        error_type="missing_param"
                    ))
                elif strict and 'type' in param_config:
                    # Validate parameter type
                    expected_type = param_config['type']
                    actual_value = actual_params[param_name]
                    
                    if not self._validate_type(actual_value, expected_type):
                        errors.append(ValidationError(
                            field=f"query_params.{param_name}",
                            message=f"Query parameter type mismatch: {param_name}",
                            expected=expected_type,
                            actual=type(actual_value).__name__,
                            error_type="type_mismatch"
                        ))
        
        return errors
    
    def _validate_schema(self, data: Any, schema: Dict[str, Any], field_name: str) -> List[ValidationError]:
        """Validate data against JSON schema"""
        errors = []
        
        try:
            # Create validator
            validator = self.schema_validator(schema)
            
            # Validate data
            validation_errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
            
            for error in validation_errors:
                field_path = f"{field_name}.{'.'.join(str(p) for p in error.path)}" if error.path else field_name
                
                errors.append(ValidationError(
                    field=field_path,
                    message=error.message,
                    expected=error.schema,
                    actual=error.instance,
                    error_type="schema_validation"
                ))
                
        except Exception as e:
            errors.append(ValidationError(
                field=field_name,
                message=f"Schema validation error: {str(e)}",
                expected=schema,
                actual=data,
                error_type="validation_error"
            ))
        
        return errors
    
    def _validate_content_type(self, headers: Dict[str, str], expected_content_type: str) -> List[ValidationError]:
        """Validate content type"""
        errors = []
        
        content_type = headers.get('Content-Type', headers.get('content-type', ''))
        
        if expected_content_type and content_type != expected_content_type:
            errors.append(ValidationError(
                field="headers.Content-Type",
                message="Content-Type mismatch",
                expected=expected_content_type,
                actual=content_type,
                error_type="content_type_mismatch"
            ))
        
        return errors
    
    def _validate_status_code(self, actual_status: int, expected_codes: List[int]) -> List[ValidationError]:
        """Validate status code"""
        errors = []
        
        if actual_status not in expected_codes:
            errors.append(ValidationError(
                field="status_code",
                message=f"Unexpected status code: {actual_status}",
                expected=expected_codes,
                actual=actual_status,
                error_type="status_code_mismatch"
            ))
        
        return errors
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_mapping.get(expected_type.lower())
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, pass validation
    
    def _create_error_report(self, validation_id: str, contract_id: str, 
                           version: str, endpoint_path: str, error_message: str) -> ValidationReport:
        """Create error validation report"""
        return ValidationReport(
            validation_id=validation_id,
            contract_id=contract_id,
            contract_version=version,
            endpoint_path=endpoint_path,
            result=ValidationResult.ERROR,
            timestamp=datetime.now(),
            duration_ms=0,
            errors=[ValidationError(
                field="validation",
                message=error_message,
                error_type="validation_error"
            )]
        )
    
    def _store_validation_report(self, report: ValidationReport):
        """Store validation report"""
        with self.lock:
            self.validation_reports.append(report)
            
            # Keep only last 1000 reports
            if len(self.validation_reports) > 1000:
                self.validation_reports = self.validation_reports[-1000:]
    
    def get_validation_reports(self, contract_id: str = None, 
                             limit: int = 100) -> List[ValidationReport]:
        """Get validation reports"""
        with self.lock:
            reports = self.validation_reports
            
            if contract_id:
                reports = [r for r in reports if r.contract_id == contract_id]
            
            return reports[-limit:]
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        with self.lock:
            if not self.validation_reports:
                return {
                    'total_validations': 0,
                    'success_rate': 0.0,
                    'average_duration_ms': 0.0,
                    'validations_by_result': {},
                    'validations_by_contract': {}
                }
            
            total_validations = len(self.validation_reports)
            passed_validations = len([r for r in self.validation_reports if r.result == ValidationResult.PASS])
            success_rate = passed_validations / total_validations if total_validations > 0 else 0.0
            
            average_duration = sum(r.duration_ms for r in self.validation_reports) / total_validations
            
            # Group by result
            validations_by_result = {}
            for report in self.validation_reports:
                result = report.result.value
                validations_by_result[result] = validations_by_result.get(result, 0) + 1
            
            # Group by contract
            validations_by_contract = {}
            for report in self.validation_reports:
                contract_key = f"{report.contract_id}:{report.contract_version}"
                validations_by_contract[contract_key] = validations_by_contract.get(contract_key, 0) + 1
            
            return {
                'total_validations': total_validations,
                'success_rate': success_rate,
                'average_duration_ms': average_duration,
                'validations_by_result': validations_by_result,
                'validations_by_contract': validations_by_contract
            }
    
    def clear_validation_reports(self):
        """Clear all validation reports"""
        with self.lock:
            self.validation_reports.clear()
    
    def validate_contract_compatibility(self, old_contract: ContractDefinition,
                                      new_contract: ContractDefinition) -> Dict[str, Any]:
        """Validate backwards compatibility between contract versions"""
        
        compatibility_issues = []
        breaking_changes = []
        
        # Check endpoints
        old_endpoints = {f"{ep.method}:{ep.path}": ep for ep in old_contract.endpoints}
        new_endpoints = {f"{ep.method}:{ep.path}": ep for ep in new_contract.endpoints}
        
        # Check for removed endpoints
        for endpoint_key in old_endpoints:
            if endpoint_key not in new_endpoints:
                breaking_changes.append(f"Endpoint removed: {endpoint_key}")
        
        # Check for modified endpoints
        for endpoint_key in old_endpoints:
            if endpoint_key in new_endpoints:
                old_ep = old_endpoints[endpoint_key]
                new_ep = new_endpoints[endpoint_key]
                
                # Check request schema compatibility
                req_issues = self._check_schema_compatibility(
                    old_ep.request_schema, new_ep.request_schema, f"{endpoint_key}.request"
                )
                compatibility_issues.extend(req_issues)
                
                # Check response schema compatibility
                resp_issues = self._check_schema_compatibility(
                    old_ep.response_schema, new_ep.response_schema, f"{endpoint_key}.response"
                )
                compatibility_issues.extend(resp_issues)
        
        # Check events
        old_events = {ev.event_type: ev for ev in old_contract.events}
        new_events = {ev.event_type: ev for ev in new_contract.events}
        
        # Check for removed events
        for event_type in old_events:
            if event_type not in new_events:
                breaking_changes.append(f"Event removed: {event_type}")
        
        # Check for modified events
        for event_type in old_events:
            if event_type in new_events:
                old_event = old_events[event_type]
                new_event = new_events[event_type]
                
                event_issues = self._check_schema_compatibility(
                    old_event.event_schema, new_event.event_schema, f"event.{event_type}"
                )
                compatibility_issues.extend(event_issues)
        
        is_compatible = len(breaking_changes) == 0
        
        return {
            'is_compatible': is_compatible,
            'breaking_changes': breaking_changes,
            'compatibility_issues': compatibility_issues,
            'old_version': old_contract.version,
            'new_version': new_contract.version
        }
    
    def _check_schema_compatibility(self, old_schema: Dict[str, Any], 
                                   new_schema: Dict[str, Any], context: str) -> List[str]:
        """Check schema compatibility between versions"""
        issues = []
        
        # Simple compatibility checks
        # In a real implementation, this would be more sophisticated
        
        # Check required fields
        old_required = set(old_schema.get('required', []))
        new_required = set(new_schema.get('required', []))
        
        # New required fields are breaking changes
        new_required_fields = new_required - old_required
        if new_required_fields:
            issues.append(f"{context}: New required fields added: {new_required_fields}")
        
        # Check properties
        old_properties = old_schema.get('properties', {})
        new_properties = new_schema.get('properties', {})
        
        # Removed properties are breaking changes
        removed_properties = set(old_properties.keys()) - set(new_properties.keys())
        if removed_properties:
            issues.append(f"{context}: Properties removed: {removed_properties}")
        
        return issues