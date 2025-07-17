"""
Pydantic models for API request/response validation.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from enum import Enum
from decimal import Decimal
from pydantic import BaseModel, Field, validator, constr, root_validator, ValidationError
import numpy as np
import pandas as pd

class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentHealthResponse(BaseModel):
    """Individual component health status."""
    name: str = Field(..., description="Component name")
    status: HealthStatus = Field(..., description="Health status")
    message: str = Field("", description="Status message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    last_check: datetime = Field(..., description="Last check timestamp")

class HealthCheckResponse(BaseModel):
    """Complete health check response."""
    status: HealthStatus = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Check timestamp")
    components: List[ComponentHealthResponse] = Field(..., description="Component statuses")
    recommendations: List[str] = Field(default_factory=list, description="Health recommendations")
    check_intervals: Dict[str, float] = Field(default_factory=dict, description="Check intervals in seconds")
    version: str = Field("1.0.0", description="API version")

class SynergyType(str, Enum):
    """Synergy pattern types."""
    TYPE_1 = "TYPE_1"
    TYPE_2 = "TYPE_2"
    TYPE_3 = "TYPE_3"
    TYPE_4 = "TYPE_4"

class MarketState(BaseModel):
    """Current market state information."""
    timestamp: datetime = Field(..., description="Market data timestamp")
    symbol: str = Field(..., description="Trading symbol")
    price: float = Field(..., ge=0, description="Current price")
    volume: float = Field(..., ge=0, description="Current volume")
    volatility: float = Field(..., ge=0, description="Current volatility")
    trend: str = Field(..., description="Market trend")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate trading symbol format."""
        if not v or not v.isupper():
            raise ValueError("Symbol must be uppercase")
        return v

class SynergyContext(BaseModel):
    """Synergy detection context."""
    synergy_type: SynergyType = Field(..., description="Type of synergy detected")
    strength: float = Field(..., ge=0, le=1, description="Synergy strength (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence (0-1)")
    pattern_data: Dict[str, Any] = Field(..., description="Pattern-specific data")
    correlation_id: str = Field(..., description="Event correlation ID")
    
class MatrixData(BaseModel):
    """Matrix data from assembler."""
    matrix_type: str = Field(..., description="Matrix type (30m/5m)")
    shape: List[int] = Field(..., description="Matrix dimensions")
    data: List[List[float]] = Field(..., description="Matrix values")
    features: List[str] = Field(..., description="Feature names")
    timestamp: datetime = Field(..., description="Matrix generation timestamp")
    
    @validator('shape')
    def validate_shape(cls, v):
        """Validate matrix shape."""
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError("Invalid matrix shape")
        return v

class StrategicDecisionRequest(BaseModel):
    """Request for strategic decision making."""
    market_state: MarketState = Field(..., description="Current market state")
    synergy_context: SynergyContext = Field(..., description="Detected synergy context")
    matrix_data: MatrixData = Field(..., description="Matrix assembler data")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    request_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")

class DecisionType(str, Enum):
    """Strategic decision types."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    EXIT = "EXIT"

class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class AgentDecision(BaseModel):
    """Individual agent decision."""
    agent_name: str = Field(..., description="Agent identifier")
    decision: DecisionType = Field(..., description="Agent's decision")
    confidence: float = Field(..., ge=0, le=1, description="Decision confidence")
    reasoning: Dict[str, Any] = Field(default_factory=dict, description="Decision reasoning")

class StrategicDecisionResponse(BaseModel):
    """Strategic decision response."""
    correlation_id: str = Field(..., description="Request correlation ID")
    decision: DecisionType = Field(..., description="Final strategic decision")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence")
    risk_level: RiskLevel = Field(..., description="Risk assessment")
    position_size: float = Field(..., ge=0, le=1, description="Recommended position size (0-1)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    agent_decisions: List[AgentDecision] = Field(..., description="Individual agent decisions")
    inference_latency_ms: float = Field(..., description="Decision latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    @validator('inference_latency_ms')
    def validate_latency(cls, v):
        """Ensure latency is within acceptable range."""
        if v > 5.0:  # 5ms threshold from PRD
            raise ValueError(f"Inference latency {v}ms exceeds 5ms threshold")
        return v

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

class MetricsResponse(BaseModel):
    """Prometheus metrics response (text format)."""
    content_type: str = Field("text/plain; version=0.0.4", description="Prometheus content type")
    
class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    event_type: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    correlation_id: str = Field(..., description="Event correlation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")

class AuthToken(BaseModel):
    """Authentication token."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(3600, description="Token expiration in seconds")

class APIKeyRequest(BaseModel):
    """API key authentication request."""
    api_key: constr(min_length=32, max_length=64) = Field(..., description="API key")
    
class RateLimitInfo(BaseModel):
    """Rate limit information."""
    limit: int = Field(..., description="Request limit")
    remaining: int = Field(..., description="Remaining requests")
    reset: datetime = Field(..., description="Reset timestamp")
    retry_after: Optional[int] = Field(None, description="Retry after seconds")


# =============================================================================
# COMPREHENSIVE DATA PIPELINE SCHEMAS
# =============================================================================

class DataQuality(str, Enum):
    """Data quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNUSABLE = "unusable"

class AssetClass(str, Enum):
    """Asset class enumeration"""
    FOREX = "forex"
    COMMODITIES = "commodities"
    EQUITIES = "equities"
    CRYPTO = "crypto"
    BONDS = "bonds"

class ValidationResult(str, Enum):
    """Validation result status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

# =============================================================================
# FINANCIAL DATA SCHEMAS
# =============================================================================

class PriceData(BaseModel):
    """Price data validation schema"""
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Volume")
    
    @validator('high')
    def validate_high(cls, v, values):
        """Validate high price against other prices"""
        if 'open' in values and v < values['open']:
            raise ValueError("High price must be >= open price")
        if 'low' in values and v < values['low']:
            raise ValueError("High price must be >= low price")
        if 'close' in values and v < values['close']:
            raise ValueError("High price must be >= close price")
        return v
    
    @validator('low')
    def validate_low(cls, v, values):
        """Validate low price against other prices"""
        if 'open' in values and v > values['open']:
            raise ValueError("Low price must be <= open price")
        if 'close' in values and v > values['close']:
            raise ValueError("Low price must be <= close price")
        return v
    
    @root_validator
    def validate_ohlc_consistency(cls, values):
        """Validate OHLC consistency"""
        o, h, l, c = values.get('open'), values.get('high'), values.get('low'), values.get('close')
        if all(x is not None for x in [o, h, l, c]):
            if not (l <= o <= h and l <= c <= h):
                raise ValueError("OHLC prices are inconsistent")
        return values

class NormalizedPriceData(BaseModel):
    """Normalized price data schema"""
    normalized_open: float = Field(..., description="Normalized opening price")
    normalized_high: float = Field(..., description="Normalized high price")
    normalized_low: float = Field(..., description="Normalized low price")
    normalized_close: float = Field(..., description="Normalized closing price")
    normalized_volume: float = Field(..., ge=0, description="Normalized volume")
    scale_factor: float = Field(..., gt=0, description="Normalization scale factor")
    
    @validator('normalized_high')
    def validate_normalized_high(cls, v, values):
        """Validate normalized high price"""
        if 'normalized_open' in values and v < values['normalized_open']:
            raise ValueError("Normalized high must be >= normalized open")
        if 'normalized_low' in values and v < values['normalized_low']:
            raise ValueError("Normalized high must be >= normalized low")
        if 'normalized_close' in values and v < values['normalized_close']:
            raise ValueError("Normalized high must be >= normalized close")
        return v

class MarketDataPoint(BaseModel):
    """Comprehensive market data point schema"""
    symbol: str = Field(..., min_length=3, max_length=20, description="Trading symbol")
    timestamp: datetime = Field(..., description="Data timestamp")
    price_data: PriceData = Field(..., description="Raw price data")
    normalized_data: NormalizedPriceData = Field(..., description="Normalized price data")
    asset_class: AssetClass = Field(..., description="Asset class")
    data_quality: DataQuality = Field(..., description="Data quality assessment")
    source: str = Field(..., min_length=1, description="Data source")
    latency_ms: float = Field(..., ge=0, le=10000, description="Data latency in milliseconds")
    
    @validator('symbol')
    def validate_symbol_format(cls, v):
        """Validate symbol format"""
        if not v.isupper():
            raise ValueError("Symbol must be uppercase")
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is not in future"""
        if v > datetime.utcnow():
            raise ValueError("Timestamp cannot be in the future")
        return v

# =============================================================================
# DATA LINEAGE SCHEMAS
# =============================================================================

class DataLineageNode(BaseModel):
    """Data lineage node schema"""
    node_id: str = Field(..., min_length=1, description="Unique node identifier")
    data_type: str = Field(..., description="Type of data")
    timestamp: datetime = Field(..., description="Node creation timestamp")
    parent_nodes: List[str] = Field(default_factory=list, description="Parent node IDs")
    child_nodes: List[str] = Field(default_factory=list, description="Child node IDs")
    transformation_applied: Optional[str] = Field(None, description="Transformation applied")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('node_id')
    def validate_node_id(cls, v):
        """Validate node ID format"""
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Node ID must be alphanumeric with hyphens/underscores")
        return v

class DataTransformation(BaseModel):
    """Data transformation schema"""
    transformation_id: str = Field(..., description="Unique transformation identifier")
    source_node_id: str = Field(..., description="Source data node ID")
    target_node_id: str = Field(..., description="Target data node ID")
    transformation_type: str = Field(..., description="Type of transformation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Transformation parameters")
    timestamp: datetime = Field(..., description="Transformation timestamp")
    duration_ms: float = Field(..., ge=0, description="Transformation duration in milliseconds")
    success: bool = Field(..., description="Transformation success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class DataLineageTrace(BaseModel):
    """Complete data lineage trace schema"""
    trace_id: str = Field(..., description="Unique trace identifier")
    root_node_id: str = Field(..., description="Root node of the trace")
    final_node_id: str = Field(..., description="Final node of the trace")
    nodes: List[DataLineageNode] = Field(..., description="All nodes in the trace")
    transformations: List[DataTransformation] = Field(..., description="All transformations in the trace")
    created_at: datetime = Field(..., description="Trace creation timestamp")
    
    @validator('nodes')
    def validate_nodes_not_empty(cls, v):
        """Validate nodes list is not empty"""
        if not v:
            raise ValueError("Nodes list cannot be empty")
        return v

# =============================================================================
# QUALITY MONITORING SCHEMAS
# =============================================================================

class QualityMetric(BaseModel):
    """Quality metric schema"""
    metric_name: str = Field(..., min_length=1, description="Metric name")
    value: float = Field(..., description="Metric value")
    threshold_warning: Optional[float] = Field(None, description="Warning threshold")
    threshold_critical: Optional[float] = Field(None, description="Critical threshold")
    timestamp: datetime = Field(..., description="Metric timestamp")
    component: str = Field(..., description="Component being measured")
    
    @validator('value')
    def validate_value_finite(cls, v):
        """Validate metric value is finite"""
        if not np.isfinite(v):
            raise ValueError("Metric value must be finite")
        return v

class QualityAssessment(BaseModel):
    """Quality assessment schema"""
    assessment_id: str = Field(..., description="Unique assessment identifier")
    component: str = Field(..., description="Component being assessed")
    metrics: List[QualityMetric] = Field(..., description="Quality metrics")
    overall_score: float = Field(..., ge=0, le=1, description="Overall quality score (0-1)")
    quality_level: DataQuality = Field(..., description="Overall quality level")
    timestamp: datetime = Field(..., description="Assessment timestamp")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")
    
    @validator('metrics')
    def validate_metrics_not_empty(cls, v):
        """Validate metrics list is not empty"""
        if not v:
            raise ValueError("Metrics list cannot be empty")
        return v

class AnomalyDetection(BaseModel):
    """Anomaly detection result schema"""
    detection_id: str = Field(..., description="Unique detection identifier")
    component: str = Field(..., description="Component where anomaly was detected")
    anomaly_type: str = Field(..., description="Type of anomaly")
    severity: str = Field(..., description="Anomaly severity level")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    timestamp: datetime = Field(..., description="Detection timestamp")
    affected_metrics: List[str] = Field(..., description="Affected metrics")
    baseline_value: float = Field(..., description="Baseline value")
    detected_value: float = Field(..., description="Detected anomalous value")
    statistical_score: float = Field(..., description="Statistical significance score")
    
    @validator('severity')
    def validate_severity(cls, v):
        """Validate severity level"""
        valid_severities = ['low', 'medium', 'high', 'critical']
        if v.lower() not in valid_severities:
            raise ValueError(f"Severity must be one of: {valid_severities}")
        return v.lower()

# =============================================================================
# PIPELINE VALIDATION SCHEMAS
# =============================================================================

class ValidationRule(BaseModel):
    """Data validation rule schema"""
    rule_id: str = Field(..., description="Unique rule identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    rule_type: str = Field(..., description="Type of validation rule")
    target_field: str = Field(..., description="Field to validate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")
    enabled: bool = Field(True, description="Whether rule is enabled")
    
    @validator('rule_type')
    def validate_rule_type(cls, v):
        """Validate rule type"""
        valid_types = ['range', 'format', 'consistency', 'statistical', 'business']
        if v.lower() not in valid_types:
            raise ValueError(f"Rule type must be one of: {valid_types}")
        return v.lower()

class ValidationReport(BaseModel):
    """Validation report schema"""
    report_id: str = Field(..., description="Unique report identifier")
    data_source: str = Field(..., description="Data source being validated")
    validation_timestamp: datetime = Field(..., description="Validation timestamp")
    total_records: int = Field(..., ge=0, description="Total number of records validated")
    passed_records: int = Field(..., ge=0, description="Number of records that passed validation")
    failed_records: int = Field(..., ge=0, description="Number of records that failed validation")
    warning_records: int = Field(..., ge=0, description="Number of records with warnings")
    validation_rules: List[ValidationRule] = Field(..., description="Applied validation rules")
    rule_results: Dict[str, ValidationResult] = Field(..., description="Results for each rule")
    error_details: List[str] = Field(default_factory=list, description="Detailed error messages")
    
    @validator('passed_records')
    def validate_passed_records(cls, v, values):
        """Validate passed records don't exceed total"""
        if 'total_records' in values and v > values['total_records']:
            raise ValueError("Passed records cannot exceed total records")
        return v
    
    @root_validator
    def validate_record_counts(cls, values):
        """Validate record counts are consistent"""
        total = values.get('total_records', 0)
        passed = values.get('passed_records', 0)
        failed = values.get('failed_records', 0)
        warnings = values.get('warning_records', 0)
        
        if passed + failed + warnings != total:
            raise ValueError("Record counts must sum to total records")
        return values

# =============================================================================
# REAL-TIME MONITORING SCHEMAS
# =============================================================================

class SystemHealthMetrics(BaseModel):
    """System health metrics schema"""
    component: str = Field(..., description="Component name")
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    disk_usage: float = Field(..., ge=0, le=100, description="Disk usage percentage")
    network_io: float = Field(..., ge=0, description="Network I/O bytes per second")
    response_time_ms: float = Field(..., ge=0, description="Response time in milliseconds")
    error_rate: float = Field(..., ge=0, le=1, description="Error rate (0-1)")
    throughput: float = Field(..., ge=0, description="Throughput (operations per second)")
    timestamp: datetime = Field(..., description="Metrics timestamp")

class ProcessControlChart(BaseModel):
    """Statistical process control chart schema"""
    chart_id: str = Field(..., description="Unique chart identifier")
    metric_name: str = Field(..., description="Metric being monitored")
    chart_type: str = Field(..., description="Type of control chart")
    data_points: List[float] = Field(..., description="Data points for the chart")
    center_line: float = Field(..., description="Center line value")
    upper_control_limit: float = Field(..., description="Upper control limit")
    lower_control_limit: float = Field(..., description="Lower control limit")
    out_of_control_points: List[int] = Field(default_factory=list, description="Indices of out-of-control points")
    timestamp: datetime = Field(..., description="Chart generation timestamp")
    
    @validator('chart_type')
    def validate_chart_type(cls, v):
        """Validate control chart type"""
        valid_types = ['x-bar', 'r-chart', 'p-chart', 'c-chart', 'ewma', 'cusum']
        if v.lower() not in valid_types:
            raise ValueError(f"Chart type must be one of: {valid_types}")
        return v.lower()
    
    @validator('data_points')
    def validate_data_points(cls, v):
        """Validate data points are finite"""
        if not all(np.isfinite(x) for x in v):
            raise ValueError("All data points must be finite")
        return v

class PredictiveQualityScore(BaseModel):
    """Predictive quality score schema"""
    score_id: str = Field(..., description="Unique score identifier")
    component: str = Field(..., description="Component being scored")
    current_quality: float = Field(..., ge=0, le=1, description="Current quality score")
    predicted_quality: float = Field(..., ge=0, le=1, description="Predicted quality score")
    prediction_horizon_minutes: int = Field(..., gt=0, description="Prediction horizon in minutes")
    confidence_interval: Tuple[float, float] = Field(..., description="Confidence interval for prediction")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    timestamp: datetime = Field(..., description="Score generation timestamp")
    
    @validator('confidence_interval')
    def validate_confidence_interval(cls, v):
        """Validate confidence interval"""
        if len(v) != 2:
            raise ValueError("Confidence interval must have exactly 2 values")
        if v[0] > v[1]:
            raise ValueError("Confidence interval lower bound must be <= upper bound")
        if not (0 <= v[0] <= 1 and 0 <= v[1] <= 1):
            raise ValueError("Confidence interval values must be between 0 and 1")
        return v

# =============================================================================
# AUTOMATED REPORTING SCHEMAS
# =============================================================================

class QualityReport(BaseModel):
    """Comprehensive quality report schema"""
    report_id: str = Field(..., description="Unique report identifier")
    report_type: str = Field(..., description="Type of quality report")
    generation_timestamp: datetime = Field(..., description="Report generation timestamp")
    time_period: Dict[str, datetime] = Field(..., description="Time period covered by report")
    components_analyzed: List[str] = Field(..., description="Components included in analysis")
    quality_assessments: List[QualityAssessment] = Field(..., description="Quality assessments")
    anomaly_detections: List[AnomalyDetection] = Field(..., description="Anomaly detections")
    validation_reports: List[ValidationReport] = Field(..., description="Validation reports")
    health_metrics: List[SystemHealthMetrics] = Field(..., description="System health metrics")
    overall_system_health: float = Field(..., ge=0, le=1, description="Overall system health score")
    recommendations: List[str] = Field(default_factory=list, description="System improvement recommendations")
    
    @validator('time_period')
    def validate_time_period(cls, v):
        """Validate time period"""
        required_keys = ['start', 'end']
        if not all(key in v for key in required_keys):
            raise ValueError(f"Time period must contain keys: {required_keys}")
        if v['start'] >= v['end']:
            raise ValueError("Start time must be before end time")
        return v