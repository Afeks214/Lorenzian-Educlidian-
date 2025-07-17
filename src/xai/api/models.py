"""
API Models and Schemas for XAI API
AGENT DELTA MISSION: Comprehensive API Data Models

This module defines comprehensive Pydantic models and schemas for request/response
validation, ensuring type safety and API documentation throughout the XAI system.

Features:
- Request/response models with validation
- Enum definitions for standardized values
- Complex nested data structures
- OpenAPI documentation support
- Type safety and serialization
- Custom validation rules

Author: Agent Delta - Integration Specialist
Version: 1.0 - API Data Models
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import UUID4, PositiveInt, NonNegativeFloat, confloat, constr


# Enums for standardized values
class ExplanationTypeEnum(str, Enum):
    """Types of explanations available"""
    FEATURE_IMPORTANCE = "FEATURE_IMPORTANCE"
    DECISION_PATH = "DECISION_PATH"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    CONFIDENCE_ANALYSIS = "CONFIDENCE_ANALYSIS"
    REGULATORY_SUMMARY = "REGULATORY_SUMMARY"


class ExplanationAudienceEnum(str, Enum):
    """Target audiences for explanations"""
    TRADER = "TRADER"
    RISK_MANAGER = "RISK_MANAGER"
    REGULATOR = "REGULATOR"
    CLIENT = "CLIENT"
    TECHNICAL = "TECHNICAL"


class QueryComplexityEnum(str, Enum):
    """Query complexity levels"""
    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"
    ANALYTICAL = "ANALYTICAL"


class QueryIntentEnum(str, Enum):
    """Query intent types"""
    PERFORMANCE_ANALYSIS = "PERFORMANCE_ANALYSIS"
    DECISION_EXPLANATION = "DECISION_EXPLANATION"
    AGENT_COMPARISON = "AGENT_COMPARISON"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    HISTORICAL_ANALYSIS = "HISTORICAL_ANALYSIS"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    MARKET_INSIGHTS = "MARKET_INSIGHTS"
    COMPLIANCE_QUERY = "COMPLIANCE_QUERY"


class ActionTypeEnum(str, Enum):
    """Trading action types"""
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    HOLD = "HOLD"
    INCREASE_LONG = "INCREASE_LONG"
    DECREASE_LONG = "DECREASE_LONG"
    INCREASE_SHORT = "INCREASE_SHORT"
    DECREASE_SHORT = "DECREASE_SHORT"


class RiskLevelEnum(str, Enum):
    """Risk level categories"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ComplianceStatusEnum(str, Enum):
    """Compliance status values"""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PENDING_REVIEW = "PENDING_REVIEW"
    UNDER_INVESTIGATION = "UNDER_INVESTIGATION"


class VisualizationTypeEnum(str, Enum):
    """Visualization types"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    PIE_CHART = "pie_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    TIME_SERIES = "time_series"
    GAUGE = "gauge"
    TABLE = "table"


# Base models
class BaseRequest(BaseModel):
    """Base request model"""
    correlation_id: Optional[str] = Field(
        None,
        description="Correlation ID for request tracking"
    )
    
    class Config:
        extra = "forbid"


class BaseResponse(BaseModel):
    """Base response model"""
    correlation_id: str = Field(
        description="Correlation ID for request tracking"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    class Config:
        extra = "forbid"


# Health and monitoring models
class HealthCheckResponse(BaseResponse):
    """Health check response"""
    status: str = Field(
        description="System health status",
        regex="^(healthy|degraded|unhealthy)$"
    )
    version: str = Field(
        description="API version"
    )
    components: Dict[str, bool] = Field(
        description="Component health status"
    )
    uptime_seconds: NonNegativeFloat = Field(
        description="System uptime in seconds"
    )
    active_connections: NonNegativeFloat = Field(
        description="Number of active WebSocket connections"
    )


# Explanation models
class ExplanationRequest(BaseRequest):
    """Request for decision explanation"""
    symbol: constr(min_length=1, max_length=10) = Field(
        description="Trading symbol (e.g., 'NQ', 'ES')"
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Decision timestamp (defaults to most recent)"
    )
    explanation_type: ExplanationTypeEnum = Field(
        default=ExplanationTypeEnum.FEATURE_IMPORTANCE,
        description="Type of explanation to generate"
    )
    audience: ExplanationAudienceEnum = Field(
        default=ExplanationAudienceEnum.TRADER,
        description="Target audience for explanation"
    )


class FeatureFactor(BaseModel):
    """Feature factor with importance score"""
    name: str = Field(description="Feature name")
    importance: confloat(ge=0.0, le=1.0) = Field(description="Importance score (0-1)")
    
    class Config:
        extra = "forbid"


class AlternativeScenario(BaseModel):
    """Alternative decision scenario"""
    scenario: str = Field(description="Scenario name")
    description: str = Field(description="Scenario description")
    confidence_change: confloat(ge=-1.0, le=1.0) = Field(
        description="Confidence change (-1 to 1)"
    )
    reasoning: str = Field(description="Scenario reasoning")
    
    class Config:
        extra = "forbid"


class ComplianceMetadata(BaseModel):
    """Compliance metadata for explanations"""
    regulatory_compliant: bool = Field(description="Regulatory compliance status")
    audit_trail_id: str = Field(description="Audit trail identifier")
    explanation_quality: confloat(ge=0.0, le=1.0) = Field(
        description="Explanation quality score"
    )
    
    class Config:
        extra = "forbid"


class ExplanationResponse(BaseResponse):
    """Response for decision explanation"""
    explanation_id: str = Field(description="Unique explanation identifier")
    symbol: str = Field(description="Trading symbol")
    timestamp: datetime = Field(description="Decision timestamp")
    explanation_type: ExplanationTypeEnum = Field(description="Explanation type")
    audience: ExplanationAudienceEnum = Field(description="Target audience")
    reasoning: str = Field(description="Natural language explanation")
    feature_importance: Dict[str, float] = Field(
        description="Feature importance scores"
    )
    top_positive_factors: List[FeatureFactor] = Field(
        description="Top positive contributing factors"
    )
    top_negative_factors: List[FeatureFactor] = Field(
        description="Top negative contributing factors"
    )
    confidence_score: confloat(ge=0.0, le=1.0) = Field(
        description="Explanation confidence score"
    )
    agent_contributions: Dict[str, Any] = Field(
        description="Individual agent contributions"
    )
    strategic_context: Dict[str, Any] = Field(
        description="Strategic MARL context"
    )
    alternative_scenarios: List[AlternativeScenario] = Field(
        description="Alternative decision scenarios"
    )
    compliance_metadata: ComplianceMetadata = Field(
        description="Compliance and audit information"
    )


# Query models
class TimeRange(BaseModel):
    """Time range specification"""
    start_time: datetime = Field(description="Start time")
    end_time: datetime = Field(description="End time")
    
    @validator('end_time')
    def end_time_after_start_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v
    
    class Config:
        extra = "forbid"


class QueryRequest(BaseRequest):
    """Natural language query request"""
    query: constr(min_length=1, max_length=1000) = Field(
        description="Natural language query"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context for the query"
    )
    time_range: Optional[TimeRange] = Field(
        None,
        description="Time range filter for the query"
    )


class Visualization(BaseModel):
    """Visualization specification"""
    type: VisualizationTypeEnum = Field(description="Visualization type")
    title: str = Field(description="Visualization title")
    data: Dict[str, Any] = Field(description="Visualization data")
    description: str = Field(description="Visualization description")
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional visualization configuration"
    )
    
    class Config:
        extra = "forbid"


class ProcessingMetadata(BaseModel):
    """Query processing metadata"""
    processing_time_ms: NonNegativeFloat = Field(
        description="Processing time in milliseconds"
    )
    data_points_analyzed: PositiveInt = Field(
        description="Number of data points analyzed"
    )
    marl_integration_used: bool = Field(
        description="Whether Strategic MARL integration was used"
    )
    
    class Config:
        extra = "forbid"


class QueryResponse(BaseResponse):
    """Natural language query response"""
    query_id: str = Field(description="Unique query identifier")
    original_query: str = Field(description="Original query text")
    interpreted_intent: QueryIntentEnum = Field(description="Interpreted query intent")
    answer: str = Field(description="Natural language answer")
    supporting_data: Dict[str, Any] = Field(description="Supporting data")
    visualizations: List[Visualization] = Field(description="Generated visualizations")
    confidence_score: confloat(ge=0.0, le=1.0) = Field(
        description="Response confidence score"
    )
    follow_up_suggestions: List[str] = Field(description="Follow-up question suggestions")
    data_sources: List[str] = Field(description="Data sources used")
    processing_metadata: ProcessingMetadata = Field(description="Processing metadata")


# Decision history models
class AgentVote(BaseModel):
    """Individual agent vote"""
    agent_id: str = Field(description="Agent identifier")
    action: ActionTypeEnum = Field(description="Recommended action")
    confidence: confloat(ge=0.0, le=1.0) = Field(description="Agent confidence")
    reasoning: Optional[str] = Field(None, description="Agent reasoning")
    
    class Config:
        extra = "forbid"


class MarketContext(BaseModel):
    """Market context information"""
    regime: str = Field(description="Market regime")
    volatility: confloat(ge=0.0) = Field(description="Market volatility")
    volume_ratio: confloat(ge=0.0) = Field(description="Volume ratio")
    trend_strength: confloat(ge=-1.0, le=1.0) = Field(description="Trend strength")
    
    class Config:
        extra = "forbid"


class PerformanceOutcome(BaseModel):
    """Performance outcome of a decision"""
    success: bool = Field(description="Whether decision was successful")
    pnl: float = Field(description="Profit/Loss from decision")
    time_to_target: Optional[NonNegativeFloat] = Field(
        None,
        description="Time to reach target in seconds"
    )
    risk_adjusted_return: Optional[float] = Field(
        None,
        description="Risk-adjusted return"
    )
    
    class Config:
        extra = "forbid"


class DecisionInfo(BaseModel):
    """Decision information"""
    decision_id: str = Field(description="Unique decision identifier")
    timestamp: datetime = Field(description="Decision timestamp")
    symbol: str = Field(description="Trading symbol")
    action: ActionTypeEnum = Field(description="Final action taken")
    confidence: confloat(ge=0.0, le=1.0) = Field(description="Decision confidence")
    agent_votes: List[AgentVote] = Field(description="Individual agent votes")
    market_context: MarketContext = Field(description="Market context")
    performance_outcome: Optional[PerformanceOutcome] = Field(
        None,
        description="Performance outcome (if available)"
    )
    explanation_summary: Optional[str] = Field(
        None,
        description="Brief explanation summary"
    )
    
    class Config:
        extra = "forbid"


class SummaryStats(BaseModel):
    """Summary statistics"""
    success_rate: confloat(ge=0.0, le=1.0) = Field(description="Success rate")
    average_confidence: confloat(ge=0.0, le=1.0) = Field(description="Average confidence")
    symbols_traded: PositiveInt = Field(description="Number of symbols traded")
    time_range: Dict[str, Optional[datetime]] = Field(description="Time range covered")
    
    class Config:
        extra = "forbid"


class DecisionHistoryResponse(BaseResponse):
    """Decision history response"""
    total_decisions: NonNegativeFloat = Field(description="Total number of decisions")
    decisions: List[DecisionInfo] = Field(description="Decision information list")
    summary_stats: SummaryStats = Field(description="Summary statistics")


# Performance analytics models
class OverallPerformance(BaseModel):
    """Overall system performance metrics"""
    score: confloat(ge=0.0, le=1.0) = Field(description="Overall performance score")
    total_trades: NonNegativeFloat = Field(description="Total number of trades")
    win_rate: confloat(ge=0.0, le=1.0) = Field(description="Win rate")
    profit_factor: confloat(ge=0.0) = Field(description="Profit factor")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    max_drawdown: confloat(ge=0.0, le=1.0) = Field(description="Maximum drawdown")
    total_pnl: float = Field(description="Total profit/loss")
    
    class Config:
        extra = "forbid"


class AgentPerformance(BaseModel):
    """Individual agent performance metrics"""
    accuracy: confloat(ge=0.0, le=1.0) = Field(description="Agent accuracy")
    confidence_calibration: confloat(ge=0.0, le=1.0) = Field(
        description="Confidence calibration score"
    )
    contribution_score: confloat(ge=0.0, le=1.0) = Field(
        description="Contribution to overall performance"
    )
    specialization_effectiveness: confloat(ge=0.0, le=1.0) = Field(
        description="Effectiveness in specialized conditions"
    )
    
    class Config:
        extra = "forbid"


class DecisionQualityMetrics(BaseModel):
    """Decision quality metrics"""
    explanation_coverage: confloat(ge=0.0, le=1.0) = Field(
        description="Percentage of decisions with explanations"
    )
    average_confidence: confloat(ge=0.0, le=1.0) = Field(
        description="Average decision confidence"
    )
    consensus_strength: confloat(ge=0.0, le=1.0) = Field(
        description="Average consensus strength"
    )
    regulatory_compliance: confloat(ge=0.0, le=1.0) = Field(
        description="Regulatory compliance score"
    )
    
    class Config:
        extra = "forbid"


class StrategicInsights(BaseModel):
    """Strategic insights from MARL system"""
    dominant_regime: str = Field(description="Dominant market regime")
    optimal_agent_weighting: List[float] = Field(
        description="Optimal agent weights [MLMI, NWRQK, Regime]"
    )
    market_conditions_suitability: confloat(ge=0.0, le=1.0) = Field(
        description="Suitability of current conditions"
    )
    synergy_detection_accuracy: confloat(ge=0.0, le=1.0) = Field(
        description="Synergy detection accuracy"
    )
    
    class Config:
        extra = "forbid"


class RiskMetrics(BaseModel):
    """Risk assessment metrics"""
    var_95: confloat(ge=0.0) = Field(description="95% Value at Risk")
    expected_shortfall: confloat(ge=0.0) = Field(description="Expected shortfall")
    volatility_adjusted_return: float = Field(description="Volatility-adjusted return")
    risk_adjusted_score: confloat(ge=0.0, le=1.0) = Field(
        description="Overall risk-adjusted score"
    )
    
    class Config:
        extra = "forbid"


class SystemHealth(BaseModel):
    """System health metrics"""
    uptime: confloat(ge=0.0, le=1.0) = Field(description="System uptime percentage")
    average_latency_ms: NonNegativeFloat = Field(description="Average latency in ms")
    error_rate: confloat(ge=0.0, le=1.0) = Field(description="Error rate")
    capacity_utilization: confloat(ge=0.0, le=1.0) = Field(
        description="Capacity utilization"
    )
    
    class Config:
        extra = "forbid"


class PerformanceAnalyticsResponse(BaseResponse):
    """Performance analytics response"""
    time_range: str = Field(description="Analysis time range")
    symbol: Optional[str] = Field(None, description="Symbol filter (if applied)")
    overall_performance: OverallPerformance = Field(description="Overall performance")
    agent_performance: Dict[str, AgentPerformance] = Field(
        description="Individual agent performance"
    )
    decision_quality_metrics: DecisionQualityMetrics = Field(
        description="Decision quality metrics"
    )
    strategic_insights: StrategicInsights = Field(description="Strategic insights")
    risk_metrics: RiskMetrics = Field(description="Risk metrics")
    system_health: SystemHealth = Field(description="System health")
    recommendations: List[str] = Field(description="Performance recommendations")


# Compliance models
class ComplianceReportRequest(BaseRequest):
    """Compliance report request"""
    start_date: datetime = Field(description="Report start date")
    end_date: datetime = Field(description="Report end date")
    report_type: str = Field(
        default="standard",
        description="Type of compliance report"
    )
    
    @validator('end_date')
    def end_date_after_start_date(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class DecisionAuditEntry(BaseModel):
    """Decision audit trail entry"""
    decision_id: str = Field(description="Decision identifier")
    timestamp: datetime = Field(description="Decision timestamp")
    symbol: str = Field(description="Trading symbol")
    action: ActionTypeEnum = Field(description="Action taken")
    reasoning: str = Field(description="Decision reasoning")
    compliance_status: ComplianceStatusEnum = Field(description="Compliance status")
    explanation_quality: confloat(ge=0.0, le=1.0) = Field(
        description="Explanation quality score"
    )
    
    class Config:
        extra = "forbid"


class RegulatoryComplianceStatus(BaseModel):
    """Regulatory compliance status"""
    overall_status: ComplianceStatusEnum = Field(description="Overall compliance status")
    mifid_ii_compliant: bool = Field(description="MiFID II compliance")
    best_execution_compliant: bool = Field(description="Best execution compliance")
    transparency_compliant: bool = Field(description="Transparency compliance")
    explanation_coverage: NonNegativeFloat = Field(
        description="Number of decisions with explanations"
    )
    
    class Config:
        extra = "forbid"


class RiskAssessment(BaseModel):
    """Risk assessment for compliance"""
    risk_level: RiskLevelEnum = Field(description="Overall risk level")
    identified_risks: List[str] = Field(description="Identified risks")
    mitigation_measures: List[str] = Field(description="Mitigation measures")
    
    class Config:
        extra = "forbid"


class ComplianceReportResponse(BaseResponse):
    """Compliance report response"""
    report_id: str = Field(description="Unique report identifier")
    generated_at: datetime = Field(description="Report generation timestamp")
    report_period: Dict[str, datetime] = Field(description="Report period")
    summary: Dict[str, Any] = Field(description="Report summary")
    decision_audit_trail: List[DecisionAuditEntry] = Field(
        description="Decision audit trail"
    )
    explanation_quality_metrics: Dict[str, Any] = Field(
        description="Explanation quality metrics"
    )
    regulatory_compliance_status: RegulatoryComplianceStatus = Field(
        description="Regulatory compliance status"
    )
    risk_assessment: RiskAssessment = Field(description="Risk assessment")
    recommendations: List[str] = Field(description="Compliance recommendations")


# WebSocket models
class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    id: str = Field(description="Message identifier")
    type: str = Field(description="Message type")
    timestamp: datetime = Field(description="Message timestamp")
    data: Dict[str, Any] = Field(description="Message data")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    
    class Config:
        extra = "forbid"


class SubscriptionRequest(BaseModel):
    """WebSocket subscription request"""
    subscription_type: str = Field(description="Subscription type")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Subscription filters"
    )
    
    class Config:
        extra = "forbid"


# Error models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    class Config:
        extra = "forbid"


class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(description="Field with validation error")
    message: str = Field(description="Validation error message")
    invalid_value: Any = Field(description="Invalid value")
    
    class Config:
        extra = "forbid"


class ValidationErrorResponse(ErrorResponse):
    """Validation error response"""
    validation_errors: List[ValidationError] = Field(
        description="List of validation errors"
    )


# Custom validators and utility functions
def generate_correlation_id() -> str:
    """Generate a new correlation ID"""
    return str(uuid.uuid4())


def validate_symbol(symbol: str) -> str:
    """Validate trading symbol"""
    if not symbol or len(symbol) < 1 or len(symbol) > 10:
        raise ValueError("Symbol must be 1-10 characters")
    return symbol.upper()


def validate_confidence_score(score: float) -> float:
    """Validate confidence score"""
    if not 0.0 <= score <= 1.0:
        raise ValueError("Confidence score must be between 0.0 and 1.0")
    return score


# Model aliases for backward compatibility
HealthResponse = HealthCheckResponse
ExplainRequest = ExplanationRequest
ExplainResponse = ExplanationResponse
QueryRequest = QueryRequest
QueryResponse = QueryResponse