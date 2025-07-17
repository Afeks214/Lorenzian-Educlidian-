"""
Comprehensive Data Schemas for Pipeline Validation
Agent Delta: Data Pipeline Transformation Specialist

This module provides centralized data schemas for bulletproof validation
across the entire trading system. Implements strict type checking, business
rules validation, and comprehensive data integrity constraints.

Key Features:
- Centralized schema definitions for all data types
- Financial domain-specific validators
- Real-time validation with performance optimization
- Extensible schema versioning and migration support
- Immutable data structures with lineage tracking
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Generic, TypeVar
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
import hashlib
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator, ValidationError
from pydantic.generics import GenericModel
import structlog

logger = structlog.get_logger(__name__)

# Type variables for generic schemas
T = TypeVar('T')
DataType = TypeVar('DataType')

# =============================================================================
# CORE ENUMERATIONS
# =============================================================================

class SchemaVersion(str, Enum):
    """Schema version enumeration"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"
    LATEST = "2.0"

class DataTier(str, Enum):
    """Data tier classification"""
    RAW = "raw"
    VALIDATED = "validated"
    NORMALIZED = "normalized"
    PROCESSED = "processed"
    AGGREGATED = "aggregated"

class DataSensitivity(str, Enum):
    """Data sensitivity classification"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class ValidationSeverity(str, Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class DataLifecycle(str, Enum):
    """Data lifecycle states"""
    CREATED = "created"
    VALIDATED = "validated"
    PROCESSED = "processed"
    ARCHIVED = "archived"
    DELETED = "deleted"

# =============================================================================
# BASE SCHEMA INFRASTRUCTURE
# =============================================================================

class BaseDataSchema(BaseModel):
    """Base schema for all data structures"""
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"
        allow_mutation = False
        validate_all = True
        use_enum_values = True
        
    # Metadata fields
    schema_version: SchemaVersion = Field(default=SchemaVersion.LATEST, description="Schema version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    data_tier: DataTier = Field(default=DataTier.RAW, description="Data tier classification")
    data_sensitivity: DataSensitivity = Field(default=DataSensitivity.INTERNAL, description="Data sensitivity level")
    lifecycle_state: DataLifecycle = Field(default=DataLifecycle.CREATED, description="Data lifecycle state")
    
    # Validation and integrity
    checksum: Optional[str] = Field(None, description="Data integrity checksum")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate data integrity checksum"""
        # Create deterministic representation of data
        data_str = self.json(sort_keys=True, exclude={'checksum', 'created_at', 'updated_at'})
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def validate_integrity(self) -> bool:
        """Validate data integrity using checksum"""
        expected_checksum = self._calculate_checksum()
        return self.checksum == expected_checksum
    
    def update_lifecycle_state(self, new_state: DataLifecycle) -> None:
        """Update lifecycle state with timestamp"""
        self.lifecycle_state = new_state
        self.updated_at = datetime.utcnow()

class VersionedSchema(BaseDataSchema):
    """Schema with version tracking and migration support"""
    
    version_history: List[Dict[str, Any]] = Field(default_factory=list, description="Version history")
    migration_applied: Optional[str] = Field(None, description="Last migration applied")
    
    def add_version_history(self, operation: str, details: Dict[str, Any]) -> None:
        """Add version history entry"""
        self.version_history.append({
            'timestamp': datetime.utcnow(),
            'operation': operation,
            'details': details,
            'schema_version': self.schema_version
        })

class ImmutableSchema(BaseDataSchema):
    """Immutable schema with lineage tracking"""
    
    class Config(BaseDataSchema.Config):
        allow_mutation = False
        frozen = True
    
    lineage_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique lineage identifier")
    parent_lineage_id: Optional[str] = Field(None, description="Parent lineage identifier")
    transformation_applied: Optional[str] = Field(None, description="Transformation applied to create this data")

# =============================================================================
# FINANCIAL DATA SCHEMAS
# =============================================================================

class CurrencyCode(str, Enum):
    """ISO 4217 currency codes"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    AUD = "AUD"
    CAD = "CAD"
    NZD = "NZD"

class ExchangeCode(str, Enum):
    """Financial exchange codes"""
    FOREX = "FOREX"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    CME = "CME"
    CBOT = "CBOT"
    NYMEX = "NYMEX"
    COMEX = "COMEX"
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"

class PriceType(str, Enum):
    """Price type classification"""
    BID = "bid"
    ASK = "ask"
    MID = "mid"
    LAST = "last"
    SETTLEMENT = "settlement"
    THEORETICAL = "theoretical"

class FinancialInstrument(ImmutableSchema):
    """Financial instrument definition"""
    
    symbol: str = Field(..., min_length=1, max_length=50, description="Trading symbol")
    isin: Optional[str] = Field(None, regex=r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', description="ISIN code")
    cusip: Optional[str] = Field(None, regex=r'^[0-9]{3}[0-9A-Z]{5}[0-9]$', description="CUSIP code")
    
    instrument_type: str = Field(..., description="Type of financial instrument")
    asset_class: str = Field(..., description="Asset class")
    currency: CurrencyCode = Field(..., description="Base currency")
    exchange: ExchangeCode = Field(..., description="Primary exchange")
    
    # Contract specifications
    contract_size: Optional[Decimal] = Field(None, description="Contract size")
    tick_size: Optional[Decimal] = Field(None, description="Minimum price increment")
    tick_value: Optional[Decimal] = Field(None, description="Value per tick")
    
    # Trading parameters
    trading_hours: Dict[str, Any] = Field(default_factory=dict, description="Trading hours")
    settlement_method: Optional[str] = Field(None, description="Settlement method")
    expiration_date: Optional[datetime] = Field(None, description="Expiration date")
    
    # Regulatory and classification
    regulatory_status: Optional[str] = Field(None, description="Regulatory status")
    sector: Optional[str] = Field(None, description="Sector classification")
    industry: Optional[str] = Field(None, description="Industry classification")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate symbol format"""
        if not v.replace('/', '').replace('-', '').replace('_', '').isalnum():
            raise ValueError("Symbol must contain only alphanumeric characters, hyphens, underscores, and forward slashes")
        return v.upper()

class PriceQuote(ImmutableSchema):
    """Price quote with comprehensive validation"""
    
    instrument: FinancialInstrument = Field(..., description="Financial instrument")
    timestamp: datetime = Field(..., description="Quote timestamp")
    
    # Price data
    price: Decimal = Field(..., gt=0, description="Quote price")
    price_type: PriceType = Field(..., description="Type of price")
    
    # Market data
    volume: Optional[Decimal] = Field(None, ge=0, description="Volume")
    open_interest: Optional[int] = Field(None, ge=0, description="Open interest")
    
    # Bid/Ask spread
    bid_price: Optional[Decimal] = Field(None, gt=0, description="Bid price")
    ask_price: Optional[Decimal] = Field(None, gt=0, description="Ask price")
    bid_size: Optional[Decimal] = Field(None, ge=0, description="Bid size")
    ask_size: Optional[Decimal] = Field(None, ge=0, description="Ask size")
    
    # Quality indicators
    data_source: str = Field(..., description="Data source")
    latency_ms: Optional[float] = Field(None, ge=0, description="Data latency in milliseconds")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence score")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is not too far in future"""
        if v > datetime.utcnow() + timedelta(seconds=60):
            raise ValueError("Timestamp cannot be more than 60 seconds in the future")
        return v
    
    @root_validator
    def validate_bid_ask_spread(cls, values):
        """Validate bid/ask spread consistency"""
        bid_price = values.get('bid_price')
        ask_price = values.get('ask_price')
        
        if bid_price is not None and ask_price is not None:
            if bid_price >= ask_price:
                raise ValueError("Bid price must be less than ask price")
        
        return values

class OHLCV(ImmutableSchema):
    """Open, High, Low, Close, Volume data"""
    
    instrument: FinancialInstrument = Field(..., description="Financial instrument")
    timestamp: datetime = Field(..., description="Bar timestamp")
    period_minutes: int = Field(..., gt=0, description="Period in minutes")
    
    # OHLC prices
    open_price: Decimal = Field(..., gt=0, description="Opening price")
    high_price: Decimal = Field(..., gt=0, description="High price")
    low_price: Decimal = Field(..., gt=0, description="Low price")
    close_price: Decimal = Field(..., gt=0, description="Closing price")
    
    # Volume and activity
    volume: Decimal = Field(..., ge=0, description="Volume")
    trade_count: Optional[int] = Field(None, ge=0, description="Number of trades")
    
    # Calculated fields
    typical_price: Optional[Decimal] = Field(None, description="Typical price (HLC/3)")
    vwap: Optional[Decimal] = Field(None, description="Volume weighted average price")
    
    # Quality metrics
    data_completeness: float = Field(1.0, ge=0, le=1, description="Data completeness ratio")
    adjustment_factor: Optional[Decimal] = Field(None, description="Price adjustment factor")
    
    @validator('high_price')
    def validate_high_price(cls, v, values):
        """Validate high price is highest"""
        if 'open_price' in values and v < values['open_price']:
            raise ValueError("High price must be >= open price")
        if 'low_price' in values and v < values['low_price']:
            raise ValueError("High price must be >= low price")
        if 'close_price' in values and v < values['close_price']:
            raise ValueError("High price must be >= close price")
        return v
    
    @validator('low_price')
    def validate_low_price(cls, v, values):
        """Validate low price is lowest"""
        if 'open_price' in values and v > values['open_price']:
            raise ValueError("Low price must be <= open price")
        if 'close_price' in values and v > values['close_price']:
            raise ValueError("Low price must be <= close price")
        return v
    
    def __post_init_post_parse__(self):
        """Calculate derived fields after validation"""
        if self.typical_price is None:
            self.typical_price = (self.high_price + self.low_price + self.close_price) / 3

# =============================================================================
# TRADING SCHEMAS
# =============================================================================

class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TradingOrder(ImmutableSchema):
    """Trading order with comprehensive validation"""
    
    order_id: str = Field(..., description="Unique order identifier")
    client_order_id: Optional[str] = Field(None, description="Client-side order identifier")
    
    # Order details
    instrument: FinancialInstrument = Field(..., description="Financial instrument")
    side: OrderSide = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    
    # Price information
    price: Optional[Decimal] = Field(None, gt=0, description="Order price")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop price")
    
    # Execution details
    status: OrderStatus = Field(default=OrderStatus.PENDING, description="Order status")
    filled_quantity: Decimal = Field(default=Decimal('0'), ge=0, description="Filled quantity")
    average_fill_price: Optional[Decimal] = Field(None, description="Average fill price")
    
    # Risk management
    position_size_limit: Optional[Decimal] = Field(None, description="Position size limit")
    risk_limit: Optional[Decimal] = Field(None, description="Risk limit")
    
    # Timing
    time_in_force: Optional[str] = Field(None, description="Time in force")
    expiration_time: Optional[datetime] = Field(None, description="Order expiration time")
    
    @validator('filled_quantity')
    def validate_filled_quantity(cls, v, values):
        """Validate filled quantity doesn't exceed order quantity"""
        if 'quantity' in values and v > values['quantity']:
            raise ValueError("Filled quantity cannot exceed order quantity")
        return v
    
    @root_validator
    def validate_order_constraints(cls, values):
        """Validate order-specific constraints"""
        order_type = values.get('order_type')
        price = values.get('price')
        stop_price = values.get('stop_price')
        
        if order_type == OrderType.LIMIT and price is None:
            raise ValueError("Limit orders must have a price")
        if order_type == OrderType.STOP and stop_price is None:
            raise ValueError("Stop orders must have a stop price")
        if order_type == OrderType.STOP_LIMIT and (price is None or stop_price is None):
            raise ValueError("Stop limit orders must have both price and stop price")
        
        return values

class TradeExecution(ImmutableSchema):
    """Trade execution record"""
    
    execution_id: str = Field(..., description="Unique execution identifier")
    order_id: str = Field(..., description="Related order identifier")
    
    # Execution details
    instrument: FinancialInstrument = Field(..., description="Financial instrument")
    side: OrderSide = Field(..., description="Trade side")
    quantity: Decimal = Field(..., gt=0, description="Executed quantity")
    price: Decimal = Field(..., gt=0, description="Execution price")
    
    # Timing
    execution_time: datetime = Field(..., description="Execution timestamp")
    
    # Financial details
    commission: Optional[Decimal] = Field(None, ge=0, description="Commission paid")
    fees: Optional[Decimal] = Field(None, ge=0, description="Additional fees")
    net_amount: Optional[Decimal] = Field(None, description="Net amount")
    
    # Market data
    market_price: Optional[Decimal] = Field(None, description="Market price at execution")
    slippage: Optional[Decimal] = Field(None, description="Price slippage")
    
    def __post_init_post_parse__(self):
        """Calculate derived fields"""
        if self.net_amount is None:
            gross_amount = self.quantity * self.price
            total_costs = (self.commission or Decimal('0')) + (self.fees or Decimal('0'))
            self.net_amount = gross_amount - total_costs

# =============================================================================
# RISK MANAGEMENT SCHEMAS
# =============================================================================

class RiskMetricType(str, Enum):
    """Risk metric type enumeration"""
    VAR = "var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    BETA = "beta"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LEVERAGE = "leverage"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"

class RiskMeasurement(ImmutableSchema):
    """Risk measurement with validation"""
    
    measurement_id: str = Field(..., description="Unique measurement identifier")
    metric_type: RiskMetricType = Field(..., description="Type of risk metric")
    
    # Measurement details
    value: Decimal = Field(..., description="Measured value")
    confidence_level: Optional[float] = Field(None, ge=0, le=1, description="Confidence level")
    time_horizon: Optional[int] = Field(None, gt=0, description="Time horizon in days")
    
    # Context
    portfolio_id: Optional[str] = Field(None, description="Portfolio identifier")
    instrument: Optional[FinancialInstrument] = Field(None, description="Instrument (if applicable)")
    
    # Calculation details
    calculation_method: str = Field(..., description="Calculation method used")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Calculation parameters")
    
    # Quality indicators
    data_quality_score: float = Field(1.0, ge=0, le=1, description="Data quality score")
    model_confidence: Optional[float] = Field(None, ge=0, le=1, description="Model confidence")
    
    @validator('confidence_level')
    def validate_confidence_level(cls, v, values):
        """Validate confidence level for VaR metrics"""
        metric_type = values.get('metric_type')
        if metric_type in [RiskMetricType.VAR, RiskMetricType.EXPECTED_SHORTFALL]:
            if v is None:
                raise ValueError(f"{metric_type.value} requires a confidence level")
            if v <= 0.5 or v >= 1.0:
                raise ValueError("Confidence level must be between 0.5 and 1.0")
        return v

# =============================================================================
# QUALITY ASSURANCE SCHEMAS
# =============================================================================

class QualityCheckType(str, Enum):
    """Quality check type enumeration"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"

class QualityCheck(BaseDataSchema):
    """Quality check definition and result"""
    
    check_id: str = Field(..., description="Unique check identifier")
    check_type: QualityCheckType = Field(..., description="Type of quality check")
    
    # Check definition
    check_name: str = Field(..., description="Human-readable check name")
    description: str = Field(..., description="Check description")
    
    # Execution
    target_schema: str = Field(..., description="Target schema for check")
    check_query: Optional[str] = Field(None, description="Check query or rule")
    
    # Results
    passed: bool = Field(..., description="Check passed")
    score: float = Field(..., ge=0, le=1, description="Quality score (0-1)")
    
    # Details
    records_checked: int = Field(..., ge=0, description="Number of records checked")
    records_passed: int = Field(..., ge=0, description="Number of records that passed")
    records_failed: int = Field(..., ge=0, description="Number of records that failed")
    
    failure_details: List[str] = Field(default_factory=list, description="Details of failures")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    @validator('records_passed')
    def validate_records_passed(cls, v, values):
        """Validate passed records count"""
        if 'records_checked' in values and v > values['records_checked']:
            raise ValueError("Passed records cannot exceed checked records")
        return v
    
    @validator('records_failed')
    def validate_records_failed(cls, v, values):
        """Validate failed records count"""
        if 'records_checked' in values and v > values['records_checked']:
            raise ValueError("Failed records cannot exceed checked records")
        return v

class DataQualityReport(BaseDataSchema):
    """Comprehensive data quality report"""
    
    report_id: str = Field(..., description="Unique report identifier")
    
    # Report scope
    data_source: str = Field(..., description="Data source assessed")
    assessment_period: Dict[str, datetime] = Field(..., description="Assessment time period")
    
    # Quality checks
    quality_checks: List[QualityCheck] = Field(..., description="Quality checks performed")
    
    # Summary metrics
    overall_score: float = Field(..., ge=0, le=1, description="Overall quality score")
    total_records: int = Field(..., ge=0, description="Total records assessed")
    
    # Recommendations
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues found")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    
    @validator('assessment_period')
    def validate_assessment_period(cls, v):
        """Validate assessment period"""
        if 'start' not in v or 'end' not in v:
            raise ValueError("Assessment period must contain 'start' and 'end' times")
        if v['start'] >= v['end']:
            raise ValueError("Assessment period start must be before end")
        return v
    
    @validator('overall_score')
    def validate_overall_score(cls, v, values):
        """Validate overall score is calculated correctly"""
        checks = values.get('quality_checks', [])
        if checks:
            calculated_score = sum(check.score for check in checks) / len(checks)
            if abs(v - calculated_score) > 0.001:
                raise ValueError("Overall score must be average of individual check scores")
        return v

# =============================================================================
# GENERIC CONTAINER SCHEMAS
# =============================================================================

class DataContainer(GenericModel, Generic[T]):
    """Generic data container with metadata"""
    
    container_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Container identifier")
    data: T = Field(..., description="Contained data")
    
    # Metadata
    data_type: str = Field(..., description="Type of contained data")
    version: str = Field(default="1.0", description="Data version")
    
    # Provenance
    source: str = Field(..., description="Data source")
    created_by: str = Field(..., description="Creator identifier")
    
    # Validation
    is_validated: bool = Field(default=False, description="Validation status")
    validation_timestamp: Optional[datetime] = Field(None, description="Validation timestamp")
    
    class Config:
        arbitrary_types_allowed = True

class BatchContainer(BaseDataSchema):
    """Container for batch data processing"""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    batch_size: int = Field(..., gt=0, description="Number of items in batch")
    
    # Processing details
    processing_started: datetime = Field(..., description="Processing start time")
    processing_completed: Optional[datetime] = Field(None, description="Processing completion time")
    
    # Status
    items_processed: int = Field(default=0, ge=0, description="Items processed successfully")
    items_failed: int = Field(default=0, ge=0, description="Items that failed processing")
    
    # Configuration
    processing_config: Dict[str, Any] = Field(default_factory=dict, description="Processing configuration")
    
    @validator('items_processed')
    def validate_items_processed(cls, v, values):
        """Validate processed items count"""
        if 'batch_size' in values and v > values['batch_size']:
            raise ValueError("Processed items cannot exceed batch size")
        return v

# =============================================================================
# SCHEMA REGISTRY AND VALIDATION
# =============================================================================

class SchemaRegistry:
    """Central registry for all data schemas"""
    
    def __init__(self):
        self.schemas: Dict[str, type] = {}
        self.validators: Dict[str, Callable] = {}
        self.migrations: Dict[str, Dict[str, Callable]] = {}
        
        # Register default schemas
        self._register_default_schemas()
    
    def _register_default_schemas(self):
        """Register default schemas"""
        schemas = {
            'financial_instrument': FinancialInstrument,
            'price_quote': PriceQuote,
            'ohlcv': OHLCV,
            'trading_order': TradingOrder,
            'trade_execution': TradeExecution,
            'risk_measurement': RiskMeasurement,
            'quality_check': QualityCheck,
            'data_quality_report': DataQualityReport,
            'batch_container': BatchContainer
        }
        
        for name, schema_class in schemas.items():
            self.register_schema(name, schema_class)
    
    def register_schema(self, name: str, schema_class: type):
        """Register a schema class"""
        self.schemas[name] = schema_class
        logger.info(f"Registered schema: {name}")
    
    def validate_data(self, schema_name: str, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against registered schema"""
        if schema_name not in self.schemas:
            return False, [f"Schema '{schema_name}' not found"]
        
        schema_class = self.schemas[schema_name]
        
        try:
            instance = schema_class(**data)
            return True, []
        except ValidationError as e:
            errors = []
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                errors.append(f"Field '{field}': {error['msg']}")
            return False, errors
    
    def get_schema(self, name: str) -> Optional[type]:
        """Get schema class by name"""
        return self.schemas.get(name)
    
    def list_schemas(self) -> List[str]:
        """List all registered schemas"""
        return list(self.schemas.keys())


# Global schema registry instance
schema_registry = SchemaRegistry()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_financial_data(data: Dict[str, Any], schema_name: str) -> Tuple[bool, List[str]]:
    """Validate financial data using schema registry"""
    return schema_registry.validate_data(schema_name, data)

def create_immutable_record(data: Dict[str, Any], schema_name: str) -> Optional[ImmutableSchema]:
    """Create immutable record from data"""
    schema_class = schema_registry.get_schema(schema_name)
    if not schema_class:
        return None
    
    try:
        return schema_class(**data)
    except ValidationError as e:
        logger.error(f"Failed to create immutable record: {e}")
        return None

def calculate_data_quality_score(checks: List[QualityCheck]) -> float:
    """Calculate overall data quality score"""
    if not checks:
        return 0.0
    
    # Weight checks by importance
    weights = {
        QualityCheckType.ACCURACY: 0.3,
        QualityCheckType.COMPLETENESS: 0.25,
        QualityCheckType.CONSISTENCY: 0.2,
        QualityCheckType.TIMELINESS: 0.15,
        QualityCheckType.VALIDITY: 0.1
    }
    
    weighted_score = 0.0
    total_weight = 0.0
    
    for check in checks:
        weight = weights.get(check.check_type, 0.1)
        weighted_score += check.score * weight
        total_weight += weight
    
    return weighted_score / total_weight if total_weight > 0 else 0.0

def generate_schema_documentation() -> Dict[str, Any]:
    """Generate documentation for all registered schemas"""
    docs = {}
    
    for name, schema_class in schema_registry.schemas.items():
        docs[name] = {
            'description': schema_class.__doc__ or "No description available",
            'fields': {},
            'validators': []
        }
        
        # Extract field information
        if hasattr(schema_class, '__fields__'):
            for field_name, field_info in schema_class.__fields__.items():
                docs[name]['fields'][field_name] = {
                    'type': str(field_info.type_),
                    'required': field_info.required,
                    'description': field_info.field_info.description or "No description"
                }
    
    return docs

# Export key components
__all__ = [
    'BaseDataSchema',
    'VersionedSchema', 
    'ImmutableSchema',
    'FinancialInstrument',
    'PriceQuote',
    'OHLCV',
    'TradingOrder',
    'TradeExecution',
    'RiskMeasurement',
    'QualityCheck',
    'DataQualityReport',
    'DataContainer',
    'BatchContainer',
    'SchemaRegistry',
    'schema_registry',
    'validate_financial_data',
    'create_immutable_record',
    'calculate_data_quality_score',
    'generate_schema_documentation'
]