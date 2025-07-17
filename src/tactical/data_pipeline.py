"""
Universal Multi-Asset Data Pipeline for Tactical MARL
AGENT 1 MISSION: Multi-Asset Data Architecture Specialist

Implements a comprehensive data pipeline capable of ingesting and normalizing
5-minute OHLCV data for diverse asset classes with production-grade reliability.

Features:
- Universal data ingestion for Forex, Commodities, Equities, Crypto
- Adaptive price normalization handling different scales (1.08 EUR/USD vs 17,000 NQ)
- Real-time data validation and quality checks
- Asset-specific preprocessing with configurable parameters
- Streaming data support with buffering capabilities
- Production-grade error handling and monitoring

Asset Classes Supported:
- FOREX: EUR/USD, GBP/USD, USD/JPY, etc.
- COMMODITIES: XAU/USD (Gold), XAG/USD (Silver), CL (Oil), etc.
- EQUITIES: NQ (Nasdaq), ES (S&P 500), RTY (Russell 2000), etc.
- CRYPTO: BTC/USD, ETH/USD, etc.

Author: Agent 1 - Multi-Asset Data Architecture Specialist
Version: 2.0 - Mission Dominion Universal Pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import warnings
from pydantic import ValidationError
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import validation schemas
try:
    from src.api.models import (
        MarketDataPoint, PriceData, NormalizedPriceData, 
        DataQuality, AssetClass, ValidationRule, ValidationReport,
        ValidationResult, QualityMetric, QualityAssessment
    )
    VALIDATION_AVAILABLE = True
except ImportError:
    logger.warning("Validation schemas not available - validation will be disabled")
    VALIDATION_AVAILABLE = False


# Use imported schemas if available, otherwise use local enums
if not VALIDATION_AVAILABLE:
    class AssetClass(Enum):
        """Supported asset classes for tactical MARL trading"""
        FOREX = "FOREX"
        COMMODITIES = "COMMODITIES" 
        EQUITIES = "EQUITIES"
        CRYPTO = "CRYPTO"
        BONDS = "BONDS"

    class DataQuality(Enum):
        """Data quality assessment levels"""
        EXCELLENT = "EXCELLENT"
        GOOD = "GOOD"
        ACCEPTABLE = "ACCEPTABLE"
        POOR = "POOR"
        UNUSABLE = "UNUSABLE"


@dataclass
class AssetMetadata:
    """Metadata container for asset-specific configuration"""
    symbol: str
    asset_class: AssetClass
    tick_size: float
    lot_size: float
    currency: str
    exchange: str
    trading_hours: Dict[str, Any]
    price_precision: int
    volume_precision: int
    
    # Asset-specific normalization parameters
    price_scale_factor: float = 1.0
    volume_scale_factor: float = 1.0
    typical_daily_range: float = 0.0
    volatility_adjustment: float = 1.0


@dataclass
class MarketDataPoint:
    """Standardized market data point structure"""
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Quality and metadata
    data_quality: DataQuality
    source: str
    latency_ms: float
    
    # Normalized values for MARL processing
    normalized_open: float = 0.0
    normalized_high: float = 0.0
    normalized_low: float = 0.0
    normalized_close: float = 0.0
    normalized_volume: float = 0.0


class UniversalDataPipeline:
    """
    Universal Multi-Asset Data Pipeline
    
    Core component for Mission Dominion Phase 1 - enables tactical MARL
    to trade across multiple asset classes with unified data processing.
    
    Key Features:
    - Asset-agnostic data ingestion and normalization
    - Real-time streaming with quality validation
    - Configurable preprocessing for different market characteristics
    - Production-grade error handling and monitoring
    - Thread-safe operations with asyncio support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Universal Data Pipeline
        
        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config or self._default_config()
        
        # Asset metadata registry
        self.asset_registry: Dict[str, AssetMetadata] = {}
        self._initialize_asset_registry()
        
        # Data buffers for real-time processing
        self.data_buffers: Dict[str, deque] = {}
        self.buffer_lock = threading.Lock()
        
        # Quality monitoring
        self.quality_stats = {
            'total_points': 0,
            'quality_distribution': {q.value: 0 for q in DataQuality},
            'latency_stats': {'min': float('inf'), 'max': 0, 'avg': 0},
            'error_count': 0
        }
        
        # Threading for concurrent processing
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )
        
        # Pipeline status
        self.is_running = False
        self.processed_count = 0
        
        # Validation engine
        self.validation_enabled = VALIDATION_AVAILABLE and self.config.get('enable_validation', True)
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.validation_reports: deque = deque(maxlen=100)
        self.validation_lock = threading.Lock()
        
        if self.validation_enabled:
            self._initialize_validation_rules()
            logger.info("Data validation engine initialized")
        
        logger.info(f"Universal Data Pipeline initialized for {len(self.asset_registry)} asset types")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for universal data pipeline"""
        return {
            'buffer_size': 1000,
            'max_workers': 4,
            'quality_threshold': DataQuality.ACCEPTABLE,
            'normalization_window': 252,  # Trading days for volatility calc
            'outlier_threshold': 5.0,     # Standard deviations
            'max_latency_ms': 100,        # Maximum acceptable latency
            'validate_prices': True,
            'validate_volumes': True,
            'enable_streaming': True,
            'enable_validation': True,    # Enable Pydantic validation
            'validation_strict': True,    # Strict validation mode
            'validation_report_frequency': 100  # Generate report every N records
        }
    
    def _initialize_asset_registry(self):
        """Initialize metadata for supported asset classes"""
        
        # FOREX Assets
        forex_assets = [
            ("EUR/USD", 0.00001, 100000, "USD", "FOREX", 4),
            ("GBP/USD", 0.00001, 100000, "USD", "FOREX", 4),
            ("USD/JPY", 0.001, 100000, "JPY", "FOREX", 2),
            ("AUD/USD", 0.00001, 100000, "USD", "FOREX", 4),
            ("USD/CHF", 0.00001, 100000, "CHF", "FOREX", 4),
        ]
        
        for symbol, tick_size, lot_size, currency, exchange, precision in forex_assets:
            self.asset_registry[symbol] = AssetMetadata(
                symbol=symbol,
                asset_class=AssetClass.FOREX,
                tick_size=tick_size,
                lot_size=lot_size,
                currency=currency,
                exchange=exchange,
                trading_hours={"start": "22:00", "end": "22:00"},  # 24/5
                price_precision=precision,
                volume_precision=0,
                price_scale_factor=10000 if precision == 4 else 100,  # Pip normalization
                volume_scale_factor=1e-6,  # Normalize to millions
                typical_daily_range=0.01,  # 100 pips typical
                volatility_adjustment=1.0
            )
        
        # COMMODITIES Assets
        commodities_assets = [
            ("XAU/USD", 0.01, 100, "USD", "COMEX", 2000.0, 50.0),   # Gold
            ("XAG/USD", 0.001, 5000, "USD", "COMEX", 25.0, 2.0),   # Silver
            ("CL", 0.01, 1000, "USD", "NYMEX", 80.0, 5.0),         # Oil
            ("GC", 0.10, 100, "USD", "COMEX", 2000.0, 50.0),       # Gold Futures
        ]
        
        for symbol, tick_size, lot_size, currency, exchange, price_level, daily_range in commodities_assets:
            self.asset_registry[symbol] = AssetMetadata(
                symbol=symbol,
                asset_class=AssetClass.COMMODITIES,
                tick_size=tick_size,
                lot_size=lot_size,
                currency=currency,
                exchange=exchange,
                trading_hours={"start": "18:00", "end": "17:00"},  # 23/6
                price_precision=2,
                volume_precision=0,
                price_scale_factor=1.0 / price_level,  # Normalize to ~1.0
                volume_scale_factor=1e-3,  # Thousands
                typical_daily_range=daily_range / price_level,
                volatility_adjustment=1.2  # Higher vol adjustment
            )
        
        # EQUITIES Assets (Futures)
        equity_assets = [
            ("NQ", 0.25, 20, "USD", "CME", 17000.0, 300.0),        # Nasdaq
            ("ES", 0.25, 50, "USD", "CME", 4500.0, 75.0),          # S&P 500
            ("RTY", 0.10, 50, "USD", "CME", 2000.0, 40.0),         # Russell 2000
            ("YM", 1.0, 5, "USD", "CBOT", 35000.0, 500.0),         # Dow Jones
        ]
        
        for symbol, tick_size, lot_size, currency, exchange, price_level, daily_range in equity_assets:
            self.asset_registry[symbol] = AssetMetadata(
                symbol=symbol,
                asset_class=AssetClass.EQUITIES,
                tick_size=tick_size,
                lot_size=lot_size,
                currency=currency,
                exchange=exchange,
                trading_hours={"start": "09:30", "end": "16:00"},  # Regular hours
                price_precision=2,
                volume_precision=0,
                price_scale_factor=1.0 / price_level,  # Normalize to ~1.0
                volume_scale_factor=1e-3,  # Thousands
                typical_daily_range=daily_range / price_level,
                volatility_adjustment=0.8  # Lower vol adjustment
            )
        
        # CRYPTO Assets
        crypto_assets = [
            ("BTC/USD", 0.01, 1, "USD", "CRYPTO", 50000.0, 2000.0),
            ("ETH/USD", 0.01, 1, "USD", "CRYPTO", 3000.0, 150.0),
        ]
        
        for symbol, tick_size, lot_size, currency, exchange, price_level, daily_range in crypto_assets:
            self.asset_registry[symbol] = AssetMetadata(
                symbol=symbol,
                asset_class=AssetClass.CRYPTO,
                tick_size=tick_size,
                lot_size=lot_size,
                currency=currency,
                exchange=exchange,
                trading_hours={"start": "00:00", "end": "24:00"},  # 24/7
                price_precision=2,
                volume_precision=4,
                price_scale_factor=1.0 / price_level,  # Normalize to ~1.0
                volume_scale_factor=1e-6,  # Millions
                typical_daily_range=daily_range / price_level,
                volatility_adjustment=2.0  # Much higher vol adjustment
            )
    
    def _initialize_validation_rules(self):
        """Initialize validation rules for data quality"""
        if not VALIDATION_AVAILABLE:
            return
            
        # General validation rules
        general_rules = [
            ValidationRule(
                rule_id="price_positive",
                rule_name="Price values must be positive",
                rule_type="range",
                target_field="price",
                parameters={"min": 0, "exclusive_min": True}
            ),
            ValidationRule(
                rule_id="volume_non_negative",
                rule_name="Volume must be non-negative",
                rule_type="range",
                target_field="volume",
                parameters={"min": 0}
            ),
            ValidationRule(
                rule_id="ohlc_consistency",
                rule_name="OHLC price consistency",
                rule_type="consistency",
                target_field="ohlc",
                parameters={"check_order": True}
            ),
            ValidationRule(
                rule_id="timestamp_validity",
                rule_name="Timestamp must not be in future",
                rule_type="format",
                target_field="timestamp",
                parameters={"max_future_seconds": 60}
            ),
            ValidationRule(
                rule_id="latency_threshold",
                rule_name="Data latency within acceptable range",
                rule_type="range",
                target_field="latency_ms",
                parameters={"max": self.config['max_latency_ms']}
            )
        ]
        
        # Apply rules to all asset classes
        for asset_class in AssetClass:
            self.validation_rules[asset_class.value] = general_rules.copy()
            
            # Asset-specific rules
            if asset_class == AssetClass.FOREX:
                self.validation_rules[asset_class.value].extend([
                    ValidationRule(
                        rule_id="forex_price_range",
                        rule_name="Forex price within reasonable range",
                        rule_type="range",
                        target_field="price",
                        parameters={"min": 0.0001, "max": 1000}
                    )
                ])
            elif asset_class == AssetClass.EQUITIES:
                self.validation_rules[asset_class.value].extend([
                    ValidationRule(
                        rule_id="equity_price_range",
                        rule_name="Equity price within reasonable range",
                        rule_type="range",
                        target_field="price",
                        parameters={"min": 0.01, "max": 100000}
                    )
                ])
    
    def _validate_with_pydantic(self, data_point: Dict[str, Any], symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate data point using Pydantic schemas
        
        Args:
            data_point: Raw data point dictionary
            symbol: Asset symbol
            
        Returns:
            Tuple[bool, List[str]]: Validation success and error messages
        """
        if not self.validation_enabled:
            return True, []
            
        errors = []
        
        try:
            # Get asset metadata
            if symbol not in self.asset_registry:
                errors.append(f"Asset {symbol} not registered")
                return False, errors
                
            metadata = self.asset_registry[symbol]
            
            # Create price data schema
            price_data = PriceData(
                open=data_point['open'],
                high=data_point['high'],
                low=data_point['low'],
                close=data_point['close'],
                volume=data_point['volume']
            )
            
            # Create normalized price data schema
            normalized_data = NormalizedPriceData(
                normalized_open=data_point['open'] * metadata.price_scale_factor,
                normalized_high=data_point['high'] * metadata.price_scale_factor,
                normalized_low=data_point['low'] * metadata.price_scale_factor,
                normalized_close=data_point['close'] * metadata.price_scale_factor,
                normalized_volume=data_point['volume'] * metadata.volume_scale_factor,
                scale_factor=metadata.price_scale_factor
            )
            
            # Create full market data point
            market_data = MarketDataPoint(
                symbol=symbol,
                timestamp=pd.to_datetime(data_point['timestamp']),
                price_data=price_data,
                normalized_data=normalized_data,
                asset_class=metadata.asset_class,
                data_quality=DataQuality.EXCELLENT,  # Will be assessed later
                source=data_point.get('source', 'unknown'),
                latency_ms=data_point.get('latency_ms', 0.0)
            )
            
            return True, []
            
        except ValidationError as e:
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                errors.append(f"Field '{field}': {error['msg']}")
            return False, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def _apply_validation_rules(self, data_point: Dict[str, Any], symbol: str) -> Tuple[bool, List[str]]:
        """
        Apply custom validation rules
        
        Args:
            data_point: Raw data point dictionary
            symbol: Asset symbol
            
        Returns:
            Tuple[bool, List[str]]: Validation success and error messages
        """
        if not self.validation_enabled or symbol not in self.asset_registry:
            return True, []
            
        metadata = self.asset_registry[symbol]
        asset_class = metadata.asset_class.value
        
        if asset_class not in self.validation_rules:
            return True, []
            
        errors = []
        
        for rule in self.validation_rules[asset_class]:
            if not rule.enabled:
                continue
                
            try:
                if rule.rule_type == "range":
                    if rule.target_field == "price":
                        price_fields = ['open', 'high', 'low', 'close']
                        for field in price_fields:
                            if field in data_point:
                                value = data_point[field]
                                if 'min' in rule.parameters and value < rule.parameters['min']:
                                    errors.append(f"Rule '{rule.rule_name}' failed: {field} value {value} below minimum {rule.parameters['min']}")
                                if 'max' in rule.parameters and value > rule.parameters['max']:
                                    errors.append(f"Rule '{rule.rule_name}' failed: {field} value {value} above maximum {rule.parameters['max']}")
                                if rule.parameters.get('exclusive_min', False) and value <= rule.parameters['min']:
                                    errors.append(f"Rule '{rule.rule_name}' failed: {field} value {value} must be greater than {rule.parameters['min']}")
                    
                    elif rule.target_field == "volume":
                        if 'volume' in data_point:
                            value = data_point['volume']
                            if 'min' in rule.parameters and value < rule.parameters['min']:
                                errors.append(f"Rule '{rule.rule_name}' failed: volume {value} below minimum {rule.parameters['min']}")
                            if 'max' in rule.parameters and value > rule.parameters['max']:
                                errors.append(f"Rule '{rule.rule_name}' failed: volume {value} above maximum {rule.parameters['max']}")
                    
                    elif rule.target_field == "latency_ms":
                        if 'latency_ms' in data_point:
                            value = data_point['latency_ms']
                            if 'max' in rule.parameters and value > rule.parameters['max']:
                                errors.append(f"Rule '{rule.rule_name}' failed: latency {value}ms above maximum {rule.parameters['max']}ms")
                
                elif rule.rule_type == "consistency":
                    if rule.target_field == "ohlc":
                        o, h, l, c = data_point.get('open', 0), data_point.get('high', 0), data_point.get('low', 0), data_point.get('close', 0)
                        if not (l <= o <= h and l <= c <= h):
                            errors.append(f"Rule '{rule.rule_name}' failed: OHLC values are inconsistent")
                
                elif rule.rule_type == "format":
                    if rule.target_field == "timestamp":
                        if 'timestamp' in data_point:
                            timestamp = pd.to_datetime(data_point['timestamp'])
                            max_future = rule.parameters.get('max_future_seconds', 60)
                            if timestamp > datetime.utcnow() + timedelta(seconds=max_future):
                                errors.append(f"Rule '{rule.rule_name}' failed: timestamp too far in future")
                        
            except Exception as e:
                errors.append(f"Rule '{rule.rule_name}' evaluation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _generate_validation_report(self, results: List[Dict[str, Any]]) -> Optional[ValidationReport]:
        """
        Generate validation report from validation results
        
        Args:
            results: List of validation results
            
        Returns:
            Optional[ValidationReport]: Generated report or None if validation disabled
        """
        if not self.validation_enabled or not results:
            return None
        
        total_records = len(results)
        passed_records = sum(1 for r in results if r['success'])
        failed_records = sum(1 for r in results if not r['success'])
        warning_records = 0  # Could be implemented based on warning conditions
        
        # Aggregate rule results
        all_rules = set()
        for result in results:
            all_rules.update(result.get('rules_applied', []))
        
        rule_results = {}
        for rule_id in all_rules:
            rule_successes = sum(1 for r in results if rule_id in r.get('rules_applied', []) and r['success'])
            rule_total = sum(1 for r in results if rule_id in r.get('rules_applied', []))
            rule_results[rule_id] = ValidationResult.PASSED if rule_successes == rule_total else ValidationResult.FAILED
        
        # Collect error details
        error_details = []
        for result in results:
            if not result['success']:
                error_details.extend(result.get('errors', []))
        
        # Create report
        report = ValidationReport(
            report_id=str(uuid.uuid4()),
            data_source="UniversalDataPipeline",
            validation_timestamp=datetime.utcnow(),
            total_records=total_records,
            passed_records=passed_records,
            failed_records=failed_records,
            warning_records=warning_records,
            validation_rules=list(self.validation_rules.get('general', [])),
            rule_results=rule_results,
            error_details=error_details[:50]  # Limit to first 50 errors
        )
        
        return report
    
    def register_asset(self, metadata: AssetMetadata) -> bool:
        """
        Register a new asset for processing
        
        Args:
            metadata: Asset metadata configuration
            
        Returns:
            bool: Success status
        """
        try:
            self.asset_registry[metadata.symbol] = metadata
            
            # Initialize data buffer
            with self.buffer_lock:
                self.data_buffers[metadata.symbol] = deque(
                    maxlen=self.config['buffer_size']
                )
            
            logger.info(f"Registered new asset: {metadata.symbol} ({metadata.asset_class.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register asset {metadata.symbol}: {e}")
            return False
    
    def validate_data_point(self, data_point: Dict[str, Any], symbol: str) -> Tuple[bool, DataQuality]:
        """
        Validate incoming data point for quality and completeness
        
        Args:
            data_point: Raw data point dictionary
            symbol: Asset symbol
            
        Returns:
            Tuple[bool, DataQuality]: Validation result and quality assessment
        """
        try:
            # Basic completeness check
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(field in data_point for field in required_fields):
                return False, DataQuality.UNUSABLE
            
            # Price validation
            if self.config['validate_prices']:
                o, h, l, c = data_point['open'], data_point['high'], data_point['low'], data_point['close']
                
                # OHLC consistency
                if not (l <= o <= h and l <= c <= h):
                    return False, DataQuality.POOR
                
                # Check for zero or negative prices
                if any(price <= 0 for price in [o, h, l, c]):
                    return False, DataQuality.UNUSABLE
                
                # Outlier detection (if we have historical data)
                if symbol in self.data_buffers and len(self.data_buffers[symbol]) > 10:
                    recent_closes = [point.close for point in list(self.data_buffers[symbol])[-10:]]
                    avg_price = np.mean(recent_closes)
                    price_deviation = abs(c - avg_price) / avg_price
                    
                    if price_deviation > 0.1:  # 10% deviation threshold
                        return False, DataQuality.POOR
            
            # Volume validation
            if self.config['validate_volumes']:
                volume = data_point['volume']
                if volume < 0:
                    return False, DataQuality.POOR
            
            # Assess quality based on completeness and consistency
            quality = DataQuality.EXCELLENT
            
            # Downgrade quality for missing optional fields
            optional_fields = ['source', 'latency_ms']
            missing_optional = sum(1 for field in optional_fields if field not in data_point)
            
            if missing_optional > 0:
                quality = DataQuality.GOOD
            
            # Check latency if provided
            if 'latency_ms' in data_point:
                latency = data_point['latency_ms']
                if latency > self.config['max_latency_ms']:
                    quality = DataQuality.ACCEPTABLE
            
            return True, quality
            
        except Exception as e:
            logger.error(f"Data validation error for {symbol}: {e}")
            return False, DataQuality.UNUSABLE
    
    def normalize_data_point(self, data_point: Dict[str, Any], symbol: str) -> MarketDataPoint:
        """
        Normalize raw data point using asset-specific parameters
        
        Args:
            data_point: Raw data point dictionary
            symbol: Asset symbol
            
        Returns:
            MarketDataPoint: Normalized data point
        """
        try:
            metadata = self.asset_registry[symbol]
            
            # Extract raw values
            timestamp = pd.to_datetime(data_point['timestamp'])
            o, h, l, c = data_point['open'], data_point['high'], data_point['low'], data_point['close']
            volume = data_point['volume']
            
            # Apply asset-specific price normalization
            price_scale = metadata.price_scale_factor
            normalized_open = o * price_scale
            normalized_high = h * price_scale
            normalized_low = l * price_scale
            normalized_close = c * price_scale
            
            # Volume normalization
            volume_scale = metadata.volume_scale_factor
            normalized_volume = volume * volume_scale
            
            # Validate and assess quality
            is_valid, quality = self.validate_data_point(data_point, symbol)
            
            # Create standardized data point
            market_data = MarketDataPoint(
                timestamp=timestamp,
                symbol=symbol,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=volume,
                data_quality=quality,
                source=data_point.get('source', 'unknown'),
                latency_ms=data_point.get('latency_ms', 0.0),
                normalized_open=normalized_open,
                normalized_high=normalized_high,
                normalized_low=normalized_low,
                normalized_close=normalized_close,
                normalized_volume=normalized_volume
            )
            
            return market_data
            
        except Exception as e:
            logger.error(f"Normalization error for {symbol}: {e}")
            raise
    
    def process_data_point(self, data_point: Dict[str, Any], symbol: str) -> Optional[MarketDataPoint]:
        """
        Process a single data point through the pipeline with comprehensive validation
        
        Args:
            data_point: Raw data point
            symbol: Asset symbol
            
        Returns:
            Optional[MarketDataPoint]: Processed data point or None if invalid
        """
        validation_start_time = time.time()
        validation_results = []
        
        try:
            # Check if asset is registered
            if symbol not in self.asset_registry:
                logger.warning(f"Unregistered asset: {symbol}")
                return None
            
            # Step 1: Pydantic schema validation
            pydantic_success, pydantic_errors = self._validate_with_pydantic(data_point, symbol)
            validation_results.append({
                'validation_type': 'pydantic',
                'success': pydantic_success,
                'errors': pydantic_errors,
                'rules_applied': ['pydantic_schema']
            })
            
            # Step 2: Custom validation rules
            rules_success, rules_errors = self._apply_validation_rules(data_point, symbol)
            validation_results.append({
                'validation_type': 'custom_rules',
                'success': rules_success,
                'errors': rules_errors,
                'rules_applied': [rule.rule_id for rule in self.validation_rules.get(self.asset_registry[symbol].asset_class.value, [])]
            })
            
            # Determine overall validation result
            overall_success = pydantic_success and rules_success
            all_errors = pydantic_errors + rules_errors
            
            if not overall_success:
                if self.config.get('validation_strict', True):
                    # Strict mode: reject data on any validation failure
                    logger.warning(f"Validation failed for {symbol}: {'; '.join(all_errors)}")
                    self.quality_stats['error_count'] += 1
                    return None
                else:
                    # Lenient mode: log warnings but continue processing
                    logger.warning(f"Validation warnings for {symbol}: {'; '.join(all_errors)}")
            
            # Step 3: Normalize data point
            normalized_point = self.normalize_data_point(data_point, symbol)
            
            # Step 4: Quality check
            quality_threshold = self.config['quality_threshold']
            quality_levels = list(DataQuality)
            
            if quality_levels.index(normalized_point.data_quality) > quality_levels.index(quality_threshold):
                logger.warning(f"Data quality below threshold for {symbol}: {normalized_point.data_quality}")
                return None
            
            # Step 5: Add to buffer
            with self.buffer_lock:
                self.data_buffers[symbol].append(normalized_point)
            
            # Step 6: Update statistics
            self._update_quality_stats(normalized_point)
            
            # Step 7: Update validation statistics
            validation_duration = (time.time() - validation_start_time) * 1000  # Convert to ms
            
            if self.validation_enabled:
                with self.validation_lock:
                    # Store validation results for reporting
                    validation_record = {
                        'symbol': symbol,
                        'timestamp': datetime.utcnow(),
                        'success': overall_success,
                        'errors': all_errors,
                        'validation_duration_ms': validation_duration,
                        'rules_applied': [r for result in validation_results for r in result.get('rules_applied', [])]
                    }
                    
                    # Generate validation report periodically
                    if self.processed_count % self.config.get('validation_report_frequency', 100) == 0:
                        recent_results = list(self.validation_reports)[-100:]  # Last 100 results
                        if recent_results:
                            report = self._generate_validation_report(recent_results)
                            if report:
                                logger.info(f"Validation report generated: {report.passed_records}/{report.total_records} passed")
            
            self.processed_count += 1
            return normalized_point
            
        except Exception as e:
            logger.error(f"Processing error for {symbol}: {e}")
            self.quality_stats['error_count'] += 1
            return None
    
    async def process_data_stream(self, data_stream: asyncio.Queue, symbol: str) -> None:
        """
        Process continuous data stream for an asset
        
        Args:
            data_stream: Async queue of data points
            symbol: Asset symbol
        """
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    # Get data with timeout
                    data_point = await asyncio.wait_for(data_stream.get(), timeout=1.0)
                    
                    # Process in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    processed_point = await loop.run_in_executor(
                        self.executor, 
                        self.process_data_point, 
                        data_point, 
                        symbol
                    )
                    
                    if processed_point:
                        logger.debug(f"Processed {symbol}: {processed_point.normalized_close:.6f}")
                    
                except asyncio.TimeoutError:
                    # No data available, continue
                    continue
                except Exception as e:
                    logger.error(f"Stream processing error for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Fatal error in data stream for {symbol}: {e}")
        finally:
            self.is_running = False
    
    def get_latest_data(self, symbol: str, count: int = 1) -> List[MarketDataPoint]:
        """
        Get latest data points for an asset
        
        Args:
            symbol: Asset symbol
            count: Number of latest points to retrieve
            
        Returns:
            List[MarketDataPoint]: Latest data points
        """
        with self.buffer_lock:
            if symbol not in self.data_buffers:
                return []
            
            buffer = self.data_buffers[symbol]
            if len(buffer) == 0:
                return []
            
            # Return last 'count' items
            start_idx = max(0, len(buffer) - count)
            return list(buffer)[start_idx:]
    
    def get_data_matrix(self, symbol: str, window_size: int = 60) -> Optional[np.ndarray]:
        """
        Get normalized data matrix for MARL processing
        
        Args:
            symbol: Asset symbol
            window_size: Number of time steps
            
        Returns:
            Optional[np.ndarray]: Data matrix (window_size, 5) or None
        """
        latest_data = self.get_latest_data(symbol, window_size)
        
        if len(latest_data) < window_size:
            logger.warning(f"Insufficient data for {symbol}: {len(latest_data)}/{window_size}")
            return None
        
        # Create matrix: [normalized_open, normalized_high, normalized_low, normalized_close, normalized_volume]
        matrix = np.array([
            [
                point.normalized_open,
                point.normalized_high,
                point.normalized_low,
                point.normalized_close,
                point.normalized_volume
            ]
            for point in latest_data[-window_size:]
        ])
        
        return matrix
    
    def _update_quality_stats(self, data_point: MarketDataPoint):
        """Update quality statistics"""
        self.quality_stats['total_points'] += 1
        self.quality_stats['quality_distribution'][data_point.data_quality.value] += 1
        
        # Update latency stats
        latency = data_point.latency_ms
        self.quality_stats['latency_stats']['min'] = min(self.quality_stats['latency_stats']['min'], latency)
        self.quality_stats['latency_stats']['max'] = max(self.quality_stats['latency_stats']['max'], latency)
        
        # Update average
        total = self.quality_stats['total_points']
        old_avg = self.quality_stats['latency_stats']['avg']
        self.quality_stats['latency_stats']['avg'] = (old_avg * (total - 1) + latency) / total
    
    def get_supported_assets(self) -> Dict[AssetClass, List[str]]:
        """
        Get all supported assets organized by asset class
        
        Returns:
            Dict[AssetClass, List[str]]: Assets by class
        """
        assets_by_class = {}
        
        for symbol, metadata in self.asset_registry.items():
            asset_class = metadata.asset_class
            if asset_class not in assets_by_class:
                assets_by_class[asset_class] = []
            assets_by_class[asset_class].append(symbol)
        
        return assets_by_class
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            'is_running': self.is_running,
            'registered_assets': len(self.asset_registry),
            'processed_count': self.processed_count,
            'quality_stats': self.quality_stats,
            'buffer_status': {
                symbol: len(buffer) for symbol, buffer in self.data_buffers.items()
            },
            'supported_asset_classes': [ac.value for ac in AssetClass]
        }
    
    def stop(self):
        """Stop the data pipeline"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        logger.info("Universal Data Pipeline stopped")


# Test and validation functions
def create_sample_data() -> Dict[str, Dict[str, Any]]:
    """Create sample data for testing different asset classes"""
    
    # Sample Forex data (EUR/USD)
    forex_sample = {
        'timestamp': '2025-07-13T10:00:00Z',
        'open': 1.0850,
        'high': 1.0865,
        'low': 1.0845,
        'close': 1.0860,
        'volume': 50000000,  # 50M units
        'source': 'test_forex',
        'latency_ms': 5.2
    }
    
    # Sample Commodities data (Gold)
    commodities_sample = {
        'timestamp': '2025-07-13T10:00:00Z',
        'open': 2050.50,
        'high': 2055.75,
        'low': 2048.25,
        'close': 2053.00,
        'volume': 125000,  # 125K oz
        'source': 'test_commodities',
        'latency_ms': 8.1
    }
    
    # Sample Equities data (NQ)
    equities_sample = {
        'timestamp': '2025-07-13T10:00:00Z',
        'open': 17250.50,
        'high': 17285.75,
        'low': 17240.25,
        'close': 17275.00,
        'volume': 85000,  # 85K contracts
        'source': 'test_equities',
        'latency_ms': 3.7
    }
    
    return {
        'EUR/USD': forex_sample,
        'XAU/USD': commodities_sample,
        'NQ': equities_sample
    }


def test_pipeline_validation():
    """Test pipeline with sample data from different asset classes"""
    print("🧪 Testing Universal Data Pipeline")
    
    # Initialize pipeline
    pipeline = UniversalDataPipeline()
    
    # Get sample data
    sample_data = create_sample_data()
    
    # Test each asset class
    for symbol, data_point in sample_data.items():
        print(f"\n📊 Testing {symbol}:")
        
        # Process data point
        result = pipeline.process_data_point(data_point, symbol)
        
        if result:
            print(f"  ✅ Processed successfully")
            print(f"  📈 Original close: {result.close}")
            print(f"  🔄 Normalized close: {result.normalized_close:.6f}")
            print(f"  📊 Quality: {result.data_quality.value}")
            print(f"  ⚡ Latency: {result.latency_ms}ms")
        else:
            print(f"  ❌ Processing failed")
    
    # Test data matrix generation
    print(f"\n📋 Pipeline Status:")
    status = pipeline.get_pipeline_status()
    print(f"  Registered assets: {status['registered_assets']}")
    print(f"  Processed points: {status['processed_count']}")
    
    # Test asset organization
    assets_by_class = pipeline.get_supported_assets()
    for asset_class, symbols in assets_by_class.items():
        print(f"  {asset_class.value}: {len(symbols)} assets")
    
    pipeline.stop()
    print("\n✅ Universal Data Pipeline validation complete!")


if __name__ == "__main__":
    test_pipeline_validation()