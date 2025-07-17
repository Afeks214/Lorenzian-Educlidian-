"""
Execution Context Vector (15D) Processing Pipeline
=================================================

This module implements the specialized 15-dimensional execution context vector processing
pipeline for the unified execution MARL system. Provides real-time context extraction,
normalization, and feature engineering for optimal execution decisions.

Technical Specifications:
- Input: Raw market and portfolio data
- Output: 15D normalized execution context vector
- Processing time: <50μs target
- Update frequency: Real-time (sub-millisecond)
- Feature engineering: Statistical, risk, and temporal features

Author: Agent 5 - Integration Validation & Production Certification
Date: 2025-07-13
Mission: Complete execution context processing pipeline implementation
"""

import numpy as np
import torch
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import structlog
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = structlog.get_logger()


@dataclass
class RawMarketData:
    """Raw market data input for context processing"""
    timestamp: datetime
    
    # Price data
    price: float
    bid: float = 0.0
    ask: float = 0.0
    mid_price: float = 0.0
    
    # Volume data
    volume: float = 0.0
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    
    # Order book data
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    spread: float = 0.0
    
    # Market microstructure
    tick_direction: int = 0  # -1, 0, 1
    trade_size: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields"""
        if self.mid_price == 0.0:
            self.mid_price = (self.bid + self.ask) / 2 if self.bid > 0 and self.ask > 0 else self.price
        
        if self.spread == 0.0:
            self.spread = self.ask - self.bid if self.ask > self.bid else 0.0


@dataclass 
class RawPortfolioData:
    """Raw portfolio data input for context processing"""
    timestamp: datetime
    
    # Portfolio values
    portfolio_value: float
    available_capital: float
    margin_used: float = 0.0
    
    # Position data
    current_position: float = 0.0
    average_entry_price: float = 0.0
    position_duration: float = 0.0  # seconds
    
    # PnL data
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    
    # Risk metrics
    var_estimate: float = 0.0
    portfolio_beta: float = 1.0
    correlation_spy: float = 0.0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0


@dataclass
class ProcessingMetrics:
    """Processing performance metrics"""
    total_processes: int = 0
    avg_processing_time_us: float = 0.0
    p95_processing_time_us: float = 0.0
    p99_processing_time_us: float = 0.0
    
    # Feature-specific timing
    feature_timings: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    feature_quality_scores: Dict[str, float] = field(default_factory=dict)
    missing_data_rates: Dict[str, float] = field(default_factory=dict)
    
    # Error tracking
    processing_errors: int = 0
    error_rate: float = 0.0


class ExecutionContextProcessor:
    """
    High-performance execution context vector processor
    
    Transforms raw market and portfolio data into 15D normalized execution context
    optimized for multi-agent decision making with <50μs processing time target.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize execution context processor"""
        self.config = config
        
        # Processing parameters
        self.lookback_periods = config.get('lookback_periods', 50)
        self.normalization_window = config.get('normalization_window', 252)
        self.feature_smoothing = config.get('feature_smoothing', 0.1)
        
        # Data storage for statistical calculations
        self.price_history = deque(maxlen=self.normalization_window)
        self.volume_history = deque(maxlen=self.normalization_window)
        self.pnl_history = deque(maxlen=self.normalization_window)
        self.volatility_history = deque(maxlen=self.normalization_window)
        
        # Real-time statistics
        self.running_stats = {
            'price_mean': 0.0,
            'price_std': 1.0,
            'volume_mean': 0.0,
            'volume_std': 1.0,
            'return_mean': 0.0,
            'return_std': 0.01,
            'volatility_mean': 0.15,
            'volatility_std': 0.05
        }
        
        # Performance tracking
        self.metrics = ProcessingMetrics()
        self.processing_times = deque(maxlen=1000)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # Feature extractors
        self._initialize_feature_extractors()
        
        logger.info("ExecutionContextProcessor initialized",
                   lookback_periods=self.lookback_periods,
                   normalization_window=self.normalization_window,
                   feature_smoothing=self.feature_smoothing)
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction functions"""
        self.feature_extractors = {
            'portfolio_features': self._extract_portfolio_features,
            'risk_features': self._extract_risk_features,
            'market_microstructure': self._extract_market_microstructure_features,
            'temporal_features': self._extract_temporal_features,
            'performance_features': self._extract_performance_features
        }
    
    async def process_execution_context(self, 
                                      market_data: RawMarketData,
                                      portfolio_data: RawPortfolioData) -> torch.Tensor:
        """
        Process raw data into 15D execution context vector
        
        Args:
            market_data: Raw market data
            portfolio_data: Raw portfolio data
            
        Returns:
            15D normalized execution context tensor
        """
        start_time = time.perf_counter()
        
        try:
            # Update historical data
            self._update_historical_data(market_data, portfolio_data)
            
            # Extract features in parallel for maximum performance
            feature_tasks = [
                self._extract_features_async('portfolio_features', market_data, portfolio_data),
                self._extract_features_async('risk_features', market_data, portfolio_data),
                self._extract_features_async('market_microstructure', market_data, portfolio_data),
                self._extract_features_async('temporal_features', market_data, portfolio_data),
                self._extract_features_async('performance_features', market_data, portfolio_data)
            ]
            
            # Wait for all feature extractions
            feature_results = await asyncio.gather(*feature_tasks)
            
            # Combine features into 15D vector
            context_vector = self._combine_features(feature_results)
            
            # Normalize vector
            normalized_vector = self._normalize_context_vector(context_vector)
            
            # Validate vector quality
            self._validate_context_vector(normalized_vector)
            
            # Update performance metrics
            processing_time = (time.perf_counter() - start_time) * 1_000_000  # microseconds
            self._update_processing_metrics(processing_time)
            
            return torch.tensor(normalized_vector, dtype=torch.float32)
            
        except Exception as e:
            logger.error("Context processing failed", error=str(e))
            self.metrics.processing_errors += 1
            self._update_error_metrics()
            
            # Return safe fallback vector
            return self._create_fallback_vector()
    
    def _update_historical_data(self, market_data: RawMarketData, portfolio_data: RawPortfolioData):
        """Update historical data for statistical calculations"""
        # Update price history
        self.price_history.append(market_data.price)
        
        # Update volume history
        self.volume_history.append(market_data.volume)
        
        # Update PnL history
        self.pnl_history.append(portfolio_data.unrealized_pnl)
        
        # Calculate and update volatility
        if len(self.price_history) >= 2:
            price_return = (self.price_history[-1] / self.price_history[-2] - 1)
            self.volatility_history.append(abs(price_return))
        
        # Update running statistics (exponential moving averages)
        alpha = self.feature_smoothing
        
        if len(self.price_history) > 1:
            self.running_stats['price_mean'] = (
                alpha * market_data.price + (1 - alpha) * self.running_stats['price_mean']
            )
            
            current_vol = np.std(list(self.volatility_history)) if len(self.volatility_history) > 5 else 0.15
            self.running_stats['volatility_mean'] = (
                alpha * current_vol + (1 - alpha) * self.running_stats['volatility_mean']
            )
    
    async def _extract_features_async(self, 
                                    feature_type: str, 
                                    market_data: RawMarketData,
                                    portfolio_data: RawPortfolioData) -> np.ndarray:
        """Extract features asynchronously"""
        extractor = self.feature_extractors[feature_type]
        
        # Run feature extraction in thread pool
        features = await asyncio.get_event_loop().run_in_executor(
            self.executor, extractor, market_data, portfolio_data
        )
        
        return features
    
    def _extract_portfolio_features(self, 
                                  market_data: RawMarketData,
                                  portfolio_data: RawPortfolioData) -> np.ndarray:
        """Extract portfolio-related features (dimensions 0-2)"""
        features = np.zeros(3)
        
        try:
            # Feature 0: Portfolio value (normalized)
            features[0] = portfolio_data.portfolio_value / 100000.0  # Normalize to $100k base
            
            # Feature 1: Available capital ratio
            if portfolio_data.portfolio_value > 0:
                features[1] = portfolio_data.available_capital / portfolio_data.portfolio_value
            else:
                features[1] = 1.0  # Default full availability
            
            # Feature 2: Current position size (normalized)
            if portfolio_data.portfolio_value > 0:
                position_value = portfolio_data.current_position * market_data.price
                features[2] = position_value / portfolio_data.portfolio_value
            else:
                features[2] = 0.0
            
        except Exception as e:
            logger.warning("Portfolio feature extraction error", error=str(e))
            features = np.array([1.0, 1.0, 0.0])  # Safe defaults
        
        return features
    
    def _extract_risk_features(self, 
                             market_data: RawMarketData,
                             portfolio_data: RawPortfolioData) -> np.ndarray:
        """Extract risk-related features (dimensions 3-6)"""
        features = np.zeros(4)
        
        try:
            # Feature 3: Unrealized PnL (normalized)
            if portfolio_data.portfolio_value > 0:
                features[3] = portfolio_data.unrealized_pnl / portfolio_data.portfolio_value
            else:
                features[3] = 0.0
            
            # Feature 4: VaR estimate (normalized)
            features[4] = portfolio_data.var_estimate
            
            # Feature 5: Expected return estimate
            if len(self.price_history) >= 10:
                recent_returns = []
                for i in range(1, min(10, len(self.price_history))):
                    ret = self.price_history[-i] / self.price_history[-(i+1)] - 1
                    recent_returns.append(ret)
                features[5] = np.mean(recent_returns) if recent_returns else 0.0
            else:
                features[5] = 0.0
            
            # Feature 6: Volatility estimate
            if len(self.volatility_history) > 0:
                features[6] = np.mean(list(self.volatility_history))
            else:
                features[6] = self.running_stats['volatility_mean']
            
        except Exception as e:
            logger.warning("Risk feature extraction error", error=str(e))
            features = np.array([0.0, 0.02, 0.0, 0.15])  # Safe defaults
        
        return features
    
    def _extract_market_microstructure_features(self, 
                                              market_data: RawMarketData,
                                              portfolio_data: RawPortfolioData) -> np.ndarray:
        """Extract market microstructure features (dimensions 7-10)"""
        features = np.zeros(4)
        
        try:
            # Feature 7: Bid-ask spread (normalized)
            if market_data.mid_price > 0:
                features[7] = market_data.spread / market_data.mid_price
            else:
                features[7] = 0.001  # Default 10 bps spread
            
            # Feature 8: Volume ratio (current vs average)
            if self.running_stats['volume_mean'] > 0:
                features[8] = market_data.volume / self.running_stats['volume_mean']
            else:
                features[8] = 1.0  # Default average volume
            
            # Feature 9: Order flow imbalance
            total_volume = market_data.bid_volume + market_data.ask_volume
            if total_volume > 0:
                features[9] = (market_data.ask_volume - market_data.bid_volume) / total_volume
            else:
                features[9] = 0.0  # Neutral flow
            
            # Feature 10: Tick direction momentum
            features[10] = float(market_data.tick_direction)
            
        except Exception as e:
            logger.warning("Market microstructure feature extraction error", error=str(e))
            features = np.array([0.001, 1.0, 0.0, 0.0])  # Safe defaults
        
        return features
    
    def _extract_temporal_features(self, 
                                 market_data: RawMarketData,
                                 portfolio_data: RawPortfolioData) -> np.ndarray:
        """Extract temporal features (dimensions 11-12)"""
        features = np.zeros(2)
        
        try:
            current_time = market_data.timestamp
            
            # Feature 11: Time since last trade (normalized to hours)
            if hasattr(self, 'last_trade_time'):
                time_since_trade = (current_time - self.last_trade_time).total_seconds() / 3600.0
                features[11] = min(time_since_trade, 24.0) / 24.0  # Normalize to daily cycle
            else:
                features[11] = 0.0
            
            if portfolio_data.current_position != 0:
                self.last_trade_time = current_time
            
            # Feature 12: Position holding time (normalized to days)
            if portfolio_data.position_duration > 0:
                holding_days = portfolio_data.position_duration / 86400.0  # Convert seconds to days
                features[12] = min(holding_days, 30.0) / 30.0  # Normalize to monthly cycle
            else:
                features[12] = 0.0
            
        except Exception as e:
            logger.warning("Temporal feature extraction error", error=str(e))
            features = np.array([0.0, 0.0])  # Safe defaults
        
        return features
    
    def _extract_performance_features(self, 
                                    market_data: RawMarketData,
                                    portfolio_data: RawPortfolioData) -> np.ndarray:
        """Extract performance features (dimensions 13-14)"""
        features = np.zeros(2)
        
        try:
            # Feature 13: Sharpe ratio (clipped)
            features[13] = np.clip(portfolio_data.sharpe_ratio, -3.0, 3.0) / 3.0  # Normalize to [-1, 1]
            
            # Feature 14: Current drawdown (normalized)
            features[14] = min(portfolio_data.current_drawdown, 1.0)  # Cap at 100% drawdown
            
        except Exception as e:
            logger.warning("Performance feature extraction error", error=str(e))
            features = np.array([0.0, 0.0])  # Safe defaults
        
        return features
    
    def _combine_features(self, feature_results: List[np.ndarray]) -> np.ndarray:
        """Combine feature arrays into 15D vector"""
        try:
            combined = np.concatenate(feature_results)
            
            # Ensure exactly 15 dimensions
            if len(combined) != 15:
                logger.warning("Feature vector dimension mismatch", 
                             expected=15, actual=len(combined))
                # Pad or truncate to 15 dimensions
                if len(combined) < 15:
                    combined = np.pad(combined, (0, 15 - len(combined)), mode='constant')
                else:
                    combined = combined[:15]
            
            return combined
            
        except Exception as e:
            logger.error("Feature combination failed", error=str(e))
            return np.zeros(15)
    
    def _normalize_context_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize context vector for neural network input"""
        try:
            # Apply tanh normalization to bound values in [-1, 1]
            normalized = np.tanh(vector)
            
            # Handle special cases
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return normalized
            
        except Exception as e:
            logger.error("Vector normalization failed", error=str(e))
            return np.zeros(15)
    
    def _validate_context_vector(self, vector: np.ndarray):
        """Validate context vector quality"""
        try:
            # Check for invalid values
            if np.any(np.isnan(vector)):
                logger.warning("Context vector contains NaN values")
                self.metrics.feature_quality_scores['nan_rate'] = np.mean(np.isnan(vector))
            
            if np.any(np.isinf(vector)):
                logger.warning("Context vector contains infinite values")
                self.metrics.feature_quality_scores['inf_rate'] = np.mean(np.isinf(vector))
            
            # Check value ranges
            if np.any(np.abs(vector) > 10):
                logger.warning("Context vector contains extreme values",
                             max_abs=np.max(np.abs(vector)))
            
            # Calculate quality score
            quality_score = 1.0
            quality_score -= np.mean(np.isnan(vector)) * 0.5  # Penalize NaN values
            quality_score -= np.mean(np.isinf(vector)) * 0.5  # Penalize infinite values
            quality_score -= min(0.3, np.mean(np.abs(vector) > 5) * 0.3)  # Penalize extreme values
            
            self.metrics.feature_quality_scores['overall'] = quality_score
            
        except Exception as e:
            logger.error("Vector validation failed", error=str(e))
    
    def _create_fallback_vector(self) -> torch.Tensor:
        """Create safe fallback vector when processing fails"""
        # Return neutral/safe values for all features
        fallback = np.array([
            1.0,    # Portfolio value (normalized)
            1.0,    # Available capital ratio
            0.0,    # Current position
            0.0,    # Unrealized PnL
            0.02,   # VaR estimate
            0.0,    # Expected return
            0.15,   # Volatility
            0.001,  # Bid-ask spread
            1.0,    # Volume ratio
            0.0,    # Order flow imbalance
            0.0,    # Tick direction
            0.0,    # Time since last trade
            0.0,    # Position holding time
            0.0,    # Sharpe ratio
            0.0     # Current drawdown
        ])
        
        return torch.tensor(fallback, dtype=torch.float32)
    
    def _update_processing_metrics(self, processing_time_us: float):
        """Update processing performance metrics"""
        self.metrics.total_processes += 1
        self.processing_times.append(processing_time_us)
        
        # Update timing statistics
        times = list(self.processing_times)
        self.metrics.avg_processing_time_us = np.mean(times)
        self.metrics.p95_processing_time_us = np.percentile(times, 95)
        self.metrics.p99_processing_time_us = np.percentile(times, 99)
    
    def _update_error_metrics(self):
        """Update error tracking metrics"""
        self.metrics.error_rate = self.metrics.processing_errors / max(1, self.metrics.total_processes)
    
    def get_processing_report(self) -> Dict[str, Any]:
        """Get comprehensive processing performance report"""
        return {
            'processing_metrics': {
                'total_processes': self.metrics.total_processes,
                'avg_processing_time_us': self.metrics.avg_processing_time_us,
                'p95_processing_time_us': self.metrics.p95_processing_time_us,
                'p99_processing_time_us': self.metrics.p99_processing_time_us,
                'processing_errors': self.metrics.processing_errors,
                'error_rate': self.metrics.error_rate
            },
            'feature_quality': self.metrics.feature_quality_scores,
            'performance_requirements': {
                'latency_target_met': self.metrics.p95_processing_time_us < 50.0,  # <50μs target
                'error_rate_acceptable': self.metrics.error_rate < 0.01,  # <1% error rate
                'quality_acceptable': self.metrics.feature_quality_scores.get('overall', 0.0) > 0.9
            },
            'running_statistics': self.running_stats,
            'data_quality': {
                'price_history_length': len(self.price_history),
                'volume_history_length': len(self.volume_history),
                'volatility_history_length': len(self.volatility_history)
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown of processor"""
        logger.info("Shutting down ExecutionContextProcessor")
        self.executor.shutdown(wait=True)
        logger.info("ExecutionContextProcessor shutdown complete")


def create_execution_context_processor(config: Dict[str, Any]) -> ExecutionContextProcessor:
    """Factory function to create execution context processor"""
    return ExecutionContextProcessor(config)


# Default configuration
DEFAULT_PROCESSOR_CONFIG = {
    'lookback_periods': 50,
    'normalization_window': 252,
    'feature_smoothing': 0.1,
    'max_workers': 4
}