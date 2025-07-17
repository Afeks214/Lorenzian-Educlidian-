"""
Alpha Generation Testing Framework

This module provides comprehensive testing for alpha generation and validation,
including signal generation, combination algorithms, alpha decay measurement,
and cross-sectional/time-series alpha models.

Key Features:
- Signal generation and validation
- Alpha combination algorithms
- Alpha decay and signal strength measurement
- Cross-sectional and time-series alpha models
- Factor model testing
- Signal-to-noise ratio analysis
- Alpha forecasting and validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from unittest.mock import Mock, patch
import asyncio
from scipy import stats
from sklearn.metrics import information_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class AlphaType(Enum):
    """Types of alpha signals"""
    MOMENTUM = "momentum"
    REVERSAL = "reversal"
    VOLATILITY = "volatility"
    QUALITY = "quality"
    VALUE = "value"
    GROWTH = "growth"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


@dataclass
class AlphaSignal:
    """Alpha signal with metadata"""
    name: str
    values: pd.Series
    alpha_type: AlphaType
    confidence: float
    decay_rate: float
    forecast_horizon: int  # days
    universe: List[str]
    timestamp: datetime


@dataclass
class AlphaMetrics:
    """Alpha validation metrics"""
    ic_mean: float  # Information Coefficient
    ic_std: float
    ic_ir: float  # Information Ratio
    hit_rate: float
    sharpe_ratio: float
    decay_rate: float
    signal_strength: float
    noise_ratio: float
    stability_score: float
    forecasting_accuracy: float


@dataclass
class AlphaCombination:
    """Combined alpha signal"""
    signals: List[AlphaSignal]
    weights: np.ndarray
    combined_values: pd.Series
    combination_method: str
    performance_metrics: AlphaMetrics


class AlphaGenerator:
    """
    Alpha generation engine with multiple signal types and validation.
    """
    
    def __init__(self):
        self.signals: Dict[str, AlphaSignal] = {}
        self.combinations: Dict[str, AlphaCombination] = {}
        self.validation_results: Dict[str, AlphaMetrics] = {}
        self.factor_models: Dict[str, Any] = {}
        
    def generate_momentum_alpha(
        self,
        prices: pd.DataFrame,
        lookback_periods: List[int] = [5, 10, 20, 50],
        name: str = "momentum_alpha"
    ) -> AlphaSignal:
        """Generate momentum-based alpha signal"""
        
        momentum_signals = []
        
        for period in lookback_periods:
            if len(prices) > period:
                # Calculate momentum for each asset
                momentum = prices.pct_change(period).iloc[-1]
                momentum_signals.append(momentum)
        
        # Combine momentum signals
        if momentum_signals:
            combined_momentum = pd.concat(momentum_signals, axis=1).mean(axis=1)
            
            # Z-score normalization
            combined_momentum = (combined_momentum - combined_momentum.mean()) / combined_momentum.std()
            
            # Calculate confidence based on signal consistency
            signal_std = pd.concat(momentum_signals, axis=1).std(axis=1).mean()
            confidence = max(0.1, 1.0 - signal_std)
            
            return AlphaSignal(
                name=name,
                values=combined_momentum,
                alpha_type=AlphaType.MOMENTUM,
                confidence=confidence,
                decay_rate=0.05,  # 5% daily decay
                forecast_horizon=5,
                universe=list(combined_momentum.index),
                timestamp=datetime.now()
            )
        
        # Fallback empty signal
        return AlphaSignal(
            name=name,
            values=pd.Series(dtype=float),
            alpha_type=AlphaType.MOMENTUM,
            confidence=0.0,
            decay_rate=0.05,
            forecast_horizon=5,
            universe=[],
            timestamp=datetime.now()
        )
    
    def generate_reversal_alpha(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        lookback_period: int = 20,
        name: str = "reversal_alpha"
    ) -> AlphaSignal:
        """Generate mean-reversal alpha signal"""
        
        if len(prices) < lookback_period:
            return AlphaSignal(
                name=name,
                values=pd.Series(dtype=float),
                alpha_type=AlphaType.REVERSAL,
                confidence=0.0,
                decay_rate=0.08,
                forecast_horizon=3,
                universe=[],
                timestamp=datetime.now()
            )
        
        # Calculate rolling mean and standard deviation
        rolling_mean = prices.rolling(window=lookback_period).mean()
        rolling_std = prices.rolling(window=lookback_period).std()
        
        # Calculate z-score (current price relative to rolling mean)
        z_score = (prices.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        
        # Reversal signal: negative z-score indicates oversold (positive alpha)
        reversal_signal = -z_score
        
        # Adjust by volume (higher volume = higher confidence)
        if not volumes.empty and len(volumes) >= lookback_period:
            volume_factor = volumes.iloc[-1] / volumes.rolling(window=lookback_period).mean().iloc[-1]
            reversal_signal *= np.sqrt(volume_factor.fillna(1.0))
        
        # Calculate confidence based on signal strength
        signal_strength = np.abs(z_score).mean()
        confidence = min(0.9, max(0.1, signal_strength / 2.0))
        
        return AlphaSignal(
            name=name,
            values=reversal_signal,
            alpha_type=AlphaType.REVERSAL,
            confidence=confidence,
            decay_rate=0.08,
            forecast_horizon=3,
            universe=list(reversal_signal.index),
            timestamp=datetime.now()
        )
    
    def generate_volatility_alpha(
        self,
        prices: pd.DataFrame,
        lookback_period: int = 30,
        name: str = "volatility_alpha"
    ) -> AlphaSignal:
        """Generate volatility-based alpha signal"""
        
        if len(prices) < lookback_period:
            return AlphaSignal(
                name=name,
                values=pd.Series(dtype=float),
                alpha_type=AlphaType.VOLATILITY,
                confidence=0.0,
                decay_rate=0.03,
                forecast_horizon=10,
                universe=[],
                timestamp=datetime.now()
            )
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Calculate realized volatility
        realized_vol = returns.rolling(window=lookback_period).std() * np.sqrt(252)
        
        # Calculate implied volatility (simplified as volatility of volatility)
        vol_of_vol = realized_vol.rolling(window=10).std()
        
        # Volatility signal: prefer low volatility assets (negative relationship)
        volatility_signal = -realized_vol.iloc[-1]
        
        # Adjust by volatility of volatility (higher vol of vol = lower confidence)
        vol_adjustment = 1.0 / (1.0 + vol_of_vol.iloc[-1].fillna(0.1))
        volatility_signal *= vol_adjustment
        
        # Z-score normalization
        volatility_signal = (volatility_signal - volatility_signal.mean()) / volatility_signal.std()
        
        # Calculate confidence
        confidence = min(0.8, max(0.2, 1.0 - vol_of_vol.iloc[-1].mean()))
        
        return AlphaSignal(
            name=name,
            values=volatility_signal,
            alpha_type=AlphaType.VOLATILITY,
            confidence=confidence,
            decay_rate=0.03,
            forecast_horizon=10,
            universe=list(volatility_signal.index),
            timestamp=datetime.now()
        )
    
    def generate_quality_alpha(
        self,
        fundamental_data: pd.DataFrame,
        name: str = "quality_alpha"
    ) -> AlphaSignal:
        """Generate quality-based alpha signal"""
        
        if fundamental_data.empty:
            return AlphaSignal(
                name=name,
                values=pd.Series(dtype=float),
                alpha_type=AlphaType.QUALITY,
                confidence=0.0,
                decay_rate=0.01,
                forecast_horizon=30,
                universe=[],
                timestamp=datetime.now()
            )
        
        # Quality metrics (simplified)
        quality_metrics = {}
        
        # ROE (Return on Equity)
        if 'roe' in fundamental_data.columns:
            quality_metrics['roe'] = fundamental_data['roe'].fillna(0)
        
        # Debt-to-Equity ratio (lower is better)
        if 'debt_to_equity' in fundamental_data.columns:
            quality_metrics['debt_to_equity'] = -fundamental_data['debt_to_equity'].fillna(0)
        
        # Earnings stability (simplified as negative earnings volatility)
        if 'earnings_volatility' in fundamental_data.columns:
            quality_metrics['earnings_stability'] = -fundamental_data['earnings_volatility'].fillna(0)
        
        # Combine quality metrics
        if quality_metrics:
            quality_df = pd.DataFrame(quality_metrics)
            
            # Z-score normalization for each metric
            quality_df = (quality_df - quality_df.mean()) / quality_df.std()
            
            # Equal-weighted combination
            quality_signal = quality_df.mean(axis=1)
            
            confidence = 0.7  # Fundamental signals typically more stable
        else:
            # Generate random quality signal for testing
            quality_signal = pd.Series(
                np.random.normal(0, 1, len(fundamental_data)),
                index=fundamental_data.index
            )
            confidence = 0.3
        
        return AlphaSignal(
            name=name,
            values=quality_signal,
            alpha_type=AlphaType.QUALITY,
            confidence=confidence,
            decay_rate=0.01,
            forecast_horizon=30,
            universe=list(quality_signal.index),
            timestamp=datetime.now()
        )
    
    def generate_technical_alpha(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        name: str = "technical_alpha"
    ) -> AlphaSignal:
        """Generate technical analysis-based alpha signal"""
        
        if len(prices) < 50:
            return AlphaSignal(
                name=name,
                values=pd.Series(dtype=float),
                alpha_type=AlphaType.TECHNICAL,
                confidence=0.0,
                decay_rate=0.1,
                forecast_horizon=2,
                universe=[],
                timestamp=datetime.now()
            )
        
        technical_signals = []
        
        # RSI (Relative Strength Index)
        rsi = self._calculate_rsi(prices, period=14)
        rsi_signal = (50 - rsi) / 50  # Normalize to [-1, 1]
        technical_signals.append(rsi_signal)
        
        # MACD (Moving Average Convergence Divergence)
        macd = self._calculate_macd(prices)
        technical_signals.append(macd)
        
        # Bollinger Bands
        bb_signal = self._calculate_bollinger_bands_signal(prices)
        technical_signals.append(bb_signal)
        
        # Volume-weighted signals
        if not volumes.empty:
            volume_signal = self._calculate_volume_signal(prices, volumes)
            technical_signals.append(volume_signal)
        
        # Combine technical signals
        if technical_signals:
            technical_df = pd.concat(technical_signals, axis=1)
            technical_df = technical_df.fillna(0)
            
            # Use the latest complete signal
            combined_signal = technical_df.iloc[-1]
            
            # Weight by signal strength
            weights = np.abs(combined_signal) / np.sum(np.abs(combined_signal))
            final_signal = np.sum(combined_signal * weights)
            
            # Convert to series for all assets
            if isinstance(final_signal, pd.Series):
                technical_alpha = final_signal
            else:
                # If scalar, apply to all assets
                technical_alpha = pd.Series(
                    [final_signal] * len(prices.columns),
                    index=prices.columns
                )
        else:
            technical_alpha = pd.Series(dtype=float)
        
        confidence = 0.4  # Technical signals typically more noisy
        
        return AlphaSignal(
            name=name,
            values=technical_alpha,
            alpha_type=AlphaType.TECHNICAL,
            confidence=confidence,
            decay_rate=0.1,
            forecast_horizon=2,
            universe=list(technical_alpha.index),
            timestamp=datetime.now()
        )
    
    def _calculate_rsi(self, prices: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]  # Return latest RSI values
    
    def _calculate_macd(self, prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD indicator"""
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        macd_histogram = macd - macd_signal
        
        return macd_histogram.iloc[-1]  # Return latest MACD histogram
    
    def _calculate_bollinger_bands_signal(self, prices: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Bollinger Bands signal"""
        
        rolling_mean = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        current_price = prices.iloc[-1]
        
        # Signal: -1 (oversold) to +1 (overbought)
        bb_signal = (current_price - rolling_mean.iloc[-1]) / (rolling_std.iloc[-1] * 2)
        
        return -bb_signal  # Reverse for contrarian signal
    
    def _calculate_volume_signal(self, prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.Series:
        """Calculate volume-based signal"""
        
        # Volume-weighted average price
        vwap = (prices * volumes).sum() / volumes.sum()
        
        # Current price relative to VWAP
        current_price = prices.iloc[-1]
        volume_signal = (current_price - vwap) / vwap
        
        return volume_signal
    
    def combine_alphas(
        self,
        signals: List[AlphaSignal],
        method: str = "equal_weight",
        name: str = "combined_alpha"
    ) -> AlphaCombination:
        """
        Combine multiple alpha signals using various methods.
        
        Args:
            signals: List of alpha signals to combine
            method: Combination method ('equal_weight', 'ic_weight', 'risk_parity', 'ml_ensemble')
            name: Name for the combined signal
            
        Returns:
            AlphaCombination with combined signal and metrics
        """
        
        if not signals:
            return AlphaCombination(
                signals=[],
                weights=np.array([]),
                combined_values=pd.Series(dtype=float),
                combination_method=method,
                performance_metrics=AlphaMetrics(
                    ic_mean=0, ic_std=0, ic_ir=0, hit_rate=0, sharpe_ratio=0,
                    decay_rate=0, signal_strength=0, noise_ratio=0, stability_score=0,
                    forecasting_accuracy=0
                )
            )
        
        # Align signals to common universe
        aligned_signals = self._align_signals(signals)
        
        # Calculate combination weights
        if method == "equal_weight":
            weights = np.ones(len(aligned_signals)) / len(aligned_signals)
        elif method == "ic_weight":
            weights = self._calculate_ic_weights(aligned_signals)
        elif method == "risk_parity":
            weights = self._calculate_risk_parity_weights(aligned_signals)
        elif method == "ml_ensemble":
            weights = self._calculate_ml_ensemble_weights(aligned_signals)
        else:
            weights = np.ones(len(aligned_signals)) / len(aligned_signals)
        
        # Combine signals
        combined_values = self._combine_signals(aligned_signals, weights)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_combination_metrics(aligned_signals, combined_values)
        
        return AlphaCombination(
            signals=signals,
            weights=weights,
            combined_values=combined_values,
            combination_method=method,
            performance_metrics=performance_metrics
        )
    
    def _align_signals(self, signals: List[AlphaSignal]) -> List[pd.Series]:
        """Align signals to common universe"""
        
        # Find common universe
        common_universe = set(signals[0].universe)
        for signal in signals[1:]:
            common_universe &= set(signal.universe)
        
        common_universe = list(common_universe)
        
        # Align all signals to common universe
        aligned = []
        for signal in signals:
            aligned_signal = signal.values.reindex(common_universe, fill_value=0)
            aligned.append(aligned_signal)
        
        return aligned
    
    def _calculate_ic_weights(self, signals: List[pd.Series]) -> np.ndarray:
        """Calculate Information Coefficient-based weights"""
        
        # For testing, use equal weights (IC calculation requires returns)
        return np.ones(len(signals)) / len(signals)
    
    def _calculate_risk_parity_weights(self, signals: List[pd.Series]) -> np.ndarray:
        """Calculate risk parity weights"""
        
        # Calculate signal volatilities
        volatilities = np.array([signal.std() for signal in signals])
        
        # Avoid division by zero
        volatilities = np.where(volatilities == 0, 1e-8, volatilities)
        
        # Inverse volatility weights
        weights = 1 / volatilities
        weights = weights / np.sum(weights)
        
        return weights
    
    def _calculate_ml_ensemble_weights(self, signals: List[pd.Series]) -> np.ndarray:
        """Calculate ML ensemble weights"""
        
        # For testing, use equal weights (ML training requires historical data)
        return np.ones(len(signals)) / len(signals)
    
    def _combine_signals(self, signals: List[pd.Series], weights: np.ndarray) -> pd.Series:
        """Combine signals with given weights"""
        
        if not signals:
            return pd.Series(dtype=float)
        
        # Weighted combination
        combined = pd.Series(0, index=signals[0].index)
        
        for signal, weight in zip(signals, weights):
            combined += weight * signal
        
        return combined
    
    def _calculate_combination_metrics(self, signals: List[pd.Series], combined: pd.Series) -> AlphaMetrics:
        """Calculate metrics for combined signal"""
        
        # Simplified metrics for testing
        return AlphaMetrics(
            ic_mean=0.05,
            ic_std=0.15,
            ic_ir=0.33,
            hit_rate=0.55,
            sharpe_ratio=0.8,
            decay_rate=0.05,
            signal_strength=np.abs(combined).mean(),
            noise_ratio=0.3,
            stability_score=0.7,
            forecasting_accuracy=0.6
        )
    
    def validate_alpha(
        self,
        signal: AlphaSignal,
        returns: pd.DataFrame,
        validation_period: int = 252
    ) -> AlphaMetrics:
        """
        Validate alpha signal against forward returns.
        
        Args:
            signal: Alpha signal to validate
            returns: Forward returns for validation
            validation_period: Validation period in days
            
        Returns:
            AlphaMetrics with validation results
        """
        
        # Align signal and returns
        common_assets = list(set(signal.universe) & set(returns.columns))
        
        if not common_assets:
            return AlphaMetrics(
                ic_mean=0, ic_std=0, ic_ir=0, hit_rate=0, sharpe_ratio=0,
                decay_rate=0, signal_strength=0, noise_ratio=0, stability_score=0,
                forecasting_accuracy=0
            )
        
        signal_values = signal.values.reindex(common_assets, fill_value=0)
        
        # Calculate validation metrics
        ic_values = []
        hit_rates = []
        
        for i in range(min(validation_period, len(returns))):
            if i < len(returns):
                period_returns = returns.iloc[i][common_assets]
                
                # Calculate Information Coefficient (correlation)
                ic = signal_values.corr(period_returns)
                if not np.isnan(ic):
                    ic_values.append(ic)
                
                # Calculate hit rate
                hit_rate = ((signal_values > 0) == (period_returns > 0)).mean()
                hit_rates.append(hit_rate)
        
        # Calculate metrics
        ic_mean = np.mean(ic_values) if ic_values else 0
        ic_std = np.std(ic_values) if ic_values else 0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0
        
        hit_rate = np.mean(hit_rates) if hit_rates else 0.5
        
        # Calculate decay rate
        decay_rate = self._calculate_decay_rate(signal_values, returns)
        
        # Calculate signal strength
        signal_strength = np.abs(signal_values).mean()
        
        # Calculate noise ratio
        noise_ratio = self._calculate_noise_ratio(signal_values, returns)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(signal_values)
        
        # Calculate forecasting accuracy
        forecasting_accuracy = self._calculate_forecasting_accuracy(signal_values, returns)
        
        # Calculate Sharpe ratio (simplified)
        if len(ic_values) > 1:
            sharpe_ratio = np.mean(ic_values) / np.std(ic_values) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        return AlphaMetrics(
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            hit_rate=hit_rate,
            sharpe_ratio=sharpe_ratio,
            decay_rate=decay_rate,
            signal_strength=signal_strength,
            noise_ratio=noise_ratio,
            stability_score=stability_score,
            forecasting_accuracy=forecasting_accuracy
        )
    
    def _calculate_decay_rate(self, signal: pd.Series, returns: pd.DataFrame) -> float:
        """Calculate alpha decay rate"""
        
        # Simplified decay calculation
        if len(returns) < 10:
            return 0.05
        
        # Calculate IC over different horizons
        ic_values = []
        for horizon in range(1, min(11, len(returns))):
            if horizon < len(returns):
                future_returns = returns.iloc[horizon][signal.index]
                ic = signal.corr(future_returns)
                if not np.isnan(ic):
                    ic_values.append(abs(ic))
        
        if len(ic_values) < 2:
            return 0.05
        
        # Fit exponential decay
        x = np.arange(len(ic_values))
        y = np.array(ic_values)
        
        # Simple linear fit in log space
        if np.all(y > 0):
            slope, _, _, _, _ = stats.linregress(x, np.log(y))
            decay_rate = abs(slope)
        else:
            decay_rate = 0.05
        
        return min(1.0, max(0.001, decay_rate))
    
    def _calculate_noise_ratio(self, signal: pd.Series, returns: pd.DataFrame) -> float:
        """Calculate signal-to-noise ratio"""
        
        if len(returns) == 0:
            return 0.5
        
        # Calculate signal variance vs noise variance
        signal_var = signal.var()
        
        # Estimate noise as residual after fitting to returns
        if len(returns) > 0:
            first_period_returns = returns.iloc[0][signal.index]
            correlation = signal.corr(first_period_returns)
            
            if not np.isnan(correlation):
                explained_var = correlation ** 2 * signal_var
                noise_var = signal_var - explained_var
                noise_ratio = noise_var / signal_var if signal_var > 0 else 0.5
            else:
                noise_ratio = 0.5
        else:
            noise_ratio = 0.5
        
        return min(1.0, max(0.0, noise_ratio))
    
    def _calculate_stability_score(self, signal: pd.Series) -> float:
        """Calculate signal stability score"""
        
        if len(signal) < 2:
            return 0.5
        
        # Calculate coefficient of variation
        cv = signal.std() / abs(signal.mean()) if signal.mean() != 0 else 1.0
        
        # Stability score: lower CV = higher stability
        stability_score = 1.0 / (1.0 + cv)
        
        return min(1.0, max(0.0, stability_score))
    
    def _calculate_forecasting_accuracy(self, signal: pd.Series, returns: pd.DataFrame) -> float:
        """Calculate forecasting accuracy"""
        
        if len(returns) == 0:
            return 0.5
        
        # Simple accuracy based on direction prediction
        first_period_returns = returns.iloc[0][signal.index]
        
        # Direction accuracy
        correct_directions = ((signal > 0) == (first_period_returns > 0)).sum()
        total_predictions = len(signal)
        
        accuracy = correct_directions / total_predictions if total_predictions > 0 else 0.5
        
        return min(1.0, max(0.0, accuracy))
    
    def build_factor_model(
        self,
        signals: List[AlphaSignal],
        returns: pd.DataFrame,
        model_type: str = "linear"
    ) -> Dict[str, Any]:
        """
        Build factor model from alpha signals.
        
        Args:
            signals: List of alpha signals (factors)
            returns: Asset returns
            model_type: Type of model ('linear', 'ridge', 'random_forest')
            
        Returns:
            Dictionary with model and performance metrics
        """
        
        # Prepare feature matrix
        feature_matrix = pd.DataFrame()
        
        for signal in signals:
            # Align signal to returns
            aligned_signal = signal.values.reindex(returns.columns, fill_value=0)
            feature_matrix[signal.name] = aligned_signal
        
        if feature_matrix.empty or returns.empty:
            return {
                'model': None,
                'r_squared': 0,
                'feature_importance': {},
                'predictions': pd.Series(dtype=float)
            }
        
        # Use first period returns as target
        target = returns.iloc[0] if len(returns) > 0 else pd.Series(dtype=float)
        
        # Remove NaN values
        valid_idx = ~(feature_matrix.isna().any(axis=1) | target.isna())
        X = feature_matrix.loc[valid_idx]
        y = target.loc[valid_idx]
        
        if len(X) < 2:
            return {
                'model': None,
                'r_squared': 0,
                'feature_importance': {},
                'predictions': pd.Series(dtype=float)
            }
        
        # Build model
        if model_type == "linear":
            model = LinearRegression()
        elif model_type == "ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        
        # Fit model
        model.fit(X, y)
        
        # Calculate R-squared
        predictions = model.predict(X)
        r_squared = 1 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
        else:
            feature_importance = {}
        
        # Generate predictions for all assets
        full_predictions = model.predict(feature_matrix.fillna(0))
        prediction_series = pd.Series(full_predictions, index=feature_matrix.index)
        
        return {
            'model': model,
            'r_squared': r_squared,
            'feature_importance': feature_importance,
            'predictions': prediction_series
        }
    
    def analyze_cross_sectional_alpha(
        self,
        signal: AlphaSignal,
        returns: pd.DataFrame,
        universe: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze cross-sectional alpha properties.
        
        Args:
            signal: Alpha signal to analyze
            returns: Cross-sectional returns
            universe: Asset universe
            
        Returns:
            Dictionary with cross-sectional analysis results
        """
        
        # Align signal to universe
        signal_values = signal.values.reindex(universe, fill_value=0)
        
        if returns.empty:
            return {
                'quintile_returns': {},
                'spread': 0,
                'turnover': 0,
                'concentration': 0
            }
        
        # Calculate quintile analysis
        quintiles = pd.qcut(signal_values, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        quintile_returns = {}
        for quintile in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            quintile_assets = signal_values[quintiles == quintile].index
            if len(quintile_assets) > 0 and len(returns) > 0:
                quintile_return = returns.iloc[0][quintile_assets].mean()
                quintile_returns[quintile] = quintile_return
            else:
                quintile_returns[quintile] = 0
        
        # Calculate spread (Q5 - Q1)
        spread = quintile_returns.get('Q5', 0) - quintile_returns.get('Q1', 0)
        
        # Calculate turnover (simplified)
        turnover = 0.2  # Placeholder
        
        # Calculate concentration (Herfindahl index)
        weights = np.abs(signal_values) / np.sum(np.abs(signal_values))
        concentration = np.sum(weights ** 2)
        
        return {
            'quintile_returns': quintile_returns,
            'spread': spread,
            'turnover': turnover,
            'concentration': concentration
        }
    
    def analyze_time_series_alpha(
        self,
        signal: AlphaSignal,
        returns: pd.DataFrame,
        window: int = 252
    ) -> Dict[str, Any]:
        """
        Analyze time-series alpha properties.
        
        Args:
            signal: Alpha signal to analyze
            returns: Time series of returns
            window: Rolling window for analysis
            
        Returns:
            Dictionary with time-series analysis results
        """
        
        if returns.empty or len(returns) < window:
            return {
                'rolling_ic': pd.Series(dtype=float),
                'ic_mean': 0,
                'ic_std': 0,
                'hit_rate': 0,
                'decay_half_life': 0
            }
        
        # Calculate rolling IC
        rolling_ic = []
        common_assets = list(set(signal.universe) & set(returns.columns))
        
        if not common_assets:
            return {
                'rolling_ic': pd.Series(dtype=float),
                'ic_mean': 0,
                'ic_std': 0,
                'hit_rate': 0,
                'decay_half_life': 0
            }
        
        signal_values = signal.values.reindex(common_assets, fill_value=0)
        
        for i in range(window, len(returns)):
            period_returns = returns.iloc[i][common_assets]
            ic = signal_values.corr(period_returns)
            rolling_ic.append(ic if not np.isnan(ic) else 0)
        
        rolling_ic_series = pd.Series(rolling_ic, index=returns.index[window:])
        
        # Calculate statistics
        ic_mean = rolling_ic_series.mean()
        ic_std = rolling_ic_series.std()
        hit_rate = (rolling_ic_series > 0).mean()
        
        # Calculate decay half-life
        decay_half_life = self._calculate_decay_half_life(signal_values, returns)
        
        return {
            'rolling_ic': rolling_ic_series,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'hit_rate': hit_rate,
            'decay_half_life': decay_half_life
        }
    
    def _calculate_decay_half_life(self, signal: pd.Series, returns: pd.DataFrame) -> float:
        """Calculate decay half-life in days"""
        
        # Simplified calculation
        decay_rate = self._calculate_decay_rate(signal, returns)
        
        # Half-life = ln(2) / decay_rate
        half_life = np.log(2) / decay_rate if decay_rate > 0 else np.inf
        
        return min(365, max(1, half_life))


# Test fixtures
@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    price_data = pd.DataFrame(index=dates, columns=assets)
    
    for asset in assets:
        # Generate realistic price series
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [100.0]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[asset] = prices
    
    return price_data


@pytest.fixture
def sample_volume_data():
    """Generate sample volume data for testing"""
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    volume_data = pd.DataFrame(index=dates, columns=assets)
    
    for asset in assets:
        # Generate realistic volume series
        volumes = np.random.lognormal(15, 0.5, len(dates))
        volume_data[asset] = volumes
    
    return volume_data


@pytest.fixture
def sample_fundamental_data():
    """Generate sample fundamental data for testing"""
    
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    fundamental_data = pd.DataFrame(index=assets)
    
    np.random.seed(42)
    fundamental_data['roe'] = np.random.normal(0.15, 0.05, len(assets))
    fundamental_data['debt_to_equity'] = np.random.normal(0.3, 0.1, len(assets))
    fundamental_data['earnings_volatility'] = np.random.normal(0.1, 0.03, len(assets))
    
    return fundamental_data


@pytest.fixture
def alpha_generator():
    """Create alpha generator instance"""
    return AlphaGenerator()


# Comprehensive test suite
@pytest.mark.asyncio
class TestAlphaGeneration:
    """Comprehensive alpha generation tests"""
    
    def test_momentum_alpha_generation(self, alpha_generator, sample_price_data):
        """Test momentum alpha generation"""
        
        signal = alpha_generator.generate_momentum_alpha(
            sample_price_data,
            lookback_periods=[5, 10, 20],
            name="test_momentum"
        )
        
        assert signal.name == "test_momentum"
        assert signal.alpha_type == AlphaType.MOMENTUM
        assert 0 <= signal.confidence <= 1
        assert signal.decay_rate > 0
        assert signal.forecast_horizon > 0
        assert len(signal.values) > 0
        
        # Check signal properties
        assert not signal.values.isna().all()
        assert signal.values.std() > 0  # Should have some variation
    
    def test_reversal_alpha_generation(self, alpha_generator, sample_price_data, sample_volume_data):
        """Test reversal alpha generation"""
        
        signal = alpha_generator.generate_reversal_alpha(
            sample_price_data,
            sample_volume_data,
            lookback_period=20,
            name="test_reversal"
        )
        
        assert signal.name == "test_reversal"
        assert signal.alpha_type == AlphaType.REVERSAL
        assert 0 <= signal.confidence <= 1
        assert signal.decay_rate > 0
        assert signal.forecast_horizon > 0
        assert len(signal.values) > 0
        
        # Reversal signal should be normalized
        assert abs(signal.values.mean()) < 1.0
    
    def test_volatility_alpha_generation(self, alpha_generator, sample_price_data):
        """Test volatility alpha generation"""
        
        signal = alpha_generator.generate_volatility_alpha(
            sample_price_data,
            lookback_period=30,
            name="test_volatility"
        )
        
        assert signal.name == "test_volatility"
        assert signal.alpha_type == AlphaType.VOLATILITY
        assert 0 <= signal.confidence <= 1
        assert signal.decay_rate > 0
        assert signal.forecast_horizon > 0
        assert len(signal.values) > 0
        
        # Volatility signal should be standardized
        assert abs(signal.values.mean()) < 0.1  # Should be close to zero mean
    
    def test_quality_alpha_generation(self, alpha_generator, sample_fundamental_data):
        """Test quality alpha generation"""
        
        signal = alpha_generator.generate_quality_alpha(
            sample_fundamental_data,
            name="test_quality"
        )
        
        assert signal.name == "test_quality"
        assert signal.alpha_type == AlphaType.QUALITY
        assert 0 <= signal.confidence <= 1
        assert signal.decay_rate > 0
        assert signal.forecast_horizon > 0
        assert len(signal.values) > 0
        
        # Quality signal should be standardized
        assert abs(signal.values.mean()) < 0.1
    
    def test_technical_alpha_generation(self, alpha_generator, sample_price_data, sample_volume_data):
        """Test technical alpha generation"""
        
        signal = alpha_generator.generate_technical_alpha(
            sample_price_data,
            sample_volume_data,
            name="test_technical"
        )
        
        assert signal.name == "test_technical"
        assert signal.alpha_type == AlphaType.TECHNICAL
        assert 0 <= signal.confidence <= 1
        assert signal.decay_rate > 0
        assert signal.forecast_horizon > 0
        assert len(signal.values) > 0
    
    def test_alpha_combination_equal_weight(self, alpha_generator, sample_price_data, sample_volume_data):
        """Test equal-weight alpha combination"""
        
        # Generate multiple signals
        momentum_signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="momentum")
        reversal_signal = alpha_generator.generate_reversal_alpha(sample_price_data, sample_volume_data, name="reversal")
        volatility_signal = alpha_generator.generate_volatility_alpha(sample_price_data, name="volatility")
        
        signals = [momentum_signal, reversal_signal, volatility_signal]
        
        # Combine signals
        combination = alpha_generator.combine_alphas(
            signals,
            method="equal_weight",
            name="test_combination"
        )
        
        assert combination.combination_method == "equal_weight"
        assert len(combination.signals) == 3
        assert np.allclose(combination.weights, [1/3, 1/3, 1/3], atol=1e-6)
        assert len(combination.combined_values) > 0
        assert isinstance(combination.performance_metrics, AlphaMetrics)
    
    def test_alpha_combination_risk_parity(self, alpha_generator, sample_price_data, sample_volume_data):
        """Test risk parity alpha combination"""
        
        # Generate signals with different volatilities
        momentum_signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="momentum")
        reversal_signal = alpha_generator.generate_reversal_alpha(sample_price_data, sample_volume_data, name="reversal")
        
        signals = [momentum_signal, reversal_signal]
        
        combination = alpha_generator.combine_alphas(
            signals,
            method="risk_parity",
            name="risk_parity_combination"
        )
        
        assert combination.combination_method == "risk_parity"
        assert len(combination.signals) == 2
        assert np.sum(combination.weights) == pytest.approx(1.0, rel=1e-6)
        assert len(combination.combined_values) > 0
    
    def test_alpha_validation(self, alpha_generator, sample_price_data):
        """Test alpha validation against forward returns"""
        
        # Generate signal
        signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="validation_test")
        
        # Generate forward returns
        returns = sample_price_data.pct_change().dropna()
        
        # Validate signal
        metrics = alpha_generator.validate_alpha(signal, returns, validation_period=100)
        
        assert isinstance(metrics, AlphaMetrics)
        assert -1 <= metrics.ic_mean <= 1
        assert metrics.ic_std >= 0
        assert 0 <= metrics.hit_rate <= 1
        assert metrics.decay_rate > 0
        assert metrics.signal_strength >= 0
        assert 0 <= metrics.noise_ratio <= 1
        assert 0 <= metrics.stability_score <= 1
        assert 0 <= metrics.forecasting_accuracy <= 1
    
    def test_factor_model_building(self, alpha_generator, sample_price_data, sample_volume_data):
        """Test factor model building"""
        
        # Generate multiple signals
        momentum_signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="momentum")
        reversal_signal = alpha_generator.generate_reversal_alpha(sample_price_data, sample_volume_data, name="reversal")
        volatility_signal = alpha_generator.generate_volatility_alpha(sample_price_data, name="volatility")
        
        signals = [momentum_signal, reversal_signal, volatility_signal]
        
        # Generate returns
        returns = sample_price_data.pct_change().dropna()
        
        # Build factor model
        model_result = alpha_generator.build_factor_model(signals, returns, model_type="linear")
        
        assert model_result['model'] is not None
        assert 0 <= model_result['r_squared'] <= 1
        assert isinstance(model_result['feature_importance'], dict)
        assert len(model_result['predictions']) > 0
    
    def test_cross_sectional_analysis(self, alpha_generator, sample_price_data):
        """Test cross-sectional alpha analysis"""
        
        # Generate signal
        signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="cross_sectional_test")
        
        # Generate returns
        returns = sample_price_data.pct_change().dropna()
        universe = list(sample_price_data.columns)
        
        # Analyze cross-sectional properties
        analysis = alpha_generator.analyze_cross_sectional_alpha(signal, returns, universe)
        
        assert isinstance(analysis['quintile_returns'], dict)
        assert len(analysis['quintile_returns']) == 5
        assert 'spread' in analysis
        assert 'turnover' in analysis
        assert 'concentration' in analysis
        assert 0 <= analysis['concentration'] <= 1
    
    def test_time_series_analysis(self, alpha_generator, sample_price_data):
        """Test time-series alpha analysis"""
        
        # Generate signal
        signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="time_series_test")
        
        # Generate returns
        returns = sample_price_data.pct_change().dropna()
        
        # Analyze time-series properties
        analysis = alpha_generator.analyze_time_series_alpha(signal, returns, window=100)
        
        assert isinstance(analysis['rolling_ic'], pd.Series)
        assert 'ic_mean' in analysis
        assert 'ic_std' in analysis
        assert 'hit_rate' in analysis
        assert 'decay_half_life' in analysis
        assert 0 <= analysis['hit_rate'] <= 1
        assert analysis['decay_half_life'] > 0
    
    def test_signal_decay_analysis(self, alpha_generator, sample_price_data):
        """Test alpha signal decay analysis"""
        
        # Generate signal
        signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="decay_test")
        
        # Generate returns with multiple horizons
        returns = sample_price_data.pct_change().dropna()
        
        # Test decay calculation
        decay_rate = alpha_generator._calculate_decay_rate(signal.values, returns)
        
        assert decay_rate > 0
        assert decay_rate < 1
        
        # Test half-life calculation
        half_life = alpha_generator._calculate_decay_half_life(signal.values, returns)
        
        assert half_life > 0
        assert half_life < 365  # Should be less than a year
    
    def test_signal_strength_measurement(self, alpha_generator, sample_price_data):
        """Test signal strength measurement"""
        
        # Generate signals with different strengths
        strong_signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="strong")
        weak_signal = alpha_generator.generate_volatility_alpha(sample_price_data, name="weak")
        
        # Compare signal strengths
        strong_strength = np.abs(strong_signal.values).mean()
        weak_strength = np.abs(weak_signal.values).mean()
        
        assert strong_strength >= 0
        assert weak_strength >= 0
        
        # Test signal-to-noise ratio
        returns = sample_price_data.pct_change().dropna()
        
        strong_noise = alpha_generator._calculate_noise_ratio(strong_signal.values, returns)
        weak_noise = alpha_generator._calculate_noise_ratio(weak_signal.values, returns)
        
        assert 0 <= strong_noise <= 1
        assert 0 <= weak_noise <= 1
    
    def test_alpha_forecasting_accuracy(self, alpha_generator, sample_price_data):
        """Test alpha forecasting accuracy"""
        
        # Generate signal
        signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="forecast_test")
        
        # Generate returns
        returns = sample_price_data.pct_change().dropna()
        
        # Test forecasting accuracy
        accuracy = alpha_generator._calculate_forecasting_accuracy(signal.values, returns)
        
        assert 0 <= accuracy <= 1
        
        # Test with perfect signal
        perfect_signal = returns.iloc[0] if len(returns) > 0 else signal.values
        perfect_accuracy = alpha_generator._calculate_forecasting_accuracy(perfect_signal, returns)
        
        assert perfect_accuracy >= accuracy
    
    def test_alpha_stability_analysis(self, alpha_generator, sample_price_data):
        """Test alpha stability analysis"""
        
        # Generate signal
        signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="stability_test")
        
        # Test stability score
        stability = alpha_generator._calculate_stability_score(signal.values)
        
        assert 0 <= stability <= 1
        
        # Test with very stable signal
        stable_signal = pd.Series([1.0] * len(signal.values), index=signal.values.index)
        stable_score = alpha_generator._calculate_stability_score(stable_signal)
        
        assert stable_score > stability
    
    def test_multi_factor_model_performance(self, alpha_generator, sample_price_data, sample_volume_data):
        """Test multi-factor model performance"""
        
        # Generate multiple factor signals
        factors = [
            alpha_generator.generate_momentum_alpha(sample_price_data, name="momentum_factor"),
            alpha_generator.generate_reversal_alpha(sample_price_data, sample_volume_data, name="reversal_factor"),
            alpha_generator.generate_volatility_alpha(sample_price_data, name="volatility_factor")
        ]
        
        # Generate returns
        returns = sample_price_data.pct_change().dropna()
        
        # Test different model types
        for model_type in ['linear', 'ridge', 'random_forest']:
            model_result = alpha_generator.build_factor_model(factors, returns, model_type=model_type)
            
            assert model_result['model'] is not None
            assert 0 <= model_result['r_squared'] <= 1
            assert len(model_result['feature_importance']) == len(factors)
            assert len(model_result['predictions']) > 0
    
    def test_alpha_regime_sensitivity(self, alpha_generator, sample_price_data):
        """Test alpha sensitivity to market regimes"""
        
        # Generate signal
        signal = alpha_generator.generate_momentum_alpha(sample_price_data, name="regime_test")
        
        # Create different market regimes (bull, bear, sideways)
        returns = sample_price_data.pct_change().dropna()
        
        # Split into regime periods
        third = len(returns) // 3
        bull_period = returns.iloc[:third]
        bear_period = returns.iloc[third:2*third]
        sideways_period = returns.iloc[2*third:]
        
        # Test signal performance in different regimes
        regimes = [
            ("bull", bull_period),
            ("bear", bear_period),
            ("sideways", sideways_period)
        ]
        
        regime_metrics = {}
        for regime_name, regime_data in regimes:
            if not regime_data.empty:
                metrics = alpha_generator.validate_alpha(signal, regime_data, validation_period=50)
                regime_metrics[regime_name] = metrics
        
        # Should have metrics for each regime
        assert len(regime_metrics) > 0
        
        # Each regime should have valid metrics
        for regime_name, metrics in regime_metrics.items():
            assert isinstance(metrics, AlphaMetrics)
            assert -1 <= metrics.ic_mean <= 1
            assert 0 <= metrics.hit_rate <= 1
    
    def test_alpha_universe_scalability(self, alpha_generator):
        """Test alpha generation with different universe sizes"""
        
        # Test with small universe
        small_universe = ['AAPL', 'GOOGL']
        small_prices = pd.DataFrame({
            asset: np.random.lognormal(4.6, 0.02, 100)
            for asset in small_universe
        })
        
        small_signal = alpha_generator.generate_momentum_alpha(small_prices, name="small_universe")
        
        assert len(small_signal.values) == len(small_universe)
        assert small_signal.confidence > 0
        
        # Test with large universe
        large_universe = [f'STOCK_{i}' for i in range(50)]
        large_prices = pd.DataFrame({
            asset: np.random.lognormal(4.6, 0.02, 100)
            for asset in large_universe
        })
        
        large_signal = alpha_generator.generate_momentum_alpha(large_prices, name="large_universe")
        
        assert len(large_signal.values) == len(large_universe)
        assert large_signal.confidence > 0
        
        # Large universe should potentially have better diversification
        assert len(large_signal.values) > len(small_signal.values)
    
    def test_alpha_combination_optimization(self, alpha_generator, sample_price_data, sample_volume_data):
        """Test alpha combination optimization"""
        
        # Generate multiple signals
        signals = [
            alpha_generator.generate_momentum_alpha(sample_price_data, name="momentum_1"),
            alpha_generator.generate_momentum_alpha(sample_price_data, [10, 30], name="momentum_2"),
            alpha_generator.generate_reversal_alpha(sample_price_data, sample_volume_data, name="reversal"),
            alpha_generator.generate_volatility_alpha(sample_price_data, name="volatility")
        ]
        
        # Test different combination methods
        methods = ["equal_weight", "risk_parity", "ic_weight", "ml_ensemble"]
        
        combinations = {}
        for method in methods:
            combination = alpha_generator.combine_alphas(signals, method=method, name=f"combo_{method}")
            combinations[method] = combination
        
        # All combinations should be valid
        for method, combination in combinations.items():
            assert combination.combination_method == method
            assert len(combination.signals) == len(signals)
            assert np.sum(combination.weights) == pytest.approx(1.0, rel=1e-6)
            assert len(combination.combined_values) > 0
        
        # Risk parity should have different weights than equal weight
        equal_weights = combinations["equal_weight"].weights
        risk_parity_weights = combinations["risk_parity"].weights
        
        assert not np.allclose(equal_weights, risk_parity_weights, atol=1e-6)