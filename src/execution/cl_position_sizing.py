"""
CL Position Sizing Algorithm
===========================

Advanced position sizing algorithm specifically designed for CL crude oil trading.
Implements dynamic sizing based on volatility, market conditions, and risk parameters.

Key Features:
- ATR-based volatility adjustment
- Kelly Criterion optimization for commodities
- Market condition-based sizing
- Correlation adjustments
- Dynamic risk management
- Session-based liquidity adjustments

Author: Agent 4 - Risk Management Mission  
Date: 2025-07-17
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    """Position sizing methods"""
    VOLATILITY_BASED = "volatility_based"
    KELLY_CRITERION = "kelly_criterion"
    FIXED_FRACTIONAL = "fixed_fractional"
    RISK_PARITY = "risk_parity"
    ADAPTIVE = "adaptive"

@dataclass
class SizingParameters:
    """Position sizing parameters"""
    base_risk_percent: float = 0.02  # 2% base risk
    max_position_size: float = 0.10  # 10% max position
    min_position_size: float = 0.005  # 0.5% min position
    volatility_lookback: int = 20
    kelly_lookback: int = 100
    confidence_threshold: float = 0.6
    safety_factor: float = 0.5
    correlation_threshold: float = 0.7
    
class CLPositionSizer:
    """
    Advanced position sizing for CL crude oil trading
    
    Implements multiple sizing algorithms optimized for commodity characteristics
    including volatility clustering, inventory impacts, and session liquidity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CL Position Sizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.params = SizingParameters(**config.get('sizing_parameters', {}))
        
        # CL-specific parameters
        self.cl_contract_size = config.get('cl_contract_size', 1000)  # barrels
        self.cl_tick_size = config.get('cl_tick_size', 0.01)
        self.cl_tick_value = config.get('cl_tick_value', 10.0)
        
        # Market condition adjustments
        self.condition_multipliers = config.get('market_condition_adjustments', {
            'trending': 1.2,
            'ranging': 0.8,
            'high_volatility': 0.7,
            'low_volatility': 1.1,
            'geopolitical_risk': 0.6,
            'inventory_shock': 0.5
        })
        
        # Session adjustments
        self.session_multipliers = config.get('session_adjustments', {
            'us_session': 1.0,
            'european_session': 0.8,
            'asian_session': 0.6,
            'overnight': 0.5
        })
        
        # Historical data for sizing
        self.price_history = []
        self.trade_history = []
        self.volatility_cache = {}
        
        # Performance tracking
        self.sizing_performance = {
            'total_trades': 0,
            'profitable_trades': 0,
            'sizing_accuracy': 0.0,
            'risk_adjusted_returns': []
        }
        
        logger.info("✅ CL Position Sizer initialized")
        logger.info(f"   📊 Base Risk: {self.params.base_risk_percent:.1%}")
        logger.info(f"   📊 Max Position: {self.params.max_position_size:.1%}")
        logger.info(f"   📊 Volatility Lookback: {self.params.volatility_lookback} periods")
    
    async def calculate_position_size(self, 
                                    signal_data: Dict[str, Any],
                                    market_data: Dict[str, Any],
                                    portfolio_data: Dict[str, Any],
                                    method: SizingMethod = SizingMethod.ADAPTIVE) -> Dict[str, Any]:
        """
        Calculate optimal position size for CL trade
        
        Args:
            signal_data: Trading signal information
            market_data: Current market data
            portfolio_data: Portfolio information
            method: Sizing method to use
            
        Returns:
            Position sizing recommendation
        """
        try:
            # Extract key parameters
            portfolio_value = portfolio_data.get('total_value', 1000000)
            entry_price = market_data.get('close', 0)
            direction = signal_data.get('direction', 'long')
            confidence = signal_data.get('confidence', 0.5)
            
            # Calculate volatility metrics
            volatility_metrics = await self._calculate_volatility_metrics(market_data)
            
            # Determine market condition
            market_condition = await self._assess_market_condition(market_data)
            
            # Calculate base position size using selected method
            if method == SizingMethod.VOLATILITY_BASED:
                base_size = await self._volatility_based_sizing(
                    volatility_metrics, entry_price, portfolio_value
                )
            elif method == SizingMethod.KELLY_CRITERION:
                base_size = await self._kelly_criterion_sizing(
                    signal_data, volatility_metrics, portfolio_value
                )
            elif method == SizingMethod.FIXED_FRACTIONAL:
                base_size = await self._fixed_fractional_sizing(
                    volatility_metrics, entry_price, portfolio_value
                )
            elif method == SizingMethod.RISK_PARITY:
                base_size = await self._risk_parity_sizing(
                    volatility_metrics, portfolio_data
                )
            else:  # ADAPTIVE
                base_size = await self._adaptive_sizing(
                    signal_data, market_data, portfolio_data, volatility_metrics
                )
            
            # Apply adjustments
            adjusted_size = await self._apply_sizing_adjustments(
                base_size, confidence, market_condition, market_data, portfolio_data
            )
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(
                adjusted_size, entry_price, direction, volatility_metrics
            )
            
            # Validate and finalize size
            final_size = await self._validate_position_size(
                adjusted_size, risk_metrics, portfolio_data
            )
            
            # Generate sizing report
            sizing_report = {
                'recommended_size': final_size,
                'method_used': method.value,
                'base_size': base_size,
                'adjustments_applied': {
                    'confidence_adjustment': confidence,
                    'market_condition': market_condition,
                    'volatility_adjustment': volatility_metrics['atr_normalized'],
                    'session_adjustment': self._get_session_multiplier()
                },
                'risk_metrics': risk_metrics,
                'volatility_metrics': volatility_metrics,
                'sizing_rationale': await self._generate_sizing_rationale(
                    final_size, base_size, method, market_condition
                ),
                'confidence_score': self._calculate_sizing_confidence(
                    signal_data, volatility_metrics, market_condition
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store for performance tracking
            self._store_sizing_decision(sizing_report)
            
            return sizing_report
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return {
                'recommended_size': 0,
                'error': str(e),
                'method_used': method.value,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _calculate_volatility_metrics(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics for CL"""
        try:
            prices = market_data.get('prices', [])
            if len(prices) < self.params.volatility_lookback:
                return {
                    'atr_20': 0.0,
                    'atr_50': 0.0,
                    'atr_normalized': 0.0,
                    'volatility_percentile': 0.0,
                    'volatility_trend': 'stable'
                }
            
            # Calculate ATR (Average True Range)
            atr_20 = self._calculate_atr(prices[-self.params.volatility_lookback:])
            atr_50 = self._calculate_atr(prices[-50:]) if len(prices) >= 50 else atr_20
            
            # Calculate realized volatility
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i]['close'] - prices[i-1]['close']) / prices[i-1]['close']
                returns.append(ret)
            
            realized_vol = np.std(returns[-self.params.volatility_lookback:]) * np.sqrt(252)
            
            # Calculate volatility percentile
            if len(prices) >= 100:
                vol_history = []
                for i in range(50, len(prices)):
                    hist_returns = []
                    for j in range(i-20, i):
                        ret = (prices[j]['close'] - prices[j-1]['close']) / prices[j-1]['close']
                        hist_returns.append(ret)
                    vol_history.append(np.std(hist_returns))
                
                volatility_percentile = np.percentile(vol_history, 
                                                    np.where(np.array(vol_history) <= realized_vol)[0].size / len(vol_history) * 100)
            else:
                volatility_percentile = 50.0
            
            # Normalize ATR
            current_price = prices[-1]['close']
            atr_normalized = atr_20 / current_price if current_price > 0 else 0.0
            
            # Determine volatility trend
            if len(prices) >= 40:
                recent_atr = self._calculate_atr(prices[-20:])
                older_atr = self._calculate_atr(prices[-40:-20])
                vol_trend = 'increasing' if recent_atr > older_atr * 1.1 else 'decreasing' if recent_atr < older_atr * 0.9 else 'stable'
            else:
                vol_trend = 'stable'
            
            return {
                'atr_20': atr_20,
                'atr_50': atr_50,
                'atr_normalized': atr_normalized,
                'realized_volatility': realized_vol,
                'volatility_percentile': volatility_percentile,
                'volatility_trend': vol_trend,
                'volatility_regime': 'high' if volatility_percentile > 75 else 'low' if volatility_percentile < 25 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {
                'atr_20': 0.0,
                'atr_50': 0.0,
                'atr_normalized': 0.0,
                'volatility_percentile': 0.0,
                'volatility_trend': 'stable'
            }
    
    def _calculate_atr(self, prices: List[Dict[str, float]]) -> float:
        """Calculate Average True Range"""
        if len(prices) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i]['high']
            low = prices[i]['low']
            prev_close = prices[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return np.mean(true_ranges) if true_ranges else 0.0
    
    async def _assess_market_condition(self, market_data: Dict[str, Any]) -> str:
        """Assess current market condition for CL"""
        try:
            prices = market_data.get('prices', [])
            if len(prices) < 20:
                return 'normal'
            
            # Calculate trend strength
            closes = [p['close'] for p in prices[-20:]]
            trend_strength = (closes[-1] - closes[0]) / closes[0]
            
            # Calculate volatility
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            volatility = np.std(returns)
            
            # Check for inventory impact
            inventory_change = market_data.get('inventory_change', 0)
            inventory_impact = abs(inventory_change) > 5000000  # 5M barrels
            
            # Check for geopolitical events (simplified)
            geopolitical_risk = market_data.get('geopolitical_risk', 0) > 0.7
            
            # Classify condition
            if geopolitical_risk:
                return 'geopolitical_risk'
            elif inventory_impact:
                return 'inventory_shock'
            elif volatility > 0.03:  # High volatility
                return 'high_volatility'
            elif volatility < 0.01:  # Low volatility
                return 'low_volatility'
            elif abs(trend_strength) > 0.05:  # Strong trend
                return 'trending'
            else:
                return 'ranging'
                
        except Exception as e:
            logger.error(f"Error assessing market condition: {e}")
            return 'normal'
    
    async def _volatility_based_sizing(self, 
                                     volatility_metrics: Dict[str, Any],
                                     entry_price: float,
                                     portfolio_value: float) -> float:
        """Calculate position size based on volatility"""
        try:
            # Base risk amount
            base_risk = portfolio_value * self.params.base_risk_percent
            
            # Volatility adjustment
            atr_normalized = volatility_metrics['atr_normalized']
            
            # Inverse volatility scaling
            # Higher volatility = smaller position
            volatility_factor = 1.0 / (1.0 + atr_normalized * 5.0)
            
            # Calculate position size
            # Risk per contract = 2 * ATR * contract size
            risk_per_contract = 2.0 * volatility_metrics['atr_20'] * self.cl_contract_size
            
            if risk_per_contract > 0:
                contracts = (base_risk * volatility_factor) / risk_per_contract
                return max(contracts, self.params.min_position_size)
            else:
                return self.params.min_position_size
                
        except Exception as e:
            logger.error(f"Error in volatility-based sizing: {e}")
            return self.params.min_position_size
    
    async def _kelly_criterion_sizing(self, 
                                    signal_data: Dict[str, Any],
                                    volatility_metrics: Dict[str, Any],
                                    portfolio_value: float) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            # Get trade statistics
            win_rate = signal_data.get('win_rate', 0.5)
            avg_win = signal_data.get('avg_win', 0.02)
            avg_loss = signal_data.get('avg_loss', 0.015)
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            if avg_loss > 0:
                b = avg_win / avg_loss  # Odds ratio
                p = win_rate
                q = 1 - win_rate
                
                kelly_fraction = (b * p - q) / b
            else:
                kelly_fraction = 0.0
            
            # Apply safety factor
            adjusted_kelly = kelly_fraction * self.params.safety_factor
            
            # Cap at maximum Kelly fraction
            max_kelly = self.config.get('kelly_criterion', {}).get('max_kelly_fraction', 0.25)
            final_kelly = min(adjusted_kelly, max_kelly)
            
            # Convert to position size
            kelly_risk = portfolio_value * final_kelly
            
            # Calculate contracts
            risk_per_contract = 2.0 * volatility_metrics['atr_20'] * self.cl_contract_size
            
            if risk_per_contract > 0 and kelly_risk > 0:
                contracts = kelly_risk / risk_per_contract
                return max(contracts, self.params.min_position_size)
            else:
                return self.params.min_position_size
                
        except Exception as e:
            logger.error(f"Error in Kelly criterion sizing: {e}")
            return self.params.min_position_size
    
    async def _fixed_fractional_sizing(self, 
                                     volatility_metrics: Dict[str, Any],
                                     entry_price: float,
                                     portfolio_value: float) -> float:
        """Calculate position size using fixed fractional method"""
        try:
            # Fixed risk amount
            risk_amount = portfolio_value * self.params.base_risk_percent
            
            # Calculate stop distance (2 * ATR)
            stop_distance = 2.0 * volatility_metrics['atr_20']
            
            # Calculate contracts
            if stop_distance > 0:
                contracts = risk_amount / (stop_distance * self.cl_contract_size)
                return max(contracts, self.params.min_position_size)
            else:
                return self.params.min_position_size
                
        except Exception as e:
            logger.error(f"Error in fixed fractional sizing: {e}")
            return self.params.min_position_size
    
    async def _risk_parity_sizing(self, 
                                volatility_metrics: Dict[str, Any],
                                portfolio_data: Dict[str, Any]) -> float:
        """Calculate position size using risk parity approach"""
        try:
            # Get portfolio volatility
            portfolio_volatility = portfolio_data.get('volatility', 0.2)  # Default 20%
            
            # CL volatility
            cl_volatility = volatility_metrics['realized_volatility']
            
            # Risk parity sizing
            # Size inversely proportional to volatility
            if cl_volatility > 0:
                volatility_ratio = portfolio_volatility / cl_volatility
                base_size = self.params.base_risk_percent * volatility_ratio
            else:
                base_size = self.params.base_risk_percent
            
            # Convert to contracts
            portfolio_value = portfolio_data.get('total_value', 1000000)
            risk_amount = portfolio_value * base_size
            
            # Calculate contracts
            risk_per_contract = 2.0 * volatility_metrics['atr_20'] * self.cl_contract_size
            
            if risk_per_contract > 0:
                contracts = risk_amount / risk_per_contract
                return max(contracts, self.params.min_position_size)
            else:
                return self.params.min_position_size
                
        except Exception as e:
            logger.error(f"Error in risk parity sizing: {e}")
            return self.params.min_position_size
    
    async def _adaptive_sizing(self, 
                             signal_data: Dict[str, Any],
                             market_data: Dict[str, Any],
                             portfolio_data: Dict[str, Any],
                             volatility_metrics: Dict[str, Any]) -> float:
        """Calculate position size using adaptive approach"""
        try:
            # Get multiple sizing methods
            vol_size = await self._volatility_based_sizing(
                volatility_metrics, market_data.get('close', 0), portfolio_data.get('total_value', 1000000)
            )
            
            kelly_size = await self._kelly_criterion_sizing(
                signal_data, volatility_metrics, portfolio_data.get('total_value', 1000000)
            )
            
            fixed_size = await self._fixed_fractional_sizing(
                volatility_metrics, market_data.get('close', 0), portfolio_data.get('total_value', 1000000)
            )
            
            # Weight based on signal confidence and market condition
            confidence = signal_data.get('confidence', 0.5)
            
            # Higher confidence = more weight on Kelly
            # Lower confidence = more weight on fixed fractional
            if confidence > 0.7:
                weights = [0.3, 0.5, 0.2]  # [vol, kelly, fixed]
            elif confidence > 0.5:
                weights = [0.4, 0.3, 0.3]
            else:
                weights = [0.5, 0.2, 0.3]
            
            # Weighted average
            adaptive_size = (vol_size * weights[0] + 
                           kelly_size * weights[1] + 
                           fixed_size * weights[2])
            
            return adaptive_size
            
        except Exception as e:
            logger.error(f"Error in adaptive sizing: {e}")
            return self.params.min_position_size
    
    async def _apply_sizing_adjustments(self, 
                                      base_size: float,
                                      confidence: float,
                                      market_condition: str,
                                      market_data: Dict[str, Any],
                                      portfolio_data: Dict[str, Any]) -> float:
        """Apply various adjustments to base position size"""
        try:
            adjusted_size = base_size
            
            # Confidence adjustment
            confidence_multiplier = min(confidence * 1.5, 1.2)  # Max 20% boost
            adjusted_size *= confidence_multiplier
            
            # Market condition adjustment
            condition_multiplier = self.condition_multipliers.get(market_condition, 1.0)
            adjusted_size *= condition_multiplier
            
            # Session adjustment
            session_multiplier = self._get_session_multiplier()
            adjusted_size *= session_multiplier
            
            # Correlation adjustment
            correlation_multiplier = await self._calculate_correlation_adjustment(
                market_data, portfolio_data
            )
            adjusted_size *= correlation_multiplier
            
            # Volatility regime adjustment
            volatility_regime = market_data.get('volatility_regime', 'normal')
            if volatility_regime == 'high':
                adjusted_size *= 0.8
            elif volatility_regime == 'low':
                adjusted_size *= 1.1
            
            return adjusted_size
            
        except Exception as e:
            logger.error(f"Error applying sizing adjustments: {e}")
            return base_size
    
    def _get_session_multiplier(self) -> float:
        """Get session-based multiplier"""
        current_hour = datetime.now().hour
        
        # US session (9 AM - 5 PM ET)
        if 9 <= current_hour <= 17:
            return self.session_multipliers['us_session']
        # European session (3 AM - 11 AM ET)
        elif 3 <= current_hour <= 11:
            return self.session_multipliers['european_session']
        # Asian session (6 PM - 2 AM ET)
        elif 18 <= current_hour <= 23 or 0 <= current_hour <= 2:
            return self.session_multipliers['asian_session']
        # Overnight
        else:
            return self.session_multipliers['overnight']
    
    async def _calculate_correlation_adjustment(self, 
                                              market_data: Dict[str, Any],
                                              portfolio_data: Dict[str, Any]) -> float:
        """Calculate correlation-based adjustment"""
        try:
            # Get existing positions
            positions = portfolio_data.get('positions', {})
            
            # Check for correlated positions
            correlated_exposure = 0
            for symbol, position in positions.items():
                if 'CL' in symbol or 'OIL' in symbol:  # Oil-related positions
                    correlated_exposure += position.get('value', 0)
            
            # Reduce size if high correlation
            portfolio_value = portfolio_data.get('total_value', 1000000)
            correlation_ratio = correlated_exposure / portfolio_value
            
            if correlation_ratio > self.params.correlation_threshold:
                # Reduce size based on correlation
                reduction_factor = 1 - (correlation_ratio - self.params.correlation_threshold)
                return max(reduction_factor, 0.5)  # Minimum 50% reduction
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating correlation adjustment: {e}")
            return 1.0
    
    async def _calculate_risk_metrics(self, 
                                    position_size: float,
                                    entry_price: float,
                                    direction: str,
                                    volatility_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk metrics for the position"""
        try:
            # Calculate stop loss
            stop_distance = 2.0 * volatility_metrics['atr_20']
            if direction.lower() == 'long':
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
            
            # Calculate risk per contract
            risk_per_contract = abs(entry_price - stop_loss) * self.cl_contract_size
            
            # Total risk
            total_risk = position_size * risk_per_contract
            
            # Position value
            position_value = position_size * entry_price * self.cl_contract_size
            
            # Take profit (1.5:1 risk/reward)
            profit_distance = stop_distance * 1.5
            if direction.lower() == 'long':
                take_profit = entry_price + profit_distance
            else:
                take_profit = entry_price - profit_distance
            
            return {
                'position_size': position_size,
                'position_value': position_value,
                'total_risk': total_risk,
                'risk_per_contract': risk_per_contract,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': 1.5,
                'max_loss_percent': total_risk / position_value if position_value > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    async def _validate_position_size(self, 
                                    position_size: float,
                                    risk_metrics: Dict[str, Any],
                                    portfolio_data: Dict[str, Any]) -> float:
        """Validate and apply final limits to position size"""
        try:
            portfolio_value = portfolio_data.get('total_value', 1000000)
            
            # Apply minimum size
            validated_size = max(position_size, self.params.min_position_size)
            
            # Apply maximum size
            validated_size = min(validated_size, self.params.max_position_size)
            
            # Check maximum position value
            max_position_value = portfolio_value * self.params.max_position_size
            position_value = risk_metrics.get('position_value', 0)
            
            if position_value > max_position_value:
                # Scale down to maximum
                scale_factor = max_position_value / position_value
                validated_size *= scale_factor
            
            # Check maximum risk
            max_risk = portfolio_value * self.params.base_risk_percent * 2  # 2x base risk limit
            total_risk = risk_metrics.get('total_risk', 0)
            
            if total_risk > max_risk:
                # Scale down to maximum risk
                scale_factor = max_risk / total_risk
                validated_size *= scale_factor
            
            return max(validated_size, self.params.min_position_size)
            
        except Exception as e:
            logger.error(f"Error validating position size: {e}")
            return self.params.min_position_size
    
    async def _generate_sizing_rationale(self, 
                                       final_size: float,
                                       base_size: float,
                                       method: SizingMethod,
                                       market_condition: str) -> str:
        """Generate human-readable rationale for sizing decision"""
        try:
            size_change = (final_size - base_size) / base_size if base_size > 0 else 0
            
            rationale = f"Position size calculated using {method.value} method. "
            
            if size_change > 0.1:
                rationale += f"Size increased by {size_change:.1%} due to "
            elif size_change < -0.1:
                rationale += f"Size decreased by {abs(size_change):.1%} due to "
            else:
                rationale += "Size maintained with minor adjustments for "
            
            rationale += f"{market_condition} market conditions. "
            
            if final_size >= self.params.max_position_size * 0.8:
                rationale += "Near maximum position size due to favorable conditions. "
            elif final_size <= self.params.min_position_size * 1.2:
                rationale += "Near minimum position size due to elevated risk. "
            
            return rationale
            
        except Exception as e:
            logger.error(f"Error generating sizing rationale: {e}")
            return f"Position size: {final_size:.2f} contracts using {method.value} method"
    
    def _calculate_sizing_confidence(self, 
                                   signal_data: Dict[str, Any],
                                   volatility_metrics: Dict[str, Any],
                                   market_condition: str) -> float:
        """Calculate confidence in sizing decision"""
        try:
            # Base confidence from signal
            base_confidence = signal_data.get('confidence', 0.5)
            
            # Volatility confidence
            vol_regime = volatility_metrics.get('volatility_regime', 'normal')
            if vol_regime == 'normal':
                vol_confidence = 0.8
            elif vol_regime == 'high':
                vol_confidence = 0.6
            else:  # low
                vol_confidence = 0.7
            
            # Market condition confidence
            condition_confidence = {
                'trending': 0.8,
                'ranging': 0.6,
                'high_volatility': 0.5,
                'low_volatility': 0.7,
                'geopolitical_risk': 0.4,
                'inventory_shock': 0.3
            }.get(market_condition, 0.6)
            
            # Weighted average
            final_confidence = (base_confidence * 0.4 + 
                             vol_confidence * 0.3 + 
                             condition_confidence * 0.3)
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating sizing confidence: {e}")
            return 0.5
    
    def _store_sizing_decision(self, sizing_report: Dict[str, Any]):
        """Store sizing decision for performance tracking"""
        try:
            self.sizing_performance['total_trades'] += 1
            
            # Store key metrics
            decision_record = {
                'timestamp': sizing_report['timestamp'],
                'size': sizing_report['recommended_size'],
                'method': sizing_report['method_used'],
                'confidence': sizing_report['confidence_score'],
                'risk_metrics': sizing_report['risk_metrics']
            }
            
            # Keep recent history
            if not hasattr(self, 'sizing_history'):
                self.sizing_history = []
            
            self.sizing_history.append(decision_record)
            if len(self.sizing_history) > 1000:
                self.sizing_history = self.sizing_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error storing sizing decision: {e}")
    
    def get_sizing_performance(self) -> Dict[str, Any]:
        """Get sizing performance metrics"""
        try:
            return {
                'total_trades': self.sizing_performance['total_trades'],
                'profitable_trades': self.sizing_performance['profitable_trades'],
                'win_rate': self.sizing_performance['profitable_trades'] / max(self.sizing_performance['total_trades'], 1),
                'sizing_accuracy': self.sizing_performance['sizing_accuracy'],
                'risk_adjusted_returns': self.sizing_performance['risk_adjusted_returns'][-20:],
                'average_position_size': np.mean([d['size'] for d in getattr(self, 'sizing_history', [])]) if hasattr(self, 'sizing_history') else 0,
                'sizing_consistency': np.std([d['size'] for d in getattr(self, 'sizing_history', [])]) if hasattr(self, 'sizing_history') else 0
            }
        except Exception as e:
            logger.error(f"Error getting sizing performance: {e}")
            return {}
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update sizing performance based on trade result"""
        try:
            if trade_result.get('profitable', False):
                self.sizing_performance['profitable_trades'] += 1
            
            # Calculate risk-adjusted return
            risk_amount = trade_result.get('risk_amount', 0)
            pnl = trade_result.get('pnl', 0)
            
            if risk_amount > 0:
                risk_adjusted_return = pnl / risk_amount
                self.sizing_performance['risk_adjusted_returns'].append(risk_adjusted_return)
            
            # Update sizing accuracy
            expected_size = trade_result.get('expected_size', 0)
            actual_size = trade_result.get('actual_size', 0)
            
            if expected_size > 0:
                accuracy = 1 - abs(actual_size - expected_size) / expected_size
                self.sizing_performance['sizing_accuracy'] = (
                    self.sizing_performance['sizing_accuracy'] * 0.9 + accuracy * 0.1
                )
            
        except Exception as e:
            logger.error(f"Error updating sizing performance: {e}")