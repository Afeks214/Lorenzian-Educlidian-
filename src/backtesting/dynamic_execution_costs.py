"""
Dynamic Execution Cost Modeling
==============================

AGENT 4 - REALISTIC EXECUTION ENGINE INTEGRATION

This module provides comprehensive execution cost modeling to replace
the simplified 10% slippage assumption with realistic, dynamic cost
calculations based on market conditions and order characteristics.

Cost Components:
- Dynamic slippage based on market conditions
- Realistic commission structure
- Exchange and regulatory fees
- Market impact modeling
- Bid-ask spread costs
- Timing and latency costs

Author: AGENT 4 - Realistic Execution Engine Integration
Date: 2025-07-17
Mission: Eliminate unrealistic fixed execution costs
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InstrumentType(Enum):
    """Supported instrument types for cost modeling"""
    FUTURES_NQ = "futures_nq"
    FUTURES_ES = "futures_es"
    STOCK_EQUITY = "stock_equity"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass
class InstrumentSpecs:
    """Instrument specifications for cost modeling"""
    instrument_type: InstrumentType
    symbol: str
    point_value: float
    tick_size: float
    tick_value: float
    typical_spread_ticks: float
    commission_per_unit: float
    exchange_fees: float
    regulatory_fees: float
    margin_requirement: float


class MarketConditionAssessment:
    """
    Assessment of current market conditions for cost modeling
    """
    
    def __init__(self):
        self.volatility_thresholds = {
            'low': 0.01,
            'medium': 0.03,
            'high': 0.05,
            'extreme': 0.08
        }
        
        self.volume_thresholds = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'extreme': 2.5
        }
    
    def assess_volatility_regime(self, volatility: float) -> str:
        """Assess volatility regime from current volatility"""
        if volatility >= self.volatility_thresholds['extreme']:
            return 'extreme'
        elif volatility >= self.volatility_thresholds['high']:
            return 'high'
        elif volatility >= self.volatility_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def assess_liquidity_regime(self, volume_ratio: float) -> str:
        """Assess liquidity regime from volume ratio"""
        if volume_ratio >= self.volume_thresholds['extreme']:
            return 'extreme'
        elif volume_ratio >= self.volume_thresholds['high']:
            return 'high'
        elif volume_ratio >= self.volume_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def assess_time_of_day_factor(self, timestamp: datetime) -> float:
        """Assess time of day liquidity factor"""
        hour = timestamp.hour
        
        # Market hours (9:30 AM - 4:00 PM EST)
        if 9 <= hour <= 16:
            if 9 <= hour <= 11 or 14 <= hour <= 16:
                return 1.0  # Peak liquidity
            else:
                return 0.85  # Normal market hours
        elif 6 <= hour <= 9 or 16 <= hour <= 18:
            return 0.6  # Pre/post market
        else:
            return 0.3  # Overnight
    
    def assess_market_stress(self, price_data: pd.Series, 
                           volume_data: Optional[pd.Series] = None) -> float:
        """Assess market stress from price and volume data"""
        try:
            # Calculate volatility-based stress
            if len(price_data) >= 2:
                returns = price_data.pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std()
                    volatility_stress = min(1.0, volatility * 20)  # Scale factor
                else:
                    volatility_stress = 0.0
            else:
                volatility_stress = 0.0
            
            # Calculate volume-based stress
            if volume_data is not None and len(volume_data) >= 2:
                volume_change = volume_data.pct_change().dropna()
                if len(volume_change) > 0:
                    volume_stress = min(1.0, abs(volume_change.iloc[-1]))
                else:
                    volume_stress = 0.0
            else:
                volume_stress = 0.0
            
            # Combine stress indicators
            overall_stress = (volatility_stress * 0.7 + volume_stress * 0.3)
            
            return min(1.0, overall_stress)
            
        except Exception as e:
            logger.warning(f"Market stress assessment failed: {e}")
            return 0.0


class DynamicSlippageModel:
    """
    Dynamic slippage model that considers market conditions, order size,
    and timing factors to provide realistic slippage estimates.
    """
    
    def __init__(self, instrument_specs: InstrumentSpecs):
        self.specs = instrument_specs
        self.condition_assessor = MarketConditionAssessment()
        
        # Base slippage parameters by instrument type
        self.base_slippage_params = {
            InstrumentType.FUTURES_NQ: {
                'market_order_base': 0.5,  # 0.5 points base slippage
                'limit_order_base': 0.25,  # 0.25 points base slippage
                'size_impact_factor': 0.1,  # 0.1 points per additional contract
                'volatility_multiplier': 2.0,
                'liquidity_multiplier': 1.5,
                'time_multiplier': 1.2
            },
            InstrumentType.FUTURES_ES: {
                'market_order_base': 0.25,
                'limit_order_base': 0.125,
                'size_impact_factor': 0.05,
                'volatility_multiplier': 1.8,
                'liquidity_multiplier': 1.3,
                'time_multiplier': 1.1
            },
            InstrumentType.STOCK_EQUITY: {
                'market_order_base': 0.01,  # 1 cent base slippage
                'limit_order_base': 0.005,  # 0.5 cent base slippage
                'size_impact_factor': 0.001,
                'volatility_multiplier': 1.5,
                'liquidity_multiplier': 2.0,
                'time_multiplier': 1.3
            }
        }
    
    def calculate_slippage(self, 
                          order_size: float,
                          order_type: str,
                          market_data: pd.Series,
                          timestamp: datetime,
                          historical_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate dynamic slippage based on market conditions
        
        Args:
            order_size: Size of the order (contracts or shares)
            order_type: Type of order ('market', 'limit', etc.)
            market_data: Current market data
            timestamp: Order timestamp
            historical_data: Historical data for context
            
        Returns:
            Dictionary with slippage components and total
        """
        try:
            # Get base parameters for instrument
            params = self.base_slippage_params.get(
                self.specs.instrument_type,
                self.base_slippage_params[InstrumentType.FUTURES_NQ]
            )
            
            # Base slippage by order type
            if order_type.lower() == 'market':
                base_slippage = params['market_order_base']
            else:
                base_slippage = params['limit_order_base']
            
            # Size impact component
            size_impact = order_size * params['size_impact_factor']
            
            # Market condition adjustments
            volatility = self._estimate_volatility(market_data, historical_data)
            volume_ratio = self._estimate_volume_ratio(market_data, historical_data)
            
            # Volatility adjustment
            volatility_regime = self.condition_assessor.assess_volatility_regime(volatility)
            volatility_multiplier = self._get_volatility_multiplier(volatility_regime, params)
            volatility_adjustment = volatility * volatility_multiplier
            
            # Liquidity adjustment
            liquidity_regime = self.condition_assessor.assess_liquidity_regime(volume_ratio)
            liquidity_multiplier = self._get_liquidity_multiplier(liquidity_regime, params)
            liquidity_adjustment = (1.0 / volume_ratio) * liquidity_multiplier
            
            # Time of day adjustment
            time_factor = self.condition_assessor.assess_time_of_day_factor(timestamp)
            time_adjustment = (1.0 - time_factor) * params['time_multiplier']
            
            # Market stress adjustment
            stress_factor = self.condition_assessor.assess_market_stress(
                market_data if isinstance(market_data, pd.Series) else pd.Series([market_data.get('Close', 0)]),
                pd.Series([market_data.get('Volume', 1000000)]) if 'Volume' in market_data else None
            )
            stress_adjustment = stress_factor * base_slippage
            
            # Calculate total slippage
            total_slippage = (
                base_slippage +
                size_impact +
                volatility_adjustment +
                liquidity_adjustment +
                time_adjustment +
                stress_adjustment
            )
            
            # Ensure minimum slippage (at least 1 tick)
            total_slippage = max(total_slippage, self.specs.tick_size)
            
            # Cap maximum slippage (prevent extreme values)
            max_slippage = base_slippage * 5  # 5x base slippage maximum
            total_slippage = min(total_slippage, max_slippage)
            
            return {
                'base_slippage': base_slippage,
                'size_impact': size_impact,
                'volatility_adjustment': volatility_adjustment,
                'liquidity_adjustment': liquidity_adjustment,
                'time_adjustment': time_adjustment,
                'stress_adjustment': stress_adjustment,
                'total_slippage': total_slippage,
                'slippage_percentage': total_slippage / market_data.get('Close', 1) if market_data.get('Close', 1) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Slippage calculation failed: {e}")
            return {
                'base_slippage': self.specs.tick_size,
                'size_impact': 0,
                'volatility_adjustment': 0,
                'liquidity_adjustment': 0,
                'time_adjustment': 0,
                'stress_adjustment': 0,
                'total_slippage': self.specs.tick_size,
                'slippage_percentage': 0.001
            }
    
    def _estimate_volatility(self, market_data: pd.Series, 
                           historical_data: Optional[pd.DataFrame] = None) -> float:
        """Estimate current volatility"""
        try:
            if historical_data is not None and len(historical_data) > 20:
                # Use historical data for better estimate
                returns = historical_data['Close'].pct_change().dropna()
                if len(returns) > 0:
                    return returns.std()
            
            # Fallback to high-low range
            if 'High' in market_data and 'Low' in market_data:
                price_range = market_data['High'] - market_data['Low']
                current_price = market_data.get('Close', market_data['High'])
                return price_range / current_price if current_price > 0 else 0.01
            
            return 0.01  # Default volatility
            
        except Exception:
            return 0.01
    
    def _estimate_volume_ratio(self, market_data: pd.Series,
                             historical_data: Optional[pd.DataFrame] = None) -> float:
        """Estimate volume ratio to average"""
        try:
            current_volume = market_data.get('Volume', 1000000)
            
            if historical_data is not None and len(historical_data) > 20:
                avg_volume = historical_data['Volume'].mean()
                return current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return 1.0  # Default ratio
            
        except Exception:
            return 1.0
    
    def _get_volatility_multiplier(self, regime: str, params: Dict[str, float]) -> float:
        """Get volatility multiplier based on regime"""
        multipliers = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'extreme': 2.0
        }
        return multipliers.get(regime, 1.0) * params['volatility_multiplier']
    
    def _get_liquidity_multiplier(self, regime: str, params: Dict[str, float]) -> float:
        """Get liquidity multiplier based on regime"""
        multipliers = {
            'low': 2.0,
            'medium': 1.0,
            'high': 0.7,
            'extreme': 0.5
        }
        return multipliers.get(regime, 1.0) * params['liquidity_multiplier']


class ComprehensiveCostModel:
    """
    Comprehensive cost model that combines all execution costs
    """
    
    def __init__(self, instrument_specs: InstrumentSpecs):
        self.specs = instrument_specs
        self.slippage_model = DynamicSlippageModel(instrument_specs)
        
        # Cost breakdown tracking
        self.cost_history = []
    
    def calculate_total_execution_costs(self,
                                      order_size: float,
                                      order_type: str,
                                      market_data: pd.Series,
                                      timestamp: datetime,
                                      historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive execution costs
        
        Returns:
            Complete cost breakdown with all components
        """
        try:
            # Calculate slippage
            slippage_breakdown = self.slippage_model.calculate_slippage(
                order_size, order_type, market_data, timestamp, historical_data
            )
            
            # Calculate commission costs
            commission_costs = self._calculate_commission_costs(order_size)
            
            # Calculate exchange and regulatory fees
            exchange_fees = self._calculate_exchange_fees(order_size)
            regulatory_fees = self._calculate_regulatory_fees(order_size)
            
            # Calculate bid-ask spread costs
            spread_costs = self._calculate_spread_costs(order_size, market_data)
            
            # Calculate market impact costs
            market_impact_costs = self._calculate_market_impact_costs(
                order_size, market_data, timestamp
            )
            
            # Calculate timing costs (opportunity cost of delays)
            timing_costs = self._calculate_timing_costs(order_size, market_data)
            
            # Total costs
            total_slippage_cost = slippage_breakdown['total_slippage'] * order_size * self.specs.point_value
            total_commission_cost = commission_costs['total_commission']
            total_fees = exchange_fees + regulatory_fees
            total_spread_cost = spread_costs['total_spread_cost']
            total_market_impact_cost = market_impact_costs['total_impact_cost']
            total_timing_cost = timing_costs['total_timing_cost']
            
            # Grand total
            total_execution_cost = (
                total_slippage_cost +
                total_commission_cost +
                total_fees +
                total_spread_cost +
                total_market_impact_cost +
                total_timing_cost
            )
            
            # Calculate cost per unit
            cost_per_unit = total_execution_cost / order_size if order_size > 0 else 0
            
            # Calculate cost as percentage of notional
            notional_value = order_size * market_data.get('Close', 1) * self.specs.point_value
            cost_percentage = (total_execution_cost / notional_value) * 100 if notional_value > 0 else 0
            
            # Create comprehensive cost breakdown
            cost_breakdown = {
                'timestamp': timestamp,
                'order_size': order_size,
                'order_type': order_type,
                'instrument': self.specs.symbol,
                
                # Slippage breakdown
                'slippage_details': slippage_breakdown,
                'slippage_cost': total_slippage_cost,
                
                # Commission breakdown
                'commission_details': commission_costs,
                'commission_cost': total_commission_cost,
                
                # Fee breakdown
                'exchange_fees': exchange_fees,
                'regulatory_fees': regulatory_fees,
                'total_fees': total_fees,
                
                # Spread breakdown
                'spread_details': spread_costs,
                'spread_cost': total_spread_cost,
                
                # Market impact breakdown
                'market_impact_details': market_impact_costs,
                'market_impact_cost': total_market_impact_cost,
                
                # Timing breakdown
                'timing_details': timing_costs,
                'timing_cost': total_timing_cost,
                
                # Summary
                'total_execution_cost': total_execution_cost,
                'cost_per_unit': cost_per_unit,
                'cost_percentage': cost_percentage,
                'notional_value': notional_value,
                
                # Quality metrics
                'execution_efficiency': self._calculate_execution_efficiency(cost_breakdown),
                'cost_ranking': self._rank_cost_components(cost_breakdown)
            }
            
            # Store cost history
            self.cost_history.append(cost_breakdown)
            
            return cost_breakdown
            
        except Exception as e:
            logger.error(f"Cost calculation failed: {e}")
            return self._create_fallback_cost_breakdown(order_size, market_data)
    
    def _calculate_commission_costs(self, order_size: float) -> Dict[str, float]:
        """Calculate commission costs"""
        base_commission = order_size * self.specs.commission_per_unit
        
        # Volume discounts (simplified)
        if order_size > 100:
            volume_discount = 0.1  # 10% discount for large orders
        elif order_size > 50:
            volume_discount = 0.05  # 5% discount for medium orders
        else:
            volume_discount = 0.0
        
        discounted_commission = base_commission * (1 - volume_discount)
        
        return {
            'base_commission': base_commission,
            'volume_discount': volume_discount,
            'total_commission': discounted_commission
        }
    
    def _calculate_exchange_fees(self, order_size: float) -> float:
        """Calculate exchange fees"""
        return order_size * self.specs.exchange_fees
    
    def _calculate_regulatory_fees(self, order_size: float) -> float:
        """Calculate regulatory fees"""
        return order_size * self.specs.regulatory_fees
    
    def _calculate_spread_costs(self, order_size: float, market_data: pd.Series) -> Dict[str, float]:
        """Calculate bid-ask spread costs"""
        try:
            # Estimate spread
            if 'Bid' in market_data and 'Ask' in market_data:
                spread = market_data['Ask'] - market_data['Bid']
            else:
                # Estimate spread from typical spread
                spread = self.specs.typical_spread_ticks * self.specs.tick_size
            
            # Half-spread cost (crossing the spread)
            half_spread_cost = (spread / 2) * order_size * self.specs.point_value
            
            return {
                'estimated_spread': spread,
                'half_spread_cost': half_spread_cost,
                'total_spread_cost': half_spread_cost
            }
            
        except Exception:
            return {
                'estimated_spread': self.specs.tick_size,
                'half_spread_cost': 0,
                'total_spread_cost': 0
            }
    
    def _calculate_market_impact_costs(self, order_size: float, 
                                     market_data: pd.Series, 
                                     timestamp: datetime) -> Dict[str, float]:
        """Calculate market impact costs"""
        try:
            # Simple market impact model
            base_impact = order_size * 0.01  # 1 basis point per unit
            
            # Time of day adjustment
            time_factor = MarketConditionAssessment().assess_time_of_day_factor(timestamp)
            time_adjusted_impact = base_impact * (1 + (1 - time_factor))
            
            # Volatility adjustment
            volatility = self._estimate_volatility(market_data)
            volatility_adjusted_impact = time_adjusted_impact * (1 + volatility * 10)
            
            # Convert to dollar cost
            current_price = market_data.get('Close', 1)
            impact_cost = volatility_adjusted_impact * current_price * self.specs.point_value
            
            return {
                'base_impact_bps': base_impact * 10000,  # Convert to basis points
                'time_adjusted_impact': time_adjusted_impact,
                'volatility_adjusted_impact': volatility_adjusted_impact,
                'total_impact_cost': impact_cost
            }
            
        except Exception:
            return {
                'base_impact_bps': 0,
                'time_adjusted_impact': 0,
                'volatility_adjusted_impact': 0,
                'total_impact_cost': 0
            }
    
    def _calculate_timing_costs(self, order_size: float, market_data: pd.Series) -> Dict[str, float]:
        """Calculate timing costs (opportunity cost of delays)"""
        try:
            # Simple timing cost model
            # Assume average delay of 100ms and price volatility
            avg_delay_seconds = 0.1  # 100ms
            
            # Estimate price volatility per second
            volatility = self._estimate_volatility(market_data)
            price_volatility_per_second = volatility / (252 * 24 * 3600)  # Annualized to per second
            
            # Calculate timing cost
            current_price = market_data.get('Close', 1)
            timing_cost = (avg_delay_seconds * price_volatility_per_second * 
                          current_price * order_size * self.specs.point_value)
            
            return {
                'avg_delay_seconds': avg_delay_seconds,
                'price_volatility_per_second': price_volatility_per_second,
                'total_timing_cost': timing_cost
            }
            
        except Exception:
            return {
                'avg_delay_seconds': 0,
                'price_volatility_per_second': 0,
                'total_timing_cost': 0
            }
    
    def _estimate_volatility(self, market_data: pd.Series) -> float:
        """Estimate volatility from market data"""
        try:
            if 'High' in market_data and 'Low' in market_data:
                price_range = market_data['High'] - market_data['Low']
                current_price = market_data.get('Close', market_data['High'])
                return price_range / current_price if current_price > 0 else 0.01
            return 0.01
        except:
            return 0.01
    
    def _calculate_execution_efficiency(self, cost_breakdown: Dict[str, Any]) -> float:
        """Calculate execution efficiency score (0-100)"""
        try:
            # Simple efficiency score based on cost percentage
            cost_percentage = cost_breakdown.get('cost_percentage', 0)
            
            if cost_percentage <= 0.05:  # <= 5 basis points
                return 100
            elif cost_percentage <= 0.1:  # <= 10 basis points
                return 80
            elif cost_percentage <= 0.2:  # <= 20 basis points
                return 60
            elif cost_percentage <= 0.5:  # <= 50 basis points
                return 40
            else:
                return 20
                
        except Exception:
            return 50
    
    def _rank_cost_components(self, cost_breakdown: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Rank cost components by magnitude"""
        try:
            components = [
                ('slippage', cost_breakdown.get('slippage_cost', 0)),
                ('commission', cost_breakdown.get('commission_cost', 0)),
                ('fees', cost_breakdown.get('total_fees', 0)),
                ('spread', cost_breakdown.get('spread_cost', 0)),
                ('market_impact', cost_breakdown.get('market_impact_cost', 0)),
                ('timing', cost_breakdown.get('timing_cost', 0))
            ]
            
            # Sort by cost magnitude
            components.sort(key=lambda x: x[1], reverse=True)
            
            return components
            
        except Exception:
            return []
    
    def _create_fallback_cost_breakdown(self, order_size: float, 
                                      market_data: pd.Series) -> Dict[str, Any]:
        """Create fallback cost breakdown on calculation failure"""
        notional_value = order_size * market_data.get('Close', 1) * self.specs.point_value
        fallback_cost = notional_value * 0.001  # 10 basis points fallback
        
        return {
            'timestamp': datetime.now(),
            'order_size': order_size,
            'order_type': 'market',
            'instrument': self.specs.symbol,
            'slippage_cost': fallback_cost * 0.6,
            'commission_cost': fallback_cost * 0.3,
            'total_fees': fallback_cost * 0.1,
            'spread_cost': 0,
            'market_impact_cost': 0,
            'timing_cost': 0,
            'total_execution_cost': fallback_cost,
            'cost_per_unit': fallback_cost / order_size if order_size > 0 else 0,
            'cost_percentage': 0.1,  # 10 basis points
            'notional_value': notional_value,
            'execution_efficiency': 60,
            'fallback_used': True
        }
    
    def get_cost_analytics(self) -> Dict[str, Any]:
        """Get comprehensive cost analytics"""
        if not self.cost_history:
            return {'error': 'No cost history available'}
        
        # Calculate aggregate metrics
        total_costs = [c['total_execution_cost'] for c in self.cost_history]
        cost_percentages = [c['cost_percentage'] for c in self.cost_history]
        efficiency_scores = [c['execution_efficiency'] for c in self.cost_history]
        
        # Component analysis
        slippage_costs = [c['slippage_cost'] for c in self.cost_history]
        commission_costs = [c['commission_cost'] for c in self.cost_history]
        
        return {
            'summary': {
                'total_executions': len(self.cost_history),
                'total_execution_costs': sum(total_costs),
                'avg_execution_cost': np.mean(total_costs),
                'avg_cost_percentage': np.mean(cost_percentages),
                'avg_efficiency_score': np.mean(efficiency_scores)
            },
            'cost_breakdown': {
                'total_slippage_costs': sum(slippage_costs),
                'total_commission_costs': sum(commission_costs),
                'avg_slippage_cost': np.mean(slippage_costs),
                'avg_commission_cost': np.mean(commission_costs)
            },
            'performance_metrics': {
                'cost_volatility': np.std(cost_percentages),
                'efficiency_consistency': np.std(efficiency_scores),
                'cost_range': [min(cost_percentages), max(cost_percentages)]
            }
        }


def create_nq_futures_cost_model() -> ComprehensiveCostModel:
    """Create cost model for NQ futures"""
    nq_specs = InstrumentSpecs(
        instrument_type=InstrumentType.FUTURES_NQ,
        symbol="NQ",
        point_value=20.0,
        tick_size=0.25,
        tick_value=5.0,
        typical_spread_ticks=1.0,
        commission_per_unit=0.50,
        exchange_fees=0.02,
        regulatory_fees=0.02,
        margin_requirement=19000.0
    )
    
    return ComprehensiveCostModel(nq_specs)


def create_es_futures_cost_model() -> ComprehensiveCostModel:
    """Create cost model for ES futures"""
    es_specs = InstrumentSpecs(
        instrument_type=InstrumentType.FUTURES_ES,
        symbol="ES",
        point_value=50.0,
        tick_size=0.25,
        tick_value=12.5,
        typical_spread_ticks=1.0,
        commission_per_unit=0.50,
        exchange_fees=0.02,
        regulatory_fees=0.02,
        margin_requirement=13000.0
    )
    
    return ComprehensiveCostModel(es_specs)


def create_stock_cost_model(symbol: str) -> ComprehensiveCostModel:
    """Create cost model for stock equity"""
    stock_specs = InstrumentSpecs(
        instrument_type=InstrumentType.STOCK_EQUITY,
        symbol=symbol,
        point_value=1.0,
        tick_size=0.01,
        tick_value=0.01,
        typical_spread_ticks=1.0,
        commission_per_unit=0.005,  # $0.005 per share
        exchange_fees=0.0001,
        regulatory_fees=0.0001,
        margin_requirement=0.5  # 50% margin
    )
    
    return ComprehensiveCostModel(stock_specs)