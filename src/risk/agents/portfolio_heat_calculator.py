"""
Portfolio Heat Calculator and Exposure Monitoring System

Implements comprehensive portfolio risk monitoring including:
- Real-time portfolio heat calculation
- Position exposure monitoring
- Sector and asset concentration tracking
- Correlation-based risk assessment
- Dynamic rebalancing recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import structlog
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import yaml

logger = structlog.get_logger()


class ExposureType(Enum):
    """Types of exposure monitoring"""
    GROSS = "gross"
    NET = "net"
    LONG = "long"
    SHORT = "short"
    SECTOR = "sector"
    ASSET = "asset"
    STRATEGY = "strategy"


@dataclass
class ExposureMetrics:
    """Exposure metrics dataclass"""
    exposure_type: ExposureType
    current_exposure: float
    limit: float
    utilization: float
    risk_level: str
    positions: List[Dict[str, Any]]
    
    def is_violation(self) -> bool:
        """Check if exposure violates limits"""
        return self.current_exposure > self.limit


@dataclass
class PortfolioHeatMetrics:
    """Portfolio heat metrics dataclass"""
    total_heat: float
    individual_heat: Dict[str, float]
    correlation_adjusted_heat: float
    risk_contribution: Dict[str, float]
    concentration_risk: float
    diversification_ratio: float
    
    def get_risk_level(self) -> str:
        """Get risk level based on heat metrics"""
        if self.total_heat > 0.15:
            return "CRITICAL"
        elif self.total_heat > 0.12:
            return "HIGH"
        elif self.total_heat > 0.08:
            return "MEDIUM"
        else:
            return "LOW"


class PortfolioHeatCalculator:
    """
    Portfolio Heat Calculator for comprehensive risk monitoring
    
    Calculates various risk metrics including:
    - Portfolio heat (total risk exposure)
    - Individual position heat contributions
    - Correlation-adjusted risk metrics
    - Concentration risk measures
    - Diversification ratios
    """
    
    def __init__(self, config_path: str = "/home/QuantNova/GrandModel/config/risk_management_config.yaml"):
        """Initialize Portfolio Heat Calculator"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.portfolio_config = self.config['portfolio_management']
        self.position_config = self.config['position_sizing']
        self.heat_config = self.portfolio_config['portfolio_heat']
        
        # Heat calculation parameters
        self.max_heat_threshold = self.heat_config['max_heat_threshold']
        self.rebalance_threshold = self.heat_config['rebalance_threshold']
        self.calculation_method = self.heat_config['heat_calculation_method']
        
        # Portfolio state
        self.positions = {}
        self.historical_returns = {}
        self.correlation_matrix = None
        self.volatility_cache = {}
        self.sector_mapping = {}
        
        # Monitoring state
        self.heat_history = []
        self.exposure_history = []
        self.rebalance_recommendations = []
        
        logger.info("Portfolio Heat Calculator initialized",
                   max_heat_threshold=self.max_heat_threshold,
                   calculation_method=self.calculation_method)
    
    def calculate_portfolio_heat(self, positions: Dict[str, Any], 
                               portfolio_value: float) -> PortfolioHeatMetrics:
        """
        Calculate comprehensive portfolio heat metrics
        
        Args:
            positions: Dictionary of current positions
            portfolio_value: Total portfolio value
            
        Returns:
            PortfolioHeatMetrics object with comprehensive metrics
        """
        try:
            # Calculate individual position heat
            individual_heat = {}
            total_heat = 0.0
            
            for symbol, position in positions.items():
                position_heat = self._calculate_position_heat(
                    symbol, position, portfolio_value)
                individual_heat[symbol] = position_heat
                total_heat += position_heat
            
            # Calculate correlation-adjusted heat
            correlation_adjusted_heat = self._calculate_correlation_adjusted_heat(
                positions, portfolio_value)
            
            # Calculate risk contribution
            risk_contribution = self._calculate_risk_contribution(
                positions, portfolio_value)
            
            # Calculate concentration risk
            concentration_risk = self._calculate_concentration_risk(
                positions, portfolio_value)
            
            # Calculate diversification ratio
            diversification_ratio = self._calculate_diversification_ratio(
                positions, portfolio_value)
            
            # Create heat metrics
            heat_metrics = PortfolioHeatMetrics(
                total_heat=total_heat,
                individual_heat=individual_heat,
                correlation_adjusted_heat=correlation_adjusted_heat,
                risk_contribution=risk_contribution,
                concentration_risk=concentration_risk,
                diversification_ratio=diversification_ratio
            )
            
            # Store in history
            self.heat_history.append({
                'timestamp': datetime.now(),
                'total_heat': total_heat,
                'correlation_adjusted_heat': correlation_adjusted_heat,
                'concentration_risk': concentration_risk,
                'diversification_ratio': diversification_ratio
            })
            
            # Keep only recent history
            if len(self.heat_history) > 1000:
                self.heat_history = self.heat_history[-1000:]
            
            logger.info("Portfolio heat calculated",
                       total_heat=total_heat,
                       correlation_adjusted_heat=correlation_adjusted_heat,
                       risk_level=heat_metrics.get_risk_level())
            
            return heat_metrics
            
        except Exception as e:
            logger.error("Error calculating portfolio heat", error=str(e))
            return self._get_default_heat_metrics()
    
    def _calculate_position_heat(self, symbol: str, position: Dict[str, Any], 
                               portfolio_value: float) -> float:
        """Calculate heat contribution from individual position"""
        try:
            position_value = position['size'] * position['current_price']
            volatility = self._get_volatility(symbol)
            
            # Basic heat calculation
            position_risk = position_value * volatility
            position_heat = position_risk / portfolio_value
            
            return position_heat
            
        except Exception as e:
            logger.error("Error calculating position heat", error=str(e), symbol=symbol)
            return 0.0
    
    def _calculate_correlation_adjusted_heat(self, positions: Dict[str, Any], 
                                           portfolio_value: float) -> float:
        """Calculate correlation-adjusted portfolio heat"""
        if self.calculation_method != "correlation_adjusted":
            return sum(self._calculate_position_heat(s, p, portfolio_value) 
                      for s, p in positions.items())
        
        try:
            # Get position weights and volatilities
            weights = {}
            volatilities = {}
            
            for symbol, position in positions.items():
                position_value = position['size'] * position['current_price']
                weights[symbol] = position_value / portfolio_value
                volatilities[symbol] = self._get_volatility(symbol)
            
            # Calculate correlation-adjusted risk
            symbols = list(positions.keys())
            n = len(symbols)
            
            if n == 0:
                return 0.0
            
            total_variance = 0.0
            
            for i in range(n):
                symbol_i = symbols[i]
                weight_i = weights[symbol_i]
                vol_i = volatilities[symbol_i]
                
                for j in range(n):
                    symbol_j = symbols[j]
                    weight_j = weights[symbol_j]
                    vol_j = volatilities[symbol_j]
                    
                    if i == j:
                        correlation = 1.0
                    else:
                        correlation = self._get_correlation(symbol_i, symbol_j)
                    
                    total_variance += weight_i * weight_j * vol_i * vol_j * correlation
            
            # Portfolio volatility (risk)
            portfolio_volatility = np.sqrt(max(0, total_variance))
            
            return portfolio_volatility
            
        except Exception as e:
            logger.error("Error calculating correlation-adjusted heat", error=str(e))
            return 0.0
    
    def _calculate_risk_contribution(self, positions: Dict[str, Any], 
                                   portfolio_value: float) -> Dict[str, float]:
        """Calculate risk contribution from each position"""
        risk_contributions = {}
        
        try:
            total_portfolio_risk = 0.0
            
            # Calculate total portfolio risk
            for symbol, position in positions.items():
                position_value = position['size'] * position['current_price']
                volatility = self._get_volatility(symbol)
                position_risk = position_value * volatility
                total_portfolio_risk += position_risk
            
            # Calculate individual contributions
            for symbol, position in positions.items():
                position_value = position['size'] * position['current_price']
                volatility = self._get_volatility(symbol)
                position_risk = position_value * volatility
                
                if total_portfolio_risk > 0:
                    contribution = position_risk / total_portfolio_risk
                else:
                    contribution = 0.0
                
                risk_contributions[symbol] = contribution
            
            return risk_contributions
            
        except Exception as e:
            logger.error("Error calculating risk contribution", error=str(e))
            return {}
    
    def _calculate_concentration_risk(self, positions: Dict[str, Any], 
                                    portfolio_value: float) -> float:
        """Calculate concentration risk (Herfindahl-Hirschman Index)"""
        try:
            if not positions:
                return 0.0
            
            # Calculate position weights
            weights = []
            for symbol, position in positions.items():
                position_value = position['size'] * position['current_price']
                weight = position_value / portfolio_value
                weights.append(weight)
            
            # Calculate HHI
            hhi = sum(w ** 2 for w in weights)
            
            # Normalize to [0, 1] where 1 is maximum concentration
            n = len(positions)
            min_hhi = 1.0 / n  # Perfectly diversified
            max_hhi = 1.0      # Completely concentrated
            
            if max_hhi > min_hhi:
                normalized_concentration = (hhi - min_hhi) / (max_hhi - min_hhi)
            else:
                normalized_concentration = 0.0
            
            return max(0.0, min(1.0, normalized_concentration))
            
        except Exception as e:
            logger.error("Error calculating concentration risk", error=str(e))
            return 0.0
    
    def _calculate_diversification_ratio(self, positions: Dict[str, Any], 
                                       portfolio_value: float) -> float:
        """Calculate diversification ratio"""
        try:
            if not positions:
                return 0.0
            
            # Calculate weighted average volatility
            total_weighted_vol = 0.0
            total_weight = 0.0
            
            for symbol, position in positions.items():
                position_value = position['size'] * position['current_price']
                weight = position_value / portfolio_value
                volatility = self._get_volatility(symbol)
                
                total_weighted_vol += weight * volatility
                total_weight += weight
            
            if total_weight > 0:
                avg_weighted_vol = total_weighted_vol / total_weight
            else:
                avg_weighted_vol = 0.0
            
            # Calculate portfolio volatility
            portfolio_vol = self._calculate_correlation_adjusted_heat(positions, portfolio_value)
            
            # Diversification ratio
            if portfolio_vol > 0:
                diversification_ratio = avg_weighted_vol / portfolio_vol
            else:
                diversification_ratio = 1.0
            
            return max(1.0, diversification_ratio)  # Should be >= 1
            
        except Exception as e:
            logger.error("Error calculating diversification ratio", error=str(e))
            return 1.0
    
    def calculate_exposure_metrics(self, positions: Dict[str, Any], 
                                 portfolio_value: float) -> Dict[ExposureType, ExposureMetrics]:
        """Calculate various exposure metrics"""
        exposure_metrics = {}
        
        try:
            # Calculate different types of exposure
            exposure_metrics[ExposureType.GROSS] = self._calculate_gross_exposure(
                positions, portfolio_value)
            exposure_metrics[ExposureType.NET] = self._calculate_net_exposure(
                positions, portfolio_value)
            exposure_metrics[ExposureType.LONG] = self._calculate_long_exposure(
                positions, portfolio_value)
            exposure_metrics[ExposureType.SHORT] = self._calculate_short_exposure(
                positions, portfolio_value)
            exposure_metrics[ExposureType.SECTOR] = self._calculate_sector_exposure(
                positions, portfolio_value)
            exposure_metrics[ExposureType.ASSET] = self._calculate_asset_exposure(
                positions, portfolio_value)
            
            # Store in history
            self.exposure_history.append({
                'timestamp': datetime.now(),
                'gross_exposure': exposure_metrics[ExposureType.GROSS].current_exposure,
                'net_exposure': exposure_metrics[ExposureType.NET].current_exposure,
                'long_exposure': exposure_metrics[ExposureType.LONG].current_exposure,
                'short_exposure': exposure_metrics[ExposureType.SHORT].current_exposure
            })
            
            return exposure_metrics
            
        except Exception as e:
            logger.error("Error calculating exposure metrics", error=str(e))
            return {}
    
    def _calculate_gross_exposure(self, positions: Dict[str, Any], 
                                portfolio_value: float) -> ExposureMetrics:
        """Calculate gross exposure (sum of absolute position values)"""
        total_exposure = 0.0
        position_list = []
        
        for symbol, position in positions.items():
            position_value = abs(position['size'] * position['current_price'])
            total_exposure += position_value
            position_list.append({
                'symbol': symbol,
                'value': position_value,
                'weight': position_value / portfolio_value
            })
        
        exposure_ratio = total_exposure / portfolio_value
        limit = self.portfolio_config['exposure_limits']['max_gross_exposure']
        
        return ExposureMetrics(
            exposure_type=ExposureType.GROSS,
            current_exposure=exposure_ratio,
            limit=limit,
            utilization=exposure_ratio / limit,
            risk_level=self._get_risk_level(exposure_ratio, limit),
            positions=position_list
        )
    
    def _calculate_net_exposure(self, positions: Dict[str, Any], 
                              portfolio_value: float) -> ExposureMetrics:
        """Calculate net exposure (sum of signed position values)"""
        total_exposure = 0.0
        position_list = []
        
        for symbol, position in positions.items():
            position_value = position['size'] * position['current_price']
            total_exposure += position_value
            position_list.append({
                'symbol': symbol,
                'value': position_value,
                'weight': position_value / portfolio_value
            })
        
        exposure_ratio = abs(total_exposure) / portfolio_value
        limit = self.portfolio_config['exposure_limits']['max_net_exposure']
        
        return ExposureMetrics(
            exposure_type=ExposureType.NET,
            current_exposure=exposure_ratio,
            limit=limit,
            utilization=exposure_ratio / limit,
            risk_level=self._get_risk_level(exposure_ratio, limit),
            positions=position_list
        )
    
    def _calculate_long_exposure(self, positions: Dict[str, Any], 
                               portfolio_value: float) -> ExposureMetrics:
        """Calculate long exposure"""
        total_exposure = 0.0
        position_list = []
        
        for symbol, position in positions.items():
            if position['size'] > 0:
                position_value = position['size'] * position['current_price']
                total_exposure += position_value
                position_list.append({
                    'symbol': symbol,
                    'value': position_value,
                    'weight': position_value / portfolio_value
                })
        
        exposure_ratio = total_exposure / portfolio_value
        limit = self.portfolio_config['exposure_limits']['max_long_exposure']
        
        return ExposureMetrics(
            exposure_type=ExposureType.LONG,
            current_exposure=exposure_ratio,
            limit=limit,
            utilization=exposure_ratio / limit,
            risk_level=self._get_risk_level(exposure_ratio, limit),
            positions=position_list
        )
    
    def _calculate_short_exposure(self, positions: Dict[str, Any], 
                                portfolio_value: float) -> ExposureMetrics:
        """Calculate short exposure"""
        total_exposure = 0.0
        position_list = []
        
        for symbol, position in positions.items():
            if position['size'] < 0:
                position_value = abs(position['size'] * position['current_price'])
                total_exposure += position_value
                position_list.append({
                    'symbol': symbol,
                    'value': position_value,
                    'weight': position_value / portfolio_value
                })
        
        exposure_ratio = total_exposure / portfolio_value
        limit = self.portfolio_config['exposure_limits']['max_short_exposure']
        
        return ExposureMetrics(
            exposure_type=ExposureType.SHORT,
            current_exposure=exposure_ratio,
            limit=limit,
            utilization=exposure_ratio / limit,
            risk_level=self._get_risk_level(exposure_ratio, limit),
            positions=position_list
        )
    
    def _calculate_sector_exposure(self, positions: Dict[str, Any], 
                                 portfolio_value: float) -> ExposureMetrics:
        """Calculate sector exposure"""
        sector_exposures = {}
        
        for symbol, position in positions.items():
            sector = self._get_sector(symbol)
            position_value = abs(position['size'] * position['current_price'])
            
            if sector not in sector_exposures:
                sector_exposures[sector] = 0.0
            sector_exposures[sector] += position_value
        
        # Find maximum sector exposure
        max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0.0
        max_exposure_ratio = max_sector_exposure / portfolio_value
        
        limit = self.portfolio_config['concentration_limits']['max_sector_exposure']
        
        position_list = [
            {'sector': sector, 'value': value, 'weight': value / portfolio_value}
            for sector, value in sector_exposures.items()
        ]
        
        return ExposureMetrics(
            exposure_type=ExposureType.SECTOR,
            current_exposure=max_exposure_ratio,
            limit=limit,
            utilization=max_exposure_ratio / limit,
            risk_level=self._get_risk_level(max_exposure_ratio, limit),
            positions=position_list
        )
    
    def _calculate_asset_exposure(self, positions: Dict[str, Any], 
                                portfolio_value: float) -> ExposureMetrics:
        """Calculate single asset exposure"""
        max_asset_exposure = 0.0
        max_symbol = None
        position_list = []
        
        for symbol, position in positions.items():
            position_value = abs(position['size'] * position['current_price'])
            exposure_ratio = position_value / portfolio_value
            
            if exposure_ratio > max_asset_exposure:
                max_asset_exposure = exposure_ratio
                max_symbol = symbol
            
            position_list.append({
                'symbol': symbol,
                'value': position_value,
                'weight': exposure_ratio
            })
        
        limit = self.portfolio_config['concentration_limits']['max_single_asset_exposure']
        
        return ExposureMetrics(
            exposure_type=ExposureType.ASSET,
            current_exposure=max_asset_exposure,
            limit=limit,
            utilization=max_asset_exposure / limit,
            risk_level=self._get_risk_level(max_asset_exposure, limit),
            positions=position_list
        )
    
    def generate_rebalancing_recommendations(self, heat_metrics: PortfolioHeatMetrics,
                                           exposure_metrics: Dict[ExposureType, ExposureMetrics],
                                           portfolio_value: float) -> List[Dict[str, Any]]:
        """Generate rebalancing recommendations"""
        recommendations = []
        
        try:
            # Check if rebalancing is needed
            if heat_metrics.total_heat <= self.rebalance_threshold:
                return recommendations
            
            # Heat-based recommendations
            if heat_metrics.total_heat > self.max_heat_threshold:
                recommendations.append({
                    'type': 'reduce_heat',
                    'reason': f'Portfolio heat {heat_metrics.total_heat:.3f} exceeds threshold {self.max_heat_threshold:.3f}',
                    'priority': 'HIGH',
                    'actions': self._generate_heat_reduction_actions(heat_metrics)
                })
            
            # Exposure-based recommendations
            for exposure_type, metrics in exposure_metrics.items():
                if metrics.is_violation():
                    recommendations.append({
                        'type': 'reduce_exposure',
                        'exposure_type': exposure_type.value,
                        'reason': f'{exposure_type.value} exposure {metrics.current_exposure:.3f} exceeds limit {metrics.limit:.3f}',
                        'priority': 'HIGH',
                        'actions': self._generate_exposure_reduction_actions(metrics)
                    })
            
            # Concentration-based recommendations
            if heat_metrics.concentration_risk > 0.7:
                recommendations.append({
                    'type': 'diversify',
                    'reason': f'High concentration risk {heat_metrics.concentration_risk:.3f}',
                    'priority': 'MEDIUM',
                    'actions': self._generate_diversification_actions(heat_metrics)
                })
            
            # Store recommendations
            self.rebalance_recommendations.extend(recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error("Error generating rebalancing recommendations", error=str(e))
            return []
    
    def _generate_heat_reduction_actions(self, heat_metrics: PortfolioHeatMetrics) -> List[Dict[str, Any]]:
        """Generate actions to reduce portfolio heat"""
        actions = []
        
        # Sort positions by heat contribution
        sorted_positions = sorted(heat_metrics.individual_heat.items(), 
                                key=lambda x: x[1], reverse=True)
        
        # Recommend reducing top contributors
        for symbol, heat in sorted_positions[:3]:  # Top 3 contributors
            if heat > 0.02:  # 2% heat threshold
                actions.append({
                    'action': 'reduce_position',
                    'symbol': symbol,
                    'current_heat': heat,
                    'target_reduction': min(0.5, heat / self.max_heat_threshold)
                })
        
        return actions
    
    def _generate_exposure_reduction_actions(self, metrics: ExposureMetrics) -> List[Dict[str, Any]]:
        """Generate actions to reduce exposure"""
        actions = []
        
        # Sort positions by exposure
        sorted_positions = sorted(metrics.positions, key=lambda x: x['weight'], reverse=True)
        
        # Recommend reducing largest exposures
        for position in sorted_positions[:2]:  # Top 2 exposures
            actions.append({
                'action': 'reduce_position',
                'symbol': position.get('symbol', 'unknown'),
                'current_weight': position['weight'],
                'target_reduction': min(0.3, position['weight'] / metrics.limit)
            })
        
        return actions
    
    def _generate_diversification_actions(self, heat_metrics: PortfolioHeatMetrics) -> List[Dict[str, Any]]:
        """Generate actions to improve diversification"""
        actions = []
        
        if heat_metrics.diversification_ratio < 1.5:
            actions.append({
                'action': 'add_uncorrelated_assets',
                'reason': f'Low diversification ratio {heat_metrics.diversification_ratio:.2f}',
                'target_ratio': 2.0
            })
        
        return actions
    
    def _get_volatility(self, symbol: str) -> float:
        """Get volatility for symbol"""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        # Default volatility
        return 0.02  # 2% default
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between symbols"""
        if self.correlation_matrix is None:
            return 0.3  # Default correlation
        
        # Simplified correlation lookup
        return 0.3
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol"""
        return self.sector_mapping.get(symbol, 'unknown')
    
    def _get_risk_level(self, current: float, limit: float) -> str:
        """Get risk level based on utilization"""
        utilization = current / limit
        
        if utilization > 1.0:
            return "CRITICAL"
        elif utilization > 0.8:
            return "HIGH"
        elif utilization > 0.6:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_default_heat_metrics(self) -> PortfolioHeatMetrics:
        """Get default heat metrics for error cases"""
        return PortfolioHeatMetrics(
            total_heat=0.0,
            individual_heat={},
            correlation_adjusted_heat=0.0,
            risk_contribution={},
            concentration_risk=0.0,
            diversification_ratio=1.0
        )
    
    def get_heat_summary(self) -> Dict[str, Any]:
        """Get portfolio heat summary"""
        if not self.heat_history:
            return {'status': 'no_data'}
        
        recent_heat = self.heat_history[-1]
        
        return {
            'current_heat': recent_heat['total_heat'],
            'max_heat_threshold': self.max_heat_threshold,
            'risk_level': self._get_risk_level(recent_heat['total_heat'], self.max_heat_threshold),
            'correlation_adjusted_heat': recent_heat['correlation_adjusted_heat'],
            'concentration_risk': recent_heat['concentration_risk'],
            'diversification_ratio': recent_heat['diversification_ratio'],
            'rebalance_needed': recent_heat['total_heat'] > self.rebalance_threshold,
            'recommendations_count': len(self.rebalance_recommendations)
        }