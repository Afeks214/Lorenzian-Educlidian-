"""
Advanced Portfolio Risk Attribution Analysis

This module provides comprehensive risk attribution analysis for portfolio components,
including:
- Component VaR attribution
- Risk factor decomposition
- Concentration risk analysis
- Marginal risk contributions
- Risk-adjusted performance metrics

Author: Agent 16 - Risk Management Enhancement Specialist
Mission: Implement production-ready risk attribution framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import scipy.stats as stats
import structlog

logger = structlog.get_logger()


class RiskFactorType(Enum):
    """Types of risk factors for attribution"""
    MARKET = "market"
    SECTOR = "sector"
    STYLE = "style"
    CURRENCY = "currency"
    CREDIT = "credit"
    LIQUIDITY = "liquidity"
    VOLATILITY = "volatility"
    SPECIFIC = "specific"


@dataclass
class RiskAttribution:
    """Risk attribution result for a portfolio component"""
    symbol: str
    position_value: float
    position_weight: float
    component_var: float
    marginal_var: float
    incremental_var: float
    var_contribution_pct: float
    risk_efficiency: float
    excess_risk: float
    beta_adjusted_risk: float
    factor_contributions: Dict[RiskFactorType, float]
    concentration_metrics: Dict[str, float]


@dataclass
class PortfolioRiskDecomposition:
    """Complete portfolio risk decomposition"""
    timestamp: datetime
    portfolio_value: float
    total_portfolio_var: float
    component_attributions: Dict[str, RiskAttribution]
    risk_factor_summary: Dict[RiskFactorType, float]
    concentration_analysis: Dict[str, float]
    diversification_metrics: Dict[str, float]
    active_risk_metrics: Dict[str, float]


class RiskAttributionAnalyzer:
    """
    Advanced risk attribution analyzer for portfolio components.
    
    Provides comprehensive analysis of how individual positions and risk factors
    contribute to overall portfolio risk.
    """
    
    def __init__(
        self,
        benchmark_returns: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02,
        enable_factor_models: bool = True
    ):
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.enable_factor_models = enable_factor_models
        
        # Risk factor mappings (simplified for demo)
        self.sector_mappings = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology', 
            'MSFT': 'Technology',
            'TSLA': 'Automotive',
            'NVDA': 'Technology',
            'JPM': 'Financial',
            'JNJ': 'Healthcare',
            'PG': 'Consumer',
            'XOM': 'Energy',
            'GE': 'Industrial'
        }
        
        # Style factors (simplified)
        self.style_factors = {
            'growth': ['AAPL', 'GOOGL', 'TSLA', 'NVDA'],
            'value': ['JPM', 'XOM', 'GE'],
            'quality': ['MSFT', 'JNJ', 'PG']
        }
        
        logger.info("RiskAttributionAnalyzer initialized",
                   has_benchmark=benchmark_returns is not None,
                   risk_free_rate=risk_free_rate,
                   enable_factor_models=enable_factor_models)
    
    def analyze_portfolio_risk_attribution(
        self,
        portfolio_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        volatilities: Dict[str, float],
        portfolio_var: float,
        component_vars: Dict[str, float],
        marginal_vars: Dict[str, float]
    ) -> PortfolioRiskDecomposition:
        """
        Perform comprehensive portfolio risk attribution analysis.
        
        Args:
            portfolio_positions: Dict of {symbol: position_value}
            portfolio_value: Total portfolio value
            asset_returns: Historical returns for each asset
            correlation_matrix: Asset correlation matrix
            volatilities: Asset volatilities
            portfolio_var: Total portfolio VaR
            component_vars: Component VaR contributions
            marginal_vars: Marginal VaR contributions
        
        Returns:
            PortfolioRiskDecomposition with detailed attribution analysis
        """
        
        logger.info("Starting portfolio risk attribution analysis",
                   portfolio_value=portfolio_value,
                   num_positions=len(portfolio_positions))
        
        # Calculate individual attributions
        component_attributions = {}
        
        for symbol, position_value in portfolio_positions.items():
            attribution = self._calculate_component_attribution(
                symbol, position_value, portfolio_value, 
                asset_returns.get(symbol, np.array([])),
                volatilities.get(symbol, 0.2),
                portfolio_var, component_vars.get(symbol, 0.0),
                marginal_vars.get(symbol, 0.0),
                correlation_matrix, portfolio_positions
            )
            component_attributions[symbol] = attribution
        
        # Calculate risk factor summary
        risk_factor_summary = self._calculate_risk_factor_summary(component_attributions)
        
        # Calculate concentration analysis
        concentration_analysis = self._calculate_concentration_analysis(
            component_attributions, portfolio_value, portfolio_var
        )
        
        # Calculate diversification metrics
        diversification_metrics = self._calculate_diversification_metrics(
            component_attributions, correlation_matrix, portfolio_var
        )
        
        # Calculate active risk metrics (if benchmark available)
        active_risk_metrics = self._calculate_active_risk_metrics(
            component_attributions, asset_returns
        )
        
        return PortfolioRiskDecomposition(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            total_portfolio_var=portfolio_var,
            component_attributions=component_attributions,
            risk_factor_summary=risk_factor_summary,
            concentration_analysis=concentration_analysis,
            diversification_metrics=diversification_metrics,
            active_risk_metrics=active_risk_metrics
        )
    
    def _calculate_component_attribution(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        asset_returns: np.ndarray,
        volatility: float,
        portfolio_var: float,
        component_var: float,
        marginal_var: float,
        correlation_matrix: np.ndarray,
        all_positions: Dict[str, float]
    ) -> RiskAttribution:
        """Calculate detailed risk attribution for a single component"""
        
        position_weight = position_value / portfolio_value
        
        # Calculate VaR contribution percentage
        var_contribution_pct = (component_var / portfolio_var) if portfolio_var > 0 else 0
        
        # Calculate risk efficiency (risk contribution vs position weight)
        risk_efficiency = var_contribution_pct / position_weight if position_weight > 0 else 0
        
        # Calculate excess risk (over/under contribution relative to weight)
        excess_risk = var_contribution_pct - position_weight
        
        # Calculate incremental VaR (approximate)
        incremental_var = self._calculate_incremental_var(
            symbol, position_value, portfolio_var, all_positions, correlation_matrix
        )
        
        # Calculate beta-adjusted risk
        beta_adjusted_risk = self._calculate_beta_adjusted_risk(
            symbol, asset_returns, volatility, position_weight
        )
        
        # Calculate factor contributions
        factor_contributions = self._calculate_factor_contributions(
            symbol, component_var, volatility, asset_returns
        )
        
        # Calculate concentration metrics
        concentration_metrics = self._calculate_position_concentration(
            symbol, position_value, portfolio_value, component_var, portfolio_var
        )
        
        return RiskAttribution(
            symbol=symbol,
            position_value=position_value,
            position_weight=position_weight,
            component_var=component_var,
            marginal_var=marginal_var,
            incremental_var=incremental_var,
            var_contribution_pct=var_contribution_pct,
            risk_efficiency=risk_efficiency,
            excess_risk=excess_risk,
            beta_adjusted_risk=beta_adjusted_risk,
            factor_contributions=factor_contributions,
            concentration_metrics=concentration_metrics
        )
    
    def _calculate_incremental_var(
        self,
        symbol: str,
        position_value: float,
        portfolio_var: float,
        all_positions: Dict[str, float],
        correlation_matrix: np.ndarray
    ) -> float:
        """Calculate incremental VaR (VaR change from removing position)"""
        
        # Simplified calculation - actual implementation would require
        # full portfolio VaR recalculation without the position
        position_weight = position_value / sum(all_positions.values())
        
        # Approximate incremental VaR as marginal VaR * position size
        # This is a simplification - true incremental VaR requires full recalculation
        incremental_var = portfolio_var * position_weight * 0.5  # Conservative estimate
        
        return incremental_var
    
    def _calculate_beta_adjusted_risk(
        self,
        symbol: str,
        asset_returns: np.ndarray,
        volatility: float,
        position_weight: float
    ) -> float:
        """Calculate beta-adjusted risk contribution"""
        
        if self.benchmark_returns is None or len(asset_returns) < 50:
            # Fallback to volatility-based risk
            return volatility * position_weight
        
        # Calculate beta relative to benchmark
        min_length = min(len(asset_returns), len(self.benchmark_returns))
        if min_length < 50:
            return volatility * position_weight
        
        asset_rets = asset_returns[-min_length:]
        bench_rets = self.benchmark_returns[-min_length:]
        
        # Calculate beta
        covariance = np.cov(asset_rets, bench_rets)[0, 1]
        benchmark_variance = np.var(bench_rets)
        
        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
        else:
            beta = 1.0
        
        # Beta-adjusted risk
        systematic_risk = abs(beta) * volatility * position_weight
        idiosyncratic_risk = volatility * position_weight * (1 - abs(beta))
        
        return systematic_risk + idiosyncratic_risk
    
    def _calculate_factor_contributions(
        self,
        symbol: str,
        component_var: float,
        volatility: float,
        asset_returns: np.ndarray
    ) -> Dict[RiskFactorType, float]:
        """Calculate risk factor contributions for the position"""
        
        factor_contributions = {}
        
        # Market factor (systematic risk)
        if self.benchmark_returns is not None and len(asset_returns) >= 50:
            market_factor = self._calculate_market_factor_contribution(
                asset_returns, component_var
            )
            factor_contributions[RiskFactorType.MARKET] = market_factor
            
            # Specific risk (idiosyncratic)
            factor_contributions[RiskFactorType.SPECIFIC] = component_var - market_factor
        else:
            # Fallback allocation
            factor_contributions[RiskFactorType.MARKET] = component_var * 0.7
            factor_contributions[RiskFactorType.SPECIFIC] = component_var * 0.3
        
        # Sector factor
        sector = self.sector_mappings.get(symbol, 'Unknown')
        if sector != 'Unknown':
            # Estimate sector contribution (simplified)
            sector_contribution = component_var * 0.2  # Assume 20% sector contribution
            factor_contributions[RiskFactorType.SECTOR] = sector_contribution
        
        # Style factor
        style_contribution = self._calculate_style_factor_contribution(symbol, component_var)
        if style_contribution > 0:
            factor_contributions[RiskFactorType.STYLE] = style_contribution
        
        # Volatility factor (based on volatility regime)
        if volatility > 0.3:  # High volatility
            factor_contributions[RiskFactorType.VOLATILITY] = component_var * 0.15
        elif volatility < 0.15:  # Low volatility
            factor_contributions[RiskFactorType.VOLATILITY] = component_var * 0.05
        else:
            factor_contributions[RiskFactorType.VOLATILITY] = component_var * 0.1
        
        return factor_contributions
    
    def _calculate_market_factor_contribution(
        self,
        asset_returns: np.ndarray,
        component_var: float
    ) -> float:
        """Calculate market factor contribution using beta"""
        
        if len(asset_returns) < 50 or len(self.benchmark_returns) < 50:
            return component_var * 0.7  # Default assumption
        
        # Align lengths
        min_length = min(len(asset_returns), len(self.benchmark_returns))
        asset_rets = asset_returns[-min_length:]
        bench_rets = self.benchmark_returns[-min_length:]
        
        # Calculate R-squared (explained variance)
        correlation = np.corrcoef(asset_rets, bench_rets)[0, 1]
        r_squared = correlation ** 2
        
        # Market factor contribution
        market_contribution = component_var * r_squared
        
        return market_contribution
    
    def _calculate_style_factor_contribution(
        self,
        symbol: str,
        component_var: float
    ) -> float:
        """Calculate style factor contribution"""
        
        style_contribution = 0.0
        
        # Check style factor membership
        for style, symbols in self.style_factors.items():
            if symbol in symbols:
                # Estimate style contribution based on factor loading
                if style == 'growth':
                    style_contribution = component_var * 0.15
                elif style == 'value':
                    style_contribution = component_var * 0.1
                elif style == 'quality':
                    style_contribution = component_var * 0.08
                break
        
        return style_contribution
    
    def _calculate_position_concentration(
        self,
        symbol: str,
        position_value: float,
        portfolio_value: float,
        component_var: float,
        portfolio_var: float
    ) -> Dict[str, float]:
        """Calculate position-specific concentration metrics"""
        
        position_weight = position_value / portfolio_value
        risk_weight = component_var / portfolio_var if portfolio_var > 0 else 0
        
        concentration_metrics = {
            'position_weight': position_weight,
            'risk_weight': risk_weight,
            'risk_concentration_ratio': risk_weight / position_weight if position_weight > 0 else 0,
            'position_size_rank': 0,  # Would be calculated relative to other positions
            'risk_contribution_rank': 0,  # Would be calculated relative to other positions
            'concentration_warning': position_weight > 0.1 or risk_weight > 0.15
        }
        
        return concentration_metrics
    
    def _calculate_risk_factor_summary(
        self,
        component_attributions: Dict[str, RiskAttribution]
    ) -> Dict[RiskFactorType, float]:
        """Calculate portfolio-level risk factor summary"""
        
        risk_factor_summary = {}
        
        # Aggregate factor contributions across all positions
        for factor_type in RiskFactorType:
            total_contribution = 0.0
            
            for attribution in component_attributions.values():
                factor_contribution = attribution.factor_contributions.get(factor_type, 0.0)
                total_contribution += factor_contribution
            
            risk_factor_summary[factor_type] = total_contribution
        
        return risk_factor_summary
    
    def _calculate_concentration_analysis(
        self,
        component_attributions: Dict[str, RiskAttribution],
        portfolio_value: float,
        portfolio_var: float
    ) -> Dict[str, float]:
        """Calculate portfolio concentration analysis"""
        
        position_weights = [attr.position_weight for attr in component_attributions.values()]
        risk_weights = [attr.var_contribution_pct for attr in component_attributions.values()]
        
        concentration_analysis = {
            'herfindahl_index_positions': sum(w**2 for w in position_weights),
            'herfindahl_index_risk': sum(w**2 for w in risk_weights),
            'effective_positions': 1 / sum(w**2 for w in position_weights) if position_weights else 0,
            'effective_risk_positions': 1 / sum(w**2 for w in risk_weights) if risk_weights else 0,
            'max_position_weight': max(position_weights) if position_weights else 0,
            'max_risk_weight': max(risk_weights) if risk_weights else 0,
            'top_5_position_concentration': sum(sorted(position_weights, reverse=True)[:5]),
            'top_5_risk_concentration': sum(sorted(risk_weights, reverse=True)[:5]),
            'concentration_ratio': sum(sorted(risk_weights, reverse=True)[:3]) / sum(sorted(position_weights, reverse=True)[:3]) if position_weights else 0
        }
        
        return concentration_analysis
    
    def _calculate_diversification_metrics(
        self,
        component_attributions: Dict[str, RiskAttribution],
        correlation_matrix: np.ndarray,
        portfolio_var: float
    ) -> Dict[str, float]:
        """Calculate portfolio diversification metrics"""
        
        position_weights = [attr.position_weight for attr in component_attributions.values()]
        volatilities = [attr.component_var / (attr.position_weight * portfolio_var) 
                       for attr in component_attributions.values() if attr.position_weight > 0]
        
        if len(position_weights) < 2:
            return {'diversification_ratio': 0.0, 'portfolio_efficiency': 0.0}
        
        # Calculate diversification ratio
        weights = np.array(position_weights)
        if len(volatilities) == len(weights) and correlation_matrix.shape[0] >= len(weights):
            vols = np.array(volatilities)
            
            # Weighted average volatility
            weighted_avg_vol = np.dot(weights, vols)
            
            # Portfolio volatility from correlation matrix
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlation_matrix[:len(weights), :len(weights)], weights)))
            
            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        else:
            diversification_ratio = 1.0
        
        # Portfolio efficiency (1 - concentration)
        herfindahl_index = sum(w**2 for w in position_weights)
        portfolio_efficiency = 1 - herfindahl_index
        
        diversification_metrics = {
            'diversification_ratio': diversification_ratio,
            'portfolio_efficiency': portfolio_efficiency,
            'effective_number_of_positions': len(position_weights),
            'diversification_benefit': max(0, diversification_ratio - 1.0)
        }
        
        return diversification_metrics
    
    def _calculate_active_risk_metrics(
        self,
        component_attributions: Dict[str, RiskAttribution],
        asset_returns: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate active risk metrics (if benchmark available)"""
        
        if self.benchmark_returns is None:
            return {'active_risk_analysis': 'No benchmark available'}
        
        # Calculate active risk contributions
        active_risk_metrics = {
            'total_active_positions': len(component_attributions),
            'active_risk_contribution': 0.0,
            'tracking_error_contribution': 0.0,
            'information_ratio_estimate': 0.0
        }
        
        total_active_contribution = 0.0
        
        for symbol, attribution in component_attributions.items():
            if symbol in asset_returns and len(asset_returns[symbol]) >= 50:
                # Calculate active contribution (simplified)
                active_contribution = attribution.beta_adjusted_risk * 0.3  # Simplified
                total_active_contribution += active_contribution
        
        active_risk_metrics['active_risk_contribution'] = total_active_contribution
        
        return active_risk_metrics
    
    def generate_risk_attribution_report(
        self,
        risk_decomposition: PortfolioRiskDecomposition
    ) -> Dict:
        """Generate comprehensive risk attribution report"""
        
        # Top risk contributors
        top_contributors = sorted(
            risk_decomposition.component_attributions.items(),
            key=lambda x: x[1].component_var,
            reverse=True
        )[:5]
        
        # Risk efficiency analysis
        risk_efficient = []
        risk_inefficient = []
        
        for symbol, attribution in risk_decomposition.component_attributions.items():
            if attribution.risk_efficiency < 0.8:  # Under-contributing to risk
                risk_efficient.append((symbol, attribution.risk_efficiency))
            elif attribution.risk_efficiency > 1.2:  # Over-contributing to risk
                risk_inefficient.append((symbol, attribution.risk_efficiency))
        
        # Concentration warnings
        concentration_warnings = []
        for symbol, attribution in risk_decomposition.component_attributions.items():
            if attribution.concentration_metrics.get('concentration_warning', False):
                concentration_warnings.append({
                    'symbol': symbol,
                    'position_weight': attribution.position_weight,
                    'risk_weight': attribution.var_contribution_pct,
                    'reason': 'High concentration risk'
                })
        
        report = {
            'report_timestamp': risk_decomposition.timestamp.isoformat(),
            'portfolio_summary': {
                'total_value': risk_decomposition.portfolio_value,
                'total_var': risk_decomposition.total_portfolio_var,
                'number_of_positions': len(risk_decomposition.component_attributions)
            },
            'top_risk_contributors': [
                {
                    'symbol': symbol,
                    'component_var': attribution.component_var,
                    'risk_contribution_pct': attribution.var_contribution_pct,
                    'position_weight': attribution.position_weight
                }
                for symbol, attribution in top_contributors
            ],
            'risk_factor_breakdown': {
                factor.value: contribution
                for factor, contribution in risk_decomposition.risk_factor_summary.items()
            },
            'concentration_analysis': risk_decomposition.concentration_analysis,
            'diversification_metrics': risk_decomposition.diversification_metrics,
            'risk_efficiency': {
                'efficient_positions': risk_efficient,
                'inefficient_positions': risk_inefficient
            },
            'concentration_warnings': concentration_warnings,
            'recommendations': self._generate_attribution_recommendations(risk_decomposition)
        }
        
        return report
    
    def _generate_attribution_recommendations(
        self,
        risk_decomposition: PortfolioRiskDecomposition
    ) -> List[str]:
        """Generate recommendations based on risk attribution analysis"""
        
        recommendations = []
        
        # Check concentration
        concentration = risk_decomposition.concentration_analysis
        if concentration['max_risk_weight'] > 0.3:
            recommendations.append(f"HIGH CONCENTRATION: Single position contributes {concentration['max_risk_weight']:.1%} of total risk")
        
        # Check diversification
        diversification = risk_decomposition.diversification_metrics
        if diversification['diversification_ratio'] < 1.2:
            recommendations.append("LOW DIVERSIFICATION: Consider adding uncorrelated positions")
        
        # Check risk efficiency
        inefficient_positions = [
            (symbol, attr.risk_efficiency)
            for symbol, attr in risk_decomposition.component_attributions.items()
            if attr.risk_efficiency > 1.5
        ]
        
        if inefficient_positions:
            recommendations.append(f"RISK INEFFICIENT: {len(inefficient_positions)} positions contribute disproportionate risk")
        
        # Check factor concentration
        factor_summary = risk_decomposition.risk_factor_summary
        market_risk = factor_summary.get(RiskFactorType.MARKET, 0)
        total_risk = sum(factor_summary.values())
        
        if market_risk / total_risk > 0.8:
            recommendations.append("HIGH SYSTEMATIC RISK: Consider adding defensive positions")
        
        if not recommendations:
            recommendations.append("Portfolio demonstrates balanced risk attribution")
        
        return recommendations


# Factory function for easy instantiation
def create_risk_attribution_analyzer(
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02
) -> RiskAttributionAnalyzer:
    """Create a risk attribution analyzer with default settings"""
    return RiskAttributionAnalyzer(
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
        enable_factor_models=True
    )