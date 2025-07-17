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
            )\n            component_attributions[symbol] = attribution\n        \n        # Calculate risk factor summary\n        risk_factor_summary = self._calculate_risk_factor_summary(component_attributions)\n        \n        # Calculate concentration analysis\n        concentration_analysis = self._calculate_concentration_analysis(\n            component_attributions, portfolio_value, portfolio_var\n        )\n        \n        # Calculate diversification metrics\n        diversification_metrics = self._calculate_diversification_metrics(\n            component_attributions, correlation_matrix, portfolio_var\n        )\n        \n        # Calculate active risk metrics (if benchmark available)\n        active_risk_metrics = self._calculate_active_risk_metrics(\n            component_attributions, asset_returns\n        )\n        \n        return PortfolioRiskDecomposition(\n            timestamp=datetime.now(),\n            portfolio_value=portfolio_value,\n            total_portfolio_var=portfolio_var,\n            component_attributions=component_attributions,\n            risk_factor_summary=risk_factor_summary,\n            concentration_analysis=concentration_analysis,\n            diversification_metrics=diversification_metrics,\n            active_risk_metrics=active_risk_metrics\n        )\n    \n    def _calculate_component_attribution(\n        self,\n        symbol: str,\n        position_value: float,\n        portfolio_value: float,\n        asset_returns: np.ndarray,\n        volatility: float,\n        portfolio_var: float,\n        component_var: float,\n        marginal_var: float,\n        correlation_matrix: np.ndarray,\n        all_positions: Dict[str, float]\n    ) -> RiskAttribution:\n        \"\"\"Calculate detailed risk attribution for a single component\"\"\"\n        \n        position_weight = position_value / portfolio_value\n        \n        # Calculate VaR contribution percentage\n        var_contribution_pct = (component_var / portfolio_var) if portfolio_var > 0 else 0\n        \n        # Calculate risk efficiency (risk contribution vs position weight)\n        risk_efficiency = var_contribution_pct / position_weight if position_weight > 0 else 0\n        \n        # Calculate excess risk (over/under contribution relative to weight)\n        excess_risk = var_contribution_pct - position_weight\n        \n        # Calculate incremental VaR (approximate)\n        incremental_var = self._calculate_incremental_var(\n            symbol, position_value, portfolio_var, all_positions, correlation_matrix\n        )\n        \n        # Calculate beta-adjusted risk\n        beta_adjusted_risk = self._calculate_beta_adjusted_risk(\n            symbol, asset_returns, volatility, position_weight\n        )\n        \n        # Calculate factor contributions\n        factor_contributions = self._calculate_factor_contributions(\n            symbol, component_var, volatility, asset_returns\n        )\n        \n        # Calculate concentration metrics\n        concentration_metrics = self._calculate_position_concentration(\n            symbol, position_value, portfolio_value, component_var, portfolio_var\n        )\n        \n        return RiskAttribution(\n            symbol=symbol,\n            position_value=position_value,\n            position_weight=position_weight,\n            component_var=component_var,\n            marginal_var=marginal_var,\n            incremental_var=incremental_var,\n            var_contribution_pct=var_contribution_pct,\n            risk_efficiency=risk_efficiency,\n            excess_risk=excess_risk,\n            beta_adjusted_risk=beta_adjusted_risk,\n            factor_contributions=factor_contributions,\n            concentration_metrics=concentration_metrics\n        )\n    \n    def _calculate_incremental_var(\n        self,\n        symbol: str,\n        position_value: float,\n        portfolio_var: float,\n        all_positions: Dict[str, float],\n        correlation_matrix: np.ndarray\n    ) -> float:\n        \"\"\"Calculate incremental VaR (VaR change from removing position)\"\"\"\n        \n        # Simplified calculation - actual implementation would require\n        # full portfolio VaR recalculation without the position\n        position_weight = position_value / sum(all_positions.values())\n        \n        # Approximate incremental VaR as marginal VaR * position size\n        # This is a simplification - true incremental VaR requires full recalculation\n        incremental_var = portfolio_var * position_weight * 0.5  # Conservative estimate\n        \n        return incremental_var\n    \n    def _calculate_beta_adjusted_risk(\n        self,\n        symbol: str,\n        asset_returns: np.ndarray,\n        volatility: float,\n        position_weight: float\n    ) -> float:\n        \"\"\"Calculate beta-adjusted risk contribution\"\"\"\n        \n        if self.benchmark_returns is None or len(asset_returns) < 50:\n            # Fallback to volatility-based risk\n            return volatility * position_weight\n        \n        # Calculate beta relative to benchmark\n        min_length = min(len(asset_returns), len(self.benchmark_returns))\n        if min_length < 50:\n            return volatility * position_weight\n        \n        asset_rets = asset_returns[-min_length:]\n        bench_rets = self.benchmark_returns[-min_length:]\n        \n        # Calculate beta\n        covariance = np.cov(asset_rets, bench_rets)[0, 1]\n        benchmark_variance = np.var(bench_rets)\n        \n        if benchmark_variance > 0:\n            beta = covariance / benchmark_variance\n        else:\n            beta = 1.0\n        \n        # Beta-adjusted risk\n        systematic_risk = abs(beta) * volatility * position_weight\n        idiosyncratic_risk = volatility * position_weight * (1 - abs(beta))\n        \n        return systematic_risk + idiosyncratic_risk\n    \n    def _calculate_factor_contributions(\n        self,\n        symbol: str,\n        component_var: float,\n        volatility: float,\n        asset_returns: np.ndarray\n    ) -> Dict[RiskFactorType, float]:\n        \"\"\"Calculate risk factor contributions for the position\"\"\"\n        \n        factor_contributions = {}\n        \n        # Market factor (systematic risk)\n        if self.benchmark_returns is not None and len(asset_returns) >= 50:\n            market_factor = self._calculate_market_factor_contribution(\n                asset_returns, component_var\n            )\n            factor_contributions[RiskFactorType.MARKET] = market_factor\n            \n            # Specific risk (idiosyncratic)\n            factor_contributions[RiskFactorType.SPECIFIC] = component_var - market_factor\n        else:\n            # Fallback allocation\n            factor_contributions[RiskFactorType.MARKET] = component_var * 0.7\n            factor_contributions[RiskFactorType.SPECIFIC] = component_var * 0.3\n        \n        # Sector factor\n        sector = self.sector_mappings.get(symbol, 'Unknown')\n        if sector != 'Unknown':\n            # Estimate sector contribution (simplified)\n            sector_contribution = component_var * 0.2  # Assume 20% sector contribution\n            factor_contributions[RiskFactorType.SECTOR] = sector_contribution\n        \n        # Style factor\n        style_contribution = self._calculate_style_factor_contribution(symbol, component_var)\n        if style_contribution > 0:\n            factor_contributions[RiskFactorType.STYLE] = style_contribution\n        \n        # Volatility factor (based on volatility regime)\n        if volatility > 0.3:  # High volatility\n            factor_contributions[RiskFactorType.VOLATILITY] = component_var * 0.15\n        elif volatility < 0.15:  # Low volatility\n            factor_contributions[RiskFactorType.VOLATILITY] = component_var * 0.05\n        else:\n            factor_contributions[RiskFactorType.VOLATILITY] = component_var * 0.1\n        \n        return factor_contributions\n    \n    def _calculate_market_factor_contribution(\n        self,\n        asset_returns: np.ndarray,\n        component_var: float\n    ) -> float:\n        \"\"\"Calculate market factor contribution using beta\"\"\"\n        \n        if len(asset_returns) < 50 or len(self.benchmark_returns) < 50:\n            return component_var * 0.7  # Default assumption\n        \n        # Align lengths\n        min_length = min(len(asset_returns), len(self.benchmark_returns))\n        asset_rets = asset_returns[-min_length:]\n        bench_rets = self.benchmark_returns[-min_length:]\n        \n        # Calculate R-squared (explained variance)\n        correlation = np.corrcoef(asset_rets, bench_rets)[0, 1]\n        r_squared = correlation ** 2\n        \n        # Market factor contribution\n        market_contribution = component_var * r_squared\n        \n        return market_contribution\n    \n    def _calculate_style_factor_contribution(\n        self,\n        symbol: str,\n        component_var: float\n    ) -> float:\n        \"\"\"Calculate style factor contribution\"\"\"\n        \n        style_contribution = 0.0\n        \n        # Check style factor membership\n        for style, symbols in self.style_factors.items():\n            if symbol in symbols:\n                # Estimate style contribution based on factor loading\n                if style == 'growth':\n                    style_contribution = component_var * 0.15\n                elif style == 'value':\n                    style_contribution = component_var * 0.1\n                elif style == 'quality':\n                    style_contribution = component_var * 0.08\n                break\n        \n        return style_contribution\n    \n    def _calculate_position_concentration(\n        self,\n        symbol: str,\n        position_value: float,\n        portfolio_value: float,\n        component_var: float,\n        portfolio_var: float\n    ) -> Dict[str, float]:\n        \"\"\"Calculate position-specific concentration metrics\"\"\"\n        \n        position_weight = position_value / portfolio_value\n        risk_weight = component_var / portfolio_var if portfolio_var > 0 else 0\n        \n        concentration_metrics = {\n            'position_weight': position_weight,\n            'risk_weight': risk_weight,\n            'risk_concentration_ratio': risk_weight / position_weight if position_weight > 0 else 0,\n            'position_size_rank': 0,  # Would be calculated relative to other positions\n            'risk_contribution_rank': 0,  # Would be calculated relative to other positions\n            'concentration_warning': position_weight > 0.1 or risk_weight > 0.15\n        }\n        \n        return concentration_metrics\n    \n    def _calculate_risk_factor_summary(\n        self,\n        component_attributions: Dict[str, RiskAttribution]\n    ) -> Dict[RiskFactorType, float]:\n        \"\"\"Calculate portfolio-level risk factor summary\"\"\"\n        \n        risk_factor_summary = {}\n        \n        # Aggregate factor contributions across all positions\n        for factor_type in RiskFactorType:\n            total_contribution = 0.0\n            \n            for attribution in component_attributions.values():\n                factor_contribution = attribution.factor_contributions.get(factor_type, 0.0)\n                total_contribution += factor_contribution\n            \n            risk_factor_summary[factor_type] = total_contribution\n        \n        return risk_factor_summary\n    \n    def _calculate_concentration_analysis(\n        self,\n        component_attributions: Dict[str, RiskAttribution],\n        portfolio_value: float,\n        portfolio_var: float\n    ) -> Dict[str, float]:\n        \"\"\"Calculate portfolio concentration analysis\"\"\"\n        \n        position_weights = [attr.position_weight for attr in component_attributions.values()]\n        risk_weights = [attr.var_contribution_pct for attr in component_attributions.values()]\n        \n        concentration_analysis = {\n            'herfindahl_index_positions': sum(w**2 for w in position_weights),\n            'herfindahl_index_risk': sum(w**2 for w in risk_weights),\n            'effective_positions': 1 / sum(w**2 for w in position_weights) if position_weights else 0,\n            'effective_risk_positions': 1 / sum(w**2 for w in risk_weights) if risk_weights else 0,\n            'max_position_weight': max(position_weights) if position_weights else 0,\n            'max_risk_weight': max(risk_weights) if risk_weights else 0,\n            'top_5_position_concentration': sum(sorted(position_weights, reverse=True)[:5]),\n            'top_5_risk_concentration': sum(sorted(risk_weights, reverse=True)[:5]),\n            'concentration_ratio': sum(sorted(risk_weights, reverse=True)[:3]) / sum(sorted(position_weights, reverse=True)[:3]) if position_weights else 0\n        }\n        \n        return concentration_analysis\n    \n    def _calculate_diversification_metrics(\n        self,\n        component_attributions: Dict[str, RiskAttribution],\n        correlation_matrix: np.ndarray,\n        portfolio_var: float\n    ) -> Dict[str, float]:\n        \"\"\"Calculate portfolio diversification metrics\"\"\"\n        \n        position_weights = [attr.position_weight for attr in component_attributions.values()]\n        volatilities = [attr.component_var / (attr.position_weight * portfolio_var) \n                       for attr in component_attributions.values() if attr.position_weight > 0]\n        \n        if len(position_weights) < 2:\n            return {'diversification_ratio': 0.0, 'portfolio_efficiency': 0.0}\n        \n        # Calculate diversification ratio\n        weights = np.array(position_weights)\n        if len(volatilities) == len(weights) and correlation_matrix.shape[0] >= len(weights):\n            vols = np.array(volatilities)\n            \n            # Weighted average volatility\n            weighted_avg_vol = np.dot(weights, vols)\n            \n            # Portfolio volatility from correlation matrix\n            portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlation_matrix[:len(weights), :len(weights)], weights)))\n            \n            diversification_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0\n        else:\n            diversification_ratio = 1.0\n        \n        # Portfolio efficiency (1 - concentration)\n        herfindahl_index = sum(w**2 for w in position_weights)\n        portfolio_efficiency = 1 - herfindahl_index\n        \n        diversification_metrics = {\n            'diversification_ratio': diversification_ratio,\n            'portfolio_efficiency': portfolio_efficiency,\n            'effective_number_of_positions': len(position_weights),\n            'diversification_benefit': max(0, diversification_ratio - 1.0)\n        }\n        \n        return diversification_metrics\n    \n    def _calculate_active_risk_metrics(\n        self,\n        component_attributions: Dict[str, RiskAttribution],\n        asset_returns: Dict[str, np.ndarray]\n    ) -> Dict[str, float]:\n        \"\"\"Calculate active risk metrics (if benchmark available)\"\"\"\n        \n        if self.benchmark_returns is None:\n            return {'active_risk_analysis': 'No benchmark available'}\n        \n        # Calculate active risk contributions\n        active_risk_metrics = {\n            'total_active_positions': len(component_attributions),\n            'active_risk_contribution': 0.0,\n            'tracking_error_contribution': 0.0,\n            'information_ratio_estimate': 0.0\n        }\n        \n        total_active_contribution = 0.0\n        \n        for symbol, attribution in component_attributions.items():\n            if symbol in asset_returns and len(asset_returns[symbol]) >= 50:\n                # Calculate active contribution (simplified)\n                active_contribution = attribution.beta_adjusted_risk * 0.3  # Simplified\n                total_active_contribution += active_contribution\n        \n        active_risk_metrics['active_risk_contribution'] = total_active_contribution\n        \n        return active_risk_metrics\n    \n    def generate_risk_attribution_report(\n        self,\n        risk_decomposition: PortfolioRiskDecomposition\n    ) -> Dict:\n        \"\"\"Generate comprehensive risk attribution report\"\"\"\n        \n        # Top risk contributors\n        top_contributors = sorted(\n            risk_decomposition.component_attributions.items(),\n            key=lambda x: x[1].component_var,\n            reverse=True\n        )[:5]\n        \n        # Risk efficiency analysis\n        risk_efficient = []\n        risk_inefficient = []\n        \n        for symbol, attribution in risk_decomposition.component_attributions.items():\n            if attribution.risk_efficiency < 0.8:  # Under-contributing to risk\n                risk_efficient.append((symbol, attribution.risk_efficiency))\n            elif attribution.risk_efficiency > 1.2:  # Over-contributing to risk\n                risk_inefficient.append((symbol, attribution.risk_efficiency))\n        \n        # Concentration warnings\n        concentration_warnings = []\n        for symbol, attribution in risk_decomposition.component_attributions.items():\n            if attribution.concentration_metrics.get('concentration_warning', False):\n                concentration_warnings.append({\n                    'symbol': symbol,\n                    'position_weight': attribution.position_weight,\n                    'risk_weight': attribution.var_contribution_pct,\n                    'reason': 'High concentration risk'\n                })\n        \n        report = {\n            'report_timestamp': risk_decomposition.timestamp.isoformat(),\n            'portfolio_summary': {\n                'total_value': risk_decomposition.portfolio_value,\n                'total_var': risk_decomposition.total_portfolio_var,\n                'number_of_positions': len(risk_decomposition.component_attributions)\n            },\n            'top_risk_contributors': [\n                {\n                    'symbol': symbol,\n                    'component_var': attribution.component_var,\n                    'risk_contribution_pct': attribution.var_contribution_pct,\n                    'position_weight': attribution.position_weight\n                }\n                for symbol, attribution in top_contributors\n            ],\n            'risk_factor_breakdown': {\n                factor.value: contribution\n                for factor, contribution in risk_decomposition.risk_factor_summary.items()\n            },\n            'concentration_analysis': risk_decomposition.concentration_analysis,\n            'diversification_metrics': risk_decomposition.diversification_metrics,\n            'risk_efficiency': {\n                'efficient_positions': risk_efficient,\n                'inefficient_positions': risk_inefficient\n            },\n            'concentration_warnings': concentration_warnings,\n            'recommendations': self._generate_attribution_recommendations(risk_decomposition)\n        }\n        \n        return report\n    \n    def _generate_attribution_recommendations(\n        self,\n        risk_decomposition: PortfolioRiskDecomposition\n    ) -> List[str]:\n        \"\"\"Generate recommendations based on risk attribution analysis\"\"\"\n        \n        recommendations = []\n        \n        # Check concentration\n        concentration = risk_decomposition.concentration_analysis\n        if concentration['max_risk_weight'] > 0.3:\n            recommendations.append(f\"HIGH CONCENTRATION: Single position contributes {concentration['max_risk_weight']:.1%} of total risk\")\n        \n        # Check diversification\n        diversification = risk_decomposition.diversification_metrics\n        if diversification['diversification_ratio'] < 1.2:\n            recommendations.append(\"LOW DIVERSIFICATION: Consider adding uncorrelated positions\")\n        \n        # Check risk efficiency\n        inefficient_positions = [\n            (symbol, attr.risk_efficiency)\n            for symbol, attr in risk_decomposition.component_attributions.items()\n            if attr.risk_efficiency > 1.5\n        ]\n        \n        if inefficient_positions:\n            recommendations.append(f\"RISK INEFFICIENT: {len(inefficient_positions)} positions contribute disproportionate risk\")\n        \n        # Check factor concentration\n        factor_summary = risk_decomposition.risk_factor_summary\n        market_risk = factor_summary.get(RiskFactorType.MARKET, 0)\n        total_risk = sum(factor_summary.values())\n        \n        if market_risk / total_risk > 0.8:\n            recommendations.append(\"HIGH SYSTEMATIC RISK: Consider adding defensive positions\")\n        \n        if not recommendations:\n            recommendations.append(\"Portfolio demonstrates balanced risk attribution\")\n        \n        return recommendations


# Factory function for easy instantiation\ndef create_risk_attribution_analyzer(\n    benchmark_returns: Optional[np.ndarray] = None,\n    risk_free_rate: float = 0.02\n) -> RiskAttributionAnalyzer:\n    \"\"\"Create a risk attribution analyzer with default settings\"\"\"\n    return RiskAttributionAnalyzer(\n        benchmark_returns=benchmark_returns,\n        risk_free_rate=risk_free_rate,\n        enable_factor_models=True\n    )