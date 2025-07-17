"""
Factor Models for Alpha Attribution and Risk Analysis

This module implements various factor models including Fama-French models
for decomposing portfolio returns and analyzing alpha generation sources.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class FactorLoadings:
    """Factor loadings and statistics"""
    loadings: Dict[str, float]
    t_statistics: Dict[str, float]
    p_values: Dict[str, float]
    standard_errors: Dict[str, float]
    alpha: float
    alpha_t_stat: float
    alpha_p_value: float
    r_squared: float
    adjusted_r_squared: float
    residuals: np.ndarray
    fitted_values: np.ndarray


@dataclass
class FactorData:
    """Factor data container"""
    market_return: np.ndarray
    smb: np.ndarray  # Small minus Big
    hml: np.ndarray  # High minus Low
    rmw: np.ndarray  # Robust minus Weak (optional)
    cma: np.ndarray  # Conservative minus Aggressive (optional)
    momentum: np.ndarray  # Momentum factor (optional)
    dates: Optional[np.ndarray] = None
    risk_free_rate: Optional[np.ndarray] = None


class FactorModelAnalyzer:
    """
    Comprehensive factor model analysis for portfolio returns
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize factor model analyzer
        
        Args:
            significance_level: Statistical significance level
        """
        self.significance_level = significance_level
        self.results: Dict[str, Any] = {}
        
    def analyze_portfolio(
        self,
        portfolio_returns: np.ndarray,
        factor_data: FactorData,
        models: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze portfolio using multiple factor models
        
        Args:
            portfolio_returns: Portfolio return series
            factor_data: Factor data
            models: List of models to run ['capm', 'ff3', 'ff4', 'ff5']
            
        Returns:
            Dictionary with analysis results
        """
        if models is None:
            models = ['capm', 'ff3', 'ff4', 'ff5']
            
        analysis = {}
        
        # Prepare excess returns
        if factor_data.risk_free_rate is not None:
            excess_returns = portfolio_returns - factor_data.risk_free_rate
            excess_market = factor_data.market_return - factor_data.risk_free_rate
        else:
            excess_returns = portfolio_returns
            excess_market = factor_data.market_return
            
        # Run different factor models
        for model_name in models:
            try:
                if model_name == 'capm':
                    result = self._run_capm(excess_returns, excess_market)
                elif model_name == 'ff3':
                    result = self._run_fama_french_3factor(excess_returns, excess_market, factor_data)
                elif model_name == 'ff4':
                    result = self._run_fama_french_4factor(excess_returns, excess_market, factor_data)
                elif model_name == 'ff5':
                    result = self._run_fama_french_5factor(excess_returns, excess_market, factor_data)
                else:
                    continue
                    
                analysis[model_name] = result
                
            except Exception as e:
                logger.warning(f"Failed to run {model_name}: {str(e)}")
                analysis[model_name] = {'error': str(e)}
        
        # Model comparison
        analysis['model_comparison'] = self._compare_models(analysis)
        
        # Rolling factor analysis
        analysis['rolling_analysis'] = self._rolling_factor_analysis(
            excess_returns, excess_market, factor_data
        )
        
        # Factor timing analysis
        analysis['factor_timing'] = self._factor_timing_analysis(
            excess_returns, excess_market, factor_data
        )
        
        # Style analysis
        analysis['style_analysis'] = self._style_analysis(
            excess_returns, excess_market, factor_data
        )
        
        self.results = analysis
        return analysis
        
    def _run_capm(self, excess_returns: np.ndarray, excess_market: np.ndarray) -> FactorLoadings:
        """Run CAPM model"""
        # Ensure same length
        min_len = min(len(excess_returns), len(excess_market))
        excess_returns = excess_returns[:min_len]
        excess_market = excess_market[:min_len]
        
        # Create design matrix
        X = np.column_stack([np.ones(len(excess_market)), excess_market])
        
        # OLS regression
        model = OLS(excess_returns, X)
        results = model.fit()
        
        return FactorLoadings(
            loadings={'market_beta': results.params[1]},
            t_statistics={'market_beta': results.tvalues[1]},
            p_values={'market_beta': results.pvalues[1]},
            standard_errors={'market_beta': results.bse[1]},
            alpha=results.params[0],
            alpha_t_stat=results.tvalues[0],
            alpha_p_value=results.pvalues[0],
            r_squared=results.rsquared,
            adjusted_r_squared=results.rsquared_adj,
            residuals=results.resid,
            fitted_values=results.fittedvalues
        )
        
    def _run_fama_french_3factor(
        self, 
        excess_returns: np.ndarray, 
        excess_market: np.ndarray,
        factor_data: FactorData
    ) -> FactorLoadings:
        """Run Fama-French 3-factor model"""
        # Ensure same length
        min_len = min(len(excess_returns), len(excess_market), len(factor_data.smb), len(factor_data.hml))
        excess_returns = excess_returns[:min_len]
        excess_market = excess_market[:min_len]
        smb = factor_data.smb[:min_len]
        hml = factor_data.hml[:min_len]
        
        # Create design matrix
        X = np.column_stack([np.ones(len(excess_market)), excess_market, smb, hml])
        
        # OLS regression
        model = OLS(excess_returns, X)
        results = model.fit()
        
        factor_names = ['market_beta', 'size_loading', 'value_loading']
        
        return FactorLoadings(
            loadings=dict(zip(factor_names, results.params[1:])),
            t_statistics=dict(zip(factor_names, results.tvalues[1:])),
            p_values=dict(zip(factor_names, results.pvalues[1:])),
            standard_errors=dict(zip(factor_names, results.bse[1:])),
            alpha=results.params[0],
            alpha_t_stat=results.tvalues[0],
            alpha_p_value=results.pvalues[0],
            r_squared=results.rsquared,
            adjusted_r_squared=results.rsquared_adj,
            residuals=results.resid,
            fitted_values=results.fittedvalues
        )
        
    def _run_fama_french_4factor(
        self, 
        excess_returns: np.ndarray, 
        excess_market: np.ndarray,
        factor_data: FactorData
    ) -> FactorLoadings:
        """Run Fama-French 4-factor model (3-factor + momentum)"""
        if factor_data.momentum is None:
            # Generate synthetic momentum factor if not provided
            momentum = self._generate_momentum_factor(excess_market)
        else:
            momentum = factor_data.momentum
            
        # Ensure same length
        min_len = min(len(excess_returns), len(excess_market), len(factor_data.smb), 
                     len(factor_data.hml), len(momentum))
        excess_returns = excess_returns[:min_len]
        excess_market = excess_market[:min_len]
        smb = factor_data.smb[:min_len]
        hml = factor_data.hml[:min_len]
        momentum = momentum[:min_len]
        
        # Create design matrix
        X = np.column_stack([np.ones(len(excess_market)), excess_market, smb, hml, momentum])
        
        # OLS regression
        model = OLS(excess_returns, X)
        results = model.fit()
        
        factor_names = ['market_beta', 'size_loading', 'value_loading', 'momentum_loading']
        
        return FactorLoadings(
            loadings=dict(zip(factor_names, results.params[1:])),
            t_statistics=dict(zip(factor_names, results.tvalues[1:])),
            p_values=dict(zip(factor_names, results.pvalues[1:])),
            standard_errors=dict(zip(factor_names, results.bse[1:])),
            alpha=results.params[0],
            alpha_t_stat=results.tvalues[0],
            alpha_p_value=results.pvalues[0],
            r_squared=results.rsquared,
            adjusted_r_squared=results.rsquared_adj,
            residuals=results.resid,
            fitted_values=results.fittedvalues
        )
        
    def _run_fama_french_5factor(
        self, 
        excess_returns: np.ndarray, 
        excess_market: np.ndarray,
        factor_data: FactorData
    ) -> FactorLoadings:
        """Run Fama-French 5-factor model"""
        # Generate synthetic factors if not provided
        if factor_data.rmw is None:
            rmw = self._generate_profitability_factor(excess_market)
        else:
            rmw = factor_data.rmw
            
        if factor_data.cma is None:
            cma = self._generate_investment_factor(excess_market)
        else:
            cma = factor_data.cma
            
        # Ensure same length
        min_len = min(len(excess_returns), len(excess_market), len(factor_data.smb), 
                     len(factor_data.hml), len(rmw), len(cma))
        excess_returns = excess_returns[:min_len]
        excess_market = excess_market[:min_len]
        smb = factor_data.smb[:min_len]
        hml = factor_data.hml[:min_len]
        rmw = rmw[:min_len]
        cma = cma[:min_len]
        
        # Create design matrix
        X = np.column_stack([np.ones(len(excess_market)), excess_market, smb, hml, rmw, cma])
        
        # OLS regression
        model = OLS(excess_returns, X)
        results = model.fit()
        
        factor_names = ['market_beta', 'size_loading', 'value_loading', 'profitability_loading', 'investment_loading']
        
        return FactorLoadings(
            loadings=dict(zip(factor_names, results.params[1:])),
            t_statistics=dict(zip(factor_names, results.tvalues[1:])),
            p_values=dict(zip(factor_names, results.pvalues[1:])),
            standard_errors=dict(zip(factor_names, results.bse[1:])),
            alpha=results.params[0],
            alpha_t_stat=results.tvalues[0],
            alpha_p_value=results.pvalues[0],
            r_squared=results.rsquared,
            adjusted_r_squared=results.rsquared_adj,
            residuals=results.resid,
            fitted_values=results.fittedvalues
        )
        
    def _generate_momentum_factor(self, market_returns: np.ndarray) -> np.ndarray:
        """Generate synthetic momentum factor"""
        # Simple momentum: 12-month return minus 1-month return
        window_long = 252  # ~12 months
        window_short = 21  # ~1 month
        
        momentum = np.zeros_like(market_returns)
        
        for i in range(window_long, len(market_returns)):
            long_return = np.prod(1 + market_returns[i-window_long:i-window_short]) - 1
            short_return = np.prod(1 + market_returns[i-window_short:i]) - 1
            momentum[i] = long_return - short_return
            
        return momentum
        
    def _generate_profitability_factor(self, market_returns: np.ndarray) -> np.ndarray:
        """Generate synthetic profitability factor (RMW)"""
        # Simple proxy: rolling correlation with market volatility
        window = 60
        rmw = np.zeros_like(market_returns)
        
        for i in range(window, len(market_returns)):
            vol = np.std(market_returns[i-window:i])
            rmw[i] = -vol + np.mean(market_returns[i-window:i])  # Profitability proxy
            
        return rmw
        
    def _generate_investment_factor(self, market_returns: np.ndarray) -> np.ndarray:
        """Generate synthetic investment factor (CMA)"""
        # Simple proxy: negative momentum (conservative vs aggressive)
        window = 126  # ~6 months
        cma = np.zeros_like(market_returns)
        
        for i in range(window, len(market_returns)):
            momentum = np.prod(1 + market_returns[i-window:i]) - 1
            cma[i] = -momentum  # Conservative vs aggressive proxy
            
        return cma
        
    def _compare_models(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different factor models"""
        comparison = {}
        
        # Extract key metrics
        models = [name for name in analysis.keys() if name != 'model_comparison']
        
        if not models:
            return {'error': 'No models to compare'}
            
        # R-squared comparison
        r_squared = {}
        adj_r_squared = {}
        alpha_significance = {}
        
        for model_name in models:
            if 'error' not in analysis[model_name]:
                result = analysis[model_name]
                r_squared[model_name] = result.r_squared
                adj_r_squared[model_name] = result.adjusted_r_squared
                alpha_significance[model_name] = result.alpha_p_value < self.significance_level
                
        comparison['r_squared'] = r_squared
        comparison['adjusted_r_squared'] = adj_r_squared
        comparison['alpha_significant'] = alpha_significance
        
        # Best model selection
        if adj_r_squared:
            best_model = max(adj_r_squared, key=adj_r_squared.get)
            comparison['best_model'] = best_model
            comparison['best_adj_r_squared'] = adj_r_squared[best_model]
            
        # Information criteria (AIC/BIC approximation)
        aic_scores = {}
        bic_scores = {}
        
        for model_name in models:
            if 'error' not in analysis[model_name]:
                result = analysis[model_name]
                n = len(result.residuals)
                k = len(result.loadings) + 1  # +1 for alpha
                
                # RSS-based approximation
                rss = np.sum(result.residuals ** 2)
                aic = n * np.log(rss / n) + 2 * k
                bic = n * np.log(rss / n) + k * np.log(n)
                
                aic_scores[model_name] = aic
                bic_scores[model_name] = bic
                
        comparison['aic_scores'] = aic_scores
        comparison['bic_scores'] = bic_scores
        
        # Best model by AIC/BIC
        if aic_scores:
            comparison['best_model_aic'] = min(aic_scores, key=aic_scores.get)
        if bic_scores:
            comparison['best_model_bic'] = min(bic_scores, key=bic_scores.get)
            
        return comparison
        
    def _rolling_factor_analysis(
        self,
        excess_returns: np.ndarray,
        excess_market: np.ndarray,
        factor_data: FactorData,
        window: int = 252
    ) -> Dict[str, Any]:
        """Rolling factor exposure analysis"""
        rolling_results = {}
        
        # Rolling beta analysis
        rolling_betas = []
        rolling_alphas = []
        rolling_r_squared = []
        
        for i in range(window, len(excess_returns)):
            window_returns = excess_returns[i-window:i]
            window_market = excess_market[i-window:i]
            
            try:
                # Simple CAPM for rolling analysis
                X = np.column_stack([np.ones(len(window_market)), window_market])
                model = OLS(window_returns, X)
                results = model.fit()
                
                rolling_alphas.append(results.params[0])
                rolling_betas.append(results.params[1])
                rolling_r_squared.append(results.rsquared)
                
            except (ValueError, TypeError, AttributeError) as e:
                rolling_alphas.append(np.nan)
                rolling_betas.append(np.nan)
                rolling_r_squared.append(np.nan)
                
        rolling_results['beta'] = {
            'values': np.array(rolling_betas),
            'mean': np.nanmean(rolling_betas),
            'std': np.nanstd(rolling_betas),
            'stability': 1 - np.nanstd(rolling_betas) / np.nanmean(np.abs(rolling_betas)) if np.nanmean(np.abs(rolling_betas)) > 0 else 0
        }
        
        rolling_results['alpha'] = {
            'values': np.array(rolling_alphas),
            'mean': np.nanmean(rolling_alphas),
            'std': np.nanstd(rolling_alphas),
            'consistency': np.nanmean(np.array(rolling_alphas) > 0)
        }
        
        rolling_results['r_squared'] = {
            'values': np.array(rolling_r_squared),
            'mean': np.nanmean(rolling_r_squared),
            'std': np.nanstd(rolling_r_squared)
        }
        
        return rolling_results
        
    def _factor_timing_analysis(
        self,
        excess_returns: np.ndarray,
        excess_market: np.ndarray,
        factor_data: FactorData
    ) -> Dict[str, Any]:
        """Analyze factor timing ability"""
        timing_results = {}
        
        # Market timing analysis (Treynor-Mazuy)
        # R_p - R_f = alpha + beta * (R_m - R_f) + gamma * (R_m - R_f)^2
        
        min_len = min(len(excess_returns), len(excess_market))
        excess_returns = excess_returns[:min_len]
        excess_market = excess_market[:min_len]
        
        # Create design matrix with squared market term
        X = np.column_stack([
            np.ones(len(excess_market)),
            excess_market,
            excess_market ** 2
        ])
        
        try:
            model = OLS(excess_returns, X)
            results = model.fit()
            
            timing_results['market_timing'] = {
                'alpha': results.params[0],
                'beta': results.params[1],
                'gamma': results.params[2],  # Timing coefficient
                'gamma_t_stat': results.tvalues[2],
                'gamma_p_value': results.pvalues[2],
                'has_timing_ability': results.pvalues[2] < self.significance_level,
                'r_squared': results.rsquared
            }
            
        except Exception as e:
            timing_results['market_timing'] = {'error': str(e)}
            
        # Factor timing for SMB and HML
        if len(factor_data.smb) >= len(excess_returns) and len(factor_data.hml) >= len(excess_returns):
            smb = factor_data.smb[:min_len]
            hml = factor_data.hml[:min_len]
            
            # SMB timing
            try:
                X_smb = np.column_stack([
                    np.ones(len(smb)),
                    excess_market,
                    smb,
                    smb * excess_market  # Interaction term
                ])
                
                model_smb = OLS(excess_returns, X_smb)
                results_smb = model_smb.fit()
                
                timing_results['size_timing'] = {
                    'interaction_coef': results_smb.params[3],
                    'interaction_t_stat': results_smb.tvalues[3],
                    'interaction_p_value': results_smb.pvalues[3],
                    'has_timing_ability': results_smb.pvalues[3] < self.significance_level
                }
                
            except Exception as e:
                timing_results['size_timing'] = {'error': str(e)}
                
            # HML timing
            try:
                X_hml = np.column_stack([
                    np.ones(len(hml)),
                    excess_market,
                    hml,
                    hml * excess_market  # Interaction term
                ])
                
                model_hml = OLS(excess_returns, X_hml)
                results_hml = model_hml.fit()
                
                timing_results['value_timing'] = {
                    'interaction_coef': results_hml.params[3],
                    'interaction_t_stat': results_hml.tvalues[3],
                    'interaction_p_value': results_hml.pvalues[3],
                    'has_timing_ability': results_hml.pvalues[3] < self.significance_level
                }
                
            except Exception as e:
                timing_results['value_timing'] = {'error': str(e)}
                
        return timing_results
        
    def _style_analysis(
        self,
        excess_returns: np.ndarray,
        excess_market: np.ndarray,
        factor_data: FactorData
    ) -> Dict[str, Any]:
        """Return-based style analysis"""
        style_results = {}
        
        # Create factor matrix
        factors = [excess_market]
        factor_names = ['market']
        
        if len(factor_data.smb) >= len(excess_returns):
            factors.append(factor_data.smb[:len(excess_returns)])
            factor_names.append('size')
            
        if len(factor_data.hml) >= len(excess_returns):
            factors.append(factor_data.hml[:len(excess_returns)])
            factor_names.append('value')
            
        if len(factors) < 2:
            return {'error': 'Insufficient factor data for style analysis'}
            
        # Constrained optimization (weights sum to 1, no short selling)
        X = np.column_stack(factors)
        
        def objective(weights):
            predicted = X @ weights
            return np.sum((excess_returns - predicted) ** 2)
            
        # Constraints: weights sum to 1, all weights >= 0
        from scipy.optimize import minimize
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
        ]
        bounds = [(0, 1) for _ in range(len(factors))]  # No short selling
        
        try:
            result = minimize(
                objective,
                x0=np.ones(len(factors)) / len(factors),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                style_weights = dict(zip(factor_names, result.x))
                
                # Calculate tracking error
                predicted = X @ result.x
                tracking_error = np.std(excess_returns - predicted)
                r_squared = 1 - np.var(excess_returns - predicted) / np.var(excess_returns)
                
                style_results['weights'] = style_weights
                style_results['tracking_error'] = tracking_error
                style_results['r_squared'] = r_squared
                style_results['success'] = True
                
                # Style classification
                if 'size' in style_weights:
                    if style_weights['size'] > 0.3:
                        style_results['size_tilt'] = 'small_cap'
                    elif style_weights['size'] < -0.3:
                        style_results['size_tilt'] = 'large_cap'
                    else:
                        style_results['size_tilt'] = 'neutral'
                        
                if 'value' in style_weights:
                    if style_weights['value'] > 0.3:
                        style_results['value_tilt'] = 'value'
                    elif style_weights['value'] < -0.3:
                        style_results['value_tilt'] = 'growth'
                    else:
                        style_results['value_tilt'] = 'neutral'
                        
            else:
                style_results['error'] = 'Optimization failed'
                
        except Exception as e:
            style_results['error'] = str(e)
            
        return style_results
        
    def create_custom_factor_model(
        self,
        portfolio_returns: np.ndarray,
        custom_factors: Dict[str, np.ndarray],
        factor_names: List[str] = None
    ) -> FactorLoadings:
        """
        Create custom factor model with user-defined factors
        
        Args:
            portfolio_returns: Portfolio return series
            custom_factors: Dictionary of factor name -> factor values
            factor_names: Optional list of factor names to use
            
        Returns:
            FactorLoadings object with results
        """
        if factor_names is None:
            factor_names = list(custom_factors.keys())
            
        # Create factor matrix
        factors = []
        for name in factor_names:
            if name in custom_factors:
                factors.append(custom_factors[name])
            else:
                raise ValueError(f"Factor {name} not found in custom_factors")
                
        # Ensure same length
        min_len = min(len(portfolio_returns), *[len(f) for f in factors])
        portfolio_returns = portfolio_returns[:min_len]
        factors = [f[:min_len] for f in factors]
        
        # Create design matrix
        X = np.column_stack([np.ones(len(portfolio_returns))] + factors)
        
        # OLS regression
        model = OLS(portfolio_returns, X)
        results = model.fit()
        
        return FactorLoadings(
            loadings=dict(zip(factor_names, results.params[1:])),
            t_statistics=dict(zip(factor_names, results.tvalues[1:])),
            p_values=dict(zip(factor_names, results.pvalues[1:])),
            standard_errors=dict(zip(factor_names, results.bse[1:])),
            alpha=results.params[0],
            alpha_t_stat=results.tvalues[0],
            alpha_p_value=results.pvalues[0],
            r_squared=results.rsquared,
            adjusted_r_squared=results.rsquared_adj,
            residuals=results.resid,
            fitted_values=results.fittedvalues
        )
        
    def generate_factor_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive factor analysis report"""
        if not self.results:
            return "No factor analysis results available"
            
        report = []
        report.append("=" * 60)
        report.append("FACTOR MODEL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model comparison
        if 'model_comparison' in self.results:
            report.append("MODEL COMPARISON")
            report.append("-" * 40)
            comparison = self.results['model_comparison']
            
            if 'adjusted_r_squared' in comparison:
                report.append("Adjusted R-squared by Model:")
                for model, r2 in comparison['adjusted_r_squared'].items():
                    report.append(f"  {model.upper()}: {r2:.4f}")
                report.append("")
                
            if 'best_model' in comparison:
                report.append(f"Best Model (Adj R²): {comparison['best_model'].upper()}")
                report.append(f"Best Adj R²: {comparison['best_adj_r_squared']:.4f}")
                report.append("")
                
        # Individual model results
        models = [name for name in self.results.keys() if name not in ['model_comparison', 'rolling_analysis', 'factor_timing', 'style_analysis']]
        
        for model_name in models:
            if 'error' not in self.results[model_name]:
                report.append(f"{model_name.upper()} MODEL RESULTS")
                report.append("-" * 40)
                
                result = self.results[model_name]
                
                # Alpha
                report.append(f"Alpha: {result.alpha:.4f}")
                report.append(f"Alpha t-stat: {result.alpha_t_stat:.3f}")
                report.append(f"Alpha p-value: {result.alpha_p_value:.4f}")
                report.append(f"Alpha significant: {result.alpha_p_value < self.significance_level}")
                report.append("")
                
                # Factor loadings
                report.append("Factor Loadings:")
                for factor, loading in result.loadings.items():
                    t_stat = result.t_statistics[factor]
                    p_val = result.p_values[factor]
                    significant = p_val < self.significance_level
                    
                    report.append(f"  {factor}: {loading:.4f} (t={t_stat:.3f}, p={p_val:.4f}, sig={significant})")
                
                report.append("")
                report.append(f"R-squared: {result.r_squared:.4f}")
                report.append(f"Adjusted R-squared: {result.adjusted_r_squared:.4f}")
                report.append("")
                
        # Rolling analysis
        if 'rolling_analysis' in self.results:
            report.append("ROLLING FACTOR ANALYSIS")
            report.append("-" * 40)
            rolling = self.results['rolling_analysis']
            
            if 'beta' in rolling:
                beta = rolling['beta']
                report.append(f"Beta Stability: {beta['stability']:.3f}")
                report.append(f"Average Beta: {beta['mean']:.3f}")
                report.append(f"Beta Volatility: {beta['std']:.3f}")
                report.append("")
                
            if 'alpha' in rolling:
                alpha = rolling['alpha']
                report.append(f"Alpha Consistency: {alpha['consistency']:.3f}")
                report.append(f"Average Alpha: {alpha['mean']:.4f}")
                report.append("")
                
        # Factor timing
        if 'factor_timing' in self.results:
            report.append("FACTOR TIMING ANALYSIS")
            report.append("-" * 40)
            timing = self.results['factor_timing']
            
            if 'market_timing' in timing and 'error' not in timing['market_timing']:
                mt = timing['market_timing']
                report.append(f"Market Timing Ability: {mt['has_timing_ability']}")
                report.append(f"Timing Coefficient: {mt['gamma']:.4f}")
                report.append(f"Timing p-value: {mt['gamma_p_value']:.4f}")
                report.append("")
                
        # Style analysis
        if 'style_analysis' in self.results:
            report.append("STYLE ANALYSIS")
            report.append("-" * 40)
            style = self.results['style_analysis']
            
            if 'weights' in style:
                report.append("Style Weights:")
                for factor, weight in style['weights'].items():
                    report.append(f"  {factor}: {weight:.3f}")
                report.append("")
                
                if 'size_tilt' in style:
                    report.append(f"Size Tilt: {style['size_tilt']}")
                if 'value_tilt' in style:
                    report.append(f"Value Tilt: {style['value_tilt']}")
                    
                report.append(f"Style R-squared: {style['r_squared']:.4f}")
                report.append(f"Tracking Error: {style['tracking_error']:.4f}")
                report.append("")
                
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text


def create_synthetic_factor_data(
    market_returns: np.ndarray,
    n_periods: int = None
) -> FactorData:
    """
    Create synthetic factor data for testing
    
    Args:
        market_returns: Market return series
        n_periods: Number of periods (defaults to length of market_returns)
        
    Returns:
        FactorData object with synthetic factors
    """
    if n_periods is None:
        n_periods = len(market_returns)
        
    # Generate correlated factors
    np.random.seed(42)  # For reproducibility
    
    # SMB (Size): negatively correlated with market
    smb = -0.3 * market_returns + np.random.normal(0, 0.02, n_periods)
    
    # HML (Value): low correlation with market
    hml = 0.1 * market_returns + np.random.normal(0, 0.015, n_periods)
    
    # Momentum: short-term reversal pattern
    momentum = np.zeros(n_periods)
    for i in range(21, n_periods):
        momentum[i] = -np.mean(market_returns[i-21:i]) + np.random.normal(0, 0.01)
        
    # RMW (Profitability): positive correlation with market
    rmw = 0.4 * market_returns + np.random.normal(0, 0.012, n_periods)
    
    # CMA (Investment): negative correlation with market
    cma = -0.2 * market_returns + np.random.normal(0, 0.008, n_periods)
    
    # Risk-free rate (constant)
    risk_free_rate = np.full(n_periods, 0.02 / 252)  # 2% annual risk-free rate
    
    return FactorData(
        market_return=market_returns,
        smb=smb,
        hml=hml,
        rmw=rmw,
        cma=cma,
        momentum=momentum,
        risk_free_rate=risk_free_rate,
        dates=np.arange(n_periods)
    )


def main():
    """Example factor model analysis"""
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    np.random.seed(42)
    n_periods = 1000
    
    # Synthetic market returns
    market_returns = np.random.normal(0.0005, 0.02, n_periods)
    
    # Create factor data
    factor_data = create_synthetic_factor_data(market_returns, n_periods)
    
    # Generate synthetic portfolio returns with factor exposures
    portfolio_returns = (
        0.001 +  # Alpha
        0.8 * (market_returns - factor_data.risk_free_rate) +  # Market beta
        0.3 * factor_data.smb +  # Size tilt
        -0.2 * factor_data.hml +  # Growth tilt
        np.random.normal(0, 0.01, n_periods)  # Idiosyncratic risk
    )
    
    # Initialize analyzer
    analyzer = FactorModelAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_portfolio(
        portfolio_returns,
        factor_data,
        models=['capm', 'ff3', 'ff4', 'ff5']
    )
    
    # Generate report
    report = analyzer.generate_factor_report("factor_analysis_report.txt")
    print(report)
    
    # Test custom factor model
    custom_factors = {
        'market': factor_data.market_return - factor_data.risk_free_rate,
        'size': factor_data.smb,
        'value': factor_data.hml
    }
    
    custom_result = analyzer.create_custom_factor_model(
        portfolio_returns,
        custom_factors,
        ['market', 'size', 'value']
    )
    
    print("\nCustom Factor Model Results:")
    print(f"Alpha: {custom_result.alpha:.4f} (p={custom_result.alpha_p_value:.4f})")
    print(f"Factor loadings: {custom_result.loadings}")
    print(f"R-squared: {custom_result.r_squared:.4f}")


if __name__ == "__main__":
    main()