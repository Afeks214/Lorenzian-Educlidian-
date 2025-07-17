"""
Regime Analysis for Market State Detection and Alpha Validation

This module implements Markov regime switching models for detecting market regimes
and analyzing regime-specific alpha generation.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize, differential_evolution
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class RegimeResults:
    """Results from regime analysis"""
    n_regimes: int
    regime_probabilities: np.ndarray
    regime_assignments: np.ndarray
    transition_matrix: np.ndarray
    regime_statistics: Dict[int, Dict[str, float]]
    log_likelihood: float
    aic: float
    bic: float
    regime_persistence: Dict[int, float]
    alpha_by_regime: Dict[int, float]
    model_type: str


@dataclass
class RegimeComparison:
    """Comparison between different regime models"""
    model_results: Dict[str, RegimeResults]
    best_model: str
    model_selection_criteria: Dict[str, Dict[str, float]]


class MarkovRegimeSwitching:
    """
    Markov Regime Switching Model for financial time series
    """
    
    def __init__(self, n_regimes: int = 2, max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize Markov Regime Switching model
        
        Args:
            n_regimes: Number of regimes
            max_iter: Maximum iterations for EM algorithm
            tol: Convergence tolerance
        """
        self.n_regimes = n_regimes
        self.max_iter = max_iter
        self.tol = tol
        self.is_fitted = False
        
        # Model parameters
        self.transition_matrix = None
        self.regime_means = None
        self.regime_variances = None
        self.initial_probs = None
        self.filtered_probs = None
        self.smoothed_probs = None
        self.log_likelihood = None
        
    def fit(self, returns: np.ndarray, verbose: bool = False) -> 'MarkovRegimeSwitching':
        """
        Fit Markov Regime Switching model using EM algorithm
        
        Args:
            returns: Time series of returns
            verbose: Print iteration details
            
        Returns:
            Fitted model
        """
        n_obs = len(returns)
        
        # Initialize parameters
        self._initialize_parameters(returns)
        
        # EM algorithm
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Forward-backward algorithm
            self._forward_backward(returns)
            
            # M-step: Update parameters
            self._update_parameters(returns)
            
            # Check convergence
            if abs(self.log_likelihood - prev_log_likelihood) < self.tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
                
            prev_log_likelihood = self.log_likelihood
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration + 1}: Log-likelihood = {self.log_likelihood:.6f}")
                
        self.is_fitted = True
        return self
        
    def _initialize_parameters(self, returns: np.ndarray):
        """Initialize model parameters"""
        n_obs = len(returns)
        
        # Use k-means clustering for initialization
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        initial_assignments = kmeans.fit_predict(returns.reshape(-1, 1))
        
        # Initialize regime means and variances
        self.regime_means = np.zeros(self.n_regimes)
        self.regime_variances = np.ones(self.n_regimes)
        
        for regime in range(self.n_regimes):
            regime_data = returns[initial_assignments == regime]
            if len(regime_data) > 0:
                self.regime_means[regime] = np.mean(regime_data)
                self.regime_variances[regime] = np.var(regime_data)
            else:
                self.regime_means[regime] = np.mean(returns)
                self.regime_variances[regime] = np.var(returns)
                
        # Initialize transition matrix (slightly favoring persistence)
        self.transition_matrix = np.full((self.n_regimes, self.n_regimes), 0.1)
        np.fill_diagonal(self.transition_matrix, 0.8)
        
        # Normalize transition matrix
        self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
        
        # Initialize initial probabilities
        self.initial_probs = np.ones(self.n_regimes) / self.n_regimes
        
        # Initialize probability arrays
        self.filtered_probs = np.zeros((n_obs, self.n_regimes))
        self.smoothed_probs = np.zeros((n_obs, self.n_regimes))
        
    def _forward_backward(self, returns: np.ndarray):
        """Forward-backward algorithm for E-step"""
        n_obs = len(returns)
        
        # Forward pass
        log_alpha = np.zeros((n_obs, self.n_regimes))
        
        # Initialize
        for regime in range(self.n_regimes):
            log_alpha[0, regime] = (np.log(self.initial_probs[regime]) + 
                                  self._log_normal_pdf(returns[0], self.regime_means[regime], 
                                                     self.regime_variances[regime]))
        
        # Forward recursion
        for t in range(1, n_obs):
            for regime in range(self.n_regimes):
                log_alpha[t, regime] = (logsumexp(log_alpha[t-1, :] + 
                                                np.log(self.transition_matrix[:, regime])) + 
                                      self._log_normal_pdf(returns[t], self.regime_means[regime], 
                                                         self.regime_variances[regime]))
        
        # Backward pass
        log_beta = np.zeros((n_obs, self.n_regimes))
        
        # Initialize (log(1) = 0)
        log_beta[n_obs-1, :] = 0
        
        # Backward recursion
        for t in range(n_obs-2, -1, -1):
            for regime in range(self.n_regimes):
                log_beta[t, regime] = logsumexp(
                    np.log(self.transition_matrix[regime, :]) + 
                    log_beta[t+1, :] + 
                    np.array([self._log_normal_pdf(returns[t+1], self.regime_means[j], 
                                                 self.regime_variances[j]) for j in range(self.n_regimes)])
                )
        
        # Compute filtered and smoothed probabilities
        log_gamma = log_alpha + log_beta
        
        # Normalize probabilities
        for t in range(n_obs):
            log_gamma[t, :] = log_gamma[t, :] - logsumexp(log_gamma[t, :])
            self.filtered_probs[t, :] = np.exp(log_alpha[t, :] - logsumexp(log_alpha[t, :]))
            self.smoothed_probs[t, :] = np.exp(log_gamma[t, :])
        
        # Compute log-likelihood
        self.log_likelihood = logsumexp(log_alpha[n_obs-1, :])
        
    def _update_parameters(self, returns: np.ndarray):
        """M-step: Update parameters"""
        n_obs = len(returns)
        
        # Update regime means and variances
        for regime in range(self.n_regimes):
            regime_weights = self.smoothed_probs[:, regime]
            total_weight = np.sum(regime_weights)
            
            if total_weight > 0:
                self.regime_means[regime] = np.sum(regime_weights * returns) / total_weight
                self.regime_variances[regime] = (np.sum(regime_weights * (returns - self.regime_means[regime])**2) / 
                                               total_weight)
            
        # Update transition matrix
        xi = np.zeros((self.n_regimes, self.n_regimes))
        
        for t in range(n_obs-1):
            for i in range(self.n_regimes):
                for j in range(self.n_regimes):
                    xi[i, j] += (self.filtered_probs[t, i] * 
                               self.transition_matrix[i, j] * 
                               self._normal_pdf(returns[t+1], self.regime_means[j], self.regime_variances[j]) * 
                               np.exp(self._compute_log_beta(returns, t+1, j)))
        
        # Normalize transition matrix
        for i in range(self.n_regimes):
            row_sum = np.sum(xi[i, :])
            if row_sum > 0:
                self.transition_matrix[i, :] = xi[i, :] / row_sum
        
        # Update initial probabilities
        self.initial_probs = self.smoothed_probs[0, :]
        
    def _compute_log_beta(self, returns: np.ndarray, t: int, regime: int) -> float:
        """Compute log beta value for specific time and regime"""
        n_obs = len(returns)
        
        if t == n_obs - 1:
            return 0.0
            
        log_beta = 0.0
        for j in range(self.n_regimes):
            log_beta += (np.log(self.transition_matrix[regime, j]) + 
                        self._log_normal_pdf(returns[t+1], self.regime_means[j], self.regime_variances[j]))
        
        return log_beta
        
    def _log_normal_pdf(self, x: float, mean: float, variance: float) -> float:
        """Log normal probability density function"""
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * (x - mean)**2 / variance
        
    def _normal_pdf(self, x: float, mean: float, variance: float) -> float:
        """Normal probability density function"""
        return np.exp(self._log_normal_pdf(x, mean, variance))
        
    def predict_regime(self, returns: np.ndarray) -> np.ndarray:
        """Predict most likely regime for each observation"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        return np.argmax(self.smoothed_probs, axis=1)
        
    def get_regime_probabilities(self) -> np.ndarray:
        """Get smoothed regime probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting probabilities")
            
        return self.smoothed_probs
        
    def compute_information_criteria(self, n_obs: int) -> Tuple[float, float]:
        """Compute AIC and BIC"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing information criteria")
            
        # Number of parameters: means + variances + transition matrix + initial probs
        n_params = (self.n_regimes * 2 +  # means and variances
                   self.n_regimes * (self.n_regimes - 1) +  # transition matrix (constrained)
                   self.n_regimes - 1)  # initial probabilities (constrained)
        
        aic = -2 * self.log_likelihood + 2 * n_params
        bic = -2 * self.log_likelihood + n_params * np.log(n_obs)
        
        return aic, bic


class RegimeAnalyzer:
    """
    Comprehensive regime analysis for financial time series
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize regime analyzer
        
        Args:
            significance_level: Statistical significance level
        """
        self.significance_level = significance_level
        self.results: Dict[str, Any] = {}
        
    def analyze_regimes(
        self,
        returns: np.ndarray,
        baseline_returns: np.ndarray = None,
        max_regimes: int = 4,
        methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive regime analysis
        
        Args:
            returns: Portfolio returns
            baseline_returns: Baseline returns for comparison
            max_regimes: Maximum number of regimes to test
            methods: List of methods ['markov', 'gaussian_mixture', 'volatility_clustering']
            
        Returns:
            Dictionary with regime analysis results
        """
        if methods is None:
            methods = ['markov', 'gaussian_mixture', 'volatility_clustering']
            
        analysis = {}
        
        # 1. Markov Regime Switching
        if 'markov' in methods:
            analysis['markov'] = self._analyze_markov_regimes(returns, max_regimes)
            
        # 2. Gaussian Mixture Models
        if 'gaussian_mixture' in methods:
            analysis['gaussian_mixture'] = self._analyze_gaussian_mixture(returns, max_regimes)
            
        # 3. Volatility clustering regimes
        if 'volatility_clustering' in methods:
            analysis['volatility_clustering'] = self._analyze_volatility_regimes(returns, max_regimes)
            
        # 4. Regime-specific alpha analysis
        if baseline_returns is not None:
            analysis['alpha_by_regime'] = self._analyze_regime_alpha(returns, baseline_returns, analysis)
            
        # 5. Regime transition analysis
        analysis['transition_analysis'] = self._analyze_regime_transitions(analysis)
        
        # 6. Model comparison
        analysis['model_comparison'] = self._compare_regime_models(analysis)
        
        # 7. Regime forecasting
        analysis['regime_forecasting'] = self._forecast_regimes(returns, analysis)
        
        self.results = analysis
        return analysis
        
    def _analyze_markov_regimes(self, returns: np.ndarray, max_regimes: int) -> Dict[str, Any]:
        """Analyze using Markov Regime Switching models"""
        markov_results = {}
        
        for n_regimes in range(2, max_regimes + 1):
            try:
                # Fit Markov model
                model = MarkovRegimeSwitching(n_regimes=n_regimes)
                model.fit(returns)
                
                # Compute information criteria
                aic, bic = model.compute_information_criteria(len(returns))
                
                # Get regime assignments
                regime_assignments = model.predict_regime(returns)
                regime_probabilities = model.get_regime_probabilities()
                
                # Compute regime statistics
                regime_statistics = {}
                for regime in range(n_regimes):
                    regime_mask = regime_assignments == regime
                    if np.any(regime_mask):
                        regime_returns = returns[regime_mask]
                        regime_statistics[regime] = {
                            'mean': np.mean(regime_returns),
                            'std': np.std(regime_returns),
                            'skewness': stats.skew(regime_returns),
                            'kurtosis': stats.kurtosis(regime_returns),
                            'count': len(regime_returns),
                            'frequency': len(regime_returns) / len(returns)
                        }
                
                # Compute regime persistence
                regime_persistence = {}
                for regime in range(n_regimes):
                    regime_persistence[regime] = model.transition_matrix[regime, regime]
                
                markov_results[f'{n_regimes}_regimes'] = RegimeResults(
                    n_regimes=n_regimes,
                    regime_probabilities=regime_probabilities,
                    regime_assignments=regime_assignments,
                    transition_matrix=model.transition_matrix,
                    regime_statistics=regime_statistics,
                    log_likelihood=model.log_likelihood,
                    aic=aic,
                    bic=bic,
                    regime_persistence=regime_persistence,
                    alpha_by_regime={},  # Will be filled later
                    model_type='markov'
                )
                
            except Exception as e:
                logger.warning(f"Failed to fit Markov model with {n_regimes} regimes: {str(e)}")
                markov_results[f'{n_regimes}_regimes'] = {'error': str(e)}
                
        return markov_results
        
    def _analyze_gaussian_mixture(self, returns: np.ndarray, max_regimes: int) -> Dict[str, Any]:
        """Analyze using Gaussian Mixture Models"""
        gm_results = {}
        
        # Create feature matrix (returns and volatility)
        rolling_vol = pd.Series(returns).rolling(window=20).std().fillna(method='bfill')
        features = np.column_stack([returns, rolling_vol])
        
        for n_regimes in range(2, max_regimes + 1):
            try:
                # Fit Gaussian Mixture Model
                gm = GaussianMixture(n_components=n_regimes, random_state=42)
                gm.fit(features)
                
                # Get regime assignments and probabilities
                regime_assignments = gm.predict(features)
                regime_probabilities = gm.predict_proba(features)
                
                # Compute regime statistics
                regime_statistics = {}
                for regime in range(n_regimes):
                    regime_mask = regime_assignments == regime
                    if np.any(regime_mask):
                        regime_returns = returns[regime_mask]
                        regime_statistics[regime] = {
                            'mean': np.mean(regime_returns),
                            'std': np.std(regime_returns),
                            'skewness': stats.skew(regime_returns),
                            'kurtosis': stats.kurtosis(regime_returns),
                            'count': len(regime_returns),
                            'frequency': len(regime_returns) / len(returns)
                        }
                
                # Approximate transition matrix
                transition_matrix = self._estimate_transition_matrix(regime_assignments, n_regimes)
                
                # Compute regime persistence
                regime_persistence = {}
                for regime in range(n_regimes):
                    regime_persistence[regime] = transition_matrix[regime, regime]
                
                gm_results[f'{n_regimes}_regimes'] = RegimeResults(
                    n_regimes=n_regimes,
                    regime_probabilities=regime_probabilities,
                    regime_assignments=regime_assignments,
                    transition_matrix=transition_matrix,
                    regime_statistics=regime_statistics,
                    log_likelihood=gm.score(features) * len(features),
                    aic=gm.aic(features),
                    bic=gm.bic(features),
                    regime_persistence=regime_persistence,
                    alpha_by_regime={},  # Will be filled later
                    model_type='gaussian_mixture'
                )
                
            except Exception as e:
                logger.warning(f"Failed to fit Gaussian Mixture with {n_regimes} regimes: {str(e)}")
                gm_results[f'{n_regimes}_regimes'] = {'error': str(e)}
                
        return gm_results
        
    def _analyze_volatility_regimes(self, returns: np.ndarray, max_regimes: int) -> Dict[str, Any]:
        """Analyze volatility-based regimes"""
        vol_results = {}
        
        # Compute rolling volatility
        rolling_vol = pd.Series(returns).rolling(window=20).std().fillna(method='bfill')
        
        for n_regimes in range(2, max_regimes + 1):
            try:
                # Use quantile-based regime classification
                quantiles = np.linspace(0, 1, n_regimes + 1)
                regime_thresholds = np.quantile(rolling_vol, quantiles)
                
                # Assign regimes based on volatility
                regime_assignments = np.digitize(rolling_vol, regime_thresholds[1:-1])
                
                # Create probability matrix (deterministic assignment)
                regime_probabilities = np.zeros((len(returns), n_regimes))
                for i, regime in enumerate(regime_assignments):
                    regime_probabilities[i, regime] = 1.0
                
                # Compute regime statistics
                regime_statistics = {}
                for regime in range(n_regimes):
                    regime_mask = regime_assignments == regime
                    if np.any(regime_mask):
                        regime_returns = returns[regime_mask]
                        regime_statistics[regime] = {
                            'mean': np.mean(regime_returns),
                            'std': np.std(regime_returns),
                            'skewness': stats.skew(regime_returns),
                            'kurtosis': stats.kurtosis(regime_returns),
                            'count': len(regime_returns),
                            'frequency': len(regime_returns) / len(returns),
                            'volatility_threshold': regime_thresholds[regime+1] if regime < n_regimes-1 else np.inf
                        }
                
                # Estimate transition matrix
                transition_matrix = self._estimate_transition_matrix(regime_assignments, n_regimes)
                
                # Compute regime persistence
                regime_persistence = {}
                for regime in range(n_regimes):
                    regime_persistence[regime] = transition_matrix[regime, regime]
                
                # Approximate log-likelihood (simplified)
                log_likelihood = -0.5 * len(returns) * np.log(2 * np.pi * np.var(returns))
                
                vol_results[f'{n_regimes}_regimes'] = RegimeResults(
                    n_regimes=n_regimes,
                    regime_probabilities=regime_probabilities,
                    regime_assignments=regime_assignments,
                    transition_matrix=transition_matrix,
                    regime_statistics=regime_statistics,
                    log_likelihood=log_likelihood,
                    aic=-2 * log_likelihood + 2 * n_regimes,
                    bic=-2 * log_likelihood + n_regimes * np.log(len(returns)),
                    regime_persistence=regime_persistence,
                    alpha_by_regime={},  # Will be filled later
                    model_type='volatility_clustering'
                )
                
            except Exception as e:
                logger.warning(f"Failed to analyze volatility regimes with {n_regimes} regimes: {str(e)}")
                vol_results[f'{n_regimes}_regimes'] = {'error': str(e)}
                
        return vol_results
        
    def _estimate_transition_matrix(self, regime_assignments: np.ndarray, n_regimes: int) -> np.ndarray:
        """Estimate transition matrix from regime assignments"""
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regime_assignments) - 1):
            current_regime = regime_assignments[i]
            next_regime = regime_assignments[i + 1]
            transition_matrix[current_regime, next_regime] += 1
            
        # Normalize rows
        for i in range(n_regimes):
            row_sum = np.sum(transition_matrix[i, :])
            if row_sum > 0:
                transition_matrix[i, :] /= row_sum
            else:
                transition_matrix[i, i] = 1.0  # Self-transition if no data
                
        return transition_matrix
        
    def _analyze_regime_alpha(
        self,
        returns: np.ndarray,
        baseline_returns: np.ndarray,
        regime_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze alpha generation by regime"""
        alpha_analysis = {}
        
        # Ensure same length
        min_len = min(len(returns), len(baseline_returns))
        returns = returns[:min_len]
        baseline_returns = baseline_returns[:min_len]
        excess_returns = returns - baseline_returns
        
        # Analyze alpha for each method and regime count
        for method_name, method_results in regime_analysis.items():
            if method_name in ['markov', 'gaussian_mixture', 'volatility_clustering']:
                alpha_analysis[method_name] = {}
                
                for regime_key, regime_result in method_results.items():
                    if 'error' not in regime_result:
                        regime_assignments = regime_result.regime_assignments[:min_len]
                        n_regimes = regime_result.n_regimes
                        
                        regime_alphas = {}
                        for regime in range(n_regimes):
                            regime_mask = regime_assignments == regime
                            if np.any(regime_mask):
                                regime_excess = excess_returns[regime_mask]
                                
                                # Statistical tests for alpha
                                if len(regime_excess) > 10:  # Minimum sample size
                                    t_stat, p_value = stats.ttest_1samp(regime_excess, 0)
                                    
                                    regime_alphas[regime] = {
                                        'alpha': np.mean(regime_excess),
                                        'alpha_std': np.std(regime_excess),
                                        't_statistic': t_stat,
                                        'p_value': p_value,
                                        'significant': p_value < self.significance_level,
                                        'observations': len(regime_excess),
                                        'sharpe_ratio': np.mean(regime_excess) / np.std(regime_excess) * np.sqrt(252) if np.std(regime_excess) > 0 else 0
                                    }
                                else:
                                    regime_alphas[regime] = {
                                        'alpha': np.mean(regime_excess) if len(regime_excess) > 0 else 0,
                                        'alpha_std': np.std(regime_excess) if len(regime_excess) > 0 else 0,
                                        't_statistic': np.nan,
                                        'p_value': np.nan,
                                        'significant': False,
                                        'observations': len(regime_excess),
                                        'sharpe_ratio': 0
                                    }
                        
                        alpha_analysis[method_name][regime_key] = regime_alphas
                        
                        # Update the original regime result
                        regime_result.alpha_by_regime = regime_alphas
                        
        return alpha_analysis
        
    def _analyze_regime_transitions(self, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime transition patterns"""
        transition_analysis = {}
        
        for method_name, method_results in regime_analysis.items():
            if method_name in ['markov', 'gaussian_mixture', 'volatility_clustering']:
                transition_analysis[method_name] = {}
                
                for regime_key, regime_result in method_results.items():
                    if 'error' not in regime_result:
                        transition_matrix = regime_result.transition_matrix
                        n_regimes = regime_result.n_regimes
                        
                        # Transition analysis
                        analysis = {
                            'persistence': {},
                            'transition_probabilities': transition_matrix.tolist(),
                            'expected_duration': {},
                            'ergodic_probabilities': {}
                        }
                        
                        # Calculate persistence and expected duration
                        for regime in range(n_regimes):
                            persistence = transition_matrix[regime, regime]
                            analysis['persistence'][regime] = persistence
                            
                            # Expected duration = 1 / (1 - persistence)
                            if persistence < 1:
                                analysis['expected_duration'][regime] = 1 / (1 - persistence)
                            else:
                                analysis['expected_duration'][regime] = float('inf')
                                
                        # Calculate ergodic probabilities (steady-state distribution)
                        try:
                            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
                            stationary_idx = np.argmax(np.real(eigenvalues))
                            stationary_dist = np.real(eigenvectors[:, stationary_idx])
                            stationary_dist = stationary_dist / np.sum(stationary_dist)
                            
                            for regime in range(n_regimes):
                                analysis['ergodic_probabilities'][regime] = stationary_dist[regime]
                                
                        except (ValueError, TypeError, AttributeError) as e:
                            # Fallback: use empirical frequencies
                            regime_counts = np.bincount(regime_result.regime_assignments, minlength=n_regimes)
                            empirical_dist = regime_counts / np.sum(regime_counts)
                            
                            for regime in range(n_regimes):
                                analysis['ergodic_probabilities'][regime] = empirical_dist[regime]
                                
                        transition_analysis[method_name][regime_key] = analysis
                        
        return transition_analysis
        
    def _compare_regime_models(self, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different regime models"""
        comparison = {}
        
        # Extract information criteria for each method
        for method_name, method_results in regime_analysis.items():
            if method_name in ['markov', 'gaussian_mixture', 'volatility_clustering']:
                comparison[method_name] = {}
                
                for regime_key, regime_result in method_results.items():
                    if 'error' not in regime_result:
                        comparison[method_name][regime_key] = {
                            'aic': regime_result.aic,
                            'bic': regime_result.bic,
                            'log_likelihood': regime_result.log_likelihood,
                            'n_regimes': regime_result.n_regimes
                        }
                        
        # Find best models
        best_models = {}
        
        for criterion in ['aic', 'bic']:
            best_score = float('inf')
            best_model = None
            
            for method_name, method_comparison in comparison.items():
                for regime_key, scores in method_comparison.items():
                    if scores[criterion] < best_score:
                        best_score = scores[criterion]
                        best_model = f"{method_name}_{regime_key}"
                        
            best_models[f'best_{criterion}'] = {
                'model': best_model,
                'score': best_score
            }
            
        comparison['best_models'] = best_models
        
        return comparison
        
    def _forecast_regimes(self, returns: np.ndarray, regime_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast future regimes"""
        forecast_results = {}
        
        # Use the best Markov model for forecasting
        if 'markov' in regime_analysis:
            best_markov = None
            best_bic = float('inf')
            
            for regime_key, regime_result in regime_analysis['markov'].items():
                if 'error' not in regime_result and regime_result.bic < best_bic:
                    best_bic = regime_result.bic
                    best_markov = regime_result
                    
            if best_markov is not None:
                # Current regime probabilities
                current_probs = best_markov.regime_probabilities[-1, :]
                
                # Forecast next period probabilities
                forecast_probs = current_probs @ best_markov.transition_matrix
                
                # Multi-step forecasts
                multi_step_forecasts = {}
                current_forecast = current_probs
                
                for horizon in [1, 5, 10, 20]:
                    for _ in range(horizon):
                        current_forecast = current_forecast @ best_markov.transition_matrix
                    multi_step_forecasts[f'{horizon}_step'] = current_forecast.copy()
                    
                forecast_results['markov'] = {
                    'next_period_probabilities': forecast_probs,
                    'multi_step_forecasts': multi_step_forecasts,
                    'most_likely_regime': np.argmax(forecast_probs),
                    'regime_confidence': np.max(forecast_probs)
                }
                
        return forecast_results
        
    def generate_regime_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive regime analysis report"""
        if not self.results:
            return "No regime analysis results available"
            
        report = []
        report.append("=" * 60)
        report.append("REGIME ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model comparison
        if 'model_comparison' in self.results:
            report.append("MODEL COMPARISON")
            report.append("-" * 40)
            comparison = self.results['model_comparison']
            
            if 'best_models' in comparison:
                best_models = comparison['best_models']
                report.append(f"Best Model (AIC): {best_models['best_aic']['model']}")
                report.append(f"AIC Score: {best_models['best_aic']['score']:.2f}")
                report.append(f"Best Model (BIC): {best_models['best_bic']['model']}")
                report.append(f"BIC Score: {best_models['best_bic']['score']:.2f}")
                report.append("")
                
        # Regime-specific alpha analysis
        if 'alpha_by_regime' in self.results:
            report.append("REGIME-SPECIFIC ALPHA ANALYSIS")
            report.append("-" * 40)
            
            for method_name, method_results in self.results['alpha_by_regime'].items():
                report.append(f"{method_name.upper()} Method:")
                
                for regime_key, regime_alphas in method_results.items():
                    report.append(f"  {regime_key}:")
                    
                    for regime_id, alpha_stats in regime_alphas.items():
                        report.append(f"    Regime {regime_id}:")
                        report.append(f"      Alpha: {alpha_stats['alpha']:.4f}")
                        report.append(f"      T-stat: {alpha_stats['t_statistic']:.3f}")
                        report.append(f"      P-value: {alpha_stats['p_value']:.4f}")
                        report.append(f"      Significant: {alpha_stats['significant']}")
                        report.append(f"      Sharpe Ratio: {alpha_stats['sharpe_ratio']:.3f}")
                        report.append(f"      Observations: {alpha_stats['observations']}")
                        report.append("")
                        
        # Transition analysis
        if 'transition_analysis' in self.results:
            report.append("REGIME TRANSITION ANALYSIS")
            report.append("-" * 40)
            
            for method_name, method_results in self.results['transition_analysis'].items():
                report.append(f"{method_name.upper()} Method:")
                
                for regime_key, transition_stats in method_results.items():
                    report.append(f"  {regime_key}:")
                    
                    # Persistence
                    report.append("    Regime Persistence:")
                    for regime_id, persistence in transition_stats['persistence'].items():
                        report.append(f"      Regime {regime_id}: {persistence:.3f}")
                        
                    # Expected duration
                    report.append("    Expected Duration:")
                    for regime_id, duration in transition_stats['expected_duration'].items():
                        if duration == float('inf'):
                            report.append(f"      Regime {regime_id}: Infinite")
                        else:
                            report.append(f"      Regime {regime_id}: {duration:.1f} periods")
                            
                    # Ergodic probabilities
                    report.append("    Long-run Probabilities:")
                    for regime_id, prob in transition_stats['ergodic_probabilities'].items():
                        report.append(f"      Regime {regime_id}: {prob:.3f}")
                        
                    report.append("")
                    
        # Regime forecasting
        if 'regime_forecasting' in self.results:
            report.append("REGIME FORECASTING")
            report.append("-" * 40)
            
            forecast = self.results['regime_forecasting']
            
            if 'markov' in forecast:
                markov_forecast = forecast['markov']
                report.append(f"Most Likely Next Regime: {markov_forecast['most_likely_regime']}")
                report.append(f"Confidence: {markov_forecast['regime_confidence']:.3f}")
                
                report.append("\nNext Period Probabilities:")
                for i, prob in enumerate(markov_forecast['next_period_probabilities']):
                    report.append(f"  Regime {i}: {prob:.3f}")
                    
                report.append("")
                
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
                
        return report_text
        
    def integrate_with_correlation_tracker(self, correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate regime analysis with existing correlation tracker
        
        Args:
            correlation_data: Data from correlation tracker
            
        Returns:
            Integrated analysis results
        """
        integration_results = {}
        
        if not self.results:
            return {'error': 'No regime analysis results available'}
            
        # Extract correlation shock events
        correlation_shocks = correlation_data.get('correlation_shocks', [])
        
        if not correlation_shocks:
            return {'error': 'No correlation shock data available'}
            
        # Map correlation shocks to regime transitions
        shock_regime_mapping = {}
        
        # Get best regime model
        best_model = None
        if 'model_comparison' in self.results and 'best_models' in self.results['model_comparison']:
            best_model_name = self.results['model_comparison']['best_models']['best_bic']['model']
            
            # Extract method and regime count
            method_name = best_model_name.split('_')[0]
            regime_key = '_'.join(best_model_name.split('_')[1:])
            
            if method_name in self.results and regime_key in self.results[method_name]:
                best_model = self.results[method_name][regime_key]
                
        if best_model is not None:
            regime_assignments = best_model.regime_assignments
            
            # Analyze regime changes around correlation shocks
            for shock in correlation_shocks:
                shock_time = shock.get('timestamp', 0)
                
                # Find regime before and after shock
                if shock_time > 0 and shock_time < len(regime_assignments):
                    regime_before = regime_assignments[max(0, shock_time - 1)]
                    regime_after = regime_assignments[min(len(regime_assignments) - 1, shock_time + 1)]
                    
                    shock_regime_mapping[shock_time] = {
                        'regime_before': regime_before,
                        'regime_after': regime_after,
                        'regime_change': regime_before != regime_after,
                        'shock_severity': shock.get('severity', 'unknown')
                    }
                    
            integration_results['shock_regime_mapping'] = shock_regime_mapping
            
            # Analyze correlation between regime changes and shocks
            regime_changes = []
            shock_times = []
            
            for i in range(1, len(regime_assignments)):
                if regime_assignments[i] != regime_assignments[i-1]:
                    regime_changes.append(i)
                    
            for shock in correlation_shocks:
                shock_times.append(shock.get('timestamp', 0))
                
            # Calculate temporal correlation
            if regime_changes and shock_times:
                # Check if regime changes coincide with shocks (within 5 periods)
                coincident_events = 0
                for change_time in regime_changes:
                    for shock_time in shock_times:
                        if abs(change_time - shock_time) <= 5:
                            coincident_events += 1
                            break
                            
                coincidence_rate = coincident_events / len(regime_changes)
                integration_results['regime_shock_correlation'] = {
                    'coincidence_rate': coincidence_rate,
                    'regime_changes': len(regime_changes),
                    'correlation_shocks': len(shock_times),
                    'coincident_events': coincident_events
                }
                
        return integration_results


def main():
    """Example regime analysis"""
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data with regime changes
    np.random.seed(42)
    n_periods = 1000
    
    # Create returns with two regimes
    regime_1_periods = 500
    regime_2_periods = 500
    
    # Low volatility regime
    regime_1_returns = np.random.normal(0.001, 0.01, regime_1_periods)
    
    # High volatility regime
    regime_2_returns = np.random.normal(-0.0005, 0.03, regime_2_periods)
    
    # Combine regimes
    returns = np.concatenate([regime_1_returns, regime_2_returns])
    
    # Generate baseline returns
    baseline_returns = np.random.normal(0.0003, 0.015, n_periods)
    
    # Initialize analyzer
    analyzer = RegimeAnalyzer()
    
    # Run regime analysis
    results = analyzer.analyze_regimes(
        returns,
        baseline_returns,
        max_regimes=4,
        methods=['markov', 'gaussian_mixture', 'volatility_clustering']
    )
    
    # Generate report
    report = analyzer.generate_regime_report("regime_analysis_report.txt")
    print(report)
    
    # Test integration with correlation tracker
    mock_correlation_data = {
        'correlation_shocks': [
            {'timestamp': 250, 'severity': 'HIGH'},
            {'timestamp': 500, 'severity': 'CRITICAL'},
            {'timestamp': 750, 'severity': 'MODERATE'}
        ]
    }
    
    integration_results = analyzer.integrate_with_correlation_tracker(mock_correlation_data)
    print("\nIntegration with Correlation Tracker:")
    print(integration_results)


if __name__ == "__main__":
    main()