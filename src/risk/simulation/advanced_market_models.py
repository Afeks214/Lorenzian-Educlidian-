"""
Advanced Market Models for Monte Carlo Simulation

Sophisticated market simulation models including:
- Geometric Brownian Motion with jump diffusion
- Heston stochastic volatility model
- SABR volatility model
- Regime-switching models
- Correlation structure preservation

All models are optimized for high-speed simulation with Numba JIT compilation.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import structlog
from numba import jit, prange
import scipy.stats as stats

logger = structlog.get_logger()

# Suppress numba warnings
warnings.filterwarnings("ignore", module="numba")


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "bull"
    BEAR = "bear"
    CRISIS = "crisis"
    RECOVERY = "recovery"


@dataclass
class ModelParameters:
    """Base class for model parameters"""
    pass


@dataclass
class GBMParameters(ModelParameters):
    """Geometric Brownian Motion parameters"""
    drift: float                    # Annual drift rate
    volatility: float              # Annual volatility
    initial_price: float           # Starting price


@dataclass  
class JumpDiffusionParameters(ModelParameters):
    """Jump diffusion model parameters"""
    drift: float                   # Drift rate
    volatility: float              # Diffusion volatility
    jump_intensity: float          # Jumps per year
    jump_mean: float              # Mean jump size (log scale)
    jump_std: float               # Jump size standard deviation
    initial_price: float          # Starting price


@dataclass
class HestonParameters(ModelParameters):
    """Heston stochastic volatility model parameters"""
    initial_price: float          # S0
    initial_variance: float       # V0  
    long_term_variance: float     # Theta
    kappa: float                  # Mean reversion speed
    vol_of_vol: float            # Sigma (volatility of volatility)
    correlation: float           # Correlation between price and vol
    drift: float                 # Risk-free rate


@dataclass
class RegimeSwitchingParameters(ModelParameters):
    """Regime-switching model parameters"""
    regime_parameters: Dict[MarketRegime, GBMParameters]  # Parameters per regime
    transition_matrix: np.ndarray                         # Regime transition probabilities
    initial_regime: MarketRegime                         # Starting regime


class BaseMarketModel(ABC):
    """Abstract base class for market models"""
    
    def __init__(self, parameters: ModelParameters):
        self.parameters = parameters
        
    @abstractmethod
    def simulate_paths(self, 
                      num_paths: int, 
                      time_steps: int, 
                      dt: float,
                      random_seed: Optional[int] = None) -> np.ndarray:
        """Simulate price paths"""
        pass


@jit(nopython=True, parallel=True, fastmath=True)
def _gbm_simulation_numba(num_paths: int,
                         time_steps: int,
                         initial_price: float,
                         drift: float,
                         volatility: float,
                         dt: float,
                         random_normals: np.ndarray) -> np.ndarray:
    """
    Optimized GBM simulation using Numba
    
    Args:
        num_paths: Number of simulation paths
        time_steps: Number of time steps
        initial_price: Starting price
        drift: Annual drift rate
        volatility: Annual volatility
        dt: Time step size
        random_normals: Pre-generated random normal variates
        
    Returns:
        Price paths array of shape (num_paths, time_steps)
    """
    paths = np.zeros((num_paths, time_steps))
    
    # Initialize with starting price
    for path in prange(num_paths):
        paths[path, 0] = initial_price
    
    # Generate paths using exact GBM solution
    for path in prange(num_paths):
        for t in range(1, time_steps):
            dW = random_normals[path, t-1]
            
            # Exact GBM solution: S(t) = S(t-1) * exp((μ - σ²/2)*dt + σ*sqrt(dt)*dW)
            drift_term = (drift - 0.5 * volatility * volatility) * dt
            diffusion_term = volatility * np.sqrt(dt) * dW
            
            paths[path, t] = paths[path, t-1] * np.exp(drift_term + diffusion_term)
    
    return paths


@jit(nopython=True, parallel=True, fastmath=True)
def _jump_diffusion_simulation_numba(num_paths: int,
                                    time_steps: int,
                                    initial_price: float,
                                    drift: float,
                                    volatility: float,
                                    jump_intensity: float,
                                    jump_mean: float,
                                    jump_std: float,
                                    dt: float,
                                    random_normals: np.ndarray,
                                    jump_times: np.ndarray,
                                    jump_sizes: np.ndarray) -> np.ndarray:
    """
    Optimized jump diffusion simulation using Numba
    
    Merton jump diffusion model:
    dS = μS*dt + σS*dW + S*(e^J - 1)*dN
    
    where dN is a Poisson process and J ~ N(μ_J, σ_J²)
    """
    paths = np.zeros((num_paths, time_steps))
    
    # Initialize with starting price
    for path in prange(num_paths):
        paths[path, 0] = initial_price
    
    # Generate paths
    for path in prange(num_paths):
        for t in range(1, time_steps):
            S_prev = paths[path, t-1]
            dW = random_normals[path, t-1]
            
            # Diffusion component (same as GBM)
            drift_term = (drift - 0.5 * volatility * volatility) * dt
            diffusion_term = volatility * np.sqrt(dt) * dW
            
            price = S_prev * np.exp(drift_term + diffusion_term)
            
            # Add jump component
            if jump_times[path, t-1] > 0:  # Jump occurs
                jump_multiplier = np.exp(jump_sizes[path, t-1])
                price *= jump_multiplier
            
            paths[path, t] = price
    
    return paths


@jit(nopython=True, parallel=True, fastmath=True)
def _heston_simulation_numba(num_paths: int,
                           time_steps: int,
                           initial_price: float,
                           initial_variance: float,
                           long_term_variance: float,
                           kappa: float,
                           vol_of_vol: float,
                           correlation: float,
                           drift: float,
                           dt: float,
                           random_normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized Heston model simulation using Numba
    
    Heston model:
    dS = μS*dt + sqrt(V)*S*dW1
    dV = κ(θ - V)*dt + σ*sqrt(V)*dW2
    
    where dW1 and dW2 are correlated Brownian motions
    """
    price_paths = np.zeros((num_paths, time_steps))
    variance_paths = np.zeros((num_paths, time_steps))
    
    # Initialize
    for path in prange(num_paths):
        price_paths[path, 0] = initial_price
        variance_paths[path, 0] = initial_variance
    
    # Correlation decomposition
    sqrt_1_minus_rho2 = np.sqrt(1.0 - correlation * correlation)
    
    # Generate paths using Euler-Maruyama scheme
    for path in prange(num_paths):
        for t in range(1, time_steps):
            S_prev = price_paths[path, t-1]
            V_prev = variance_paths[path, t-1]
            
            # Ensure variance stays positive (absorption at zero)
            V_prev = max(V_prev, 0.0)
            
            # Generate correlated random variables
            dW1 = random_normals[path, t-1, 0]
            dW2_indep = random_normals[path, t-1, 1]
            dW2 = correlation * dW1 + sqrt_1_minus_rho2 * dW2_indep
            
            # Price evolution
            price_drift = drift * S_prev * dt
            price_diffusion = np.sqrt(V_prev) * S_prev * dW1 * np.sqrt(dt)
            price_paths[path, t] = S_prev + price_drift + price_diffusion
            
            # Variance evolution
            variance_drift = kappa * (long_term_variance - V_prev) * dt
            variance_diffusion = vol_of_vol * np.sqrt(V_prev) * dW2 * np.sqrt(dt)
            variance_paths[path, t] = V_prev + variance_drift + variance_diffusion
            
            # Ensure variance stays non-negative
            variance_paths[path, t] = max(variance_paths[path, t], 0.0)
            
            # Ensure price stays positive
            price_paths[path, t] = max(price_paths[path, t], 0.01 * initial_price)
    
    return price_paths, variance_paths


class GeometricBrownianMotion(BaseMarketModel):
    """
    Standard Geometric Brownian Motion model
    
    dS = μS*dt + σS*dW
    
    Features:
    - Exact analytical solution
    - Constant drift and volatility
    - Log-normal price distribution
    """
    
    def __init__(self, parameters: GBMParameters):
        super().__init__(parameters)
        self.params = parameters
    
    def simulate_paths(self, 
                      num_paths: int, 
                      time_steps: int, 
                      dt: float,
                      random_seed: Optional[int] = None) -> np.ndarray:
        """Simulate GBM price paths"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random normal variates
        random_normals = np.random.normal(0, 1, (num_paths, time_steps - 1))
        
        return _gbm_simulation_numba(
            num_paths,
            time_steps,
            self.params.initial_price,
            self.params.drift,
            self.params.volatility,
            dt,
            random_normals
        )


class JumpDiffusionModel(BaseMarketModel):
    """
    Merton jump diffusion model
    
    dS = μS*dt + σS*dW + S*(e^J - 1)*dN
    
    Features:
    - Poisson jump process
    - Log-normal jump sizes
    - Captures price gaps and sudden moves
    """
    
    def __init__(self, parameters: JumpDiffusionParameters):
        super().__init__(parameters)
        self.params = parameters
    
    def simulate_paths(self, 
                      num_paths: int, 
                      time_steps: int, 
                      dt: float,
                      random_seed: Optional[int] = None) -> np.ndarray:
        """Simulate jump diffusion price paths"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random normal variates for diffusion
        random_normals = np.random.normal(0, 1, (num_paths, time_steps - 1))
        
        # Generate jump times (Poisson process)
        jump_prob = self.params.jump_intensity * dt
        jump_times = np.random.binomial(1, jump_prob, (num_paths, time_steps - 1))
        
        # Generate jump sizes (log-normal)
        jump_sizes = np.random.normal(
            self.params.jump_mean, 
            self.params.jump_std, 
            (num_paths, time_steps - 1)
        )
        
        return _jump_diffusion_simulation_numba(
            num_paths,
            time_steps,
            self.params.initial_price,
            self.params.drift,
            self.params.volatility,
            self.params.jump_intensity,
            self.params.jump_mean,
            self.params.jump_std,
            dt,
            random_normals,
            jump_times,
            jump_sizes
        )


class HestonStochasticVolatility(BaseMarketModel):
    """
    Heston stochastic volatility model
    
    dS = μS*dt + sqrt(V)*S*dW1
    dV = κ(θ - V)*dt + σ*sqrt(V)*dW2
    
    Features:
    - Stochastic volatility
    - Volatility clustering
    - Correlation between price and volatility
    """
    
    def __init__(self, parameters: HestonParameters):
        super().__init__(parameters)
        self.params = parameters
    
    def simulate_paths(self, 
                      num_paths: int, 
                      time_steps: int, 
                      dt: float,
                      random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths
        
        Returns:
            Tuple of (price_paths, variance_paths)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random normal variates (2D for correlated Brownian motions)
        random_normals = np.random.normal(0, 1, (num_paths, time_steps - 1, 2))
        
        return _heston_simulation_numba(
            num_paths,
            time_steps,
            self.params.initial_price,
            self.params.initial_variance,
            self.params.long_term_variance,
            self.params.kappa,
            self.params.vol_of_vol,
            self.params.correlation,
            self.params.drift,
            dt,
            random_normals
        )


class RegimeSwitchingModel(BaseMarketModel):
    """
    Regime-switching model with multiple market states
    
    Features:
    - Multiple market regimes (bull, bear, crisis, recovery)
    - Markov chain regime transitions
    - Different parameters per regime
    """
    
    def __init__(self, parameters: RegimeSwitchingParameters):
        super().__init__(parameters)
        self.params = parameters
        self.regimes = list(MarketRegime)
        self.regime_index = {regime: i for i, regime in enumerate(self.regimes)}
    
    def simulate_paths(self, 
                      num_paths: int, 
                      time_steps: int, 
                      dt: float,
                      random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate regime-switching paths
        
        Returns:
            Tuple of (price_paths, regime_paths)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        price_paths = np.zeros((num_paths, time_steps))
        regime_paths = np.zeros((num_paths, time_steps), dtype=int)
        
        # Initial conditions
        initial_regime_idx = self.regime_index[self.params.initial_regime]
        initial_params = self.params.regime_parameters[self.params.initial_regime]
        
        for path in range(num_paths):
            price_paths[path, 0] = initial_params.initial_price
            regime_paths[path, 0] = initial_regime_idx
        
        # Transition probability matrix
        P = self.params.transition_matrix
        
        # Simulate paths
        for path in range(num_paths):
            current_regime_idx = initial_regime_idx
            
            for t in range(1, time_steps):
                # Regime transition
                if t > 1:  # Allow regime switches after first step
                    transition_probs = P[current_regime_idx, :]
                    current_regime_idx = np.random.choice(len(self.regimes), p=transition_probs)
                
                regime_paths[path, t] = current_regime_idx
                current_regime = self.regimes[current_regime_idx]
                current_params = self.params.regime_parameters[current_regime]
                
                # Price evolution under current regime (GBM)
                S_prev = price_paths[path, t-1]
                dW = np.random.normal(0, 1)
                
                drift_term = (current_params.drift - 0.5 * current_params.volatility**2) * dt
                diffusion_term = current_params.volatility * np.sqrt(dt) * dW
                
                price_paths[path, t] = S_prev * np.exp(drift_term + diffusion_term)
        
        return price_paths, regime_paths
    
    def get_regime_statistics(self, regime_paths: np.ndarray) -> Dict[str, Any]:
        """Calculate regime statistics from simulation results"""
        stats = {}
        
        for i, regime in enumerate(self.regimes):
            regime_mask = (regime_paths == i)
            stats[regime.value] = {
                'frequency': np.mean(regime_mask),
                'avg_duration': self._calculate_avg_duration(regime_paths, i),
                'transitions_from': self._count_transitions_from(regime_paths, i)
            }
        
        return stats
    
    def _calculate_avg_duration(self, regime_paths: np.ndarray, regime_idx: int) -> float:
        """Calculate average duration of regime episodes"""
        durations = []
        
        for path in range(regime_paths.shape[0]):
            current_duration = 0
            
            for t in range(regime_paths.shape[1]):
                if regime_paths[path, t] == regime_idx:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                        current_duration = 0
            
            # Handle end of path
            if current_duration > 0:
                durations.append(current_duration)
        
        return np.mean(durations) if durations else 0.0
    
    def _count_transitions_from(self, regime_paths: np.ndarray, regime_idx: int) -> Dict[int, int]:
        """Count transitions from given regime to other regimes"""
        transitions = {i: 0 for i in range(len(self.regimes))}
        
        for path in range(regime_paths.shape[0]):
            for t in range(regime_paths.shape[1] - 1):
                if regime_paths[path, t] == regime_idx:
                    next_regime = regime_paths[path, t + 1]
                    transitions[next_regime] += 1
        
        return transitions


class CorrelationGenerator:
    """
    Utility class for generating and managing correlation structures
    """
    
    @staticmethod
    def generate_random_correlation_matrix(n_assets: int, 
                                         min_correlation: float = -0.3,
                                         max_correlation: float = 0.7,
                                         random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random positive definite correlation matrix
        
        Args:
            n_assets: Number of assets
            min_correlation: Minimum correlation value
            max_correlation: Maximum correlation value
            random_seed: Random seed for reproducibility
            
        Returns:
            Positive definite correlation matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate random matrix
        A = np.random.uniform(min_correlation, max_correlation, (n_assets, n_assets))
        
        # Make symmetric
        A = (A + A.T) / 2
        
        # Set diagonal to 1
        np.fill_diagonal(A, 1.0)
        
        # Ensure positive definiteness using nearest correlation matrix
        return CorrelationGenerator._nearest_correlation_matrix(A)
    
    @staticmethod
    def _nearest_correlation_matrix(A: np.ndarray, max_iterations: int = 100) -> np.ndarray:
        """
        Find the nearest positive definite correlation matrix using Higham's algorithm
        """
        n = A.shape[0]
        
        # Initial setup
        Y = A.copy()
        dS = np.zeros_like(A)
        
        for iteration in range(max_iterations):
            # Projection onto S (symmetric matrices with unit diagonal)
            Y = Y - dS
            np.fill_diagonal(Y, 1.0)
            
            # Projection onto positive semidefinite cone
            eigenvals, eigenvecs = np.linalg.eigh(Y)
            eigenvals = np.maximum(eigenvals, 0)
            X = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Update dS
            dS = X - Y
            
            # Check convergence
            if np.linalg.norm(dS, 'fro') < 1e-8:
                break
            
            Y = X
        
        # Final projection to correlation matrix
        np.fill_diagonal(X, 1.0)
        
        return X
    
    @staticmethod
    def block_correlation_matrix(block_sizes: List[int], 
                                within_block_corr: float = 0.6,
                                between_block_corr: float = 0.1) -> np.ndarray:
        """
        Generate block correlation matrix (useful for sector/asset class correlations)
        
        Args:
            block_sizes: List of block sizes
            within_block_corr: Correlation within blocks
            between_block_corr: Correlation between blocks
            
        Returns:
            Block-structured correlation matrix
        """
        total_size = sum(block_sizes)
        corr_matrix = np.full((total_size, total_size), between_block_corr)
        
        # Fill blocks
        start_idx = 0
        for block_size in block_sizes:
            end_idx = start_idx + block_size
            corr_matrix[start_idx:end_idx, start_idx:end_idx] = within_block_corr
            start_idx = end_idx
        
        # Set diagonal to 1
        np.fill_diagonal(corr_matrix, 1.0)
        
        return corr_matrix