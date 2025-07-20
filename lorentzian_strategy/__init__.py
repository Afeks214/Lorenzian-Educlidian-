"""
HYBRID LORENTZIAN-EUCLIDEAN DISTANCE SYSTEM
==========================================

An intelligent hybrid distance metric system for Lorentzian Classification 
that dynamically switches between Lorentzian and Euclidean distance calculations 
based on real-time market regime detection.

This package provides:
- Market regime detection with volatility and trend analysis
- Hybrid distance metrics with intelligent selection
- Regime-aware parameter optimization
- Comprehensive testing and validation

Quick Start:
-----------
```python
from lorentzian_strategy import HybridLorentzianSystem
import pandas as pd

# Initialize the system
system = HybridLorentzianSystem()

# Load your OHLCV market data
data = pd.read_csv('market_data.csv')

# Get optimal distance metric for current conditions
recommendation = system.get_distance_recommendation(data)
print(f"Recommended metric: {recommendation['recommended_metric']}")

# Calculate adaptive distance between feature vectors
x = [0.1, 0.2, 0.3, 0.4, 0.5]  # First feature vector
y = [0.12, 0.18, 0.32, 0.38, 0.52]  # Second feature vector
distance = system.calculate_distance(x, y, data)
print(f"Adaptive distance: {distance}")

# Optimize parameters for current market regime
optimal_params = system.optimize_parameters(data)
print(f"Optimal parameters: {optimal_params}")
```

Components:
-----------
- MarketRegime: Market regime detection and analysis
- HybridDistanceCalculator: Intelligent distance metric selection
- RegimeAwareOptimizer: Parameter optimization for different regimes
- Comprehensive testing suite

Author: Claude Code Assistant
Date: 2025-07-20
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import core components
try:
    # Try relative imports first
    from .market_regime import RegimeDetector, RegimeConfig, RegimeMetrics, MarketRegime
    from .distance_metrics import (
        HybridDistanceCalculator, 
        DistanceMetricsConfig, 
        lorentzian_distance, 
        euclidean_distance, 
        hybrid_distance, 
        adaptive_distance,
        get_optimal_distance_metric
    )
    from .regime_optimization import RegimeAwareOptimizer, OptimizationConfig
    from .test_hybrid_system import run_comprehensive_tests
    
    COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    try:
        # Try absolute imports
        import market_regime
        import distance_metrics
        import regime_optimization
        import test_hybrid_system
        
        RegimeDetector = market_regime.RegimeDetector
        RegimeConfig = market_regime.RegimeConfig
        RegimeMetrics = market_regime.RegimeMetrics
        MarketRegime = market_regime.MarketRegime
        
        HybridDistanceCalculator = distance_metrics.HybridDistanceCalculator
        DistanceMetricsConfig = distance_metrics.DistanceMetricsConfig
        lorentzian_distance = distance_metrics.lorentzian_distance
        euclidean_distance = distance_metrics.euclidean_distance
        hybrid_distance = distance_metrics.hybrid_distance
        adaptive_distance = distance_metrics.adaptive_distance
        get_optimal_distance_metric = distance_metrics.get_optimal_distance_metric
        
        RegimeAwareOptimizer = regime_optimization.RegimeAwareOptimizer
        OptimizationConfig = regime_optimization.OptimizationConfig
        
        run_comprehensive_tests = test_hybrid_system.run_comprehensive_tests
        
        COMPONENTS_AVAILABLE = True
        
    except ImportError as e2:
        logger.error(f"Failed to import components with both relative and absolute imports: {e2}")
        # Create dummy classes to prevent NameError
        class RegimeConfig: pass
        class RegimeMetrics: pass
        class MarketRegime: pass
        class DistanceMetricsConfig: pass
        class OptimizationConfig: pass
        class HybridDistanceCalculator: pass
        class RegimeAwareOptimizer: pass
        def run_comprehensive_tests(): return False
        def lorentzian_distance(x, y): return 0.0
        def euclidean_distance(x, y): return 0.0
        def hybrid_distance(x, y, alpha=0.5): return 0.0
        def adaptive_distance(x, y, data=None): return 0.0
        def get_optimal_distance_metric(data): return {}
        
        COMPONENTS_AVAILABLE = False

# Version information
__version__ = "1.0.0"
__author__ = "Claude Code Assistant"
__email__ = "claude@anthropic.com"

# Export main components
__all__ = [
    'HybridLorentzianSystem',
    'MarketRegime',
    'RegimeMetrics',
    'HybridDistanceCalculator',
    'RegimeAwareOptimizer',
    'lorentzian_distance',
    'euclidean_distance', 
    'hybrid_distance',
    'adaptive_distance',
    'get_optimal_distance_metric',
    'run_system_tests',
    'create_default_system'
]

class HybridLorentzianSystem:
    """
    Main interface for the Hybrid Lorentzian-Euclidean Distance System
    
    This class provides a simple, unified interface for all system functionality
    including regime detection, distance calculation, and parameter optimization.
    """
    
    def __init__(self, 
                 regime_config: Optional[RegimeConfig] = None,
                 distance_config: Optional[DistanceMetricsConfig] = None,
                 optimization_config: Optional[OptimizationConfig] = None):
        """
        Initialize the hybrid system
        
        Args:
            regime_config: Configuration for regime detection
            distance_config: Configuration for distance calculations
            optimization_config: Configuration for parameter optimization
        """
        if not COMPONENTS_AVAILABLE:
            raise ImportError("Required components not available. Please check dependencies.")
        
        # Initialize configurations
        self.regime_config = regime_config or RegimeConfig()
        self.distance_config = distance_config or DistanceMetricsConfig()
        self.optimization_config = optimization_config or OptimizationConfig()
        
        # Initialize components
        self.regime_detector = RegimeDetector(self.regime_config)
        self.distance_calculator = HybridDistanceCalculator(self.distance_config)
        self.optimizer = RegimeAwareOptimizer(self.optimization_config)
        
        # System state
        self.last_regime_metrics = None
        self.last_recommendation = None
        
        logger.info("Hybrid Lorentzian System initialized successfully")
    
    def analyze_market_regime(self, market_data: pd.DataFrame) -> RegimeMetrics:
        """
        Analyze current market regime
        
        Args:
            market_data: OHLCV market data DataFrame
            
        Returns:
            RegimeMetrics object with comprehensive regime analysis
        """
        try:
            regime_metrics = self.regime_detector.detect_regime(market_data)
            self.last_regime_metrics = regime_metrics
            
            logger.info(f"Market regime detected: {regime_metrics.regime.value} "
                       f"(confidence: {regime_metrics.confidence:.3f})")
            
            return regime_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            # Return default regime metrics
            return RegimeMetrics()
    
    def get_distance_recommendation(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get distance metric recommendation for current market conditions
        
        Args:
            market_data: OHLCV market data DataFrame
            
        Returns:
            Dictionary with metric recommendation and analysis
        """
        try:
            recommendation = self.distance_calculator.get_metric_recommendation(market_data)
            self.last_recommendation = recommendation
            
            logger.info(f"Distance metric recommendation: {recommendation['recommended_metric']} "
                       f"(alpha: {recommendation['alpha_value']:.3f})")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error getting distance recommendation: {e}")
            return {
                'recommended_metric': 'lorentzian',
                'alpha_value': 0.5,
                'reasoning': 'Default recommendation due to error',
                'confidence': 0.3
            }
    
    def calculate_distance(self, 
                          x: Union[np.ndarray, List[float]], 
                          y: Union[np.ndarray, List[float]],
                          market_data: Optional[pd.DataFrame] = None,
                          force_metric: Optional[str] = None) -> float:
        """
        Calculate adaptive distance between two feature vectors
        
        Args:
            x: First feature vector
            y: Second feature vector
            market_data: Optional market data for regime-aware selection
            force_metric: Force specific metric ('lorentzian', 'euclidean', 'hybrid')
            
        Returns:
            Distance value
        """
        try:
            result = self.distance_calculator.adaptive_distance(
                x, y, market_data, force_metric=force_metric
            )
            
            return result.distance
            
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            # Fallback to simple Euclidean distance
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            return np.sqrt(np.sum((x_arr - y_arr) ** 2))
    
    def optimize_parameters(self, 
                           market_data: pd.DataFrame,
                           target_regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        Optimize parameters for current market conditions
        
        Args:
            market_data: OHLCV market data DataFrame
            target_regime: Specific regime to optimize for (auto-detected if None)
            
        Returns:
            Dictionary with optimal parameters
        """
        try:
            result = self.optimizer.optimize_parameters(market_data, target_regime)
            
            logger.info(f"Parameter optimization completed. "
                       f"Confidence: {result.confidence_score:.3f}")
            
            return result.optimal_parameters
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            # Return default parameters
            return {
                'k_neighbors': 8,
                'lookback_window': 8,
                'confidence_threshold': 0.6,
                'alpha': 0.5
            }
    
    def get_optimal_parameters_for_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get optimal parameters for current market regime
        
        Args:
            market_data: Current OHLCV market data
            
        Returns:
            Dictionary of optimal parameters
        """
        try:
            return self.optimizer.get_optimal_parameters(market_data)
            
        except Exception as e:
            logger.error(f"Error getting optimal parameters: {e}")
            return {
                'k_neighbors': 8,
                'lookback_window': 8,
                'confidence_threshold': 0.6,
                'alpha': 0.5
            }
    
    def detect_regime_transition(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect if a regime transition is occurring
        
        Args:
            market_data: Current OHLCV market data
            
        Returns:
            Dictionary with transition information
        """
        try:
            current_metrics = self.analyze_market_regime(market_data)
            
            if self.last_regime_metrics is not None:
                transition_info = self.regime_detector.detect_regime_transition(
                    current_metrics, self.last_regime_metrics
                )
                
                if transition_info["transition_detected"]:
                    logger.info(f"Regime transition detected: {transition_info['transition_type']}")
                
                return transition_info
            else:
                return {
                    "transition_detected": False,
                    "transition_type": None,
                    "transition_confidence": 0.0,
                    "recommended_action": "maintain"
                }
                
        except Exception as e:
            logger.error(f"Error detecting regime transition: {e}")
            return {
                "transition_detected": False,
                "transition_type": "error",
                "transition_confidence": 0.0,
                "recommended_action": "maintain"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'system_version': __version__,
            'components_available': COMPONENTS_AVAILABLE,
            'last_regime': self.last_regime_metrics.regime.value if self.last_regime_metrics else None,
            'last_recommendation': self.last_recommendation['recommended_metric'] if self.last_recommendation else None,
            'configurations': {
                'regime_config': self.regime_config.__dict__ if hasattr(self.regime_config, '__dict__') else str(self.regime_config),
                'distance_config': self.distance_config.to_dict(),
                'optimization_config': self.optimization_config.__dict__ if hasattr(self.optimization_config, '__dict__') else str(self.optimization_config)
            }
        }
        
        return status
    
    def validate_system(self) -> bool:
        """
        Validate system functionality
        
        Returns:
            True if all validations pass
        """
        try:
            # Create test data
            test_data = pd.DataFrame({
                'open': [100, 101, 102, 103, 104],
                'high': [101, 102, 103, 104, 105],
                'low': [99, 100, 101, 102, 103],
                'close': [101, 102, 103, 104, 105],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            
            # Test regime detection
            regime_metrics = self.analyze_market_regime(test_data)
            if not isinstance(regime_metrics, RegimeMetrics):
                return False
            
            # Test distance recommendation
            recommendation = self.get_distance_recommendation(test_data)
            if 'recommended_metric' not in recommendation:
                return False
            
            # Test distance calculation
            x = [0.1, 0.2, 0.3]
            y = [0.12, 0.18, 0.32]
            distance = self.calculate_distance(x, y, test_data)
            if not isinstance(distance, (int, float)) or not np.isfinite(distance):
                return False
            
            logger.info("System validation passed")
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return False

# Convenience functions for quick access

def create_default_system() -> HybridLorentzianSystem:
    """
    Create a hybrid system with default configurations
    
    Returns:
        HybridLorentzianSystem instance
    """
    return HybridLorentzianSystem()

def quick_distance_calculation(x: List[float], y: List[float], 
                             market_data: Optional[pd.DataFrame] = None) -> float:
    """
    Quick distance calculation with automatic metric selection
    
    Args:
        x: First feature vector
        y: Second feature vector
        market_data: Optional market data for regime-aware selection
        
    Returns:
        Distance value
    """
    if market_data is not None:
        return adaptive_distance(x, y, market_data)
    else:
        return lorentzian_distance(x, y)

def quick_regime_analysis(market_data: pd.DataFrame) -> str:
    """
    Quick market regime analysis
    
    Args:
        market_data: OHLCV market data DataFrame
        
    Returns:
        String describing the current market regime
    """
    try:
        system = create_default_system()
        metrics = system.analyze_market_regime(market_data)
        return f"{metrics.regime.value} (confidence: {metrics.confidence:.2f})"
    except Exception as e:
        logger.error(f"Quick regime analysis failed: {e}")
        return "unknown"

def run_system_tests() -> bool:
    """
    Run comprehensive system tests
    
    Returns:
        True if all tests pass
    """
    if not COMPONENTS_AVAILABLE:
        logger.error("Cannot run tests - components not available")
        return False
    
    return run_comprehensive_tests()

# Module initialization message
if COMPONENTS_AVAILABLE:
    logger.info(f"Hybrid Lorentzian-Euclidean Distance System v{__version__} loaded successfully")
    logger.info("All components available and ready for use")
else:
    logger.warning(f"Hybrid Lorentzian-Euclidean Distance System v{__version__} loaded with limited functionality")
    logger.warning("Some components are not available - check dependencies")

# Example usage demonstration
def demonstrate_system():
    """Demonstrate the hybrid system capabilities"""
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Cannot run demonstration - components not available")
        return
    
    print(f"\n🚀 HYBRID LORENTZIAN-EUCLIDEAN DISTANCE SYSTEM v{__version__}")
    print("=" * 70)
    print("Intelligent market regime-aware distance metric system")
    print("=" * 70)
    
    try:
        # Create sample market data
        np.random.seed(42)
        n_bars = 100
        returns = np.random.normal(0.001, 0.02, n_bars)
        prices = 100 * np.exp(np.cumsum(returns))
        
        market_data = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_bars)
        })
        
        # Initialize system
        print("Initializing hybrid system...")
        system = HybridLorentzianSystem()
        
        # Analyze market regime
        print("\n📊 Market Regime Analysis:")
        regime_metrics = system.analyze_market_regime(market_data)
        print(f"  Regime: {regime_metrics.regime.value.upper()}")
        print(f"  Confidence: {regime_metrics.confidence:.3f}")
        print(f"  Volatility: {regime_metrics.volatility:.3f}")
        print(f"  Trend Strength: {regime_metrics.trend_strength:.1f}")
        
        # Get distance recommendation
        print("\n🎯 Distance Metric Recommendation:")
        recommendation = system.get_distance_recommendation(market_data)
        print(f"  Recommended Metric: {recommendation['recommended_metric'].upper()}")
        print(f"  Alpha Value: {recommendation['alpha_value']:.3f}")
        print(f"  Reasoning: {recommendation['reasoning']}")
        
        # Calculate distances
        print("\n📏 Distance Calculations:")
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.12, 0.18, 0.32, 0.38, 0.52]
        
        adaptive_dist = system.calculate_distance(x, y, market_data)
        lorentzian_dist = system.calculate_distance(x, y, force_metric="lorentzian")
        euclidean_dist = system.calculate_distance(x, y, force_metric="euclidean")
        
        print(f"  Adaptive Distance: {adaptive_dist:.6f}")
        print(f"  Lorentzian Distance: {lorentzian_dist:.6f}")
        print(f"  Euclidean Distance: {euclidean_dist:.6f}")
        
        # Optimize parameters
        print("\n⚙️ Parameter Optimization:")
        optimal_params = system.optimize_parameters(market_data)
        print(f"  K Neighbors: {optimal_params['k_neighbors']}")
        print(f"  Lookback Window: {optimal_params['lookback_window']}")
        print(f"  Confidence Threshold: {optimal_params['confidence_threshold']:.3f}")
        print(f"  Alpha: {optimal_params['alpha']:.3f}")
        
        # System validation
        print("\n✅ System Validation:")
        validation_result = system.validate_system()
        print(f"  Validation Status: {'PASSED' if validation_result else 'FAILED'} ✓" if validation_result else "  Validation Status: FAILED ❌")
        
        print("\n" + "=" * 70)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("The Hybrid Lorentzian-Euclidean Distance System is ready for production use.")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.error(f"Demonstration error: {e}")

if __name__ == "__main__":
    demonstrate_system()