"""
Base Classes and Interfaces for Lorentzian Trading Strategy
Provides foundation classes for all strategy components.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from ..config.config import get_config
from ..utils.logging_config import get_logger, log_function_call


@dataclass
class Signal:
    """Trading signal data structure"""
    timestamp: datetime
    signal_type: str  # 'long', 'short', 'exit_long', 'exit_short'
    confidence: float  # 0.0 to 1.0
    price: float
    features: Dict[str, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Position data structure"""
    entry_time: datetime
    entry_price: float
    position_type: str  # 'long' or 'short'
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_open(self) -> bool:
        """Check if position is still open"""
        return self.exit_time is None
    
    @property
    def duration(self) -> Optional[float]:
        """Get position duration in hours"""
        if self.exit_time:
            return (self.exit_time - self.entry_time).total_seconds() / 3600
        return None


@dataclass
class FeatureVector:
    """Feature vector for ML classification"""
    timestamp: datetime
    features: np.ndarray
    feature_names: List[str]
    target: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if len(self.features) != len(self.feature_names):
            raise ValueError("Features and feature_names must have same length")


class BaseComponent(ABC):
    """Base class for all strategy components"""
    
    def __init__(self, config=None, logger_name: str = None):
        self.config = config or get_config()
        self.logger = get_logger(logger_name or self.__class__.__name__)
        self._initialized = False
        self._state = {}
    
    def initialize(self):
        """Initialize component - called once before processing"""
        if not self._initialized:
            self._setup()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    def _setup(self):
        """Component-specific setup logic"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state for serialization"""
        return self._state.copy()
    
    def set_state(self, state: Dict[str, Any]):
        """Set component state from serialization"""
        self._state = state.copy()
    
    def reset(self):
        """Reset component to initial state"""
        self._state = {}
        self._initialized = False


class BaseDataProcessor(BaseComponent):
    """Base class for data processing components"""
    
    def __init__(self, config=None):
        super().__init__(config, "lorentzian_strategy.data")
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process input data and return transformed data"""
        pass
    
    def validate_input(self, data: pd.DataFrame):
        """Validate input data format"""
        required_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    @log_function_call(logger=logging.getLogger("lorentzian_strategy.data"))
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Callable interface for processing"""
        self.initialize()
        self.validate_input(data)
        return self.process(data)


class BaseFeatureExtractor(BaseComponent):
    """Base class for feature extraction components"""
    
    def __init__(self, config=None):
        super().__init__(config, "lorentzian_strategy.features")
        self.feature_names = []
    
    @abstractmethod
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        pass
    
    def create_feature_vector(self, data: pd.DataFrame, timestamp: datetime) -> FeatureVector:
        """Create feature vector for specific timestamp"""
        features = self.extract_features(data)
        return FeatureVector(
            timestamp=timestamp,
            features=features,
            feature_names=self.get_feature_names()
        )


class BaseClassifier(BaseComponent):
    """Base class for classification components"""
    
    def __init__(self, config=None):
        super().__init__(config, "lorentzian_strategy.core")
    
    @abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Train the classifier"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        pass


class BaseDistanceMetric(BaseComponent):
    """Base class for distance metric implementations"""
    
    def __init__(self, config=None):
        super().__init__(config, "lorentzian_strategy.core")
    
    @abstractmethod
    def calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate distance between two feature vectors"""
        pass
    
    @abstractmethod
    def calculate_distances(self, query: np.ndarray, database: np.ndarray) -> np.ndarray:
        """Calculate distances from query to all vectors in database"""
        pass


class BaseSignalGenerator(BaseComponent):
    """Base class for signal generation components"""
    
    def __init__(self, config=None):
        super().__init__(config, "lorentzian_strategy.signals")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals from data"""
        pass
    
    def should_enter_long(self, features: Dict[str, float], **kwargs) -> Tuple[bool, float]:
        """Check if should enter long position"""
        return False, 0.0
    
    def should_enter_short(self, features: Dict[str, float], **kwargs) -> Tuple[bool, float]:
        """Check if should enter short position"""
        return False, 0.0
    
    def should_exit_position(self, position: Position, current_data: Dict[str, float]) -> bool:
        """Check if should exit current position"""
        return False


class BaseStrategy(BaseComponent):
    """Base class for complete trading strategies"""
    
    def __init__(self, config=None):
        super().__init__(config, "lorentzian_strategy.strategy")
        self.positions: List[Position] = []
        self.signals: List[Signal] = []
        self.feature_extractor: Optional[BaseFeatureExtractor] = None
        self.classifier: Optional[BaseClassifier] = None
        self.signal_generator: Optional[BaseSignalGenerator] = None
    
    def add_component(self, component: BaseComponent, component_type: str):
        """Add component to strategy"""
        if component_type == "feature_extractor":
            self.feature_extractor = component
        elif component_type == "classifier":
            self.classifier = component
        elif component_type == "signal_generator":
            self.signal_generator = component
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    @abstractmethod
    def run_backtest(self, data: pd.DataFrame, start_date: str = None, 
                    end_date: str = None) -> Dict[str, Any]:
        """Run backtest on historical data"""
        pass
    
    def get_current_positions(self) -> List[Position]:
        """Get currently open positions"""
        return [pos for pos in self.positions if pos.is_open]
    
    def get_closed_positions(self) -> List[Position]:
        """Get closed positions"""
        return [pos for pos in self.positions if not pos.is_open]


class BaseRiskManager(BaseComponent):
    """Base class for risk management components"""
    
    def __init__(self, config=None):
        super().__init__(config, "lorentzian_strategy.risk")
    
    @abstractmethod
    def check_position_size(self, signal: Signal, current_capital: float) -> float:
        """Determine appropriate position size"""
        pass
    
    @abstractmethod
    def check_risk_limits(self, positions: List[Position], current_capital: float) -> bool:
        """Check if risk limits are exceeded"""
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """Calculate stop loss level"""
        pass
    
    @abstractmethod
    def calculate_take_profit(self, entry_price: float, position_type: str) -> float:
        """Calculate take profit level"""
        pass


class BaseOptimizer(BaseComponent):
    """Base class for parameter optimization"""
    
    def __init__(self, config=None):
        super().__init__(config, "lorentzian_strategy.optimization")
    
    @abstractmethod
    def optimize_parameters(self, strategy: BaseStrategy, data: pd.DataFrame, 
                          parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        pass


class ValidationMixin:
    """Mixin class for adding validation capabilities"""
    
    def validate_data_continuity(self, data: pd.DataFrame, max_gap_hours: int = 2):
        """Validate data continuity"""
        time_diff = data['Timestamp'].diff()
        expected_interval = pd.Timedelta(minutes=30)
        large_gaps = time_diff > pd.Timedelta(hours=max_gap_hours)
        
        if large_gaps.any():
            gap_count = large_gaps.sum()
            self.logger.warning(f"Found {gap_count} large time gaps in data")
            return False
        return True
    
    def validate_feature_quality(self, features: np.ndarray, max_nan_ratio: float = 0.05):
        """Validate feature quality"""
        if features.size == 0:
            raise ValueError("Empty feature array")
        
        nan_ratio = np.isnan(features).sum() / features.size
        if nan_ratio > max_nan_ratio:
            raise ValueError(f"Too many NaN values in features: {nan_ratio:.2%} > {max_nan_ratio:.2%}")
        
        if np.isinf(features).any():
            raise ValueError("Infinite values found in features")
        
        return True


class CacheableMixin:
    """Mixin class for adding caching capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self.cache_enabled = True
    
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        import hashlib
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def get_from_cache(self, key: str) -> Any:
        """Get value from cache"""
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        return None
    
    def set_cache(self, key: str, value: Any):
        """Set value in cache"""
        if self.cache_enabled:
            self._cache[key] = value
    
    def clear_cache(self):
        """Clear all cached values"""
        self._cache.clear()


# Utility functions for component management
def create_component_pipeline(*components: BaseComponent) -> List[BaseComponent]:
    """Create a pipeline of components"""
    for component in components:
        component.initialize()
    return list(components)


def validate_component_compatibility(component1: BaseComponent, component2: BaseComponent) -> bool:
    """Validate that two components are compatible"""
    # Basic compatibility check - can be extended
    return True