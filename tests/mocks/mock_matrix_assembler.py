"""
Mock Matrix Assembler for testing

Provides deterministic matrix data for unit testing the Strategic Environment
and other components that depend on matrix inputs.
"""

import numpy as np
from typing import Optional, Dict, Any
from src.core.minimal_dependencies import MinimalComponentBase


class MockMatrixAssembler(MinimalComponentBase):
    """
    Mock matrix assembler returning deterministic test data
    
    Generates predictable matrices for testing agent observations,
    feature extraction, and environment behavior.
    """
    
    def __init__(self, name: str = "mock_matrix_assembler", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Default shape from Strategic MARL spec
        self.matrix_shape = config.get("matrix_shape", [48, 13]) if config else [48, 13]
        self.feature_names = [
            "sma_20", "sma_50", "rsi", "macd", "macd_signal",
            "bb_upper", "bb_lower", "volume", "atr", "mlmi",
            "mmd_1", "mmd_2", "mmd_3"
        ]
        
        # Test scenarios
        self.scenario = "default"
        self.step_count = 0
        
    def set_scenario(self, scenario: str) -> None:
        """Set test scenario for deterministic outputs"""
        self.scenario = scenario
        self.step_count = 0
        
    def get_current_matrix(self) -> np.ndarray:
        """
        Get current matrix based on scenario
        
        Returns deterministic matrices for different test scenarios.
        """
        self.step_count += 1
        
        if self.scenario == "default":
            # Default: Random but seeded for reproducibility
            np.random.seed(42 + self.step_count)
            return np.random.randn(*self.matrix_shape).astype(np.float32)
            
        elif self.scenario == "bullish":
            # Bullish trend scenario
            matrix = np.zeros(self.matrix_shape, dtype=np.float32)
            # SMA20 > SMA50 (bullish)
            matrix[:, 0] = np.linspace(1.0, 1.5, self.matrix_shape[0])  # sma_20
            matrix[:, 1] = np.linspace(0.8, 1.2, self.matrix_shape[0])  # sma_50
            # RSI trending up
            matrix[:, 2] = np.linspace(40, 70, self.matrix_shape[0])    # rsi
            # MLMI bullish
            matrix[:, 9] = np.linspace(0.4, 0.8, self.matrix_shape[0])  # mlmi
            # MMD features positive
            matrix[:, 10:13] = 0.5
            return matrix
            
        elif self.scenario == "bearish":
            # Bearish trend scenario
            matrix = np.zeros(self.matrix_shape, dtype=np.float32)
            # SMA20 < SMA50 (bearish)
            matrix[:, 0] = np.linspace(1.2, 0.8, self.matrix_shape[0])  # sma_20
            matrix[:, 1] = np.linspace(1.0, 1.0, self.matrix_shape[0])  # sma_50
            # RSI trending down
            matrix[:, 2] = np.linspace(60, 30, self.matrix_shape[0])    # rsi
            # MLMI bearish
            matrix[:, 9] = np.linspace(0.6, 0.2, self.matrix_shape[0])  # mlmi
            # MMD features negative
            matrix[:, 10:13] = -0.5
            return matrix
            
        elif self.scenario == "neutral":
            # Neutral/consolidation scenario
            matrix = np.zeros(self.matrix_shape, dtype=np.float32)
            # All indicators near neutral
            matrix[:, :] = 0.0
            matrix[:, 2] = 50  # RSI at neutral
            matrix[:, 9] = 0.5  # MLMI at neutral
            return matrix
            
        elif self.scenario == "high_volatility":
            # High volatility scenario
            np.random.seed(42 + self.step_count)
            matrix = np.random.randn(*self.matrix_shape) * 2.0  # Higher variance
            return matrix.astype(np.float32)
            
        else:
            # Unknown scenario - return zeros
            return np.zeros(self.matrix_shape, dtype=np.float32)
            
    def get_feature_names(self) -> list:
        """Get feature names"""
        return self.feature_names
        
    def reset(self) -> None:
        """Reset assembler state"""
        self.step_count = 0