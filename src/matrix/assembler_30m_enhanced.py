"""Enhanced MatrixAssembler30m with integrated MMD features"""

from .assembler_30m import MatrixAssembler30m
import numpy as np
from typing import Dict, Any, Optional, List

class MatrixAssembler30mEnhanced(MatrixAssembler30m):
    """Enhanced 30m assembler with integrated regime features"""
    
    def __init__(self, name: str, kernel):
        super().__init__(name, kernel)
        
        # Add MMD features
        self.mmd_features = ['mmd_1', 'mmd_2', 'mmd_3', 'volatility', 'volume_profile']
        self.feature_names.extend(self.mmd_features)
        self.n_features = len(self.feature_names)
    
    def extract_features(self, feature_store: Dict[str, Any]) -> Optional[List[float]]:
        # Get base features
        base_features = super().extract_features(feature_store)
        if base_features is None:
            return None
        
        # Add MMD features
        mmd_data = feature_store.get('mmd_features', [0.0] * 3)[:3]
        volatility = feature_store.get('volatility_30', 0.0)
        volume_profile = feature_store.get('volume_profile_skew', 0.0)
        
        regime_features = list(mmd_data) + [volatility, volume_profile]
        return base_features + regime_features