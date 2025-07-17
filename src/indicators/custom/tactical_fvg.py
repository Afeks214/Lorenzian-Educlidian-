"""
Enhanced Fair Value Gap (FVG) Detection for Tactical System

Implements the exact mathematical formulas specified in the PRD:
- Body size filter with configurable multiplier
- Gap detection with precise conditions
- Mitigation tracking and age calculation
- Level proximity analysis

Mathematical Implementation:
- Bullish FVG: Low[i] > High[i-2] AND Body[i-1] > avg_body_size × body_multiplier
- Bearish FVG: High[i] < Low[i-2] AND Body[i-1] > avg_body_size × body_multiplier
- Body[i-1] = |Close[i-1] - Open[i-1]|
- avg_body_size = mean(|Close[j] - Open[j]|) for j in [i-lookback, i-1]

Author: Quantitative Engineer
Version: 1.0
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FVGLevel:
    """Container for FVG level information"""
    fvg_type: str  # 'bullish' or 'bearish'
    upper_level: float
    lower_level: float
    creation_bar: int
    mitigation_bar: Optional[int] = None
    age: int = 0
    is_active: bool = True


@dataclass
class FVGDetectionResult:
    """Container for FVG detection results"""
    fvg_bullish_active: bool
    fvg_bearish_active: bool
    fvg_nearest_level: float
    fvg_age: int
    fvg_mitigation_signal: bool
    active_fvgs: List[FVGLevel]
    detection_metadata: Dict[str, Any]


class TacticalFVGDetector:
    """
    Enhanced FVG Detector for Tactical System
    
    Implements exact mathematical formulas from PRD with:
    - Precise body size filtering
    - Gap detection with configurable parameters
    - Mitigation tracking
    - Age calculation
    - Level proximity analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Tactical FVG Detector
        
        Args:
            config: Configuration dictionary with detection parameters
        """
        self.config = config or self._default_config()
        
        # Extract parameters from config
        self.lookback_period = self.config.get('lookback_period', 10)
        self.body_multiplier = self.config.get('body_multiplier', 1.5)
        self.max_fvg_age = self.config.get('max_fvg_age', 50)
        self.mitigation_lookback = self.config.get('mitigation_lookback', 20)
        
        # Active FVG tracking
        self.active_fvgs: List[FVGLevel] = []
        self.bar_count = 0
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'bullish_detections': 0,
            'bearish_detections': 0,
            'mitigated_fvgs': 0
        }
        
        logger.info(f"TacticalFVGDetector initialized with lookback={self.lookback_period}, "
                   f"body_multiplier={self.body_multiplier}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for FVG detection"""
        return {
            'lookback_period': 10,
            'body_multiplier': 1.5,
            'max_fvg_age': 50,
            'mitigation_lookback': 20
        }
    
    def detect_fvg_5min(
        self,
        prices: np.ndarray,
        volumes: np.ndarray = None
    ) -> FVGDetectionResult:
        """
        Detect FVG patterns in 5-minute data using exact PRD formulas
        
        Args:
            prices: OHLC price array shape (n, 4) [Open, High, Low, Close]
            volumes: Volume array shape (n,) - optional
            
        Returns:
            FVGDetectionResult with detection details
        """
        if len(prices) < 3:
            return self._create_empty_result()
        
        try:
            # Update bar count
            self.bar_count = len(prices)
            
            # Calculate body sizes for filtering
            bodies = self._calculate_body_sizes(prices)
            
            # Calculate average body size
            avg_body_size = self._calculate_average_body_size(bodies)
            
            # Detect new FVGs
            new_fvgs = self._detect_new_fvgs(prices, bodies, avg_body_size)
            
            # Add new FVGs to active list
            self.active_fvgs.extend(new_fvgs)
            
            # Update existing FVGs
            self._update_active_fvgs(prices)
            
            # Clean up old FVGs
            self._cleanup_old_fvgs()
            
            # Generate detection result
            result = self._generate_detection_result(prices)
            
            # Update statistics
            self._update_detection_stats(new_fvgs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in FVG detection: {e}")
            return self._create_empty_result()
    
    def _calculate_body_sizes(self, prices: np.ndarray) -> np.ndarray:
        """Calculate body sizes: |Close - Open|"""
        return np.abs(prices[:, 3] - prices[:, 0])  # |Close - Open|
    
    def _calculate_average_body_size(self, bodies: np.ndarray) -> float:
        """Calculate average body size over lookback period"""
        if len(bodies) < self.lookback_period:
            return np.mean(bodies) if len(bodies) > 0 else 0.001
        
        # Use last lookback_period bars
        recent_bodies = bodies[-self.lookback_period:]
        avg_body = np.mean(recent_bodies)
        
        # Ensure minimum value to avoid division by zero
        return max(avg_body, 0.001)
    
    def _detect_new_fvgs(
        self,
        prices: np.ndarray,
        bodies: np.ndarray,
        avg_body_size: float
    ) -> List[FVGLevel]:
        """Detect new FVGs using exact PRD formulas"""
        new_fvgs = []
        
        if len(prices) < 3:
            return new_fvgs
        
        # Check last 3 bars for FVG pattern
        i = len(prices) - 1  # Current bar index
        
        # Extract required price levels
        high_i_minus_2 = prices[i-2, 1]  # High[i-2]
        low_i_minus_2 = prices[i-2, 2]   # Low[i-2]
        
        body_i_minus_1 = bodies[i-1]     # Body[i-1]
        
        high_i = prices[i, 1]            # High[i]
        low_i = prices[i, 2]             # Low[i]
        
        # Body size filter
        body_threshold = avg_body_size * self.body_multiplier
        
        if body_i_minus_1 <= body_threshold:
            return new_fvgs
        
        # Bullish FVG detection: Low[i] > High[i-2]
        if low_i > high_i_minus_2:
            bullish_fvg = FVGLevel(
                fvg_type='bullish',
                upper_level=low_i,
                lower_level=high_i_minus_2,
                creation_bar=i,
                age=0,
                is_active=True
            )
            new_fvgs.append(bullish_fvg)
            
            logger.debug(f"Bullish FVG detected at bar {i}: "
                        f"Low[{i}]={low_i:.4f} > High[{i-2}]={high_i_minus_2:.4f}")
        
        # Bearish FVG detection: High[i] < Low[i-2]
        if high_i < low_i_minus_2:
            bearish_fvg = FVGLevel(
                fvg_type='bearish',
                upper_level=low_i_minus_2,
                lower_level=high_i,
                creation_bar=i,
                age=0,
                is_active=True
            )
            new_fvgs.append(bearish_fvg)
            
            logger.debug(f"Bearish FVG detected at bar {i}: "
                        f"High[{i}]={high_i:.4f} < Low[{i-2}]={low_i_minus_2:.4f}")
        
        return new_fvgs
    
    def _update_active_fvgs(self, prices: np.ndarray):
        """Update active FVGs with age and mitigation status"""
        if len(prices) == 0:
            return
        
        current_bar = len(prices) - 1
        current_high = prices[current_bar, 1]
        current_low = prices[current_bar, 2]
        
        for fvg in self.active_fvgs:
            if not fvg.is_active:
                continue
            
            # Update age
            fvg.age = current_bar - fvg.creation_bar
            
            # Check for mitigation
            if fvg.mitigation_bar is None:
                is_mitigated = False
                
                if fvg.fvg_type == 'bullish':
                    # Bullish FVG mitigated when price goes below lower level
                    if current_low <= fvg.lower_level:
                        is_mitigated = True
                else:  # bearish
                    # Bearish FVG mitigated when price goes above upper level
                    if current_high >= fvg.upper_level:
                        is_mitigated = True
                
                if is_mitigated:
                    fvg.mitigation_bar = current_bar
                    fvg.is_active = False
                    logger.debug(f"{fvg.fvg_type.title()} FVG mitigated at bar {current_bar}")
            
            # Deactivate very old FVGs
            if fvg.age > self.max_fvg_age:
                fvg.is_active = False
    
    def _cleanup_old_fvgs(self):
        """Remove old and mitigated FVGs"""
        # Keep only active FVGs or recently mitigated ones
        self.active_fvgs = [
            fvg for fvg in self.active_fvgs
            if fvg.is_active or (fvg.mitigation_bar is not None and fvg.age < self.mitigation_lookback)
        ]
    
    def _generate_detection_result(self, prices: np.ndarray) -> FVGDetectionResult:
        """Generate FVG detection result"""
        # Find active FVGs
        active_bullish = [fvg for fvg in self.active_fvgs if fvg.fvg_type == 'bullish' and fvg.is_active]
        active_bearish = [fvg for fvg in self.active_fvgs if fvg.fvg_type == 'bearish' and fvg.is_active]
        
        # Check for mitigation signal (FVG just mitigated)
        mitigation_signal = False
        just_mitigated = [
            fvg for fvg in self.active_fvgs
            if fvg.mitigation_bar is not None and fvg.mitigation_bar == len(prices) - 1
        ]
        mitigation_signal = len(just_mitigated) > 0
        
        # Find nearest FVG level
        nearest_level = 0.0
        nearest_distance = float('inf')
        
        if len(prices) > 0:
            current_price = prices[-1, 3]  # Close price
            
            for fvg in self.active_fvgs:
                if not fvg.is_active:
                    continue
                
                # Calculate distance to FVG levels
                distances = [
                    abs(current_price - fvg.upper_level),
                    abs(current_price - fvg.lower_level)
                ]
                
                min_distance = min(distances)
                if min_distance < nearest_distance:
                    nearest_distance = min_distance
                    # Choose the closer level
                    if distances[0] < distances[1]:
                        nearest_level = fvg.upper_level
                    else:
                        nearest_level = fvg.lower_level
        
        # Calculate average age of active FVGs
        avg_age = 0
        active_fvgs = [fvg for fvg in self.active_fvgs if fvg.is_active]
        if active_fvgs:
            avg_age = int(np.mean([fvg.age for fvg in active_fvgs]))
        
        # Create metadata
        metadata = {
            'total_active_fvgs': len(active_fvgs),
            'bullish_count': len(active_bullish),
            'bearish_count': len(active_bearish),
            'just_mitigated_count': len(just_mitigated),
            'nearest_distance': nearest_distance if nearest_distance != float('inf') else 0.0,
            'detection_stats': self.detection_stats.copy()
        }
        
        return FVGDetectionResult(
            fvg_bullish_active=len(active_bullish) > 0,
            fvg_bearish_active=len(active_bearish) > 0,
            fvg_nearest_level=nearest_level,
            fvg_age=avg_age,
            fvg_mitigation_signal=mitigation_signal,
            active_fvgs=active_fvgs,
            detection_metadata=metadata
        )
    
    def _update_detection_stats(self, new_fvgs: List[FVGLevel]):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += len(new_fvgs)
        
        for fvg in new_fvgs:
            if fvg.fvg_type == 'bullish':
                self.detection_stats['bullish_detections'] += 1
            else:
                self.detection_stats['bearish_detections'] += 1
        
        # Count mitigated FVGs
        mitigated_count = sum(1 for fvg in self.active_fvgs if fvg.mitigation_bar is not None)
        self.detection_stats['mitigated_fvgs'] = mitigated_count
    
    def _create_empty_result(self) -> FVGDetectionResult:
        """Create empty FVG detection result"""
        return FVGDetectionResult(
            fvg_bullish_active=False,
            fvg_bearish_active=False,
            fvg_nearest_level=0.0,
            fvg_age=0,
            fvg_mitigation_signal=False,
            active_fvgs=[],
            detection_metadata={}
        )
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of FVG detection performance"""
        active_count = len([fvg for fvg in self.active_fvgs if fvg.is_active])
        
        summary = {
            'active_fvgs': active_count,
            'total_fvgs_tracked': len(self.active_fvgs),
            'detection_stats': self.detection_stats,
            'configuration': {
                'lookback_period': self.lookback_period,
                'body_multiplier': self.body_multiplier,
                'max_fvg_age': self.max_fvg_age
            }
        }
        
        return summary
    
    def reset(self):
        """Reset detector state"""
        self.active_fvgs.clear()
        self.bar_count = 0
        self.detection_stats = {
            'total_detections': 0,
            'bullish_detections': 0,
            'bearish_detections': 0,
            'mitigated_fvgs': 0
        }
        
        logger.info("TacticalFVGDetector reset")


@njit
def calculate_momentum_5_bar(prices: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate 5-bar momentum using exact PRD formula
    
    Formula: momentum_5 = ((P_current - P_5bars_ago) / P_5bars_ago) × 100
    Normalized: tanh(momentum_5 / 5.0)
    
    Args:
        prices: Close prices array
        lookback: Number of bars to look back (default: 5)
        
    Returns:
        Momentum array with tanh normalization
    """
    n = len(prices)
    momentum = np.zeros(n, dtype=np.float32)
    
    for i in range(lookback, n):
        p_current = prices[i]
        p_ago = prices[i - lookback]
        
        if p_ago != 0:
            # Calculate percentage change
            pct_change = ((p_current - p_ago) / p_ago) * 100.0
            
            # Clip to [-10%, +10%] range
            pct_change = max(-10.0, min(10.0, pct_change))
            
            # Apply tanh normalization
            momentum[i] = np.tanh(pct_change / 5.0)
    
    return momentum


@njit
def calculate_volume_ratio_ema(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Calculate volume ratio using 20-period EMA with exact PRD formula
    
    Formula:
    - EMA_volume[i] = α × Volume[i] + (1-α) × EMA_volume[i-1]
    - α = 2 / (period + 1) = 0.095 for 20-period
    - volume_ratio = Volume_current / EMA_volume
    - log_ratio = log(1 + max(0, volume_ratio - 1))
    - normalized_ratio = tanh(log_ratio)
    
    Args:
        volumes: Volume array
        period: EMA period (default: 20)
        
    Returns:
        Volume ratio array with log transform and tanh normalization
    """
    n = len(volumes)
    if n == 0:
        return np.array([], dtype=np.float32)
    
    # Calculate EMA alpha
    alpha = 2.0 / (period + 1)
    
    # Initialize arrays
    ema_volume = np.zeros(n, dtype=np.float32)
    volume_ratio = np.zeros(n, dtype=np.float32)
    
    # Initialize EMA with first volume
    ema_volume[0] = max(volumes[0], 1.0)  # Avoid zero
    volume_ratio[0] = 1.0
    
    # Calculate EMA and ratios
    for i in range(1, n):
        # Update EMA
        ema_volume[i] = alpha * volumes[i] + (1 - alpha) * ema_volume[i-1]
        
        # Ensure minimum EMA value
        ema_volume[i] = max(ema_volume[i], 1.0)
        
        # Calculate volume ratio
        ratio = volumes[i] / ema_volume[i]
        
        # Apply log transform: log(1 + max(0, ratio - 1))
        log_ratio = np.log(1.0 + max(0.0, ratio - 1.0))
        
        # Apply tanh normalization
        volume_ratio[i] = np.tanh(log_ratio)
    
    return volume_ratio


def create_tactical_fvg_detector(config: Optional[Dict[str, Any]] = None) -> TacticalFVGDetector:
    """
    Factory function to create tactical FVG detector
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured TacticalFVGDetector instance
    """
    return TacticalFVGDetector(config)