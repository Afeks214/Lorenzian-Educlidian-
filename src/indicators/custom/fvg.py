"""
Fair Value Gap (FVG) Detection Implementation
EXTRACTED EXACTLY from Strategy_Implementation.ipynb cells 2cc4dd54 and 64b01841
Applied ONLY to 5-minute data as per notebook
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Dict, Any
from src.indicators.base import BaseIndicator
from src.core.minimal_dependencies import EventBus, BarData


def detect_real_fvg(df, min_gap_ticks=2, min_gap_percent=0.01, max_age_bars=50):
    """
    Detects REAL Fair Value Gaps (FVGs) from actual price data.
    
    A Fair Value Gap occurs when:
    - Bullish FVG: The low of candle 3 is ABOVE the high of candle 1 (true gap in price)
    - Bearish FVG: The high of candle 3 is BELOW the low of candle 1 (true gap in price)
    - The gap must meet minimum size requirements
    
    Parameters:
        df: DataFrame with OHLC data
        min_gap_ticks: Minimum gap size in price points
        min_gap_percent: Minimum gap size as percentage of price
        max_age_bars: Maximum bars an FVG remains active before expiring
    
    Returns:
        dict with FVG detection results
    """
    if len(df) < 3:
        return {
            'bull_fvg_detected': np.zeros(len(df), dtype=bool),
            'bear_fvg_detected': np.zeros(len(df), dtype=bool),
            'bull_fvg_top': np.full(len(df), np.nan),
            'bull_fvg_bottom': np.full(len(df), np.nan),
            'bear_fvg_top': np.full(len(df), np.nan),
            'bear_fvg_bottom': np.full(len(df), np.nan),
            'bull_fvg_age': np.zeros(len(df), dtype=int),
            'bear_fvg_age': np.zeros(len(df), dtype=int),
            'is_bull_fvg_active': np.zeros(len(df), dtype=bool),
            'is_bear_fvg_active': np.zeros(len(df), dtype=bool)
        }
    
    n = len(df)
    
    # Initialize output arrays
    bull_fvg_detected = np.zeros(n, dtype=bool)
    bear_fvg_detected = np.zeros(n, dtype=bool)
    bull_fvg_top = np.full(n, np.nan)
    bull_fvg_bottom = np.full(n, np.nan)
    bear_fvg_top = np.full(n, np.nan)
    bear_fvg_bottom = np.full(n, np.nan)
    bull_fvg_age = np.zeros(n, dtype=int)
    bear_fvg_age = np.zeros(n, dtype=int)
    is_bull_fvg_active = np.zeros(n, dtype=bool)
    is_bear_fvg_active = np.zeros(n, dtype=bool)
    
    # Extract OHLC arrays for faster access
    high_arr = df['High'].values
    low_arr = df['Low'].values
    close_arr = df['Close'].values
    
    # Track active FVGs
    active_bull_fvgs = []  # List of (top, bottom, birth_idx)
    active_bear_fvgs = []  # List of (top, bottom, birth_idx)
    
    # Detect FVGs starting from index 2 (need 3 candles)
    for i in range(2, n):
        current_price = close_arr[i]
        
        # === REAL FVG DETECTION ===
        candle1_high = high_arr[i-2]  # First candle high
        candle1_low = low_arr[i-2]    # First candle low
        candle3_high = high_arr[i]    # Third candle (current) high
        candle3_low = low_arr[i]      # Third candle (current) low
        
        # Bullish FVG: Current candle low > First candle high (TRUE GAP)
        if candle3_low > candle1_high:
            gap_size = candle3_low - candle1_high
            gap_percent = gap_size / current_price
            
            # Validate gap meets minimum requirements
            if gap_size >= min_gap_ticks and gap_percent >= min_gap_percent:
                bull_fvg_detected[i] = True
                bull_fvg_bottom[i] = candle1_high  # Support level
                bull_fvg_top[i] = candle3_low      # Resistance level
                
                # Add to active FVGs
                active_bull_fvgs.append((candle3_low, candle1_high, i))
        
        # Bearish FVG: Current candle high < First candle low (TRUE GAP)
        elif candle3_high < candle1_low:
            gap_size = candle1_low - candle3_high
            gap_percent = gap_size / current_price
            
            # Validate gap meets minimum requirements
            if gap_size >= min_gap_ticks and gap_percent >= min_gap_percent:
                bear_fvg_detected[i] = True
                bear_fvg_top[i] = candle1_low      # Resistance level
                bear_fvg_bottom[i] = candle3_high  # Support level
                
                # Add to active FVGs
                active_bear_fvgs.append((candle1_low, candle3_high, i))
        
        # === FVG INVALIDATION LOGIC ===
        current_high = high_arr[i]
        current_low = low_arr[i]
        
        # Check bullish FVG invalidation (price breaks below support)
        active_bull_fvgs = [
            fvg for fvg in active_bull_fvgs 
            if not (current_low <= fvg[1] or (i - fvg[2]) > max_age_bars)
        ]
        
        # Check bearish FVG invalidation (price breaks above resistance)
        active_bear_fvgs = [
            fvg for fvg in active_bear_fvgs 
            if not (current_high >= fvg[0] or (i - fvg[2]) > max_age_bars)
        ]
        
        # === UPDATE ACTIVE STATUS ===
        if active_bull_fvgs:
            is_bull_fvg_active[i] = True
            # Use the most recent (youngest) FVG for current levels
            youngest_bull = max(active_bull_fvgs, key=lambda x: x[2])
            bull_fvg_top[i] = youngest_bull[0]
            bull_fvg_bottom[i] = youngest_bull[1]
            bull_fvg_age[i] = i - youngest_bull[2]
        
        if active_bear_fvgs:
            is_bear_fvg_active[i] = True
            # Use the most recent (youngest) FVG for current levels
            youngest_bear = max(active_bear_fvgs, key=lambda x: x[2])
            bear_fvg_top[i] = youngest_bear[0]
            bear_fvg_bottom[i] = youngest_bear[1]
            bear_fvg_age[i] = i - youngest_bear[2]
    
    return {
        'bull_fvg_detected': bull_fvg_detected,
        'bear_fvg_detected': bear_fvg_detected,
        'bull_fvg_top': bull_fvg_top,
        'bull_fvg_bottom': bull_fvg_bottom,
        'bear_fvg_top': bear_fvg_top,
        'bear_fvg_bottom': bear_fvg_bottom,
        'bull_fvg_age': bull_fvg_age,
        'bear_fvg_age': bear_fvg_age,
        'is_bull_fvg_active': is_bull_fvg_active,
        'is_bear_fvg_active': is_bear_fvg_active
    }


@njit
def detect_real_fvg_numba(high, low, close, n, min_gap_ticks=2.0, min_gap_percent=0.01, max_age_bars=50):
    """
    HIGH-PERFORMANCE REAL FVG DETECTION using Numba JIT compilation
    
    Detects authentic Fair Value Gaps from actual OHLC price data:
    - Bullish FVG: Candle 3 low > Candle 1 high (true price gap)
    - Bearish FVG: Candle 3 high < Candle 1 low (true price gap)
    - Includes gap size validation and proper invalidation logic
    
    This replaces the synthetic FVG generation with REAL market structure analysis.
    """
    # Initialize output arrays
    bull_fvg_detected = np.zeros(n, dtype=np.bool_)
    bear_fvg_detected = np.zeros(n, dtype=np.bool_)
    is_bull_fvg_active = np.zeros(n, dtype=np.bool_)
    is_bear_fvg_active = np.zeros(n, dtype=np.bool_)
    bull_fvg_top = np.full(n, np.nan)
    bull_fvg_bottom = np.full(n, np.nan)
    bear_fvg_top = np.full(n, np.nan)
    bear_fvg_bottom = np.full(n, np.nan)
    
    # Track active FVGs with simple arrays (Numba-compatible)
    max_concurrent_fvgs = 100  # Reasonable limit for performance
    
    # Bullish FVG tracking
    bull_fvg_tops = np.full(max_concurrent_fvgs, np.nan)
    bull_fvg_bottoms = np.full(max_concurrent_fvgs, np.nan)
    bull_fvg_birth_bars = np.full(max_concurrent_fvgs, -1, dtype=np.int32)
    bull_fvg_count = 0
    
    # Bearish FVG tracking
    bear_fvg_tops = np.full(max_concurrent_fvgs, np.nan)
    bear_fvg_bottoms = np.full(max_concurrent_fvgs, np.nan)
    bear_fvg_birth_bars = np.full(max_concurrent_fvgs, -1, dtype=np.int32)
    bear_fvg_count = 0
    
    # Process each bar starting from index 2 (need 3 candles for FVG)
    for i in range(2, n):
        current_price = close[i]
        current_high = high[i]
        current_low = low[i]
        
        # === REAL FVG DETECTION ===
        candle1_high = high[i-2]  # First candle (2 bars ago)
        candle1_low = low[i-2]
        candle3_high = high[i]    # Third candle (current)
        candle3_low = low[i]
        
        # BULLISH FVG: Current low > First high (TRUE GAP UP)
        if candle3_low > candle1_high:
            gap_size = candle3_low - candle1_high
            gap_percent = gap_size / current_price if current_price > 0 else 0
            
            # Validate gap meets minimum requirements
            if gap_size >= min_gap_ticks and gap_percent >= min_gap_percent:
                bull_fvg_detected[i] = True
                
                # Store FVG details
                if bull_fvg_count < max_concurrent_fvgs:
                    bull_fvg_tops[bull_fvg_count] = candle3_low      # Resistance
                    bull_fvg_bottoms[bull_fvg_count] = candle1_high  # Support
                    bull_fvg_birth_bars[bull_fvg_count] = i
                    bull_fvg_count += 1
        
        # BEARISH FVG: Current high < First low (TRUE GAP DOWN)
        elif candle3_high < candle1_low:
            gap_size = candle1_low - candle3_high
            gap_percent = gap_size / current_price if current_price > 0 else 0
            
            # Validate gap meets minimum requirements
            if gap_size >= min_gap_ticks and gap_percent >= min_gap_percent:
                bear_fvg_detected[i] = True
                
                # Store FVG details
                if bear_fvg_count < max_concurrent_fvgs:
                    bear_fvg_tops[bear_fvg_count] = candle1_low      # Resistance
                    bear_fvg_bottoms[bear_fvg_count] = candle3_high  # Support
                    bear_fvg_birth_bars[bear_fvg_count] = i
                    bear_fvg_count += 1
        
        # === FVG INVALIDATION AND CLEANUP ===
        
        # Check bullish FVG invalidation
        new_bull_count = 0
        for j in range(bull_fvg_count):
            if bull_fvg_birth_bars[j] >= 0:  # Valid FVG
                age = i - bull_fvg_birth_bars[j]
                support_level = bull_fvg_bottoms[j]
                
                # Invalidate if price breaks below support or too old
                if current_low <= support_level or age > max_age_bars:
                    # Invalidate this FVG
                    bull_fvg_birth_bars[j] = -1
                else:
                    # Keep this FVG, compact arrays
                    if new_bull_count != j:
                        bull_fvg_tops[new_bull_count] = bull_fvg_tops[j]
                        bull_fvg_bottoms[new_bull_count] = bull_fvg_bottoms[j]
                        bull_fvg_birth_bars[new_bull_count] = bull_fvg_birth_bars[j]
                    new_bull_count += 1
        bull_fvg_count = new_bull_count
        
        # Check bearish FVG invalidation
        new_bear_count = 0
        for j in range(bear_fvg_count):
            if bear_fvg_birth_bars[j] >= 0:  # Valid FVG
                age = i - bear_fvg_birth_bars[j]
                resistance_level = bear_fvg_tops[j]
                
                # Invalidate if price breaks above resistance or too old
                if current_high >= resistance_level or age > max_age_bars:
                    # Invalidate this FVG
                    bear_fvg_birth_bars[j] = -1
                else:
                    # Keep this FVG, compact arrays
                    if new_bear_count != j:
                        bear_fvg_tops[new_bear_count] = bear_fvg_tops[j]
                        bear_fvg_bottoms[new_bear_count] = bear_fvg_bottoms[j]
                        bear_fvg_birth_bars[new_bear_count] = bear_fvg_birth_bars[j]
                    new_bear_count += 1
        bear_fvg_count = new_bear_count
        
        # === UPDATE ACTIVE STATUS ===
        
        # Set active flags and levels
        if bull_fvg_count > 0:
            is_bull_fvg_active[i] = True
            # Use most recent (youngest) bullish FVG
            youngest_idx = 0
            youngest_bar = bull_fvg_birth_bars[0]
            for j in range(1, bull_fvg_count):
                if bull_fvg_birth_bars[j] > youngest_bar:
                    youngest_bar = bull_fvg_birth_bars[j]
                    youngest_idx = j
            
            bull_fvg_top[i] = bull_fvg_tops[youngest_idx]
            bull_fvg_bottom[i] = bull_fvg_bottoms[youngest_idx]
        
        if bear_fvg_count > 0:
            is_bear_fvg_active[i] = True
            # Use most recent (youngest) bearish FVG
            youngest_idx = 0
            youngest_bar = bear_fvg_birth_bars[0]
            for j in range(1, bear_fvg_count):
                if bear_fvg_birth_bars[j] > youngest_bar:
                    youngest_bar = bear_fvg_birth_bars[j]
                    youngest_idx = j
            
            bear_fvg_top[i] = bear_fvg_tops[youngest_idx]
            bear_fvg_bottom[i] = bear_fvg_bottoms[youngest_idx]
    
    return (bull_fvg_detected, bear_fvg_detected, is_bull_fvg_active, is_bear_fvg_active,
            bull_fvg_top, bull_fvg_bottom, bear_fvg_top, bear_fvg_bottom)


# Backward compatibility function (same interface as before)
@njit
def generate_fvg_data_fast(high, low, n):
    """
    REPLACED: Now uses REAL FVG detection instead of synthetic generation
    Maintains same interface for strategy compatibility
    """
    # Use realistic defaults for NQ futures based on data analysis
    close = (high + low) / 2.0  # Approximate close from high/low
    min_gap_ticks = 1.0    # 1 point minimum gap (realistic for NQ)
    min_gap_percent = 0.0005  # 5 basis points minimum (realistic)
    max_age_bars = 20  # Same as original for compatibility
    
    results = detect_real_fvg_numba(high, low, close, n, min_gap_ticks, min_gap_percent, max_age_bars)
    
    # Return only the 4 arrays expected by existing code
    return results[0], results[1], results[2], results[3]  # bull_detected, bear_detected, bull_active, bear_active


class FVGDetector(BaseIndicator):
    """FVG Detector - ONLY processes 5-minute data as per notebook"""
    
    def __init__(self, config: Dict[str, Any], event_bus: EventBus):
        super().__init__(config, event_bus)
        self.threshold = config.get('threshold', 0.001)
        self.lookback_period = config.get('lookback_period', 10)
        self.body_multiplier = config.get('body_multiplier', 1.5)
    
    def calculate_5m(self, bar: BarData) -> Dict[str, Any]:
        """REAL FVG detection on 5-minute bars using authentic market structure analysis"""
        self.update_5m_history(bar)
        if len(self.history_5m) < 3:
            return {
                'fvg_bullish_active': False, 
                'fvg_bearish_active': False,
                'fvg_nearest_level': 0.0,
                'fvg_age': 0,
                'fvg_mitigation_signal': False,
                'bull_fvg_top': 0.0,
                'bull_fvg_bottom': 0.0,
                'bear_fvg_top': 0.0,
                'bear_fvg_bottom': 0.0
            }
        
        # Convert to DataFrame for real FVG detection
        df = pd.DataFrame([{
            'High': b.high, 'Low': b.low, 'Open': b.open, 'Close': b.close
        } for b in self.history_5m])
        
        # Use NEW REAL FVG detection with actual market structure analysis
        high_array = df['High'].values
        low_array = df['Low'].values
        close_array = df['Close'].values
        n = len(df)
        
        # Apply REAL FVG detection algorithm
        (bull_fvg_detected, bear_fvg_detected, is_bull_fvg_active, is_bear_fvg_active,
         bull_fvg_top, bull_fvg_bottom, bear_fvg_top, bear_fvg_bottom) = detect_real_fvg_numba(
            high_array, low_array, close_array, n,
            min_gap_ticks=1.0,      # 1 point minimum gap (realistic for NQ)
            min_gap_percent=0.0005, # 5 basis points minimum (realistic)
            max_age_bars=30         # Maximum 30 bars (2.5 hours) active time
        )
        
        current_idx = n - 1
        
        # Determine mitigation signals (when FVG gets filled)
        fvg_mitigation = False
        if current_idx > 0:
            # Bullish FVG mitigation: was active, now inactive due to price break
            if (is_bull_fvg_active[current_idx-1] and not is_bull_fvg_active[current_idx] and
                not np.isnan(bull_fvg_bottom[current_idx-1]) and low_array[current_idx] <= bull_fvg_bottom[current_idx-1]):
                fvg_mitigation = True
            
            # Bearish FVG mitigation: was active, now inactive due to price break
            elif (is_bear_fvg_active[current_idx-1] and not is_bear_fvg_active[current_idx] and
                  not np.isnan(bear_fvg_top[current_idx-1]) and high_array[current_idx] >= bear_fvg_top[current_idx-1]):
                fvg_mitigation = True
        
        # Determine nearest significant level
        nearest_level = 0.0
        if is_bull_fvg_active[current_idx] and not np.isnan(bull_fvg_bottom[current_idx]):
            nearest_level = bull_fvg_bottom[current_idx]  # Support level
        elif is_bear_fvg_active[current_idx] and not np.isnan(bear_fvg_top[current_idx]):
            nearest_level = bear_fvg_top[current_idx]     # Resistance level
        
        # Calculate age of active FVG
        fvg_age = 0
        if is_bull_fvg_active[current_idx]:
            # Find when this bullish FVG was detected
            for i in range(current_idx, max(0, current_idx-30), -1):
                if bull_fvg_detected[i]:
                    fvg_age = current_idx - i
                    break
        elif is_bear_fvg_active[current_idx]:
            # Find when this bearish FVG was detected
            for i in range(current_idx, max(0, current_idx-30), -1):
                if bear_fvg_detected[i]:
                    fvg_age = current_idx - i
                    break
        
        return {
            'fvg_bullish_active': bool(is_bull_fvg_active[current_idx]),
            'fvg_bearish_active': bool(is_bear_fvg_active[current_idx]),
            'fvg_nearest_level': float(nearest_level),
            'fvg_age': int(fvg_age),
            'fvg_mitigation_signal': bool(fvg_mitigation),
            'bull_fvg_top': float(bull_fvg_top[current_idx]) if not np.isnan(bull_fvg_top[current_idx]) else 0.0,
            'bull_fvg_bottom': float(bull_fvg_bottom[current_idx]) if not np.isnan(bull_fvg_bottom[current_idx]) else 0.0,
            'bear_fvg_top': float(bear_fvg_top[current_idx]) if not np.isnan(bear_fvg_top[current_idx]) else 0.0,
            'bear_fvg_bottom': float(bear_fvg_bottom[current_idx]) if not np.isnan(bear_fvg_bottom[current_idx]) else 0.0
        }
    
    
    def get_current_values(self) -> Dict[str, Any]:
        return {'fvg_bullish_active': False, 'fvg_bearish_active': False}
    
    def reset(self) -> None:
        self.history_5m = []