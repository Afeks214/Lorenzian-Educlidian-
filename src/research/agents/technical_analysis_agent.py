"""
Technical Analysis Research Agent
=================================

Agent 2: Advanced technical indicator research and signal generation
- Multi-timeframe analysis, indicator optimization, signal validation
- <8ms research latency target
- Integration with existing IndicatorEngine and matrix assemblers
"""

import asyncio
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import structlog
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger()

@dataclass
class TechnicalAnalysisResult:
    """Technical analysis research results"""
    timestamp: datetime = field(default_factory=datetime.now)
    signals: Dict[str, float] = field(default_factory=dict)
    indicators: Dict[str, float] = field(default_factory=dict)
    patterns: List[str] = field(default_factory=list)
    support_resistance: Dict[str, float] = field(default_factory=dict)
    momentum_analysis: Dict[str, float] = field(default_factory=dict)
    volatility_analysis: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    confidence_score: float = 0.0

class TechnicalAnalysisAgent:
    """Agent 2: Technical Analysis Research Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "TechnicalAnalysisAgent"
        self.agent_id = 2
        self.max_processing_time_ms = 8.0
        self.executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix='tech_analysis')
        
        # Initialize technical indicators
        self.indicators = {
            'sma': self._sma,
            'ema': self._ema,
            'rsi': self._rsi,
            'macd': self._macd,
            'bollinger': self._bollinger_bands,
            'stochastic': self._stochastic,
            'atr': self._atr,
            'adx': self._adx
        }
        
        # Performance metrics
        self.metrics = {'total_analyses': 0, 'avg_processing_time_ms': 0.0}
        
        logger.info(f"TechnicalAnalysisAgent initialized")
    
    async def research_technical_analysis(self, market_data: Dict[str, Any]) -> TechnicalAnalysisResult:
        """Perform parallel technical analysis research"""
        start_time = time.perf_counter()
        
        try:
            # Parallel processing tasks
            tasks = [
                self._analyze_indicators_async(market_data),
                self._detect_patterns_async(market_data),
                self._find_support_resistance_async(market_data),
                self._analyze_momentum_async(market_data),
                self._analyze_volatility_async(market_data),
                self._generate_signals_async(market_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return TechnicalAnalysisResult(
                signals=results[5] if not isinstance(results[5], Exception) else {},
                indicators=results[0] if not isinstance(results[0], Exception) else {},
                patterns=results[1] if not isinstance(results[1], Exception) else [],
                support_resistance=results[2] if not isinstance(results[2], Exception) else {},
                momentum_analysis=results[3] if not isinstance(results[3], Exception) else {},
                volatility_analysis=results[4] if not isinstance(results[4], Exception) else {},
                processing_time_ms=processing_time,
                confidence_score=0.8
            )
            
        except Exception as e:
            logger.error(f"Technical analysis failed", error=str(e))
            return TechnicalAnalysisResult(processing_time_ms=(time.perf_counter() - start_time) * 1000)
    
    async def _analyze_indicators_async(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze technical indicators asynchronously"""
        loop = asyncio.get_event_loop()
        
        def analyze_indicators():
            prices = np.array(market_data.get('prices', []))
            volumes = np.array(market_data.get('volumes', []))
            
            if len(prices) < 50:
                return {}
            
            return {
                'sma_20': self._sma(prices, 20),
                'sma_50': self._sma(prices, 50),
                'ema_12': self._ema(prices, 12),
                'ema_26': self._ema(prices, 26),
                'rsi_14': self._rsi(prices, 14),
                'macd': self._macd(prices),
                'bb_upper': self._bollinger_bands(prices)[0],
                'bb_lower': self._bollinger_bands(prices)[1],
                'stoch_k': self._stochastic(prices)[0],
                'atr_14': self._atr(prices, 14),
                'adx_14': self._adx(prices, 14)
            }
        
        return await loop.run_in_executor(self.executor, analyze_indicators)
    
    async def _detect_patterns_async(self, market_data: Dict[str, Any]) -> List[str]:
        """Detect chart patterns asynchronously"""
        loop = asyncio.get_event_loop()
        
        def detect_patterns():
            prices = np.array(market_data.get('prices', []))
            if len(prices) < 20:
                return []
            
            patterns = []
            
            # Simple pattern detection
            if self._is_ascending_triangle(prices):
                patterns.append("ascending_triangle")
            if self._is_descending_triangle(prices):
                patterns.append("descending_triangle")
            if self._is_head_and_shoulders(prices):
                patterns.append("head_and_shoulders")
            if self._is_double_top(prices):
                patterns.append("double_top")
            if self._is_double_bottom(prices):
                patterns.append("double_bottom")
            
            return patterns
        
        return await loop.run_in_executor(self.executor, detect_patterns)
    
    async def _find_support_resistance_async(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Find support and resistance levels asynchronously"""
        loop = asyncio.get_event_loop()
        
        def find_support_resistance():
            prices = np.array(market_data.get('prices', []))
            if len(prices) < 20:
                return {}
            
            # Find local minima and maxima
            support_levels = []
            resistance_levels = []
            
            for i in range(2, len(prices) - 2):
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    support_levels.append(prices[i])
                elif prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    resistance_levels.append(prices[i])
            
            return {
                'support_1': np.mean(support_levels[-3:]) if len(support_levels) >= 3 else 0,
                'support_2': np.mean(support_levels[-6:-3]) if len(support_levels) >= 6 else 0,
                'resistance_1': np.mean(resistance_levels[-3:]) if len(resistance_levels) >= 3 else 0,
                'resistance_2': np.mean(resistance_levels[-6:-3]) if len(resistance_levels) >= 6 else 0
            }
        
        return await loop.run_in_executor(self.executor, find_support_resistance)
    
    async def _analyze_momentum_async(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze momentum asynchronously"""
        loop = asyncio.get_event_loop()
        
        def analyze_momentum():
            prices = np.array(market_data.get('prices', []))
            if len(prices) < 20:
                return {}
            
            # Calculate various momentum indicators
            roc_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 else 0
            roc_20 = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 20 else 0
            
            return {
                'roc_10': roc_10,
                'roc_20': roc_20,
                'momentum_10': prices[-1] - prices[-11] if len(prices) > 10 else 0,
                'momentum_20': prices[-1] - prices[-21] if len(prices) > 20 else 0,
                'price_velocity': np.mean(np.diff(prices[-10:])) if len(prices) > 10 else 0
            }
        
        return await loop.run_in_executor(self.executor, analyze_momentum)
    
    async def _analyze_volatility_async(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze volatility asynchronously"""
        loop = asyncio.get_event_loop()
        
        def analyze_volatility():
            prices = np.array(market_data.get('prices', []))
            if len(prices) < 20:
                return {}
            
            returns = np.diff(np.log(prices))
            
            return {
                'realized_vol_10': np.std(returns[-10:]) * np.sqrt(252) if len(returns) > 10 else 0,
                'realized_vol_20': np.std(returns[-20:]) * np.sqrt(252) if len(returns) > 20 else 0,
                'parkinson_vol': self._parkinson_volatility(market_data),
                'garman_klass_vol': self._garman_klass_volatility(market_data)
            }
        
        return await loop.run_in_executor(self.executor, analyze_volatility)
    
    async def _generate_signals_async(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate trading signals asynchronously"""
        loop = asyncio.get_event_loop()
        
        def generate_signals():
            prices = np.array(market_data.get('prices', []))
            if len(prices) < 50:
                return {}
            
            # Generate various signals
            sma_20 = self._sma(prices, 20)
            sma_50 = self._sma(prices, 50)
            rsi = self._rsi(prices, 14)
            macd_signal = self._macd(prices)
            
            signals = {
                'sma_cross': 1.0 if sma_20 > sma_50 else -1.0,
                'rsi_signal': 1.0 if rsi < 30 else (-1.0 if rsi > 70 else 0.0),
                'macd_signal': 1.0 if macd_signal > 0 else -1.0,
                'price_momentum': 1.0 if prices[-1] > prices[-5] else -1.0,
                'volatility_breakout': 1.0 if self._atr(prices, 14) > np.mean([self._atr(prices, 14)]) * 1.5 else 0.0
            }
            
            # Aggregate signal
            signals['aggregate'] = np.mean(list(signals.values()))
            
            return signals
        
        return await loop.run_in_executor(self.executor, generate_signals)
    
    # Technical indicator implementations
    def _sma(self, prices: np.ndarray, period: int) -> float:
        """Simple Moving Average"""
        return np.mean(prices[-period:]) if len(prices) >= period else 0.0
    
    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return 0.0
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    
    def _macd(self, prices: np.ndarray) -> float:
        """MACD Signal"""
        if len(prices) < 26:
            return 0.0
        
        ema_12 = self._ema(prices, 12)
        ema_26 = self._ema(prices, 26)
        return ema_12 - ema_26
    
    def _bollinger_bands(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float]:
        """Bollinger Bands"""
        if len(prices) < period:
            return 0.0, 0.0
        
        sma = self._sma(prices, period)
        std = np.std(prices[-period:])
        
        return sma + 2 * std, sma - 2 * std
    
    def _stochastic(self, prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Stochastic Oscillator"""
        if len(prices) < period:
            return 50.0, 50.0
        
        low_min = np.min(prices[-period:])
        high_max = np.max(prices[-period:])
        
        if high_max == low_min:
            return 50.0, 50.0
        
        k = 100 * (prices[-1] - low_min) / (high_max - low_min)
        d = k  # Simplified
        
        return k, d
    
    def _atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Average True Range"""
        if len(prices) < period + 1:
            return 0.0
        
        # Simplified ATR using price differences
        true_ranges = []
        for i in range(1, len(prices)):
            tr = abs(prices[i] - prices[i-1])
            true_ranges.append(tr)
        
        return np.mean(true_ranges[-period:])
    
    def _adx(self, prices: np.ndarray, period: int = 14) -> float:
        """Average Directional Index"""
        if len(prices) < period + 1:
            return 0.0
        
        # Simplified ADX calculation
        price_changes = np.diff(prices)
        positive_changes = np.where(price_changes > 0, price_changes, 0)
        negative_changes = np.where(price_changes < 0, -price_changes, 0)
        
        avg_positive = np.mean(positive_changes[-period:])
        avg_negative = np.mean(negative_changes[-period:])
        
        if avg_positive + avg_negative == 0:
            return 0.0
        
        return 100 * abs(avg_positive - avg_negative) / (avg_positive + avg_negative)
    
    def _parkinson_volatility(self, market_data: Dict[str, Any]) -> float:
        """Parkinson volatility estimator"""
        highs = market_data.get('highs', [])
        lows = market_data.get('lows', [])
        
        if len(highs) < 20 or len(lows) < 20:
            return 0.0
        
        log_ratios = np.log(np.array(highs[-20:]) / np.array(lows[-20:]))
        return np.sqrt(np.mean(log_ratios ** 2) / (4 * np.log(2))) * np.sqrt(252)
    
    def _garman_klass_volatility(self, market_data: Dict[str, Any]) -> float:
        """Garman-Klass volatility estimator"""
        opens = market_data.get('opens', [])
        highs = market_data.get('highs', [])
        lows = market_data.get('lows', [])
        closes = market_data.get('closes', [])
        
        if len(opens) < 20 or len(highs) < 20 or len(lows) < 20 or len(closes) < 20:
            return 0.0
        
        # Simplified Garman-Klass
        hl_ratios = np.log(np.array(highs[-20:]) / np.array(lows[-20:]))
        co_ratios = np.log(np.array(closes[-20:]) / np.array(opens[-20:]))
        
        gk_estimator = 0.5 * hl_ratios**2 - (2*np.log(2) - 1) * co_ratios**2
        return np.sqrt(np.mean(gk_estimator)) * np.sqrt(252)
    
    # Pattern detection methods
    def _is_ascending_triangle(self, prices: np.ndarray) -> bool:
        """Detect ascending triangle pattern"""
        if len(prices) < 20:
            return False
        
        recent_highs = []
        recent_lows = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                recent_highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                recent_lows.append(prices[i])
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return False
        
        # Check if highs are relatively flat and lows are ascending
        high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        return abs(high_slope) < 0.001 and low_slope > 0.001
    
    def _is_descending_triangle(self, prices: np.ndarray) -> bool:
        """Detect descending triangle pattern"""
        if len(prices) < 20:
            return False
        
        recent_highs = []
        recent_lows = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                recent_highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                recent_lows.append(prices[i])
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return False
        
        # Check if lows are relatively flat and highs are descending
        high_slope = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        return abs(low_slope) < 0.001 and high_slope < -0.001
    
    def _is_head_and_shoulders(self, prices: np.ndarray) -> bool:
        """Detect head and shoulders pattern"""
        if len(prices) < 30:
            return False
        
        # Find significant peaks
        peaks = []
        for i in range(5, len(prices) - 5):
            if all(prices[i] > prices[i-j] for j in range(1, 6)) and \
               all(prices[i] > prices[i+j] for j in range(1, 6)):
                peaks.append((i, prices[i]))
        
        if len(peaks) < 3:
            return False
        
        # Check if middle peak is highest (head) and outer peaks are similar (shoulders)
        peaks.sort(key=lambda x: x[1], reverse=True)
        head = peaks[0]
        shoulders = peaks[1:3]
        
        # Check if head is in the middle
        head_pos = head[0]
        shoulder_positions = [s[0] for s in shoulders]
        
        return min(shoulder_positions) < head_pos < max(shoulder_positions)
    
    def _is_double_top(self, prices: np.ndarray) -> bool:
        """Detect double top pattern"""
        if len(prices) < 20:
            return False
        
        # Find the two highest peaks
        peaks = []
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append(prices[i])
        
        if len(peaks) < 2:
            return False
        
        peaks.sort(reverse=True)
        return abs(peaks[0] - peaks[1]) / peaks[0] < 0.02  # Within 2%
    
    def _is_double_bottom(self, prices: np.ndarray) -> bool:
        """Detect double bottom pattern"""
        if len(prices) < 20:
            return False
        
        # Find the two lowest troughs
        troughs = []
        for i in range(2, len(prices) - 2):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append(prices[i])
        
        if len(troughs) < 2:
            return False
        
        troughs.sort()
        return abs(troughs[0] - troughs[1]) / troughs[0] < 0.02  # Within 2%
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            'agent_name': self.name,
            'agent_id': self.agent_id,
            'metrics': self.metrics.copy()
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.executor.shutdown(wait=True)
        logger.info(f"TechnicalAnalysisAgent shutdown complete")

def create_technical_analysis_agent(config: Dict[str, Any]) -> TechnicalAnalysisAgent:
    """Create Technical Analysis Agent"""
    return TechnicalAnalysisAgent(config)