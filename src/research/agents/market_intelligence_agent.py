"""
Market Intelligence Research Agent
==================================

Agent 1: Real-time market regime detection and pattern analysis
- Parallel processing for regime detection, correlation analysis, volatility clustering
- <10ms research latency target
- Integration with existing RegimeDetectionAgent and StrategicMARLComponent

Author: 7-Agent Parallel Research System
Date: 2025-07-17
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
import json

logger = structlog.get_logger()

@dataclass
class MarketIntelligenceResult:
    """Market intelligence research results"""
    timestamp: datetime = field(default_factory=datetime.now)
    regime_detected: str = "ranging"
    regime_confidence: float = 0.0
    volatility_clustering: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    market_stress_level: float = 0.0
    trend_strength: float = 0.0
    liquidity_conditions: Dict[str, float] = field(default_factory=dict)
    macro_factors: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    confidence_score: float = 0.0

@dataclass
class RegimeSignals:
    """Real-time regime detection signals"""
    volatility_regime: str = "normal"
    trend_regime: str = "sideways"
    momentum_regime: str = "neutral"
    volume_regime: str = "normal"
    correlation_regime: str = "stable"
    market_microstructure: str = "normal"

class MarketIntelligenceAgent:
    """
    Market Intelligence Research Agent for real-time regime detection
    
    Features:
    - Real-time regime detection with <10ms latency
    - Parallel volatility clustering analysis
    - Correlation matrix monitoring
    - Market stress level assessment
    - Trend strength measurement
    - Liquidity condition analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "MarketIntelligenceAgent"
        self.agent_id = 1
        
        # Performance targets
        self.max_processing_time_ms = config.get('max_processing_time_ms', 10.0)
        
        # Initialize neural networks for regime detection
        self.regime_detector = self._create_regime_detector()
        self.volatility_clusterer = self._create_volatility_clusterer()
        self.correlation_monitor = self._create_correlation_monitor()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix='market_intel')
        
        # Performance metrics
        self.metrics = {
            'total_analyses': 0,
            'avg_processing_time_ms': 0.0,
            'regime_accuracy': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Caching system
        self.cache = {}
        self.cache_ttl = 5.0  # 5 seconds
        
        # Historical data for analysis
        self.price_history = deque(maxlen=1000)
        self.volatility_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=500)
        
        logger.info(f"MarketIntelligenceAgent initialized", 
                   max_latency_ms=self.max_processing_time_ms,
                   thread_pool_workers=4)
    
    def _create_regime_detector(self) -> nn.Module:
        """Create neural network for regime detection"""
        return nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # 5 regime types
        )
    
    def _create_volatility_clusterer(self) -> nn.Module:
        """Create neural network for volatility clustering"""
        return nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Low, Medium, High volatility clusters
        )
    
    def _create_correlation_monitor(self) -> nn.Module:
        """Create neural network for correlation monitoring"""
        return nn.Sequential(
            nn.Linear(9, 32),  # 3x3 correlation matrix flattened
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # Correlation strength levels
        )
    
    async def research_market_intelligence(self, 
                                         market_data: Dict[str, Any],
                                         context: Optional[Dict[str, Any]] = None) -> MarketIntelligenceResult:
        """
        Perform parallel market intelligence research
        
        Args:
            market_data: Real-time market data
            context: Additional context for analysis
            
        Returns:
            MarketIntelligenceResult with comprehensive analysis
        """
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(market_data)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Parallel processing tasks
            tasks = [
                self._detect_regime_async(market_data),
                self._analyze_volatility_clustering_async(market_data),
                self._monitor_correlations_async(market_data),
                self._assess_market_stress_async(market_data),
                self._measure_trend_strength_async(market_data),
                self._analyze_liquidity_conditions_async(market_data),
                self._evaluate_macro_factors_async(market_data)
            ]
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            regime_result = results[0] if not isinstance(results[0], Exception) else {"regime": "unknown", "confidence": 0.0}
            volatility_result = results[1] if not isinstance(results[1], Exception) else {}
            correlation_result = results[2] if not isinstance(results[2], Exception) else np.eye(3)
            stress_result = results[3] if not isinstance(results[3], Exception) else 0.0
            trend_result = results[4] if not isinstance(results[4], Exception) else 0.0
            liquidity_result = results[5] if not isinstance(results[5], Exception) else {}
            macro_result = results[6] if not isinstance(results[6], Exception) else {}
            
            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Create result
            result = MarketIntelligenceResult(
                timestamp=datetime.now(),
                regime_detected=regime_result["regime"],
                regime_confidence=regime_result["confidence"],
                volatility_clustering=volatility_result,
                correlation_matrix=correlation_result,
                market_stress_level=stress_result,
                trend_strength=trend_result,
                liquidity_conditions=liquidity_result,
                macro_factors=macro_result,
                processing_time_ms=processing_time,
                confidence_score=self._calculate_overall_confidence(regime_result, volatility_result, correlation_result)
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update metrics
            self._update_metrics(processing_time, result)
            
            # Validate performance
            if processing_time > self.max_processing_time_ms:
                logger.warning(f"Market intelligence processing exceeded target",
                              actual_ms=processing_time,
                              target_ms=self.max_processing_time_ms)
            
            return result
            
        except Exception as e:
            logger.error(f"Market intelligence research failed", error=str(e))
            
            # Return safe fallback
            processing_time = (time.perf_counter() - start_time) * 1000
            return MarketIntelligenceResult(
                processing_time_ms=processing_time,
                confidence_score=0.0
            )
    
    async def _detect_regime_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market regime asynchronously"""
        loop = asyncio.get_event_loop()
        
        def detect_regime():
            # Extract features from market data
            features = self._extract_regime_features(market_data)
            
            # Neural network inference
            with torch.no_grad():
                regime_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                regime_probs = torch.softmax(self.regime_detector(regime_tensor), dim=-1)
                
                # Regime mapping
                regimes = ["trending_up", "trending_down", "ranging", "volatile", "breakout"]
                regime_idx = torch.argmax(regime_probs).item()
                confidence = regime_probs[0, regime_idx].item()
                
                return {
                    "regime": regimes[regime_idx],
                    "confidence": confidence,
                    "probabilities": regime_probs.squeeze().tolist()
                }
        
        return await loop.run_in_executor(self.executor, detect_regime)
    
    async def _analyze_volatility_clustering_async(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze volatility clustering asynchronously"""
        loop = asyncio.get_event_loop()
        
        def analyze_volatility():
            # Extract volatility features
            volatility_features = self._extract_volatility_features(market_data)
            
            # Clustering analysis
            with torch.no_grad():
                vol_tensor = torch.tensor(volatility_features, dtype=torch.float32).unsqueeze(0)
                cluster_probs = torch.softmax(self.volatility_clusterer(vol_tensor), dim=-1)
                
                return {
                    "low_vol_prob": cluster_probs[0, 0].item(),
                    "medium_vol_prob": cluster_probs[0, 1].item(),
                    "high_vol_prob": cluster_probs[0, 2].item(),
                    "current_volatility": np.std(volatility_features[-20:]) if len(volatility_features) > 20 else 0.0,
                    "volatility_trend": self._calculate_volatility_trend(volatility_features)
                }
        
        return await loop.run_in_executor(self.executor, analyze_volatility)
    
    async def _monitor_correlations_async(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Monitor correlations asynchronously"""
        loop = asyncio.get_event_loop()
        
        def monitor_correlations():
            # Extract correlation features
            correlation_data = self._extract_correlation_data(market_data)
            
            # Calculate correlation matrix
            if len(correlation_data) > 10:
                correlation_matrix = np.corrcoef(correlation_data.T)
                
                # Neural network analysis
                with torch.no_grad():
                    corr_tensor = torch.tensor(correlation_matrix.flatten(), dtype=torch.float32).unsqueeze(0)
                    if corr_tensor.shape[1] == 9:  # 3x3 matrix
                        strength_probs = torch.softmax(self.correlation_monitor(corr_tensor), dim=-1)
                
                return correlation_matrix
            else:
                return np.eye(3)
        
        return await loop.run_in_executor(self.executor, monitor_correlations)
    
    async def _assess_market_stress_async(self, market_data: Dict[str, Any]) -> float:
        """Assess market stress level asynchronously"""
        loop = asyncio.get_event_loop()
        
        def assess_stress():
            # Calculate stress indicators
            volatility = market_data.get('volatility', 0.0)
            volume_spike = market_data.get('volume_ratio', 1.0)
            bid_ask_spread = market_data.get('spread', 0.0)
            
            # Composite stress score
            stress_score = (
                min(volatility * 5, 1.0) * 0.4 +
                min(abs(volume_spike - 1.0) * 2, 1.0) * 0.3 +
                min(bid_ask_spread * 1000, 1.0) * 0.3
            )
            
            return float(np.clip(stress_score, 0.0, 1.0))
        
        return await loop.run_in_executor(self.executor, assess_stress)
    
    async def _measure_trend_strength_async(self, market_data: Dict[str, Any]) -> float:
        """Measure trend strength asynchronously"""
        loop = asyncio.get_event_loop()
        
        def measure_trend():
            # Extract price data
            prices = market_data.get('prices', [])
            if len(prices) < 20:
                return 0.0
            
            # Calculate trend strength using multiple indicators
            price_array = np.array(prices[-50:])
            
            # Linear regression slope
            x = np.arange(len(price_array))
            slope = np.polyfit(x, price_array, 1)[0]
            
            # Normalize trend strength
            price_range = np.max(price_array) - np.min(price_array)
            normalized_slope = slope / price_range if price_range > 0 else 0.0
            
            return float(np.clip(abs(normalized_slope), 0.0, 1.0))
        
        return await loop.run_in_executor(self.executor, measure_trend)
    
    async def _analyze_liquidity_conditions_async(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze liquidity conditions asynchronously"""
        loop = asyncio.get_event_loop()
        
        def analyze_liquidity():
            volume = market_data.get('volume', 0.0)
            avg_volume = market_data.get('avg_volume', 1.0)
            bid_ask_spread = market_data.get('spread', 0.0)
            
            return {
                "volume_ratio": volume / avg_volume if avg_volume > 0 else 1.0,
                "spread_normalized": bid_ask_spread * 10000,  # Convert to basis points
                "liquidity_score": self._calculate_liquidity_score(volume, avg_volume, bid_ask_spread),
                "market_depth": market_data.get('market_depth', 0.5)
            }
        
        return await loop.run_in_executor(self.executor, analyze_liquidity)
    
    async def _evaluate_macro_factors_async(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate macro factors asynchronously"""
        loop = asyncio.get_event_loop()
        
        def evaluate_macro():
            # Simplified macro factor evaluation
            return {
                "vix_level": market_data.get('vix', 20.0),
                "yield_curve": market_data.get('yield_curve', 0.0),
                "dollar_strength": market_data.get('dxy', 100.0),
                "commodity_trend": market_data.get('commodity_index', 0.0),
                "economic_surprise": market_data.get('economic_surprise', 0.0)
            }
        
        return await loop.run_in_executor(self.executor, evaluate_macro)
    
    def _extract_regime_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for regime detection"""
        # Extract 48 features from market data
        features = np.zeros(48)
        
        # Price-based features
        prices = market_data.get('prices', [])
        if len(prices) >= 20:
            features[0] = np.mean(prices[-20:])  # Recent price average
            features[1] = np.std(prices[-20:])   # Recent volatility
            features[2] = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] != 0 else 0  # Return
        
        # Volume features
        volumes = market_data.get('volumes', [])
        if len(volumes) >= 20:
            features[3] = np.mean(volumes[-20:])
            features[4] = np.std(volumes[-20:])
        
        # Fill remaining features with market data
        for i in range(5, 48):
            features[i] = market_data.get(f'feature_{i}', 0.0)
        
        return features
    
    def _extract_volatility_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract volatility features"""
        prices = market_data.get('prices', [])
        if len(prices) < 100:
            return np.zeros(100)
        
        # Calculate returns
        returns = np.diff(np.log(prices[-101:]))
        
        # Pad if necessary
        if len(returns) < 100:
            returns = np.pad(returns, (0, 100 - len(returns)), 'constant')
        
        return returns
    
    def _extract_correlation_data(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract data for correlation analysis"""
        # Extract 3 series for correlation analysis
        prices = market_data.get('prices', [])
        volumes = market_data.get('volumes', [])
        
        if len(prices) < 50:
            return np.random.randn(50, 3)
        
        # Create synthetic correlated series
        price_returns = np.diff(np.log(prices[-51:]))
        volume_series = np.array(volumes[-50:]) if len(volumes) >= 50 else np.ones(50)
        
        # Third series - synthetic
        synthetic_series = price_returns * 0.7 + np.random.randn(50) * 0.3
        
        return np.column_stack([price_returns, volume_series, synthetic_series])
    
    def _calculate_volatility_trend(self, volatility_features: np.ndarray) -> float:
        """Calculate volatility trend"""
        if len(volatility_features) < 20:
            return 0.0
        
        recent_vol = np.std(volatility_features[-10:])
        past_vol = np.std(volatility_features[-20:-10])
        
        if past_vol == 0:
            return 0.0
        
        return (recent_vol - past_vol) / past_vol
    
    def _calculate_liquidity_score(self, volume: float, avg_volume: float, spread: float) -> float:
        """Calculate overall liquidity score"""
        volume_component = min(volume / avg_volume, 2.0) / 2.0 if avg_volume > 0 else 0.5
        spread_component = max(0.0, 1.0 - spread * 10000)  # Lower spread = higher liquidity
        
        return (volume_component + spread_component) / 2.0
    
    def _calculate_overall_confidence(self, regime_result: Dict, volatility_result: Dict, correlation_result: np.ndarray) -> float:
        """Calculate overall confidence score"""
        regime_confidence = regime_result.get("confidence", 0.0)
        volatility_confidence = max(volatility_result.get("high_vol_prob", 0.0), 
                                   volatility_result.get("medium_vol_prob", 0.0),
                                   volatility_result.get("low_vol_prob", 0.0))
        
        # Correlation confidence based on matrix stability
        correlation_confidence = 1.0 - np.std(correlation_result) if correlation_result.size > 0 else 0.0
        
        return (regime_confidence + volatility_confidence + correlation_confidence) / 3.0
    
    def _generate_cache_key(self, market_data: Dict[str, Any]) -> str:
        """Generate cache key for market data"""
        # Simple hash of key market data points
        key_data = {
            'last_price': market_data.get('prices', [0])[-1] if market_data.get('prices') else 0,
            'volume': market_data.get('volume', 0),
            'timestamp': int(time.time() / self.cache_ttl)  # Round to cache TTL
        }
        return str(hash(str(key_data)))
    
    def _get_cached_result(self, cache_key: str) -> Optional[MarketIntelligenceResult]:
        """Get cached result if available"""
        if cache_key in self.cache:
            timestamp, result = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: MarketIntelligenceResult):
        """Cache result with timestamp"""
        self.cache[cache_key] = (time.time(), result)
        
        # Clean old cache entries
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]
    
    def _update_metrics(self, processing_time: float, result: MarketIntelligenceResult):
        """Update performance metrics"""
        self.metrics['total_analyses'] += 1
        self.metrics['avg_processing_time_ms'] = (
            self.metrics['avg_processing_time_ms'] * 0.9 + processing_time * 0.1
        )
        
        # Update cache hit rate
        cache_hits = sum(1 for k, (t, r) in self.cache.items() if time.time() - t < self.cache_ttl)
        self.metrics['cache_hit_rate'] = cache_hits / max(1, len(self.cache))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            'agent_name': self.name,
            'agent_id': self.agent_id,
            'metrics': self.metrics.copy(),
            'config': {
                'max_processing_time_ms': self.max_processing_time_ms,
                'cache_ttl': self.cache_ttl,
                'thread_pool_workers': self.executor._max_workers
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.executor.shutdown(wait=True)
        logger.info(f"MarketIntelligenceAgent shutdown complete")


# Factory function
def create_market_intelligence_agent(config: Dict[str, Any]) -> MarketIntelligenceAgent:
    """Create Market Intelligence Agent with configuration"""
    return MarketIntelligenceAgent(config)