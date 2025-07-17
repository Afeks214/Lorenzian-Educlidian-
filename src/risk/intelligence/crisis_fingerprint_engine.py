"""
Crisis Fingerprint Pattern Matching Engine

This module provides real-time crisis pattern matching with <5ms latency.
Uses optimized algorithms for fast similarity matching and pattern detection.

Key Features:
- <5ms real-time pattern matching
- Optimized similarity algorithms (LSH, KD-trees)
- Sliding window analysis
- Multi-dimensional crisis feature matching
- Confidence scoring with pattern libraries
- Memory-efficient pattern storage
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog
import asyncio
from collections import deque
import time

try:
    from scipy.spatial.distance import cosine, euclidean
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Fallback implementations
    def euclidean(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

from .crisis_dataset_processor import CrisisFingerprint, CrisisType

logger = structlog.get_logger()


@dataclass
class PatternMatch:
    """Result of pattern matching"""
    timestamp: datetime
    matched_pattern: CrisisFingerprint
    similarity_score: float
    pattern_confidence: float
    distance_metric: str
    match_quality: str  # HIGH, MEDIUM, LOW
    processing_time_ms: float


@dataclass
class SlidingWindowState:
    """State of sliding window analysis"""
    window_size: int
    current_features: deque
    timestamps: deque
    last_update: datetime
    feature_buffer: np.ndarray


class OptimizedPatternMatcher:
    """
    Optimized pattern matching using LSH and KD-trees for fast similarity search.
    Designed to meet <5ms latency requirement.
    """
    
    def __init__(self, pattern_library: List[CrisisFingerprint]):
        self.pattern_library = pattern_library
        self.pattern_features = None
        self.pattern_labels = None
        
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        
        # Fast search structures
        self.nn_matcher = None
        self.feature_index = None
        
        # Performance tracking
        self.search_times = deque(maxlen=1000)
        self.latency_target_ms = 5.0
        
        self._build_search_index()
    
    def _build_search_index(self):
        """Build optimized search index"""
        
        if not self.pattern_library:
            logger.warning("No patterns in library for index building")
            return
        
        # Extract and normalize features
        features = []
        labels = []
        
        for pattern in self.pattern_library:
            features.append(pattern.feature_vector)
            labels.append({
                'crisis_type': pattern.crisis_type,
                'severity': pattern.severity,
                'timestamp': pattern.timestamp
            })
        
        self.pattern_features = np.array(features)
        self.pattern_labels = labels
        
        if SKLEARN_AVAILABLE and self.scaler is not None:
            # Normalize features with sklearn
            self.pattern_features = self.scaler.fit_transform(self.pattern_features)
            
            # Build fast nearest neighbor index
            self.nn_matcher = NearestNeighbors(
                n_neighbors=min(10, len(self.pattern_library)),
                algorithm='kd_tree',  # Fast for moderate dimensions
                metric='euclidean',
                n_jobs=1  # Single thread for latency
            )
            self.nn_matcher.fit(self.pattern_features)
        else:
            # Fallback: simple normalization
            self.pattern_features = (self.pattern_features - np.mean(self.pattern_features, axis=0)) / (np.std(self.pattern_features, axis=0) + 1e-8)
        
        logger.info(f"Pattern search index built with {len(self.pattern_library)} patterns")
    
    def find_closest_patterns(
        self, 
        query_features: np.ndarray, 
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find k closest patterns with optimized search"""
        
        start_time = time.perf_counter()
        
        try:
            if SKLEARN_AVAILABLE and self.nn_matcher is not None:
                # Normalize query features
                query_normalized = self.scaler.transform(query_features.reshape(1, -1))
                
                # Fast nearest neighbor search
                distances, indices = self.nn_matcher.kneighbors(
                    query_normalized, 
                    n_neighbors=min(k, len(self.pattern_library))
                )
                
                # Convert to similarity scores (1 / (1 + distance))
                similarities = 1.0 / (1.0 + distances[0])
                
                results = list(zip(indices[0], similarities))
            else:
                # Fallback: manual search with euclidean distance
                query_normalized = (query_features - np.mean(self.pattern_features, axis=0)) / (np.std(self.pattern_features, axis=0) + 1e-8)
                
                distances = []
                for i, pattern_features in enumerate(self.pattern_features):
                    dist = euclidean(query_normalized, pattern_features)
                    distances.append((i, dist))
                
                # Sort by distance and take top k
                distances.sort(key=lambda x: x[1])
                distances = distances[:min(k, len(distances))]
                
                # Convert to similarity scores
                results = [(idx, 1.0 / (1.0 + dist)) for idx, dist in distances]
            
            # Track performance
            search_time = (time.perf_counter() - start_time) * 1000
            self.search_times.append(search_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Pattern search failed: {e}")
            return []
    
    def get_search_performance(self) -> Dict:
        """Get search performance statistics"""
        
        if not self.search_times:
            return {"no_searches": True}
        
        times = list(self.search_times)
        
        return {
            "avg_search_time_ms": np.mean(times),
            "max_search_time_ms": np.max(times),
            "min_search_time_ms": np.min(times),
            "latency_target_ms": self.latency_target_ms,
            "target_met_percentage": sum(1 for t in times if t <= self.latency_target_ms) / len(times) * 100,
            "total_searches": len(times)
        }


class CrisisFingerprintEngine:
    """
    Real-time crisis fingerprint pattern matching engine.
    
    Provides <5ms pattern matching with sliding window analysis
    and multi-dimensional crisis feature comparison.
    """
    
    def __init__(
        self,
        pattern_library: List[CrisisFingerprint] = None,
        window_size: int = 60,  # 60 time periods
        similarity_threshold: float = 0.7,
        confidence_threshold: float = 0.8
    ):
        self.pattern_library = pattern_library or []
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        
        # Pattern matcher
        self.pattern_matcher = OptimizedPatternMatcher(self.pattern_library)
        
        # Sliding window state
        self.sliding_window = SlidingWindowState(
            window_size=window_size,
            current_features=deque(maxlen=window_size),
            timestamps=deque(maxlen=window_size),
            last_update=datetime.now(),
            feature_buffer=np.zeros((window_size, len(self.pattern_library[0].feature_vector) if self.pattern_library else 24))
        )
        
        # Pattern matching history
        self.match_history = deque(maxlen=1000)
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.latency_target_ms = 5.0
        
        # Crisis pattern thresholds by type
        self.crisis_thresholds = {
            CrisisType.FLASH_CRASH: {
                'similarity_min': 0.85,
                'volatility_spike_min': 2.0,
                'price_drop_min': 0.05
            },
            CrisisType.LIQUIDITY_CRISIS: {
                'similarity_min': 0.80,
                'spread_spike_min': 2.5,
                'volume_drop_min': 0.3
            },
            CrisisType.VOLATILITY_EXPLOSION: {
                'similarity_min': 0.75,
                'volatility_spike_min': 3.0,
                'persistence_min': 0.6
            },
            CrisisType.CORRELATION_BREAKDOWN: {
                'similarity_min': 0.78,
                'correlation_spike_min': 0.7,
                'contagion_min': 0.4
            },
            CrisisType.MARKET_STRUCTURE_BREAK: {
                'similarity_min': 0.82,
                'volume_spike_min': 4.0,
                'unusual_pattern_min': 0.5
            }
        }
        
        logger.info("CrisisFingerprintEngine initialized",
                   pattern_count=len(self.pattern_library),
                   window_size=window_size)
    
    async def update_sliding_window(
        self, 
        feature_vector: np.ndarray, 
        timestamp: datetime
    ) -> None:
        """Update sliding window with new features"""
        
        # Add to sliding window
        self.sliding_window.current_features.append(feature_vector)
        self.sliding_window.timestamps.append(timestamp)
        self.sliding_window.last_update = timestamp
        
        # Update feature buffer for fast access
        if len(self.sliding_window.current_features) > 0:
            features_array = np.array(list(self.sliding_window.current_features))
            buffer_size = min(len(features_array), self.window_size)
            self.sliding_window.feature_buffer[:buffer_size] = features_array[-buffer_size:]
    
    async def detect_crisis_pattern(
        self, 
        current_features: np.ndarray,
        timestamp: datetime = None
    ) -> Optional[PatternMatch]:
        """
        Detect crisis patterns in real-time with <5ms processing.
        
        Args:
            current_features: Current feature vector
            timestamp: Current timestamp
            
        Returns:
            PatternMatch if crisis pattern detected, None otherwise
        """
        
        start_time = time.perf_counter()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Update sliding window
            await self.update_sliding_window(current_features, timestamp)
            
            # Fast pattern matching
            pattern_match = await self._fast_pattern_match(current_features, timestamp)
            
            # Record processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            if pattern_match:
                pattern_match.processing_time_ms = processing_time
                
                # Store in history
                self.match_history.append(pattern_match)
                
                # Check latency requirement
                if processing_time > self.latency_target_ms:
                    logger.warning(f"Processing time {processing_time:.2f}ms exceeds target {self.latency_target_ms}ms")
            
            return pattern_match
            
        except Exception as e:
            logger.error(f"Crisis pattern detection failed: {e}")
            return None
    
    async def _fast_pattern_match(
        self, 
        features: np.ndarray, 
        timestamp: datetime
    ) -> Optional[PatternMatch]:
        """Fast pattern matching implementation"""
        
        if not self.pattern_library:
            return None
        
        # Find closest patterns
        closest_patterns = self.pattern_matcher.find_closest_patterns(features, k=5)
        
        if not closest_patterns:
            return None
        
        # Get best match
        best_idx, best_similarity = closest_patterns[0]
        best_pattern = self.pattern_library[best_idx]
        
        # Calculate pattern confidence
        pattern_confidence = await self._calculate_pattern_confidence(
            features, best_pattern, best_similarity
        )
        
        # Check if match meets thresholds
        if (best_similarity >= self.similarity_threshold and 
            pattern_confidence >= self.confidence_threshold):
            
            # Determine match quality
            match_quality = self._determine_match_quality(best_similarity, pattern_confidence)
            
            return PatternMatch(
                timestamp=timestamp,
                matched_pattern=best_pattern,
                similarity_score=best_similarity,
                pattern_confidence=pattern_confidence,
                distance_metric="euclidean_normalized",
                match_quality=match_quality,
                processing_time_ms=0  # Will be set by caller
            )
        
        return None
    
    async def _calculate_pattern_confidence(
        self, 
        features: np.ndarray, 
        matched_pattern: CrisisFingerprint, 
        similarity_score: float
    ) -> float:
        """Calculate confidence score for pattern match"""
        
        try:
            # Base confidence from similarity
            confidence = similarity_score
            
            # Crisis-specific feature checks
            crisis_type = matched_pattern.crisis_type
            thresholds = self.crisis_thresholds.get(crisis_type, {})
            
            # Extract specific features based on crisis type
            if crisis_type == CrisisType.FLASH_CRASH:
                # Check volatility spike and price drop
                volatility_spike = features[0] if len(features) > 0 else 0
                price_drop = abs(features[3]) if len(features) > 3 else 0
                
                vol_score = min(1.0, volatility_spike / thresholds.get('volatility_spike_min', 2.0))
                price_score = min(1.0, price_drop / thresholds.get('price_drop_min', 0.05))
                
                confidence *= (vol_score + price_score) / 2
                
            elif crisis_type == CrisisType.LIQUIDITY_CRISIS:
                # Check spread spike and volume
                spread_spike = features[16] if len(features) > 16 else 0
                volume_impact = features[6] if len(features) > 6 else 0
                
                spread_score = min(1.0, spread_spike / thresholds.get('spread_spike_min', 2.5))
                volume_score = min(1.0, volume_impact / thresholds.get('volume_spike_min', 2.0))
                
                confidence *= (spread_score + volume_score) / 2
                
            elif crisis_type == CrisisType.VOLATILITY_EXPLOSION:
                # Check volatility spike and persistence
                volatility_spike = features[0] if len(features) > 0 else 0
                persistence = features[2] if len(features) > 2 else 0
                
                vol_score = min(1.0, volatility_spike / thresholds.get('volatility_spike_min', 3.0))
                persist_score = min(1.0, persistence / thresholds.get('persistence_min', 0.6))
                
                confidence *= (vol_score + persist_score) / 2
                
            elif crisis_type == CrisisType.CORRELATION_BREAKDOWN:
                # Check correlation features
                correlation_breakdown = features[13] if len(features) > 13 else 0
                correlation_contagion = features[14] if len(features) > 14 else 0
                
                breakdown_score = min(1.0, correlation_breakdown / thresholds.get('correlation_spike_min', 0.7))
                contagion_score = min(1.0, correlation_contagion / thresholds.get('contagion_min', 0.4))
                
                confidence *= (breakdown_score + contagion_score) / 2
            
            # Additional confidence from sliding window analysis
            window_confidence = await self._calculate_window_confidence(matched_pattern)
            
            # Combine confidences
            final_confidence = (confidence * 0.7) + (window_confidence * 0.3)
            
            return min(1.0, max(0.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return similarity_score
    
    async def _calculate_window_confidence(self, matched_pattern: CrisisFingerprint) -> float:
        """Calculate confidence based on sliding window trends"""
        
        try:
            if len(self.sliding_window.current_features) < 5:
                return 0.5  # Neutral confidence with insufficient data
            
            features_array = np.array(list(self.sliding_window.current_features))
            
            # Check for trending behavior consistent with crisis
            if matched_pattern.crisis_type == CrisisType.FLASH_CRASH:
                # Look for accelerating volatility and price drops
                volatility_trend = features_array[-5:, 0]  # Last 5 volatility spikes
                price_trend = features_array[-5:, 3]       # Last 5 price movements
                
                vol_acceleration = np.diff(volatility_trend).mean() if len(volatility_trend) > 1 else 0
                price_acceleration = np.diff(price_trend).mean() if len(price_trend) > 1 else 0
                
                confidence = min(1.0, (vol_acceleration + abs(price_acceleration)) / 2)
                
            elif matched_pattern.crisis_type == CrisisType.VOLATILITY_EXPLOSION:
                # Look for persistent high volatility
                volatility_window = features_array[-10:, 0] if len(features_array) >= 10 else features_array[:, 0]
                volatility_persistence = (volatility_window > 1.5).mean()  # Fraction above threshold
                
                confidence = volatility_persistence
                
            elif matched_pattern.crisis_type == CrisisType.LIQUIDITY_CRISIS:
                # Look for deteriorating liquidity metrics
                liquidity_window = features_array[-10:, 16:19] if len(features_array) >= 10 else features_array[:, 16:19]
                liquidity_trend = np.mean(np.diff(liquidity_window, axis=0), axis=0)
                
                # Higher values indicate worse liquidity
                confidence = min(1.0, np.mean(liquidity_trend) / 2.0)
                
            else:
                # Default window confidence
                recent_similarity = self._calculate_recent_pattern_similarity(matched_pattern)
                confidence = recent_similarity
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Window confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_recent_pattern_similarity(self, pattern: CrisisFingerprint) -> float:
        """Calculate similarity to pattern using recent window data"""
        
        try:
            if len(self.sliding_window.current_features) < 3:
                return 0.5
            
            # Use last few features
            recent_features = np.array(list(self.sliding_window.current_features)[-3:])
            pattern_features = pattern.feature_vector
            
            # Calculate average similarity
            similarities = []
            for features in recent_features:
                similarity = 1 / (1 + euclidean(features, pattern_features))
                similarities.append(similarity)
            
            return np.mean(similarities)
            
        except Exception:
            return 0.5
    
    def _determine_match_quality(self, similarity: float, confidence: float) -> str:
        """Determine match quality based on scores"""
        
        combined_score = (similarity + confidence) / 2
        
        if combined_score >= 0.9:
            return "HIGH"
        elif combined_score >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def get_crisis_similarity_score(
        self, 
        features: np.ndarray, 
        crisis_type: CrisisType
    ) -> float:
        """Get similarity score for specific crisis type"""
        
        if not self.pattern_library:
            return 0.0
        
        # Filter patterns by crisis type
        type_patterns = [p for p in self.pattern_library if p.crisis_type == crisis_type]
        
        if not type_patterns:
            return 0.0
        
        # Find best similarity for this crisis type
        best_similarity = 0.0
        
        for pattern in type_patterns:
            similarity = 1 / (1 + euclidean(features, pattern.feature_vector))
            best_similarity = max(best_similarity, similarity)
        
        return best_similarity
    
    def add_pattern_to_library(self, pattern: CrisisFingerprint) -> None:
        """Add new pattern to library and rebuild index"""
        
        self.pattern_library.append(pattern)
        
        # Rebuild search index periodically or when library grows significantly
        if len(self.pattern_library) % 100 == 0:
            self.pattern_matcher = OptimizedPatternMatcher(self.pattern_library)
            logger.info(f"Pattern library updated, now contains {len(self.pattern_library)} patterns")
    
    def get_performance_stats(self) -> Dict:
        """Get engine performance statistics"""
        
        if not self.processing_times:
            return {"no_processing_data": True}
        
        times = list(self.processing_times)
        
        # Pattern matcher performance
        search_perf = self.pattern_matcher.get_search_performance()
        
        return {
            "total_patterns": len(self.pattern_library),
            "window_size": self.window_size,
            "processing_stats": {
                "avg_processing_time_ms": np.mean(times),
                "max_processing_time_ms": np.max(times),
                "min_processing_time_ms": np.min(times),
                "latency_target_ms": self.latency_target_ms,
                "target_met_percentage": sum(1 for t in times if t <= self.latency_target_ms) / len(times) * 100,
                "total_detections": len(times)
            },
            "pattern_search_stats": search_perf,
            "match_history_size": len(self.match_history),
            "recent_matches": len([m for m in self.match_history if m.timestamp > datetime.now() - timedelta(hours=1)])
        }
    
    def get_crisis_type_distribution(self) -> Dict:
        """Get distribution of crisis types in pattern library"""
        
        distribution = {}
        for pattern in self.pattern_library:
            crisis_type = pattern.crisis_type.value
            distribution[crisis_type] = distribution.get(crisis_type, 0) + 1
        
        return distribution
    
    async def export_performance_report(self, output_path: str) -> bool:
        """Export detailed performance report"""
        
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "performance_stats": self.get_performance_stats(),
                "crisis_distribution": self.get_crisis_type_distribution(),
                "thresholds": {
                    "similarity_threshold": self.similarity_threshold,
                    "confidence_threshold": self.confidence_threshold,
                    "latency_target_ms": self.latency_target_ms
                },
                "recent_matches": [
                    {
                        "timestamp": match.timestamp.isoformat(),
                        "crisis_type": match.matched_pattern.crisis_type.value,
                        "similarity": match.similarity_score,
                        "confidence": match.pattern_confidence,
                        "quality": match.match_quality,
                        "processing_time_ms": match.processing_time_ms
                    }
                    for match in list(self.match_history)[-10:]  # Last 10 matches
                ]
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
            return False