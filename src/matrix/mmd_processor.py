"""
MMD (Market Microstructure Detector) Processor with PCA/t-SNE Dimensionality Reduction

Processes high-dimensional MMD features and reduces them to 3 principal components
for integration into the 30m matrix assembler.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import deque
import logging
import threading
import warnings

from src.utils.logger import get_logger

warnings.filterwarnings('ignore')


class MMDProcessor:
    """
    Market Microstructure Detector with dimensionality reduction.
    
    Processes complex market microstructure data and reduces to 3 components:
    - PC1: Primary microstructure trend
    - PC2: Secondary pattern component  
    - PC3: Volatility/noise component
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 method: str = 'pca',
                 update_frequency: int = 10):
        """
        Initialize MMD processor.
        
        Args:
            window_size: Number of samples for rolling PCA/t-SNE
            method: 'pca' or 'tsne' for dimensionality reduction
            update_frequency: Update reduction model every N samples
        """
        self.logger = get_logger(self.__class__.__name__)
        self.window_size = window_size
        self.method = method.lower()
        self.update_frequency = update_frequency
        
        # Data storage
        self.feature_buffer = deque(maxlen=window_size)
        self.reduced_buffer = deque(maxlen=window_size)
        
        # Dimensionality reduction models
        self.scaler = StandardScaler()
        self.reducer = None
        self._init_reducer()
        
        # State tracking
        self.n_updates = 0
        self.last_model_update = 0
        self.is_fitted = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        
        self.logger.info(
            f"MMD Processor initialized with method={method}, "
            f"window_size={window_size}"
        )
    
    def _init_reducer(self) -> None:
        """Initialize the dimensionality reduction model."""
        if self.method == 'pca':
            self.reducer = PCA(n_components=3, random_state=42)
        elif self.method == 'tsne':
            self.reducer = TSNE(
                n_components=3,
                random_state=42,
                n_iter=300,
                perplexity=min(30, max(5, self.window_size // 4))
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def calculate_mmd_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate comprehensive MMD features from market data.
        
        Args:
            market_data: Dictionary containing market data (price, volume, etc.)
            
        Returns:
            High-dimensional MMD feature vector
        """
        features = []
        
        try:
            # Price-based features
            price = market_data.get('close', 0.0)
            open_price = market_data.get('open', price)
            high = market_data.get('high', price)
            low = market_data.get('low', price)
            volume = market_data.get('volume', 0.0)
            
            # Basic price features
            if price > 0:
                price_range = (high - low) / price if price > 0 else 0.0
                body_ratio = abs(price - open_price) / price if price > 0 else 0.0
                upper_shadow = (high - max(price, open_price)) / price if price > 0 else 0.0
                lower_shadow = (min(price, open_price) - low) / price if price > 0 else 0.0
            else:
                price_range = body_ratio = upper_shadow = lower_shadow = 0.0
            
            features.extend([price_range, body_ratio, upper_shadow, lower_shadow])
            
            # Volume features
            features.extend([
                np.log1p(volume) if volume > 0 else 0.0,
                volume / (price * (high - low)) if price > 0 and high > low else 0.0  # Volume/Price*Range
            ])
            
            # Advanced microstructure features
            spread = market_data.get('spread', 0.0)
            bid_volume = market_data.get('bid_volume', volume / 2)
            ask_volume = market_data.get('ask_volume', volume / 2)
            
            # Order flow imbalance
            total_volume = bid_volume + ask_volume
            if total_volume > 0:
                order_flow_imbalance = (bid_volume - ask_volume) / total_volume
            else:
                order_flow_imbalance = 0.0
            
            features.extend([
                spread / price if price > 0 else 0.0,
                order_flow_imbalance,
                np.log1p(bid_volume) if bid_volume > 0 else 0.0,
                np.log1p(ask_volume) if ask_volume > 0 else 0.0
            ])
            
            # Volatility proxies
            volatility_proxy = price_range * np.sqrt(volume) if volume > 0 else 0.0
            features.append(volatility_proxy)
            
            # Liquidity measures
            if volume > 0 and price_range > 0:
                amihud_illiquidity = abs(price - open_price) / (volume * price)
                kyle_lambda = price_range / np.sqrt(volume)
            else:
                amihud_illiquidity = kyle_lambda = 0.0
            
            features.extend([amihud_illiquidity, kyle_lambda])
            
            # Market impact measures
            features.extend([
                market_data.get('tick_direction', 0.0),  # +1 uptick, -1 downtick
                market_data.get('trade_size_avg', 0.0),
                market_data.get('trade_frequency', 0.0)
            ])
            
            # Technical indicators as MMD features
            features.extend([
                market_data.get('rsi', 50.0) / 100.0,  # Normalize RSI
                market_data.get('bb_position', 0.5),   # Bollinger band position
                market_data.get('macd_signal', 0.0),
                market_data.get('volume_sma_ratio', 1.0)
            ])
            
            # Ensure we have at least 20 features for meaningful PCA
            while len(features) < 20:
                features.append(0.0)
            
            # Convert to numpy array and handle inf/nan
            feature_array = np.array(features, dtype=np.float32)
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return feature_array
            
        except Exception as e:
            self.logger.error(f"Error calculating MMD features: {e}")
            # Return zero vector of standard size
            return np.zeros(20, dtype=np.float32)
    
    def process_30m_mmd(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Process MMD for 30-minute timeframe and return 3 principal components.
        
        Args:
            market_data: 30-minute bar market data
            
        Returns:
            3-component reduced MMD vector [PC1, PC2, PC3]
        """
        import time
        start_time = time.time()
        
        try:
            with self._lock:
                # Calculate high-dimensional MMD features
                mmd_features = self.calculate_mmd_features(market_data)
                
                # Store in buffer
                self.feature_buffer.append(mmd_features)
                self.n_updates += 1
                
                # Check if we need to update the reduction model
                if (self.n_updates - self.last_model_update) >= self.update_frequency:
                    self._update_reduction_model()
                
                # Apply dimensionality reduction
                reduced_features = self._apply_reduction(mmd_features)
                self.reduced_buffer.append(reduced_features)
                
                # Track performance
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                
                if self.n_updates % 100 == 0:
                    avg_time = np.mean(list(self.processing_times))
                    self.logger.debug(f"MMD processing time: {avg_time:.2f}ms avg")
                
                return reduced_features
                
        except Exception as e:
            self.logger.error(f"Error in MMD processing: {e}")
            return np.zeros(3, dtype=np.float32)
    
    def _update_reduction_model(self) -> None:
        """Update the PCA/t-SNE model with recent data."""
        try:
            if len(self.feature_buffer) < 10:
                return
            
            # Prepare data matrix
            X = np.array(list(self.feature_buffer))
            
            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit reduction model
            if self.method == 'pca':
                self.reducer.fit(X_scaled)
                self.is_fitted = True
                
                # Log explained variance
                explained_var = self.reducer.explained_variance_ratio_
                self.logger.debug(
                    f"PCA explained variance: "
                    f"PC1={explained_var[0]:.3f}, "
                    f"PC2={explained_var[1]:.3f}, "
                    f"PC3={explained_var[2]:.3f}"
                )
                
            elif self.method == 'tsne':
                # t-SNE doesn't have a fit method, we'll recompute each time
                self.is_fitted = True
                
            self.last_model_update = self.n_updates
            self.logger.debug(f"Updated {self.method.upper()} model with {len(X)} samples")
            
        except Exception as e:
            self.logger.error(f"Error updating reduction model: {e}")
    
    def _apply_reduction(self, features: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to features."""
        try:
            if not self.is_fitted or len(self.feature_buffer) < 3:
                # Return zero vector if not ready
                return np.zeros(3, dtype=np.float32)
            
            # Standardize single sample
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            if self.method == 'pca':
                # Transform using fitted PCA
                reduced = self.reducer.transform(features_scaled)
                return reduced[0].astype(np.float32)
                
            elif self.method == 'tsne':
                # For t-SNE, we need to recompute on the entire buffer
                # This is computationally expensive, so we'll use a simplified approach
                # for real-time processing
                if len(self.feature_buffer) >= 10:
                    X = np.array(list(self.feature_buffer))
                    X_scaled = self.scaler.fit_transform(X)
                    
                    # Use last fitted t-SNE or compute new one
                    embedded = self.reducer.fit_transform(X_scaled)
                    return embedded[-1].astype(np.float32)  # Return last point
                else:
                    return np.zeros(3, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error applying reduction: {e}")
            return np.zeros(3, dtype=np.float32)
    
    def get_component_interpretation(self) -> Dict[str, str]:
        """Get interpretation of the 3 principal components."""
        if not self.is_fitted or self.method != 'pca':
            return {
                "PC1": "Primary market microstructure trend",
                "PC2": "Secondary pattern component",
                "PC3": "Volatility and noise component"
            }
        
        try:
            # Analyze PCA components
            components = self.reducer.components_
            
            # Feature names for interpretation
            feature_names = [
                "price_range", "body_ratio", "upper_shadow", "lower_shadow",
                "log_volume", "volume_price_ratio", "spread_ratio", "order_flow_imbalance",
                "bid_volume", "ask_volume", "volatility_proxy", "amihud_illiquidity",
                "kyle_lambda", "tick_direction", "trade_size", "trade_frequency",
                "rsi_norm", "bb_position", "macd_signal", "volume_sma_ratio"
            ]
            
            interpretation = {}
            for i in range(3):
                # Find top contributors to this component
                component = components[i]
                top_indices = np.argsort(np.abs(component))[-3:][::-1]
                top_features = [feature_names[idx] for idx in top_indices if idx < len(feature_names)]
                
                if i == 0:
                    interpretation[f"PC{i+1}"] = f"Primary trend: {', '.join(top_features)}"
                elif i == 1:
                    interpretation[f"PC{i+1}"] = f"Secondary pattern: {', '.join(top_features)}"
                else:
                    interpretation[f"PC{i+1}"] = f"Volatility component: {', '.join(top_features)}"
            
            return interpretation
            
        except Exception as e:
            self.logger.error(f"Error interpreting components: {e}")
            return {
                "PC1": "Primary market microstructure trend",
                "PC2": "Secondary pattern component", 
                "PC3": "Volatility and noise component"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        with self._lock:
            stats = {
                "method": self.method,
                "window_size": self.window_size,
                "n_updates": self.n_updates,
                "is_fitted": self.is_fitted,
                "buffer_size": len(self.feature_buffer),
                "last_model_update": self.last_model_update
            }
            
            if self.processing_times:
                times = list(self.processing_times)
                stats["performance"] = {
                    "avg_processing_time_ms": np.mean(times),
                    "max_processing_time_ms": np.max(times),
                    "p95_processing_time_ms": np.percentile(times, 95)
                }
            
            if self.is_fitted and self.method == 'pca':
                stats["explained_variance"] = self.reducer.explained_variance_ratio_.tolist()
            
            return stats
    
    def reset(self) -> None:
        """Reset the processor state."""
        with self._lock:
            self.feature_buffer.clear()
            self.reduced_buffer.clear()
            self.n_updates = 0
            self.last_model_update = 0
            self.is_fitted = False
            self.processing_times.clear()
            
            # Reinitialize models
            self.scaler = StandardScaler()
            self._init_reducer()
            
            self.logger.info("MMD Processor reset")