"""
Data preprocessing pipeline for large datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
import logging
import time
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import pickle
import joblib
from pathlib import Path
import threading
import queue

from ..core.config import DataPipelineConfig
from ..core.exceptions import DataPreprocessingException
from ..core.data_loader import DataChunk
from ..streaming.data_streamer import DataStreamer

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    memory_threshold_mb: float = 1000.0
    enable_caching: bool = True
    cache_directory: str = "/tmp/preprocessing_cache"
    
    # Data cleaning options
    remove_duplicates: bool = True
    handle_missing_values: str = "interpolate"  # "drop", "fill", "interpolate"
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation"
    
    # Data transformation options
    normalize_features: bool = True
    scale_features: bool = True
    encode_categorical: bool = True
    
    # Feature engineering options
    create_technical_indicators: bool = True
    create_time_features: bool = True
    create_lag_features: bool = True
    lag_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Performance options
    enable_vectorization: bool = True
    use_numba: bool = True
    enable_gpu_acceleration: bool = False

class DataProcessor:
    """
    High-performance data processor for large datasets
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None,
                 pipeline_config: Optional[DataPipelineConfig] = None):
        self.config = config or PreprocessingConfig()
        self.pipeline_config = pipeline_config or DataPipelineConfig()
        
        # Initialize components
        self.data_streamer = DataStreamer(self.pipeline_config)
        self.transformers = {}
        self.feature_engineers = {}
        self.stats = ProcessingStats()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup caching
        self._setup_caching()
    
    def _setup_caching(self):
        """Setup caching directory"""
        if self.config.enable_caching:
            Path(self.config.cache_directory).mkdir(parents=True, exist_ok=True)
    
    def process_file(self, file_path: str, 
                    preprocessing_steps: Optional[List[str]] = None,
                    **kwargs) -> Iterator[DataChunk]:
        """
        Process a single file through the preprocessing pipeline
        """
        preprocessing_steps = preprocessing_steps or [
            "clean_data",
            "handle_missing_values", 
            "detect_outliers",
            "normalize_features",
            "engineer_features"
        ]
        
        try:
            # Stream and process data
            for chunk in self.data_streamer.stream_file(file_path, **kwargs):
                processed_chunk = self._process_chunk(chunk, preprocessing_steps)
                if processed_chunk is not None:
                    yield processed_chunk
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise DataPreprocessingException(f"Failed to process file {file_path}: {str(e)}")
    
    def process_multiple_files(self, file_paths: List[str],
                             preprocessing_steps: Optional[List[str]] = None,
                             parallel: bool = True,
                             **kwargs) -> Iterator[DataChunk]:
        """
        Process multiple files through the preprocessing pipeline
        """
        if parallel and self.config.enable_parallel_processing:
            yield from self._process_multiple_files_parallel(
                file_paths, preprocessing_steps, **kwargs
            )
        else:
            yield from self._process_multiple_files_sequential(
                file_paths, preprocessing_steps, **kwargs
            )
    
    def _process_multiple_files_sequential(self, file_paths: List[str],
                                         preprocessing_steps: Optional[List[str]] = None,
                                         **kwargs) -> Iterator[DataChunk]:
        """Process files sequentially"""
        for file_path in file_paths:
            yield from self.process_file(file_path, preprocessing_steps, **kwargs)
    
    def _process_multiple_files_parallel(self, file_paths: List[str],
                                       preprocessing_steps: Optional[List[str]] = None,
                                       **kwargs) -> Iterator[DataChunk]:
        """Process files in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit processing tasks
            future_to_file = {
                executor.submit(self._process_file_to_list, file_path, preprocessing_steps, **kwargs): file_path
                for file_path in file_paths
            }
            
            # Collect results
            for future in future_to_file:
                file_path = future_to_file[future]
                try:
                    chunks = future.result()
                    for chunk in chunks:
                        yield chunk
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    raise DataPreprocessingException(f"Failed to process file {file_path}: {str(e)}")
    
    def _process_file_to_list(self, file_path: str, 
                            preprocessing_steps: Optional[List[str]] = None,
                            **kwargs) -> List[DataChunk]:
        """Process file and return list of chunks"""
        return list(self.process_file(file_path, preprocessing_steps, **kwargs))
    
    def _process_chunk(self, chunk: DataChunk, 
                      preprocessing_steps: List[str]) -> Optional[DataChunk]:
        """Process a single data chunk"""
        try:
            start_time = time.time()
            data = chunk.data.copy()
            
            # Apply preprocessing steps
            for step in preprocessing_steps:
                data = self._apply_preprocessing_step(data, step)
                if data is None or data.empty:
                    return None
            
            # Create processed chunk
            processed_chunk = DataChunk(
                data=data,
                chunk_id=chunk.chunk_id,
                start_row=chunk.start_row,
                end_row=chunk.end_row,
                file_path=chunk.file_path,
                timestamp=time.time(),
                memory_usage=0  # Will be calculated
            )
            
            # Update stats
            processing_time = time.time() - start_time
            self.stats.update_processing_time(processing_time)
            self.stats.chunks_processed += 1
            
            return processed_chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
            return None
    
    def _apply_preprocessing_step(self, data: pd.DataFrame, step: str) -> pd.DataFrame:
        """Apply a single preprocessing step"""
        if step == "clean_data":
            return self._clean_data(data)
        elif step == "handle_missing_values":
            return self._handle_missing_values(data)
        elif step == "detect_outliers":
            return self._detect_outliers(data)
        elif step == "normalize_features":
            return self._normalize_features(data)
        elif step == "scale_features":
            return self._scale_features(data)
        elif step == "engineer_features":
            return self._engineer_features(data)
        elif step == "encode_categorical":
            return self._encode_categorical(data)
        else:
            logger.warning(f"Unknown preprocessing step: {step}")
            return data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing duplicates and invalid values"""
        # Remove duplicates
        if self.config.remove_duplicates:
            data = data.drop_duplicates()
        
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Remove rows with invalid numeric values
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data = data[~data[col].isin([np.inf, -np.inf])]
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        if self.config.handle_missing_values == "drop":
            return data.dropna()
        elif self.config.handle_missing_values == "fill":
            return self._fill_missing_values(data)
        elif self.config.handle_missing_values == "interpolate":
            return self._interpolate_missing_values(data)
        else:
            return data
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate strategies"""
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # Fill numeric columns with median
                data[column].fillna(data[column].median(), inplace=True)
            elif data[column].dtype == 'object':
                # Fill categorical columns with mode
                mode_value = data[column].mode()
                if not mode_value.empty:
                    data[column].fillna(mode_value[0], inplace=True)
        
        return data
    
    def _interpolate_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Linear interpolation for numeric columns
            data[col] = data[col].interpolate(method='linear')
        
        # Forward fill for remaining missing values
        data = data.fillna(method='ffill')
        
        return data
    
    def _detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers"""
        if not self.config.outlier_detection:
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if self.config.outlier_method == "iqr":
                data = self._remove_outliers_iqr(data, col)
            elif self.config.outlier_method == "zscore":
                data = self._remove_outliers_zscore(data, col)
            elif self.config.outlier_method == "isolation":
                data = self._remove_outliers_isolation(data, col)
        
        return data
    
    def _remove_outliers_iqr(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    def _remove_outliers_zscore(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using Z-score method"""
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        return data[z_scores < 3]
    
    def _remove_outliers_isolation(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest
            
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = isolation_forest.fit_predict(data[[column]])
            
            return data[outliers != -1]
        except ImportError:
            logger.warning("scikit-learn not available, skipping isolation forest outlier detection")
            return data
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to [0, 1] range"""
        if not self.config.normalize_features:
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            min_val = data[col].min()
            max_val = data[col].max()
            
            if max_val != min_val:
                data[col] = (data[col] - min_val) / (max_val - min_val)
        
        return data
    
    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features to have zero mean and unit variance"""
        if not self.config.scale_features:
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            if std_val != 0:
                data[col] = (data[col] - mean_val) / std_val
        
        return data
    
    def _encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        if not self.config.encode_categorical:
            return data
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            # Simple label encoding
            unique_values = data[col].unique()
            value_to_code = {value: i for i, value in enumerate(unique_values)}
            data[col] = data[col].map(value_to_code)
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features"""
        if 'timestamp' in data.columns:
            data = self._create_time_features(data)
        
        if self.config.create_technical_indicators:
            data = self._create_technical_indicators(data)
        
        if self.config.create_lag_features:
            data = self._create_lag_features(data)
        
        return data
    
    def _create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        if not self.config.create_time_features:
            return data
        
        if 'timestamp' in data.columns:
            # Convert to datetime if needed
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Extract time features
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['month'] = data['timestamp'].dt.month
            data['quarter'] = data['timestamp'].dt.quarter
            data['year'] = data['timestamp'].dt.year
            
            # Create cyclical features
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        return data
    
    def _create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for financial data"""
        if not self.config.create_technical_indicators:
            return data
        
        # Assume OHLCV data
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in price_columns if col in data.columns]
        
        if 'close' in available_columns:
            # Moving averages
            data['sma_5'] = data['close'].rolling(window=5).mean()
            data['sma_10'] = data['close'].rolling(window=10).mean()
            data['sma_20'] = data['close'].rolling(window=20).mean()
            
            # Exponential moving averages
            data['ema_5'] = data['close'].ewm(span=5).mean()
            data['ema_10'] = data['close'].ewm(span=10).mean()
            data['ema_20'] = data['close'].ewm(span=20).mean()
            
            # Returns
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # Volatility
            data['volatility'] = data['returns'].rolling(window=20).std()
            
            # RSI
            data['rsi'] = self._calculate_rsi(data['close'])
        
        if 'high' in available_columns and 'low' in available_columns:
            # True range
            data['tr'] = self._calculate_true_range(data)
            data['atr'] = data['tr'].rolling(window=14).mean()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range
    
    def _create_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        if not self.config.create_lag_features:
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            for lag in self.config.lag_periods:
                data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return data
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'chunks_processed': self.stats.chunks_processed,
            'avg_processing_time': self.stats.get_avg_processing_time(),
            'total_processing_time': self.stats.get_total_processing_time(),
            'throughput': self.stats.get_throughput()
        }
    
    def save_transformer(self, name: str, transformer: Any, file_path: str):
        """Save a transformer to disk"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(transformer, f)
            
            self.transformers[name] = file_path
            logger.info(f"Transformer '{name}' saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving transformer: {str(e)}")
    
    def load_transformer(self, name: str, file_path: str) -> Any:
        """Load a transformer from disk"""
        try:
            with open(file_path, 'rb') as f:
                transformer = pickle.load(f)
            
            self.transformers[name] = transformer
            logger.info(f"Transformer '{name}' loaded from {file_path}")
            return transformer
        except Exception as e:
            logger.error(f"Error loading transformer: {str(e)}")
            return None


class ProcessingStats:
    """Statistics for data processing operations"""
    
    def __init__(self):
        self.chunks_processed = 0
        self.processing_times = []
        self.start_time = time.time()
    
    def update_processing_time(self, processing_time: float):
        """Update processing time statistics"""
        self.processing_times.append(processing_time)
        
        # Keep only recent times (last 100)
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-100:]
    
    def get_avg_processing_time(self) -> float:
        """Get average processing time"""
        return np.mean(self.processing_times) if self.processing_times else 0.0
    
    def get_total_processing_time(self) -> float:
        """Get total processing time"""
        return sum(self.processing_times)
    
    def get_throughput(self) -> float:
        """Get processing throughput"""
        elapsed = time.time() - self.start_time
        return self.chunks_processed / elapsed if elapsed > 0 else 0.0
    
    def reset(self):
        """Reset statistics"""
        self.chunks_processed = 0
        self.processing_times = []
        self.start_time = time.time()


# Utility functions for custom preprocessing
def create_custom_transformer(transform_func: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable:
    """Create a custom transformer function"""
    def transformer(data: pd.DataFrame) -> pd.DataFrame:
        try:
            return transform_func(data)
        except Exception as e:
            logger.error(f"Error in custom transformer: {str(e)}")
            return data
    
    return transformer

def create_feature_selector(features: List[str]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Create a feature selector"""
    def selector(data: pd.DataFrame) -> pd.DataFrame:
        available_features = [f for f in features if f in data.columns]
        return data[available_features] if available_features else data
    
    return selector

def create_data_validator(validation_func: Callable[[pd.DataFrame], bool]) -> Callable:
    """Create a data validator"""
    def validator(data: pd.DataFrame) -> pd.DataFrame:
        try:
            if validation_func(data):
                return data
            else:
                logger.warning("Data validation failed")
                return data
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return data
    
    return validator