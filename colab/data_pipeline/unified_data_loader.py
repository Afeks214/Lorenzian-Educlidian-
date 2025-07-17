"""
Unified Data Loader for NQ Dataset Processing

Provides a common data loading interface for both execution engine and risk management notebooks
with support for chunked loading, validation, and preprocessing.
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
from pathlib import Path
import time
import hashlib
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import mmap
import pickle
import json
from functools import lru_cache

# Performance monitoring
class LoadingMetrics:
    """Track data loading performance metrics"""
    
    def __init__(self):
        self.load_times = []
        self.chunk_sizes = []
        self.memory_usage = []
        self.validation_times = []
        self.preprocessing_times = []
        
    def record_load_time(self, duration: float, chunk_size: int, memory_mb: float):
        """Record loading performance metrics"""
        self.load_times.append(duration)
        self.chunk_sizes.append(chunk_size)
        self.memory_usage.append(memory_mb)
    
    def record_validation_time(self, duration: float):
        """Record validation performance"""
        self.validation_times.append(duration)
    
    def record_preprocessing_time(self, duration: float):
        """Record preprocessing performance"""
        self.preprocessing_times.append(duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        return {
            'avg_load_time': np.mean(self.load_times) if self.load_times else 0,
            'max_load_time': max(self.load_times) if self.load_times else 0,
            'avg_chunk_size': np.mean(self.chunk_sizes) if self.chunk_sizes else 0,
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'avg_validation_time': np.mean(self.validation_times) if self.validation_times else 0,
            'avg_preprocessing_time': np.mean(self.preprocessing_times) if self.preprocessing_times else 0,
            'total_loads': len(self.load_times)
        }

@dataclass
class DataValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    
class DataValidator:
    """Validate NQ dataset integrity and quality"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_data(self, df: pd.DataFrame) -> DataValidationResult:
        """Comprehensive data validation"""
        errors = []
        warnings = []
        
        # Required columns check
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Data quality checks
        if len(df) == 0:
            errors.append("Empty dataset")
        
        # Null value checks
        null_counts = df.isnull().sum()
        if null_counts.any():
            warnings.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Price data validation
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                if (df[col] <= 0).any():
                    errors.append(f"Non-positive values in {col}")
                
                # Check for extreme values
                q99 = df[col].quantile(0.99)
                q1 = df[col].quantile(0.01)
                extreme_ratio = q99 / q1
                if extreme_ratio > 10:  # Price range too wide
                    warnings.append(f"Extreme price range in {col}: {extreme_ratio:.2f}")
        
        # Volume validation
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                errors.append("Negative volume values")
        
        # Timestamp validation
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    if df['timestamp'].isnull().any():
                        errors.append("Invalid timestamp format")
                except:
                    errors.append("Invalid timestamp format")
        
        # Statistical summary
        statistics = {}
        if len(df) > 0:
            statistics = {
                'rows': len(df),
                'columns': len(df.columns),
                'date_range': {
                    'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                    'end': df['timestamp'].max() if 'timestamp' in df.columns else None
                },
                'price_stats': {
                    'mean_close': df['close'].mean() if 'close' in df.columns else None,
                    'std_close': df['close'].std() if 'close' in df.columns else None,
                    'min_close': df['close'].min() if 'close' in df.columns else None,
                    'max_close': df['close'].max() if 'close' in df.columns else None
                },
                'volume_stats': {
                    'mean_volume': df['volume'].mean() if 'volume' in df.columns else None,
                    'total_volume': df['volume'].sum() if 'volume' in df.columns else None
                }
            }
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=statistics
        )

class DataPreprocessor:
    """Preprocess NQ data for ML training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing pipeline"""
        start_time = time.time()
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # 1. Sort by timestamp
        if 'timestamp' in processed_df.columns:
            processed_df = processed_df.sort_values('timestamp')
        
        # 2. Calculate returns
        if 'close' in processed_df.columns:
            processed_df['returns'] = processed_df['close'].pct_change()
            processed_df['log_returns'] = np.log(processed_df['close'] / processed_df['close'].shift(1))
        
        # 3. Calculate technical indicators
        processed_df = self._add_technical_indicators(processed_df)
        
        # 4. Calculate volatility features
        processed_df = self._add_volatility_features(processed_df)
        
        # 5. Add time-based features
        processed_df = self._add_time_features(processed_df)
        
        # 6. Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # 7. Normalize features if requested
        if self.config.get('normalize_features', True):
            processed_df = self._normalize_features(processed_df)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Data preprocessing completed in {processing_time:.2f}s")
        
        return processed_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        if 'close' in df.columns:
            # Moving averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_30'] = df['close'].rolling(window=30).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            df['ema_30'] = df['close'].ewm(span=30).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        if 'returns' in df.columns:
            # Rolling volatility
            df['volatility_10'] = df['returns'].rolling(window=10).std()
            df['volatility_30'] = df['returns'].rolling(window=30).std()
            
            # GARCH-like features
            df['vol_ratio'] = df['volatility_10'] / df['volatility_30']
            
            # Realized volatility
            df['realized_vol'] = df['returns'].rolling(window=24).std() * np.sqrt(252)
        
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # True Range and Average True Range
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(window=14).mean()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            
            # Market session indicators
            df['market_open'] = (df['hour'] >= 9) & (df['hour'] < 16)
            df['pre_market'] = (df['hour'] >= 4) & (df['hour'] < 9)
            df['after_market'] = (df['hour'] >= 16) & (df['hour'] < 20)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Forward fill for price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Fill remaining NaN values with appropriate defaults
        df = df.fillna(0)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for ML training"""
        # Features to normalize (exclude price levels, keep price-based ratios)
        normalize_cols = [col for col in df.columns if col not in 
                         ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for col in normalize_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                # Z-score normalization
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
        
        return df

class UnifiedDataLoader:
    """Unified data loader for both execution engine and risk management notebooks"""
    
    def __init__(self, 
                 data_dir: str = "/home/QuantNova/GrandModel/colab/data/",
                 chunk_size: int = 10000,
                 cache_enabled: bool = True,
                 validation_enabled: bool = True,
                 preprocessing_enabled: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize unified data loader
        
        Args:
            data_dir: Directory containing NQ data files
            chunk_size: Size of data chunks for processing
            cache_enabled: Enable caching for repeated loads
            validation_enabled: Enable data validation
            preprocessing_enabled: Enable data preprocessing
            config: Configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.cache_enabled = cache_enabled
        self.validation_enabled = validation_enabled
        self.preprocessing_enabled = preprocessing_enabled
        
        # Configuration
        self.config = config or {}
        
        # Initialize components
        self.validator = DataValidator(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.metrics = LoadingMetrics()
        
        # Cache setup
        self.cache_dir = self.data_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Available data files
        self.available_files = self._discover_data_files()
        
        self.logger.info(f"UnifiedDataLoader initialized with {len(self.available_files)} files")
        
        # Data completeness monitoring
        self.completeness_monitor = self._initialize_completeness_monitor()
    
    def _discover_data_files(self) -> Dict[str, Path]:
        """Discover available NQ data files"""
        files = {}
        
        # Look for NQ data files with various naming patterns
        for file_path in self.data_dir.glob("*NQ*.csv"):
            filename = file_path.name
            
            # Parse timeframe from filename
            if "30 min" in filename or "30min" in filename:
                files['30min'] = file_path
            elif "5 min" in filename or "5min" in filename:
                if "extended" in filename.lower():
                    files['5min_extended'] = file_path
                else:
                    files['5min'] = file_path
        
        return files
    
    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes"""
        return list(self.available_files.keys())
    
    def _get_cache_path(self, timeframe: str, chunk_idx: int = None) -> Path:
        """Get cache file path for dataset"""
        base_name = f"{timeframe}"
        if chunk_idx is not None:
            base_name += f"_chunk_{chunk_idx}"
        
        return self.cache_dir / f"{base_name}.pkl"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of data file for cache invalidation"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _load_from_cache(self, cache_path: Path, file_hash: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        if not self.cache_enabled or not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Check if cache is valid
            if cached_data.get('file_hash') == file_hash:
                self.logger.info(f"Loading from cache: {cache_path}")
                return cached_data['data']
                
        except Exception as e:
            self.logger.warning(f"Cache load failed: {e}")
        
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path, file_hash: str):
        """Save data to cache"""
        if not self.cache_enabled:
            return
        
        try:
            cached_data = {
                'data': data,
                'file_hash': file_hash,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"Saved to cache: {cache_path}")
            
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
    
    def load_data(self, 
                  timeframe: str = "30min",
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load NQ data for specified timeframe
        
        Args:
            timeframe: Data timeframe ('30min', '5min', '5min_extended')
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            columns: Specific columns to load
        
        Returns:
            Loaded and processed DataFrame
        """
        start_time = time.time()
        
        if timeframe not in self.available_files:
            raise ValueError(f"Timeframe '{timeframe}' not available. Available: {list(self.available_files.keys())}")
        
        file_path = self.available_files[timeframe]
        file_hash = self._calculate_file_hash(file_path)
        cache_path = self._get_cache_path(timeframe)
        
        # Try to load from cache
        df = self._load_from_cache(cache_path, file_hash)
        
        if df is None:
            # Load from file
            self.logger.info(f"Loading {timeframe} data from {file_path}")
            
            # First read a sample to detect the actual column names
            sample_df = pd.read_csv(file_path, nrows=5)
            
            # Map actual column names to standard names (case-insensitive)
            column_mapping = {}
            standard_cols = {'timestamp': ['timestamp', 'time', 'datetime', 'date'],
                           'open': ['open'],
                           'high': ['high'], 
                           'low': ['low'],
                           'close': ['close'],
                           'volume': ['volume']}
            
            for std_name, possible_names in standard_cols.items():
                for col in sample_df.columns:
                    if col.lower() in [name.lower() for name in possible_names]:
                        column_mapping[col] = std_name
                        break
            
            # Read with optimized dtypes based on actual column names
            dtype_map = {}
            timestamp_col = None
            for actual_col, std_col in column_mapping.items():
                if std_col in ['open', 'high', 'low', 'close']:
                    dtype_map[actual_col] = np.float32
                elif std_col == 'volume':
                    dtype_map[actual_col] = np.int64
                elif std_col == 'timestamp':
                    timestamp_col = actual_col
            
            # Read full file  
            df = pd.read_csv(file_path, dtype=dtype_map)
            
            # Handle timestamp parsing with specific format
            if timestamp_col:
                # Clean invalid timestamp values first
                df = df[df[timestamp_col] != '0']  # Remove "0" strings
                df = df[df[timestamp_col] != 0]    # Remove numeric 0
                df = df.dropna(subset=[timestamp_col])  # Remove nulls
                
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                df = df.dropna(subset=[timestamp_col])  # Remove any remaining invalid timestamps
            else:
                # Use first column as timestamp if no timestamp column found
                first_col = df.columns[0]
                df = df[df[first_col] != '0']
                df = df[df[first_col] != 0]
                df = df.dropna(subset=[first_col])
                
                df[first_col] = pd.to_datetime(df[first_col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                df = df.dropna(subset=[first_col])
                column_mapping[first_col] = 'timestamp'
            
            # Rename columns to standard names
            df = df.rename(columns=column_mapping)
            
            # Data validation
            if self.validation_enabled:
                validation_start = time.time()
                validation_result = self.validator.validate_data(df)
                validation_time = time.time() - validation_start
                
                self.metrics.record_validation_time(validation_time)
                
                if not validation_result.is_valid:
                    raise ValueError(f"Data validation failed: {validation_result.errors}")
                
                if validation_result.warnings:
                    self.logger.warning(f"Data validation warnings: {validation_result.warnings}")
            
            # Data preprocessing
            if self.preprocessing_enabled:
                preprocessing_start = time.time()
                df = self.preprocessor.preprocess_data(df)
                preprocessing_time = time.time() - preprocessing_start
                
                self.metrics.record_preprocessing_time(preprocessing_time)
            
            # Save to cache
            self._save_to_cache(df, cache_path, file_hash)
        
        # Apply filters
        if start_date or end_date:
            df = self._apply_date_filters(df, start_date, end_date)
        
        if columns:
            available_cols = [col for col in columns if col in df.columns]
            if len(available_cols) != len(columns):
                missing = [col for col in columns if col not in df.columns]
                self.logger.warning(f"Missing columns: {missing}")
            df = df[available_cols]
        
        # Record metrics
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        load_time = time.time() - start_time
        self.metrics.record_load_time(load_time, len(df), memory_usage)
        
        self.logger.info(f"Loaded {len(df)} rows in {load_time:.2f}s ({memory_usage:.1f} MB)")
        
        return df
    
    def _apply_date_filters(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Apply date range filters"""
        if 'timestamp' not in df.columns:
            self.logger.warning("No timestamp column for date filtering")
            return df
        
        mask = pd.Series(True, index=df.index)
        
        if start_date:
            mask &= df['timestamp'] >= pd.to_datetime(start_date)
        
        if end_date:
            mask &= df['timestamp'] <= pd.to_datetime(end_date)
        
        return df[mask]
    
    def load_chunked_data(self, 
                          timeframe: str = "30min",
                          chunk_size: Optional[int] = None,
                          **kwargs) -> Iterator[pd.DataFrame]:
        """
        Load data in chunks for memory-efficient processing
        
        Args:
            timeframe: Data timeframe
            chunk_size: Size of each chunk (override default)
            **kwargs: Additional arguments for load_data
        
        Yields:
            DataFrame chunks
        """
        chunk_size = chunk_size or self.chunk_size
        
        # Load full dataset first to get total size
        full_df = self.load_data(timeframe, **kwargs)
        total_rows = len(full_df)
        
        self.logger.info(f"Loading {total_rows} rows in chunks of {chunk_size}")
        
        # Yield chunks
        for i in range(0, total_rows, chunk_size):
            chunk = full_df.iloc[i:i + chunk_size]
            yield chunk
    
    def load_multiple_timeframes(self, 
                                timeframes: List[str],
                                align_timestamps: bool = True,
                                **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load multiple timeframes simultaneously
        
        Args:
            timeframes: List of timeframes to load
            align_timestamps: Whether to align timestamps across timeframes
            **kwargs: Additional arguments for load_data
        
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        results = {}
        
        # Load each timeframe
        for tf in timeframes:
            if tf in self.available_files:
                results[tf] = self.load_data(tf, **kwargs)
            else:
                self.logger.warning(f"Timeframe {tf} not available")
        
        # Align timestamps if requested
        if align_timestamps and len(results) > 1:
            results = self._align_timestamps(results)
        
        return results
    
    def _align_timestamps(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align timestamps across different timeframes - FIXED synchronization"""
        # Find common timestamp range
        min_timestamp = None
        max_timestamp = None
        
        for df in data_dict.values():
            if 'timestamp' in df.columns:
                ts_min = df['timestamp'].min()
                ts_max = df['timestamp'].max()
                
                if min_timestamp is None or ts_min > min_timestamp:
                    min_timestamp = ts_min
                if max_timestamp is None or ts_max < max_timestamp:
                    max_timestamp = ts_max
        
        # Filter all dataframes to common range
        aligned_data = {}
        for timeframe, df in data_dict.items():
            if 'timestamp' in df.columns:
                mask = (df['timestamp'] >= min_timestamp) & (df['timestamp'] <= max_timestamp)
                aligned_df = df[mask].copy()
                
                # Ensure timestamps are properly sorted and de-duplicated
                aligned_df = aligned_df.sort_values('timestamp')
                aligned_df = aligned_df.drop_duplicates(subset=['timestamp'])
                
                # For multi-timeframe synchronization, ensure consistent intervals
                if timeframe == '5min':
                    # Ensure 5-minute intervals are consistent
                    aligned_df = aligned_df.resample('5min', on='timestamp').first().dropna()
                elif timeframe == '30min':
                    # Ensure 30-minute intervals are consistent 
                    aligned_df = aligned_df.resample('30min', on='timestamp').first().dropna()
                
                aligned_data[timeframe] = aligned_df
                
                self.logger.info(f"Aligned {timeframe}: {len(aligned_df)} bars from {aligned_df['timestamp'].min()} to {aligned_df['timestamp'].max()}")
            else:
                aligned_data[timeframe] = df
        
        return aligned_data
    
    def get_data_statistics(self, timeframe: str = "30min") -> Dict[str, Any]:
        """Get comprehensive statistics for a dataset"""
        df = self.load_data(timeframe)
        
        stats = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'dtypes': df.dtypes.value_counts().to_dict()
            },
            'time_range': {},
            'price_statistics': {},
            'volume_statistics': {},
            'feature_statistics': {}
        }
        
        # Time range analysis
        if 'timestamp' in df.columns:
            stats['time_range'] = {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max(),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
                'frequency': self._infer_frequency(df['timestamp'])
            }
        
        # Price statistics
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                stats['price_statistics'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skew': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                }
        
        # Volume statistics
        if 'volume' in df.columns:
            stats['volume_statistics'] = {
                'mean': df['volume'].mean(),
                'std': df['volume'].std(),
                'min': df['volume'].min(),
                'max': df['volume'].max(),
                'total': df['volume'].sum()
            }
        
        # Feature statistics
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                stats['feature_statistics'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'null_count': df[col].isnull().sum()
                }
        
        return stats
    
    def _infer_frequency(self, timestamps: pd.Series) -> str:
        """Infer the frequency of timestamps"""
        if len(timestamps) < 2:
            return "unknown"
        
        # Calculate mode of time differences
        diffs = timestamps.diff().dropna()
        if len(diffs) == 0:
            return "unknown"
        
        mode_diff = diffs.mode().iloc[0]
        
        # Convert to common frequency strings
        if mode_diff == pd.Timedelta(minutes=1):
            return "1min"
        elif mode_diff == pd.Timedelta(minutes=5):
            return "5min"
        elif mode_diff == pd.Timedelta(minutes=30):
            return "30min"
        elif mode_diff == pd.Timedelta(hours=1):
            return "1hour"
        elif mode_diff == pd.Timedelta(days=1):
            return "1day"
        else:
            return str(mode_diff)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get data loading performance metrics"""
        return self.metrics.get_summary()
    
    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
        self.logger.info("Cache cleared")
    
    def preload_data(self, timeframes: List[str]):
        """Preload data into cache for faster access"""
        for tf in timeframes:
            if tf in self.available_files:
                self.logger.info(f"Preloading {tf} data...")
                self.load_data(tf)
        
        self.logger.info("Data preloading completed")
    
    def _initialize_completeness_monitor(self) -> Dict[str, Any]:
        """Initialize data completeness monitoring"""
        return {
            'total_expected_bars': {},
            'actual_bars_loaded': {},
            'missing_periods': {},
            'data_quality_scores': {},
            'last_monitoring_update': None
        }
    
    def monitor_data_completeness(self, timeframe: str) -> Dict[str, Any]:
        """
        Monitor data completeness for a specific timeframe
        """
        if timeframe not in self.available_files:
            return {'error': f'Timeframe {timeframe} not available'}
        
        try:
            # Load data for analysis
            df = self.load_data(timeframe)
            
            # Calculate completeness metrics
            total_rows = len(df)
            date_range = df['timestamp'].max() - df['timestamp'].min() if 'timestamp' in df.columns else None
            
            # Expected bars calculation based on timeframe
            if timeframe == '5min' and date_range:
                expected_bars = int(date_range.total_seconds() / 300)  # 5 minutes = 300 seconds
            elif timeframe == '30min' and date_range:
                expected_bars = int(date_range.total_seconds() / 1800)  # 30 minutes = 1800 seconds
            else:
                expected_bars = total_rows  # Fallback
            
            # Data quality checks
            null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            
            # Detect missing time periods
            missing_periods = self._detect_missing_periods(df, timeframe)
            
            # Calculate completeness score
            completeness_score = (total_rows / expected_bars) * 100 if expected_bars > 0 else 0
            quality_score = max(0, 100 - null_percentage)
            overall_score = (completeness_score + quality_score) / 2
            
            # Update monitoring data
            self.completeness_monitor.update({
                'total_expected_bars': {timeframe: expected_bars},
                'actual_bars_loaded': {timeframe: total_rows},
                'missing_periods': {timeframe: missing_periods},
                'data_quality_scores': {timeframe: {
                    'completeness_score': completeness_score,
                    'quality_score': quality_score,
                    'overall_score': overall_score,
                    'null_percentage': null_percentage
                }},
                'last_monitoring_update': datetime.now().isoformat()
            })
            
            # Log results
            self.logger.info(f"Data completeness monitoring for {timeframe}:")
            self.logger.info(f"  Expected bars: {expected_bars:,}")
            self.logger.info(f"  Actual bars: {total_rows:,}")
            self.logger.info(f"  Completeness: {completeness_score:.1f}%")
            self.logger.info(f"  Quality score: {quality_score:.1f}%")
            self.logger.info(f"  Overall score: {overall_score:.1f}%")
            
            if missing_periods:
                self.logger.warning(f"  Missing periods detected: {len(missing_periods)}")
            
            return self.completeness_monitor
            
        except Exception as e:
            self.logger.error(f"Data completeness monitoring failed: {e}")
            return {'error': str(e)}
    
    def _detect_missing_periods(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """
        Detect missing time periods in the data
        """
        if 'timestamp' not in df.columns or len(df) < 2:
            return []
        
        # Expected frequency
        freq_map = {
            '5min': '5T',
            '30min': '30T',
            '5min_extended': '5T'
        }
        
        expected_freq = freq_map.get(timeframe, '5T')
        
        try:
            # Create expected time range
            start_time = df['timestamp'].min()
            end_time = df['timestamp'].max()
            expected_range = pd.date_range(start=start_time, end=end_time, freq=expected_freq)
            
            # Find missing timestamps
            missing_timestamps = expected_range.difference(df['timestamp'])
            
            # Group consecutive missing periods
            missing_periods = []
            if len(missing_timestamps) > 0:
                current_start = None
                current_end = None
                
                for ts in missing_timestamps:
                    if current_start is None:
                        current_start = ts
                        current_end = ts
                    elif (ts - current_end).total_seconds() <= pd.Timedelta(expected_freq).total_seconds() * 2:
                        current_end = ts
                    else:
                        # End of consecutive period
                        missing_periods.append({
                            'start': current_start.isoformat(),
                            'end': current_end.isoformat(),
                            'duration': str(current_end - current_start),
                            'missing_bars': len(pd.date_range(current_start, current_end, freq=expected_freq))
                        })
                        current_start = ts
                        current_end = ts
                
                # Add final period
                if current_start is not None:
                    missing_periods.append({
                        'start': current_start.isoformat(),
                        'end': current_end.isoformat(),
                        'duration': str(current_end - current_start),
                        'missing_bars': len(pd.date_range(current_start, current_end, freq=expected_freq))
                    })
            
            return missing_periods
            
        except Exception as e:
            self.logger.warning(f"Missing period detection failed: {e}")
            return []
    
    def get_completeness_report(self) -> Dict[str, Any]:
        """Get comprehensive data completeness report"""
        report = {
            'summary': {},
            'detailed_metrics': self.completeness_monitor,
            'recommendations': []
        }
        
        # Generate summary
        if self.completeness_monitor.get('data_quality_scores'):
            scores = []
            for timeframe, metrics in self.completeness_monitor['data_quality_scores'].items():
                scores.append(metrics['overall_score'])
                report['summary'][timeframe] = {
                    'status': 'GOOD' if metrics['overall_score'] > 85 else 'FAIR' if metrics['overall_score'] > 70 else 'POOR',
                    'score': metrics['overall_score']
                }
            
            report['summary']['average_score'] = sum(scores) / len(scores) if scores else 0
        
        # Generate recommendations
        for timeframe, metrics in self.completeness_monitor.get('data_quality_scores', {}).items():
            if metrics['completeness_score'] < 90:
                report['recommendations'].append(f"Consider filling missing data gaps for {timeframe}")
            if metrics['quality_score'] < 95:
                report['recommendations'].append(f"Address data quality issues in {timeframe} (null values)")
        
        return report