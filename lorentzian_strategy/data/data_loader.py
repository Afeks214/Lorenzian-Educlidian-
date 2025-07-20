"""
Robust Data Loading and Validation System for Lorentzian Trading Strategy
Handles NQ futures data with comprehensive validation and preprocessing.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass

from ..config.config import get_config


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_records: int
    date_range: Tuple[datetime, datetime]
    missing_values: Dict[str, int]
    duplicates: int
    ohlc_violations: Dict[str, int]
    zero_volume_records: int
    time_gaps: Dict[str, int]
    statistical_summary: Dict[str, Dict[str, float]]
    quality_score: float
    issues: List[str]
    recommendations: List[str]


class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def validate_ohlc_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate OHLC data integrity"""
        issues = []
        violations = {}
        
        # Check High >= Low
        high_low_valid = df['High'] >= df['Low']
        violations['high_low'] = (~high_low_valid).sum()
        if violations['high_low'] > 0:
            issues.append(f"{violations['high_low']} records where High < Low")
        
        # Check Open within High-Low range
        open_valid = (df['Open'] >= df['Low']) & (df['Open'] <= df['High'])
        violations['open_range'] = (~open_valid).sum()
        if violations['open_range'] > 0:
            issues.append(f"{violations['open_range']} records where Open outside High-Low range")
        
        # Check Close within High-Low range
        close_valid = (df['Close'] >= df['Low']) & (df['Close'] <= df['High'])
        violations['close_range'] = (~close_valid).sum()
        if violations['close_range'] > 0:
            issues.append(f"{violations['close_range']} records where Close outside High-Low range")
        
        # Check for negative prices
        negative_prices = (df[['Open', 'High', 'Low', 'Close']] < 0).any(axis=1)
        violations['negative_prices'] = negative_prices.sum()
        if violations['negative_prices'] > 0:
            issues.append(f"{violations['negative_prices']} records with negative prices")
        
        # Check for unrealistic price jumps (>50% in one bar)
        price_changes = df['Close'].pct_change().abs()
        large_jumps = price_changes > 0.5
        violations['large_jumps'] = large_jumps.sum()
        if violations['large_jumps'] > 0:
            issues.append(f"{violations['large_jumps']} records with >50% price jumps")
        
        return {
            'violations': violations,
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def validate_timestamps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate timestamp data"""
        issues = []
        
        # Check for duplicates
        duplicates = df['Timestamp'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate timestamps")
        
        # Check chronological order
        if not df['Timestamp'].is_monotonic_increasing:
            issues.append("Timestamps are not in chronological order")
        
        # Analyze time gaps
        time_diff = df['Timestamp'].diff()
        expected_interval = pd.Timedelta(minutes=30)  # 30-minute bars
        
        # Count different interval types
        gap_analysis = {}
        gap_analysis['normal_30min'] = (time_diff == expected_interval).sum()
        gap_analysis['1hour'] = (time_diff == pd.Timedelta(hours=1)).sum()
        gap_analysis['weekend_gaps'] = (time_diff > pd.Timedelta(days=1)).sum()
        gap_analysis['missing_bars'] = (
            (time_diff > expected_interval) & 
            (time_diff < pd.Timedelta(days=1))
        ).sum()
        
        return {
            'duplicates': duplicates,
            'gap_analysis': gap_analysis,
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def validate_volume_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate volume data"""
        issues = []
        
        # Check for negative volume
        negative_volume = (df['Volume'] < 0).sum()
        if negative_volume > 0:
            issues.append(f"{negative_volume} records with negative volume")
        
        # Count zero volume records
        zero_volume = (df['Volume'] == 0).sum()
        if zero_volume > 0:
            issues.append(f"{zero_volume} records with zero volume")
        
        # Check for extremely high volume (outliers)
        volume_q99 = df['Volume'].quantile(0.99)
        extreme_volume = (df['Volume'] > volume_q99 * 10).sum()
        if extreme_volume > 0:
            issues.append(f"{extreme_volume} records with extremely high volume")
        
        return {
            'negative_volume': negative_volume,
            'zero_volume': zero_volume,
            'extreme_volume': extreme_volume,
            'issues': issues,
            'valid': len(issues) == 0
        }
    
    def generate_quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """Generate comprehensive data quality report"""
        self.logger.info("Generating data quality report...")
        
        # Basic statistics
        total_records = len(df)
        date_range = (df['Timestamp'].min(), df['Timestamp'].max())
        missing_values = df.isnull().sum().to_dict()
        
        # Validation results
        ohlc_validation = self.validate_ohlc_data(df)
        timestamp_validation = self.validate_timestamps(df)
        volume_validation = self.validate_volume_data(df)
        
        # Collect all issues
        all_issues = []
        all_issues.extend(ohlc_validation['issues'])
        all_issues.extend(timestamp_validation['issues'])
        all_issues.extend(volume_validation['issues'])
        
        # Statistical summary
        numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        statistical_summary = {}
        for col in numerical_cols:
            statistical_summary[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q50': float(df[col].quantile(0.50)),
                'q75': float(df[col].quantile(0.75))
            }
        
        # Calculate quality score (0-100)
        total_violations = sum(ohlc_validation['violations'].values())
        quality_score = max(0, 100 - (total_violations / total_records * 100))
        
        # Generate recommendations
        recommendations = []
        if ohlc_validation['violations']['high_low'] > 0:
            recommendations.append("Fix High < Low violations by swapping values")
        if timestamp_validation['duplicates'] > 0:
            recommendations.append("Remove duplicate timestamps")
        if volume_validation['zero_volume'] > 50:
            recommendations.append("Consider removing or interpolating zero volume records")
        if quality_score < 95:
            recommendations.append("Data quality is below recommended threshold")
        
        return DataQualityReport(
            total_records=total_records,
            date_range=date_range,
            missing_values=missing_values,
            duplicates=timestamp_validation['duplicates'],
            ohlc_violations=ohlc_validation['violations'],
            zero_volume_records=volume_validation['zero_volume'],
            time_gaps=timestamp_validation['gap_analysis'],
            statistical_summary=statistical_summary,
            quality_score=quality_score,
            issues=all_issues,
            recommendations=recommendations
        )


class DataProcessor:
    """Data preprocessing and cleaning pipeline"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess data"""
        self.logger.info("Cleaning data...")
        original_length = len(df)
        
        # Copy to avoid modifying original
        df_clean = df.copy()
        
        # Fix OHLC violations
        df_clean = self._fix_ohlc_violations(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove duplicates
        if self.config.data.remove_duplicates:
            df_clean = df_clean.drop_duplicates(subset=['Timestamp'])
        
        # Sort by timestamp
        df_clean = df_clean.sort_values('Timestamp').reset_index(drop=True)
        
        self.logger.info(f"Cleaned data: {original_length} -> {len(df_clean)} records")
        return df_clean
    
    def _fix_ohlc_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix OHLC data violations"""
        # Fix High < Low by swapping
        mask = df['High'] < df['Low']
        if mask.any():
            self.logger.warning(f"Fixing {mask.sum()} High < Low violations")
            df.loc[mask, ['High', 'Low']] = df.loc[mask, ['Low', 'High']].values
        
        # Fix Open outside High-Low range
        df['Open'] = df['Open'].clip(df['Low'], df['High'])
        
        # Fix Close outside High-Low range
        df['Close'] = df['Close'].clip(df['Low'], df['High'])
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration"""
        method = self.config.data.fill_missing_method
        
        if method == "forward":
            df = df.fillna(method='ffill')
        elif method == "backward":
            df = df.fillna(method='bfill')
        elif method == "interpolate":
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].interpolate()
            df['Volume'] = df['Volume'].fillna(0)
        elif method == "drop":
            df = df.dropna()
        
        return df


class DataLoader:
    """Main data loading and management system"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.validator = DataValidator(config)
        self.processor = DataProcessor(config)
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        self.config.ensure_directories()
    
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load raw data from CSV file"""
        file_path = file_path or self.config.data.source_file
        self.logger.info(f"Loading raw data from: {file_path}")
        
        try:
            # Load with chunk processing for large files
            if self.config.data.chunk_size:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.config.data.chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
            
            # Parse timestamps
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, format='mixed')
            
            self.logger.info(f"Loaded {len(df)} records from {df['Timestamp'].min()} to {df['Timestamp'].max()}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def get_cache_path(self, data_hash: str, suffix: str = "") -> str:
        """Generate cache file path based on data hash"""
        cache_dir = Path(self.config.data.cache_dir)
        return str(cache_dir / f"data_{data_hash}{suffix}.pkl")
    
    def get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash for dataframe to enable caching"""
        # Use shape, first/last timestamps, and basic stats for hash
        hash_data = {
            'shape': df.shape,
            'first_timestamp': str(df['Timestamp'].iloc[0]),
            'last_timestamp': str(df['Timestamp'].iloc[-1]),
            'close_mean': float(df['Close'].mean()),
            'volume_sum': int(df['Volume'].sum())
        }
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()[:12]
    
    def save_cache(self, df: pd.DataFrame, cache_path: str):
        """Save dataframe to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
            self.logger.debug(f"Saved cache to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def load_cache(self, cache_path: str) -> Optional[pd.DataFrame]:
        """Load dataframe from cache"""
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    df = pickle.load(f)
                self.logger.debug(f"Loaded cache from {cache_path}")
                return df
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
        return None
    
    def load_and_process_data(self, file_path: Optional[str] = None, 
                             use_cache: bool = None) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Load, validate, and process data with caching"""
        use_cache = use_cache if use_cache is not None else self.config.data.cache_enabled
        
        # Load raw data
        df_raw = self.load_raw_data(file_path)
        data_hash = self.get_data_hash(df_raw)
        
        # Check cache
        cache_path = self.get_cache_path(data_hash, "_processed")
        report_cache_path = self.get_cache_path(data_hash, "_report")
        
        if use_cache:
            cached_df = self.load_cache(cache_path)
            cached_report = self.load_cache(report_cache_path)
            if cached_df is not None and cached_report is not None:
                self.logger.info("Using cached processed data")
                return cached_df, cached_report
        
        # Generate quality report on raw data
        quality_report = self.validator.generate_quality_report(df_raw)
        
        # Process data
        df_processed = self.processor.clean_data(df_raw)
        
        # Save to cache
        if use_cache:
            self.save_cache(df_processed, cache_path)
            self.save_cache(quality_report, report_cache_path)
        
        return df_processed, quality_report
    
    def save_quality_report(self, report: DataQualityReport, output_path: Optional[str] = None):
        """Save quality report to JSON file"""
        if output_path is None:
            validation_dir = Path(self.config.data.validation_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = validation_dir / f"quality_report_{timestamp}.json"
        
        # Convert to dict for JSON serialization
        report_dict = {
            'total_records': report.total_records,
            'date_range': [report.date_range[0].isoformat(), report.date_range[1].isoformat()],
            'missing_values': report.missing_values,
            'duplicates': report.duplicates,
            'ohlc_violations': report.ohlc_violations,
            'zero_volume_records': report.zero_volume_records,
            'time_gaps': report.time_gaps,
            'statistical_summary': report.statistical_summary,
            'quality_score': report.quality_score,
            'issues': report.issues,
            'recommendations': report.recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Quality report saved to: {output_path}")
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data information"""
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': {
                'start': df['Timestamp'].min().isoformat(),
                'end': df['Timestamp'].max().isoformat(),
                'duration_days': (df['Timestamp'].max() - df['Timestamp'].min()).days
            },
            'price_range': {
                'min': float(df['Low'].min()),
                'max': float(df['High'].max()),
                'current': float(df['Close'].iloc[-1])
            },
            'volume_stats': {
                'total': int(df['Volume'].sum()),
                'avg_daily': float(df.groupby(df['Timestamp'].dt.date)['Volume'].sum().mean()),
                'max_bar': int(df['Volume'].max())
            }
        }


# Convenience function for quick data loading
def load_nq_data(config=None, use_cache: bool = True) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Quick function to load NQ data with default settings"""
    loader = DataLoader(config)
    return loader.load_and_process_data(use_cache=use_cache)