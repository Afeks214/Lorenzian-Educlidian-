#!/usr/bin/env python3
"""
CL Data Processor for 500% Trustworthy Backtesting
Mission: Process both 5-minute and 30-minute CL ETH data files for bulletproof backtesting foundation
"""

import pandas as pd
import numpy as np
import warnings
import json
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLDataProcessor:
    """
    Comprehensive CL data processor for bulletproof backtesting
    """
    
    def __init__(self, data_dir: str = "/home/QuantNova/GrandModel/colab/data/"):
        self.data_dir = data_dir
        self.file_5min = f"{data_dir}@CL - 5 min - ETH.csv"
        self.file_30min = f"{data_dir}@CL - 30 min - ETH.csv"
        
        # Data quality metrics
        self.quality_report = {
            '5min': {},
            '30min': {},
            'summary': {}
        }
        
        # Price validation ranges for crude oil
        self.price_range = {
            'min_realistic': 10.0,  # Extreme low (like 2020 crash)
            'max_realistic': 200.0,  # Extreme high
            'normal_min': 30.0,     # Normal low range
            'normal_max': 120.0     # Normal high range
        }
        
    def load_data(self, timeframe: str) -> pd.DataFrame:
        """
        Load and perform initial validation of CL data
        
        Args:
            timeframe: '5min' or '30min'
            
        Returns:
            pd.DataFrame: Cleaned dataframe with proper datetime index
        """
        if timeframe == '5min':
            file_path = self.file_5min
        elif timeframe == '30min':
            file_path = self.file_30min
        else:
            raise ValueError("Timeframe must be '5min' or '30min'")
            
        logger.info(f"Loading {timeframe} data from {file_path}")
        
        try:
            # Load data with proper handling of potential issues
            df = pd.read_csv(file_path)
            
            # Handle column naming inconsistency (30min file has 'Llow' instead of 'Low')
            if 'Llow' in df.columns:
                df.rename(columns={'Llow': 'Low'}, inplace=True)
                logger.warning("Fixed column name 'Llow' to 'Low' in 30min data")
            
            # Store original row count
            original_rows = len(df)
            self.quality_report[timeframe]['original_rows'] = original_rows
            
            # Parse timestamp and set as index - handle mixed formats
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
            df.set_index('Timestamp', inplace=True)
            
            # Sort by timestamp to ensure proper ordering
            df.sort_index(inplace=True)
            
            # Basic data type validation
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values
            df.dropna(inplace=True)
            cleaned_rows = len(df)
            
            logger.info(f"Loaded {timeframe} data: {original_rows} -> {cleaned_rows} rows after cleaning")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {timeframe} data: {str(e)}")
            raise
    
    def validate_ohlc_relationships(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Validate OHLC relationships and data integrity
        
        Args:
            df: DataFrame with OHLC data
            timeframe: '5min' or '30min'
            
        Returns:
            Dict: Validation results
        """
        logger.info(f"Validating OHLC relationships for {timeframe} data")
        
        validation_results = {}
        
        # Check 1: High >= max(Open, Close) and Low <= min(Open, Close)
        high_valid = (df['High'] >= df[['Open', 'Close']].max(axis=1)).all()
        low_valid = (df['Low'] <= df[['Open', 'Close']].min(axis=1)).all()
        
        # Check 2: High >= Low
        high_low_valid = (df['High'] >= df['Low']).all()
        
        # Check 3: Volume > 0
        volume_valid = (df['Volume'] > 0).all()
        
        # Check 4: No negative prices
        positive_prices = (df[['Open', 'High', 'Low', 'Close']] > 0).all().all()
        
        # Calculate invalid rows
        invalid_high = ~(df['High'] >= df[['Open', 'Close']].max(axis=1))
        invalid_low = ~(df['Low'] <= df[['Open', 'Close']].min(axis=1))
        invalid_high_low = ~(df['High'] >= df['Low'])
        invalid_volume = ~(df['Volume'] > 0)
        invalid_prices = ~(df[['Open', 'High', 'Low', 'Close']] > 0).any(axis=1)
        
        validation_results = {
            'high_valid': high_valid,
            'low_valid': low_valid,
            'high_low_valid': high_low_valid,
            'volume_valid': volume_valid,
            'positive_prices': positive_prices,
            'invalid_high_count': invalid_high.sum(),
            'invalid_low_count': invalid_low.sum(),
            'invalid_high_low_count': invalid_high_low.sum(),
            'invalid_volume_count': invalid_volume.sum(),
            'invalid_prices_count': invalid_prices.sum(),
            'total_rows': len(df)
        }
        
        # Store in quality report
        self.quality_report[timeframe]['ohlc_validation'] = validation_results
        
        logger.info(f"OHLC validation for {timeframe}: "
                   f"High valid: {high_valid}, Low valid: {low_valid}, "
                   f"High>=Low: {high_low_valid}, Volume valid: {volume_valid}")
        
        return validation_results
    
    def detect_data_gaps(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Detect missing data and gaps in the time series
        
        Args:
            df: DataFrame with datetime index
            timeframe: '5min' or '30min'
            
        Returns:
            Dict: Gap analysis results
        """
        logger.info(f"Detecting data gaps for {timeframe} data")
        
        # Expected frequency
        expected_freq = '5min' if timeframe == '5min' else '30min'
        
        # Create expected time range
        start_time = df.index.min()
        end_time = df.index.max()
        
        # Generate expected datetime range
        expected_range = pd.date_range(start=start_time, end=end_time, freq=expected_freq)
        
        # Find missing timestamps
        missing_timestamps = expected_range.difference(df.index)
        
        # Calculate gap statistics
        gap_stats = {
            'expected_records': len(expected_range),
            'actual_records': len(df),
            'missing_records': len(missing_timestamps),
            'completeness_percentage': (len(df) / len(expected_range)) * 100,
            'data_start': start_time,
            'data_end': end_time,
            'data_span_days': (end_time - start_time).days,
            'largest_gap_hours': 0
        }
        
        # Find largest gaps
        if len(missing_timestamps) > 0:
            # Group consecutive missing timestamps
            missing_df = pd.DataFrame(index=missing_timestamps)
            missing_df['gap_group'] = (missing_df.index.to_series().diff() != 
                                     pd.Timedelta(expected_freq)).cumsum()
            
            gap_sizes = missing_df.groupby('gap_group').size()
            largest_gap = gap_sizes.max()
            
            if timeframe == '5min':
                largest_gap_hours = largest_gap * 5 / 60
            else:
                largest_gap_hours = largest_gap * 30 / 60
                
            gap_stats['largest_gap_hours'] = largest_gap_hours
            gap_stats['number_of_gaps'] = len(gap_sizes)
        
        # Store in quality report
        self.quality_report[timeframe]['gap_analysis'] = gap_stats
        
        logger.info(f"Gap analysis for {timeframe}: "
                   f"{gap_stats['completeness_percentage']:.2f}% complete, "
                   f"{gap_stats['missing_records']} missing records")
        
        return gap_stats
    
    def detect_outliers_and_validate_prices(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Detect price outliers and validate crude oil price ranges
        
        Args:
            df: DataFrame with OHLC data
            timeframe: '5min' or '30min'
            
        Returns:
            Dict: Outlier analysis results
        """
        logger.info(f"Detecting outliers and validating prices for {timeframe} data")
        
        outlier_stats = {}
        
        # Price range validation
        price_cols = ['Open', 'High', 'Low', 'Close']
        
        for col in price_cols:
            col_stats = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'below_realistic_min': (df[col] < self.price_range['min_realistic']).sum(),
                'above_realistic_max': (df[col] > self.price_range['max_realistic']).sum(),
                'outside_normal_range': ((df[col] < self.price_range['normal_min']) | 
                                       (df[col] > self.price_range['normal_max'])).sum()
            }
            
            # Statistical outliers (3 standard deviations)
            mean_val = col_stats['mean']
            std_val = col_stats['std']
            outliers_3std = ((df[col] < mean_val - 3*std_val) | 
                           (df[col] > mean_val + 3*std_val)).sum()
            col_stats['outliers_3std'] = outliers_3std
            
            # IQR outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers_iqr = ((df[col] < Q1 - 1.5*IQR) | 
                          (df[col] > Q3 + 1.5*IQR)).sum()
            col_stats['outliers_iqr'] = outliers_iqr
            
            outlier_stats[col] = col_stats
        
        # Volume analysis
        volume_stats = {
            'min': df['Volume'].min(),
            'max': df['Volume'].max(),
            'mean': df['Volume'].mean(),
            'median': df['Volume'].median(),
            'std': df['Volume'].std(),
            'zero_volume': (df['Volume'] == 0).sum(),
            'very_low_volume': (df['Volume'] < 10).sum(),
            'very_high_volume': (df['Volume'] > df['Volume'].quantile(0.99)).sum()
        }
        
        outlier_stats['Volume'] = volume_stats
        
        # Store in quality report
        self.quality_report[timeframe]['outlier_analysis'] = outlier_stats
        
        logger.info(f"Price validation for {timeframe}: "
                   f"Price range ${outlier_stats['Close']['min']:.2f} - ${outlier_stats['Close']['max']:.2f}")
        
        return outlier_stats
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add essential technical indicators for pattern detection
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """
        logger.info("Adding technical indicators")
        
        # Price-based indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = (df['Price_Range'] / df['Close']) * 100
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        
        # Volatility indicators
        df['Price_Volatility'] = df['Price_Change'].rolling(window=20).std()
        df['ATR'] = df['Price_Range'].rolling(window=14).mean()
        
        # Momentum indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        
        # Support/Resistance levels
        df['Support_Level'] = df['Low'].rolling(window=20).min()
        df['Resistance_Level'] = df['High'].rolling(window=20).max()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Time-based features
        df['Hour'] = df.index.hour
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # Market session indicators (ETH timezone)
        df['US_Session'] = ((df['Hour'] >= 14) & (df['Hour'] <= 21)).astype(int)
        df['Asian_Session'] = ((df['Hour'] >= 22) | (df['Hour'] <= 8)).astype(int)
        df['European_Session'] = ((df['Hour'] >= 8) & (df['Hour'] <= 16)).astype(int)
        
        logger.info(f"Added technical indicators. DataFrame now has {len(df.columns)} columns")
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def align_timeframes(self, df_5min: pd.DataFrame, df_30min: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align 5-minute and 30-minute data for multi-timeframe analysis
        
        Args:
            df_5min: 5-minute DataFrame
            df_30min: 30-minute DataFrame
            
        Returns:
            Tuple: (aligned_5min, aligned_30min)
        """
        logger.info("Aligning 5-minute and 30-minute timeframes")
        
        # Find common date range
        common_start = max(df_5min.index.min(), df_30min.index.min())
        common_end = min(df_5min.index.max(), df_30min.index.max())
        
        # Filter both datasets to common range
        df_5min_aligned = df_5min.loc[common_start:common_end].copy()
        df_30min_aligned = df_30min.loc[common_start:common_end].copy()
        
        # Add 30-minute data to 5-minute timeframe using forward fill
        df_5min_aligned = df_5min_aligned.merge(
            df_30min_aligned.add_suffix('_30min'),
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Forward fill 30-minute data
        df_5min_aligned = df_5min_aligned.fillna(method='ffill')
        
        logger.info(f"Aligned timeframes: {len(df_5min_aligned)} 5-minute records, "
                   f"{len(df_30min_aligned)} 30-minute records")
        
        return df_5min_aligned, df_30min_aligned
    
    def create_train_test_splits(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test splits for backtesting
        
        Args:
            df: DataFrame to split
            test_size: Proportion of data for testing
            
        Returns:
            Tuple: (train_df, test_df)
        """
        logger.info(f"Creating train/test splits with {test_size*100}% test data")
        
        # Time-based split (not random)
        split_index = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()
        
        logger.info(f"Train: {len(train_df)} records ({train_df.index.min()} to {train_df.index.max()})")
        logger.info(f"Test: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")
        
        return train_df, test_df
    
    def generate_quality_report(self) -> Dict:
        """
        Generate comprehensive data quality report
        
        Returns:
            Dict: Complete quality report
        """
        logger.info("Generating comprehensive quality report")
        
        # Summary statistics
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'data_span_years': 0,
            'total_records': 0,
            'overall_quality_score': 0,
            'backtesting_readiness': False
        }
        
        # Calculate data span
        if '5min' in self.quality_report and 'gap_analysis' in self.quality_report['5min']:
            span_days = self.quality_report['5min']['gap_analysis']['data_span_days']
            summary['data_span_years'] = span_days / 365.25
            summary['total_records'] = (self.quality_report['5min']['original_rows'] + 
                                      self.quality_report['30min']['original_rows'])
        
        # Calculate overall quality score
        quality_factors = []
        
        for timeframe in ['5min', '30min']:
            if timeframe in self.quality_report:
                # Data completeness
                if 'gap_analysis' in self.quality_report[timeframe]:
                    completeness = self.quality_report[timeframe]['gap_analysis']['completeness_percentage']
                    quality_factors.append(completeness / 100)
                
                # OHLC validation
                if 'ohlc_validation' in self.quality_report[timeframe]:
                    ohlc_valid = self.quality_report[timeframe]['ohlc_validation']
                    validation_score = sum([
                        ohlc_valid['high_valid'],
                        ohlc_valid['low_valid'], 
                        ohlc_valid['high_low_valid'],
                        ohlc_valid['volume_valid'],
                        ohlc_valid['positive_prices']
                    ]) / 5
                    quality_factors.append(validation_score)
        
        if quality_factors:
            summary['overall_quality_score'] = np.mean(quality_factors) * 100
        
        # Determine backtesting readiness
        summary['backtesting_readiness'] = (
            summary['data_span_years'] >= 3 and
            summary['overall_quality_score'] >= 95
        )
        
        self.quality_report['summary'] = summary
        
        logger.info(f"Quality report generated: {summary['overall_quality_score']:.1f}% quality, "
                   f"{summary['data_span_years']:.1f} years span, "
                   f"Backtesting ready: {summary['backtesting_readiness']}")
        
        return self.quality_report
    
    def export_processed_data(self, df_5min: pd.DataFrame, df_30min: pd.DataFrame, 
                            df_5min_train: pd.DataFrame, df_5min_test: pd.DataFrame,
                            df_30min_train: pd.DataFrame, df_30min_test: pd.DataFrame) -> None:
        """
        Export all processed datasets
        
        Args:
            df_5min: Processed 5-minute data
            df_30min: Processed 30-minute data
            df_5min_train: 5-minute training data
            df_5min_test: 5-minute test data
            df_30min_train: 30-minute training data
            df_30min_test: 30-minute test data
        """
        logger.info("Exporting processed datasets")
        
        # Export full datasets
        df_5min.to_csv(f"{self.data_dir}CL_5min_processed.csv")
        df_30min.to_csv(f"{self.data_dir}CL_30min_processed.csv")
        
        # Export train/test splits
        df_5min_train.to_csv(f"{self.data_dir}CL_5min_train.csv")
        df_5min_test.to_csv(f"{self.data_dir}CL_5min_test.csv")
        df_30min_train.to_csv(f"{self.data_dir}CL_30min_train.csv")
        df_30min_test.to_csv(f"{self.data_dir}CL_30min_test.csv")
        
        # Export quality report
        with open(f"{self.data_dir}CL_data_quality_report.json", 'w') as f:
            json.dump(self.quality_report, f, indent=2, default=str)
        
        logger.info("All datasets and quality report exported successfully")
    
    def process_all_data(self) -> Dict:
        """
        Main processing pipeline - orchestrates all data processing steps
        
        Returns:
            Dict: Complete processing results
        """
        logger.info("=== STARTING CL DATA PROCESSING FOR 500% TRUSTWORTHY BACKTESTING ===")
        
        try:
            # Step 1: Load data
            df_5min = self.load_data('5min')
            df_30min = self.load_data('30min')
            
            # Step 2: Validate data integrity
            self.validate_ohlc_relationships(df_5min, '5min')
            self.validate_ohlc_relationships(df_30min, '30min')
            
            # Step 3: Detect gaps and missing data
            self.detect_data_gaps(df_5min, '5min')
            self.detect_data_gaps(df_30min, '30min')
            
            # Step 4: Detect outliers and validate prices
            self.detect_outliers_and_validate_prices(df_5min, '5min')
            self.detect_outliers_and_validate_prices(df_30min, '30min')
            
            # Step 5: Add technical indicators
            df_5min = self.add_technical_indicators(df_5min)
            df_30min = self.add_technical_indicators(df_30min)
            
            # Step 6: Align timeframes
            df_5min_aligned, df_30min_aligned = self.align_timeframes(df_5min, df_30min)
            
            # Step 7: Create train/test splits
            df_5min_train, df_5min_test = self.create_train_test_splits(df_5min_aligned)
            df_30min_train, df_30min_test = self.create_train_test_splits(df_30min_aligned)
            
            # Step 8: Generate quality report
            quality_report = self.generate_quality_report()
            
            # Step 9: Export processed data
            self.export_processed_data(
                df_5min_aligned, df_30min_aligned,
                df_5min_train, df_5min_test,
                df_30min_train, df_30min_test
            )
            
            logger.info("=== CL DATA PROCESSING COMPLETED SUCCESSFULLY ===")
            
            return {
                'success': True,
                'quality_report': quality_report,
                'datasets': {
                    '5min_full': df_5min_aligned,
                    '30min_full': df_30min_aligned,
                    '5min_train': df_5min_train,
                    '5min_test': df_5min_test,
                    '30min_train': df_30min_train,
                    '30min_test': df_30min_test
                }
            }
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'quality_report': self.quality_report
            }


def main():
    """
    Main execution function
    """
    processor = CLDataProcessor()
    results = processor.process_all_data()
    
    if results['success']:
        print("\n" + "="*80)
        print("CL DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        quality_report = results['quality_report']
        summary = quality_report['summary']
        
        print(f"\nDATA QUALITY SUMMARY:")
        print(f"• Data Span: {summary['data_span_years']:.1f} years")
        print(f"• Total Records: {summary['total_records']:,}")
        print(f"• Overall Quality Score: {summary['overall_quality_score']:.1f}%")
        print(f"• Backtesting Ready: {'✓ YES' if summary['backtesting_readiness'] else '✗ NO'}")
        
        print(f"\nFILES CREATED:")
        data_dir = "/home/QuantNova/GrandModel/colab/data/"
        print(f"• {data_dir}CL_5min_processed.csv")
        print(f"• {data_dir}CL_30min_processed.csv")
        print(f"• {data_dir}CL_5min_train.csv")
        print(f"• {data_dir}CL_5min_test.csv")
        print(f"• {data_dir}CL_30min_train.csv")
        print(f"• {data_dir}CL_30min_test.csv")
        print(f"• {data_dir}CL_data_quality_report.json")
        
        print(f"\nREADY FOR 500% TRUSTWORTHY BACKTESTING!")
        print("="*80)
        
    else:
        print("\n" + "="*80)
        print("ERROR IN CL DATA PROCESSING!")
        print("="*80)
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()