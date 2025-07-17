#!/usr/bin/env python3
"""
Synthetic 5-Year Dataset Generator for Performance Validation
============================================================

This module generates comprehensive synthetic datasets for testing system performance
with 5-year historical data at different time intervals (5-min and 30-min).

Features:
- Realistic market microstructure modeling
- Multiple market regime simulation
- Stress testing scenarios
- Memory-efficient data generation
- Configurable dataset sizes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
import json
import time
import psutil
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SyntheticDataGenerator:
    """
    High-performance synthetic market data generator for large-scale testing.
    
    Generates realistic 5-year datasets with:
    - Realistic volatility clustering
    - Market regime transitions
    - Intraday patterns
    - Volume correlations
    - Stress testing scenarios
    """
    
    def __init__(self, base_price: float = 16000.0, seed: int = 42):
        """Initialize the synthetic data generator."""
        np.random.seed(seed)
        self.base_price = base_price
        self.current_price = base_price
        
        # Market regime parameters
        self.regimes = {
            'bull': {'drift': 0.15, 'volatility': 0.20, 'probability': 0.30},
            'bear': {'drift': -0.12, 'volatility': 0.35, 'probability': 0.25},
            'sideways': {'drift': 0.02, 'volatility': 0.15, 'probability': 0.35},
            'crisis': {'drift': -0.45, 'volatility': 0.80, 'probability': 0.10}
        }
        
        self.current_regime = 'sideways'
        self.regime_duration = 0
        
        # Performance tracking
        self.generation_stats = {
            'total_records': 0,
            'memory_usage_mb': 0,
            'generation_time_seconds': 0,
            'records_per_second': 0
        }
    
    def _get_regime_transition(self) -> str:
        """Determine regime transitions based on current state."""
        self.regime_duration += 1
        
        # Regime persistence logic
        if self.regime_duration < 100:  # Minimum regime duration
            return self.current_regime
        
        # Transition probabilities
        transition_prob = 0.01 + (self.regime_duration - 100) * 0.0001
        
        if np.random.random() < transition_prob:
            # Choose new regime
            regime_weights = [self.regimes[r]['probability'] for r in self.regimes.keys()]
            new_regime = np.random.choice(list(self.regimes.keys()), p=regime_weights)
            
            if new_regime != self.current_regime:
                self.regime_duration = 0
                return new_regime
        
        return self.current_regime
    
    def _generate_intraday_pattern(self, hour: int, minute: int) -> float:
        """Generate realistic intraday volatility patterns."""
        # Market opening hours volatility spike
        if 9 <= hour <= 10:
            return 1.5
        elif 15 <= hour <= 16:  # Market close
            return 1.3
        elif 0 <= hour <= 6:  # Overnight
            return 0.6
        else:
            return 1.0
    
    def _generate_price_move(self, dt: datetime, interval_minutes: int) -> Tuple[float, float, float, float, int]:
        """Generate realistic OHLCV data for a single interval."""
        
        # Update regime
        self.current_regime = self._get_regime_transition()
        regime_params = self.regimes[self.current_regime]
        
        # Intraday pattern adjustment
        intraday_factor = self._generate_intraday_pattern(dt.hour, dt.minute)
        
        # Calculate price movement
        dt_fraction = interval_minutes / (365.25 * 24 * 60)  # Convert to years
        drift = regime_params['drift'] * dt_fraction
        volatility = regime_params['volatility'] * intraday_factor * np.sqrt(dt_fraction)
        
        # Generate random walk
        price_change = np.random.normal(drift, volatility)
        new_price = self.current_price * (1 + price_change)
        
        # Generate OHLC from base price movement
        high_factor = 1 + abs(np.random.normal(0, volatility/4))
        low_factor = 1 - abs(np.random.normal(0, volatility/4))
        
        open_price = self.current_price
        close_price = new_price
        high_price = max(open_price, close_price) * high_factor
        low_price = min(open_price, close_price) * low_factor
        
        # Generate volume (correlated with volatility)
        base_volume = 50000
        volume_factor = 1 + abs(price_change) * 10  # Higher volume on big moves
        volume = int(base_volume * volume_factor * np.random.uniform(0.5, 2.0))
        
        self.current_price = close_price
        
        return open_price, high_price, low_price, close_price, volume
    
    def generate_dataset(self, 
                        start_date: str = "2019-01-01",
                        end_date: str = "2024-01-01",
                        interval_minutes: int = 5,
                        output_file: Optional[str] = None,
                        chunk_size: int = 100000) -> pd.DataFrame:
        """
        Generate synthetic dataset for specified time range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval_minutes: Time interval in minutes (5 or 30)
            output_file: Optional output CSV file path
            chunk_size: Number of records per chunk for memory efficiency
            
        Returns:
            DataFrame with OHLCV data
        """
        
        print(f"Generating synthetic dataset: {start_date} to {end_date}, {interval_minutes}min intervals")
        
        start_time = time.time()
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Calculate total intervals
        total_minutes = int((end_dt - start_dt).total_seconds() / 60)
        total_intervals = total_minutes // interval_minutes
        
        print(f"Total intervals to generate: {total_intervals:,}")
        
        # Initialize data storage
        data_chunks = []
        current_chunk = []
        
        # Generate data
        current_dt = start_dt
        records_generated = 0
        
        while current_dt < end_dt:
            # Generate OHLCV for current interval
            open_price, high_price, low_price, close_price, volume = self._generate_price_move(current_dt, interval_minutes)
            
            # Create record
            record = {
                'Date': current_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': volume
            }
            
            current_chunk.append(record)
            records_generated += 1
            
            # Process chunk if full
            if len(current_chunk) >= chunk_size:
                chunk_df = pd.DataFrame(current_chunk)
                data_chunks.append(chunk_df)
                current_chunk = []
                
                # Memory management
                if len(data_chunks) >= 10:  # Write to disk periodically
                    if output_file:
                        combined_chunk = pd.concat(data_chunks, ignore_index=True)
                        if records_generated == len(combined_chunk):  # First write
                            combined_chunk.to_csv(output_file, index=False)
                        else:  # Append
                            combined_chunk.to_csv(output_file, mode='a', header=False, index=False)
                        data_chunks = []
            
            # Progress tracking
            if records_generated % 100000 == 0:
                progress = (records_generated / total_intervals) * 100
                elapsed = time.time() - start_time
                rate = records_generated / elapsed
                print(f"Progress: {progress:.1f}% ({records_generated:,} records) - Rate: {rate:.0f} records/sec")
            
            # Move to next interval
            current_dt += timedelta(minutes=interval_minutes)
        
        # Process final chunk
        if current_chunk:
            chunk_df = pd.DataFrame(current_chunk)
            data_chunks.append(chunk_df)
        
        # Combine all chunks
        if data_chunks:
            final_df = pd.concat(data_chunks, ignore_index=True)
            
            # Save final data
            if output_file:
                if os.path.exists(output_file):
                    final_df.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    final_df.to_csv(output_file, index=False)
        else:
            final_df = pd.DataFrame()
        
        # Calculate final statistics
        end_time = time.time()
        generation_time = end_time - start_time
        
        self.generation_stats.update({
            'total_records': records_generated,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'generation_time_seconds': generation_time,
            'records_per_second': records_generated / generation_time if generation_time > 0 else 0
        })
        
        print(f"\nGeneration complete!")
        print(f"Records generated: {records_generated:,}")
        print(f"Generation time: {generation_time:.1f} seconds")
        print(f"Rate: {self.generation_stats['records_per_second']:.0f} records/second")
        print(f"Memory usage: {self.generation_stats['memory_usage_mb']:.1f} MB")
        
        return final_df
    
    def generate_stress_test_dataset(self, 
                                   output_file: str,
                                   stress_scenarios: List[str] = None) -> Dict:
        """
        Generate specialized datasets for stress testing.
        
        Args:
            output_file: Output file path
            stress_scenarios: List of stress scenarios to include
            
        Returns:
            Dictionary with stress test statistics
        """
        
        if stress_scenarios is None:
            stress_scenarios = ['flash_crash', 'volatility_spike', 'trend_reversal', 'gap_events']
        
        print(f"Generating stress test dataset with scenarios: {stress_scenarios}")
        
        # Generate base dataset (1 year)
        base_df = self.generate_dataset(
            start_date="2023-01-01",
            end_date="2024-01-01",
            interval_minutes=5,
            output_file=None
        )
        
        stress_results = {}
        
        for scenario in stress_scenarios:
            print(f"Applying stress scenario: {scenario}")
            
            if scenario == 'flash_crash':
                # Simulate flash crash
                crash_indices = np.random.choice(len(base_df), size=10, replace=False)
                for idx in crash_indices:
                    base_df.iloc[idx:idx+12, base_df.columns.get_loc('Low')] *= 0.95
                    base_df.iloc[idx:idx+12, base_df.columns.get_loc('Close')] *= 0.97
                    base_df.iloc[idx:idx+12, base_df.columns.get_loc('Volume')] *= 5
            
            elif scenario == 'volatility_spike':
                # Simulate volatility spikes
                spike_indices = np.random.choice(len(base_df), size=50, replace=False)
                for idx in spike_indices:
                    volatility_factor = np.random.uniform(1.5, 3.0)
                    base_df.iloc[idx:idx+6, base_df.columns.get_loc('High')] *= volatility_factor
                    base_df.iloc[idx:idx+6, base_df.columns.get_loc('Low')] /= volatility_factor
                    base_df.iloc[idx:idx+6, base_df.columns.get_loc('Volume')] *= 2
            
            elif scenario == 'trend_reversal':
                # Simulate sharp trend reversals
                reversal_points = np.random.choice(len(base_df), size=5, replace=False)
                for idx in reversal_points:
                    trend_factor = np.random.choice([-1, 1]) * 0.1
                    for i in range(min(100, len(base_df) - idx)):
                        base_df.iloc[idx + i, base_df.columns.get_loc('Close')] *= (1 + trend_factor * i/100)
            
            elif scenario == 'gap_events':
                # Simulate gap events
                gap_indices = np.random.choice(len(base_df), size=20, replace=False)
                for idx in gap_indices:
                    if idx > 0:
                        prev_close = base_df.iloc[idx-1, base_df.columns.get_loc('Close')]
                        gap_factor = np.random.uniform(0.98, 1.02)
                        base_df.iloc[idx, base_df.columns.get_loc('Open')] = prev_close * gap_factor
                        base_df.iloc[idx, base_df.columns.get_loc('High')] = max(
                            base_df.iloc[idx, base_df.columns.get_loc('High')],
                            base_df.iloc[idx, base_df.columns.get_loc('Open')]
                        )
                        base_df.iloc[idx, base_df.columns.get_loc('Low')] = min(
                            base_df.iloc[idx, base_df.columns.get_loc('Low')],
                            base_df.iloc[idx, base_df.columns.get_loc('Open')]
                        )
        
        # Save stress test dataset
        base_df.to_csv(output_file, index=False)
        
        stress_results['file_path'] = output_file
        stress_results['total_records'] = len(base_df)
        stress_results['scenarios_applied'] = stress_scenarios
        stress_results['file_size_mb'] = os.path.getsize(output_file) / 1024 / 1024
        
        return stress_results

def main():
    """Main execution function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate synthetic datasets for performance validation')
    parser.add_argument('--start-date', default='2019-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=int, choices=[5, 30], default=5, help='Time interval in minutes')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--stress-test', action='store_true', help='Generate stress test dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)
    
    if args.stress_test:
        # Generate stress test dataset
        stress_results = generator.generate_stress_test_dataset(args.output)
        
        # Save stress test metadata
        metadata_file = args.output.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'stress_test_results': stress_results,
                'generation_stats': generator.generation_stats
            }, f, indent=2)
        
        print(f"Stress test dataset saved to: {args.output}")
        print(f"Metadata saved to: {metadata_file}")
    
    else:
        # Generate normal dataset
        df = generator.generate_dataset(
            start_date=args.start_date,
            end_date=args.end_date,
            interval_minutes=args.interval,
            output_file=args.output
        )
        
        # Save generation metadata
        metadata_file = args.output.replace('.csv', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'dataset_info': {
                    'start_date': args.start_date,
                    'end_date': args.end_date,
                    'interval_minutes': args.interval,
                    'total_records': len(df) if df is not None else 0,
                    'file_size_mb': os.path.getsize(args.output) / 1024 / 1024 if os.path.exists(args.output) else 0
                },
                'generation_stats': generator.generation_stats
            }, f, indent=2)
        
        print(f"Dataset saved to: {args.output}")
        print(f"Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    main()