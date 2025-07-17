#!/usr/bin/env python3
"""
🚨 RED TEAM EXTREME MARKET DATA ATTACK SUITE
Agent 3 Mission: Attack Strategic MARL with extreme and malformed market data

This module creates extreme market data scenarios designed to:
- Test numerical stability with NaN, infinity, and extreme values
- Challenge data validation and sanitization
- Expose edge cases in feature calculation
- Test system resilience to corrupted data streams
- Verify graceful degradation under data quality attacks

MISSION: Prove the system can handle real-world data corruption and extreme market conditions.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import math

class ExtremeDataGenerator:
    """
    Generates extreme and malformed market data for attacking Strategic MARL system.
    """
    
    def __init__(self, base_price: float = 15000.0):
        self.base_price = base_price
        self.output_dir = "adversarial_tests/extreme_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def attack_1_nan_values(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 ATTACK 1: NaN VALUES ATTACK
        
        Inject NaN values in various combinations to test data validation.
        """
        print("🚨 GENERATING NaN VALUES ATTACK...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        for i in range(bars):
            # Randomly inject NaN values
            nan_probability = 0.3  # 30% chance of NaN in any field
            
            open_price = self.base_price * (1 + np.random.normal(0, 0.01))
            close_price = open_price * (1 + np.random.normal(0, 0.02))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(500, 2000)
            
            # Strategic NaN injection patterns
            if np.random.random() < nan_probability:
                if i % 4 == 0:  # NaN in price data
                    open_price = np.nan
                elif i % 4 == 1:
                    high_price = np.nan
                elif i % 4 == 2:
                    low_price = np.nan
                elif i % 4 == 3:
                    close_price = np.nan
            
            if np.random.random() < 0.2:  # 20% chance of NaN volume
                volume = np.nan
            
            # Create complete NaN bars occasionally
            if i % 20 == 0 and i > 0:  # Every 20th bar is completely NaN
                open_price = close_price = high_price = low_price = volume = np.nan
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'attack_type': 'nan_values'
            })
        
        return pd.DataFrame(data)
    
    def attack_2_infinity_values(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 ATTACK 2: INFINITY VALUES ATTACK
        
        Inject positive and negative infinity values to test numerical handling.
        """
        print("🚨 GENERATING INFINITY VALUES ATTACK...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        for i in range(bars):
            # Base normal data
            open_price = self.base_price * (1 + np.random.normal(0, 0.01))
            close_price = open_price * (1 + np.random.normal(0, 0.02))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(500, 2000)
            
            # Strategic infinity injection
            if i % 10 == 0:  # Every 10th bar has infinity
                if i % 20 == 0:  # Positive infinity
                    open_price = np.inf
                    high_price = np.inf
                else:  # Negative infinity
                    low_price = -np.inf
                    close_price = -np.inf
            
            if i % 15 == 0:  # Infinite volume
                volume = np.inf
            
            # Mixed infinity patterns
            if i % 30 == 0:  # Complex infinity pattern
                open_price = np.inf
                close_price = -np.inf
                high_price = np.inf
                low_price = -np.inf
                volume = np.inf
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'attack_type': 'infinity_values'
            })
        
        return pd.DataFrame(data)
    
    def attack_3_zero_negative_values(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 ATTACK 3: ZERO AND NEGATIVE VALUES ATTACK
        
        Test system with impossible market data: negative prices, zero volume.
        """
        print("🚨 GENERATING ZERO/NEGATIVE VALUES ATTACK...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        for i in range(bars):
            # Start with normal data
            open_price = self.base_price * (1 + np.random.normal(0, 0.01))
            close_price = open_price * (1 + np.random.normal(0, 0.02))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(500, 2000)
            
            # Strategic impossible values
            if i % 8 == 0:  # Negative prices
                open_price = -abs(open_price)
                close_price = -abs(close_price)
            
            if i % 12 == 0:  # Zero prices
                if i % 24 == 0:
                    open_price = 0.0
                else:
                    close_price = 0.0
            
            if i % 15 == 0:  # Zero or negative volume
                if i % 30 == 0:
                    volume = 0
                else:
                    volume = -np.random.randint(100, 1000)
            
            # Impossible high/low relationships
            if i % 20 == 0:
                high_price = low_price - 1000  # High < Low (impossible)
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'attack_type': 'zero_negative'
            })
        
        return pd.DataFrame(data)
    
    def attack_4_extreme_volatility(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 ATTACK 4: EXTREME VOLATILITY ATTACK
        
        Create extreme price movements that could cause overflow/underflow.
        """
        print("🚨 GENERATING EXTREME VOLATILITY ATTACK...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        current_price = self.base_price
        
        for i in range(bars):
            # Extreme volatility patterns
            if i % 5 == 0:  # Extreme upward spike
                multiplier = np.random.uniform(10, 1000)  # 10x to 1000x price
                open_price = current_price
                close_price = current_price * multiplier
                high_price = close_price * 1.1
                low_price = open_price * 0.9
                
            elif i % 5 == 1:  # Extreme downward crash
                divisor = np.random.uniform(10, 1000)
                open_price = current_price
                close_price = current_price / divisor
                high_price = open_price * 1.1
                low_price = close_price * 0.9
                
            elif i % 5 == 2:  # Flash crash and recovery
                open_price = current_price
                close_price = current_price  # Ends where it started
                low_price = current_price / 100  # Flash crash to 1% of value
                high_price = current_price * 1.01
                
            elif i % 5 == 3:  # Massive range bar
                open_price = current_price
                close_price = current_price * np.random.uniform(0.5, 2.0)
                high_price = max(open_price, close_price) * np.random.uniform(5, 50)
                low_price = min(open_price, close_price) / np.random.uniform(5, 50)
                
            else:  # Normal-ish movement
                open_price = current_price
                close_price = current_price * (1 + np.random.normal(0, 0.1))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.05)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.05)))
            
            # Extreme volume
            if i % 3 == 0:
                volume = np.random.randint(1, 10) * 10**np.random.randint(6, 12)  # Up to trillions
            else:
                volume = np.random.randint(100, 1000)
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'attack_type': 'extreme_volatility'
            })
            
            current_price = close_price
        
        return pd.DataFrame(data)
    
    def attack_5_precision_attacks(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 ATTACK 5: PRECISION EDGE CASES ATTACK
        
        Test floating-point precision limits and subnormal numbers.
        """
        print("🚨 GENERATING PRECISION EDGE CASES ATTACK...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        for i in range(bars):
            # Different precision attacks based on bar
            if i % 10 == 0:  # Subnormal numbers
                open_price = 4.9406564584124654e-324  # Smallest positive subnormal
                close_price = 2.2250738585072009e-308  # Smallest positive normal
                high_price = close_price
                low_price = open_price
                volume = 1
                
            elif i % 10 == 1:  # Maximum float64 values
                open_price = 1.7976931348623157e+308  # Near max float64
                close_price = open_price * 0.999999
                high_price = 1.7976931348623157e+308
                low_price = close_price
                volume = int(9.223372036854775807e+18)  # Max int64
                
            elif i % 10 == 2:  # Precision loss scenarios
                base = 1e15
                open_price = base + 0.1  # Will lose precision
                close_price = base + 0.2
                high_price = base + 0.3
                low_price = base + 0.05
                volume = base + 1
                
            elif i % 10 == 3:  # Denormal arithmetic
                tiny1 = 1e-320
                tiny2 = 1e-319
                open_price = self.base_price + tiny1
                close_price = self.base_price + tiny2
                high_price = self.base_price + tiny2 * 2
                low_price = self.base_price + tiny1 / 2
                volume = 1000 + tiny1
                
            elif i % 10 == 4:  # Epsilon edge cases
                eps = np.finfo(float).eps  # Machine epsilon
                open_price = self.base_price
                close_price = self.base_price + eps
                high_price = self.base_price + eps * 2
                low_price = self.base_price - eps
                volume = 1000
                
            else:  # Normal precision but challenging calculations
                open_price = self.base_price * (1 + np.random.uniform(-1e-10, 1e-10))
                close_price = open_price * (1 + np.random.uniform(-1e-12, 1e-12))
                high_price = max(open_price, close_price) * (1 + 1e-15)
                low_price = min(open_price, close_price) * (1 - 1e-15)
                volume = np.random.randint(1, 10)
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'attack_type': 'precision_edge_cases'
            })
        
        return pd.DataFrame(data)
    
    def attack_6_corrupted_timestamps(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 ATTACK 6: CORRUPTED TIMESTAMPS ATTACK
        
        Test with malformed, duplicate, and out-of-order timestamps.
        """
        print("🚨 GENERATING CORRUPTED TIMESTAMPS ATTACK...")
        
        # Generate base data first
        base_dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        for i in range(bars):
            # Base market data
            open_price = self.base_price * (1 + np.random.normal(0, 0.01))
            close_price = open_price * (1 + np.random.normal(0, 0.02))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(500, 2000)
            
            # Corrupt timestamps based on pattern
            if i % 15 == 0:  # Duplicate timestamps
                date = base_dates[max(0, i-1)]
            elif i % 20 == 0:  # Out of order (future date)
                date = base_dates[i] + timedelta(days=365)
            elif i % 25 == 0:  # Out of order (past date)
                date = base_dates[0] - timedelta(days=10)
            elif i % 30 == 0:  # Invalid date
                try:
                    date = datetime(1900, 1, 1)  # Very old date
                except (ValueError, TypeError, AttributeError, KeyError) as e:
                    date = base_dates[i]
            elif i % 35 == 0:  # Massive future date
                date = datetime(2099, 12, 31)
            else:
                date = base_dates[i]
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'attack_type': 'corrupted_timestamps'
            })
        
        # Add some duplicate rows
        if len(data) > 10:
            data.append(data[5].copy())  # Exact duplicate
            data.append(data[10].copy())  # Another duplicate
        
        return pd.DataFrame(data)
    
    def attack_7_mixed_chaos(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 ATTACK 7: MIXED CHAOS ATTACK
        
        Combine multiple attack vectors for maximum chaos.
        """
        print("🚨 GENERATING MIXED CHAOS ATTACK...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        current_price = self.base_price
        
        for i in range(bars):
            # Random combination of attacks
            attack_vector = np.random.choice([
                'nan_inf', 'negative_zero', 'extreme_vol', 
                'precision', 'normal_chaos', 'impossible'
            ])
            
            if attack_vector == 'nan_inf':
                # Mix of NaN and infinity
                values = [current_price, np.nan, np.inf, -np.inf]
                open_price = np.random.choice(values)
                close_price = np.random.choice(values)
                high_price = np.random.choice(values)
                low_price = np.random.choice(values)
                volume = np.random.choice([1000, np.nan, np.inf, 0])
                
            elif attack_vector == 'negative_zero':
                # Negative or zero impossible values
                open_price = -abs(current_price) if np.random.random() < 0.5 else 0
                close_price = -abs(current_price) if np.random.random() < 0.5 else 0
                high_price = -abs(current_price) if np.random.random() < 0.5 else low_price - 1000
                low_price = 0
                volume = -np.random.randint(100, 1000)
                
            elif attack_vector == 'extreme_vol':
                # Extreme movements
                mult = np.random.uniform(0.001, 1000)
                open_price = current_price
                close_price = current_price * mult
                high_price = max(open_price, close_price) * np.random.uniform(1, 100)
                low_price = min(open_price, close_price) / np.random.uniform(1, 100)
                volume = np.random.randint(1, 10) * 10**np.random.randint(1, 15)
                
            elif attack_vector == 'precision':
                # Precision attacks
                eps = np.finfo(float).eps
                open_price = current_price + eps
                close_price = current_price + eps * 0.5
                high_price = current_price + eps * 2
                low_price = current_price - eps
                volume = 1e-300  # Extremely small
                
            elif attack_vector == 'impossible':
                # Impossible market conditions
                open_price = np.inf
                close_price = -np.inf
                high_price = np.nan
                low_price = np.inf  # High < Low
                volume = np.nan
                
            else:  # normal_chaos
                # Normal data with slight chaos
                open_price = current_price * (1 + np.random.normal(0, 0.5))
                close_price = open_price * (1 + np.random.normal(0, 0.5))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.3)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.3)))
                volume = np.random.randint(1, 10000)
            
            # Corrupted timestamp occasionally
            if np.random.random() < 0.1:
                date = dates[np.random.randint(0, len(dates))]  # Random timestamp
            else:
                date = dates[i]
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'attack_vector': attack_vector,
                'attack_type': 'mixed_chaos'
            })
            
            # Update current price (if valid)
            if not (np.isnan(close_price) or np.isinf(close_price) or close_price <= 0):
                current_price = close_price
        
        return pd.DataFrame(data)
    
    def generate_all_extreme_attacks(self) -> Dict[str, str]:
        """
        Generate all extreme data attack scenarios.
        """
        print("🚨" * 30)
        print("EXTREME MARKET DATA ATTACK SUITE")
        print("🚨" * 30)
        
        attacks = {
            'nan_values': self.attack_1_nan_values,
            'infinity_values': self.attack_2_infinity_values,
            'zero_negative': self.attack_3_zero_negative_values,
            'extreme_volatility': self.attack_4_extreme_volatility,
            'precision_edge_cases': self.attack_5_precision_attacks,
            'corrupted_timestamps': self.attack_6_corrupted_timestamps,
            'mixed_chaos': self.attack_7_mixed_chaos,
        }
        
        attack_files = {}
        
        for attack_name, attack_func in attacks.items():
            try:
                print(f"\n🎯 Executing {attack_name} attack...")
                data = attack_func()
                
                file_path = os.path.join(self.output_dir, f"extreme_{attack_name}.csv")
                data.to_csv(file_path, index=False)
                attack_files[attack_name] = file_path
                
                print(f"✅ {attack_name} attack saved to {file_path}")
                print(f"   📊 Bars: {len(data)}")
                
                # Check for problematic values
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_cols:
                    if col in data.columns:
                        nan_count = data[col].isna().sum()
                        inf_count = np.isinf(data[col]).sum()
                        neg_count = (data[col] < 0).sum() if data[col].dtype in ['float64', 'int64'] else 0
                        
                        if nan_count > 0 or inf_count > 0 or neg_count > 0:
                            print(f"   🚨 {col}: NaN={nan_count}, Inf={inf_count}, Negative={neg_count}")
                
            except Exception as e:
                print(f"❌ Failed to generate {attack_name}: {e}")
                attack_files[attack_name] = None
        
        return attack_files

def test_data_processing(data_file: str) -> Dict[str, Any]:
    """
    Test how well the system handles extreme data.
    """
    print(f"\n🔍 TESTING EXTREME DATA: {data_file}")
    
    results = {
        'file': data_file,
        'loadable': False,
        'has_nan': False,
        'has_inf': False,
        'has_negative_prices': False,
        'has_zero_volume': False,
        'data_shape': None,
        'error_message': None
    }
    
    try:
        # Load the data
        data = pd.read_csv(data_file)
        results['loadable'] = True
        results['data_shape'] = data.shape
        
        # Check for problematic values
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_cols:
            if col in data.columns:
                col_data = pd.to_numeric(data[col], errors='coerce')
                
                if col_data.isna().any():
                    results['has_nan'] = True
                
                if np.isinf(col_data).any():
                    results['has_inf'] = True
                
                if col in ['Open', 'High', 'Low', 'Close']:
                    if (col_data <= 0).any():
                        results['has_negative_prices'] = True
                
                if col == 'Volume':
                    if (col_data <= 0).any():
                        results['has_zero_volume'] = True
        
        print(f"✅ Data loaded successfully: {data.shape}")
        print(f"   🚨 NaN values: {results['has_nan']}")
        print(f"   🚨 Infinity values: {results['has_inf']}")
        print(f"   🚨 Negative prices: {results['has_negative_prices']}")
        print(f"   🚨 Zero/negative volume: {results['has_zero_volume']}")
        
    except Exception as e:
        results['error_message'] = str(e)
        print(f"❌ Failed to process data: {e}")
    
    return results

def run_extreme_data_attack_suite():
    """
    Execute the complete extreme data attack suite.
    """
    print("🚨" * 40)
    print("RED TEAM EXTREME DATA ATTACK EXECUTION")
    print("🚨" * 40)
    
    generator = ExtremeDataGenerator()
    attack_files = generator.generate_all_extreme_attacks()
    
    print("\n" + "="*80)
    print("🎯 TESTING SYSTEM RESPONSE TO EXTREME DATA")
    print("="*80)
    
    test_results = []
    
    for attack_name, file_path in attack_files.items():
        if file_path:
            result = test_data_processing(file_path)
            test_results.append(result)
    
    print("\n" + "="*80)
    print("📊 EXTREME DATA ATTACK RESULTS SUMMARY")
    print("="*80)
    
    for result in test_results:
        attack_name = os.path.basename(result['file']).replace('extreme_', '').replace('.csv', '')
        print(f"\n🎯 {attack_name.upper()}:")
        print(f"   📁 File: {result['file']}")
        print(f"   📊 Shape: {result['data_shape']}")
        print(f"   ✅ Loadable: {result['loadable']}")
        
        if result['loadable']:
            danger_level = sum([
                result['has_nan'],
                result['has_inf'], 
                result['has_negative_prices'],
                result['has_zero_volume']
            ])
            print(f"   🚨 Danger Level: {danger_level}/4")
        
        if result['error_message']:
            print(f"   💬 Error: {result['error_message']}")
    
    # Summary statistics
    total_attacks = len([r for r in test_results if r['loadable']])
    dangerous_data = len([r for r in test_results if r.get('has_nan') or r.get('has_inf')])
    
    print(f"\n📈 EXTREME DATA ATTACK SUMMARY:")
    print(f"   Total Attack Files: {len(test_results)}")
    print(f"   Successfully Generated: {total_attacks}")
    print(f"   Contains Dangerous Values: {dangerous_data}")
    print(f"   Attack Success Rate: {(dangerous_data/max(total_attacks,1))*100:.1f}%")
    
    return test_results

if __name__ == "__main__":
    run_extreme_data_attack_suite()