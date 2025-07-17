#!/usr/bin/env python3
"""
AGENT 2: BACKTEST DATA QUALITY ANALYSIS
Focused validation of the exact 291,373 bars used in backtesting (2021-2025 filtered data)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class BacktestDataQualityAnalyzer:
    def __init__(self, data_path="/home/QuantNova/GrandModel/data/historical/NQ - 5 min.csv"):
        self.data_path = data_path
        self.full_df = None
        self.backtest_df = None
        self.analysis_results = {}
        
    def load_and_filter_data(self):
        """Load data and apply the same filtering as the backtest system"""
        print("📊 Loading and filtering data exactly as backtest system...")
        
        # Load full dataset
        self.full_df = pd.read_csv(self.data_path)
        self.full_df['Timestamp'] = pd.to_datetime(self.full_df['Timestamp'], format='mixed', dayfirst=True)
        self.full_df = self.full_df.set_index('Timestamp')
        
        # Apply exact same filtering as backtest
        start_date = pd.to_datetime('2021-01-01')
        end_date = pd.to_datetime('2025-07-01')
        self.backtest_df = self.full_df[(self.full_df.index >= start_date) & (self.full_df.index <= end_date)]
        
        print(f"✅ Full dataset: {len(self.full_df):,} bars")
        print(f"✅ Backtest dataset: {len(self.backtest_df):,} bars")
        print(f"📅 Backtest period: {self.backtest_df.index.min()} to {self.backtest_df.index.max()}")
        
        # Verify this matches the claimed 291,373 bars
        expected_bars = 291373
        actual_bars = len(self.backtest_df)
        
        if actual_bars == expected_bars:
            print(f"✅ VERIFIED: Exactly {expected_bars:,} bars as claimed in backtest")
        else:
            print(f"⚠️  DISCREPANCY: Expected {expected_bars:,} bars, got {actual_bars:,}")
            print(f"   Difference: {actual_bars - expected_bars:,} bars")
        
        return actual_bars == expected_bars
    
    def analyze_data_quality(self):
        """Comprehensive quality analysis of the 291,373 backtest bars"""
        print(f"\n🔍 ANALYZING {len(self.backtest_df):,} BACKTEST BARS")
        print("="*60)
        
        # 1. COMPLETENESS ANALYSIS
        print("\n1️⃣ COMPLETENESS ANALYSIS")
        missing_counts = self.backtest_df.isnull().sum()
        total_missing = missing_counts.sum()
        completeness_pct = (1 - total_missing / (len(self.backtest_df) * len(self.backtest_df.columns))) * 100
        
        print(f"   Missing values per column:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"     {col}: {count:,} ({count/len(self.backtest_df)*100:.3f}%)")
        
        if total_missing == 0:
            print(f"   ✅ PERFECT: 0 missing values in {len(self.backtest_df):,} bars")
        else:
            print(f"   ⚠️  Missing: {total_missing:,} values ({100-completeness_pct:.3f}%)")
        
        # 2. TIMESTAMP INTEGRITY
        print("\n2️⃣ TIMESTAMP INTEGRITY")
        time_diffs = self.backtest_df.index.to_series().diff().dropna()
        expected_interval = pd.Timedelta(minutes=5)
        
        regular_intervals = (time_diffs == expected_interval).sum()
        irregular_intervals = len(time_diffs) - regular_intervals
        regular_pct = regular_intervals / len(time_diffs) * 100
        
        gaps = time_diffs[time_diffs > expected_interval]
        weekends_markets_closed = time_diffs[time_diffs > pd.Timedelta(hours=12)]
        
        print(f"   Regular 5-min intervals: {regular_intervals:,} ({regular_pct:.2f}%)")
        print(f"   Irregular intervals: {irregular_intervals:,}")
        print(f"   Data gaps > 5min: {len(gaps):,}")
        print(f"   Weekend/market closure gaps: {len(weekends_markets_closed):,}")
        
        # 3. PRICE DATA INTEGRITY
        print("\n3️⃣ PRICE DATA INTEGRITY")
        
        # OHLC relationship validation
        high_low_violations = (self.backtest_df['High'] < self.backtest_df['Low']).sum()
        open_violations = ((self.backtest_df['Open'] > self.backtest_df['High']) | 
                          (self.backtest_df['Open'] < self.backtest_df['Low'])).sum()
        close_violations = ((self.backtest_df['Close'] > self.backtest_df['High']) | 
                           (self.backtest_df['Close'] < self.backtest_df['Low'])).sum()
        
        print(f"   High < Low violations: {high_low_violations}")
        print(f"   Open outside H/L range: {open_violations}")
        print(f"   Close outside H/L range: {close_violations}")
        
        # Price statistics
        price_stats = {
            'min_price': float(self.backtest_df['Low'].min()),
            'max_price': float(self.backtest_df['High'].max()),
            'mean_price': float(self.backtest_df['Close'].mean()),
            'price_range': float(self.backtest_df['High'].max() - self.backtest_df['Low'].min())
        }
        
        print(f"   Price range: ${price_stats['min_price']:.2f} - ${price_stats['max_price']:.2f}")
        print(f"   Average close: ${price_stats['mean_price']:.2f}")
        
        # 4. VOLUME ANALYSIS
        print("\n4️⃣ VOLUME ANALYSIS")
        volume_stats = self.backtest_df['Volume'].describe()
        zero_volume = (self.backtest_df['Volume'] == 0).sum()
        negative_volume = (self.backtest_df['Volume'] < 0).sum()
        
        print(f"   Volume range: {volume_stats['min']:.0f} - {volume_stats['max']:.0f}")
        print(f"   Average volume: {volume_stats['mean']:.0f}")
        print(f"   Zero volume bars: {zero_volume:,}")
        print(f"   Negative volume bars: {negative_volume:,}")
        
        # 5. MARKET MICROSTRUCTURE ANALYSIS
        print("\n5️⃣ MARKET MICROSTRUCTURE ANALYSIS")
        
        # Calculate spreads (High - Low)
        spreads = self.backtest_df['High'] - self.backtest_df['Low']
        spread_stats = spreads.describe()
        
        # Price changes
        price_changes = self.backtest_df['Close'].pct_change().dropna()
        extreme_moves = abs(price_changes) > 0.05  # >5% moves
        
        print(f"   Average spread: ${spread_stats['mean']:.2f}")
        print(f"   Max spread: ${spread_stats['max']:.2f}")
        print(f"   Extreme moves (>5%): {extreme_moves.sum():,}")
        print(f"   Price volatility: {price_changes.std()*100:.3f}%")
        
        # 6. DATA CONSISTENCY CHECKS
        print("\n6️⃣ DATA CONSISTENCY CHECKS")
        
        # Check for duplicate timestamps
        duplicates = self.backtest_df.index.duplicated().sum()
        
        # Check for impossible prices (negative, etc.)
        negative_prices = ((self.backtest_df[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1)).sum()
        
        # Check for data freeze (consecutive identical prices)
        consecutive_same_close = (self.backtest_df['Close'] == self.backtest_df['Close'].shift(1)).sum()
        
        print(f"   Duplicate timestamps: {duplicates:,}")
        print(f"   Negative/zero prices: {negative_prices:,}")
        print(f"   Consecutive same close: {consecutive_same_close:,}")
        
        # 7. TRADING HOURS ANALYSIS
        print("\n7️⃣ TRADING HOURS ANALYSIS")
        
        # Analyze trading hours (NQ trades almost 24/5)
        self.backtest_df['hour'] = self.backtest_df.index.hour
        self.backtest_df['weekday'] = self.backtest_df.index.dayofweek
        
        # Count bars by hour
        hourly_counts = self.backtest_df['hour'].value_counts().sort_index()
        quiet_hours = hourly_counts[hourly_counts < hourly_counts.median() * 0.5]
        
        print(f"   24-hour coverage: {len(hourly_counts)} different hours")
        print(f"   Quiet hours (<50% median): {len(quiet_hours)} hours")
        print(f"   Weekend data: {(self.backtest_df['weekday'] >= 5).sum():,} bars")
        
        # Store results
        self.analysis_results = {
            'dataset_summary': {
                'total_bars': len(self.backtest_df),
                'date_range': {
                    'start': str(self.backtest_df.index.min()),
                    'end': str(self.backtest_df.index.max()),
                    'days': (self.backtest_df.index.max() - self.backtest_df.index.min()).days
                }
            },
            'completeness': {
                'missing_values': int(total_missing),
                'completeness_pct': float(completeness_pct)
            },
            'timestamp_quality': {
                'regular_intervals': int(regular_intervals),
                'regular_pct': float(regular_pct),
                'gaps_count': len(gaps)
            },
            'price_integrity': {
                'ohlc_violations': int(high_low_violations + open_violations + close_violations),
                'price_stats': price_stats
            },
            'volume_quality': {
                'zero_volume_bars': int(zero_volume),
                'negative_volume_bars': int(negative_volume),
                'avg_volume': float(volume_stats['mean'])
            },
            'microstructure': {
                'extreme_moves': int(extreme_moves.sum()),
                'avg_spread': float(spread_stats['mean']),
                'volatility_pct': float(price_changes.std()*100)
            },
            'consistency': {
                'duplicates': int(duplicates),
                'negative_prices': int(negative_prices),
                'consecutive_same': int(consecutive_same_close)
            }
        }
        
        return self.analysis_results
    
    def calculate_trustworthiness_score(self):
        """Calculate trustworthiness score for the 291,373 backtest bars"""
        print(f"\n🏆 TRUSTWORTHINESS CALCULATION FOR {len(self.backtest_df):,} BARS")
        print("="*60)
        
        scores = {}
        
        # Completeness (25 points)
        completeness = self.analysis_results['completeness']['completeness_pct']
        scores['completeness'] = min(25, completeness * 0.25)
        
        # Timestamp integrity (25 points)
        regular_pct = self.analysis_results['timestamp_quality']['regular_pct']
        scores['timestamps'] = min(25, regular_pct * 0.25)
        
        # Price integrity (25 points)
        violations = self.analysis_results['price_integrity']['ohlc_violations']
        violation_rate = violations / len(self.backtest_df)
        scores['price_integrity'] = max(0, 25 - (violation_rate * 10000))
        
        # Consistency (15 points)
        duplicates = self.analysis_results['consistency']['duplicates']
        negative_prices = self.analysis_results['consistency']['negative_prices']
        total_issues = duplicates + negative_prices
        issue_rate = total_issues / len(self.backtest_df)
        scores['consistency'] = max(0, 15 - (issue_rate * 5000))
        
        # Volume quality (10 points)
        zero_vol_rate = self.analysis_results['volume_quality']['zero_volume_bars'] / len(self.backtest_df)
        scores['volume'] = max(0, 10 - (zero_vol_rate * 200))
        
        total_score = sum(scores.values())
        
        print(f"\n📊 SCORING BREAKDOWN:")
        print(f"   Completeness:     {scores['completeness']:.1f}/25")
        print(f"   Timestamps:       {scores['timestamps']:.1f}/25")
        print(f"   Price Integrity:  {scores['price_integrity']:.1f}/25") 
        print(f"   Consistency:      {scores['consistency']:.1f}/15")
        print(f"   Volume Quality:   {scores['volume']:.1f}/10")
        print(f"   " + "="*30)
        print(f"   TOTAL SCORE:      {total_score:.1f}/100")
        
        # Grade assignment
        if total_score >= 95: grade = "A+"
        elif total_score >= 90: grade = "A"
        elif total_score >= 85: grade = "A-"
        elif total_score >= 80: grade = "B+"
        elif total_score >= 75: grade = "B"
        elif total_score >= 70: grade = "B-"
        else: grade = "C or below"
        
        print(f"   GRADE:            {grade}")
        
        return total_score, grade
    
    def generate_final_report(self):
        """Generate final comprehensive report"""
        print(f"\n{'='*80}")
        print("📋 FINAL DATA QUALITY REPORT - 291,373 BACKTEST BARS")
        print(f"{'='*80}")
        
        score, grade = self.calculate_trustworthiness_score()
        
        print(f"\n🎯 EXECUTIVE SUMMARY:")
        print(f"   • Dataset: {len(self.backtest_df):,} NQ futures 5-minute bars")
        print(f"   • Period: {self.backtest_df.index.min().strftime('%Y-%m-%d')} to {self.backtest_df.index.max().strftime('%Y-%m-%d')}")
        print(f"   • Data Quality Score: {score:.1f}/100 ({grade})")
        
        # Key findings
        print(f"\n✅ STRENGTHS:")
        if self.analysis_results['completeness']['missing_values'] == 0:
            print(f"   • Perfect completeness - no missing values")
        if self.analysis_results['price_integrity']['ohlc_violations'] == 0:
            print(f"   • Perfect OHLC integrity - all price relationships valid")
        if self.analysis_results['timestamp_quality']['regular_pct'] > 98:
            print(f"   • Excellent timestamp regularity ({self.analysis_results['timestamp_quality']['regular_pct']:.1f}%)")
        if self.analysis_results['consistency']['duplicates'] == 0:
            print(f"   • No duplicate timestamps")
        
        # Areas for attention (if any)
        issues = []
        if self.analysis_results['timestamp_quality']['gaps_count'] > 0:
            issues.append(f"Data gaps present: {self.analysis_results['timestamp_quality']['gaps_count']:,}")
        if self.analysis_results['volume_quality']['zero_volume_bars'] > 0:
            issues.append(f"Zero volume bars: {self.analysis_results['volume_quality']['zero_volume_bars']:,}")
        if self.analysis_results['microstructure']['extreme_moves'] > 0:
            issues.append(f"Extreme price moves: {self.analysis_results['microstructure']['extreme_moves']:,}")
        
        if issues:
            print(f"\n⚠️  AREAS FOR ATTENTION:")
            for issue in issues:
                print(f"   • {issue}")
        
        print(f"\n🏅 FINAL ASSESSMENT:")
        if score >= 90:
            assessment = "EXCELLENT - Data is of institutional quality and suitable for production trading"
        elif score >= 80:
            assessment = "GOOD - Data quality is solid with minor issues that don't affect reliability"
        elif score >= 70:
            assessment = "ACCEPTABLE - Data is usable but may require additional preprocessing"
        else:
            assessment = "NEEDS IMPROVEMENT - Data quality issues may affect backtest reliability"
        
        print(f"   {assessment}")
        
        # Save detailed report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"/home/QuantNova/GrandModel/results/nq_backtest/backtest_data_quality_report_{timestamp}.json"
        
        final_report = {
            'metadata': {
                'analysis_timestamp': timestamp,
                'dataset_bars': len(self.backtest_df),
                'analysis_type': 'Backtest Data Quality Validation'
            },
            'quality_score': score,
            'grade': grade,
            'assessment': assessment,
            'detailed_results': self.analysis_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved: {report_path}")
        
        return final_report

def main():
    """Main execution"""
    print("🚀 AGENT 2: BACKTEST DATA QUALITY ANALYSIS")
    print("🎯 Validating the exact 291,373 bars used in backtesting")
    print("="*80)
    
    analyzer = BacktestDataQualityAnalyzer()
    
    # Load and filter data exactly as backtest does
    data_verified = analyzer.load_and_filter_data()
    
    if not data_verified:
        print("❌ Data verification failed - bar count mismatch")
        return
    
    # Perform comprehensive analysis
    analyzer.analyze_data_quality()
    
    # Generate final report
    report = analyzer.generate_final_report()
    
    return analyzer, report

if __name__ == "__main__":
    analyzer, report = main()