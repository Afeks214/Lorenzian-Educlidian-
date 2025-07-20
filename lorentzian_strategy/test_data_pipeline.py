"""
Test Data Pipeline for Lorentzian Trading Strategy
Validates data loading, feature extraction, and quality reporting.
"""

import sys
import os
sys.path.append('/home/QuantNova/GrandModel')

import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import our modules
from lorentzian_strategy import initialize_package, DataLoader, create_feature_extractor
from lorentzian_strategy.config.config import get_config


def test_data_pipeline():
    """Test the complete data pipeline"""
    print("=" * 60)
    print("LORENTZIAN STRATEGY DATA PIPELINE TEST")
    print("=" * 60)
    
    # Initialize package
    print("\n1. Initializing package...")
    try:
        config = initialize_package(environment="development")
        print("✓ Package initialized successfully")
        print(f"   - Data source: {config.data.source_file}")
        print(f"   - Cache enabled: {config.data.cache_enabled}")
        print(f"   - Optimization: Numba={config.optimization.use_numba}")
    except Exception as e:
        print(f"✗ Package initialization failed: {e}")
        return False
    
    # Test data loading
    print("\n2. Testing data loading...")
    try:
        loader = DataLoader(config)
        df, quality_report = loader.load_and_process_data(use_cache=False)
        
        print("✓ Data loaded successfully")
        print(f"   - Records: {len(df):,}")
        print(f"   - Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        print(f"   - Quality score: {quality_report.quality_score:.1f}%")
        
        if quality_report.issues:
            print(f"   - Issues found: {len(quality_report.issues)}")
            for issue in quality_report.issues[:3]:  # Show first 3 issues
                print(f"     * {issue}")
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    # Test feature extraction
    print("\n3. Testing feature extraction...")
    try:
        feature_extractor = create_feature_extractor(config)
        
        # Extract features for a subset of data
        test_subset = df.iloc[-1000:].copy()  # Last 1000 records
        feature_matrix, timestamps = feature_extractor.extract_feature_matrix(test_subset)
        
        print("✓ Feature extraction successful")
        print(f"   - Feature matrix shape: {feature_matrix.shape}")
        print(f"   - Feature names: {feature_extractor.get_feature_names()}")
        print(f"   - Sample features (last row): {feature_matrix[-1][:5]}")
        
        # Check for NaN or infinite values
        nan_count = np.isnan(feature_matrix).sum()
        inf_count = np.isinf(feature_matrix).sum()
        print(f"   - NaN values: {nan_count}")
        print(f"   - Infinite values: {inf_count}")
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False
    
    # Test data quality report
    print("\n4. Testing data quality reporting...")
    try:
        # Save quality report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/home/QuantNova/GrandModel/lorentzian_strategy/data/validation/test_quality_report_{timestamp}.json"
        loader.save_quality_report(quality_report, report_path)
        
        print("✓ Quality report saved successfully")
        print(f"   - Report path: {report_path}")
        print(f"   - OHLC violations: {sum(quality_report.ohlc_violations.values())}")
        print(f"   - Zero volume records: {quality_report.zero_volume_records}")
        print(f"   - Duplicates: {quality_report.duplicates}")
        
    except Exception as e:
        print(f"✗ Quality reporting failed: {e}")
        return False
    
    # Performance test
    print("\n5. Testing performance...")
    try:
        import time
        
        # Time data loading
        start_time = time.time()
        df_cached, _ = loader.load_and_process_data(use_cache=True)
        load_time = time.time() - start_time
        
        # Time feature extraction
        start_time = time.time()
        sample_features = feature_extractor.extract_features(df.iloc[-200:])
        feature_time = time.time() - start_time
        
        print("✓ Performance test completed")
        print(f"   - Data loading time: {load_time:.3f}s")
        print(f"   - Feature extraction time: {feature_time:.3f}s")
        print(f"   - Features per second: {1/feature_time:.1f}")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False
    
    # Test configuration validation
    print("\n6. Testing configuration validation...")
    try:
        config.validate()
        print("✓ Configuration validation passed")
        
        # Test different environments
        test_configs = ["development", "production", "testing"]
        for env in test_configs:
            test_config = get_config(environment=env)
            test_config.validate()
            print(f"   - {env} environment: ✓")
            
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    
    # Summary statistics
    print(f"\nSUMMARY:")
    print(f"- Total records processed: {len(df):,}")
    print(f"- Date range: {(df['Timestamp'].max() - df['Timestamp'].min()).days} days")
    print(f"- Data quality score: {quality_report.quality_score:.1f}%")
    print(f"- Features extracted: {len(feature_extractor.get_feature_names())}")
    print(f"- Feature matrix shape: {feature_matrix.shape}")
    print(f"- Memory efficiency: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    return True


def detailed_data_analysis():
    """Perform detailed analysis of the NQ data"""
    print("\n" + "=" * 60)
    print("DETAILED NQ DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    config = get_config()
    loader = DataLoader(config)
    df, quality_report = loader.load_and_process_data()
    
    # Detailed statistics
    print(f"\nData Overview:")
    print(f"- Total records: {len(df):,}")
    print(f"- Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"- Duration: {(df['Timestamp'].max() - df['Timestamp'].min()).days} days")
    
    # Price statistics
    print(f"\nPrice Analysis:")
    print(f"- Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    print(f"- Current price: ${df['Close'].iloc[-1]:.2f}")
    print(f"- Average daily range: ${((df['High'] - df['Low']) / df['Close'] * 100).mean():.2f}%")
    
    # Volume analysis
    print(f"\nVolume Analysis:")
    print(f"- Total volume: {df['Volume'].sum():,}")
    print(f"- Average volume per bar: {df['Volume'].mean():.0f}")
    print(f"- Max volume: {df['Volume'].max():,}")
    
    # Time gaps analysis
    time_diff = df['Timestamp'].diff()
    print(f"\nTime Gaps Analysis:")
    print(f"- Normal 30min intervals: {(time_diff == pd.Timedelta(minutes=30)).sum():,}")
    print(f"- 1-hour gaps: {(time_diff == pd.Timedelta(hours=1)).sum():,}")
    print(f"- Weekend gaps (>1 day): {(time_diff > pd.Timedelta(days=1)).sum():,}")
    
    # Quality issues
    print(f"\nData Quality Issues:")
    for issue in quality_report.issues:
        print(f"- {issue}")
    
    if quality_report.recommendations:
        print(f"\nRecommendations:")
        for rec in quality_report.recommendations:
            print(f"- {rec}")


if __name__ == "__main__":
    # Run tests
    success = test_data_pipeline()
    
    if success:
        # Run detailed analysis
        detailed_data_analysis()
    else:
        print("\nTests failed. Please check the error messages above.")
        sys.exit(1)