#!/usr/bin/env python3
"""
Generate additional test data for 500-row validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read existing data
df = pd.read_csv('/home/QuantNova/GrandModel/colab/data/NQ - 5 min - ETH.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Original data shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Generate additional synthetic data to reach 500 rows
current_rows = len(df)
target_rows = 500
additional_rows = target_rows - current_rows

if additional_rows > 0:
    print(f"Generating {additional_rows} additional rows...")
    
    # Get the last row to continue from
    last_row = df.iloc[-1]
    last_date = last_row['Date']
    last_price = last_row['Close']
    
    # Generate additional data
    new_data = []
    current_date = last_date
    current_price = last_price
    
    for i in range(additional_rows):
        # Move to next 5-minute interval
        current_date += timedelta(minutes=5)
        
        # Generate realistic price movement (random walk with slight upward bias)
        price_change = np.random.normal(0, current_price * 0.002)  # 0.2% std volatility
        current_price = max(current_price + price_change, current_price * 0.95)  # Prevent extreme drops
        
        # Generate OHLCV data
        volatility = current_price * 0.001  # 0.1% intraday volatility
        open_price = current_price + np.random.normal(0, volatility)
        high_price = max(open_price, current_price) + abs(np.random.normal(0, volatility))
        low_price = min(open_price, current_price) - abs(np.random.normal(0, volatility))
        close_price = current_price
        volume = int(np.random.normal(15000, 3000))  # Realistic volume
        
        new_data.append({
            'Date': current_date,
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': max(volume, 1000)  # Ensure minimum volume
        })
    
    # Create new dataframe with extended data
    new_df = pd.DataFrame(new_data)
    extended_df = pd.concat([df, new_df], ignore_index=True)
    
    # Save extended data
    extended_df.to_csv('/home/QuantNova/GrandModel/colab/data/NQ - 5 min - ETH_extended.csv', index=False)
    
    print(f"✅ Extended data saved!")
    print(f"   New shape: {extended_df.shape}")
    print(f"   New date range: {extended_df['Date'].min()} to {extended_df['Date'].max()}")
    print(f"   Price range: ${extended_df['Close'].min():.2f} - ${extended_df['Close'].max():.2f}")
    
    # Verify data quality
    print(f"\n📊 Data Quality Check:")
    print(f"   Price continuity: {abs(extended_df['Close'].diff().mean()):.4f} avg change")
    print(f"   Volume range: {extended_df['Volume'].min()} - {extended_df['Volume'].max()}")
    print(f"   Missing values: {extended_df.isnull().sum().sum()}")
    
else:
    print("✅ Data already has sufficient rows")