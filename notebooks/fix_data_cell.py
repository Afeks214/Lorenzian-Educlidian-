import json

# Read the notebook
with open('Synergy_1_MLMI_FVG_NWRQK.ipynb', 'r') as f:
    nb = json.load(f)

# Create proper data loading cell
data_loading_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        "# Cell 2a: Optimized Data Loading with Configuration\n",
        "\n",
        "# Import data loading functions\n",
        "from data_loader import load_data_optimized, validate_dataframe\n",
        "\n",
        "# Load data files with error handling\n",
        "print(\"Loading data files using configuration...\")\n",
        "print(f\"5m data path: {config.data_5m_path}\")\n",
        "print(f\"30m data path: {config.data_30m_path}\")\n",
        "\n",
        "try:\n",
        "    # Load 5-minute data\n",
        "    print(\"\\nLoading 5-minute data...\")\n",
        "    df_5m = load_data_optimized(config.data_5m_path, '5m')\n",
        "    \n",
        "    # Load 30-minute data\n",
        "    print(\"\\nLoading 30-minute data...\")\n",
        "    df_30m = load_data_optimized(config.data_30m_path, '30m')\n",
        "    \n",
        "    # Verify time alignment\n",
        "    print(\"\\nVerifying time alignment...\")\n",
        "    \n",
        "    # Find overlapping period\n",
        "    start_time = max(df_5m.index[0], df_30m.index[0])\n",
        "    end_time = min(df_5m.index[-1], df_30m.index[-1])\n",
        "    \n",
        "    if start_time >= end_time:\n",
        "        raise ValueError(\"No overlapping time period between 5m and 30m data\")\n",
        "    \n",
        "    # Trim dataframes to overlapping period\n",
        "    df_5m = df_5m[start_time:end_time]\n",
        "    df_30m = df_30m[start_time:end_time]\n",
        "    \n",
        "    print(f\"\\nAligned data period: {start_time} to {end_time}\")\n",
        "    print(f\"5-minute bars after alignment: {len(df_5m):,}\")\n",
        "    print(f\"30-minute bars after alignment: {len(df_30m):,}\")\n",
        "    \n",
        "    # Verify reasonable ratio\n",
        "    ratio = len(df_5m) / len(df_30m)\n",
        "    expected_ratio = 6  # 30min / 5min\n",
        "    if abs(ratio - expected_ratio) > 1:\n",
        "        print(f\"Warning: Unexpected timeframe ratio: {ratio:.2f} (expected ~{expected_ratio})\")\n",
        "    \n",
        "    print(f\"\\n5-minute data: {df_5m.index[0]} to {df_5m.index[-1]}\")\n",
        "    print(f\"30-minute data: {df_30m.index[0]} to {df_30m.index[-1]}\")\n",
        "    \n",
        "    # Final validation\n",
        "    print(\"\\nData loading completed successfully!\")\n",
        "    \n",
        "except Exception as e:\n",
        "    print(f\"\\nFatal error during data loading: {str(e)}\")\n",
        "    print(\"Cannot proceed with analysis. Please check your data files.\")\n",
        "    raise"
    ]
}

# Replace the problematic cell
if len(nb['cells']) > 2:
    nb['cells'][2] = data_loading_cell

# Save the fixed notebook
with open('Synergy_1_MLMI_FVG_NWRQK.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
    
print('Data loading cell properly fixed!')