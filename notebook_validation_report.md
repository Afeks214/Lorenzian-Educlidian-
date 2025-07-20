# 📋 NOTEBOOK CELL FLOW VALIDATION REPORT
============================================================
🕐 Validation Date: 2025-07-20 06:40:50
🎯 Notebooks Tested: 5

## 📊 SUMMARY STATISTICS
------------------------------
✅ Successful Notebooks: 1/5 (20.0%)
📱 Total Cells: 119
✅ Successful Cells: 16/119 (13.4%)
⏱️ Total Execution Time: 22.99s

## 📓 INDIVIDUAL NOTEBOOK RESULTS
----------------------------------------
### risk_management_mappo_training.ipynb
**Status**: ❌ FAIL (7.1% success rate)
**Cells**: 1/25 successful
**Execution Time**: 13.02s
**Memory Peak**: 466.6MB
**Critical Issues**:
  - Cell 2: Command '['/usr/bin/python3', '-m', 'pip', 'install', 'gymnasium']' returned non-zero exit status 1.
  - Cell 3: No module named 'dask'
  - Cell 5: name 'DATA_DIR' is not defined
  - Cell 7: name 'jit' is not defined
  - Cell 9: name 'nn' is not defined
  - Cell 11: name 'nn' is not defined
  - Cell 13: name 'perf_monitor' is not defined
  - Cell 15: name 'perf_monitor' is not defined
  - Cell 17: name 'RiskEnvironment' is not defined
  - Cell 18: name 'env' is not defined
  - Cell 20: name 'data_loader' is not defined
  - Cell 21: name 'data_loader' is not defined
  - Cell 23: name 'trainer' is not defined
**Cell Details**:
  ❌ Cell 2: Command '['/usr/bin/python3', '-m', 'pip', 'install', 'gymnasium']' returned non-zero exit status 1....
  ❌ Cell 3: No module named 'dask'...
  ❌ Cell 5: name 'DATA_DIR' is not defined...
  ❌ Cell 7: name 'jit' is not defined...
  ❌ Cell 9: name 'nn' is not defined...
  ❌ Cell 11: name 'nn' is not defined...
  ❌ Cell 13: name 'perf_monitor' is not defined...
  ❌ Cell 15: name 'perf_monitor' is not defined...
  ❌ Cell 17: name 'RiskEnvironment' is not defined...
  ❌ Cell 18: name 'env' is not defined...
  ❌ Cell 20: name 'data_loader' is not defined...
  ❌ Cell 21: name 'data_loader' is not defined...
  ❌ Cell 23: name 'trainer' is not defined...

### execution_engine_mappo_training.ipynb
**Status**: ❌ FAIL (9.1% success rate)
**Cells**: 1/24 successful
**Execution Time**: 0.02s
**Memory Peak**: 467.4MB
**Critical Issues**:
  - Cell 4: list() takes no keyword arguments
  - Cell 6: name 'nn' is not defined
  - Cell 8: name 'dataclass' is not defined
  - Cell 10: name 'Tuple' is not defined
  - Cell 12: name 'MassiveDatasetLoader' is not defined
  - Cell 14: name 'List' is not defined
  - Cell 16: name 'env' is not defined
  - Cell 18: name 'training_summary' is not defined
  - Cell 20: name 'torch' is not defined
  - Cell 22: name 'total_avg_latency' is not defined
**Cell Details**:
  ❌ Cell 4: list() takes no keyword arguments...
  ❌ Cell 6: name 'nn' is not defined...
  ❌ Cell 8: name 'dataclass' is not defined...
  ❌ Cell 10: name 'Tuple' is not defined...
  ❌ Cell 12: name 'MassiveDatasetLoader' is not defined...
  ❌ Cell 14: name 'List' is not defined...
  ❌ Cell 16: name 'env' is not defined...
  ❌ Cell 18: name 'training_summary' is not defined...
  ❌ Cell 20: name 'torch' is not defined...
  ❌ Cell 22: name 'total_avg_latency' is not defined...

### strategic_mappo_training.ipynb
**Status**: ✅ PASS (83.3% success rate)
**Cells**: 5/12 successful
**Execution Time**: 0.14s
**Memory Peak**: 469.2MB
**Critical Issues**:
  - Cell 11: name 'calculate_optimal_batch_size' is not defined

### tactical_mappo_training.ipynb
**Status**: ❌ FAIL (36.8% success rate)
**Cells**: 7/32 successful
**Execution Time**: 9.68s
**Memory Peak**: 549.9MB
**Critical Issues**:
  - Cell 12: name 'torch' is not defined
  - Cell 14: name 'create_large_dataset_simulation' is not defined
  - Cell 15: name 'df' is not defined
  - Cell 17: name 'gpu_optimizer' is not defined
  - Cell 19: name 'df' is not defined
  - Cell 20: name 'df' is not defined
  - Cell 21: name 'df' is not defined
  - Cell 23: name 'trainer' is not defined
  - Cell 24: name 'trainer' is not defined
  - Cell 26: name 'trainer' is not defined
  - Cell 28: name 'trainer' is not defined
  - Cell 30: name 'trainer' is not defined
**Cell Details**:
  ❌ Cell 12: name 'torch' is not defined...
  ❌ Cell 14: name 'create_large_dataset_simulation' is not defined...
  ❌ Cell 15: name 'df' is not defined...
  ❌ Cell 17: name 'gpu_optimizer' is not defined...
  ❌ Cell 19: name 'df' is not defined...
  ❌ Cell 20: name 'df' is not defined...
  ❌ Cell 21: name 'df' is not defined...
  ❌ Cell 23: name 'trainer' is not defined...
  ❌ Cell 24: name 'trainer' is not defined...
  ❌ Cell 26: name 'trainer' is not defined...
  ❌ Cell 28: name 'trainer' is not defined...
  ❌ Cell 30: name 'trainer' is not defined...

### xai_trading_explanations_training.ipynb
**Status**: ❌ FAIL (16.7% success rate)
**Cells**: 2/26 successful
**Execution Time**: 0.14s
**Memory Peak**: 554.8MB
**Critical Issues**:
  - Cell 2: invalid syntax (<string>, line 67)
  - Cell 4: unexpected character after line continuation character (<string>, line 321)
  - Cell 6: invalid syntax (<string>, line 1)
  - Cell 11: No module named 'nltk'
  - Cell 13: name 'dataclass' is not defined
  - Cell 15: name 'dataclass' is not defined
  - Cell 17: name 'TradingDecision' is not defined
  - Cell 19: name 'nn' is not defined
  - Cell 21: name 'Dict' is not defined
  - Cell 23: name 'performance_monitor' is not defined
**Cell Details**:
  ❌ Cell 2: invalid syntax (<string>, line 67)...
  ❌ Cell 4: unexpected character after line continuation character (<string>, line 321)...
  ❌ Cell 6: invalid syntax (<string>, line 1)...
  ❌ Cell 11: No module named 'nltk'...
  ❌ Cell 13: name 'dataclass' is not defined...
  ❌ Cell 15: name 'dataclass' is not defined...
  ❌ Cell 17: name 'TradingDecision' is not defined...
  ❌ Cell 19: name 'nn' is not defined...
  ❌ Cell 21: name 'Dict' is not defined...
  ❌ Cell 23: name 'performance_monitor' is not defined...

## 🔧 RECOMMENDATIONS
--------------------
⚠️ The following notebooks need attention:
  - risk_management_mappo_training.ipynb: Cell 2: Command '['/usr/bin/python3', '-m', 'pip', 'install', 'gymnasium']' returned non-zero exit status 1.
  - risk_management_mappo_training.ipynb: Cell 3: No module named 'dask'
  - risk_management_mappo_training.ipynb: Cell 5: name 'DATA_DIR' is not defined
  - execution_engine_mappo_training.ipynb: Cell 4: list() takes no keyword arguments
  - execution_engine_mappo_training.ipynb: Cell 6: name 'nn' is not defined
  - execution_engine_mappo_training.ipynb: Cell 8: name 'dataclass' is not defined
  - tactical_mappo_training.ipynb: Cell 12: name 'torch' is not defined
  - tactical_mappo_training.ipynb: Cell 14: name 'create_large_dataset_simulation' is not defined
  - tactical_mappo_training.ipynb: Cell 15: name 'df' is not defined
  - xai_trading_explanations_training.ipynb: Cell 2: invalid syntax (<string>, line 67)
  - xai_trading_explanations_training.ipynb: Cell 4: unexpected character after line continuation character (<string>, line 321)
  - xai_trading_explanations_training.ipynb: Cell 6: invalid syntax (<string>, line 1)

## 🎯 VALIDATION TARGETS
-------------------------
- **Minimum Success Rate**: 80% of cells must execute successfully
- **Memory Usage**: < 2GB peak memory per notebook
- **Execution Time**: < 300s total per notebook
- **Critical Errors**: No import/module errors