# Performance Validation Report

Generated: 2025-07-15 16:36:13

## System Information

- **cpu_count**: 2
- **cpu_freq_mhz**: 2445.432
- **memory_total_gb**: 7.757411956787109
- **python_version**: 3.12.3 (main, Jun 18 2025, 17:59:45) [GCC 13.3.0]
- **platform**: linux
- **timestamp**: 2025-07-15T16:36:02.547780

## Benchmark Results

### data_loading

- **Success**: True
- **Dataset Size**: 525,888 records
- **Total Time**: 2.03 seconds
- **Peak Memory**: 5188.8 MB
- **Average CPU**: 77.4%
- **Throughput**: 258866 records/second

### training_simulation

- **Success**: True
- **Dataset Size**: 10,000 records
- **Total Time**: 1.04 seconds
- **Peak Memory**: 5189.0 MB
- **Average CPU**: 35.2%
- **Throughput**: 96134 records/second

### training_simulation

- **Success**: True
- **Dataset Size**: 50,000 records
- **Total Time**: 1.22 seconds
- **Peak Memory**: 5221.8 MB
- **Average CPU**: 55.9%
- **Throughput**: 411177 records/second

### training_simulation

- **Success**: True
- **Dataset Size**: 100,000 records
- **Total Time**: 1.23 seconds
- **Peak Memory**: 5223.9 MB
- **Average CPU**: 49.9%
- **Throughput**: 814476 records/second

### training_simulation

- **Success**: True
- **Dataset Size**: 500,000 records
- **Total Time**: 1.97 seconds
- **Peak Memory**: 5222.5 MB
- **Average CPU**: 41.8%
- **Throughput**: 2537887 records/second

## Bottleneck Analysis

### Memory Issues

- No issues detected

### Cpu Issues

- No issues detected

### Throughput Issues

- No issues detected

### Stability Issues

- No issues detected

## Scaling Projections

### Time Complexity

- **Slope**: 0.175
- **R²**: 0.901
- **Complexity Class**: Sub-linear (better than O(n))

### 5-Year Dataset Projections

#### 5min_1year

- **Dataset Size**: 105,120 records
- **Projected Time**: 0.00 hours
- **Projected Memory**: 5.09 GB

#### 5min_5years

- **Dataset Size**: 525,600 records
- **Projected Time**: 0.00 hours
- **Projected Memory**: 5.08 GB

#### 30min_1year

- **Dataset Size**: 17,520 records
- **Projected Time**: 0.00 hours
- **Projected Memory**: 5.09 GB

#### 30min_5years

- **Dataset Size**: 87,600 records
- **Projected Time**: 0.00 hours
- **Projected Memory**: 5.09 GB

## Recommendations

**General Recommendations:**
- Regular performance monitoring in production
- Implement auto-scaling based on workload
- Use performance profiling tools for optimization
- Consider cloud-based solutions for scalability