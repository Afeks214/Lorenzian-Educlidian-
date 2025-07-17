# Strategic MARL Component <50ms Latency Benchmark Suite

This comprehensive benchmark suite validates the <50ms latency target for `strategic_marl_component.py` with detailed performance analytics, regression detection, and continuous monitoring capabilities.

## Features

### 🚀 Core Benchmarking
- **End-to-end inference latency measurement** with P50/P95/P99 statistics
- **Memory usage tracking** during inference operations
- **CPU and GPU monitoring** with resource utilization analysis
- **Load testing** with concurrent request scenarios
- **Mathematical validation** of latency targets

### 📊 Performance Analytics
- **Automated performance regression detection** against historical baselines
- **Continuous benchmarking integration** for CI/CD pipelines
- **Performance trend analysis** and alerting
- **Comprehensive reporting** with actionable recommendations

### 🔍 Advanced Monitoring
- **Real-time performance monitoring** during test execution
- **Line-by-line profiling** for detailed bottleneck analysis
- **Memory leak detection** with allocation tracking
- **Concurrency stress testing** with configurable load patterns

## Quick Start

### 1. Basic Usage

```python
from tests.performance.test_strategic_marl_latency_benchmark import (
    StrategicMARLBenchmark, 
    LatencyBenchmarkConfig
)

# Configure benchmark
config = LatencyBenchmarkConfig(
    target_latency_ms=50.0,
    p95_latency_ms=75.0,
    p99_latency_ms=100.0,
    benchmark_iterations=100
)

# Create benchmark suite
benchmark = StrategicMARLBenchmark(config)

# Run single inference benchmark
result = await benchmark.run_single_inference_benchmark()

print(f"P50 Latency: {result.metrics.p50_latency_ms:.2f}ms")
print(f"P95 Latency: {result.metrics.p95_latency_ms:.2f}ms")
print(f"Passed: {result.passed}")
```

### 2. Running Tests

```bash
# Run all benchmark tests
python -m pytest tests/performance/test_strategic_marl_latency_benchmark.py -v

# Run specific test
python -m pytest tests/performance/test_strategic_marl_latency_benchmark.py::TestStrategicMARLLatencyBenchmark::test_single_inference_latency_target -v

# Run with performance markers
python -m pytest tests/performance/test_strategic_marl_latency_benchmark.py -m "performance" -v
```

### 3. Demo Script

```bash
# Run comprehensive demo
python tests/performance/demo_strategic_marl_benchmark.py

# Run standalone benchmark
python tests/performance/test_strategic_marl_latency_benchmark.py
```

## Configuration

### LatencyBenchmarkConfig

```python
@dataclass
class LatencyBenchmarkConfig:
    target_latency_ms: float = 50.0           # Main latency target
    p95_latency_ms: float = 75.0              # P95 latency target
    p99_latency_ms: float = 100.0             # P99 latency target
    min_throughput_ops_per_sec: float = 100.0 # Minimum throughput
    max_memory_increase_mb: float = 50.0      # Memory usage limit
    max_cpu_usage_percent: float = 80.0       # CPU usage limit
    max_gpu_memory_mb: float = 256.0          # GPU memory limit
    warmup_iterations: int = 10               # Warmup iterations
    benchmark_iterations: int = 100           # Benchmark iterations
    load_test_duration_sec: int = 60          # Load test duration
    concurrent_users: int = 10                # Concurrent users
    regression_threshold_percent: float = 20.0 # Regression threshold
```

## Benchmark Types

### 1. Single Inference Latency Benchmark

Tests individual inference latency with detailed metrics:

```python
result = await benchmark.run_single_inference_benchmark()
```

**Validates:**
- P50 latency < 50ms
- P95 latency < 75ms
- P99 latency < 100ms
- Error rate < 5%
- Memory usage within limits

### 2. Load Testing Benchmark

Tests performance under concurrent load:

```python
result = await benchmark.run_load_testing_benchmark()
```

**Validates:**
- Throughput ≥ 100 ops/sec
- P95 latency < 75ms under load
- Error rate < 10% under load
- CPU/GPU usage within limits

### 3. Performance Regression Detection

Detects performance regressions against historical baselines:

```python
regression_detected, messages = benchmark.detect_performance_regression(result)
```

**Features:**
- Automatic baseline calculation
- Configurable regression thresholds
- Trend analysis over time
- Early warning system

## Performance Metrics

### PerformanceMetrics

```python
@dataclass
class PerformanceMetrics:
    timestamp: datetime
    test_name: str
    latency_ms: List[float]           # Raw latency measurements
    memory_usage_mb: List[float]      # Memory usage over time
    cpu_usage_percent: List[float]    # CPU usage over time
    gpu_memory_mb: List[float]        # GPU memory usage
    throughput_ops_per_sec: float     # Operations per second
    error_rate_percent: float         # Error rate percentage
    success_count: int                # Successful operations
    failure_count: int                # Failed operations
    p50_latency_ms: float            # 50th percentile latency
    p95_latency_ms: float            # 95th percentile latency
    p99_latency_ms: float            # 99th percentile latency
    max_latency_ms: float            # Maximum latency
    min_latency_ms: float            # Minimum latency
    avg_latency_ms: float            # Average latency
    std_latency_ms: float            # Standard deviation
```

## Advanced Features

### 1. Real-time Performance Monitoring

```python
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Run benchmarks...

memory_usage, cpu_usage, gpu_memory = monitor.stop_monitoring()
```

### 2. Comprehensive Reporting

```python
report = benchmark.generate_performance_report(results)
print(report)
```

Sample report output:
```
================================================================================
STRATEGIC MARL COMPONENT PERFORMANCE BENCHMARK REPORT
================================================================================
Generated: 2024-01-15 14:30:00
Target Latency: 50.0ms

Overall Status: PASSED

Test: single_inference_benchmark
  Status: PASSED
  P50 Latency: 35.20ms
  P95 Latency: 45.80ms
  P99 Latency: 48.50ms
  Throughput: 250.5 ops/sec
  Error Rate: 0.0%

Test: load_testing_benchmark
  Status: PASSED
  P50 Latency: 38.40ms
  P95 Latency: 52.10ms
  P99 Latency: 58.90ms
  Throughput: 180.2 ops/sec
  Error Rate: 1.2%
```

### 3. Historical Data Management

```python
# Automatic historical data management
benchmark._save_historical_results(result)
historical_data = benchmark._load_historical_results()
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Strategic MARL Performance Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  performance-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio
    
    - name: Run performance benchmarks
      run: |
        python -m pytest tests/performance/test_strategic_marl_latency_benchmark.py -v
    
    - name: Generate performance report
      run: |
        python tests/performance/demo_strategic_marl_benchmark.py
    
    - name: Upload performance report
      uses: actions/upload-artifact@v2
      with:
        name: performance-report
        path: /tmp/strategic_marl_demo_report.txt
```

## Performance Targets

### Primary Targets
- **P50 Latency**: < 50ms
- **P95 Latency**: < 75ms
- **P99 Latency**: < 100ms
- **Throughput**: ≥ 100 ops/sec
- **Error Rate**: < 5%

### Resource Limits
- **Memory Increase**: < 50MB
- **CPU Usage**: < 80%
- **GPU Memory**: < 256MB

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH=/path/to/GrandModel:$PYTHONPATH
   ```

2. **Missing Dependencies**
   ```bash
   pip install psutil torch memory-profiler line-profiler
   ```

3. **Permission Errors**
   ```bash
   # Create temp directory
   mkdir -p /tmp/strategic_marl_benchmarks
   chmod 755 /tmp/strategic_marl_benchmarks
   ```

### Performance Optimization Tips

1. **Reduce Warmup Iterations**
   ```python
   config.warmup_iterations = 5  # For faster testing
   ```

2. **Adjust Benchmark Iterations**
   ```python
   config.benchmark_iterations = 50  # For quicker feedback
   ```

3. **Optimize Load Testing**
   ```python
   config.load_test_duration_sec = 30  # Shorter duration
   config.concurrent_users = 5        # Fewer users
   ```

## API Reference

### Core Classes

- `StrategicMARLBenchmark`: Main benchmark suite class
- `LatencyBenchmarkConfig`: Configuration management
- `PerformanceMetrics`: Metrics collection and analysis
- `BenchmarkResult`: Result aggregation and reporting
- `PerformanceMonitor`: Real-time monitoring

### Test Classes

- `TestStrategicMARLLatencyBenchmark`: Primary test suite
- Async test methods for comprehensive validation
- Configuration validation tests
- Performance regression tests

## Contributing

### Adding New Benchmarks

1. Create new test method in `TestStrategicMARLLatencyBenchmark`
2. Follow naming convention: `test_<benchmark_name>`
3. Use async/await for asynchronous operations
4. Include proper assertions and error handling

### Extending Metrics

1. Update `PerformanceMetrics` dataclass
2. Modify collection logic in benchmark methods
3. Update reporting format
4. Add validation in tests

## License

This benchmark suite is part of the GrandModel project and follows the same licensing terms.

## Support

For questions or issues with the benchmark suite:
1. Check the troubleshooting section
2. Review test output for detailed error messages
3. Examine historical results for regression patterns
4. Contact the development team for advanced support

---

**Note**: This benchmark suite is designed to validate the <50ms latency target for strategic_marl_component.py. All performance targets and limits are configurable based on your specific requirements.