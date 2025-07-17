# Unified Data Pipeline System for NQ Dataset Processing

A comprehensive data pipeline system designed for massive NQ dataset processing that can be seamlessly shared between execution engine and risk management notebooks.

## 🚀 Key Features

### 1. **Unified Data Loading**
- **Common Interface**: Single API for both execution engine and risk management notebooks
- **Multiple Timeframes**: Support for 30min, 5min, and extended datasets
- **Chunked Processing**: Memory-efficient processing of large datasets
- **Data Validation**: Comprehensive validation with customizable rules
- **Preprocessing Pipeline**: Automatic feature engineering and normalization

### 2. **Memory Optimization**
- **Shared Memory Pools**: Efficient data sharing between processes
- **Intelligent Caching**: LRU/LFU/FIFO caching with persistence
- **Memory Monitoring**: Real-time memory usage tracking with alerts
- **Memory Mapping**: Efficient large file access with mmap
- **Automatic Cleanup**: Garbage collection and resource management

### 3. **Data Flow Coordination**
- **Inter-notebook Communication**: Seamless data sharing between notebooks
- **Stream Processing**: Real-time data streams with buffering
- **Concurrent Processing**: Multi-threaded and multi-process support
- **Data Consistency**: Checksum validation and consistency checks
- **Synchronization**: Coordinated data access and updates

### 4. **Performance Monitoring**
- **Real-time Metrics**: Live performance monitoring dashboard
- **Benchmarking Suite**: Comprehensive performance testing
- **Load Testing**: Stress testing with various data sizes
- **Performance Alerts**: Threshold-based alerting system
- **Detailed Reporting**: HTML reports with visualizations

### 5. **Scalability Features**
- **Multi-GPU Support**: Parallel processing across multiple GPUs
- **Distributed Processing**: Scale across multiple nodes
- **Auto-scaling**: Dynamic resource allocation based on load
- **Data Parallelism**: Efficient data distribution strategies
- **Resource Optimization**: Intelligent resource management

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ (32GB+ recommended for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: SSD with 100GB+ free space

### Software Requirements
- **Python**: 3.8+
- **PyTorch**: 1.9+
- **CUDA**: 11.0+ (if using GPU)
- **Operating System**: Linux, macOS, or Windows

### Python Dependencies
```bash
pip install pandas numpy torch matplotlib seaborn psutil scikit-learn
```

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/QuantNova/GrandModel.git
cd GrandModel/colab/data_pipeline
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up data directory**:
```bash
mkdir -p /home/QuantNova/GrandModel/colab/data/
# Place your NQ CSV files in this directory
```

4. **Configure the system**:
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your specific settings
```

## 🚀 Quick Start

### Basic Usage

```python
from unified_data_loader import UnifiedDataLoader
from memory_manager import MemoryManager
from performance_monitor import PerformanceMonitor

# Initialize components
data_loader = UnifiedDataLoader()
memory_manager = MemoryManager()
performance_monitor = PerformanceMonitor()

# Load data
data_30min = data_loader.load_data('30min')
data_5min = data_loader.load_data('5min')

# Store in shared memory
memory_manager.store_data('nq_30min', data_30min)
memory_manager.store_data('nq_5min', data_5min)

# Monitor performance
with PerformanceTimer(performance_monitor, 'processing_time'):
    # Your processing code here
    pass
```

### Advanced Usage with Coordination

```python
from data_flow_coordinator import DataFlowCoordinator, create_notebook_client
from scalability_manager import ScalabilityManager

# Set up coordination
coordinator = DataFlowCoordinator()
notebook_client = create_notebook_client('execution_engine', 'execution', coordinator)

# Set up scalability
scalability_manager = ScalabilityManager()
scalability_manager.initialize_system('multi_gpu')

# Create data stream
stream = notebook_client.create_data_stream(
    'market_data',
    DataStreamType.MARKET_DATA,
    ['risk_management']
)

# Process data with scaling
result = scalability_manager.process_large_dataset(
    data_tensor,
    processing_function,
    batch_size=2000
)
```

## 📊 Data Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Data Pipeline System                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  │ Execution Engine│    │ Risk Management │    │   Other         │
│  │    Notebook     │    │    Notebook     │    │ Notebooks       │
│  └─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
│            │                      │                      │
│            └──────────────────────┼──────────────────────┘
│                                   │
│  ┌─────────────────────────────────▼─────────────────────────────────┐
│  │                    Data Flow Coordinator                         │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │  │ Data Streams│  │Synchronizer │  │ Consistency │  │ Concurrent  │
│  │  │             │  │             │  │   Checker   │  │ Processor   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│  └─────────────────────────────────────────────────────────────────┘
│                                   │
│  ┌─────────────────────────────────▼─────────────────────────────────┐
│  │                    Unified Data Loader                           │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │  │   Loader    │  │ Validator   │  │Preprocessor │  │   Cache     │
│  │  │             │  │             │  │             │  │             │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│  └─────────────────────────────────────────────────────────────────┘
│                                   │
│  ┌─────────────────────────────────▼─────────────────────────────────┐
│  │                    Memory Manager                                │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │  │ Shared Pool │  │  Monitor    │  │ Memory Map  │  │ Optimizer   │
│  │  │             │  │             │  │             │  │             │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│  └─────────────────────────────────────────────────────────────────┘
│                                   │
│  ┌─────────────────────────────────▼─────────────────────────────────┐
│  │                  Scalability Manager                             │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │  │ Multi-GPU   │  │ Distributed │  │ Auto-Scaler │  │ Optimizer   │
│  │  │ Processor   │  │ Processor   │  │             │  │             │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│  └─────────────────────────────────────────────────────────────────┘
│                                   │
│  ┌─────────────────────────────────▼─────────────────────────────────┐
│  │                Performance Monitor                               │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  │  │ Metrics     │  │ Benchmarks  │  │ Dashboard   │  │ Alerts      │
│  │  │ Collector   │  │             │  │             │  │             │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
│  └─────────────────────────────────────────────────────────────────┘
│                                   │
│  ┌─────────────────────────────────▼─────────────────────────────────┐
│  │                      Data Storage                                │
│  │          NQ 30min Data    │    NQ 5min Data    │    Cache        │
│  └─────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

The system uses a YAML configuration file (`config.yaml`) for all settings:

```yaml
# Data Loading Configuration
data_loading:
  data_dir: "/home/QuantNova/GrandModel/colab/data/"
  chunk_size: 10000
  cache_enabled: true
  validation_enabled: true
  preprocessing_enabled: true

# Memory Management Configuration
memory_management:
  shared_pool_size_gb: 4.0
  enable_monitoring: true
  eviction_policy: "lru"

# Performance Monitoring
performance_monitoring:
  enable_dashboard: true
  max_history: 10000
  alert_thresholds:
    data_load_time: 10.0
    memory_usage: 0.9

# Scalability Configuration
scalability:
  max_workers: 8
  enable_gpu_acceleration: true
  auto_scaling:
    enabled: true
    thresholds:
      cpu_usage: 0.8
      memory_usage: 0.8
```

## 📊 Performance Monitoring

### Built-in Metrics
- **Data Loading**: Load times, validation times, preprocessing times
- **Memory Usage**: System memory, shared pool usage, GPU memory
- **Throughput**: Processing rates, stream message rates
- **System Health**: CPU usage, memory usage, disk usage, GPU utilization

### Dashboard Features
- **Real-time Visualization**: Live updating charts and graphs
- **Performance Alerts**: Threshold-based notifications
- **Benchmark Reports**: Detailed performance analysis
- **Export Capabilities**: JSON, CSV, HTML reports

### Benchmarking Suite
```python
# Create benchmark suite
benchmark_suite = performance_monitor.create_benchmark_suite(data_loader)

# Run comprehensive benchmarks
loading_results = benchmark_suite.benchmark_loading_performance(['30min', '5min'])
chunked_results = benchmark_suite.benchmark_chunked_loading('30min', [1000, 5000, 10000])
caching_results = benchmark_suite.benchmark_caching_performance('30min')
```

## 🚀 Scalability Features

### Multi-GPU Processing
```python
# Initialize multi-GPU processor
scalability_manager = ScalabilityManager()
scalability_manager.initialize_system('multi_gpu')

# Process data across multiple GPUs
result = scalability_manager.process_large_dataset(
    data_tensor,
    processing_function,
    batch_size=2000
)
```

### Distributed Processing
```python
# Set up distributed processing
os.environ['WORLD_SIZE'] = '4'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# Initialize distributed system
scalability_manager.initialize_system('distributed')
```

### Auto-scaling
```python
# Enable auto-scaling
config = ScalingConfiguration(
    auto_scaling_enabled=True,
    scaling_thresholds={
        'cpu_usage': 0.8,
        'memory_usage': 0.8,
        'gpu_utilization': 0.8
    }
)
```

## 🔄 Data Flow Coordination

### Notebook Registration
```python
# Register notebooks with coordinator
execution_client = create_notebook_client('execution_engine', 'execution', coordinator)
risk_client = create_notebook_client('risk_management', 'risk', coordinator)
```

### Data Streams
```python
# Create data stream
stream = execution_client.create_data_stream(
    'market_data',
    DataStreamType.MARKET_DATA,
    ['risk_management']
)

# Publish data
stream.publish(data, metadata={'timestamp': time.time()})

# Consume data
messages = stream.get_messages(max_messages=10)
```

### Data Synchronization
```python
# Synchronize data between notebooks
success = execution_client.sync_data(
    'risk_management',
    'processed_features',
    feature_data
)
```

## 🧪 Testing and Validation

### Unit Tests
```bash
# Run unit tests
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run performance tests
python -m pytest tests/performance/ -v
```

### Validation Scripts
```python
# Validate data integrity
from unified_data_loader import DataValidator

validator = DataValidator(config)
result = validator.validate_data(data, 'nq_30min')
print(f"Validation result: {result.is_valid}")
```

## 📈 Performance Optimization

### Memory Optimization
- **Shared Memory Pools**: Reduce memory duplication
- **Intelligent Caching**: LRU/LFU cache with persistence
- **Memory Monitoring**: Real-time usage tracking
- **Automatic Cleanup**: Garbage collection optimization

### Processing Optimization
- **Chunked Processing**: Handle large datasets efficiently
- **Parallel Processing**: Multi-threaded and multi-process
- **GPU Acceleration**: CUDA-optimized operations
- **Batch Optimization**: Optimal batch size selection

### I/O Optimization
- **Memory Mapping**: Efficient file access
- **Compressed Storage**: Reduced disk usage
- **Async Operations**: Non-blocking I/O
- **Connection Pooling**: Reusable connections

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce chunk size in configuration
   - Enable memory monitoring
   - Use memory mapping for large files

2. **Slow Loading Performance**
   - Enable caching
   - Increase chunk size
   - Check disk I/O performance

3. **GPU Memory Issues**
   - Reduce batch size
   - Enable GPU memory optimization
   - Use gradient checkpointing

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable performance profiling
config['development']['profiling']['enabled'] = True
```

## 🔐 Security Considerations

### Data Security
- **Encryption**: Optional AES-256-GCM encryption
- **Access Control**: Role-based permissions
- **Audit Logging**: Comprehensive audit trails
- **Data Validation**: Input sanitization

### System Security
- **Process Isolation**: Sandboxed processing
- **Resource Limits**: Memory and CPU limits
- **Network Security**: Encrypted communication
- **Dependency Management**: Secure dependencies

## 📚 Examples

### Example 1: Basic Data Loading
```python
# Load and process 30-minute data
data_loader = UnifiedDataLoader()
data_30min = data_loader.load_data('30min')
print(f"Loaded {len(data_30min)} rows")
```

### Example 2: Memory-Optimized Processing
```python
# Process large dataset with memory optimization
memory_manager = MemoryManager()
memory_manager.store_data('nq_data', data)

# Process in chunks
for chunk in data_loader.load_chunked_data('30min', chunk_size=5000):
    # Process chunk
    result = process_chunk(chunk)
    # Store result
    memory_manager.store_data(f'result_{chunk.index[0]}', result)
```

### Example 3: Real-time Stream Processing
```python
# Set up real-time stream
stream = execution_client.create_data_stream(
    'realtime_data',
    DataStreamType.MARKET_DATA,
    ['risk_management']
)

# Process stream in real-time
while True:
    messages = stream.get_messages(max_messages=100)
    for message in messages:
        # Process message
        result = process_message(message.data)
        # Send to output stream
        output_stream.publish(result)
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest`
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Write comprehensive tests

### Testing Guidelines
- Write unit tests for all new features
- Include integration tests for complex features
- Add performance benchmarks for critical paths
- Ensure 90%+ test coverage

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **QuantNova Team**: Core development and architecture
- **PyTorch Team**: Deep learning framework support
- **NumPy/Pandas Teams**: Data processing libraries
- **Open Source Community**: Various utility libraries

## 📞 Support

For support, please:
1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/QuantNova/GrandModel/issues)
3. Create a [new issue](https://github.com/QuantNova/GrandModel/issues/new)
4. Contact the development team

## 🚀 Future Roadmap

### Version 2.0 (Planned)
- **Cloud Integration**: AWS/GCP/Azure support
- **Advanced Analytics**: ML-based optimization
- **Real-time Streaming**: Apache Kafka integration
- **Web Interface**: Browser-based dashboard

### Version 3.0 (Future)
- **Quantum Computing**: Quantum optimization support
- **Edge Computing**: IoT device support
- **Blockchain Integration**: Distributed ledger support
- **AI-powered Optimization**: Automated tuning

---

**Built with ❤️ by the QuantNova Team**

*Last updated: 2024-01-15*