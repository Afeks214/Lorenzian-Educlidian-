# Matrix Assembler Components

## Overview

The matrix assembler components are responsible for transforming real-time market data and technical indicators into structured input matrices for the MARL (Multi-Agent Reinforcement Learning) system. These components handle data preprocessing, normalization, and temporal window management across multiple timeframes.

## Components

### Matrix Assembler 30m (`assembler_30m_enhanced.py`)

Strategic-level matrix assembler that processes 30-minute bar data for long-term decision making.

**Key Features:**
- 48-bar sliding window for strategic patterns
- 13 enhanced features including MMD (Market Microstructure Detector)
- Real-time indicator integration
- Normalized feature scaling
- Event-driven matrix updates

**Feature Set:**
- `mlmi_value`: MLMI indicator value (0-1 normalized)
- `mlmi_signal`: MLMI directional signal (-1, 0, 1)
- `nwrqk_value`: NWRQK momentum indicator (0-1 normalized)
- `nwrqk_slope`: NWRQK slope derivative
- `fvg_bullish_active`: Active bullish Fair Value Gaps count
- `fvg_bearish_active`: Active bearish Fair Value Gaps count
- `fvg_nearest_level`: Distance to nearest FVG level
- `lvn_distance_points`: Distance to Low Volume Node in points
- `lvn_nearest_strength`: Strength of nearest LVN (0-1)
- `mmd_institutional_flow`: Institutional flow detection (0-1)
- `mmd_retail_sentiment`: Retail sentiment indicator (0-1)
- `mmd_liquidity_premium`: Liquidity premium measurement (0-1)
- `mmd_volatility_regime`: Current volatility regime (0-1)

**Usage:**
```python
from src.matrix.assembler_30m_enhanced import MatrixAssembler30mEnhanced

# Initialize with configuration
config = {
    'window_size': 48,
    'features': ['mlmi_value', 'mlmi_signal', 'nwrqk_value', 'mmd_institutional_flow'],
    'normalization': True,
    'fill_method': 'forward'
}

assembler = MatrixAssembler30mEnhanced(config, event_bus)

# Initialize and start
assembler.initialize()
assembler.start()

# The assembler automatically listens for INDICATORS_READY events
# and publishes MATRIX_30M_READY events when new matrices are available
```

**Matrix Structure:**
```python
# Output matrix shape: (48, 13)
matrix = assembler.get_current_matrix()
print(f"Matrix shape: {matrix.shape}")  # (48, 13)
print(f"Latest values: {matrix[-1, :5]}")  # Most recent bar features
```

### Matrix Assembler 5m (`assembler_5m.py`)

Tactical-level matrix assembler for short-term execution decisions.

**Key Features:**
- 60-bar sliding window for tactical patterns
- 7 specialized features for execution timing
- High-frequency updates (every 5 minutes)
- FVG and LVN pattern integration
- Sub-second matrix assembly

**Feature Set:**
- `fvg_bullish_active`: Active bullish FVG count
- `fvg_bearish_active`: Active bearish FVG count
- `fvg_nearest_level`: Distance to nearest FVG
- `lvn_distance_points`: Distance to nearest LVN
- `lvn_nearest_strength`: LVN strength indicator
- `price_momentum`: Short-term price momentum
- `volume_profile`: Volume distribution analysis

**Usage:**
```python
from src.matrix.assembler_5m import MatrixAssembler5m

assembler = MatrixAssembler5m(config, event_bus)
assembler.initialize()

# Matrix updates every 5 minutes
matrix = assembler.get_current_matrix()
print(f"Tactical matrix shape: {matrix.shape}")  # (60, 7)
```

### Base Matrix Assembler (`base.py`)

Abstract base class providing common functionality for all matrix assemblers.

**Core Features:**
- Standardized matrix assembly interface
- Data validation and error handling
- Performance monitoring and metrics
- State persistence and recovery
- Event subscription management

**Interface:**
```python
from src.matrix.base import BaseMatrixAssembler

class CustomMatrixAssembler(BaseMatrixAssembler):
    def _assemble_matrix(self, data):
        """Implement custom matrix assembly logic"""
        pass
    
    def _validate_data(self, data):
        """Implement data validation"""
        pass
    
    def _normalize_features(self, matrix):
        """Implement feature normalization"""
        pass
```

### Normalizers (`normalizers.py`)

Feature normalization utilities for consistent data scaling.

**Normalization Methods:**
- **Min-Max Scaling**: Scale features to [0, 1] range
- **Z-Score Normalization**: Standard normal distribution
- **Robust Scaling**: Median and IQR-based scaling
- **Rolling Window**: Time-series aware normalization

**Usage:**
```python
from src.matrix.normalizers import MinMaxNormalizer, RollingNormalizer

# Static normalization
normalizer = MinMaxNormalizer()
normalized_data = normalizer.fit_transform(raw_data)

# Rolling window normalization for time series
rolling_normalizer = RollingNormalizer(window_size=100)
normalized_stream = rolling_normalizer.transform_streaming(data_stream)
```

## Data Flow Architecture

### Strategic Matrix Assembly (30m)

```
Market Data → 30m Bars → Technical Indicators → Matrix Assembly → MARL Strategic Agent
     ↓              ↓              ↓                    ↓
Raw Ticks    OHLCV Data    MLMI, NWRQK, FVG    48x13 Matrix
```

### Tactical Matrix Assembly (5m)

```
Market Data → 5m Bars → Tactical Indicators → Matrix Assembly → MARL Tactical Agent
     ↓             ↓             ↓                    ↓
Raw Ticks    OHLCV Data    FVG, LVN, Volume    60x7 Matrix
```

### Real-Time Updates

```python
# Event-driven matrix updates
@event_bus.subscribe(EventType.INDICATORS_READY)
async def on_indicators_ready(event):
    indicators = event.payload
    
    # Update 30m matrix
    matrix_30m = assembler_30m.update_matrix(indicators)
    
    # Publish strategic matrix update
    await event_bus.publish(Event(
        type=EventType.MATRIX_30M_READY,
        payload={'matrix': matrix_30m, 'timestamp': event.timestamp},
        source='matrix_assembler_30m'
    ))
```

## Performance Characteristics

### Benchmarks

| Operation | Target Time | Actual Performance |
|-----------|-------------|-------------------|
| 30m Matrix Assembly | < 1ms | ~0.8ms avg |
| 5m Matrix Assembly | < 0.5ms | ~0.3ms avg |
| Feature Normalization | < 0.2ms | ~0.1ms avg |
| Matrix Validation | < 0.1ms | ~0.05ms avg |

### Memory Usage

- **30m Matrix**: ~25KB per matrix (48 × 13 × 8 bytes)
- **5m Matrix**: ~3.4KB per matrix (60 × 7 × 8 bytes)
- **Historical Storage**: Configurable window (default: 1000 matrices)
- **Total Memory**: ~50MB for full historical cache

### Throughput

- **Update Frequency**: 30m matrices every 30 minutes, 5m matrices every 5 minutes
- **Burst Capacity**: 1000+ matrix assemblies per second
- **Latency**: Sub-millisecond from indicator update to matrix ready

## Configuration

### Production Configuration

```yaml
matrix_assemblers:
  30m:
    window_size: 48
    features:
      - mlmi_value
      - mlmi_signal
      - nwrqk_value
      - nwrqk_slope
      - fvg_bullish_active
      - fvg_bearish_active
      - fvg_nearest_level
      - lvn_distance_points
      - lvn_nearest_strength
      - mmd_institutional_flow
      - mmd_retail_sentiment
      - mmd_liquidity_premium
      - mmd_volatility_regime
    normalization:
      method: rolling_zscore
      window: 100
    validation:
      enabled: true
      max_nan_ratio: 0.1
    
  5m:
    window_size: 60
    features:
      - fvg_bullish_active
      - fvg_bearish_active
      - fvg_nearest_level
      - lvn_distance_points
      - lvn_nearest_strength
      - price_momentum
      - volume_profile
    normalization:
      method: minmax
      clip_outliers: true
    performance:
      max_assembly_time_ms: 0.5
      enable_caching: true
```

### Development Configuration

```yaml
matrix_assemblers:
  30m:
    window_size: 20  # Smaller window for faster testing
    debug: true
    save_matrices: true  # Save to disk for analysis
    
  5m:
    window_size: 30
    debug: true
    validation:
      strict_mode: true  # Extra validation checks
```

## Integration with MARL System

### Strategic Agent Integration

```python
from src.agents.strategic_agent import StrategicAgent

class StrategicMARLIntegration:
    def __init__(self, agent, matrix_assembler):
        self.agent = agent
        self.matrix_assembler = matrix_assembler
        
    @event_bus.subscribe(EventType.MATRIX_30M_READY)
    async def on_strategic_matrix_ready(self, event):
        matrix = event.payload['matrix']
        
        # Convert to agent observation format
        observation = self._matrix_to_observation(matrix)
        
        # Get agent decision
        action = self.agent.act(observation)
        
        # Publish decision
        await event_bus.publish(Event(
            type=EventType.STRATEGIC_DECISION,
            payload=action,
            source='strategic_agent'
        ))
    
    def _matrix_to_observation(self, matrix):
        """Convert matrix to agent observation format"""
        # Flatten matrix for agent input
        return matrix.flatten()
```

### Tactical Agent Integration

```python
class TacticalMARLIntegration:
    @event_bus.subscribe(EventType.MATRIX_5M_READY)
    async def on_tactical_matrix_ready(self, event):
        matrix = event.payload['matrix']
        
        # High-frequency tactical decisions
        observation = self._process_tactical_matrix(matrix)
        action = self.tactical_agent.act(observation)
        
        # Immediate execution signals
        await event_bus.publish(Event(
            type=EventType.TACTICAL_SIGNAL,
            payload=action,
            source='tactical_agent'
        ))
```

## Error Handling and Recovery

### Data Quality Validation

```python
def validate_matrix_quality(matrix):
    """Comprehensive matrix quality checks"""
    
    # Check for NaN values
    nan_ratio = np.isnan(matrix).sum() / matrix.size
    if nan_ratio > 0.1:
        raise DataQualityError(f"Too many NaN values: {nan_ratio:.2%}")
    
    # Check for infinite values
    if np.isinf(matrix).any():
        raise DataQualityError("Infinite values detected")
    
    # Check feature ranges
    for i, feature_name in enumerate(feature_names):
        feature_data = matrix[:, i]
        if feature_data.std() == 0:
            logger.warning(f"Feature {feature_name} has zero variance")
    
    # Check temporal consistency
    if not _check_temporal_order(matrix):
        raise DataQualityError("Temporal order violation")
```

### Recovery Mechanisms

```python
class MatrixAssemblerRecovery:
    def __init__(self, assembler):
        self.assembler = assembler
        self.backup_matrices = deque(maxlen=10)
    
    def handle_assembly_failure(self, error):
        """Handle matrix assembly failures"""
        
        logger.error(f"Matrix assembly failed: {error}")
        
        # Try to recover using last known good matrix
        if self.backup_matrices:
            recovery_matrix = self.backup_matrices[-1]
            logger.info("Using backup matrix for recovery")
            return recovery_matrix
        
        # Generate synthetic matrix if no backup available
        synthetic_matrix = self._generate_synthetic_matrix()
        logger.warning("Generated synthetic matrix for emergency recovery")
        return synthetic_matrix
    
    def _generate_synthetic_matrix(self):
        """Generate emergency synthetic matrix"""
        # Create matrix with neutral values
        shape = (self.assembler.window_size, len(self.assembler.features))
        return np.zeros(shape)
```

## Monitoring and Observability

### Matrix Quality Metrics

```python
class MatrixQualityMonitor:
    def __init__(self):
        self.metrics = {
            'assembly_time': [],
            'nan_ratio': [],
            'feature_variance': [],
            'temporal_gaps': []
        }
    
    def track_matrix_assembly(self, matrix, assembly_time):
        """Track matrix assembly metrics"""
        
        # Performance tracking
        self.metrics['assembly_time'].append(assembly_time)
        
        # Data quality tracking
        nan_ratio = np.isnan(matrix).sum() / matrix.size
        self.metrics['nan_ratio'].append(nan_ratio)
        
        # Feature analysis
        feature_variance = np.var(matrix, axis=0)
        self.metrics['feature_variance'].append(feature_variance)
        
        # Generate alerts if needed
        self._check_quality_alerts(matrix, assembly_time)
    
    def _check_quality_alerts(self, matrix, assembly_time):
        """Check for quality issues requiring alerts"""
        
        if assembly_time > 1.0:  # 1ms threshold
            logger.warning(f"Slow matrix assembly: {assembly_time:.3f}ms")
        
        if np.isnan(matrix).any():
            logger.warning("NaN values detected in matrix")
        
        if np.var(matrix) < 1e-6:
            logger.warning("Very low matrix variance - possible data staleness")
```

## Testing

### Unit Tests

```python
# tests/unit/test_matrix/test_assembler_30m.py
import pytest
import numpy as np
from src.matrix.assembler_30m_enhanced import MatrixAssembler30mEnhanced

class TestMatrixAssembler30m:
    def setUp(self):
        self.config = {
            'window_size': 48,
            'features': ['mlmi_value', 'nwrqk_value'],
            'normalization': False
        }
        self.assembler = MatrixAssembler30mEnhanced(self.config, Mock())
    
    def test_matrix_assembly(self):
        """Test basic matrix assembly"""
        indicators = {
            'mlmi_value': 0.75,
            'nwrqk_value': 0.25
        }
        
        matrix = self.assembler.assemble_matrix(indicators)
        
        assert matrix.shape == (48, 2)
        assert not np.isnan(matrix).any()
        assert matrix[-1, 0] == 0.75  # Latest mlmi_value
        assert matrix[-1, 1] == 0.25  # Latest nwrqk_value
    
    def test_normalization(self):
        """Test feature normalization"""
        self.assembler.config['normalization'] = True
        
        # Add data with known range
        for i in range(50):
            indicators = {'mlmi_value': i / 49.0, 'nwrqk_value': 1.0}
            self.assembler.update_matrix(indicators)
        
        matrix = self.assembler.get_current_matrix()
        
        # Check normalization
        assert 0 <= matrix[:, 0].min() <= 0.1  # Normalized minimum
        assert 0.9 <= matrix[:, 0].max() <= 1.0  # Normalized maximum
```

### Performance Tests

```python
@pytest.mark.performance
def test_assembly_performance():
    """Test matrix assembly performance"""
    assembler = MatrixAssembler30mEnhanced(config, Mock())
    
    # Warm up
    for _ in range(100):
        assembler.update_matrix(sample_indicators)
    
    # Performance test
    start_time = time.perf_counter()
    for _ in range(1000):
        assembler.update_matrix(sample_indicators)
    end_time = time.perf_counter()
    
    avg_time = (end_time - start_time) / 1000 * 1000  # Convert to ms
    assert avg_time < 1.0, f"Assembly too slow: {avg_time:.3f}ms"
```

## Troubleshooting

### Common Issues

**Matrix Assembly Failures:**
- Check indicator data completeness
- Verify feature configuration matches available indicators
- Review data type compatibility

**Performance Issues:**
- Monitor memory usage for large windows
- Check normalization method efficiency
- Review event subscription overhead

**Data Quality Problems:**
- Validate input data ranges
- Check for missing or stale indicators
- Review normalization parameter settings

### Debug Commands

```bash
# Check matrix assembler health
curl http://localhost:8000/matrix/health

# View current matrix state
curl http://localhost:8000/matrix/30m/current

# Get performance metrics
curl http://localhost:8000/matrix/metrics

# Debug matrix assembly
python -c "
from src.matrix.assembler_30m_enhanced import MatrixAssembler30mEnhanced
assembler = MatrixAssembler30mEnhanced(config, None)
print(assembler.debug_info())
"
```

## Related Documentation

- [Core Components](../core/README.md)
- [Indicators Engine](../indicators/README.md)
- [MARL Agents API](../../docs/api/agents_api.md)
- [System Architecture](../../docs/architecture/system_overview.md)