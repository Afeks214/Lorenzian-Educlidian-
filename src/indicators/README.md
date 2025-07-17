# Technical Indicators Engine

## Overview

The indicators engine provides a comprehensive suite of technical analysis tools specifically designed for high-frequency trading and MARL-based decision making. The system features both standard and custom indicators optimized for real-time processing and integration with the matrix assembler components.

## Architecture

### Core Components

- **Indicator Engine (`engine.py`)**: Central orchestrator for all indicator calculations
- **Base Indicator (`base.py`)**: Abstract base class for all indicators
- **Custom Indicators (`custom/`)**: Specialized proprietary indicators
- **Event Integration**: Real-time event-driven indicator updates

## Custom Indicators

### MLMI - Market Learning Momentum Indicator (`custom/mlmi.py`)

Advanced momentum indicator using machine learning-derived features.

**Key Features:**
- Adaptive momentum calculation based on market regime
- Signal strength measurement (0-1 normalized)
- Directional bias detection (-1, 0, 1)
- Real-time regime adaptation

**Mathematical Foundation:**
```
MLMI = tanh(α * momentum_score + β * trend_strength)
Signal = sign(MLMI) if |MLMI| > threshold else 0
```

**Usage:**
```python
from src.indicators.custom.mlmi import MLMIIndicator

mlmi = MLMIIndicator(config={
    'period': 20,
    'smoothing': 0.1,
    'threshold': 0.3
})

# Calculate indicator values
mlmi_value = mlmi.calculate(price_data)
mlmi_signal = mlmi.get_signal()

print(f"MLMI Value: {mlmi_value:.3f}")
print(f"MLMI Signal: {mlmi_signal}")
```

**Configuration:**
```yaml
mlmi:
  period: 20              # Lookback period for momentum calculation
  smoothing: 0.1          # Exponential smoothing factor
  threshold: 0.3          # Signal threshold for binary classification
  adaptive: true          # Enable adaptive threshold based on volatility
  regime_detection: true  # Enable market regime detection
```

### NWRQK - Neural Wave Recognition Quantum Kit (`custom/nwrqk.py`)

Quantum-inspired wave pattern recognition for trend analysis.

**Key Features:**
- Multi-timeframe wave decomposition
- Quantum-inspired pattern matching
- Trend strength quantification
- Slope analysis for momentum direction

**Usage:**
```python
from src.indicators.custom.nwrqk import NWRQKIndicator

nwrqk = NWRQKIndicator(config={
    'window_size': 50,
    'wave_periods': [5, 10, 20],
    'quantum_levels': 8
})

# Real-time calculation
nwrqk_value = nwrqk.calculate(ohlc_data)
nwrqk_slope = nwrqk.get_slope()

print(f"NWRQK Value: {nwrqk_value:.3f}")
print(f"NWRQK Slope: {nwrqk_slope:.4f}")
```

**Wave Decomposition:**
```python
# Access wave components
wave_components = nwrqk.get_wave_components()
for period, amplitude in wave_components.items():
    print(f"Wave {period}: {amplitude:.3f}")
```

### FVG - Fair Value Gap Detector (`custom/fvg.py`)

Advanced gap detection system for identifying institutional order imbalances.

**Key Features:**
- Real-time gap identification
- Gap strength classification (weak, medium, strong)
- Active gap tracking with expiration
- Distance measurement to nearest gaps

**Gap Types:**
- **Bullish FVG**: Price gap above previous high indicating buying pressure
- **Bearish FVG**: Price gap below previous low indicating selling pressure
- **Mitigation**: When price returns to fill a previously identified gap

**Usage:**
```python
from src.indicators.custom.fvg import FVGIndicator

fvg = FVGIndicator(config={
    'min_gap_size': 2.0,     # Minimum gap size in points
    'max_active_gaps': 10,   # Maximum number of tracked gaps
    'gap_timeout': 3600      # Gap expiration time in seconds
})

# Detect gaps in real-time
fvg_data = fvg.calculate(ohlc_data)

print(f"Bullish FVGs: {fvg_data['bullish_active']}")
print(f"Bearish FVGs: {fvg_data['bearish_active']}")
print(f"Nearest Gap Level: {fvg_data['nearest_level']:.2f}")
print(f"Distance to Gap: {fvg_data['distance_points']:.1f} points")
```

**Gap Analysis:**
```python
# Get detailed gap information
active_gaps = fvg.get_active_gaps()
for gap in active_gaps:
    print(f"Gap: {gap.direction} at {gap.level:.2f} "
          f"(strength: {gap.strength}, age: {gap.age}s)")
```

### LVN - Low Volume Node Detector (`custom/lvn.py`)

Volume profile analysis for identifying support/resistance levels.

**Key Features:**
- Volume-price distribution analysis
- Low volume node identification
- Support/resistance strength measurement
- Real-time level updates

**Usage:**
```python
from src.indicators.custom.lvn import LVNIndicator

lvn = LVNIndicator(config={
    'volume_threshold': 0.3,  # Threshold for low volume classification
    'price_bins': 100,        # Number of price levels for analysis
    'lookback_periods': 500   # Volume history for analysis
})

# Calculate volume nodes
lvn_data = lvn.calculate(ohlcv_data)

print(f"Distance to LVN: {lvn_data['distance_points']:.1f} points")
print(f"LVN Strength: {lvn_data['nearest_strength']:.3f}")
```

**Volume Profile Analysis:**
```python
# Get complete volume profile
volume_profile = lvn.get_volume_profile()
for price_level, volume in volume_profile.items():
    if volume < lvn.volume_threshold:
        print(f"Low Volume Node at {price_level:.2f}: {volume:.0f}")
```

### MMD - Market Microstructure Detector (`custom/mmd.py`)

Advanced microstructure analysis for institutional flow detection.

**Key Features:**
- Institutional vs retail flow classification
- Liquidity premium measurement
- Volatility regime detection
- Order flow toxicity analysis

**Usage:**
```python
from src.indicators.custom.mmd import MMDIndicator

mmd = MMDIndicator(config={
    'tick_analysis_window': 1000,
    'flow_threshold': 0.6,
    'volatility_lookback': 100
})

# Microstructure analysis
mmd_data = mmd.calculate(tick_data)

print(f"Institutional Flow: {mmd_data['institutional_flow']:.3f}")
print(f"Retail Sentiment: {mmd_data['retail_sentiment']:.3f}")
print(f"Liquidity Premium: {mmd_data['liquidity_premium']:.3f}")
print(f"Volatility Regime: {mmd_data['volatility_regime']:.3f}")
```

### Tactical FVG (`custom/tactical_fvg.py`)

High-frequency Fair Value Gap detector optimized for tactical execution.

**Key Features:**
- Sub-second gap detection
- Micro-structure gap analysis
- Execution-focused gap classification
- Real-time mitigation tracking

**Usage:**
```python
from src.indicators.custom.tactical_fvg import TacticalFVGIndicator

tactical_fvg = TacticalFVGIndicator(config={
    'min_gap_ticks': 1,      # Minimum gap size in ticks
    'update_frequency': 100, # Update frequency in milliseconds
    'micro_gap_threshold': 0.5
})

# High-frequency gap detection
tactical_data = tactical_fvg.calculate(tick_data)
```

## Indicator Engine

### Engine Configuration

```python
from src.indicators.engine import IndicatorEngine

engine_config = {
    'indicators': {
        'mlmi': {
            'enabled': True,
            'period': 20,
            'update_frequency': 'bar_close'
        },
        'nwrqk': {
            'enabled': True,
            'window_size': 50,
            'update_frequency': 'tick'
        },
        'fvg': {
            'enabled': True,
            'min_gap_size': 2.0,
            'update_frequency': 'tick'
        }
    },
    'performance': {
        'max_calculation_time_ms': 10,
        'enable_caching': True,
        'parallel_processing': True
    }
}

engine = IndicatorEngine(engine_config, event_bus)
```

### Real-Time Updates

```python
class IndicatorEngine:
    def __init__(self, config, event_bus):
        self.indicators = self._initialize_indicators(config)
        self.event_bus = event_bus
        
        # Subscribe to market data events
        event_bus.subscribe(EventType.NEW_TICK, self.on_new_tick)
        event_bus.subscribe(EventType.NEW_5MIN_BAR, self.on_new_bar)
        event_bus.subscribe(EventType.NEW_30MIN_BAR, self.on_new_bar)
    
    async def on_new_tick(self, event):
        """Process new tick data"""
        tick_data = event.payload
        
        # Update tick-based indicators
        updated_indicators = {}
        
        for name, indicator in self.indicators.items():
            if indicator.update_frequency == 'tick':
                try:
                    value = indicator.calculate(tick_data)
                    updated_indicators[name] = value
                except Exception as e:
                    logger.error(f"Error calculating {name}: {e}")
        
        # Publish indicator updates
        if updated_indicators:
            await self.event_bus.publish(Event(
                type=EventType.INDICATORS_UPDATED,
                payload=updated_indicators,
                source='indicator_engine'
            ))
    
    async def on_new_bar(self, event):
        """Process new bar data"""
        bar_data = event.payload
        
        # Calculate all indicators
        all_indicators = {}
        
        for name, indicator in self.indicators.items():
            try:
                value = indicator.calculate(bar_data)
                all_indicators[name] = value
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
        
        # Publish complete indicator set
        await self.event_bus.publish(Event(
            type=EventType.INDICATORS_READY,
            payload=all_indicators,
            source='indicator_engine'
        ))
```

## Performance Optimization

### Calculation Performance

```python
class OptimizedIndicator(BaseIndicator):
    def __init__(self, config):
        super().__init__(config)
        
        # Pre-allocate arrays for performance
        self.data_buffer = np.zeros(config['buffer_size'])
        self.result_cache = {}
        
        # Vectorized operations
        self.vectorized_calc = np.vectorize(self._scalar_calculation)
    
    def calculate(self, data):
        """Optimized calculation with caching"""
        
        # Check cache first
        data_hash = hash(data.tobytes())
        if data_hash in self.result_cache:
            return self.result_cache[data_hash]
        
        # Vectorized calculation
        result = self.vectorized_calc(data)
        
        # Cache result
        self.result_cache[data_hash] = result
        
        # Limit cache size
        if len(self.result_cache) > 1000:
            self.result_cache.pop(next(iter(self.result_cache)))
        
        return result
```

### Memory Management

```python
class MemoryEfficientIndicator:
    def __init__(self, config):
        self.max_history = config.get('max_history', 1000)
        self.data_history = deque(maxlen=self.max_history)
        
        # Use memory-mapped arrays for large datasets
        if config.get('use_memmap', False):
            self.large_buffer = np.memmap(
                'indicator_buffer.dat',
                dtype=np.float64,
                mode='w+',
                shape=(self.max_history, 10)
            )
    
    def update(self, new_data):
        """Memory-efficient data update"""
        self.data_history.append(new_data)
        
        # Update memory-mapped buffer if used
        if hasattr(self, 'large_buffer'):
            self.large_buffer[-1] = new_data
```

## Event Integration

### Event-Driven Architecture

```python
@event_bus.subscribe(EventType.INDICATORS_READY)
async def on_indicators_ready(event):
    """Handle complete indicator calculations"""
    indicators = event.payload
    
    # Forward to matrix assemblers
    await event_bus.publish(Event(
        type=EventType.MATRIX_UPDATE_REQUIRED,
        payload=indicators,
        source='indicator_processor'
    ))

@event_bus.subscribe(EventType.NEW_TICK)
async def on_new_tick(event):
    """Handle high-frequency tick updates"""
    tick = event.payload
    
    # Update tick-sensitive indicators only
    tick_indicators = ['fvg', 'tactical_fvg', 'mmd']
    
    for indicator_name in tick_indicators:
        indicator = indicator_engine.get_indicator(indicator_name)
        if indicator:
            await indicator.update_async(tick)
```

### Performance Monitoring

```python
class IndicatorPerformanceMonitor:
    def __init__(self):
        self.calculation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
    def track_calculation(self, indicator_name, calculation_time):
        """Track indicator calculation performance"""
        self.calculation_times[indicator_name].append(calculation_time)
        
        # Keep only recent measurements
        if len(self.calculation_times[indicator_name]) > 1000:
            self.calculation_times[indicator_name] = \
                self.calculation_times[indicator_name][-500:]
        
        # Alert on performance degradation
        if calculation_time > 10.0:  # 10ms threshold
            logger.warning(
                f"Slow indicator calculation: {indicator_name} "
                f"took {calculation_time:.2f}ms"
            )
    
    def get_performance_report(self):
        """Generate performance report"""
        report = {}
        
        for indicator_name, times in self.calculation_times.items():
            if times:
                report[indicator_name] = {
                    'avg_time_ms': np.mean(times),
                    'max_time_ms': np.max(times),
                    'p95_time_ms': np.percentile(times, 95),
                    'error_count': self.error_counts[indicator_name]
                }
        
        return report
```

## Configuration Examples

### Production Configuration

```yaml
indicators:
  engine:
    performance:
      max_calculation_time_ms: 10
      enable_caching: true
      parallel_processing: true
      max_memory_mb: 512
    
    error_handling:
      max_retries: 3
      fallback_to_default: true
      log_errors: true
  
  mlmi:
    enabled: true
    period: 20
    smoothing: 0.1
    threshold: 0.3
    adaptive: true
    update_frequency: bar_close
    
  nwrqk:
    enabled: true
    window_size: 50
    wave_periods: [5, 10, 20, 40]
    quantum_levels: 8
    update_frequency: bar_close
    
  fvg:
    enabled: true
    min_gap_size: 2.0
    max_active_gaps: 20
    gap_timeout: 7200  # 2 hours
    update_frequency: tick
    
  lvn:
    enabled: true
    volume_threshold: 0.3
    price_bins: 200
    lookback_periods: 1000
    update_frequency: bar_close
    
  mmd:
    enabled: true
    tick_analysis_window: 2000
    flow_threshold: 0.6
    volatility_lookback: 200
    update_frequency: tick
    
  tactical_fvg:
    enabled: true
    min_gap_ticks: 1
    update_frequency: 100  # 100ms
    micro_gap_threshold: 0.5
```

### Development Configuration

```yaml
indicators:
  engine:
    debug: true
    save_calculations: true  # Save to disk for analysis
    performance:
      enable_profiling: true
      
  # Reduced settings for faster testing
  mlmi:
    period: 10
    debug: true
    
  nwrqk:
    window_size: 25
    wave_periods: [5, 10]
    
  fvg:
    min_gap_size: 1.0
    max_active_gaps: 10
```

## Testing

### Unit Tests

```python
# tests/unit/test_indicators/test_mlmi.py
import pytest
import numpy as np
from src.indicators.custom.mlmi import MLMIIndicator

class TestMLMIIndicator:
    def setUp(self):
        self.config = {
            'period': 20,
            'smoothing': 0.1,
            'threshold': 0.3
        }
        self.mlmi = MLMIIndicator(self.config)
    
    def test_calculation_accuracy(self):
        """Test MLMI calculation accuracy"""
        # Generate test data
        prices = np.random.randn(100).cumsum() + 100
        
        # Calculate MLMI
        mlmi_value = self.mlmi.calculate(prices)
        
        # Verify output range
        assert -1 <= mlmi_value <= 1
        assert isinstance(mlmi_value, float)
    
    def test_signal_generation(self):
        """Test MLMI signal generation"""
        # Strong uptrend data
        uptrend_prices = np.linspace(100, 110, 50)
        
        self.mlmi.calculate(uptrend_prices)
        signal = self.mlmi.get_signal()
        
        # Should generate bullish signal
        assert signal > 0
    
    def test_performance(self):
        """Test calculation performance"""
        prices = np.random.randn(1000).cumsum() + 100
        
        start_time = time.perf_counter()
        for _ in range(100):
            self.mlmi.calculate(prices[-100:])
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 100 * 1000  # Convert to ms
        assert avg_time < 1.0, f"MLMI calculation too slow: {avg_time:.3f}ms"
```

### Integration Tests

```python
@pytest.mark.integration
def test_indicator_engine_integration():
    """Test complete indicator engine integration"""
    engine = IndicatorEngine(test_config, mock_event_bus)
    engine.initialize()
    
    # Generate test market data
    test_data = generate_test_ohlcv_data(100)
    
    # Process data through engine
    for bar in test_data:
        engine.process_bar(bar)
    
    # Verify all indicators calculated
    indicators = engine.get_latest_indicators()
    
    assert 'mlmi_value' in indicators
    assert 'nwrqk_value' in indicators
    assert 'fvg_bullish_active' in indicators
    
    # Verify indicator values are reasonable
    assert not np.isnan(indicators['mlmi_value'])
    assert indicators['fvg_bullish_active'] >= 0
```

### Performance Tests

```python
@pytest.mark.performance
def test_indicator_throughput():
    """Test indicator engine throughput"""
    engine = IndicatorEngine(production_config, mock_event_bus)
    
    # Generate high-frequency test data
    tick_data = generate_tick_data(10000)  # 10k ticks
    
    start_time = time.perf_counter()
    
    for tick in tick_data:
        engine.process_tick(tick)
    
    end_time = time.perf_counter()
    
    # Calculate throughput
    throughput = len(tick_data) / (end_time - start_time)
    
    assert throughput > 1000, f"Throughput too low: {throughput:.0f} ticks/sec"
```

## Troubleshooting

### Common Issues

**Slow Indicator Calculations:**
- Check indicator configuration parameters
- Review data buffer sizes
- Monitor memory usage
- Consider enabling caching

**NaN or Infinite Values:**
- Validate input data quality
- Check for division by zero in calculations
- Review normalization parameters
- Implement data validation

**Memory Leaks:**
- Monitor indicator buffer sizes
- Check cache size limits
- Review data history retention
- Use memory profiling tools

### Debug Commands

```bash
# Check indicator engine health
curl http://localhost:8000/indicators/health

# Get current indicator values
curl http://localhost:8000/indicators/current

# View performance metrics
curl http://localhost:8000/indicators/performance

# Debug specific indicator
python -c "
from src.indicators.custom.mlmi import MLMIIndicator
indicator = MLMIIndicator(config)
print(indicator.debug_info())
"
```

### Performance Tuning

```python
# Enable performance profiling
config = {
    'indicators': {
        'engine': {
            'enable_profiling': True,
            'profile_output': 'indicators_profile.txt'
        }
    }
}

# Get performance recommendations
engine = IndicatorEngine(config, event_bus)
recommendations = engine.get_optimization_recommendations()
```

## Related Documentation

- [Matrix Assemblers](../matrix/README.md)
- [Core Events System](../core/README.md)
- [MARL Agents API](../../docs/api/agents_api.md)
- [Performance Guidelines](../../docs/guides/performance_guide.md)