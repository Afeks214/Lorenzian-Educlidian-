# Synergy Detection System

## Overview

The synergy detection system identifies convergent signals across multiple technical indicators and timeframes to generate high-confidence trading opportunities. This component uses advanced pattern recognition algorithms to detect four distinct synergy patterns that indicate optimal entry and exit points for the MARL trading system.

## Pattern Types

### TYPE_1: Momentum Alignment
**Description:** Confluence of directional momentum across multiple timeframes
**Confidence:** High (>0.8)
**Duration:** 15-30 minutes

**Detection Criteria:**
- MLMI signal alignment across 5m and 30m timeframes
- NWRQK momentum confirmation
- Volume surge above average
- No conflicting FVG levels nearby

```python
def detect_type_1_pattern(self, indicators):
    """Detect momentum alignment pattern"""
    
    # Check MLMI alignment
    mlmi_5m_signal = indicators.get('mlmi_5m_signal', 0)
    mlmi_30m_signal = indicators.get('mlmi_30m_signal', 0)
    mlmi_aligned = (mlmi_5m_signal * mlmi_30m_signal) > 0
    
    # Check NWRQK confirmation
    nwrqk_slope = indicators.get('nwrqk_slope', 0)
    nwrqk_confirms = (nwrqk_slope * mlmi_5m_signal) > 0
    
    # Check volume surge
    volume_ratio = indicators.get('volume_ratio', 1.0)
    volume_surge = volume_ratio > 1.5
    
    # Check FVG conflicts
    fvg_conflicts = self._check_fvg_conflicts(indicators, mlmi_5m_signal)
    
    if mlmi_aligned and nwrqk_confirms and volume_surge and not fvg_conflicts:
        return SynergyPattern(
            type='TYPE_1',
            confidence=self._calculate_type_1_confidence(indicators),
            direction=1 if mlmi_5m_signal > 0 else -1,
            duration_estimate=20 * 60,  # 20 minutes
            entry_conditions=self._get_type_1_entry_conditions(indicators)
        )
    
    return None
```

### TYPE_2: Gap Momentum Convergence
**Description:** FVG levels aligned with momentum indicators for precise entries
**Confidence:** Very High (>0.9)
**Duration:** 5-15 minutes

**Detection Criteria:**
- Price approaching significant FVG level
- MLMI momentum in gap direction
- LVN support/resistance nearby
- High institutional flow detected (MMD)

```python
def detect_type_2_pattern(self, indicators):
    """Detect gap momentum convergence pattern"""
    
    # Check FVG proximity
    fvg_distance = indicators.get('fvg_nearest_distance', float('inf'))
    approaching_gap = fvg_distance < 5.0  # Within 5 points
    
    # Check momentum direction
    mlmi_signal = indicators.get('mlmi_signal', 0)
    gap_direction = indicators.get('fvg_nearest_direction', 0)
    momentum_aligned = (mlmi_signal * gap_direction) > 0
    
    # Check LVN confluence
    lvn_distance = indicators.get('lvn_distance_points', float('inf'))
    lvn_nearby = lvn_distance < 3.0  # Within 3 points
    
    # Check institutional flow
    institutional_flow = indicators.get('mmd_institutional_flow', 0)
    institutional_active = institutional_flow > 0.7
    
    if approaching_gap and momentum_aligned and lvn_nearby and institutional_active:
        return SynergyPattern(
            type='TYPE_2',
            confidence=self._calculate_type_2_confidence(indicators),
            direction=gap_direction,
            duration_estimate=10 * 60,  # 10 minutes
            entry_level=indicators.get('fvg_nearest_level'),
            stop_loss=self._calculate_gap_stop_loss(indicators)
        )
    
    return None
```

### TYPE_3: Mean Reversion Setup
**Description:** Oversold/overbought conditions with reversal confluence
**Confidence:** Medium-High (0.7-0.85)
**Duration:** 30-60 minutes

**Detection Criteria:**
- MLMI extreme values (>0.8 or <-0.8)
- Price at significant LVN level
- Divergence between price and NWRQK
- Retail sentiment extreme (MMD)

### TYPE_4: Breakout Confirmation
**Description:** Multi-timeframe breakout with volume and momentum confirmation
**Confidence:** High (0.8-0.95)
**Duration:** 60-120 minutes

**Detection Criteria:**
- Price breaking significant LVN resistance/support
- Volume surge >200% of average
- MLMI and NWRQK alignment in breakout direction
- No conflicting FVG levels above/below

## Core Components

### Synergy Detector (`detector.py`)

Main detection engine that processes indicator data and identifies patterns.

```python
from src.synergy.detector import SynergyDetector

class SynergyDetector:
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        self.pattern_detectors = self._initialize_pattern_detectors()
        self.active_patterns = []
        self.pattern_history = deque(maxlen=1000)
        
        # Subscribe to indicator updates
        event_bus.subscribe(EventType.INDICATORS_READY, self.on_indicators_ready)
    
    async def on_indicators_ready(self, event):
        """Process new indicator data for pattern detection"""
        indicators = event.payload
        
        # Detect all pattern types
        detected_patterns = []
        
        for detector_name, detector in self.pattern_detectors.items():
            pattern = detector.detect(indicators)
            if pattern:
                detected_patterns.append(pattern)
        
        # Filter and rank patterns
        high_confidence_patterns = [
            p for p in detected_patterns 
            if p.confidence >= self.config['min_confidence']
        ]
        
        # Publish synergy events
        for pattern in high_confidence_patterns:
            await self._publish_synergy_event(pattern)
        
        # Update active patterns
        self._update_active_patterns(high_confidence_patterns)
    
    async def _publish_synergy_event(self, pattern):
        """Publish synergy detection event"""
        synergy_event = Event(
            type=EventType.SYNERGY_DETECTED,
            payload={
                'pattern_type': pattern.type,
                'confidence': pattern.confidence,
                'direction': pattern.direction,
                'entry_level': pattern.entry_level,
                'stop_loss': pattern.stop_loss,
                'duration_estimate': pattern.duration_estimate,
                'timestamp': datetime.now()
            },
            source='synergy_detector'
        )
        
        await self.event_bus.publish(synergy_event)
```

### Pattern Base (`patterns.py`)

Base classes and data structures for pattern definitions.

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

@dataclass
class SynergyPattern:
    """Base synergy pattern data structure"""
    type: str                    # Pattern type (TYPE_1, TYPE_2, etc.)
    confidence: float           # Confidence score (0-1)
    direction: int              # Direction: 1 (bullish), -1 (bearish)
    timestamp: datetime         # Detection timestamp
    duration_estimate: int      # Expected duration in seconds
    entry_level: Optional[float] = None    # Optimal entry price
    stop_loss: Optional[float] = None      # Stop loss level
    take_profit: Optional[float] = None    # Take profit level
    metadata: Dict[str, Any] = None        # Additional pattern data
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate expiry time
        self.expires_at = self.timestamp + timedelta(seconds=self.duration_estimate)
    
    def is_expired(self, current_time: datetime = None) -> bool:
        """Check if pattern has expired"""
        if current_time is None:
            current_time = datetime.now()
        return current_time > self.expires_at
    
    def get_risk_reward_ratio(self) -> Optional[float]:
        """Calculate risk/reward ratio if levels are available"""
        if not all([self.entry_level, self.stop_loss, self.take_profit]):
            return None
        
        risk = abs(self.entry_level - self.stop_loss)
        reward = abs(self.take_profit - self.entry_level)
        
        return reward / risk if risk > 0 else None

class PatternDetectorBase:
    """Base class for pattern detectors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_history = deque(maxlen=100)
    
    def detect(self, indicators: Dict[str, float]) -> Optional[SynergyPattern]:
        """Detect pattern in indicator data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _calculate_confidence(self, indicators: Dict[str, float]) -> float:
        """Calculate pattern confidence score"""
        raise NotImplementedError
    
    def _validate_pattern(self, pattern: SynergyPattern) -> bool:
        """Validate detected pattern"""
        # Basic validation
        if not (0 <= pattern.confidence <= 1):
            return False
        
        if pattern.direction not in [-1, 1]:
            return False
        
        # Risk management validation
        if pattern.entry_level and pattern.stop_loss:
            risk_percent = abs(pattern.entry_level - pattern.stop_loss) / pattern.entry_level
            if risk_percent > self.config.get('max_risk_percent', 0.02):  # 2% max risk
                return False
        
        return True
```

### Sequence Analysis (`sequence.py`)

Advanced pattern sequence analysis for detecting complex multi-bar patterns.

```python
class SequenceAnalyzer:
    """Analyze sequences of indicator values for complex patterns"""
    
    def __init__(self, config):
        self.config = config
        self.sequence_buffer = deque(maxlen=config['max_sequence_length'])
        self.known_sequences = self._load_known_sequences()
    
    def add_indicators(self, indicators: Dict[str, float]):
        """Add new indicator set to sequence buffer"""
        self.sequence_buffer.append({
            'timestamp': datetime.now(),
            'indicators': indicators.copy()
        })
    
    def detect_sequences(self) -> List[SynergyPattern]:
        """Detect complex multi-bar patterns"""
        detected_patterns = []
        
        if len(self.sequence_buffer) < self.config['min_sequence_length']:
            return detected_patterns
        
        # Analyze recent sequence
        recent_sequence = list(self.sequence_buffer)[-self.config['analysis_window']:]
        
        # Check for known patterns
        for pattern_name, pattern_definition in self.known_sequences.items():
            if self._matches_sequence_pattern(recent_sequence, pattern_definition):
                pattern = self._create_sequence_pattern(pattern_name, recent_sequence)
                if pattern:
                    detected_patterns.append(pattern)
        
        return detected_patterns
    
    def _matches_sequence_pattern(self, sequence, pattern_definition):
        """Check if sequence matches known pattern"""
        if len(sequence) < len(pattern_definition['conditions']):
            return False
        
        for i, condition in enumerate(pattern_definition['conditions']):
            bar_data = sequence[-(len(pattern_definition['conditions']) - i)]
            
            if not self._evaluate_condition(bar_data['indicators'], condition):
                return False
        
        return True
    
    def _evaluate_condition(self, indicators, condition):
        """Evaluate single condition against indicator data"""
        for indicator_name, criteria in condition.items():
            if indicator_name not in indicators:
                return False
            
            value = indicators[indicator_name]
            
            # Check range conditions
            if 'min' in criteria and value < criteria['min']:
                return False
            if 'max' in criteria and value > criteria['max']:
                return False
            
            # Check trend conditions
            if 'trend' in criteria:
                # Implementation depends on specific trend requirements
                pass
        
        return True
```

## Configuration

### Production Configuration

```yaml
synergy_detection:
  # Pattern detection settings
  min_confidence: 0.75          # Minimum confidence for pattern activation
  max_active_patterns: 5        # Maximum concurrent active patterns
  pattern_timeout: 3600         # Pattern expiry time in seconds
  
  # Pattern-specific settings
  type_1:                       # Momentum Alignment
    enabled: true
    min_confidence: 0.8
    volume_surge_threshold: 1.5
    max_fvg_distance: 10.0
    
  type_2:                       # Gap Momentum Convergence  
    enabled: true
    min_confidence: 0.9
    fvg_proximity_threshold: 5.0
    lvn_proximity_threshold: 3.0
    min_institutional_flow: 0.7
    
  type_3:                       # Mean Reversion Setup
    enabled: true
    min_confidence: 0.7
    mlmi_extreme_threshold: 0.8
    max_lvn_distance: 2.0
    
  type_4:                       # Breakout Confirmation
    enabled: true
    min_confidence: 0.8
    volume_surge_threshold: 2.0
    min_breakout_distance: 5.0
  
  # Risk management
  risk_management:
    max_risk_percent: 0.02      # 2% maximum risk per pattern
    min_risk_reward_ratio: 2.0  # Minimum 2:1 risk/reward
    max_position_overlap: 2     # Maximum overlapping positions
  
  # Performance settings
  performance:
    max_detection_time_ms: 5    # Maximum detection time
    enable_sequence_analysis: true
    sequence_buffer_size: 100
    enable_pattern_learning: true

  # Monitoring
  monitoring:
    log_all_detections: true
    save_pattern_history: true
    performance_tracking: true
```

### Development Configuration

```yaml
synergy_detection:
  debug: true
  min_confidence: 0.6           # Lower threshold for testing
  save_debug_data: true
  
  # Enhanced logging for development
  logging:
    log_level: DEBUG
    save_indicator_snapshots: true
    pattern_analysis_output: logs/pattern_analysis.json
```

## Usage Examples

### Basic Pattern Detection

```python
from src.synergy.detector import SynergyDetector

# Initialize detector
config = load_config('synergy_detection')
detector = SynergyDetector(config, event_bus)

# Subscribe to synergy events
@event_bus.subscribe(EventType.SYNERGY_DETECTED)
async def on_synergy_detected(event):
    pattern = event.payload
    
    print(f"Synergy Pattern Detected!")
    print(f"Type: {pattern['pattern_type']}")
    print(f"Confidence: {pattern['confidence']:.3f}")
    print(f"Direction: {'BULLISH' if pattern['direction'] > 0 else 'BEARISH'}")
    
    if pattern.get('entry_level'):
        print(f"Entry Level: {pattern['entry_level']:.2f}")
    if pattern.get('stop_loss'):
        print(f"Stop Loss: {pattern['stop_loss']:.2f}")

# Start detection
detector.initialize()
detector.start()
```

### Custom Pattern Detection

```python
class CustomPatternDetector(PatternDetectorBase):
    """Custom pattern detector implementation"""
    
    def detect(self, indicators: Dict[str, float]) -> Optional[SynergyPattern]:
        """Detect custom pattern logic"""
        
        # Custom detection logic
        mlmi_value = indicators.get('mlmi_value', 0)
        nwrqk_value = indicators.get('nwrqk_value', 0)
        fvg_active = indicators.get('fvg_bullish_active', 0)
        
        # Example: Strong momentum with gap support
        if (mlmi_value > 0.7 and 
            nwrqk_value > 0.6 and 
            fvg_active > 0):
            
            confidence = self._calculate_confidence(indicators)
            
            return SynergyPattern(
                type='CUSTOM_MOMENTUM',
                confidence=confidence,
                direction=1,
                timestamp=datetime.now(),
                duration_estimate=30 * 60,  # 30 minutes
                metadata={'mlmi': mlmi_value, 'nwrqk': nwrqk_value}
            )
        
        return None
    
    def _calculate_confidence(self, indicators: Dict[str, float]) -> float:
        """Calculate confidence for custom pattern"""
        # Implement custom confidence calculation
        base_confidence = 0.7
        
        # Boost confidence based on indicator strength
        mlmi_boost = indicators.get('mlmi_value', 0) * 0.2
        volume_boost = min(indicators.get('volume_ratio', 1.0) - 1.0, 0.1)
        
        return min(base_confidence + mlmi_boost + volume_boost, 1.0)

# Register custom detector
detector.register_pattern_detector('custom_momentum', CustomPatternDetector(config))
```

### Pattern Analytics

```python
class SynergyAnalytics:
    """Analytics for synergy pattern performance"""
    
    def __init__(self, detector):
        self.detector = detector
        self.pattern_outcomes = defaultdict(list)
    
    def track_pattern_outcome(self, pattern_id: str, outcome: Dict[str, Any]):
        """Track the outcome of a detected pattern"""
        self.pattern_outcomes[pattern_id].append(outcome)
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern performance statistics"""
        stats = {}
        
        for pattern_type in ['TYPE_1', 'TYPE_2', 'TYPE_3', 'TYPE_4']:
            pattern_outcomes = [
                outcome for outcomes in self.pattern_outcomes.values()
                for outcome in outcomes
                if outcome['pattern_type'] == pattern_type
            ]
            
            if pattern_outcomes:
                stats[pattern_type] = {
                    'total_detected': len(pattern_outcomes),
                    'success_rate': sum(1 for o in pattern_outcomes if o['profitable']) / len(pattern_outcomes),
                    'avg_return': np.mean([o['return_pct'] for o in pattern_outcomes]),
                    'avg_duration': np.mean([o['duration_minutes'] for o in pattern_outcomes]),
                    'confidence_correlation': np.corrcoef(
                        [o['confidence'] for o in pattern_outcomes],
                        [o['return_pct'] for o in pattern_outcomes]
                    )[0, 1]
                }
        
        return stats
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate recommendations for pattern detection optimization"""
        recommendations = []
        stats = self.get_pattern_statistics()
        
        for pattern_type, data in stats.items():
            if data['success_rate'] < 0.6:
                recommendations.append(
                    f"Consider increasing confidence threshold for {pattern_type} "
                    f"(current success rate: {data['success_rate']:.1%})"
                )
            
            if data['confidence_correlation'] < 0.3:
                recommendations.append(
                    f"Review confidence calculation for {pattern_type} "
                    f"(low correlation with returns: {data['confidence_correlation']:.3f})"
                )
        
        return recommendations
```

## Performance Optimization

### Real-Time Detection Optimization

```python
class OptimizedSynergyDetector:
    """Performance-optimized synergy detector"""
    
    def __init__(self, config, event_bus):
        self.config = config
        self.event_bus = event_bus
        
        # Pre-compile detection functions
        self.compiled_detectors = self._compile_detectors()
        
        # Use numpy for vectorized operations
        self.indicator_buffer = np.zeros((100, 20))  # 100 bars, 20 indicators
        self.buffer_index = 0
        
        # Pattern cache for recently calculated patterns
        self.pattern_cache = LRUCache(maxsize=1000)
    
    def _compile_detectors(self):
        """Pre-compile detection functions for better performance"""
        import numba
        
        @numba.jit(nopython=True)
        def fast_type_1_detection(mlmi_5m, mlmi_30m, nwrqk_slope, volume_ratio):
            """Compiled TYPE_1 pattern detection"""
            alignment_score = mlmi_5m * mlmi_30m
            momentum_score = nwrqk_slope * mlmi_5m
            volume_score = volume_ratio - 1.0
            
            if alignment_score > 0.5 and momentum_score > 0.3 and volume_score > 0.5:
                confidence = min(0.8 + (alignment_score + momentum_score + volume_score) * 0.1, 1.0)
                return confidence
            return 0.0
        
        return {'type_1': fast_type_1_detection}
    
    async def process_indicators_fast(self, indicators):
        """High-performance indicator processing"""
        
        # Update circular buffer
        indicator_array = self._indicators_to_array(indicators)
        self.indicator_buffer[self.buffer_index] = indicator_array
        self.buffer_index = (self.buffer_index + 1) % 100
        
        # Use compiled detectors
        detected_patterns = []
        
        for detector_name, detector_func in self.compiled_detectors.items():
            if detector_name == 'type_1':
                confidence = detector_func(
                    indicators.get('mlmi_5m_signal', 0),
                    indicators.get('mlmi_30m_signal', 0),
                    indicators.get('nwrqk_slope', 0),
                    indicators.get('volume_ratio', 1.0)
                )
                
                if confidence > self.config['min_confidence']:
                    pattern = SynergyPattern(
                        type='TYPE_1',
                        confidence=confidence,
                        direction=1 if indicators.get('mlmi_5m_signal', 0) > 0 else -1,
                        timestamp=datetime.now(),
                        duration_estimate=20 * 60
                    )
                    detected_patterns.append(pattern)
        
        return detected_patterns
```

## Testing

### Unit Tests

```python
# tests/unit/test_synergy/test_detector.py
import pytest
from src.synergy.detector import SynergyDetector
from src.synergy.patterns import SynergyPattern

class TestSynergyDetector:
    def setUp(self):
        self.config = {
            'min_confidence': 0.75,
            'type_1': {'enabled': True, 'min_confidence': 0.8}
        }
        self.detector = SynergyDetector(self.config, Mock())
    
    def test_type_1_detection(self):
        """Test TYPE_1 pattern detection"""
        # Strong momentum alignment indicators
        indicators = {
            'mlmi_5m_signal': 1.0,
            'mlmi_30m_signal': 1.0,
            'nwrqk_slope': 0.5,
            'volume_ratio': 2.0,
            'fvg_nearest_distance': 15.0  # No conflicts
        }
        
        patterns = self.detector.detect_patterns(indicators)
        
        assert len(patterns) >= 1
        type_1_patterns = [p for p in patterns if p.type == 'TYPE_1']
        assert len(type_1_patterns) == 1
        
        pattern = type_1_patterns[0]
        assert pattern.confidence >= 0.8
        assert pattern.direction == 1
    
    def test_pattern_filtering(self):
        """Test pattern confidence filtering"""
        # Weak signals
        indicators = {
            'mlmi_5m_signal': 0.3,
            'mlmi_30m_signal': 0.2,
            'nwrqk_slope': 0.1
        }
        
        patterns = self.detector.detect_patterns(indicators)
        
        # Should not detect any patterns due to low confidence
        assert len(patterns) == 0
    
    def test_pattern_expiry(self):
        """Test pattern expiry mechanism"""
        pattern = SynergyPattern(
            type='TYPE_1',
            confidence=0.85,
            direction=1,
            timestamp=datetime.now() - timedelta(hours=2),
            duration_estimate=30 * 60  # 30 minutes
        )
        
        assert pattern.is_expired()
```

### Integration Tests

```python
@pytest.mark.integration
def test_synergy_detector_integration():
    """Test complete synergy detection integration"""
    detector = SynergyDetector(production_config, event_bus)
    detector.initialize()
    
    # Generate test market scenario
    test_scenario = create_test_market_scenario('strong_uptrend')
    
    detected_patterns = []
    
    @event_bus.subscribe(EventType.SYNERGY_DETECTED)
    async def capture_patterns(event):
        detected_patterns.append(event.payload)
    
    # Process test scenario
    for indicator_update in test_scenario:
        await detector.on_indicators_ready(
            Event(type=EventType.INDICATORS_READY, payload=indicator_update)
        )
    
    # Verify pattern detection
    assert len(detected_patterns) > 0
    
    # Verify pattern quality
    for pattern in detected_patterns:
        assert pattern['confidence'] >= production_config['min_confidence']
        assert pattern['pattern_type'] in ['TYPE_1', 'TYPE_2', 'TYPE_3', 'TYPE_4']
```

## Troubleshooting

### Common Issues

**No Patterns Detected:**
- Check indicator data quality and completeness
- Verify confidence thresholds are not too high
- Review pattern detection criteria
- Check for missing required indicators

**False Positive Patterns:**
- Increase confidence thresholds
- Add additional filtering criteria
- Review risk management settings
- Implement pattern validation

**Performance Issues:**
- Enable pattern caching
- Use compiled detection functions
- Reduce sequence analysis window
- Optimize indicator calculations

### Debug Commands

```bash
# Check synergy detector health
curl http://localhost:8000/synergy/health

# View active patterns
curl http://localhost:8000/synergy/active

# Get pattern statistics
curl http://localhost:8000/synergy/statistics

# Debug pattern detection
python -c "
from src.synergy.detector import SynergyDetector
detector = SynergyDetector(config, None)
print(detector.debug_pattern_detection(test_indicators))
"
```

## Related Documentation

- [Indicators Engine](../indicators/README.md)
- [Matrix Assemblers](../matrix/README.md)
- [MARL Agents API](../../docs/api/agents_api.md)
- [Event System](../core/README.md)