# PettingZoo Environment Tests

This directory contains comprehensive tests for all four PettingZoo environments in the GrandModel system.

## Test Structure

### Individual Environment Tests

1. **`test_strategic_env.py`** - Tests for Strategic MARL Environment (3 agents)
   - MLMI Expert Agent
   - NWRQK Expert Agent  
   - Regime Detection Agent

2. **`test_tactical_env.py`** - Tests for Tactical Market Environment (3 agents)
   - FVG Agent
   - Momentum Agent
   - Entry Optimization Agent

3. **`test_risk_env.py`** - Tests for Risk Management Environment (4 agents)
   - Position Sizing Agent
   - Stop/Target Agent
   - Risk Monitor Agent
   - Portfolio Optimizer Agent

4. **`test_execution_env.py`** - Tests for Execution Environment (5 agents)
   - Position Sizing Agent
   - Stop/Target Agent
   - Risk Monitor Agent
   - Portfolio Optimizer Agent
   - Routing Agent

### Comprehensive Test Suite

- **`test_all_environments.py`** - Cross-environment tests and comparisons
- **`pytest_environments.ini`** - Test configuration with markers

## Test Categories

### 1. PettingZoo API Compliance Tests
```python
@pytest.mark.pettingzoo
def test_pettingzoo_api_compliance():
    """Test PettingZoo API compliance using official test suite"""
```

### 2. Environment Logic Tests
```python
@pytest.mark.strategic
def test_agent_turn_sequence():
    """Test agent turn sequence and state machine"""
```

### 3. Performance Tests
```python
@pytest.mark.performance
def test_step_latency():
    """Test step execution latency"""
```

### 4. Integration Tests
```python
@pytest.mark.integration
def test_unified_execution():
    """Test integration with unified execution system"""
```

### 5. Stability Tests
```python
@pytest.mark.stability
def test_memory_stability():
    """Test memory stability over multiple episodes"""
```

## Running Tests

### Run All Environment Tests
```bash
pytest tests/test_*_env.py -v
```

### Run Specific Environment Tests
```bash
# Strategic environment only
pytest tests/test_strategic_env.py -v

# Tactical environment only
pytest tests/test_tactical_env.py -v

# Risk environment only
pytest tests/test_risk_env.py -v

# Execution environment only
pytest tests/test_execution_env.py -v
```

### Run Tests by Category
```bash
# PettingZoo compliance tests only
pytest -m pettingzoo -v

# Performance tests only
pytest -m performance -v

# Integration tests only
pytest -m integration -v

# Skip slow tests
pytest -m "not slow" -v
```

### Run Comprehensive Test Suite
```bash
# All environments comparison
pytest tests/test_all_environments.py -v

# With detailed output
pytest tests/test_all_environments.py -v -s
```

### Run with Coverage
```bash
pytest tests/test_*_env.py --cov=src --cov-report=html --cov-report=term-missing
```

## Test Features

### 1. PettingZoo API Compliance
- Official PettingZoo API test suite integration
- Action/observation space validation
- Environment lifecycle testing
- Turn-based execution validation

### 2. Environment-Specific Testing
- Agent coordination and state machines
- Observation space correctness
- Action space validation
- Reward calculation testing
- Episode termination conditions

### 3. Performance Benchmarking
- Reset time measurement
- Step execution latency
- Memory usage monitoring
- Throughput analysis

### 4. Error Handling
- Invalid action recovery
- Exception handling
- Error state recovery
- Graceful degradation

### 5. Deterministic Behavior
- Seed-based reproducibility
- Consistent agent selection
- Deterministic rewards
- State consistency

## Environment Specifications

### Strategic Environment
- **Agents**: 3 (mlmi_expert, nwrqk_expert, regime_expert)
- **Action Space**: Box(3,) - probability distributions
- **Observation Space**: Dict with agent_features, shared_context, market_matrix
- **Episode Length**: Configurable (default: 1000 steps)

### Tactical Environment  
- **Agents**: 3 (fvg_agent, momentum_agent, entry_opt_agent)
- **Action Space**: Discrete(3) - bearish/neutral/bullish
- **Observation Space**: Box(60, 7) - market matrix
- **Episode Length**: Configurable (default: 1000 steps)

### Risk Environment
- **Agents**: 4 (position_sizing, stop_target, risk_monitor, portfolio_optimizer)
- **Action Space**: Mixed (discrete and continuous)
- **Observation Space**: Box(10,) - normalized risk state
- **Episode Length**: Configurable (default: 1000 steps)

### Execution Environment
- **Agents**: 5 (position_sizing, stop_target, risk_monitor, portfolio_optimizer, routing)
- **Action Space**: Mixed (discrete and continuous)
- **Observation Space**: Agent-specific dimensions
- **Episode Length**: Configurable (default: 1000 steps)

## Test Data and Fixtures

### Configuration Fixtures
```python
@pytest.fixture
def config():
    """Standard test configuration"""
    return {
        'max_steps': 100,
        'initial_capital': 100000.0,
        # ... other settings
    }
```

### Environment Fixtures
```python
@pytest.fixture
def test_env(config):
    """Create test environment instance"""
    return EnvironmentClass(config)
```

## Performance Benchmarks

### Expected Performance Targets
- **Reset Time**: < 1 second
- **Step Time**: < 100ms average
- **Memory Growth**: < 2x per episode
- **PettingZoo Compliance**: 100% pass rate

### Monitoring Metrics
- Decision latency
- Memory usage
- Episode completion rate
- Error recovery rate

## Integration Points

### 1. Matrix Assemblers
- Strategic: MatrixAssembler30mEnhanced
- Tactical: MatrixAssembler5m
- Risk: CorrelationTracker integration
- Execution: UnifiedExecutionMARLSystem

### 2. Reward Systems
- Strategic: Ensemble decision rewards
- Tactical: FVG pattern rewards
- Risk: Portfolio risk rewards
- Execution: Execution quality rewards

### 3. State Processors
- Normalization and validation
- Feature extraction
- State consistency checks
- Performance monitoring

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH=/home/QuantNova/GrandModel:$PYTHONPATH
   ```

2. **PettingZoo API Failures**
   - Check action/observation space definitions
   - Verify agent lifecycle management
   - Ensure proper termination handling

3. **Performance Issues**
   - Profile step execution
   - Check memory leaks
   - Optimize matrix operations

4. **Environment Errors**
   - Validate configuration
   - Check component initialization
   - Verify data flow

### Debug Mode
```bash
# Run with debug output
pytest tests/test_strategic_env.py -v -s --tb=long

# Run single test with debugging
pytest tests/test_strategic_env.py::TestStrategicEnvironment::test_agent_turn_sequence -v -s
```

## Contributing

### Adding New Tests
1. Follow naming convention: `test_<functionality>`
2. Use appropriate markers: `@pytest.mark.strategic`
3. Include docstrings with test purpose
4. Add performance assertions
5. Test error conditions

### Test Guidelines
- Test one functionality per test function
- Use descriptive test names
- Include both positive and negative cases
- Mock external dependencies when needed
- Ensure tests are deterministic

## Requirements

### Dependencies
- pytest >= 6.0
- pytest-timeout
- pytest-cov (optional)
- pettingzoo
- numpy
- torch

### Installation
```bash
pip install -r requirements_test.txt
```

## Results and Reporting

### Test Reports
- Terminal output with `-v` flag
- HTML coverage reports with `--cov-report=html`
- JUnit XML with `--junitxml=report.xml`

### Performance Reports
- Step timing analysis
- Memory usage profiles
- Environment comparison metrics
- Compliance validation results