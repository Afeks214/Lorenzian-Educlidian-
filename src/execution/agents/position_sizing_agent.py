"""
Position Sizing Agent (π₁) for Execution Engine MARL System

Implements optimal position sizing using modified Kelly Criterion with risk adjustment.
Designed for ultra-low latency (<200μs) execution with discrete action space.

Mathematical Foundation:
f* = (bp - q) / b - λ * σ²

Where:
- f*: Optimal fraction of capital to risk
- b: Expected payoff ratio (average_win / average_loss)
- p: Probability of winning trade (from tactical confidence)
- q: Probability of losing trade (1 - p)
- λ: Risk aversion parameter (default: 2.0)
- σ²: Portfolio volatility

Action Space: Discrete(5) → {0: 0 contracts, 1: 1 contract, 2: 2 contracts, 3: 3 contracts, 4: 5 contracts}
Observation Space: Box(-∞, +∞, (15,)) → Full execution context
Target Inference Latency: <200μs
Maximum Account Risk: 2%
Maximum Position Size: 5 contracts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List
import time
import structlog
from dataclasses import dataclass
from enum import IntEnum

from src.core.event_bus import EventBus, Event, EventType
from src.core.events import Event as CoreEvent

logger = structlog.get_logger()


class PositionSizeAction(IntEnum):
    """Discrete position size actions"""
    ZERO_CONTRACTS = 0      # 0 contracts
    ONE_CONTRACT = 1        # 1 contract
    TWO_CONTRACTS = 2       # 2 contracts  
    THREE_CONTRACTS = 3     # 3 contracts
    FIVE_CONTRACTS = 4      # 5 contracts


@dataclass
class ExecutionContext:
    """
    15-dimensional execution context vector
    
    Features:
    0-2: Market microstructure (bid_ask_spread, order_book_imbalance, market_impact)
    3-5: Volatility measures (realized_vol, implied_vol, vol_of_vol) 
    6-8: Liquidity measures (market_depth, volume_profile, liquidity_cost)
    9-11: Risk metrics (portfolio_var, correlation_risk, leverage_ratio)
    12-14: Performance metrics (pnl_unrealized, drawdown_current, confidence_score)
    """
    # Market microstructure
    bid_ask_spread: float = 0.0
    order_book_imbalance: float = 0.0  
    market_impact: float = 0.0
    
    # Volatility measures
    realized_vol: float = 0.0
    implied_vol: float = 0.0
    vol_of_vol: float = 0.0
    
    # Liquidity measures  
    market_depth: float = 0.0
    volume_profile: float = 0.0
    liquidity_cost: float = 0.0
    
    # Risk metrics
    portfolio_var: float = 0.0
    correlation_risk: float = 0.0
    leverage_ratio: float = 0.0
    
    # Performance metrics
    pnl_unrealized: float = 0.0
    drawdown_current: float = 0.0
    confidence_score: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input"""
        return torch.tensor([
            self.bid_ask_spread, self.order_book_imbalance, self.market_impact,
            self.realized_vol, self.implied_vol, self.vol_of_vol,
            self.market_depth, self.volume_profile, self.liquidity_cost,
            self.portfolio_var, self.correlation_risk, self.leverage_ratio,
            self.pnl_unrealized, self.drawdown_current, self.confidence_score
        ], dtype=torch.float32)


@dataclass 
class KellyCalculationResult:
    """Result of Kelly Criterion calculation"""
    optimal_fraction: float
    win_probability: float
    expected_payoff_ratio: float
    risk_adjustment: float
    volatility_penalty: float
    final_position_size: int
    calculation_time_ns: int


class PositionSizingNetwork(nn.Module):
    """
    Ultra-fast neural network for position sizing decisions
    
    Architecture: 15D input → 256→128→64→5 output
    Target inference time: <200μs
    """
    
    def __init__(self, input_dim: int = 15, hidden_dims: List[int] = None, output_dim: int = 5):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build sequential network for maximum speed
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),  # In-place for memory efficiency
                nn.LayerNorm(hidden_dim)  # Stabilizes training
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize for fast convergence
        self._initialize_weights()
        
        # JIT compilation for speed
        self._compiled = False
        
    def _initialize_weights(self):
        """Initialize weights for fast convergence and stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimization for inference speed
        
        Args:
            x: Input tensor [batch_size, 15] or [15] for single inference
            
        Returns:
            Action probabilities [batch_size, 5] or [5]
        """
        # Ensure correct input shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Forward pass
        logits = self.network(x)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        if squeeze_output:
            probs = probs.squeeze(0)
            
        return probs
    
    def compile_for_inference(self):
        """Compile network for maximum inference speed"""
        if not self._compiled:
            try:
                # JIT script compilation
                example_input = torch.randn(1, self.input_dim)
                self.traced_model = torch.jit.trace(self, example_input)
                self._compiled = True
                logger.info("Position sizing network compiled for inference")
            except Exception as e:
                logger.warning("Failed to compile network", error=str(e))
                
    def fast_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Ultra-fast inference path"""
        if self._compiled and hasattr(self, 'traced_model'):
            # Ensure correct input shape for traced model
            if x.dim() == 1:
                result = self.traced_model(x.unsqueeze(0))
                return result.squeeze(0)  # Remove batch dimension for single input
            else:
                return self.traced_model(x)
        else:
            return self.forward(x)


class KellyCalculator:
    """
    High-performance Kelly Criterion calculator with risk adjustment
    
    Implements: f* = (bp - q) / b - λ * σ²
    """
    
    def __init__(self, risk_aversion: float = 2.0, max_position_fraction: float = 0.25):
        self.risk_aversion = risk_aversion
        self.max_position_fraction = max_position_fraction
        
    def calculate_optimal_size(self,
                             confidence: float,
                             expected_payoff_ratio: float,
                             account_equity: float,
                             current_volatility: float,
                             contract_value: float = 20.0) -> KellyCalculationResult:
        """
        Calculate optimal position size using modified Kelly Criterion
        
        Args:
            confidence: Tactical decision confidence [0.0, 1.0]
            expected_payoff_ratio: E[Win] / E[Loss] ratio
            account_equity: Current account value in USD
            current_volatility: Portfolio volatility estimate
            contract_value: Value per contract in USD
            
        Returns:
            KellyCalculationResult with optimal position size
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Input validation
            confidence = max(0.0, min(1.0, confidence))
            expected_payoff_ratio = max(0.001, expected_payoff_ratio)  # Prevent division by zero
            current_volatility = max(0.001, current_volatility)
            
            # Kelly fraction calculation
            win_prob = confidence
            loss_prob = 1.0 - confidence
            
            if expected_payoff_ratio <= 0 or win_prob <= 0:
                return self._zero_position_result(start_time)
                
            # Basic Kelly fraction: f = (bp - q) / b
            basic_kelly_fraction = (
                (expected_payoff_ratio * win_prob - loss_prob) / expected_payoff_ratio
            )
            
            # Risk adjustment for volatility: f* = f - λ * σ²
            volatility_penalty = self.risk_aversion * (current_volatility ** 2)
            adjusted_fraction = basic_kelly_fraction - volatility_penalty
            
            # Apply safety constraints
            optimal_fraction = max(0.0, min(self.max_position_fraction, adjusted_fraction))
            
            # Convert to position size in contracts
            max_dollar_risk = account_equity * optimal_fraction
            max_contracts = int(max_dollar_risk / contract_value)
            
            # Discrete action space mapping: {0, 1, 2, 3, 5}
            position_size = self._map_to_discrete_action(max_contracts)
            
            end_time = time.perf_counter_ns()
            
            return KellyCalculationResult(
                optimal_fraction=optimal_fraction,
                win_probability=win_prob,
                expected_payoff_ratio=expected_payoff_ratio,
                risk_adjustment=volatility_penalty,
                volatility_penalty=volatility_penalty,
                final_position_size=position_size,
                calculation_time_ns=end_time - start_time
            )
            
        except Exception as e:
            logger.error("Error in Kelly calculation", error=str(e))
            return self._zero_position_result(start_time)
    
    def _map_to_discrete_action(self, contracts: int) -> int:
        """Map continuous contract count to discrete action space"""
        if contracts <= 0:
            return 0  # 0 contracts
        elif contracts == 1:
            return 1  # 1 contract
        elif contracts == 2:
            return 2  # 2 contracts
        elif contracts <= 3:
            return 3  # 3 contracts
        else:
            return 4  # 5 contracts (maximum)
    
    def _zero_position_result(self, start_time: int) -> KellyCalculationResult:
        """Return zero position result for error cases"""
        end_time = time.perf_counter_ns()
        return KellyCalculationResult(
            optimal_fraction=0.0,
            win_probability=0.0,
            expected_payoff_ratio=0.0,
            risk_adjustment=0.0,
            volatility_penalty=0.0,
            final_position_size=0,
            calculation_time_ns=end_time - start_time
        )


class PositionSizingAgent:
    """
    Position Sizing Agent (π₁) for Execution Engine MARL System
    
    High-performance agent for optimal position sizing with:
    - Modified Kelly Criterion calculation
    - <200μs inference latency
    - Safety constraints (max 2% account risk, max 5 contracts)
    - Real-time performance monitoring
    """
    
    def __init__(self, config: Dict[str, Any], event_bus: Optional[EventBus] = None):
        """
        Initialize Position Sizing Agent
        
        Args:
            config: Agent configuration
            event_bus: Event bus for communication
        """
        self.config = config
        self.event_bus = event_bus
        
        # Agent parameters
        self.max_account_risk = config.get('max_account_risk', 0.02)  # 2%
        self.max_contracts = config.get('max_contracts', 5)
        self.risk_aversion = config.get('risk_aversion', 2.0)
        self.learning_rate = config.get('learning_rate', 1e-4)
        
        # Initialize components
        self.network = PositionSizingNetwork(
            input_dim=15,
            hidden_dims=config.get('hidden_dims', [256, 128, 64]),
            output_dim=5
        )
        
        self.kelly_calculator = KellyCalculator(
            risk_aversion=self.risk_aversion,
            max_position_fraction=self.max_account_risk
        )
        
        # Performance tracking
        self.inference_times = []
        self.decisions_made = 0
        self.safety_violations = 0
        self.kelly_calculations = 0
        
        # Compile network for speed
        self.network.compile_for_inference()
        
        logger.info("Position Sizing Agent (π₁) initialized",
                   max_account_risk=self.max_account_risk,
                   max_contracts=self.max_contracts,
                   risk_aversion=self.risk_aversion)
    
    def decide_position_size(self, 
                           execution_context: ExecutionContext,
                           account_equity: float,
                           expected_payoff_ratio: float = 1.5) -> Tuple[int, Dict[str, Any]]:
        """
        Decide optimal position size for current market conditions
        
        Args:
            execution_context: 15-dimensional execution context
            account_equity: Current account equity in USD
            expected_payoff_ratio: Expected win/loss ratio
            
        Returns:
            Tuple of (position_size_contracts, decision_info)
        """
        start_time = time.perf_counter_ns()
        
        try:
            # Convert context to tensor
            context_tensor = execution_context.to_tensor()
            
            # Neural network inference
            with torch.no_grad():
                action_probs = self.network.fast_inference(context_tensor)
                
            # Get most probable action
            action = torch.argmax(action_probs).item()
            confidence = action_probs[action].item() if action_probs.dim() == 1 else action_probs[0, action].item()
            
            # Kelly Criterion calculation for validation
            kelly_result = self.kelly_calculator.calculate_optimal_size(
                confidence=confidence,
                expected_payoff_ratio=expected_payoff_ratio,
                account_equity=account_equity,
                current_volatility=execution_context.realized_vol
            )
            
            # Map action to contract count
            position_size = self._action_to_contracts(action)
            
            # Apply safety constraints
            position_size = self._apply_safety_constraints(
                position_size, account_equity, execution_context
            )
            
            # Performance tracking
            end_time = time.perf_counter_ns()
            inference_time_ns = end_time - start_time
            self.inference_times.append(inference_time_ns)
            self.decisions_made += 1
            self.kelly_calculations += 1
            
            # Decision info
            decision_info = {
                'action': action,
                'confidence': confidence,
                'action_probs': action_probs.flatten().tolist(),
                'kelly_fraction': kelly_result.optimal_fraction,
                'kelly_position_size': kelly_result.final_position_size,
                'inference_time_ns': inference_time_ns,
                'inference_time_us': inference_time_ns / 1000,
                'safety_applied': position_size != self._action_to_contracts(action),
                'expected_payoff_ratio': expected_payoff_ratio,
                'volatility_penalty': kelly_result.volatility_penalty
            }
            
            # Log performance if needed
            if self.decisions_made % 1000 == 0:
                self._log_performance_stats()
                
            # Emit event if event bus available
            if self.event_bus:
                self._emit_decision_event(position_size, decision_info)
                
            return position_size, decision_info
            
        except Exception as e:
            logger.error("Error in position sizing decision", error=str(e))
            return 0, {'error': str(e), 'inference_time_ns': time.perf_counter_ns() - start_time}
    
    def _action_to_contracts(self, action: int) -> int:
        """Map action index to contract count"""
        action_map = {
            0: 0,  # 0 contracts
            1: 1,  # 1 contract
            2: 2,  # 2 contracts
            3: 3,  # 3 contracts
            4: 5   # 5 contracts
        }
        return action_map.get(action, 0)
    
    def _apply_safety_constraints(self, 
                                position_size: int,
                                account_equity: float,
                                execution_context: ExecutionContext) -> int:
        """Apply safety constraints to position size"""
        original_size = position_size
        
        # Max contracts constraint
        position_size = min(position_size, self.max_contracts)
        
        # Max account risk constraint (2%)
        contract_value = 20.0  # Assumed contract value
        max_risk_dollars = account_equity * self.max_account_risk
        max_risk_contracts = int(max_risk_dollars / contract_value)
        position_size = min(position_size, max_risk_contracts)
        
        # Portfolio VaR constraint
        if execution_context.portfolio_var > 0.015:  # 1.5% VaR threshold
            position_size = max(0, position_size - 1)  # Reduce by 1 contract
            
        # High correlation risk constraint
        if execution_context.correlation_risk > 0.8:
            position_size = max(0, position_size - 1)  # Reduce by 1 contract
            
        # Current drawdown constraint
        if execution_context.drawdown_current > 0.1:  # 10% drawdown
            position_size = max(0, position_size - 2)  # Reduce by 2 contracts
            
        # Track safety violations
        if position_size != original_size:
            self.safety_violations += 1
            
        return max(0, position_size)
    
    def _emit_decision_event(self, position_size: int, decision_info: Dict[str, Any]):
        """Emit position sizing decision event"""
        try:
            event = CoreEvent(
                type=EventType.AGENT_DECISION,
                data={
                    'agent': 'position_sizing_agent',
                    'position_size': position_size,
                    'decision_info': decision_info,
                    'timestamp': time.time()
                }
            )
            self.event_bus.emit(event)
        except Exception as e:
            logger.warning("Failed to emit decision event", error=str(e))
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        if not self.inference_times:
            return
            
        avg_time_ns = np.mean(self.inference_times[-1000:])  # Last 1000 decisions
        max_time_ns = np.max(self.inference_times[-1000:])
        p95_time_ns = np.percentile(self.inference_times[-1000:], 95)
        
        safety_violation_rate = self.safety_violations / self.decisions_made
        
        logger.info("Position Sizing Agent Performance",
                   decisions_made=self.decisions_made,
                   avg_inference_time_us=avg_time_ns / 1000,
                   max_inference_time_us=max_time_ns / 1000,
                   p95_inference_time_us=p95_time_ns / 1000,
                   safety_violation_rate=f"{safety_violation_rate:.3f}",
                   target_met=avg_time_ns < 200_000)  # 200μs target
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.inference_times:
            return {}
            
        recent_times = self.inference_times[-1000:] if len(self.inference_times) > 1000 else self.inference_times
        
        return {
            'total_decisions': self.decisions_made,
            'total_kelly_calculations': self.kelly_calculations,
            'safety_violations': self.safety_violations,
            'safety_violation_rate': self.safety_violations / max(1, self.decisions_made),
            'avg_inference_time_ns': np.mean(recent_times),
            'avg_inference_time_us': np.mean(recent_times) / 1000,
            'max_inference_time_us': np.max(recent_times) / 1000,
            'p50_inference_time_us': np.percentile(recent_times, 50) / 1000,
            'p95_inference_time_us': np.percentile(recent_times, 95) / 1000,
            'p99_inference_time_us': np.percentile(recent_times, 99) / 1000,
            'target_200us_met': np.mean(recent_times) < 200_000,
            'network_compiled': self.network._compiled
        }
    
    def update_parameters(self, new_config: Dict[str, Any]):
        """Update agent parameters dynamically"""
        if 'max_account_risk' in new_config:
            self.max_account_risk = new_config['max_account_risk']
            
        if 'max_contracts' in new_config:
            self.max_contracts = new_config['max_contracts']
            
        if 'risk_aversion' in new_config:
            self.risk_aversion = new_config['risk_aversion']
            self.kelly_calculator.risk_aversion = new_config['risk_aversion']
            
        logger.info("Position Sizing Agent parameters updated", **new_config)
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.inference_times.clear()
        self.decisions_made = 0
        self.safety_violations = 0
        self.kelly_calculations = 0
        logger.info("Position Sizing Agent metrics reset")
    
    def validate_kelly_implementation(self, test_cases: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Validate Kelly Criterion implementation against known test cases
        
        Args:
            test_cases: List of test cases with 'confidence', 'payoff_ratio', 'expected_fraction'
            
        Returns:
            Validation results
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            kelly_result = self.kelly_calculator.calculate_optimal_size(
                confidence=test_case['confidence'],
                expected_payoff_ratio=test_case['payoff_ratio'],
                account_equity=10000.0,  # Standard test equity
                current_volatility=0.2   # Standard test volatility
            )
            
            expected = test_case['expected_fraction']
            actual = kelly_result.optimal_fraction
            error = abs(actual - expected)
            error_pct = (error / max(0.001, expected)) * 100
            
            results.append({
                'test_case': i,
                'expected': expected,
                'actual': actual,
                'error': error,
                'error_pct': error_pct,
                'calculation_time_ns': kelly_result.calculation_time_ns
            })
        
        # Summary statistics
        errors = [r['error'] for r in results]
        calc_times = [r['calculation_time_ns'] for r in results]
        
        validation_summary = {
            'total_tests': len(test_cases),
            'max_error': max(errors) if errors else 0,
            'avg_error': np.mean(errors) if errors else 0,
            'max_error_pct': max([r['error_pct'] for r in results]) if results else 0,
            'avg_calculation_time_ns': np.mean(calc_times) if calc_times else 0,
            'all_tests_passed': max(errors) < 0.05 if errors else False,  # 5% tolerance
            'detailed_results': results
        }
        
        logger.info("Kelly Criterion validation completed",
                   tests=len(test_cases),
                   max_error=validation_summary['max_error'],
                   avg_error=validation_summary['avg_error'],
                   passed=validation_summary['all_tests_passed'])
        
        return validation_summary


# Factory function for easy instantiation
def create_position_sizing_agent(config: Dict[str, Any], event_bus: Optional[EventBus] = None) -> PositionSizingAgent:
    """Create and initialize a Position Sizing Agent"""
    return PositionSizingAgent(config, event_bus)


# Performance benchmark for validation
def benchmark_position_sizing_performance(agent: PositionSizingAgent, 
                                        num_iterations: int = 10000) -> Dict[str, Any]:
    """
    Benchmark position sizing agent performance
    
    Args:
        agent: Position sizing agent to benchmark
        num_iterations: Number of iterations to run
        
    Returns:
        Performance benchmark results
    """
    logger.info("Starting position sizing performance benchmark", iterations=num_iterations)
    
    # Reset metrics
    agent.reset_metrics()
    
    # Generate random execution contexts for testing
    contexts = []
    for _ in range(num_iterations):
        context = ExecutionContext(
            bid_ask_spread=np.random.uniform(0.0001, 0.001),
            order_book_imbalance=np.random.uniform(-0.5, 0.5),
            market_impact=np.random.uniform(0.0001, 0.01),
            realized_vol=np.random.uniform(0.1, 0.5),
            implied_vol=np.random.uniform(0.1, 0.6),
            vol_of_vol=np.random.uniform(0.01, 0.1),
            market_depth=np.random.uniform(100, 10000),
            volume_profile=np.random.uniform(0.1, 2.0),
            liquidity_cost=np.random.uniform(0.0001, 0.01),
            portfolio_var=np.random.uniform(0.005, 0.025),
            correlation_risk=np.random.uniform(0.1, 0.9),
            leverage_ratio=np.random.uniform(1.0, 3.0),
            pnl_unrealized=np.random.uniform(-1000, 1000),
            drawdown_current=np.random.uniform(0.0, 0.2),
            confidence_score=np.random.uniform(0.3, 0.9)
        )
        contexts.append(context)
    
    # Run benchmark
    start_time = time.perf_counter()
    
    for context in contexts:
        position_size, decision_info = agent.decide_position_size(
            execution_context=context,
            account_equity=10000.0,
            expected_payoff_ratio=1.5
        )
    
    end_time = time.perf_counter()
    total_time_s = end_time - start_time
    
    # Get performance metrics
    metrics = agent.get_performance_metrics()
    
    # Benchmark results
    benchmark_results = {
        'total_iterations': num_iterations,
        'total_time_s': total_time_s,
        'iterations_per_second': num_iterations / total_time_s,
        'avg_time_per_iteration_us': (total_time_s / num_iterations) * 1_000_000,
        'target_200us_met': metrics.get('target_200us_met', False),
        'performance_metrics': metrics
    }
    
    logger.info("Position sizing performance benchmark completed",
               iterations_per_second=benchmark_results['iterations_per_second'],
               avg_time_us=benchmark_results['avg_time_per_iteration_us'],
               target_met=benchmark_results['target_200us_met'])
    
    return benchmark_results