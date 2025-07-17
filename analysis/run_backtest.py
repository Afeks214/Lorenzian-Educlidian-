"""
Backtesting Framework for Strategic MARL System

Allows comparison of different agents (baseline vs MARL) on historical
or synthetic market data.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import yaml
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import asyncio
from datetime import datetime, timedelta
import warnings
from numba import jit, cuda
from collections import deque
from scipy import stats
from itertools import product
import pickle
import os
from contextlib import contextmanager
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import components
from environment.strategic_env import StrategicMarketEnv
from baselines import RuleBasedAgent, RandomAgent
from models.architectures import MLMIActor, NWRQKActor, MMDActor
from tests.mocks import MockMatrixAssembler, MockSynergyDetector
from .metrics import calculate_all_metrics, PerformanceMetrics, compare_performance
from .market_simulation import MarketSimulator, Order, OrderType, OrderSide, MarketImpactModel
from .risk_integration import RiskIntegrator, create_risk_integrator, RiskMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Enhanced configuration for backtesting"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 10000.0
    position_size: float = 0.1  # Fraction of capital per trade
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    data_source: str = "synthetic"  # "synthetic" or "historical"
    episodes: int = 100
    max_steps_per_episode: int = 1000
    random_seed: Optional[int] = 42
    
    # Walk-forward optimization
    walk_forward_enabled: bool = False
    train_period_days: int = 252  # 1 year
    test_period_days: int = 63   # 3 months
    optimization_lookback: int = 504  # 2 years
    
    # Multi-timeframe settings
    multi_timeframe: bool = False
    tactical_timeframe: str = "5m"  # 5-minute bars
    strategic_timeframe: str = "30m"  # 30-minute bars
    alignment_method: str = "interpolate"  # "interpolate" or "downsample"
    
    # Monte Carlo simulation
    monte_carlo_runs: int = 1000
    monte_carlo_enabled: bool = False
    confidence_intervals: List[float] = field(default_factory=lambda: [0.05, 0.95])
    
    # Parallel execution
    parallel_enabled: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 10
    
    # Market simulation
    realistic_execution: bool = True
    market_impact_enabled: bool = True
    latency_simulation: bool = True
    
    # Risk management
    dynamic_position_sizing: bool = True
    kelly_criterion: bool = True
    correlation_adjustment: bool = True
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    
    # Performance optimization
    jit_compilation: bool = True
    use_gpu: bool = False
    memory_efficient: bool = True
    cache_enabled: bool = True
    
    
@dataclass
class BacktestResult:
    """Enhanced results from a backtest run"""
    agent_name: str
    equity_curve: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    actions: List[np.ndarray]
    metrics: PerformanceMetrics
    execution_time: float
    config: BacktestConfig
    
    # Walk-forward results
    walk_forward_results: Optional[List[Dict[str, Any]]] = None
    optimization_history: Optional[List[Dict[str, Any]]] = None
    
    # Multi-timeframe results
    tactical_metrics: Optional[PerformanceMetrics] = None
    strategic_metrics: Optional[PerformanceMetrics] = None
    timeframe_correlation: Optional[float] = None
    
    # Monte Carlo results
    monte_carlo_stats: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    
    # Risk analysis
    risk_metrics: Optional[Dict[str, float]] = None
    drawdown_analysis: Optional[Dict[str, Any]] = None
    position_sizing_stats: Optional[Dict[str, float]] = None
    
    # Execution analysis
    execution_stats: Optional[Dict[str, float]] = None
    market_impact_cost: Optional[float] = None
    latency_impact: Optional[float] = None
    
    # Performance stats
    memory_usage_mb: Optional[float] = None
    cpu_usage_pct: Optional[float] = None
    gpu_utilization: Optional[float] = None
    

class BacktestRunner:
    """
    Enhanced backtesting engine with walk-forward optimization, multi-timeframe analysis,
    Monte Carlo simulation, and parallel execution capabilities.
    """
    
    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced backtest runner
        
        Args:
            env_config: Configuration for the environment
        """
        self.env_config = env_config or self._get_default_env_config()
        self.results: Dict[str, BacktestResult] = {}
        
        # Initialize components
        self.market_simulator = None
        self.risk_integrator = None
        self.performance_cache = {}
        
        # Initialize market simulator and risk integrator
        self._initialize_components()
        
        # Walk-forward optimization state
        self.optimization_results = {}
        self.parameter_history = []
        
        # Multi-timeframe state
        self.tactical_results = {}
        self.strategic_results = {}
        
        # Monte Carlo state
        self.mc_results = {}
        self.bootstrap_samples = []
        
        # Performance monitoring
        self.execution_stats = {
            'total_backtests': 0,
            'avg_time_per_backtest': 0.0,
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_utilization': []
        }
        
    def _get_default_env_config(self) -> Dict[str, Any]:
        """Get default environment configuration"""
        return {
            'matrix_shape': [48, 13],
            'max_timesteps': 1000,
            'feature_indices': {
                'mlmi_expert': [0, 1, 9, 10],
                'nwrqk_expert': [2, 3, 4, 5],
                'regime_expert': [10, 11, 12]
            }
        }
    
    def _initialize_components(self):
        """Initialize market simulator and risk integrator"""
        
        # Default symbols for testing
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        # Initialize market simulator
        try:
            self.market_simulator = MarketSimulator(
                symbols=symbols,
                initial_prices={symbol: 100.0 for symbol in symbols},
                volatilities={symbol: 0.02 for symbol in symbols},
                spreads={symbol: 0.01 for symbol in symbols},
                market_impact_model=MarketImpactModel(
                    model_type="square_root",
                    temporary_impact=0.001,
                    permanent_impact=0.0005
                )
            )
            logger.info("Market simulator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize market simulator: {e}")
            self.market_simulator = None
        
        # Initialize risk integrator  
        try:
            self.risk_integrator = create_risk_integrator(
                symbols=symbols,
                initial_capital=10000.0,
                risk_config={
                    'max_drawdown_limit': 0.20,
                    'kelly_multiplier': 0.25,
                    'max_position_size': 0.3,
                    'correlation_adjustment': True,
                    'volatility_adjustment': True
                }
            )
            logger.info("Risk integrator initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize risk integrator: {e}")
            self.risk_integrator = None
        
    def run_baseline_agents(
        self, 
        config: BacktestConfig
    ) -> Dict[str, BacktestResult]:
        """
        Run baseline agents (rule-based and random)
        
        Args:
            config: Backtest configuration
            
        Returns:
            Dictionary of results by agent name
        """
        results = {}
        
        # Rule-based agent
        logger.info("Running rule-based agent backtest...")
        rule_agent = RuleBasedAgent()
        results['rule_based'] = self._run_agent(rule_agent, config, "rule_based")
        
        # Enhanced rule-based agent
        logger.info("Running enhanced rule-based agent backtest...")
        enhanced_agent = RuleBasedAgent({'volatility_threshold': 1.5})
        results['enhanced_rule'] = self._run_agent(enhanced_agent, config, "enhanced_rule")
        
        # Random agent
        logger.info("Running random agent backtest...")
        random_agent = RandomAgent({'random_seed': config.random_seed})
        results['random'] = self._run_agent(random_agent, config, "random")
        
        return results
        
    def run_marl_agent(
        self,
        model_checkpoint: Union[str, Dict[str, Any]],
        config: BacktestConfig
    ) -> BacktestResult:
        """
        Run MARL agent from checkpoint
        
        Args:
            model_checkpoint: Path to checkpoint or loaded state dict
            config: Backtest configuration
            
        Returns:
            Backtest result
        """
        logger.info("Running MARL agent backtest...")
        
        # Load models
        if isinstance(model_checkpoint, str):
            checkpoint = torch.load(model_checkpoint)
        else:
            checkpoint = model_checkpoint
            
        # Create MARL agent wrapper
        marl_agent = MARLAgentWrapper(checkpoint, self.env_config)
        
        # Run backtest
        result = self._run_agent(marl_agent, config, "marl")
        
        return result
        
    def _run_agent(
        self,
        agent: Any,
        config: BacktestConfig,
        agent_name: str
    ) -> BacktestResult:
        """
        Run a single agent through backtest
        
        Args:
            agent: Agent to test
            config: Backtest configuration
            agent_name: Name for results
            
        Returns:
            Backtest result
        """
        start_time = time.time()
        
        # Set random seed
        if config.random_seed:
            np.random.seed(config.random_seed)
            
        # Initialize tracking
        equity_curve = [config.initial_capital]
        returns = []
        positions = []
        actions = []
        
        # Create environment
        env = StrategicMarketEnv(self.env_config)
        
        # Use mock components for synthetic data
        if config.data_source == "synthetic":
            env.matrix_assembler = MockMatrixAssembler()
            env.synergy_detector = MockSynergyDetector()
            
        # Run episodes
        for episode in range(config.episodes):
            env.reset()
            
            # Set scenario for synthetic data
            if config.data_source == "synthetic":
                # Vary scenarios
                scenarios = ['bullish', 'bearish', 'neutral', 'high_volatility']
                scenario = scenarios[episode % len(scenarios)]
                env.matrix_assembler.set_scenario(scenario)
                
                # Vary synergy patterns
                synergy_scenarios = ['none', 'type1_bullish', 'type1_bearish', 'type2_bullish']
                synergy_scenario = synergy_scenarios[episode % len(synergy_scenarios)]
                env.synergy_detector.set_scenario(synergy_scenario)
                
            episode_returns = []
            episode_positions = []
            
            # Run episode
            step_count = 0
            for agent_idx in env.agent_iter():
                if step_count >= config.max_steps_per_episode:
                    break
                    
                # Get observation
                obs = env.observe(agent_idx)
                
                # Get action from agent
                if hasattr(agent, 'get_action'):
                    action = agent.get_action(obs)
                else:
                    # For environment agents
                    action = np.array([0.33, 0.34, 0.33])
                    
                actions.append(action)
                
                # Step environment
                env.step(action)
                
                # Calculate returns every 3 steps (full cycle)
                if (step_count + 1) % 3 == 0:
                    # Simulate trading logic
                    position = self._calculate_position(actions[-3:], config)
                    pnl = self._simulate_pnl(position, env, config)
                    
                    # Update equity
                    current_equity = equity_curve[-1]
                    new_equity = current_equity * (1 + pnl)
                    equity_curve.append(new_equity)
                    
                    episode_returns.append(pnl)
                    episode_positions.append(position)
                    
                step_count += 1
                
                if env.terminations[agent_idx]:
                    break
                    
            # Record episode results
            if episode_returns:
                returns.extend(episode_returns)
                positions.extend(episode_positions)
                
        # Calculate metrics
        equity_curve = np.array(equity_curve)
        returns = np.array(returns)
        positions = np.array(positions)
        
        metrics = calculate_all_metrics(
            equity_curve=equity_curve,
            returns=returns,
            periods_per_year=252 * 78  # Assuming 78 5-min bars per day
        )
        
        execution_time = time.time() - start_time
        
        result = BacktestResult(
            agent_name=agent_name,
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            actions=actions,
            metrics=metrics,
            execution_time=execution_time,
            config=config
        )
        
        self.results[agent_name] = result
        
        logger.info(f"Completed {agent_name} backtest in {execution_time:.2f}s")
        logger.info(f"Performance: {metrics}")
        
        return result
        
    def _calculate_position(
        self,
        agent_actions: List[np.ndarray],
        config: BacktestConfig
    ) -> float:
        """
        Calculate position from agent actions with risk management
        
        Args:
            agent_actions: List of 3 agent actions
            config: Backtest configuration
            
        Returns:
            Position size (-1 to 1)
        """
        # Average the agent actions
        avg_action = np.mean(agent_actions, axis=0)
        
        # Convert to position
        # action[0] = bearish, action[1] = neutral, action[2] = bullish
        base_position = (avg_action[2] - avg_action[0]) * config.position_size
        
        # Apply risk management if enabled
        if config.dynamic_position_sizing and self.risk_integrator:
            # Calculate signal strength
            signal_strength = avg_action[2] - avg_action[0]  # -1 to 1
            
            # Estimate win probability and payout ratio from recent performance
            win_prob = 0.55  # Default optimistic estimate
            payout_ratio = 1.2  # Default 1.2:1 reward/risk
            
            # Calculate risk-adjusted position
            risk_adjusted_position = self.risk_integrator.calculate_position_size(
                symbol='AAPL',  # Default symbol
                signal_strength=signal_strength,
                win_probability=win_prob,
                payout_ratio=payout_ratio,
                current_price=100.0,  # Default price
                volatility=0.02  # Default volatility
            )
            
            # Use risk-adjusted position if available
            if risk_adjusted_position != 0:
                return risk_adjusted_position
        
        return np.clip(base_position, -1.0, 1.0)
        
    def _simulate_pnl(
        self,
        position: float,
        env: StrategicMarketEnv,
        config: BacktestConfig = None
    ) -> float:
        """
        Enhanced P&L simulation with realistic execution costs
        
        Args:
            position: Position size
            env: Environment (for market data)
            config: Backtest configuration
            
        Returns:
            P&L as percentage return
        """
        
        # Use realistic execution if enabled
        if config and config.realistic_execution and self.market_simulator:
            return self._simulate_realistic_pnl(position, config)
        
        # Fallback to simple simulation
        market_return = np.random.normal(0.0001, 0.001)  # Small random return
        
        # Apply position
        pnl = position * market_return
        
        # Add transaction costs if position changed
        if position != 0:
            pnl -= config.transaction_cost if config else 0.001
            
        return pnl
    
    def _simulate_realistic_pnl(
        self,
        position: float,
        config: BacktestConfig
    ) -> float:
        """
        Simulate P&L using realistic market execution
        
        Args:
            position: Position size
            config: Backtest configuration
            
        Returns:
            P&L as percentage return
        """
        
        if not self.market_simulator:
            return self._simulate_pnl(position, None, config)
        
        # Update market data
        self.market_simulator.update_market_data(1.0)
        
        # Simulate order execution
        symbol = 'AAPL'  # Default symbol for testing
        order_side = OrderSide.BUY if position > 0 else OrderSide.SELL
        
        # Create market order
        order = Order(
            order_id=f"order_{datetime.now().timestamp()}",
            symbol=symbol,
            side=order_side,
            order_type=OrderType.MARKET,
            quantity=abs(position * 1000),  # Scale to shares
            timestamp=datetime.now()
        )
        
        # Submit order
        self.market_simulator.submit_order(order)
        
        # Wait for execution (simplified)
        time.sleep(0.01)  # Small delay for execution
        
        # Get execution stats
        fills = self.market_simulator.get_fills(order.order_id)
        
        if fills:
            # Calculate total execution cost
            total_cost = sum(fill.commission + fill.market_impact for fill in fills)
            total_notional = sum(fill.quantity * fill.price for fill in fills)
            
            # Convert to percentage
            execution_cost_pct = total_cost / total_notional if total_notional > 0 else 0
            
            # Simulate market return
            market_return = np.random.normal(0.0001, 0.001)
            
            # Apply position and subtract execution costs
            pnl = position * market_return - execution_cost_pct
            
            return pnl
        
        # Fallback if no fills
        return self._simulate_pnl(position, None, config)
        
    def compare_results(
        self,
        agent_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare results across agents
        
        Args:
            agent_names: List of agents to compare (all if None)
            
        Returns:
            Comparison DataFrame
        """
        if agent_names is None:
            agent_names = list(self.results.keys())
            
        comparison_data = {}
        for name in agent_names:
            if name in self.results:
                result = self.results[name]
                comparison_data[name] = result.metrics.to_dict()
                comparison_data[name]['execution_time'] = result.execution_time
                
        df = pd.DataFrame(comparison_data).T
        
        # Add ranking
        ranking_metrics = ['sharpe_ratio', 'total_return', 'calmar_ratio']
        for metric in ranking_metrics:
            if metric in df.columns:
                df[f'{metric}_rank'] = df[metric].rank(ascending=False)
                
        return df
        
    def run_walk_forward_optimization(
        self,
        agent_factory: Callable,
        config: BacktestConfig,
        parameter_space: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization
        
        Args:
            agent_factory: Function to create agent with parameters
            config: Backtest configuration
            parameter_space: Dictionary of parameter ranges to optimize
            
        Returns:
            Walk-forward optimization results
        """
        logger.info("Starting walk-forward optimization")
        
        # Generate parameter combinations
        param_names = list(parameter_space.keys())
        param_values = list(parameter_space.values())
        param_combinations = list(product(*param_values))
        
        # Create time windows
        total_periods = config.optimization_lookback + config.test_period_days
        windows = []
        
        for i in range(0, total_periods - config.train_period_days, config.test_period_days):
            train_start = i
            train_end = i + config.train_period_days
            test_start = train_end
            test_end = min(test_start + config.test_period_days, total_periods)
            
            if test_end > test_start:
                windows.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end
                })
        
        wfo_results = []
        
        for window_idx, window in enumerate(windows):
            logger.info(f"Processing window {window_idx + 1}/{len(windows)}")
            
            # Optimize on training window
            best_params = self._optimize_parameters(
                agent_factory, config, param_combinations, param_names, window
            )
            
            # Test on out-of-sample window
            test_result = self._test_parameters(
                agent_factory, config, best_params, window
            )
            
            wfo_results.append({
                'window': window_idx,
                'train_period': (window['train_start'], window['train_end']),
                'test_period': (window['test_start'], window['test_end']),
                'best_params': best_params,
                'test_metrics': test_result
            })
        
        return {
            'results': wfo_results,
            'parameter_stability': self._analyze_parameter_stability(wfo_results),
            'performance_consistency': self._analyze_performance_consistency(wfo_results)
        }
    
    def _optimize_parameters(
        self,
        agent_factory: Callable,
        config: BacktestConfig,
        param_combinations: List[Tuple],
        param_names: List[str],
        window: Dict[str, int]
    ) -> Dict[str, Any]:
        """Optimize parameters on training window"""
        
        best_sharpe = -np.inf
        best_params = {}
        
        # Use parallel processing if enabled
        if config.parallel_enabled:
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {}
                
                for params in param_combinations:
                    param_dict = dict(zip(param_names, params))
                    future = executor.submit(
                        self._evaluate_parameters,
                        agent_factory, param_dict, config, window
                    )
                    futures[future] = param_dict
                
                for future in as_completed(futures):
                    param_dict = futures[future]
                    try:
                        result = future.result()
                        if result['sharpe_ratio'] > best_sharpe:
                            best_sharpe = result['sharpe_ratio']
                            best_params = param_dict
                    except Exception as e:
                        logger.error(f"Parameter evaluation failed: {e}")
        else:
            # Sequential processing
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                result = self._evaluate_parameters(
                    agent_factory, param_dict, config, window
                )
                if result['sharpe_ratio'] > best_sharpe:
                    best_sharpe = result['sharpe_ratio']
                    best_params = param_dict
        
        return best_params
    
    def _evaluate_parameters(
        self,
        agent_factory: Callable,
        params: Dict[str, Any],
        config: BacktestConfig,
        window: Dict[str, int]
    ) -> Dict[str, float]:
        """Evaluate parameter set on given window"""
        
        # Create agent with parameters
        agent = agent_factory(params)
        
        # Run limited backtest on window
        window_config = BacktestConfig(
            **config.__dict__,
            episodes=config.episodes // 10,  # Reduced for optimization
            max_steps_per_episode=config.max_steps_per_episode // 2
        )
        
        result = self._run_agent(agent, window_config, f"opt_{hash(str(params))}")
        
        return {
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'total_return': result.metrics.total_return,
            'max_drawdown': result.metrics.max_drawdown,
            'calmar_ratio': result.metrics.calmar_ratio
        }
    
    def _test_parameters(
        self,
        agent_factory: Callable,
        config: BacktestConfig,
        params: Dict[str, Any],
        window: Dict[str, int]
    ) -> Dict[str, float]:
        """Test optimized parameters on out-of-sample window"""
        
        agent = agent_factory(params)
        result = self._run_agent(agent, config, f"test_{hash(str(params))}")
        
        return {
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'total_return': result.metrics.total_return,
            'max_drawdown': result.metrics.max_drawdown,
            'calmar_ratio': result.metrics.calmar_ratio
        }
    
    def _analyze_parameter_stability(self, wfo_results: List[Dict]) -> Dict[str, float]:
        """Analyze parameter stability across windows"""
        
        param_stability = {}
        
        # Extract all parameters
        all_params = {}
        for result in wfo_results:
            for param, value in result['best_params'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)
        
        # Calculate stability metrics
        for param, values in all_params.items():
            if len(values) > 1:
                param_stability[f"{param}_std"] = np.std(values)
                param_stability[f"{param}_cv"] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        
        return param_stability
    
    def _analyze_performance_consistency(self, wfo_results: List[Dict]) -> Dict[str, float]:
        """Analyze performance consistency across windows"""
        
        sharpe_ratios = [r['test_metrics']['sharpe_ratio'] for r in wfo_results]
        returns = [r['test_metrics']['total_return'] for r in wfo_results]
        
        return {
            'sharpe_consistency': np.std(sharpe_ratios),
            'return_consistency': np.std(returns),
            'positive_periods': sum(1 for r in returns if r > 0) / len(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_return': np.mean(returns)
        }
    
    def run_multi_timeframe_analysis(
        self,
        agent: Any,
        config: BacktestConfig
    ) -> Dict[str, BacktestResult]:
        """
        Run multi-timeframe backtesting analysis
        
        Args:
            agent: Agent to test
            config: Backtest configuration
            
        Returns:
            Results for each timeframe
        """
        logger.info("Starting multi-timeframe analysis")
        
        results = {}
        
        # Tactical timeframe (5-minute)
        tactical_config = BacktestConfig(
            **config.__dict__,
            max_steps_per_episode=config.max_steps_per_episode * 6,  # More steps for shorter timeframe
            episodes=config.episodes // 2  # Fewer episodes due to more steps
        )
        
        tactical_result = self._run_agent(agent, tactical_config, f"{agent.__class__.__name__}_tactical")
        results['tactical'] = tactical_result
        
        # Strategic timeframe (30-minute)
        strategic_config = BacktestConfig(
            **config.__dict__,
            max_steps_per_episode=config.max_steps_per_episode,
            episodes=config.episodes
        )
        
        strategic_result = self._run_agent(agent, strategic_config, f"{agent.__class__.__name__}_strategic")
        results['strategic'] = strategic_result
        
        # Analyze correlation between timeframes
        correlation = self._calculate_timeframe_correlation(tactical_result, strategic_result)
        
        # Combined analysis
        combined_result = self._combine_timeframe_results(tactical_result, strategic_result, correlation)
        results['combined'] = combined_result
        
        return results
    
    def _calculate_timeframe_correlation(
        self,
        tactical_result: BacktestResult,
        strategic_result: BacktestResult
    ) -> float:
        """Calculate correlation between timeframe returns"""
        
        # Align returns by resampling
        min_len = min(len(tactical_result.returns), len(strategic_result.returns))
        tactical_aligned = tactical_result.returns[:min_len]
        strategic_aligned = strategic_result.returns[:min_len]
        
        if len(tactical_aligned) > 1 and len(strategic_aligned) > 1:
            correlation = np.corrcoef(tactical_aligned, strategic_aligned)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _combine_timeframe_results(
        self,
        tactical_result: BacktestResult,
        strategic_result: BacktestResult,
        correlation: float
    ) -> BacktestResult:
        """Combine results from multiple timeframes"""
        
        # Weight by inverse correlation (diversification benefit)
        tactical_weight = 0.6 if correlation < 0.5 else 0.4
        strategic_weight = 1.0 - tactical_weight
        
        # Combine equity curves
        min_len = min(len(tactical_result.equity_curve), len(strategic_result.equity_curve))
        combined_equity = (
            tactical_weight * tactical_result.equity_curve[:min_len] +
            strategic_weight * strategic_result.equity_curve[:min_len]
        )
        
        # Combine returns
        min_ret_len = min(len(tactical_result.returns), len(strategic_result.returns))
        combined_returns = (
            tactical_weight * tactical_result.returns[:min_ret_len] +
            strategic_weight * strategic_result.returns[:min_ret_len]
        )
        
        # Calculate combined metrics
        combined_metrics = calculate_all_metrics(
            equity_curve=combined_equity,
            returns=combined_returns,
            periods_per_year=252 * 78
        )
        
        return BacktestResult(
            agent_name=f"{tactical_result.agent_name}_combined",
            equity_curve=combined_equity,
            returns=combined_returns,
            positions=tactical_result.positions,  # Use tactical positions
            actions=tactical_result.actions,
            metrics=combined_metrics,
            execution_time=tactical_result.execution_time + strategic_result.execution_time,
            config=tactical_result.config,
            tactical_metrics=tactical_result.metrics,
            strategic_metrics=strategic_result.metrics,
            timeframe_correlation=correlation
        )
    
    def run_monte_carlo_simulation(
        self,
        agent: Any,
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for robust performance estimation
        
        Args:
            agent: Agent to test
            config: Backtest configuration
            
        Returns:
            Monte Carlo simulation results
        """
        logger.info(f"Starting Monte Carlo simulation with {config.monte_carlo_runs} runs")
        
        mc_results = []
        
        # Use parallel processing
        if config.parallel_enabled:
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                futures = []
                
                for run in range(config.monte_carlo_runs):
                    # Create config with different random seed
                    mc_config = BacktestConfig(
                        **config.__dict__,
                        random_seed=config.random_seed + run if config.random_seed else run,
                        episodes=config.episodes // 10  # Reduced for MC
                    )
                    
                    future = executor.submit(
                        self._run_agent,
                        agent, mc_config, f"mc_{run}"
                    )
                    futures.append(future)
                
                # Collect results with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Monte Carlo"):
                    try:
                        result = future.result()
                        mc_results.append(result)
                    except Exception as e:
                        logger.error(f"Monte Carlo run failed: {e}")
        else:
            # Sequential processing with progress bar
            for run in tqdm(range(config.monte_carlo_runs), desc="Monte Carlo"):
                mc_config = BacktestConfig(
                    **config.__dict__,
                    random_seed=config.random_seed + run if config.random_seed else run,
                    episodes=config.episodes // 10
                )
                
                try:
                    result = self._run_agent(agent, mc_config, f"mc_{run}")
                    mc_results.append(result)
                except Exception as e:
                    logger.error(f"Monte Carlo run {run} failed: {e}")
        
        # Analyze MC results
        return self._analyze_monte_carlo_results(mc_results, config)
    
    def _analyze_monte_carlo_results(
        self,
        mc_results: List[BacktestResult],
        config: BacktestConfig
    ) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        
        # Extract key metrics
        total_returns = [r.metrics.total_return for r in mc_results]
        sharpe_ratios = [r.metrics.sharpe_ratio for r in mc_results]
        max_drawdowns = [r.metrics.max_drawdown for r in mc_results]
        calmar_ratios = [r.metrics.calmar_ratio for r in mc_results]
        
        # Calculate statistics
        stats = {}
        
        for metric_name, values in [
            ('total_return', total_returns),
            ('sharpe_ratio', sharpe_ratios),
            ('max_drawdown', max_drawdowns),
            ('calmar_ratio', calmar_ratios)
        ]:
            values_clean = [v for v in values if not np.isnan(v) and not np.isinf(v)]
            
            if values_clean:
                stats[metric_name] = {
                    'mean': np.mean(values_clean),
                    'std': np.std(values_clean),
                    'min': np.min(values_clean),
                    'max': np.max(values_clean),
                    'median': np.median(values_clean),
                    'percentile_5': np.percentile(values_clean, 5),
                    'percentile_95': np.percentile(values_clean, 95),
                    'positive_rate': sum(1 for v in values_clean if v > 0) / len(values_clean)
                }
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for interval in config.confidence_intervals:
            lower_pct = (1 - interval) / 2 * 100
            upper_pct = (1 + interval) / 2 * 100
            
            confidence_intervals[f"{interval:.0%}"] = {
                'total_return': (
                    np.percentile(total_returns, lower_pct),
                    np.percentile(total_returns, upper_pct)
                ),
                'sharpe_ratio': (
                    np.percentile(sharpe_ratios, lower_pct),
                    np.percentile(sharpe_ratios, upper_pct)
                )
            }
        
        # Risk analysis
        risk_metrics = {
            'probability_of_loss': sum(1 for r in total_returns if r < 0) / len(total_returns),
            'expected_shortfall_5': np.mean([r for r in total_returns if r <= np.percentile(total_returns, 5)]),
            'tail_ratio': np.percentile(total_returns, 95) / abs(np.percentile(total_returns, 5)) if np.percentile(total_returns, 5) != 0 else 0,
            'stability_ratio': np.std(total_returns) / np.mean(total_returns) if np.mean(total_returns) != 0 else 0
        }
        
        return {
            'statistics': stats,
            'confidence_intervals': confidence_intervals,
            'risk_metrics': risk_metrics,
            'total_runs': len(mc_results),
            'successful_runs': len(mc_results)
        }


class MARLAgentWrapper:
    """
    Wrapper to make MARL models compatible with backtest interface
    """
    
    def __init__(self, checkpoint: Dict[str, Any], env_config: Dict[str, Any]):
        """
        Initialize MARL agent wrapper
        
        Args:
            checkpoint: Model checkpoint
            env_config: Environment configuration
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.models = {}
        
        # Create and load MLMI actor
        self.models['mlmi_expert'] = MLMIActor(
            input_dim=len(env_config['feature_indices']['mlmi_expert']),
            hidden_dims=[256, 128, 64]
        )
        
        # Create and load NWRQK actor
        self.models['nwrqk_expert'] = NWRQKActor(
            input_dim=6,  # NWRQK uses 6D features
            hidden_dims=[256, 128, 64]
        )
        
        # Create and load MMD actor
        self.models['regime_expert'] = MMDActor(
            input_dim=len(env_config['feature_indices']['regime_expert']),
            hidden_dims=[128, 64, 32]
        )
        
        # Load state dicts if available
        if 'models' in checkpoint:
            for name, state_dict in checkpoint['models'].items():
                if name in self.models and name != 'critic':
                    self.models[name].load_state_dict(state_dict)
                    
        # Move to device
        for model in self.models.values():
            model.to(self.device)
            model.eval()
            
        self.current_agent = 'mlmi_expert'
        self.agent_cycle = ['mlmi_expert', 'nwrqk_expert', 'regime_expert']
        self.agent_idx = 0
        
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Get action from current agent"""
        # Determine which agent should act
        agent = self.agent_cycle[self.agent_idx % 3]
        self.agent_idx += 1
        
        # Get model
        model = self.models[agent]
        
        # Prepare observation
        features = observation['features']
        shared_context = observation['shared_context']
        combined = np.concatenate([features, shared_context])
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
        
        # Get action
        with torch.no_grad():
            output = model(obs_tensor, deterministic=True)  # Deterministic for testing
            action_probs = output['action_probs'].cpu().numpy()[0]
            
        return action_probs


def main():
    """Enhanced example usage of backtesting framework"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create backtest runner
    runner = BacktestRunner()
    
    # Configure enhanced backtest
    config = BacktestConfig(
        episodes=50,
        max_steps_per_episode=300,
        initial_capital=10000,
        position_size=0.1,
        random_seed=42,
        
        # Enhanced features
        walk_forward_enabled=False,  # Disable for demo
        multi_timeframe=True,
        monte_carlo_enabled=True,
        monte_carlo_runs=100,  # Reduced for demo
        parallel_enabled=True,
        
        # Risk management
        dynamic_position_sizing=True,
        kelly_criterion=True,
        correlation_adjustment=True,
        max_drawdown_limit=0.15,
        
        # Market simulation
        realistic_execution=True,
        market_impact_enabled=True,
        latency_simulation=True,
        
        # Performance optimization
        jit_compilation=True,
        memory_efficient=True
    )
    
    # Run baseline agents
    print("Running baseline agents with enhanced features...")
    baseline_results = runner.run_baseline_agents(config)
    
    # Run multi-timeframe analysis
    print("\nRunning multi-timeframe analysis...")
    rule_agent = RuleBasedAgent()
    timeframe_results = runner.run_multi_timeframe_analysis(rule_agent, config)
    
    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    mc_results = runner.run_monte_carlo_simulation(rule_agent, config)
    
    # Compare results
    comparison = runner.compare_results()
    print("\nBacktest Results Comparison:")
    print(comparison)
    
    # Print Monte Carlo statistics
    if mc_results:
        print("\nMonte Carlo Statistics:")
        for metric, stats in mc_results.get('statistics', {}).items():
            print(f"{metric}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  95% CI: [{stats['percentile_5']:.4f}, {stats['percentile_95']:.4f}]")
    
    # Print timeframe correlation
    if timeframe_results.get('combined'):
        correlation = timeframe_results['combined'].timeframe_correlation
        print(f"\nTimeframe Correlation: {correlation:.3f}")
    
    # Print risk metrics
    if runner.risk_integrator:
        risk_summary = runner.risk_integrator.get_risk_summary()
        print("\nRisk Summary:")
        for key, value in risk_summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    # Print execution statistics
    if runner.market_simulator:
        exec_stats = runner.market_simulator.get_execution_stats()
        print("\nExecution Statistics:")
        for key, value in exec_stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    # Save results
    comparison.to_csv("enhanced_backtest_results.csv")
    logger.info("Results saved to enhanced_backtest_results.csv")
    

if __name__ == "__main__":
    main()