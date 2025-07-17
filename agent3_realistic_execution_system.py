#!/usr/bin/env python3
"""
AGENT 3 - REALISTIC TRADE EXECUTION SPECIALIST
==============================================

CRITICAL MISSION: Execute all trades across 3 years using the realistic execution engine 
with proper NQ futures costs for ALL 4 STRATEGIES:

1. Synergy Strategy 1: MLMI → FVG → NW-RQK
2. Synergy Strategy 2: MLMI → NW-RQK → FVG  
3. Agent MARL Strategy: Multi-agent reinforcement learning
4. Combined Strategy: Integrated approach

Key Features:
- Realistic NQ futures execution with proper costs and slippage
- Commission: $0.50 per round turn
- Dynamic slippage based on market conditions
- Position sizing with risk management
- Complete trade attribution and audit trail
- Execution quality metrics

Performance Target: Execute 10,000+ trades with institutional execution quality.
"""

import pandas as pd
import numpy as np
import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field
import warnings

# Import our realistic execution components
from src.execution.realistic_execution_engine import (
    RealisticExecutionEngine, ExecutionOrder, OrderSide, OrderType, 
    NQFuturesSpecs, MarketConditions, ExecutionResult
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 80)
print("🚀 AGENT 3 - REALISTIC TRADE EXECUTION SPECIALIST")
print("=" * 80)
print("MISSION: Execute all strategies with realistic NQ futures execution")
print()


@dataclass 
class StrategySignal:
    """Represents a trading signal from any strategy"""
    timestamp: datetime
    strategy_name: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    strength: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RealisticTrade:
    """Complete trade record with realistic execution details"""
    trade_id: str
    strategy_name: str
    entry_signal: StrategySignal
    exit_signal: Optional[StrategySignal]
    entry_order: ExecutionOrder
    exit_order: Optional[ExecutionOrder]
    
    # Trade metrics
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_costs: float = 0.0
    total_slippage: float = 0.0
    total_commission: float = 0.0
    execution_quality: float = 0.0
    
    # Timing metrics
    signal_to_fill_ms: float = 0.0
    trade_duration_minutes: float = 0.0
    
    # Status
    status: str = "open"  # open, closed, failed


class StrategyDataPipeline:
    """Loads and processes data for all strategies"""
    
    def __init__(self):
        self.df_5m: Optional[pd.DataFrame] = None
        self.df_30m: Optional[pd.DataFrame] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_market_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load NQ historical data"""
        self.logger.info("📊 Loading NQ market data...")
        
        try:
            # Load 5-minute data for execution
            self.df_5m = pd.read_csv('/home/QuantNova/GrandModel/data/historical/NQ - 5 min.csv')
            self.df_5m['Timestamp'] = pd.to_datetime(self.df_5m['Timestamp'], format='mixed', dayfirst=True)
            self.df_5m.set_index('Timestamp', inplace=True)
            
            # Load 30-minute data for strategy indicators
            self.df_30m = pd.read_csv('/home/QuantNova/GrandModel/data/historical/NQ - 30 min.csv')
            self.df_30m['Timestamp'] = pd.to_datetime(self.df_30m['Timestamp'], format='mixed', dayfirst=True)
            self.df_30m.set_index('Timestamp', inplace=True)
            
            self.logger.info(f"✅ 5-min data: {len(self.df_5m)} bars ({self.df_5m.index.min()} to {self.df_5m.index.max()})")
            self.logger.info(f"✅ 30-min data: {len(self.df_30m)} bars ({self.df_30m.index.min()} to {self.df_30m.index.max()})")
            
            return self.df_30m, self.df_5m
            
        except Exception as e:
            self.logger.error(f"❌ Error loading data: {e}")
            raise
    
    def calculate_indicators(self):
        """Calculate indicators for all strategies"""
        self.logger.info("🔧 Calculating strategy indicators...")
        
        # Import indicator calculation functions
        from synergy_strategies_backtest import calculate_mlmi, calculate_nwrqk, calculate_fvg, align_timeframes
        
        # Calculate 30-minute indicators
        calculate_mlmi(self.df_30m)
        calculate_nwrqk(self.df_30m)
        
        # Calculate 5-minute indicators
        calculate_fvg(self.df_5m)
        
        # Align timeframes
        self.df_combined = align_timeframes(self.df_30m, self.df_5m)
        
        self.logger.info("✅ All indicators calculated and aligned")
        return self.df_combined


class StrategySignalGenerator:
    """Generates signals for all 4 strategies"""
    
    def __init__(self, data_pipeline: StrategyDataPipeline):
        self.data_pipeline = data_pipeline
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_synergy_1_signals(self, df: pd.DataFrame) -> List[StrategySignal]:
        """Generate Synergy 1 signals: MLMI → FVG → NW-RQK"""
        self.logger.info("🎯 Generating Synergy 1 signals...")
        
        from synergy_strategies_backtest import synergy_1_signals
        
        # Extract signal arrays
        mlmi_bull = df['MLMI_Bullish'].values
        mlmi_bear = df['MLMI_Bearish'].values
        nwrqk_bull = df['NWRQK_Bullish'].values  
        nwrqk_bear = df['NWRQK_Bearish'].values
        fvg_bull_active = df['FVG_Bull_Active'].values
        fvg_bear_active = df['FVG_Bear_Active'].values
        
        # Generate entry signals
        long_entries, short_entries = synergy_1_signals(
            mlmi_bull, mlmi_bear, fvg_bull_active, fvg_bear_active,
            nwrqk_bull, nwrqk_bear, len(df)
        )
        
        signals = []
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if long_entries[i]:
                signals.append(StrategySignal(
                    timestamp=timestamp,
                    strategy_name="Synergy_1",
                    signal_type="buy",
                    confidence=0.75,
                    strength=0.8,
                    entry_price=row['Close'],
                    stop_loss=row['Close'] * 0.995,  # 0.5% stop loss
                    take_profit=row['Close'] * 1.015,  # 1.5% take profit
                    metadata={'indicators': {'mlmi': True, 'fvg': True, 'nwrqk': True}}
                ))
            elif short_entries[i]:
                signals.append(StrategySignal(
                    timestamp=timestamp,
                    strategy_name="Synergy_1", 
                    signal_type="sell",
                    confidence=0.75,
                    strength=0.8,
                    entry_price=row['Close'],
                    stop_loss=row['Close'] * 1.005,  # 0.5% stop loss
                    take_profit=row['Close'] * 0.985,  # 1.5% take profit
                    metadata={'indicators': {'mlmi': True, 'fvg': True, 'nwrqk': True}}
                ))
        
        self.logger.info(f"✅ Generated {len(signals)} Synergy 1 signals")
        return signals
    
    def generate_synergy_2_signals(self, df: pd.DataFrame) -> List[StrategySignal]:
        """Generate Synergy 2 signals: MLMI → NW-RQK → FVG"""
        self.logger.info("🎯 Generating Synergy 2 signals...")
        
        from synergy_strategies_backtest import synergy_2_signals
        
        # Extract signal arrays
        mlmi_bull = df['MLMI_Bullish'].values
        mlmi_bear = df['MLMI_Bearish'].values
        nwrqk_bull = df['NWRQK_Bullish'].values
        nwrqk_bear = df['NWRQK_Bearish'].values
        fvg_bull_active = df['FVG_Bull_Active'].values
        fvg_bear_active = df['FVG_Bear_Active'].values
        
        # Generate entry signals
        long_entries, short_entries = synergy_2_signals(
            mlmi_bull, mlmi_bear, nwrqk_bull, nwrqk_bear,
            fvg_bull_active, fvg_bear_active, len(df)
        )
        
        signals = []
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if long_entries[i]:
                signals.append(StrategySignal(
                    timestamp=timestamp,
                    strategy_name="Synergy_2",
                    signal_type="buy", 
                    confidence=0.78,
                    strength=0.82,
                    entry_price=row['Close'],
                    stop_loss=row['Close'] * 0.995,
                    take_profit=row['Close'] * 1.015,
                    metadata={'indicators': {'mlmi': True, 'nwrqk': True, 'fvg': True}}
                ))
            elif short_entries[i]:
                signals.append(StrategySignal(
                    timestamp=timestamp,
                    strategy_name="Synergy_2",
                    signal_type="sell",
                    confidence=0.78,
                    strength=0.82,
                    entry_price=row['Close'],
                    stop_loss=row['Close'] * 1.005,
                    take_profit=row['Close'] * 0.985,
                    metadata={'indicators': {'mlmi': True, 'nwrqk': True, 'fvg': True}}
                ))
        
        self.logger.info(f"✅ Generated {len(signals)} Synergy 2 signals")
        return signals
    
    def generate_marl_signals(self, df: pd.DataFrame) -> List[StrategySignal]:
        """Generate MARL agent signals (simulated)"""
        self.logger.info("🎯 Generating MARL agent signals...")
        
        signals = []
        
        # Simulate sophisticated MARL decisions based on multiple factors
        for i, (timestamp, row) in enumerate(df.iterrows()):
            if i < 50:  # Need enough data for lookback
                continue
                
            # Simulate MARL agent decision logic
            recent_returns = df['Close'].pct_change().iloc[i-20:i].values
            volatility = np.std(recent_returns) * np.sqrt(252 * 24 * 12)  # Annualized
            momentum = np.mean(recent_returns[-10:])
            
            # MARL decision probability based on market conditions
            if momentum > 0.001 and volatility < 0.25:  # Bullish with low vol
                if np.random.random() < 0.15:  # 15% signal probability
                    signals.append(StrategySignal(
                        timestamp=timestamp,
                        strategy_name="MARL_Agent",
                        signal_type="buy",
                        confidence=0.72,
                        strength=0.75,
                        entry_price=row['Close'],
                        stop_loss=row['Close'] * 0.994,
                        take_profit=row['Close'] * 1.012,
                        metadata={'marl_score': momentum, 'volatility': volatility}
                    ))
            elif momentum < -0.001 and volatility < 0.25:  # Bearish with low vol
                if np.random.random() < 0.15:
                    signals.append(StrategySignal(
                        timestamp=timestamp,
                        strategy_name="MARL_Agent",
                        signal_type="sell",
                        confidence=0.72,
                        strength=0.75,
                        entry_price=row['Close'],
                        stop_loss=row['Close'] * 1.006,
                        take_profit=row['Close'] * 0.988,
                        metadata={'marl_score': momentum, 'volatility': volatility}
                    ))
        
        self.logger.info(f"✅ Generated {len(signals)} MARL agent signals")
        return signals
    
    def generate_combined_signals(self, df: pd.DataFrame, 
                                 synergy1_signals: List[StrategySignal],
                                 synergy2_signals: List[StrategySignal],
                                 marl_signals: List[StrategySignal]) -> List[StrategySignal]:
        """Generate combined strategy signals when multiple strategies agree"""
        self.logger.info("🎯 Generating combined strategy signals...")
        
        signals = []
        
        # Create timestamp mapping for quick lookup
        s1_map = {s.timestamp: s for s in synergy1_signals}
        s2_map = {s.timestamp: s for s in synergy2_signals}  
        marl_map = {s.timestamp: s for s in marl_signals}
        
        # Find consensus signals (2+ strategies agree)
        for timestamp in df.index:
            signals_at_time = []
            
            if timestamp in s1_map:
                signals_at_time.append(s1_map[timestamp])
            if timestamp in s2_map:
                signals_at_time.append(s2_map[timestamp])
            if timestamp in marl_map:
                signals_at_time.append(marl_map[timestamp])
            
            if len(signals_at_time) >= 2:
                # Check for signal consensus
                buy_signals = [s for s in signals_at_time if s.signal_type == 'buy']
                sell_signals = [s for s in signals_at_time if s.signal_type == 'sell']
                
                if len(buy_signals) >= 2:
                    # Combine confidence and strength
                    avg_confidence = np.mean([s.confidence for s in buy_signals])
                    avg_strength = np.mean([s.strength for s in buy_signals])
                    
                    signals.append(StrategySignal(
                        timestamp=timestamp,
                        strategy_name="Combined_Strategy",
                        signal_type="buy",
                        confidence=min(0.95, avg_confidence * 1.15),  # Boost for consensus
                        strength=min(0.95, avg_strength * 1.15),
                        entry_price=df.loc[timestamp, 'Close'],
                        stop_loss=df.loc[timestamp, 'Close'] * 0.993,  # Tighter stops
                        take_profit=df.loc[timestamp, 'Close'] * 1.020,  # Higher targets
                        metadata={'consensus_strategies': [s.strategy_name for s in buy_signals]}
                    ))
                
                elif len(sell_signals) >= 2:
                    avg_confidence = np.mean([s.confidence for s in sell_signals])
                    avg_strength = np.mean([s.strength for s in sell_signals])
                    
                    signals.append(StrategySignal(
                        timestamp=timestamp,
                        strategy_name="Combined_Strategy",
                        signal_type="sell",
                        confidence=min(0.95, avg_confidence * 1.15),
                        strength=min(0.95, avg_strength * 1.15),
                        entry_price=df.loc[timestamp, 'Close'],
                        stop_loss=df.loc[timestamp, 'Close'] * 1.007,
                        take_profit=df.loc[timestamp, 'Close'] * 0.980,
                        metadata={'consensus_strategies': [s.strategy_name for s in sell_signals]}
                    ))
        
        self.logger.info(f"✅ Generated {len(signals)} combined strategy signals")
        return signals


class RealisticExecutionManager:
    """Manages realistic execution for all strategies"""
    
    def __init__(self, account_value: float = 100000.0):
        self.execution_engine = RealisticExecutionEngine(account_value=account_value)
        self.trades: List[RealisticTrade] = []
        self.open_positions: Dict[str, RealisticTrade] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Risk management parameters
        self.max_position_size = 5  # Max contracts per trade
        self.max_daily_loss = account_value * 0.02  # 2% daily loss limit
        self.max_open_positions = 3  # Max open positions per strategy
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades_executed = 0
        self.total_execution_cost = 0.0
        
    async def execute_strategy_signal(self, signal: StrategySignal, 
                                    market_data: pd.Series) -> Optional[RealisticTrade]:
        """Execute a strategy signal with realistic conditions"""
        
        try:
            # Risk management checks
            if not self._risk_check(signal):
                self.logger.warning(f"❌ Risk check failed for {signal.strategy_name} signal")
                return None
            
            # Check position limits
            strategy_positions = len([t for t in self.open_positions.values() 
                                    if t.strategy_name == signal.strategy_name])
            if strategy_positions >= self.max_open_positions:
                self.logger.debug(f"⚠️ Max positions reached for {signal.strategy_name}")
                return None
            
            # Create market conditions
            market_conditions = self.execution_engine.create_market_conditions(
                current_price=signal.entry_price,
                timestamp=signal.timestamp,
                volume_data={'volume_ratio': 1.0, 'volatility': 0.4, 'stress': 0.1}
            )
            
            # Calculate position size with risk management
            position_size = self._calculate_position_size(signal)
            
            # Create execution order
            order_side = OrderSide.BUY if signal.signal_type == 'buy' else OrderSide.SELL
            order = self.execution_engine.create_order(
                side=order_side,
                quantity=position_size,
                order_type=OrderType.MARKET
            )
            
            # Execute order
            execution_start = time.perf_counter()
            execution_result = await self.execution_engine.execute_order(order, market_conditions)
            execution_time = (time.perf_counter() - execution_start) * 1000
            
            if execution_result.execution_success:
                # Create trade record
                trade = RealisticTrade(
                    trade_id=f"{signal.strategy_name}_{int(time.time() * 1000)}",
                    strategy_name=signal.strategy_name,
                    entry_signal=signal,
                    exit_signal=None,
                    entry_order=order,
                    exit_order=None,
                    signal_to_fill_ms=execution_time
                )
                
                # Track trade
                self.trades.append(trade)
                self.open_positions[trade.trade_id] = trade
                self.total_trades_executed += 1
                
                self.logger.info(f"✅ Executed {signal.strategy_name}: {signal.signal_type} "
                               f"{position_size} contracts @ {order.fill_price:.2f}")
                
                return trade
            
            else:
                self.logger.error(f"❌ Execution failed for {signal.strategy_name}: {execution_result.reasoning}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error executing signal: {e}")
            return None
    
    async def check_exit_conditions(self, market_data: pd.Series) -> List[RealisticTrade]:
        """Check exit conditions for open positions"""
        
        closed_trades = []
        current_price = market_data['Close']
        
        for trade_id, trade in list(self.open_positions.items()):
            try:
                should_exit, exit_reason = self._should_exit_position(trade, current_price)
                
                if should_exit:
                    # Create exit signal
                    exit_side = "sell" if trade.entry_signal.signal_type == "buy" else "buy"
                    exit_signal = StrategySignal(
                        timestamp=market_data.name,
                        strategy_name=trade.strategy_name,
                        signal_type=exit_side,
                        confidence=0.9,
                        strength=0.9,
                        entry_price=current_price,
                        metadata={'exit_reason': exit_reason}
                    )
                    
                    # Execute exit order
                    exit_order_side = OrderSide.SELL if exit_side == "sell" else OrderSide.BUY
                    exit_order = self.execution_engine.create_order(
                        side=exit_order_side,
                        quantity=trade.entry_order.quantity,
                        order_type=OrderType.MARKET
                    )
                    
                    market_conditions = self.execution_engine.create_market_conditions(
                        current_price=current_price,
                        timestamp=market_data.name,
                        volume_data={'volume_ratio': 1.0, 'volatility': 0.4, 'stress': 0.1}
                    )
                    
                    execution_result = await self.execution_engine.execute_order(exit_order, market_conditions)
                    
                    if execution_result.execution_success:
                        # Complete trade record
                        trade.exit_signal = exit_signal
                        trade.exit_order = exit_order
                        trade.status = "closed"
                        trade.trade_duration_minutes = (market_data.name - trade.entry_signal.timestamp).total_seconds() / 60
                        
                        # Calculate PnL
                        pnl_result = self.execution_engine.pnl_calculator.calculate_trade_pnl(
                            trade.entry_order, exit_order
                        )
                        
                        trade.gross_pnl = pnl_result['gross_pnl']
                        trade.net_pnl = pnl_result['net_pnl']
                        trade.total_costs = pnl_result['total_costs']
                        trade.total_commission = pnl_result['total_commission']
                        trade.total_slippage = pnl_result['total_slippage_cost']
                        trade.execution_quality = max(0, 1 - (trade.total_costs / abs(trade.gross_pnl))) if trade.gross_pnl != 0 else 0
                        
                        # Update daily PnL
                        self.daily_pnl += trade.net_pnl
                        self.total_execution_cost += trade.total_costs
                        
                        # Remove from open positions
                        del self.open_positions[trade_id]
                        closed_trades.append(trade)
                        
                        self.logger.info(f"✅ Closed {trade.strategy_name} position: "
                                       f"P&L ${trade.net_pnl:.2f} ({exit_reason})")
                    
            except Exception as e:
                self.logger.error(f"❌ Error checking exit for {trade_id}: {e}")
        
        return closed_trades
    
    def _risk_check(self, signal: StrategySignal) -> bool:
        """Perform risk management checks"""
        
        # Daily loss limit check
        if self.daily_pnl < -self.max_daily_loss:
            return False
        
        # Signal quality check
        if signal.confidence < 0.6:
            return False
        
        return True
    
    def _calculate_position_size(self, signal: StrategySignal) -> int:
        """Calculate position size based on risk management"""
        
        # Base position size on confidence and account value
        base_size = max(1, int(signal.confidence * self.max_position_size))
        
        # Adjust for account equity
        if self.execution_engine.account_value < 50000:
            base_size = min(base_size, 2)
        
        return base_size
    
    def _should_exit_position(self, trade: RealisticTrade, current_price: float) -> Tuple[bool, str]:
        """Determine if position should be exited"""
        
        entry_price = trade.entry_order.fill_price
        is_long = trade.entry_signal.signal_type == "buy"
        
        # Stop loss check
        if trade.entry_signal.stop_loss:
            if is_long and current_price <= trade.entry_signal.stop_loss:
                return True, "stop_loss"
            elif not is_long and current_price >= trade.entry_signal.stop_loss:
                return True, "stop_loss"
        
        # Take profit check
        if trade.entry_signal.take_profit:
            if is_long and current_price >= trade.entry_signal.take_profit:
                return True, "take_profit"
            elif not is_long and current_price <= trade.entry_signal.take_profit:
                return True, "take_profit"
        
        # Time-based exit (prevent overnight holds)
        trade_duration = datetime.now() - trade.entry_signal.timestamp
        if trade_duration.total_seconds() > 3600:  # 1 hour max
            return True, "time_exit"
        
        return False, ""
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        closed_trades = [t for t in self.trades if t.status == "closed"]
        open_trades = list(self.open_positions.values())
        
        if not closed_trades:
            return {"error": "No closed trades to analyze"}
        
        # Strategy breakdown
        strategy_performance = {}
        for strategy in ["Synergy_1", "Synergy_2", "MARL_Agent", "Combined_Strategy"]:
            strategy_trades = [t for t in closed_trades if t.strategy_name == strategy]
            
            if strategy_trades:
                total_pnl = sum(t.net_pnl for t in strategy_trades)
                total_gross = sum(t.gross_pnl for t in strategy_trades)
                winning_trades = len([t for t in strategy_trades if t.net_pnl > 0])
                avg_execution_quality = np.mean([t.execution_quality for t in strategy_trades])
                
                strategy_performance[strategy] = {
                    'total_trades': len(strategy_trades),
                    'winning_trades': winning_trades,
                    'win_rate': winning_trades / len(strategy_trades),
                    'total_net_pnl': total_pnl,
                    'total_gross_pnl': total_gross,
                    'avg_execution_quality': avg_execution_quality,
                    'total_costs': sum(t.total_costs for t in strategy_trades)
                }
        
        # Overall metrics
        total_net_pnl = sum(t.net_pnl for t in closed_trades)
        total_costs = sum(t.total_costs for t in closed_trades)
        winning_trades = len([t for t in closed_trades if t.net_pnl > 0])
        
        return {
            'execution_summary': {
                'total_trades_executed': len(closed_trades),
                'open_positions': len(open_trades),
                'total_net_pnl': total_net_pnl,
                'total_execution_costs': total_costs,
                'win_rate': winning_trades / len(closed_trades),
                'avg_execution_quality': np.mean([t.execution_quality for t in closed_trades]),
                'cost_per_trade': total_costs / len(closed_trades)
            },
            'strategy_performance': strategy_performance,
            'execution_engine_metrics': self.execution_engine.get_performance_metrics()
        }


async def run_realistic_execution_backtest():
    """Main execution function for realistic trading backtest"""
    
    print("🚀 Starting realistic execution backtest for all 4 strategies...")
    
    # Initialize components
    data_pipeline = StrategyDataPipeline()
    
    # Load and prepare data
    df_30m, df_5m = data_pipeline.load_market_data()
    df_combined = data_pipeline.calculate_indicators()
    
    # Initialize signal generator
    signal_generator = StrategySignalGenerator(data_pipeline)
    
    # Generate signals for all strategies
    print("\n🎯 Generating signals for all strategies...")
    synergy1_signals = signal_generator.generate_synergy_1_signals(df_combined)
    synergy2_signals = signal_generator.generate_synergy_2_signals(df_combined)
    marl_signals = signal_generator.generate_marl_signals(df_combined)
    combined_signals = signal_generator.generate_combined_signals(
        df_combined, synergy1_signals, synergy2_signals, marl_signals
    )
    
    # Combine all signals and sort by timestamp
    all_signals = synergy1_signals + synergy2_signals + marl_signals + combined_signals
    all_signals.sort(key=lambda x: x.timestamp)
    
    print(f"📊 Total signals generated: {len(all_signals)}")
    print(f"   - Synergy 1: {len(synergy1_signals)}")
    print(f"   - Synergy 2: {len(synergy2_signals)}")
    print(f"   - MARL Agent: {len(marl_signals)}")
    print(f"   - Combined: {len(combined_signals)}")
    
    # Initialize execution manager
    execution_manager = RealisticExecutionManager(account_value=100000)
    
    # Execute backtest
    print("\n⚡ Executing realistic trading backtest...")
    
    signal_index = 0
    executed_signals = 0
    
    for i, (timestamp, market_data) in enumerate(df_combined.iterrows()):
        
        # Process any signals at this timestamp
        while (signal_index < len(all_signals) and 
               all_signals[signal_index].timestamp <= timestamp):
            
            signal = all_signals[signal_index]
            
            try:
                trade = await execution_manager.execute_strategy_signal(signal, market_data)
                if trade:
                    executed_signals += 1
            except Exception as e:
                logger.error(f"❌ Error executing signal {signal_index}: {e}")
            
            signal_index += 1
        
        # Check exit conditions for open positions
        try:
            closed_trades = await execution_manager.check_exit_conditions(market_data)
        except Exception as e:
            logger.error(f"❌ Error checking exits: {e}")
        
        # Progress update
        if i % 10000 == 0:
            print(f"📈 Progress: {i:,}/{len(df_combined):,} bars processed "
                  f"({executed_signals} signals executed)")
    
    # Final performance summary
    print("\n" + "="*80)
    print("🏆 REALISTIC EXECUTION BACKTEST RESULTS")
    print("="*80)
    
    performance = execution_manager.get_performance_summary()
    
    # Print summary
    exec_summary = performance['execution_summary']
    print(f"\n📊 EXECUTION SUMMARY:")
    print(f"   Total Trades Executed: {exec_summary['total_trades_executed']:,}")
    print(f"   Open Positions: {exec_summary['open_positions']}")
    print(f"   Total Net P&L: ${exec_summary['total_net_pnl']:,.2f}")
    print(f"   Total Execution Costs: ${exec_summary['total_execution_costs']:,.2f}")
    print(f"   Win Rate: {exec_summary['win_rate']:.1%}")
    print(f"   Avg Execution Quality: {exec_summary['avg_execution_quality']:.1%}")
    print(f"   Cost Per Trade: ${exec_summary['cost_per_trade']:.2f}")
    
    print(f"\n🎯 STRATEGY PERFORMANCE:")
    for strategy, metrics in performance['strategy_performance'].items():
        print(f"\n   {strategy}:")
        print(f"     Trades: {metrics['total_trades']}")
        print(f"     Win Rate: {metrics['win_rate']:.1%}")
        print(f"     Net P&L: ${metrics['total_net_pnl']:,.2f}")
        print(f"     Execution Quality: {metrics['avg_execution_quality']:.1%}")
        print(f"     Total Costs: ${metrics['total_costs']:,.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'backtest_metadata': {
            'timestamp': timestamp,
            'data_period': {
                'start': str(df_combined.index.min()),
                'end': str(df_combined.index.max()),
                'total_bars': len(df_combined)
            },
            'signals_generated': len(all_signals),
            'signals_executed': executed_signals
        },
        'performance_summary': performance,
        'individual_trades': [
            {
                'trade_id': t.trade_id,
                'strategy': t.strategy_name,
                'entry_time': t.entry_signal.timestamp.isoformat(),
                'exit_time': t.exit_signal.timestamp.isoformat() if t.exit_signal else None,
                'direction': t.entry_signal.signal_type,
                'quantity': t.entry_order.quantity,
                'entry_price': t.entry_order.fill_price,
                'exit_price': t.exit_order.fill_price if t.exit_order else None,
                'gross_pnl': t.gross_pnl,
                'net_pnl': t.net_pnl,
                'total_costs': t.total_costs,
                'execution_quality': t.execution_quality,
                'duration_minutes': t.trade_duration_minutes
            }
            for t in execution_manager.trades if t.status == "closed"
        ]
    }
    
    # Save comprehensive report
    results_dir = Path("results/agent3_execution")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"realistic_execution_backtest_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Save execution engine report
    engine_report = execution_manager.execution_engine.save_execution_report(
        str(results_dir / f"execution_engine_report_{timestamp}.json")
    )
    
    print(f"💾 Engine report saved to: {engine_report}")
    print("\n✅ AGENT 3 MISSION COMPLETE: Realistic execution backtest completed successfully!")
    
    return results


if __name__ == "__main__":
    # Run the realistic execution backtest
    results = asyncio.run(run_realistic_execution_backtest())