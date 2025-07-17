"""
Emergency Action Execution System for Risk Monitor Agent

This module provides the execution framework for emergency risk management actions
with real-time portfolio protection and automated intervention capabilities.

Key Features:
- Immediate position reduction protocols
- Emergency liquidation system
- Hedge creation and management
- Real-time execution monitoring
- Comprehensive audit trail
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import structlog
import time

from src.core.events import Event, EventType, EventBus

logger = structlog.get_logger()


class ActionPriority(Enum):
    """Action execution priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class ExecutionStatus(Enum):
    """Action execution status"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Position:
    """Position data for emergency actions"""
    symbol: str
    quantity: float
    market_value: float
    side: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_contribution: float  # Contribution to portfolio risk


@dataclass
class ActionInstruction:
    """Emergency action instruction"""
    action_id: str
    action_type: str
    priority: ActionPriority
    positions_affected: List[str]
    reduction_percentage: float
    execution_method: str
    time_limit_ms: float
    reason: str
    timestamp: datetime


@dataclass
class ExecutionResult:
    """Action execution result"""
    action_id: str
    status: ExecutionStatus
    execution_time_ms: float
    positions_processed: int
    volume_executed: float
    slippage_cost: float
    success_rate: float
    error_message: Optional[str]
    timestamp: datetime


class EmergencyActionExecutor:
    """
    Emergency Action Execution Engine
    
    Handles real-time execution of emergency risk management actions
    with microsecond precision and comprehensive monitoring.
    """
    
    def __init__(self, event_bus: EventBus, config: Dict[str, Any]):
        self.event_bus = event_bus
        self.config = config
        
        # Current portfolio state
        self.positions: Dict[str, Position] = {}
        self.portfolio_value: float = 0.0
        self.cash_available: float = 0.0
        
        # Execution parameters
        self.max_execution_time_ms = config.get('max_execution_time_ms', 5000)  # 5 seconds
        self.slippage_tolerance = config.get('slippage_tolerance', 0.001)  # 0.1%
        self.minimum_position_size = config.get('minimum_position_size', 100)
        
        # Emergency protocols
        self.emergency_liquidation_enabled = config.get('emergency_liquidation_enabled', True)
        self.hedge_instruments = config.get('hedge_instruments', ['SPY', 'VIX'])
        self.max_hedge_notional = config.get('max_hedge_notional', 1000000)  # $1M
        
        # Execution tracking
        self.active_actions: Dict[str, ActionInstruction] = {}
        self.execution_history: List[ExecutionResult] = []
        self.execution_queue: List[ActionInstruction] = []
        
        # Performance monitoring
        self.total_executions = 0
        self.successful_executions = 0
        self.execution_times: List[float] = []
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        self.logger = logger.bind(component="EmergencyActionExecutor")
        self.logger.info("Emergency Action Executor initialized")
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions for portfolio updates"""
        self.event_bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)
        self.event_bus.subscribe(EventType.EMERGENCY_STOP, self._handle_emergency_stop)
    
    def _handle_position_update(self, event: Event):
        """Handle position updates from portfolio system"""
        try:
            position_data = event.payload
            
            if hasattr(position_data, 'positions'):
                # Full portfolio update
                self.positions = {}
                total_value = 0.0
                
                for pos_data in position_data.positions:
                    position = Position(
                        symbol=pos_data.symbol,
                        quantity=pos_data.quantity,
                        market_value=pos_data.market_value,
                        side=pos_data.side if hasattr(pos_data, 'side') else 'long',
                        entry_price=pos_data.entry_price if hasattr(pos_data, 'entry_price') else pos_data.price,
                        current_price=pos_data.price,
                        unrealized_pnl=pos_data.unrealized_pnl if hasattr(pos_data, 'unrealized_pnl') else 0.0,
                        risk_contribution=pos_data.risk_contribution if hasattr(pos_data, 'risk_contribution') else 0.0
                    )
                    self.positions[pos_data.symbol] = position
                    total_value += abs(pos_data.market_value)
                
                self.portfolio_value = total_value
                
        except Exception as e:
            self.logger.error("Error handling position update", error=str(e))
    
    def _handle_emergency_stop(self, event: Event):
        """Handle emergency stop events"""
        self.logger.critical("Emergency stop received - executing immediate liquidation")
        
        # Create emergency liquidation instruction
        action_instruction = ActionInstruction(
            action_id=f"emergency_stop_{int(time.time())}",
            action_type="CLOSE_ALL",
            priority=ActionPriority.EMERGENCY,
            positions_affected=list(self.positions.keys()),
            reduction_percentage=1.0,  # 100% liquidation
            execution_method="MARKET",
            time_limit_ms=3000,  # 3 seconds maximum
            reason="Emergency stop triggered",
            timestamp=datetime.now()
        )
        
        # Execute immediately
        asyncio.create_task(self.execute_action(action_instruction))
    
    async def execute_reduce_position(
        self, 
        reduction_percentage: float,
        priority: ActionPriority = ActionPriority.HIGH,
        positions_to_reduce: Optional[List[str]] = None
    ) -> ExecutionResult:
        """
        Execute position reduction action
        
        Args:
            reduction_percentage: Percentage to reduce (0.0 to 1.0)
            priority: Execution priority
            positions_to_reduce: Specific positions to reduce (None = all positions)
            
        Returns:
            ExecutionResult with execution details
        """
        action_id = f"reduce_{int(time.time())}"
        
        if positions_to_reduce is None:
            positions_to_reduce = list(self.positions.keys())
        
        action_instruction = ActionInstruction(
            action_id=action_id,
            action_type="REDUCE_POSITION",
            priority=priority,
            positions_affected=positions_to_reduce,
            reduction_percentage=reduction_percentage,
            execution_method="SMART_ROUTING",
            time_limit_ms=self.max_execution_time_ms,
            reason=f"Risk reduction: {reduction_percentage:.1%}",
            timestamp=datetime.now()
        )
        
        return await self.execute_action(action_instruction)
    
    async def execute_close_all(
        self,
        priority: ActionPriority = ActionPriority.CRITICAL
    ) -> ExecutionResult:
        """
        Execute emergency liquidation of all positions
        
        Args:
            priority: Execution priority
            
        Returns:
            ExecutionResult with execution details
        """
        action_id = f"close_all_{int(time.time())}"
        
        action_instruction = ActionInstruction(
            action_id=action_id,
            action_type="CLOSE_ALL",
            priority=priority,
            positions_affected=list(self.positions.keys()),
            reduction_percentage=1.0,  # 100% liquidation
            execution_method="MARKET",
            time_limit_ms=3000,  # Emergency liquidation - 3 seconds max
            reason="Emergency liquidation",
            timestamp=datetime.now()
        )
        
        return await self.execute_action(action_instruction)
    
    async def execute_hedge(
        self,
        hedge_ratio: float = 0.8,
        priority: ActionPriority = ActionPriority.HIGH
    ) -> ExecutionResult:
        """
        Execute hedge creation to offset portfolio risk
        
        Args:
            hedge_ratio: Percentage of portfolio to hedge (0.0 to 1.0)
            priority: Execution priority
            
        Returns:
            ExecutionResult with execution details
        """
        action_id = f"hedge_{int(time.time())}"
        
        action_instruction = ActionInstruction(
            action_id=action_id,
            action_type="HEDGE",
            priority=priority,
            positions_affected=self.hedge_instruments,
            reduction_percentage=hedge_ratio,
            execution_method="HEDGE_STRATEGY",
            time_limit_ms=self.max_execution_time_ms,
            reason=f"Portfolio hedge: {hedge_ratio:.1%}",
            timestamp=datetime.now()
        )
        
        return await self.execute_action(action_instruction)
    
    async def execute_action(self, action_instruction: ActionInstruction) -> ExecutionResult:
        """
        Execute emergency action with real-time monitoring
        
        Args:
            action_instruction: Action to execute
            
        Returns:
            ExecutionResult with execution details
        """
        start_time = time.time()
        action_id = action_instruction.action_id
        
        try:
            # Add to active actions
            self.active_actions[action_id] = action_instruction
            
            # Log action start
            self.logger.info("Executing emergency action",
                           action_id=action_id,
                           action_type=action_instruction.action_type,
                           priority=action_instruction.priority.name)
            
            # Execute based on action type
            if action_instruction.action_type == "REDUCE_POSITION":
                result = await self._execute_position_reduction(action_instruction)
            elif action_instruction.action_type == "CLOSE_ALL":
                result = await self._execute_liquidation(action_instruction)
            elif action_instruction.action_type == "HEDGE":
                result = await self._execute_hedge_creation(action_instruction)
            else:
                raise ValueError(f"Unknown action type: {action_instruction.action_type}")
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time
            
            # Update tracking
            self.total_executions += 1
            if result.status == ExecutionStatus.COMPLETED:
                self.successful_executions += 1
            
            self.execution_times.append(execution_time)
            self.execution_history.append(result)
            
            # Cleanup
            if action_id in self.active_actions:
                del self.active_actions[action_id]
            
            # Publish execution result
            self.event_bus.publish(
                self.event_bus.create_event(
                    EventType.RISK_UPDATE,
                    {
                        'execution_result': asdict(result),
                        'action_instruction': asdict(action_instruction)
                    },
                    'EmergencyActionExecutor'
                )
            )
            
            self.logger.info("Emergency action completed",
                           action_id=action_id,
                           status=result.status.value,
                           execution_time_ms=execution_time)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            error_result = ExecutionResult(
                action_id=action_id,
                status=ExecutionStatus.FAILED,
                execution_time_ms=execution_time,
                positions_processed=0,
                volume_executed=0.0,
                slippage_cost=0.0,
                success_rate=0.0,
                error_message=str(e),
                timestamp=datetime.now()
            )
            
            self.execution_history.append(error_result)
            
            # Cleanup
            if action_id in self.active_actions:
                del self.active_actions[action_id]
            
            self.logger.error("Emergency action failed",
                            action_id=action_id,
                            error=str(e),
                            execution_time_ms=execution_time)
            
            return error_result
    
    async def _execute_position_reduction(self, instruction: ActionInstruction) -> ExecutionResult:
        """Execute position reduction with smart routing"""
        
        positions_processed = 0
        total_volume = 0.0
        total_slippage = 0.0
        
        for symbol in instruction.positions_affected:
            if symbol not in self.positions:
                continue
            
            position = self.positions[symbol]
            reduction_amount = abs(position.quantity) * instruction.reduction_percentage
            
            # Simulate execution (in real implementation, call trading system)
            await asyncio.sleep(0.001)  # Simulate network latency
            
            # Update position
            new_quantity = position.quantity * (1 - instruction.reduction_percentage)
            self.positions[symbol].quantity = new_quantity
            self.positions[symbol].market_value *= (1 - instruction.reduction_percentage)
            
            positions_processed += 1
            total_volume += abs(reduction_amount * position.current_price)
            total_slippage += abs(reduction_amount * position.current_price) * self.slippage_tolerance
            
            self.logger.debug("Position reduced",
                            symbol=symbol,
                            original_quantity=position.quantity,
                            new_quantity=new_quantity,
                            reduction_pct=f"{instruction.reduction_percentage:.1%}")
        
        success_rate = positions_processed / max(1, len(instruction.positions_affected))
        
        return ExecutionResult(
            action_id=instruction.action_id,
            status=ExecutionStatus.COMPLETED if success_rate > 0.8 else ExecutionStatus.FAILED,
            execution_time_ms=0,  # Will be set by caller
            positions_processed=positions_processed,
            volume_executed=total_volume,
            slippage_cost=total_slippage,
            success_rate=success_rate,
            error_message=None,
            timestamp=datetime.now()
        )
    
    async def _execute_liquidation(self, instruction: ActionInstruction) -> ExecutionResult:
        """Execute emergency liquidation of all positions"""
        
        positions_processed = 0
        total_volume = 0.0
        total_slippage = 0.0
        
        # Sort positions by risk contribution (highest risk first)
        sorted_positions = sorted(
            self.positions.items(),
            key=lambda x: abs(x[1].risk_contribution),
            reverse=True
        )
        
        for symbol, position in sorted_positions:
            # Simulate market order execution
            await asyncio.sleep(0.0005)  # Ultra-fast execution
            
            volume = abs(position.quantity * position.current_price)
            slippage = volume * (self.slippage_tolerance * 2)  # Higher slippage for market orders
            
            # Clear position
            self.positions[symbol].quantity = 0
            self.positions[symbol].market_value = 0
            
            positions_processed += 1
            total_volume += volume
            total_slippage += slippage
            
            self.logger.debug("Position liquidated",
                            symbol=symbol,
                            volume=volume,
                            slippage=slippage)
        
        success_rate = 1.0  # Emergency liquidation always succeeds
        
        return ExecutionResult(
            action_id=instruction.action_id,
            status=ExecutionStatus.COMPLETED,
            execution_time_ms=0,  # Will be set by caller
            positions_processed=positions_processed,
            volume_executed=total_volume,
            slippage_cost=total_slippage,
            success_rate=success_rate,
            error_message=None,
            timestamp=datetime.now()
        )
    
    async def _execute_hedge_creation(self, instruction: ActionInstruction) -> ExecutionResult:
        """Execute hedge position creation"""
        
        hedge_notional = self.portfolio_value * instruction.reduction_percentage
        hedge_notional = min(hedge_notional, self.max_hedge_notional)
        
        positions_created = 0
        total_volume = 0.0
        total_slippage = 0.0
        
        # Create hedge positions in specified instruments
        for hedge_instrument in self.hedge_instruments:
            if hedge_notional <= 0:
                break
            
            # Calculate hedge size (simplified)
            instrument_notional = hedge_notional / len(self.hedge_instruments)
            
            # Simulate hedge execution
            await asyncio.sleep(0.002)  # Hedge creation takes slightly longer
            
            slippage = instrument_notional * self.slippage_tolerance
            
            # Create synthetic hedge position
            hedge_symbol = f"{hedge_instrument}_HEDGE"
            if hedge_symbol not in self.positions:
                self.positions[hedge_symbol] = Position(
                    symbol=hedge_symbol,
                    quantity=0,
                    market_value=0,
                    side='short',
                    entry_price=100.0,  # Simplified
                    current_price=100.0,
                    unrealized_pnl=0.0,
                    risk_contribution=-instrument_notional / self.portfolio_value  # Negative correlation
                )
            
            # Update hedge position
            self.positions[hedge_symbol].quantity -= instrument_notional / 100.0  # Short position
            self.positions[hedge_symbol].market_value -= instrument_notional
            
            positions_created += 1
            total_volume += instrument_notional
            total_slippage += slippage
            
            self.logger.debug("Hedge position created",
                            instrument=hedge_instrument,
                            notional=instrument_notional,
                            slippage=slippage)
        
        success_rate = positions_created / max(1, len(self.hedge_instruments))
        
        return ExecutionResult(
            action_id=instruction.action_id,
            status=ExecutionStatus.COMPLETED if success_rate > 0.5 else ExecutionStatus.FAILED,
            execution_time_ms=0,  # Will be set by caller
            positions_processed=positions_created,
            volume_executed=total_volume,
            slippage_cost=total_slippage,
            success_rate=success_rate,
            error_message=None,
            timestamp=datetime.now()
        )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution performance statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 1.0,
                "avg_execution_time_ms": 0.0,
                "total_volume_executed": 0.0,
                "total_slippage_cost": 0.0
            }
        
        successful = [r for r in self.execution_history if r.status == ExecutionStatus.COMPLETED]
        
        avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0.0
        total_volume = sum(r.volume_executed for r in self.execution_history)
        total_slippage = sum(r.slippage_cost for r in self.execution_history)
        
        return {
            "total_executions": self.total_executions,
            "success_rate": len(successful) / len(self.execution_history),
            "avg_execution_time_ms": avg_execution_time,
            "total_volume_executed": total_volume,
            "total_slippage_cost": total_slippage,
            "active_actions": len(self.active_actions),
            "portfolio_value": self.portfolio_value,
            "positions_count": len(self.positions)
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        if not self.positions:
            return {"portfolio_value": 0.0, "positions_count": 0, "total_risk": 0.0}
        
        total_risk = sum(abs(pos.risk_contribution) for pos in self.positions.values())
        long_positions = sum(1 for pos in self.positions.values() if pos.quantity > 0)
        short_positions = sum(1 for pos in self.positions.values() if pos.quantity < 0)
        
        return {
            "portfolio_value": self.portfolio_value,
            "positions_count": len(self.positions),
            "long_positions": long_positions,
            "short_positions": short_positions,
            "total_risk": total_risk,
            "cash_available": self.cash_available,
            "largest_position": max(
                (abs(pos.market_value) for pos in self.positions.values()),
                default=0.0
            )
        }