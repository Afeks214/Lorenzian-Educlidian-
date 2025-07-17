"""
Dynamic Stop-Loss and Take-Profit Agent

Implements intelligent stop-loss and take-profit management with:
- Volatility-adjusted stop levels
- Minimum 1.5:1 risk/reward ratio
- Trailing stops and targets
- Market condition adaptive adjustments
- Partial profit taking
- Dynamic level updates based on market structure
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import structlog
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import yaml

logger = structlog.get_logger()


class StopTargetType(Enum):
    """Types of stop/target orders"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TRAILING_PROFIT = "trailing_profit"
    PARTIAL_PROFIT = "partial_profit"


class MarketStructure(Enum):
    """Market structure conditions"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATING = "consolidating"


@dataclass
class StopTargetLevel:
    """Stop or target level specification"""
    level_type: StopTargetType
    price: float
    size: float  # Position size affected
    volatility_adjusted: bool
    trail_distance: Optional[float] = None
    activation_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_level(self, new_price: float, current_price: float) -> bool:
        """Update trailing level if applicable"""
        if self.level_type in [StopTargetType.TRAILING_STOP, StopTargetType.TRAILING_PROFIT]:
            if self.trail_distance:
                old_price = self.price
                self.price = new_price
                self.last_updated = datetime.now()
                return old_price != new_price
        return False


@dataclass
class PositionRiskProfile:
    """Risk profile for a position"""
    symbol: str
    entry_price: float
    current_price: float
    position_size: float
    volatility: float
    market_structure: MarketStructure
    stop_levels: List[StopTargetLevel] = field(default_factory=list)
    target_levels: List[StopTargetLevel] = field(default_factory=list)
    unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    def calculate_risk_reward(self) -> float:
        """Calculate current risk/reward ratio"""
        if not self.stop_levels or not self.target_levels:
            return 0.0
        
        stop_distance = abs(self.current_price - self.stop_levels[0].price)
        target_distance = abs(self.target_levels[0].price - self.current_price)
        
        if stop_distance > 0:
            return target_distance / stop_distance
        return 0.0


class DynamicStopTargetAgent:
    """
    Dynamic Stop-Loss and Take-Profit Agent
    
    Features:
    - Volatility-adjusted stop-loss levels (2% of position minimum)
    - Take-profit targets with 1.5:1 minimum risk/reward
    - Trailing stops and profits
    - Partial profit taking at predefined levels
    - Market structure adaptive adjustments
    - Real-time level updates
    """
    
    def __init__(self, config_path: str = "/home/QuantNova/GrandModel/config/risk_management_config.yaml"):
        """Initialize Dynamic Stop-Target Agent"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.risk_config = self.config['risk_controls']
        self.stop_config = self.risk_config['stop_loss']
        self.target_config = self.risk_config['take_profit']
        
        # Stop-loss parameters
        self.default_stop_percent = self.stop_config['default_percent']
        self.max_stop_percent = self.stop_config['max_stop_percent']
        self.min_stop_percent = self.stop_config['min_stop_percent']
        self.volatility_adjusted = self.stop_config['volatility_adjusted']
        self.trailing_stop_enabled = self.stop_config['trailing_stop']
        
        # Take-profit parameters
        self.min_risk_reward_ratio = self.target_config['risk_reward_ratio']
        self.volatility_adjusted_targets = self.target_config['volatility_adjusted']
        self.partial_profit_levels = self.target_config['partial_profit_levels']
        self.trailing_profit_enabled = self.target_config['trailing_profit']
        
        # Position tracking
        self.position_profiles: Dict[str, PositionRiskProfile] = {}
        self.executed_stops_targets = []
        self.performance_metrics = {
            'total_stops_hit': 0,
            'total_targets_hit': 0,
            'avg_risk_reward_realized': 0.0,
            'trailing_stop_improvements': 0,
            'partial_profit_executions': 0
        }
        
        logger.info("Dynamic Stop-Target Agent initialized",
                   default_stop_percent=self.default_stop_percent,
                   min_risk_reward_ratio=self.min_risk_reward_ratio,
                   trailing_enabled=self.trailing_stop_enabled)
    
    def create_position_levels(self, symbol: str, entry_price: float, 
                             position_size: float, direction: str,
                             volatility: float = None,
                             market_structure: MarketStructure = MarketStructure.RANGING) -> PositionRiskProfile:
        """
        Create stop-loss and take-profit levels for a new position
        
        Args:
            symbol: Trading symbol
            entry_price: Position entry price
            position_size: Position size (positive for long, negative for short)
            direction: Position direction ("long" or "short")
            volatility: Asset volatility (optional)
            market_structure: Current market structure
            
        Returns:
            PositionRiskProfile with configured levels
        """
        try:
            # Get or estimate volatility
            if volatility is None:
                volatility = self._estimate_volatility(symbol)
            
            # Create position profile
            profile = PositionRiskProfile(
                symbol=symbol,
                entry_price=entry_price,
                current_price=entry_price,
                position_size=position_size,
                volatility=volatility,
                market_structure=market_structure
            )
            
            # Calculate stop-loss level
            stop_level = self._calculate_stop_loss_level(
                entry_price, volatility, direction, market_structure)
            
            # Calculate take-profit levels
            target_levels = self._calculate_take_profit_levels(
                entry_price, stop_level, direction, volatility, market_structure)
            
            # Create stop-loss order
            stop_order = StopTargetLevel(
                level_type=StopTargetType.TRAILING_STOP if self.trailing_stop_enabled else StopTargetType.STOP_LOSS,
                price=stop_level,
                size=abs(position_size),
                volatility_adjusted=self.volatility_adjusted,
                trail_distance=self._calculate_trailing_distance(volatility) if self.trailing_stop_enabled else None
            )
            profile.stop_levels.append(stop_order)
            
            # Create take-profit orders
            for i, (target_price, target_size) in enumerate(target_levels):
                target_order = StopTargetLevel(
                    level_type=StopTargetType.PARTIAL_PROFIT if i < len(target_levels) - 1 else StopTargetType.TAKE_PROFIT,
                    price=target_price,
                    size=target_size,
                    volatility_adjusted=self.volatility_adjusted_targets,
                    risk_reward_ratio=self._calculate_risk_reward_ratio(entry_price, stop_level, target_price)
                )
                profile.target_levels.append(target_order)
            
            # Store position profile
            self.position_profiles[symbol] = profile
            
            logger.info("Position levels created",
                       symbol=symbol,
                       entry_price=entry_price,
                       stop_level=stop_level,
                       target_levels=[t[0] for t in target_levels],
                       risk_reward_ratio=profile.calculate_risk_reward())
            
            return profile
            
        except Exception as e:
            logger.error("Error creating position levels", error=str(e), symbol=symbol)
            return self._create_default_profile(symbol, entry_price, position_size, volatility or 0.02)
    
    def _calculate_stop_loss_level(self, entry_price: float, volatility: float, 
                                 direction: str, market_structure: MarketStructure) -> float:
        """Calculate stop-loss level based on volatility and market structure"""
        try:
            # Base stop percentage
            base_stop = self.default_stop_percent
            
            # Volatility adjustment
            if self.volatility_adjusted:
                volatility_multiplier = 1.0 + (volatility - 0.02) * 2.0  # Adjust around 2% base volatility
                base_stop *= volatility_multiplier
            
            # Market structure adjustment
            structure_multiplier = self._get_market_structure_stop_multiplier(market_structure)
            adjusted_stop = base_stop * structure_multiplier
            
            # Apply bounds
            final_stop = max(self.min_stop_percent, min(self.max_stop_percent, adjusted_stop))
            
            # Calculate stop price
            if direction.lower() == "long":
                stop_price = entry_price * (1 - final_stop)
            else:  # short
                stop_price = entry_price * (1 + final_stop)
            
            return stop_price
            
        except Exception as e:
            logger.error("Error calculating stop-loss level", error=str(e))
            return entry_price * 0.98 if direction.lower() == "long" else entry_price * 1.02
    
    def _calculate_take_profit_levels(self, entry_price: float, stop_level: float, 
                                    direction: str, volatility: float,
                                    market_structure: MarketStructure) -> List[Tuple[float, float]]:
        """Calculate take-profit levels with partial profit taking"""
        try:
            stop_distance = abs(entry_price - stop_level)
            target_levels = []
            
            # Calculate primary target based on risk/reward ratio
            base_target_distance = stop_distance * self.min_risk_reward_ratio
            
            # Market structure adjustment
            structure_multiplier = self._get_market_structure_target_multiplier(market_structure)
            adjusted_target_distance = base_target_distance * structure_multiplier
            
            # Volatility adjustment
            if self.volatility_adjusted_targets:
                volatility_multiplier = 1.0 + (volatility - 0.02) * 1.5
                adjusted_target_distance *= volatility_multiplier
            
            # Calculate target price
            if direction.lower() == "long":
                primary_target = entry_price + adjusted_target_distance
            else:  # short
                primary_target = entry_price - adjusted_target_distance
            
            # Create partial profit levels
            remaining_size = 1.0  # 100% of position
            
            for i, profit_level in enumerate(self.partial_profit_levels):
                if direction.lower() == "long":
                    target_price = entry_price + (adjusted_target_distance * profit_level)
                else:
                    target_price = entry_price - (adjusted_target_distance * profit_level)
                
                # Size for this level (percentage of remaining position)
                if i < len(self.partial_profit_levels) - 1:
                    level_size = remaining_size * 0.5  # Take 50% of remaining
                    remaining_size -= level_size
                else:
                    level_size = remaining_size  # Take all remaining
                
                target_levels.append((target_price, level_size))
            
            # Add final target if no partial levels hit primary target
            if not target_levels or target_levels[-1][0] != primary_target:
                target_levels.append((primary_target, remaining_size))
            
            return target_levels
            
        except Exception as e:
            logger.error("Error calculating take-profit levels", error=str(e))
            # Default fallback
            if direction.lower() == "long":
                return [(entry_price * 1.03, 1.0)]  # 3% profit target
            else:
                return [(entry_price * 0.97, 1.0)]  # 3% profit target
    
    def update_position_levels(self, symbol: str, current_price: float, 
                             volume: float = None) -> Dict[str, Any]:
        """
        Update position levels based on current market conditions
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            volume: Current volume (optional)
            
        Returns:
            Dictionary with update information
        """
        if symbol not in self.position_profiles:
            return {'status': 'no_position', 'symbol': symbol}
        
        try:
            profile = self.position_profiles[symbol]
            profile.current_price = current_price
            
            # Update unrealized PnL
            if profile.position_size > 0:  # Long position
                profile.unrealized_pnl = (current_price - profile.entry_price) * profile.position_size
            else:  # Short position
                profile.unrealized_pnl = (profile.entry_price - current_price) * abs(profile.position_size)
            
            # Update max favorable/adverse excursions
            if profile.unrealized_pnl > profile.max_favorable_excursion:
                profile.max_favorable_excursion = profile.unrealized_pnl
            if profile.unrealized_pnl < profile.max_adverse_excursion:
                profile.max_adverse_excursion = profile.unrealized_pnl
            
            updates = {
                'symbol': symbol,
                'current_price': current_price,
                'unrealized_pnl': profile.unrealized_pnl,
                'stop_updates': [],
                'target_updates': [],
                'triggered_levels': []
            }
            
            # Update trailing stops
            for stop_level in profile.stop_levels:
                if stop_level.level_type == StopTargetType.TRAILING_STOP:
                    if self._update_trailing_stop(stop_level, current_price, profile):
                        updates['stop_updates'].append({
                            'old_price': stop_level.price,
                            'new_price': stop_level.price,
                            'type': 'trailing_stop'
                        })
                        self.performance_metrics['trailing_stop_improvements'] += 1
            
            # Update trailing profits
            for target_level in profile.target_levels:
                if target_level.level_type == StopTargetType.TRAILING_PROFIT:
                    if self._update_trailing_profit(target_level, current_price, profile):
                        updates['target_updates'].append({
                            'old_price': target_level.price,
                            'new_price': target_level.price,
                            'type': 'trailing_profit'
                        })
            
            # Check for triggered levels
            triggered_stops = self._check_triggered_stops(profile, current_price)
            triggered_targets = self._check_triggered_targets(profile, current_price)
            
            updates['triggered_levels'] = triggered_stops + triggered_targets
            
            # Update performance metrics
            self.performance_metrics['total_stops_hit'] += len(triggered_stops)
            self.performance_metrics['total_targets_hit'] += len(triggered_targets)
            
            return updates
            
        except Exception as e:
            logger.error("Error updating position levels", error=str(e), symbol=symbol)
            return {'status': 'error', 'symbol': symbol, 'error': str(e)}
    
    def _update_trailing_stop(self, stop_level: StopTargetLevel, current_price: float, 
                            profile: PositionRiskProfile) -> bool:
        """Update trailing stop level"""
        if not stop_level.trail_distance:
            return False
        
        try:
            if profile.position_size > 0:  # Long position
                # Move stop up if price moves favorably
                new_stop_price = current_price - stop_level.trail_distance
                if new_stop_price > stop_level.price:
                    stop_level.price = new_stop_price
                    stop_level.last_updated = datetime.now()
                    return True
            else:  # Short position
                # Move stop down if price moves favorably
                new_stop_price = current_price + stop_level.trail_distance
                if new_stop_price < stop_level.price:
                    stop_level.price = new_stop_price
                    stop_level.last_updated = datetime.now()
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Error updating trailing stop", error=str(e))
            return False
    
    def _update_trailing_profit(self, target_level: StopTargetLevel, current_price: float, 
                              profile: PositionRiskProfile) -> bool:
        """Update trailing profit level"""
        if not target_level.trail_distance:
            return False
        
        try:
            if profile.position_size > 0:  # Long position
                # Move target down if price moves unfavorably but maintain trailing distance
                new_target_price = current_price + target_level.trail_distance
                if new_target_price < target_level.price:
                    target_level.price = new_target_price
                    target_level.last_updated = datetime.now()
                    return True
            else:  # Short position
                # Move target up if price moves unfavorably but maintain trailing distance
                new_target_price = current_price - target_level.trail_distance
                if new_target_price > target_level.price:
                    target_level.price = new_target_price
                    target_level.last_updated = datetime.now()
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Error updating trailing profit", error=str(e))
            return False
    
    def _check_triggered_stops(self, profile: PositionRiskProfile, 
                             current_price: float) -> List[Dict[str, Any]]:
        """Check for triggered stop levels"""
        triggered = []
        
        for stop_level in profile.stop_levels:
            if profile.position_size > 0:  # Long position
                if current_price <= stop_level.price:
                    triggered.append({
                        'type': 'stop_loss',
                        'price': stop_level.price,
                        'size': stop_level.size,
                        'pnl': (stop_level.price - profile.entry_price) * profile.position_size
                    })
            else:  # Short position
                if current_price >= stop_level.price:
                    triggered.append({
                        'type': 'stop_loss',
                        'price': stop_level.price,
                        'size': stop_level.size,
                        'pnl': (profile.entry_price - stop_level.price) * abs(profile.position_size)
                    })
        
        return triggered
    
    def _check_triggered_targets(self, profile: PositionRiskProfile, 
                               current_price: float) -> List[Dict[str, Any]]:
        """Check for triggered target levels"""
        triggered = []
        
        for target_level in profile.target_levels:
            if profile.position_size > 0:  # Long position
                if current_price >= target_level.price:
                    triggered.append({
                        'type': 'take_profit',
                        'price': target_level.price,
                        'size': target_level.size,
                        'pnl': (target_level.price - profile.entry_price) * target_level.size
                    })
            else:  # Short position
                if current_price <= target_level.price:
                    triggered.append({
                        'type': 'take_profit',
                        'price': target_level.price,
                        'size': target_level.size,
                        'pnl': (profile.entry_price - target_level.price) * target_level.size
                    })
        
        return triggered
    
    def _get_market_structure_stop_multiplier(self, market_structure: MarketStructure) -> float:
        """Get stop multiplier based on market structure"""
        multipliers = {
            MarketStructure.TRENDING_UP: 0.8,      # Tighter stops in trends
            MarketStructure.TRENDING_DOWN: 0.8,
            MarketStructure.RANGING: 1.0,          # Normal stops in ranges
            MarketStructure.VOLATILE: 1.5,         # Wider stops in volatility
            MarketStructure.CONSOLIDATING: 1.2     # Slightly wider in consolidation
        }
        return multipliers.get(market_structure, 1.0)
    
    def _get_market_structure_target_multiplier(self, market_structure: MarketStructure) -> float:
        """Get target multiplier based on market structure"""
        multipliers = {
            MarketStructure.TRENDING_UP: 1.3,      # Bigger targets in trends
            MarketStructure.TRENDING_DOWN: 1.3,
            MarketStructure.RANGING: 0.9,          # Smaller targets in ranges
            MarketStructure.VOLATILE: 1.1,         # Slightly bigger in volatility
            MarketStructure.CONSOLIDATING: 0.8     # Smaller in consolidation
        }
        return multipliers.get(market_structure, 1.0)
    
    def _calculate_trailing_distance(self, volatility: float) -> float:
        """Calculate trailing distance based on volatility"""
        base_distance = self.default_stop_percent
        volatility_adjustment = volatility * 2.0  # Scale with volatility
        return base_distance + volatility_adjustment
    
    def _calculate_risk_reward_ratio(self, entry_price: float, stop_price: float, 
                                   target_price: float) -> float:
        """Calculate risk/reward ratio"""
        try:
            risk = abs(entry_price - stop_price)
            reward = abs(target_price - entry_price)
            return reward / risk if risk > 0 else 0.0
        except:
            return 0.0
    
    def _estimate_volatility(self, symbol: str) -> float:
        """Estimate volatility for symbol"""
        # In practice, this would use historical data
        return 0.02  # 2% default volatility
    
    def _create_default_profile(self, symbol: str, entry_price: float, 
                              position_size: float, volatility: float) -> PositionRiskProfile:
        """Create default position profile for error cases"""
        return PositionRiskProfile(
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            position_size=position_size,
            volatility=volatility,
            market_structure=MarketStructure.RANGING
        )
    
    def get_position_summary(self, symbol: str) -> Dict[str, Any]:
        """Get position summary including current levels"""
        if symbol not in self.position_profiles:
            return {'status': 'no_position', 'symbol': symbol}
        
        profile = self.position_profiles[symbol]
        
        return {
            'symbol': symbol,
            'entry_price': profile.entry_price,
            'current_price': profile.current_price,
            'position_size': profile.position_size,
            'unrealized_pnl': profile.unrealized_pnl,
            'max_favorable_excursion': profile.max_favorable_excursion,
            'max_adverse_excursion': profile.max_adverse_excursion,
            'current_risk_reward': profile.calculate_risk_reward(),
            'stop_levels': [
                {
                    'type': level.level_type.value,
                    'price': level.price,
                    'size': level.size,
                    'trail_distance': level.trail_distance
                }
                for level in profile.stop_levels
            ],
            'target_levels': [
                {
                    'type': level.level_type.value,
                    'price': level.price,
                    'size': level.size,
                    'risk_reward_ratio': level.risk_reward_ratio
                }
                for level in profile.target_levels
            ]
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for stop/target management"""
        total_executions = (self.performance_metrics['total_stops_hit'] + 
                          self.performance_metrics['total_targets_hit'])
        
        if total_executions > 0:
            target_hit_rate = self.performance_metrics['total_targets_hit'] / total_executions
        else:
            target_hit_rate = 0.0
        
        return {
            'total_stops_hit': self.performance_metrics['total_stops_hit'],
            'total_targets_hit': self.performance_metrics['total_targets_hit'],
            'target_hit_rate': target_hit_rate,
            'trailing_stop_improvements': self.performance_metrics['trailing_stop_improvements'],
            'partial_profit_executions': self.performance_metrics['partial_profit_executions'],
            'avg_risk_reward_realized': self.performance_metrics['avg_risk_reward_realized'],
            'active_positions': len(self.position_profiles)
        }
    
    def close_position(self, symbol: str, close_price: float) -> Dict[str, Any]:
        """Close position and calculate final metrics"""
        if symbol not in self.position_profiles:
            return {'status': 'no_position', 'symbol': symbol}
        
        profile = self.position_profiles[symbol]
        
        # Calculate final PnL
        if profile.position_size > 0:  # Long position
            final_pnl = (close_price - profile.entry_price) * profile.position_size
        else:  # Short position
            final_pnl = (profile.entry_price - close_price) * abs(profile.position_size)
        
        # Create close summary
        close_summary = {
            'symbol': symbol,
            'entry_price': profile.entry_price,
            'close_price': close_price,
            'position_size': profile.position_size,
            'final_pnl': final_pnl,
            'max_favorable_excursion': profile.max_favorable_excursion,
            'max_adverse_excursion': profile.max_adverse_excursion,
            'hold_time': datetime.now() - profile.stop_levels[0].created_at if profile.stop_levels else None
        }
        
        # Remove from active positions
        del self.position_profiles[symbol]
        
        # Add to executed history
        self.executed_stops_targets.append(close_summary)
        
        return close_summary