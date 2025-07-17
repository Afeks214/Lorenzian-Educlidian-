"""
Execution Rules and Market Impact Modeling Agent

Implements execution rules and market impact considerations:
- Entry/exit execution logic
- Market impact modeling
- Realistic fill modeling
- Execution cost estimation
- Volume and spread constraints
- Participation rate limits
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import structlog
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import yaml

logger = structlog.get_logger()


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ExecutionQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"


@dataclass
class ExecutionRule:
    """Execution rule specification"""
    rule_type: str
    condition: str
    action: str
    priority: int
    enabled: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if rule condition is met"""
        # Simplified rule evaluation
        return True


@dataclass
class MarketImpactModel:
    """Market impact model parameters"""
    model_type: str  # "square_root", "linear", "power_law"
    participation_rate: float
    volatility: float
    adv: float  # Average daily volume
    impact_coefficient: float
    
    def calculate_impact(self, order_size: float, volume: float) -> float:
        """Calculate market impact for order"""
        if self.model_type == "square_root":
            return self.impact_coefficient * np.sqrt(order_size / volume)
        elif self.model_type == "linear":
            return self.impact_coefficient * (order_size / volume)
        else:
            return self.impact_coefficient * (order_size / volume) ** 0.6


@dataclass
class ExecutionCosts:
    """Execution cost breakdown"""
    commission: float
    spread_cost: float
    market_impact: float
    slippage: float
    opportunity_cost: float
    total_cost: float
    
    def calculate_total(self) -> float:
        """Calculate total execution cost"""
        self.total_cost = (self.commission + self.spread_cost + 
                          self.market_impact + self.slippage + self.opportunity_cost)
        return self.total_cost


class ExecutionRulesAgent:
    """
    Execution Rules and Market Impact Agent
    
    Features:
    - Execution rule engine
    - Market impact modeling
    - Fill probability estimation
    - Execution cost calculation
    - Volume and spread constraints
    - Participation rate monitoring
    """
    
    def __init__(self, config_path: str = "/home/QuantNova/GrandModel/config/risk_management_config.yaml"):
        """Initialize Execution Rules Agent"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.execution_config = self.config['execution_rules']
        self.entry_rules = self.execution_config['entry_rules']
        self.exit_rules = self.execution_config['exit_rules']
        self.market_impact_config = self.execution_config['market_impact']
        self.fill_modeling_config = self.execution_config['fill_modeling']
        self.cost_config = self.execution_config['cost_estimation']
        
        # Execution parameters
        self.min_signal_confidence = self.entry_rules['min_signal_confidence']
        self.max_position_correlation = self.entry_rules['max_position_correlation']
        self.volume_threshold = self.entry_rules['volume_threshold']
        self.spread_threshold = self.entry_rules['spread_threshold']
        self.participation_rate = self.market_impact_config['participation_rate']
        self.impact_threshold = self.market_impact_config['impact_threshold']
        
        # Execution rules
        self.execution_rules = self._initialize_execution_rules()
        
        # Market impact models
        self.impact_models = {}
        
        # Execution metrics
        self.execution_metrics = {
            'total_orders': 0,
            'successful_executions': 0,
            'blocked_orders': 0,
            'avg_execution_cost': 0.0,
            'avg_market_impact': 0.0,
            'avg_fill_rate': 0.0,
            'rule_violations': 0
        }
        
        logger.info("Execution Rules Agent initialized",
                   min_signal_confidence=self.min_signal_confidence,
                   participation_rate=self.participation_rate,
                   impact_threshold=self.impact_threshold)
    
    def _initialize_execution_rules(self) -> List[ExecutionRule]:
        """Initialize execution rules"""
        rules = [
            ExecutionRule(
                rule_type="entry",
                condition="signal_confidence >= min_threshold",
                action="allow_entry",
                priority=1
            ),
            ExecutionRule(
                rule_type="entry",
                condition="volume >= volume_threshold",
                action="allow_entry",
                priority=2
            ),
            ExecutionRule(
                rule_type="entry",
                condition="spread <= spread_threshold",
                action="allow_entry",
                priority=3
            ),
            ExecutionRule(
                rule_type="exit",
                condition="stop_loss_triggered",
                action="force_exit",
                priority=1
            ),
            ExecutionRule(
                rule_type="exit",
                condition="take_profit_triggered",
                action="force_exit",
                priority=2
            ),
            ExecutionRule(
                rule_type="risk",
                condition="market_impact > impact_threshold",
                action="reduce_order_size",
                priority=1
            )
        ]
        return rules
    
    def evaluate_entry_execution(self, symbol: str, order_size: float, 
                                signal_confidence: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate entry execution rules
        
        Args:
            symbol: Trading symbol
            order_size: Proposed order size
            signal_confidence: Signal confidence (0-1)
            market_data: Current market data
            
        Returns:
            Execution evaluation result
        """
        try:
            result = {
                'symbol': symbol,
                'order_size': order_size,
                'signal_confidence': signal_confidence,
                'execution_allowed': True,
                'rule_violations': [],
                'execution_quality': ExecutionQuality.GOOD,
                'recommended_adjustments': [],
                'estimated_costs': None,
                'market_impact': None
            }
            
            # Extract market data
            current_price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            spread = market_data.get('spread', 0)
            bid_size = market_data.get('bid_size', 0)
            ask_size = market_data.get('ask_size', 0)
            
            # Check signal confidence
            if signal_confidence < self.min_signal_confidence:
                result['execution_allowed'] = False
                result['rule_violations'].append({
                    'rule': 'min_signal_confidence',
                    'value': signal_confidence,
                    'threshold': self.min_signal_confidence
                })
            
            # Check volume threshold
            if volume < self.volume_threshold:
                result['execution_allowed'] = False
                result['rule_violations'].append({
                    'rule': 'volume_threshold',
                    'value': volume,
                    'threshold': self.volume_threshold
                })
            
            # Check spread threshold
            if current_price > 0:
                spread_pct = spread / current_price
                if spread_pct > self.spread_threshold:
                    result['execution_allowed'] = False
                    result['rule_violations'].append({
                        'rule': 'spread_threshold',
                        'value': spread_pct,
                        'threshold': self.spread_threshold
                    })
            
            # Calculate market impact
            market_impact = self._calculate_market_impact(
                symbol, order_size, volume, current_price)
            result['market_impact'] = market_impact
            
            # Check market impact threshold
            if market_impact > self.impact_threshold:
                result['recommended_adjustments'].append({
                    'type': 'reduce_size',
                    'reason': 'high_market_impact',
                    'current_impact': market_impact,
                    'threshold': self.impact_threshold,
                    'recommended_size': order_size * (self.impact_threshold / market_impact)
                })
            
            # Calculate execution costs
            execution_costs = self._calculate_execution_costs(
                symbol, order_size, current_price, market_data)
            result['estimated_costs'] = execution_costs
            
            # Determine execution quality
            result['execution_quality'] = self._assess_execution_quality(
                signal_confidence, market_impact, spread_pct, volume)
            
            # Update metrics
            self.execution_metrics['total_orders'] += 1
            if result['execution_allowed']:
                self.execution_metrics['successful_executions'] += 1
            else:
                self.execution_metrics['blocked_orders'] += 1
            
            self.execution_metrics['rule_violations'] += len(result['rule_violations'])
            
            return result
            
        except Exception as e:
            logger.error("Error evaluating entry execution", error=str(e), symbol=symbol)
            return {'error': str(e)}
    
    def evaluate_exit_execution(self, symbol: str, order_size: float, 
                               exit_reason: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate exit execution rules"""
        try:
            result = {
                'symbol': symbol,
                'order_size': order_size,
                'exit_reason': exit_reason,
                'execution_allowed': True,
                'urgency_level': 'normal',
                'recommended_order_type': OrderType.LIMIT,
                'execution_strategy': 'standard',
                'estimated_costs': None
            }
            
            # Determine urgency based on exit reason
            if exit_reason in ['stop_loss', 'margin_call', 'risk_limit']:
                result['urgency_level'] = 'high'
                result['recommended_order_type'] = OrderType.MARKET
                result['execution_strategy'] = 'aggressive'
            elif exit_reason in ['take_profit', 'rebalance']:
                result['urgency_level'] = 'low'
                result['recommended_order_type'] = OrderType.LIMIT
                result['execution_strategy'] = 'patient'
            
            # Calculate execution costs
            current_price = market_data.get('price', 0)
            execution_costs = self._calculate_execution_costs(
                symbol, order_size, current_price, market_data)
            result['estimated_costs'] = execution_costs
            
            return result
            
        except Exception as e:
            logger.error("Error evaluating exit execution", error=str(e), symbol=symbol)
            return {'error': str(e)}
    
    def _calculate_market_impact(self, symbol: str, order_size: float, 
                               volume: float, price: float) -> float:
        """Calculate market impact for order"""
        try:
            if volume <= 0 or price <= 0:
                return 0.0
            
            # Get or create impact model for symbol
            if symbol not in self.impact_models:
                self.impact_models[symbol] = MarketImpactModel(
                    model_type=self.market_impact_config['impact_model'],
                    participation_rate=self.participation_rate,
                    volatility=0.02,  # Default 2% volatility
                    adv=volume * 100,  # Estimate average daily volume
                    impact_coefficient=0.01
                )
            
            model = self.impact_models[symbol]
            impact = model.calculate_impact(abs(order_size), volume)
            
            # Convert to price impact
            price_impact = impact * price
            
            return price_impact
            
        except Exception as e:
            logger.error("Error calculating market impact", error=str(e), symbol=symbol)
            return 0.0
    
    def _calculate_execution_costs(self, symbol: str, order_size: float, 
                                 price: float, market_data: Dict[str, Any]) -> ExecutionCosts:
        """Calculate comprehensive execution costs"""
        try:
            # Commission cost
            commission = self.cost_config['commission_per_share'] * abs(order_size)
            
            # Spread cost
            spread = market_data.get('spread', 0)
            spread_cost = spread * abs(order_size) * self.cost_config['spread_cost_multiplier']
            
            # Market impact cost
            volume = market_data.get('volume', 1)
            market_impact = self._calculate_market_impact(symbol, order_size, volume, price)
            market_impact_cost = market_impact * abs(order_size) * self.cost_config['market_impact_multiplier']
            
            # Slippage cost
            volatility = market_data.get('volatility', 0.02)
            base_slippage = self.fill_modeling_config['base_slippage']
            volatility_multiplier = self.fill_modeling_config['volatility_multiplier']
            slippage = base_slippage * (1 + volatility * volatility_multiplier)
            slippage_cost = slippage * price * abs(order_size)
            
            # Opportunity cost (if enabled)
            opportunity_cost = 0.0
            if self.cost_config['opportunity_cost_enabled']:
                # Simplified opportunity cost calculation
                opportunity_cost = 0.001 * price * abs(order_size)  # 0.1% of notional
            
            costs = ExecutionCosts(
                commission=commission,
                spread_cost=spread_cost,
                market_impact=market_impact_cost,
                slippage=slippage_cost,
                opportunity_cost=opportunity_cost,
                total_cost=0.0
            )
            
            costs.calculate_total()
            
            return costs
            
        except Exception as e:
            logger.error("Error calculating execution costs", error=str(e), symbol=symbol)
            return ExecutionCosts(0, 0, 0, 0, 0, 0)
    
    def _assess_execution_quality(self, signal_confidence: float, market_impact: float, 
                                spread_pct: float, volume: float) -> ExecutionQuality:
        """Assess execution quality based on market conditions"""
        try:
            score = 0
            
            # Signal confidence factor
            if signal_confidence > 0.8:
                score += 3
            elif signal_confidence > 0.6:
                score += 2
            elif signal_confidence > 0.4:
                score += 1
            
            # Market impact factor
            if market_impact < self.impact_threshold * 0.5:
                score += 3
            elif market_impact < self.impact_threshold:
                score += 2
            elif market_impact < self.impact_threshold * 2:
                score += 1
            
            # Spread factor
            if spread_pct < self.spread_threshold * 0.5:
                score += 2
            elif spread_pct < self.spread_threshold:
                score += 1
            
            # Volume factor
            if volume > self.volume_threshold * 2:
                score += 2
            elif volume > self.volume_threshold:
                score += 1
            
            # Determine quality
            if score >= 8:
                return ExecutionQuality.EXCELLENT
            elif score >= 6:
                return ExecutionQuality.GOOD
            elif score >= 4:
                return ExecutionQuality.AVERAGE
            else:
                return ExecutionQuality.POOR
                
        except Exception as e:
            logger.error("Error assessing execution quality", error=str(e))
            return ExecutionQuality.AVERAGE
    
    def estimate_fill_probability(self, symbol: str, order_type: OrderType, 
                                order_size: float, limit_price: float, 
                                market_data: Dict[str, Any]) -> float:
        """Estimate probability of order fill"""
        try:
            if order_type == OrderType.MARKET:
                return 0.99  # Market orders almost always fill
            
            current_price = market_data.get('price', 0)
            volume = market_data.get('volume', 0)
            spread = market_data.get('spread', 0)
            
            if current_price <= 0:
                return 0.0
            
            # Distance from market price
            price_distance = abs(limit_price - current_price) / current_price
            
            # Base fill probability
            if price_distance == 0:
                fill_prob = 0.95  # At market price
            elif price_distance < 0.001:  # Within 0.1%
                fill_prob = 0.9
            elif price_distance < 0.005:  # Within 0.5%
                fill_prob = 0.7
            elif price_distance < 0.01:   # Within 1%
                fill_prob = 0.5
            else:
                fill_prob = 0.2
            
            # Adjust for order size vs volume
            if volume > 0:
                size_factor = min(1.0, volume / (abs(order_size) * 10))
                fill_prob *= size_factor
            
            # Adjust for spread
            if spread > 0 and current_price > 0:
                spread_factor = min(1.0, 1.0 - (spread / current_price) * 10)
                fill_prob *= spread_factor
            
            # Apply partial fill probability
            partial_fill_prob = self.fill_modeling_config['partial_fill_probability']
            if np.random.random() < partial_fill_prob:
                fill_prob *= 0.7  # Partial fill reduces effective probability
            
            return max(0.0, min(1.0, fill_prob))
            
        except Exception as e:
            logger.error("Error estimating fill probability", error=str(e), symbol=symbol)
            return 0.5  # Default 50% probability
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        total_orders = self.execution_metrics['total_orders']
        
        return {
            'total_orders': total_orders,
            'successful_executions': self.execution_metrics['successful_executions'],
            'blocked_orders': self.execution_metrics['blocked_orders'],
            'success_rate': self.execution_metrics['successful_executions'] / max(1, total_orders),
            'block_rate': self.execution_metrics['blocked_orders'] / max(1, total_orders),
            'avg_execution_cost': self.execution_metrics['avg_execution_cost'],
            'avg_market_impact': self.execution_metrics['avg_market_impact'],
            'avg_fill_rate': self.execution_metrics['avg_fill_rate'],
            'rule_violations': self.execution_metrics['rule_violations'],
            'violation_rate': self.execution_metrics['rule_violations'] / max(1, total_orders)
        }
    
    def validate_execution_rules(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate execution rules against a list of orders"""
        violations = []
        warnings = []
        
        for order in orders:
            try:
                symbol = order.get('symbol', '')
                size = order.get('size', 0)
                confidence = order.get('confidence', 0)
                market_data = order.get('market_data', {})
                
                # Evaluate execution
                evaluation = self.evaluate_entry_execution(symbol, size, confidence, market_data)
                
                if not evaluation.get('execution_allowed', True):
                    violations.append({
                        'symbol': symbol,
                        'order_size': size,
                        'violations': evaluation.get('rule_violations', [])
                    })
                
                # Check for warnings
                if evaluation.get('execution_quality') == ExecutionQuality.POOR:
                    warnings.append({
                        'symbol': symbol,
                        'issue': 'poor_execution_quality',
                        'quality': evaluation.get('execution_quality').value
                    })
                
            except Exception as e:
                violations.append({
                    'symbol': order.get('symbol', 'unknown'),
                    'error': str(e)
                })
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings,
            'total_orders_checked': len(orders)
        }