"""
Benchmark Agents

Collection of traditional benchmark strategies for baseline comparison.
These agents implement standard investment strategies used in practice.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from .rule_based_agent import RuleBasedAgent


class BuyAndHoldAgent(RuleBasedAgent):
    """
    Buy and Hold agent
    
    Features:
    - Simple buy-and-hold strategy
    - Optional rebalancing periods
    - Configurable initial allocation
    - Performance tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Buy and hold parameters
        self.initial_allocation = self.config.get('initial_allocation', [0.05, 0.05, 0.9])  # [bear, neutral, bull]
        self.rebalance_period = self.config.get('rebalance_period', None)  # None for no rebalancing
        self.rebalance_threshold = self.config.get('rebalance_threshold', 0.1)  # 10% drift threshold
        
        # Position tracking
        self.current_allocation = np.array(self.initial_allocation)
        self.step_count = 0
        self.last_rebalance = 0
        
        # Performance tracking
        self.initial_value = 1.0
        self.current_value = 1.0
        self.returns_history = []
        
    def should_rebalance(self) -> bool:
        """Check if rebalancing is needed"""
        if self.rebalance_period is None:
            return False
            
        # Time-based rebalancing
        if self.step_count - self.last_rebalance >= self.rebalance_period:
            return True
            
        # Drift-based rebalancing
        allocation_drift = np.abs(self.current_allocation - self.initial_allocation)
        if np.any(allocation_drift > self.rebalance_threshold):
            return True
            
        return False
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate buy-and-hold action"""
        self.step_count += 1
        
        # Check if rebalancing is needed
        if self.should_rebalance():
            self.current_allocation = np.array(self.initial_allocation)
            self.last_rebalance = self.step_count
            
        # Ensure valid probability distribution
        action = self.current_allocation / self.current_allocation.sum()
        
        # Record action
        self.action_history.append(action)
        
        return action
    
    def update_performance(self, returns: float):
        """Update performance metrics"""
        self.current_value *= (1 + returns)
        self.returns_history.append(returns)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        base_stats = super().get_statistics()
        
        # Buy and hold specific statistics
        total_return = (self.current_value - self.initial_value) / self.initial_value
        
        if self.returns_history:
            avg_return = np.mean(self.returns_history)
            volatility = np.std(self.returns_history)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
        else:
            avg_return = 0.0
            volatility = 0.0
            sharpe_ratio = 0.0
        
        base_stats.update({
            'buy_and_hold': {
                'total_return': total_return,
                'avg_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'rebalance_count': max(0, self.step_count - self.last_rebalance) if self.rebalance_period else 0
            }
        })
        
        return base_stats
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.current_allocation = np.array(self.initial_allocation)
        self.step_count = 0
        self.last_rebalance = 0
        self.current_value = 1.0
        self.returns_history = []


class EqualWeightAgent(RuleBasedAgent):
    """
    Equal Weight agent
    
    Features:
    - Equal weight allocation across all actions
    - Optional exclusion of neutral position
    - Periodic rebalancing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Equal weight parameters
        self.include_neutral = self.config.get('include_neutral', True)
        self.rebalance_period = self.config.get('rebalance_period', 1)  # Rebalance every step
        self.neutral_weight = self.config.get('neutral_weight', 0.2)  # Weight for neutral when excluded
        
        # Calculate equal weights
        if self.include_neutral:
            self.target_allocation = np.array([1/3, 1/3, 1/3])  # Equal weight
        else:
            # Exclude neutral, equal weight between bull/bear
            self.target_allocation = np.array([0.4, self.neutral_weight, 0.4])
        
        self.step_count = 0
        
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate equal weight action"""
        self.step_count += 1
        
        # Always return target allocation (equal weight)
        action = self.target_allocation.copy()
        
        # Ensure valid probability distribution
        action = action / action.sum()
        
        # Record action
        self.action_history.append(action)
        
        return action
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        base_stats = super().get_statistics()
        
        base_stats.update({
            'equal_weight': {
                'target_allocation': self.target_allocation.tolist(),
                'include_neutral': self.include_neutral,
                'rebalance_period': self.rebalance_period
            }
        })
        
        return base_stats
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.step_count = 0


class MarketCapWeightedAgent(RuleBasedAgent):
    """
    Market Cap Weighted agent
    
    Features:
    - Weights based on market capitalization proxy
    - Dynamic weight adjustment
    - Minimum weight constraints
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Market cap parameters
        self.market_cap_proxy = self.config.get('market_cap_proxy', 'volume')  # 'volume' or 'price'
        self.min_weight = self.config.get('min_weight', 0.05)  # Minimum 5% weight
        self.max_weight = self.config.get('max_weight', 0.8)   # Maximum 80% weight
        self.lookback_period = self.config.get('lookback_period', 20)
        
        # Weight calculation
        self.use_momentum_bias = self.config.get('use_momentum_bias', True)
        self.momentum_weight = self.config.get('momentum_weight', 0.3)
        
        # History tracking
        self.market_data_history = []
        self.weight_history = []
        
    def extract_market_data(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Extract market data for weight calculation"""
        features = observation.get('features', np.array([]))
        shared_context = observation.get('shared_context', np.array([]))
        
        market_data = {
            'price': 0.0,
            'volume': 1.0,
            'volatility': 0.15  # Default volatility
        }
        
        if len(features) > 0:
            market_data['price'] = features[0]
            if len(features) > 4:
                market_data['volume'] = features[4]
        
        if len(shared_context) > 2:
            market_data['volatility'] = np.exp(shared_context[2])
        
        return market_data
    
    def calculate_market_cap_weights(self) -> np.ndarray:
        """Calculate weights based on market cap proxy"""
        if len(self.market_data_history) < 2:
            return np.array([1/3, 1/3, 1/3])  # Equal weight default
        
        # Calculate market cap proxy
        if self.market_cap_proxy == 'volume':
            # Use volume-weighted approach
            recent_volumes = [data['volume'] for data in self.market_data_history[-self.lookback_period:]]
            avg_volume = np.mean(recent_volumes)
            
            # Higher volume -> higher weight for active positions
            if avg_volume > 1.0:
                # High volume -> favor active positions
                base_weights = np.array([0.4, 0.2, 0.4])
            else:
                # Low volume -> more conservative
                base_weights = np.array([0.3, 0.4, 0.3])
                
        elif self.market_cap_proxy == 'price':
            # Use price momentum as proxy
            recent_prices = [data['price'] for data in self.market_data_history[-self.lookback_period:]]
            
            if len(recent_prices) >= 2:
                price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                
                if price_momentum > 0.05:  # Strong positive momentum
                    base_weights = np.array([0.1, 0.1, 0.8])
                elif price_momentum < -0.05:  # Strong negative momentum
                    base_weights = np.array([0.8, 0.1, 0.1])
                else:  # Neutral momentum
                    base_weights = np.array([0.3, 0.4, 0.3])
            else:
                base_weights = np.array([1/3, 1/3, 1/3])
        else:
            base_weights = np.array([1/3, 1/3, 1/3])
        
        # Apply momentum bias if enabled
        if self.use_momentum_bias and len(self.market_data_history) >= 5:
            recent_data = self.market_data_history[-5:]
            prices = [data['price'] for data in recent_data]
            momentum = (prices[-1] - prices[0]) / prices[0]
            
            # Adjust weights based on momentum
            momentum_adjustment = np.array([
                -momentum * self.momentum_weight,  # Bearish adjustment
                0.0,  # Neutral unchanged
                momentum * self.momentum_weight   # Bullish adjustment
            ])
            
            base_weights += momentum_adjustment
        
        # Apply min/max constraints
        base_weights = np.clip(base_weights, self.min_weight, self.max_weight)
        
        # Normalize to sum to 1
        base_weights = base_weights / base_weights.sum()
        
        return base_weights
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate market cap weighted action"""
        # Extract market data
        market_data = self.extract_market_data(observation)
        self.market_data_history.append(market_data)
        
        # Maintain history limit
        if len(self.market_data_history) > self.lookback_period * 2:
            self.market_data_history = self.market_data_history[-self.lookback_period * 2:]
        
        # Calculate weights
        action = self.calculate_market_cap_weights()
        
        # Record action and weights
        self.action_history.append(action)
        self.weight_history.append(action.copy())
        
        return action
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        base_stats = super().get_statistics()
        
        if self.weight_history:
            avg_weights = np.mean(self.weight_history, axis=0)
            weight_volatility = np.std(self.weight_history, axis=0)
        else:
            avg_weights = np.array([0, 0, 0])
            weight_volatility = np.array([0, 0, 0])
        
        base_stats.update({
            'market_cap_weighted': {
                'avg_weights': avg_weights.tolist(),
                'weight_volatility': weight_volatility.tolist(),
                'market_cap_proxy': self.market_cap_proxy,
                'use_momentum_bias': self.use_momentum_bias
            }
        })
        
        return base_stats
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.market_data_history = []
        self.weight_history = []


class SectorRotationAgent(RuleBasedAgent):
    """
    Sector Rotation agent
    
    Features:
    - Rotates between different market positions based on regime
    - Economic cycle awareness
    - Volatility-based regime detection
    - Momentum-based rotation signals
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Sector rotation parameters
        self.rotation_period = self.config.get('rotation_period', 20)  # Minimum periods between rotations
        self.volatility_threshold = self.config.get('volatility_threshold', 0.2)
        self.momentum_threshold = self.config.get('momentum_threshold', 0.05)
        
        # Regime detection
        self.regime_lookback = self.config.get('regime_lookback', 30)
        self.use_volatility_regime = self.config.get('use_volatility_regime', True)
        self.use_momentum_regime = self.config.get('use_momentum_regime', True)
        
        # Sector allocations for different regimes
        self.regime_allocations = {
            'bull_market': np.array([0.05, 0.05, 0.9]),    # Aggressive growth
            'bear_market': np.array([0.9, 0.05, 0.05]),    # Defensive
            'neutral_market': np.array([0.2, 0.6, 0.2]),   # Balanced
            'high_volatility': np.array([0.1, 0.8, 0.1]),  # Conservative
            'low_volatility': np.array([0.3, 0.1, 0.6])    # Risk-on
        }
        
        # State tracking
        self.current_regime = 'neutral_market'
        self.last_rotation = 0
        self.step_count = 0
        self.market_history = []
        self.regime_history = []
        
    def extract_market_conditions(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Extract market conditions for regime detection"""
        features = observation.get('features', np.array([]))
        shared_context = observation.get('shared_context', np.array([]))
        
        conditions = {
            'price': 0.0,
            'volatility': 0.15,
            'trend': 0.0,
            'volume': 1.0
        }
        
        if len(features) > 0:
            conditions['price'] = features[0]
            if len(features) > 4:
                conditions['volume'] = features[4]
        
        if len(shared_context) > 2:
            conditions['volatility'] = np.exp(shared_context[2])
        
        return conditions
    
    def detect_market_regime(self) -> str:
        """Detect current market regime"""
        if len(self.market_history) < self.regime_lookback:
            return 'neutral_market'
        
        recent_data = self.market_history[-self.regime_lookback:]
        
        # Calculate market metrics
        prices = [data['price'] for data in recent_data]
        volatilities = [data['volatility'] for data in recent_data]
        
        # Momentum calculation
        if len(prices) >= 2:
            momentum = (prices[-1] - prices[0]) / prices[0]
        else:
            momentum = 0.0
        
        # Volatility calculation
        avg_volatility = np.mean(volatilities)
        
        # Regime detection logic
        if self.use_volatility_regime and avg_volatility > self.volatility_threshold:
            return 'high_volatility'
        elif self.use_volatility_regime and avg_volatility < self.volatility_threshold / 2:
            return 'low_volatility'
        elif self.use_momentum_regime and momentum > self.momentum_threshold:
            return 'bull_market'
        elif self.use_momentum_regime and momentum < -self.momentum_threshold:
            return 'bear_market'
        else:
            return 'neutral_market'
    
    def should_rotate(self, new_regime: str) -> bool:
        """Check if rotation should occur"""
        # Always rotate if regime changed
        if new_regime != self.current_regime:
            return True
        
        # Time-based rotation check
        if self.step_count - self.last_rotation >= self.rotation_period:
            return True
        
        return False
    
    def calculate_transition_weights(self, target_allocation: np.ndarray) -> np.ndarray:
        """Calculate smooth transition weights"""
        if not self.action_history:
            return target_allocation
        
        current_allocation = self.action_history[-1]
        
        # Smooth transition over multiple steps
        transition_speed = 0.3  # 30% of the way to target each step
        
        new_allocation = (
            (1 - transition_speed) * current_allocation +
            transition_speed * target_allocation
        )
        
        return new_allocation
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate sector rotation action"""
        self.step_count += 1
        
        # Extract market conditions
        market_conditions = self.extract_market_conditions(observation)
        self.market_history.append(market_conditions)
        
        # Maintain history limit
        if len(self.market_history) > self.regime_lookback * 2:
            self.market_history = self.market_history[-self.regime_lookback * 2:]
        
        # Detect market regime
        detected_regime = self.detect_market_regime()
        
        # Check if rotation is needed
        if self.should_rotate(detected_regime):
            self.current_regime = detected_regime
            self.last_rotation = self.step_count
        
        # Get target allocation for current regime
        target_allocation = self.regime_allocations.get(
            self.current_regime, 
            self.regime_allocations['neutral_market']
        )
        
        # Calculate smooth transition
        action = self.calculate_transition_weights(target_allocation)
        
        # Ensure valid probability distribution
        action = action / action.sum()
        
        # Record action and regime
        self.action_history.append(action)
        self.regime_history.append(self.current_regime)
        
        return action
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        base_stats = super().get_statistics()
        
        # Regime statistics
        if self.regime_history:
            regime_counts = {}
            for regime in self.regime_history:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            regime_distribution = {
                regime: count / len(self.regime_history) 
                for regime, count in regime_counts.items()
            }
        else:
            regime_distribution = {}
        
        rotations_count = len(set(self.regime_history)) - 1 if len(self.regime_history) > 1 else 0
        
        base_stats.update({
            'sector_rotation': {
                'current_regime': self.current_regime,
                'regime_distribution': regime_distribution,
                'rotations_count': rotations_count,
                'avg_rotation_period': self.rotation_period,
                'volatility_threshold': self.volatility_threshold,
                'momentum_threshold': self.momentum_threshold
            }
        })
        
        return base_stats
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.current_regime = 'neutral_market'
        self.last_rotation = 0
        self.step_count = 0
        self.market_history = []
        self.regime_history = []


class RiskParityAgent(RuleBasedAgent):
    """
    Risk Parity agent
    
    Features:
    - Equal risk contribution from each position
    - Volatility-based position sizing
    - Dynamic risk budgeting
    - Correlation-aware allocation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Risk parity parameters
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        self.risk_target = self.config.get('risk_target', 0.15)  # 15% target risk
        self.min_weight = self.config.get('min_weight', 0.05)
        self.max_weight = self.config.get('max_weight', 0.8)
        
        # Risk calculation
        self.use_realized_volatility = self.config.get('use_realized_volatility', True)
        self.volatility_adjustment = self.config.get('volatility_adjustment', 1.0)
        
        # History tracking
        self.returns_history = []
        self.volatility_history = []
        self.risk_contributions = []
        
    def calculate_position_volatilities(self) -> np.ndarray:
        """Calculate volatility for each position"""
        if len(self.returns_history) < self.volatility_lookback:
            # Default volatilities
            return np.array([0.2, 0.1, 0.2])  # Bear, Neutral, Bull
        
        # Use return history to estimate volatilities
        recent_returns = self.returns_history[-self.volatility_lookback:]
        
        # Estimate volatilities (simplified approach)
        # In practice, you'd have separate return series for each position
        base_volatility = np.std(recent_returns)
        
        # Assume different volatilities for different positions
        position_volatilities = np.array([
            base_volatility * 1.2,  # Bear position (higher vol)
            base_volatility * 0.6,  # Neutral position (lower vol)
            base_volatility * 1.1   # Bull position (moderate vol)
        ])
        
        return position_volatilities
    
    def calculate_risk_parity_weights(self) -> np.ndarray:
        """Calculate risk parity weights"""
        # Get position volatilities
        volatilities = self.calculate_position_volatilities()
        
        # Risk parity: weight inversely proportional to volatility
        inverse_volatilities = 1.0 / (volatilities + 1e-6)  # Add small epsilon
        
        # Normalize to get weights
        weights = inverse_volatilities / inverse_volatilities.sum()
        
        # Apply min/max constraints
        weights = np.clip(weights, self.min_weight, self.max_weight)
        
        # Renormalize after clipping
        weights = weights / weights.sum()
        
        return weights
    
    def calculate_risk_contributions(self, weights: np.ndarray, volatilities: np.ndarray) -> np.ndarray:
        """Calculate risk contribution of each position"""
        # Risk contribution = weight * volatility / portfolio_volatility
        portfolio_volatility = np.sqrt(np.sum((weights * volatilities) ** 2))
        
        if portfolio_volatility > 0:
            risk_contributions = (weights * volatilities) / portfolio_volatility
        else:
            risk_contributions = weights / weights.sum()
        
        return risk_contributions
    
    def update_returns(self, observation: Dict[str, Any]):
        """Update returns history"""
        features = observation.get('features', np.array([]))
        
        if len(features) > 0 and len(self.returns_history) > 0:
            # Calculate return (simplified)
            current_price = features[0]
            if hasattr(self, 'previous_price'):
                return_value = (current_price - self.previous_price) / self.previous_price
                self.returns_history.append(return_value)
            
            self.previous_price = current_price
        elif len(features) > 0:
            self.previous_price = features[0]
        
        # Maintain history limit
        if len(self.returns_history) > self.volatility_lookback * 2:
            self.returns_history = self.returns_history[-self.volatility_lookback * 2:]
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        """Generate risk parity action"""
        # Update returns history
        self.update_returns(observation)
        
        # Calculate risk parity weights
        weights = self.calculate_risk_parity_weights()
        
        # Calculate risk contributions
        volatilities = self.calculate_position_volatilities()
        risk_contributions = self.calculate_risk_contributions(weights, volatilities)
        
        # Store risk contributions
        self.risk_contributions.append(risk_contributions)
        self.volatility_history.append(volatilities)
        
        # Record action
        self.action_history.append(weights)
        
        return weights
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        base_stats = super().get_statistics()
        
        if self.risk_contributions:
            avg_risk_contributions = np.mean(self.risk_contributions, axis=0)
            risk_contribution_std = np.std(self.risk_contributions, axis=0)
        else:
            avg_risk_contributions = np.array([0, 0, 0])
            risk_contribution_std = np.array([0, 0, 0])
        
        if self.volatility_history:
            avg_volatilities = np.mean(self.volatility_history, axis=0)
        else:
            avg_volatilities = np.array([0, 0, 0])
        
        base_stats.update({
            'risk_parity': {
                'avg_risk_contributions': avg_risk_contributions.tolist(),
                'risk_contribution_std': risk_contribution_std.tolist(),
                'avg_volatilities': avg_volatilities.tolist(),
                'risk_target': self.risk_target,
                'volatility_lookback': self.volatility_lookback
            }
        })
        
        return base_stats
    
    def reset(self):
        """Reset agent state"""
        super().reset()
        self.returns_history = []
        self.volatility_history = []
        self.risk_contributions = []
        if hasattr(self, 'previous_price'):
            delattr(self, 'previous_price')