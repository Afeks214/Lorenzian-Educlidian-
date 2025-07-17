"""
Correlation-Based Position Limits Agent

Implements correlation-based risk management including:
- Maximum 0.7 correlation threshold between positions
- Dynamic correlation matrix calculation
- Position size adjustments based on correlation
- Correlation cluster detection
- Diversification enforcement
- Real-time correlation monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
import structlog
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import yaml
from collections import defaultdict
from itertools import combinations

logger = structlog.get_logger()


class CorrelationLevel(Enum):
    """Correlation level classification"""
    LOW = "low"           # < 0.3
    MODERATE = "moderate" # 0.3 - 0.5
    HIGH = "high"         # 0.5 - 0.7
    EXCESSIVE = "excessive" # > 0.7


class CorrelationAction(Enum):
    """Actions to take for correlation violations"""
    NONE = "none"
    REDUCE_NEWER_POSITION = "reduce_newer_position"
    REDUCE_SMALLER_POSITION = "reduce_smaller_position"
    REDUCE_BOTH_POSITIONS = "reduce_both_positions"
    BLOCK_NEW_POSITION = "block_new_position"
    CLOSE_CLUSTER = "close_cluster"


@dataclass
class CorrelationPair:
    """Correlation pair information"""
    symbol1: str
    symbol2: str
    correlation: float
    correlation_level: CorrelationLevel
    lookback_period: int
    last_updated: datetime
    position_size1: float = 0.0
    position_size2: float = 0.0
    combined_risk: float = 0.0
    
    def needs_action(self, threshold: float) -> bool:
        """Check if correlation pair needs action"""
        return self.correlation > threshold


@dataclass
class CorrelationCluster:
    """Correlation cluster information"""
    symbols: Set[str]
    avg_correlation: float
    max_correlation: float
    total_exposure: float
    cluster_risk: float
    formed_at: datetime
    
    def add_symbol(self, symbol: str, correlation: float):
        """Add symbol to cluster"""
        self.symbols.add(symbol)
        # Recalculate average correlation
        # This is simplified - in practice would use full correlation matrix
        
    def remove_symbol(self, symbol: str):
        """Remove symbol from cluster"""
        self.symbols.discard(symbol)


@dataclass
class CorrelationMetrics:
    """Correlation monitoring metrics"""
    total_pairs_monitored: int = 0
    high_correlation_pairs: int = 0
    excessive_correlation_pairs: int = 0
    correlation_violations: int = 0
    position_reductions: int = 0
    blocked_positions: int = 0
    clusters_detected: int = 0
    avg_portfolio_correlation: float = 0.0
    max_portfolio_correlation: float = 0.0
    diversification_score: float = 1.0


class CorrelationLimitsAgent:
    """
    Correlation-Based Position Limits Agent
    
    Features:
    - Real-time correlation matrix calculation
    - Position size adjustments based on correlation
    - Correlation cluster detection
    - Maximum correlation threshold enforcement
    - Diversification scoring and monitoring
    - Dynamic correlation-based position limits
    """
    
    def __init__(self, config_path: str = "/home/QuantNova/GrandModel/config/risk_management_config.yaml"):
        """Initialize Correlation Limits Agent"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.correlation_config = self.config['risk_controls']['correlation_limits']
        
        # Correlation parameters
        self.max_correlation = self.correlation_config['max_correlation']
        self.correlation_lookback = self.correlation_config['correlation_lookback']
        self.max_correlated_positions = self.correlation_config['max_correlated_positions']
        self.correlation_adjustment_factor = self.correlation_config['correlation_adjustment_factor']
        
        # Correlation tracking
        self.correlation_matrix = {}
        self.correlation_pairs: Dict[Tuple[str, str], CorrelationPair] = {}
        self.correlation_clusters: List[CorrelationCluster] = []
        self.historical_returns: Dict[str, List[float]] = {}
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Position tracking
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.position_correlations: Dict[str, Dict[str, float]] = {}
        
        # Metrics and monitoring
        self.correlation_metrics = CorrelationMetrics()
        self.correlation_violations = []
        self.correlation_actions = []
        
        # Callbacks
        self.position_adjustment_callback = None
        self.position_block_callback = None
        
        logger.info("Correlation Limits Agent initialized",
                   max_correlation=self.max_correlation,
                   lookback_days=self.correlation_lookback,
                   max_correlated_positions=self.max_correlated_positions)
    
    def update_price_data(self, symbol: str, price: float, timestamp: datetime = None):
        """Update price data for correlation calculation"""
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Add to price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append((timestamp, price))
            
            # Keep only recent data
            cutoff_time = timestamp - timedelta(days=self.correlation_lookback)
            self.price_history[symbol] = [
                (t, p) for t, p in self.price_history[symbol] 
                if t >= cutoff_time
            ]
            
            # Calculate returns
            self._calculate_returns(symbol)
            
            # Update correlation matrix
            self._update_correlation_matrix(symbol)
            
        except Exception as e:
            logger.error("Error updating price data", error=str(e), symbol=symbol)
    
    def _calculate_returns(self, symbol: str):
        """Calculate returns for symbol"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
                return
            
            prices = [p for _, p in self.price_history[symbol]]
            returns = []
            
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    return_value = (prices[i] - prices[i-1]) / prices[i-1]
                    returns.append(return_value)
            
            self.historical_returns[symbol] = returns
            
        except Exception as e:
            logger.error("Error calculating returns", error=str(e), symbol=symbol)
    
    def _update_correlation_matrix(self, updated_symbol: str):
        """Update correlation matrix after price update"""
        try:
            # Update correlations for the updated symbol with all other symbols
            for other_symbol in self.historical_returns:
                if other_symbol != updated_symbol:
                    correlation = self._calculate_correlation(updated_symbol, other_symbol)
                    if correlation is not None:
                        # Store in matrix (symmetric)
                        if updated_symbol not in self.correlation_matrix:
                            self.correlation_matrix[updated_symbol] = {}
                        if other_symbol not in self.correlation_matrix:
                            self.correlation_matrix[other_symbol] = {}
                        
                        self.correlation_matrix[updated_symbol][other_symbol] = correlation
                        self.correlation_matrix[other_symbol][updated_symbol] = correlation
                        
                        # Update correlation pair
                        pair_key = tuple(sorted([updated_symbol, other_symbol]))
                        self._update_correlation_pair(pair_key, correlation)
            
            # Update portfolio correlation metrics
            self._update_portfolio_correlation_metrics()
            
        except Exception as e:
            logger.error("Error updating correlation matrix", error=str(e))
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Calculate correlation between two symbols"""
        try:
            if (symbol1 not in self.historical_returns or 
                symbol2 not in self.historical_returns):
                return None
            
            returns1 = self.historical_returns[symbol1]
            returns2 = self.historical_returns[symbol2]
            
            if len(returns1) < 20 or len(returns2) < 20:  # Minimum data requirement
                return None
            
            # Align returns by length
            min_length = min(len(returns1), len(returns2))
            returns1 = returns1[-min_length:]
            returns2 = returns2[-min_length:]
            
            # Calculate correlation
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation):
                return None
            
            return correlation
            
        except Exception as e:
            logger.error("Error calculating correlation", error=str(e), 
                        symbol1=symbol1, symbol2=symbol2)
            return None
    
    def _update_correlation_pair(self, pair_key: Tuple[str, str], correlation: float):
        """Update correlation pair information"""
        try:
            symbol1, symbol2 = pair_key
            
            # Classify correlation level
            if abs(correlation) > 0.7:
                level = CorrelationLevel.EXCESSIVE
            elif abs(correlation) > 0.5:
                level = CorrelationLevel.HIGH
            elif abs(correlation) > 0.3:
                level = CorrelationLevel.MODERATE
            else:
                level = CorrelationLevel.LOW
            
            # Get position sizes
            pos1_size = self.current_positions.get(symbol1, {}).get('size', 0.0)
            pos2_size = self.current_positions.get(symbol2, {}).get('size', 0.0)
            
            # Create or update correlation pair
            if pair_key not in self.correlation_pairs:
                self.correlation_pairs[pair_key] = CorrelationPair(
                    symbol1=symbol1,
                    symbol2=symbol2,
                    correlation=correlation,
                    correlation_level=level,
                    lookback_period=self.correlation_lookback,
                    last_updated=datetime.now(),
                    position_size1=pos1_size,
                    position_size2=pos2_size
                )
            else:
                pair = self.correlation_pairs[pair_key]
                pair.correlation = correlation
                pair.correlation_level = level
                pair.last_updated = datetime.now()
                pair.position_size1 = pos1_size
                pair.position_size2 = pos2_size
            
            # Calculate combined risk
            self._calculate_combined_risk(pair_key)
            
        except Exception as e:
            logger.error("Error updating correlation pair", error=str(e))
    
    def _calculate_combined_risk(self, pair_key: Tuple[str, str]):
        """Calculate combined risk for correlation pair"""
        try:
            if pair_key not in self.correlation_pairs:
                return
            
            pair = self.correlation_pairs[pair_key]
            
            # Simplified combined risk calculation
            # In practice, this would use volatility and position values
            individual_risk1 = abs(pair.position_size1) * 0.02  # 2% volatility assumption
            individual_risk2 = abs(pair.position_size2) * 0.02
            
            # Correlation adjustment
            correlation_factor = 1.0 + pair.correlation * 0.5  # Increase risk for positive correlation
            combined_risk = (individual_risk1 + individual_risk2) * correlation_factor
            
            pair.combined_risk = combined_risk
            
        except Exception as e:
            logger.error("Error calculating combined risk", error=str(e))
    
    def check_position_correlation(self, symbol: str, proposed_size: float) -> Dict[str, Any]:
        """
        Check correlation constraints for a proposed position
        
        Args:
            symbol: Symbol for new/modified position
            proposed_size: Proposed position size
            
        Returns:
            Dictionary with correlation check results
        """
        try:
            result = {
                'symbol': symbol,
                'proposed_size': proposed_size,
                'correlation_violations': [],
                'recommended_adjustments': [],
                'action_required': False,
                'max_allowed_size': proposed_size,
                'blocking_correlations': []
            }
            
            # Check correlations with existing positions
            for existing_symbol, position in self.current_positions.items():
                if existing_symbol == symbol or position.get('size', 0) == 0:
                    continue
                
                # Get correlation
                correlation = self._get_correlation(symbol, existing_symbol)
                if correlation is None:
                    continue
                
                # Check if correlation exceeds threshold
                if abs(correlation) > self.max_correlation:
                    violation = {
                        'correlated_symbol': existing_symbol,
                        'correlation': correlation,
                        'existing_size': position.get('size', 0),
                        'violation_level': 'EXCESSIVE' if abs(correlation) > 0.8 else 'HIGH'
                    }
                    result['correlation_violations'].append(violation)
                    result['action_required'] = True
                    
                    # Calculate size adjustment
                    adjustment = self._calculate_size_adjustment(
                        symbol, existing_symbol, correlation, proposed_size, position.get('size', 0))
                    result['recommended_adjustments'].append(adjustment)
                    
                    # Update max allowed size
                    if adjustment.get('max_new_size', proposed_size) < result['max_allowed_size']:
                        result['max_allowed_size'] = adjustment.get('max_new_size', proposed_size)
            
            # Check for correlation clusters
            cluster_check = self._check_correlation_clusters(symbol, proposed_size)
            if cluster_check['in_cluster']:
                result['cluster_info'] = cluster_check
                if cluster_check['cluster_risk'] > 0.15:  # 15% cluster risk limit
                    result['action_required'] = True
                    result['blocking_correlations'].append('cluster_risk_exceeded')
            
            return result
            
        except Exception as e:
            logger.error("Error checking position correlation", error=str(e), symbol=symbol)
            return {'error': str(e)}
    
    def _calculate_size_adjustment(self, new_symbol: str, existing_symbol: str, 
                                 correlation: float, proposed_size: float, 
                                 existing_size: float) -> Dict[str, Any]:
        """Calculate position size adjustment for correlation constraint"""
        try:
            adjustment = {
                'type': 'size_adjustment',
                'reason': f'High correlation ({correlation:.3f}) with {existing_symbol}',
                'original_size': proposed_size,
                'max_new_size': proposed_size,
                'existing_adjustment': None
            }
            
            # Calculate adjustment based on correlation level
            if abs(correlation) > 0.8:
                # Very high correlation - reduce by 75%
                adjustment['max_new_size'] = proposed_size * 0.25
                adjustment['existing_adjustment'] = {
                    'symbol': existing_symbol,
                    'current_size': existing_size,
                    'recommended_size': existing_size * 0.5  # Reduce existing by 50%
                }
            elif abs(correlation) > 0.7:
                # High correlation - reduce by 50%
                adjustment['max_new_size'] = proposed_size * 0.5
            else:
                # Moderate correlation - reduce by 25%
                adjustment['max_new_size'] = proposed_size * 0.75
            
            return adjustment
            
        except Exception as e:
            logger.error("Error calculating size adjustment", error=str(e))
            return {'error': str(e)}
    
    def _check_correlation_clusters(self, symbol: str, proposed_size: float) -> Dict[str, Any]:
        """Check if symbol would join or form a correlation cluster"""
        try:
            result = {
                'in_cluster': False,
                'cluster_symbols': [],
                'cluster_correlation': 0.0,
                'cluster_risk': 0.0,
                'cluster_action': CorrelationAction.NONE
            }
            
            # Find symbols with high correlation to the new symbol
            highly_correlated = []
            for existing_symbol in self.current_positions:
                if existing_symbol == symbol:
                    continue
                
                correlation = self._get_correlation(symbol, existing_symbol)
                if correlation is not None and abs(correlation) > 0.6:
                    highly_correlated.append({
                        'symbol': existing_symbol,
                        'correlation': correlation,
                        'size': self.current_positions[existing_symbol].get('size', 0)
                    })
            
            # Check if this forms a cluster (3+ highly correlated positions)
            if len(highly_correlated) >= 2:
                result['in_cluster'] = True
                result['cluster_symbols'] = [item['symbol'] for item in highly_correlated]
                result['cluster_correlation'] = np.mean([abs(item['correlation']) for item in highly_correlated])
                
                # Calculate cluster risk
                cluster_exposure = sum(abs(item['size']) for item in highly_correlated)
                cluster_exposure += abs(proposed_size)  # Add new position
                result['cluster_risk'] = cluster_exposure * result['cluster_correlation']
                
                # Determine action
                if result['cluster_risk'] > 0.2:  # 20% cluster risk limit
                    result['cluster_action'] = CorrelationAction.CLOSE_CLUSTER
                elif result['cluster_risk'] > 0.15:  # 15% cluster risk limit
                    result['cluster_action'] = CorrelationAction.REDUCE_BOTH_POSITIONS
                elif result['cluster_risk'] > 0.1:  # 10% cluster risk limit
                    result['cluster_action'] = CorrelationAction.REDUCE_NEWER_POSITION
            
            return result
            
        except Exception as e:
            logger.error("Error checking correlation clusters", error=str(e))
            return {'error': str(e)}
    
    def apply_correlation_limits(self, symbol: str, proposed_size: float) -> Dict[str, Any]:
        """
        Apply correlation limits and execute necessary actions
        
        Args:
            symbol: Symbol for position
            proposed_size: Proposed position size
            
        Returns:
            Dictionary with applied actions and final position size
        """
        try:
            # Check correlation constraints
            correlation_check = self.check_position_correlation(symbol, proposed_size)
            
            if correlation_check.get('error'):
                return correlation_check
            
            result = {
                'symbol': symbol,
                'original_size': proposed_size,
                'final_size': proposed_size,
                'actions_taken': [],
                'violations_resolved': 0,
                'correlation_adjustments': []
            }
            
            # Apply size adjustments if needed
            if correlation_check['action_required']:
                result['final_size'] = correlation_check['max_allowed_size']
                
                # Execute adjustments
                for adjustment in correlation_check['recommended_adjustments']:
                    action_result = self._execute_correlation_adjustment(adjustment)
                    result['actions_taken'].append(action_result)
                    
                    if action_result.get('success'):
                        result['violations_resolved'] += 1
                
                # Handle cluster actions
                if 'cluster_info' in correlation_check:
                    cluster_action = self._execute_cluster_action(
                        correlation_check['cluster_info'])
                    result['actions_taken'].append(cluster_action)
            
            # Update position
            self.current_positions[symbol] = {
                'size': result['final_size'],
                'timestamp': datetime.now()
            }
            
            # Update metrics
            self._update_correlation_metrics(correlation_check)
            
            return result
            
        except Exception as e:
            logger.error("Error applying correlation limits", error=str(e), symbol=symbol)
            return {'error': str(e)}
    
    def _execute_correlation_adjustment(self, adjustment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute correlation adjustment action"""
        try:
            action_result = {
                'type': 'correlation_adjustment',
                'success': False,
                'adjustment': adjustment
            }
            
            if self.position_adjustment_callback:
                # Call external adjustment function
                success = self.position_adjustment_callback(adjustment)
                action_result['success'] = success
                
                if success:
                    self.correlation_metrics.position_reductions += 1
                    
                    # Log the action
                    self.correlation_actions.append({
                        'timestamp': datetime.now(),
                        'action': 'position_adjustment',
                        'details': adjustment
                    })
            
            return action_result
            
        except Exception as e:
            logger.error("Error executing correlation adjustment", error=str(e))
            return {'error': str(e)}
    
    def _execute_cluster_action(self, cluster_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cluster action"""
        try:
            action = cluster_info.get('cluster_action', CorrelationAction.NONE)
            
            action_result = {
                'type': 'cluster_action',
                'action': action.value,
                'success': False,
                'cluster_info': cluster_info
            }
            
            if action == CorrelationAction.CLOSE_CLUSTER:
                # Close all positions in cluster
                if self.position_adjustment_callback:
                    for symbol in cluster_info['cluster_symbols']:
                        close_adjustment = {
                            'symbol': symbol,
                            'original_size': self.current_positions.get(symbol, {}).get('size', 0),
                            'max_new_size': 0,
                            'type': 'cluster_close'
                        }
                        self.position_adjustment_callback(close_adjustment)
                    
                    action_result['success'] = True
                    self.correlation_metrics.clusters_detected += 1
            
            return action_result
            
        except Exception as e:
            logger.error("Error executing cluster action", error=str(e))
            return {'error': str(e)}
    
    def _update_correlation_metrics(self, correlation_check: Dict[str, Any]):
        """Update correlation metrics"""
        try:
            if correlation_check.get('correlation_violations'):
                self.correlation_metrics.correlation_violations += len(correlation_check['correlation_violations'])
                
                # Count high correlation pairs
                for violation in correlation_check['correlation_violations']:
                    if violation['violation_level'] == 'EXCESSIVE':
                        self.correlation_metrics.excessive_correlation_pairs += 1
                    else:
                        self.correlation_metrics.high_correlation_pairs += 1
            
            # Update total pairs monitored
            self.correlation_metrics.total_pairs_monitored = len(self.correlation_pairs)
            
        except Exception as e:
            logger.error("Error updating correlation metrics", error=str(e))
    
    def _update_portfolio_correlation_metrics(self):
        """Update portfolio-wide correlation metrics"""
        try:
            if not self.correlation_pairs:
                return
            
            # Calculate average portfolio correlation
            active_pairs = [
                pair for pair in self.correlation_pairs.values()
                if pair.position_size1 != 0 and pair.position_size2 != 0
            ]
            
            if active_pairs:
                self.correlation_metrics.avg_portfolio_correlation = np.mean([
                    abs(pair.correlation) for pair in active_pairs
                ])
                
                self.correlation_metrics.max_portfolio_correlation = max([
                    abs(pair.correlation) for pair in active_pairs
                ])
                
                # Calculate diversification score
                self.correlation_metrics.diversification_score = max(0.0, 
                    1.0 - self.correlation_metrics.avg_portfolio_correlation)
            
        except Exception as e:
            logger.error("Error updating portfolio correlation metrics", error=str(e))
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get correlation between two symbols"""
        try:
            if symbol1 in self.correlation_matrix and symbol2 in self.correlation_matrix[symbol1]:
                return self.correlation_matrix[symbol1][symbol2]
            return None
        except:
            return None
    
    def register_callbacks(self, position_adjustment_callback=None, position_block_callback=None):
        """Register callbacks for correlation actions"""
        self.position_adjustment_callback = position_adjustment_callback
        self.position_block_callback = position_block_callback
        
        logger.info("Correlation callbacks registered")
    
    def get_correlation_status(self) -> Dict[str, Any]:
        """Get current correlation status"""
        return {
            'total_pairs_monitored': self.correlation_metrics.total_pairs_monitored,
            'high_correlation_pairs': self.correlation_metrics.high_correlation_pairs,
            'excessive_correlation_pairs': self.correlation_metrics.excessive_correlation_pairs,
            'avg_portfolio_correlation': self.correlation_metrics.avg_portfolio_correlation,
            'max_portfolio_correlation': self.correlation_metrics.max_portfolio_correlation,
            'diversification_score': self.correlation_metrics.diversification_score,
            'correlation_violations': self.correlation_metrics.correlation_violations,
            'position_reductions': self.correlation_metrics.position_reductions,
            'clusters_detected': self.correlation_metrics.clusters_detected,
            'active_positions': len([p for p in self.current_positions.values() if p.get('size', 0) != 0])
        }
    
    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get current correlation matrix"""
        return self.correlation_matrix.copy()
    
    def get_high_correlation_pairs(self, threshold: float = None) -> List[Dict[str, Any]]:
        """Get pairs with high correlation"""
        if threshold is None:
            threshold = self.max_correlation
        
        high_pairs = []
        for pair in self.correlation_pairs.values():
            if abs(pair.correlation) > threshold:
                high_pairs.append({
                    'symbol1': pair.symbol1,
                    'symbol2': pair.symbol2,
                    'correlation': pair.correlation,
                    'correlation_level': pair.correlation_level.value,
                    'position_size1': pair.position_size1,
                    'position_size2': pair.position_size2,
                    'combined_risk': pair.combined_risk,
                    'last_updated': pair.last_updated.isoformat()
                })
        
        return sorted(high_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def validate_correlation_limits(self) -> Dict[str, Any]:
        """Validate current positions against correlation limits"""
        violations = []
        warnings = []
        
        try:
            # Check all pairs
            for pair in self.correlation_pairs.values():
                if (pair.position_size1 != 0 and pair.position_size2 != 0 and 
                    abs(pair.correlation) > self.max_correlation):
                    violations.append({
                        'type': 'correlation_violation',
                        'symbol1': pair.symbol1,
                        'symbol2': pair.symbol2,
                        'correlation': pair.correlation,
                        'threshold': self.max_correlation
                    })
                
                elif (pair.position_size1 != 0 and pair.position_size2 != 0 and 
                      abs(pair.correlation) > self.max_correlation * 0.8):
                    warnings.append({
                        'type': 'correlation_warning',
                        'symbol1': pair.symbol1,
                        'symbol2': pair.symbol2,
                        'correlation': pair.correlation,
                        'threshold': self.max_correlation * 0.8
                    })
            
            # Check portfolio diversification
            if self.correlation_metrics.diversification_score < 0.5:
                warnings.append({
                    'type': 'poor_diversification',
                    'diversification_score': self.correlation_metrics.diversification_score,
                    'threshold': 0.5
                })
            
            return {
                'is_valid': len(violations) == 0,
                'violations': violations,
                'warnings': warnings,
                'total_pairs': len(self.correlation_pairs),
                'high_correlation_pairs': self.correlation_metrics.high_correlation_pairs,
                'avg_correlation': self.correlation_metrics.avg_portfolio_correlation
            }
            
        except Exception as e:
            logger.error("Error validating correlation limits", error=str(e))
            return {'error': str(e)}